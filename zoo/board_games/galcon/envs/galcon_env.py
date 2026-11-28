import copy
import ctypes
import math
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gymnasium import spaces

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from zoo.board_games.galcon.envs.rule_bot import (
    GalconFixedPolicyBot,
    GalconMCTSBot,
    GalconPassBot,
    GalconRandomBot,
    build_galcon_bot,
)
from zoo.board_games.galcon.envs.cpp_sim import (
    BASE_ITEM_USERS,
    DEFAULT_PHYSICS_SUBSTEPS,
    EnvCppSimHandle,
    GalconSim as _CppSimFallback,
    HAS_FAST_SIM_CPP,
    PRODUCTION_DIVISOR,
    SHIP_RADIUS as _SIM_SHIP_RADIUS,
    physical_ship_count,
    planet_radius_from_production,
    spawn_fleet_units,
    step_physics_cpp,
)


@dataclass(slots=True)
class Planet:
    id: int
    owner: int
    ships: float
    x: float
    y: float
    production: float
    radius: float
    neutral: bool


class Fleet:
    __slots__ = (
        'id',
        'owner',
        'ships',
        'x',
        'y',
        'source',
        'target',
        'radius',
        'num_units',
        '_units',
        '_unit_headings',
        '_cpp_env',
        '_cpp_gen',
    )

    def __init__(
        self,
        id: int,
        owner: int,
        ships: float,
        x: float,
        y: float,
        source: int,
        target: int,
        radius: float,
        num_units: int = 1,
        units: Optional[List[List[float]]] = None,
        unit_headings: Optional[List[List[float]]] = None,
    ) -> None:
        self.id = id
        self.owner = owner
        self.ships = ships
        self.x = x
        self.y = y
        self.source = source
        self.target = target
        self.radius = radius
        self.num_units = num_units
        self._units = units
        self._unit_headings = unit_headings
        self._cpp_env: Optional['GalconEnv'] = None
        self._cpp_gen: int = -1

    @property
    def units(self) -> Optional[List[List[float]]]:
        if self._units is None and self._cpp_env is not None:
            self._cpp_env._materialize_fleet_units(self)
        return self._units

    @units.setter
    def units(self, value: Optional[List[List[float]]]) -> None:
        self._units = value
        if self._cpp_env is not None:
            self._cpp_env._cpp_units_materialized = True
            self._cpp_env = None

    @property
    def unit_headings(self) -> Optional[List[List[float]]]:
        if self._unit_headings is None and self._cpp_env is not None:
            self._cpp_env._materialize_fleet_units(self)
        return self._unit_headings

    @unit_headings.setter
    def unit_headings(self, value: Optional[List[List[float]]]) -> None:
        self._unit_headings = value
        if self._cpp_env is not None:
            self._cpp_env._cpp_units_materialized = True
            self._cpp_env = None

    def __repr__(self) -> str:
        return (
            f'Fleet(id={self.id!r}, owner={self.owner!r}, ships={self.ships!r}, '
            f'x={self.x!r}, y={self.y!r}, source={self.source!r}, target={self.target!r}, '
            f'radius={self.radius!r}, num_units={self.num_units!r})'
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fleet):
            return NotImplemented
        return (
            self.id == other.id
            and self.owner == other.owner
            and self.ships == other.ships
            and self.x == other.x
            and self.y == other.y
            and self.source == other.source
            and self.target == other.target
            and self.radius == other.radius
            and self.num_units == other.num_units
            and self.units == other.units
            and self.unit_headings == other.unit_headings
        )

    def __getstate__(self):
        return (
            self.id,
            self.owner,
            self.ships,
            self.x,
            self.y,
            self.source,
            self.target,
            self.radius,
            self.num_units,
            self.units,
            self.unit_headings,
        )

    def __setstate__(self, state) -> None:
        (
            self.id,
            self.owner,
            self.ships,
            self.x,
            self.y,
            self.source,
            self.target,
            self.radius,
            self.num_units,
            self._units,
            self._unit_headings,
        ) = state
        self._cpp_env = None
        self._cpp_gen = -1


@ENV_REGISTRY.register('galcon')
class GalconEnv(BaseEnv):
    """
    Overview:
        A LightZero-compatible Galcon-style two-player environment.

        Grid action / observation v1:
        - action 0 is pass.
        - actions 1..grid_cells^2 encode source grid cell -> target grid cell.
        - source grid cell must contain a friendly planet (or 1+ fleets, with fleet redirection enabled)
        - target grid cell must contain a planet.
        - send action sends a fixed ratio of source ships (for now)
        - observations are channel-first grid tensors.
    """

    PLAYER_NEUTRAL = 0
    PLAYER_1 = 1
    PLAYER_2 = 2

    # production=100 means 120 ships/min = 2 ships/sec => production / 50 ships/sec.
    PRODUCTION_TO_SHIPS_PER_SECOND_DIVISOR = PRODUCTION_DIVISOR
    # TODO: For now, fleet radius is just a constant.
    DUMMY_FLEET_RADIUS = 10.0
    SHIP_RADIUS = _SIM_SHIP_RADIUS
    # Real Galcon 2 planets_settle padding per planet: 8 + sqrt(0.5) (~8.707 px),
    # giving a minimum surface-to-surface gap of 2 * (8 + sqrt(0.5)) = 16 + sqrt(2) (~17.414 px).
    PLANETS_SETTLE_PADDING = 8.0 + math.sqrt(0.5)
    PLANETS_SETTLE_DELTA = 2.0 * PLANETS_SETTLE_PADDING - 2.0 * SHIP_RADIUS  # 4 + sqrt(2) ~= 5.4142
    # Number of attempts to settle planets (usually finishes in 1 or 2 attempts, and it will break early if successful)
    MAX_SETTLE_ATTEMPTS = 100

    PLANET_LOCAL_X_CHANNEL = 0
    PLANET_LOCAL_Y_CHANNEL = 1
    PLANET_FRIENDLY_SHIPS_CHANNEL = 2
    PLANET_ENEMY_SHIPS_CHANNEL = 3
    PLANET_NEUTRAL_SHIPS_CHANNEL = 4
    PLANET_FRIENDLY_PRODUCTION_CHANNEL = 5
    PLANET_ENEMY_PRODUCTION_CHANNEL = 6
    # neutrals don't produce ships, but we still want to include production so we know how good the planet is
    PLANET_NEUTRAL_PRODUCTION_CHANNEL = 7
    PLANET_CHANNEL_COUNT = 8

    FLEET_FEATURE_COUNT = 5
    # LANDING_BUCKETS = (
    #     (0.0, 1.0),
    #     (1.0, 2.0),
    #     (2.0, 4.0),
    #     (4.0, 6.0),
    #     (6.0, 8.0),
    #     (8.0, 10.0),
    #     (10.0, math.inf),
    # )
    # TODO: undo for the "real" runs
    LANDING_BUCKETS = (
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, math.inf),
    )
    LANDING_FEATURE_COUNT_PER_BUCKET = 4
    # Offset applied to the map RNG seed when dynamic_seed=False (evaluator envs) so that
    # collector and evaluator workers seeded with the same base seed (e.g. 0..N in train_muzero)
    # draw independent random map sequences when map_seed=None.
    EVAL_MAP_RNG_SEED_OFFSET = 1_000_000

    config = dict(
        env_id='Galcon',
        battle_mode='self_play_mode',
        bot_action_type='random',  # {'random', 'fixed', 'pass', 'mcts', 'waffle', 'waffleopt', 'sparky', 'sparkyn', 'greedy', 'expand', 'pressure', 'arcade', 'counter'}
        # Pareto-optimal fast pure C++ MCTS bot defaults (~5.5 ms/move; scale to 64 sims for +113 Elo at ~15 ms/move)
        mcts_num_simulations=16,
        mcts_c_puct=0.5,
        mcts_max_actions=10,
        mcts_max_depth=8,
        mcts_eval_horizon=40,
        mcts_rollout_ticks=0,
        mcts_prior_temp=1.5,
        num_planets=8,
        min_send_ships=1,
        send_ratio=0.5,
        tick_seconds=0.25,
        fleet_speed=40.0,
        max_episode_steps=200,
        timeout_reward_scale=1.0,
        map_seed=None,
        # Max grid size could be as large as (12 + 12 + 6 + 6 + 0.5) / sqrt(2) = 25.8
        grid_square_size=25.0,
        grid_max_x=200.0,
        grid_max_y=125.0,
        # If 1, homes always spawn on an ellipse touching the sides of the box. If in between, the ellipse shrinks proportionally.
        home_distance_fraction = 0.8,
        neutral_min_cost = 0,
        neutral_max_cost = 50,
        neutral_min_production = 15,
        neutral_max_production = 100,
        fleet_top_k=3,
        enable_fleet_redirection=False,
        # Use calibrated multi-ship C++17 Galcon 2 physics engine (falls back to bit-exact Python sim if g++ unavailable).
        use_cpp_sim=True,
        physics_substeps=DEFAULT_PHYSICS_SUBSTEPS,
        # TODO: settings for home ships, home prod, etc.
        # TODO: Base some of these settings (max_expected_ships, max_expected_production, etc. on these other settings)
        # Used for normalizing ship counts, the largest "blob" of ships we are likely to see (on any planet, grid cell, or fleet)
        # Only near the end of the game, where it doesn't matter much, might we see 250+ ships.
        max_expected_ships=250.0,
        max_expected_production=100.0,
        max_fleet_radius=500.0,
        # Note: on a 400 by 240 map, the longest possible flight is about 11.5 seconds
        # TODO: This should be calculated based on map size and ship speed.
        max_expected_eta=20.0,
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        manager=dict(shared_memory=False),
        channel_last=False,
        scale=True,
        replay_name_suffix='',
        render_mode=None,
        replay_path=None,
        # options are: {'mp4', 'gif'}. Only relevant for 'image_savefile_mode'
        replay_format='gif',
        replay_screen_scaling=9,
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        prob_random_action_in_bot=0.,
        prob_pass_action_in_bot=0.,
        # Eval can only output -1 to 1 (i.e. do not stop training)
        stop_value=1e9,
        debug_legal_actions=False,
        debug_observation=False,
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: Optional[EasyDict] = None) -> None:
        default_config = self.default_config()
        if cfg is not None:
            default_config.update(cfg)
        self.cfg = default_config

        self.num_planets = self.cfg.num_planets
        self.pass_action = 0
        self.min_send_ships = float(self.cfg.min_send_ships)
        self.send_ratio = float(self.cfg.send_ratio)
        self.tick_seconds = float(self.cfg.tick_seconds)
        self.fleet_speed = float(self.cfg.fleet_speed)
        self.max_episode_steps = int(self.cfg.max_episode_steps)
        self.timeout_reward_scale = float(self.cfg.get('timeout_reward_scale', 1.0))
        self.map_seed = self.cfg.get('map_seed', None)

        self.grid_square_size = float(self.cfg.grid_square_size)
        self.grid_max_x = float(self.cfg.grid_max_x)
        self.grid_max_y = float(self.cfg.grid_max_y)
        self.grid_width = int(math.ceil((2 * self.grid_max_x) / self.grid_square_size))
        self.grid_height = int(math.ceil((2 * self.grid_max_y) / self.grid_square_size))
        self.grid_cell_count = self.grid_width * self.grid_height
        self.pair_action_count = self.grid_cell_count * self.grid_cell_count
        self.enable_fleet_redirection = bool(self.cfg.get('enable_fleet_redirection', False))
        self.use_cpp_sim = bool(self.cfg.get('use_cpp_sim', True))
        self.physics_substeps = max(1, int(self.cfg.get('physics_substeps', DEFAULT_PHYSICS_SUBSTEPS)))
        self.total_num_actions = (
            2 * self.pair_action_count + 1 if self.enable_fleet_redirection else self.pair_action_count + 1
        )

        self.home_distance_fraction = float(self.cfg.home_distance_fraction)

        self.neutral_min_cost = float(self.cfg.neutral_min_cost)
        self.neutral_max_cost = float(self.cfg.neutral_max_cost)
        self.neutral_min_production = float(self.cfg.neutral_min_production)
        self.neutral_max_production = float(self.cfg.neutral_max_production)

        self.fleet_top_k = int(self.cfg.fleet_top_k)
        self.max_expected_ships = float(self.cfg.max_expected_ships)
        self.max_expected_production = float(self.cfg.max_expected_production)
        self.max_fleet_radius = float(self.cfg.max_fleet_radius)
        self.max_expected_eta = float(self.cfg.max_expected_eta)

        self.fleet_slot_count_per_side = self.fleet_top_k + 1
        self.fleet_channel_count = 2 * self.fleet_slot_count_per_side * self.FLEET_FEATURE_COUNT
        self.landing_schedule_channel_count = len(self.LANDING_BUCKETS) * self.LANDING_FEATURE_COUNT_PER_BUCKET
        self.observation_channel_count = (
                self.PLANET_CHANNEL_COUNT + self.fleet_channel_count + self.landing_schedule_channel_count
        )

        self.channel_last = self.cfg.channel_last
        self.scale = self.cfg.scale
        self.battle_mode = self.cfg.battle_mode
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']

        self.players = [self.PLAYER_1, self.PLAYER_2]
        self._current_player = self.PLAYER_1
        self._step_count = 0
        self._next_fleet_id = self.num_planets

        self._action_space = spaces.Discrete(self.total_num_actions)
        self._reward_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self._observation_shape = (self.observation_channel_count, self.grid_height, self.grid_width)
        self._observation_space = spaces.Dict(
            {
                'observation': spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=self._observation_shape,
                    dtype=np.float32,
                ),
                'action_mask': spaces.Box(low=0, high=1, shape=(self.total_num_actions,), dtype=np.int8),
                'current_player_index': spaces.Discrete(2),
                'to_play': spaces.Box(low=-1, high=2, shape=(), dtype=np.int8),
            }
        )

        # Set the parameters related to replay rendering.
        self.screen_scaling = self.cfg.replay_screen_scaling
        # options = {None, 'state_realtime_mode', 'image_realtime_mode', 'image_savefile_mode'}
        self.render_mode = self.cfg.render_mode
        self.replay_name_suffix = self.cfg.replay_name_suffix
        self.replay_path = self.cfg.replay_path
        self.replay_format = self.cfg.replay_format
        self.screen = None
        self.frames: List[np.ndarray] = []

        self.prob_random_agent = self.cfg.prob_random_agent
        self.prob_expert_agent = self.cfg.prob_expert_agent
        self.prob_random_action_in_bot = self.cfg.prob_random_action_in_bot
        self.prob_pass_action_in_bot = self.cfg.prob_pass_action_in_bot
        self.bot_action_type = self.cfg.bot_action_type
        self.bot = build_galcon_bot(self, self.bot_action_type)

        self.debug_legal_actions = self.cfg.debug_legal_actions
        self.debug_observation = self.cfg.debug_observation

        self.planets: List[Planet] = []
        self.fleets: List[Fleet] = []
        self._current_map_seed: Optional[int] = None
        self._cpp_sim: Optional[EnvCppSimHandle] = None
        self._cpp_gen: int = 0
        self._cpp_units_materialized: bool = False
        self._cpp_cfg_sig: Optional[Tuple[float, float, int]] = None
        self._cpp_planets_sorted: List[Planet] = []
        self._cpp_pid_to_idx: Dict[int, int] = {}
        self._cpp_p_snap: List[Tuple[Any, ...]] = []
        self._cpp_f_snap: List[Tuple[Any, ...]] = []
        self._init_cpp_bucket_buffers()
        self.reset()

    def _init_cpp_bucket_buffers(self) -> None:
        self._cpp_buckets_ref = self.LANDING_BUCKETS
        n_b = len(self.LANDING_BUCKETS)
        self._cpp_n_buckets = n_b
        self._cpp_bucket_lowers = (ctypes.c_double * max(1, n_b))()
        self._cpp_bucket_uppers = (ctypes.c_double * max(1, n_b))()
        for i, (lo, hi) in enumerate(self.LANDING_BUCKETS):
            self._cpp_bucket_lowers[i] = float(lo)
            self._cpp_bucket_uppers[i] = float(hi)
        self._ptr_cpp_bucket_lowers = ctypes.addressof(self._cpp_bucket_lowers)
        self._ptr_cpp_bucket_uppers = ctypes.addressof(self._cpp_bucket_uppers)

    def reset(self, start_player_index: int = 0, replay_name_suffix: Optional[str] = None) -> dict:
        self._step_count = 0
        self.players = [self.PLAYER_1, self.PLAYER_2]
        self._current_player = self.players[start_player_index]
        self._next_fleet_id = self.num_planets
        self.frames = []
        if replay_name_suffix is not None:
            self.replay_name_suffix = replay_name_suffix

        episode_map_seed = self._resolve_map_seed()
        self._current_map_seed = episode_map_seed

        # Also resets planets and fleets
        self._generate_map(map_seed=episode_map_seed)
        self._settle_planets(map_seed=episode_map_seed)
        if self.use_cpp_sim and HAS_FAST_SIM_CPP:
            self._init_or_reset_cpp_sim()
        elif self._cpp_sim is not None:
            self._cpp_sim.close()
            self._cpp_sim = None
            self._cpp_p_snap = []
            self._cpp_f_snap = []
        if hasattr(self.bot, 'reset'):
            self.bot.reset()

        self._init_behavioral_metrics()
        return self.observe()

    def _init_map_rng(self, seed: int, dynamic_seed: bool) -> np.random.RandomState:
        map_rng_seed = int(seed) if dynamic_seed else (int(seed) + self.EVAL_MAP_RNG_SEED_OFFSET) % (2**31 - 1)
        return np.random.RandomState(map_rng_seed)

    def _resolve_map_seed(self) -> int:
        if self.map_seed is not None:
            return int(self.map_seed)
        if hasattr(self, '_seed'):
            if not hasattr(self, '_map_rng') or self._map_rng is None:
                self._map_rng = self._init_map_rng(self._seed, getattr(self, '_dynamic_seed', True))
            return int(self._map_rng.randint(0, 2**31 - 1))
        return int(np.random.RandomState().randint(0, 2**31 - 1))

    def _generate_map(self, map_seed: Optional[int] = None) -> None:
        """
        Generate a seeded mirrored Galcon map.

        For num_planets=8:
        - 2 home planets
        - 6 neutral planets as 3 mirrored pairs
        """
        if self.num_planets < 2:
            raise ValueError('GalconEnv requires at least 2 planets.')

        if self.num_planets % 2 != 0:
            raise ValueError('GalconEnv currently expects an even number of neutral planets.')

        if map_seed is None:
            map_seed = self._resolve_map_seed()
        rng = np.random.RandomState(map_seed)

        # spawn angle for the homes
        a = rng.random_sample() * 2 * math.pi
        # distance from the center of the map for the homes
        home_x = self.grid_max_x * self.home_distance_fraction * math.cos(a)
        home_y = self.grid_max_y * self.home_distance_fraction * math.sin(a)

        planets = [
            Planet(
                id=0,
                owner=self.PLAYER_1,
                ships=100.0,
                x=home_x,
                y=home_y,
                production=100.0,
                radius=24.0,
                neutral=False,
            ),
            Planet(
                id=1,
                owner=self.PLAYER_2,
                ships=100.0,
                x=-home_x,
                y=-home_y,
                production=100.0,
                radius=24.0,
                neutral=False,
            ),
        ]

        next_planet_id = 2
        neutral_pairs = (self.num_planets - 2) // 2
        for _ in range(neutral_pairs):
            x = -self.grid_max_x + (2 * self.grid_max_x) * rng.random_sample()
            y = -self.grid_max_y + (2 * self.grid_max_y) * rng.random_sample()
            neutral_ships = self.neutral_min_cost + (self.neutral_max_cost - self.neutral_min_cost) * rng.random_sample()
            production = self.neutral_min_production + (self.neutral_max_production - self.neutral_min_production) * rng.random_sample()
            radius = self._radius_from_production(production)

            planets.append(
                Planet(
                    id=next_planet_id,
                    owner=self.PLAYER_NEUTRAL,
                    ships=float(neutral_ships),
                    x=float(x),
                    y=float(y),
                    production=float(production),
                    radius=float(radius),
                    neutral=True,
                )
            )
            next_planet_id += 1

            planets.append(
                Planet(
                    id=next_planet_id,
                    owner=self.PLAYER_NEUTRAL,
                    ships=float(neutral_ships),
                    x=float(-x),
                    y=float(-y),
                    production=float(production),
                    radius=float(radius),
                    neutral=True,
                )
            )
            next_planet_id += 1

        self.planets = planets
        self.fleets = []

    def _settle_planets(self, map_seed: Optional[int] = None):
        # Settle planets to prevent overlap
        min_gap = 2 * self.SHIP_RADIUS + self.PLANETS_SETTLE_DELTA

        if map_seed is None:
            map_seed = self._resolve_map_seed()
        rng = np.random.RandomState(map_seed)

        # Clamp all planets within map boundaries before checking overlaps
        for i in range(0, len(self.planets), 2):
            p_i = self.planets[i]
            margin_x = p_i.radius + self.SHIP_RADIUS
            margin_y = p_i.radius + self.SHIP_RADIUS
            p_i.x = max(-self.grid_max_x + margin_x, min(self.grid_max_x - margin_x, p_i.x))
            p_i.y = max(-self.grid_max_y + margin_y, min(self.grid_max_y - margin_y, p_i.y))
            p_odd = self.planets[i + 1]
            p_odd.x = -p_i.x
            p_odd.y = -p_i.y

        # Run relaxation simulation to settle the positions of neutral planets
        for attempt_num in range(self.MAX_SETTLE_ATTEMPTS):
            moved = False
            displacements = {}
            for i in range(0, len(self.planets), 2):  # Only adjust the first of each neutral pair (even IDs)
                p_i = self.planets[i]
                dx_total = 0.0
                dy_total = 0.0

                for j in range(len(self.planets)):
                    if j == i:
                        continue
                    p_j = self.planets[j]

                    dist = math.hypot(p_i.x - p_j.x, p_i.y - p_j.y)
                    min_dist = p_i.radius + p_j.radius + min_gap

                    # If the planets are at least the minimum distance apart + half of the "tolerance", it's close enough. 
                    if dist < min_dist:
                        overlap = min_dist - dist
                        if dist < 1e-4:
                            # Perturb in a random direction if they are exactly at the same position
                            angle = rng.uniform(0, 2 * math.pi)
                            ux, uy = math.cos(angle), math.sin(angle)
                        else:
                            ux = (p_i.x - p_j.x) / dist
                            uy = (p_i.y - p_j.y) / dist

                        step = 0.5 * overlap + 0.01 # add a tiny bit more to deal with floating point numbers
                        dx_total += ux * step
                        dy_total += uy * step
                        moved = True

                displacements[i] = (dx_total, dy_total)

            if not moved:
                break

            for i in range(0, len(self.planets), 2):
                p_i = self.planets[i]
                dx, dy = displacements[i]
                p_i.x += dx
                p_i.y += dy

                # Keep within map boundaries
                margin_x = p_i.radius + self.SHIP_RADIUS
                margin_y = p_i.radius + self.SHIP_RADIUS
                p_i.x = max(-self.grid_max_x + margin_x, min(self.grid_max_x - margin_x, p_i.x))
                p_i.y = max(-self.grid_max_y + margin_y, min(self.grid_max_y - margin_y, p_i.y))

                # Update the mirrored partner
                p_odd = self.planets[i + 1]
                p_odd.x = -p_i.x
                p_odd.y = -p_i.y

            # Print warning if the planets are not settled on the final attempt
            if attempt_num == self.MAX_SETTLE_ATTEMPTS - 1:
                # find the minimum distance between any two planets
                min_dist = float('inf')
                for i in range(len(self.planets)):
                    for j in range(i + 1, len(self.planets)):
                        dist = math.hypot(self.planets[i].x - self.planets[j].x, self.planets[i].y - self.planets[j].y)
                        min_dist = min(min_dist, dist)
                
                # warning
                if min_dist < 1:
                    print(f'WARNING: Planets are too close. Min distance: {min_dist}')
            


    @staticmethod
    def _radius_from_production(production: float) -> float:
        return planet_radius_from_production(production)

    def step(self, action: int) -> BaseEnvTimestep:
        if self.battle_mode == 'self_play_mode':
            return self._player_step(action, flag='agent')
        elif self.battle_mode in ['play_with_bot_mode', 'eval_mode']:
            timestep_player1 = self._player_step(action, flag='agent', need_obs=False)
            if timestep_player1.done:
                # # Player 1 has finished, so player 2 will play next
                # timestep_player1.obs['to_play'] = self.PLAYER_2
                return timestep_player1

            bot_action = self.bot_action()
            timestep_player2 = self._player_step(bot_action, flag='bot', need_obs=True)
            # In play_with_bot_mode / eval_mode, observe() emits to_play=-1 because each env.step()
            # advances a full (agent + bot) turn and all transitions stay in Player 1's perspective.
            timestep_player2.obs['to_play'] = -1

            # Invert reward from Player 2's (bot's) perspective to Player 1's (agent's) perspective.
            neg_reward = -timestep_player2.reward
            if timestep_player2.done:
                timestep_player2.info['eval_episode_return'] = -timestep_player2.info['eval_episode_return']
            timestep_player2 = timestep_player2._replace(reward=neg_reward)

            return timestep_player2

        else:
            raise ValueError(f'Unsupported battle_mode: {self.battle_mode}')

    def _player_step(self, action: int, flag: str, need_obs: bool = True) -> BaseEnvTimestep:
        if not hasattr(self, '_bm_early_steps'):
            self._init_behavioral_metrics()
        self._record_action_turn_metrics(self._current_player, action)
        if 0 < action < self.total_num_actions:
            self._apply_action(action)

        self._advance_one_tick()
        self._step_count += 1

        # Single-pass planet & fleet statistics for elimination, timeout, and state info.
        p1_id = self.PLAYER_1
        p2_id = self.PLAYER_2
        p1_alive = False
        p2_alive = False
        p1_prod = 0.0
        p2_prod = 0.0
        p1_p_ships = 0.0
        p2_p_ships = 0.0
        for p in self.planets:
            own = p.owner
            if own == p1_id:
                p1_alive = True
                p1_prod += p.production
                p1_p_ships += p.ships
            elif own == p2_id:
                p2_alive = True
                p2_prod += p.production
                p2_p_ships += p.ships

        p1_f_ships = 0.0
        p2_f_ships = 0.0
        for fl in self.fleets:
            own = fl.owner
            if own == p1_id:
                p1_alive = True
                p1_f_ships += fl.ships
            elif own == p2_id:
                p2_alive = True
                p2_f_ships += fl.ships

        p1_prod_r = float(round(p1_prod, 2))
        p2_prod_r = float(round(p2_prod, 2))
        p1_ships_r = float(round(p1_p_ships + p1_f_ships, 2))
        p2_ships_r = float(round(p2_p_ships + p2_f_ships, 2))

        if p1_alive and not p2_alive:
            winner_by_elimination = p1_id
        elif p2_alive and not p1_alive:
            winner_by_elimination = p2_id
        else:
            winner_by_elimination = -1

        if winner_by_elimination != -1:
            done = True
            winner = winner_by_elimination
        elif self._step_count >= self.max_episode_steps:
            done = True
            if p1_prod_r > p2_prod_r:
                winner = p1_id
            elif p2_prod_r > p1_prod_r:
                winner = p2_id
            elif p1_ships_r > p2_ships_r:
                winner = p1_id
            elif p2_ships_r > p1_ships_r:
                winner = p2_id
            else:
                winner = -1
        else:
            done = False
            winner = -1

        if done:
            ended_by_timeout = (winner_by_elimination == -1 and self._step_count >= self.max_episode_steps)
            reward_mag = self.timeout_reward_scale if ended_by_timeout else 1.0
            if winner == -1:
                reward = np.array(0, dtype=np.float32)
                eval_return = np.array(0, dtype=np.float32)
            elif winner == self._current_player:
                reward = np.array(reward_mag, dtype=np.float32)
                eval_return = np.array(1, dtype=np.float32)
            else:
                reward = np.array(-reward_mag, dtype=np.float32)
                eval_return = np.array(-1, dtype=np.float32)
        else:
            ended_by_timeout = False
            reward = np.array(0, dtype=np.float32)
            eval_return = None

        self._update_tick_behavioral_metrics(
            p1_prod=p1_prod,
            p2_prod=p2_prod,
            p1_p_ships=p1_p_ships,
            p2_p_ships=p2_p_ships,
            p1_f_ships=p1_f_ships,
            p2_f_ships=p2_f_ships,
            done=done,
        )

        info = {
            'winner': winner if done else -1,
            'step_count': self._step_count,
            'simulated_seconds': self._step_count * self.tick_seconds,
            'production_player_1': p1_prod_r,
            'production_player_2': p2_prod_r,
            'ships_player_1': p1_ships_r,
            'ships_player_2': p2_ships_r,
        }

        self._current_player = self.next_player

        if need_obs or done or self.debug_legal_actions or self.debug_observation:
            obs = self.observe()
        else:
            obs = None

        # Render the new step.
        if self.render_mode is not None:
            self.render(self.render_mode)
        if done:
            info['eval_episode_return'] = eval_return
            metrics_by_player, summary_metrics = self._finalize_behavioral_metrics(
                winner=winner, ended_by_timeout=ended_by_timeout
            )
            info['galcon_metrics_by_player'] = metrics_by_player
            info['galcon_metrics'] = summary_metrics
            if self.render_mode == 'image_savefile_mode':
                self.save_render_output(replay_name_suffix=self.replay_name_suffix, replay_path=self.replay_path,
                                        format=self.replay_format)

        return BaseEnvTimestep(obs, reward, done, info)

    def _fleets_at_cell(self, grid_x: int, grid_y: int, owner: Optional[int] = None) -> List[Fleet]:
        matched = []
        grid_max_x = self.grid_max_x
        grid_max_y = self.grid_max_y
        sq = self.grid_square_size
        w_m1 = self.grid_width - 1
        h_m1 = self.grid_height - 1
        floor = math.floor
        for fleet in self.fleets:
            if owner is not None and fleet.owner != owner:
                continue
            gx = floor((fleet.x + grid_max_x) / sq)
            if gx < 0:
                gx = 0
            elif gx > w_m1:
                gx = w_m1
            if gx != grid_x:
                continue
            gy = floor((fleet.y + grid_max_y) / sq)
            if gy < 0:
                gy = 0
            elif gy > h_m1:
                gy = h_m1
            if gy == grid_y:
                matched.append(fleet)
        return matched

    def _apply_action(self, action: int) -> None:
        action_type, source_x, source_y, target_x, target_y = self.decode_action_with_type(action)
        if action_type == 'pass':
            return

        target = self._planet_at_cell(target_x, target_y)
        if target is None:
            return

        if action_type == 'redirect':
            if not self.enable_fleet_redirection:
                return
            cur_player = self._current_player
            target_id = target.id
            grid_max_x = self.grid_max_x
            grid_max_y = self.grid_max_y
            sq = self.grid_square_size
            w_m1 = self.grid_width - 1
            h_m1 = self.grid_height - 1
            floor = math.floor
            cpp_sim = self._cpp_sim
            dst_idx = self._cpp_pid_to_idx.get(target_id, -1) if cpp_sim is not None else -1
            f_snap = self._cpp_f_snap
            n_snap = len(f_snap)
            redirected_ships = 0.0
            redirected_count = 0
            for idx, fleet in enumerate(self.fleets):
                if fleet.owner != cur_player:
                    continue
                gx = floor((fleet.x + grid_max_x) / sq)
                if gx < 0:
                    gx = 0
                elif gx > w_m1:
                    gx = w_m1
                if gx != source_x:
                    continue
                gy = floor((fleet.y + grid_max_y) / sq)
                if gy < 0:
                    gy = 0
                elif gy > h_m1:
                    gy = h_m1
                if gy != source_y:
                    continue
                fleet.target = target_id
                redirected_count += 1
                redirected_ships += float(fleet.ships)
                if cpp_sim is not None and fleet._cpp_env is self and dst_idx >= 0:
                    if cpp_sim.set_fleet_target(fleet.id, dst_idx):
                        if idx < n_snap and f_snap[idx][0] is fleet:
                            s = f_snap[idx]
                            f_snap[idx] = (s[0], s[1], s[2], s[3], s[4], s[5], target_id, s[7])
            if redirected_count > 0:
                self._record_applied_action_metrics(
                    player=cur_player,
                    target=target,
                    source=None,
                    ships_sent=redirected_ships,
                    is_redirect=True,
                )
            return

        # action_type == 'send': planet launch only (does not redirect flying fleets in this cell)
        source = self._planet_at_cell(source_x, source_y, owner=self._current_player)
        if source is None:
            return
        if source.id == target.id:
            return
        if source.neutral:
            return
        if source.owner != self._current_player:
            return
        if not self._can_send_from_planet(source):
            return

        ships_to_send = self._ships_to_send(source.ships)
        self._record_applied_action_metrics(
            player=self._current_player,
            target=target,
            source=source,
            ships_sent=float(ships_to_send),
            is_redirect=False,
        )
        if self.use_cpp_sim:
            self._spawn_cpp_fleet(source, target, ships_to_send)
            return

        source.ships -= ships_to_send
        source.ships = max(0.0, source.ships)

        spawn_dx, spawn_dy = self._vector_components(source.x, source.y, target.x, target.y, source.radius)

        self.fleets.append(
            Fleet(
                id=self._next_fleet_id,
                owner=source.owner,
                ships=float(ships_to_send),
                x=float(source.x + spawn_dx),
                y=float(source.y + spawn_dy),
                source=source.id,
                target=target.id,
                radius=self.DUMMY_FLEET_RADIUS,
            )
        )
        self._next_fleet_id += 1

    def _init_behavioral_metrics(self) -> None:
        """Initialize lightweight per-episode behavioral counters for both players."""
        tick_s = max(float(self.tick_seconds), 1e-6)
        max_sec = max(1, int(self.max_episode_steps)) * tick_s
        early_t_sec = min(10.0, 0.25 * max_sec)
        early_payback_t_sec = min(15.0, 0.35 * max_sec)
        self._bm_early_steps: int = max(1, int(round(early_t_sec / tick_s)))
        self._bm_early_payback_steps: int = max(self._bm_early_steps, int(round(early_payback_t_sec / tick_s)))
        self._bm_early_snapshot_taken: bool = False

        p1_home = None
        p2_home = None
        p1_prod = 0.0
        p2_prod = 0.0
        p1_ships = 0.0
        p2_ships = 0.0
        for p in self.planets:
            if p.owner == self.PLAYER_1:
                if p1_home is None:
                    p1_home = p
                p1_prod += float(p.production)
                p1_ships += float(p.ships)
            elif p.owner == self.PLAYER_2:
                if p2_home is None:
                    p2_home = p
                p2_prod += float(p.production)
                p2_ships += float(p.ships)

        self._bm_home_id: Dict[int, Optional[int]] = {
            self.PLAYER_1: p1_home.id if p1_home is not None else None,
            self.PLAYER_2: p2_home.id if p2_home is not None else None,
        }
        self._bm_home_pos: Dict[int, Optional[Tuple[float, float]]] = {
            self.PLAYER_1: (float(p1_home.x), float(p1_home.y)) if p1_home is not None else None,
            self.PLAYER_2: (float(p2_home.x), float(p2_home.y)) if p2_home is not None else None,
        }

        # Initial neutral planet metadata: pid -> (init_prod, init_payback_sec, own_half_owner)
        self._bm_neutral_meta: Dict[int, Tuple[float, float, int]] = {}
        self._bm_first_capturer: Dict[int, int] = {}
        self._bm_first_capture_step: Dict[int, int] = {}
        h1_pos = self._bm_home_pos[self.PLAYER_1]
        h2_pos = self._bm_home_pos[self.PLAYER_2]
        for p in self.planets:
            if p.owner == self.PLAYER_NEUTRAL:
                prod = float(p.production)
                rate = float(self._production_to_ships_per_second(prod))
                payback = (float(p.ships) / rate) if rate > 0.0 else float('inf')
                half_owner = 0
                if h1_pos is not None and h2_pos is not None:
                    d1 = (p.x - h1_pos[0]) ** 2 + (p.y - h1_pos[1]) ** 2
                    d2 = (p.x - h2_pos[0]) ** 2 + (p.y - h2_pos[1]) ** 2
                    if d1 < d2 - 1e-6:
                        half_owner = self.PLAYER_1
                    elif d2 < d1 - 1e-6:
                        half_owner = self.PLAYER_2
                self._bm_neutral_meta[p.id] = (prod, payback, half_owner)
                self._bm_first_capturer[p.id] = 0
                self._bm_first_capture_step[p.id] = -1

        self._bm_early_cheap_left: Dict[int, float] = {self.PLAYER_1: 0.0, self.PLAYER_2: 0.0}
        self._bm_early_home_rush_ships: Dict[int, float] = {self.PLAYER_1: 0.0, self.PLAYER_2: 0.0}
        self._bm_pass_history: Dict[int, List[int]] = {self.PLAYER_1: [], self.PLAYER_2: []}
        self._bm_ahead_turns: Dict[int, int] = {self.PLAYER_1: 0, self.PLAYER_2: 0}
        self._bm_ahead_pass_turns: Dict[int, int] = {self.PLAYER_1: 0, self.PLAYER_2: 0}

        self._bm_non_pass: Dict[int, int] = {self.PLAYER_1: 0, self.PLAYER_2: 0}
        self._bm_target_neutral: Dict[int, int] = {self.PLAYER_1: 0, self.PLAYER_2: 0}
        self._bm_target_enemy: Dict[int, int] = {self.PLAYER_1: 0, self.PLAYER_2: 0}
        self._bm_target_reinforce: Dict[int, int] = {self.PLAYER_1: 0, self.PLAYER_2: 0}
        self._bm_target_defend: Dict[int, int] = {self.PLAYER_1: 0, self.PLAYER_2: 0}
        self._bm_target_forward_relay: Dict[int, int] = {self.PLAYER_1: 0, self.PLAYER_2: 0}
        self._bm_redirects: Dict[int, int] = {self.PLAYER_1: 0, self.PLAYER_2: 0}
        self._bm_planet_sends: Dict[int, int] = {self.PLAYER_1: 0, self.PLAYER_2: 0}
        self._bm_send_frac_sum: Dict[int, float] = {self.PLAYER_1: 0.0, self.PLAYER_2: 0.0}

        self._bm_rear_share_sum: Dict[int, float] = {self.PLAYER_1: 0.0, self.PLAYER_2: 0.0}
        self._bm_rear_share_ticks: Dict[int, int] = {self.PLAYER_1: 0, self.PLAYER_2: 0}
        self._bm_flight_share_sum: Dict[int, float] = {self.PLAYER_1: 0.0, self.PLAYER_2: 0.0}
        self._bm_flight_share_ticks: Dict[int, int] = {self.PLAYER_1: 0, self.PLAYER_2: 0}
        self._bm_p1_prod_share_hist: List[float] = []

        self._bm_prev_prod: Dict[int, float] = {self.PLAYER_1: p1_prod, self.PLAYER_2: p2_prod}
        self._bm_prev_ships: Dict[int, float] = {self.PLAYER_1: p1_ships, self.PLAYER_2: p2_ships}

    def _record_action_turn_metrics(self, player: int, action: int) -> None:
        if player not in (self.PLAYER_1, self.PLAYER_2):
            return
        is_pass = 1 if int(action) == 0 else 0
        self._bm_pass_history[player].append(is_pass)

        opp = self.PLAYER_2 if player == self.PLAYER_1 else self.PLAYER_1
        my_prod = self._bm_prev_prod[player]
        tot_prod = my_prod + self._bm_prev_prod[opp]
        my_ships = self._bm_prev_ships[player]
        tot_ships = my_ships + self._bm_prev_ships[opp]
        is_ahead = (
            (tot_prod > 0.0 and (my_prod / tot_prod) >= 0.60)
            or (tot_ships > 0.0 and (my_ships / tot_ships) >= 0.60)
        )
        if is_ahead:
            self._bm_ahead_turns[player] += 1
            if is_pass:
                self._bm_ahead_pass_turns[player] += 1

    def _record_applied_action_metrics(
        self,
        player: int,
        target: Planet,
        source: Optional[Planet],
        ships_sent: float,
        is_redirect: bool,
    ) -> None:
        if not hasattr(self, '_bm_non_pass') or player not in (self.PLAYER_1, self.PLAYER_2):
            return
        opp = self.PLAYER_2 if player == self.PLAYER_1 else self.PLAYER_1
        self._bm_non_pass[player] += 1
        if is_redirect:
            self._bm_redirects[player] += 1

        if target.owner == self.PLAYER_NEUTRAL:
            self._bm_target_neutral[player] += 1
        elif target.owner == opp:
            self._bm_target_enemy[player] += 1
        elif target.owner == player:
            self._bm_target_reinforce[player] += 1
            # Defend: own target planet has at least one incoming enemy fleet
            is_under_attack = any(
                fl.owner == opp and fl.target == target.id for fl in self.fleets
            )
            if is_under_attack:
                self._bm_target_defend[player] += 1
            elif source is not None:
                opp_home_pos = self._bm_home_pos.get(opp)
                if opp_home_pos is not None:
                    d_src = (source.x - opp_home_pos[0]) ** 2 + (source.y - opp_home_pos[1]) ** 2
                    d_tgt = (target.x - opp_home_pos[0]) ** 2 + (target.y - opp_home_pos[1]) ** 2
                    if d_tgt < d_src - 1e-6:
                        self._bm_target_forward_relay[player] += 1

        if not is_redirect and source is not None:
            self._bm_planet_sends[player] += 1
            if source.ships > 1e-6:
                self._bm_send_frac_sum[player] += min(1.0, max(0.0, float(ships_sent) / float(source.ships)))
            if (
                self._step_count < self._bm_early_steps
                and self._bm_home_id.get(opp) is not None
                and target.id == self._bm_home_id[opp]
            ):
                self._bm_early_home_rush_ships[player] += float(ships_sent)

    def _update_tick_behavioral_metrics(
        self,
        p1_prod: float,
        p2_prod: float,
        p1_p_ships: float,
        p2_p_ships: float,
        p1_f_ships: float,
        p2_f_ships: float,
        done: bool,
    ) -> None:
        # 1. Track first-capture ownership of initially neutral planets
        first_cap = self._bm_first_capturer
        first_step = self._bm_first_capture_step
        p1_planets = []
        p2_planets = []
        for p in self.planets:
            own = p.owner
            if own == self.PLAYER_1:
                p1_planets.append(p)
                if p.id in first_cap and first_cap[p.id] == 0:
                    first_cap[p.id] = self.PLAYER_1
                    first_step[p.id] = self._step_count
            elif own == self.PLAYER_2:
                p2_planets.append(p)
                if p.id in first_cap and first_cap[p.id] == 0:
                    first_cap[p.id] = self.PLAYER_2
                    first_step[p.id] = self._step_count

        # 2. Snapshot own-half cheap neutrals (payback <= 20s) still uncaptured at early checkpoint
        if (not self._bm_early_snapshot_taken) and (self._step_count >= self._bm_early_steps or done):
            self._bm_early_snapshot_taken = True
            left_p1 = 0.0
            left_p2 = 0.0
            for pid, (prod, payback, half_owner) in self._bm_neutral_meta.items():
                if first_cap[pid] == 0 and payback <= 20.0:
                    if half_owner == self.PLAYER_1:
                        left_p1 += prod
                    elif half_owner == self.PLAYER_2:
                        left_p2 += prod
            self._bm_early_cheap_left[self.PLAYER_1] = left_p1
            self._bm_early_cheap_left[self.PLAYER_2] = left_p2

        # 3. Ships-in-flight share
        p1_tot = p1_p_ships + p1_f_ships
        if p1_tot > 1e-6:
            self._bm_flight_share_sum[self.PLAYER_1] += p1_f_ships / p1_tot
            self._bm_flight_share_ticks[self.PLAYER_1] += 1
        p2_tot = p2_p_ships + p2_f_ships
        if p2_tot > 1e-6:
            self._bm_flight_share_sum[self.PLAYER_2] += p2_f_ships / p2_tot
            self._bm_flight_share_ticks[self.PLAYER_2] += 1

        # 4. Rear-planet ship share (landed ships on planets strictly farther from nearest enemy than median)
        if p1_planets and p2_planets:
            if len(p1_planets) >= 2 and p1_p_ships > 1e-6:
                dists = [
                    min((u.x - v.x) ** 2 + (u.y - v.y) ** 2 for v in p2_planets)
                    for u in p1_planets
                ]
                s_dists = sorted(dists)
                n = len(s_dists)
                med = 0.5 * (s_dists[n // 2 - 1] + s_dists[n // 2]) if n % 2 == 0 else s_dists[n // 2]
                rear_ships = sum(u.ships for u, d in zip(p1_planets, dists) if d > med + 1e-6)
                self._bm_rear_share_sum[self.PLAYER_1] += rear_ships / p1_p_ships
                self._bm_rear_share_ticks[self.PLAYER_1] += 1
            if len(p2_planets) >= 2 and p2_p_ships > 1e-6:
                dists = [
                    min((u.x - v.x) ** 2 + (u.y - v.y) ** 2 for v in p1_planets)
                    for u in p2_planets
                ]
                s_dists = sorted(dists)
                n = len(s_dists)
                med = 0.5 * (s_dists[n // 2 - 1] + s_dists[n // 2]) if n % 2 == 0 else s_dists[n // 2]
                rear_ships = sum(u.ships for u, d in zip(p2_planets, dists) if d > med + 1e-6)
                self._bm_rear_share_sum[self.PLAYER_2] += rear_ships / p2_p_ships
                self._bm_rear_share_ticks[self.PLAYER_2] += 1

        # 5. Production share history & cached state for next turn's pass_rate_when_ahead check
        tot_prod = p1_prod + p2_prod
        self._bm_p1_prod_share_hist.append((p1_prod / tot_prod) if tot_prod > 1e-6 else 0.5)
        self._bm_prev_prod[self.PLAYER_1] = p1_prod
        self._bm_prev_prod[self.PLAYER_2] = p2_prod
        self._bm_prev_ships[self.PLAYER_1] = p1_tot
        self._bm_prev_ships[self.PLAYER_2] = p2_tot

    def _finalize_behavioral_metrics(
        self, winner: int, ended_by_timeout: bool
    ) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
        """Compute clean finite float behavioral metric dicts on episode end."""
        final_step = max(1, int(self._step_count))
        half_step = final_step // 2
        hist = self._bm_p1_prod_share_hist
        idx_10s = min(len(hist) - 1, max(0, self._bm_early_steps - 1)) if hist else 0
        idx_half = (len(hist) // 2) if hist else 0
        p1_share_10s = float(hist[idx_10s]) if hist else 0.5
        p1_share_half = float(hist[idx_half]) if hist else 0.5

        by_player: Dict[int, Dict[str, float]] = {}
        for p in (self.PLAYER_1, self.PLAYER_2):
            opp = self.PLAYER_2 if p == self.PLAYER_1 else self.PLAYER_1
            m: Dict[str, float] = {}

            # A. Opening & Expansion Economy
            early_prod = 0.0
            total_cap_prod = 0.0
            late_cap_prod = 0.0
            stolen_prod = 0.0
            early_paybacks: List[float] = []
            for pid, (prod, payback, half_owner) in self._bm_neutral_meta.items():
                cap_by = self._bm_first_capturer[pid]
                cap_step = self._bm_first_capture_step[pid]
                if cap_by == p:
                    total_cap_prod += prod
                    if cap_step <= self._bm_early_steps:
                        early_prod += prod
                    if cap_step <= self._bm_early_payback_steps and math.isfinite(payback):
                        early_paybacks.append(payback)
                    if cap_step > half_step:
                        late_cap_prod += prod
                elif cap_by == opp and half_owner == p:
                    stolen_prod += prod

            m['early_neutral_prod_taken'] = float(early_prod)
            m['early_cheap_neutrals_left'] = float(self._bm_early_cheap_left[p])
            if early_paybacks:
                m['mean_capture_payback_sec'] = float(sum(early_paybacks) / len(early_paybacks))
            m['early_home_rush_ships'] = float(self._bm_early_home_rush_ships[p])
            m['stolen_own_half_prod'] = float(stolen_prod)
            if total_cap_prod > 1e-6:
                m['late_neutral_capture_share'] = float(late_cap_prod / total_cap_prod)

            # B. Action Mix & Tempo
            ph = self._bm_pass_history[p]
            n_turns = len(ph)
            if n_turns > 0:
                i1 = max(1, int(round(0.25 * n_turns)))
                i2 = max(i1, int(round(0.65 * n_turns)))
                seg_open = ph[:i1]
                seg_mid = ph[i1:i2]
                seg_late = ph[i2:]
                if seg_open:
                    m['pass_rate_opening'] = float(sum(seg_open) / len(seg_open))
                if seg_mid:
                    m['pass_rate_mid'] = float(sum(seg_mid) / len(seg_mid))
                if seg_late:
                    m['pass_rate_late'] = float(sum(seg_late) / len(seg_late))

            if self._bm_ahead_turns[p] > 0:
                m['pass_rate_when_ahead'] = float(self._bm_ahead_pass_turns[p] / self._bm_ahead_turns[p])

            np_cnt = self._bm_non_pass[p]
            if np_cnt > 0:
                m['target_neutral_share'] = float(self._bm_target_neutral[p] / np_cnt)
                m['target_enemy_share'] = float(self._bm_target_enemy[p] / np_cnt)
                m['target_reinforce_share'] = float(self._bm_target_reinforce[p] / np_cnt)
                m['target_defend_share'] = float(self._bm_target_defend[p] / np_cnt)
                m['target_forward_relay_share'] = float(self._bm_target_forward_relay[p] / np_cnt)
                if self.enable_fleet_redirection:
                    m['redirect_share'] = float(self._bm_redirects[p] / np_cnt)

            if self._bm_planet_sends[p] > 0:
                m['send_fraction_mean'] = float(self._bm_send_frac_sum[p] / self._bm_planet_sends[p])

            # C. Force Allocation & Positional Efficiency
            if self._bm_rear_share_ticks[p] > 0:
                m['rear_planet_ship_share'] = float(
                    self._bm_rear_share_sum[p] / self._bm_rear_share_ticks[p]
                )
            if self._bm_flight_share_ticks[p] > 0:
                m['ships_in_flight_share'] = float(
                    self._bm_flight_share_sum[p] / self._bm_flight_share_ticks[p]
                )
            m['prod_share_at_10s'] = p1_share_10s if p == self.PLAYER_1 else (1.0 - p1_share_10s)
            m['prod_share_at_half'] = p1_share_half if p == self.PLAYER_1 else (1.0 - p1_share_half)

            # D. Game Decisiveness
            m['timeout_rate'] = 1.0 if ended_by_timeout else 0.0
            if winner == p:
                m['avg_win_steps'] = float(final_step)
            elif winner != -1:
                m['avg_loss_steps'] = float(final_step)

            by_player[p] = m

        if self.battle_mode in ('play_with_bot_mode', 'eval_mode'):
            summary = dict(by_player[self.PLAYER_1])
        else:
            # Self-play mode: average symmetric metrics across P1 and P2, plus p1_win_rate
            summary = {}
            all_keys = set(by_player[self.PLAYER_1].keys()) | set(by_player[self.PLAYER_2].keys())
            for k in sorted(all_keys):
                vals = [
                    by_player[p][k]
                    for p in (self.PLAYER_1, self.PLAYER_2)
                    if k in by_player[p]
                ]
                if vals:
                    summary[k] = float(sum(vals) / len(vals))
            if winner == self.PLAYER_1:
                summary['p1_win_rate'] = 1.0
            elif winner == self.PLAYER_2:
                summary['p1_win_rate'] = 0.0
            else:
                summary['p1_win_rate'] = 0.5

        return by_player, summary

    def _init_or_reset_cpp_sim(self) -> None:
        planets = self.planets
        if not planets or len(planets) > 64:
            if self._cpp_sim is not None:
                self._cpp_sim.close()
                self._cpp_sim = None
            self._cpp_p_snap = []
            self._cpp_f_snap = []
            return

        if all(p.id == i for i, p in enumerate(planets)):
            planets_sorted = list(planets)
        else:
            planets_sorted = sorted(planets, key=lambda p: p.id)

        if self._cpp_sim is None:
            self._cpp_sim = EnvCppSimHandle.create(
                planets_sorted,
                width=2.0 * self.grid_max_x,
                height=2.0 * self.grid_max_y,
                fleet_speed=self.fleet_speed,
                tick_dt=self.tick_seconds,
                physics_substeps=self.physics_substeps,
                allow_redirect=self.enable_fleet_redirection,
            )
        else:
            ok = self._cpp_sim.update_planets(
                planets_sorted,
                fleet_speed=self.fleet_speed,
                tick_dt=self.tick_seconds,
                physics_substeps=self.physics_substeps,
                clear_fleets=True,
            )
            if not ok:
                self._cpp_sim.close()
                self._cpp_sim = None

        if self._cpp_sim is not None:
            self._cpp_planets_sorted = planets_sorted
            self._cpp_pid_to_idx = {p.id: i for i, p in enumerate(planets_sorted)}
            self._cpp_cfg_sig = (self.tick_seconds, self.fleet_speed, self.physics_substeps)
            self._cpp_p_snap = [
                (p, p.id, p.owner, p.ships, p.x, p.y, p.production, p.radius) for p in planets
            ]
            self._cpp_f_snap = []
            self._cpp_units_materialized = False
            self._cpp_gen += 1
        else:
            self._cpp_p_snap = []
            self._cpp_f_snap = []

    def _materialize_fleet_units(self, fleet: Fleet) -> None:
        if self._cpp_sim is None or fleet._cpp_gen != self._cpp_gen:
            fleet._cpp_env = None
            return
        units, headings = self._cpp_sim.read_fleet_units(fleet.id)
        if units is not None:
            fleet._units = units
            fleet._unit_headings = headings
            self._cpp_units_materialized = True
        else:
            fleet._cpp_env = None

    def _materialize_all_cpp_fleets(self) -> None:
        if self._cpp_sim is None:
            return
        for fl in self.fleets:
            if fl._cpp_env is self and (fl._units is None or fl._unit_headings is None):
                self._materialize_fleet_units(fl)
            fl._cpp_env = None

    def _sync_cpp_sim(self) -> bool:
        if not HAS_FAST_SIM_CPP:
            return False
        planets = self.planets
        n_planets = len(planets)
        if n_planets <= 0 or n_planets > 64:
            self._materialize_all_cpp_fleets()
            if self._cpp_sim is not None:
                self._cpp_sim.close()
                self._cpp_sim = None
            return False

        cfg_sig = (self.tick_seconds, self.fleet_speed, self.physics_substeps)
        p_snap = self._cpp_p_snap
        planets_match = (
            self._cpp_sim is not None
            and self._cpp_cfg_sig == cfg_sig
            and n_planets == len(p_snap)
        )
        if planets_match:
            for p, s in zip(planets, p_snap):
                if (
                    p is not s[0]
                    or p.id != s[1]
                    or p.owner != s[2]
                    or p.ships != s[3]
                    or p.x != s[4]
                    or p.y != s[5]
                    or p.production != s[6]
                    or p.radius != s[7]
                ):
                    planets_match = False
                    break

        fleets = self.fleets
        f_snap = self._cpp_f_snap
        fleets_match = (
            self._cpp_sim is not None
            and not self._cpp_units_materialized
            and len(fleets) == len(f_snap)
        )
        if fleets_match:
            for fl, s in zip(fleets, f_snap):
                if (
                    fl is not s[0]
                    or fl._cpp_env is not self
                    or fl.id != s[1]
                    or fl.owner != s[2]
                    or fl.ships != s[3]
                    or fl.x != s[4]
                    or fl.y != s[5]
                    or fl.target != s[6]
                    or fl.num_units != s[7]
                ):
                    fleets_match = False
                    break

        if planets_match and fleets_match:
            return True

        # Before overwriting C++ fleet buffers, preserve any still-lazy units on fleets
        # that remain in self.fleets while another fleet was added/mutated.
        if self._cpp_sim is not None and not fleets_match:
            for fl in fleets:
                if fl._cpp_env is self and (fl._units is None or fl._unit_headings is None):
                    self._materialize_fleet_units(fl)

        if all(p.id == i for i, p in enumerate(planets)):
            planets_sorted = list(planets)
        else:
            planets_sorted = sorted(planets, key=lambda p: p.id)
        pid_to_idx = {p.id: i for i, p in enumerate(planets_sorted)}

        if self._cpp_sim is None:
            self._cpp_sim = EnvCppSimHandle.create(
                planets_sorted,
                width=2.0 * self.grid_max_x,
                height=2.0 * self.grid_max_y,
                fleet_speed=self.fleet_speed,
                tick_dt=self.tick_seconds,
                physics_substeps=self.physics_substeps,
                allow_redirect=self.enable_fleet_redirection,
            )
            if self._cpp_sim is None:
                return False
            fleets_match = False
        elif not planets_match:
            if not self._cpp_sim.update_planets(
                planets_sorted,
                fleet_speed=self.fleet_speed,
                tick_dt=self.tick_seconds,
                physics_substeps=self.physics_substeps,
                clear_fleets=False,
            ):
                self._materialize_all_cpp_fleets()
                self._cpp_sim.close()
                self._cpp_sim = None
                return False
            if pid_to_idx != self._cpp_pid_to_idx:
                for fl in fleets:
                    if fl._cpp_env is self and (fl._units is None or fl._unit_headings is None):
                        self._materialize_fleet_units(fl)
                fleets_match = False

        if not fleets_match:
            planet_by_id = {p.id: p for p in planets_sorted}
            if not self._cpp_sim.load_fleets(fleets, planet_by_id, pid_to_idx):
                self._materialize_all_cpp_fleets()
                self._cpp_sim.close()
                self._cpp_sim = None
                return False
            self._cpp_gen += 1
            gen = self._cpp_gen
            any_kept_units = False
            for fl in fleets:
                if fl._units is not None or fl._unit_headings is not None:
                    any_kept_units = True
                fl._cpp_env = self
                fl._cpp_gen = gen
            self._cpp_units_materialized = any_kept_units

        self._cpp_planets_sorted = planets_sorted
        self._cpp_pid_to_idx = pid_to_idx
        self._cpp_cfg_sig = cfg_sig
        self._cpp_p_snap = [
            (p, p.id, p.owner, p.ships, p.x, p.y, p.production, p.radius) for p in planets
        ]
        self._cpp_f_snap = [
            (fl, fl.id, fl.owner, fl.ships, fl.x, fl.y, fl.target, fl.num_units) for fl in fleets
        ]
        return True

    def _spawn_cpp_fleet(self, source: Planet, target: Planet, ships_to_send: float) -> None:
        """Launch a multi-unit physical fleet matching calibrated Galcon 2 spawn geometry."""
        if self._sync_cpp_sim():
            src_idx = self._cpp_pid_to_idx.get(source.id, -1)
            dst_idx = self._cpp_pid_to_idx.get(target.id, -1)
            if src_idx >= 0 and dst_idx >= 0:
                fid = self._next_fleet_id
                rc, cx, cy, fl_radius, parts = self._cpp_sim.spawn_fleet_exact(
                    source.owner,
                    src_idx,
                    dst_idx,
                    float(ships_to_send),
                    fid,
                )
                if rc == 1:
                    return
                if rc == 0:
                    source.ships = max(0.0, source.ships - ships_to_send)
                    self._next_fleet_id = fid + 1
                    fl = Fleet(
                        id=fid,
                        owner=source.owner,
                        ships=float(ships_to_send),
                        x=float(cx),
                        y=float(cy),
                        source=source.id,
                        target=target.id,
                        radius=float(fl_radius),
                        num_units=int(parts),
                    )
                    fl._cpp_env = self
                    fl._cpp_gen = self._cpp_gen
                    self.fleets.append(fl)
                    # Keep planet & fleet snapshots synchronized with C++ state
                    p_snap = self._cpp_p_snap
                    if 0 <= source.id < len(p_snap) and p_snap[source.id][0] is source:
                        p_snap[source.id] = (
                            source, source.id, source.owner, source.ships,
                            source.x, source.y, source.production, source.radius,
                        )
                    else:
                        for i, s in enumerate(p_snap):
                            if s[0] is source:
                                p_snap[i] = (
                                    source, source.id, source.owner, source.ships,
                                    source.x, source.y, source.production, source.radius,
                                )
                                break
                    self._cpp_f_snap.append(
                        (fl, fl.id, fl.owner, fl.ships, fl.x, fl.y, fl.target, fl.num_units)
                    )
                    return
                # Capacity overflow in fixed POD SimState: materialize & fall back to Python/vector path
                self._materialize_all_cpp_fleets()
                self._cpp_sim.close()
                self._cpp_sim = None

        total_planet_ships = sum(max(0.0, p.ships) for p in self.planets)
        item_count = (
            BASE_ITEM_USERS
            + len(self.planets)
            + len(self.fleets)
            + sum(max(1, f.num_units) for f in self.fleets)
        )
        parts = physical_ship_count(ships_to_send, total_planet_ships, item_count)
        if parts <= 0:
            return

        source.ships = max(0.0, source.ships - ships_to_send)

        fid = self._next_fleet_id
        self._next_fleet_id += 1
        units, unit_headings, cx, cy, fl_radius = spawn_fleet_units(
            fid,
            parts,
            source.x,
            source.y,
            source.radius,
            target.x,
            target.y,
        )

        self.fleets.append(
            Fleet(
                id=fid,
                owner=source.owner,
                ships=float(ships_to_send),
                x=float(cx),
                y=float(cy),
                source=source.id,
                target=target.id,
                radius=float(fl_radius),
                num_units=parts,
                units=units,
                unit_headings=unit_headings,
            )
        )

    def _advance_one_tick_cpp(self) -> None:
        """Advance multi-ship C++17 (or bit-exact Python fallback) physics in-place on self.planets & self.fleets."""
        if self._sync_cpp_sim():
            self.fleets = self._cpp_sim.step_physics_and_read(self._cpp_planets_sorted, self.fleets)
            self._cpp_gen += 1
            gen = self._cpp_gen
            for fl in self.fleets:
                fl._cpp_gen = gen
            self._cpp_units_materialized = False
            self._cpp_p_snap = [
                (p, p.id, p.owner, p.ships, p.x, p.y, p.production, p.radius) for p in self.planets
            ]
            self._cpp_f_snap = [
                (fl, fl.id, fl.owner, fl.ships, fl.x, fl.y, fl.target, fl.num_units) for fl in self.fleets
            ]
            return

        adapter = object.__new__(_CppSimFallback)
        adapter.planets = {p.id: p for p in self.planets}
        adapter.fleets = {f.id: f for f in self.fleets}
        adapter.tick_dt = self.tick_seconds
        adapter.fleet_speed = self.fleet_speed
        adapter.physics_substeps = self.physics_substeps

        if HAS_FAST_SIM_CPP:
            step_physics_cpp(adapter)
        else:
            adapter._step_physics_python()

        for p in self.planets:
            p.neutral = (p.owner == self.PLAYER_NEUTRAL)
        self.fleets = [f for f in self.fleets if f.id in adapter.fleets]

    def _advance_one_tick(self) -> None:
        if self.use_cpp_sim:
            self._advance_one_tick_cpp()
            return
        # Legacy point-fleet tick update
        self._produce_ships()
        self._move_fleets()

    def _produce_ships(self) -> None:
        for planet in self.planets:
            if planet.neutral:
                continue
            if planet.owner == self.PLAYER_NEUTRAL:
                continue
            planet.ships += self._production_to_ships_per_second(planet.production) * self.tick_seconds

    def _production_to_ships_per_second(self, production: float) -> float:
        return production / self.PRODUCTION_TO_SHIPS_PER_SECOND_DIVISOR

    def _move_fleets(self) -> None:
        fleet_update_distance = self.fleet_speed * self.tick_seconds
        remaining_fleets = []

        for fleet in self.fleets:
            target = self.planets[fleet.target]
            distance_to_target = self._distance(fleet.x, fleet.y, target.x, target.y)

            if distance_to_target - fleet_update_distance < target.radius:
                self._land_fleet(fleet)
            else:
                dx, dy = self._vector_components(fleet.x, fleet.y, target.x, target.y, fleet_update_distance)
                fleet.x += dx
                fleet.y += dy
                remaining_fleets.append(fleet)

        self.fleets = remaining_fleets

    def _land_fleet(self, fleet: Fleet) -> None:
        target = self.planets[fleet.target]

        if fleet.owner == target.owner:
            target.ships += fleet.ships
            return

        diff = target.ships - fleet.ships
        if diff < 0:
            target.ships = -diff
            target.owner = fleet.owner
            target.neutral = False
        else:
            target.ships = diff

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x1 - x2, y1 - y2)

    @staticmethod
    def _vector_components(x1: float, y1: float, x2: float, y2: float, distance: float) -> Tuple[float, float]:
        angle = math.atan2(y2 - y1, x2 - x1)
        return distance * math.cos(angle), distance * math.sin(angle)

    # Given world coordinates (x, y), return the equivalent grid (x, y)
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        grid_x = math.floor((x + self.grid_max_x) / self.grid_square_size)
        if grid_x < 0:
            grid_x = 0
        elif grid_x >= self.grid_width:
            grid_x = self.grid_width - 1
        grid_y = math.floor((y + self.grid_max_y) / self.grid_square_size)
        if grid_y < 0:
            grid_y = 0
        elif grid_y >= self.grid_height:
            grid_y = self.grid_height - 1
        return grid_x, grid_y

    # Cells are indexed numerically, starting at index 0 = (0, 0), index 1 = (0, 1), index grid_width = (1, 0), ...
    def _grid_to_cell_index(self, grid_x: int, grid_y: int) -> int:
        return grid_y * self.grid_width + grid_x

    def _cell_index_to_grid(self, cell_index: int) -> Tuple[int, int]:
        grid_y = cell_index // self.grid_width
        grid_x = cell_index % self.grid_width
        return grid_x, grid_y

    # returns the (x, y) world coordinate of the center of cell (x, y)
    def _grid_cell_center(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        x = -self.grid_max_x + (grid_x + 0.5) * self.grid_square_size
        y = -self.grid_max_y + (grid_y + 0.5) * self.grid_square_size
        return x, y

    def _local_grid_offset(self, x: float, y: float) -> Tuple[float, float]:
        grid_x, grid_y = self._world_to_grid(x, y)
        sq = self.grid_square_size
        nx = (x - (-self.grid_max_x + grid_x * sq)) / sq
        if nx <= 0.0:
            nx = 0.0
        elif nx >= 1.0:
            nx = 1.0
        ny = (y - (-self.grid_max_y + grid_y * sq)) / sq
        if ny <= 0.0:
            ny = 0.0
        elif ny >= 1.0:
            ny = 1.0
        return nx, ny

    def _normalize_world_x(self, x: float) -> float:
        val = (x + self.grid_max_x) / (2.0 * self.grid_max_x)
        if val <= 0.0:
            return 0.0
        if val >= 1.0:
            return 1.0
        return val

    def _normalize_world_y(self, y: float) -> float:
        val = (y + self.grid_max_y) / (2.0 * self.grid_max_y)
        if val <= 0.0:
            return 0.0
        if val >= 1.0:
            return 1.0
        return val

    def _linear_encode(self, x: float, max_expected_value: float) -> float:
        val = x / max_expected_value
        if val <= 0.0:
            return 0.0
        if val >= 1.0:
            return 1.0
        return val

    # (x + 1) / (max_expected_x + 1). max_expected_x is the largest value the network is likely to ever see.
    def _log_encode(self, value: float, max_expected_value: float) -> float:
        value = max(0.0, float(value))
        max_expected_value = max(1.0, float(max_expected_value))
        return math.log(value + 1.0) / math.log(max_expected_value + 1.0)

    def _encode_ships(self, ships: float) -> float:
        val = ships / self.max_expected_ships
        if val <= 0.0:
            return 0.0
        if val >= 1.0:
            return 1.0
        return val

    def _encode_eta(self, eta_seconds: float) -> float:
        val = eta_seconds / self.max_expected_eta
        if val <= 0.0:
            return 0.0
        if val >= 1.0:
            return 1.0
        return val

    def _fleet_eta_seconds(self, fleet: Fleet) -> float:
        target = self.planets[fleet.target]
        distance_to_target_edge = math.hypot(fleet.x - target.x, fleet.y - target.y) - target.radius
        if distance_to_target_edge <= 0.0:
            return 0.0
        return distance_to_target_edge / (self.fleet_speed if self.fleet_speed > 1e-6 else 1e-6)

    def _planet_at_cell(self, grid_x: int, grid_y: int, owner: Optional[int] = None) -> Optional[Planet]:
        grid_max_x = self.grid_max_x
        grid_max_y = self.grid_max_y
        sq = self.grid_square_size
        w_m1 = self.grid_width - 1
        h_m1 = self.grid_height - 1
        floor = math.floor

        best: Optional[Planet] = None
        count = 0
        for planet in self.planets:
            if owner is not None and planet.owner != owner:
                continue
            gx = floor((planet.x + grid_max_x) / sq)
            if gx < 0:
                gx = 0
            elif gx > w_m1:
                gx = w_m1
            if gx != grid_x:
                continue
            gy = floor((planet.y + grid_max_y) / sq)
            if gy < 0:
                gy = 0
            elif gy > h_m1:
                gy = h_m1
            if gy != grid_y:
                continue
            count += 1
            if best is None or planet.id < best.id:
                best = planet

        if count > 1:
            logging.warning(
                'WARNING: More than one planet found in grid cell. This should not happen for grid cell size <= 25.'
            )
        return best

    def encode_action(
        self, source_x: int, source_y: int, target_x: int, target_y: int, is_redirect: bool = False
    ) -> int:
        source_cell = source_y * self.grid_width + source_x
        target_cell = target_y * self.grid_width + target_x
        base_action = source_cell * self.grid_cell_count + target_cell + 1
        return base_action + self.pair_action_count if is_redirect else base_action

    def decode_action_with_type(
        self, action: int
    ) -> Tuple[str, Optional[int], Optional[int], Optional[int], Optional[int]]:
        if action == self.pass_action:
            return 'pass', None, None, None, None

        shifted_action = int(action) - 1
        if shifted_action >= self.pair_action_count:
            action_type = 'redirect'
            shifted_action -= self.pair_action_count
        else:
            action_type = 'send'

        source_cell = shifted_action // self.grid_cell_count
        target_cell = shifted_action % self.grid_cell_count

        grid_width = self.grid_width
        return (
            action_type,
            source_cell % grid_width,
            source_cell // grid_width,
            target_cell % grid_width,
            target_cell // grid_width,
        )

    def decode_action(self, action: int) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        _, source_x, source_y, target_x, target_y = self.decode_action_with_type(action)
        return source_x, source_y, target_x, target_y

    def _ships_to_send(self, source_ships: float) -> float:
        ships = float(source_ships)
        if ships <= 0.0:
            return 0.0
        scaled = ships * self.send_ratio
        return float(math.floor(ships if ships < scaled else scaled))

    def _can_send_from_planet(self, planet: Planet) -> bool:
        ships = planet.ships
        if ships <= 0.0:
            return False
        scaled = ships * self.send_ratio
        ships_to_send = math.floor(ships if ships < scaled else scaled)
        return ships_to_send >= self.min_send_ships and ships_to_send > 0

    @property
    def legal_actions(self) -> List[int]:
        legal = [self.pass_action]

        grid_max_x = self.grid_max_x
        grid_max_y = self.grid_max_y
        sq = self.grid_square_size
        grid_width = self.grid_width
        w_m1 = grid_width - 1
        h_m1 = self.grid_height - 1
        grid_cell_count = self.grid_cell_count
        cur_player = self._current_player
        send_ratio = self.send_ratio
        min_send_ships = self.min_send_ships
        floor = math.floor

        target_by_cell: Dict[Tuple[int, int], Planet] = {}
        friendly_by_cell: Dict[Tuple[int, int], Planet] = {}
        friendly_source_cells = set()

        for planet in self.planets:
            gx = floor((planet.x + grid_max_x) / sq)
            if gx < 0:
                gx = 0
            elif gx > w_m1:
                gx = w_m1
            gy = floor((planet.y + grid_max_y) / sq)
            if gy < 0:
                gy = 0
            elif gy > h_m1:
                gy = h_m1
            cell = (gx, gy)

            prev_t = target_by_cell.get(cell)
            if prev_t is None or planet.id < prev_t.id:
                target_by_cell[cell] = planet

            if planet.owner == cur_player:
                prev_f = friendly_by_cell.get(cell)
                if prev_f is None or planet.id < prev_f.id:
                    friendly_by_cell[cell] = planet
                ships = planet.ships
                if ships > 0.0:
                    scaled = ships * send_ratio
                    to_send = floor(ships if ships < scaled else scaled)
                    if to_send >= min_send_ships and to_send > 0:
                        friendly_source_cells.add(cell)

        sorted_targets = [
            (ty * grid_width + tx, target_by_cell[(tx, ty)].id)
            for (tx, ty) in sorted(target_by_cell.keys())
        ]

        for sx, sy in sorted(friendly_source_cells):
            src_id = friendly_by_cell[(sx, sy)].id
            row_base = (sy * grid_width + sx) * grid_cell_count + 1
            for tgt_cell_idx, tgt_id in sorted_targets:
                if src_id != tgt_id:
                    legal.append(row_base + tgt_cell_idx)

        if self.enable_fleet_redirection and self.fleets:
            # Map each cell containing friendly fleets to either the unique target_id shared
            # by all friendly fleets in that cell, or -1 if >=2 distinct targets exist.
            fleet_cell_target: Dict[Tuple[int, int], int] = {}
            for fleet in self.fleets:
                if fleet.owner != cur_player:
                    continue
                gx = floor((fleet.x + grid_max_x) / sq)
                if gx < 0:
                    gx = 0
                elif gx > w_m1:
                    gx = w_m1
                gy = floor((fleet.y + grid_max_y) / sq)
                if gy < 0:
                    gy = 0
                elif gy > h_m1:
                    gy = h_m1
                cell = (gx, gy)
                prev_tgt = fleet_cell_target.get(cell)
                if prev_tgt is None:
                    fleet_cell_target[cell] = fleet.target
                elif prev_tgt != -1 and prev_tgt != fleet.target:
                    fleet_cell_target[cell] = -1

            redir_offset = self.pair_action_count + 1
            for (sx, sy) in sorted(fleet_cell_target.keys()):
                only_tgt = fleet_cell_target[(sx, sy)]
                row_base = (sy * grid_width + sx) * grid_cell_count + redir_offset
                if only_tgt == -1:
                    for tgt_cell_idx, _ in sorted_targets:
                        legal.append(row_base + tgt_cell_idx)
                else:
                    for tgt_cell_idx, tgt_id in sorted_targets:
                        if tgt_id != only_tgt:
                            legal.append(row_base + tgt_cell_idx)

        return legal

    def _fill_action_mask(self, action_mask: np.ndarray) -> None:
        action_mask[self.pass_action] = 1

        grid_max_x = self.grid_max_x
        grid_max_y = self.grid_max_y
        sq = self.grid_square_size
        grid_width = self.grid_width
        w_m1 = grid_width - 1
        h_m1 = self.grid_height - 1
        grid_cell_count = self.grid_cell_count
        cur_player = self._current_player
        send_ratio = self.send_ratio
        min_send_ships = self.min_send_ships
        floor = math.floor

        target_by_cell: Dict[int, int] = {}
        friendly_by_cell: Dict[int, int] = {}
        friendly_source_cells = set()

        for planet in self.planets:
            gx = floor((planet.x + grid_max_x) / sq)
            if gx < 0:
                gx = 0
            elif gx > w_m1:
                gx = w_m1
            gy = floor((planet.y + grid_max_y) / sq)
            if gy < 0:
                gy = 0
            elif gy > h_m1:
                gy = h_m1
            cell_idx = gy * grid_width + gx

            prev_tid = target_by_cell.get(cell_idx)
            if prev_tid is None or planet.id < prev_tid:
                target_by_cell[cell_idx] = planet.id

            if planet.owner == cur_player:
                prev_fid = friendly_by_cell.get(cell_idx)
                if prev_fid is None or planet.id < prev_fid:
                    friendly_by_cell[cell_idx] = planet.id
                ships = planet.ships
                if ships > 0.0:
                    scaled = ships * send_ratio
                    to_send = floor(ships if ships < scaled else scaled)
                    if to_send >= min_send_ships and to_send > 0:
                        friendly_source_cells.add(cell_idx)

        target_items = list(target_by_cell.items())
        for src_cell_idx in friendly_source_cells:
            src_id = friendly_by_cell[src_cell_idx]
            row_base = src_cell_idx * grid_cell_count + 1
            for tgt_cell_idx, tgt_id in target_items:
                if src_id != tgt_id:
                    action_mask[row_base + tgt_cell_idx] = 1

        if self.enable_fleet_redirection and self.fleets:
            fleet_cell_target: Dict[int, int] = {}
            for fleet in self.fleets:
                if fleet.owner != cur_player:
                    continue
                gx = floor((fleet.x + grid_max_x) / sq)
                if gx < 0:
                    gx = 0
                elif gx > w_m1:
                    gx = w_m1
                gy = floor((fleet.y + grid_max_y) / sq)
                if gy < 0:
                    gy = 0
                elif gy > h_m1:
                    gy = h_m1
                cell_idx = gy * grid_width + gx
                prev_tgt = fleet_cell_target.get(cell_idx)
                if prev_tgt is None:
                    fleet_cell_target[cell_idx] = fleet.target
                elif prev_tgt != -1 and prev_tgt != fleet.target:
                    fleet_cell_target[cell_idx] = -1

            redir_offset = self.pair_action_count + 1
            for src_cell_idx, only_tgt in fleet_cell_target.items():
                row_base = src_cell_idx * grid_cell_count + redir_offset
                if only_tgt == -1:
                    for tgt_cell_idx, _ in target_items:
                        action_mask[row_base + tgt_cell_idx] = 1
                else:
                    for tgt_cell_idx, tgt_id in target_items:
                        if tgt_id != only_tgt:
                            action_mask[row_base + tgt_cell_idx] = 1

    def observe(self) -> dict:
        if self.LANDING_BUCKETS is not self._cpp_buckets_ref:
            self._init_cpp_bucket_buffers()
        encoded_in_cpp = False
        if (
            self.use_cpp_sim
            and self.grid_cell_count <= 1024
            and self._cpp_n_buckets <= 16
            and self._sync_cpp_sim()
        ):
            observation = np.empty(self._observation_shape, dtype=np.float32)
            action_mask = np.empty(self.total_num_actions, dtype=np.int8)
            encoded_in_cpp = self._cpp_sim.encode_obs_and_mask(
                self._current_player,
                self.grid_width,
                self.grid_height,
                self.grid_max_x,
                self.grid_max_y,
                self.grid_square_size,
                self.fleet_top_k,
                self.max_expected_ships,
                self.max_expected_production,
                self.max_fleet_radius,
                self.max_expected_eta,
                self.send_ratio,
                self.min_send_ships,
                self.enable_fleet_redirection,
                self._cpp_n_buckets,
                self._ptr_cpp_bucket_lowers,
                self._ptr_cpp_bucket_uppers,
                observation.ctypes.data,
                action_mask.ctypes.data,
            )

        if not encoded_in_cpp:
            action_mask = np.zeros(self.total_num_actions, dtype=np.int8)
            self._fill_action_mask(action_mask)
            observation = self.current_state()

        if self.battle_mode in ['play_with_bot_mode', 'eval_mode']:
            # Single-agent perspective (each step executes agent + bot): disable MCTS minimax
            # negation and buffer alternating sign flips. self_play_mode policies remap -1 -> 1 in _forward_eval.
            to_play = -1
        else:
            to_play = self._current_player

        if self.debug_legal_actions:
            print(self.legal_actions)

        if self.debug_observation:
            # Summing the map and flipping it makes it visible as a grid of numbers. You can almost make out what is
            # happening just by looking at the grid of numbers changing.
            print("Observation summed across channels:")
            print(np.flip(np.round(sum(observation), 2), axis=1))

        return {
            'observation': observation,
            'action_mask': action_mask,
            'current_player_index': self.current_player_index,
            'to_play': to_play,
        }

    def current_state(self) -> np.ndarray:
        """
        Encode the current game state as a channel-first grid tensor.

        Channel layout:
            planet channels:
                0 local x offset inside grid square
                1 local y offset inside grid square
                2 planet ships (if friendly)
                3 planet ships (if enemy)
                4 planet ships (if neutral)
                5 planet production (if friendly)
                6 planet production (if enemy)
                7 planet production (if neutral)

            fleet channels:
                For friendly fleets, then enemy fleets:
                    K largest fleets by ship count plus one remainder summary slot.
                    Each slot has:
                        target x
                        target y
                        radius
                        ships (not broken up into enemy ships or friendly ships, since the friendly/enemy slots are already separated
                        ETA (estimated time of arrival, in seconds)

            landing schedule channels:
                For each bucket:
                    friendly ships landing
                    enemy ships landing
                    max(friendly - enemy, 0)
                    max(enemy - friendly, 0)
        """
        if self.LANDING_BUCKETS is not self._cpp_buckets_ref:
            self._init_cpp_bucket_buffers()
        if (
            self.use_cpp_sim
            and self.grid_cell_count <= 1024
            and self._cpp_n_buckets <= 16
            and self._sync_cpp_sim()
        ):
            obs = np.empty(self._observation_shape, dtype=np.float32)
            if self._cpp_sim.encode_obs_and_mask(
                self._current_player,
                self.grid_width,
                self.grid_height,
                self.grid_max_x,
                self.grid_max_y,
                self.grid_square_size,
                self.fleet_top_k,
                self.max_expected_ships,
                self.max_expected_production,
                self.max_fleet_radius,
                self.max_expected_eta,
                self.send_ratio,
                self.min_send_ships,
                self.enable_fleet_redirection,
                self._cpp_n_buckets,
                self._ptr_cpp_bucket_lowers,
                self._ptr_cpp_bucket_uppers,
                obs.ctypes.data,
                0,
            ):
                return obs

        obs = np.zeros(self._observation_shape, dtype=np.float32)

        self._encode_planets(obs)
        if self.fleets:
            self._encode_fleets(obs)
            self._encode_landing_schedule(obs)

        return obs

    def _encode_planets(self, obs: np.ndarray) -> None:
        grid_max_x = self.grid_max_x
        grid_max_y = self.grid_max_y
        sq = self.grid_square_size
        w_m1 = self.grid_width - 1
        h_m1 = self.grid_height - 1
        max_ships = self.max_expected_ships
        max_prod = self.max_expected_production
        cur_player = self._current_player
        neutral_player = self.PLAYER_NEUTRAL
        floor = math.floor

        ch_lx = self.PLANET_LOCAL_X_CHANNEL
        ch_ly = self.PLANET_LOCAL_Y_CHANNEL
        ch_fs = self.PLANET_FRIENDLY_SHIPS_CHANNEL
        ch_fp = self.PLANET_FRIENDLY_PRODUCTION_CHANNEL
        ch_ns = self.PLANET_NEUTRAL_SHIPS_CHANNEL
        ch_np = self.PLANET_NEUTRAL_PRODUCTION_CHANNEL
        ch_es = self.PLANET_ENEMY_SHIPS_CHANNEL
        ch_ep = self.PLANET_ENEMY_PRODUCTION_CHANNEL

        for planet in self.planets:
            px = planet.x
            py = planet.y
            gx = floor((px + grid_max_x) / sq)
            if gx < 0:
                gx = 0
            elif gx > w_m1:
                gx = w_m1
            gy = floor((py + grid_max_y) / sq)
            if gy < 0:
                gy = 0
            elif gy > h_m1:
                gy = h_m1

            lx = (px - (-grid_max_x + gx * sq)) / sq
            if lx <= 0.0:
                lx = 0.0
            elif lx >= 1.0:
                lx = 1.0
            ly = (py - (-grid_max_y + gy * sq)) / sq
            if ly <= 0.0:
                ly = 0.0
            elif ly >= 1.0:
                ly = 1.0

            obs[ch_lx, gy, gx] = lx
            obs[ch_ly, gy, gx] = ly

            s_val = planet.ships / max_ships
            if s_val <= 0.0:
                s_val = 0.0
            elif s_val >= 1.0:
                s_val = 1.0
            p_val = planet.production / max_prod
            if p_val <= 0.0:
                p_val = 0.0
            elif p_val >= 1.0:
                p_val = 1.0

            own = planet.owner
            if own == cur_player:
                obs[ch_fs, gy, gx] = s_val
                obs[ch_fp, gy, gx] = p_val
            elif own == neutral_player:
                obs[ch_ns, gy, gx] = s_val
                obs[ch_np, gy, gx] = p_val
            else:
                obs[ch_es, gy, gx] = s_val
                obs[ch_ep, gy, gx] = p_val

    def _encode_fleets(self, obs: np.ndarray) -> None:
        fleets = self.fleets
        if not fleets:
            return

        grid_max_x = self.grid_max_x
        grid_max_y = self.grid_max_y
        span_x = 2.0 * grid_max_x
        span_y = 2.0 * grid_max_y
        sq = self.grid_square_size
        w_m1 = self.grid_width - 1
        h_m1 = self.grid_height - 1
        cur_player = self._current_player
        planets = self.planets
        speed = self.fleet_speed if self.fleet_speed > 1e-6 else 1e-6
        max_rad = self.max_fleet_radius if self.max_fleet_radius > 1e-6 else 1e-6
        max_ships = self.max_expected_ships
        max_eta = self.max_expected_eta
        top_k = self.fleet_top_k
        feat_cnt = self.FLEET_FEATURE_COUNT
        floor = math.floor
        hypot = math.hypot

        friendly_by_cell: Dict[Tuple[int, int], List[Fleet]] = {}
        enemy_by_cell: Dict[Tuple[int, int], List[Fleet]] = {}

        for fleet in fleets:
            gx = floor((fleet.x + grid_max_x) / sq)
            if gx < 0:
                gx = 0
            elif gx > w_m1:
                gx = w_m1
            gy = floor((fleet.y + grid_max_y) / sq)
            if gy < 0:
                gy = 0
            elif gy > h_m1:
                gy = h_m1
            cell = (gx, gy)
            if fleet.owner == cur_player:
                lst = friendly_by_cell.get(cell)
                if lst is None:
                    friendly_by_cell[cell] = [fleet]
                else:
                    lst.append(fleet)
            else:
                lst = enemy_by_cell.get(cell)
                if lst is None:
                    enemy_by_cell[cell] = [fleet]
                else:
                    lst.append(fleet)

        fleet_base_channel = self.PLANET_CHANNEL_COUNT
        enemy_base_channel = fleet_base_channel + self.fleet_slot_count_per_side * feat_cnt

        for side_map, base_ch in ((friendly_by_cell, fleet_base_channel), (enemy_by_cell, enemy_base_channel)):
            for (gx, gy), cell_fleets in side_map.items():
                n_fl = len(cell_fleets)
                if n_fl > 1:
                    cell_fleets.sort(key=lambda fl: fl.ships, reverse=True)
                n_top = top_k if n_fl > top_k else n_fl
                for slot_idx in range(n_top):
                    fl = cell_fleets[slot_idx]
                    tgt = planets[fl.target]
                    dist_edge = hypot(fl.x - tgt.x, fl.y - tgt.y) - tgt.radius
                    eta = (dist_edge / speed) if dist_edge > 0.0 else 0.0
                    ch = base_ch + slot_idx * feat_cnt

                    vx = (tgt.x + grid_max_x) / span_x
                    obs[ch, gy, gx] = 0.0 if vx <= 0.0 else (1.0 if vx >= 1.0 else vx)
                    vy = (tgt.y + grid_max_y) / span_y
                    obs[ch + 1, gy, gx] = 0.0 if vy <= 0.0 else (1.0 if vy >= 1.0 else vy)
                    vr = fl.radius / max_rad
                    obs[ch + 2, gy, gx] = 0.0 if vr <= 0.0 else (1.0 if vr >= 1.0 else vr)
                    vs = fl.ships / max_ships
                    obs[ch + 3, gy, gx] = 0.0 if vs <= 0.0 else (1.0 if vs >= 1.0 else vs)
                    ve = eta / max_eta
                    obs[ch + 4, gy, gx] = 0.0 if ve <= 0.0 else (1.0 if ve >= 1.0 else ve)

                if n_fl > top_k:
                    rem = cell_fleets[top_k:]
                    total_ships = 0.0
                    for fl in rem:
                        total_ships += fl.ships
                    if total_ships > 0.0:
                        w_tx = 0.0
                        w_ty = 0.0
                        w_rad = 0.0
                        w_eta = 0.0
                        for fl in rem:
                            tgt = planets[fl.target]
                            w = fl.ships / total_ships
                            w_tx += tgt.x * w
                            w_ty += tgt.y * w
                            w_rad += fl.radius * w
                            dist_edge = hypot(fl.x - tgt.x, fl.y - tgt.y) - tgt.radius
                            if dist_edge > 0.0:
                                w_eta += (dist_edge / speed) * w
                        ch = base_ch + top_k * feat_cnt
                        vx = (w_tx + grid_max_x) / span_x
                        obs[ch, gy, gx] = 0.0 if vx <= 0.0 else (1.0 if vx >= 1.0 else vx)
                        vy = (w_ty + grid_max_y) / span_y
                        obs[ch + 1, gy, gx] = 0.0 if vy <= 0.0 else (1.0 if vy >= 1.0 else vy)
                        vr = w_rad / max_rad
                        obs[ch + 2, gy, gx] = 0.0 if vr <= 0.0 else (1.0 if vr >= 1.0 else vr)
                        vs = total_ships / max_ships
                        obs[ch + 3, gy, gx] = 0.0 if vs <= 0.0 else (1.0 if vs >= 1.0 else vs)
                        ve = w_eta / max_eta
                        obs[ch + 4, gy, gx] = 0.0 if ve <= 0.0 else (1.0 if ve >= 1.0 else ve)

    # Encode top-K fleets and a "summary" channel for a single player (friendly OR enemy)
    def _encode_top_k_fleets(
            self,
            obs: np.ndarray,
            grid_x: int,
            grid_y: int,
            fleets: List[Fleet],
            base_channel: int,
    ) -> None:
        if len(fleets) == 0:
            return

        fleets = sorted(fleets, key=lambda fleet: fleet.ships, reverse=True)
        top_fleets = fleets[:self.fleet_top_k]
        remainder_fleets = fleets[self.fleet_top_k:]

        for slot_index, fleet in enumerate(top_fleets):
            self._encode_single_fleet_slot(
                obs=obs,
                grid_x=grid_x,
                grid_y=grid_y,
                fleet=fleet,
                base_channel=base_channel + slot_index * self.FLEET_FEATURE_COUNT,
            )

        if len(remainder_fleets) > 0:
            summary_channel = base_channel + self.fleet_top_k * self.FLEET_FEATURE_COUNT
            self._encode_fleet_summary_slot(
                obs=obs,
                grid_x=grid_x,
                grid_y=grid_y,
                fleets=remainder_fleets,
                base_channel=summary_channel,
            )

    def _encode_single_fleet_slot(
            self,
            obs: np.ndarray,
            grid_x: int,
            grid_y: int,
            fleet: Fleet,
            base_channel: int,
    ) -> None:
        target = self.planets[fleet.target]
        eta_seconds = self._fleet_eta_seconds(fleet)

        obs[base_channel + 0, grid_y, grid_x] = self._normalize_world_x(target.x)
        obs[base_channel + 1, grid_y, grid_x] = self._normalize_world_y(target.y)
        obs[base_channel + 2, grid_y, grid_x] = self._linear_encode(fleet.radius, max(self.max_fleet_radius, 1e-6))
        obs[base_channel + 3, grid_y, grid_x] = self._encode_ships(fleet.ships)
        obs[base_channel + 4, grid_y, grid_x] = self._encode_eta(eta_seconds)

    def _encode_fleet_summary_slot(
            self,
            obs: np.ndarray,
            grid_x: int,
            grid_y: int,
            fleets: List[Fleet],
            base_channel: int,
    ) -> None:
        total_ships = float(sum(fleet.ships for fleet in fleets))
        if total_ships <= 0:
            return

        weighted_target_x = 0.0
        weighted_target_y = 0.0
        weighted_radius = 0.0
        weighted_eta = 0.0

        for fleet in fleets:
            target = self.planets[fleet.target]
            weight = fleet.ships / total_ships
            weighted_target_x += target.x * weight
            weighted_target_y += target.y * weight
            weighted_radius += fleet.radius * weight
            weighted_eta += self._fleet_eta_seconds(fleet) * weight

        obs[base_channel + 0, grid_y, grid_x] = self._normalize_world_x(weighted_target_x)
        obs[base_channel + 1, grid_y, grid_x] = self._normalize_world_y(weighted_target_y)
        obs[base_channel + 2, grid_y, grid_x] = self._linear_encode(weighted_radius, max(self.max_fleet_radius, 1e-6))
        obs[base_channel + 3, grid_y, grid_x] = self._encode_ships(total_ships)
        obs[base_channel + 4, grid_y, grid_x] = self._encode_eta(weighted_eta)

    def _encode_landing_schedule(self, obs: np.ndarray) -> None:
        fleets = self.fleets
        if not fleets:
            return

        landing_base_channel = self.PLANET_CHANNEL_COUNT + self.fleet_channel_count
        buckets = self.LANDING_BUCKETS
        n_buckets = len(buckets)
        feat_per_bucket = self.LANDING_FEATURE_COUNT_PER_BUCKET
        planets = self.planets
        cur_player = self._current_player
        speed = self.fleet_speed if self.fleet_speed > 1e-6 else 1e-6
        max_ships = self.max_expected_ships
        grid_max_x = self.grid_max_x
        grid_max_y = self.grid_max_y
        sq = self.grid_square_size
        w_m1 = self.grid_width - 1
        h_m1 = self.grid_height - 1
        floor = math.floor
        hypot = math.hypot

        default_3_buckets = (
            n_buckets == 3
            and buckets[0] == (0.0, 1.0)
            and buckets[1] == (1.0, 2.0)
            and buckets[2] == (2.0, math.inf)
        )

        # Accumulate incoming ships in a single O(F) pass, only touching planets with incoming fleets.
        # Each entry stores [friendly_0..B-1, enemy_0..B-1].
        by_target: Dict[int, List[float]] = {}
        two_b = 2 * n_buckets
        for fleet in fleets:
            tid = fleet.target
            target = planets[tid]
            dist_edge = hypot(fleet.x - target.x, fleet.y - target.y) - target.radius
            eta = (dist_edge / speed) if dist_edge > 0.0 else 0.0
            if default_3_buckets:
                if eta < 1.0:
                    b_idx = 0
                elif eta < 2.0:
                    b_idx = 1
                else:
                    b_idx = 2
            else:
                b_idx = self._landing_bucket_index(eta)

            acc = by_target.get(tid)
            if acc is None:
                acc = [0.0] * two_b
                by_target[tid] = acc
            if fleet.owner == cur_player:
                acc[b_idx] += fleet.ships
            else:
                acc[n_buckets + b_idx] += fleet.ships

        for tid, acc in by_target.items():
            planet = planets[tid]
            gx = floor((planet.x + grid_max_x) / sq)
            if gx < 0:
                gx = 0
            elif gx > w_m1:
                gx = w_m1
            gy = floor((planet.y + grid_max_y) / sq)
            if gy < 0:
                gy = 0
            elif gy > h_m1:
                gy = h_m1

            for b_idx in range(n_buckets):
                friendly = acc[b_idx]
                enemy = acc[n_buckets + b_idx]
                if friendly == 0.0 and enemy == 0.0:
                    continue
                ch = landing_base_channel + b_idx * feat_per_bucket
                if friendly > 0.0:
                    vf = friendly / max_ships
                    obs[ch, gy, gx] = 1.0 if vf >= 1.0 else vf
                if enemy > 0.0:
                    ve = enemy / max_ships
                    obs[ch + 1, gy, gx] = 1.0 if ve >= 1.0 else ve
                diff = friendly - enemy
                if diff > 0.0:
                    vd = diff / max_ships
                    obs[ch + 2, gy, gx] = 1.0 if vd >= 1.0 else vd
                elif diff < 0.0:
                    vd = (-diff) / max_ships
                    obs[ch + 3, gy, gx] = 1.0 if vd >= 1.0 else vd

    def _landing_bucket_index(self, eta_seconds: float) -> int:
        for bucket_index, (lower, upper) in enumerate(self.LANDING_BUCKETS):
            if lower <= eta_seconds < upper:
                return bucket_index
        return len(self.LANDING_BUCKETS) - 1

    def get_done_winner(self) -> Tuple[bool, int]:
        winner_by_elimination = self._winner_by_elimination()
        if winner_by_elimination != -1:
            return True, winner_by_elimination

        if self._step_count >= self.max_episode_steps:
            return True, self._winner_by_timeout()

        return False, -1

    def _winner_by_elimination(self) -> int:
        p1_alive = self._player_has_planets_or_fleets(self.PLAYER_1)
        p2_alive = self._player_has_planets_or_fleets(self.PLAYER_2)

        if p1_alive and not p2_alive:
            return self.PLAYER_1
        if p2_alive and not p1_alive:
            return self.PLAYER_2
        return -1

    def _player_has_planets_or_fleets(self, player: int) -> bool:
        return any(p.owner == player for p in self.planets) or any(f.owner == player for f in self.fleets)

    def _winner_by_timeout(self) -> int:
        p1_production = self._total_production(self.PLAYER_1)
        p2_production = self._total_production(self.PLAYER_2)

        if p1_production > p2_production:
            return self.PLAYER_1
        if p2_production > p1_production:
            return self.PLAYER_2

        p1_ships = self._total_ships(self.PLAYER_1)
        p2_ships = self._total_ships(self.PLAYER_2)

        if p1_ships > p2_ships:
            return self.PLAYER_1
        if p2_ships > p1_ships:
            return self.PLAYER_2

        return -1

    # total production, rounded to nearest 1/100th
    def _total_production(self, player: int) -> float:
        return float(round(sum(p.production for p in self.planets if p.owner == player), 2))

    # total ships, rounded to nearest 1/100th
    def _total_ships(self, player: int) -> float:
        planet_ships = sum(p.ships for p in self.planets if p.owner == player)
        fleet_ships = sum(f.ships for f in self.fleets if f.owner == player)
        return float(round(planet_ships + fleet_ships, 2))

    def _get_state_info(self, winner: int) -> dict:
        return {
            'winner': winner,
            'step_count': self._step_count,
            'simulated_seconds': self._step_count * self.tick_seconds,
            'production_player_1': self._total_production(self.PLAYER_1),
            'production_player_2': self._total_production(self.PLAYER_2),
            'ships_player_1': self._total_ships(self.PLAYER_1),
            'ships_player_2': self._total_ships(self.PLAYER_2),
        }

    def _pick_cpp_bot_action(self, mode: int, rng_val: int = 0) -> int:
        """Fast C++ action picker for fixed (mode=0) and random (mode=1) rule bots (-1 on fallback)."""
        if self.use_cpp_sim and self.grid_cell_count <= 1024 and self._sync_cpp_sim():
            return self._cpp_sim.pick_bot_action(
                self._current_player,
                self.grid_width,
                self.grid_height,
                self.grid_max_x,
                self.grid_max_y,
                self.grid_square_size,
                self.send_ratio,
                self.min_send_ships,
                self.enable_fleet_redirection,
                mode,
                rng_val,
            )
        return -1

    def random_action(self) -> int:
        return int(np.random.choice(self.legal_actions))

    def bot_action(self) -> int:
        if np.random.rand() < self.prob_random_action_in_bot:
            return self.random_action()
        if np.random.rand() < self.prob_pass_action_in_bot:
            return self.pass_action
        return self.bot.get_action()

    def action_to_string(self, action: int) -> str:
        source_x, source_y, target_x, target_y = self.decode_action(action)
        if source_x is None and source_y is None and target_x is None and target_y is None:
            return 'Pass'
        return (
            f'Send {self.send_ratio:.0%} ships '
            f'from grid ({source_x}, {source_y}) to grid ({target_x}, {target_y})'
        )

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        self._map_rng = self._init_map_rng(seed, dynamic_seed)
        np.random.seed(seed)

    def close(self) -> None:
        if getattr(self, '_cpp_sim', None) is not None:
            self._cpp_sim.close()
            self._cpp_sim = None
            self._cpp_p_snap = []
            self._cpp_f_snap = []

    def __del__(self) -> None:
        try:
            cpp_sim = getattr(self, '_cpp_sim', None)
            if cpp_sim is not None:
                cpp_sim.close()
                self._cpp_sim = None
        except Exception:
            pass

    def __deepcopy__(self, memo: dict) -> 'GalconEnv':
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        skip_keys = {
            '_cpp_sim',
            '_cpp_p_snap',
            '_cpp_f_snap',
            '_cpp_planets_sorted',
            '_cpp_pid_to_idx',
            '_cpp_cfg_sig',
            '_cpp_bucket_lowers',
            '_cpp_bucket_uppers',
            '_ptr_cpp_bucket_lowers',
            '_ptr_cpp_bucket_uppers',
        }
        for k, v in self.__dict__.items():
            if k in skip_keys:
                continue
            setattr(result, k, copy.deepcopy(v, memo))
        result._cpp_sim = None
        result._cpp_gen = 0
        result._cpp_units_materialized = False
        result._cpp_cfg_sig = None
        result._cpp_planets_sorted = []
        result._cpp_pid_to_idx = {}
        result._cpp_p_snap = []
        result._cpp_f_snap = []
        result._init_cpp_bucket_buffers()
        return result

    def render(self, mode: Optional[str] = None) -> None:
        """
        Overview:
            Render the current Galcon game state.
        Arguments:
            - mode (:obj:`str`): Rendering mode. Options:
                - None / 'state_realtime_mode': Print state to console.
                - 'image_realtime_mode': Display as a matplotlib figure in real time.
                - 'image_savefile_mode': Capture frame for later saving as GIF/MP4.
        """
        if mode is None or mode == 'state_realtime_mode':
            print(f'Galcon step={self._step_count}, current_player={self._current_player}')
            print('planets:', self.planets)
            print('fleets:', self.fleets)
            return

        frame = self._render_to_rgb_array()
        self.frames.append(frame)

        if mode == 'image_realtime_mode':
            plt.imshow(frame)
            plt.axis('off')
            plt.draw()
            plt.pause(0.001)

    def _render_to_rgb_array(self) -> np.ndarray:
        """
        Overview:
            Render the current Galcon game state to an RGB numpy array.

        Color scheme:
            - Player 1 planets/fleets: blue (#4A90D9)
            - Player 2 planets/fleets: red (#E05A5A)
            - Neutral planets: gray (#888888)
            - Background: dark (#1A1A2E)

        Planets are drawn as filled circles sized proportionally to their radius.
        Fleets are drawn as solid triangles oriented toward their target, labeled with ship count.
        """
        # ---- color palette ----
        COLOR_BG = '#1A1A2E'
        COLOR_P1 = '#4A90D9'   # blue  – player 1
        COLOR_P2 = '#E05A5A'   # red   – player 2
        COLOR_NEUTRAL = '#888888'
        COLOR_TEXT = '#FFFFFF'
        COLOR_BORDER = '#CCCCCC'

        def owner_color(owner: int) -> str:
            if owner == self.PLAYER_1:
                return COLOR_P1
            elif owner == self.PLAYER_2:
                return COLOR_P2
            return COLOR_NEUTRAL

        fig_w_in = 10.0
        fig_h_in = fig_w_in * (2 * self.grid_max_y) / (2 * self.grid_max_x)
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
        fig.patch.set_facecolor(COLOR_BG)
        ax.set_facecolor(COLOR_BG)

        ax.set_xlim(-self.grid_max_x, self.grid_max_x)
        ax.set_ylim(-self.grid_max_y, self.grid_max_y)
        ax.set_aspect('equal')
        ax.axis('off')

        # ---- draw planets ----
        for planet in self.planets:
            color = owner_color(planet.owner)
            # Scale circle radius from world units to data units
            circle = plt.Circle(
                (planet.x, planet.y),
                radius=planet.radius,
                color=color,
                alpha=0.85,
                zorder=2,
            )
            ax.add_patch(circle)
            # Thin border ring
            ring = plt.Circle(
                (planet.x, planet.y),
                radius=planet.radius,
                color=COLOR_BORDER,
                fill=False,
                linewidth=0.8,
                alpha=0.5,
                zorder=3,
            )
            ax.add_patch(ring)
            # Ship count label inside the planet
            ship_str = f'{int(planet.ships)}'
            ax.text(
                planet.x, planet.y,
                ship_str,
                color=COLOR_TEXT,
                fontsize=20,
                ha='center', va='center',
                fontweight='bold',
                zorder=4,
            )

        # ---- draw fleets ----
        for fleet in self.fleets:
            f_color = owner_color(fleet.owner)
            if fleet.units and len(fleet.units) > 1:
                # Multi-ship C++ flock: draw each physical ship oriented along its live heading
                headings = fleet.unit_headings
                has_headings = headings is not None and len(headings) == len(fleet.units)
                target_planet = self.planets[fleet.target]
                unit_size = 4.2
                for idx_u, (ux, uy) in enumerate(fleet.units):
                    if has_headings:
                        hx, hy = headings[idx_u]
                        u_ang = math.atan2(hy, hx)
                    else:
                        u_ang = math.atan2(target_planet.y - uy, target_planet.x - ux)
                    tip = np.array([ux + unit_size * math.cos(u_ang),
                                    uy + unit_size * math.sin(u_ang)])
                    left = np.array([ux + unit_size * 0.65 * math.cos(u_ang + math.radians(140)),
                                     uy + unit_size * 0.65 * math.sin(u_ang + math.radians(140))])
                    right = np.array([ux + unit_size * 0.65 * math.cos(u_ang - math.radians(140)),
                                      uy + unit_size * 0.65 * math.sin(u_ang - math.radians(140))])
                    ax.add_patch(Polygon(
                        [tip, left, right],
                        closed=True,
                        color=f_color,
                        alpha=0.88,
                        zorder=5,
                    ))

                ax.text(
                    fleet.x, fleet.y,
                    str(int(fleet.ships)),
                    color=COLOR_TEXT,
                    fontsize=15,
                    ha='center', va='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=COLOR_BG, alpha=0.55, edgecolor='none'),
                    zorder=6,
                )
            else:
                target_planet = self.planets[fleet.target]
                if fleet.unit_headings and len(fleet.unit_headings) >= 1:
                    hx, hy = fleet.unit_headings[0]
                    angle = math.atan2(hy, hx)
                else:
                    dx = target_planet.x - fleet.x
                    dy = target_planet.y - fleet.y
                    dist = math.hypot(dx, dy)
                    angle = 0.0 if dist < 1e-6 else math.atan2(dy, dx)

                # Triangle size proportional to ship count, capped for readability
                tri_size = max(4.0, min(14.0, 4.0 + fleet.ships / 15.0))

                # Triangle pointing in direction of travel:
                #   tip at the front, two base corners behind
                tip = np.array([fleet.x + tri_size * math.cos(angle),
                                fleet.y + tri_size * math.sin(angle)])
                left_angle = angle + math.radians(140)
                right_angle = angle - math.radians(140)
                left  = np.array([fleet.x + tri_size * 0.65 * math.cos(left_angle),
                                   fleet.y + tri_size * 0.65 * math.sin(left_angle)])
                right = np.array([fleet.x + tri_size * 0.65 * math.cos(right_angle),
                                   fleet.y + tri_size * 0.65 * math.sin(right_angle)])

                triangle = Polygon(
                    [tip, left, right],
                    closed=True,
                    color=f_color,
                    alpha=0.9,
                    zorder=5,
                )
                ax.add_patch(triangle)

                # Ship count label at the centroid of the triangle
                centroid_x = (tip[0] + left[0] + right[0]) / 3.0
                centroid_y = (tip[1] + left[1] + right[1]) / 3.0
                ax.text(
                    centroid_x, centroid_y,
                    str(int(fleet.ships)),
                    color=COLOR_TEXT,
                    fontsize=20,
                    ha='center', va='center',
                    fontweight='bold',
                    zorder=6,
                )

        # ---- HUD ----
        p1_ships = int(self._total_ships(self.PLAYER_1))
        p2_ships = int(self._total_ships(self.PLAYER_2))
        p1_prod  = int(self._total_production(self.PLAYER_1))
        p2_prod  = int(self._total_production(self.PLAYER_2))
        hud = (
            f'Step {self._step_count}   '
            f'P1 \u25cf  ships={p1_ships} prod={p1_prod}   '
            f'P2 \u25cf  ships={p2_ships} prod={p2_prod}'
        )
        ax.set_title(hud, color=COLOR_TEXT, fontsize=20, pad=4)

        # ---- capture ----
        fig.tight_layout(pad=0.2)
        fig.canvas.draw()
        
        rgba = np.asarray(fig.canvas.buffer_rgba())

        plt.close(fig)

        return rgba

    def save_render_output(
        self,
        replay_name_suffix: str = '',
        replay_path: Optional[str] = None,
        format: str = 'gif',
    ) -> None:
        """
        Overview:
            Save the accumulated rendered frames to a GIF or MP4 file.
        Arguments:
            - replay_name_suffix (:obj:`str`): Suffix appended to the filename.
            - replay_path (:obj:`str`): Directory to save the file. Defaults to current directory.
            - format (:obj:`str`): 'gif' or 'mp4'.
        """
        if not self.frames:
            logging.warning('save_render_output called but no frames were captured.')
            return

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # At the end of the episode, save the frames.
        if replay_name_suffix == '':
            if replay_path is None:
                filename = f'galcon_{os.getpid()}_{timestamp}.{format}'
            else:
                os.makedirs(replay_path, exist_ok=True)
                filename = os.path.join(
                    replay_path,
                    f'galcon_{os.getpid()}_{timestamp}.{format}'
                )
        else:
            if replay_path is None:
                filename = f'galcon_{replay_name_suffix}_{os.getpid()}_{timestamp}.{format}'
            else:
                os.makedirs(replay_path, exist_ok=True)
                filename = os.path.join(replay_path, f'galcon_{replay_name_suffix}_{os.getpid()}_{timestamp}.{format}')

        if format == 'gif':
            imageio.mimsave(
                filename,
                self.frames,
                format='GIF',
                duration=int(round(self.tick_seconds * 1000)),
                loop=0,
            )
        elif format == 'mp4':
            imageio.mimsave(
                filename,
                self.frames,
                fps=int(round(1.0 / self.tick_seconds)),
                codec='libx264',
                pixelformat='yuv420p',
            )
        else:
            raise ValueError(f'Unsupported format: {format}')

        logging.info(f'Galcon replay saved to {filename}')
        self.frames = []

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def current_player_index(self) -> int:
        return self.players.index(self._current_player)

    @property
    def next_player(self) -> int:
        return self.players[0] if self._current_player == self.players[1] else self.players[1]

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return 'LightZero Galcon Env'

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.battle_mode = 'eval_mode'
        return [cfg for _ in range(evaluator_env_num)]

    _ACTION_REFLECT_PERM_CACHE: Dict[Tuple[int, int, int, bool, bool], np.ndarray] = {}

    @classmethod
    def _get_action_reflection_perm(
        cls,
        grid_h: int,
        grid_w: int,
        num_actions: int,
        reflect_x: bool,
        reflect_y: bool,
    ) -> np.ndarray:
        """Return a cached action-index permutation for reflecting Galcon actions.

        Supports arbitrary grid dimensions ``(grid_h, grid_w)`` and both planet-only
        (``N + 1``) and planet+redirect (``2N + 1``) action spaces, where
        ``N = (grid_w * grid_h) ** 2``.
        """
        key = (int(grid_h), int(grid_w), int(num_actions), bool(reflect_x), bool(reflect_y))
        cached = cls._ACTION_REFLECT_PERM_CACHE.get(key)
        if cached is not None:
            return cached

        grid_cell_count = int(grid_w) * int(grid_h)
        pair_action_count = grid_cell_count * grid_cell_count
        if int(num_actions) == pair_action_count + 1:
            has_redirect_layer = False
        elif int(num_actions) == 2 * pair_action_count + 1:
            has_redirect_layer = True
        else:
            raise ValueError(
                f"Unexpected Galcon action_space_size={num_actions} for grid ({grid_h}, {grid_w}): "
                f"expected {pair_action_count + 1} or {2 * pair_action_count + 1}."
            )

        if not reflect_x and not reflect_y:
            perm = np.arange(int(num_actions), dtype=np.int64)
        else:
            shifted = np.arange(pair_action_count, dtype=np.int64)
            source_cell = shifted // grid_cell_count
            target_cell = shifted % grid_cell_count

            source_y = source_cell // grid_w
            source_x = source_cell % grid_w
            target_y = target_cell // grid_w
            target_x = target_cell % grid_w

            if reflect_x:
                source_x = grid_w - 1 - source_x
                target_x = grid_w - 1 - target_x
            if reflect_y:
                source_y = grid_h - 1 - source_y
                target_y = grid_h - 1 - target_y

            new_source_cell = source_y * grid_w + source_x
            new_target_cell = target_y * grid_w + target_x
            perm = np.zeros(int(num_actions), dtype=np.int64)
            base_perm = new_source_cell * grid_cell_count + new_target_cell + 1
            perm[1:pair_action_count + 1] = base_perm
            if has_redirect_layer:
                perm[pair_action_count + 1:] = base_perm + pair_action_count

        perm.setflags(write=False)
        cls._ACTION_REFLECT_PERM_CACHE[key] = perm
        return perm

    @classmethod
    def _reflect_obs_channels_inplace(cls, obs: np.ndarray, reflect_x: bool, reflect_y: bool) -> None:
        """Invert planet local x/y and fleet target x/y coordinate channels in-place on ``obs`` ``(..., C, H, W)``."""
        C = int(obs.shape[-3])
        planet_channels = cls.PLANET_CHANNEL_COUNT
        landing_channels = len(cls.LANDING_BUCKETS) * cls.LANDING_FEATURE_COUNT_PER_BUCKET
        fleet_features = cls.FLEET_FEATURE_COUNT
        fleet_channels = C - planet_channels - landing_channels
        if fleet_channels < 0 or fleet_channels % (2 * fleet_features) != 0:
            raise ValueError(
                f"Unexpected Galcon observation channel count C={C} "
                f"(planet={planet_channels}, landing={landing_channels}, "
                f"fleet_features={fleet_features})"
            )
        slot_count_per_side = fleet_channels // (2 * fleet_features)

        # Planet channels: local x/y offsets within cell if any planet info exists in channels 2..planet_channels
        planet_mask = np.sum(obs[..., 2:planet_channels, :, :], axis=-3) > 0
        if reflect_x:
            lx = obs[..., cls.PLANET_LOCAL_X_CHANNEL, :, :]
            np.subtract(1.0, lx, out=lx)
            lx *= planet_mask
        if reflect_y:
            ly = obs[..., cls.PLANET_LOCAL_Y_CHANNEL, :, :]
            np.subtract(1.0, ly, out=ly)
            ly *= planet_mask

        # Fleet channels (both friendly and enemy slots): target world x/y normalized in [0, 1]
        if slot_count_per_side > 0:
            fleet_end = planet_channels + 2 * slot_count_per_side * fleet_features
            fleet_view = obs[..., planet_channels:fleet_end, :, :]
            # Channel offset +3 is ships (> 0 iff a fleet is present in this slot)
            fleet_mask = fleet_view[..., 3::fleet_features, :, :] > 0
            if reflect_x:
                tx = fleet_view[..., 0::fleet_features, :, :]
                np.subtract(1.0, tx, out=tx)
                tx *= fleet_mask
            if reflect_y:
                ty = fleet_view[..., 1::fleet_features, :, :]
                np.subtract(1.0, ty, out=ty)
                ty *= fleet_mask

    @classmethod
    def _reflect_obs_array(cls, obs: np.ndarray, reflect_x: bool, reflect_y: bool) -> np.ndarray:
        """Reflect an observation tensor of shape ``(..., C, H, W)`` across X and/or Y.

        Dynamically infers fleet slot count from ``C`` so it remains valid across any
        observation channel configuration (planet + fleet slots + landing buckets).
        """
        if obs.ndim < 3:
            raise ValueError(f"Expected obs with ndim >= 3 (..., C, H, W), got shape {obs.shape}")

        if not reflect_x and not reflect_y:
            # Validate channel count even on identity reflection
            C = int(obs.shape[-3])
            fleet_channels = C - cls.PLANET_CHANNEL_COUNT - len(cls.LANDING_BUCKETS) * cls.LANDING_FEATURE_COUNT_PER_BUCKET
            if fleet_channels < 0 or fleet_channels % (2 * cls.FLEET_FEATURE_COUNT) != 0:
                raise ValueError(f"Unexpected Galcon observation channel count C={C}")
            return np.array(obs, copy=True)

        # A. Flip spatial grids (copy to ensure contiguous positive-stride array)
        out = obs
        if reflect_x:
            out = np.flip(out, axis=-1)
        if reflect_y:
            out = np.flip(out, axis=-2)
        out = np.ascontiguousarray(out)

        # B. Invert x/y coordinate channels on occupied cells in-place
        cls._reflect_obs_channels_inplace(out, reflect_x=reflect_x, reflect_y=reflect_y)
        return out

    @staticmethod
    def _remap_action_array(perm: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Remap discrete action IDs via ``perm``, preserving negative padding sentinels (e.g. -1)."""
        arr = np.asarray(actions, dtype=np.int64)
        valid = arr >= 0
        if np.all(valid):
            return perm[arr]
        safe = np.where(valid, arr, 0)
        return np.where(valid, perm[safe], arr)

    @classmethod
    def reflect_batch(
        cls,
        obs_batch: np.ndarray,
        action_batch: np.ndarray,
        sampled_actions_batch: Optional[np.ndarray],
        reflect_x_mask: np.ndarray,
        reflect_y_mask: np.ndarray,
        image_channel: Optional[int] = None,
        action_space_size: Optional[int] = None,
        inplace: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Overview:
            Vectorized on-the-fly X/Y symmetry reflection for a sampled training mini-batch.
            Returns reflected ``(obs_batch, action_batch, sampled_actions_batch)`` arrays.
            When ``inplace=False`` (default), returns fresh arrays without mutating caller inputs;
            when ``inplace=True``, mutates freshly allocated caller arrays in-place to avoid
            extra full-batch copies on the hot sampling path.

            Supports ``obs_batch`` shaped either ``(B, T * C, H, W)`` (stacked frames + unroll
            steps as produced by ``SampledEfficientZeroGameBuffer``) or ``(B, T, C, H, W)``,
            with arbitrary channel count ``C`` (dynamically infers fleet slots) and arbitrary
            grid size ``(H, W)`` / planet-only or redirect action space.
        """
        rx = np.asarray(reflect_x_mask, dtype=np.bool_)
        ry = np.asarray(reflect_y_mask, dtype=np.bool_)
        obs_in = np.asarray(obs_batch)
        act_in = np.asarray(action_batch)
        sa_in = None if sampled_actions_batch is None else np.asarray(sampled_actions_batch)

        B = int(obs_in.shape[0])
        if rx.shape != (B,) or ry.shape != (B,):
            raise ValueError(
                f"reflect_x_mask {rx.shape} and reflect_y_mask {ry.shape} must both have shape ({B},)"
            )

        if not np.any(rx) and not np.any(ry):
            if inplace:
                return obs_in, act_in, sa_in
            return (
                np.array(obs_in, copy=True),
                np.array(act_in, copy=True),
                None if sa_in is None else np.array(sa_in, copy=True),
            )

        was_4d = (obs_in.ndim == 4)
        obs_work = obs_in if (inplace and obs_in.flags.writeable) else np.array(obs_in, copy=True)
        if was_4d:
            _, total_c, H, W = obs_work.shape
            C = int(image_channel) if image_channel is not None else int(total_c)
            if C <= 0 or total_c % C != 0:
                raise ValueError(
                    f"obs_batch channel dim {total_c} is not divisible by image_channel={C}"
                )
            T = total_c // C
            obs_5d = obs_work.reshape(B, T, C, H, W)
        elif obs_work.ndim == 5:
            _, T, C, H, W = obs_work.shape
            obs_5d = obs_work
        else:
            raise ValueError(f"Expected 4D or 5D obs_batch, got shape {obs_in.shape}")

        pair_action_count = (int(W) * int(H)) ** 2
        num_actions = int(action_space_size) if action_space_size is not None else (pair_action_count + 1)

        act_out = act_in if (inplace and act_in.flags.writeable) else np.array(act_in, copy=True)
        sa_out = None if sa_in is None else (
            sa_in if (inplace and sa_in.flags.writeable) else np.array(sa_in, copy=True)
        )

        # Flip + invert coordinate channels per orientation group.
        # Uses zero-scatter contiguous slice views whenever orientation rows form a contiguous block.
        for do_x, do_y in ((True, False), (False, True), (True, True)):
            idx = np.flatnonzero((rx == do_x) & (ry == do_y))
            if idx.size == 0:
                continue
            is_contiguous = (int(idx[-1]) - int(idx[0]) + 1 == int(idx.size))
            if is_contiguous:
                sel = slice(int(idx[0]), int(idx[-1]) + 1)
                view = obs_5d[sel]
                if do_x and do_y:
                    view[...] = view[:, :, :, ::-1, ::-1].copy()
                elif do_x:
                    view[...] = view[:, :, :, :, ::-1].copy()
                else:
                    view[...] = view[:, :, :, ::-1, :].copy()
                cls._reflect_obs_channels_inplace(view, reflect_x=do_x, reflect_y=do_y)
            else:
                sel = idx
                if do_x and do_y:
                    sub = obs_5d[idx, :, :, ::-1, ::-1]
                elif do_x:
                    sub = obs_5d[idx, :, :, :, ::-1]
                else:
                    sub = obs_5d[idx, :, :, ::-1, :]
                cls._reflect_obs_channels_inplace(sub, reflect_x=do_x, reflect_y=do_y)
                obs_5d[idx] = sub

            perm = cls._get_action_reflection_perm(H, W, num_actions, reflect_x=do_x, reflect_y=do_y)
            act_out[sel] = cls._remap_action_array(perm, act_out[sel]).astype(act_out.dtype, copy=False)
            if sa_out is not None:
                sa_out[sel] = cls._remap_action_array(perm, sa_out[sel]).astype(sa_out.dtype, copy=False)

        obs_out = obs_5d.reshape(obs_in.shape) if was_4d else obs_5d
        return obs_out, act_out, sa_out

    @classmethod
    def reflect_game_segment(cls, game_segment: Any, reflect_x: bool, reflect_y: bool) -> Any:
        """
        Overview:
            Return a deep-copied GameSegment reflected across the x-axis (left-right),
            y-axis (top-bottom), or both. Updates spatial grids, relative cell/world
            coordinate channels for planets and fleets, actions, action masks, and
            Sampled EfficientZero root_sampled_actions.
        """
        reflected = copy.deepcopy(game_segment)

        obs_segment = np.asarray(reflected.obs_segment)
        # obs_segment shape: (T, C, H, W)
        _, _, H, W = obs_segment.shape

        reflected.obs_segment = cls._reflect_obs_array(obs_segment, reflect_x=reflect_x, reflect_y=reflect_y)

        # C. Reflect actions, action masks, and root_sampled_actions
        # Supports both planet-only (N + 1) and planet+redirect (2N + 1) action spaces
        grid_cell_count = W * H
        pair_action_count = grid_cell_count * grid_cell_count
        mask_len = None
        if (
            hasattr(reflected, 'action_mask_segment')
            and reflected.action_mask_segment is not None
            and len(reflected.action_mask_segment) > 0
        ):
            mask_len = len(reflected.action_mask_segment[0])
        elif getattr(reflected, 'action_space_size', None) is not None:
            mask_len = int(reflected.action_space_size)
        has_redirect_layer = (mask_len == 2 * pair_action_count + 1)
        num_actions = (2 * pair_action_count + 1) if has_redirect_layer else (pair_action_count + 1)

        perm = cls._get_action_reflection_perm(H, W, num_actions, reflect_x=reflect_x, reflect_y=reflect_y)

        if hasattr(reflected, 'action_segment') and reflected.action_segment is not None:
            reflected.action_segment = cls._remap_action_array(perm, reflected.action_segment)

        if hasattr(reflected, 'action_mask_segment') and reflected.action_mask_segment is not None:
            if isinstance(reflected.action_mask_segment, np.ndarray):
                reflected.action_mask_segment = reflected.action_mask_segment[:, perm]
            else:
                reflected.action_mask_segment = [mask[perm] for mask in reflected.action_mask_segment]

        if hasattr(reflected, 'root_sampled_actions') and reflected.root_sampled_actions is not None:
            if isinstance(reflected.root_sampled_actions, np.ndarray):
                reflected.root_sampled_actions = cls._remap_action_array(perm, reflected.root_sampled_actions)
            else:
                reflected.root_sampled_actions = [
                    cls._remap_action_array(perm, actions) for actions in reflected.root_sampled_actions
                ]

        return reflected

    @staticmethod
    def validate_config(cfg: dict) -> None:
        """
        Overview:
            Validate Galcon training config invariants at startup.

            Guards against accidental desync between ``env.max_episode_steps`` and
            ``policy.td_steps`` / ``policy.game_segment_length`` (e.g. when a wrapper
            script overrides ``env.max_episode_steps`` without updating the policy
            fields, which would silently enable n-step bootstrapping and/or split
            episodes across segments).

            Set ``cfg.policy.allow_episode_horizon_mismatch = True`` to opt into
            intentional bootstrapping or multi-segment experiments.
        """
        env_cfg = cfg.get('env', {}) if hasattr(cfg, 'get') else getattr(cfg, 'env', {})
        policy_cfg = cfg.get('policy', {}) if hasattr(cfg, 'get') else getattr(cfg, 'policy', {})

        max_episode_steps = (
            env_cfg.get('max_episode_steps')
            if hasattr(env_cfg, 'get')
            else getattr(env_cfg, 'max_episode_steps', None)
        )
        if max_episode_steps is None:
            return
        max_episode_steps = int(max_episode_steps)

        td_steps = policy_cfg.get('td_steps') if hasattr(policy_cfg, 'get') else getattr(policy_cfg, 'td_steps', None)
        seg_len = (
            policy_cfg.get('game_segment_length')
            if hasattr(policy_cfg, 'get')
            else getattr(policy_cfg, 'game_segment_length', None)
        )

        mismatches = []
        if td_steps is not None and int(td_steps) != max_episode_steps:
            mismatches.append(f'policy.td_steps={td_steps} != env.max_episode_steps={max_episode_steps}')
        if seg_len is not None and int(seg_len) != max_episode_steps:
            mismatches.append(f'policy.game_segment_length={seg_len} != env.max_episode_steps={max_episode_steps}')

        if not mismatches:
            return

        allow_mismatch = bool(
            policy_cfg.get('allow_episode_horizon_mismatch', False)
            if hasattr(policy_cfg, 'get')
            else getattr(policy_cfg, 'allow_episode_horizon_mismatch', False)
        )
        detail = '; '.join(mismatches)
        if allow_mismatch:
            logging.info(
                f'[Galcon] Proceeding with intentional episode-horizon mismatch '
                f'(allow_episode_horizon_mismatch=True): {detail}'
            )
            return

        raise ValueError(
            f'Galcon episode-horizon mismatch detected ({detail}). '
            f'Update policy.td_steps and policy.game_segment_length to match env.max_episode_steps '
            f'({max_episode_steps}) to keep single-segment Monte Carlo outcomes, or set '
            f'policy.allow_episode_horizon_mismatch=True if this mismatch is intentional '
            f'(e.g., for an n-step bootstrapping or multi-segment experiment).'
        )
