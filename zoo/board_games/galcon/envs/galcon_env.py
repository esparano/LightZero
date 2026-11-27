import copy
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

from zoo.board_games.galcon.envs.rule_bot import GalconFixedPolicyBot, GalconRandomBot, GalconPassBot


@dataclass
class Planet:
    id: int
    owner: int
    ships: float
    x: float
    y: float
    production: float
    radius: float
    neutral: bool


@dataclass
class Fleet:
    id: int
    owner: int
    ships: float
    x: float
    y: float
    source: int
    target: int
    radius: float


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
    PRODUCTION_TO_SHIPS_PER_SECOND_DIVISOR = 50.0
    # TODO: For now, fleet radius is just a constant.
    DUMMY_FLEET_RADIUS = 10.0
    SHIP_RADIUS = 6.0
    # Leave an extra bit of space between planets (beyond 2 * SHIP_RADIUS) so that ships can always fit through
    PLANETS_SETTLE_DELTA = 0.5
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
        bot_action_type='random',  # {'random', 'fixed'}
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

        if self.bot_action_type == 'random':
            self.bot = GalconRandomBot(self)
        elif self.bot_action_type == 'fixed':
            self.bot = GalconFixedPolicyBot(self)
        elif self.bot_action_type == 'pass':
            self.bot = GalconPassBot(self)
        else:
            raise ValueError(f'Unsupported Galcon bot_action_type: {self.bot_action_type}')

        self.debug_legal_actions = self.cfg.debug_legal_actions
        self.debug_observation = self.cfg.debug_observation

        self.planets: List[Planet] = []
        self.fleets: List[Fleet] = []
        self._current_map_seed: Optional[int] = None
        self.reset()

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
        return (production * 12.0 / 5.0 + 168.0) / 17.0

    def step(self, action: int) -> BaseEnvTimestep:
        if self.battle_mode == 'self_play_mode':
            return self._player_step(action, flag='agent')
        elif self.battle_mode in ['play_with_bot_mode', 'eval_mode']:
            timestep_player1 = self._player_step(action, flag='agent')
            if timestep_player1.done:
                # # Player 1 has finished, so player 2 will play next
                # timestep_player1.obs['to_play'] = self.PLAYER_2
                return timestep_player1

            bot_action = self.bot_action()
            timestep_player2 = self._player_step(bot_action, flag='bot')
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

    def _player_step(self, action: int, flag: str) -> BaseEnvTimestep:
        if action not in self.legal_actions:
            action = self.pass_action

        self._apply_action(action)
        self._advance_one_tick()
        self._step_count += 1

        winner_by_elimination = self._winner_by_elimination()
        done, winner = self.get_done_winner()

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
            reward = np.array(0, dtype=np.float32)
            eval_return = None

        info = self._get_state_info(winner if done else -1)

        self._current_player = self.next_player
        
        obs = self.observe()
        
        # Render the new step.
        if self.render_mode is not None:
            self.render(self.render_mode)
        if done:
            info['eval_episode_return'] = eval_return
            if self.render_mode == 'image_savefile_mode':
                self.save_render_output(replay_name_suffix=self.replay_name_suffix, replay_path=self.replay_path,
                                        format=self.replay_format)            

        return BaseEnvTimestep(obs, reward, done, info)

    def _fleets_at_cell(self, grid_x: int, grid_y: int, owner: Optional[int] = None) -> List[Fleet]:
        matched = []
        for fleet in self.fleets:
            if owner is not None and fleet.owner != owner:
                continue
            if self._world_to_grid(fleet.x, fleet.y) == (grid_x, grid_y):
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
            for fleet in self._fleets_at_cell(source_x, source_y, owner=self._current_player):
                fleet.target = target.id
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
        if source.ships < self.min_send_ships:
            return

        ships_to_send = min(source.ships, source.ships * self.send_ratio)
        if ships_to_send <= 0:
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

    def _advance_one_tick(self) -> None:
        # TODO: some day, move fleets and adjust production gradually (produce in between fleets landing)
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
        grid_x = int(math.floor((x - (-self.grid_max_x)) / self.grid_square_size))
        grid_y = int(math.floor((y - (-self.grid_max_y)) / self.grid_square_size))
        grid_x = int(np.clip(grid_x, 0, self.grid_width - 1))
        grid_y = int(np.clip(grid_y, 0, self.grid_height - 1))
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
        cell_min_x = -self.grid_max_x + grid_x * self.grid_square_size
        cell_min_y = -self.grid_max_y + grid_y * self.grid_square_size
        normalized_x = self._linear_encode(x - cell_min_x, self.grid_square_size)
        normalized_y = self._linear_encode(y - cell_min_y, self.grid_square_size)
        return normalized_x, normalized_y

    def _normalize_world_x(self, x: float) -> float:
        return self._linear_encode(x - (-self.grid_max_x), 2 * self.grid_max_x)

    def _normalize_world_y(self, y: float) -> float:
        return self._linear_encode(y - (-self.grid_max_y), 2 * self.grid_max_y)

    def _linear_encode(self, x: float, max_expected_value: float) -> float:
        return float(np.clip(x / max_expected_value, 0.0, 1.0))

    # (x + 1) / (max_expected_x + 1). max_expected_x is the largest value the network is likely to ever see.
    def _log_encode(self, value: float, max_expected_value: float) -> float:
        value = max(0.0, float(value))
        max_expected_value = max(1.0, float(max_expected_value))
        return float(np.log(value + 1.0) / np.log(max_expected_value + 1.0))

    def _encode_ships(self, ships: float) -> float:
        return self._linear_encode(ships, self.max_expected_ships)

    def _encode_eta(self, eta_seconds: float) -> float:
        return self._linear_encode(eta_seconds, self.max_expected_eta)

    def _fleet_eta_seconds(self, fleet: Fleet) -> float:
        target = self.planets[fleet.target]
        distance_to_target_edge = max(0.0, self._distance(fleet.x, fleet.y, target.x, target.y) - target.radius)
        return distance_to_target_edge / max(self.fleet_speed, 1e-6)

    def _planet_at_cell(self, grid_x: int, grid_y: int, owner: Optional[int] = None) -> Optional[Planet]:
        matching_planets = []
        for planet in self.planets:
            planet_grid_x, planet_grid_y = self._world_to_grid(planet.x, planet.y)
            if planet_grid_x != grid_x or planet_grid_y != grid_y:
                continue
            if owner is not None and planet.owner != owner:
                continue
            matching_planets.append(planet)

        if len(matching_planets) == 0:
            return None

        if len(matching_planets) > 1: 
            logging.warning(
                'WARNING: More than one planet found in grid cell. This should not happen for grid cell size <= 25.'
            )

        # Deterministic tie-breaker if multiple planets share a grid square.
        matching_planets.sort(key=lambda p: p.id)
        return matching_planets[0]

    def encode_action(
        self, source_x: int, source_y: int, target_x: int, target_y: int, is_redirect: bool = False
    ) -> int:
        source_cell = self._grid_to_cell_index(source_x, source_y)
        target_cell = self._grid_to_cell_index(target_x, target_y)
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

        source_x, source_y = self._cell_index_to_grid(source_cell)
        target_x, target_y = self._cell_index_to_grid(target_cell)
        return action_type, source_x, source_y, target_x, target_y

    def decode_action(self, action: int) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        _, source_x, source_y, target_x, target_y = self.decode_action_with_type(action)
        return source_x, source_y, target_x, target_y

    # TODO: This is expensive and may be called more than once. Good candidate for optimization.
    @property
    def legal_actions(self) -> List[int]:
        legal = [self.pass_action]
        legal_set = {self.pass_action}

        friendly_source_cells = set()
        target_cells = set()

        for planet in self.planets:
            grid_x, grid_y = self._world_to_grid(planet.x, planet.y)
            if planet.owner == self._current_player and planet.ships >= self.min_send_ships:
                friendly_source_cells.add((grid_x, grid_y))
            target_cells.add((grid_x, grid_y))

        # Note: Sorting seems to be not strictly necessary...
        for source_x, source_y in sorted(friendly_source_cells):
            source = self._planet_at_cell(source_x, source_y, owner=self._current_player)
            if source is None:
                continue

            for target_x, target_y in sorted(target_cells):
                target = self._planet_at_cell(target_x, target_y)
                if target is None:
                    continue
                if source.id == target.id:
                    continue

                act = self.encode_action(source_x, source_y, target_x, target_y)
                if act not in legal_set:
                    legal_set.add(act)
                    legal.append(act)

        if self.enable_fleet_redirection:
            friendly_fleet_cells = {}
            for fleet in self.fleets:
                if fleet.owner == self._current_player:
                    cell = self._world_to_grid(fleet.x, fleet.y)
                    friendly_fleet_cells.setdefault(cell, []).append(fleet)

            for (source_x, source_y), cell_fleets in sorted(friendly_fleet_cells.items()):
                for target_x, target_y in sorted(target_cells):
                    target = self._planet_at_cell(target_x, target_y)
                    if target is None:
                        continue
                    # Only legal if at least one friendly fleet in this cell isn't already heading to target
                    if not any(f.target != target.id for f in cell_fleets):
                        continue
                    act = self.encode_action(source_x, source_y, target_x, target_y, is_redirect=True)
                    if act not in legal_set:
                        legal_set.add(act)
                        legal.append(act)

        return legal

    def observe(self) -> dict:
        action_mask = np.zeros(self.total_num_actions, dtype=np.int8)
        for action in self.legal_actions:
            action_mask[action] = 1

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
        obs = np.zeros(self._observation_shape, dtype=np.float32)

        self._encode_planets(obs)
        self._encode_fleets(obs)
        self._encode_landing_schedule(obs)

        return obs

    def _encode_planets(self, obs: np.ndarray) -> None:
        for planet in self.planets:
            grid_x, grid_y = self._world_to_grid(planet.x, planet.y)
            local_x, local_y = self._local_grid_offset(planet.x, planet.y)

            obs[self.PLANET_LOCAL_X_CHANNEL, grid_y, grid_x] = local_x
            obs[self.PLANET_LOCAL_Y_CHANNEL, grid_y, grid_x] = local_y

            if planet.owner == self._current_player:
                obs[self.PLANET_FRIENDLY_SHIPS_CHANNEL, grid_y, grid_x] = float(
                    np.clip(planet.ships / self.max_expected_ships, 0.0, 1.0)
                )
                obs[self.PLANET_FRIENDLY_PRODUCTION_CHANNEL, grid_y, grid_x] = float(
                    np.clip(planet.production / self.max_expected_production, 0.0, 1.0)
                )
            elif planet.owner == self.PLAYER_NEUTRAL:
                obs[self.PLANET_NEUTRAL_SHIPS_CHANNEL, grid_y, grid_x] = float(
                    np.clip(planet.ships / self.max_expected_ships, 0.0, 1.0)
                )
                obs[self.PLANET_NEUTRAL_PRODUCTION_CHANNEL, grid_y, grid_x] = float(
                    np.clip(planet.production / self.max_expected_production, 0.0, 1.0)
                )
            else:
                obs[self.PLANET_ENEMY_SHIPS_CHANNEL, grid_y, grid_x] = float(
                    np.clip(planet.ships / self.max_expected_ships, 0.0, 1.0)
                )
                obs[self.PLANET_ENEMY_PRODUCTION_CHANNEL, grid_y, grid_x] = float(
                    np.clip(planet.production / self.max_expected_production, 0.0, 1.0)
                )

    def _encode_fleets(self, obs: np.ndarray) -> None:
        fleets_by_cell: Dict[Tuple[int, int], List[Fleet]] = {}

        for fleet in self.fleets:
            grid_x, grid_y = self._world_to_grid(fleet.x, fleet.y)
            fleets_by_cell.setdefault((grid_x, grid_y), []).append(fleet)

        fleet_base_channel = self.PLANET_CHANNEL_COUNT

        for (grid_x, grid_y), fleets in fleets_by_cell.items():
            friendly_fleets = [fleet for fleet in fleets if fleet.owner == self._current_player]
            enemy_fleets = [fleet for fleet in fleets if fleet.owner != self._current_player]

            self._encode_top_k_fleets(
                obs=obs,
                grid_x=grid_x,
                grid_y=grid_y,
                fleets=friendly_fleets,
                base_channel=fleet_base_channel,
            )

            enemy_base_channel = fleet_base_channel + self.fleet_slot_count_per_side * self.FLEET_FEATURE_COUNT
            self._encode_top_k_fleets(
                obs=obs,
                grid_x=grid_x,
                grid_y=grid_y,
                fleets=enemy_fleets,
                base_channel=enemy_base_channel,
            )

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
        landing_base_channel = self.PLANET_CHANNEL_COUNT + self.fleet_channel_count

        for planet in self.planets:
            grid_x, grid_y = self._world_to_grid(planet.x, planet.y)
            friendly_by_bucket = np.zeros(len(self.LANDING_BUCKETS), dtype=np.float32)
            enemy_by_bucket = np.zeros(len(self.LANDING_BUCKETS), dtype=np.float32)

            for fleet in self.fleets:
                if fleet.target != planet.id:
                    continue

                bucket_index = self._landing_bucket_index(self._fleet_eta_seconds(fleet))
                if fleet.owner == self._current_player:
                    friendly_by_bucket[bucket_index] += fleet.ships
                else:
                    enemy_by_bucket[bucket_index] += fleet.ships

            for bucket_index in range(len(self.LANDING_BUCKETS)):
                friendly = float(friendly_by_bucket[bucket_index])
                enemy = float(enemy_by_bucket[bucket_index])
                bucket_channel = landing_base_channel + bucket_index * self.LANDING_FEATURE_COUNT_PER_BUCKET

                obs[bucket_channel + 0, grid_y, grid_x] = self._encode_ships(friendly)
                obs[bucket_channel + 1, grid_y, grid_x] = self._encode_ships(enemy)
                obs[bucket_channel + 2, grid_y, grid_x] = self._encode_ships(max(friendly - enemy, 0.0))
                obs[bucket_channel + 3, grid_y, grid_x] = self._encode_ships(max(enemy - friendly, 0.0))

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
        pass

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
            target_planet = self.planets[fleet.target]
            dx = target_planet.x - fleet.x
            dy = target_planet.y - fleet.y
            dist = math.hypot(dx, dy)

            if dist < 1e-6:
                angle = 0.0
            else:
                angle = math.atan2(dy, dx)

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
                color=owner_color(fleet.owner),
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
        T, C, H, W = obs_segment.shape

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

        # A. Flip spatial grids (copy to ensure contiguous positive-stride array)
        if reflect_x:
            obs_segment = np.flip(obs_segment, axis=3)
        if reflect_y:
            obs_segment = np.flip(obs_segment, axis=2)
        obs_segment = np.ascontiguousarray(obs_segment)

        # B. Invert x/y coordinate channels on occupied cells
        # Planet channels: local x/y offsets within cell if any planet info exists in channels 2..7
        planet_mask = np.sum(obs_segment[:, 2:planet_channels], axis=1) > 0  # (T, H, W)
        if reflect_x:
            obs_segment[:, cls.PLANET_LOCAL_X_CHANNEL] = np.where(
                planet_mask, 1.0 - obs_segment[:, cls.PLANET_LOCAL_X_CHANNEL], 0.0
            )
        if reflect_y:
            obs_segment[:, cls.PLANET_LOCAL_Y_CHANNEL] = np.where(
                planet_mask, 1.0 - obs_segment[:, cls.PLANET_LOCAL_Y_CHANNEL], 0.0
            )

        # Fleet channels (both friendly and enemy slots): target world x/y normalized in [0, 1]
        for slot in range(2 * slot_count_per_side):
            base_ch = planet_channels + slot * fleet_features
            # Channel offset +3 is ships (> 0 iff a fleet is present in this slot)
            fleet_mask = obs_segment[:, base_ch + 3] > 0
            if reflect_x:
                obs_segment[:, base_ch + 0] = np.where(fleet_mask, 1.0 - obs_segment[:, base_ch + 0], 0.0)
            if reflect_y:
                obs_segment[:, base_ch + 1] = np.where(fleet_mask, 1.0 - obs_segment[:, base_ch + 1], 0.0)

        reflected.obs_segment = obs_segment

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

        shifted = np.arange(pair_action_count, dtype=np.int64)
        source_cell = shifted // grid_cell_count
        target_cell = shifted % grid_cell_count

        source_y = source_cell // W
        source_x = source_cell % W
        target_y = target_cell // W
        target_x = target_cell % W

        if reflect_x:
            source_x = W - 1 - source_x
            target_x = W - 1 - target_x
        if reflect_y:
            source_y = H - 1 - source_y
            target_y = H - 1 - target_y

        new_source_cell = source_y * W + source_x
        new_target_cell = target_y * W + target_x
        perm = np.zeros(num_actions, dtype=np.int64)
        base_perm = new_source_cell * grid_cell_count + new_target_cell + 1
        perm[1:pair_action_count + 1] = base_perm
        if has_redirect_layer:
            perm[pair_action_count + 1:] = base_perm + pair_action_count

        if hasattr(reflected, 'action_segment') and reflected.action_segment is not None:
            reflected.action_segment = perm[np.asarray(reflected.action_segment, dtype=np.int64)]

        if hasattr(reflected, 'action_mask_segment') and reflected.action_mask_segment is not None:
            if isinstance(reflected.action_mask_segment, np.ndarray):
                reflected.action_mask_segment = reflected.action_mask_segment[:, perm]
            else:
                reflected.action_mask_segment = [mask[perm] for mask in reflected.action_mask_segment]

        if hasattr(reflected, 'root_sampled_actions') and reflected.root_sampled_actions is not None:
            if isinstance(reflected.root_sampled_actions, np.ndarray):
                reflected.root_sampled_actions = perm[reflected.root_sampled_actions.astype(np.int64)]
            else:
                reflected.root_sampled_actions = [
                    perm[np.asarray(actions, dtype=np.int64)] for actions in reflected.root_sampled_actions
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
