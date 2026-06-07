import copy
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gymnasium import spaces

from zoo.board_games.galcon.envs.rule_bot import GalconFixedPolicyBot, GalconRandomBot


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

        Phase 2 mechanics:
        - action 0 is pass.
        - actions 1..N*N are source-target send actions.
        - send action sends a fixed ratio of source ships.
        - one player action advances one simulation tick.
        - timeout winner is decided by production, then total ships.
    """

    PLAYER_1 = 1
    PLAYER_2 = 2
    NEUTRAL = 0

    # production=100 means 120 ships/min = 2 ships/sec => production / 50 ships/sec.
    PRODUCTION_TO_SHIPS_PER_SECOND_DIVISOR = 50.0

    config = dict(
        env_id='Galcon',
        battle_mode='self_play_mode',
        bot_action_type='random',  # {'random', 'fixed'}
        num_planets=8,
        min_send_ships=1,
        send_ratio=0.5,
        tick_seconds=0.25,
        fleet_speed=40.0,
        # TODO: update radius
        fleet_radius=5,
        max_episode_steps=200,
        map_seed=None,
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        manager=dict(shared_memory=False),
        channel_last=False,
        scale=True,
        render_mode=None,
        replay_path=None,
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        prob_random_action_in_bot=0.,
        stop_value=1,
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
        self.total_num_actions = self.num_planets * self.num_planets + 1
        self.min_send_ships = float(self.cfg.min_send_ships)
        self.send_ratio = float(self.cfg.send_ratio)
        self.tick_seconds = float(self.cfg.tick_seconds)
        self.fleet_speed = float(self.cfg.fleet_speed)
        self.fleet_radius = float(self.cfg.fleet_radius)
        self.max_episode_steps = int(self.cfg.max_episode_steps)
        self.map_seed = self.cfg.get('map_seed', None)

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

        # Placeholder grid observation. Exact channels/grid spec will be finalized later.
        self._observation_shape = (1, self.num_planets, self.num_planets)
        self._observation_space = spaces.Dict(
            {
                'observation': spaces.Box(low=0, high=1, shape=self._observation_shape, dtype=np.float32),
                'action_mask': spaces.Box(low=0, high=1, shape=(self.total_num_actions,), dtype=np.int8),
                'current_player_index': spaces.Discrete(2),
                'to_play': spaces.Box(low=-1, high=2, shape=(), dtype=np.int8),
            }
        )

        self.prob_random_agent = self.cfg.prob_random_agent
        self.prob_expert_agent = self.cfg.prob_expert_agent
        self.prob_random_action_in_bot = self.cfg.prob_random_action_in_bot
        self.bot_action_type = self.cfg.bot_action_type

        if self.bot_action_type == 'random':
            self.bot = GalconRandomBot(self)
        elif self.bot_action_type == 'fixed':
            self.bot = GalconFixedPolicyBot(self)
        else:
            raise ValueError(f'Unsupported Galcon bot_action_type: {self.bot_action_type}')

        self.planets: List[Planet] = []
        self.fleets: List[Fleet] = []
        self.reset()

    def reset(self, start_player_index: int = 0, replay_name_suffix: Optional[str] = None) -> dict:
        self._step_count = 0
        self.players = [self.PLAYER_1, self.PLAYER_2]
        self._current_player = self.players[start_player_index]
        self._next_fleet_id = self.num_planets

        self._generate_map()

        return self.observe()

    def _generate_map(self) -> None:
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

        if self.map_seed is not None:
            rng = np.random.RandomState(self.map_seed)
        elif hasattr(self, '_seed'):
            rng = np.random.RandomState(self._seed)
        else:
            rng = np.random.RandomState()

        planets = [
            Planet(
                id=0,
                owner=self.PLAYER_1,
                ships=100.0,
                x=-180.0,
                y=0.0,
                production=100.0,
                radius=24.0,
                neutral=False,
            ),
            Planet(
                id=1,
                owner=self.PLAYER_2,
                ships=100.0,
                x=180.0,
                y=0.0,
                production=100.0,
                radius=24.0,
                neutral=False,
            ),
        ]

        next_planet_id = 2
        neutral_pairs = (self.num_planets - 2) // 2
        for _ in range(neutral_pairs):
            x = (rng.random_sample() * 2.0 - 1.0) * 200.0
            y = (rng.random_sample() * 2.0 - 1.0) * 120.0
            neutral_ships = rng.random_sample() * 50.0
            production = rng.random_sample() * 85.0 + 15.0
            radius = self._radius_from_production(production)

            planets.append(
                Planet(
                    id=next_planet_id,
                    owner=self.NEUTRAL,
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
                    owner=self.NEUTRAL,
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

    @staticmethod
    def _radius_from_production(production: float) -> float:
        return (production * 12.0 / 5.0 + 168.0) / 17.0

    def step(self, action: int) -> BaseEnvTimestep:
        if self.battle_mode == 'self_play_mode':
            return self._player_step(action, flag='agent')

        elif self.battle_mode in ['play_with_bot_mode', 'eval_mode']:
            timestep_player1 = self._player_step(action, flag='agent')
            if timestep_player1.done:
                timestep_player1.obs['to_play'] = -1
                return timestep_player1

            bot_action = self.bot_action()
            timestep_player2 = self._player_step(bot_action, flag='bot')

            timestep_player2.info['eval_episode_return'] = -timestep_player2.reward
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)
            timestep_player2.obs['to_play'] = -1

            return timestep_player2

        else:
            raise ValueError(f'Unsupported battle_mode: {self.battle_mode}')

    def _player_step(self, action: int, flag: str) -> BaseEnvTimestep:
        if action not in self.legal_actions:
            action = self.pass_action

        self._apply_action(action)
        self._advance_one_tick()
        self._step_count += 1

        done, winner = self.get_done_winner()

        if done:
            if winner == -1:
                reward = np.array(0).astype(np.float32)
            elif winner == self._current_player:
                reward = np.array(1).astype(np.float32)
            else:
                reward = np.array(-1).astype(np.float32)
        else:
            reward = np.array(0).astype(np.float32)

        info = self._get_state_info(winner if done else -1)
        if done:
            info['eval_episode_return'] = reward

        self._current_player = self.next_player
        obs = self.observe()

        return BaseEnvTimestep(obs, reward, done, info)

    def _apply_action(self, action: int) -> None:
        source_id, target_id = self.decode_action(action)
        if source_id is None and target_id is None:
            return

        source = self.planets[source_id]
        target = self.planets[target_id]

        if source.owner != self._current_player:
            return
        if source_id == target_id:
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
                radius=self.fleet_radius,
            )
        )
        self._next_fleet_id += 1

    def _advance_one_tick(self) -> None:
        self._add_production()
        self._move_fleets()

    def _add_production(self) -> None:
        for planet in self.planets:
            if planet.neutral:
                continue
            if planet.owner == self.NEUTRAL:
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

    def decode_action(self, action: int) -> Tuple[Optional[int], Optional[int]]:
        if action == self.pass_action:
            return None, None

        shifted_action = action - 1
        source = shifted_action // self.num_planets
        target = shifted_action % self.num_planets
        return source, target

    @property
    def legal_actions(self) -> List[int]:
        legal = [self.pass_action]
        for action in range(1, self.total_num_actions):
            source, target = self.decode_action(action)
            if source == target:
                continue
            if self.planets[source].owner != self._current_player:
                continue
            if self.planets[source].ships < self.min_send_ships:
                continue
            legal.append(action)

        return legal

    def observe(self) -> dict:
        action_mask = np.zeros(self.total_num_actions, dtype=np.int8)
        for action in self.legal_actions:
            action_mask[action] = 1

        observation = self.current_state()

        if self.battle_mode in ['play_with_bot_mode', 'eval_mode']:
            to_play = -1
        else:
            to_play = self._current_player

        return {
            'observation': observation,
            'action_mask': action_mask,
            'current_player_index': self.current_player_index,
            'to_play': to_play,
        }

    def current_state(self) -> np.ndarray:
        """
        Placeholder grid observation.

        Proper grid-based planet/fleet channels will be designed in the model/observation phase.
        """
        obs = np.zeros(self._observation_shape, dtype=np.float32)

        for planet in self.planets:
            idx = planet.id
            owner_value = planet.owner / 2.0 if self.scale else planet.owner
            obs[0, idx, idx] = owner_value

        return obs

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

    def _total_production(self, player: int) -> float:
        return float(sum(p.production for p in self.planets if p.owner == player))

    def _total_ships(self, player: int) -> float:
        planet_ships = sum(p.ships for p in self.planets if p.owner == player)
        fleet_ships = sum(f.ships for f in self.fleets if f.owner == player)
        return float(planet_ships + fleet_ships)

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
        return self.bot.get_action()

    def action_to_string(self, action: int) -> str:
        source, target = self.decode_action(action)
        if source is None and target is None:
            return 'Pass'
        return f'Send {self.send_ratio:.0%} ships from planet {source} to planet {target}'

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(seed)

    def close(self) -> None:
        pass

    def render(self, mode: Optional[str] = None) -> None:
        print(f'Galcon step={self._step_count}, current_player={self._current_player}')
        print('planets:', self.planets)
        print('fleets:', self.fleets)

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