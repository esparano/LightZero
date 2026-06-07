import copy
from typing import List, Optional, Tuple

import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gymnasium import spaces

from zoo.board_games.galcon.envs.rule_bot import GalconFixedPolicyBot, GalconRandomBot


@ENV_REGISTRY.register('galcon')
class GalconEnv(BaseEnv):
    """
    Overview:
        A LightZero-compatible Galcon-style two-player environment.

        Phase 1 scaffold only:
        - Fixed small discrete action space: source_planet * num_planets + target_planet.
        - Actions send a fixed ratio of ships.
        - Exact planet/fleet/map/combat mechanics will be filled in later.
    """

    config = dict(
        env_id='Galcon',
        battle_mode='self_play_mode',
        bot_action_type='random',  # {'random', 'fixed'}
        num_planets=8,
        min_send_ships=1,
        send_ratio=0.5,
        tick_seconds=0.25,
        max_episode_steps=200,
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
        # +1 for the "pass" action
        self.total_num_actions = self.num_planets * self.num_planets + 1
        self.min_send_ships = self.cfg.min_send_ships
        self.send_ratio = self.cfg.send_ratio
        self.tick_seconds = self.cfg.tick_seconds
        self.max_episode_steps = self.cfg.max_episode_steps

        self.channel_last = self.cfg.channel_last
        self.scale = self.cfg.scale
        self.battle_mode = self.cfg.battle_mode
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']

        self.players = [1, 2]
        self._current_player = 1
        self._step_count = 0

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

        self.planets = []
        self.fleets = []
        self.reset()

    def reset(self, start_player_index: int = 0, replay_name_suffix: Optional[str] = None) -> dict:
        self._step_count = 0
        self.players = [1, 2]
        self._current_player = self.players[start_player_index]

        self._generate_map()

        return self.observe()

    def _generate_map(self) -> None:
        """
        TODO: Placeholder seeded/random map generation.
        """
        self.planets = [
            {
                'id': i,
                'owner': 0,
                'ships': 0,
                'production': 0,
            }
            for i in range(self.num_planets)
        ]
        self.fleets = []

        if self.num_planets >= 2:
            self.planets[0]['owner'] = 1
            self.planets[0]['ships'] = 10
            self.planets[0]['production'] = 1

            self.planets[1]['owner'] = 2
            self.planets[1]['ships'] = 10
            self.planets[1]['production'] = 1

    def step(self, action: int) -> BaseEnvTimestep:
        if self.battle_mode == 'self_play_mode':
            timestep = self._player_step(action, flag='agent')

            return timestep

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
            action = self.random_action()

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

        info = {}
        if done:
            info['eval_episode_return'] = reward

        self._current_player = self.next_player
        obs = self.observe()

        return BaseEnvTimestep(obs, reward, done, info)

    def _apply_action(self, action: int) -> None:
        """
        Placeholder action application.

        Later:
        - decode source/target
        - send fixed 50% ships
        - create fleet
        """
        source, target = self.decode_action(action)
        if source is None and target is None:
            return

        _ = source, target

    def _advance_one_tick(self) -> None:
        """
        Placeholder time update.

        Later:
        - move fleets
        - resolve arrivals
        - grow planets
        """
        pass

    def decode_action(self, action: int) -> Tuple[int, int]:
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
            if self.planets[source]['owner'] != self._current_player:
                continue
            if self.planets[source]['ships'] < self.min_send_ships:
                continue
            legal.append(action)

        # Keep env usable during scaffolding.
        if len(legal) == 0:
            return [0]

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

        The proper grid-based planet/fleet channels will be designed in a later phase.
        """
        obs = np.zeros(self._observation_shape, dtype=np.float32)

        for planet in self.planets:
            idx = planet['id']
            # TODO: This code doesn't make sense
            owner_value = planet['owner'] / 2.0 if self.scale else planet['owner']
            obs[0, idx, idx] = owner_value

        return obs

    def get_done_winner(self) -> Tuple[bool, int]:
        """
        Placeholder winner logic.

        Later:
        - elimination winner
        - at time limit: higher production wins; if tied, more ships wins; if tied, draw.
        """
        if self._step_count >= self.max_episode_steps:
            return True, self._winner_by_timeout()

        return False, -1

    def _winner_by_timeout(self) -> int:
        p1_production = sum(p['production'] for p in self.planets if p['owner'] == 1)
        p2_production = sum(p['production'] for p in self.planets if p['owner'] == 2)

        if p1_production > p2_production:
            return 1
        if p2_production > p1_production:
            return 2

        p1_ships = sum(p['ships'] for p in self.planets if p['owner'] == 1)
        p2_ships = sum(p['ships'] for p in self.planets if p['owner'] == 2)

        if p1_ships > p2_ships:
            return 1
        if p2_ships > p1_ships:
            return 2

        return -1

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
        # Hack
        return 0 if self._current_player == 1 else 1

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
