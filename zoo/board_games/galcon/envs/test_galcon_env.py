import numpy as np
import pytest
from easydict import EasyDict
from gymnasium import spaces

from zoo.board_games.galcon.envs.galcon_env import GalconEnv


@pytest.mark.envtest
class TestGalconEnv:

    def setup_method(self) -> None:
        self.cfg = EasyDict(
            battle_mode='self_play_mode',
            bot_action_type='random',
            num_planets=8,
            min_send_ships=1,
            send_ratio=0.5,
            tick_seconds=0.25,
            fleet_speed=40.0,
            max_episode_steps=20,
            map_seed=0,
            channel_last=False,
            scale=True,
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            prob_random_action_in_bot=0.,
            render_mode=None,
            replay_path=None,
        )

    def test_reset(self) -> None:
        env = GalconEnv(EasyDict(self.cfg))
        obs = env.reset()

        assert isinstance(obs, dict)
        assert 'observation' in obs
        assert 'action_mask' in obs
        assert 'to_play' in obs
        assert obs['observation'].shape == (1, 8, 8)
        assert obs['action_mask'].shape == (8 * 8 + 1,)

    def test_action_space(self) -> None:
        env = GalconEnv(EasyDict(self.cfg))
        env.reset()

        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 8 * 8 + 1

    def test_random_action_is_legal(self) -> None:
        env = GalconEnv(EasyDict(self.cfg))
        env.reset()

        action = env.random_action()
        assert action in env.legal_actions

    def test_pass_action(self) -> None:
        env = GalconEnv(EasyDict(self.cfg))
        env.reset()

        assert env.pass_action == 0
        assert env.pass_action in env.legal_actions
        assert env.decode_action(env.pass_action) == (None, None)
        assert env.action_to_string(env.pass_action) == 'Pass'

    def test_send_action_creates_fleet(self) -> None:
        env = GalconEnv(EasyDict(self.cfg))
        env.reset()

        action = 2  # shifted action 1 => source=0, target=1 for num_planets=8
        source_ships_before = env.planets[0].ships

        assert action in env.legal_actions

        obs, reward, done, info = env.step(action)

        # The ships have changed due to production in the env.step
        expected_produced_ships = (env.planets[0].production /
                                   env.PRODUCTION_TO_SHIPS_PER_SECOND_DIVISOR * env.tick_seconds)

        assert len(env.fleets) == 1
        assert env.planets[0].ships == source_ships_before * (1 - env.send_ratio) + expected_produced_ships
        assert env.fleets[0].owner == GalconEnv.PLAYER_1
        assert env.fleets[0].source == 0
        assert env.fleets[0].target == 1

    def test_production_adds_ships(self) -> None:
        env = GalconEnv(EasyDict(self.cfg))
        env.reset()

        before = env.planets[0].ships
        env.step(env.pass_action)
        after = env.planets[0].ships

        expected_delta = env.planets[0].production / env.PRODUCTION_TO_SHIPS_PER_SECOND_DIVISOR * env.tick_seconds
        assert np.isclose(after - before, expected_delta)

    def test_seeded_map_is_deterministic(self) -> None:
        env1 = GalconEnv(EasyDict(self.cfg))
        env2 = GalconEnv(EasyDict(self.cfg))

        env1.reset()
        env2.reset()

        planets1 = [(p.x, p.y, p.ships, p.production, p.radius) for p in env1.planets]
        planets2 = [(p.x, p.y, p.ships, p.production, p.radius) for p in env2.planets]

        assert planets1 == planets2

    def test_timeout_winner_by_production(self) -> None:
        cfg = EasyDict(self.cfg)
        cfg.max_episode_steps = 1
        env = GalconEnv(cfg)
        env.reset()

        # Give player 1 a single planet
        env.planets[2].owner = GalconEnv.PLAYER_1
        env.planets[2].neutral = False

        obs, reward, done, info = env.step(env.pass_action)

        assert done
        assert info['winner'] == GalconEnv.PLAYER_1
        assert reward == np.array(1).astype(np.float32)

    def test_self_play_rollout(self) -> None:
        env = GalconEnv(EasyDict(self.cfg))
        obs = env.reset()

        done = False
        step = 0
        while not done and step < self.cfg.max_episode_steps + 5:
            action = env.random_action()
            obs, reward, done, info = env.step(action)
            step += 1

        assert done
        assert step <= self.cfg.max_episode_steps + 5