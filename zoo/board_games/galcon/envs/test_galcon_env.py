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
            max_episode_steps=20,
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