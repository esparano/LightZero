import pytest
from easydict import EasyDict

from zoo.board_games.galcon.envs.galcon_env import GalconEnv


@pytest.mark.unittest
class TestGalconBots:

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

    def test_random_bot_vs_random_bot(self) -> None:
        cfg = EasyDict(self.cfg)
        cfg.bot_action_type = 'random'
        env = GalconEnv(cfg)
        env.reset()

        done = False
        step = 0
        # TODO: Why + 5?
        while not done and step < cfg.max_episode_steps + 5:
            action = env.bot_action()
            obs, reward, done, info = env.step(action)
            step += 1

        assert done

    def test_fixed_bot_vs_fixed_bot(self) -> None:
        cfg = EasyDict(self.cfg)
        cfg.bot_action_type = 'fixed'
        env = GalconEnv(cfg)
        env.reset()

        done = False
        step = 0
        while not done and step < cfg.max_episode_steps + 5:
            action = env.bot_action()
            obs, reward, done, info = env.step(action)
            step += 1

        assert done