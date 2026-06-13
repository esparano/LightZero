import pytest
from easydict import EasyDict
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from zoo.board_games.galcon.envs.galcon_env import GalconEnv

# Run all tests: 
# pytest zoo/board_games/galcon/envs/

@pytest.mark.unittest
class TestGalconBots:

    def setup_method(self) -> None:
        self.cfg = GalconEnv.default_config() 
        self.cfg.update(dict(
        ))

    def test_random_bot_vs_random_bot(self) -> None:
        cfg = self.cfg
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
        cfg = self.cfg
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