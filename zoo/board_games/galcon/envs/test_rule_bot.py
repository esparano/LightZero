import pytest
from easydict import EasyDict
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from zoo.board_games.galcon.envs.galcon_env import GalconEnv
from zoo.board_games.galcon.envs.rule_bot import GalconFixedPolicyBot, GalconRandomBot, GalconPassBot

# Run all tests: 
# pytest zoo/board_games/galcon/envs/

@pytest.mark.unittest
class TestGalconRuleBots:

    def setup_method(self) -> None:
        self.cfg = GalconEnv.default_config() 
        self.cfg.update(dict(
        ))

    def test_pass_bot_passes(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        bot = GalconPassBot(env)
        action = bot.get_action()

        assert action in env.legal_actions
        assert action == env.pass_action

    def test_random_bot_returns_legal_action(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        bot = GalconRandomBot(env)
        action = bot.get_action()

        assert action in env.legal_actions

    def test_fixed_policy_bot_returns_legal_action(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()
        bot = GalconFixedPolicyBot(env)
        action = bot.get_action()

        assert action in env.legal_actions

    def test_random_and_fixed_bots_can_step(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        random_bot = GalconRandomBot(env)
        action = random_bot.get_action()
        obs, reward, done, info = env.step(action)

        fixed_bot = GalconFixedPolicyBot(env)
        action = fixed_bot.get_action()
        obs, reward, done, info = env.step(action)

        assert isinstance(obs, dict)

    def test_random_bot_prefers_non_pass_when_available(self) -> None:
        env = GalconEnv(self.cfg)
        self.cfg.update(dict(
            num_planets = 10
        ))
        env.reset()

        bot = GalconRandomBot(env)
        action = bot.get_action()

        assert action in env.legal_actions
        assert action != env.pass_action


    def test_fixed_policy_bot_prefers_non_pass_when_available(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        bot = GalconFixedPolicyBot(env)
        action = bot.get_action()

        assert action in env.legal_actions
        assert action != env.pass_action