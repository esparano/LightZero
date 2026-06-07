import pytest
from easydict import EasyDict

from zoo.board_games.galcon.envs.galcon_env import GalconEnv
from zoo.board_games.galcon.envs.rule_bot import GalconFixedPolicyBot, GalconRandomBot


@pytest.mark.unittest
class TestGalconRuleBots:

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

    def test_random_bot_returns_legal_action(self) -> None:
        env = GalconEnv(EasyDict(self.cfg))
        env.reset()

        bot = GalconRandomBot(env)
        action = bot.get_action()

        assert action in env.legal_actions

    def test_fixed_policy_bot_returns_legal_action(self) -> None:
        env = GalconEnv(EasyDict(self.cfg))
        env.reset()

        bot = GalconFixedPolicyBot(env)
        action = bot.get_action()

        assert action in env.legal_actions

    def test_random_and_fixed_bots_can_step(self) -> None:
        env = GalconEnv(EasyDict(self.cfg))
        env.reset()

        random_bot = GalconRandomBot(env)
        action = random_bot.get_action()
        obs, reward, done, info = env.step(action)

        fixed_bot = GalconFixedPolicyBot(env)
        action = fixed_bot.get_action()
        obs, reward, done, info = env.step(action)

        assert isinstance(obs, dict)