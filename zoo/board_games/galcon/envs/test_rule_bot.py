import pytest
from easydict import EasyDict

from zoo.board_games.galcon.envs.galcon_env import GalconEnv
from zoo.board_games.galcon.envs.rule_bot import GalconFixedPolicyBot, GalconRandomBot

# Run all tests: 
# pytest zoo/board_games/galcon/envs/

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
            fleet_speed=40.0,
            max_episode_steps=20,
            map_seed=0,
            grid_square_size=25.0,
            grid_max_x=200.0,
            grid_max_y=125.0,
            home_distance_fraction=0.8,
            neutral_min_cost = 0,
            neutral_max_cost = 50,
            neutral_min_production = 15,
            neutral_max_production = 100,
            fleet_top_k=3,
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

    def test_random_bot_prefers_non_pass_when_available(self) -> None:
        env = GalconEnv(EasyDict(self.cfg))
        env.reset()

        bot = GalconRandomBot(env)
        action = bot.get_action()

        assert action in env.legal_actions
        assert action != env.pass_action


    def test_fixed_policy_bot_prefers_non_pass_when_available(self) -> None:
        env = GalconEnv(EasyDict(self.cfg))
        env.reset()

        bot = GalconFixedPolicyBot(env)
        action = bot.get_action()

        assert action in env.legal_actions
        assert action != env.pass_action