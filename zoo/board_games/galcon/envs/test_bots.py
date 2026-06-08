import pytest
from easydict import EasyDict

from zoo.board_games.galcon.envs.galcon_env import GalconEnv

# Run all tests: 
# pytest zoo/board_games/galcon/envs/

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
            fleet_speed=40.0,
            max_episode_steps=20,
            map_seed=0,
            grid_square_size=20.0,
            grid_min_x=-200.0,
            grid_max_x=200.0,
            grid_min_y=-120.0,
            grid_max_y=120.0,
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