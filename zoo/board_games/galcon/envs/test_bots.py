import pytest
from easydict import EasyDict
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from zoo.board_games.galcon.envs.galcon_env import GalconEnv
from zoo.board_games.galcon.envs.rule_bot import GalconFixedPolicyBot, GalconMCTSBot

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

    def test_mcts_bot_vs_mcts_bot(self) -> None:
        cfg = EasyDict(self.cfg)
        cfg.bot_action_type = 'mcts'
        cfg.map_seed = 101
        cfg.max_episode_steps = 30
        cfg.mcts_num_simulations = 16
        env = GalconEnv(cfg)
        env.reset()

        done = False
        step = 0
        while not done and step < cfg.max_episode_steps + 5:
            action = env.bot_action()
            assert action in env.legal_actions
            obs, reward, done, info = env.step(action)
            step += 1

        assert done

    def test_mcts_bot_beats_fixed_bot_in_eval_mode(self) -> None:
        # 1) MCTS as Player 1 vs Fixed as env bot (Player 2)
        cfg1 = EasyDict(self.cfg)
        cfg1.battle_mode = 'eval_mode'
        cfg1.bot_action_type = 'fixed'
        cfg1.map_seed = 100
        cfg1.max_episode_steps = 200
        cfg1.mcts_num_simulations = 32
        env1 = GalconEnv(cfg1)
        env1.reset()
        mcts_p1 = GalconMCTSBot(env1)

        done = False
        info = {}
        while not done:
            _, _, done, info = env1.step(mcts_p1.get_action())
        assert float(info['eval_episode_return']) > 0.0

        # 2) Fixed as Player 1 vs MCTS as env bot (Player 2, exercising bot_action_type='mcts')
        cfg2 = EasyDict(self.cfg)
        cfg2.battle_mode = 'eval_mode'
        cfg2.bot_action_type = 'mcts'
        cfg2.map_seed = 100
        cfg2.max_episode_steps = 200
        cfg2.mcts_num_simulations = 32
        env2 = GalconEnv(cfg2)
        env2.reset()
        fixed_p1 = GalconFixedPolicyBot(env2)

        done = False
        info = {}
        while not done:
            _, _, done, info = env2.step(fixed_p1.get_action())
        assert float(info['eval_episode_return']) < 0.0

    def test_cpp_bots_beat_fixed_bot_in_eval_mode(self) -> None:
        from zoo.board_games.galcon.envs.rule_bot import GalconCppBot, HAS_CPP_BOTS

        if not HAS_CPP_BOTS:
            pytest.skip("C++ bots_bridge requires g++ on Linux/WSL")

        for bot_type in ('waffle', 'waffleopt', 'sparky', 'sparkyn'):
            cfg = EasyDict(self.cfg)
            cfg.battle_mode = 'eval_mode'
            cfg.bot_action_type = 'fixed'
            cfg.map_seed = 100
            cfg.max_episode_steps = 200
            env = GalconEnv(cfg)
            env.reset()
            p1_bot = GalconCppBot(env, bot_type=bot_type)
            done = False
            info = {}
            while not done:
                _, _, done, info = env.step(p1_bot.get_action())
            assert float(info['eval_episode_return']) > 0.0, f"{bot_type} failed to beat fixed bot on seed 100"
