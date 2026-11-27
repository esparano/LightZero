import pytest
from easydict import EasyDict
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from zoo.board_games.galcon.envs.galcon_env import GalconEnv, Planet
from zoo.board_games.galcon.envs.rule_bot import (
    GalconFixedPolicyBot,
    GalconMCTSBot,
    GalconPassBot,
    GalconRandomBot,
)

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

    def test_mcts_bot_returns_legal_action(self) -> None:
        cfg = EasyDict(self.cfg)
        cfg.map_seed = 42
        cfg.mcts_num_simulations = 16
        env = GalconEnv(cfg)
        env.reset()

        bot = GalconMCTSBot(env)
        action = bot.get_action()

        assert action in env.legal_actions

    def test_mcts_bot_captures_cheap_high_prod_neutral(self) -> None:
        cfg = EasyDict(self.cfg)
        cfg.num_planets = 4
        cfg.mcts_num_simulations = 24
        env = GalconEnv(cfg)
        env.reset()
        # Deterministic 4-planet setup: P1 home at (-100, 0), juicy cheap neutral right next to it
        env.planets = [
            Planet(id=0, owner=env.PLAYER_1, ships=100.0, x=-100.0, y=0.0, production=100.0, radius=24.0, neutral=False),
            Planet(id=1, owner=env.PLAYER_2, ships=100.0, x=150.0, y=0.0, production=100.0, radius=24.0, neutral=False),
            Planet(id=2, owner=env.PLAYER_NEUTRAL, ships=5.0, x=-40.0, y=0.0, production=95.0, radius=23.3, neutral=True),
            Planet(id=3, owner=env.PLAYER_NEUTRAL, ships=50.0, x=90.0, y=60.0, production=20.0, radius=12.7, neutral=True),
        ]
        env.fleets = []
        env._current_player = env.PLAYER_1

        bot = GalconMCTSBot(env)
        action = bot.get_action()
        expected_send = env.encode_action(
            *env._world_to_grid(-100.0, 0.0),
            *env._world_to_grid(-40.0, 0.0),
            is_redirect=False,
        )

        assert action in env.legal_actions
        assert action == expected_send

    def test_mcts_bot_can_step_with_redirects_enabled(self) -> None:
        cfg = EasyDict(self.cfg)
        cfg.map_seed = 7
        cfg.enable_fleet_redirection = True
        cfg.mcts_num_simulations = 16
        env = GalconEnv(cfg)
        env.reset()

        bot = GalconMCTSBot(env)
        for _ in range(4):
            action = bot.get_action()
            assert action in env.legal_actions
            obs, _, done, _ = env.step(action)
            assert isinstance(obs, dict)
            if done:
                break

    def test_mcts_bot_defaults_and_future_ships_eval_parity(self) -> None:
        from zoo.board_games.galcon.envs.cpp_sim import FastGalconCppSim, GalconSim, HAS_FAST_SIM_CPP, SendOrder, SimPlanet
        from zoo.board_games.galcon.envs.cpp_sim.sim import evaluate_galcon_state

        env = GalconEnv(self.cfg)
        bot = GalconMCTSBot(env)
        assert bot.search_engine.num_simulations == 16
        assert bot.search_engine.rollout_ticks == 0
        assert bot.search_engine.c_puct == pytest.approx(0.5)

        # Verify C++ and Python future-ships evaluators match and penalize expensive neutral capture
        def _eval_after_send(neutral_ships: float):
            planets = [
                SimPlanet(0, 1, 100.0, -100.0, 0.0, 100.0, 24.0),
                SimPlanet(1, 2, 100.0, 150.0, 0.0, 100.0, 24.0),
                SimPlanet(2, 0, neutral_ships, -30.0, 0.0, 20.0, 13.0),
            ]
            sim = GalconSim(planets, [], width=400.0, height=250.0, use_cpp=True)
            sim.step({1: [SendOrder(0, 2, 50)]})
            py_val = evaluate_galcon_state(sim, player=1, horizon=40)
            if HAS_FAST_SIM_CPP:
                cpp = FastGalconCppSim.from_sim(sim)
                try:
                    cpp_val = cpp.evaluate_heuristic(1, horizon=40)
                finally:
                    cpp.close()
                assert cpp_val == pytest.approx(py_val, abs=1e-5)
            return py_val

        val_cheap = _eval_after_send(5.0)
        val_expensive = _eval_after_send(49.0)
        assert val_cheap > val_expensive + 0.05

