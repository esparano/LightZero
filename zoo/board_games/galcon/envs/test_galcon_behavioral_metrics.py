import math

import numpy as np
import pytest
from easydict import EasyDict

from lzero.entry.async_split.actor import SelfPlayActor
from lzero.entry.async_split.elo_evaluator import AsyncEloEvaluator
from lzero.entry.async_split.evaluator import AsyncEvaluator
from lzero.entry.async_split.learner import AsyncLearner
from lzero.worker.head_to_head import EnvBotMatchPlayer, HeadToHeadRunner
from zoo.board_games.galcon.envs.galcon_env import GalconEnv


def _make_galcon_cfg(
    battle_mode: str = 'self_play_mode',
    use_cpp_sim: bool = True,
    max_episode_steps: int = 80,
    enable_fleet_redirection: bool = False,
    bot_action_type: str = 'expand',
    map_seed: int = 7,
) -> EasyDict:
    cfg = GalconEnv.default_config()
    cfg.battle_mode = battle_mode
    cfg.use_cpp_sim = use_cpp_sim
    cfg.num_planets = 6
    cfg.grid_width = 8
    cfg.grid_height = 8
    cfg.max_episode_steps = max_episode_steps
    cfg.enable_fleet_redirection = enable_fleet_redirection
    cfg.bot_action_type = bot_action_type
    cfg.map_seed = map_seed
    cfg.render_mode = None
    return cfg


def _encode_planet_send(env: GalconEnv, src_planet, tgt_planet) -> int:
    sx, sy = env._world_to_grid(src_planet.x, src_planet.y)
    tx, ty = env._world_to_grid(tgt_planet.x, tgt_planet.y)
    return int(env.encode_action(sx, sy, tx, ty))


def _tb_tags(tb_dir) -> set:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    return set(ea.Tags().get('scalars', []))


@pytest.mark.unittest
class TestGalconBehavioralMetrics:

    @pytest.mark.parametrize('use_cpp_sim', [False, True])
    def test_episode_emits_finite_metrics_only_on_done_and_uses_pass_rate_when_ahead(self, use_cpp_sim):
        env = GalconEnv(_make_galcon_cfg(
            battle_mode='self_play_mode',
            use_cpp_sim=use_cpp_sim,
            max_episode_steps=60,
            bot_action_type='expand',
        ))
        env.reset()

        last_ts = None
        for step_idx in range(60):
            # Alternate between expand bot actions for P1 and passive/simple actions for P2
            if env._current_player == env.PLAYER_1:
                act = env.bot_action()
            else:
                act = 0  # PASS so P1 pulls ahead and triggers pass_rate_when_ahead if P1 passes
                if step_idx % 6 == 1:
                    act = env.bot_action()
            ts = env.step(act)
            if not ts.done:
                assert 'galcon_metrics' not in ts.info
                assert 'galcon_metrics_by_player' not in ts.info
            else:
                last_ts = ts
                break

        assert last_ts is not None and last_ts.done
        by_player = last_ts.info['galcon_metrics_by_player']
        summary = last_ts.info['galcon_metrics']

        assert set(by_player.keys()) == {1, 2}
        for p in (1, 2):
            m = by_player[p]
            assert 'hold_rate_when_ahead' not in m
            for k, v in m.items():
                assert isinstance(v, float), f'{k} is not float: {type(v)}'
                assert math.isfinite(v), f'{k} is not finite: {v}'
            # Core unconditional metrics must always be present
            for req in (
                'early_neutral_prod_taken',
                'early_cheap_neutrals_left',
                'early_home_rush_ships',
                'stolen_own_half_prod',
                'pass_rate_opening',
                'pass_rate_mid',
                'pass_rate_late',
                'ships_in_flight_share',
                'prod_share_at_10s',
                'prod_share_at_half',
                'timeout_rate',
            ):
                assert req in m, f'Missing {req} for player {p}'
            # Redirect share must be omitted when enable_fleet_redirection=False
            assert 'redirect_share' not in m
            # Win/loss step metrics are mutually exclusive per player
            assert not ('avg_win_steps' in m and 'avg_loss_steps' in m)

        # Self-play summary includes p1_win_rate and never hold_rate_when_ahead
        assert 'p1_win_rate' in summary
        assert summary['p1_win_rate'] in (0.0, 0.5, 1.0)
        assert 'hold_rate_when_ahead' not in summary

    def test_exact_early_rush_pass_when_ahead_and_redirect_metrics(self):
        env = GalconEnv(_make_galcon_cfg(
            battle_mode='self_play_mode',
            use_cpp_sim=True,
            max_episode_steps=12,
            enable_fleet_redirection=True,
            map_seed=3,
        ))
        env.reset()

        p1_home = next(p for p in env.planets if p.owner == env.PLAYER_1)
        p2_home = next(p for p in env.planets if p.owner == env.PLAYER_2)

        # Step 1 (P1): send 50% of home (50 ships) directly at P2 home -> early_home_rush_ships
        rush_action = _encode_planet_send(env, p1_home, p2_home)
        ts = env.step(rush_action)
        assert not ts.done

        # Step 2 (P2): PASS
        ts = env.step(0)
        assert not ts.done

        # Step 3 (P1): redirect that in-flight fleet back to own home (reinforce + redirect)
        assert len(env.fleets) >= 1
        fl = env.fleets[0]
        fx, fy = env._world_to_grid(fl.x, fl.y)
        hx, hy = env._world_to_grid(p1_home.x, p1_home.y)
        redir_action = int(env.encode_action(fx, fy, hx, hy, is_redirect=True))
        ts = env.step(redir_action)

        # Force P1 way ahead in ships & prod, then have P1 PASS for remaining turns -> pass_rate_when_ahead == 1.0
        env._bm_prev_prod[env.PLAYER_1] = 200.0
        env._bm_prev_prod[env.PLAYER_2] = 50.0
        while not ts.done:
            if env._current_player == env.PLAYER_1:
                env._bm_prev_prod[env.PLAYER_1] = 200.0
                env._bm_prev_prod[env.PLAYER_2] = 50.0
            ts = env.step(0)

        m1 = ts.info['galcon_metrics_by_player'][1]
        assert m1['early_home_rush_ships'] == pytest.approx(50.0, abs=1.0)
        assert m1['send_fraction_mean'] == pytest.approx(0.5, abs=1e-4)
        assert m1['target_enemy_share'] == pytest.approx(0.5, abs=1e-6)
        assert m1['target_reinforce_share'] == pytest.approx(0.5, abs=1e-6)
        assert m1['redirect_share'] == pytest.approx(0.5, abs=1e-6)
        assert m1['pass_rate_when_ahead'] == pytest.approx(1.0, abs=1e-6)
        assert m1['timeout_rate'] == 1.0

    def test_eval_mode_reports_player1_metrics_even_on_bot_turn_done(self):
        env = GalconEnv(_make_galcon_cfg(
            battle_mode='eval_mode',
            use_cpp_sim=True,
            max_episode_steps=10,  # even step count -> ends on bot's (P2) turn!
            bot_action_type='pass',
        ))
        env.reset()
        ts = None
        for _ in range(10):
            ts = env.step(0)
            if ts.done:
                break
        assert ts is not None and ts.done
        # In eval_mode, galcon_metrics must match Player 1's metrics exactly
        assert ts.info['galcon_metrics'] == ts.info['galcon_metrics_by_player'][1]
        assert ts.info['galcon_metrics']['pass_rate_opening'] == pytest.approx(1.0)
        assert ts.info['galcon_metrics']['timeout_rate'] == 1.0

    def test_head_to_head_runner_aggregates_candidate_behavior_and_seat_split(self):
        env_cfg = _make_galcon_cfg(
            battle_mode='self_play_mode',
            use_cpp_sim=True,
            max_episode_steps=40,
            bot_action_type='random',
        )
        runner = HeadToHeadRunner(env_fn=GalconEnv, env_cfg=env_cfg, env_num=2)
        try:
            player_a = EnvBotMatchPlayer(bot_type='expand', player_id='bot:expand')
            player_b = EnvBotMatchPlayer(bot_type='pass', player_id='bot:pass')
            summary = runner.play_match(player_a=player_a, player_b=player_b, num_games=4, seed=11)
        finally:
            runner.close()

        assert summary.games_a_as_p1 == 2
        assert summary.games_a_as_p2 == 2
        assert 'early_neutral_prod_taken' in summary.behavior_a
        assert len(summary.behavior_a['early_neutral_prod_taken']) == 4
        assert 'hold_rate_when_ahead' not in summary.behavior_a
        # Expand bot should beat pass bot and take neutral production
        assert summary.wins_a >= 3
        assert float(np.mean(summary.behavior_a['early_neutral_prod_taken'])) > 0.0

    def test_async_split_logs_behavior_scalars_to_tensorboard(self, tmp_path):
        from lzero.entry.tests.test_async_split_e2e import _make_tiny_galcon_async_cfg

        run_dir = tmp_path / 'galcon_behavior_tb'
        main_cfg, create_cfg = _make_tiny_galcon_async_cfg()
        main_cfg.env.max_episode_steps = 16
        main_cfg.policy.game_segment_length = 16
        main_cfg.policy.td_steps = 16

        learner = AsyncLearner(
            input_cfg=(main_cfg, create_cfg),
            run_dir=run_dir,
            seed=0,
            device='cpu',
            publish_every_n_updates=1,
            save_ckpt_every_n_updates=0,
            max_replay_ratio=0.5,
            min_transitions=4,
            use_bg_reader=False,
        )
        actor = SelfPlayActor(
            input_cfg=(main_cfg, create_cfg),
            run_dir=run_dir,
            actor_id='actor_0',
            seed=10,
            device='cpu',
            env_manager_type='base',
            weight_sync_interval=1,
        )
        evaluator = AsyncEvaluator(
            input_cfg=(main_cfg, create_cfg),
            run_dir=run_dir,
            seed=20,
            device='cpu',
            evaluator_env_num=2,
            n_evaluator_episode=2,
            env_manager_type='base',
            min_iter_gap=0,
            eval_mode='both',
            elo_games_per_opponent=2,
            elo_opponents_per_eval=1,
            elo_anchor_bots='pass',
        )
        try:
            actor.step_once(n_episode=2)
            learner.ingest_pending_segments()
            assert learner.train_step() is not None
            learner.tb_logger.flush()

            eval_res = evaluator.eval_once_if_updated()
            assert eval_res is not None
        finally:
            actor.close()
            evaluator.close()
            learner.close()

        learner_tags = _tb_tags(run_dir / 'log' / 'learner')
        eval_tags = _tb_tags(run_dir / 'log' / 'evaluator')
        elo_tags = _tb_tags(run_dir / 'log' / 'elo_evaluator')

        assert 'async_learner/behavior_early_neutral_prod_taken' in learner_tags
        assert 'async_learner/behavior_p1_win_rate' in learner_tags
        assert 'async_learner/behavior_timeout_rate' in learner_tags

        assert 'evaluator_iter/behavior_early_neutral_prod_taken' in eval_tags
        assert 'evaluator_iter/behavior_timeout_rate' in eval_tags

        assert 'elo_iter/behavior_early_neutral_prod_taken' in elo_tags
        assert 'elo_iter/behavior_p1_minus_p2_win_rate' in elo_tags
        assert 'elo_iter/behavior_timeout_rate' in elo_tags
        for tag_set in (learner_tags, eval_tags, elo_tags):
            assert not any('hold_rate_when_ahead' in t for t in tag_set)
