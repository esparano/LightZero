# -*- coding: utf-8 -*-
"""CPU-only unit and end-to-end tests for paired head-to-head matches, live
AsyncEloEvaluator / AsyncEvaluator elo_ladder & both modes, and offline
checkpoint tournaments on tiny Galcon.
"""

import copy
import os
from pathlib import Path
from typing import List, Tuple

import pytest
import torch
from ding.config import compile_config
from ding.envs import BaseEnvTimestep, get_vec_env_setting
from ding.policy import create_policy

from lzero.entry.async_split import (
    AsyncEloEvaluator,
    AsyncEvaluator,
    OfflineTournament,
    discover_checkpoints_in_dir,
    parse_checkpoint_spec,
)
from lzero.entry.async_split.cli import build_parser
from lzero.entry.async_split.io_utils import (
    AsyncSplitDirs,
    atomic_torch_save,
    cpu_state_dict,
    detect_existing_run_artifacts,
    publish_weights,
)
from lzero.entry.tests.test_async_split_e2e import _make_tiny_galcon_async_cfg
from lzero.worker.head_to_head import (
    EnvBotMatchPlayer,
    HeadToHeadRunner,
    PolicyMatchPlayer,
    _extract_winner_player,
)


def _read_tb_tags(log_dir: Path) -> List[str]:
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

    tags = set()
    for ef in sorted(log_dir.glob('events.out.tfevents.*')):
        for event in EventFileLoader(str(ef)).Load():
            if not event.HasField('summary'):
                continue
            for val in event.summary.value:
                tags.add(val.tag)
    return sorted(tags)


def _save_dummy_ckpt(policy, path: Path, train_iter: int, perturb: float = 0.0) -> Path:
    sd = cpu_state_dict(policy._model.state_dict())
    if perturb != 0.0:
        for k, v in sd.items():
            if torch.is_tensor(v) and v.is_floating_point():
                sd[k] = v + perturb
    return atomic_torch_save(
        {'model': sd, 'train_iter': int(train_iter), 'envstep': int(train_iter * 10), 'last_iter': int(train_iter)},
        path,
    )


@pytest.mark.unittest
def test_head_to_head_paired_map_seeds_and_bots():
    main_cfg, create_cfg = _make_tiny_galcon_async_cfg()
    main_cfg.env.map_seed = None  # Random maps so pair seed matching is genuinely tested
    main_cfg.env.max_episode_steps = 12
    compiled = compile_config(
        copy.deepcopy(main_cfg), seed=0, env=None, auto=True, create_cfg=copy.deepcopy(create_cfg), save_cfg=False
    )
    env_fn, collector_env_cfg, _ = get_vec_env_setting(compiled.env)
    runner = HeadToHeadRunner(env_fn=env_fn, env_cfg=collector_env_cfg[0], env_num=2)
    try:
        bot_rand = EnvBotMatchPlayer('random')
        bot_pass = EnvBotMatchPlayer('pass')
        summary = runner.play_match(bot_rand, bot_pass, num_games=6, seed=42)
        assert summary.total_games == 6
        assert len(summary.map_seeds) == 6
        # Duplicate-bridge invariant: first 3 games (A=P1) and next 3 games (A=P2) use identical map seeds
        assert sorted(summary.map_seeds[:3]) == sorted(summary.map_seeds[3:])
        assert len(set(summary.map_seeds[:3])) >= 2
        assert summary.total_steps > 0
        # Active expansion (random) beats pure pass on Galcon
        assert summary.win_rate_a >= 0.5
    finally:
        runner.close()


@pytest.mark.unittest
def test_head_to_head_policy_vs_bot_and_policy_vs_policy():
    main_cfg, create_cfg = _make_tiny_galcon_async_cfg()
    main_cfg.policy.device = 'cpu'
    compiled = compile_config(
        copy.deepcopy(main_cfg), seed=0, env=None, auto=True, create_cfg=copy.deepcopy(create_cfg), save_cfg=False
    )
    env_fn, collector_env_cfg, _ = get_vec_env_setting(compiled.env)
    policy_a = create_policy(copy.deepcopy(compiled.policy), model=None, enable_field=['learn', 'eval'])
    policy_b = create_policy(copy.deepcopy(compiled.policy), model=None, enable_field=['learn', 'eval'])

    runner = HeadToHeadRunner(env_fn=env_fn, env_cfg=collector_env_cfg[0], env_num=2)
    try:
        pa = PolicyMatchPlayer('iter_0', policy_a, compiled.policy)
        pb = PolicyMatchPlayer('iter_10', policy_b, compiled.policy)
        bot = EnvBotMatchPlayer('random')

        s_bot = runner.play_match(pa, bot, num_games=4, seed=7)
        assert s_bot.total_games == 4
        assert s_bot.wins_a + s_bot.draws + s_bot.losses_a == 4

        s_pvp = runner.play_match(pa, pb, num_games=4, seed=11)
        assert s_pvp.total_games == 4
        assert s_pvp.wins_a + s_pvp.draws + s_pvp.losses_a == 4
    finally:
        runner.close()


@pytest.mark.unittest
def test_async_elo_evaluator_live_promotion_resume_and_tb(tmp_path):
    run_dir = tmp_path / 'elo_live_run'
    cfg_pair = _make_tiny_galcon_async_cfg()
    dirs = AsyncSplitDirs.from_root(run_dir)
    dirs.ensure_exists()

    elo_eval = AsyncEloEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=123,
        device='cpu',
        evaluator_env_num=2,
        games_per_opponent=2,
        opponents_per_eval=3,
        max_ladder_size=4,
        min_iter_gap=0,
    )
    try:
        # 1. Publish iter_0 weights and evaluate -> promotes seed rung & saves elo_ckpt_best.pth.tar
        publish_weights(
            dirs.latest_weights_path,
            elo_eval.candidate_policy._model.state_dict(),
            train_iter=0,
            envstep=0,
        )
        res0 = elo_eval.eval_once_if_updated()
        assert res0 is not None
        assert res0['candidate_id'] == 'iter_0'
        assert res0['promoted'] is True
        assert res0['is_new_best'] is True
        assert (dirs.elo_rungs_dir / 'iter_0.pth.tar').is_file()
        assert (dirs.ckpt_dir / 'elo_ckpt_best.pth.tar').is_file()
        assert dirs.elo_state_path.is_file()
        assert any('elo_ladder/state.json' in item for item in detect_existing_run_artifacts(dirs))

        # 2. Publish iter_2 weights -> evaluates against both bot:random and iter_0
        st = dirs.latest_weights_path.stat()
        publish_weights(
            dirs.latest_weights_path,
            elo_eval.candidate_policy._model.state_dict(),
            train_iter=2,
            envstep=40,
        )
        os.utime(dirs.latest_weights_path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000))
        res2 = elo_eval.eval_once_if_updated()
        assert res2 is not None
        assert res2['train_iter'] == 2
        opps_played = {m['opponent_id'] for m in res2['matches']}
        assert {'bot:random', 'iter_0'}.issubset(opps_played)

        saved_best_elo = elo_eval.ladder.best_elo
        saved_match_count = len(elo_eval.ladder.matches)
    finally:
        elo_eval.close()

    # Verify TensorBoard logged Elo tags
    tb_tags = _read_tb_tags(dirs.log_dir / 'elo_evaluator')
    assert 'elo_iter/rating' in tb_tags
    assert 'elo_iter/rating_std' in tb_tags
    assert 'elo_iter/best_rating' in tb_tags
    assert 'elo_step/rating' in tb_tags

    # 3. Resume in a fresh AsyncEloEvaluator instance
    elo_eval_resumed = AsyncEloEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=123,
        device='cpu',
        evaluator_env_num=2,
        games_per_opponent=2,
        opponents_per_eval=3,
        max_ladder_size=4,
        min_iter_gap=0,
        resume=True,
    )
    try:
        assert elo_eval_resumed._last_eval_iter == 2
        assert elo_eval_resumed._num_evals == 2
        assert len(elo_eval_resumed.ladder.matches) == saved_match_count
        assert elo_eval_resumed.ladder.best_elo == pytest.approx(saved_best_elo, abs=1e-6)
        # Same iter_2 weights must NOT be re-evaluated after resume
        assert elo_eval_resumed.eval_once_if_updated() is None

        # Publish iter_4 -> resumes cleanly
        st = dirs.latest_weights_path.stat()
        publish_weights(
            dirs.latest_weights_path,
            elo_eval_resumed.candidate_policy._model.state_dict(),
            train_iter=4,
            envstep=80,
        )
        os.utime(dirs.latest_weights_path, ns=(st.st_atime_ns, st.st_mtime_ns + 4_000_000))
        res4 = elo_eval_resumed.eval_once_if_updated()
        assert res4 is not None
        assert res4['train_iter'] == 4
        assert elo_eval_resumed._num_evals == 3
    finally:
        elo_eval_resumed.close()


@pytest.mark.unittest
def test_async_evaluator_modes_elo_ladder_and_both(tmp_path):
    cfg_pair = _make_tiny_galcon_async_cfg()

    # Mode 1: elo_ladder
    run_elo = tmp_path / 'mode_elo_ladder'
    dirs_elo = AsyncSplitDirs.from_root(run_elo)
    dirs_elo.ensure_exists()
    ev_elo = AsyncEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_elo,
        seed=0,
        device='cpu',
        evaluator_env_num=2,
        eval_mode='elo_ladder',
        elo_games_per_opponent=2,
        elo_opponents_per_eval=2,
    )
    try:
        publish_weights(dirs_elo.latest_weights_path, ev_elo.policy._model.state_dict(), train_iter=1, envstep=10)
        out = ev_elo.eval_once_if_updated()
        assert out is not None
        assert 'elo_result' in out
        assert (dirs_elo.ckpt_dir / 'elo_ckpt_best.pth.tar').is_file()
        assert not (dirs_elo.ckpt_dir / 'ckpt_best.pth.tar').is_file()
    finally:
        ev_elo.close()

    # Mode 2: both (runs bot eval AND elo_ladder eval on the same weight snapshot)
    run_both = tmp_path / 'mode_both'
    dirs_both = AsyncSplitDirs.from_root(run_both)
    dirs_both.ensure_exists()
    ev_both = AsyncEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_both,
        seed=0,
        device='cpu',
        evaluator_env_num=2,
        n_evaluator_episode=2,
        env_manager_type='base',
        eval_mode='both',
        elo_games_per_opponent=2,
        elo_opponents_per_eval=2,
    )
    try:
        publish_weights(dirs_both.latest_weights_path, ev_both.policy._model.state_dict(), train_iter=1, envstep=10)
        out = ev_both.eval_once_if_updated()
        assert out is not None
        assert 'elo_result' in out
        assert (dirs_both.ckpt_dir / 'ckpt_best.pth.tar').is_file()
        assert (dirs_both.ckpt_dir / 'elo_ckpt_best.pth.tar').is_file()
        assert dirs_both.evaluator_state_path.is_file()
        assert dirs_both.elo_state_path.is_file()
    finally:
        ev_both.close()


@pytest.mark.unittest
def test_offline_tournament_discovery_ladder_resume_and_round_robin(tmp_path):
    cfg_pair = _make_tiny_galcon_async_cfg()
    cfg_pair[0].policy.device = 'cpu'
    compiled = compile_config(
        copy.deepcopy(cfg_pair[0]), seed=0, env=None, auto=True, create_cfg=copy.deepcopy(cfg_pair[1]), save_cfg=False
    )
    policy = create_policy(copy.deepcopy(compiled.policy), model=None, enable_field=['learn', 'eval'])

    ckpt_dir = tmp_path / 'run_a' / 'ckpt'
    ckpt_dir.mkdir(parents=True)
    for idx, it in enumerate((0, 100, 200, 300)):
        _save_dummy_ckpt(policy, ckpt_dir / f'iteration_{it}.pth.tar', train_iter=it, perturb=0.01 * idx)

    # Test discovery with every_n and max_checkpoints
    all_specs = discover_checkpoints_in_dir(ckpt_dir)
    assert [s.train_iter for s in all_specs] == [0, 100, 200, 300]

    sub_every = discover_checkpoints_in_dir(ckpt_dir, every_n=2)
    assert [s.train_iter for s in sub_every] == [0, 200, 300]

    sub_max = discover_checkpoints_in_dir(ckpt_dir, max_checkpoints=3)
    assert len(sub_max) == 3
    assert sub_max[0].train_iter == 0 and sub_max[-1].train_iter == 300

    # Test parse_checkpoint_spec with label=path@config
    parsed = parse_checkpoint_spec(f'custom_a={ckpt_dir / "iteration_100.pth.tar"}@{tmp_path / "dummy_cfg.py"}')
    assert len(parsed) == 1
    assert parsed[0].player_id == 'custom_a'
    assert parsed[0].config_path == (tmp_path / 'dummy_cfg.py').resolve()

    # Run ladder/Swiss tournament on 3 checkpoints + bot:random
    specs_3 = all_specs[:3]
    out_ladder = tmp_path / 'tourn_ladder'
    t1 = OfflineTournament(
        default_input_cfg=cfg_pair,
        checkpoints=specs_3,
        output_dir=out_ladder,
        anchor_bots=('random',),
        games_per_pair=2,
        mode='ladder',
        num_rounds=2,
        seed=5,
        device='cpu',
        env_num=2,
        write_tb=True,
    )
    res1 = t1.run()
    assert res1['matches_played'] >= 3
    assert (out_ladder / 'tournament_state.json').is_file()
    assert 'bot:random' in res1['leaderboard_text']
    tb_tags = _read_tb_tags(out_ladder / 'tb')
    assert 'tournament_iter/rating' in tb_tags

    # Resume same ladder tournament with same num_rounds=2 -> must be a 0-match no-op!
    total_before = res1['total_matches']
    t1_ladder_resume = OfflineTournament(
        default_input_cfg=cfg_pair,
        checkpoints=specs_3,
        output_dir=out_ladder,
        anchor_bots=('random',),
        games_per_pair=2,
        mode='ladder',
        num_rounds=2,
        seed=5,
        device='cpu',
        env_num=2,
        resume=True,
    )
    res_noop = t1_ladder_resume.run()
    assert res_noop['matches_played'] == 0
    assert res_noop['total_matches'] == total_before

    # Resume same tournament switching to mode='round_robin' -> finishes only unplayed pairs
    t1_resume = OfflineTournament(
        default_input_cfg=cfg_pair,
        checkpoints=specs_3,
        output_dir=out_ladder,
        anchor_bots=('random',),
        games_per_pair=2,
        mode='round_robin',
        seed=5,
        device='cpu',
        env_num=2,
        resume=True,
    )
    res_rr = t1_resume.run()
    # 4 participants (bot:random + 3 ckpts) => C(4, 2) = 6 total unique pairs
    assert res_rr['total_matches'] == 6
    assert res_rr['matches_played'] == 6 - total_before


@pytest.mark.unittest
def test_extract_winner_player_lightzero_self_play_invariants():
    # Standard LightZero self_play_mode env without info['winner'] (e.g. TicTacToe, Connect4, Chess):
    # timestep.reward is from acting_player's perspective (+1 if acting_player won),
    # while info['eval_episode_return'] is from Player 1's perspective (-1 if Player 2 won).
    ts_p2_wins_on_p2_turn = BaseEnvTimestep(obs={}, reward=1.0, done=True, info={'eval_episode_return': -1.0})
    assert _extract_winner_player(ts_p2_wins_on_p2_turn, acting_player=2) == 2

    ts_p1_wins_on_p2_turn = BaseEnvTimestep(obs={}, reward=-1.0, done=True, info={'eval_episode_return': 1.0})
    assert _extract_winner_player(ts_p1_wins_on_p2_turn, acting_player=2) == 1

    ts_draw = BaseEnvTimestep(obs={}, reward=0.0, done=True, info={'eval_episode_return': 0.0})
    assert _extract_winner_player(ts_draw, acting_player=1) == -1

    # Explicit info['winner'] (Galcon) always takes precedence
    ts_explicit = BaseEnvTimestep(obs={}, reward=-1.0, done=True, info={'winner': 2, 'eval_episode_return': -1.0})
    assert _extract_winner_player(ts_explicit, acting_player=1) == 2


@pytest.mark.unittest
def test_parse_checkpoint_spec_edge_cases_and_bot_id_collision(tmp_path):
    cfg_pair = _make_tiny_galcon_async_cfg()
    tricky_dir = tmp_path / 'run_lr=1e-4@exp'
    tricky_dir.mkdir(parents=True)
    ckpt = tricky_dir / 'ckpt_best.pth.tar'
    ckpt.write_bytes(b'placeholder')

    # Paths containing '=' and '@' must parse cleanly without misinterpreting as label/config
    parsed = parse_checkpoint_spec(str(ckpt))
    assert len(parsed) == 1
    assert parsed[0].ckpt_path == ckpt.resolve()
    assert parsed[0].config_path is None
    assert parsed[0].player_id == 'ckpt_best'

    # Explicit collision with reserved bot ID must fail loudly instead of silently dropping checkpoint
    bot_collision = parse_checkpoint_spec(f'bot:random={ckpt}')
    with pytest.raises(ValueError, match='collides with reserved bot player ID'):
        OfflineTournament(
            default_input_cfg=cfg_pair,
            checkpoints=bot_collision,
            output_dir=tmp_path / 'tourn_collision',
            anchor_bots=('random',),
            device='cpu',
            env_num=1,
        )

    # Globs and @config combined with '=' or '@' in directory names must parse cleanly
    glob_dir = tmp_path / 'run_lr=1e-4'
    glob_dir.mkdir(parents=True)
    for it in (100, 200):
        (glob_dir / f'iteration_{it}.pth.tar').write_bytes(b'x')
    cfg_file = tmp_path / 'custom_cfg.py'
    cfg_file.write_text('# dummy\n', encoding='utf-8')

    parsed_glob = parse_checkpoint_spec(str(glob_dir / 'iteration_*.pth.tar'))
    assert [s.train_iter for s in parsed_glob] == [100, 200]

    parsed_eq_at_cfg = parse_checkpoint_spec(f'{glob_dir / "iteration_100.pth.tar"}@{cfg_file}')
    assert len(parsed_eq_at_cfg) == 1
    assert parsed_eq_at_cfg[0].train_iter == 100
    assert parsed_eq_at_cfg[0].config_path == cfg_file.resolve()

    at_dir = tmp_path / 'exp@v2'
    at_dir.mkdir(parents=True)
    (at_dir / 'iteration_50.pth.tar').write_bytes(b'x')
    parsed_at_glob = parse_checkpoint_spec(str(at_dir / 'iteration_*.pth.tar'))
    assert [s.train_iter for s in parsed_at_glob] == [50]
    assert parsed_at_glob[0].config_path is None


@pytest.mark.unittest
def test_offline_tournament_resume_new_checkpoints_and_mid_round_crash(tmp_path):
    cfg_pair = _make_tiny_galcon_async_cfg()
    cfg_pair[0].policy.device = 'cpu'
    compiled = compile_config(
        copy.deepcopy(cfg_pair[0]), seed=0, env=None, auto=True, create_cfg=copy.deepcopy(cfg_pair[1]), save_cfg=False
    )
    policy = create_policy(copy.deepcopy(compiled.policy), model=None, enable_field=['learn', 'eval'])

    ckpt_dir = tmp_path / 'ckpts'
    ckpt_dir.mkdir(parents=True)
    for idx, it in enumerate((0, 100, 200, 300)):
        _save_dummy_ckpt(policy, ckpt_dir / f'iteration_{it}.pth.tar', train_iter=it, perturb=0.01 * idx)
    all_specs = discover_checkpoints_in_dir(ckpt_dir)

    # 1. Run ladder tournament on first 3 checkpoints (num_rounds=2 -> 3 chain + 2 Swiss = 5 matches)
    out_dir = tmp_path / 'tourn_inc'
    t1 = OfflineTournament(
        default_input_cfg=cfg_pair,
        checkpoints=all_specs[:3],
        output_dir=out_dir,
        anchor_bots=('random',),
        games_per_pair=2,
        mode='ladder',
        num_rounds=2,
        seed=9,
        device='cpu',
        env_num=2,
    )
    r1 = t1.run()
    assert r1['total_matches'] in (4, 5)

    # 2. Add iter_300 and --resume -> must connect iter_300 in chain + run catch-up Swiss match!
    t2 = OfflineTournament(
        default_input_cfg=cfg_pair,
        checkpoints=all_specs[:4],
        output_dir=out_dir,
        anchor_bots=('random',),
        games_per_pair=2,
        mode='ladder',
        num_rounds=2,
        seed=9,
        device='cpu',
        env_num=2,
        resume=True,
    )
    r2 = t2.run()
    assert r2['matches_played'] >= 2  # at least chain edge + 1 Swiss catch-up match for iter_300
    est_300 = next(row for row in r2['leaderboard'] if row['player_id'] == 'iter_300')
    assert est_300['games'] >= 4

    # 3. Re-running with --resume and NO new checkpoints must play 0 matches
    t3 = OfflineTournament(
        default_input_cfg=cfg_pair,
        checkpoints=all_specs[:4],
        output_dir=out_dir,
        anchor_bots=('random',),
        games_per_pair=2,
        mode='ladder',
        num_rounds=2,
        seed=9,
        device='cpu',
        env_num=2,
        resume=True,
    )
    r3 = t3.run()
    assert r3['matches_played'] == 0

    # 4. Simulate mid-round Swiss crash: pending_swiss_pairs has 1 remaining pair in round 1
    from lzero.entry.async_split.io_utils import atomic_json_save, read_json

    crash_dir = tmp_path / 'tourn_mid_crash'
    t_seed = OfflineTournament(
        default_input_cfg=cfg_pair,
        checkpoints=all_specs[:3],
        output_dir=crash_dir,
        anchor_bots=('random',),
        games_per_pair=2,
        mode='ladder',
        num_rounds=1,  # Round 1 chain only (3 matches)
        seed=9,
        device='cpu',
        env_num=2,
    )
    t_seed.run()

    state = read_json(crash_dir / 'tournament_state.json')
    state['completed_swiss_rounds'] = 0
    state['pending_swiss_pairs'] = [['iter_0', 'iter_200']]
    state['pending_is_catchup'] = False
    atomic_json_save(state, crash_dir / 'tournament_state.json')

    t_recover = OfflineTournament(
        default_input_cfg=cfg_pair,
        checkpoints=all_specs[:3],
        output_dir=crash_dir,
        anchor_bots=('random',),
        games_per_pair=2,
        mode='ladder',
        num_rounds=2,  # 1 Swiss round total; finishing the 1 pending pair completes it
        seed=9,
        device='cpu',
        env_num=2,
        resume=True,
    )
    r_rec = t_recover.run()
    # Must play ONLY the 1 pending pair (not re-plan a full 2-pair Swiss round!)
    assert r_rec['matches_played'] == 1
    assert r_rec['total_matches'] == 4


@pytest.mark.unittest
def test_retroactive_best_rung_inversion_updates_elo_ckpt_best(tmp_path):
    # Regression test: if active rung iter_200 retroactively overtakes initial best_player_id
    # (iter_100) during iter_300's evaluation, elo_ckpt_best.pth.tar must be updated to iter_200
    # rather than overwritten with weaker candidate iter_300.
    from lzero.worker.elo_rating import PlayerEntry
    from lzero.worker.head_to_head import MatchSummary

    cfg_pair = _make_tiny_galcon_async_cfg()
    run_dir = tmp_path / 'retro_best_run'
    elo_eval = AsyncEloEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        evaluator_env_num=1,
        games_per_opponent=10,
        opponents_per_eval=4,
        anchor_bots=('random',),
    )
    try:
        # Save distinct rung checkpoints for iter_100 and iter_200
        p100 = elo_eval.dirs.elo_rungs_dir / 'iter_100.pth.tar'
        p200 = elo_eval.dirs.elo_rungs_dir / 'iter_200.pth.tar'
        _save_dummy_ckpt(elo_eval.candidate_policy, p100, train_iter=100, perturb=0.0)
        _save_dummy_ckpt(elo_eval.candidate_policy, p200, train_iter=200, perturb=0.5)

        elo_eval.ladder.register_player(
            PlayerEntry('iter_100', kind='checkpoint', ckpt_path=str(p100), train_iter=100, active=True)
        )
        elo_eval.ladder.register_player(
            PlayerEntry('iter_200', kind='checkpoint', ckpt_path=str(p200), train_iter=200, active=True)
        )
        elo_eval.ladder.record_match('iter_100', 'bot:random', wins_a=9, draws=0, losses_a=1)
        elo_eval.ladder.record_match('iter_200', 'bot:random', wins_a=8, draws=0, losses_a=2)
        elo_eval.ladder.fit_ratings()
        elo_eval.ladder.best_player_id = 'iter_100'
        elo_eval.ladder.best_elo = elo_eval.ladder.get_elo('iter_100')

        # Stub runner.play_match so iter_300 beats iter_100 (9-1), loses to iter_200 (1-9), beats random (8-2)
        outcomes = {
            'bot:random': (8, 0, 2),
            'iter_100': (9, 0, 1),
            'iter_200': (1, 0, 9),
        }

        def _fake_play_match(player_a, player_b, num_games, seed):
            w, d, l = outcomes[player_b.player_id]
            return MatchSummary(
                player_a=player_a.player_id,
                player_b=player_b.player_id,
                wins_a=w,
                draws=d,
                losses_a=l,
                p1_wins=w,
                p2_wins=l,
                total_steps=100,
                elapsed_sec=0.01,
                map_seeds=[1] * num_games,
            )

        elo_eval.runner.play_match = _fake_play_match
        res = elo_eval.evaluate_loaded_candidate(train_iter=300, envstep=3000)

        # iter_200 must retroactively become best_player_id & elo_ckpt_best.pth.tar (NOT iter_300!)
        assert res['is_new_best'] is False
        assert elo_eval.ladder.best_player_id == 'iter_200'
        assert elo_eval.ladder.best_elo == pytest.approx(elo_eval.ladder.get_elo('iter_200'), abs=1e-6)
        assert elo_eval.ladder.best_elo > elo_eval.ladder.get_elo('iter_300')
        saved_best = torch.load(str(elo_eval.dirs.ckpt_dir / 'elo_ckpt_best.pth.tar'), map_location='cpu', weights_only=False)
        assert saved_best['train_iter'] == 200
    finally:
        elo_eval.close()


@pytest.mark.unittest
def test_offline_tournament_mid_catchup_crash_and_incremental_ordering(tmp_path):
    from lzero.entry.async_split.io_utils import atomic_json_save, read_json

    cfg_pair = _make_tiny_galcon_async_cfg()
    cfg_pair[0].policy.device = 'cpu'
    compiled = compile_config(
        copy.deepcopy(cfg_pair[0]), seed=0, env=None, auto=True, create_cfg=copy.deepcopy(cfg_pair[1]), save_cfg=False
    )
    policy = create_policy(copy.deepcopy(compiled.policy), model=None, enable_field=['learn', 'eval'])

    ckpt_dir = tmp_path / 'ckpts'
    ckpt_dir.mkdir(parents=True)
    for idx, it in enumerate((0, 100, 200, 300)):
        _save_dummy_ckpt(policy, ckpt_dir / f'iteration_{it}.pth.tar', train_iter=it, perturb=0.01 * idx)
    all_specs = discover_checkpoints_in_dir(ckpt_dir)

    out_dir = tmp_path / 'tourn_catchup_crash'
    t1 = OfflineTournament(
        default_input_cfg=cfg_pair,
        checkpoints=all_specs[:3],
        output_dir=out_dir,
        anchor_bots=('random',),
        games_per_pair=2,
        mode='ladder',
        num_rounds=3,  # chain + 2 Swiss rounds
        seed=11,
        device='cpu',
        env_num=2,
    )
    t1.run()

    # Passing ONLY the new checkpoint iter_300 on --resume must place it chronologically
    # AFTER iter_200 (not prepended before iter_0 next to bot:random).
    t2 = OfflineTournament(
        default_input_cfg=cfg_pair,
        checkpoints=[all_specs[3]],
        output_dir=out_dir,
        anchor_bots=('random',),
        games_per_pair=2,
        mode='ladder',
        num_rounds=3,
        seed=11,
        device='cpu',
        env_num=2,
        resume=True,
    )
    try:
        assert t2._ordered_player_ids == ['bot:random', 'iter_0', 'iter_100', 'iter_200', 'iter_300']
        assert 'iter_300' in t2._catchup_player_ids
        assert t2._catchup_rounds_remaining == 2
        # Simulate crash right after iter_300 was persisted to state (e.g., after chain edge finished)
        t2._chain_completed = True
        t2._save_state()
    finally:
        t2.close()

    # Re-resume: iter_300 is ALREADY in ladder.players, so _newly_added_ids is empty,
    # yet persisted catchup_player_ids + catchup_rounds_remaining must still execute both catch-up rounds!
    t3 = OfflineTournament(
        default_input_cfg=cfg_pair,
        checkpoints=[all_specs[3]],
        output_dir=out_dir,
        anchor_bots=('random',),
        games_per_pair=2,
        mode='ladder',
        num_rounds=3,
        seed=11,
        device='cpu',
        env_num=2,
        resume=True,
    )
    r3 = t3.run()
    assert r3['matches_played'] == 2
    state_after = read_json(out_dir / 'tournament_state.json')
    assert state_after['catchup_rounds_remaining'] == 0
    assert state_after['catchup_player_ids'] == []


@pytest.mark.unittest
def test_async_evaluator_closes_env_on_partial_init_failure(tmp_path, monkeypatch):
    import lzero.entry.async_split.evaluator as eval_mod

    cfg_pair = _make_tiny_galcon_async_cfg()
    closed_flags = {'env_closed': False}

    real_create_env_manager = eval_mod.create_env_manager

    def _wrapped_create_env_manager(*args, **kwargs):
        mgr = real_create_env_manager(*args, **kwargs)
        orig_close = mgr.close

        def _tracked_close():
            closed_flags['env_closed'] = True
            return orig_close()

        mgr.close = _tracked_close
        return mgr

    def _failing_create_policy(*args, **kwargs):
        raise RuntimeError('simulated policy init failure')

    monkeypatch.setattr(eval_mod, 'create_env_manager', _wrapped_create_env_manager)
    monkeypatch.setattr(eval_mod, 'create_policy', _failing_create_policy)

    with pytest.raises(RuntimeError, match='simulated policy init failure'):
        AsyncEvaluator(
            input_cfg=cfg_pair,
            run_dir=tmp_path / 'partial_init_fail',
            seed=0,
            device='cpu',
            evaluator_env_num=1,
            n_evaluator_episode=1,
            env_manager_type='base',
            eval_mode='bot',
        )
    assert closed_flags['env_closed'] is True


@pytest.mark.unittest
def test_elo_evaluator_promotion_uses_post_refit_champion_score(tmp_path):
    from lzero.worker.elo_rating import PlayerEntry
    from lzero.worker.head_to_head import MatchSummary

    cfg_pair = _make_tiny_galcon_async_cfg()
    run_dir = tmp_path / 'champ_score_align'
    elo_eval = AsyncEloEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        evaluator_env_num=1,
        games_per_opponent=10,
        opponents_per_eval=4,
        promotion_win_rate=0.55,
        promotion_elo_margin=50.0,
        anchor_bots=('random',),
    )
    try:
        p100 = elo_eval.dirs.elo_rungs_dir / 'iter_100.pth.tar'
        p200 = elo_eval.dirs.elo_rungs_dir / 'iter_200.pth.tar'
        _save_dummy_ckpt(elo_eval.candidate_policy, p100, train_iter=100)
        _save_dummy_ckpt(elo_eval.candidate_policy, p200, train_iter=200)

        elo_eval.ladder.register_player(
            PlayerEntry('iter_100', kind='checkpoint', ckpt_path=str(p100), train_iter=100, active=True)
        )
        elo_eval.ladder.register_player(
            PlayerEntry('iter_200', kind='checkpoint', ckpt_path=str(p200), train_iter=200, active=True)
        )
        elo_eval.ladder.record_match('iter_100', 'bot:random', wins_a=8, draws=0, losses_a=2)
        elo_eval.ladder.record_match('iter_200', 'bot:random', wins_a=8, draws=0, losses_a=2)
        elo_eval.ladder.record_match('iter_200', 'iter_100', wins_a=6, draws=0, losses_a=4)
        elo_eval.ladder.fit_ratings()
        assert elo_eval.ladder.get_champion().player_id == 'iter_200'

        # iter_300 beats pre-refit champion iter_200 (8-2 -> 80%), loses to iter_100 (4-6 -> 40%),
        # and beats random (9-1 -> 90%). After refit, iter_100 overtakes iter_200 as true champion,
        # and iter_300 is slightly above iter_100 (< +50 Elo) while having LOST 4-6 to iter_100 ->
        # must NOT falsely promote via beat_champion using stale iter_200's 80% score!
        outcomes = {
            'bot:random': (9, 0, 1),
            'iter_100': (4, 0, 6),
            'iter_200': (8, 0, 2),
        }

        def _fake_play_match(player_a, player_b, num_games, seed):
            w, d, l = outcomes[player_b.player_id]
            return MatchSummary(
                player_a=player_a.player_id,
                player_b=player_b.player_id,
                wins_a=w,
                draws=d,
                losses_a=l,
                p1_wins=w,
                p2_wins=l,
                total_steps=100,
                elapsed_sec=0.01,
                map_seeds=[1] * num_games,
            )

        elo_eval.runner.play_match = _fake_play_match
        res = elo_eval.evaluate_loaded_candidate(train_iter=300, envstep=3000)

        assert elo_eval.ladder.get_elo('iter_100') > elo_eval.ladder.get_elo('iter_200')
        assert res['win_rate_vs_champion'] == pytest.approx(0.4, abs=1e-6)
        assert res['promoted'] is False
    finally:
        elo_eval.close()


@pytest.mark.unittest
def test_offline_tournament_preserves_custom_order_and_inflight_catchup_on_resume(tmp_path):
    from lzero.entry.async_split.io_utils import atomic_json_save, read_json

    cfg_pair = _make_tiny_galcon_async_cfg()
    cfg_pair[0].policy.device = 'cpu'
    compiled = compile_config(
        copy.deepcopy(cfg_pair[0]), seed=0, env=None, auto=True, create_cfg=copy.deepcopy(cfg_pair[1]), save_cfg=False
    )
    policy = create_policy(copy.deepcopy(compiled.policy), model=None, enable_field=['learn', 'eval'])

    ckpt_dir = tmp_path / 'custom_ckpts'
    ckpt_dir.mkdir(parents=True)
    # Intentionally non-alphabetical custom labels ('early', 'mid', 'late', 'final')
    specs = []
    for idx, label in enumerate(('early', 'mid', 'late', 'final')):
        p = _save_dummy_ckpt(policy, ckpt_dir / f'{label}.pth.tar', train_iter=idx * 10, perturb=0.01 * idx)
        specs.extend(parse_checkpoint_spec(f'{label}={p}'))

    out_dir = tmp_path / 'tourn_custom_order'
    t1 = OfflineTournament(
        default_input_cfg=cfg_pair,
        checkpoints=specs[:3],  # early, mid, late
        output_dir=out_dir,
        anchor_bots=('random',),
        games_per_pair=2,
        mode='ladder',
        num_rounds=1,  # chain only
        seed=13,
        device='cpu',
        env_num=1,
    )
    t1.run()

    # Simulate an in-flight regular Swiss round when resuming with partial overlap ['late', 'final']
    state = read_json(out_dir / 'tournament_state.json')
    assert state['ordered_player_ids'] == ['bot:random', 'early', 'mid', 'late']
    state['completed_swiss_rounds'] = 0
    state['pending_swiss_pairs'] = [['bot:random', 'mid']]
    state['pending_is_catchup'] = False
    atomic_json_save(state, out_dir / 'tournament_state.json')

    t2 = OfflineTournament(
        default_input_cfg=cfg_pair,
        checkpoints=specs[2:],  # partial overlap: 'late' (existing) + 'final' (new)
        output_dir=out_dir,
        anchor_bots=('random',),
        games_per_pair=2,
        mode='ladder',
        num_rounds=2,  # 1 Swiss round total (currently in-flight without 'final')
        seed=13,
        device='cpu',
        env_num=1,
        resume=True,
    )
    try:
        # Exact original order must be preserved despite sort_keys=True and partial overlap!
        assert t2._ordered_player_ids == ['bot:random', 'early', 'mid', 'late', 'final']
        # In-flight regular Swiss round planned without 'final' -> 'final' must get 1 catch-up Swiss round!
        assert t2._catchup_rounds_remaining == 1
        assert 'final' in t2._catchup_player_ids
    finally:
        t2.close()


@pytest.mark.unittest
def test_async_evaluator_both_mode_resume_desync_skips_duplicate_elo(tmp_path):
    cfg_pair = _make_tiny_galcon_async_cfg()
    run_dir = tmp_path / 'both_desync'
    dirs = AsyncSplitDirs.from_root(run_dir)
    dirs.ensure_exists()

    ev1 = AsyncEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        evaluator_env_num=2,
        n_evaluator_episode=2,
        env_manager_type='base',
        eval_mode='both',
        elo_games_per_opponent=2,
        elo_opponents_per_eval=2,
    )
    try:
        publish_weights(dirs.latest_weights_path, ev1.policy._model.state_dict(), train_iter=5, envstep=50)
        res1 = ev1.eval_once_if_updated()
        assert res1 is not None and 'elo_result' in res1
    finally:
        ev1.close()

    # Simulate crash after elo_ladder/state.json saved iter 5 while evaluator_state.json is still at iter 4
    from lzero.entry.async_split.io_utils import atomic_json_save
    atomic_json_save({'last_eval_iter': 4, 'num_evals': 1, 'max_episode_return': 0.0}, dirs.evaluator_state_path)

    ev2 = AsyncEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        evaluator_env_num=2,
        n_evaluator_episode=2,
        env_manager_type='base',
        eval_mode='both',
        elo_games_per_opponent=2,
        elo_opponents_per_eval=2,
        resume=True,
    )
    try:
        res2 = ev2.eval_once_if_updated()
        assert res2 is not None
        # Elo evaluator must NOT duplicate iter_5 as iter_5_r2!
        assert 'elo_result' not in res2
        assert 'iter_5_r2' not in ev2.elo_evaluator.ladder.players
        assert ev2.elo_evaluator._num_evals == 1
    finally:
        ev2.close()


@pytest.mark.unittest
def test_cli_parses_elo_evaluator_launch_and_tournament_args():
    parser = build_parser()

    eval_args = parser.parse_args([
        'evaluator',
        '--eval-mode', 'elo_ladder',
        '--elo-games-per-opponent', '12',
        '--elo-opponents-per-eval', '5',
        '--elo-max-ladder-size', '12',
        '--elo-promotion-win-rate', '0.6',
        '--elo-promotion-margin', '40',
        '--elo-anchor-bots', 'random,fixed',
    ])
    assert eval_args.eval_mode == 'elo_ladder'
    assert eval_args.elo_games_per_opponent == 12
    assert eval_args.elo_anchor_bots == 'random,fixed'

    launch_args = parser.parse_args([
        'launch',
        '--num-actors', '2',
        '--eval-mode', 'both',
        '--elo-games-per-opponent', '8',
    ])
    assert launch_args.eval_mode == 'both'
    assert launch_args.elo_games_per_opponent == 8

    tourn_args = parser.parse_args([
        'tournament',
        '--ckpt-dir', 'data_sez/run1/ckpt',
        '--mode', 'ladder',
        '--num-rounds', '4',
        '--games-per-pair', '10',
        '--anchor-bots', 'random',
        '--write-tb',
        '--resume',
    ])
    assert tourn_args.command == 'tournament'
    assert tourn_args.ckpt_dir == ['data_sez/run1/ckpt']
    assert tourn_args.mode == 'ladder'
    assert tourn_args.write_tb is True
    assert tourn_args.resume is True


@pytest.mark.unittest
def test_head_to_head_and_elo_evaluator_support_mcts_anchor_bot(tmp_path):
    """Verify MCTS bot (default and parameterized mcts:<sims>) works in HeadToHeadRunner,
    AsyncEloEvaluator (default Galcon anchors include both random and mcts), and CLI defaults.
    """
    parser = build_parser()
    assert parser.parse_args(['evaluator']).elo_anchor_bots == 'random,mcts:16'
    assert parser.parse_args(['launch']).elo_anchor_bots == 'random,mcts:16'
    assert parser.parse_args(['tournament']).anchor_bots == 'random,mcts:16'

    cfg_pair = _make_tiny_galcon_async_cfg()
    cfg_pair[0].env.max_episode_steps = 12
    cfg_pair[0].env.mcts_num_simulations = 16
    compiled = compile_config(
        copy.deepcopy(cfg_pair[0]), seed=0, env=None, auto=True, create_cfg=copy.deepcopy(cfg_pair[1]), save_cfg=False
    )
    env_fn, collector_env_cfg, _ = get_vec_env_setting(compiled.env)
    runner = HeadToHeadRunner(env_fn=env_fn, env_cfg=collector_env_cfg[0], env_num=2)
    try:
        bot_mcts = EnvBotMatchPlayer('mcts:16')
        bot_pass = EnvBotMatchPlayer('pass')
        summary = runner.play_match(bot_mcts, bot_pass, num_games=2, seed=19)
        assert summary.total_games == 2
        assert summary.win_rate_a >= 0.5
    finally:
        runner.close()

    run_dir = tmp_path / 'elo_mcts_default_anchors'
    dirs = AsyncSplitDirs.from_root(run_dir)
    dirs.ensure_exists()
    elo_eval = AsyncEloEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=7,
        device='cpu',
        evaluator_env_num=2,
        games_per_opponent=2,
        opponents_per_eval=4,
        max_ladder_size=4,
        min_iter_gap=0,
    )
    try:
        assert elo_eval.ladder.anchor_bot_types == ['random', 'mcts:16']
        assert elo_eval.ladder.anchor_id == 'bot:random'
        publish_weights(
            dirs.latest_weights_path,
            elo_eval.candidate_policy._model.state_dict(),
            train_iter=0,
            envstep=0,
        )
        res = elo_eval.eval_once_if_updated()
        assert res is not None
        opps_played = {m['opponent_id'] for m in res['matches']}
        assert {'bot:random', 'bot:mcts:16'}.issubset(opps_played)
    finally:
        elo_eval.close()

    tb_tags = _read_tb_tags(dirs.log_dir / 'elo_evaluator')
    assert 'elo_iter/win_rate_vs_bot_random' in tb_tags
    assert 'elo_iter/win_rate_vs_bot_mcts_16' in tb_tags

