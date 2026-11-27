# -*- coding: utf-8 -*-
"""Unit and end-to-end tests for async_split restart state recovery."""

import copy
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import torch

from lzero.entry.async_split import AsyncEvaluator, AsyncLearner, SelfPlayActor
from lzero.entry.async_split.cli import (
    _stat_signature,
    _wait_for_learner_weights_refresh,
    build_parser,
)
from lzero.entry.async_split.io_utils import (
    AsyncSplitDirs,
    atomic_json_save,
    atomic_pickle_save,
    detect_existing_run_artifacts,
    extract_optimizer_step,
    find_latest_checkpoint,
    parse_checkpoint_iter,
    publish_weights,
    read_json,
    write_segment_batch,
)
from lzero.entry.async_split.tb_utils import (
    inspect_tb_max_steps,
    is_envstep_tag,
    list_tb_event_files,
    truncate_tb_events_after,
)
from lzero.entry.tests.test_async_split_e2e import _make_tiny_galcon_async_cfg


def _read_tb_scalar_steps_in_walltime_order(log_dir: Path, tag: str) -> List[Tuple[float, int, float]]:
    """Read (wall_time, step, value) across all TensorBoard event files in wall-time order."""
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

    records: List[Tuple[float, int, float]] = []
    event_files = sorted(log_dir.glob('events.out.tfevents.*'))
    for ef in event_files:
        for event in EventFileLoader(str(ef)).Load():
            if not event.HasField('summary'):
                continue
            for val in event.summary.value:
                if val.tag == tag:
                    records.append((float(event.wall_time), int(event.step), float(val.simple_value)))
    records.sort(key=lambda r: r[0])
    return records


@pytest.mark.unittest
def test_find_latest_checkpoint_numeric_order_and_detect_artifacts(tmp_path):
    dirs = AsyncSplitDirs.from_root(tmp_path / 'run')
    dirs.ensure_exists()

    # Pre-staged actor segments must NOT trigger existing-run guardrail
    write_segment_batch(
        dirs.incoming_dir,
        segments=['seg'],
        metas=[{'done': True}],
        actor_id='actor_0',
        seq=1,
        envsteps=10,
        episodes=1,
        weight_version=0,
    )
    assert detect_existing_run_artifacts(dirs) == []
    assert find_latest_checkpoint(dirs.ckpt_dir) is None

    # Sibling evaluator creating its empty event file first must NOT block fresh learner startup
    (dirs.log_dir / 'evaluator').mkdir(parents=True, exist_ok=True)
    (dirs.log_dir / 'evaluator' / 'events.out.tfevents.12345').write_bytes(b'')
    assert detect_existing_run_artifacts(dirs, include_evaluator_logs=False) == []
    assert len(detect_existing_run_artifacts(dirs, include_evaluator_logs=True)) == 1

    # Create checkpoints where lexicographical order != numeric order
    for name in (
        'iteration_9000.pth.tar',
        'iteration_99000.pth.tar',
        'iteration_100000.pth.tar',
        'ckpt_best.pth.tar',
        'WM_ckpt_best.pth.tar',
        '.iteration_999999.pth.tar.tmp.123',
    ):
        (dirs.ckpt_dir / name).write_bytes(b'x')

    latest = find_latest_checkpoint(dirs.ckpt_dir)
    assert latest is not None
    assert latest.name == 'iteration_100000.pth.tar'
    assert parse_checkpoint_iter(latest) == 100000
    assert parse_checkpoint_iter(dirs.ckpt_dir / 'ckpt_best.pth.tar') is None
    assert parse_checkpoint_iter(dirs.ckpt_dir / 'WM_ckpt_best.pth.tar') is None

    artifacts = detect_existing_run_artifacts(dirs)
    assert any('ckpt/' in item for item in artifacts)


@pytest.mark.unittest
def test_fresh_run_refuses_existing_run_dir_unless_resume_or_overwrite(tmp_path):
    run_dir = tmp_path / 'guarded_run'
    cfg_pair = _make_tiny_galcon_async_cfg()

    # Resuming an empty directory must fail fast
    with pytest.raises(FileNotFoundError, match='Cannot resume'):
        AsyncLearner(input_cfg=cfg_pair, run_dir=run_dir, device='cpu', resume=True, use_bg_reader=False)

    learner = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        save_ckpt_every_n_updates=0,
        use_bg_reader=False,
    )
    learner.close()

    # Starting fresh again in the same run_dir without --resume or --overwrite must fail fast
    with pytest.raises(FileExistsError, match='--resume'):
        AsyncLearner(input_cfg=cfg_pair, run_dir=run_dir, device='cpu', use_bg_reader=False)

    # Explicit overwrite=True is allowed and starts at train_iter == 0
    learner_overwrite = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        overwrite=True,
        save_ckpt_every_n_updates=0,
        use_bg_reader=False,
    )
    try:
        assert learner_overwrite.learner.train_iter == 0
    finally:
        learner_overwrite.close()


@pytest.mark.unittest
def test_async_split_graceful_restart_recovers_full_state_and_monotonic_tb(tmp_path):
    """End-to-end restart test:
    1. Run learner + actor + evaluator for 3 updates, then shut down gracefully.
    2. Resume learner and evaluator in the same run_dir with resume=True (auto-selecting latest ckpt).
    3. Verify train_iter, envsteps, episodes, files_ingested, weights, optimizer, replay buffer,
       evaluator best score / last_eval_iter, and monotonic TensorBoard steps across restart.
    """
    run_dir = tmp_path / 'galcon_async_resume_e2e'
    cfg_pair = _make_tiny_galcon_async_cfg()
    cfg_pair[0].policy.learn = dict(learner=dict(hook=dict(log_show_after_iter=1)))

    # --- Session 1 ---
    learner1 = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        publish_every_n_updates=1,
        save_ckpt_every_n_updates=2,
        max_replay_ratio=0.5,
        min_transitions=8,
        use_bg_reader=False,
    )
    actor = SelfPlayActor(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        actor_id='actor_0',
        seed=10,
        device='cpu',
        env_manager_type='base',
        weight_sync_interval=1,
    )
    evaluator1 = AsyncEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=20,
        device='cpu',
        evaluator_env_num=2,
        n_evaluator_episode=2,
        env_manager_type='base',
        min_iter_gap=0,
    )

    try:
        actor.step_once(n_episode=2)
        learner1.ingest_pending_segments()
        for _ in range(3):
            lv = learner1.train_step()
            assert lv is not None
            assert np.isfinite(float(lv['total_loss']))

        assert learner1.learner.train_iter == 3
        assert (learner1.dirs.ckpt_dir / 'iteration_2.pth.tar').is_file()
        iter2_mtime_before = (learner1.dirs.ckpt_dir / 'iteration_2.pth.tar').stat().st_mtime_ns

        st = learner1.dirs.latest_weights_path.stat()
        os.utime(learner1.dirs.latest_weights_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        eval1 = evaluator1.eval_once_if_updated()
        assert eval1 is not None
        assert eval1['train_iter'] == 3
        assert evaluator1.dirs.evaluator_state_path.is_file()

        pre_envsteps = learner1.total_ingested_envsteps
        pre_episodes = learner1.total_ingested_episodes
        pre_files = learner1.total_files_ingested
        pre_buf_segments = learner1.replay_buffer.get_num_of_game_segments()
        pre_buf_transitions = learner1.replay_buffer.get_num_of_transitions()
        pre_model_sd = {k: v.detach().cpu().clone() for k, v in learner1.policy._learn_model.state_dict().items()}
        pre_best_return = float(evaluator1.evaluator._max_episode_return)
    finally:
        evaluator1.close()
        learner1.close()

    # Graceful shutdown must have persisted checkpoint, learner_state.json, and replay_buffer.pkl
    assert (learner1.dirs.ckpt_dir / 'iteration_3.pth.tar').is_file()
    assert learner1.dirs.learner_state_path.is_file()
    assert learner1.dirs.replay_buffer_path.is_file()

    # --- Session 2 (resume=True) ---
    learner2 = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        resume=True,
        publish_every_n_updates=1,
        save_ckpt_every_n_updates=2,
        max_replay_ratio=0.5,
        min_transitions=8,
        use_bg_reader=False,
    )
    evaluator2 = AsyncEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=20,
        device='cpu',
        evaluator_env_num=2,
        n_evaluator_episode=2,
        env_manager_type='base',
        min_iter_gap=0,
        resume=True,
    )

    try:
        # All learner counters, weights, and replay buffer must be restored
        assert learner2.learner.train_iter == 3
        assert learner2.total_ingested_envsteps == pre_envsteps
        assert learner2.total_ingested_episodes == pre_episodes
        assert learner2.total_files_ingested == pre_files
        assert learner2.replay_buffer.get_num_of_game_segments() == pre_buf_segments
        assert learner2.replay_buffer.get_num_of_transitions() == pre_buf_transitions

        for k, v in pre_model_sd.items():
            assert torch.allclose(learner2.policy._learn_model.state_dict()[k].cpu(), v)

        # Published latest_weights.pt must carry train_iter == 3 (never reset to 0)
        published = torch.load(str(learner2.dirs.latest_weights_path), map_location='cpu', weights_only=False)
        assert published['train_iter'] == 3
        assert published['envstep'] == pre_envsteps

        # Evaluator restored its best score and last_eval_iter, and skips duplicate eval at iter 3
        assert evaluator2._last_eval_iter == 3
        assert pytest.approx(float(evaluator2.evaluator._max_episode_return)) == pre_best_return
        assert evaluator2.eval_once_if_updated() is None

        # Throttler blocks stale training before fresh actor transitions arrive
        assert learner2.train_step() is None

        # Actor syncs resumed weights (iter 3), collects fresh episodes, learner continues to iter 5
        st = learner2.dirs.latest_weights_path.stat()
        os.utime(learner2.dirs.latest_weights_path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000))
        actor.step_once(n_episode=2)
        assert actor.current_weight_iter == 3

        learner2.ingest_pending_segments()
        assert learner2.total_ingested_envsteps > pre_envsteps
        for _ in range(2):
            lv = learner2.train_step()
            assert lv is not None
            assert np.isfinite(float(lv['total_loss']))

        assert learner2.learner.train_iter == 5
        assert (learner2.dirs.ckpt_dir / 'iteration_5.pth.tar').is_file()
        # Earlier iteration_2 checkpoint must remain untouched
        assert (learner2.dirs.ckpt_dir / 'iteration_2.pth.tar').stat().st_mtime_ns == iter2_mtime_before

        # Evaluator evaluates iter 5, but must NOT overwrite ckpt_best if historical max is higher
        evaluator2.evaluator._max_episode_return = 999.0
        best_ckpt = evaluator2.dirs.ckpt_dir / 'ckpt_best.pth.tar'
        best_before = torch.load(str(best_ckpt), map_location='cpu', weights_only=False)
        st = learner2.dirs.latest_weights_path.stat()
        os.utime(learner2.dirs.latest_weights_path, ns=(st.st_atime_ns, st.st_mtime_ns + 3_000_000))
        eval2 = evaluator2.eval_once_if_updated()
        assert eval2 is not None
        assert eval2['train_iter'] == 5
        best_after = torch.load(str(best_ckpt), map_location='cpu', weights_only=False)
        assert best_after['train_iter'] == best_before['train_iter'] == 3
    finally:
        actor.close()
        evaluator2.close()
        learner2.close()

    # Verify TensorBoard learner steps are strictly monotonic across both event files (no jump back to 0!)
    value_loss_steps = [
        step for _, step, _ in _read_tb_scalar_steps_in_walltime_order(
            learner2.dirs.log_dir / 'learner', 'learner_iter/value_loss_avg'
        )
    ]
    assert value_loss_steps == [0, 1, 2, 3, 4]

    async_steps = [
        step for _, step, _ in _read_tb_scalar_steps_in_walltime_order(
            learner2.dirs.log_dir / 'learner', 'async_learner/ingested_envsteps'
        )
    ]
    assert async_steps == sorted(async_steps)
    assert min(async_steps) >= 1 and max(async_steps) == 5

    eval_steps = [
        step for _, step, _ in _read_tb_scalar_steps_in_walltime_order(
            learner2.dirs.log_dir / 'evaluator', 'evaluator_iter/eval_episode_return_mean'
        )
    ]
    assert eval_steps == [3, 5]


@pytest.mark.unittest
def test_resume_falls_back_to_checkpoint_counters_and_skips_stale_buffer(tmp_path):
    """Verify resuming a legacy/crashed checkpoint (missing learner_state.json, stale replay_buffer.pkl)
    still restores train_iter and last_step from the .pth.tar file and refuses stale buffer data.
    """
    run_dir = tmp_path / 'legacy_resume'
    cfg_pair = _make_tiny_galcon_async_cfg()

    learner1 = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        save_ckpt_every_n_updates=0,
        use_bg_reader=False,
    )
    try:
        # Fabricate a legacy checkpoint saved at iter=42, last_step=12345 (no learner_state.json)
        ckpt_state = learner1.policy.learn_mode.state_dict()
        ckpt_state['last_iter'] = 42
        ckpt_state['last_step'] = 12345
        torch.save(ckpt_state, str(learner1.dirs.ckpt_dir / 'iteration_42.pth.tar'))
    finally:
        learner1.close()

    # Remove learner_state.json and write a mismatched replay_buffer.pkl (train_iter=10 != 42)
    if learner1.dirs.learner_state_path.exists():
        learner1.dirs.learner_state_path.unlink()
    atomic_pickle_save(
        {
            'version': 1,
            'train_iter': 10,
            'game_segment_buffer': ['stale_seg'],
            'game_pos_priorities': np.array([1.0]),
            'game_segment_game_pos_look_up': [(0, 0)],
            'num_of_collected_episodes': 1,
            'base_idx': 0,
            'clear_time': 0.0,
        },
        learner1.dirs.replay_buffer_path,
    )

    learner2 = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        resume=True,
        save_ckpt_every_n_updates=0,
        use_bg_reader=False,
    )
    try:
        assert learner2.learner.train_iter == 42
        assert learner2.total_ingested_envsteps == 12345
        # Stale buffer (iter 10 vs 42) must NOT be loaded
        assert learner2.replay_buffer.get_num_of_game_segments() == 0
    finally:
        learner2.close()


@pytest.mark.unittest
def test_resume_restores_lr_scheduler_state(tmp_path):
    """Verify cos_lr_scheduler and piecewise_decay_lr_scheduler state survives checkpoint + resume."""
    for scheduler_key in ('cos_lr_scheduler', 'piecewise_decay_lr_scheduler'):
        run_dir = tmp_path / f'lr_resume_{scheduler_key}'
        cfg_pair = _make_tiny_galcon_async_cfg()
        cfg_pair[0].policy[scheduler_key] = True
        cfg_pair[0].policy.threshold_training_steps_for_final_lr = 4

        actor = SelfPlayActor(
            input_cfg=cfg_pair,
            run_dir=run_dir,
            actor_id='actor_lr',
            seed=10,
            device='cpu',
            env_manager_type='base',
        )
        learner1 = AsyncLearner(
            input_cfg=cfg_pair,
            run_dir=run_dir,
            seed=0,
            device='cpu',
            save_ckpt_every_n_updates=0,
            max_replay_ratio=None,
            min_transitions=4,
            use_bg_reader=False,
        )
        try:
            actor.step_once(n_episode=2)
            learner1.ingest_pending_segments()
            for _ in range(3):
                assert learner1.train_step() is not None
            expected_last_epoch = learner1.policy.lr_scheduler.last_epoch
            expected_lr = learner1.policy._optimizer.param_groups[0]['lr']
            assert expected_last_epoch == 3
        finally:
            actor.close()
            learner1.close()

        learner2 = AsyncLearner(
            input_cfg=cfg_pair,
            run_dir=run_dir,
            seed=0,
            device='cpu',
            resume=True,
            save_ckpt_every_n_updates=0,
            max_replay_ratio=None,
            min_transitions=4,
            use_bg_reader=False,
        )
        try:
            assert learner2.policy.lr_scheduler.last_epoch == expected_last_epoch
            assert pytest.approx(learner2.policy._optimizer.param_groups[0]['lr']) == expected_lr
        finally:
            learner2.close()


@pytest.mark.unittest
def test_cli_parser_resume_and_overwrite_flags():
    parser = build_parser()

    l_args = parser.parse_args(['learner', '--run-dir', '/tmp/r', '--resume', '--no-persist-buffer'])
    assert l_args.resume is True
    assert l_args.overwrite is False
    assert l_args.no_persist_buffer is True

    e_args = parser.parse_args(['evaluator', '--run-dir', '/tmp/r', '--resume'])
    assert e_args.resume is True

    launch_args = parser.parse_args(['launch', '--run-dir', '/tmp/r', '--resume'])
    assert launch_args.resume is True
    assert launch_args.overwrite is False
    assert launch_args.no_persist_buffer is False

    launch_ow = parser.parse_args(['launch', '--run-dir', '/tmp/r', '--overwrite', '--no-persist-buffer'])
    assert launch_ow.resume is False
    assert launch_ow.overwrite is True
    assert launch_ow.no_persist_buffer is True


@pytest.mark.unittest
def test_close_skips_checkpoint_save_when_weights_are_non_finite(tmp_path):
    """Ensure graceful shutdown never writes a NaN-poisoned checkpoint that --resume would pick up."""
    run_dir = tmp_path / 'nan_shutdown'
    cfg_pair = _make_tiny_galcon_async_cfg()

    learner = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        save_ckpt_every_n_updates=0,
        use_bg_reader=False,
    )
    # Corrupt a parameter with NaN before closing
    with torch.no_grad():
        first_param = next(learner.policy._learn_model.parameters())
        first_param.fill_(float('nan'))

    learner.close()
    assert find_latest_checkpoint(learner.dirs.ckpt_dir) is None
    assert not learner.dirs.learner_state_path.exists()
    assert not learner.dirs.replay_buffer_path.exists()


@pytest.mark.unittest
def test_resume_refuses_optimizer_step_ahead_of_last_iter(tmp_path):
    """A checkpoint whose optimizer step exceeds last_iter (from a legacy counter reset)
    must be rejected on --resume instead of silently overwriting checkpoints and drawing
    backward TensorBoard lines.
    """
    run_dir = tmp_path / 'reset_counter_run'
    cfg_pair = _make_tiny_galcon_async_cfg()

    learner1 = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        save_ckpt_every_n_updates=0,
        use_bg_reader=False,
    )
    try:
        ckpt_state = learner1.policy.learn_mode.state_dict()
        ckpt_state['last_iter'] = 39251
        ckpt_state['last_step'] = 981295
        ckpt_state['optimizer'] = {'state': {0: {'step': torch.tensor(139426.0)}}, 'param_groups': []}
        assert extract_optimizer_step(ckpt_state) == 139426
        bad_ckpt = learner1.dirs.ckpt_dir / 'iteration_39251.pth.tar'
        torch.save(ckpt_state, str(bad_ckpt))
    finally:
        learner1.close()

    with pytest.raises(ValueError, match='optimizer state has step=139426'):
        AsyncLearner(
            input_cfg=cfg_pair,
            run_dir=run_dir,
            seed=0,
            device='cpu',
            resume=True,
            model_path=str(bad_ckpt),
            save_ckpt_every_n_updates=0,
            use_bg_reader=False,
        )


@pytest.mark.unittest
def test_resume_refuses_explicit_model_path_older_than_latest_checkpoint(tmp_path):
    """Explicit --resume --model-path pointing to an earlier iteration than existing
    checkpoints in the same run_dir must fail fast instead of regressing steps.
    """
    run_dir = tmp_path / 'step_regression_run'
    cfg_pair = _make_tiny_galcon_async_cfg()

    learner1 = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        save_ckpt_every_n_updates=0,
        use_bg_reader=False,
    )
    try:
        base_sd = learner1.policy.learn_mode.state_dict()
        older_ckpt = learner1.dirs.ckpt_dir / 'iteration_10.pth.tar'
        newer_ckpt = learner1.dirs.ckpt_dir / 'iteration_50.pth.tar'
        torch.save({**base_sd, 'last_iter': 10, 'last_step': 200}, str(older_ckpt))
        torch.save({**base_sd, 'last_iter': 50, 'last_step': 1000}, str(newer_ckpt))
    finally:
        learner1.close()

    with pytest.raises(ValueError, match='newer checkpoint'):
        AsyncLearner(
            input_cfg=cfg_pair,
            run_dir=run_dir,
            seed=0,
            device='cpu',
            resume=True,
            model_path=str(older_ckpt),
            save_ckpt_every_n_updates=0,
            use_bg_reader=False,
        )


@pytest.mark.unittest
def test_unclean_crash_rollback_prunes_orphaned_tb_events_and_clamps_eval_state(tmp_path):
    """Simulate a hard crash (SIGKILL) between periodic checkpoints:
    - Periodic checkpoint saved at iter 2, then learner continues to iter 3 and evaluator
      evaluates iter 2 and iter 3 before crashing without graceful shutdown.
    - Resuming from iteration_2.pth.tar must prune orphaned events (> iter 2 / > envstep@2)
      across BOTH iter-axis and envstep-axis tags without wiping valid envstep history,
      roll back evaluator_state.json from 3 -> 2, and produce strictly non-decreasing
      TensorBoard steps across every tag when continuing to iter 5.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = tmp_path / 'crash_rollback_e2e'
    cfg_pair = _make_tiny_galcon_async_cfg()
    cfg_pair[0].policy.learn = dict(learner=dict(hook=dict(log_show_after_iter=1)))

    actor = SelfPlayActor(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        actor_id='actor_0',
        seed=10,
        device='cpu',
        env_manager_type='base',
        weight_sync_interval=1,
    )
    learner1 = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        publish_every_n_updates=1,
        save_ckpt_every_n_updates=2,
        max_replay_ratio=None,
        min_transitions=4,
        use_bg_reader=False,
    )
    evaluator1 = AsyncEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=20,
        device='cpu',
        evaluator_env_num=2,
        n_evaluator_episode=2,
        env_manager_type='base',
        min_iter_gap=0,
    )

    try:
        actor.step_once(n_episode=2)
        learner1.ingest_pending_segments()
        for _ in range(2):
            assert learner1.train_step() is not None
        assert learner1.learner.train_iter == 2
        assert (learner1.dirs.ckpt_dir / 'iteration_2.pth.tar').is_file()
        envsteps_at_2 = learner1.total_ingested_envsteps

        # Evaluate at iter 2 (valid pre-checkpoint eval)
        st = learner1.dirs.latest_weights_path.stat()
        os.utime(learner1.dirs.latest_weights_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        ev_at_2 = evaluator1.eval_once_if_updated()
        assert ev_at_2 is not None and ev_at_2['train_iter'] == 2

        # Ingest more data and train to iter 3 (NOT a periodic checkpoint multiple -> no iter 3 ckpt)
        actor.step_once(n_episode=2)
        learner1.ingest_pending_segments()
        assert learner1.total_ingested_envsteps > envsteps_at_2
        assert learner1.train_step() is not None
        assert learner1.learner.train_iter == 3
        assert not (learner1.dirs.ckpt_dir / 'iteration_3.pth.tar').is_file()

        # Evaluator also advances past checkpoint 2 to iter 3 before crash
        st = learner1.dirs.latest_weights_path.stat()
        os.utime(learner1.dirs.latest_weights_path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000))
        ev_at_3 = evaluator1.eval_once_if_updated()
        assert ev_at_3 is not None and ev_at_3['train_iter'] == 3
        assert read_json(learner1.dirs.evaluator_state_path)['last_eval_iter'] == 3

        # Simulate SIGKILL: flush writers WITHOUT calling learner1.close() so no iteration_3.pth.tar is saved
        learner1.tb_logger.flush()
        learner1.tb_logger.close()
        evaluator1.close()
    except Exception:
        actor.close()
        raise

    # Confirm orphaned iter-3 events exist on disk before resume
    assert inspect_tb_max_steps(run_dir / 'log' / 'learner')['max_iter'] == 3
    assert inspect_tb_max_steps(run_dir / 'log' / 'learner')['max_envstep'] > envsteps_at_2
    assert inspect_tb_max_steps(run_dir / 'log' / 'evaluator')['max_iter'] == 3

    # --- Session 2: Resume after crash (auto-picks iteration_2.pth.tar) ---
    learner2 = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        resume=True,
        publish_every_n_updates=1,
        save_ckpt_every_n_updates=2,
        max_replay_ratio=None,
        min_transitions=4,
        use_bg_reader=False,
    )
    evaluator2 = AsyncEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=20,
        device='cpu',
        evaluator_env_num=2,
        n_evaluator_episode=2,
        env_manager_type='base',
        min_iter_gap=0,
        resume=True,
    )

    try:
        assert learner2.learner.train_iter == 2
        assert learner2.total_ingested_envsteps == envsteps_at_2
        # Evaluator state must have been rolled back from 3 -> 2
        assert evaluator2._last_eval_iter == 2
        assert read_json(learner2.dirs.evaluator_state_path)['last_eval_iter'] == 2
        # Published weights must already reflect reconciled iter 2
        published = torch.load(str(learner2.dirs.latest_weights_path), map_location='cpu', weights_only=False)
        assert published['train_iter'] == 2
        assert published['envstep'] == envsteps_at_2

        # Continue training 2->5 and evaluate at 5
        actor.step_once(n_episode=2)
        learner2.ingest_pending_segments()
        for _ in range(3):
            assert learner2.train_step() is not None
        assert learner2.learner.train_iter == 5

        st = learner2.dirs.latest_weights_path.stat()
        os.utime(learner2.dirs.latest_weights_path, ns=(st.st_atime_ns, st.st_mtime_ns + 3_000_000))
        ev_at_5 = evaluator2.eval_once_if_updated()
        assert ev_at_5 is not None and ev_at_5['train_iter'] == 5
    finally:
        actor.close()
        evaluator2.close()
        learner2.close()

    # Check EVERY scalar tag in BOTH learner and evaluator via TensorBoard's real EventAccumulator
    for sub in ('learner', 'evaluator'):
        ea = EventAccumulator(str(run_dir / 'log' / sub), size_guidance={'scalars': 0})
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        assert len(tags) > 0
        for tag in tags:
            steps = [e.step for e in ea.Scalars(tag)]
            assert steps == sorted(steps), f'Backward step jump in {sub}/{tag}: {steps}'

    # Verify exact continuity on both iter-axis AND envstep-axis (no wiped envstep history!)
    ea_learner = EventAccumulator(str(run_dir / 'log' / 'learner'), size_guidance={'scalars': 0})
    ea_learner.Reload()
    assert [e.step for e in ea_learner.Scalars('learner_iter/value_loss_avg')] == [0, 1, 2, 3, 4]
    learner_step_steps = [e.step for e in ea_learner.Scalars('learner_step/value_loss_avg')]
    assert len(learner_step_steps) == 5
    assert learner_step_steps[0] == envsteps_at_2 and learner_step_steps[-1] > envsteps_at_2

    ea_eval = EventAccumulator(str(run_dir / 'log' / 'evaluator'), size_guidance={'scalars': 0})
    ea_eval.Reload()
    assert [e.step for e in ea_eval.Scalars('evaluator_iter/eval_episode_return_mean')] == [2, 5]
    eval_step_steps = [e.step for e in ea_eval.Scalars('evaluator_step/eval_episode_return_mean')]
    assert len(eval_step_steps) == 2 and eval_step_steps[0] == envsteps_at_2 and eval_step_steps[1] > envsteps_at_2


@pytest.mark.unittest
def test_evaluator_resume_infers_last_eval_iter_from_tb_when_state_json_missing(tmp_path):
    """If evaluator_state.json is absent when resuming a legacy run, AsyncEvaluator must
    infer last_eval_iter from existing TensorBoard event files so it never logs backwards.
    """
    run_dir = tmp_path / 'eval_tb_fallback'
    cfg_pair = _make_tiny_galcon_async_cfg()

    learner = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        save_ckpt_every_n_updates=0,
        use_bg_reader=False,
    )
    evaluator1 = AsyncEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=20,
        device='cpu',
        evaluator_env_num=2,
        n_evaluator_episode=2,
        env_manager_type='base',
        min_iter_gap=0,
    )
    try:
        publish_weights(
            learner.dirs.latest_weights_path,
            learner.policy._learn_model.state_dict(),
            train_iter=100,
            envstep=2500,
        )
        res = evaluator1.eval_once_if_updated()
        assert res is not None and res['train_iter'] == 100
    finally:
        evaluator1.close()
        learner.close()

    # Delete evaluator_state.json to simulate a legacy pre-resume run
    learner.dirs.evaluator_state_path.unlink()
    assert not learner.dirs.evaluator_state_path.exists()

    # Publish a lower train_iter (e.g. stale/regressed weights)
    publish_weights(
        learner.dirs.latest_weights_path,
        learner.policy._learn_model.state_dict(),
        train_iter=40,
        envstep=1000,
    )

    evaluator2 = AsyncEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=20,
        device='cpu',
        evaluator_env_num=2,
        n_evaluator_episode=2,
        env_manager_type='base',
        min_iter_gap=0,
        resume=True,
    )
    try:
        assert evaluator2._last_eval_iter == 100
        # Must refuse to evaluate train_iter=40 <= 100
        assert evaluator2.eval_once_if_updated() is None
    finally:
        evaluator2.close()


@pytest.mark.unittest
def test_tb_utils_tag_classification_clean_noop_and_corrupt_tail(tmp_path):
    from tensorboardX import SummaryWriter

    assert is_envstep_tag('learner_step/value_loss_avg') is True
    assert is_envstep_tag('evaluator_step/eval_episode_return_mean') is True
    assert is_envstep_tag('evaluator_step_task0/reward_mean') is True
    assert is_envstep_tag('learner_iter/value_loss_avg') is False
    assert is_envstep_tag('async_learner/ingested_envsteps') is False
    assert is_envstep_tag('evaluator_iter/avg_envstep_per_episode') is False

    log_dir = tmp_path / 'tb_dir'
    w1 = SummaryWriter(str(log_dir), filename_suffix='.session1')
    w1.add_scalar('learner_iter/loss', 1.0, 0)
    w1.add_scalar('learner_iter/loss', 0.9, 1)
    w1.add_scalar('async_learner/ingested_envsteps', 100.0, 2)
    w1.add_scalar('learner_step/loss', 0.9, 100)
    w1.flush()
    w1.close()

    ef1 = list_tb_event_files(log_dir)[0]
    mtime_before = ef1.stat().st_mtime_ns

    # Clean cutoff at (iter=2, envstep=100) must leave ef1 completely untouched
    stats_clean = truncate_tb_events_after(log_dir, max_iter=2, max_envstep=100)
    assert stats_clean == {'files_modified': 0, 'files_removed': 0, 'events_dropped': 0}
    assert ef1.stat().st_mtime_ns == mtime_before

    # Write a second file whose events are ALL post-checkpoint, plus corrupt tail on ef1
    w2 = SummaryWriter(str(log_dir), filename_suffix='.session2')
    w2.add_scalar('learner_iter/loss', 0.8, 2)
    w2.add_scalar('async_learner/ingested_envsteps', 150.0, 3)
    w2.add_scalar('learner_step/loss', 0.8, 150)
    w2.flush()
    w2.close()
    assert len(list_tb_event_files(log_dir)) == 2

    with open(ef1, 'ab') as f:
        f.write(b'\x99\x88\x77\x66\x55')  # simulated SIGKILL partial trailing bytes

    stats_prune = truncate_tb_events_after(log_dir, max_iter=2, max_envstep=100)
    assert stats_prune['files_removed'] == 1
    assert stats_prune['files_modified'] == 1
    assert stats_prune['events_dropped'] == 3
    assert len(list_tb_event_files(log_dir)) == 1
    assert inspect_tb_max_steps(log_dir) == {'max_iter': 2, 'max_envstep': 100}


@pytest.mark.unittest
def test_wait_for_learner_weights_refresh_sync_and_early_exit(tmp_path):
    weights_path = tmp_path / 'latest_weights.pt'
    weights_path.write_bytes(b'old_weights')
    prev_sig = _stat_signature(weights_path)

    class _FakeProc:
        def __init__(self, rc=None):
            self._rc = rc
            self.calls = 0

        def poll(self):
            self.calls += 1
            return self._rc

    # Case 1: Learner exits early (e.g., guardrail error) -> returns immediately without waiting
    failed_proc = _FakeProc(rc=1)
    ready, rc = _wait_for_learner_weights_refresh(
        weights_path=weights_path,
        prev_sig=prev_sig,
        learner_proc=failed_proc,
        stop_check=lambda: False,
        timeout_sec=5.0,
        poll_interval=0.01,
    )
    assert ready is False
    assert rc == 1

    # Case 2: Learner atomically replaces weights file -> detected promptly
    running_proc = _FakeProc(rc=None)
    tmp_new = tmp_path / '.latest_weights.pt.tmp'
    tmp_new.write_bytes(b'fresh_reconciled_weights')
    os.replace(str(tmp_new), str(weights_path))
    st = weights_path.stat()
    os.utime(weights_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    ready2, rc2 = _wait_for_learner_weights_refresh(
        weights_path=weights_path,
        prev_sig=prev_sig,
        learner_proc=running_proc,
        stop_check=lambda: False,
        timeout_sec=5.0,
        poll_interval=0.01,
    )
    assert ready2 is True
    assert rc2 is None
