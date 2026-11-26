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
from lzero.entry.async_split.cli import build_parser
from lzero.entry.async_split.io_utils import (
    AsyncSplitDirs,
    atomic_json_save,
    atomic_pickle_save,
    detect_existing_run_artifacts,
    find_latest_checkpoint,
    parse_checkpoint_iter,
    write_segment_batch,
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

    # Create checkpoints where lexicographical order != numeric order
    for name in (
        'iteration_9000.pth.tar',
        'iteration_99000.pth.tar',
        'iteration_100000.pth.tar',
        'WM_ckpt_best.pth.tar',
        '.iteration_999999.pth.tar.tmp.123',
    ):
        (dirs.ckpt_dir / name).write_bytes(b'x')

    latest = find_latest_checkpoint(dirs.ckpt_dir)
    assert latest is not None
    assert latest.name == 'iteration_100000.pth.tar'
    assert parse_checkpoint_iter(latest) == 100000
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

        # Evaluator evaluates iter 5, but must NOT overwrite WM_ckpt_best if historical max is higher
        evaluator2.evaluator._max_episode_return = 999.0
        best_ckpt = evaluator2.dirs.ckpt_dir / 'WM_ckpt_best.pth.tar'
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
