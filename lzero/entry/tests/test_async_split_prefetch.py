# -*- coding: utf-8 -*-
"""Unit and integration tests for configurable background batch prefetching in ``AsyncLearner``."""

import threading
import time

import numpy as np
import pytest
import torch
from easydict import EasyDict

from lzero.entry.async_split.actor import SelfPlayActor
from lzero.entry.async_split.cli import build_parser
from lzero.entry.async_split.learner import (
    AsyncLearner,
    _BackgroundBatchPrefetcher,
    _PrefetchPolicyProxy,
    _RWLock,
)
from zoo.board_games.galcon.config.galcon_sez_sp_mode_config_SIMPLE_1map import (
    create_config as _base_create_cfg,
    main_config as _base_main_cfg,
)


def _make_tiny_galcon_cfg():
    import copy

    cfg = EasyDict(copy.deepcopy(_base_main_cfg))
    create_cfg = EasyDict(copy.deepcopy(_base_create_cfg))
    cfg.env.num_planets = 4
    cfg.env.grid_max_x = 50.0
    cfg.env.grid_max_y = 50.0
    cfg.env.grid_square_size = 25.0
    cfg.env.max_episode_steps = 8
    cfg.env.collector_env_num = 2
    cfg.env.evaluator_env_num = 2
    cfg.env.n_evaluator_episode = 2

    cfg.policy.cuda = False
    cfg.policy.device = 'cpu'
    cfg.policy.num_simulations = 4
    cfg.policy.n_episode = 2
    cfg.policy.batch_size = 4
    cfg.policy.game_segment_length = 8
    cfg.policy.td_steps = 8
    cfg.policy.num_unroll_steps = 2
    cfg.policy.lstm_horizon_len = 2
    cfg.policy.replay_buffer_size = 200
    cfg.policy.target_update_freq = 2
    cfg.policy.model.observation_shape = (40, 4, 4)
    cfg.policy.model.image_channel = 40
    cfg.policy.model.action_space_size = 4 * 4 * 4 * 4 + 1
    cfg.policy.model.num_of_sampled_actions = 4
    cfg.policy.model.num_res_blocks = 1
    cfg.policy.model.num_channels = 8
    cfg.policy.model.lstm_hidden_size = 16
    cfg.policy.model.reward_head_channels = 4
    cfg.policy.model.value_head_channels = 4
    cfg.policy.model.policy_head_channels = 4
    cfg.policy.model.reward_head_hidden_channels = [16]
    cfg.policy.model.value_head_hidden_channels = [16]
    cfg.policy.model.policy_head_hidden_channels = [16]
    cfg.policy.model.embedded_action_dim = 8
    create_cfg.env_manager.type = 'base'
    return cfg, create_cfg


@pytest.mark.unittest
class TestAsyncSplitPrefetchPrimitives:

    def test_rwlock_allows_concurrent_readers_and_excludes_writer(self):
        rw = _RWLock()
        active_readers = 0
        max_concurrent_readers = 0
        writer_saw_readers = False
        guard = threading.Lock()

        def _reader():
            nonlocal active_readers, max_concurrent_readers
            for _ in range(10):
                with rw.read_lock():
                    with guard:
                        active_readers += 1
                        max_concurrent_readers = max(max_concurrent_readers, active_readers)
                    time.sleep(0.005)
                    with guard:
                        active_readers -= 1

        def _writer():
            nonlocal writer_saw_readers
            for _ in range(5):
                with rw.write_lock():
                    with guard:
                        if active_readers != 0:
                            writer_saw_readers = True
                    time.sleep(0.005)

        threads = [threading.Thread(target=_reader) for _ in range(3)] + [threading.Thread(target=_writer)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive()

        assert not writer_saw_readers
        assert max_concurrent_readers >= 2

    def test_prefetch_policy_proxy_isolates_target_model_and_syncs(self):
        class _DummyPolicy:
            def __init__(self):
                self._target_model = torch.nn.Linear(4, 2)
                self._target_model.train()
                self.extra_attr = 'ok'

        pol = _DummyPolicy()
        proxy = _PrefetchPolicyProxy(pol)
        assert proxy.extra_attr == 'ok'
        assert proxy._target_model is not pol._target_model
        assert proxy._target_model.training is False

        # Mutating source model's mode and weights does not affect proxy until sync_from().
        with torch.no_grad():
            pol._target_model.weight.fill_(7.0)
        pol._target_model.train()
        assert not torch.allclose(proxy._target_model.weight, pol._target_model.weight)
        assert proxy._target_model.training is False

        proxy.sync_from(pol)
        assert torch.allclose(proxy._target_model.weight, pol._target_model.weight)
        assert proxy._target_model.training is False

    def test_cli_parses_prefetch_flags(self):
        parser = build_parser()
        args_learner = parser.parse_args(['learner', '--prefetch-workers', '2', '--prefetch-queue-size', '6'])
        assert args_learner.prefetch_workers == 2
        assert args_learner.prefetch_queue_size == 6

        args_launch = parser.parse_args(['launch', '--prefetch-workers', '1'])
        assert args_launch.prefetch_workers == 1
        assert args_launch.prefetch_queue_size is None

        args_default = parser.parse_args(['learner'])
        assert args_default.prefetch_workers == 0

    def test_launch_default_worker_thread_env(self):
        from lzero.entry.async_split.cli import (
            _build_worker_env,
            _compute_default_worker_threads,
            build_parser,
        )

        # 32 CPUs, 4 actors + 1 eval + 1 learner = 6 workers -> min(4, 32 // 6) = 4
        assert _compute_default_worker_threads(num_actors=4, include_evaluator=True, cpu_count=32) == 4
        # 16 CPUs, 6 workers -> 2
        assert _compute_default_worker_threads(num_actors=4, include_evaluator=True, cpu_count=16) == 2
        # 4 CPUs, 6 workers -> floor at 1
        assert _compute_default_worker_threads(num_actors=4, include_evaluator=True, cpu_count=4) == 1

        # Defaults populate unset OMP/MKL/OPENBLAS without clobbering caller-provided vars
        env_auto = _build_worker_env(
            num_actors=4,
            include_evaluator=True,
            worker_threads=None,
            base_env={'PATH': '/usr/bin', 'OMP_NUM_THREADS': '2'},
            cpu_count=32,
        )
        assert env_auto['OMP_NUM_THREADS'] == '2'
        assert env_auto['MKL_NUM_THREADS'] == '4'
        assert env_auto['OPENBLAS_NUM_THREADS'] == '4'

        # Explicit --worker-threads overrides all three
        env_override = _build_worker_env(
            num_actors=4,
            include_evaluator=True,
            worker_threads=3,
            base_env={'OMP_NUM_THREADS': '16'},
            cpu_count=32,
        )
        assert env_override['OMP_NUM_THREADS'] == '3'
        assert env_override['MKL_NUM_THREADS'] == '3'
        assert env_override['OPENBLAS_NUM_THREADS'] == '3'

        with pytest.raises(ValueError):
            _build_worker_env(num_actors=4, worker_threads=0, base_env={})

        parser = build_parser()
        assert parser.parse_args(['launch']).worker_threads is None
        assert parser.parse_args(['launch', '--worker-threads', '3']).worker_threads == 3


@pytest.mark.unittest
class TestAsyncLearnerPrefetchIntegration:

    @pytest.mark.parametrize('prefetch_workers', [0, 1, 2])
    def test_learner_runs_with_configurable_prefetch_workers(self, tmp_path, prefetch_workers):
        cfg, create_cfg = _make_tiny_galcon_cfg()
        run_dir = tmp_path / f'run_pw_{prefetch_workers}'

        learner = AsyncLearner(
            input_cfg=(cfg, create_cfg),
            run_dir=run_dir,
            seed=0,
            device='cpu',
            publish_every_n_updates=2,
            save_ckpt_every_n_updates=0,
            max_replay_ratio=None,
            min_transitions=8,
            poll_interval=0.05,
            use_bg_reader=False,
            prefetch_workers=prefetch_workers,
            prefetch_queue_size=3 if prefetch_workers > 0 else None,
        )
        actor = SelfPlayActor(
            input_cfg=(cfg, create_cfg),
            run_dir=run_dir,
            actor_id='actor_0',
            seed=10,
            device='cpu',
            collector_env_num=2,
            n_episode=2,
            num_simulations=4,
            env_manager_type='base',
            wait_for_initial_weights=False,
        )
        try:
            actor.step_once()
            actor.step_once()
            summary = learner.run(
                max_train_iter=5,
                max_idle_seconds=10.0,
                install_signal_handlers=False,
            )
            assert summary['train_iter'] == 5
            assert summary['ingested_episodes'] >= 4
            assert learner._batch_prefetcher is None  # cleaned up in close()
        finally:
            actor.close()
            learner.close()

    def test_prefetch_workers_zero_spawns_no_prefetcher(self, tmp_path):
        cfg, create_cfg = _make_tiny_galcon_cfg()
        learner = AsyncLearner(
            input_cfg=(cfg, create_cfg),
            run_dir=tmp_path / 'run_pw_0',
            seed=0,
            device='cpu',
            prefetch_workers=0,
        )
        try:
            learner.start_prefetcher()
            assert learner._batch_prefetcher is None
        finally:
            learner.close()

    def test_learner_prefetch_with_reanalyze_caps_workers_and_trains(self, tmp_path):
        cfg, create_cfg = _make_tiny_galcon_cfg()
        cfg.policy.reanalyze_ratio = 0.5
        cfg.policy.reanalyze_outdated = True
        cfg.policy.reanalyze_noise = False
        cfg.policy.policy_target_type = 'completed_q'
        run_dir = tmp_path / 'run_reanalyze_prefetch'

        learner = AsyncLearner(
            input_cfg=(cfg, create_cfg),
            run_dir=run_dir,
            seed=0,
            device='cpu',
            publish_every_n_updates=2,
            save_ckpt_every_n_updates=0,
            max_replay_ratio=None,
            min_transitions=8,
            poll_interval=0.05,
            use_bg_reader=False,
            prefetch_workers=2,
            prefetch_queue_size=2,
        )
        actor = SelfPlayActor(
            input_cfg=(cfg, create_cfg),
            run_dir=run_dir,
            actor_id='actor_0',
            seed=10,
            device='cpu',
            collector_env_num=2,
            n_episode=2,
            num_simulations=4,
            env_manager_type='base',
            wait_for_initial_weights=False,
        )
        try:
            learner.start_prefetcher()
            assert learner._batch_prefetcher is not None
            assert learner._batch_prefetcher.num_workers == 1
            actor.step_once()
            actor.step_once()
            summary = learner.run(
                max_train_iter=4,
                max_idle_seconds=15.0,
                install_signal_handlers=False,
            )
            assert summary['train_iter'] == 4
            assert summary['ingested_episodes'] >= 4
        finally:
            actor.close()
            learner.close()

    def test_learner_prefetch_with_reanalyze_surfaces_worker_error(self, tmp_path):
        """A crash inside the background prefetcher must raise instead of silently falling back to main-thread sample."""
        cfg, create_cfg = _make_tiny_galcon_cfg()
        cfg.policy.reanalyze_ratio = 0.5
        cfg.policy.reanalyze_outdated = True
        run_dir = tmp_path / 'run_reanalyze_err'

        learner = AsyncLearner(
            input_cfg=(cfg, create_cfg),
            run_dir=run_dir,
            seed=0,
            device='cpu',
            save_ckpt_every_n_updates=0,
            max_replay_ratio=None,
            min_transitions=4,
            poll_interval=0.02,
            use_bg_reader=False,
            prefetch_workers=1,
            prefetch_queue_size=2,
        )
        actor = SelfPlayActor(
            input_cfg=(cfg, create_cfg),
            run_dir=run_dir,
            actor_id='actor_0',
            seed=10,
            device='cpu',
            collector_env_num=2,
            n_episode=2,
            num_simulations=4,
            env_manager_type='base',
            wait_for_initial_weights=False,
        )
        try:
            actor.step_once()
            learner.ingest_pending_segments()

            def _boom(*_a, **_kw):
                raise ValueError('simulated GPU OOM in prefetch worker')

            learner.replay_buffer.sample = _boom
            learner.start_prefetcher()
            # Wait until worker thread hits _boom and exits.
            deadline = time.time() + 5.0
            while learner._batch_prefetcher.is_alive() and time.time() < deadline:
                time.sleep(0.02)
            assert not learner._batch_prefetcher.is_alive()

            with pytest.raises(RuntimeError, match='simulated GPU OOM in prefetch worker'):
                learner._sample_train_batch(int(cfg.policy.batch_size))
        finally:
            actor.close()
            learner.close()

    def test_learner_prefetch_with_reanalyze_stop_does_not_launch_duplicate_sample(self, tmp_path):
        """request_stop() while waiting on a slow reanalyze worker must return None without concurrent main-thread sample."""
        cfg, create_cfg = _make_tiny_galcon_cfg()
        cfg.policy.reanalyze_ratio = 0.5
        cfg.policy.reanalyze_outdated = True
        run_dir = tmp_path / 'run_reanalyze_stop'

        learner = AsyncLearner(
            input_cfg=(cfg, create_cfg),
            run_dir=run_dir,
            seed=0,
            device='cpu',
            save_ckpt_every_n_updates=0,
            max_replay_ratio=None,
            min_transitions=4,
            poll_interval=0.02,
            use_bg_reader=False,
            prefetch_workers=1,
            prefetch_queue_size=2,
        )
        actor = SelfPlayActor(
            input_cfg=(cfg, create_cfg),
            run_dir=run_dir,
            actor_id='actor_0',
            seed=10,
            device='cpu',
            collector_env_num=2,
            n_episode=2,
            num_simulations=4,
            env_manager_type='base',
            wait_for_initial_weights=False,
        )
        try:
            actor.step_once()
            learner.ingest_pending_segments()

            entered = threading.Event()
            release = threading.Event()
            active = 0
            max_concurrent = 0
            guard = threading.Lock()
            orig_sample = learner.replay_buffer.sample

            def _slow_sample(bs, pol):
                nonlocal active, max_concurrent
                with guard:
                    active += 1
                    max_concurrent = max(max_concurrent, active)
                entered.set()
                release.wait(timeout=5.0)
                try:
                    return orig_sample(bs, pol)
                finally:
                    with guard:
                        active -= 1

            learner.replay_buffer.sample = _slow_sample
            learner.start_prefetcher()
            assert entered.wait(timeout=5.0)

            def _trigger_stop():
                time.sleep(0.05)
                learner.request_stop()

            stopper = threading.Thread(target=_trigger_stop)
            stopper.start()
            try:
                batch = learner._sample_train_batch(int(cfg.policy.batch_size))
                stopper.join(timeout=2.0)
            finally:
                release.set()

            assert batch is None
            assert max_concurrent == 1
        finally:
            actor.close()
            learner.close()
