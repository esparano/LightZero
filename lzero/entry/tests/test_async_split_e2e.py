# -*- coding: utf-8 -*-
"""CPU-only end-to-end smoke tests for the decoupled Galcon SampledEfficientZero
actor-learner-evaluator split (no Ray, no GPU).
"""

import os
from typing import Tuple

import numpy as np
import pytest
import torch
from easydict import EasyDict

from lzero.entry.async_split import AsyncEvaluator, AsyncLearner, SelfPlayActor
from lzero.entry.async_split.cli import build_parser
from zoo.board_games.galcon.envs.galcon_env import GalconEnv


def _make_tiny_galcon_async_cfg() -> Tuple[EasyDict, EasyDict]:
    """Tiny 4x4 Galcon + SEZ config for fast CPU-only end-to-end tests."""
    grid_max = 50.0
    grid_square_size = 25.0
    grid_w = int(round((2 * grid_max) / grid_square_size))
    grid_h = int(round((2 * grid_max) / grid_square_size))
    action_space_size = grid_w * grid_h * grid_w * grid_h + 1

    # Probe exact channel count from GalconEnv
    probe_env_cfg = GalconEnv.default_config()
    probe_env_cfg.update(
        dict(
            battle_mode='self_play_mode',
            num_planets=4,
            grid_square_size=grid_square_size,
            grid_max_x=grid_max,
            grid_max_y=grid_max,
            fleet_top_k=1,
            max_episode_steps=8,
            map_seed=2,
            channel_last=False,
            scale=True,
        )
    )
    probe_obs = GalconEnv(probe_env_cfg).reset()
    c, h, w = probe_obs['observation'].shape

    main_cfg = EasyDict(
        dict(
            exp_name='galcon_async_test',
            env=dict(
                battle_mode='self_play_mode',
                bot_action_type='random',
                num_planets=4,
                min_send_ships=1,
                send_ratio=0.5,
                tick_seconds=0.25,
                fleet_speed=40.0,
                max_episode_steps=8,
                map_seed=2,
                grid_square_size=grid_square_size,
                grid_max_x=grid_max,
                grid_max_y=grid_max,
                home_distance_fraction=0.8,
                neutral_min_cost=0,
                neutral_max_cost=10,
                neutral_min_production=15,
                neutral_max_production=100,
                fleet_top_k=1,
                channel_last=False,
                scale=True,
                collector_env_num=2,
                evaluator_env_num=2,
                n_evaluator_episode=2,
                stop_value=2,
                manager=dict(shared_memory=False),
                agent_vs_human=False,
                prob_random_agent=0,
                prob_expert_agent=0,
                prob_random_action_in_bot=0.0,
                prob_pass_action_in_bot=0.0,
                render_mode=None,
            ),
            policy=dict(
                model=dict(
                    model_type='conv',
                    observation_shape=(c, h, w),
                    image_channel=c,
                    frame_stack_num=1,
                    action_space_size=action_space_size,
                    continuous_action_space=False,
                    num_of_sampled_actions=4,
                    downsample=False,
                    self_supervised_learning_loss=True,
                    num_res_blocks=1,
                    num_channels=8,
                    lstm_hidden_size=16,
                    reward_head_channels=4,
                    value_head_channels=4,
                    policy_head_channels=4,
                    reward_head_hidden_channels=[16],
                    value_head_hidden_channels=[16],
                    policy_head_hidden_channels=[16],
                    categorical_distribution=True,
                    reward_support_range=(-1.0, 2.0, 1.0),
                    value_support_range=(-1.0, 2.0, 1.0),
                    discrete_action_encoding_type='one_hot',
                    norm_type='BN',
                    embed_actions=True,
                    embedded_action_dim=8,
                ),
                cuda=False,
                env_type='board_games',
                action_type='varied_action_space',
                battle_mode='self_play_mode',
                mcts_ctree=True,
                game_segment_length=8,
                update_per_collect=2,
                batch_size=4,
                optim_type='AdamW',
                piecewise_decay_lr_scheduler=False,
                learning_rate=1e-3,
                grad_clip_value=1.0,
                num_simulations=4,
                manual_temperature_decay=False,
                fixed_temperature_value=1.0,
                reanalyze_ratio=0.0,
                td_steps=4,
                allow_episode_horizon_mismatch=True,
                discount_factor=0.99,
                num_unroll_steps=2,
                lstm_horizon_len=2,
                policy_loss_type='cross_entropy',
                use_priority=False,
                n_episode=2,
                eval_freq=1000,
                replay_buffer_size=500,
                collector_env_num=2,
                evaluator_env_num=2,
                symmetric_augment_x=True,
                symmetric_augment_y=True,
                symmetric_augment_type='Galcon',
            ),
        )
    )
    create_cfg = EasyDict(
        dict(
            env=dict(
                type='galcon',
                import_names=['zoo.board_games.galcon.envs.galcon_env'],
            ),
            env_manager=dict(type='base'),
            policy=dict(
                type='sampled_efficientzero',
                import_names=['lzero.policy.sampled_efficientzero'],
            ),
        )
    )
    return main_cfg, create_cfg


@pytest.mark.unittest
def test_galcon_async_split_e2e_cpu_smoke(tmp_path):
    """End-to-end CPU-only smoke test:
    1. Learner starts and publishes initial weights (iter 0).
    2. Actor syncs weights, collects self-play Galcon episodes, and writes atomic segment batch.
    3. Learner ingests segments with 4x Galcon symmetry augmentation, trains, and publishes updated weights.
    4. Actor reloads updated weights (parameters match learner).
    5. Evaluator polls updated weights, evaluates in eval_mode, and saves ckpt_best.pth.tar.
    """
    run_dir = tmp_path / 'galcon_async_e2e'
    cfg_pair = _make_tiny_galcon_async_cfg()

    learner = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        publish_every_n_updates=2,
        save_ckpt_every_n_updates=2,
        max_replay_ratio=0.5,
        min_transitions=8,
        archive_segments=True,
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
    evaluator = AsyncEvaluator(
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
        # 1. Learner published iter-0 weights on startup
        assert learner.dirs.latest_weights_path.is_file()

        # 2. Actor syncs iter-0 weights and generates a self-play segment batch
        batch_info = actor.step_once(n_episode=2)
        assert batch_info['path'].is_file()
        assert batch_info['num_segments'] >= 2
        assert batch_info['episodes'] >= 2
        assert batch_info['envsteps'] > 0
        assert actor.current_weight_iter == 0

        # 3. Learner ingests segments -> 4x symmetry augmentation applied!
        ingest_stats = learner.ingest_pending_segments()
        assert ingest_stats.files_ingested == 1
        assert ingest_stats.raw_segments == batch_info['num_segments']
        assert learner.replay_buffer.get_num_of_game_segments() == 4 * ingest_stats.raw_segments
        assert not batch_info['path'].exists()
        assert (learner.dirs.processed_dir / batch_info['path'].name).is_file()

        # 4. Learner trains 2 updates -> publishes iter-2 weights & saves iteration_2.pth.tar
        losses = []
        for _ in range(2):
            lv = learner.train_step()
            assert lv is not None
            assert np.isfinite(float(lv['total_loss']))
            losses.append(lv)
        assert learner.learner.train_iter == 2
        assert (learner.dirs.ckpt_dir / 'iteration_2.pth.tar').is_file()

        # Ensure mtime_ns differs from initial iter-0 write even on coarse-mtime filesystems
        st = learner.dirs.latest_weights_path.stat()
        os.utime(learner.dirs.latest_weights_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

        # 5. Actor reloads iter-2 weights and matches learner parameters
        reloaded_meta = actor.sync_weights()
        assert reloaded_meta is not None
        assert reloaded_meta['train_iter'] == 2
        assert actor.current_weight_iter == 2
        for (k1, p1), (k2, p2) in zip(
            learner.policy._learn_model.state_dict().items(),
            actor.policy._model.state_dict().items(),
        ):
            assert k1 == k2
            assert torch.allclose(p1.cpu(), p2.cpu())

        # 6. Evaluator detects iter-2 weights, runs 2 eval episodes, and saves ckpt_best.pth.tar
        eval_result = evaluator.eval_once_if_updated()
        assert eval_result is not None
        assert eval_result['train_iter'] == 2
        assert eval_result['envstep'] == learner.total_ingested_envsteps
        best_ckpt = evaluator.dirs.ckpt_dir / 'ckpt_best.pth.tar'
        assert best_ckpt.is_file()
        saved_payload = torch.load(str(best_ckpt), map_location='cpu', weights_only=False)
        assert 'model' in saved_payload
        assert saved_payload['train_iter'] == 2
    finally:
        actor.close()
        evaluator.close()
        learner.close()


@pytest.mark.unittest
def test_learner_replay_ratio_throttling_blocks_stale_overtraining(tmp_path):
    """Verify AsyncLearner pauses updates once max_replay_ratio is reached and resumes on new data."""
    run_dir = tmp_path / 'galcon_async_throttle'
    cfg_pair = _make_tiny_galcon_async_cfg()

    learner = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        publish_every_n_updates=10,
        save_ckpt_every_n_updates=0,
        max_replay_ratio=0.1,  # Strict cap: 1 update per 10 raw transitions
        min_transitions=8,
        use_bg_reader=False,
    )
    actor = SelfPlayActor(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        actor_id='actor_throttle',
        seed=11,
        device='cpu',
        env_manager_type='base',
    )

    try:
        # Before any data: throttled by warmup
        assert learner.train_step() is None

        # Ingest 1 batch
        actor.step_once(n_episode=2)
        stats1 = learner.ingest_pending_segments()
        expected_budget = int(stats1.raw_transitions * 0.1 + 1e-9)
        assert expected_budget >= 1

        updates_done = 0
        while learner.train_step() is not None:
            updates_done += 1
            assert updates_done <= expected_budget + 1

        assert updates_done == expected_budget
        # Next update must be throttled
        assert learner.train_step() is None

        # Adding fresh actor data unblocks training
        actor.step_once(n_episode=2)
        stats2 = learner.ingest_pending_segments()
        assert stats2.raw_transitions > 0
        assert learner.train_step() is not None
    finally:
        actor.close()
        learner.close()


@pytest.mark.unittest
def test_learner_background_reader_ingests_and_trains(tmp_path):
    """Verify AsyncLearner.run() with background reader thread ingests actor batches and reaches max_train_iter."""
    run_dir = tmp_path / 'galcon_async_bg_reader'
    cfg_pair = _make_tiny_galcon_async_cfg()

    actor = SelfPlayActor(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        actor_id='actor_bg',
        seed=12,
        device='cpu',
        env_manager_type='base',
    )
    try:
        actor.step_once(n_episode=2)
    finally:
        actor.close()

    learner = AsyncLearner(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=0,
        device='cpu',
        publish_every_n_updates=2,
        save_ckpt_every_n_updates=0,
        max_replay_ratio=0.5,
        min_transitions=8,
        poll_interval=0.05,
        use_bg_reader=True,
    )
    summary = learner.run(
        max_train_iter=2,
        max_idle_seconds=5.0,
        install_signal_handlers=False,
    )
    assert summary['train_iter'] == 2
    assert summary['files_ingested'] == 1
    assert summary['buffer_segments'] >= 8  # 4x symmetry on >= 2 raw segments
    assert learner.dirs.latest_weights_path.is_file()


@pytest.mark.unittest
def test_cli_parser_subcommands():
    parser = build_parser()
    a_args = parser.parse_args(['actor', '--run-dir', '/tmp/r', '--actor-id', 'a1', '--device', 'cpu'])
    assert a_args.command == 'actor'
    assert a_args.actor_id == 'a1'
    assert a_args.device == 'cpu'

    l_args = parser.parse_args(['learner', '--run-dir', '/tmp/r', '--publish-every', '50', '--max-replay-ratio', '0.25'])
    assert l_args.command == 'learner'
    assert l_args.publish_every == 50
    assert l_args.max_replay_ratio == 0.25

    e_args = parser.parse_args(['evaluator', '--run-dir', '/tmp/r', '--n-episode', '20', '--bot-action-type', 'random'])
    assert e_args.command == 'evaluator'
    assert e_args.n_episode == 20
    assert e_args.bot_action_type == 'random'

    launch_args = parser.parse_args(['launch', '--run-dir', '/tmp/r', '--num-actors', '3'])
    assert launch_args.command == 'launch'
    assert launch_args.num_actors == 3


@pytest.mark.unittest
def test_build_worker_env_hides_cuda_only_for_cpu_device():
    from lzero.entry.async_split.cli import _build_worker_env

    cpu_env = _build_worker_env(
        num_actors=4,
        include_evaluator=True,
        worker_threads=1,
        base_env={'CUDA_VISIBLE_DEVICES': '0'},
        device='cpu',
    )
    assert cpu_env['CUDA_VISIBLE_DEVICES'] == ''
    assert cpu_env['OMP_NUM_THREADS'] == '1'

    cpu_env_unset = _build_worker_env(
        num_actors=4,
        include_evaluator=True,
        worker_threads=1,
        base_env={},
        device='cpu',
    )
    assert cpu_env_unset['CUDA_VISIBLE_DEVICES'] == ''

    for dev in ('cuda', 'auto', None):
        non_cpu_env = _build_worker_env(
            num_actors=4,
            include_evaluator=True,
            worker_threads=1,
            base_env={'CUDA_VISIBLE_DEVICES': '0'},
            device=dev,
        )
        assert non_cpu_env['CUDA_VISIBLE_DEVICES'] == '0'

        non_cpu_unset = _build_worker_env(
            num_actors=4,
            include_evaluator=True,
            worker_threads=1,
            base_env={},
            device=dev,
        )
        assert 'CUDA_VISIBLE_DEVICES' not in non_cpu_unset


@pytest.mark.unittest
def test_spawned_cpu_worker_env_sees_zero_cuda_devices(tmp_path):
    """Verify pre-spawn CUDA_VISIBLE_DEVICES='' from _build_worker_env(device='cpu')
    completely hides CUDA (is_available() == False, device_count() == 0, is_initialized() == False)
    even across SelfPlayActor, AsyncEvaluator, and AsyncLearner(device='cpu').
    """
    import subprocess
    import sys
    from lzero.entry.async_split.cli import _build_worker_env

    worker_env = _build_worker_env(
        num_actors=2,
        include_evaluator=True,
        worker_threads=1,
        device='cpu',
    )
    script = f"""
import torch
assert not torch.cuda.is_available(), 'CUDA should be hidden for CPU worker!'
assert torch.cuda.device_count() == 0
from lzero.entry.async_split import AsyncEvaluator, AsyncLearner, SelfPlayActor
from lzero.entry.tests.test_async_split_e2e import _make_tiny_galcon_async_cfg

run_dir = {str(tmp_path / 'pre_spawn_hide_cuda')!r}
cfg_pair = _make_tiny_galcon_async_cfg()
learner = AsyncLearner(
    input_cfg=cfg_pair,
    run_dir=run_dir,
    seed=0,
    device='cpu',
    publish_every_n_updates=1,
    save_ckpt_every_n_updates=0,
    min_transitions=4,
    use_bg_reader=False,
)
actor = SelfPlayActor(
    input_cfg=cfg_pair,
    run_dir=run_dir,
    actor_id='cpu_actor_0',
    seed=10,
    device='cpu',
    env_manager_type='base',
)
evaluator = AsyncEvaluator(
    input_cfg=cfg_pair,
    run_dir=run_dir,
    seed=20,
    device='cpu',
    evaluator_env_num=2,
    n_evaluator_episode=2,
    env_manager_type='base',
    min_iter_gap=0,
    eval_mode='both',
    elo_games_per_opponent=2,
    elo_opponents_per_eval=2,
    elo_anchor_bots='random',
)
try:
    actor.step_once(n_episode=2)
    learner.ingest_pending_segments()
    learner.train_step()
    evaluator.eval_once_if_updated()
    assert not torch.cuda.is_available()
    assert not torch.cuda.is_initialized()
finally:
    actor.close()
    evaluator.close()
    learner.close()
"""
    proc = subprocess.run(
        [sys.executable, '-c', script],
        env=worker_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f'Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}'


@pytest.mark.unittest
def test_run_launch_isolates_cuda_per_worker_device(tmp_path, monkeypatch):
    import lzero.entry.async_split.cli as cli_mod

    cfg_pair = _make_tiny_galcon_async_cfg()
    monkeypatch.setattr(cli_mod, 'load_config_from_py', lambda _path: cfg_pair)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')

    spawned = []

    class _FakeProc:
        def __init__(self, cmd, env):
            self.cmd = cmd
            self.env = env
            self._rc = 0 if cmd[3] == 'learner' else None

        def poll(self):
            return self._rc

        def terminate(self):
            self._rc = 0

        def wait(self, timeout=None):
            self._rc = 0
            return 0

        def kill(self):
            self._rc = -9

    def _fake_popen(cmd, stdout=None, stderr=None, env=None):
        proc = _FakeProc(cmd, dict(env or {}))
        spawned.append(proc)
        return proc

    monkeypatch.setattr(cli_mod.subprocess, 'Popen', _fake_popen)

    parser = build_parser()
    args = parser.parse_args(
        [
            'launch',
            '--config',
            'dummy.py',
            '--run-dir',
            str(tmp_path / 'run_cpu_actors'),
            '--num-actors',
            '2',
            '--learner-device',
            'cuda',
            '--actor-device',
            'cpu',
            '--eval-device',
            'cpu',
            '--worker-threads',
            '1',
        ]
    )
    rc = cli_mod._run_launch(args)
    assert rc == 0
    assert len(spawned) == 4  # 1 learner + 2 actors + 1 evaluator

    by_role = {}
    for p in spawned:
        role = p.cmd[3]
        by_role.setdefault(role, []).append(p.env.get('CUDA_VISIBLE_DEVICES'))

    assert by_role['learner'] == ['0']
    assert by_role['actor'] == ['', '']
    assert by_role['evaluator'] == ['']

    # Verify GPU actors preserve CUDA_VISIBLE_DEVICES while CPU evaluator still hides it
    spawned.clear()
    args_gpu_actors = parser.parse_args(
        [
            'launch',
            '--config',
            'dummy.py',
            '--run-dir',
            str(tmp_path / 'run_gpu_actors'),
            '--num-actors',
            '1',
            '--learner-device',
            'cuda',
            '--actor-device',
            'cuda',
            '--eval-device',
            'cpu',
            '--worker-threads',
            '1',
        ]
    )
    rc = cli_mod._run_launch(args_gpu_actors)
    assert rc == 0
    by_role2 = {}
    for p in spawned:
        role = p.cmd[3]
        by_role2.setdefault(role, []).append(p.env.get('CUDA_VISIBLE_DEVICES'))
    assert by_role2['learner'] == ['0']
    assert by_role2['actor'] == ['0']
    assert by_role2['evaluator'] == ['']


@pytest.mark.unittest
def test_cpu_actor_and_evaluator_do_not_initialize_cuda_in_process(tmp_path):
    """Regression test: CPU SelfPlayActor / AsyncEvaluator must not trigger torch.cuda._lazy_init()
    (e.g. via EasyTimer defaulting to cuda=True) even when CUDA is visible in the process.
    """
    import subprocess
    import sys

    script = f"""
import torch
from lzero.entry.async_split import AsyncEvaluator, AsyncLearner, SelfPlayActor
from lzero.entry.tests.test_async_split_e2e import _make_tiny_galcon_async_cfg

run_dir = {str(tmp_path / 'no_cuda_init')!r}
cfg_pair = _make_tiny_galcon_async_cfg()
learner = AsyncLearner(
    input_cfg=cfg_pair,
    run_dir=run_dir,
    seed=0,
    device='cpu',
    publish_every_n_updates=1,
    save_ckpt_every_n_updates=0,
    min_transitions=4,
    use_bg_reader=False,
)
actor = SelfPlayActor(
    input_cfg=cfg_pair,
    run_dir=run_dir,
    actor_id='cpu_actor_0',
    seed=10,
    device='cpu',
    env_manager_type='base',
)
evaluator = AsyncEvaluator(
    input_cfg=cfg_pair,
    run_dir=run_dir,
    seed=20,
    device='cpu',
    evaluator_env_num=2,
    n_evaluator_episode=2,
    env_manager_type='base',
    min_iter_gap=0,
    eval_mode='both',
    elo_games_per_opponent=2,
    elo_opponents_per_eval=2,
    elo_anchor_bots='random',
)
try:
    actor.step_once(n_episode=2)
    evaluator.eval_once_if_updated()
    assert not torch.cuda.is_initialized(), 'CPU actor/evaluator unexpectedly initialized CUDA!'
finally:
    actor.close()
    evaluator.close()
    learner.close()
"""
    proc = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f'Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}'


@pytest.mark.unittest
def test_galcon_async_split_on_the_fly_symmetry_e2e(tmp_path):
    """AsyncLearner with symmetric_augment_on_the_fly='auto' wires GalconEnv.reflect_batch,
    stores 1x raw segments (no 4x deepcopy on ingest), and trains cleanly with finite loss.
    """
    run_dir = tmp_path / 'galcon_async_otf_sym'
    main_cfg, create_cfg = _make_tiny_galcon_async_cfg()
    main_cfg.policy.symmetric_augment_on_the_fly = 'auto'

    learner = AsyncLearner(
        input_cfg=(main_cfg, create_cfg),
        run_dir=run_dir,
        seed=0,
        device='cpu',
        publish_every_n_updates=2,
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
    try:
        assert learner.replay_buffer.reflect_batch_fn is not None
        batch_info = actor.step_once(n_episode=2)
        ingest_stats = learner.ingest_pending_segments()
        assert ingest_stats.files_ingested == 1
        assert learner.replay_buffer.get_num_of_game_segments() == ingest_stats.raw_segments
        assert ingest_stats.augmented_transitions == ingest_stats.raw_transitions

        for _ in range(2):
            lv = learner.train_step()
            assert lv is not None
            assert np.isfinite(float(lv['total_loss']))
        assert learner.learner.train_iter == 2
    finally:
        actor.close()
        learner.close()
