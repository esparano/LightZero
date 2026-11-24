# -*- coding: utf-8 -*-
"""Unit tests for async_split atomic I/O utilities and replay-ratio throttling."""

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from lzero.entry.async_split.io_utils import (
    AsyncSplitDirs,
    WeightWatcher,
    consume_segment_file,
    list_incoming_segment_files,
    load_config_from_py,
    publish_weights,
    read_segment_batch,
    write_segment_batch,
)
from lzero.entry.async_split.replay_throttle import ReplayRatioThrottler


@pytest.mark.unittest
def test_atomic_weight_publish_and_watcher_reload(tmp_path):
    dirs = AsyncSplitDirs.from_root(tmp_path / 'run')
    dirs.ensure_exists()

    src_model = nn.Linear(4, 2)
    dst_model = nn.Linear(4, 2)
    with torch.no_grad():
        src_model.weight.fill_(1.5)
        src_model.bias.fill_(-0.25)
        dst_model.weight.zero_()
        dst_model.bias.zero_()

    watcher = WeightWatcher(dirs.latest_weights_path)
    assert watcher.has_update() is False
    assert watcher.poll_and_load(dst_model, device='cpu') is None

    # Publish version 10
    meta = publish_weights(dirs.latest_weights_path, src_model.state_dict(), train_iter=10, envstep=200)
    assert meta['train_iter'] == 10
    assert meta['envstep'] == 200
    assert not list(dirs.weights_dir.glob('*.tmp.*'))

    loaded = watcher.poll_and_load(dst_model, device='cpu')
    assert loaded is not None
    assert loaded['train_iter'] == 10
    assert loaded['envstep'] == 200
    assert torch.allclose(dst_model.weight, src_model.weight)
    assert torch.allclose(dst_model.bias, src_model.bias)

    # Second poll without new publish returns None
    assert watcher.poll_and_load(dst_model, device='cpu') is None

    # Publish version 20 with modified weights
    with torch.no_grad():
        src_model.weight.fill_(3.0)
    publish_weights(dirs.latest_weights_path, src_model.state_dict(), train_iter=20, envstep=400)
    # Bump mtime_ns explicitly in case filesystem has coarse timestamp resolution
    st = dirs.latest_weights_path.stat()
    os.utime(dirs.latest_weights_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    loaded2 = watcher.poll_and_load(dst_model, device='cpu')
    assert loaded2 is not None
    assert loaded2['train_iter'] == 20
    assert torch.allclose(dst_model.weight, torch.full_like(dst_model.weight, 3.0))


@pytest.mark.unittest
def test_atomic_segment_batch_write_list_read_consume(tmp_path):
    dirs = AsyncSplitDirs.from_root(tmp_path / 'run')
    dirs.ensure_exists(create_processed=True)

    # Stray hidden temp file must be ignored by listing
    (dirs.incoming_dir / '.partial.seg.pkl.tmp.123').write_bytes(b'incomplete')

    p1 = write_segment_batch(
        dirs.incoming_dir,
        segments=['seg_a'],
        metas=[{'done': True}],
        actor_id='actor-0',
        seq=1,
        envsteps=50,
        episodes=1,
        weight_version=0,
    )
    p2 = write_segment_batch(
        dirs.incoming_dir,
        segments=['seg_b'],
        metas=[{'done': True}],
        actor_id='actor-0',
        seq=2,
        envsteps=60,
        episodes=1,
        weight_version=5,
    )

    listed = list_incoming_segment_files(dirs.incoming_dir)
    assert listed == [p1, p2]

    batch1 = read_segment_batch(p1)
    assert batch1['segments'] == ['seg_a']
    assert batch1['envsteps'] == 50
    assert batch1['weight_version'] == 0

    # Consume with archive
    consume_segment_file(p1, archive_dir=dirs.processed_dir)
    assert not p1.exists()
    assert (dirs.processed_dir / p1.name).exists()

    # Consume with delete
    consume_segment_file(p2, archive_dir=None)
    assert not p2.exists()
    assert list_incoming_segment_files(dirs.incoming_dir) == []


@pytest.mark.unittest
def test_replay_ratio_throttler_warmup_throttle_and_unblock():
    throttler = ReplayRatioThrottler(min_transitions=100, max_replay_ratio=0.5)

    # Warmup phase: insufficient buffer transitions
    allowed, reason = throttler.can_update(sampleable_buffer_transitions=50)
    assert (allowed, reason) == (False, 'warmup')

    # Ingest 20 raw transitions (say 80 augmented, buffer has 120 sampleable)
    throttler.record_ingested(raw_transitions=20, augmented_transitions=80)
    assert throttler.allowed_updates_remaining == 10  # floor(20 * 0.5)

    for _ in range(10):
        allowed, reason = throttler.can_update(sampleable_buffer_transitions=120)
        assert (allowed, reason) == (True, 'ok')
        throttler.record_update()

    # Now at 10 / 20 = 0.5 -> must throttle
    assert pytest.approx(throttler.current_replay_ratio) == 0.5
    assert throttler.allowed_updates_remaining == 0
    allowed, reason = throttler.can_update(sampleable_buffer_transitions=120)
    assert (allowed, reason) == (False, 'throttled')

    # Ingesting more raw data unlocks additional updates
    throttler.record_ingested(raw_transitions=10, augmented_transitions=40)
    assert throttler.allowed_updates_remaining == 5
    allowed, reason = throttler.can_update(sampleable_buffer_transitions=160)
    assert (allowed, reason) == (True, 'ok')


@pytest.mark.unittest
def test_replay_ratio_throttler_unthrottled_mode():
    throttler = ReplayRatioThrottler(min_transitions=10, max_replay_ratio=None)
    assert throttler.can_update(5) == (False, 'warmup')
    assert throttler.allowed_updates_remaining is None
    assert throttler.can_update(10) == (True, 'ok')
    throttler.record_update(1000)
    assert throttler.can_update(10) == (True, 'ok')


@pytest.mark.unittest
def test_load_config_from_py_galcon_simple():
    cfg_path = Path('zoo/board_games/galcon/config/galcon_sez_sp_mode_config_SIMPLE.py')
    main_cfg, create_cfg = load_config_from_py(cfg_path)
    assert create_cfg.policy.type == 'sampled_efficientzero'
    assert create_cfg.env.type == 'galcon'
    assert main_cfg.policy.symmetric_augment_type == 'Galcon'
    # Verify returned configs are deep copies (mutations don't leak across loads)
    main_cfg.policy.batch_size = 7
    main_cfg2, _ = load_config_from_py(cfg_path)
    assert main_cfg2.policy.batch_size != 7


@pytest.mark.unittest
def test_compute_actor_learner_balance():
    from lzero.entry.async_split.replay_throttle import compute_actor_learner_balance

    # Learner starved (frequently throttled, empty queue, ratio at cap) -> negative
    bal, status = compute_actor_learner_balance(
        throttle_rate=0.85, pending_files=0, current_replay_ratio=0.5, max_replay_ratio=0.5
    )
    assert bal <= -0.5
    assert status == 'need_more_actors'

    # Sweet spot (rarely throttled, empty queue, ratio just under cap) -> near 0
    bal, status = compute_actor_learner_balance(
        throttle_rate=0.1, pending_files=0, current_replay_ratio=0.45, max_replay_ratio=0.5
    )
    assert -0.5 < bal < 0.5
    assert status == 'balanced'

    # Actors outpacing learner (backlog piling up + low replay ratio) -> positive
    bal, status = compute_actor_learner_balance(
        throttle_rate=0.0, pending_files=6, current_replay_ratio=0.1, max_replay_ratio=0.5
    )
    assert bal >= 0.5
    assert status == 'learner_bottleneck'
