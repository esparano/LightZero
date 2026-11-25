"""Decoupled learner worker (no Ray).

Ingests ``GameSegment`` batch files written by ``SelfPlayActor`` processes into
LightZero's ``SampledEfficientZeroGameBuffer`` (preserving 4x ``reflect_game_segment``
symmetry augmentation and ``GalconEnv`` ``to_play``/sign-flip invariants), trains
with configurable replay-ratio throttling, logs standard TensorBoard metrics,
saves periodic checkpoints, and atomically publishes ``latest_weights.pt``.
"""

import contextlib
import copy
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import psutil
import torch
from ding.config import compile_config
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner
from ditk import logging
from easydict import EasyDict
from tensorboardX import SummaryWriter

from lzero.policy import visit_count_temperature

from .io_utils import (
    AsyncSplitDirs,
    consume_segment_file,
    list_incoming_segment_files,
    publish_weights,
    read_segment_batch,
)
from .replay_throttle import ReplayRatioThrottler, compute_actor_learner_balance


def _ding_exp_name(root_dir: Path) -> str:
    return os.path.relpath(str(root_dir), os.getcwd())


def _select_game_buffer_cls(policy_type: str):
    if policy_type in ['muzero', 'muzero_context', 'muzero_rnn_full_obs']:
        from lzero.mcts import MuZeroGameBuffer as GameBuffer
    elif policy_type == 'efficientzero':
        from lzero.mcts import EfficientZeroGameBuffer as GameBuffer
    elif policy_type == 'sampled_efficientzero':
        from lzero.mcts import SampledEfficientZeroGameBuffer as GameBuffer
    elif policy_type == 'sampled_muzero':
        from lzero.mcts import SampledMuZeroGameBuffer as GameBuffer
    elif policy_type == 'gumbel_muzero':
        from lzero.mcts import GumbelMuZeroGameBuffer as GameBuffer
    elif policy_type == 'stochastic_muzero':
        from lzero.mcts import StochasticMuZeroGameBuffer as GameBuffer
    else:
        raise ValueError(f'Unsupported policy type for AsyncLearner: {policy_type!r}')
    return GameBuffer


@dataclass
class IngestStats:
    files_ingested: int = 0
    raw_segments: int = 0
    raw_transitions: int = 0
    augmented_transitions: int = 0
    episodes: int = 0
    weight_versions: Optional[List[int]] = None


class _BackgroundSegmentReader:
    """Reads and deserializes incoming ``*.seg.pkl`` files off the learner thread.

    The replay buffer itself is never touched from this thread; deserialized
    payloads are passed through a bounded queue to the main learner thread.
    """

    def __init__(self, incoming_dir: Path, poll_interval: float = 0.2, max_queue_size: int = 32) -> None:
        self.incoming_dir = incoming_dir
        self.poll_interval = max(0.02, float(poll_interval))
        self.queue: 'queue.Queue[Tuple[Path, Dict[str, Any]]]' = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._queued_paths: Set[Path] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name='async-segment-reader', daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def mark_consumed(self, path: Path) -> None:
        with self._lock:
            self._queued_paths.discard(path)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            found_any = False
            for path in list_incoming_segment_files(self.incoming_dir):
                if self._stop_event.is_set():
                    break
                with self._lock:
                    if path in self._queued_paths:
                        continue
                try:
                    payload = read_segment_batch(path)
                except (FileNotFoundError, EOFError, OSError) as exc:
                    logging.warning(f'[AsyncLearner] Transient read error on {path.name}: {exc}')
                    continue
                except Exception as exc:
                    logging.error(f'[AsyncLearner] Failed to deserialize {path.name}: {exc}')
                    continue

                with self._lock:
                    self._queued_paths.add(path)
                while not self._stop_event.is_set():
                    try:
                        self.queue.put((path, payload), timeout=0.2)
                        found_any = True
                        break
                    except queue.Full:
                        continue
            if not found_any:
                self._stop_event.wait(self.poll_interval)


class _RWLock:
    """Writer-preferring reader-writer lock for concurrent replay sampling + exclusive ingestion."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False

    @contextlib.contextmanager
    def read_lock(self) -> Iterator[None]:
        with self._cond:
            while self._writer_active or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextlib.contextmanager
    def write_lock(self) -> Iterator[None]:
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers > 0:
                    self._cond.wait()
                self._writer_active = True
            finally:
                self._writers_waiting -= 1
        try:
            yield
        finally:
            with self._cond:
                self._writer_active = False
                self._cond.notify_all()


def _unwrap_model(model: Any) -> Any:
    """Unwrap DI-engine model wrappers (e.g. TargetNetworkWrapper) down to the raw nn.Module."""
    while hasattr(model, '_model') and model._model is not model:
        model = model._model
    return model


class _PrefetchPolicyProxy:
    """Thread-isolated view of ``policy`` for background ``replay_buffer.sample()`` workers.

    Owns a private eval-mode deepcopy of the underlying ``policy._target_model`` module
    so ``_forward_learn()`` toggling ``policy._target_model.train()`` or calling
    ``_target_model.update()`` never races with background sampling threads.
    """

    def __init__(self, policy: Any) -> None:
        self._policy = policy
        src_target = getattr(policy, '_target_model', None)
        if src_target is not None:
            self._target_model = copy.deepcopy(_unwrap_model(src_target))
            self._target_model.eval()
        else:
            self._target_model = None

    def sync_from(self, policy: Any) -> None:
        src_target = getattr(policy, '_target_model', None)
        if self._target_model is not None and src_target is not None:
            self._target_model.load_state_dict(_unwrap_model(src_target).state_dict())
            self._target_model.eval()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)


class _BackgroundBatchPrefetcher:
    """Samples training batches on background thread(s) to overlap CPU buffer work with GPU training."""

    def __init__(
        self,
        replay_buffer: Any,
        policy_proxy: _PrefetchPolicyProxy,
        batch_size: int,
        min_transitions: int,
        buffer_rwlock: _RWLock,
        target_rwlock: _RWLock,
        num_workers: int = 1,
        max_queue_size: Optional[int] = None,
        poll_interval: float = 0.05,
    ) -> None:
        self._replay_buffer = replay_buffer
        self._policy_proxy = policy_proxy
        self._batch_size = int(batch_size)
        self._min_transitions = int(min_transitions)
        self._buffer_rwlock = buffer_rwlock
        self._target_rwlock = target_rwlock
        self.num_workers = max(1, int(num_workers))
        q_size = int(max_queue_size) if max_queue_size is not None else max(2, self.num_workers * 2)
        self.max_queue_size = max(1, q_size)
        self._poll_interval = max(0.01, float(poll_interval))
        self.queue: 'queue.Queue[Any]' = queue.Queue(maxsize=self.max_queue_size)
        self._stop_event = threading.Event()
        self._error_lock = threading.Lock()
        self._last_error: Optional[BaseException] = None
        self._threads: List[threading.Thread] = [
            threading.Thread(
                target=self._worker_loop,
                name=f'async-batch-prefetch-{i}',
                daemon=True,
            )
            for i in range(self.num_workers)
        ]

    @property
    def last_error(self) -> Optional[BaseException]:
        with self._error_lock:
            return self._last_error

    def start(self) -> None:
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Drain queue so workers blocked on queue.put() wake up promptly.
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=2.0)

    def get_batch(self, timeout: float = 0.0) -> Optional[Any]:
        err = self.last_error
        if err is not None:
            raise RuntimeError(f'Background batch prefetch worker failed: {err}') from err
        try:
            if timeout > 0:
                return self.queue.get(timeout=timeout)
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                with self._buffer_rwlock.read_lock():
                    sampleable = self._replay_buffer.get_num_of_sampleable_transitions()
                    total_in_buf = self._replay_buffer.get_num_of_transitions()
                    if sampleable < self._min_transitions or total_in_buf <= self._batch_size:
                        batch = None
                    else:
                        with self._target_rwlock.read_lock():
                            batch = self._replay_buffer.sample(self._batch_size, self._policy_proxy)
            except Exception as exc:
                with self._error_lock:
                    if self._last_error is None:
                        self._last_error = exc
                logging.error(f'[AsyncLearner] Batch prefetch worker error: {exc}')
                self._stop_event.set()
                return

            if batch is None:
                self._stop_event.wait(self._poll_interval)
                continue

            while not self._stop_event.is_set():
                try:
                    self.queue.put(batch, timeout=0.1)
                    break
                except queue.Full:
                    continue


class AsyncLearner:
    """Standalone GPU/CPU learner worker for decoupled MuZero / SampledEfficientZero."""

    def __init__(
        self,
        input_cfg: Tuple[dict, dict],
        run_dir: os.PathLike | str,
        seed: int = 0,
        device: str = 'auto',
        model_path: Optional[str] = None,
        publish_every_n_updates: int = 25,
        save_ckpt_every_n_updates: int = 1000,
        max_replay_ratio: Optional[float] = 0.5,
        min_transitions: Optional[int] = None,
        poll_interval: float = 0.25,
        archive_segments: bool = False,
        use_bg_reader: bool = True,
        prefetch_workers: int = 0,
        prefetch_queue_size: Optional[int] = None,
    ) -> None:
        self.dirs = AsyncSplitDirs.from_root(run_dir)
        self.dirs.ensure_exists(create_processed=archive_segments)
        self.publish_every_n_updates = max(1, int(publish_every_n_updates))
        self.save_ckpt_every_n_updates = max(0, int(save_ckpt_every_n_updates))
        self.poll_interval = float(poll_interval)
        self.archive_segments = bool(archive_segments)
        self.use_bg_reader = bool(use_bg_reader)
        self.prefetch_workers = max(0, int(prefetch_workers))
        self.prefetch_queue_size = (
            max(1, int(prefetch_queue_size)) if prefetch_queue_size is not None else None
        )
        self._stop_requested = False
        self._closed = False

        cfg = EasyDict(copy.deepcopy(input_cfg[0]))
        create_cfg = EasyDict(copy.deepcopy(input_cfg[1]))

        if device == 'auto':
            use_cuda = bool(cfg.policy.get('cuda', False) and torch.cuda.is_available())
        else:
            use_cuda = (device == 'cuda' and torch.cuda.is_available())
        cfg.policy.cuda = use_cuda
        cfg.policy.device = 'cuda' if use_cuda else 'cpu'

        cfg.exp_name = _ding_exp_name(self.dirs.root_dir)
        self.cfg = compile_config(
            cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=False
        )
        self.cfg.exp_name = _ding_exp_name(self.dirs.root_dir)
        self.create_cfg = create_cfg
        set_pkg_seed(self.cfg.seed, use_cuda=self.cfg.policy.cuda)

        # SampledEfficientZeroPolicy._forward_learn logs _collect_mcts_temperature (must be float for DI-engine monitor).
        self.policy = create_policy(self.cfg.policy, model=None, enable_field=['learn', 'collect'])
        self.policy._collect_mcts_temperature = float(
            visit_count_temperature(
                self.cfg.policy.manual_temperature_decay,
                self.cfg.policy.fixed_temperature_value,
                self.cfg.policy.threshold_training_steps_for_final_temperature,
                trained_steps=0,
            )
        )
        if model_path is not None:
            ckpt = torch.load(model_path, map_location=self.cfg.policy.device, weights_only=False)
            if 'optimizer' in ckpt and 'target_model' in ckpt:
                self.policy.learn_mode.load_state_dict(ckpt)
            elif 'model' in ckpt:
                self.policy._learn_model.load_state_dict(ckpt['model'])
                if hasattr(self.policy, '_target_model') and self.policy._target_model is not None:
                    self.policy._target_model.load_state_dict(ckpt['model'])
            else:
                raise ValueError(f'Unrecognized checkpoint format at {model_path}')

        self.tb_logger = SummaryWriter(str(self.dirs.log_dir / 'learner'))
        self.learner = BaseLearner(
            self.cfg.policy.learn.learner,
            self.policy.learn_mode,
            self.tb_logger,
            exp_name=self.cfg.exp_name,
        )

        # Initialize replay buffer with exact env_fn.reflect_game_segment attachment
        buffer_cls = _select_game_buffer_cls(self.create_cfg.policy.type)
        self.replay_buffer = buffer_cls(self.cfg.policy)
        env_fn, _, _ = get_vec_env_setting(self.cfg.env)
        if hasattr(env_fn, 'validate_config'):
            env_fn.validate_config(self.cfg)
        if hasattr(env_fn, 'reflect_game_segment'):
            self.replay_buffer.reflect_fn = env_fn.reflect_game_segment

        effective_min = (
            int(min_transitions)
            if min_transitions is not None
            else int(self.cfg.policy.batch_size) + 1
        )
        self.throttler = ReplayRatioThrottler(
            min_transitions=max(effective_min, int(self.cfg.policy.batch_size) + 1),
            max_replay_ratio=max_replay_ratio,
        )

        self.total_ingested_envsteps = 0
        self.total_ingested_episodes = 0
        self.total_files_ingested = 0
        self._last_published_iter = -1
        self._last_saved_ckpt_iter = 0
        self._recent_weight_lags: List[int] = []
        self._recent_throttled: List[float] = []
        self._last_balance_warn_time = 0.0
        self._bg_reader: Optional[_BackgroundSegmentReader] = None
        self._buffer_rwlock = _RWLock()
        self._target_rwlock = _RWLock()
        self._prefetch_policy_proxy: Optional[_PrefetchPolicyProxy] = None
        self._batch_prefetcher: Optional[_BackgroundBatchPrefetcher] = None
        self._target_update_freq = max(
            1,
            int(
                self.cfg.policy.get(
                    'target_update_freq',
                    self.cfg.policy.get('target_model_update_freq', 100),
                )
            ),
        )

        self.learner.call_hook('before_run')
        # Publish initial weights immediately so actors and evaluator can sync at iter 0.
        self.publish_current_weights()

    def publish_current_weights(self) -> Dict[str, Any]:
        meta = publish_weights(
            self.dirs.latest_weights_path,
            self.policy._learn_model.state_dict(),
            train_iter=self.learner.train_iter,
            envstep=self.total_ingested_envsteps,
        )
        self._last_published_iter = int(self.learner.train_iter)
        return meta

    def _ingest_payload(self, path: Path, payload: Dict[str, Any], stats: IngestStats) -> None:
        segments = payload.get('segments', [])
        metas = payload.get('metas', [])
        if segments:
            with self._buffer_rwlock.write_lock():
                before_transitions = self.replay_buffer.get_num_of_transitions()
                self.replay_buffer.push_game_segments((segments, metas))
                after_push_transitions = self.replay_buffer.get_num_of_transitions()
                self.replay_buffer.remove_oldest_data_to_fit()

                raw_transitions = int(payload.get('envsteps', 0))
                if raw_transitions <= 0:
                    raw_transitions = sum(
                        self.replay_buffer._game_segment_transition_count(seg) for seg in segments
                    )
                augmented_transitions = max(raw_transitions, after_push_transitions - before_transitions)
            episodes = int(payload.get('episodes', sum(1 for m in metas if bool(m.get('done', False)))))

            self.total_ingested_envsteps += raw_transitions
            self.total_ingested_episodes += episodes
            self.throttler.record_ingested(raw_transitions, augmented_transitions)

            weight_version = int(payload.get('weight_version', 0))
            lag = max(0, int(self.learner.train_iter) - weight_version)
            self._recent_weight_lags.append(lag)
            if len(self._recent_weight_lags) > 100:
                self._recent_weight_lags = self._recent_weight_lags[-100:]

            stats.raw_segments += len(segments)
            stats.raw_transitions += raw_transitions
            stats.augmented_transitions += augmented_transitions
            stats.episodes += episodes
            if stats.weight_versions is not None:
                stats.weight_versions.append(weight_version)

        archive_dir = self.dirs.processed_dir if self.archive_segments else None
        consume_segment_file(path, archive_dir=archive_dir)
        if self._bg_reader is not None:
            self._bg_reader.mark_consumed(path)
        stats.files_ingested += 1
        self.total_files_ingested += 1

    def ingest_pending_segments(self, max_files: Optional[int] = None) -> IngestStats:
        """Ingest available segment batches into the replay buffer."""
        stats = IngestStats(weight_versions=[])
        if self._bg_reader is not None:
            while max_files is None or stats.files_ingested < max_files:
                try:
                    path, payload = self._bg_reader.queue.get_nowait()
                except queue.Empty:
                    break
                self._ingest_payload(path, payload, stats)
            return stats

        files = list_incoming_segment_files(self.dirs.incoming_dir)
        if max_files is not None:
            files = files[:max_files]
        for path in files:
            try:
                payload = read_segment_batch(path)
            except (FileNotFoundError, EOFError, OSError) as exc:
                logging.warning(f'[AsyncLearner] Transient read error on {path.name}: {exc}')
                continue
            self._ingest_payload(path, payload, stats)
        return stats

    def start_prefetcher(self) -> None:
        """Start background batch prefetch worker thread(s) when ``prefetch_workers > 0``."""
        if self.prefetch_workers <= 0 or self._batch_prefetcher is not None:
            return
        self._prefetch_policy_proxy = _PrefetchPolicyProxy(self.policy)
        self._batch_prefetcher = _BackgroundBatchPrefetcher(
            replay_buffer=self.replay_buffer,
            policy_proxy=self._prefetch_policy_proxy,
            batch_size=int(self.cfg.policy.batch_size),
            min_transitions=self.throttler.min_transitions,
            buffer_rwlock=self._buffer_rwlock,
            target_rwlock=self._target_rwlock,
            num_workers=self.prefetch_workers,
            max_queue_size=self.prefetch_queue_size,
            poll_interval=min(0.05, self.poll_interval),
        )
        self._batch_prefetcher.start()
        logging.info(
            f'[AsyncLearner] Started background batch prefetcher '
            f'(workers={self._batch_prefetcher.num_workers}, '
            f'queue_size={self._batch_prefetcher.max_queue_size})'
        )

    def _sync_prefetch_target_model(self) -> None:
        if self._prefetch_policy_proxy is None:
            return
        with self._target_rwlock.write_lock():
            self._prefetch_policy_proxy.sync_from(self.policy)

    def _sample_train_batch(self, batch_size: int) -> Any:
        if self._batch_prefetcher is not None:
            batch = self._batch_prefetcher.get_batch(timeout=0.05)
            if batch is not None:
                return batch
        with self._buffer_rwlock.read_lock():
            return self.replay_buffer.sample(batch_size, self.policy)

    def train_step(self) -> Optional[Dict[str, Any]]:
        """Run one gradient update if warmup and replay-ratio constraints allow."""
        batch_size = int(self.cfg.policy.batch_size)
        with self._buffer_rwlock.read_lock():
            sampleable = self.replay_buffer.get_num_of_sampleable_transitions()
            total_in_buf = self.replay_buffer.get_num_of_transitions()
        allowed, reason = self.throttler.can_update(sampleable)
        if not allowed or total_in_buf <= batch_size:
            return None

        self.policy._collect_mcts_temperature = float(
            visit_count_temperature(
                self.cfg.policy.manual_temperature_decay,
                self.cfg.policy.fixed_temperature_value,
                self.cfg.policy.threshold_training_steps_for_final_temperature,
                trained_steps=int(self.learner.train_iter),
            )
        )
        train_data = self._sample_train_batch(batch_size)
        log_vars = self.learner.train(train_data, self.total_ingested_envsteps)

        if self.cfg.policy.use_priority and log_vars and 'value_priority_orig' in log_vars[0]:
            with self._buffer_rwlock.write_lock():
                self.replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        self.throttler.record_update(1)
        cur_iter = int(self.learner.train_iter)

        if self._prefetch_policy_proxy is not None and cur_iter % self._target_update_freq == 0:
            self._sync_prefetch_target_model()

        if cur_iter - self._last_published_iter >= self.publish_every_n_updates:
            self.publish_current_weights()

        if (
            self.save_ckpt_every_n_updates > 0
            and cur_iter > 0
            and cur_iter - self._last_saved_ckpt_iter >= self.save_ckpt_every_n_updates
        ):
            self.learner.save_checkpoint(f'iteration_{cur_iter}.pth.tar')
            self._last_saved_ckpt_iter = cur_iter

        if cur_iter % max(1, self.publish_every_n_updates) == 0:
            self._log_async_metrics(throttled=0.0)

        return log_vars[0] if log_vars else {}

    def _count_pending_segment_files(self) -> int:
        disk_count = len(list_incoming_segment_files(self.dirs.incoming_dir))
        bg_count = self._bg_reader.queue.qsize() if self._bg_reader is not None else 0
        return max(disk_count, bg_count)

    def _log_async_metrics(self, throttled: float) -> None:
        self._recent_throttled.append(float(throttled))
        if len(self._recent_throttled) > 50:
            self._recent_throttled = self._recent_throttled[-50:]
        throttle_rate = sum(self._recent_throttled) / len(self._recent_throttled)
        pending_files = self._count_pending_segment_files()
        balance, status = compute_actor_learner_balance(
            throttle_rate=throttle_rate,
            pending_files=pending_files,
            current_replay_ratio=self.throttler.current_replay_ratio,
            max_replay_ratio=self.throttler.max_replay_ratio,
        )

        now = time.time()
        if status != 'balanced' and (now - self._last_balance_warn_time) >= 60.0:
            self._last_balance_warn_time = now
            if status == 'need_more_actors':
                logging.warning(
                    f'[AsyncLearner] BOTTLENECK=ACTORS (balance={balance:+.2f}, '
                    f'throttle_rate={throttle_rate:.0%}, incoming_queue={pending_files}): '
                    'Learner is starved for games — add more self-play actors.'
                )
            else:
                logging.warning(
                    f'[AsyncLearner] BOTTLENECK=LEARNER (balance={balance:+.2f}, '
                    f'incoming_queue={pending_files}, replay_ratio={self.throttler.current_replay_ratio:.3f}'
                    f'/{self.throttler.max_replay_ratio}): '
                    'Actors are outpacing the learner — reduce actors or use a faster learner device.'
                )

        if self.tb_logger is None:
            return
        step = self.total_ingested_envsteps
        cur_iter = int(self.learner.train_iter)
        self.tb_logger.add_scalar('async_learner/actor_learner_balance', balance, cur_iter)
        self.tb_logger.add_scalar('async_learner/throttle_rate', throttle_rate, cur_iter)
        self.tb_logger.add_scalar('async_learner/incoming_queue_files', pending_files, cur_iter)
        self.tb_logger.add_scalar('async_learner/ingested_envsteps', step, cur_iter)
        self.tb_logger.add_scalar('async_learner/ingested_episodes', self.total_ingested_episodes, cur_iter)
        self.tb_logger.add_scalar('async_learner/replay_ratio', self.throttler.current_replay_ratio, cur_iter)
        self.tb_logger.add_scalar(
            'async_learner/buffer_transitions', self.replay_buffer.get_num_of_transitions(), cur_iter
        )
        self.tb_logger.add_scalar(
            'async_learner/buffer_segments', self.replay_buffer.get_num_of_game_segments(), cur_iter
        )
        self.tb_logger.add_scalar('async_learner/throttled', throttled, cur_iter)
        if self._recent_weight_lags:
            mean_lag = sum(self._recent_weight_lags) / len(self._recent_weight_lags)
            self.tb_logger.add_scalar('async_learner/weight_lag_mean', mean_lag, cur_iter)
        if self._batch_prefetcher is not None:
            self.tb_logger.add_scalar(
                'async_learner/prefetch_queue_size', self._batch_prefetcher.queue.qsize(), cur_iter
            )
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
        self.tb_logger.add_scalar('async_learner/rss_mb', rss_mb, cur_iter)

    def request_stop(self, *_args: Any) -> None:
        self._stop_requested = True

    def run(
        self,
        max_train_iter: int = int(1e10),
        max_env_step: int = int(1e10),
        max_idle_seconds: Optional[float] = None,
        install_signal_handlers: bool = True,
    ) -> Dict[str, Any]:
        """Run the async learner loop until limits are reached or stopped."""
        if install_signal_handlers:
            signal.signal(signal.SIGINT, self.request_stop)
            signal.signal(signal.SIGTERM, self.request_stop)

        if self.use_bg_reader:
            self._bg_reader = _BackgroundSegmentReader(
                self.dirs.incoming_dir, poll_interval=self.poll_interval
            )
            self._bg_reader.start()

        self.start_prefetcher()

        last_progress_time = time.time()
        last_throttle_log_time = 0.0
        try:
            while not self._stop_requested:
                ingest_stats = self.ingest_pending_segments()
                if ingest_stats.files_ingested > 0:
                    last_progress_time = time.time()

                # Run a burst of updates while allowed, interspersing ingestion checks
                did_update = False
                for _ in range(self.publish_every_n_updates):
                    if self._stop_requested:
                        break
                    if self.learner.train_iter >= max_train_iter or self.total_ingested_envsteps >= max_env_step:
                        break
                    result = self.train_step()
                    if result is None:
                        break
                    did_update = True
                    last_progress_time = time.time()

                if self.learner.train_iter >= max_train_iter or self.total_ingested_envsteps >= max_env_step:
                    break

                if not did_update:
                    now = time.time()
                    if now - last_throttle_log_time >= 5.0:
                        sampleable = self.replay_buffer.get_num_of_sampleable_transitions()
                        _, reason = self.throttler.can_update(sampleable)
                        logging.info(
                            f'[AsyncLearner] Waiting ({reason}): iter={self.learner.train_iter}, '
                            f'envsteps={self.total_ingested_envsteps}, '
                            f'buffer_sampleable={sampleable}/{self.throttler.min_transitions}, '
                            f'replay_ratio={self.throttler.current_replay_ratio:.3f}'
                        )
                        if reason == 'throttled':
                            self._log_async_metrics(throttled=1.0)
                        last_throttle_log_time = now

                    if max_idle_seconds is not None and (now - last_progress_time) >= max_idle_seconds:
                        logging.info('[AsyncLearner] Max idle time reached; exiting.')
                        break
                    time.sleep(self.poll_interval)
        finally:
            self.publish_current_weights()
            self.close()

        return {
            'train_iter': int(self.learner.train_iter),
            'ingested_envsteps': int(self.total_ingested_envsteps),
            'ingested_episodes': int(self.total_ingested_episodes),
            'files_ingested': int(self.total_files_ingested),
            'buffer_segments': int(self.replay_buffer.get_num_of_game_segments()),
            'buffer_transitions': int(self.replay_buffer.get_num_of_transitions()),
            'replay_ratio': float(self.throttler.current_replay_ratio),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._batch_prefetcher is not None:
            self._batch_prefetcher.stop()
            self._batch_prefetcher = None
        if self._bg_reader is not None:
            self._bg_reader.stop()
            self._bg_reader = None
        try:
            self.learner.call_hook('after_run')
        except Exception:
            pass
        if self.tb_logger is not None:
            try:
                self.tb_logger.flush()
                self.tb_logger.close()
            except Exception:
                pass
