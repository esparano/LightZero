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
    atomic_json_save,
    atomic_pickle_save,
    consume_segment_file,
    detect_existing_run_artifacts,
    extract_optimizer_step,
    find_latest_checkpoint,
    find_non_finite_tensors,
    list_incoming_segment_files,
    parse_checkpoint_iter,
    publish_weights,
    read_json,
    read_pickle,
    read_segment_batch,
)
from .replay_throttle import ReplayRatioThrottler, compute_actor_learner_balance
from .tb_utils import inspect_tb_max_steps, truncate_tb_events_after


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
                t.join(timeout=10.0)

    def is_alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)

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
        resume: bool = False,
        overwrite: bool = False,
        persist_replay_buffer: bool = True,
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
        self.resume = bool(resume)
        self.overwrite = bool(overwrite)
        self.persist_replay_buffer = bool(persist_replay_buffer)

        if not self.resume and not self.overwrite:
            # Exclude log/evaluator/ so a concurrently spawned CPU evaluator creating its
            # event file before a CUDA learner finishes importing does not falsely trip the guard.
            existing_artifacts = detect_existing_run_artifacts(self.dirs, include_evaluator_logs=False)
            if existing_artifacts:
                summary = ', '.join(existing_artifacts)
                raise FileExistsError(
                    f"Run directory '{self.dirs.root_dir}' already contains existing run artifacts "
                    f'({summary}). Pass --resume to continue this run, choose a new --run-dir, '
                    'or pass --overwrite to start fresh in this directory.'
                )

        resume_ckpt_path: Optional[Path] = None
        if self.resume:
            if model_path is not None:
                resume_ckpt_path = Path(model_path)
                if not resume_ckpt_path.is_file():
                    raise FileNotFoundError(
                        f'Cannot resume: specified checkpoint does not exist: {resume_ckpt_path}'
                    )
            else:
                resume_ckpt_path = find_latest_checkpoint(self.dirs.ckpt_dir)
                if resume_ckpt_path is None:
                    raise FileNotFoundError(
                        f"Cannot resume: no 'iteration_<N>.pth.tar' checkpoint found in '{self.dirs.ckpt_dir}'."
                    )

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

        # Validate env config before mutating logs/weights or loading large checkpoints/buffers.
        env_fn, _, _ = get_vec_env_setting(self.cfg.env)
        if hasattr(env_fn, 'validate_config'):
            env_fn.validate_config(self.cfg)

        # SampledEfficientZeroPolicy._forward_learn logs _collect_mcts_temperature (must be float for DI-engine monitor).
        self.policy = create_policy(self.cfg.policy, model=None, enable_field=['learn', 'collect'])

        resumed_iter = 0
        resumed_envsteps = 0
        has_resumed_envsteps = False
        resumed_episodes = 0
        resumed_files = 0
        ckpt_to_load = resume_ckpt_path if self.resume else (Path(model_path) if model_path is not None else None)
        if ckpt_to_load is not None:
            ckpt = torch.load(str(ckpt_to_load), map_location=self.cfg.policy.device, weights_only=False)
            if self.resume:
                if ckpt.get('last_iter') is not None:
                    resumed_iter = int(ckpt['last_iter'])
                elif ckpt.get('train_iter') is not None:
                    resumed_iter = int(ckpt['train_iter'])
                else:
                    parsed_iter = parse_checkpoint_iter(ckpt_to_load)
                    resumed_iter = int(parsed_iter) if parsed_iter is not None else 0

                opt_step = extract_optimizer_step(ckpt)
                if opt_step is not None and opt_step > resumed_iter:
                    raise ValueError(
                        f"Cannot --resume from '{ckpt_to_load}': checkpoint reports "
                        f'last_iter={resumed_iter}, but optimizer state has step={opt_step}. '
                        'This indicates the iteration counter was previously reset (e.g. restarted '
                        'without --resume). Resuming into this run directory would overwrite '
                        'existing checkpoints and cause backward horizontal lines in TensorBoard. '
                        'Repair the checkpoint/logs first or start in a new --run-dir.'
                    )

                if model_path is not None:
                    highest_ckpt = find_latest_checkpoint(self.dirs.ckpt_dir)
                    if highest_ckpt is not None:
                        highest_iter = parse_checkpoint_iter(highest_ckpt)
                        if highest_iter is not None and resumed_iter < highest_iter:
                            raise ValueError(
                                f"Cannot --resume into '{self.dirs.root_dir}' from '{ckpt_to_load.name}' "
                                f'(train_iter={resumed_iter}) because newer checkpoint '
                                f"'{highest_ckpt.name}' (train_iter={highest_iter}) already exists in "
                                f"'{self.dirs.ckpt_dir}'. Resuming from an earlier iteration in the same "
                                'run directory would overwrite checkpoints and create backward lines in '
                                'TensorBoard. Use a new --run-dir to branch from an earlier checkpoint, '
                                'or archive/remove newer checkpoints first.'
                            )

                if ckpt.get('last_step') is not None:
                    resumed_envsteps = int(ckpt['last_step'])
                    has_resumed_envsteps = True
                elif ckpt.get('envstep') is not None:
                    resumed_envsteps = int(ckpt['envstep'])
                    has_resumed_envsteps = True

                saved_state = self._read_learner_state_safe()
                if saved_state is not None and int(saved_state.get('train_iter', -1)) == resumed_iter:
                    resumed_envsteps = max(resumed_envsteps, int(saved_state.get('total_ingested_envsteps', 0)))
                    has_resumed_envsteps = True
                    resumed_episodes = int(saved_state.get('total_ingested_episodes', 0))
                    resumed_files = int(saved_state.get('total_files_ingested', 0))

            if 'optimizer' in ckpt and 'target_model' in ckpt:
                self.policy.learn_mode.load_state_dict(ckpt)
            elif 'model' in ckpt:
                self.policy._learn_model.load_state_dict(ckpt['model'])
                if hasattr(self.policy, '_target_model') and self.policy._target_model is not None:
                    self.policy._target_model.load_state_dict(ckpt['model'])
            else:
                raise ValueError(f'Unrecognized checkpoint format at {ckpt_to_load}')

            if (
                self.resume
                and resumed_iter > 0
                and hasattr(self.policy, '_target_model')
                and hasattr(self.policy._target_model, 'reset_state')
            ):
                self.policy._target_model.reset_state(target_update_count=resumed_iter)

        if self.resume:
            self._reconcile_resumed_logs_and_eval_state(
                resumed_iter=resumed_iter,
                resumed_envsteps=resumed_envsteps if has_resumed_envsteps else None,
            )

        self.policy._collect_mcts_temperature = float(
            visit_count_temperature(
                self.cfg.policy.manual_temperature_decay,
                self.cfg.policy.fixed_temperature_value,
                self.cfg.policy.threshold_training_steps_for_final_temperature,
                trained_steps=resumed_iter,
            )
        )

        self.tb_logger = SummaryWriter(str(self.dirs.log_dir / 'learner'))
        self.learner = BaseLearner(
            self.cfg.policy.learn.learner,
            self.policy.learn_mode,
            self.tb_logger,
            exp_name=self.cfg.exp_name,
        )
        if self.resume and resumed_iter > 0:
            self.learner._last_iter.update(resumed_iter)
        if self.resume and resumed_envsteps > 0:
            self.learner._collector_envstep = resumed_envsteps

        # Initialize replay buffer with exact env_fn.reflect_game_segment attachment
        buffer_cls = _select_game_buffer_cls(self.create_cfg.policy.type)
        self.replay_buffer = buffer_cls(self.cfg.policy)
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

        self.total_ingested_envsteps = resumed_envsteps
        self.total_ingested_episodes = resumed_episodes
        self.total_files_ingested = resumed_files
        self._last_published_iter = resumed_iter - 1
        self._last_saved_ckpt_iter = resumed_iter
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
        # Publish initial (or resumed) weights before loading replay_buffer.pkl so actors
        # and evaluator immediately observe the reconciled train_iter rather than stale weights.
        self.publish_current_weights()

        if self.resume and self.persist_replay_buffer:
            self._load_replay_buffer_snapshot(expected_iter=resumed_iter)

        if self.resume:
            logging.info(
                f'[AsyncLearner] Resumed from {resume_ckpt_path}: '
                f'train_iter={int(self.learner.train_iter)}, '
                f'ingested_envsteps={self.total_ingested_envsteps}, '
                f'ingested_episodes={self.total_ingested_episodes}, '
                f'buffer_transitions={self.replay_buffer.get_num_of_transitions()}'
            )

    def _reconcile_resumed_logs_and_eval_state(
        self,
        resumed_iter: int,
        resumed_envsteps: Optional[int],
    ) -> None:
        """Prune orphaned post-checkpoint TensorBoard events (e.g. after an unclean crash)
        and roll back ``evaluator_state.json`` if it advanced past ``resumed_iter``.
        """
        learner_stats = truncate_tb_events_after(
            self.dirs.log_dir / 'learner',
            max_iter=resumed_iter,
            max_envstep=resumed_envsteps,
            pre_increment_iter_prefixes=('learner_iter/',),
        )
        if learner_stats['events_dropped'] > 0:
            logging.warning(
                f'[AsyncLearner] Pruned {learner_stats["events_dropped"]} orphaned post-checkpoint '
                f'learner TensorBoard event(s) beyond train_iter={resumed_iter}'
                + (f', envstep={resumed_envsteps}' if resumed_envsteps is not None else '')
                + f' (modified={learner_stats["files_modified"]}, removed={learner_stats["files_removed"]}).'
            )

        eval_log_dir = self.dirs.log_dir / 'evaluator'
        eval_stats = truncate_tb_events_after(
            eval_log_dir,
            max_iter=resumed_iter,
            max_envstep=resumed_envsteps,
            pre_increment_iter_prefixes=(),
        )
        if eval_stats['events_dropped'] > 0:
            logging.warning(
                f'[AsyncLearner] Pruned {eval_stats["events_dropped"]} orphaned post-checkpoint '
                f'evaluator TensorBoard event(s) beyond train_iter={resumed_iter}'
                + (f', envstep={resumed_envsteps}' if resumed_envsteps is not None else '')
                + f' (modified={eval_stats["files_modified"]}, removed={eval_stats["files_removed"]}).'
            )

        try:
            eval_state = read_json(self.dirs.evaluator_state_path)
        except Exception:
            eval_state = None
        if (
            eval_state is not None
            and eval_state.get('last_eval_iter') is not None
            and int(eval_state['last_eval_iter']) > resumed_iter
        ):
            old_eval_iter = int(eval_state['last_eval_iter'])
            clamped_iter = inspect_tb_max_steps(eval_log_dir)['max_iter']
            if clamped_iter is not None and clamped_iter > resumed_iter:
                clamped_iter = resumed_iter
            eval_state['last_eval_iter'] = clamped_iter
            eval_state['timestamp'] = time.time()
            try:
                atomic_json_save(eval_state, self.dirs.evaluator_state_path)
                logging.warning(
                    f'[AsyncLearner] Rolled back evaluator_state.json last_eval_iter '
                    f'from {old_eval_iter} to {clamped_iter} to match resumed train_iter={resumed_iter}.'
                )
            except Exception as exc:
                logging.warning(
                    f'[AsyncLearner] Failed to update {self.dirs.evaluator_state_path.name}: {exc}'
                )

    def _read_learner_state_safe(self) -> Optional[Dict[str, Any]]:
        try:
            return read_json(self.dirs.learner_state_path)
        except Exception as exc:
            logging.warning(f'[AsyncLearner] Ignoring unreadable {self.dirs.learner_state_path.name}: {exc}')
            return None

    def _save_learner_state(self, checkpoint_name: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {
            'version': 1,
            'train_iter': int(self.learner.train_iter),
            'total_ingested_envsteps': int(self.total_ingested_envsteps),
            'total_ingested_episodes': int(self.total_ingested_episodes),
            'total_files_ingested': int(self.total_files_ingested),
            'checkpoint_name': checkpoint_name,
            'timestamp': time.time(),
        }
        try:
            atomic_json_save(payload, self.dirs.learner_state_path)
        except Exception as exc:
            logging.warning(f'[AsyncLearner] Failed to save {self.dirs.learner_state_path.name}: {exc}')

    def _save_replay_buffer_snapshot(self) -> bool:
        with self._buffer_rwlock.write_lock():
            if self.replay_buffer.get_num_of_game_segments() <= 0:
                return False
            snapshot = {
                'version': 1,
                'train_iter': int(self.learner.train_iter),
                'game_segment_buffer': self.replay_buffer.game_segment_buffer,
                'game_pos_priorities': self.replay_buffer.game_pos_priorities,
                'game_segment_game_pos_look_up': self.replay_buffer.game_segment_game_pos_look_up,
                'num_of_collected_episodes': int(self.replay_buffer.num_of_collected_episodes),
                'base_idx': int(self.replay_buffer.base_idx),
                'clear_time': float(self.replay_buffer.clear_time),
            }
            try:
                atomic_pickle_save(snapshot, self.dirs.replay_buffer_path)
                logging.info(
                    f'[AsyncLearner] Saved replay buffer snapshot to {self.dirs.replay_buffer_path} '
                    f'(train_iter={snapshot["train_iter"]}, '
                    f'segments={len(snapshot["game_segment_buffer"])}, '
                    f'transitions={len(snapshot["game_segment_game_pos_look_up"])})'
                )
                return True
            except Exception as exc:
                logging.warning(f'[AsyncLearner] Failed to save replay buffer snapshot: {exc}')
                return False

    def _load_replay_buffer_snapshot(self, expected_iter: int) -> bool:
        if not self.dirs.replay_buffer_path.is_file():
            return False
        try:
            snapshot = read_pickle(self.dirs.replay_buffer_path)
        except Exception as exc:
            logging.warning(f'[AsyncLearner] Could not read replay buffer snapshot; starting cold: {exc}')
            return False
        if not isinstance(snapshot, dict):
            logging.warning('[AsyncLearner] Invalid replay buffer snapshot format; starting cold.')
            return False
        snap_iter = snapshot.get('train_iter')
        if snap_iter is not None and int(snap_iter) != int(expected_iter):
            logging.warning(
                f'[AsyncLearner] Replay buffer snapshot train_iter={snap_iter} does not match '
                f'resumed checkpoint train_iter={expected_iter}; skipping stale buffer.'
            )
            return False
        segments = snapshot.get('game_segment_buffer')
        priorities = snapshot.get('game_pos_priorities')
        lookup = snapshot.get('game_segment_game_pos_look_up')
        if segments is None or priorities is None or lookup is None or len(priorities) != len(lookup):
            logging.warning('[AsyncLearner] Corrupted/misaligned replay buffer snapshot; starting cold.')
            return False
        self.replay_buffer.game_segment_buffer = list(segments)
        self.replay_buffer.game_pos_priorities = priorities
        self.replay_buffer.game_segment_game_pos_look_up = list(lookup)
        self.replay_buffer.num_of_collected_episodes = int(snapshot.get('num_of_collected_episodes', 0))
        self.replay_buffer.base_idx = int(snapshot.get('base_idx', 0))
        self.replay_buffer.clear_time = float(snapshot.get('clear_time', 0.0))
        self.replay_buffer.remove_oldest_data_to_fit()
        logging.info(
            f'[AsyncLearner] Restored replay buffer snapshot: '
            f'segments={self.replay_buffer.get_num_of_game_segments()}, '
            f'transitions={self.replay_buffer.get_num_of_transitions()}'
        )
        return True

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

    def _sample_uses_mcts(self) -> bool:
        reanalyze_ratio = float(self.cfg.policy.get('reanalyze_ratio', 0.0) or 0.0)
        return reanalyze_ratio > 0.0 or bool(self.cfg.policy.get('use_root_value', False))

    def start_prefetcher(self) -> None:
        """Start background batch prefetch worker thread(s) when ``prefetch_workers > 0``."""
        if self.prefetch_workers <= 0 or self._batch_prefetcher is not None:
            return
        effective_workers = self.prefetch_workers
        if self._sample_uses_mcts() and effective_workers > 1:
            logging.info(
                f'[AsyncLearner] Capping prefetch_workers from {effective_workers} to 1 '
                f'because replay sampling runs MCTS (reanalyze_ratio={self.cfg.policy.get("reanalyze_ratio", 0.0)}, '
                f'use_root_value={bool(self.cfg.policy.get("use_root_value", False))}).'
            )
            effective_workers = 1
        self._prefetch_policy_proxy = _PrefetchPolicyProxy(self.policy)
        self._batch_prefetcher = _BackgroundBatchPrefetcher(
            replay_buffer=self.replay_buffer,
            policy_proxy=self._prefetch_policy_proxy,
            batch_size=int(self.cfg.policy.batch_size),
            min_transitions=self.throttler.min_transitions,
            buffer_rwlock=self._buffer_rwlock,
            target_rwlock=self._target_rwlock,
            num_workers=effective_workers,
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
            if self._sample_uses_mcts():
                # Always surface any worker error immediately, even if the worker thread has already exited.
                batch = self._batch_prefetcher.get_batch(timeout=0.0)
                if batch is not None:
                    return batch
                with self._buffer_rwlock.read_lock():
                    can_prefetch = (
                        self.replay_buffer.get_num_of_sampleable_transitions()
                        >= self._batch_prefetcher._min_transitions
                        and self.replay_buffer.get_num_of_transitions()
                        > self._batch_prefetcher._batch_size
                    )
                if can_prefetch:
                    # Reanalyze batches take >50ms; wait for the prefetch worker instead of timing out
                    # and launching a duplicate concurrent GPU MCTS reanalyze on the main thread.
                    while not self._stop_requested and self._batch_prefetcher.is_alive():
                        batch = self._batch_prefetcher.get_batch(timeout=0.1)
                        if batch is not None:
                            return batch
                    # Final non-blocking check surfaces any error or batch produced right before exit.
                    batch = self._batch_prefetcher.get_batch(timeout=0.0)
                    if batch is not None:
                        return batch
                    if self._stop_requested:
                        return None
            else:
                batch = self._batch_prefetcher.get_batch(timeout=0.05)
                if batch is not None:
                    return batch
        if self._stop_requested:
            return None
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
        if train_data is None:
            return None
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
            ckpt_name = f'iteration_{cur_iter}.pth.tar'
            self.learner.save_checkpoint(ckpt_name)
            self._save_learner_state(checkpoint_name=ckpt_name)
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

                if (
                    self._stop_requested
                    or self.learner.train_iter >= max_train_iter
                    or self.total_ingested_envsteps >= max_env_step
                ):
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
            try:
                self.publish_current_weights()
            except FloatingPointError as exc:
                logging.error(f'[AsyncLearner] Skipping final weight publish on shutdown: {exc}')
            finally:
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
        self.learner._collector_envstep = int(self.total_ingested_envsteps)
        bad_keys = find_non_finite_tensors(self.policy._learn_model.state_dict())
        if bad_keys:
            preview = ', '.join(bad_keys[:5])
            logging.error(
                f'[AsyncLearner] Skipping shutdown checkpoint/state save due to non-finite weights '
                f'({len(bad_keys)} bad tensors: {preview}).'
            )
        else:
            try:
                self.learner.call_hook('after_run')
            except Exception as exc:
                logging.warning(f'[AsyncLearner] after_run hook failed during close: {exc}')
            self._save_learner_state(checkpoint_name=f'iteration_{int(self.learner.train_iter)}.pth.tar')
            if self.persist_replay_buffer:
                self._save_replay_buffer_snapshot()
        if self.tb_logger is not None:
            try:
                self.tb_logger.flush()
                self.tb_logger.close()
            except Exception:
                pass
