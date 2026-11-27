"""Filesystem I/O, atomic serialization, weight watching, and config loading
for the lightweight decoupled actor-learner-evaluator split (no Ray).
"""

import copy
import importlib.util
import json
import os
import pickle
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from easydict import EasyDict

_ITER_CKPT_RE = re.compile(r'^iteration_(\d+)\.pth\.tar$')


@dataclass(frozen=True)
class AsyncSplitDirs:
    """Standard directory layout for an async actor-learner-evaluator run.

    All workers (local or remote over a shared filesystem) coordinate solely
    through atomic file operations inside ``root_dir``.
    """

    root_dir: Path
    weights_dir: Path
    latest_weights_path: Path
    ckpt_dir: Path
    learner_state_path: Path
    replay_buffer_path: Path
    evaluator_state_path: Path
    segments_dir: Path
    incoming_dir: Path
    processed_dir: Path
    log_dir: Path

    @classmethod
    def from_root(cls, root_dir: os.PathLike | str) -> 'AsyncSplitDirs':
        root = Path(root_dir).resolve()
        weights_dir = root / 'weights'
        ckpt_dir = root / 'ckpt'
        segments_dir = root / 'segments'
        return cls(
            root_dir=root,
            weights_dir=weights_dir,
            latest_weights_path=weights_dir / 'latest_weights.pt',
            ckpt_dir=ckpt_dir,
            learner_state_path=ckpt_dir / 'learner_state.json',
            replay_buffer_path=ckpt_dir / 'replay_buffer.pkl',
            evaluator_state_path=ckpt_dir / 'evaluator_state.json',
            segments_dir=segments_dir,
            incoming_dir=segments_dir / 'incoming',
            processed_dir=segments_dir / 'processed',
            log_dir=root / 'log',
        )

    def ensure_exists(self, create_processed: bool = False) -> None:
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if create_processed:
            self.processed_dir.mkdir(parents=True, exist_ok=True)


def _atomic_tmp_path(target_path: Path) -> Path:
    token = f'{os.getpid()}.{uuid.uuid4().hex[:10]}'
    return target_path.parent / f'.{target_path.name}.tmp.{token}'


def cpu_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Detach and move all tensors in a model state_dict to CPU."""
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def atomic_torch_save(payload: Any, target_path: os.PathLike | str) -> Path:
    """Atomically write a PyTorch payload via temp file + ``os.replace``."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _atomic_tmp_path(target)
    try:
        torch.save(payload, str(tmp_path))
        os.replace(str(tmp_path), str(target))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return target


def atomic_json_save(payload: Dict[str, Any], target_path: os.PathLike | str) -> Path:
    """Atomically write a JSON dictionary via temp file + ``os.replace``."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _atomic_tmp_path(target)
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(str(tmp_path), str(target))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return target


def read_json(path: os.PathLike | str) -> Optional[Dict[str, Any]]:
    """Read a JSON dictionary if the file exists, otherwise return ``None``."""
    target = Path(path)
    if not target.is_file():
        return None
    with open(target, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f'Expected JSON object in {target}, got {type(data).__name__}')
    return data


def atomic_pickle_save(payload: Any, target_path: os.PathLike | str) -> Path:
    """Atomically write a pickled payload via temp file + ``os.replace``."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _atomic_tmp_path(target)
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(str(tmp_path), str(target))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return target


def read_pickle(path: os.PathLike | str) -> Any:
    """Read a pickled payload written by ``atomic_pickle_save``."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def parse_checkpoint_iter(path: os.PathLike | str) -> Optional[int]:
    """Extract iteration number from ``iteration_<N>.pth.tar``, or ``None``."""
    match = _ITER_CKPT_RE.match(Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def extract_optimizer_step(ckpt: Mapping[str, Any]) -> Optional[int]:
    """Return the maximum integer optimizer ``step`` recorded in ``ckpt['optimizer']``, or ``None``.

    PyTorch adaptive optimizers (e.g. Adam / AdamW) persist a monotonic per-parameter ``step``
    counter in ``optimizer['state'][param_id]['step']``. Comparing this against ``last_iter``
    detects checkpoints whose iteration counter was previously reset (e.g. legacy restarts
    without ``--resume``).
    """
    opt = ckpt.get('optimizer')
    if not isinstance(opt, Mapping):
        return None
    state = opt.get('state')
    if not isinstance(state, Mapping) or not state:
        return None
    max_step: Optional[int] = None
    for param_state in state.values():
        if not isinstance(param_state, Mapping) or 'step' not in param_state:
            continue
        raw_step = param_state['step']
        try:
            if torch.is_tensor(raw_step):
                if raw_step.numel() != 1:
                    continue
                val = int(raw_step.item())
            else:
                val = int(raw_step)
        except (TypeError, ValueError):
            continue
        if max_step is None or val > max_step:
            max_step = val
    return max_step


def find_latest_checkpoint(ckpt_dir: os.PathLike | str) -> Optional[Path]:
    """Return the ``iteration_<N>.pth.tar`` file with the highest numeric ``N``."""
    directory = Path(ckpt_dir)
    if not directory.is_dir():
        return None
    best_iter = -1
    best_path: Optional[Path] = None
    for entry in directory.iterdir():
        if not entry.is_file() or entry.name.startswith('.'):
            continue
        it = parse_checkpoint_iter(entry)
        if it is not None and it > best_iter:
            best_iter = it
            best_path = entry
    return best_path


def detect_existing_run_artifacts(
    dirs: AsyncSplitDirs,
    include_evaluator_logs: bool = True,
) -> List[str]:
    """Return descriptions of existing run artifacts that would be overwritten or
    mixed with TensorBoard history if starting a fresh run without ``--resume``.
    """
    found: List[str] = []
    if dirs.latest_weights_path.is_file():
        found.append(str(dirs.latest_weights_path.relative_to(dirs.root_dir)))
    if dirs.ckpt_dir.is_dir():
        ckpt_files = sorted(
            p.name
            for p in dirs.ckpt_dir.iterdir()
            if p.is_file()
            and not p.name.startswith('.')
            and (
                p.name.endswith('.pth.tar')
                or p.name in {'learner_state.json', 'evaluator_state.json', 'replay_buffer.pkl'}
            )
        )
        if ckpt_files:
            preview = ', '.join(ckpt_files[:3])
            suffix = f' (+{len(ckpt_files) - 3} more)' if len(ckpt_files) > 3 else ''
            found.append(f'ckpt/{preview}{suffix}')
    log_subdirs = ('learner', 'evaluator') if include_evaluator_logs else ('learner',)
    for sub in log_subdirs:
        sub_dir = dirs.log_dir / sub
        if sub_dir.is_dir():
            tb_events = [p for p in sub_dir.iterdir() if p.is_file() and p.name.startswith('events.out.tfevents.')]
            if tb_events:
                found.append(f'log/{sub}/ ({len(tb_events)} TensorBoard event file(s))')
    return found


def find_non_finite_tensors(state_dict: Dict[str, Any]) -> List[str]:
    """Return keys of any floating-point tensors containing NaN or Inf values."""
    bad_keys: List[str] = []
    for k, v in state_dict.items():
        if torch.is_tensor(v) and v.is_floating_point():
            if not bool(torch.isfinite(v).all().item()):
                bad_keys.append(str(k))
    return bad_keys


def publish_weights(
    weights_path: os.PathLike | str,
    model_state_dict: Dict[str, Any],
    train_iter: int,
    envstep: int,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Atomically publish CPU model weights + metadata for actors and evaluator.

    Raises FloatingPointError without touching ``weights_path`` if any floating-point
    tensor contains NaN/Inf, preventing corrupted weights from segfaulting C++ MCTS actors.
    """
    cpu_sd = cpu_state_dict(model_state_dict)
    bad_keys = find_non_finite_tensors(cpu_sd)
    if bad_keys:
        preview = ', '.join(bad_keys[:5])
        suffix = f' (+{len(bad_keys) - 5} more)' if len(bad_keys) > 5 else ''
        raise FloatingPointError(
            f'Refusing to publish non-finite weights at train_iter={int(train_iter)} '
            f'({len(bad_keys)} bad tensors: {preview}{suffix}).'
        )
    payload: Dict[str, Any] = {
        'model': cpu_sd,
        'train_iter': int(train_iter),
        'version': int(train_iter),
        'envstep': int(envstep),
        'timestamp': time.time(),
    }
    if extra_meta:
        payload.update(extra_meta)
    atomic_torch_save(payload, weights_path)
    return {k: v for k, v in payload.items() if k != 'model'}


class WeightWatcher:
    """Polls an atomically updated weights file and reloads a model when changed."""

    def __init__(self, weights_path: os.PathLike | str) -> None:
        self.weights_path = Path(weights_path)
        self.last_mtime_ns: Optional[int] = None
        self.last_size: Optional[int] = None
        self.last_ino: Optional[int] = None
        self.last_version: int = -1
        self.last_meta: Dict[str, Any] = {'train_iter': 0, 'version': -1, 'envstep': 0}

    def has_update(self) -> bool:
        try:
            st = self.weights_path.stat()
        except FileNotFoundError:
            return False
        if self.last_mtime_ns is None:
            return True
        return (
            st.st_mtime_ns != self.last_mtime_ns
            or st.st_size != self.last_size
            or st.st_ino != self.last_ino
        )

    def poll_and_load(
        self,
        model: torch.nn.Module,
        device: str = 'cpu',
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Reload ``model`` if ``weights_path`` has been updated.

        Returns metadata dict (``train_iter``, ``version``, ``envstep``, ...)
        when a new weight version was loaded, otherwise ``None``.
        """
        if not force and not self.has_update():
            return None

        try:
            st = self.weights_path.stat()
            payload = torch.load(str(self.weights_path), map_location=device, weights_only=False)
        except (FileNotFoundError, EOFError, RuntimeError, OSError):
            return None

        if not isinstance(payload, dict) or 'model' not in payload:
            return None

        version = int(payload.get('version', payload.get('train_iter', 0)))
        if (
            not force
            and version <= self.last_version
            and self.last_mtime_ns == st.st_mtime_ns
            and self.last_ino == st.st_ino
        ):
            return None

        model.load_state_dict(payload['model'])
        self.last_mtime_ns = st.st_mtime_ns
        self.last_size = st.st_size
        self.last_ino = st.st_ino
        self.last_version = version
        meta = {k: v for k, v in payload.items() if k != 'model'}
        meta.setdefault('train_iter', version)
        meta.setdefault('version', version)
        meta.setdefault('envstep', 0)
        self.last_meta = meta
        return meta


def write_segment_batch(
    incoming_dir: os.PathLike | str,
    segments: List[Any],
    metas: List[Dict[str, Any]],
    actor_id: str,
    seq: int,
    envsteps: int,
    episodes: int,
    weight_version: int,
) -> Path:
    """Atomically write a batch of ``GameSegment`` objects + metadata to ``incoming_dir``."""
    incoming = Path(incoming_dir)
    incoming.mkdir(parents=True, exist_ok=True)
    ts_ms = int(time.time() * 1000)
    safe_actor = str(actor_id).replace('/', '_').replace('\\', '_')
    filename = f'{ts_ms:013d}_{safe_actor}_{int(seq):06d}_{uuid.uuid4().hex[:8]}.seg.pkl'
    target = incoming / filename
    tmp_path = _atomic_tmp_path(target)

    payload = {
        'segments': segments,
        'metas': metas,
        'actor_id': str(actor_id),
        'seq': int(seq),
        'envsteps': int(envsteps),
        'episodes': int(episodes),
        'weight_version': int(weight_version),
        'timestamp': time.time(),
    }
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(str(tmp_path), str(target))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return target


def list_incoming_segment_files(incoming_dir: os.PathLike | str) -> List[Path]:
    """Return finalized ``*.seg.pkl`` files in chronological filename order."""
    incoming = Path(incoming_dir)
    if not incoming.exists():
        return []
    files = [
        p for p in incoming.iterdir()
        if p.is_file() and p.name.endswith('.seg.pkl') and not p.name.startswith('.')
    ]
    files.sort(key=lambda p: p.name)
    return files


def read_segment_batch(path: os.PathLike | str) -> Dict[str, Any]:
    """Read a serialized segment batch file produced by ``write_segment_batch``."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def consume_segment_file(
    path: os.PathLike | str,
    archive_dir: Optional[os.PathLike | str] = None,
) -> None:
    """Archive or delete a segment batch file after ingestion."""
    src = Path(path)
    if not src.exists():
        return
    if archive_dir is not None:
        dst_dir = Path(archive_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        os.replace(str(src), str(dst_dir / src.name))
    else:
        src.unlink()


def load_config_from_py(config_path: os.PathLike | str) -> Tuple[EasyDict, EasyDict]:
    """Load deep-copied ``(main_config, create_config)`` from a LightZero Python config file."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Config file not found: {path}')
    module_name = f'_lz_async_cfg_{uuid.uuid4().hex}'
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load config spec from: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'main_config') or not hasattr(module, 'create_config'):
        raise AttributeError(
            f'Config file {path} must define both `main_config` and `create_config`.'
        )
    return EasyDict(copy.deepcopy(module.main_config)), EasyDict(copy.deepcopy(module.create_config))
