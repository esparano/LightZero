"""TensorBoard event inspection and crash-rollback truncation utilities for async_split.

LightZero's ``BaseLearner`` and ``MuZeroEvaluator`` log two distinct step axes into
the same ``SummaryWriter`` directory:
  - iteration-indexed tags (e.g. ``learner_iter/*``, ``async_learner/*``, ``evaluator_iter/*``)
  - envstep-indexed tags (e.g. ``learner_step/*``, ``evaluator_step/*``)

TensorBoard's built-in ``SessionLog.START`` / ``purge_step`` purges all tags using a
single step threshold (``by_tags=False``), which would erase ``*_step/*`` history when
given ``train_iter``. The helpers here inspect and atomically prune orphaned
post-checkpoint events per tag axis while leaving clean event files untouched.
"""

import os
import struct
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from google.protobuf.message import DecodeError
from tensorboard.compat.proto import event_pb2
from tensorboardX.record_writer import masked_crc32c


def is_envstep_tag(tag: str) -> bool:
    """Return True if a TensorBoard scalar tag uses ``envstep`` rather than ``train_iter``."""
    prefix = tag.split('/', 1)[0]
    return prefix.endswith('_step') or '_step_task' in prefix


def list_tb_event_files(log_dir: os.PathLike | str) -> List[Path]:
    """Return existing ``events.out.tfevents.*`` files in chronological filename order."""
    directory = Path(log_dir)
    if not directory.is_dir():
        return []
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and not p.name.startswith('.') and p.name.startswith('events.out.tfevents.')
    ]
    files.sort(key=lambda p: p.name)
    return files


def _encode_tfrecord(payload: bytes) -> bytes:
    header = struct.pack('<Q', len(payload))
    header_crc = struct.pack('<I', masked_crc32c(header))
    data_crc = struct.pack('<I', masked_crc32c(payload))
    return header + header_crc + payload + data_crc


def _tb_rewrite_tmp_path(target_path: Path) -> Path:
    """Create a temp path whose basename does NOT contain ``tfevents``.

    TensorBoard's ``IsTensorFlowEventsFile`` checks ``'tfevents' in basename``, so
    temporary rewrite files must omit that substring to avoid being tailed live.
    """
    token = f'{os.getpid()}.{uuid.uuid4().hex[:10]}'
    return target_path.parent / f'.tb_rewrite.tmp.{token}'


def inspect_tb_max_steps(log_dir: os.PathLike | str) -> Dict[str, Optional[int]]:
    """Fast-scan ``events.out.tfevents.*`` in ``log_dir`` for max iter-axis and envstep-axis steps."""
    max_iter: Optional[int] = None
    max_envstep: Optional[int] = None

    for path in list_tb_event_files(log_dir):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        pos = 0
        end = len(data)
        ev = event_pb2.Event()
        while pos + 12 <= end:
            length = struct.unpack_from('<Q', data, pos)[0]
            rec_start = pos + 12
            rec_end = rec_start + length
            if rec_end + 4 > end:
                break
            try:
                ev.ParseFromString(data[rec_start:rec_end])
            except DecodeError:
                break
            if ev.HasField('summary'):
                step = int(ev.step)
                for val in ev.summary.value:
                    if is_envstep_tag(val.tag):
                        if max_envstep is None or step > max_envstep:
                            max_envstep = step
                    else:
                        if max_iter is None or step > max_iter:
                            max_iter = step
            pos = rec_end + 4

    return {'max_iter': max_iter, 'max_envstep': max_envstep}


def _should_drop_summary_value(
    tag: str,
    step: int,
    max_iter: int,
    max_envstep: Optional[int],
    pre_increment_iter_prefixes: Tuple[str, ...],
) -> bool:
    if is_envstep_tag(tag):
        return max_envstep is not None and step > max_envstep
    if pre_increment_iter_prefixes and tag.startswith(pre_increment_iter_prefixes):
        return step >= max_iter
    return step > max_iter


def truncate_tb_events_after(
    log_dir: os.PathLike | str,
    max_iter: int,
    max_envstep: Optional[int] = None,
    pre_increment_iter_prefixes: Tuple[str, ...] = ('learner_iter/',),
) -> Dict[str, int]:
    """Prune orphaned post-checkpoint TensorBoard summary events atomically per tag axis.

    For a checkpoint saved at ``(train_iter=max_iter, envstep=max_envstep)``:
      - ``learner_iter/*`` (logged by ``BaseLearner`` before ``_last_iter.add(1)``) drops
        events with ``step >= max_iter``.
      - Other iter-indexed tags (``async_learner/*``, ``evaluator_iter/*``) drop events
        with ``step > max_iter``.
      - Envstep-indexed tags (``learner_step/*``, ``evaluator_step/*``) drop events with
        ``step > max_envstep`` (when ``max_envstep`` is provided).

    Files containing zero orphaned events and no corrupt trailing bytes are left untouched.
    """
    max_iter = int(max_iter)
    max_envstep = int(max_envstep) if max_envstep is not None else None

    files_modified = 0
    files_removed = 0
    events_dropped = 0

    for path in list_tb_event_files(log_dir):
        try:
            data = path.read_bytes()
        except OSError:
            continue

        kept_chunks: List[bytes] = []
        file_dropped = 0
        orig_summary_events = 0
        kept_summary_events = 0
        truncated_tail = False

        pos = 0
        end = len(data)
        while pos < end:
            if pos + 12 > end:
                truncated_tail = True
                break
            length = struct.unpack_from('<Q', data, pos)[0]
            rec_start = pos + 12
            rec_end = rec_start + length
            if rec_end + 4 > end:
                truncated_tail = True
                break
            raw_rec = data[pos : rec_end + 4]
            ev = event_pb2.Event()
            try:
                ev.ParseFromString(data[rec_start:rec_end])
            except DecodeError:
                truncated_tail = True
                break

            if not ev.HasField('summary') or len(ev.summary.value) == 0:
                kept_chunks.append(raw_rec)
            else:
                orig_summary_events += 1
                step = int(ev.step)
                drop_flags = [
                    _should_drop_summary_value(
                        v.tag,
                        step,
                        max_iter=max_iter,
                        max_envstep=max_envstep,
                        pre_increment_iter_prefixes=pre_increment_iter_prefixes,
                    )
                    for v in ev.summary.value
                ]
                if not any(drop_flags):
                    kept_chunks.append(raw_rec)
                    kept_summary_events += 1
                elif all(drop_flags):
                    file_dropped += 1
                else:
                    filtered = event_pb2.Event()
                    filtered.CopyFrom(ev)
                    del filtered.summary.value[:]
                    for v, should_drop in zip(ev.summary.value, drop_flags):
                        if not should_drop:
                            filtered.summary.value.add().CopyFrom(v)
                    kept_chunks.append(_encode_tfrecord(filtered.SerializeToString()))
                    kept_summary_events += 1
                    file_dropped += 1

            pos = rec_end + 4

        if file_dropped == 0 and not truncated_tail:
            continue

        events_dropped += file_dropped
        if orig_summary_events > 0 and kept_summary_events == 0:
            try:
                path.unlink()
                files_removed += 1
            except OSError:
                pass
            continue

        tmp_path = _tb_rewrite_tmp_path(path)
        try:
            with open(tmp_path, 'wb') as f:
                for chunk in kept_chunks:
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp_path), str(path))
            files_modified += 1
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    return {
        'files_modified': files_modified,
        'files_removed': files_removed,
        'events_dropped': events_dropped,
    }
