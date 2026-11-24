"""Replay-ratio throttling for the decoupled async learner.

Prevents a fast GPU learner from overfitting stale replay buffer data when it
outpaces CPU self-play actors.
"""

import math
from typing import Optional, Tuple


class ReplayRatioThrottler:
    """Controls when the learner may execute gradient updates.

    Replay ratio is defined as:
        ``total_updates / total_raw_transitions_ingested``

    Using raw (pre-symmetry) ingested transitions keeps the ratio directly
    comparable to synchronous ``update_per_collect / collected_envsteps``
    regardless of whether 4x symmetry augmentation is enabled.
    """

    def __init__(
        self,
        min_transitions: int,
        max_replay_ratio: Optional[float] = None,
        initial_updates: int = 0,
        initial_raw_transitions: int = 0,
    ) -> None:
        if min_transitions < 1:
            raise ValueError(f'min_transitions must be >= 1, got {min_transitions}')
        if max_replay_ratio is not None and max_replay_ratio <= 0:
            raise ValueError(f'max_replay_ratio must be > 0 or None, got {max_replay_ratio}')
        self.min_transitions = int(min_transitions)
        self.max_replay_ratio = float(max_replay_ratio) if max_replay_ratio is not None else None
        self.total_updates = int(initial_updates)
        self.total_raw_transitions = int(initial_raw_transitions)
        self.total_augmented_transitions = 0

    def record_ingested(self, raw_transitions: int, augmented_transitions: Optional[int] = None) -> None:
        """Record newly ingested transitions from actor segment batches."""
        raw = max(0, int(raw_transitions))
        self.total_raw_transitions += raw
        if augmented_transitions is None:
            self.total_augmented_transitions += raw
        else:
            self.total_augmented_transitions += max(0, int(augmented_transitions))

    def record_update(self, n: int = 1) -> None:
        """Record completed learner gradient update(s)."""
        self.total_updates += max(0, int(n))

    @property
    def current_replay_ratio(self) -> float:
        """Current updates per raw ingested transition."""
        if self.total_raw_transitions <= 0:
            return 0.0 if self.total_updates == 0 else float('inf')
        return self.total_updates / float(self.total_raw_transitions)

    @property
    def allowed_updates_remaining(self) -> Optional[int]:
        """Number of updates still permitted under ``max_replay_ratio``, or ``None`` if unthrottled."""
        if self.max_replay_ratio is None:
            return None
        max_allowed = int(math.floor(self.total_raw_transitions * self.max_replay_ratio + 1e-9))
        return max(0, max_allowed - self.total_updates)

    def can_update(self, sampleable_buffer_transitions: int) -> Tuple[bool, str]:
        """Return ``(allowed, reason)`` where reason is ``'ok'``, ``'warmup'``, or ``'throttled'``."""
        if int(sampleable_buffer_transitions) < self.min_transitions:
            return False, 'warmup'
        if self.max_replay_ratio is not None:
            remaining = self.allowed_updates_remaining
            if remaining is not None and remaining <= 0:
                return False, 'throttled'
        return True, 'ok'
