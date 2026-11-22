# -*- coding: utf-8 -*-
"""
Unit tests proving the board-game value-target sign-flip fix in
``EfficientZeroGameBuffer._compute_target_reward_value`` and
``SampledEfficientZeroGameBuffer._compute_target_reward_value``.

In a two-player alternating board game (``to_play`` in ``{1, 2}``), the reward of
transition ``current_index + i`` is from the perspective of the player to play at
that transition, while the value target is from the perspective of the player to
play at ``current_index``. The reward sign must be positive when
``to_play_list[current_index + i] == to_play_list[current_index]`` and negated
otherwise.

The pre-fix code checked ``to_play_list[base_index] == to_play_list[i]``, mixing
a segment-global index (``base_index``) with a 0-based slice offset (``i``), which
inverted every reward sign whenever ``base_index`` was odd (e.g. ``state_index = 1``).
"""

import numpy as np
import pytest
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_efficientzero import EfficientZeroGameBuffer
from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer


class _StubModelOutput:
    """Minimal stand-in for network output on the ``model.training=True`` /
    ``use_root_value=False`` path, where only ``.value`` is consumed."""

    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)


class _StubModel:
    """Stub model returning a constant bootstrap value for every state."""

    training = True

    def __init__(self, bootstrap_value=0.0):
        self._bootstrap_value = float(bootstrap_value)

    def initial_inference(self, m_obs):
        n = int(m_obs.shape[0])
        return _StubModelOutput(np.full(n, self._bootstrap_value, dtype=np.float32))


def _make_buffer(buffer_cls, num_unroll_steps=0, discount_factor=1.0, lstm_horizon_len=1000):
    buf = object.__new__(buffer_cls)
    buf._cfg = EasyDict(
        dict(
            env_type='board_games',
            device='cpu',
            mini_infer_size=16,
            use_root_value=False,
            discount_factor=discount_factor,
            lstm_horizon_len=lstm_horizon_len,
            num_unroll_steps=num_unroll_steps,
            model=dict(
                model_type='mlp',
                continuous_action_space=False,
                action_space_size=5,
            ),
        )
    )
    return buf


def _build_context(rewards, to_play, state_index, td_steps, num_unroll_steps=0):
    """Build the 8-tuple ``reward_value_context`` expected by ``_compute_target_reward_value``."""
    n_transitions = num_unroll_steps + 1
    seg_len = len(rewards)
    # prepare_observation(..., model_type='mlp') expects [B, S, O]
    value_obs_list = [np.zeros((1, 4), dtype=np.float32) for _ in range(n_transitions)]
    value_mask = [1.0] * n_transitions
    pos_in_game_segment_list = [state_index]
    rewards_list = [np.asarray(rewards, dtype=np.float32)]
    game_segment_lens = [seg_len]
    td_steps_list = np.full(n_transitions, td_steps, dtype=np.int32)
    action_mask_segment = [
        [np.ones(5, dtype=np.int8) for _ in range(seg_len)]
    ]
    to_play_segment = [list(to_play)]
    return [
        value_obs_list,
        value_mask,
        pos_in_game_segment_list,
        rewards_list,
        game_segment_lens,
        td_steps_list,
        action_mask_segment,
        to_play_segment,
    ]


@pytest.mark.unittest
@pytest.mark.parametrize('buffer_cls', [SampledEfficientZeroGameBuffer, EfficientZeroGameBuffer])
class TestBoardGameValueTargetSignFlip:

    def test_current_player_reward_positive_at_odd_state_index(self, buffer_cls):
        """At state_index=1 (Player 2), Player 2's own reward at index 1 must be +0.5.
        Pre-fix code compared to_play[1]==to_play[0] (2==1 -> False) and produced -0.5.
        """
        buf = _make_buffer(buffer_cls, num_unroll_steps=0, discount_factor=1.0)
        ctx = _build_context(
            rewards=[0.0, 0.5, 0.0],
            to_play=[1, 2, 1],
            state_index=1,
            td_steps=2,
            num_unroll_steps=0,
        )
        _, target_values = buf._compute_target_reward_value(ctx, _StubModel(0.0))
        target_values = np.asarray(target_values, dtype=np.float32)
        assert target_values.shape == (1, 1)
        np.testing.assert_allclose(target_values, np.array([[0.5]], dtype=np.float32), rtol=1e-5, atol=1e-6)

    def test_opponent_lookahead_reward_negated_at_odd_state_index(self, buffer_cls):
        """At state_index=1 (Player 2), Player 1's reward at index 2 must be -0.6.
        Pre-fix code compared to_play[1]==to_play[1] (2==2 -> True) and produced +0.6.
        """
        buf = _make_buffer(buffer_cls, num_unroll_steps=0, discount_factor=1.0)
        ctx = _build_context(
            rewards=[0.0, 0.0, 0.6],
            to_play=[1, 2, 1],
            state_index=1,
            td_steps=2,
            num_unroll_steps=0,
        )
        _, target_values = buf._compute_target_reward_value(ctx, _StubModel(0.0))
        target_values = np.asarray(target_values, dtype=np.float32)
        assert target_values.shape == (1, 1)
        np.testing.assert_allclose(target_values, np.array([[-0.6]], dtype=np.float32), rtol=1e-5, atol=1e-6)

    def test_alternating_signs_across_unroll_with_horizon_reset(self, buffer_cls):
        """Across a 3-step unroll (indices 0, 1, 2) with terminal +1.0 for P1 at index 2
        and lstm_horizon_len=1, target values must alternate [+1.0, -1.0, +1.0].
        """
        buf = _make_buffer(buffer_cls, num_unroll_steps=2, discount_factor=1.0, lstm_horizon_len=1)
        ctx = _build_context(
            rewards=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            to_play=[1, 2, 1, 2, 1, 2],
            state_index=0,
            td_steps=3,
            num_unroll_steps=2,
        )
        _, target_values = buf._compute_target_reward_value(ctx, _StubModel(0.0))
        target_values = np.asarray(target_values, dtype=np.float32)
        assert target_values.shape == (1, 3)
        np.testing.assert_allclose(
            target_values,
            np.array([[1.0, -1.0, 1.0]], dtype=np.float32),
            rtol=1e-5,
            atol=1e-6,
        )
