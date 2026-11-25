# -*- coding: utf-8 -*-
"""
Unit tests for the fast paths in
``SampledEfficientZeroGameBuffer._compute_target_reward_value``:

1. When ``value_mask`` is all zeros (e.g., Galcon with ``td_steps == max_episode_steps``),
   neither target-model inference nor ``legal_actions`` extraction is executed, while
   discounted alternating-player Monte Carlo value targets and value prefixes remain exact.
2. When ``use_root_value=False`` with a mixed ``value_mask``, ``_preprocess_to_play_and_action_mask``
   is skipped and ``model.initial_inference`` is only called on unmasked (``value_mask != 0``)
   observations, producing results identical to evaluating every observation.
3. On ``model.training=False`` with ``use_root_value=False``, only ``m_output.value`` is
   transformed/detached (no unnecessary D2H copies of ``latent_state`` / ``policy_logits``).
"""

import numpy as np
import pytest
import torch
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer
from lzero.policy import DiscreteSupport, inverse_scalar_transform


class _RecordingOutput:
    def __init__(self, value, include_extras=True):
        self.value = value
        if include_extras:
            self.latent_state = None
            self.policy_logits = None
            self.reward_hidden_state = None


class _RecordingModel:
    """Stub model that records every ``initial_inference`` call and batch size."""

    def __init__(self, values_fn, training=True, fail_on_call=False):
        self._values_fn = values_fn
        self.training = training
        self.fail_on_call = fail_on_call
        self.call_batch_sizes = []

    def initial_inference(self, m_obs):
        if self.fail_on_call:
            raise AssertionError('model.initial_inference should not be called when value_mask is all zeros')
        self.call_batch_sizes.append(int(m_obs.shape[0]))
        return _RecordingOutput(self._values_fn(m_obs), include_extras=False)


def _make_buffer(num_unroll_steps=2, discount_factor=0.99, lstm_horizon_len=5, use_root_value=False):
    buf = object.__new__(SampledEfficientZeroGameBuffer)
    buf._cfg = EasyDict(
        dict(
            env_type='board_games',
            device='cpu',
            mini_infer_size=16,
            use_root_value=use_root_value,
            discount_factor=discount_factor,
            lstm_horizon_len=lstm_horizon_len,
            num_unroll_steps=num_unroll_steps,
            model=dict(
                model_type='mlp',
                continuous_action_space=False,
                action_space_size=4097,
            ),
        )
    )
    buf.value_support = DiscreteSupport(-1.0, 2.0, 1.0)
    return buf


def _build_context(rewards, to_play, state_index, td_steps, value_mask, obs_markers=None, num_unroll_steps=2):
    n_transitions = num_unroll_steps + 1
    seg_len = len(rewards)
    if obs_markers is None:
        obs_markers = list(range(n_transitions))
    value_obs_list = [np.full((1, 4), float(m), dtype=np.float32) for m in obs_markers]
    pos_in_game_segment_list = [state_index]
    rewards_list = [np.asarray(rewards, dtype=np.float32)]
    game_segment_lens = [seg_len]
    td_steps_list = np.full(n_transitions, td_steps, dtype=np.int32)
    action_mask_segment = [[np.ones(4097, dtype=np.int8) for _ in range(seg_len)]]
    to_play_segment = [list(to_play)]
    return [
        value_obs_list,
        list(value_mask),
        pos_in_game_segment_list,
        rewards_list,
        game_segment_lens,
        td_steps_list,
        action_mask_segment,
        to_play_segment,
    ]


@pytest.mark.unittest
class TestSEZRewardValueFastPath:

    def test_all_zero_value_mask_skips_model_and_action_mask_preprocessing(self, monkeypatch):
        """When value_mask is all zeros (td_steps >= game_len), skip both inference and legal_actions."""
        buf = _make_buffer(num_unroll_steps=2, discount_factor=0.9, lstm_horizon_len=5, use_root_value=False)

        def _forbid_preprocess(*args, **kwargs):
            raise AssertionError('_preprocess_to_play_and_action_mask should be skipped')

        monkeypatch.setattr(buf, '_preprocess_to_play_and_action_mask', _forbid_preprocess)

        # 4-step game: P1 (0), P2 (1), P1 (2), P2 (3) wins with +1.0 at index 3.
        ctx = _build_context(
            rewards=[0.0, 0.0, 0.0, 1.0],
            to_play=[1, 2, 1, 2],
            state_index=1,
            td_steps=4,
            value_mask=[0, 0, 0],
            num_unroll_steps=2,
        )
        model = _RecordingModel(values_fn=lambda obs: np.zeros(obs.shape[0]), fail_on_call=True)

        value_prefixes, target_values = buf._compute_target_reward_value(ctx, model)

        assert model.call_batch_sizes == []
        # state_index=1 (P2): reward at idx 3 (P2, same player) -> +1.0 * 0.9^2 = +0.81
        # current_index=2 (P1): reward at idx 3 (P2, opp player) -> -1.0 * 0.9^1 = -0.90
        # current_index=3 (P2): reward at idx 3 (P2, same player) -> +1.0 * 0.9^0 = +1.00
        np.testing.assert_allclose(target_values, np.array([[0.81, -0.9, 1.0]]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(value_prefixes, np.array([[0.0, 0.0, 1.0]]), rtol=1e-5, atol=1e-6)

    def test_mixed_value_mask_only_infers_unmasked_obs_and_skips_legal_actions(self, monkeypatch):
        """With use_root_value=False and partial value_mask, infer only active entries and match dense result."""
        buf = _make_buffer(num_unroll_steps=3, discount_factor=0.5, lstm_horizon_len=5, use_root_value=False)

        def _forbid_preprocess(*args, **kwargs):
            raise AssertionError('_preprocess_to_play_and_action_mask should be skipped when use_root_value=False')

        monkeypatch.setattr(buf, '_preprocess_to_play_and_action_mask', _forbid_preprocess)

        # 4 transitions with markers [10, 20, 30, 40]; only indices 0 and 2 are unmasked.
        ctx = _build_context(
            rewards=[0.0, 0.0, 0.0, 0.0, 0.0],
            to_play=[1, 2, 1, 2, 1],
            state_index=0,
            td_steps=2,  # even -> positive bootstrap sign
            value_mask=[1, 0, 1, 0],
            obs_markers=[0.2, 0.9, -0.4, 0.7],
            num_unroll_steps=3,
        )
        # Model returns obs[:, 0] directly so we verify exact index scattering.
        model = _RecordingModel(
            values_fn=lambda obs: obs[:, 0].detach().cpu().numpy().astype(np.float32),
            training=True,
        )

        _, target_values = buf._compute_target_reward_value(ctx, model)

        # Only 2 active entries should be passed to initial_inference (not 4).
        assert model.call_batch_sizes == [2]
        # discount^2 = 0.25 -> active indices [0, 2] get [0.2 * 0.25, -0.4 * 0.25] = [0.05, -0.10]
        np.testing.assert_allclose(target_values, np.array([[0.05, 0.0, -0.10, 0.0]]), rtol=1e-5, atol=1e-6)

    def test_eval_mode_value_transform_only_consumes_value_head(self):
        """On model.training=False with use_root_value=False, categorical value logits transform cleanly without extra attributes."""
        buf = _make_buffer(num_unroll_steps=1, discount_factor=0.8, lstm_horizon_len=5, use_root_value=False)
        ctx = _build_context(
            rewards=[0.0, 0.0, 0.0, 0.0],
            to_play=[1, 2, 1, 2],
            state_index=0,
            td_steps=1,  # odd -> negated bootstrap sign in board_games
            value_mask=[1, 0],
            num_unroll_steps=1,
        )
        # Support is [-1, 0, +1]; logits putting all mass on +1 -> scalar value = +1.0.
        def _cat_logits(obs):
            logits = torch.full((obs.shape[0], 3), -1e9, dtype=torch.float32)
            logits[:, 2] = 0.0
            return logits

        model = _RecordingModel(values_fn=_cat_logits, training=False)

        _, target_values = buf._compute_target_reward_value(ctx, model)

        expected_v = float(inverse_scalar_transform(_cat_logits(torch.zeros((1, 4))), buf.value_support).item())
        assert model.call_batch_sizes == [1]
        # odd td_steps=1 -> -0.8 * expected_v at index 0; masked 0.0 at index 1
        np.testing.assert_allclose(target_values, np.array([[-0.8 * expected_v, 0.0]]), rtol=1e-5, atol=1e-6)
