# -*- coding: utf-8 -*-
"""
Unit tests for multi-segment ``GameSegment.pad_over()`` and collector
``pad_and_save_last_trajectory()`` across board-game segment boundaries.

Verifies that when a two-player episode exceeds ``game_segment_length`` and
splits across two ``GameSegment`` objects:
1. ``to_play_segment`` is padded alongside ``reward_segment`` so
   ``_compute_target_reward_value`` never raises ``IndexError`` and applies
   exact two-player sign flips across the boundary.
2. ``root_sampled_actions`` is padded alongside ``child_visit_segment`` so
   ``SampledEfficientZeroGameBuffer`` pairs next-segment visit counts with the
   real next-segment sampled actions (rather than random dummy actions).
3. ``action_mask_segment`` is padded alongside ``obs_segment`` for reanalyze
   and board-game legal-action masking across the boundary.
"""

import numpy as np
import pytest
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer
from lzero.mcts.buffer.game_segment import GameSegment
from lzero.worker.muzero_collector import MuZeroCollector
from lzero.worker.muzero_segment_collector import MuZeroSegmentCollector


def _make_segment_cfg(game_segment_length=4, num_unroll_steps=2, td_steps=2, K=3, action_space_size=5):
    return EasyDict(
        dict(
            game_segment_length=game_segment_length,
            num_unroll_steps=num_unroll_steps,
            td_steps=td_steps,
            discount_factor=1.0,
            gray_scale=False,
            transform2string=False,
            sampled_algo=True,
            gumbel_algo=False,
            use_ture_chance_label_in_chance_encoder=False,
            model=dict(
                frame_stack_num=1,
                observation_shape=(1, 4),
                image_channel=1,
                action_space_size=action_space_size,
                continuous_action_space=False,
                num_of_sampled_actions=K,
            ),
        )
    )


def _populate_segment(cfg, start_step, length, rewards, to_play_vals, sampled_base=0):
    seg = GameSegment(action_space=None, game_segment_length=cfg.game_segment_length, config=cfg)
    K = cfg.model.num_of_sampled_actions
    A = cfg.model.action_space_size
    init_obs = [np.full((1, 4), float(start_step), dtype=np.float32)]
    seg.reset(init_obs)
    for idx in range(length):
        step = start_step + idx
        action = step % A
        next_obs = np.full((1, 4), float(step + 1), dtype=np.float32)
        mask = np.zeros(A, dtype=np.int8)
        mask[action] = 1
        mask[(action + 1) % A] = 1
        seg.append(action, next_obs, rewards[idx], mask, to_play_vals[idx])
        sampled = np.array([(sampled_base + idx + j) % A for j in range(K)], dtype=np.int64)
        visits = [10 if j == 0 else 1 for j in range(K)]
        seg.store_search_stats(visits, root_value=0.1 * step, root_sampled_actions=sampled)
    return seg


@pytest.mark.unittest
class TestGameSegmentPadOver:

    def test_pad_over_extends_to_play_action_mask_and_root_sampled_actions(self):
        cfg = _make_segment_cfg(game_segment_length=4, num_unroll_steps=2, td_steps=2, K=3, action_space_size=5)
        seg1 = _populate_segment(cfg, start_step=0, length=4, rewards=[0.0, 0.0, 0.0, 0.0], to_play_vals=[1, 2, 1, 2], sampled_base=0)
        seg2 = _populate_segment(cfg, start_step=4, length=4, rewards=[0.5, -0.25, 1.0, 0.0], to_play_vals=[1, 2, 1, 2], sampled_base=2)

        unroll_plus_td = cfg.num_unroll_steps + cfg.td_steps
        seg1.pad_over(
            seg2.obs_segment[1:1 + unroll_plus_td],
            seg2.reward_segment[:unroll_plus_td - 1],
            seg2.action_segment[:unroll_plus_td],
            seg2.root_value_segment[:unroll_plus_td],
            seg2.child_visit_segment[:unroll_plus_td],
            next_segment_to_play=seg2.to_play_segment[:unroll_plus_td],
            next_segment_action_masks=seg2.action_mask_segment[:unroll_plus_td],
            next_segment_root_sampled_actions=seg2.root_sampled_actions[:unroll_plus_td],
        )

        assert len(seg1.to_play_segment) == 4 + unroll_plus_td
        assert seg1.to_play_segment[4:] == seg2.to_play_segment[:unroll_plus_td]
        assert len(seg1.action_mask_segment) == 4 + unroll_plus_td
        assert len(seg1.root_sampled_actions) == 4 + unroll_plus_td
        for a, b in zip(seg1.root_sampled_actions[4:], seg2.root_sampled_actions[:unroll_plus_td]):
            np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize('collector_cls', [MuZeroCollector, MuZeroSegmentCollector])
    def test_collector_pad_and_save_last_trajectory_pads_cross_segment_fields(self, collector_cls):
        cfg = _make_segment_cfg(game_segment_length=4, num_unroll_steps=2, td_steps=2, K=3, action_space_size=5)
        seg1 = _populate_segment(cfg, start_step=0, length=4, rewards=[0.0, 0.0, 0.0, 0.0], to_play_vals=[1, 2, 1, 2], sampled_base=0)
        seg2 = _populate_segment(cfg, start_step=4, length=4, rewards=[0.5, 0.0, 1.0, 0.0], to_play_vals=[1, 2, 1, 2], sampled_base=3)

        collector = object.__new__(collector_cls)
        collector._end_flag = True
        collector.policy_config = cfg
        collector.unroll_plus_td_steps = cfg.num_unroll_steps + cfg.td_steps
        collector.game_segment_pool = []

        last_game_segments = [seg1]
        last_game_priorities = [np.ones(4, dtype=np.float32)]
        game_segments = [seg2]
        done = np.array([False])

        collector.pad_and_save_last_trajectory(0, last_game_segments, last_game_priorities, game_segments, done)

        assert len(collector.game_segment_pool) == 1
        saved_seg = collector.game_segment_pool[0][0]
        assert saved_seg.valid_transition_count == 4
        assert len(saved_seg.to_play_segment) == 8
        assert list(saved_seg.to_play_segment) == [1, 2, 1, 2, 1, 2, 1, 2]
        assert len(saved_seg.action_mask_segment) == 8
        assert len(saved_seg.root_sampled_actions) == 8
        np.testing.assert_array_equal(saved_seg.root_sampled_actions[4], seg2.root_sampled_actions[0])

    def test_cross_segment_value_target_sign_flip_and_sampled_actions_alignment(self):
        """At the last transition (state_index=3, to_play=2) of seg1, bootstrapping
        td_steps=2 reaches into seg2's padded rewards (index 4, to_play=1, reward=+0.5).
        Must not raise IndexError and must negate opponent's reward to -0.5.
        """
        cfg = _make_segment_cfg(game_segment_length=4, num_unroll_steps=2, td_steps=2, K=3, action_space_size=5)
        seg1 = _populate_segment(cfg, start_step=0, length=4, rewards=[0.0, 0.0, 0.0, 0.2], to_play_vals=[1, 2, 1, 2], sampled_base=0)
        seg2 = _populate_segment(cfg, start_step=4, length=4, rewards=[0.5, 0.0, 0.0, 0.0], to_play_vals=[1, 2, 1, 2], sampled_base=3)

        collector = object.__new__(MuZeroCollector)
        collector._end_flag = True
        collector.policy_config = cfg
        collector.unroll_plus_td_steps = cfg.num_unroll_steps + cfg.td_steps
        collector.game_segment_pool = []
        collector.pad_and_save_last_trajectory(
            0, [seg1], [np.ones(4, dtype=np.float32)], [seg2], np.array([False])
        )
        padded_seg = collector.game_segment_pool[0][0]

        class _ZeroModel:
            training = True

            def initial_inference(self, m_obs):
                return EasyDict(value=np.zeros(int(m_obs.shape[0]), dtype=np.float32))

        buf = object.__new__(SampledEfficientZeroGameBuffer)
        buf._cfg = EasyDict(
            dict(
                env_type='board_games',
                device='cpu',
                mini_infer_size=16,
                use_root_value=False,
                discount_factor=1.0,
                lstm_horizon_len=100,
                num_unroll_steps=2,
                model=dict(model_type='mlp', continuous_action_space=False, action_space_size=5),
            )
        )

        # Build reward_value_context starting at state_index=3 (the boundary of seg1)
        n_transitions = 3  # num_unroll_steps + 1
        ctx = [
            [np.zeros((1, 4), dtype=np.float32) for _ in range(n_transitions)],
            [1.0] * n_transitions,
            [3],  # pos_in_game_segment = 3 (last real step of seg1)
            [padded_seg.reward_segment],
            [padded_seg.valid_transition_count],
            np.full(n_transitions, 2, dtype=np.int32),
            [padded_seg.action_mask_segment],
            [padded_seg.to_play_segment],
        ]
        _, target_values = buf._compute_target_reward_value(ctx, _ZeroModel())
        target_values = np.asarray(target_values, dtype=np.float32)

        # At unroll step 0 (current_index=3, to_play=2):
        #   i=0 -> index 3 (to_play=2 == 2): +0.2
        #   i=1 -> index 4 (to_play=1 != 2): -0.5  (from seg2!)
        #   total = 0.2 - 0.5 = -0.3
        np.testing.assert_allclose(target_values[0, 0], -0.3, atol=1e-5)

        # Verify unroll slice of root_sampled_actions across boundary [3, 4, 5] matches seg2[0], seg2[1]
        sliced_sampled = padded_seg.root_sampled_actions[3:6]
        assert len(sliced_sampled) == 3
        np.testing.assert_array_equal(sliced_sampled[1], seg2.root_sampled_actions[0])
        np.testing.assert_array_equal(sliced_sampled[2], seg2.root_sampled_actions[1])
