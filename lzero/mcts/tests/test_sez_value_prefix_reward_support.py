# -*- coding: utf-8 -*-
"""
Regression tests for BUG-10: ``value_prefix`` must be decoded using
``reward_inverse_scalar_transform_handle`` (``reward_support_range``), NOT
``value_inverse_scalar_transform_handle`` (``value_support_range``), across:
1. ``SampledEfficientZeroMCTSCtree.search`` (``mcts_ctree_sampled.py``)
2. ``SampledEfficientZeroMCTSPtree.search`` (``mcts_ptree_sampled.py``)
3. ``SampledEfficientZeroPolicy._forward_learn`` (``sampled_efficientzero.py``)
4. Bit-for-bit equivalence when ``reward_support_range == value_support_range``.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer
from lzero.mcts.buffer.game_segment import GameSegment
from lzero.mcts.tree_search.mcts_ctree_sampled import SampledEfficientZeroMCTSCtree
from lzero.mcts.tree_search.mcts_ptree_sampled import SampledEfficientZeroMCTSPtree
from lzero.policy.sampled_efficientzero import SampledEfficientZeroPolicy


class _StubRecurrentOutput:
    def __init__(self, latent_state, policy_logits, value, value_prefix, reward_hidden_state):
        self.latent_state = latent_state
        self.policy_logits = policy_logits
        self.value = value
        self.value_prefix = value_prefix
        self.reward_hidden_state = reward_hidden_state


class _CategoricalStubModel(nn.Module):
    """Stub model returning fixed categorical logits for value and value_prefix."""

    def __init__(self, action_space_size, value_logits_row, value_prefix_logits_row, hidden_dim=4):
        super().__init__()
        self.action_space_size = action_space_size
        self.value_logits_row = list(value_logits_row)
        self.value_prefix_logits_row = list(value_prefix_logits_row)
        self.hidden_dim = hidden_dim

    def recurrent_inference(self, latent_states, reward_hidden_state, actions):
        batch_size = latent_states.shape[0]
        next_latents = latent_states.clone()
        policy_logits = torch.zeros((batch_size, self.action_space_size), dtype=torch.float32)
        value = torch.tensor([self.value_logits_row] * batch_size, dtype=torch.float32)
        value_prefix = torch.tensor([self.value_prefix_logits_row] * batch_size, dtype=torch.float32)
        c_out = torch.zeros((1, batch_size, self.hidden_dim), dtype=torch.float32)
        h_out = torch.zeros((1, batch_size, self.hidden_dim), dtype=torch.float32)
        return _StubRecurrentOutput(
            latent_state=next_latents,
            policy_logits=policy_logits,
            value=value,
            value_prefix=value_prefix,
            reward_hidden_state=(c_out, h_out),
        )


def _make_mcts_cfg(
    reward_support_range,
    value_support_range,
    action_space_size=1,
    num_of_sampled_actions=1,
    num_simulations=1,
    discount_factor=1.0,
):
    return EasyDict(
        dict(
            device='cpu',
            env_type='not_board_games',
            discount_factor=discount_factor,
            num_simulations=num_simulations,
            lstm_horizon_len=5,
            pb_c_base=19652,
            pb_c_init=1.25,
            value_delta_max=0.01,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.0,
            model=dict(
                continuous_action_space=False,
                action_space_size=action_space_size,
                num_of_sampled_actions=num_of_sampled_actions,
                categorical_distribution=True,
                reward_support_range=reward_support_range,
                value_support_range=value_support_range,
            ),
        )
    )


def _run_mcts_search(mcts_cls, cfg, model, legal_actions):
    roots = mcts_cls.roots(
        1,
        [legal_actions],
        cfg.model.action_space_size,
        cfg.model.num_of_sampled_actions,
        False,
    )
    root_policy_logits = [0.0] * cfg.model.action_space_size
    roots.prepare_no_noise([0.0], [root_policy_logits], [-1])

    latent_roots = np.zeros((1, 3), dtype=np.float32)
    reward_hidden_roots = (
        np.zeros((1, 1, model.hidden_dim), dtype=np.float32),
        np.zeros((1, 1, model.hidden_dim), dtype=np.float32),
    )

    mcts = mcts_cls(cfg)
    mcts.search(roots, model, latent_roots, reward_hidden_roots, to_play_batch=[-1])
    return mcts, roots


def _make_policy_cfg(reward_support_range, value_support_range, monitor_extra_statistics=True):
    grid_h, grid_w, channels = 2, 2, 4
    action_space_size = grid_h * grid_w * grid_h * grid_w + 1
    cfg = SampledEfficientZeroPolicy.default_config()
    cfg.update(
        dict(
            model=dict(
                model_type='conv',
                observation_shape=(channels, grid_h, grid_w),
                image_channel=channels,
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
                reward_support_range=reward_support_range,
                value_support_range=value_support_range,
                discrete_action_encoding_type='one_hot',
                norm_type='BN',
                embed_actions=True,
                embedded_action_dim=8,
            ),
            cuda=False,
            device='cpu',
            env_type='board_games',
            action_type='varied_action_space',
            battle_mode='self_play_mode',
            mcts_ctree=False,
            monitor_extra_statistics=monitor_extra_statistics,
            use_augmentation=False,
            update_per_collect=1,
            batch_size=2,
            optim_type='AdamW',
            learning_rate=0.001,
            weight_decay=1e-4,
            momentum=0.9,
            grad_clip_value=10,
            piecewise_decay_lr_scheduler=False,
            num_unroll_steps=2,
            td_steps=2,
            discount_factor=0.99,
            reanalyze_ratio=0.0,
            replay_buffer_size=100,
            priority_prob_alpha=0.6,
            priority_prob_beta=0.4,
            use_priority=False,
            game_segment_length=10,
            policy_entropy_weight=0.001,
            ssl_loss_weight=1.0,
            reanalyze_outdated=True,
            use_root_value=False,
            mini_infer_size=128,
            sample_type='transition',
            target_model_update_freq=100,
        )
    )
    return cfg


def _make_dummy_segment(cfg, num_steps=6):
    channels, grid_h, grid_w = cfg.model.observation_shape
    k_sampled = cfg.model.num_of_sampled_actions
    action_space_size = cfg.model.action_space_size

    segment_cfg = EasyDict(
        dict(
            num_unroll_steps=cfg.num_unroll_steps,
            td_steps=cfg.td_steps,
            discount_factor=cfg.discount_factor,
            gray_scale=False,
            transform2string=False,
            sampled_algo=True,
            gumbel_algo=False,
            use_ture_chance_label_in_chance_encoder=False,
            model=dict(
                frame_stack_num=1,
                observation_shape=cfg.model.observation_shape,
                image_channel=channels,
                action_space_size=action_space_size,
                continuous_action_space=False,
                num_of_sampled_actions=k_sampled,
            ),
        )
    )
    segment = GameSegment(action_space=None, game_segment_length=cfg.game_segment_length, config=segment_cfg)

    for i in range(num_steps):
        segment.obs_segment.append(np.random.randn(channels, grid_h, grid_w).astype(np.float32))
        action = (i + 1) % action_space_size
        segment.action_segment.append(action)
        segment.action_mask_segment.append(np.ones(action_space_size, dtype=np.int8))
        sampled_actions = np.array([action if j == 0 else (j % action_space_size) for j in range(k_sampled)])
        segment.root_sampled_actions.append(sampled_actions)
        segment.child_visit_segment.append([10 if j == 0 else 2 for j in range(k_sampled)])
        segment.root_value_segment.append(0.1 * i)
        segment.reward_segment.append(0.0 if i < num_steps - 1 else 1.0)
        segment.to_play_segment.append(1 if i % 2 == 0 else 2)

    segment.valid_transition_count = num_steps
    return segment


@pytest.mark.unittest
@pytest.mark.parametrize(
    'mcts_cls',
    [SampledEfficientZeroMCTSCtree, SampledEfficientZeroMCTSPtree],
    ids=['ctree', 'ptree'],
)
class TestSEZMCTSValuePrefixRewardSupport:

    def test_search_asymmetric_support_sizes_decodes_value_prefix_with_reward_support(self, mcts_cls):
        """With reward_support size 3 (-1, 0, 1) and value_support size 11 (-5..5),
        search must neither crash with a broadcast RuntimeError nor mis-decode value_prefix.
        """
        # Value logits peaked at atom 0.0 (index 5 of 11); value_prefix peaked at atom +1.0 (index 2 of 3).
        value_logits = [-20.0] * 11
        value_logits[5] = 20.0
        vp_logits = [-20.0, -20.0, 20.0]

        cfg = _make_mcts_cfg(
            reward_support_range=(-1.0, 2.0, 1.0),
            value_support_range=(-5.0, 6.0, 1.0),
            action_space_size=1,
            num_of_sampled_actions=1,
            num_simulations=1,
            discount_factor=1.0,
        )
        model = _CategoricalStubModel(
            action_space_size=1,
            value_logits_row=value_logits,
            value_prefix_logits_row=vp_logits,
        )
        mcts, roots = _run_mcts_search(mcts_cls, cfg, model, legal_actions=[0])

        expected_vp = mcts.reward_inverse_scalar_transform_handle(
            torch.tensor([vp_logits], dtype=torch.float32)
        ).item()
        child_q = float(roots.get_children_q_values(1.0)[0][0])
        root_val = float(roots.get_values()[0])
        assert np.isclose(child_q, expected_vp, atol=1e-4), (
            f"Expected child Q {expected_vp} (from reward_support), got {child_q}"
        )
        # prepare_no_noise initializes root.visit_count=1 with value_sum=0, so after 1 sim root.value == expected_vp / 2.
        assert np.isclose(root_val, expected_vp / 2.0, atol=1e-4), (
            f"Expected root value {expected_vp / 2.0} (from reward_support), got {root_val}"
        )

        # Also verify multi-step expansion with asymmetric supports completes cleanly.
        cfg_multi = _make_mcts_cfg(
            reward_support_range=(-1.0, 2.0, 1.0),
            value_support_range=(-5.0, 6.0, 1.0),
            action_space_size=2,
            num_of_sampled_actions=2,
            num_simulations=4,
            discount_factor=0.99,
        )
        model_multi = _CategoricalStubModel(
            action_space_size=2,
            value_logits_row=value_logits,
            value_prefix_logits_row=vp_logits,
        )
        _, roots_multi = _run_mcts_search(mcts_cls, cfg_multi, model_multi, legal_actions=[0, 1])
        assert sum(roots_multi.get_distributions()[0]) == 4

    def test_search_same_size_different_range_uses_reward_support(self, mcts_cls):
        """With equal support sizes (3) but distinct ranges — reward [-2, 0, 2] vs
        value [-10, 0, 10] — using value_inverse_scalar_transform_handle silently
        mis-scales value_prefix (~119.5 instead of ~7.97). Verify reward_support is used.
        """
        value_logits = [-20.0, 20.0, -20.0]  # peaked at 0.0 -> h^-1(0.0) = 0.0
        vp_logits = [-20.0, -20.0, 20.0]     # peaked at index 2 (+2.0 reward vs +10.0 value)

        cfg = _make_mcts_cfg(
            reward_support_range=(-2.0, 3.0, 2.0),
            value_support_range=(-10.0, 11.0, 10.0),
            action_space_size=1,
            num_of_sampled_actions=1,
            num_simulations=1,
            discount_factor=1.0,
        )
        model = _CategoricalStubModel(
            action_space_size=1,
            value_logits_row=value_logits,
            value_prefix_logits_row=vp_logits,
        )
        mcts, roots = _run_mcts_search(mcts_cls, cfg, model, legal_actions=[0])

        vp_tensor = torch.tensor([vp_logits], dtype=torch.float32)
        expected_reward_vp = mcts.reward_inverse_scalar_transform_handle(vp_tensor).item()
        wrong_value_vp = mcts.value_inverse_scalar_transform_handle(vp_tensor).item()
        child_q = float(roots.get_children_q_values(1.0)[0][0])
        root_val = float(roots.get_values()[0])

        assert abs(expected_reward_vp - wrong_value_vp) > 50.0
        assert np.isclose(child_q, expected_reward_vp, atol=1e-4), (
            f"Expected value_prefix decoded via reward_support ({expected_reward_vp}), "
            f"got {child_q} (value_support would give {wrong_value_vp})"
        )
        assert np.isclose(root_val, expected_reward_vp / 2.0, atol=1e-4), (
            f"Expected root value {expected_reward_vp / 2.0} (reward_support), "
            f"got {root_val} (value_support would give {wrong_value_vp / 2.0})"
        )


@pytest.mark.unittest
class TestSEZPolicyForwardLearnValuePrefixRewardSupport:

    def test_forward_learn_asymmetric_support_sizes_monitor_extra_statistics(self):
        """With monitor_extra_statistics=True and reward_support size 3 vs value_support size 11,
        _forward_learn must complete without RuntimeError and decode predicted_value_prefixs
        using reward_inverse_scalar_transform_handle.
        """
        torch.manual_seed(0)
        np.random.seed(0)
        cfg = _make_policy_cfg(
            reward_support_range=(-1.0, 2.0, 1.0),
            value_support_range=(-5.0, 6.0, 1.0),
            monitor_extra_statistics=True,
        )
        policy = SampledEfficientZeroPolicy(cfg)
        policy._init_learn()

        buffer = SampledEfficientZeroGameBuffer(cfg)
        seg1 = _make_dummy_segment(cfg, num_steps=6)
        seg2 = _make_dummy_segment(cfg, num_steps=6)
        meta = {'done': True, 'unroll_plus_td_steps': 4, 'priorities': np.ones(6)}
        buffer.push_game_segments(([seg1, seg2], [meta, meta]))
        train_data = buffer.sample(batch_size=2, policy=policy)

        recorded_vp = []
        recorded_val = []
        orig_reward_handle = policy.reward_inverse_scalar_transform_handle
        orig_value_handle = policy.value_inverse_scalar_transform_handle

        def _spy_reward(tensor):
            out = orig_reward_handle(tensor)
            recorded_vp.append(out.detach().clone())
            return out

        def _spy_value(tensor):
            out = orig_value_handle(tensor)
            recorded_val.append(out.detach().clone())
            return out

        policy.reward_inverse_scalar_transform_handle = _spy_reward
        policy.value_inverse_scalar_transform_handle = _spy_value

        log_vars = policy._forward_learn(train_data)

        unroll_steps = cfg.num_unroll_steps
        assert len(recorded_vp) == unroll_steps, (
            f"Expected {unroll_steps} calls to reward_inverse_scalar_transform_handle, got {len(recorded_vp)}"
        )
        assert len(recorded_val) == 1 + unroll_steps, (
            f"Expected {1 + unroll_steps} calls to value_inverse_scalar_transform_handle, got {len(recorded_val)}"
        )

        expected_vp_mean = torch.cat(recorded_vp, dim=0).mean().item()
        expected_val_mean = torch.cat(recorded_val, dim=0).mean().item()
        assert np.isfinite(log_vars['total_loss'])
        assert np.isfinite(log_vars['value_prefix_loss'])
        assert np.isclose(log_vars['predicted_value_prefixs'], expected_vp_mean, atol=1e-6)
        assert np.isclose(log_vars['predicted_values'], expected_val_mean, atol=1e-6)

    def test_forward_learn_same_size_different_range_uses_reward_support(self):
        """With equal support sizes (3) but distinct ranges — reward [-2, 0, 2] vs
        value [-10, 0, 10] — verify predicted_value_prefixs in _forward_learn uses
        reward_support rather than silently decoding on value_support.
        """
        torch.manual_seed(1)
        np.random.seed(1)
        cfg = _make_policy_cfg(
            reward_support_range=(-2.0, 3.0, 2.0),
            value_support_range=(-10.0, 11.0, 10.0),
            monitor_extra_statistics=True,
        )
        policy = SampledEfficientZeroPolicy(cfg)
        policy._init_learn()

        # Force dynamics reward head output logits to [-20, -20, 20] (peaked at atom index 2).
        reward_linears = [
            m for m in policy._learn_model.dynamics_network.fc_reward_head.modules()
            if isinstance(m, nn.Linear)
        ]
        assert len(reward_linears) > 0
        last_reward_linear = reward_linears[-1]
        with torch.no_grad():
            last_reward_linear.weight.zero_()
            last_reward_linear.bias.copy_(torch.tensor([-20.0, -20.0, 20.0]))

        buffer = SampledEfficientZeroGameBuffer(cfg)
        seg1 = _make_dummy_segment(cfg, num_steps=6)
        seg2 = _make_dummy_segment(cfg, num_steps=6)
        meta = {'done': True, 'unroll_plus_td_steps': 4, 'priorities': np.ones(6)}
        buffer.push_game_segments(([seg1, seg2], [meta, meta]))
        train_data = buffer.sample(batch_size=2, policy=policy)

        log_vars = policy._forward_learn(train_data)

        vp_logits = torch.tensor([[-20.0, -20.0, 20.0]], dtype=torch.float32)
        expected_reward_vp = policy.reward_inverse_scalar_transform_handle(vp_logits).item()
        wrong_value_vp = policy.value_inverse_scalar_transform_handle(vp_logits).item()

        assert abs(expected_reward_vp - wrong_value_vp) > 50.0
        assert np.isclose(log_vars['predicted_value_prefixs'], expected_reward_vp, atol=1e-3), (
            f"Expected predicted_value_prefixs={expected_reward_vp} (reward_support), "
            f"got {log_vars['predicted_value_prefixs']} (value_support would give {wrong_value_vp})"
        )


@pytest.mark.unittest
def test_equal_reward_and_value_supports_produce_bit_identical_outputs():
    """When reward_support_range == value_support_range (e.g. Galcon (-1., 2., 1.)),
    reward_inverse_scalar_transform_handle and value_inverse_scalar_transform_handle
    must produce bit-for-bit identical outputs across MCTS ctree, MCTS ptree, and Policy.
    """
    support_range = (-1.0, 2.0, 1.0)
    mcts_cfg = _make_mcts_cfg(
        reward_support_range=support_range,
        value_support_range=support_range,
    )
    ctree = SampledEfficientZeroMCTSCtree(mcts_cfg)
    ptree = SampledEfficientZeroMCTSPtree(mcts_cfg)

    policy_cfg = _make_policy_cfg(
        reward_support_range=support_range,
        value_support_range=support_range,
        monitor_extra_statistics=True,
    )
    policy = SampledEfficientZeroPolicy(policy_cfg)
    policy._init_learn()

    torch.manual_seed(123)
    logits_2d = torch.randn(8, 3, dtype=torch.float32)
    logits_1d = torch.randn(3, dtype=torch.float32)

    for obj in (ctree, ptree, policy):
        assert torch.equal(obj.reward_support.arange, obj.value_support.arange)

        out_r_2d = obj.reward_inverse_scalar_transform_handle(logits_2d.clone())
        out_v_2d = obj.value_inverse_scalar_transform_handle(logits_2d.clone())
        assert torch.equal(out_r_2d, out_v_2d)

        out_r_1d = obj.reward_inverse_scalar_transform_handle(logits_1d.clone())
        out_v_1d = obj.value_inverse_scalar_transform_handle(logits_1d.clone())
        assert torch.equal(out_r_1d, out_v_1d)
