import numpy as np
import pytest
import torch
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer
from lzero.mcts.buffer.game_segment import GameSegment
from lzero.policy.sampled_efficientzero import SampledEfficientZeroPolicy


def _make_test_policy_and_buffer_cfg(grid_h=4, grid_w=4, channels=8, num_sampled_actions=4):
    action_space_size = grid_h * grid_w * grid_h * grid_w + 1
    cfg = SampledEfficientZeroPolicy.default_config()

    custom_cfg = dict(
        model=dict(
            model_type='conv',
            observation_shape=(channels, grid_h, grid_w),
            image_channel=channels,
            frame_stack_num=1,
            action_space_size=action_space_size,
            continuous_action_space=False,
            num_of_sampled_actions=num_sampled_actions,
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
            reward_support_range=(-1.0, 2.0, 1.0),
            value_support_range=(-1.0, 2.0, 1.0),
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
    cfg.update(custom_cfg)
    return cfg


def _make_dummy_segment(cfg, num_steps=6):
    grid_h, grid_w = cfg.model.observation_shape[1], cfg.model.observation_shape[2]
    channels = cfg.model.observation_shape[0]
    K = cfg.model.num_of_sampled_actions
    action_space_size = cfg.model.action_space_size

    segment_cfg = EasyDict(dict(
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
            num_of_sampled_actions=K,
        ),
    ))
    segment = GameSegment(action_space=None, game_segment_length=cfg.game_segment_length, config=segment_cfg)

    for i in range(num_steps):
        obs = np.random.randn(channels, grid_h, grid_w).astype(np.float32)
        segment.obs_segment.append(obs)

        action = (i + 1) % action_space_size
        segment.action_segment.append(action)

        action_mask = np.ones(action_space_size, dtype=np.int8)
        segment.action_mask_segment.append(action_mask)

        sampled_actions = np.array([action if j == 0 else (j % action_space_size) for j in range(K)])
        segment.root_sampled_actions.append(sampled_actions)

        # Child visits distribution over K sampled actions
        visits = [10 if j == 0 else 2 for j in range(K)]
        segment.child_visit_segment.append(visits)

        segment.root_value_segment.append(0.1 * i)
        segment.reward_segment.append(0.0 if i < num_steps - 1 else 1.0)
        segment.to_play_segment.append(1 if i % 2 == 0 else 2)

    # Valid transition count metadata
    segment.valid_transition_count = num_steps
    return segment


@pytest.mark.unittest
class TestSampledEfficientZeroPolicyLearn:

    def test_forward_learn_single_step(self):
        cfg = _make_test_policy_and_buffer_cfg()
        policy = SampledEfficientZeroPolicy(cfg)
        policy._init_learn()

        buffer = SampledEfficientZeroGameBuffer(cfg)

        seg1 = _make_dummy_segment(cfg, num_steps=6)
        seg2 = _make_dummy_segment(cfg, num_steps=6)

        meta1 = {'done': True, 'unroll_plus_td_steps': 4, 'priorities': np.ones(6)}
        meta2 = {'done': True, 'unroll_plus_td_steps': 4, 'priorities': np.ones(6)}

        buffer.push_game_segments(([seg1, seg2], [meta1, meta2]))

        # Sample batch from buffer
        train_data = buffer.sample(batch_size=2, policy=policy)

        # Execute learning step
        log_vars = policy._forward_learn(train_data)

        assert isinstance(log_vars, dict)
        assert 'total_loss' in log_vars
        assert 'policy_loss' in log_vars
        assert 'value_loss' in log_vars
        assert 'consistency_loss' in log_vars

        # Ensure no NaN / Inf values in any loss
        for key in ['total_loss', 'policy_loss', 'value_loss', 'consistency_loss']:
            val = float(log_vars[key])
            assert not np.isnan(val), f"{key} is NaN!"
            assert not np.isinf(val), f"{key} is Inf!"
