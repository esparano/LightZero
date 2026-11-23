# -*- coding: utf-8 -*-
"""
End-to-end integration test for Galcon + SampledEfficientZero (mcts_ctree=True)
+ SampledEfficientZeroGameBuffer (with X/Y symmetry augmentation and multi-segment
``pad_over``) + ``_forward_collect``, ``_forward_eval``, and ``_forward_learn``.
"""

import numpy as np
import pytest
import torch
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer
from lzero.mcts.buffer.game_segment import GameSegment
from lzero.policy.sampled_efficientzero import SampledEfficientZeroPolicy
from lzero.worker.muzero_collector import MuZeroCollector
from zoo.board_games.galcon.envs.galcon_env import GalconEnv


def _make_small_galcon_env_and_policy_cfg():
    # Small 4x4 grid (grid_max=50, grid_square_size=25 -> 4x4) for fast CPU integration testing
    env_cfg = GalconEnv.default_config()
    env_cfg.update(
        dict(
            battle_mode='self_play_mode',
            bot_action_type='random',
            num_planets=4,
            min_send_ships=1,
            send_ratio=0.5,
            tick_seconds=0.25,
            fleet_speed=40.0,
            max_episode_steps=12,
            map_seed=2,
            grid_square_size=25.0,
            grid_max_x=50.0,
            grid_max_y=50.0,
            home_distance_fraction=0.8,
            neutral_min_cost=0,
            neutral_max_cost=10,
            neutral_min_production=15,
            neutral_max_production=100,
            fleet_top_k=1,
            channel_last=False,
            scale=True,
        )
    )
    env = GalconEnv(env_cfg)
    obs0 = env.reset()
    C, H, W = obs0['observation'].shape
    A = int(obs0['action_mask'].shape[0])
    K = 4

    policy_cfg = SampledEfficientZeroPolicy.default_config()
    policy_cfg.update(
        dict(
            model=dict(
                model_type='conv',
                observation_shape=(C, H, W),
                image_channel=C,
                frame_stack_num=1,
                action_space_size=A,
                continuous_action_space=False,
                num_of_sampled_actions=K,
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
            mcts_ctree=True,
            use_augmentation=False,
            num_simulations=4,
            update_per_collect=1,
            batch_size=4,
            optim_type='AdamW',
            learning_rate=1e-3,
            weight_decay=1e-4,
            grad_clip_value=1.0,
            piecewise_decay_lr_scheduler=False,
            # Force multi-segment split (game_segment_length=4 < 10 plies)
            game_segment_length=4,
            num_unroll_steps=2,
            lstm_horizon_len=2,
            td_steps=2,
            discount_factor=1.0,
            reanalyze_ratio=0.0,
            replay_buffer_size=200,
            priority_prob_alpha=0.6,
            priority_prob_beta=0.4,
            use_priority=False,
            policy_loss_type='cross_entropy',
            use_root_value=False,
            mini_infer_size=64,
            sample_type='transition',
            target_model_update_freq=100,
            symmetric_augment_x=True,
            symmetric_augment_y=True,
            symmetric_augment_type='Galcon',
        )
    )
    return env_cfg, policy_cfg


@pytest.mark.unittest
def test_galcon_sez_ctree_multi_segment_symmetry_e2e():
    """End-to-end smoke test: GalconEnv -> SEZ ctree collect -> multi-segment pad_over
    -> symmetry-augmented buffer push (4x) -> sample -> _forward_learn -> _forward_eval.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    env_cfg, policy_cfg = _make_small_galcon_env_and_policy_cfg()

    policy = SampledEfficientZeroPolicy(policy_cfg)
    policy._init_collect()
    policy._init_learn()
    policy._init_eval()

    env = GalconEnv(env_cfg)
    obs_dict = env.reset()

    seg_cfg = EasyDict(
        dict(
            game_segment_length=policy_cfg.game_segment_length,
            num_unroll_steps=policy_cfg.num_unroll_steps,
            td_steps=policy_cfg.td_steps,
            discount_factor=policy_cfg.discount_factor,
            gray_scale=False,
            transform2string=False,
            sampled_algo=True,
            gumbel_algo=False,
            use_ture_chance_label_in_chance_encoder=False,
            model=dict(
                frame_stack_num=1,
                observation_shape=policy_cfg.model.observation_shape,
                image_channel=policy_cfg.model.image_channel,
                action_space_size=policy_cfg.model.action_space_size,
                continuous_action_space=False,
                num_of_sampled_actions=policy_cfg.model.num_of_sampled_actions,
            ),
        )
    )

    # Collect 2 consecutive segments (4 plies + 4 plies) from a real Galcon self-play game
    segments = []
    for _ in range(2):
        seg = GameSegment(action_space=None, game_segment_length=policy_cfg.game_segment_length, config=seg_cfg)
        seg.reset([obs_dict['observation']])
        for _step in range(policy_cfg.game_segment_length):
            obs_t = torch.from_numpy(obs_dict['observation']).float().unsqueeze(0)
            out = policy._forward_collect(
                obs_t,
                action_mask=[obs_dict['action_mask']],
                temperature=1.0,
                to_play=[obs_dict['to_play']],
            )[0]
            action = int(out['action'])
            assert obs_dict['action_mask'][action] == 1, f"Illegal Galcon action {action} selected!"

            ts = env.step(action)
            seg.append(action, ts.obs['observation'], float(ts.reward), obs_dict['action_mask'], obs_dict['to_play'])
            seg.store_search_stats(
                out['visit_count_distributions'],
                root_value=float(out['searched_value']),
                root_sampled_actions=out['root_sampled_actions'],
            )
            obs_dict = env.reset() if ts.done else ts.obs
        segments.append(seg)

    # Pad segment 0 with segment 1 via MuZeroCollector.pad_and_save_last_trajectory
    collector = object.__new__(MuZeroCollector)
    collector._end_flag = True
    collector.policy_config = seg_cfg
    collector.unroll_plus_td_steps = policy_cfg.num_unroll_steps + policy_cfg.td_steps
    collector.game_segment_pool = []
    collector.pad_and_save_last_trajectory(
        0, [segments[0]], [np.ones(4, dtype=np.float32)], [segments[1]], np.array([False])
    )
    padded_seg0 = collector.game_segment_pool[0][0]
    segments[1].valid_transition_count = 4
    segments[1].game_segment_to_array()

    # Push into symmetry-augmented buffer (2 segments -> 8 segments with X/Y reflection)
    buf = SampledEfficientZeroGameBuffer(policy_cfg)
    metas = [
        {'done': False, 'unroll_plus_td_steps': 4, 'priorities': np.ones(4, dtype=np.float32)},
        {'done': True, 'unroll_plus_td_steps': 4, 'priorities': np.ones(4, dtype=np.float32)},
    ]
    buf.push_game_segments(([padded_seg0, segments[1]], metas))
    assert buf.get_num_of_game_segments() == 8

    # Sample and run _forward_learn (exercises ctree 3D sampled_actions + cross-segment to_play)
    train_data = buf.sample(batch_size=policy_cfg.batch_size, policy=policy)
    log_vars = policy._forward_learn(train_data)
    for key in ['total_loss', 'policy_loss', 'value_loss', 'consistency_loss']:
        val = float(log_vars[key])
        assert np.isfinite(val), f"Non-finite {key}: {val}"

    # Verify _forward_eval in eval_mode produces legal actions without error
    eval_cfg = GalconEnv.default_config()
    eval_cfg.update(env_cfg)
    eval_cfg.battle_mode = 'eval_mode'
    eval_env = GalconEnv(eval_cfg)
    eval_obs = eval_env.reset()
    eval_out = policy._forward_eval(
        torch.from_numpy(eval_obs['observation']).float().unsqueeze(0),
        action_mask=[eval_obs['action_mask']],
        to_play=[eval_obs['to_play']],
    )[0]
    eval_action = int(eval_out['action'])
    assert eval_obs['action_mask'][eval_action] == 1
