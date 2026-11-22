import numpy as np
import pytest
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer
from lzero.mcts.buffer.game_segment import GameSegment


def _create_mock_galcon_segment(grid_h=4, grid_w=4, channels=60, num_steps=5):
    action_space_size = grid_w * grid_h * grid_w * grid_h + 1
    config = EasyDict(dict(
        num_unroll_steps=2,
        td_steps=2,
        discount_factor=0.99,
        gray_scale=False,
        transform2string=False,
        sampled_algo=True,
        gumbel_algo=False,
        use_ture_chance_label_in_chance_encoder=False,
        model=dict(
            frame_stack_num=1,
            observation_shape=(channels, grid_h, grid_w),
            image_channel=channels,
            action_space_size=action_space_size,
            continuous_action_space=False,
            num_of_sampled_actions=4,
        ),
    ))
    segment = GameSegment(action_space=None, game_segment_length=10, config=config)

    for _ in range(num_steps):
        obs = np.zeros((channels, grid_h, grid_w), dtype=np.float32)
        segment.obs_segment.append(obs)
        segment.action_segment.append(1)

        action_mask = np.zeros(action_space_size, dtype=np.int8)
        action_mask[0] = 1
        action_mask[1] = 1
        segment.action_mask_segment.append(action_mask)

        segment.root_sampled_actions.append(np.array([0, 1, 0, 0]))
        segment.child_visit_segment.append([5, 15, 0, 0])
        segment.root_value_segment.append(0.5)
        segment.reward_segment.append(0.0)

    return segment, config


def _make_buffer(sym_x: bool, sym_y: bool):
    grid_w, grid_h = 4, 4
    action_space_size = grid_w * grid_h * grid_w * grid_h + 1
    buffer_cfg = EasyDict(dict(
        batch_size=2,
        replay_buffer_size=100,
        env_type='board_games',
        action_type='varied_action_space',
        use_priority=False,
        priority_prob_alpha=0.6,
        priority_prob_beta=0.4,
        game_segment_length=10,
        num_unroll_steps=2,
        reanalyze_ratio=0.0,
        device='cpu',
        symmetric_augment_x=sym_x,
        symmetric_augment_y=sym_y,
        symmetric_augment_type='Galcon',
        model=dict(
            action_space_size=action_space_size,
            continuous_action_space=False,
            num_of_sampled_actions=4,
            value_support_range=(-10, 10, 1),
            reward_support_range=(-10, 10, 1),
        ),
    ))
    return SampledEfficientZeroGameBuffer(buffer_cfg)


@pytest.mark.unittest
class TestGalconBufferSymmetry:

    @pytest.mark.parametrize('sym_x,sym_y,expected_count', [
        (False, False, 1),  # no augmentation -> 1 segment
        (True, False, 2),   # X-only reflection -> 2 segments (orig + X)
        (False, True, 2),   # Y-only reflection -> 2 segments (orig + Y)
        (True, True, 4),    # X and Y reflection -> 4 segments (orig + X + Y + XY)
    ])
    def test_game_buffer_push_populates_expected_segment_count(self, sym_x, sym_y, expected_count):
        buffer = _make_buffer(sym_x=sym_x, sym_y=sym_y)
        segment, _ = _create_mock_galcon_segment()
        meta = {'done': True, 'unroll_plus_td_steps': 4, 'priorities': np.ones(5)}

        buffer.push_game_segments(([segment], [meta]))

        assert len(buffer.game_segment_buffer) == expected_count
