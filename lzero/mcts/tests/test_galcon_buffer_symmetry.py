import numpy as np
import pytest
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer import reflect_game_segment
from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer
from lzero.mcts.buffer.game_segment import GameSegment


def _create_mock_galcon_segment(grid_h=4, grid_w=4, channels=60, num_steps=3):
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

    for step in range(num_steps):
        obs = np.zeros((channels, grid_h, grid_w), dtype=np.float32)
        # Place a friendly planet at (x=1, y=2)
        obs[0, 2, 1] = 0.25  # local_x
        obs[1, 2, 1] = 0.75  # local_y
        obs[2, 2, 1] = 100.0  # friendly ships
        obs[5, 2, 1] = 50.0   # friendly production

        # Place a friendly fleet at (x=0, y=1) pointing to (0.2, 0.8)
        obs[8, 1, 0] = 0.2   # target_x
        obs[9, 1, 0] = 0.8   # target_y
        obs[11, 1, 0] = 20.0 # fleet ships

        segment.obs_segment.append(obs)

        # Action: source (0, 1) -> cell 1*4+0 = 4; target (1, 2) -> cell 2*4+1 = 9
        action = 4 * (grid_h * grid_w) + 9 + 1
        segment.action_segment.append(action)

        action_mask = np.zeros(action_space_size, dtype=np.int8)
        action_mask[0] = 1
        action_mask[action] = 1
        segment.action_mask_segment.append(action_mask)

        # 4 sampled actions: pass (0), real action, and two dummies
        segment.root_sampled_actions.append(np.array([0, action, 0, 0]))
        segment.child_visit_segment.append([5, 15, 0, 0])
        segment.root_value_segment.append(0.5)
        segment.reward_segment.append(0.0)

    return segment, config


@pytest.mark.unittest
class TestGalconBufferSymmetry:

    def test_reflect_game_segment_x_flip(self):
        segment, _ = _create_mock_galcon_segment(grid_h=4, grid_w=4)
        policy_cfg = EasyDict(dict(symmetric_augment_type='Galcon'))

        reflected = reflect_game_segment(policy_cfg, segment, reflect_x=True, reflect_y=False)

        assert len(reflected.obs_segment) == len(segment.obs_segment)
        ref_obs = reflected.obs_segment[0]

        # Planet originally at (x=1, y=2) -> X-flip: x' = 4 - 1 - 1 = 2, y' = 2
        assert ref_obs[2, 2, 2] == 100.0  # friendly ships
        assert np.isclose(ref_obs[0, 2, 2], 1.0 - 0.25)  # local_x inverted
        assert np.isclose(ref_obs[1, 2, 2], 0.75)        # local_y unchanged

        # Fleet originally at (x=0, y=1) -> X-flip: x' = 4 - 1 - 0 = 3, y' = 1
        assert ref_obs[11, 1, 3] == 20.0  # fleet ships
        assert np.isclose(ref_obs[8, 1, 3], 1.0 - 0.2)  # target_x inverted
        assert np.isclose(ref_obs[9, 1, 3], 0.8)        # target_y unchanged

        # Action: source (3, 1) -> cell 1*4+3 = 7; target (2, 2) -> cell 2*4+2 = 10
        expected_action = 7 * 16 + 10 + 1
        assert reflected.action_segment[0] == expected_action
        assert reflected.action_mask_segment[0][expected_action] == 1
        assert reflected.action_mask_segment[0][0] == 1  # pass stays legal
        assert reflected.root_sampled_actions[0][1] == expected_action

    def test_reflect_game_segment_xy_flip(self):
        segment, _ = _create_mock_galcon_segment(grid_h=4, grid_w=4)
        policy_cfg = EasyDict(dict(symmetric_augment_type='Galcon'))

        reflected = reflect_game_segment(policy_cfg, segment, reflect_x=True, reflect_y=True)
        ref_obs = reflected.obs_segment[0]

        # Planet originally at (x=1, y=2) -> XY-flip: x' = 2, y' = 4 - 1 - 2 = 1
        assert ref_obs[2, 1, 2] == 100.0
        assert np.isclose(ref_obs[0, 1, 2], 1.0 - 0.25)
        assert np.isclose(ref_obs[1, 1, 2], 1.0 - 0.75)

        # Fleet originally at (x=0, y=1) -> XY-flip: x' = 3, y' = 4 - 1 - 1 = 2
        assert ref_obs[11, 2, 3] == 20.0
        assert np.isclose(ref_obs[8, 2, 3], 1.0 - 0.2)
        assert np.isclose(ref_obs[9, 2, 3], 1.0 - 0.8)

        # Action: source (3, 2) -> cell 2*4+3 = 11; target (2, 1) -> cell 1*4+2 = 6
        expected_action = 11 * 16 + 6 + 1
        assert reflected.action_segment[0] == expected_action
        assert reflected.action_mask_segment[0][expected_action] == 1
        assert reflected.root_sampled_actions[0][1] == expected_action

    def test_game_buffer_push_with_symmetry(self):
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
            symmetric_augment_x=True,
            symmetric_augment_y=True,
            symmetric_augment_type='Galcon',
            model=dict(
                action_space_size=action_space_size,
                continuous_action_space=False,
                num_of_sampled_actions=4,
                value_support_range=(-10, 10, 1),
                reward_support_range=(-10, 10, 1),
            ),
        ))

        buffer = SampledEfficientZeroGameBuffer(buffer_cfg)
        segment, _ = _create_mock_galcon_segment(grid_h=4, grid_w=4, num_steps=5)
        meta = {'done': True, 'unroll_plus_td_steps': 4, 'priorities': np.ones(5)}

        buffer.push_game_segments(([segment], [meta]))

        # 1 original segment pushed with X and Y symmetry should yield 4 segments in buffer
        assert len(buffer.game_segment_buffer) == 4
