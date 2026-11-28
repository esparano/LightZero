import numpy as np
import pytest
from easydict import EasyDict

from ding.torch_utils import to_list
from lzero.mcts.buffer.game_buffer_efficientzero import EfficientZeroGameBuffer
from lzero.mcts.buffer.game_buffer_muzero import MuZeroGameBuffer
from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer
from lzero.mcts.buffer.game_buffer_sampled_muzero import SampledMuZeroGameBuffer

config = EasyDict(
    dict(
        batch_size=10,
        transition_num=20,
        priority_prob_alpha=0.6,
        priority_prob_beta=0.4,
        replay_buffer_size=10000,
        env_type='not_board_games',
        use_priority=True,
        action_type='fixed_action_space',
        game_segment_length=20,
        model=dict(
            action_space_size=6,
            value_support_range=(-10, 10, 1),
            reward_support_range=(-10, 10, 1),
        ),
    )
)


def _make_varied_action_config(action_space_size=100, num_of_sampled_actions=3):
    return EasyDict(
        dict(
            batch_size=2,
            priority_prob_alpha=0.6,
            priority_prob_beta=0.4,
            replay_buffer_size=100,
            env_type='board_games',
            use_priority=False,
            action_type='varied_action_space',
            game_segment_length=10,
            num_unroll_steps=0,
            model=dict(
                action_space_size=action_space_size,
                continuous_action_space=False,
                num_of_sampled_actions=num_of_sampled_actions,
                value_support_range=(-10, 10, 1),
                reward_support_range=(-10, 10, 1),
            ),
        )
    )


def _make_sparse_policy_context(action_space_size=100):
    action_mask = np.zeros(action_space_size, dtype=np.int8)
    action_mask[[7, 42, 99]] = 1
    return [
        [0],
        [[[0.2, 0.3, 0.5]]],
        [1],
        [[action_mask]],
        [[-1]],
    ]


@pytest.mark.unittest
def test_push():
    buffer = EfficientZeroGameBuffer(config)
    # fake data
    data = [[1, 1, 1] for _ in range(10)]  # (s,a,r)
    meta = {'done': True, 'unroll_plus_td_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    # _push_game_segment
    for i in range(20):
        buffer._push_game_segment(to_list(np.multiply(i, data)), meta)
    assert buffer.get_num_of_game_segments() == 20

    # push_game_segments
    buffer.push_game_segments([[data, data], [meta, meta]])
    assert buffer.get_num_of_game_segments() == 22

    # Clear
    del buffer.game_segment_buffer[:]
    assert buffer.get_num_of_game_segments() == 0

    # _push_game_segment
    for i in range(5):
        buffer._push_game_segment(to_list(np.multiply(i, data)), meta)


@pytest.mark.unittest
def test_update_priority():
    buffer = EfficientZeroGameBuffer(config)
    # fake data
    data = [[1, 1, 1] for _ in range(10)]  # (s,a,r)
    meta = {'done': True, 'unroll_plus_td_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    # _push_game_segment
    for i in range(20):
        buffer._push_game_segment(to_list(np.multiply(i, data)), meta)
    assert buffer.get_num_of_game_segments() == 20

    # fake data
    indices = [0, 1]
    make_time = [999, 1000]
    train_data = [[[], [], [], indices, [], make_time], []]
    # train_data = [current_batch, target_batch]
    # current_batch = [obs_lst, action_lst, mask_lst, batch_index_list, weights, make_time_lst]
    batch_priorities = [0.999, 0.8]

    buffer.update_priority(train_data, batch_priorities)

    assert buffer.game_pos_priorities[0] == 0.999


@pytest.mark.unittest
def test_sample_orig_data():
    buffer = EfficientZeroGameBuffer(config)

    # fake data
    data_1 = [[1, 1, 1] for i in range(10)]  # (s,a,r)
    meta_1 = {'done': True, 'unroll_plus_td_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    data_2 = [[1, 1, 1] for i in range(10, 20)]  # (s,a,r)
    meta_2 = {'done': True, 'unroll_plus_td_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    # push
    buffer._push_game_segment(data_1, meta_1)
    buffer._push_game_segment(data_2, meta_2)

    context = buffer._sample_orig_data(batch_size=2)
    # context = (game_lst, game_pos_lst, indices_lst, weights, make_time)
    print(context)


@pytest.mark.unittest
def test_non_sampled_varied_action_policy_maps_to_action_ids():
    buffer = MuZeroGameBuffer(_make_varied_action_config())

    target_policies = buffer._compute_target_policy_non_reanalyzed(
        _make_sparse_policy_context(), policy_shape=100
    )

    assert target_policies.shape == (1, 1, 100)
    np.testing.assert_allclose(target_policies[0, 0, [7, 42, 99]], [0.2, 0.3, 0.5])
    assert target_policies[0, 0].sum() == pytest.approx(1.0)


@pytest.mark.unittest
@pytest.mark.parametrize('buffer_cls', [SampledMuZeroGameBuffer, SampledEfficientZeroGameBuffer])
def test_sampled_varied_action_policy_uses_sampled_action_indices(buffer_cls):
    buffer = buffer_cls(_make_varied_action_config())

    target_policies = buffer._compute_target_policy_non_reanalyzed(
        _make_sparse_policy_context(), policy_shape=3
    )

    assert target_policies.shape == (1, 1, 3)
    np.testing.assert_allclose(target_policies[0, 0], [0.2, 0.3, 0.5])


def _reference_compute_target_reward_value(
    reward_value_context,
    bootstrap_values,
    *,
    discount_factor: float,
    num_unroll_steps: int,
    lstm_horizon_len: int,
    env_type: str,
):
    """Exact naive reference matching the pre-optimization _compute_target_reward_value loop."""
    (
        _value_obs_list,
        value_mask,
        pos_in_game_segment_list,
        rewards_list,
        game_segment_lens,
        td_steps_list,
        _action_mask_segment,
        to_play_segment,
    ) = reward_value_context

    is_2p = (
        env_type == 'board_games'
        and len(to_play_segment) > 0
        and len(to_play_segment[0]) > 0
        and to_play_segment[0][0] in (1, 2)
    )
    if is_2p:
        vals = [
            float(v) * (discount_factor ** int(td)) * float(m) * (1.0 if int(td) % 2 == 0 else -1.0)
            for v, td, m in zip(bootstrap_values, td_steps_list, value_mask)
        ]
    else:
        vals = [
            float(v) * (discount_factor ** int(td)) * float(m)
            for v, td, m in zip(bootstrap_values, td_steps_list, value_mask)
        ]

    batch_value_prefixs, batch_target_values = [], []
    horizon_id = 0
    value_index = 0
    for seg_len, reward_list, state_index, to_play_list in zip(
        game_segment_lens, rewards_list, pos_in_game_segment_list, to_play_segment
    ):
        target_values = []
        target_value_prefixs = []
        value_prefix = 0.0
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + int(td_steps_list[value_index])
            value = vals[value_index]
            for i, reward in enumerate(reward_list[current_index:bootstrap_index]):
                r = float(reward)
                if is_2p:
                    if to_play_list[current_index + i] == to_play_list[current_index]:
                        value += r * (discount_factor ** i)
                    else:
                        value += -r * (discount_factor ** i)
                else:
                    value += r * (discount_factor ** i)

            if horizon_id % lstm_horizon_len == 0:
                value_prefix = 0.0
            horizon_id += 1

            if current_index < seg_len:
                target_values.append(value)
                value_prefix += float(reward_list[current_index])
                target_value_prefixs.append(value_prefix)
            else:
                target_values.append(0.0)
                target_value_prefixs.append(value_prefix)

            value_index += 1

        batch_value_prefixs.append(target_value_prefixs)
        batch_target_values.append(target_values)

    return np.asarray(batch_value_prefixs, dtype=np.float64), np.asarray(batch_target_values, dtype=np.float64)


@pytest.mark.unittest
@pytest.mark.parametrize(
    'scenario,reward_as_array',
    [
        ('sparse_terminal_2p', False),
        ('sparse_terminal_2p', True),
        ('sparse_multi_2p', False),
        ('sparse_multi_2p', True),
        ('dense_1p_bootstrap', False),
        ('dense_2p_non_alternating', True),
    ],
)
def test_sez_compute_target_reward_value_matches_reference(scenario, reward_as_array):
    import torch

    is_board = scenario != 'dense_1p_bootstrap'
    num_unroll_steps = 5
    lstm_horizon_len = 4  # intentionally not dividing (num_unroll_steps + 1) == 6
    discount_factor = 0.99

    cfg = EasyDict(
        dict(
            batch_size=3,
            priority_prob_alpha=0.6,
            priority_prob_beta=0.4,
            replay_buffer_size=100,
            env_type='board_games' if is_board else 'not_board_games',
            use_priority=False,
            action_type='varied_action_space' if is_board else 'fixed_action_space',
            game_segment_length=30,
            num_unroll_steps=num_unroll_steps,
            td_steps=12 if scenario == 'dense_1p_bootstrap' else 30,
            lstm_horizon_len=lstm_horizon_len,
            discount_factor=discount_factor,
            use_root_value=False,
            mini_infer_size=8,
            device='cpu',
            model=dict(
                model_type='mlp',
                observation_shape=4,
                frame_stack_num=1,
                categorical_distribution=False,
                continuous_action_space=False,
                action_space_size=5,
                num_of_sampled_actions=3,
                value_support_range=(-10, 10, 1),
                reward_support_range=(-10, 10, 1),
            ),
        )
    )
    buf = SampledEfficientZeroGameBuffer(cfg)

    rng = np.random.RandomState(42)
    seg_lens = [20, 14, 10]
    state_indices = [3, 11, 7]  # seg 1 and 2 hit padded tails (current_index >= seg_len)
    rewards_list = []
    to_play_segment = []
    for seg_idx, L in enumerate(seg_lens):
        if scenario == 'sparse_terminal_2p':
            # seg 0: zero rewards (mode 0), seg 1: single terminal +1 (mode 1), seg 2: single terminal -1 (mode 1)
            r = [0.0] * L
            if seg_idx == 1:
                r[-1] = 1.0
            elif seg_idx == 2:
                r[-1] = -1.0
            tp = [1 if t % 2 == 0 else 2 for t in range(L)]
        elif scenario == 'sparse_multi_2p':
            # Multiple sparse rewards (mode 2) + non-alternating consecutive turns
            r = [0.0] * L
            r[min(4, L - 1)] = 0.25
            r[min(8, L - 1)] = -0.5
            r[-1] = 1.0
            tp = [1, 1, 2, 1, 2, 2, 1, 2, 1, 1] * 3
            tp = tp[:L]
        elif scenario == 'dense_1p_bootstrap':
            # Dense rewards (nnz > 8 -> mode 3) with single-player to_play=-1
            r = [float(x) for x in rng.uniform(-1.0, 1.0, size=L)]
            tp = [-1] * L
        else:
            # dense_2p_non_alternating: dense rewards (mode 3) + 2p zero-sum with non-alternating turns
            r = [float(x) for x in rng.uniform(-1.0, 1.0, size=L)]
            tp = [1 if (t // 2) % 2 == 0 else 2 for t in range(L)]
        rewards_list.append(np.asarray(r, dtype=np.float32) if reward_as_array else r)
        to_play_segment.append(tp)

    unroll_stride = num_unroll_steps + 1
    value_obs_list = []
    value_mask = []
    td_steps_list = []
    expected_bootstrap_vals = []
    for b, (L, s_idx) in enumerate(zip(seg_lens, state_indices)):
        td = min(cfg.td_steps, max(1, L - s_idx))
        for k in range(unroll_stride):
            td_steps_list.append(td)
            boot_idx = s_idx + k + td
            if boot_idx < L:
                value_mask.append(1)
                obs_val = float(b * 10 + k + 1) * 0.1
                value_obs_list.append([np.full((4,), obs_val, dtype=np.float32)])
                expected_bootstrap_vals.append(obs_val)
            else:
                value_mask.append(0)
                value_obs_list.append([np.zeros((4,), dtype=np.float32)])
                expected_bootstrap_vals.append(0.0)

    action_mask_segment = [[np.ones(5, dtype=np.int8) for _ in range(L)] for L in seg_lens]
    context = [
        value_obs_list,
        value_mask,
        state_indices,
        rewards_list,
        seg_lens,
        td_steps_list,
        action_mask_segment,
        to_play_segment,
    ]

    class _DummyModel:
        training = False

        def initial_inference(self, obs):
            # Return first channel as raw scalar value (transformed via inverse_scalar_transform in buffer).
            out = EasyDict(value=obs[:, :1])
            return out

    from lzero.policy import inverse_scalar_transform

    expected_transformed_vals = [
        float(
            inverse_scalar_transform(
                torch.tensor([[v]], dtype=torch.float32),
                buf.value_support,
                categorical_distribution=False,
            ).item()
        )
        if m
        else 0.0
        for v, m in zip(expected_bootstrap_vals, value_mask)
    ]

    vp_opt, tv_opt = buf._compute_target_reward_value(context, _DummyModel())
    vp_ref, tv_ref = _reference_compute_target_reward_value(
        context,
        expected_transformed_vals,
        discount_factor=discount_factor,
        num_unroll_steps=num_unroll_steps,
        lstm_horizon_len=lstm_horizon_len,
        env_type=cfg.env_type,
    )

    np.testing.assert_allclose(vp_opt, vp_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tv_opt, tv_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.unittest
def test_sez_fast_conv_obs_policy_and_pickle_roundtrip():
    import pickle

    from lzero.mcts.buffer.game_segment import GameSegment
    from lzero.mcts.utils import prepare_observation

    cfg = EasyDict(
        dict(
            batch_size=3,
            priority_prob_alpha=0.6,
            priority_prob_beta=0.4,
            replay_buffer_size=100,
            env_type='board_games',
            use_priority=False,
            action_type='varied_action_space',
            game_segment_length=12,
            num_unroll_steps=4,
            td_steps=10,
            lstm_horizon_len=5,
            discount_factor=0.99,
            use_root_value=False,
            reanalyze_ratio=0.0,
            reanalyze_outdated=False,
            mcts_ctree=True,
            mini_infer_size=8,
            device='cpu',
            gray_scale=False,
            transform2string=False,
            sampled_algo=True,
            gumbel_algo=False,
            use_ture_chance_label_in_chance_encoder=False,
            model=dict(
                model_type='conv',
                observation_shape=(3, 4, 4),
                image_channel=3,
                frame_stack_num=2,
                categorical_distribution=False,
                continuous_action_space=False,
                action_space_size=10,
                num_of_sampled_actions=3,
                value_support_range=(-10, 10, 1),
                reward_support_range=(-10, 10, 1),
            ),
        )
    )
    buf = SampledEfficientZeroGameBuffer(cfg)

    def _make_seg(seg_len, seed_offset):
        seg = GameSegment(
            action_space=10,
            game_segment_length=12,
            config=cfg,
            task_id=None,
        )
        init_frames = [np.full((3, 4, 4), float(seed_offset + i), dtype=np.float32) for i in range(2)]
        seg.reset(init_frames)
        for t in range(seg_len):
            nxt = np.full((3, 4, 4), float(seed_offset + 10 + t), dtype=np.float32)
            seg.append(
                action=t % 10,
                obs=nxt,
                reward=1.0 if t == seg_len - 1 else 0.0,
                action_mask=np.ones(10, dtype=np.int8),
                to_play=1 if t % 2 == 0 else 2,
            )
            # Mix normal distributions and an empty distribution to test uniform fallback
            seg.child_visit_segment.append([] if t == 1 else [0.5, 0.3, 0.2])
            seg.root_value_segment.append(0.0)
            seg.root_sampled_actions.append(np.array([[0], [1], [2]], dtype=np.int64))
        seg.game_segment_to_array()
        return seg

    segs = [_make_seg(10, 0), _make_seg(6, 100), _make_seg(12, 200)]
    metas = [
        {'done': True, 'unroll_plus_td_steps': 0, 'priorities': np.ones(len(s), dtype=np.float32)}
        for s in segs
    ]
    buf.push_game_segments((segs, metas))

    # Exercise _make_batch fast conv obs path and compare obs bit-for-bit with prepare_observation(get_unroll_obs)
    np.random.seed(7)
    reward_ctx, policy_re_ctx, policy_non_re_ctx, current_batch = buf._make_batch(batch_size=3, reanalyze_ratio=0.0)
    assert policy_re_ctx is None
    obs_fast = current_batch[0]

    # Reconstruct exact reference obs from the same sampled segments & positions
    pos_list = reward_ctx[2]
    # Re-run _sample_orig_data with same seed to recover exact game_lst
    np.random.seed(7)
    game_lst, pos_list_2, _, _, _ = buf._sample_orig_data(3)
    assert pos_list == pos_list_2
    obs_ref = prepare_observation(
        [g.get_unroll_obs(p, num_unroll_steps=cfg.num_unroll_steps, padding=True) for g, p in zip(game_lst, pos_list)],
        'conv',
    )
    np.testing.assert_array_equal(obs_fast, obs_ref)

    # Check _compute_target_policy_non_reanalyzed handles empty (uniform), normal, and padded tail (zeros)
    p_ctx = [
        [0, 4],
        [segs[0].child_visit_segment, segs[1].child_visit_segment],
        [len(segs[0]), len(segs[1])],
        [segs[0].action_mask_segment, segs[1].action_mask_segment],
        [segs[0].to_play_segment, segs[1].to_play_segment],
    ]
    pol = buf._compute_target_policy_non_reanalyzed(p_ctx, policy_shape=3)
    assert pol.shape == (2, 5, 3)
    np.testing.assert_allclose(pol[0, 0], [0.5, 0.3, 0.2])
    np.testing.assert_allclose(pol[0, 1], [1.0 / 3, 1.0 / 3, 1.0 / 3])  # empty child_visit -> uniform
    np.testing.assert_allclose(pol[1, 0], [0.5, 0.3, 0.2])
    np.testing.assert_allclose(pol[1, 1], [0.5, 0.3, 0.2])
    np.testing.assert_allclose(pol[1, 2:], 0.0)  # padded tail beyond seg_len=6 (state_index=4 -> valid k=0,1)

    # __getstate__ / __setstate__ strips non-picklable lock & caches and rebuilds discount cache cleanly
    buf._get_discount_powers(16)
    state = buf.__getstate__()
    assert '_segment_update_lock' not in state
    assert '_discount_powers' not in state
    assert '_discount_powers_list' not in state
    restored = SampledEfficientZeroGameBuffer.__new__(SampledEfficientZeroGameBuffer)
    restored.__setstate__(state)
    pows, pows_list = restored._get_discount_powers(10)
    np.testing.assert_allclose(pows[:5], [0.99 ** i for i in range(5)])
    assert len(pows_list) == len(pows)


@pytest.mark.unittest
def test_uniform_sample_orig_data_respects_zero_priority_mask_and_uniqueness():
    cfg = EasyDict(
        dict(
            batch_size=32,
            priority_prob_alpha=0.6,
            priority_prob_beta=0.4,
            replay_buffer_size=1000,
            env_type='not_board_games',
            use_priority=False,
            action_type='fixed_action_space',
            game_segment_length=25,
            model=dict(
                action_space_size=6,
                value_support_range=(-10, 10, 1),
                reward_support_range=(-10, 10, 1),
            ),
        )
    )
    buf = SampledEfficientZeroGameBuffer(cfg)
    # 20 segments x 25 transitions = 500 transitions (batch_size*4=128 < 400 -> fast O(batch_size) path)
    for seg_i in range(20):
        seg = [[seg_i, t, 0.0] for t in range(25)]
        prios = np.ones(25, dtype=np.float32)
        prios[-5:] = 0.0  # structurally masked segment tails (100 total masked transitions)
        buf._push_game_segment(seg, {'done': True, 'unroll_plus_td_steps': 0, 'priorities': prios})

    priorities = np.asarray(buf.game_pos_priorities)
    assert buf.get_num_of_transitions() == 500
    assert buf.get_num_of_sampleable_transitions() == 400

    np.random.seed(123)
    first_indices = None
    for trial in range(10):
        _, _, indices, weights, make_time = buf._sample_orig_data(batch_size=32)
        assert len(indices) == 32
        assert len(set(indices.tolist())) == 32
        assert np.all(priorities[indices] > 0)
        np.testing.assert_array_equal(weights, np.ones(32, dtype=np.float64))
        assert all(t > 0 for t in make_time)
        if trial == 0:
            first_indices = indices.copy()

    # Deterministic under np.random.seed
    np.random.seed(123)
    _, _, repeat_indices, _, _ = buf._sample_orig_data(batch_size=32)
    np.testing.assert_array_equal(first_indices, repeat_indices)
