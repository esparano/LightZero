import numpy as np
import pytest
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer
from lzero.mcts.buffer.game_segment import GameSegment
from zoo.board_games.galcon.envs.galcon_env import GalconEnv


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

    def test_game_buffer_push_with_explicit_reflect_fn(self):
        buffer = _make_buffer(sym_x=True, sym_y=True)
        buffer._cfg.pop('symmetric_augment_type', None)
        buffer.reflect_fn = GalconEnv.reflect_game_segment

        segment, _ = _create_mock_galcon_segment()
        meta = {'done': True, 'unroll_plus_td_steps': 4, 'priorities': np.ones(5)}

        buffer.push_game_segments(([segment], [meta]))

        assert len(buffer.game_segment_buffer) == 4


def _build_rich_galcon_segment(
    grid_h: int = 8,
    grid_w: int = 8,
    fleet_slots_per_side: int = 2,
    include_redirect: bool = False,
    num_steps: int = 6,
    num_unroll_steps: int = 2,
    frame_stack_num: int = 1,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    planet_ch = GalconEnv.PLANET_CHANNEL_COUNT
    fleet_feat = GalconEnv.FLEET_FEATURE_COUNT
    landing_ch = len(GalconEnv.LANDING_BUCKETS) * GalconEnv.LANDING_FEATURE_COUNT_PER_BUCKET
    channels = planet_ch + 2 * fleet_slots_per_side * fleet_feat + landing_ch

    pair_actions = (grid_h * grid_w) ** 2
    action_space_size = (2 * pair_actions + 1) if include_redirect else (pair_actions + 1)
    k_samp = 4

    config = EasyDict(dict(
        num_unroll_steps=num_unroll_steps,
        td_steps=num_steps,
        discount_factor=0.99,
        gray_scale=False,
        transform2string=False,
        sampled_algo=True,
        gumbel_algo=False,
        use_ture_chance_label_in_chance_encoder=False,
        model=dict(
            frame_stack_num=frame_stack_num,
            observation_shape=(channels, grid_h, grid_w),
            image_channel=channels,
            action_space_size=action_space_size,
            continuous_action_space=False,
            num_of_sampled_actions=k_samp,
        ),
    ))
    segment = GameSegment(action_space=None, game_segment_length=num_steps, config=config)

    for step in range(num_steps):
        obs = np.zeros((channels, grid_h, grid_w), dtype=np.float32)
        # Populate asymmetric planet cells (with non-trivial local x/y offsets) and empty cells
        for _ in range(3):
            y, x = int(rng.integers(0, grid_h)), int(rng.integers(0, grid_w))
            obs[GalconEnv.PLANET_LOCAL_X_CHANNEL, y, x] = float(rng.uniform(0.05, 0.95))
            obs[GalconEnv.PLANET_LOCAL_Y_CHANNEL, y, x] = float(rng.uniform(0.05, 0.95))
            obs[2:planet_ch, y, x] = rng.uniform(0.1, 1.0, size=(planet_ch - 2,)).astype(np.float32)
        # Populate asymmetric fleet slots (both friendly and enemy)
        for slot in range(2 * fleet_slots_per_side):
            base_ch = planet_ch + slot * fleet_feat
            y, x = int(rng.integers(0, grid_h)), int(rng.integers(0, grid_w))
            obs[base_ch + 0, y, x] = float(rng.uniform(0.05, 0.95))  # target_x
            obs[base_ch + 1, y, x] = float(rng.uniform(0.05, 0.95))  # target_y
            obs[base_ch + 2, y, x] = float(rng.uniform(0.1, 1.0))    # eta/dist
            obs[base_ch + 3, y, x] = float(rng.uniform(0.1, 1.0))    # ships > 0
            obs[base_ch + 4, y, x] = float(rng.uniform(0.1, 1.0))
        # Populate landing buckets
        obs[channels - landing_ch:, :, :] = rng.uniform(0.0, 1.0, size=(landing_ch, grid_h, grid_w)).astype(np.float32)

        segment.obs_segment.append(obs)
        # Mix PASS (0), planet sends, and redirect actions (if enabled)
        if step == 0:
            act = 0
        elif include_redirect and step % 2 == 1:
            act = int(rng.integers(pair_actions + 1, action_space_size))
        else:
            act = int(rng.integers(1, pair_actions + 1))
        segment.action_segment.append(act)

        action_mask = np.zeros(action_space_size, dtype=np.int8)
        action_mask[0] = 1
        action_mask[act] = 1
        segment.action_mask_segment.append(action_mask)

        # Include a mix of PASS, real actions, and -1 sentinel padding
        sa = np.array([0, act, int(rng.integers(1, action_space_size)), -1], dtype=np.int64)
        segment.root_sampled_actions.append(sa)
        segment.child_visit_segment.append([0.2, 0.5, 0.3, 0.0])
        segment.root_value_segment.append(0.25)
        segment.reward_segment.append(0.0 if step < num_steps - 1 else 1.0)
        segment.to_play_segment.append(1 if step % 2 == 0 else 2)

    return segment, config, channels, action_space_size


class _DummyPolicy:
    def __init__(self):
        import torch
        self._target_model = torch.nn.Linear(1, 1)


@pytest.mark.unittest
class TestGalconOnTheFlyBatchSymmetry:

    @pytest.mark.parametrize('grid_h,grid_w,fleet_slots,include_redirect', [
        (4, 4, 0, False),   # minimal C=20 (0 fleet slots)
        (8, 8, 2, False),   # C=40 (current simplified obs)
        (8, 8, 4, False),   # C=60 (higher fleet slot count)
        (6, 8, 3, True),    # non-square grid, C=50, planet+redirect (2N+1) action space
    ])
    def test_reflect_batch_matches_reflect_game_segment_bit_for_bit(
        self, grid_h, grid_w, fleet_slots, include_redirect
    ):
        num_unroll = 2
        seg, _, channels, action_space_size = _build_rich_galcon_segment(
            grid_h=grid_h,
            grid_w=grid_w,
            fleet_slots_per_side=fleet_slots,
            include_redirect=include_redirect,
            num_steps=5,
            num_unroll_steps=num_unroll,
        )

        # Build a 4-row batch where each row is the same unroll window from pos=0,
        # assigned each of the 4 distinct orientations (F,F), (T,F), (F,T), (T,T).
        orientations = [(False, False), (True, False), (False, True), (True, True)]
        obs_unroll = np.stack(seg.obs_segment[:num_unroll + 1], axis=0)  # (U+1, C, H, W)
        act_unroll = np.asarray(seg.action_segment[:num_unroll], dtype=np.int64)
        sa_unroll = np.stack(seg.root_sampled_actions[:num_unroll + 1], axis=0)[..., None]  # ctree (U+1, K, 1)

        obs_batch_4d = np.stack(
            [obs_unroll.reshape((num_unroll + 1) * channels, grid_h, grid_w) for _ in orientations],
            axis=0,
        )
        act_batch = np.stack([act_unroll for _ in orientations], axis=0)
        sa_batch = np.stack([sa_unroll for _ in orientations], axis=0)

        obs_orig_copy = obs_batch_4d.copy()
        act_orig_copy = act_batch.copy()
        sa_orig_copy = sa_batch.copy()

        rx_mask = np.array([o[0] for o in orientations], dtype=np.bool_)
        ry_mask = np.array([o[1] for o in orientations], dtype=np.bool_)

        obs_out, act_out, sa_out = GalconEnv.reflect_batch(
            obs_batch_4d,
            act_batch,
            sa_batch,
            rx_mask,
            ry_mask,
            image_channel=channels,
            action_space_size=action_space_size,
        )

        # Caller inputs must not be mutated
        np.testing.assert_array_equal(obs_batch_4d, obs_orig_copy)
        np.testing.assert_array_equal(act_batch, act_orig_copy)
        np.testing.assert_array_equal(sa_batch, sa_orig_copy)

        # Every orientation must match reflect_game_segment bit-for-bit
        for row_idx, (rx, ry) in enumerate(orientations):
            ref_seg = GalconEnv.reflect_game_segment(seg, reflect_x=rx, reflect_y=ry)
            exp_obs = np.asarray(ref_seg.obs_segment[:num_unroll + 1]).reshape(
                (num_unroll + 1) * channels, grid_h, grid_w
            )
            exp_act = np.asarray(ref_seg.action_segment[:num_unroll], dtype=np.int64)
            exp_sa = np.stack(ref_seg.root_sampled_actions[:num_unroll + 1], axis=0)[..., None]

            np.testing.assert_array_equal(obs_out[row_idx], exp_obs)
            np.testing.assert_array_equal(act_out[row_idx], exp_act)
            np.testing.assert_array_equal(sa_out[row_idx], exp_sa)
            # Sentinel -1 slots must be preserved as -1
            assert np.all(sa_out[row_idx, :, -1, 0] == -1)

        # 5D obs_batch and 3D ptree sampled_actions must also match identically
        obs_batch_5d = obs_batch_4d.reshape(4, num_unroll + 1, channels, grid_h, grid_w)
        sa_batch_ptree = sa_batch[..., 0]
        obs_out_5d, act_out_5d, sa_out_ptree = GalconEnv.reflect_batch(
            obs_batch_5d,
            act_batch,
            sa_batch_ptree,
            rx_mask,
            ry_mask,
            image_channel=channels,
            action_space_size=action_space_size,
        )
        np.testing.assert_array_equal(obs_out_5d.reshape(obs_batch_4d.shape), obs_out)
        np.testing.assert_array_equal(act_out_5d, act_out)
        np.testing.assert_array_equal(sa_out_ptree, sa_out[..., 0])

    @pytest.mark.parametrize('sym_x,sym_y,expected_orientations', [
        (True, False, {(False, False), (True, False)}),
        (False, True, {(False, False), (False, True)}),
        (True, True, {(False, False), (True, False), (False, True), (True, True)}),
    ])
    def test_buffer_on_the_fly_stores_1x_and_samples_exact_reflections(
        self, sym_x, sym_y, expected_orientations
    ):
        num_unroll = 2
        seg, _, channels, action_space_size = _build_rich_galcon_segment(
            grid_h=4,
            grid_w=4,
            fleet_slots_per_side=3,  # C=50 (non-40 channel count!)
            include_redirect=False,
            num_steps=5,
            num_unroll_steps=num_unroll,
        )
        buffer_cfg = EasyDict(dict(
            batch_size=32,
            replay_buffer_size=100,
            env_type='board_games',
            action_type='varied_action_space',
            use_priority=False,
            priority_prob_alpha=0.6,
            priority_prob_beta=0.4,
            game_segment_length=5,
            num_unroll_steps=num_unroll,
            td_steps=5,
            discount_factor=0.99,
            lstm_horizon_len=2,
            reanalyze_ratio=0.0,
            use_root_value=False,
            mcts_ctree=True,
            device='cpu',
            symmetric_augment_x=sym_x,
            symmetric_augment_y=sym_y,
            symmetric_augment_type='Galcon',
            symmetric_augment_on_the_fly='auto',
            model=dict(
                model_type='conv',
                frame_stack_num=1,
                image_channel=channels,
                observation_shape=(channels, 4, 4),
                action_space_size=action_space_size,
                continuous_action_space=False,
                num_of_sampled_actions=4,
                categorical_distribution=True,
                value_support_range=(-10, 10, 1),
                reward_support_range=(-10, 10, 1),
            ),
        ))
        buf = SampledEfficientZeroGameBuffer(buffer_cfg)
        meta = {'done': True, 'unroll_plus_td_steps': 7, 'priorities': np.ones(5)}
        # Push 8 copies (40 transitions >= batch_size=32) so sampling succeeds without replacement.
        buf.push_game_segments(([seg] * 8, [meta] * 8))

        # Must store only 1x raw segments (8, NOT 32)!
        assert buf.get_num_of_game_segments() == 8
        assert buf.get_num_of_transitions() == 40

        # Precompute exact ground-truth reflected segments for all 4 orientations
        gt_segs = {
            (rx, ry): GalconEnv.reflect_game_segment(seg, reflect_x=rx, reflect_y=ry)
            for rx in (False, True)
            for ry in (False, True)
        }

        np.random.seed(123)
        train_data = buf.sample(32, _DummyPolicy())
        obs_batch, act_batch, sa_batch, _, batch_index_list, _, _ = train_data[0]

        seen_orientations = set()
        for row_idx in range(32):
            pos = int(buf.game_segment_game_pos_look_up[int(batch_index_list[row_idx])][1])
            if pos + num_unroll >= 5:
                continue  # skip padded tail rows where random padding actions are drawn
            matched = []
            for orient, ref_seg in gt_segs.items():
                exp_obs = np.asarray(ref_seg.obs_segment[pos:pos + num_unroll + 1]).reshape(
                    (num_unroll + 1) * channels, 4, 4
                )
                exp_act = np.asarray(ref_seg.action_segment[pos:pos + num_unroll], dtype=np.int64)
                exp_sa = np.stack(ref_seg.root_sampled_actions[pos:pos + num_unroll + 1], axis=0)
                if sa_batch.ndim == 4 and exp_sa.ndim == 2:
                    exp_sa = exp_sa[..., None]
                if (
                    np.array_equal(obs_batch[row_idx], exp_obs)
                    and np.array_equal(act_batch[row_idx], exp_act)
                    and np.array_equal(sa_batch[row_idx], exp_sa)
                ):
                    matched.append(orient)
            assert len(matched) == 1, f"Row {row_idx} (pos={pos}) matched {matched}"
            assert matched[0] in expected_orientations
            seen_orientations.add(matched[0])

        assert seen_orientations == expected_orientations

    def test_on_the_fly_reanalyze_guards(self):
        # Explicit True + reanalyze_ratio > 0 must raise ValueError
        buf_strict = _make_buffer(sym_x=True, sym_y=True)
        buf_strict._cfg.symmetric_augment_on_the_fly = True
        buf_strict._cfg.reanalyze_ratio = 0.5
        seg, _ = _create_mock_galcon_segment()
        meta = {'done': True, 'unroll_plus_td_steps': 4, 'priorities': np.ones(5)}
        with pytest.raises(ValueError, match='reanalyze_ratio'):
            buf_strict.push_game_segments(([seg], [meta]))

        # 'auto' + reanalyze_ratio > 0 falls back cleanly to 4x segment storage
        buf_auto = _make_buffer(sym_x=True, sym_y=True)
        buf_auto._cfg.symmetric_augment_on_the_fly = 'auto'
        buf_auto._cfg.reanalyze_ratio = 0.5
        buf_auto.push_game_segments(([seg], [meta]))
        assert len(buf_auto.game_segment_buffer) == 4
