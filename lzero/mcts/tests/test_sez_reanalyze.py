# -*- coding: utf-8 -*-
"""Regression tests for SampledEfficientZeroGameBuffer reanalyze (CTree & PTree)."""

import copy
import threading

import numpy as np
import pytest
import torch
from ding.config import compile_config
from ding.policy import create_policy
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer
from lzero.mcts.buffer.game_segment import GameSegment
from lzero.policy import DiscreteSupport
from zoo.board_games.galcon.config.galcon_sez_sp_mode_config_SIMPLE_1map import (
    create_config as _galcon_create_cfg,
    main_config as _galcon_main_cfg,
)
from zoo.board_games.galcon.envs.galcon_env import GalconEnv


def _build_tiny_galcon_setup(
    mcts_ctree: bool,
    reanalyze_ratio: float,
    policy_target_type: str = 'completed_q',
    num_unroll_steps: int = 2,
    batch_size: int = 4,
    reanalyze_noise: bool = False,
    symmetric_augment: bool = False,
):
    mc = EasyDict(copy.deepcopy(_galcon_main_cfg))
    cc = EasyDict(copy.deepcopy(_galcon_create_cfg))

    mc.env.num_planets = 4
    mc.env.grid_max_x = 50.0
    mc.env.grid_max_y = 50.0
    mc.env.grid_square_size = 25.0
    mc.env.max_episode_steps = 8

    mc.policy.cuda = False
    mc.policy.device = 'cpu'
    mc.policy.mcts_ctree = mcts_ctree
    mc.policy.batch_size = batch_size
    mc.policy.num_simulations = 4
    mc.policy.game_segment_length = 8
    mc.policy.td_steps = 8
    mc.policy.num_unroll_steps = num_unroll_steps
    mc.policy.lstm_horizon_len = max(1, num_unroll_steps)
    mc.policy.reanalyze_ratio = reanalyze_ratio
    mc.policy.reanalyze_outdated = True
    mc.policy.reanalyze_noise = reanalyze_noise
    mc.policy.policy_target_type = policy_target_type
    mc.policy.symmetric_augment_x = symmetric_augment
    mc.policy.symmetric_augment_y = symmetric_augment
    mc.policy.model.observation_shape = (40, 4, 4)
    mc.policy.model.image_channel = 40
    mc.policy.model.action_space_size = 4 * 4 * 4 * 4 + 1
    mc.policy.model.num_of_sampled_actions = 4
    mc.policy.model.num_res_blocks = 1
    mc.policy.model.num_channels = 8
    mc.policy.model.lstm_hidden_size = 16
    mc.policy.model.reward_head_channels = 4
    mc.policy.model.value_head_channels = 4
    mc.policy.model.policy_head_channels = 4
    mc.policy.model.reward_head_hidden_channels = [16]
    mc.policy.model.value_head_hidden_channels = [16]
    mc.policy.model.policy_head_hidden_channels = [16]
    mc.policy.model.embedded_action_dim = 8
    mc.policy.model.policy_head_type = 'factored_spatial'
    mc.policy.model.policy_head_embed_dim = 8
    cc.env_manager.type = 'base'

    cfg = compile_config(mc, create_cfg=cc, auto=True, save_cfg=False)
    policy = create_policy(cfg.policy, enable_field=['learn', 'collect', 'eval'])
    buf = SampledEfficientZeroGameBuffer(cfg.policy)
    env = GalconEnv(cfg.env)
    return cfg, policy, buf, env


def _collect_segments(cfg, policy, env, num_episodes: int = 2, steps_per_ep: int = 6):
    segs, metas = [], []
    for _ in range(num_episodes):
        obs = env.reset()
        seg = GameSegment(
            env.action_space,
            game_segment_length=cfg.policy.game_segment_length,
            config=cfg.policy,
            task_id=None,
        )
        seg.reset([obs['observation']])
        for step in range(steps_per_ep):
            data_t = torch.from_numpy(obs['observation']).unsqueeze(0).float()
            out = policy.collect_mode.forward(
                data_t,
                [obs['action_mask']],
                temperature=1.0,
                to_play=[obs['to_play']],
                ready_env_id=[0],
            )
            act = out[0]['action']
            seg.store_search_stats(
                out[0]['visit_count_distributions'],
                out[0]['searched_value'],
                out[0]['root_sampled_actions'],
            )
            ts = env.step(act)
            seg.append(act, ts.obs['observation'], ts.reward, obs['action_mask'], obs['to_play'], timestep=step)
            obs = ts.obs
            if ts.done:
                break
        seg.game_segment_to_array()
        segs.append(seg)
        metas.append({'priorities': np.ones(len(seg), dtype=np.float32), 'done': True, 'unroll_plus_td_steps': 1})
    return segs, metas


class _DeterministicShiftModel:
    """Stub model that always strongly prefers legal action ``preferred_action`` over stale actions."""

    def __init__(self, action_space_size: int, preferred_action: int):
        self.action_space_size = action_space_size
        self.preferred_action = preferred_action
        self.training = False

    def to(self, _device):
        return self

    def eval(self):
        self.training = False
        return self

    def initial_inference(self, obs):
        B = obs.shape[0]
        logits = torch.full((B, self.action_space_size), -10.0, dtype=torch.float32)
        logits[:, self.preferred_action] = 10.0
        value = torch.tensor([[0.0, 20.0, 0.0] for _ in range(B)], dtype=torch.float32)
        latent = torch.zeros((B, 4), dtype=torch.float32)
        h = torch.zeros((1, B, 4), dtype=torch.float32)
        c = torch.zeros((1, B, 4), dtype=torch.float32)
        return EasyDict(
            value=value,
            value_prefix=np.zeros((B, 1), dtype=np.float32),
            policy_logits=logits,
            latent_state=latent,
            reward_hidden_state=(h, c),
        )

    def recurrent_inference(self, latent_states, reward_hidden_state, actions):
        B = latent_states.shape[0]
        logits = torch.full((B, self.action_space_size), -10.0, dtype=torch.float32)
        logits[:, self.preferred_action] = 10.0
        value = torch.tensor([[0.0, 20.0, 0.0] for _ in range(B)], dtype=torch.float32)
        value_prefix = torch.tensor([[0.0, 20.0, 0.0] for _ in range(B)], dtype=torch.float32)
        return EasyDict(
            value=value,
            value_prefix=value_prefix,
            policy_logits=logits,
            latent_state=latent_states,
            reward_hidden_state=reward_hidden_state,
        )


def _make_stub_segment(action_space_size: int, k: int, seg_len: int, mcts_ctree: bool, legal_actions_list, stale_actions, num_unroll_steps: int = 1):
    """Create a minimal GameSegment with deliberately stale root_sampled_actions and child_visit."""
    seg_cfg = EasyDict(
        dict(
            num_unroll_steps=num_unroll_steps,
            td_steps=seg_len,
            discount_factor=0.99,
            model=dict(
                frame_stack_num=1,
                observation_shape=(4,),
                action_space_size=action_space_size,
                continuous_action_space=False,
            ),
            gray_scale=False,
            transform2string=False,
            sampled_algo=True,
            gumbel_algo=False,
            use_ture_chance_label_in_chance_encoder=False,
        )
    )
    seg = GameSegment(
        action_space=action_space_size,
        game_segment_length=seg_len,
        config=seg_cfg,
        task_id=None,
    )
    seg.reset([np.zeros((4,), dtype=np.float32)])
    mask = np.zeros(action_space_size, dtype=np.int8)
    for a in legal_actions_list:
        mask[a] = 1
    stale_arr = np.array(stale_actions, dtype=np.int64).reshape(k, 1) if mcts_ctree else np.array(stale_actions, dtype=np.int64)
    for step in range(seg_len):
        # Put 100% probability mass on the LAST slot (k - 1) of stale_actions so it sits at a
        # strictly different slot index than the refreshed preferred_action (which peaks at slot 0).
        # That way any stale/refreshed mismatch or torn read deterministically maps to the wrong action!
        stale_dist = [0.0] * (k - 1) + [1.0]
        seg.store_search_stats(stale_dist, 0.0, stale_arr.copy())
        seg.append(legal_actions_list[0], np.zeros((4,), dtype=np.float32), 0.0, mask.copy(), 1 if step % 2 == 0 else 2, timestep=step)
    seg.game_segment_to_array()
    return seg


def _make_stub_buffer(
    action_space_size: int,
    k: int,
    seg_len: int,
    mcts_ctree: bool,
    reanalyze_ratio: float,
    num_unroll_steps: int = 1,
    batch_size: int = 2,
    reanalyze_noise: bool = False,
    root_noise_weight: float = 0.0,
):
    buf_cfg = EasyDict(
        dict(
            device='cpu',
            env_type='board_games',
            action_type='varied_action_space',
            mcts_ctree=mcts_ctree,
            replay_buffer_size=100,
            batch_size=batch_size,
            priority_prob_alpha=0.6,
            priority_prob_beta=0.4,
            use_priority=False,
            use_root_value=False,
            game_segment_length=seg_len,
            td_steps=seg_len,
            num_unroll_steps=num_unroll_steps,
            lstm_horizon_len=max(1, num_unroll_steps),
            discount_factor=0.99,
            mini_infer_size=16,
            num_simulations=4,
            pb_c_base=19652,
            pb_c_init=1.25,
            value_delta_max=0.01,
            root_dirichlet_alpha=0.3,
            root_noise_weight=root_noise_weight,
            reanalyze_ratio=reanalyze_ratio,
            reanalyze_outdated=True,
            reanalyze_noise=reanalyze_noise,
            policy_target_type='completed_q',
            completed_q_c_visit=50.0,
            completed_q_c_scale=1.0,
            symmetric_augment_x=False,
            symmetric_augment_y=False,
            model=dict(
                model_type='mlp',
                frame_stack_num=1,
                continuous_action_space=False,
                action_space_size=action_space_size,
                num_of_sampled_actions=k,
                categorical_distribution=True,
                value_support_range=(-1.0, 2.0, 1.0),
                reward_support_range=(-1.0, 2.0, 1.0),
            ),
        )
    )
    buf = SampledEfficientZeroGameBuffer(buf_cfg)
    buf.value_support = DiscreteSupport(-1.0, 2.0, 1.0)
    buf.reward_support = DiscreteSupport(-1.0, 2.0, 1.0)
    return buf


@pytest.mark.unittest
@pytest.mark.parametrize('mcts_ctree', [True, False])
def test_reanalyze_updates_root_sampled_actions_in_lockstep_with_child_visit(mcts_ctree):
    """Reanalyzing a segment must update root_sampled_actions alongside child_visit so later non-reanalyzed samples stay aligned."""
    action_space_size = 10
    k = 4
    seg_len = 4
    # Legal actions are {5, 6, 7, 8}; stale stored actions are {0, 1, 2, 3} (completely disjoint!).
    legal_actions = [5, 6, 7, 8]
    stale_actions = [0, 1, 2, 3]
    preferred_action = 7

    buf = _make_stub_buffer(
        action_space_size=action_space_size,
        k=k,
        seg_len=seg_len,
        mcts_ctree=mcts_ctree,
        reanalyze_ratio=1.0,
        num_unroll_steps=1,
        batch_size=2,
    )
    seg = _make_stub_segment(action_space_size, k, seg_len, mcts_ctree, legal_actions, stale_actions)
    buf.push_game_segments(([seg], [{'priorities': np.ones(seg_len, dtype=np.float32), 'done': True, 'unroll_plus_td_steps': 1}]))

    policy = EasyDict(_target_model=_DeterministicShiftModel(action_space_size, preferred_action))

    # 1. Reanalyze all sampled transitions (reanalyze_ratio=1.0)
    train_data_re = buf.sample(2, policy)
    re_sampled_actions = train_data_re[0][2]
    re_target_policies = train_data_re[1][2]
    assert re_sampled_actions.dtype != object
    expected_shape = (2, 2, k, 1) if mcts_ctree else (2, 2, k)
    assert re_sampled_actions.shape == expected_shape
    assert re_target_policies.shape == (2, 2, k)

    # Every reanalyzed transition in the segment must now have refreshed legal root_sampled_actions
    # paired with refreshed child_visit_segment where preferred_action carries the highest mass.
    refreshed_count = 0
    for t in range(seg_len):
        stored_actions = np.asarray(seg.root_sampled_actions[t]).reshape(-1)
        stored_probs = np.asarray(seg.child_visit_segment[t], dtype=np.float64)
        if set(int(a) for a in stored_actions) != set(stale_actions):
            refreshed_count += 1
            assert set(int(a) for a in stored_actions).issubset(set(legal_actions))
            best_slot = int(np.argmax(stored_probs))
            assert int(stored_actions[best_slot]) == preferred_action
    assert refreshed_count > 0, 'Expected at least one segment transition to be refreshed in-place'

    # 2. Switch to non-reanalyzed sampling (reanalyze_ratio=0.0): every sample read back from
    #    the segment must still pair its stored child_visit with matching root_sampled_actions.
    buf._cfg.reanalyze_ratio = 0.0
    train_data_non_re = buf.sample(2, policy)
    non_re_actions = train_data_non_re[0][2]
    non_re_policies = train_data_non_re[1][2]
    non_re_masks = train_data_non_re[0][3]
    for b in range(2):
        for u in range(2):
            if non_re_masks[b, u] > 0:
                acts = np.asarray(non_re_actions[b, u]).reshape(-1)
                probs = np.asarray(non_re_policies[b, u], dtype=np.float64)
                best_act = int(acts[int(np.argmax(probs))])
                # Either untouched (peaks at slot k-1 -> stale_actions[-1]) or refreshed (peaks at slot 0 -> preferred_action).
                # Before the fix, refreshed child_visit (slot 0) paired with stale_actions -> stale_actions[0] != stale_actions[-1]!
                if set(int(a) for a in acts) == set(stale_actions):
                    assert best_act == stale_actions[-1]
                else:
                    assert best_act == preferred_action


@pytest.mark.unittest
@pytest.mark.parametrize('mcts_ctree', [True, False])
def test_galcon_sez_reanalyze_sample_and_learn_step(mcts_ctree):
    """End-to-end Galcon SEZ sample(reanalyze_ratio=0.5) + _forward_learn works for both CTree and PTree."""
    cfg, policy, buf, env = _build_tiny_galcon_setup(
        mcts_ctree=mcts_ctree,
        reanalyze_ratio=0.5,
        policy_target_type='completed_q',
        num_unroll_steps=2,
        batch_size=4,
    )
    segs, metas = _collect_segments(cfg, policy, env, num_episodes=2, steps_per_ep=6)
    buf.push_game_segments((segs, metas))

    for _ in range(2):
        train_data = buf.sample(4, policy)
        sampled_actions = train_data[0][2]
        target_policies = train_data[1][2]
        assert sampled_actions.dtype != object
        assert sampled_actions.shape == ((4, 3, 4, 1) if mcts_ctree else (4, 3, 4))
        assert target_policies.shape == (4, 3, 4)
        out = policy.learn_mode.forward(train_data)
        assert np.isfinite(out['total_loss'])


@pytest.mark.unittest
def test_reanalyze_edge_cases_zero_reanalyze_num_and_single_transition():
    """Handle int(batch_size * reanalyze_ratio) == 0 and transition_batch_size == 1 cleanly."""
    action_space_size = 8
    k = 4
    seg_len = 4
    legal_actions = [1, 2, 3, 4]
    policy = EasyDict(_target_model=_DeterministicShiftModel(action_space_size, preferred_action=3))

    # Case 1: reanalyze_ratio > 0 but int(batch_size * reanalyze_ratio) == 0
    buf_tiny_ratio = _make_stub_buffer(
        action_space_size=action_space_size,
        k=k,
        seg_len=seg_len,
        mcts_ctree=True,
        reanalyze_ratio=0.1,
        num_unroll_steps=1,
        batch_size=4,
    )
    segs = [_make_stub_segment(action_space_size, k, seg_len, True, legal_actions, legal_actions) for _ in range(2)]
    metas = [{'priorities': np.ones(seg_len, dtype=np.float32), 'done': True, 'unroll_plus_td_steps': 1} for _ in range(2)]
    buf_tiny_ratio.push_game_segments((segs, metas))
    train_data = buf_tiny_ratio.sample(4, policy)
    assert train_data[1][2].shape == (4, 2, k)

    # Case 2: transition_batch_size == 1 (batch_size=1, num_unroll_steps=0, reanalyze_ratio=1.0)
    buf_single = _make_stub_buffer(
        action_space_size=action_space_size,
        k=k,
        seg_len=seg_len,
        mcts_ctree=True,
        reanalyze_ratio=1.0,
        num_unroll_steps=0,
        batch_size=1,
    )
    seg = _make_stub_segment(action_space_size, k, seg_len, True, legal_actions, legal_actions)
    buf_single.push_game_segments(([seg], [{'priorities': np.ones(seg_len, dtype=np.float32), 'done': True, 'unroll_plus_td_steps': 1}]))
    train_data_1 = buf_single.sample(1, policy)
    assert train_data_1[1][2].shape == (1, 1, k)
    assert np.isclose(train_data_1[1][2][0, 0].sum(), 1.0, atol=1e-6)


@pytest.mark.unittest
def test_reanalyze_concurrent_sample_no_torn_reads():
    """Concurrent sample() calls with reanalyze_ratio=0.5 never pair stale actions with refreshed probabilities."""
    action_space_size = 10
    k = 4
    seg_len = 5
    legal_actions = [5, 6, 7, 8]
    stale_actions = [0, 1, 2, 3]
    preferred_action = 6

    buf = _make_stub_buffer(
        action_space_size=action_space_size,
        k=k,
        seg_len=seg_len,
        mcts_ctree=True,
        reanalyze_ratio=0.5,
        num_unroll_steps=1,
        batch_size=4,
    )
    segs = [_make_stub_segment(action_space_size, k, seg_len, True, legal_actions, stale_actions) for _ in range(2)]
    metas = [{'priorities': np.ones(seg_len, dtype=np.float32), 'done': True, 'unroll_plus_td_steps': 1} for _ in range(2)]
    buf.push_game_segments((segs, metas))

    errors = []

    def _worker():
        policy = EasyDict(_target_model=_DeterministicShiftModel(action_space_size, preferred_action))
        try:
            for _ in range(10):
                train_data = buf.sample(4, policy)
                acts_batch = train_data[0][2]
                mask_batch = train_data[0][3]
                pol_batch = train_data[1][2]
                for b in range(4):
                    for u in range(2):
                        if mask_batch[b, u] > 0:
                            acts = np.asarray(acts_batch[b, u]).reshape(-1)
                            probs = np.asarray(pol_batch[b, u], dtype=np.float64)
                            best_act = int(acts[int(np.argmax(probs))])
                            if set(int(a) for a in acts) == set(stale_actions):
                                assert best_act == stale_actions[-1], f'Torn read: stale acts {acts} with probs {probs}'
                            else:
                                assert best_act == preferred_action, f'Torn read: refreshed acts {acts} with probs {probs}'
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)
        assert not t.is_alive()
    assert not errors, f'Concurrent reanalyze errors: {errors}'


@pytest.mark.unittest
def test_sez_buffer_deepcopy_and_short_root_sampled_actions_bounds():
    """GameBuffer supports copy.deepcopy despite its internal lock, and guards short root_sampled_actions segments."""
    action_space_size = 8
    k = 4
    seg_len = 4
    legal_actions = [1, 2, 3, 4]
    buf = _make_stub_buffer(
        action_space_size=action_space_size,
        k=k,
        seg_len=seg_len,
        mcts_ctree=True,
        reanalyze_ratio=1.0,
        num_unroll_steps=1,
        batch_size=1,
    )
    # 1. copy.deepcopy succeeds and creates an independent lock
    buf_copy = copy.deepcopy(buf)
    assert buf_copy._get_segment_update_lock() is not buf._get_segment_update_lock()

    # 2. Legacy segment whose root_sampled_actions is shorter than len(seg) must not raise IndexError
    seg = _make_stub_segment(action_space_size, k, seg_len, True, legal_actions, legal_actions)
    seg.root_sampled_actions = seg.root_sampled_actions[:1]
    buf.push_game_segments(([seg], [{'priorities': np.ones(seg_len, dtype=np.float32), 'done': True, 'unroll_plus_td_steps': 1}]))
    policy = EasyDict(_target_model=_DeterministicShiftModel(action_space_size, preferred_action=3))
    train_data = buf.sample(1, policy)
    assert train_data[1][2].shape == (1, 2, k)


@pytest.mark.unittest
@pytest.mark.parametrize('mcts_ctree', [True, False])
def test_galcon_sez_reanalyze_with_4x_symmetry_augmentation(mcts_ctree):
    """Partial reanalyze preserves integer dtype, shape, and legal-action alignment on 4x symmetry-augmented Galcon segments."""
    cfg, policy, buf, env = _build_tiny_galcon_setup(
        mcts_ctree=mcts_ctree,
        reanalyze_ratio=0.5,
        policy_target_type='completed_q',
        num_unroll_steps=2,
        batch_size=4,
        reanalyze_noise=False,
        symmetric_augment=True,
    )
    buf.reflect_fn = GalconEnv.reflect_game_segment

    segs, metas = _collect_segments(cfg, policy, env, num_episodes=1, steps_per_ep=6)
    buf.push_game_segments((segs, metas))
    assert buf.get_num_of_game_segments() == 4
    initial_dtypes = [np.asarray(seg.root_sampled_actions[0]).dtype for seg in buf.game_segment_buffer]
    # Reflected segments (indices 1..3) are stored as integer arrays by GalconEnv.reflect_game_segment.
    for ref_dtype in initial_dtypes[1:]:
        assert np.issubdtype(ref_dtype, np.integer)

    expected_item_shape = (4, 1) if mcts_ctree else (4,)
    for _ in range(3):
        train_data = buf.sample(4, policy)
        sampled_actions = train_data[0][2]
        target_policies = train_data[1][2]
        assert sampled_actions.dtype != object
        assert np.all(sampled_actions == np.rint(sampled_actions))
        assert sampled_actions.shape == ((4, 3, 4, 1) if mcts_ctree else (4, 3, 4))
        assert target_policies.shape == (4, 3, 4)
        out = policy.learn_mode.forward(train_data)
        assert np.isfinite(out['total_loss'])

    # Verify every stored segment retains its original dtype/shape (int64 for reflected copies) and legal actions after in-place write-back.
    for seg, init_dtype in zip(buf.game_segment_buffer, initial_dtypes):
        for t in range(len(seg)):
            stored = np.asarray(seg.root_sampled_actions[t])
            assert stored.dtype == init_dtype
            assert stored.shape == expected_item_shape
            legal_set = set(np.nonzero(seg.action_mask_segment[t])[0].tolist())
            probs = np.asarray(seg.child_visit_segment[t], dtype=np.float64)
            assert np.all(np.isfinite(probs))
            assert np.isclose(probs.sum(), 1.0, atol=1e-5)
            for act, prob in zip(stored.reshape(-1), probs):
                if prob > 0:
                    assert int(act) in legal_set

    # Verify non-reanalyzed readback after in-place updates still learns cleanly.
    buf._cfg.reanalyze_ratio = 0.0
    train_data_non_re = buf.sample(4, policy)
    out_non_re = policy.learn_mode.forward(train_data_non_re)
    assert np.isfinite(out_non_re['total_loss'])


@pytest.mark.unittest
@pytest.mark.parametrize('mcts_ctree', [True, False])
def test_reanalyze_with_dirichlet_noise_and_sparse_legal_actions(mcts_ctree):
    """reanalyze_noise=True works cleanly for both CTree and PTree, including when len(legal_actions) < K."""
    action_space_size = 10
    k = 4
    seg_len = 4
    # Strictly fewer legal actions (2) than K (4), testing Dirichlet noise clamping/padding.
    legal_actions = [5, 7]
    stale_actions = [0, 1, 2, 3]
    buf = _make_stub_buffer(
        action_space_size=action_space_size,
        k=k,
        seg_len=seg_len,
        mcts_ctree=mcts_ctree,
        reanalyze_ratio=0.5,
        num_unroll_steps=1,
        batch_size=2,
        reanalyze_noise=True,
        root_noise_weight=0.25,
    )
    seg = _make_stub_segment(action_space_size, k, seg_len, mcts_ctree, legal_actions, stale_actions)
    buf.push_game_segments(([seg], [{'priorities': np.ones(seg_len, dtype=np.float32), 'done': True, 'unroll_plus_td_steps': 1}]))
    policy = EasyDict(_target_model=_DeterministicShiftModel(action_space_size, preferred_action=7))

    train_data = buf.sample(2, policy)
    re_acts = train_data[0][2][0]
    re_masks = train_data[0][3][0]
    re_pols = train_data[1][2][0]
    assert np.issubdtype(re_acts.dtype, np.integer)
    for u in range(2):
        if re_masks[u] > 0:
            probs = np.asarray(re_pols[u], dtype=np.float64)
            assert np.all(np.isfinite(probs))
            assert np.all(probs >= 0.0)
            assert np.isclose(probs.sum(), 1.0, atol=1e-5)
            for act, prob in zip(np.asarray(re_acts[u]).reshape(-1), probs):
                if prob > 0:
                    assert int(act) in legal_actions

    # End-to-end Galcon SEZ sample + learn with reanalyze_noise=True (policy default).
    cfg, galcon_policy, galcon_buf, env = _build_tiny_galcon_setup(
        mcts_ctree=mcts_ctree,
        reanalyze_ratio=0.5,
        policy_target_type='completed_q',
        num_unroll_steps=2,
        batch_size=4,
        reanalyze_noise=True,
        symmetric_augment=False,
    )
    segs, metas = _collect_segments(cfg, galcon_policy, env, num_episodes=2, steps_per_ep=6)
    galcon_buf.push_game_segments((segs, metas))
    galcon_batch = galcon_buf.sample(4, galcon_policy)
    out = galcon_policy.learn_mode.forward(galcon_batch)
    assert np.isfinite(out['total_loss'])
