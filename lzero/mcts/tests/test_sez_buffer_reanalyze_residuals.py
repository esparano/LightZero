from types import SimpleNamespace
import numpy as np
import pytest
import torch
from easydict import EasyDict

import lzero.mcts.buffer.game_buffer_sampled_efficientzero as sez_buf_mod
from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer
from lzero.mcts.buffer.game_segment import GameSegment
from lzero.policy.scaling_transform import DiscreteSupport, phi_transform, scalar_transform


def _make_buffer_cfg(
    *,
    use_root_value: bool = False,
    categorical_distribution: bool = True,
    reanalyze_ratio: float = 0.0,
    mcts_ctree: bool = False,
    td_steps: int = 1,
    num_unroll_steps: int = 3,
    action_space_size: int = 6,
    num_of_sampled_actions: int = 3,
):
    return EasyDict(
        dict(
            device='cpu',
            batch_size=2,
            env_type='board_games',
            action_type='varied_action_space',
            game_segment_length=8,
            num_unroll_steps=num_unroll_steps,
            td_steps=td_steps,
            discount_factor=1.0,
            lstm_horizon_len=5,
            reanalyze_ratio=reanalyze_ratio,
            reanalyze_outdated=True,
            reanalyze_noise=False,
            use_root_value=use_root_value,
            use_priority=False,
            use_priority_IS_weight=False,
            priority_prob_alpha=0.6,
            priority_prob_beta=0.4,
            replay_buffer_size=100,
            mini_infer_size=64,
            mcts_ctree=mcts_ctree,
            num_simulations=2,
            pb_c_base=19652,
            pb_c_init=1.25,
            value_delta_max=0.01,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            gray_scale=False,
            transform2string=False,
            sampled_algo=True,
            gumbel_algo=False,
            use_ture_chance_label_in_chance_encoder=False,
            policy_target_type='visit_count',
            model=dict(
                model_type='mlp',
                frame_stack_num=1,
                observation_shape=4,
                action_space_size=action_space_size,
                continuous_action_space=False,
                num_of_sampled_actions=num_of_sampled_actions,
                categorical_distribution=categorical_distribution,
                value_support_range=(-1.0, 2.0, 1.0),
                reward_support_range=(-1.0, 2.0, 1.0),
                lstm_hidden_size=4,
            ),
        )
    )


def _make_segment(cfg, length: int, action_masks=None, to_plays=None, rewards=None):
    seg = GameSegment(
        action_space=None,
        game_segment_length=cfg.game_segment_length,
        config=cfg,
    )
    init_obs = [np.zeros((4,), dtype=np.float32)]
    seg.reset(init_obs)
    k_sampled = cfg.model.num_of_sampled_actions
    for t in range(length):
        obs_next = np.full((4,), float(t + 1), dtype=np.float32)
        rew = float(rewards[t]) if rewards is not None else 0.0
        am = (
            np.asarray(action_masks[t], dtype=np.int8)
            if action_masks is not None
            else np.ones((cfg.model.action_space_size,), dtype=np.int8)
        )
        tp = int(to_plays[t]) if to_plays is not None else (1 if t % 2 == 0 else 2)
        seg.append(0, obs_next, rew, action_mask=am, to_play=tp)
        seg.child_visit_segment.append([1.0 / k_sampled] * k_sampled)
        seg.root_value_segment.append(0.0)
        if cfg.mcts_ctree:
            seg.root_sampled_actions.append(np.arange(k_sampled, dtype=np.int64).reshape(k_sampled, 1))
        else:
            seg.root_sampled_actions.append(np.arange(k_sampled, dtype=np.int64))
    seg.game_segment_to_array()
    return seg


class _SpyTargetModel:
    def __init__(self, action_space_size: int, categorical_distribution: bool, value_scalar: float = 0.4):
        self.training = False
        self.action_space_size = action_space_size
        self.categorical_distribution = categorical_distribution
        self.value_scalar = value_scalar
        self.support = DiscreteSupport(-1.0, 2.0, 1.0, 'cpu')
        self.initial_inference_calls = []

    def to(self, device):
        return self

    def eval(self):
        self.training = False
        return self

    def _encode_value(self, batch_size: int) -> torch.Tensor:
        if self.categorical_distribution:
            scaled = scalar_transform(torch.full((batch_size, 1), self.value_scalar, dtype=torch.float32))
            probs = phi_transform(self.support, scaled).clamp_min(1e-6)
            return torch.log(probs)
        # Raw scalar mode: shape (B, 1) (not support_size=3!)
        return scalar_transform(torch.full((batch_size, 1), self.value_scalar, dtype=torch.float32))

    def initial_inference(self, obs: torch.Tensor):
        batch = obs.shape[0]
        self.initial_inference_calls.append(batch)
        return SimpleNamespace(
            latent_state=torch.zeros((batch, 4), dtype=torch.float32),
            value=self._encode_value(batch),
            value_prefix=torch.zeros((batch, 1), dtype=torch.float32),
            policy_logits=torch.zeros((batch, self.action_space_size), dtype=torch.float32),
            reward_hidden_state=(
                torch.zeros((1, batch, 4), dtype=torch.float32),
                torch.zeros((1, batch, 4), dtype=torch.float32),
            ),
        )

    def recurrent_inference(self, latent_state, reward_hidden_state, actions):
        batch = latent_state.shape[0]
        if self.categorical_distribution:
            vp = torch.tensor([[-10.0, 10.0, -10.0]], dtype=torch.float32).repeat(batch, 1)
        else:
            vp = torch.zeros((batch, 1), dtype=torch.float32)
        return SimpleNamespace(
            latent_state=torch.zeros_like(latent_state),
            value=self._encode_value(batch),
            value_prefix=vp,
            policy_logits=torch.zeros((batch, self.action_space_size), dtype=torch.float32),
            reward_hidden_state=(
                torch.zeros_like(reward_hidden_state[0]),
                torch.zeros_like(reward_hidden_state[1]),
            ),
        )


@pytest.mark.unittest
def test_use_root_value_indexes_bootstrap_to_play_and_action_mask_and_skips_padding(monkeypatch):
    """BUG-7d: With use_root_value=True, MCTS runs on bootstrap states (pos + k + td_steps),
    so it must use to_play and action_mask at pos + k + td_steps (not pos + k) and skip
    value_mask == 0 padded slots.
    """
    cfg = _make_buffer_cfg(
        use_root_value=True,
        categorical_distribution=True,
        reanalyze_ratio=0.0,
        mcts_ctree=False,
        td_steps=1,
        num_unroll_steps=3,
        action_space_size=6,
        num_of_sampled_actions=2,
    )
    # Episode length = 4: states 0, 1, 2, 3 have distinct action masks and alternating to_play [1, 2, 1, 2].
    # Starting at pos=1 with td_steps=1, num_unroll_steps=3 (4 unroll slots k=0,1,2,3):
    #   k=0 -> bootstrap_idx=2 (valid, to_play=1, legal=[2, 3])
    #   k=1 -> bootstrap_idx=3 (valid, to_play=2, legal=[4, 5])
    #   k=2 -> bootstrap_idx=4 (padded, value_mask=0 -> must be skipped!)
    #   k=3 -> bootstrap_idx=5 (padded, value_mask=0 -> must be skipped!)
    masks = [
        [1, 1, 0, 0, 0, 0],  # t=0: [0, 1]
        [1, 0, 0, 0, 0, 1],  # t=1: [0, 5]
        [0, 0, 1, 1, 0, 0],  # t=2: [2, 3]
        [0, 0, 0, 0, 1, 1],  # t=3: [4, 5]
    ]
    to_plays = [1, 2, 1, 2]
    seg = _make_segment(cfg, length=4, action_masks=masks, to_plays=to_plays)
    buf = SampledEfficientZeroGameBuffer(cfg)
    model = _SpyTargetModel(action_space_size=6, categorical_distribution=True, value_scalar=0.5)

    captured = {}
    orig_roots = sez_buf_mod.MCTSPtree.roots
    orig_search = sez_buf_mod.MCTSPtree.search

    def _spy_roots(root_num, legal_actions, *args, **kwargs):
        captured['root_num'] = root_num
        captured['legal_actions'] = legal_actions
        return orig_roots(root_num, legal_actions, *args, **kwargs)

    def _spy_search(self, roots, m, latent_state_roots, reward_hidden_state_roots, to_play):
        captured['to_play'] = list(to_play)
        return orig_search(self, roots, m, latent_state_roots, reward_hidden_state_roots, to_play)

    monkeypatch.setattr(sez_buf_mod.MCTSPtree, 'roots', staticmethod(_spy_roots))
    monkeypatch.setattr(sez_buf_mod.MCTSPtree, 'search', _spy_search)

    reward_value_context = buf._prepare_reward_value_context(
        batch_index_list=[0],
        game_segment_list=[seg],
        pos_in_game_segment_list=[1],
        total_transitions=4,
    )
    buf._compute_target_reward_value(reward_value_context, model)

    # Only the 2 unmasked bootstrap slots (k=0 -> idx 2, k=1 -> idx 3) should run inference & MCTS
    assert sum(model.initial_inference_calls) == 2
    assert captured['root_num'] == 2
    assert captured['to_play'] == [1, 2]
    assert captured['legal_actions'] == [[2, 3], [4, 5]]


@pytest.mark.unittest
@pytest.mark.parametrize('mode', ['value_no_root', 'value_with_root', 'policy_reanalyze'])
def test_categorical_distribution_false_does_not_crash_in_buffer(monkeypatch, mode):
    """BUG-7e: inverse_scalar_transform calls in SampledEfficientZeroGameBuffer must forward
    categorical_distribution=self._cfg.model.categorical_distribution so scalar-value models
    decode true scalar values instead of collapsing them via softmax over support atoms.
    """
    cfg = _make_buffer_cfg(
        use_root_value=(mode == 'value_with_root'),
        categorical_distribution=False,
        reanalyze_ratio=(1.0 if mode == 'policy_reanalyze' else 0.0),
        mcts_ctree=False,
        td_steps=1,
        num_unroll_steps=2,
        action_space_size=4,
        num_of_sampled_actions=2,
    )
    seg = _make_segment(cfg, length=5)
    buf = SampledEfficientZeroGameBuffer(cfg)
    model = _SpyTargetModel(action_space_size=4, categorical_distribution=False, value_scalar=0.5)

    recorded_calls = []
    orig_inv = sez_buf_mod.inverse_scalar_transform

    def _spy_inv(*args, **kwargs):
        out = orig_inv(*args, **kwargs)
        recorded_calls.append((kwargs.get('categorical_distribution', True), out.detach().clone()))
        return out

    monkeypatch.setattr(sez_buf_mod, 'inverse_scalar_transform', _spy_inv)

    if mode in ('value_no_root', 'value_with_root'):
        ctx = buf._prepare_reward_value_context(
            batch_index_list=[0],
            game_segment_list=[seg],
            pos_in_game_segment_list=[0],
            total_transitions=5,
        )
        vprefixes, tvals = buf._compute_target_reward_value(ctx, model)
        assert vprefixes.shape == (1, 3)
        assert tvals.shape == (1, 3)
        assert np.all(np.isfinite(tvals))
        if mode == 'value_no_root':
            # td_steps=1 in 2-player mode negates bootstrap value (+0.5 -> -0.5)
            assert np.allclose(tvals, -0.5, atol=1e-3)
    else:
        ctx = buf._prepare_policy_reanalyzed_context(
            batch_index_list=[0],
            game_segment_list=[seg],
            pos_in_game_segment_list=[0],
        )
        policies, sampled_actions = buf._compute_target_policy_reanalyzed(ctx, model)
        assert policies.shape == (1, 3, 2)
        assert sampled_actions.shape[0] == 3
        assert np.allclose(policies.sum(axis=-1), 1.0)

    assert recorded_calls, 'Expected inverse_scalar_transform to be called'
    for cat_flag, decoded in recorded_calls:
        assert cat_flag is False
        assert torch.allclose(decoded, torch.full_like(decoded, 0.5), atol=1e-3)


@pytest.mark.unittest
@pytest.mark.parametrize('mcts_ctree', [False, True])
def test_policy_reanalyze_skips_masked_padding_slots(monkeypatch, mcts_ctree):
    """BUG-7f: _compute_target_policy_reanalyzed must skip GPU initial_inference and MCTS search
    on policy_mask == 0 padding slots while preserving exact outputs and valid action indices.
    """
    cfg = _make_buffer_cfg(
        use_root_value=False,
        categorical_distribution=True,
        reanalyze_ratio=1.0,
        mcts_ctree=mcts_ctree,
        td_steps=1,
        num_unroll_steps=4,
        action_space_size=6,
        num_of_sampled_actions=3,
    )
    # Segment length = 3; starting at pos=1 with num_unroll_steps=4 gives 5 unroll slots:
    #   k=0 (idx 1): valid (policy_mask=1)
    #   k=1 (idx 2): valid (policy_mask=1)
    #   k=2,3,4 (idx 3,4,5): out-of-trajectory padding (policy_mask=0 -> must be skipped!)
    seg = _make_segment(cfg, length=3)
    buf = SampledEfficientZeroGameBuffer(cfg)
    model = _SpyTargetModel(action_space_size=6, categorical_distribution=True, value_scalar=0.25)

    captured_roots = {}
    tree_cls = sez_buf_mod.MCTSCtree if mcts_ctree else sez_buf_mod.MCTSPtree
    orig_roots = tree_cls.roots

    def _spy_roots(root_num, legal_actions, *args, **kwargs):
        captured_roots['root_num'] = root_num
        captured_roots['legal_actions_len'] = len(legal_actions)
        return orig_roots(root_num, legal_actions, *args, **kwargs)

    monkeypatch.setattr(tree_cls, 'roots', staticmethod(_spy_roots))

    ctx = buf._prepare_policy_reanalyzed_context(
        batch_index_list=[0],
        game_segment_list=[seg],
        pos_in_game_segment_list=[1],
    )
    policies, sampled_actions = buf._compute_target_policy_reanalyzed(ctx, model)

    # Only the 2 active unmasked slots should run model.initial_inference and MCTS
    assert sum(model.initial_inference_calls) == 2
    assert captured_roots['root_num'] == 2
    assert captured_roots['legal_actions_len'] == 2

    # Output shapes cover all 5 unroll slots; active slots sum to 1, padded slots are 0
    assert policies.shape == (1, 5, 3)
    assert np.allclose(policies[0, :2].sum(axis=-1), 1.0)
    assert np.allclose(policies[0, 2:], 0.0)
    assert sampled_actions.shape[0] == 5
    # Padded sampled actions must be valid in-range integer indices (never out-of-bounds for torch.gather)
    assert np.all((sampled_actions >= 0) & (sampled_actions < cfg.model.action_space_size))


@pytest.mark.unittest
def test_policy_reanalyze_all_masked_skips_model_and_mcts_completely():
    """BUG-7f edge case: if every transition in a reanalyze context has policy_mask == 0,
    no inference or MCTS search runs and zeroed outputs with matching shapes are returned.
    """
    cfg = _make_buffer_cfg(
        use_root_value=False,
        categorical_distribution=True,
        reanalyze_ratio=1.0,
        mcts_ctree=False,
        td_steps=1,
        num_unroll_steps=2,
        action_space_size=4,
        num_of_sampled_actions=2,
    )
    seg = _make_segment(cfg, length=2)
    buf = SampledEfficientZeroGameBuffer(cfg)
    model = _SpyTargetModel(action_space_size=4, categorical_distribution=True)

    ctx = buf._prepare_policy_reanalyzed_context(
        batch_index_list=[0],
        game_segment_list=[seg],
        pos_in_game_segment_list=[2],  # past segment length -> all policy_mask == 0
    )
    policies, sampled_actions = buf._compute_target_policy_reanalyzed(ctx, model)

    assert model.initial_inference_calls == []
    assert policies.shape == (1, 3, 2)
    assert np.allclose(policies, 0.0)
    assert sampled_actions.shape == (3, 2)
