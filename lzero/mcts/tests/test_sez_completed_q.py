# -*- coding: utf-8 -*-
"""
Unit tests for Gumbel-style ``completed_q`` policy targets in Sampled EfficientZero,
covering both ``SampledEfficientZeroMCTSCtree`` and ``SampledEfficientZeroMCTSPtree``,
minimax sign-flipping, duplicate-slot zero padding, and policy collect/eval integration.
"""

import numpy as np
import pytest
import torch
from easydict import EasyDict

from lzero.mcts.tree_search.mcts_ctree_sampled import SampledEfficientZeroMCTSCtree
from lzero.mcts.tree_search.mcts_ptree_sampled import SampledEfficientZeroMCTSPtree
from lzero.policy.sampled_efficientzero import (
    SampledEfficientZeroPolicy,
    compute_completed_q_policy,
)


class _StubModelOutput:
    def __init__(self, latent_state, policy_logits, value, value_prefix, reward_hidden_state):
        self.latent_state = latent_state
        self.policy_logits = policy_logits
        self.value = value
        self.value_prefix = value_prefix
        self.reward_hidden_state = reward_hidden_state


class _TwoActionStubModel(torch.nn.Module):
    """Minimal discrete SEZ model where root action determines subtree value cleanly across depths."""

    def __init__(self, action_space_size=6, root_logits=None, child_values=None):
        super().__init__()
        self.action_space_size = action_space_size
        self.root_logits = root_logits or [2.0, 0.0, -5.0, -5.0, -5.0, -5.0]
        # Value at depth 1 from the depth-1 player's perspective (negative = good for root in 2-player mode)
        self.child_values = child_values or {0: 0.2, 1: -0.8}
        self.value_support_size = 3
        self.reward_support_size = 3
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def initial_inference(self, obs):
        B = obs.shape[0]
        # latent[:, 0] = depth, latent[:, 1] = root action (-1 at depth 0)
        latent = torch.zeros((B, 4), dtype=torch.float32)
        latent[:, 1] = -1.0
        logits = torch.tensor([self.root_logits for _ in range(B)], dtype=torch.float32)
        value = torch.zeros((B, 1), dtype=torch.float32)
        value_prefix = [0.0 for _ in range(B)]
        rhs = (torch.zeros((1, B, 4)), torch.zeros((1, B, 4)))
        return _StubModelOutput(latent, logits, value, value_prefix, rhs)

    def recurrent_inference(self, latent_states, reward_hidden_state, actions):
        B = latent_states.shape[0]
        next_latents = torch.zeros((B, 4), dtype=torch.float32)
        values = []
        for b in range(B):
            depth = int(round(float(latent_states[b, 0].item())))
            root_act = int(round(float(latent_states[b, 1].item())))
            act = int(actions[b].reshape(-1)[0].item())
            if depth == 0:
                root_act = act
            next_depth = depth + 1
            next_latents[b, 0] = float(next_depth)
            next_latents[b, 1] = float(root_act)
            # Keep depth-1 value consistent regardless of deeper rollouts (alternating sign at even/odd plies)
            base_v = float(self.child_values.get(root_act, 0.0))
            if next_depth % 2 == 1:
                values.append([base_v])
            else:
                values.append([-base_v])
        value = torch.tensor(values, dtype=torch.float32)
        value_prefix = torch.zeros((B, 1), dtype=torch.float32)
        logits = torch.zeros((B, self.action_space_size), dtype=torch.float32)
        rhs = (torch.zeros((1, B, 4)), torch.zeros((1, B, 4)))
        return _StubModelOutput(next_latents, logits, value, value_prefix, rhs)


def _make_mcts_cfg(
    action_space_size=6,
    num_of_sampled_actions=4,
    num_simulations=12,
    discount_factor=0.99,
    env_type='board_games',
):
    return EasyDict(
        dict(
            device='cpu',
            env_type=env_type,
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
                categorical_distribution=False,
                value_support_range=(-1.0, 2.0, 1.0),
                reward_support_range=(-1.0, 2.0, 1.0),
            ),
        )
    )


def _extract_action_ints(sampled_actions_row):
    out = []
    for a in sampled_actions_row:
        if hasattr(a, 'value'):
            out.append(int(np.asarray(a.value).item()))
        elif isinstance(a, (list, tuple, np.ndarray)):
            out.append(int(np.asarray(a).reshape(-1)[0]))
        else:
            out.append(int(a))
    return out


# ---------------------------------------------------------------------------
# 1. Pure math tests for compute_completed_q_policy
# ---------------------------------------------------------------------------


def test_completed_q_overcomes_prior_trap():
    """Even when action 0 dominates prior & visit counts, a higher-Q action wins completed_q."""
    policy_logits = np.array([-0.98, -3.5, -4.0, -2.82, -18.0, -18.0])
    sampled_actions = np.array([0, 3, 1, 2, 2, 2])  # 4 unique + 2 padded duplicates
    visit_counts = [16, 6, 2, 1, 0, 0]
    child_q_values = [0.129, 0.259, 0.05, -0.10]  # only 4 unique entries

    probs = compute_completed_q_policy(
        visit_counts=visit_counts,
        child_q_values=child_q_values,
        sampled_actions=sampled_actions,
        policy_logits=policy_logits,
        raw_value=0.10,
        c_visit=50.0,
        c_scale=1.0,
    )

    assert len(probs) == 6
    # Trailing padded duplicate slots must stay exact 0.0
    assert probs[4] == 0.0 and probs[5] == 0.0
    assert np.isclose(sum(probs), 1.0, atol=1e-6)
    # Action 3 (index 1, Q=+0.259) must beat Action 0 (index 0, Q=+0.129) despite 16 vs 6 visits
    assert probs[1] > probs[0]


def test_completed_q_unvisited_gets_zero_advantage():
    """Unvisited actions are completed with v_mix so their logit shift is exactly 0."""
    policy_logits = np.array([1.0, 1.0, 1.0])
    sampled_actions = np.array([0, 1, 2])
    # Action 0 visited with Q = +0.5; actions 1 & 2 unvisited -> completed with v_mix < 0.5
    probs = compute_completed_q_policy(
        visit_counts=[5, 0, 0],
        child_q_values=[0.5, 0.0, 0.0],
        sampled_actions=sampled_actions,
        policy_logits=policy_logits,
        raw_value=0.0,
        c_visit=50.0,
        c_scale=1.0,
    )
    assert np.isclose(sum(probs), 1.0, atol=1e-6)
    assert probs[0] > probs[1]
    assert np.isclose(probs[1], probs[2], atol=1e-7)


def test_completed_q_large_reward_scale_is_numerically_stable():
    """Large Q magnitudes (e.g. Atari-scale) never produce NaN or Inf."""
    probs = compute_completed_q_policy(
        visit_counts=[10, 10],
        child_q_values=[500.0, 450.0],
        sampled_actions=np.array([0, 1]),
        policy_logits=np.array([0.0, 0.0]),
        raw_value=475.0,
        c_visit=50.0,
        c_scale=1.0,
    )
    assert np.all(np.isfinite(probs))
    assert np.isclose(sum(probs), 1.0, atol=1e-6)
    assert probs[0] > probs[1]


# ---------------------------------------------------------------------------
# 2. ctree & ptree get_children_q_values sign-flip and padding tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('mcts_cls', [SampledEfficientZeroMCTSCtree, SampledEfficientZeroMCTSPtree])
@pytest.mark.parametrize('to_play', [1, -1])
def test_mcts_get_children_q_values_sign_and_padding(mcts_cls, to_play):
    """Verify minimax sign flip for to_play=1 vs no flip for to_play=-1, and unique prefix length."""
    gamma = 0.99
    cfg = _make_mcts_cfg(
        action_space_size=6,
        num_of_sampled_actions=4,  # K=4 > 2 legal actions -> tests padding!
        num_simulations=16,
        discount_factor=gamma,
    )
    model = _TwoActionStubModel(
        action_space_size=6,
        root_logits=[0.0, 0.0, -5.0, -5.0, -5.0, -5.0],
        child_values={0: 0.6, 1: -0.6},
    )
    if to_play == -1:
        # In 1-player mode (no minimax sign flip), leaf values don't alternate sign across depth
        orig_rec = model.recurrent_inference
        def _rec_1p(latent_states, reward_hidden_state, actions):
            out = orig_rec(latent_states, reward_hidden_state, actions)
            B = latent_states.shape[0]
            vals = [[float(model.child_values.get(int(round(float(out.latent_state[b, 1].item()))), 0.0))] for b in range(B)]
            out.value = torch.tensor(vals, dtype=torch.float32)
            return out
        model.recurrent_inference = _rec_1p
    mcts = mcts_cls(cfg)
    roots = mcts_cls.roots(1, [[0, 1]], 6, 4, False)
    roots.prepare_no_noise([0.0], [model.root_logits], [to_play])

    latent = np.zeros((1, 4), dtype=np.float32)
    latent[:, 1] = -1.0
    rhs = (np.zeros((1, 1, 4), dtype=np.float32), np.zeros((1, 1, 4), dtype=np.float32))
    mcts.search(roots, model, latent, rhs, [to_play])

    dists = roots.get_distributions()[0]
    q_vals = roots.get_children_q_values(gamma)[0]
    sampled = _extract_action_ints(roots.get_sampled_actions()[0])

    # Distributions padded to K=4, q_values unpadded to num_unique=2
    assert len(dists) == 4
    assert dists[2] == 0 and dists[3] == 0
    assert len(q_vals) == 2

    q_by_act = {sampled[j]: q_vals[j] for j in range(len(q_vals))}
    if to_play == 1:
        # Two-player minimax: child.value=-0.6 (opponent losing) -> root Q ≈ +0.6*gamma
        assert q_by_act[1] > 0.3
        assert q_by_act[0] < -0.3
    else:
        # Single-player (to_play=-1): no negation -> child.value=+0.6 -> root Q ≈ +0.6*gamma
        assert q_by_act[0] > 0.3
        assert q_by_act[1] < -0.3


# ---------------------------------------------------------------------------
# 3. End-to-end SampledEfficientZeroPolicy collect & eval integration tests
# ---------------------------------------------------------------------------


def _make_policy_cfg(mcts_ctree=True, policy_target_type='completed_q', continuous=False):
    cfg = EasyDict(SampledEfficientZeroPolicy.default_config())
    cfg.cuda = False
    cfg.device = 'cpu'
    cfg.mcts_ctree = mcts_ctree
    cfg.env_type = 'board_games'
    cfg.battle_mode = 'self_play_mode'
    cfg.action_type = 'varied_action_space'
    cfg.num_simulations = 10
    cfg.discount_factor = 0.99
    cfg.root_noise_weight = 0.0
    cfg.policy_target_type = policy_target_type
    cfg.completed_q_c_visit = 50.0
    cfg.completed_q_c_scale = 1.0
    cfg.model.continuous_action_space = continuous
    cfg.model.action_space_size = 6
    cfg.model.num_of_sampled_actions = 4
    cfg.model.categorical_distribution = False
    cfg.model.value_support_range = (-1.0, 2.0, 1.0)
    cfg.model.reward_support_range = (-1.0, 2.0, 1.0)
    return cfg


def test_policy_target_type_validation():
    bad_cfg = _make_policy_cfg(policy_target_type='invalid_mode')
    with pytest.raises(ValueError, match='policy_target_type'):
        SampledEfficientZeroPolicy(bad_cfg, model=_TwoActionStubModel(), enable_field=['learn', 'collect'])

    cont_cfg = _make_policy_cfg(policy_target_type='completed_q', continuous=True)
    with pytest.raises(NotImplementedError, match='discrete'):
        SampledEfficientZeroPolicy(cont_cfg, model=_TwoActionStubModel(), enable_field=['eval'])


@pytest.mark.parametrize('mcts_ctree', [True, False])
def test_policy_completed_q_collect_and_eval_override_prior_trap(mcts_ctree):
    """With strong prior on losing action 0 and few sims (10), completed_q selects & targets winning action 1."""
    model = _TwoActionStubModel(
        action_space_size=6,
        root_logits=[2.2, 0.0, -10.0, -10.0, -10.0, -10.0],
        child_values={0: 0.4, 1: -0.8},  # in 2-player mode, action 1 has Q=+0.8 vs action 0 Q=-0.4
    )
    cfg = _make_policy_cfg(mcts_ctree=mcts_ctree, policy_target_type='completed_q')
    policy = SampledEfficientZeroPolicy(cfg, model=model, enable_field=['learn', 'collect', 'eval'])

    obs = torch.zeros((1, 4), dtype=torch.float32)
    mask = [np.array([1, 1, 0, 0, 0, 0], dtype=np.int8)]
    ready_env_id = np.array([0])

    # Eval mode: greedy argmax over completed_q must choose action 1 (winning) over action 0 (prior trap)
    eval_out = policy.eval_mode.forward(obs, action_mask=mask, to_play=[1], ready_env_id=ready_env_id)
    assert eval_out[0]['action'] == 1
    eval_dist = eval_out[0]['visit_count_distributions']
    assert len(eval_dist) == 4
    assert np.isclose(sum(eval_dist), 1.0, atol=1e-6)
    assert eval_dist[2] == 0.0 and eval_dist[3] == 0.0

    # Collect mode: stored policy target must put >50% mass on winning action 1
    collect_out = policy.collect_mode.forward(
        obs, action_mask=mask, temperature=1.0, to_play=[1], epsilon=0.0, ready_env_id=ready_env_id
    )
    collect_dist = collect_out[0]['visit_count_distributions']
    sampled = _extract_action_ints(collect_out[0]['root_sampled_actions'])
    prob_by_act = {sampled[j]: collect_dist[j] for j in range(2)}
    assert np.isclose(sum(collect_dist), 1.0, atol=1e-6)
    assert prob_by_act[1] > prob_by_act[0]
    assert prob_by_act[1] > 0.5


def test_policy_default_visit_count_unchanged():
    """Default policy_target_type='visit_count' continues emitting raw integer visit counts."""
    model = _TwoActionStubModel()
    cfg = _make_policy_cfg(mcts_ctree=True, policy_target_type='visit_count')
    policy = SampledEfficientZeroPolicy(cfg, model=model, enable_field=['learn', 'eval'])

    obs = torch.zeros((1, 4), dtype=torch.float32)
    mask = [np.array([1, 1, 0, 0, 0, 0], dtype=np.int8)]
    eval_out = policy.eval_mode.forward(obs, action_mask=mask, to_play=[1], ready_env_id=np.array([0]))
    dist = eval_out[0]['visit_count_distributions']
    assert sum(dist) == cfg.num_simulations
    assert all(isinstance(v, (int, np.integer)) for v in dist)


# ---------------------------------------------------------------------------
# 4. GameBuffer reanalyze integration tests for completed_q
# ---------------------------------------------------------------------------


class _ReanalyzeStubModel(_TwoActionStubModel):
    """Stub model whose initial_inference outputs categorical value logits for GameBuffer reanalyze."""

    def initial_inference(self, obs):
        out = super().initial_inference(obs)
        B = obs.shape[0]
        # Under DiscreteSupport(-1, 2, 1) -> support [-1, 0, 1], logits [0, 20, 0] decode to raw value 0.0
        out.value = torch.tensor([[0.0, 20.0, 0.0] for _ in range(B)], dtype=torch.float32)
        out.value_prefix = np.zeros((B, 1), dtype=np.float32)
        return out


def _make_reanalyze_buffer(policy_target_type='completed_q', continuous=False):
    from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer
    from lzero.policy import DiscreteSupport

    buf = object.__new__(SampledEfficientZeroGameBuffer)
    buf._cfg = EasyDict(
        dict(
            device='cpu',
            env_type='board_games',
            action_type='varied_action_space',
            mcts_ctree=True,
            mini_infer_size=16,
            num_unroll_steps=1,
            num_simulations=10,
            discount_factor=0.99,
            lstm_horizon_len=5,
            pb_c_base=19652,
            pb_c_init=1.25,
            value_delta_max=0.01,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.0,
            reanalyze_noise=False,
            policy_target_type=policy_target_type,
            completed_q_c_visit=50.0,
            completed_q_c_scale=1.0,
            model=dict(
                model_type='mlp',
                continuous_action_space=continuous,
                action_space_size=6,
                num_of_sampled_actions=4,
                categorical_distribution=False,
                value_support_range=(-1.0, 2.0, 1.0),
                reward_support_range=(-1.0, 2.0, 1.0),
            ),
        )
    )
    buf.value_support = DiscreteSupport(-1.0, 2.0, 1.0)
    return buf


def _make_reanalyze_context():
    # 1 game segment with num_unroll_steps=1 -> 2 transitions: step 0 valid (mask=1), step 1 padded (mask=0)
    policy_obs_list = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
    policy_mask = [1, 0]
    pos_in_game_segment_list = [0]
    batch_index_list = [0]
    child_visits = [[[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]]
    root_values = [[0.0, 0.0]]
    game_segment_lens = [1]
    action_mask_segment = [[np.array([1, 1, 0, 0, 0, 0], dtype=np.int8), np.array([1, 1, 0, 0, 0, 0], dtype=np.int8)]]
    to_play_segment = [[1, 2]]
    return (
        [
            policy_obs_list,
            policy_mask,
            pos_in_game_segment_list,
            batch_index_list,
            child_visits,
            root_values,
            game_segment_lens,
            action_mask_segment,
            to_play_segment,
        ],
        child_visits,
    )


def test_reanalyze_completed_q_overrides_prior_trap_and_updates_segment():
    """GameBuffer reanalyze with completed_q overrides prior trap and stores completed_q into child_visits."""
    model = _ReanalyzeStubModel(
        action_space_size=6,
        root_logits=[1.8, 0.0, -10.0, -10.0, -10.0, -10.0],
        # In 2-player mode, action 0 has Q=0.00 and action 1 has Q=+0.20:
        # strong prior on 0 gives it visit-count plurality at 10 sims, while action 1 is still visited
        # and has strictly higher Q so completed_q flips majority mass to action 1.
        child_values={0: 0.0, 1: -0.20},
    )
    model.eval()

    # 1. completed_q mode
    buf_cq = _make_reanalyze_buffer(policy_target_type='completed_q')
    ctx_cq, child_visits_cq = _make_reanalyze_context()
    target_policies_cq, sampled_cq = buf_cq._compute_target_policy_reanalyzed(ctx_cq, model)

    assert target_policies_cq.shape == (1, 2, 4)
    # Valid transition (step 0): winning action 1 must beat prior-trapped action 0 and trailing padding must be 0
    step0_policy = target_policies_cq[0, 0]
    step0_sampled = _extract_action_ints(sampled_cq[0])
    prob_by_act = {step0_sampled[j]: float(step0_policy[j]) for j in range(2)}
    assert np.isclose(np.sum(step0_policy), 1.0, atol=1e-6)
    assert step0_policy[2] == 0.0 and step0_policy[3] == 0.0
    assert prob_by_act[1] > prob_by_act[0]
    assert prob_by_act[1] > 0.5
    # GameSegment child_visits must also be updated with the completed_q distribution
    assert np.allclose(child_visits_cq[0][0], step0_policy, atol=1e-6)
    # Masked transition (step 1, policy_mask=0) must stay exact zeros
    assert np.all(target_policies_cq[0, 1] == 0.0)

    # 2. Default visit_count mode remains unchanged (prior-trapped action 0 still holds visit plurality)
    buf_vc = _make_reanalyze_buffer(policy_target_type='visit_count')
    ctx_vc, _ = _make_reanalyze_context()
    target_policies_vc, sampled_vc = buf_vc._compute_target_policy_reanalyzed(ctx_vc, model)
    vc_step0 = target_policies_vc[0, 0]
    vc_sampled = _extract_action_ints(sampled_vc[0])
    vc_by_act = {vc_sampled[j]: float(vc_step0[j]) for j in range(2)}
    assert np.isclose(np.sum(vc_step0), 1.0, atol=1e-6)
    assert vc_by_act[0] > vc_by_act[1]


def test_reanalyze_completed_q_validation():
    model = _ReanalyzeStubModel()
    model.eval()

    bad_buf = _make_reanalyze_buffer(policy_target_type='bad_mode')
    ctx_bad, _ = _make_reanalyze_context()
    with pytest.raises(ValueError, match='policy_target_type'):
        bad_buf._compute_target_policy_reanalyzed(ctx_bad, model)

    cont_buf = _make_reanalyze_buffer(policy_target_type='completed_q', continuous=True)
    ctx_cont, _ = _make_reanalyze_context()
    with pytest.raises(NotImplementedError, match='discrete'):
        cont_buf._compute_target_policy_reanalyzed(ctx_cont, model)
