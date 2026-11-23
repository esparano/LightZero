# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for Sampled EfficientZero MCTS ``search()`` across both
the Python implementation (``SampledEfficientZeroMCTSPtree``) and the compiled C++
implementation (``SampledEfficientZeroMCTSCtree``).

Covers:
1. Multi-step search expanding past depth 1 (non-root leaf nodes with ``legal_actions=None``).
2. Two-player minimax value negation at depth 1 (prefers opponent-losing over opponent-winning).
3. Two-player 2-ply minimax trap detection (avoids a move that allows an opponent winning reply).
4. LSTM reward hidden-state reset every ``lstm_horizon_len`` steps during search.
5. Behavioral parity between ``ptree`` and ``ctree`` on a deterministic two-player board game
   with sparse legal actions.
"""

import numpy as np
import pytest
import torch
from easydict import EasyDict

from lzero.mcts.tree_search.mcts_ctree_sampled import SampledEfficientZeroMCTSCtree
from lzero.mcts.tree_search.mcts_ptree_sampled import SampledEfficientZeroMCTSPtree


class _StubRecurrentOutput:
    def __init__(self, latent_state, policy_logits, value, value_prefix, reward_hidden_state):
        self.latent_state = latent_state
        self.policy_logits = policy_logits
        self.value = value
        self.value_prefix = value_prefix
        self.reward_hidden_state = reward_hidden_state


def _make_mcts_cfg(
    action_space_size=2,
    num_of_sampled_actions=2,
    num_simulations=20,
    lstm_horizon_len=5,
    discount_factor=1.0,
    env_type='board_games',
):
    return EasyDict(
        dict(
            device='cpu',
            env_type=env_type,
            discount_factor=discount_factor,
            num_simulations=num_simulations,
            lstm_horizon_len=lstm_horizon_len,
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
                value_support_range=(-10.0, 11.0, 1.0),
                reward_support_range=(-10.0, 11.0, 1.0),
            ),
        )
    )


def _extract_action_ints(sampled_actions_row):
    """Normalize sampled action entries from ptree (Action objects) or ctree (list/int)."""
    out = []
    for a in sampled_actions_row:
        if hasattr(a, 'value'):
            out.append(int(np.asarray(a.value).item()))
        elif isinstance(a, (list, tuple, np.ndarray)):
            out.append(int(np.asarray(a).reshape(-1)[0]))
        else:
            out.append(int(a))
    return out


def _best_action_from_roots(roots):
    sampled = _extract_action_ints(roots.get_sampled_actions()[0])
    visits = list(roots.get_distributions()[0])
    # Aggregate visits by action id in case of padded duplicates
    totals = {}
    for act, v in zip(sampled, visits):
        totals[act] = totals.get(act, 0) + v
    return max(totals.items(), key=lambda kv: kv[1])[0], totals


class _DepthTrackingStubModel(torch.nn.Module):
    """Stub model encoding (depth, first_action, second_action) in the latent state."""

    def __init__(self, action_space_size=2, value_fn=None, policy_fn=None, hidden_dim=4):
        super().__init__()
        self.action_space_size = action_space_size
        self.value_fn = value_fn or (lambda d, a1, a2: 0.0)
        self.policy_fn = policy_fn or (lambda d, a1, a2: [0.0] * action_space_size)
        self.hidden_dim = hidden_dim
        self.recorded_depths = []
        self.recorded_hidden_norms = []  # (depth_before_step, c_norm, h_norm)

    def recurrent_inference(self, latent_states, reward_hidden_state, actions):
        # latent_states shape: (B, 3) -> [depth, first_action, second_action]
        B = latent_states.shape[0]
        c_in, h_in = reward_hidden_state
        next_latents = []
        values = []
        value_prefixes = []
        policies = []

        for b in range(B):
            depth = int(round(float(latent_states[b, 0].item())))
            a1 = int(round(float(latent_states[b, 1].item())))
            a2 = int(round(float(latent_states[b, 2].item())))
            act = int(actions[b].reshape(-1)[0].item())

            self.recorded_depths.append(depth)
            self.recorded_hidden_norms.append(
                (depth, float(c_in[:, b, :].abs().sum().item()), float(h_in[:, b, :].abs().sum().item()))
            )

            next_depth = depth + 1
            next_a1 = act if depth == 0 else a1
            next_a2 = act if depth == 1 else a2
            next_latents.append([float(next_depth), float(next_a1), float(next_a2)])
            values.append([float(self.value_fn(next_depth, next_a1, next_a2))])
            value_prefixes.append([0.0])
            policies.append(self.policy_fn(next_depth, next_a1, next_a2))

        c_out = torch.ones((1, B, self.hidden_dim), dtype=torch.float32)
        h_out = torch.ones((1, B, self.hidden_dim), dtype=torch.float32)
        return _StubRecurrentOutput(
            latent_state=torch.tensor(next_latents, dtype=torch.float32),
            policy_logits=torch.tensor(policies, dtype=torch.float32),
            value=torch.tensor(values, dtype=torch.float32),
            value_prefix=torch.tensor(value_prefixes, dtype=torch.float32),
            reward_hidden_state=(c_out, h_out),
        )


def _run_search(mcts_cls, cfg, model, legal_actions, root_policy_logits, to_play_batch):
    roots = mcts_cls.roots(
        1,
        [legal_actions],
        cfg.model.action_space_size,
        cfg.model.num_of_sampled_actions,
        False,
    )
    roots.prepare_no_noise([0.0], [root_policy_logits], list(to_play_batch))

    latent_roots = np.array([[0.0, -1.0, -1.0]], dtype=np.float32)
    h_dim = model.hidden_dim
    reward_hidden_roots = (
        np.zeros((1, 1, h_dim), dtype=np.float32),
        np.zeros((1, 1, h_dim), dtype=np.float32),
    )

    mcts = mcts_cls(cfg)
    mcts.search(roots, model, latent_roots, reward_hidden_roots, to_play_batch=list(to_play_batch))
    return roots


@pytest.mark.unittest
@pytest.mark.parametrize('mcts_cls', [SampledEfficientZeroMCTSPtree, SampledEfficientZeroMCTSCtree])
class TestSEZMCTSSearch:

    def test_search_expands_past_depth_one(self, mcts_cls):
        """Non-root leaf nodes (where legal_actions is None) must expand so search
        proceeds past depth 1."""
        torch.manual_seed(0)
        cfg = _make_mcts_cfg(action_space_size=2, num_of_sampled_actions=2, num_simulations=12)
        model = _DepthTrackingStubModel(action_space_size=2)
        _run_search(mcts_cls, cfg, model, legal_actions=[0, 1], root_policy_logits=[0.0, 0.0], to_play_batch=[1])

        # Depth 0 = root->child; depth >= 1 = child->grandchild (requires non-root expansion!)
        assert max(model.recorded_depths) >= 2, f"Expected depth >= 2, got max {max(model.recorded_depths)}"

    def test_two_player_minimax_prefers_opponent_losing_move(self, mcts_cls):
        """At depth 1 (Player 2's turn):
        - Action 0 leaves P2 with value -0.9 (P1 winning +0.9).
        - Action 1 leaves P2 with value +0.9 (P2 winning, P1 losing -0.9).
        Minimax backup must negate P2's values so P1 picks Action 0 and gets a positive root value.
        """
        torch.manual_seed(0)
        cfg = _make_mcts_cfg(action_space_size=2, num_of_sampled_actions=2, num_simulations=24)

        def value_fn(depth, a1, a2):
            if depth % 2 == 1:  # Player 2's perspective
                return -0.9 if a1 == 0 else +0.9
            else:  # Player 1's perspective
                return +0.9 if a1 == 0 else -0.9

        model = _DepthTrackingStubModel(action_space_size=2, value_fn=value_fn)
        roots = _run_search(
            mcts_cls, cfg, model, legal_actions=[0, 1], root_policy_logits=[0.0, 0.0], to_play_batch=[1]
        )

        best_act, totals = _best_action_from_roots(roots)
        root_val = float(roots.get_values()[0])
        assert best_act == 0, f"Expected winning action 0, got {best_act} with visits {totals}"
        assert totals[0] > totals[1], f"Expected more visits on action 0, got {totals}"
        assert root_val > 0.3, f"Expected positive root value for P1, got {root_val}"

    def test_two_player_depth2_avoids_trap_with_opponent_winning_reply(self, mcts_cls):
        """2-ply minimax trap:
        - Action 0: neutral at depth 1 (P2 val = 0.0), but every P2 reply gives P1 +0.5.
        - Action 1: tempting at depth 1 (P2 val = -0.2), BUT P2 has a killer reply (a2=1)
          that gives P1 -1.0 (P2 wins!).
        With enough simulations, MCTS must discover P2's killer reply and choose Action 0.
        """
        torch.manual_seed(0)
        cfg = _make_mcts_cfg(action_space_size=2, num_of_sampled_actions=2, num_simulations=50)

        def value_fn(depth, a1, a2):
            if depth == 1:  # Player 2's turn
                return 0.0 if a1 == 0 else -0.2
            # depth >= 2: alternating perspectives
            if a1 == 0:
                p1_val = +0.5
            else:
                p1_val = -1.0 if a2 == 1 else +0.2
            return p1_val if depth % 2 == 0 else -p1_val

        model = _DepthTrackingStubModel(action_space_size=2, value_fn=value_fn)
        roots = _run_search(
            mcts_cls, cfg, model, legal_actions=[0, 1], root_policy_logits=[0.0, 0.0], to_play_batch=[1]
        )

        best_act, totals = _best_action_from_roots(roots)
        assert best_act == 0, f"MCTS fell into 2-ply trap! Visits: {totals}"

    def test_lstm_hidden_state_resets_at_horizon_len(self, mcts_cls):
        """With K=1 (forced single line) and lstm_horizon_len=2:
        - depth 0 input: initial zero hidden state
        - depth 1 input: non-zero hidden state carried from depth 0
        - depth 2 input: RESET to zero (because search_len=2 % 2 == 0 after depth 1)
        - depth 3 input: non-zero hidden state carried from depth 2
        """
        torch.manual_seed(0)
        cfg = _make_mcts_cfg(
            action_space_size=1, num_of_sampled_actions=1, num_simulations=5, lstm_horizon_len=2
        )
        model = _DepthTrackingStubModel(action_space_size=1)
        _run_search(mcts_cls, cfg, model, legal_actions=[0], root_policy_logits=[0.0], to_play_batch=[1])

        by_depth = {d: (c_norm, h_norm) for d, c_norm, h_norm in model.recorded_hidden_norms}
        assert by_depth[0] == (0.0, 0.0), f"Root hidden state should be zero, got {by_depth[0]}"
        assert by_depth[1][0] > 0.0 and by_depth[1][1] > 0.0, f"Depth 1 should carry hidden state, got {by_depth[1]}"
        assert by_depth[2] == (0.0, 0.0), f"Depth 2 should reset hidden state at horizon=2, got {by_depth[2]}"
        assert by_depth[3][0] > 0.0 and by_depth[3][1] > 0.0, f"Depth 3 should carry hidden state, got {by_depth[3]}"

    def test_search_respects_policy_priors_when_values_equal(self, mcts_cls):
        """When all leaf values are 0.0 and root logit for action 1 is much larger
        than action 0 (+5.0 vs -5.0), PUCT must concentrate visits on action 1.
        Catches the C++ cnode.cpp std::string::compare('density') bug that forced uniform priors.
        """
        torch.manual_seed(0)
        cfg = _make_mcts_cfg(action_space_size=2, num_of_sampled_actions=2, num_simulations=20)
        model = _DepthTrackingStubModel(action_space_size=2, value_fn=lambda d, a1, a2: 0.0)
        roots = _run_search(
            mcts_cls, cfg, model, legal_actions=[0, 1], root_policy_logits=[-5.0, 5.0], to_play_batch=[1]
        )
        best_act, totals = _best_action_from_roots(roots)
        assert best_act == 1, f"Expected prior-favored action 1, got {best_act} with visits {totals}"
        assert totals[1] >= 15, f"Expected >= 15/20 visits on high-prior action 1, got {totals}"


@pytest.mark.unittest
def test_ptree_and_ctree_parity_on_two_player_board_game():
    """Verify Python ptree and C++ ctree agree on best action, legal-action masking,
    and root value on the exact same deterministic two-player board-game stub."""
    cfg = _make_mcts_cfg(action_space_size=5, num_of_sampled_actions=2, num_simulations=30)

    def value_fn(depth, a1, a2):
        # Among legal actions {1, 3}, action 3 is strongly winning for P1; illegal actions 0,2,4
        # have huge bait values that must NEVER be selected at the root.
        if a1 == 3:
            p1_val = +0.8
        elif a1 == 1:
            p1_val = -0.8
        else:
            p1_val = +10.0  # illegal bait
        return p1_val if depth % 2 == 0 else -p1_val

    # Give illegal actions huge root logits to ensure both ptree & ctree mask them out.
    root_logits = [100.0, 0.0, 100.0, 0.0, 100.0]

    torch.manual_seed(0)
    ptree_roots = _run_search(
        SampledEfficientZeroMCTSPtree,
        cfg,
        _DepthTrackingStubModel(action_space_size=5, value_fn=value_fn),
        legal_actions=[1, 3],
        root_policy_logits=root_logits,
        to_play_batch=[1],
    )
    torch.manual_seed(0)
    ctree_roots = _run_search(
        SampledEfficientZeroMCTSCtree,
        cfg,
        _DepthTrackingStubModel(action_space_size=5, value_fn=value_fn),
        legal_actions=[1, 3],
        root_policy_logits=root_logits,
        to_play_batch=[1],
    )

    p_best, p_totals = _best_action_from_roots(ptree_roots)
    c_best, c_totals = _best_action_from_roots(ctree_roots)
    p_val = float(ptree_roots.get_values()[0])
    c_val = float(ctree_roots.get_values()[0])

    assert set(p_totals.keys()).issubset({1, 3}), f"ptree sampled illegal action: {p_totals}"
    assert set(c_totals.keys()).issubset({1, 3}), f"ctree sampled illegal action: {c_totals}"
    assert p_best == 3 and c_best == 3, f"Expected both to pick 3; ptree={p_totals}, ctree={c_totals}"
    assert abs(p_val - c_val) < 0.1, f"ptree ({p_val}) and ctree ({c_val}) root values diverged!"
