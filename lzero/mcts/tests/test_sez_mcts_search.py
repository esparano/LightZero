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


def _legacy_ctree_search_reference(mcts, roots, model, latent_state_roots_np, reward_hidden_state_roots_np, to_play_batch):
    """Exact pre-optimization CPU-list reference implementation of SampledEfficientZeroMCTSCtree.search
    used to verify bit-for-bit parity of the device-resident latent pool.
    """
    import copy
    from lzero.mcts.ctree.ctree_sampled_efficientzero import ezs_tree as tree_sampled_efficientzero
    from lzero.policy import to_detach_cpu_numpy

    recorded_inputs = []
    recorded_backprop = []
    with torch.no_grad():
        model.eval()
        batch_size = roots.num
        device = mcts._cfg.device
        pb_c_base, pb_c_init, discount_factor = mcts._cfg.pb_c_base, mcts._cfg.pb_c_init, mcts._cfg.discount_factor

        latent_state_batch_in_search_path = [latent_state_roots_np]
        reward_hidden_state_c_pool = [reward_hidden_state_roots_np[0]]
        reward_hidden_state_h_pool = [reward_hidden_state_roots_np[1]]

        min_max_stats_lst = tree_sampled_efficientzero.MinMaxStatsList(batch_size)
        min_max_stats_lst.set_delta(mcts._cfg.value_delta_max)

        for simulation_index in range(mcts._cfg.num_simulations):
            latent_states = []
            hidden_states_c_reward = []
            hidden_states_h_reward = []
            results = tree_sampled_efficientzero.ResultsWrapper(num=batch_size)

            if mcts._cfg.env_type == 'not_board_games':
                ix_list, iy_list, last_actions, virtual_to_play_batch = tree_sampled_efficientzero.batch_traverse(
                    roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                    to_play_batch, mcts._cfg.model.continuous_action_space,
                )
            else:
                ix_list, iy_list, last_actions, virtual_to_play_batch = tree_sampled_efficientzero.batch_traverse(
                    roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                    copy.deepcopy(to_play_batch), mcts._cfg.model.continuous_action_space,
                )

            search_lens = results.get_search_len()
            for ix, iy in zip(ix_list, iy_list):
                latent_states.append(latent_state_batch_in_search_path[ix][iy])
                hidden_states_c_reward.append(reward_hidden_state_c_pool[ix][0][iy])
                hidden_states_h_reward.append(reward_hidden_state_h_pool[ix][0][iy])
            latent_states_t = torch.from_numpy(np.asarray(latent_states)).to(device)
            c_t = torch.from_numpy(np.asarray(hidden_states_c_reward)).to(device).unsqueeze(0)
            h_t = torch.from_numpy(np.asarray(hidden_states_h_reward)).to(device).unsqueeze(0)
            if mcts._cfg.model.continuous_action_space:
                act_t = torch.from_numpy(np.asarray(last_actions)).to(device)
            else:
                act_t = torch.from_numpy(np.asarray(last_actions)).to(device).long()

            recorded_inputs.append((
                latent_states_t.detach().cpu().numpy().copy(),
                c_t.detach().cpu().numpy().copy(),
                h_t.detach().cpu().numpy().copy(),
                act_t.detach().cpu().numpy().copy(),
            ))

            network_output = model.recurrent_inference(latent_states_t, (c_t, h_t), act_t)
            [lat_np, logits_np, val_np, vp_np] = to_detach_cpu_numpy([
                network_output.latent_state,
                network_output.policy_logits,
                mcts.value_inverse_scalar_transform_handle(network_output.value),
                mcts.reward_inverse_scalar_transform_handle(network_output.value_prefix),
            ])
            rhs_np = (
                network_output.reward_hidden_state[0].detach().cpu().numpy(),
                network_output.reward_hidden_state[1].detach().cpu().numpy(),
            )
            latent_state_batch_in_search_path.append(lat_np)
            vp_pool = vp_np.reshape(-1).tolist()
            val_pool = val_np.reshape(-1).tolist()
            logits_pool = logits_np.tolist()

            reset_idx = (np.array(search_lens) % mcts._cfg.lstm_horizon_len == 0)
            rhs_np[0][:, reset_idx, :] = 0
            rhs_np[1][:, reset_idx, :] = 0
            is_reset_list = reset_idx.astype(np.int32).tolist()
            reward_hidden_state_c_pool.append(rhs_np[0])
            reward_hidden_state_h_pool.append(rhs_np[1])

            recorded_backprop.append((vp_pool, val_pool, logits_pool, is_reset_list))
            tree_sampled_efficientzero.batch_backpropagate(
                simulation_index + 1, discount_factor, vp_pool, val_pool, logits_pool,
                min_max_stats_lst, results, is_reset_list, virtual_to_play_batch,
            )

    return recorded_inputs, recorded_backprop


def _parity_devices():
    devs = ['cpu']
    if torch.cuda.is_available():
        devs.append('cuda')
    return devs


@pytest.mark.unittest
@pytest.mark.parametrize('device_str', _parity_devices())
@pytest.mark.parametrize('pass_tensors', [True, False])
def test_device_resident_latent_pool_bit_for_bit_parity_deterministic(device_str, pass_tensors):
    """Verify device-resident MCTS latent pool matches the legacy CPU-list search
    BIT-FOR-BIT end-to-end on a real SampledEfficientZeroModel (conv + LSTM),
    across both CPU and CUDA and both tensor and NumPy root inputs.

    Uses K=1 (1 action) so C++ Efraimidis-Spirakis action ordering is 100% deterministic
    and independent roots_ref / roots_new trees can be compared directly without trace replay.
    """
    from lzero.model.sampled_efficientzero_model import SampledEfficientZeroModel

    torch.manual_seed(1234)
    np.random.seed(1234)
    B = 6
    grid_h, grid_w = 4, 4
    C = 12
    action_space_size = 1
    k_samp = 1
    num_sims = 12
    lstm_h = 3  # exercises multiple LSTM horizon resets within 12 sims

    model = SampledEfficientZeroModel(
        observation_shape=(C, grid_h, grid_w),
        action_space_size=action_space_size,
        num_res_blocks=1,
        num_channels=16,
        reward_head_channels=4,
        value_head_channels=4,
        policy_head_channels=4,
        reward_head_hidden_channels=[16],
        value_head_hidden_channels=[16],
        policy_head_hidden_channels=[16],
        reward_support_range=(-5.0, 6.0, 1.0),
        value_support_range=(-5.0, 6.0, 1.0),
        categorical_distribution=True,
        last_linear_layer_init_zero=False,
        downsample=False,
        lstm_hidden_size=32,
        continuous_action_space=False,
        discrete_action_encoding_type='one_hot',
        norm_type='LN',
        state_norm=True,
    ).to(device_str)
    model.eval()

    cfg = _make_mcts_cfg(
        action_space_size=action_space_size,
        num_of_sampled_actions=k_samp,
        num_simulations=num_sims,
        lstm_horizon_len=lstm_h,
        discount_factor=0.99,
        env_type='board_games',
    )
    cfg.device = device_str
    cfg.model.categorical_distribution = True
    cfg.model.value_support_range = (-5.0, 6.0, 1.0)
    cfg.model.reward_support_range = (-5.0, 6.0, 1.0)

    obs = torch.randn((B, C, grid_h, grid_w), device=device_str)
    with torch.no_grad():
        init_out = model.initial_inference(obs)
    root_lat_t = init_out.latent_state.detach().clone()
    root_rhs_t = (
        init_out.reward_hidden_state[0].detach().clone(),
        init_out.reward_hidden_state[1].detach().clone(),
    )
    root_vp = [0.0] * B
    root_logits = init_out.policy_logits.detach().cpu().numpy().tolist()
    legal_actions = [list(range(action_space_size)) for _ in range(B)]
    to_play_batch = [1 if i % 2 == 0 else 2 for i in range(B)]

    # 1. Run exact legacy CPU-list reference
    mcts_ref = SampledEfficientZeroMCTSCtree(cfg)
    roots_ref = SampledEfficientZeroMCTSCtree.roots(B, legal_actions, action_space_size, k_samp, False)
    roots_ref.prepare_no_noise(root_vp, root_logits, list(to_play_batch))
    ref_inputs, _ = _legacy_ctree_search_reference(
        mcts_ref,
        roots_ref,
        model,
        root_lat_t.cpu().numpy().copy(),
        (root_rhs_t[0].cpu().numpy().copy(), root_rhs_t[1].cpu().numpy().copy()),
        list(to_play_batch),
    )

    # 2. Run new device-resident SampledEfficientZeroMCTSCtree.search, spying on recurrent_inference inputs
    new_inputs = []
    orig_rec = model.recurrent_inference

    def _spy_rec(lat, rhs, act):
        new_inputs.append((
            lat.detach().cpu().numpy().copy(),
            rhs[0].detach().cpu().numpy().copy(),
            rhs[1].detach().cpu().numpy().copy(),
            act.detach().cpu().numpy().copy(),
        ))
        return orig_rec(lat, rhs, act)

    model.recurrent_inference = _spy_rec
    try:
        mcts_new = SampledEfficientZeroMCTSCtree(cfg)
        roots_new = SampledEfficientZeroMCTSCtree.roots(B, legal_actions, action_space_size, k_samp, False)
        roots_new.prepare_no_noise(root_vp, root_logits, list(to_play_batch))
        lat_in = root_lat_t if pass_tensors else root_lat_t.cpu().numpy().copy()
        rhs_in = (
            root_rhs_t
            if pass_tensors
            else (root_rhs_t[0].cpu().numpy().copy(), root_rhs_t[1].cpu().numpy().copy())
        )
        mcts_new.search(roots_new, model, lat_in, rhs_in, to_play_batch=list(to_play_batch))
    finally:
        model.recurrent_inference = orig_rec

    assert len(new_inputs) == len(ref_inputs) == num_sims
    for sim_i in range(num_sims):
        for tensor_i in range(4):
            np.testing.assert_array_equal(new_inputs[sim_i][tensor_i], ref_inputs[sim_i][tensor_i])

    assert roots_new.get_distributions() == roots_ref.get_distributions()
    assert roots_new.get_sampled_actions() == roots_ref.get_sampled_actions()
    assert roots_new.get_values() == roots_ref.get_values()
    assert roots_new.get_children_q_values(0.99) == roots_ref.get_children_q_values(0.99)


@pytest.mark.unittest
@pytest.mark.parametrize('device_str', _parity_devices())
def test_device_resident_latent_pool_bit_for_bit_parity_galcon_4097_factored(device_str):
    """Lockstep 50-sim replay parity on full 8x8 Galcon (C=40, 4097 actions, factored_spatial head,
    K=16, B=8, lstm_horizon_len=5): verifies every recurrent_inference input and every backprop
    payload produced by the GPU/device-resident pool matches the legacy CPU-list pool BIT-FOR-BIT.
    """
    from lzero.mcts.ctree.ctree_sampled_efficientzero import ezs_tree as tree_sampled_efficientzero
    from lzero.model.sampled_efficientzero_model import SampledEfficientZeroModel
    from lzero.policy import to_detach_cpu_numpy

    torch.manual_seed(77)
    np.random.seed(77)
    B = 8
    C, H, W = 40, 8, 8
    action_space_size = (H * W) ** 2 + 1  # 4097
    k_samp = 16
    num_sims = 50
    lstm_h = 2  # triggers partial-batch LSTM horizon resets across heterogeneous branch depths

    model = SampledEfficientZeroModel(
        observation_shape=(C, H, W),
        action_space_size=action_space_size,
        num_res_blocks=1,
        num_channels=32,
        reward_head_channels=8,
        value_head_channels=8,
        policy_head_channels=8,
        reward_head_hidden_channels=[32],
        value_head_hidden_channels=[32],
        policy_head_hidden_channels=[32],
        reward_support_range=(-10.0, 11.0, 1.0),
        value_support_range=(-10.0, 11.0, 1.0),
        categorical_distribution=True,
        last_linear_layer_init_zero=False,
        downsample=False,
        lstm_hidden_size=64,
        continuous_action_space=False,
        discrete_action_encoding_type='one_hot',
        norm_type='GN',
        state_norm=True,
        policy_head_type='factored_spatial',
        policy_head_embed_dim=16,
    ).to(device_str)
    model.eval()

    cfg = _make_mcts_cfg(
        action_space_size=action_space_size,
        num_of_sampled_actions=k_samp,
        num_simulations=num_sims,
        lstm_horizon_len=lstm_h,
        discount_factor=0.99,
        env_type='board_games',
    )
    cfg.device = device_str
    cfg.model.categorical_distribution = True
    cfg.model.value_support_range = (-10.0, 11.0, 1.0)
    cfg.model.reward_support_range = (-10.0, 11.0, 1.0)

    obs = torch.randn((B, C, H, W), device=device_str)
    with torch.no_grad():
        init_out = model.initial_inference(obs)
    root_lat_t = init_out.latent_state.detach().clone()
    root_rhs_t = (
        init_out.reward_hidden_state[0].detach().clone(),
        init_out.reward_hidden_state[1].detach().clone(),
    )
    root_vp = [0.0] * B
    root_logits = init_out.policy_logits.detach().cpu().numpy().tolist()
    # Sparse legal actions like real Galcon (~25 legal moves per env)
    rng = np.random.default_rng(77)
    legal_actions = [sorted([0] + rng.choice(np.arange(1, action_space_size), size=24, replace=False).tolist()) for _ in range(B)]
    to_play_batch = [1 if i % 2 == 0 else 2 for i in range(B)]

    # Capture exact traversal trace + model inputs + backprop payloads during real device-resident search
    trace_steps = []
    new_rec_inputs = []
    new_backprop_payloads = []

    orig_traverse = tree_sampled_efficientzero.batch_traverse
    orig_backprop = tree_sampled_efficientzero.batch_backpropagate
    orig_rec = model.recurrent_inference

    def _spy_traverse(*args, **kwargs):
        res = orig_traverse(*args, **kwargs)
        results_wrapper = args[5]
        trace_steps.append((list(res[0]), list(res[1]), list(res[2]), list(results_wrapper.get_search_len())))
        return res

    def _spy_rec(lat, rhs, act):
        new_rec_inputs.append((
            lat.detach().cpu().numpy().copy(),
            rhs[0].detach().cpu().numpy().copy(),
            rhs[1].detach().cpu().numpy().copy(),
            act.detach().cpu().numpy().copy(),
        ))
        return orig_rec(lat, rhs, act)

    def _spy_backprop(cur_idx, df, vp_pool, val_pool, logits_pool, mms, results, is_reset_list, vtp):
        vp_list = vp_pool.tolist() if isinstance(vp_pool, np.ndarray) else list(vp_pool)
        val_list = val_pool.tolist() if isinstance(val_pool, np.ndarray) else list(val_pool)
        logits_list = logits_pool.tolist() if isinstance(logits_pool, np.ndarray) else [list(r) for r in logits_pool]
        rst_list = is_reset_list.tolist() if isinstance(is_reset_list, np.ndarray) else list(is_reset_list)
        new_backprop_payloads.append((vp_list, val_list, logits_list, rst_list))
        return orig_backprop(cur_idx, df, vp_pool, val_pool, logits_pool, mms, results, is_reset_list, vtp)

    tree_sampled_efficientzero.batch_traverse = _spy_traverse
    tree_sampled_efficientzero.batch_backpropagate = _spy_backprop
    model.recurrent_inference = _spy_rec
    try:
        mcts = SampledEfficientZeroMCTSCtree(cfg)
        roots = SampledEfficientZeroMCTSCtree.roots(B, legal_actions, action_space_size, k_samp, False)
        roots.prepare_no_noise(root_vp, root_logits, list(to_play_batch))
        mcts.search(roots, model, root_lat_t, root_rhs_t, to_play_batch=list(to_play_batch))
    finally:
        tree_sampled_efficientzero.batch_traverse = orig_traverse
        tree_sampled_efficientzero.batch_backpropagate = orig_backprop
        model.recurrent_inference = orig_rec

    assert len(trace_steps) == num_sims

    # Replay the exact same 50-sim traversal trace through the legacy CPU-list pool
    lat_pool_cpu = [root_lat_t.cpu().numpy().copy()]
    c_pool_cpu = [root_rhs_t[0].cpu().numpy().copy()]
    h_pool_cpu = [root_rhs_t[1].cpu().numpy().copy()]

    saw_partial_reset = False
    with torch.no_grad():
        for sim_i, (ix_list, iy_list, last_actions, search_lens) in enumerate(trace_steps):
            lat_ref = torch.from_numpy(np.asarray([lat_pool_cpu[ix][iy] for ix, iy in zip(ix_list, iy_list)])).to(device_str)
            c_ref = torch.from_numpy(np.asarray([c_pool_cpu[ix][0][iy] for ix, iy in zip(ix_list, iy_list)])).to(device_str).unsqueeze(0)
            h_ref = torch.from_numpy(np.asarray([h_pool_cpu[ix][0][iy] for ix, iy in zip(ix_list, iy_list)])).to(device_str).unsqueeze(0)
            act_ref = torch.from_numpy(np.asarray(last_actions)).to(device_str).long()

            # 1. Exact bit-for-bit input parity to recurrent_inference
            np.testing.assert_array_equal(new_rec_inputs[sim_i][0], lat_ref.cpu().numpy())
            np.testing.assert_array_equal(new_rec_inputs[sim_i][1], c_ref.cpu().numpy())
            np.testing.assert_array_equal(new_rec_inputs[sim_i][2], h_ref.cpu().numpy())
            np.testing.assert_array_equal(new_rec_inputs[sim_i][3], act_ref.cpu().numpy())

            out_ref = model.recurrent_inference(lat_ref, (c_ref, h_ref), act_ref)
            [lat_np, logits_np, val_np, vp_np] = to_detach_cpu_numpy([
                out_ref.latent_state,
                out_ref.policy_logits,
                mcts.value_inverse_scalar_transform_handle(out_ref.value),
                mcts.reward_inverse_scalar_transform_handle(out_ref.value_prefix),
            ])
            rhs_np = (
                out_ref.reward_hidden_state[0].detach().cpu().numpy(),
                out_ref.reward_hidden_state[1].detach().cpu().numpy(),
            )
            reset_idx = (np.array(search_lens) % lstm_h == 0)
            if 0 < int(np.sum(reset_idx)) < B:
                saw_partial_reset = True
            rhs_np[0][:, reset_idx, :] = 0
            rhs_np[1][:, reset_idx, :] = 0
            lat_pool_cpu.append(lat_np)
            c_pool_cpu.append(rhs_np[0])
            h_pool_cpu.append(rhs_np[1])

            # 2. Exact bit-for-bit backprop payload parity (value_prefix, value, policy_logits, is_reset)
            got_vp, got_val, got_logits, got_reset = new_backprop_payloads[sim_i]
            assert got_vp == vp_np.reshape(-1).tolist()
            assert got_val == val_np.reshape(-1).tolist()
            assert got_logits == logits_np.tolist()
            assert got_reset == reset_idx.astype(np.int32).tolist()

    assert saw_partial_reset, "Expected at least one simulation with partial-batch LSTM reset (0 < resets < B)"


@pytest.mark.unittest
@pytest.mark.parametrize('device_str', _parity_devices())
def test_device_resident_latent_pool_bit_for_bit_parity_continuous(device_str):
    """Verify continuous_action_space=True path (not_board_games + float action vectors) matches
    the legacy CPU-list search BIT-FOR-BIT on both CPU and CUDA."""
    from lzero.mcts.ctree.ctree_sampled_efficientzero import ezs_tree as tree_sampled_efficientzero
    from lzero.model.sampled_efficientzero_model_mlp import SampledEfficientZeroModelMLP
    from lzero.policy import to_detach_cpu_numpy

    torch.manual_seed(99)
    np.random.seed(99)
    B = 4
    obs_dim = 8
    act_dim = 3
    k_samp = 4
    num_sims = 12
    lstm_h = 2

    model = SampledEfficientZeroModelMLP(
        observation_shape=obs_dim,
        action_space_size=act_dim,
        latent_state_dim=16,
        reward_head_hidden_channels=[16],
        value_head_hidden_channels=[16],
        policy_head_hidden_channels=[16],
        reward_support_range=(-5.0, 6.0, 1.0),
        value_support_range=(-5.0, 6.0, 1.0),
        categorical_distribution=True,
        last_linear_layer_init_zero=False,
        lstm_hidden_size=16,
        continuous_action_space=True,
        norm_type='LN',
        state_norm=True,
    ).to(device_str)
    model.eval()

    cfg = _make_mcts_cfg(
        action_space_size=act_dim,
        num_of_sampled_actions=k_samp,
        num_simulations=num_sims,
        lstm_horizon_len=lstm_h,
        discount_factor=0.99,
        env_type='not_board_games',
    )
    cfg.device = device_str
    cfg.model.continuous_action_space = True
    cfg.model.categorical_distribution = True
    cfg.model.value_support_range = (-5.0, 6.0, 1.0)
    cfg.model.reward_support_range = (-5.0, 6.0, 1.0)

    obs = torch.randn((B, obs_dim), device=device_str)
    with torch.no_grad():
        init_out = model.initial_inference(obs)
    root_lat_t = init_out.latent_state.detach().clone()
    root_rhs_t = (
        init_out.reward_hidden_state[0].detach().clone(),
        init_out.reward_hidden_state[1].detach().clone(),
    )
    root_vp = [0.0] * B
    root_logits = init_out.policy_logits.detach().cpu().numpy().tolist()
    legal_actions = [[-1] * k_samp for _ in range(B)]
    to_play_batch = [-1] * B

    trace_steps = []
    new_rec_inputs = []
    new_backprop_payloads = []

    orig_traverse = tree_sampled_efficientzero.batch_traverse
    orig_backprop = tree_sampled_efficientzero.batch_backpropagate
    orig_rec = model.recurrent_inference

    def _spy_traverse(*args, **kwargs):
        res = orig_traverse(*args, **kwargs)
        results_wrapper = args[5]
        trace_steps.append((list(res[0]), list(res[1]), [list(a) for a in res[2]], list(results_wrapper.get_search_len())))
        return res

    def _spy_rec(lat, rhs, act):
        new_rec_inputs.append((
            lat.detach().cpu().numpy().copy(),
            rhs[0].detach().cpu().numpy().copy(),
            rhs[1].detach().cpu().numpy().copy(),
            act.detach().cpu().numpy().copy(),
        ))
        return orig_rec(lat, rhs, act)

    def _spy_backprop(cur_idx, df, vp_pool, val_pool, logits_pool, mms, results, is_reset_list, vtp):
        vp_list = vp_pool.tolist() if isinstance(vp_pool, np.ndarray) else list(vp_pool)
        val_list = val_pool.tolist() if isinstance(val_pool, np.ndarray) else list(val_pool)
        logits_list = logits_pool.tolist() if isinstance(logits_pool, np.ndarray) else [list(r) for r in logits_pool]
        rst_list = is_reset_list.tolist() if isinstance(is_reset_list, np.ndarray) else list(is_reset_list)
        new_backprop_payloads.append((vp_list, val_list, logits_list, rst_list))
        return orig_backprop(cur_idx, df, vp_pool, val_pool, logits_pool, mms, results, is_reset_list, vtp)

    tree_sampled_efficientzero.batch_traverse = _spy_traverse
    tree_sampled_efficientzero.batch_backpropagate = _spy_backprop
    model.recurrent_inference = _spy_rec
    try:
        mcts = SampledEfficientZeroMCTSCtree(cfg)
        roots = SampledEfficientZeroMCTSCtree.roots(B, legal_actions, act_dim, k_samp, True)
        roots.prepare_no_noise(root_vp, root_logits, list(to_play_batch))
        mcts.search(roots, model, root_lat_t, root_rhs_t, to_play_batch=list(to_play_batch))
    finally:
        tree_sampled_efficientzero.batch_traverse = orig_traverse
        tree_sampled_efficientzero.batch_backpropagate = orig_backprop
        model.recurrent_inference = orig_rec

    lat_pool_cpu = [root_lat_t.cpu().numpy().copy()]
    c_pool_cpu = [root_rhs_t[0].cpu().numpy().copy()]
    h_pool_cpu = [root_rhs_t[1].cpu().numpy().copy()]

    with torch.no_grad():
        for sim_i, (ix_list, iy_list, last_actions, search_lens) in enumerate(trace_steps):
            lat_ref = torch.from_numpy(np.asarray([lat_pool_cpu[ix][iy] for ix, iy in zip(ix_list, iy_list)])).to(device_str)
            c_ref = torch.from_numpy(np.asarray([c_pool_cpu[ix][0][iy] for ix, iy in zip(ix_list, iy_list)])).to(device_str).unsqueeze(0)
            h_ref = torch.from_numpy(np.asarray([h_pool_cpu[ix][0][iy] for ix, iy in zip(ix_list, iy_list)])).to(device_str).unsqueeze(0)
            act_ref = torch.from_numpy(np.asarray(last_actions)).to(device_str)

            np.testing.assert_array_equal(new_rec_inputs[sim_i][0], lat_ref.cpu().numpy())
            np.testing.assert_array_equal(new_rec_inputs[sim_i][1], c_ref.cpu().numpy())
            np.testing.assert_array_equal(new_rec_inputs[sim_i][2], h_ref.cpu().numpy())
            np.testing.assert_array_equal(new_rec_inputs[sim_i][3], act_ref.cpu().numpy())

            out_ref = model.recurrent_inference(lat_ref, (c_ref, h_ref), act_ref)
            [lat_np, logits_np, val_np, vp_np] = to_detach_cpu_numpy([
                out_ref.latent_state,
                out_ref.policy_logits,
                mcts.value_inverse_scalar_transform_handle(out_ref.value),
                mcts.reward_inverse_scalar_transform_handle(out_ref.value_prefix),
            ])
            rhs_np = (
                out_ref.reward_hidden_state[0].detach().cpu().numpy(),
                out_ref.reward_hidden_state[1].detach().cpu().numpy(),
            )
            reset_idx = (np.array(search_lens) % lstm_h == 0)
            rhs_np[0][:, reset_idx, :] = 0
            rhs_np[1][:, reset_idx, :] = 0
            lat_pool_cpu.append(lat_np)
            c_pool_cpu.append(rhs_np[0])
            h_pool_cpu.append(rhs_np[1])

            got_vp, got_val, got_logits, got_reset = new_backprop_payloads[sim_i]
            assert got_vp == vp_np.reshape(-1).tolist()
            assert got_val == val_np.reshape(-1).tolist()
            assert got_logits == logits_np.tolist()
            assert got_reset == reset_idx.astype(np.int32).tolist()


@pytest.mark.unittest
def test_batch_backpropagate_numpy_strided_matches_list_bit_for_bit():
    """Verify zero-copy strided float32 NumPy views passed to ezs_tree.batch_backpropagate
    produce bit-for-bit identical tree state to the legacy Python list path."""
    from lzero.mcts.ctree.ctree_sampled_efficientzero import ezs_tree

    B = 4
    A = 1
    K = 1
    legal = [[0] for _ in range(B)]
    root_vp = [0.0] * B
    root_logits = [[0.0] for _ in range(B)]
    to_play = [1, 2, 1, 2]

    roots_np = ezs_tree.Roots(B, legal, A, K, False)
    roots_list = ezs_tree.Roots(B, legal, A, K, False)
    roots_np.prepare_no_noise(root_vp, root_logits, list(to_play))
    roots_list.prepare_no_noise(root_vp, root_logits, list(to_play))

    mms_np = ezs_tree.MinMaxStatsList(B)
    mms_list = ezs_tree.MinMaxStatsList(B)
    mms_np.set_delta(0.01)
    mms_list.set_delta(0.01)

    rng = np.random.default_rng(123)
    for sim in range(8):
        res_np = ezs_tree.ResultsWrapper(num=B)
        res_list = ezs_tree.ResultsWrapper(num=B)
        ix_np, iy_np, act_np, vtp_np = ezs_tree.batch_traverse(
            roots_np, 19652, 1.25, 0.99, mms_np, res_np, list(to_play), False
        )
        ix_l, iy_l, act_l, vtp_l = ezs_tree.batch_traverse(
            roots_list, 19652, 1.25, 0.99, mms_list, res_list, list(to_play), False
        )
        assert (ix_np, iy_np, act_np, vtp_np) == (ix_l, iy_l, act_l, vtp_l)

        # Packed (B, 2 + A) float32 buffer matching mcts_ctree_sampled.py layout
        packed = rng.standard_normal((B, 2 + A)).astype(np.float32)
        reset_arr = np.array([1 if (sim + b) % 3 == 0 else 0 for b in range(B)], dtype=np.int32)

        ezs_tree.batch_backpropagate(
            sim + 1, 0.99, packed[:, 1], packed[:, 0], packed[:, 2:], mms_np, res_np, reset_arr, vtp_np
        )
        ezs_tree.batch_backpropagate(
            sim + 1, 0.99, packed[:, 1].tolist(), packed[:, 0].tolist(), packed[:, 2:].tolist(),
            mms_list, res_list, reset_arr.tolist(), vtp_l
        )

    assert roots_np.get_distributions() == roots_list.get_distributions()
    assert roots_np.get_sampled_actions() == roots_list.get_sampled_actions()
    assert roots_np.get_values() == roots_list.get_values()
    assert roots_np.get_children_q_values(0.99) == roots_list.get_children_q_values(0.99)


@pytest.mark.unittest
def test_log_space_topk_efraimidis_spirakis_sampling_on_4097_actions():
    """Verify the C++ log-space Efraimidis-Spirakis + size-K min-heap sampler on A=4097 actions:
    - samples exactly K=16 unique actions without replacement
    - strongly favors high-logit actions (no underflow corruption on 4097 actions)
    - preserves exact visit counts across multi-depth search."""
    from lzero.mcts.ctree.ctree_sampled_efficientzero import ezs_tree

    B = 8
    A = 4097
    K = 16
    hot_actions = {0, 17, 512, 2048, 4096}
    logits = np.full((B, A), -2.0, dtype=np.float32)
    for a in hot_actions:
        logits[:, a] = 8.0

    # Unrestricted legal_actions ([-1]) exercises the non-root/unmasked A=4097 fast path
    roots = ezs_tree.Roots(B, [[-1] for _ in range(B)], A, K, False)
    roots.prepare_no_noise([0.0] * B, logits.tolist(), [1] * B)

    sampled_batch = roots.get_sampled_actions()
    assert len(sampled_batch) == B
    for b in range(B):
        acts = [int(a[0]) for a in sampled_batch[b]]
        assert len(acts) == K
        assert len(set(acts)) == K, f"Expected {K} unique actions without replacement, got {acts}"
        assert all(0 <= a < A for a in acts)
        assert hot_actions.issubset(set(acts)), f"High-logit actions {hot_actions} missing from {acts}"

    # Run 25 sims via zero-copy strided NumPy batch_backpropagate and verify visit totals
    mms = ezs_tree.MinMaxStatsList(B)
    mms.set_delta(0.01)
    packed = np.zeros((B, 2 + A), dtype=np.float32)
    packed[:, 2:] = logits
    reset_arr = np.zeros(B, dtype=np.int32)
    for sim in range(25):
        res = ezs_tree.ResultsWrapper(num=B)
        _, _, _, vtp = ezs_tree.batch_traverse(roots, 19652, 1.25, 0.99, mms, res, [1] * B, False)
        ezs_tree.batch_backpropagate(sim + 1, 0.99, packed[:, 1], packed[:, 0], packed[:, 2:], mms, res, reset_arr, vtp)

    dists = roots.get_distributions()
    for b in range(B):
        assert sum(dists[b]) == 25, f"Expected 25 total root child visits, got {sum(dists[b])}"
