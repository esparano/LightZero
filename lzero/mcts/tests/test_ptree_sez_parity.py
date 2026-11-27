import numpy as np
import pytest
import torch
from easydict import EasyDict

from lzero.mcts.ptree import ptree_sez as tree_sez
from lzero.mcts.ptree.minimax import MinMaxStatsList
from lzero.mcts.tree_search.mcts_ptree_sampled import SampledEfficientZeroMCTSPtree


class _DummyRecurrentOutput:
    def __init__(self, latent_state, reward_hidden_state, value, value_prefix, policy_logits):
        self.latent_state = latent_state
        self.reward_hidden_state = reward_hidden_state
        self.value = value
        self.value_prefix = value_prefix
        self.policy_logits = policy_logits


class _DummyModel:
    def eval(self):
        return self

    def recurrent_inference(self, latent_states, reward_hidden_state, actions):
        batch = latent_states.shape[0]
        return _DummyRecurrentOutput(
            latent_state=torch.zeros_like(latent_states),
            reward_hidden_state=(
                torch.zeros_like(reward_hidden_state[0]),
                torch.zeros_like(reward_hidden_state[1]),
            ),
            value=torch.zeros((batch, 3), dtype=torch.float32),
            value_prefix=torch.zeros((batch, 3), dtype=torch.float32),
            policy_logits=torch.zeros((batch, 2), dtype=torch.float32),
        )


@pytest.mark.unittest
def test_ptree_batch_backpropagate_to_play_none_uses_single_player_sentinel():
    """BUG-18: batch_backpropagate(..., to_play=None) expanded with to_play=-1 but passed 0 to
    backpropagate(), raising AssertionError ('to_play is 0!') instead of running single-player backprop.
    """
    roots = tree_sez.Roots(
        root_num=1,
        legal_actions_list=[[0, 1]],
        action_space_size=2,
        num_of_sampled_actions=2,
        continuous_action_space=False,
    )
    roots.prepare_no_noise(
        value_prefixs=[0.0],
        policies=[[0.0, 0.0]],
        to_play=[-1],
    )
    root = roots.roots[0]
    child = root.get_child(root.legal_actions[0])

    results = tree_sez.SearchResults(num=1)
    results.nodes = [child]
    results.search_paths = [[root, child]]
    min_max_stats_lst = MinMaxStatsList(1)

    tree_sez.batch_backpropagate(
        simulation_index=1,
        discount_factor=1.0,
        value_prefixs=[0.25],
        values=[0.5],
        policies=[[0.0, 0.0]],
        min_max_stats_lst=min_max_stats_lst,
        results=results,
        is_reset_list=[0],
        to_play=None,
    )

    assert child.to_play == -1
    assert child.visit_count == 1
    assert root.visit_count == 2
    assert np.isclose(child.value, 0.5)
    assert np.isclose(root.value_sum, 0.75)
    assert np.isclose(root.value, 0.375)


@pytest.mark.unittest
def test_mcts_ptree_sampled_search_sets_value_delta_max_on_min_max_stats(monkeypatch):
    """BUG-17: SampledEfficientZeroMCTSPtree.search() must call min_max_stats_lst.set_delta(value_delta_max)
    to match SampledEfficientZeroMCTSCtree.search().
    """
    cfg = EasyDict(
        dict(
            device='cpu',
            env_type='not_board_games',
            num_simulations=2,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=1.0,
            value_delta_max=0.25,
            lstm_horizon_len=5,
            model=dict(
                categorical_distribution=True,
                value_support_range=(-1.0, 2.0, 1.0),
                reward_support_range=(-1.0, 2.0, 1.0),
                action_space_size=2,
                continuous_action_space=False,
                num_of_sampled_actions=2,
            ),
        )
    )
    mcts = SampledEfficientZeroMCTSPtree(cfg)
    roots = mcts.roots(
        root_num=1,
        legal_action_list=[[0, 1]],
        action_space_size=2,
        num_of_sampled_actions=2,
        continuous_action_space=False,
    )
    roots.prepare_no_noise(
        value_prefixs=[0.0],
        policies=[[0.0, 0.0]],
        to_play=[-1],
    )

    observed_deltas = []
    orig_batch_traverse = tree_sez.batch_traverse

    def _spy_batch_traverse(*args, **kwargs):
        min_max_stats_lst = args[4]
        observed_deltas.extend(s.value_delta_max for s in min_max_stats_lst.stats_lst)
        return orig_batch_traverse(*args, **kwargs)

    monkeypatch.setattr(tree_sez, 'batch_traverse', _spy_batch_traverse)

    mcts.search(
        roots=roots,
        model=_DummyModel(),
        latent_state_roots=[np.zeros((4,), dtype=np.float32)],
        reward_hidden_state_roots=(
            np.zeros((1, 1, 4), dtype=np.float32),
            np.zeros((1, 1, 4), dtype=np.float32),
        ),
        to_play_batch=[-1],
    )

    assert observed_deltas, 'Expected batch_traverse to be invoked at least once'
    assert all(np.isclose(d, 0.25) for d in observed_deltas), (
        f'Expected value_delta_max=0.25 on MinMaxStatsList, got {observed_deltas}'
    )
