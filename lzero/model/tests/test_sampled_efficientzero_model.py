from itertools import product

import pytest
import torch
from ding.torch_utils import is_differentiable

from lzero.model.sampled_efficientzero_model import (
    DynamicsNetwork,
    PredictionNetwork,
    SampledEfficientZeroModel,
)
from lzero.model.sampled_efficientzero_model_mlp import SampledEfficientZeroModelMLP

batch_size = [100, 10]
num_res_blocks = [3, 4, 5]
num_channels = [10]
lstm_hidden_size = [64]
action_space_size = [2, 3]
reward_head_channels = [2]
reward_head_hidden_channels = [[16]]
output_support_size = [2]
flatten_input_size_for_reward_head = [180]
dynamics_network_args = list(
    product(
        batch_size, num_res_blocks, num_channels, lstm_hidden_size, action_space_size, reward_head_channels,
        reward_head_hidden_channels, output_support_size, flatten_input_size_for_reward_head
    )
)

value_head_channels = [8]
policy_head_channels = [8]
value_head_hidden_channels = [[
    16,
]]
policy_head_hidden_channels = [[
    16,
]]
observation_shape = [1, 3, 3]

prediction_network_args = list(
    product(
        action_space_size,
        batch_size,
        num_res_blocks,
        num_channels,
        value_head_channels,
        policy_head_channels,
        value_head_hidden_channels,
        policy_head_hidden_channels,
        output_support_size,
    )
)


@pytest.mark.unittest
class TestSampledEfficientZeroModel:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, list):
            loss = sum([t.sum() for t in outputs])
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

    @pytest.mark.parametrize(
        'action_space_size, batch_size, num_res_blocks, num_channels, value_head_channels, policy_head_channels, value_head_hidden_channels, policy_head_hidden_channels, output_support_size',
        prediction_network_args
    )
    def test_prediction_network(
        self, action_space_size, batch_size, num_res_blocks, num_channels, value_head_channels, policy_head_channels,
        value_head_hidden_channels, policy_head_hidden_channels, output_support_size
    ):
        obs = torch.rand(batch_size, num_channels, 3, 3)
        flatten_input_size_for_value_head = value_head_channels * observation_shape[1] * observation_shape[2]
        flatten_input_size_for_policy_head = policy_head_channels * observation_shape[1] * observation_shape[2]
        prediction_network = PredictionNetwork(
            observation_shape=observation_shape,
            continuous_action_space=True,
            action_space_size=action_space_size,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            value_head_channels=value_head_channels,
            policy_head_channels=policy_head_channels,
            value_head_hidden_channels=value_head_hidden_channels,
            policy_head_hidden_channels=policy_head_hidden_channels,
            output_support_size=output_support_size,
            flatten_input_size_for_value_head=flatten_input_size_for_value_head,
            flatten_input_size_for_policy_head=flatten_input_size_for_policy_head,
            last_linear_layer_init_zero=True,
        )
        policy, value = prediction_network(obs)
        assert policy.shape == torch.Size([batch_size, action_space_size * 2])
        assert value.shape == torch.Size([batch_size, output_support_size])

    @pytest.mark.parametrize(
        'batch_size, num_res_blocks, num_channels, lstm_hidden_size, action_space_size, reward_head_channels, reward_head_hidden_channels, output_support_size,'
        'flatten_input_size_for_reward_head', dynamics_network_args
    )
    def test_dynamics_network(
        self, batch_size, num_res_blocks, num_channels, lstm_hidden_size, action_space_size, reward_head_channels,
        reward_head_hidden_channels, output_support_size, flatten_input_size_for_reward_head
    ):
        print('=' * 20)
        print(
            batch_size, num_res_blocks, num_channels, lstm_hidden_size, action_space_size, reward_head_channels,
            reward_head_hidden_channels, output_support_size, flatten_input_size_for_reward_head
        )
        print('=' * 20)
        observation_shape = [1, 3, 3]
        flatten_input_size_for_reward_head = reward_head_channels * observation_shape[1] * observation_shape[2]
        state_action_embedding = torch.rand(batch_size, num_channels, observation_shape[1], observation_shape[2])
        dynamics_network = DynamicsNetwork(
            observation_shape=observation_shape,
            action_encoding_dim=action_space_size,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            lstm_hidden_size=lstm_hidden_size,
            reward_head_channels=reward_head_channels,
            reward_head_hidden_channels=reward_head_hidden_channels,
            output_support_size=output_support_size,
            flatten_input_size_for_reward_head=flatten_input_size_for_reward_head
        )
        next_state, reward_hidden_state, value_prefix = dynamics_network(
            state_action_embedding,
            (torch.randn(1, batch_size, lstm_hidden_size), torch.randn(1, batch_size, lstm_hidden_size))
        )
        assert next_state.shape == torch.Size([batch_size, num_channels - action_space_size, 3, 3])
        assert reward_hidden_state[0].shape == torch.Size([1, batch_size, lstm_hidden_size])
        assert reward_hidden_state[1].shape == torch.Size([1, batch_size, lstm_hidden_size])
        assert value_prefix.shape == torch.Size([batch_size, output_support_size])

    def test_state_norm_defaults_true_and_bounds_latent_state(self):
        # Conv model default
        conv_model = SampledEfficientZeroModel(
            observation_shape=(4, 4, 4),
            action_space_size=5,
            num_res_blocks=1,
            num_channels=8,
            lstm_hidden_size=16,
            reward_head_channels=4,
            value_head_channels=4,
            policy_head_channels=4,
            reward_head_hidden_channels=[16],
            value_head_hidden_channels=[16],
            policy_head_hidden_channels=[16],
            reward_support_range=(-1., 2., 1.),
            value_support_range=(-1., 2., 1.),
            proj_hid=32,
            proj_out=32,
            pred_hid=16,
            pred_out=32,
            continuous_action_space=False,
        )
        assert conv_model.state_norm is True
        obs = torch.randn(4, 4, 4, 4) * 10.0
        init_out = conv_model.initial_inference(obs)
        assert init_out.latent_state.min().item() >= -1e-6
        assert init_out.latent_state.max().item() <= 1.0 + 1e-6

        actions = torch.randint(0, 5, (4,))
        rec_out = conv_model.recurrent_inference(init_out.latent_state, init_out.reward_hidden_state, actions)
        assert rec_out.latent_state.min().item() >= -1e-6
        assert rec_out.latent_state.max().item() <= 1.0 + 1e-6
        (rec_out.latent_state.sum() + rec_out.value.sum() + rec_out.policy_logits.sum()).backward()

        # MLP model default
        mlp_model = SampledEfficientZeroModelMLP(
            observation_shape=8,
            action_space_size=4,
            latent_state_dim=16,
            lstm_hidden_size=16,
            reward_head_hidden_channels=[16],
            value_head_hidden_channels=[16],
            policy_head_hidden_channels=[16],
            reward_support_range=(-1., 2., 1.),
            value_support_range=(-1., 2., 1.),
            proj_hid=32,
            proj_out=32,
            pred_hid=16,
            pred_out=32,
            continuous_action_space=False,
        )
        assert mlp_model.state_norm is True
        mlp_obs = torch.randn(4, 8) * 10.0
        mlp_init = mlp_model.initial_inference(mlp_obs)
        assert mlp_init.latent_state.min().item() >= -1e-6
        assert mlp_init.latent_state.max().item() <= 1.0 + 1e-6

        mlp_rec = mlp_model.recurrent_inference(mlp_init.latent_state, mlp_init.reward_hidden_state, torch.randint(0, 4, (4,)))
        assert mlp_rec.latent_state.min().item() >= -1e-6
        assert mlp_rec.latent_state.max().item() <= 1.0 + 1e-6
        (mlp_rec.latent_state.sum() + mlp_rec.value.sum() + mlp_rec.policy_logits.sum()).backward()

    def test_dynamics_network_norm_type_ln_uses_no_batchnorm(self):
        """Regression test: DynamicsNetwork(..., norm_type='LN') must not hardcode BatchNorm
        in resblocks or norm_value_prefix, and must support batch_size=1 in train() mode.
        """
        conv_model = SampledEfficientZeroModel(
            observation_shape=(4, 4, 4),
            action_space_size=5,
            num_res_blocks=2,
            num_channels=8,
            lstm_hidden_size=16,
            reward_head_channels=4,
            value_head_channels=4,
            policy_head_channels=4,
            reward_head_hidden_channels=[16],
            value_head_hidden_channels=[16],
            policy_head_hidden_channels=[16],
            reward_support_range=(-1., 2., 1.),
            value_support_range=(-1., 2., 1.),
            proj_hid=32,
            proj_out=32,
            pred_hid=16,
            pred_out=32,
            continuous_action_space=False,
            norm_type='LN',
        )
        dyn = conv_model.dynamics_network
        bn_modules = [
            name for name, mod in dyn.named_modules()
            if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
        ]
        assert bn_modules == [], f"Unexpected BatchNorm layers in DynamicsNetwork under norm_type='LN': {bn_modules}"
        assert isinstance(dyn.norm_value_prefix, torch.nn.LayerNorm)

        # Batch size 1 in train() mode must succeed without BatchNorm ValueError
        conv_model.train()
        obs = torch.randn(1, 4, 4, 4)
        init_out = conv_model.initial_inference(obs)
        rec_out = conv_model.recurrent_inference(
            init_out.latent_state, init_out.reward_hidden_state, torch.zeros((1,), dtype=torch.long)
        )
        assert rec_out.latent_state.shape == torch.Size([1, 8, 4, 4])
        assert rec_out.value_prefix.shape == torch.Size([1, 3])

    def test_dynamics_network_norm_type_bn_unchanged(self):
        net = DynamicsNetwork(
            observation_shape=(4, 4, 4),
            action_encoding_dim=2,
            num_res_blocks=2,
            num_channels=10,
            reward_head_channels=4,
            reward_head_hidden_channels=[16],
            output_support_size=5,
            flatten_input_size_for_reward_head=4 * 4 * 4,
            downsample=False,
            lstm_hidden_size=16,
            norm_type='BN',
        )
        assert isinstance(net.norm_value_prefix, torch.nn.BatchNorm1d)
        resblock_bns = [
            name for name, mod in net.resblocks.named_modules()
            if isinstance(mod, torch.nn.BatchNorm2d)
        ]
        assert len(resblock_bns) == 4

    @pytest.mark.parametrize('policy_head_type', ['flat', 'factored_spatial'])
    def test_sampled_efficientzero_norm_type_gn(self, policy_head_type):
        action_space_size = 1 + 4 * 4 * 4 * 4
        conv_model = SampledEfficientZeroModel(
            observation_shape=(4, 4, 4),
            action_space_size=action_space_size,
            num_res_blocks=2,
            num_channels=8,
            lstm_hidden_size=16,
            reward_head_channels=4,
            value_head_channels=4,
            policy_head_channels=4,
            reward_head_hidden_channels=[16],
            value_head_hidden_channels=[16],
            policy_head_hidden_channels=[16],
            reward_support_range=(-1., 2., 1.),
            value_support_range=(-1., 2., 1.),
            proj_hid=32,
            proj_out=32,
            pred_hid=16,
            pred_out=32,
            continuous_action_space=False,
            norm_type='GN',
            embed_actions=True,
            embedded_action_dim=4,
            policy_head_type=policy_head_type,
        )

        # All 5 outer 2D conv norm layers must use GroupNorm(1, C) instead of per-pixel LayerNorm([C, H, W])
        outer_norms = [
            conv_model.representation_network.norm,
            conv_model.dynamics_network.norm_common,
            conv_model.dynamics_network.norm_reward,
            conv_model.prediction_network.norm_value,
            conv_model.prediction_network.norm_policy,
        ]
        for norm in outer_norms:
            assert isinstance(norm, torch.nn.GroupNorm)
            assert norm.num_groups == 1

        # Zero BatchNorm and zero 3D spatial LayerNorm anywhere in repr / dynamics / prediction
        for sub_name, sub_mod in [
            ('representation_network', conv_model.representation_network),
            ('dynamics_network', conv_model.dynamics_network),
            ('prediction_network', conv_model.prediction_network),
        ]:
            bns = [
                n for n, m in sub_mod.named_modules()
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
            ]
            assert bns == [], f"Unexpected BatchNorm in {sub_name} under norm_type='GN': {bns}"
            spatial_lns = [
                n for n, m in sub_mod.named_modules()
                if isinstance(m, torch.nn.LayerNorm) and len(m.normalized_shape) > 1
            ]
            assert spatial_lns == [], f"Unexpected spatial LayerNorm in {sub_name} under norm_type='GN': {spatial_lns}"

        # 1D LSTM & MLP heads still use 1D LayerNorm
        assert isinstance(conv_model.dynamics_network.norm_value_prefix, torch.nn.LayerNorm)
        assert conv_model.dynamics_network.norm_value_prefix.normalized_shape == (16,)

        # Batch size 1 in train() mode must succeed end-to-end (forward + backward)
        conv_model.train()
        obs = torch.randn(1, 4, 4, 4)
        init_out = conv_model.initial_inference(obs)
        rec_out = conv_model.recurrent_inference(
            init_out.latent_state, init_out.reward_hidden_state, torch.zeros((1,), dtype=torch.long)
        )
        assert init_out.policy_logits.shape == torch.Size([1, action_space_size])
        assert rec_out.latent_state.shape == torch.Size([1, 8, 4, 4])
        assert rec_out.value_prefix.shape == torch.Size([1, 3])
        (init_out.value.sum() + rec_out.value_prefix.sum() + rec_out.policy_logits.sum()).backward()

        # Verify downsample=True path also supports norm_type='GN'
        from lzero.model.common import RepresentationNetwork
        ds_repr = RepresentationNetwork(
            observation_shape=(3, 64, 64),
            num_res_blocks=1,
            num_channels=8,
            downsample=True,
            norm_type='GN',
        )
        assert isinstance(ds_repr.downsample_net.norm1, torch.nn.GroupNorm)
        assert ds_repr.downsample_net.norm1.num_groups == 1

        # Verify invalid norm_type raises ValueError cleanly
        with pytest.raises(ValueError, match="Unsupported norm_type"):
            SampledEfficientZeroModel(
                observation_shape=(4, 4, 4),
                action_space_size=action_space_size,
                num_res_blocks=1,
                num_channels=8,
                norm_type='INVALID',
            )


if __name__ == "__main__":
    batch_size = 2
    num_res_blocks = 3
    num_channels = 10
    lstm_hidden_size = 64
    action_space_size = 5
    reward_head_channels = 2
    reward_head_hidden_channels = [16]
    output_support_size = 2
    observation_shape = [1, 3, 3]
    # flatten_input_size_for_reward_head = 180
    flatten_input_size_for_reward_head = reward_head_channels * observation_shape[1] * observation_shape[2]

    state_action_embedding = torch.rand(batch_size, num_channels, observation_shape[1], observation_shape[2])
    dynamics_network = DynamicsNetwork(
        observation_shape=observation_shape,
        action_encoding_dim=action_space_size,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        reward_head_channels=reward_head_channels,
        reward_head_hidden_channels=reward_head_hidden_channels,
        output_support_size=output_support_size,
        flatten_input_size_for_reward_head=flatten_input_size_for_reward_head
    )
    next_state, reward_hidden_state, value_prefix = dynamics_network(
        state_action_embedding,
        (torch.randn(1, batch_size, lstm_hidden_size), torch.randn(1, batch_size, lstm_hidden_size))
    )
    assert next_state.shape == torch.Size([batch_size, num_channels - action_space_size, 3, 3])
    assert reward_hidden_state[0].shape == torch.Size([1, batch_size, lstm_hidden_size])
    assert reward_hidden_state[1].shape == torch.Size([1, batch_size, lstm_hidden_size])
    assert value_prefix.shape == torch.Size([batch_size, output_support_size])
