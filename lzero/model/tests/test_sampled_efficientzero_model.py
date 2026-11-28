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

    @pytest.mark.parametrize('dtype,tol', [(torch.float64, 1e-12), (torch.float32, 1e-5)])
    @pytest.mark.parametrize('action_2d', [False, True])
    @pytest.mark.parametrize('norm_type', ['BN', 'LN', 'GN'])
    def test_dynamics_action_embedding_matches_dense_one_hot(self, dtype, tol, action_2d, norm_type):
        import copy

        torch.manual_seed(17)
        action_space_size = 65  # 1 + 4*4*4 (covers pass=0, duplicates, and last action=64)
        model_fast = SampledEfficientZeroModel(
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
            proj_hid=16,
            proj_out=16,
            pred_hid=16,
            pred_out=16,
            continuous_action_space=False,
            discrete_action_encoding_type='one_hot',
            norm_type=norm_type,
            embed_actions=True,
            embedded_action_dim=6,
            state_norm=True,
        ).to(dtype=dtype)
        model_ref = copy.deepcopy(model_fast)
        model_fast.train()
        model_ref.train()

        # Exact state_dict keys and shapes (action_conv.weight reused directly)
        assert list(model_fast.state_dict().keys()) == list(model_ref.state_dict().keys())
        assert 'dynamics_network.action_conv.weight' in model_fast.state_dict()

        # Deliberately include duplicate actions (0, 7, 7, 15, 0, 64) to exercise grad scatter-add
        actions_1d = torch.tensor([0, 7, 7, 15, 0, action_space_size - 1], dtype=torch.long)
        actions_in = actions_1d.unsqueeze(-1) if action_2d else actions_1d
        B = actions_1d.shape[0]

        latent_fast = torch.randn(B, 8, 4, 4, dtype=dtype, requires_grad=True)
        h_fast = torch.randn(1, B, 16, dtype=dtype, requires_grad=True)
        c_fast = torch.randn(1, B, 16, dtype=dtype, requires_grad=True)

        latent_ref = latent_fast.detach().clone().requires_grad_(True)
        h_ref = h_fast.detach().clone().requires_grad_(True)
        c_ref = c_fast.detach().clone().requires_grad_(True)

        # Fast path via SampledEfficientZeroModel._dynamics
        ns_fast, (nh_fast, nc_fast), vp_fast = model_fast._dynamics(latent_fast, (h_fast, c_fast), actions_in)

        # Reference legacy dense one-hot + action_conv path
        one_hot = torch.zeros(B, action_space_size, dtype=dtype)
        one_hot.scatter_(1, actions_1d.unsqueeze(-1), 1.0)
        act_enc = one_hot.unsqueeze(-1).unsqueeze(-1).expand(B, action_space_size, 4, 4)
        sae = torch.cat((latent_ref, act_enc), dim=1)
        ns_ref_raw, (nh_ref, nc_ref), vp_ref = model_ref.dynamics_network(sae, (h_ref, c_ref))
        from lzero.model.utils import renormalize
        ns_ref = renormalize(ns_ref_raw)

        # Verify forward outputs match tightly
        assert torch.allclose(ns_fast, ns_ref, rtol=tol, atol=tol)
        assert torch.allclose(nh_fast, nh_ref, rtol=tol, atol=tol)
        assert torch.allclose(nc_fast, nc_ref, rtol=tol, atol=tol)
        assert torch.allclose(vp_fast, vp_ref, rtol=tol, atol=tol)

        # Backward with deterministic non-uniform weights to test full Jacobian & duplicate-action grad accumulation
        w_ns = torch.randn_like(ns_fast)
        w_nh = torch.randn_like(nh_fast)
        w_nc = torch.randn_like(nc_fast)
        w_vp = torch.randn_like(vp_fast)

        loss_fast = (ns_fast * w_ns).sum() + (nh_fast * w_nh).sum() + (nc_fast * w_nc).sum() + (vp_fast * w_vp).sum()
        loss_ref = (ns_ref * w_ns).sum() + (nh_ref * w_nh).sum() + (nc_ref * w_nc).sum() + (vp_ref * w_vp).sum()
        loss_fast.backward()
        loss_ref.backward()

        assert torch.allclose(latent_fast.grad, latent_ref.grad, rtol=tol, atol=tol)
        assert torch.allclose(h_fast.grad, h_ref.grad, rtol=tol, atol=tol)
        assert torch.allclose(c_fast.grad, c_ref.grad, rtol=tol, atol=tol)

        for (name, p_fast), (_, p_ref) in zip(
            model_fast.dynamics_network.named_parameters(),
            model_ref.dynamics_network.named_parameters(),
        ):
            assert p_fast.grad is not None and p_ref.grad is not None, f'missing grad on {name}'
            assert torch.allclose(p_fast.grad, p_ref.grad, rtol=tol, atol=tol), (
                f'grad mismatch on {name}: max_abs={(p_fast.grad - p_ref.grad).abs().max().item()}'
            )

    def test_dynamics_action_embedding_multistep_policy_learn_parity(self):
        import copy
        import types
        import numpy as np
        from ding.utils import deep_merge_dicts
        from lzero.model.utils import renormalize
        from lzero.policy.sampled_efficientzero import SampledEfficientZeroPolicy

        base_cfg = SampledEfficientZeroPolicy.default_config()
        pol_cfg = deep_merge_dicts(
            base_cfg,
            dict(
                model=dict(
                    observation_shape=(4, 4, 4),
                    frame_stack_num=1,
                    image_channel=4,
                    action_space_size=65,
                    num_of_sampled_actions=4,
                    continuous_action_space=False,
                    discrete_action_encoding_type='one_hot',
                    model_type='conv',
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
                    proj_hid=16,
                    proj_out=16,
                    pred_hid=16,
                    pred_out=16,
                    self_supervised_learning_loss=True,
                    categorical_distribution=True,
                    downsample=False,
                    norm_type='LN',
                    embed_actions=True,
                    embedded_action_dim=6,
                    state_norm=True,
                ),
                cuda=False,
                device='cpu',
                batch_size=6,
                num_unroll_steps=3,
                lstm_horizon_len=3,
                td_steps=3,
                learning_rate=1e-3,
                weight_decay=1e-4,
                optim_type='AdamW',
                grad_clip_value=5.0,
                ssl_loss_weight=2.0,
                policy_loss_weight=1.0,
                value_loss_weight=0.25,
                reward_loss_weight=1.0,
                policy_entropy_weight=0.0,
                use_augmentation=False,
                cos_lr_scheduler=False,
                piecewise_decay_lr_scheduler=False,
                multi_gpu=False,
                monitor_extra_statistics=False,
                manual_temperature_decay=False,
                fixed_temperature_value=1.0,
                threshold_training_steps_for_final_temperature=100,
                target_update_freq=2,
                policy_loss_type='KL',
                normalize_prob_of_sampled_actions=False,
                mcts_ctree=True,
            ),
        )

        torch.manual_seed(42)
        pol_fast = SampledEfficientZeroPolicy(copy.deepcopy(pol_cfg), enable_field=['learn', 'collect'])
        pol_ref = SampledEfficientZeroPolicy(copy.deepcopy(pol_cfg), enable_field=['learn', 'collect'])
        pol_ref._learn_model.load_state_dict(copy.deepcopy(pol_fast._learn_model.state_dict()))
        pol_ref._target_model.load_state_dict(copy.deepcopy(pol_fast._target_model.state_dict()))
        pol_fast._collect_mcts_temperature = 1.0
        pol_ref._collect_mcts_temperature = 1.0

        # Force pol_ref._learn_model._dynamics to run the exact legacy dense one-hot path
        def _legacy_dynamics(self_m, latent_state, reward_hidden_state, action):
            act = action.unsqueeze(-1) if len(action.shape) == 1 else action
            one_hot = torch.zeros(act.shape[0], self_m.action_space_size, device=act.device)
            one_hot.scatter_(1, act.long(), 1)
            act_enc = one_hot.unsqueeze(-1).unsqueeze(-1).expand(
                latent_state.shape[0], self_m.action_space_size, latent_state.shape[2], latent_state.shape[3]
            ).to(latent_state.device).float()
            sae = torch.cat((latent_state, act_enc), dim=1)
            ns, nrh, vp = self_m.dynamics_network(sae, reward_hidden_state)
            return (renormalize(ns) if self_m.state_norm else ns), nrh, vp

        pol_ref._learn_model._dynamics = types.MethodType(_legacy_dynamics, pol_ref._learn_model)

        # Verify identical optimizer parameter groups
        assert len(pol_fast._optimizer.param_groups) == len(pol_ref._optimizer.param_groups)
        for g_f, g_r in zip(pol_fast._optimizer.param_groups, pol_ref._optimizer.param_groups):
            assert len(g_f['params']) == len(g_r['params'])
            assert g_f['weight_decay'] == g_r['weight_decay']

        rng = np.random.RandomState(7)
        for step in range(5):
            B, U, K = 6, 3, 4
            obs = rng.randn(B, 4 * (1 + U), 4, 4).astype(np.float32)
            # Include duplicate actions in every unroll step
            actions = np.array([[0, 5, 64], [0, 5, 12], [7, 7, 0], [15, 5, 64], [0, 12, 7], [64, 7, 0]], dtype=np.int64)
            rsa = rng.randint(0, 65, size=(B, U + 1, K, 1)).astype(np.int64)
            mask = np.ones((B, U + 1), dtype=np.float32)
            indices = np.arange(B, dtype=np.int64)
            weights = np.ones((B,), dtype=np.float32)
            make_time = np.ones((B,), dtype=np.float64)
            tvp = rng.uniform(-0.5, 0.5, size=(B, U + 1)).astype(np.float32)
            tv = rng.uniform(-0.5, 0.5, size=(B, U + 1)).astype(np.float32)
            tp = np.full((B, U + 1, K), 1.0 / K, dtype=np.float32)

            batch = [[obs, actions, rsa, mask, indices, weights, make_time], [tvp, tv, tp]]
            ret_f = pol_fast._forward_learn(copy.deepcopy(batch))
            ret_r = pol_ref._forward_learn(copy.deepcopy(batch))

            for key in ('weighted_total_loss', 'policy_loss', 'value_loss', 'value_prefix_loss', 'consistency_loss', 'total_grad_norm_before_clip'):
                if step == 0:
                    assert ret_f[key] == pytest.approx(ret_r[key], rel=1e-5, abs=1e-6), f'step 0 {key}: {ret_f[key]} vs {ret_r[key]}'
                else:
                    assert ret_f[key] == pytest.approx(ret_r[key], rel=1e-4, abs=1e-5), f'step {step} {key}: {ret_f[key]} vs {ret_r[key]}'

        # Biases feeding directly into BatchNorm1d have exact mathematical gradient 0,
        # so AdamW amplifies 1e-8 fp32 ULP roundoff into ~1e-4..1e-3 steps on those dead
        # biases (and their paired BN running_mean). Verify all functional network tensors
        # (representation, dynamics incl. action_conv, prediction) and all projection/pred weights.
        dead_pre_bn_keys = {
            'projection.0.bias', 'projection.1.running_mean',
            'projection.3.bias', 'projection.4.running_mean',
            'projection.6.bias', 'projection.7.running_mean',
            'projection.7.bias', 'prediction_head.0.bias', 'prediction_head.1.running_mean',
        }
        for name, p_f in pol_fast._learn_model.state_dict().items():
            if name in dead_pre_bn_keys:
                continue
            p_r = pol_ref._learn_model.state_dict()[name]
            assert torch.allclose(p_f, p_r, rtol=1e-4, atol=1e-5), (
                f'param mismatch after 5 steps on {name}: max_abs={(p_f - p_r).abs().max().item()}'
            )
        # Verify inference outputs (initial + recurrent) match tightly after 5 AdamW steps
        pol_fast._learn_model.eval()
        pol_ref._learn_model.eval()
        with torch.no_grad():
            probe_obs = torch.randn(4, 4, 4, 4)
            out_f = pol_fast._learn_model.initial_inference(probe_obs)
            out_r = pol_ref._learn_model.initial_inference(probe_obs)
            assert torch.allclose(out_f.value, out_r.value, rtol=1e-4, atol=1e-5)
            assert torch.allclose(out_f.policy_logits, out_r.policy_logits, rtol=1e-4, atol=1e-5)
            rec_act = torch.tensor([0, 7, 7, 64], dtype=torch.long)
            rec_f = pol_fast._learn_model.recurrent_inference(out_f.latent_state, out_f.reward_hidden_state, rec_act)
            rec_r = pol_ref._learn_model.recurrent_inference(out_r.latent_state, out_r.reward_hidden_state, rec_act)
            assert torch.allclose(rec_f.latent_state, rec_r.latent_state, rtol=1e-4, atol=1e-5)
            assert torch.allclose(rec_f.value_prefix, rec_r.value_prefix, rtol=1e-4, atol=1e-5)
            assert torch.allclose(rec_f.policy_logits, rec_r.policy_logits, rtol=1e-4, atol=1e-5)

        # Verify guardrail rejections on mismatched action count and embed_actions=False
        with pytest.raises(ValueError, match='Expected 1 discrete action index'):
            pol_fast._learn_model._dynamics(
                torch.zeros(2, 8, 4, 4),
                (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)),
                torch.zeros(3, dtype=torch.long),
            )
        dyn_no_embed = DynamicsNetwork(
            observation_shape=(4, 4, 4),
            action_encoding_dim=2,
            num_res_blocks=1,
            num_channels=10,
            reward_head_channels=4,
            reward_head_hidden_channels=[16],
            output_support_size=3,
            flatten_input_size_for_reward_head=4 * 4 * 4,
            downsample=False,
            lstm_hidden_size=16,
            embed_actions=False,
        )
        with pytest.raises(ValueError, match='embed_actions=True'):
            dyn_no_embed(
                torch.zeros(2, 8, 4, 4),
                (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)),
                discrete_action_indices=torch.zeros(2, dtype=torch.long),
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
