import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer
from lzero.mcts.buffer.game_segment import GameSegment
from lzero.model.sampled_efficientzero_model import (
    FactoredSpatialPolicyHead,
    PredictionNetwork,
    SampledEfficientZeroModel,
)
from lzero.policy.sampled_efficientzero import SampledEfficientZeroPolicy
from zoo.board_games.galcon.envs.galcon_env import GalconEnv


def _make_sez_model(
    obs_shape=(8, 4, 4),
    action_space_size=None,
    num_pair_layers=1,
    policy_head_type='factored_spatial',
    policy_head_embed_dim=16,
    last_linear_layer_init_zero=True,
    norm_type='LN',
    downsample=False,
    continuous_action_space=False,
    policy_head_channels=6,
    policy_head_hidden_channels=(32,),
    policy_head_pass_hidden_channels=None,
):
    if action_space_size is None:
        h, w = obs_shape[1], obs_shape[2]
        n = h * w
        action_space_size = 1 + num_pair_layers * n * n
    return SampledEfficientZeroModel(
        observation_shape=obs_shape,
        action_space_size=action_space_size,
        num_res_blocks=1,
        num_channels=8,
        lstm_hidden_size=16,
        reward_head_channels=4,
        value_head_channels=4,
        policy_head_channels=policy_head_channels,
        reward_head_hidden_channels=[16],
        value_head_hidden_channels=[16],
        policy_head_hidden_channels=list(policy_head_hidden_channels),
        reward_support_range=(-1.0, 2.0, 1.0),
        value_support_range=(-1.0, 2.0, 1.0),
        proj_hid=16,
        proj_out=16,
        pred_hid=16,
        pred_out=16,
        self_supervised_learning_loss=True,
        categorical_distribution=True,
        last_linear_layer_init_zero=last_linear_layer_init_zero,
        downsample=downsample,
        continuous_action_space=continuous_action_space,
        num_of_sampled_actions=4,
        norm_type=norm_type,
        embed_actions=True,
        embedded_action_dim=8,
        policy_head_type=policy_head_type,
        policy_head_embed_dim=policy_head_embed_dim,
        policy_head_pass_hidden_channels=policy_head_pass_hidden_channels,
    )


def _galcon_reflection_action_perm(
    grid_h: int, grid_w: int, num_pair_layers: int, reflect_x: bool, reflect_y: bool
) -> np.ndarray:
    """Compute exact Galcon action permutation using GalconEnv.reflect_game_segment."""
    n = grid_h * grid_w
    action_space_size = 1 + num_pair_layers * n * n
    # C = 40 matches GalconEnv default with fleet_top_k=1 (8 planet + 20 fleet + 12 landing).
    seg_cfg = EasyDict(
        dict(
            num_unroll_steps=1,
            td_steps=1,
            discount_factor=0.99,
            gray_scale=False,
            transform2string=False,
            sampled_algo=True,
            gumbel_algo=False,
            use_ture_chance_label_in_chance_encoder=False,
            model=dict(
                frame_stack_num=1,
                observation_shape=(40, grid_h, grid_w),
                image_channel=40,
                action_space_size=action_space_size,
                continuous_action_space=False,
                num_of_sampled_actions=4,
            ),
        )
    )
    seg = GameSegment(action_space=None, game_segment_length=2, config=seg_cfg)
    seg.obs_segment.append(np.zeros((40, grid_h, grid_w), dtype=np.float32))
    seg.action_segment = np.arange(action_space_size, dtype=np.int64)
    seg.action_mask_segment = [np.ones(action_space_size, dtype=np.int8)]
    seg.root_sampled_actions = [np.zeros(4, dtype=np.int64)]

    reflected = GalconEnv.reflect_game_segment(seg, reflect_x=reflect_x, reflect_y=reflect_y)
    return np.asarray(reflected.action_segment, dtype=np.int64)


@pytest.mark.unittest
class TestFactoredSpatialPolicyHead:

    @pytest.mark.parametrize('grid_h,grid_w', [(4, 4), (3, 5)])
    @pytest.mark.parametrize('num_pair_layers', [1, 2])
    @pytest.mark.parametrize('norm_type', ['LN', 'BN', 'GN'])
    def test_output_shape_and_exact_action_layout(self, grid_h, grid_w, num_pair_layers, norm_type):
        torch.manual_seed(0)
        n = grid_h * grid_w
        action_space_size = 1 + num_pair_layers * n * n
        model = _make_sez_model(
            obs_shape=(8, grid_h, grid_w),
            action_space_size=action_space_size,
            num_pair_layers=num_pair_layers,
            policy_head_type='factored_spatial',
            policy_head_embed_dim=12,
            last_linear_layer_init_zero=False,
            norm_type=norm_type,
        )
        head = model.prediction_network.fc_policy_head
        assert isinstance(head, FactoredSpatialPolicyHead)
        assert head.num_pair_layers == num_pair_layers

        batch_size = 3
        obs = torch.randn(batch_size, 8, grid_h, grid_w)
        init_out = model.initial_inference(obs)
        assert init_out.policy_logits.shape == (batch_size, action_space_size)
        assert torch.isfinite(init_out.policy_logits).all()

        actions = torch.randint(0, action_space_size, (batch_size,))
        rec_out = model.recurrent_inference(init_out.latent_state, init_out.reward_hidden_state, actions)
        assert rec_out.policy_logits.shape == (batch_size, action_space_size)
        assert torch.isfinite(rec_out.policy_logits).all()

    def test_exact_formula_matches_galcon_row_major_action_indexing(self):
        torch.manual_seed(7)
        grid_h, grid_w, c_p, embed_dim, num_pair_layers = 3, 5, 6, 8, 2
        n = grid_h * grid_w
        action_space_size = 1 + num_pair_layers * n * n
        head = FactoredSpatialPolicyHead(
            in_channels=c_p,
            grid_height=grid_h,
            grid_width=grid_w,
            action_space_size=action_space_size,
            embed_dim=embed_dim,
            pass_hidden_channels=[16],
            norm_type='LN',
            last_linear_layer_init_zero=False,
        )
        with torch.no_grad():
            head.rel_dist_bias.copy_(torch.randn_like(head.rel_dist_bias))

        head.eval()
        feats = torch.randn(2, c_p, grid_h, grid_w)
        logits = head(feats)

        u_map = head.src_proj(feats).reshape(2, num_pair_layers, embed_dim, grid_h, grid_w)
        v_map = head.dst_proj(feats).reshape(2, num_pair_layers, embed_dim, grid_h, grid_w)
        s_map = head.src_bias_proj(feats)  # (B, L, H, W)
        t_map = head.dst_bias_proj(feats)  # (B, L, H, W)

        # Spot-check multiple (layer, src_xy, dst_xy) tuples using Galcon row-major cell = y * W + x
        checks = [
            (0, (0, 0), (0, 0)),
            (0, (4, 2), (1, 0)),
            (0, (2, 1), (3, 2)),
            (1, (0, 2), (4, 0)),
            (1, (3, 1), (1, 2)),
        ]
        for layer, (sx, sy), (tx, ty) in checks:
            src_cell = sy * grid_w + sx
            dst_cell = ty * grid_w + tx
            action_id = 1 + layer * n * n + src_cell * n + dst_cell
            expected = (
                (u_map[:, layer, :, sy, sx] * v_map[:, layer, :, ty, tx]).sum(dim=-1) / (embed_dim ** 0.5)
                + s_map[:, layer, sy, sx]
                + t_map[:, layer, ty, tx]
                + head.rel_dist_bias[layer, abs(sy - ty), abs(sx - tx)]
            )
            assert torch.allclose(logits[:, action_id], expected, atol=1e-5)

    def test_invalid_config_validation_errors(self):
        # Unsupported policy_head_type
        with pytest.raises(ValueError, match='policy_head_type'):
            _make_sez_model(policy_head_type='invalid_head')

        with pytest.raises(ValueError, match='policy_head_type'):
            PredictionNetwork(
                observation_shape=(8, 4, 4),
                continuous_action_space=False,
                action_space_size=257,
                num_res_blocks=1,
                num_channels=8,
                value_head_channels=4,
                policy_head_channels=4,
                value_head_hidden_channels=[16],
                policy_head_hidden_channels=[16],
                output_support_size=3,
                flatten_input_size_for_value_head=64,
                flatten_input_size_for_policy_head=64,
                policy_head_type='invalid_head',
            )

        # Continuous action space not supported
        with pytest.raises(ValueError, match='discrete action spaces'):
            _make_sez_model(continuous_action_space=True, policy_head_type='factored_spatial')

        # downsample=True fails fast on both non-Atari grids (e.g. 4x4) and Atari grids (64x64)
        for obs_shape, act_size in [((8, 4, 4), 257), ((4, 64, 64), 1 + (64 * 64) ** 2)]:
            with pytest.raises(ValueError, match='downsample=False'):
                _make_sez_model(
                    obs_shape=obs_shape,
                    action_space_size=act_size,
                    downsample=True,
                    policy_head_type='factored_spatial',
                )

        # action_space_size - 1 must be a positive multiple of (H * W)^2
        for bad_action_size in [1, 256, 258, 300, 512]:
            with pytest.raises(ValueError, match='positive multiple'):
                _make_sez_model(
                    obs_shape=(8, 4, 4),
                    action_space_size=bad_action_size,
                    policy_head_type='factored_spatial',
                )

        # Non-positive embed_dim
        for bad_dim in [0, -8]:
            with pytest.raises(ValueError, match='policy_head_embed_dim'):
                _make_sez_model(
                    obs_shape=(8, 4, 4),
                    action_space_size=257,
                    policy_head_type='factored_spatial',
                    policy_head_embed_dim=bad_dim,
                )

        # Non-3D observation_shape
        with pytest.raises(ValueError, match='3D'):
            _make_sez_model(
                obs_shape=8,
                action_space_size=65,
                policy_head_type='factored_spatial',
            )

    @pytest.mark.parametrize('num_pair_layers', [1, 2])
    @pytest.mark.parametrize('norm_type', ['LN', 'BN'])
    def test_last_linear_layer_init_zero_yields_exact_zero_logits(self, num_pair_layers, norm_type):
        torch.manual_seed(11)
        model = _make_sez_model(
            obs_shape=(8, 4, 4),
            num_pair_layers=num_pair_layers,
            policy_head_type='factored_spatial',
            last_linear_layer_init_zero=True,
            norm_type=norm_type,
        )
        obs = torch.randn(4, 8, 4, 4) * 5.0
        out = model.initial_inference(obs)
        assert torch.equal(out.policy_logits, torch.zeros_like(out.policy_logits))

        # Sanity contrast: last_linear_layer_init_zero=False yields non-zero initial logits
        model_nonzero = _make_sez_model(
            obs_shape=(8, 4, 4),
            num_pair_layers=num_pair_layers,
            policy_head_type='factored_spatial',
            last_linear_layer_init_zero=False,
            norm_type=norm_type,
        )
        out_nonzero = model_nonzero.initial_inference(obs)
        assert out_nonzero.policy_logits.abs().max().item() > 1e-3

    @pytest.mark.parametrize('grid_h,grid_w', [(4, 4), (3, 5)])
    @pytest.mark.parametrize('num_pair_layers', [1, 2])
    @pytest.mark.parametrize('norm_type', ['LN', 'BN'])
    def test_exact_4x_xy_reflection_equivariance(self, grid_h, grid_w, num_pair_layers, norm_type):
        torch.manual_seed(23)
        c_p = 8
        n = grid_h * grid_w
        action_space_size = 1 + num_pair_layers * n * n
        head = FactoredSpatialPolicyHead(
            in_channels=c_p,
            grid_height=grid_h,
            grid_width=grid_w,
            action_space_size=action_space_size,
            embed_dim=16,
            pass_hidden_channels=[32],
            norm_type=norm_type,
            last_linear_layer_init_zero=False,
        )
        # Populate rel_dist_bias with non-zero random weights to verify distance-bias equivariance too.
        with torch.no_grad():
            head.rel_dist_bias.copy_(torch.randn_like(head.rel_dist_bias))

        head.eval()
        feats = torch.randn(4, c_p, grid_h, grid_w)
        logits_orig = head(feats)

        for reflect_x, reflect_y in [(False, False), (True, False), (False, True), (True, True)]:
            perm = _galcon_reflection_action_perm(grid_h, grid_w, num_pair_layers, reflect_x, reflect_y)
            assert perm[0] == 0

            feats_ref = feats
            if reflect_x:
                feats_ref = torch.flip(feats_ref, dims=[3])
            if reflect_y:
                feats_ref = torch.flip(feats_ref, dims=[2])

            logits_ref = head(feats_ref)

            # PASS logit must be strictly invariant to X/Y reflections
            assert torch.allclose(logits_ref[:, 0], logits_orig[:, 0], atol=1e-5)

            # Send logits must match exact Galcon-reflected action indexing (perm is an involution)
            expected_ref = logits_orig[:, torch.from_numpy(perm)]
            assert torch.allclose(logits_ref, expected_ref, atol=1e-5)

            # Non-identity flips must actually permute non-trivial send logits
            if reflect_x or reflect_y:
                assert (logits_ref[:, 1:] - logits_orig[:, 1:]).abs().max().item() > 1e-2

    def test_gradient_flow_with_last_linear_layer_init_zero_and_without(self):
        torch.manual_seed(31)
        grid_h, grid_w, c_p, embed_dim, num_pair_layers = 3, 4, 6, 8, 2
        n = grid_h * grid_w
        action_space_size = 1 + num_pair_layers * n * n

        # Non-uniform target over PASS and send actions across both pair layers and distinct distances
        target_actions = torch.tensor([0, 1 + 0 * n + 5, 1 + n * n + 2 * n + 9, 1 + 7 * n + 1], dtype=torch.long)
        feats_base = torch.randn(4, c_p, grid_h, grid_w)

        # 1. With last_linear_layer_init_zero=False: all components get non-zero grad on step 0
        head_nz = FactoredSpatialPolicyHead(
            in_channels=c_p,
            grid_height=grid_h,
            grid_width=grid_w,
            action_space_size=action_space_size,
            embed_dim=embed_dim,
            pass_hidden_channels=[16],
            norm_type='LN',
            last_linear_layer_init_zero=False,
        )
        feats_nz = feats_base.clone().requires_grad_(True)
        loss_nz = F.cross_entropy(head_nz(feats_nz), target_actions)
        loss_nz.backward()

        for name, param in head_nz.named_parameters():
            assert param.grad is not None, f"Missing grad for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"
            assert param.grad.abs().sum().item() > 0.0, f"Zero grad for {name}"
        assert feats_nz.grad is not None and feats_nz.grad.abs().sum().item() > 0.0

        # 2. With last_linear_layer_init_zero=True:
        #    - Step 0 output is identically 0.0, yet dst_proj (v), src_bias_proj (s), dst_bias_proj (t),
        #      rel_dist_bias (b), and the PASS output linear layer ALL get non-zero gradients (no saddle point).
        #    - After 1 optimizer step, EVERY parameter including src_proj (u) and hidden PASS layers
        #      gets non-zero gradients, and loss decreases.
        head_z = FactoredSpatialPolicyHead(
            in_channels=c_p,
            grid_height=grid_h,
            grid_width=grid_w,
            action_space_size=action_space_size,
            embed_dim=embed_dim,
            pass_hidden_channels=[16],
            norm_type='LN',
            last_linear_layer_init_zero=True,
        )
        optimizer = torch.optim.AdamW(head_z.parameters(), lr=1e-2)

        logits_step0 = head_z(feats_base)
        assert torch.equal(logits_step0, torch.zeros_like(logits_step0))
        loss_step0 = F.cross_entropy(logits_step0, target_actions)
        optimizer.zero_grad()
        loss_step0.backward()

        assert head_z.dst_proj.weight.grad.abs().sum().item() > 0.0
        assert head_z.dst_proj.bias.grad.abs().sum().item() > 0.0
        assert head_z.src_bias_proj.weight.grad.abs().sum().item() > 0.0
        assert head_z.dst_bias_proj.weight.grad.abs().sum().item() > 0.0
        assert head_z.rel_dist_bias.grad.abs().sum().item() > 0.0
        pass_linears = [m for m in head_z.pass_head.modules() if isinstance(m, nn.Linear)]
        assert pass_linears[-1].weight.grad.abs().sum().item() > 0.0
        assert pass_linears[-1].bias.grad.abs().sum().item() > 0.0

        optimizer.step()

        # Step 1: now v and PASS final layer are non-zero -> u, v, s, t, b, full PASS MLP, and input all get grad
        feats_step1 = feats_base.clone().requires_grad_(True)
        optimizer.zero_grad()
        loss_step1 = F.cross_entropy(head_z(feats_step1), target_actions)
        loss_step1.backward()

        for name, param in head_z.named_parameters():
            assert param.grad is not None, f"Step 1 missing grad for {name}"
            assert torch.isfinite(param.grad).all(), f"Step 1 non-finite grad for {name}"
            assert param.grad.abs().sum().item() > 0.0, f"Step 1 zero grad for {name}"
        assert feats_step1.grad is not None and feats_step1.grad.abs().sum().item() > 0.0
        assert loss_step1.item() < loss_step0.item()

    @pytest.mark.parametrize('mcts_ctree', [False, True])
    def test_e2e_sez_policy_learn_and_mcts_inference_factored_head(self, mcts_ctree):
        torch.manual_seed(0)
        np.random.seed(0)

        env_cfg = GalconEnv.default_config()
        env_cfg.update(
            dict(
                battle_mode='self_play_mode',
                bot_action_type='random',
                num_planets=4,
                min_send_ships=1,
                send_ratio=0.5,
                tick_seconds=0.25,
                fleet_speed=40.0,
                max_episode_steps=10,
                map_seed=2,
                grid_square_size=25.0,
                grid_max_x=50.0,
                grid_max_y=50.0,
                home_distance_fraction=0.8,
                neutral_min_cost=0,
                neutral_max_cost=10,
                neutral_min_production=15,
                neutral_max_production=100,
                fleet_top_k=1,
                channel_last=False,
                scale=True,
            )
        )
        env = GalconEnv(env_cfg)
        obs0 = env.reset()
        c, h, w = obs0['observation'].shape
        action_space_size = int(obs0['action_mask'].shape[0])
        k_sampled = 4

        policy_cfg = SampledEfficientZeroPolicy.default_config()
        policy_cfg.update(
            dict(
                model=dict(
                    model_type='conv',
                    observation_shape=(c, h, w),
                    image_channel=c,
                    frame_stack_num=1,
                    action_space_size=action_space_size,
                    continuous_action_space=False,
                    num_of_sampled_actions=k_sampled,
                    downsample=False,
                    self_supervised_learning_loss=True,
                    num_res_blocks=1,
                    num_channels=8,
                    lstm_hidden_size=16,
                    reward_head_channels=4,
                    value_head_channels=4,
                    policy_head_channels=6,
                    reward_head_hidden_channels=[16],
                    value_head_hidden_channels=[16],
                    policy_head_hidden_channels=[16],
                    categorical_distribution=True,
                    reward_support_range=(-1.0, 2.0, 1.0),
                    value_support_range=(-1.0, 2.0, 1.0),
                    discrete_action_encoding_type='one_hot',
                    norm_type='BN',
                    embed_actions=True,
                    embedded_action_dim=8,
                    policy_head_type='factored_spatial',
                    policy_head_embed_dim=16,
                ),
                cuda=False,
                device='cpu',
                env_type='board_games',
                action_type='varied_action_space',
                battle_mode='self_play_mode',
                mcts_ctree=mcts_ctree,
                use_augmentation=False,
                num_simulations=4,
                update_per_collect=1,
                batch_size=4,
                optim_type='AdamW',
                learning_rate=1e-3,
                weight_decay=1e-4,
                grad_clip_value=1.0,
                piecewise_decay_lr_scheduler=False,
                game_segment_length=6,
                num_unroll_steps=2,
                lstm_horizon_len=2,
                td_steps=2,
                discount_factor=1.0,
                reanalyze_ratio=0.0,
                replay_buffer_size=200,
                priority_prob_alpha=0.6,
                priority_prob_beta=0.4,
                use_priority=False,
                policy_loss_type='cross_entropy',
                use_root_value=False,
                mini_infer_size=64,
                sample_type='transition',
                target_model_update_freq=100,
                symmetric_augment_x=True,
                symmetric_augment_y=True,
                symmetric_augment_type='Galcon',
            )
        )

        policy = SampledEfficientZeroPolicy(policy_cfg)
        policy._init_collect()
        policy._init_learn()
        policy._init_eval()

        head = policy._learn_model.prediction_network.fc_policy_head
        assert isinstance(head, FactoredSpatialPolicyHead)

        seg_cfg = EasyDict(
            dict(
                game_segment_length=policy_cfg.game_segment_length,
                num_unroll_steps=policy_cfg.num_unroll_steps,
                td_steps=policy_cfg.td_steps,
                discount_factor=policy_cfg.discount_factor,
                gray_scale=False,
                transform2string=False,
                sampled_algo=True,
                gumbel_algo=False,
                use_ture_chance_label_in_chance_encoder=False,
                model=dict(
                    frame_stack_num=1,
                    observation_shape=policy_cfg.model.observation_shape,
                    image_channel=policy_cfg.model.image_channel,
                    action_space_size=policy_cfg.model.action_space_size,
                    continuous_action_space=False,
                    num_of_sampled_actions=k_sampled,
                ),
            )
        )

        seg = GameSegment(action_space=None, game_segment_length=policy_cfg.game_segment_length, config=seg_cfg)
        obs_dict = obs0
        seg.reset([obs_dict['observation']])
        for _ in range(policy_cfg.game_segment_length):
            obs_t = torch.from_numpy(obs_dict['observation']).float().unsqueeze(0)
            out = policy._forward_collect(
                obs_t,
                action_mask=[obs_dict['action_mask']],
                temperature=1.0,
                to_play=[obs_dict['to_play']],
            )[0]
            action = int(out['action'])
            assert obs_dict['action_mask'][action] == 1
            ts = env.step(action)
            seg.append(action, ts.obs['observation'], float(ts.reward), obs_dict['action_mask'], obs_dict['to_play'])
            seg.store_search_stats(
                out['visit_count_distributions'],
                root_value=float(out['searched_value']),
                root_sampled_actions=out['root_sampled_actions'],
            )
            obs_dict = env.reset() if ts.done else ts.obs

        seg.valid_transition_count = policy_cfg.game_segment_length
        seg.game_segment_to_array()

        buf = SampledEfficientZeroGameBuffer(policy_cfg)
        meta = {'done': True, 'unroll_plus_td_steps': 4, 'priorities': np.ones(policy_cfg.game_segment_length)}
        buf.push_game_segments(([seg], [meta]))
        assert buf.get_num_of_game_segments() == 4

        initial_params = {k: v.detach().clone() for k, v in head.named_parameters()}
        for _ in range(2):
            train_data = buf.sample(batch_size=policy_cfg.batch_size, policy=policy)
            log_vars = policy._forward_learn(train_data)
            for key in ['total_loss', 'policy_loss', 'value_loss', 'consistency_loss']:
                assert np.isfinite(float(log_vars[key])), f"Non-finite {key}: {log_vars[key]}"

        # After 2 AdamW steps, all factored head components (including u after v becomes non-zero) should update.
        for name, param in head.named_parameters():
            assert not torch.equal(param.detach(), initial_params[name]), (
                f"FactoredSpatialPolicyHead parameter '{name}' did not update during _forward_learn"
            )

        # Verify MCTS eval inference produces a legal action
        eval_out = policy._forward_eval(
            torch.from_numpy(obs0['observation']).float().unsqueeze(0),
            action_mask=[obs0['action_mask']],
            to_play=[obs0['to_play']],
        )[0]
        assert obs0['action_mask'][int(eval_out['action'])] == 1

    def test_pass_mlp_width_cap_and_explicit_override(self):
        # Legacy flat-head width [512] is automatically capped to max(2 * C_p, 64) = 64
        m_capped = _make_sez_model(
            policy_head_channels=16,
            policy_head_hidden_channels=[512],
            policy_head_pass_hidden_channels=None,
        )
        capped_linears = [m for m in m_capped.prediction_network.fc_policy_head.pass_head.modules() if isinstance(m, nn.Linear)]
        assert capped_linears[0].in_features == 32
        assert capped_linears[0].out_features == 64
        assert capped_linears[-1].out_features == 1

        # Smaller policy_head_hidden_channels ([16]) are preserved verbatim
        m_small = _make_sez_model(
            policy_head_channels=6,
            policy_head_hidden_channels=[16],
            policy_head_pass_hidden_channels=None,
        )
        small_linears = [m for m in m_small.prediction_network.fc_policy_head.pass_head.modules() if isinstance(m, nn.Linear)]
        assert small_linears[0].out_features == 16

        # Explicit policy_head_pass_hidden_channels overrides the default cap
        m_override = _make_sez_model(
            policy_head_channels=16,
            policy_head_hidden_channels=[512],
            policy_head_pass_hidden_channels=[256],
        )
        override_linears = [m for m in m_override.prediction_network.fc_policy_head.pass_head.modules() if isinstance(m, nn.Linear)]
        assert override_linears[0].out_features == 256

    def test_default_flat_policy_head_unchanged(self):
        default_model_cfg = SampledEfficientZeroPolicy.default_config().model
        assert default_model_cfg.policy_head_type == 'flat'
        assert default_model_cfg.policy_head_embed_dim == 32
        assert default_model_cfg.policy_head_pass_hidden_channels is None

        # Default discrete model matches explicit policy_head_type='flat' bit-for-bit
        torch.manual_seed(99)
        m_default = SampledEfficientZeroModel(
            observation_shape=(4, 4, 4),
            action_space_size=17,
            num_res_blocks=1,
            num_channels=8,
            lstm_hidden_size=16,
            reward_head_channels=4,
            value_head_channels=4,
            policy_head_channels=4,
            reward_head_hidden_channels=[16],
            value_head_hidden_channels=[16],
            policy_head_hidden_channels=[16],
            reward_support_range=(-1.0, 2.0, 1.0),
            value_support_range=(-1.0, 2.0, 1.0),
            proj_hid=16,
            proj_out=16,
            pred_hid=16,
            pred_out=16,
            continuous_action_space=False,
        )
        torch.manual_seed(99)
        m_flat = SampledEfficientZeroModel(
            observation_shape=(4, 4, 4),
            action_space_size=17,
            num_res_blocks=1,
            num_channels=8,
            lstm_hidden_size=16,
            reward_head_channels=4,
            value_head_channels=4,
            policy_head_channels=4,
            reward_head_hidden_channels=[16],
            value_head_hidden_channels=[16],
            policy_head_hidden_channels=[16],
            reward_support_range=(-1.0, 2.0, 1.0),
            value_support_range=(-1.0, 2.0, 1.0),
            proj_hid=16,
            proj_out=16,
            pred_hid=16,
            pred_out=16,
            continuous_action_space=False,
            policy_head_type='flat',
        )

        assert m_default.policy_head_type == 'flat'
        assert not isinstance(m_default.prediction_network.fc_policy_head, FactoredSpatialPolicyHead)
        assert list(m_default.state_dict().keys()) == list(m_flat.state_dict().keys())
        for key in m_default.state_dict():
            assert torch.equal(m_default.state_dict()[key], m_flat.state_dict()[key])

        obs = torch.randn(3, 4, 4, 4)
        out_default = m_default.initial_inference(obs)
        out_flat = m_flat.initial_inference(obs)
        assert torch.equal(out_default.policy_logits, out_flat.policy_logits)
        assert torch.equal(out_default.value, out_flat.value)

        # Continuous + downsample=True flat model continues to work unchanged
        m_cont = SampledEfficientZeroModel(
            observation_shape=(3, 64, 64),
            action_space_size=2,
            num_res_blocks=1,
            num_channels=8,
            lstm_hidden_size=16,
            reward_head_channels=4,
            value_head_channels=4,
            policy_head_channels=4,
            reward_head_hidden_channels=[16],
            value_head_hidden_channels=[16],
            policy_head_hidden_channels=[16],
            reward_support_range=(-1.0, 2.0, 1.0),
            value_support_range=(-1.0, 2.0, 1.0),
            proj_hid=16,
            proj_out=16,
            pred_hid=16,
            pred_out=16,
            downsample=True,
            continuous_action_space=True,
            norm_type='BN',
        )
        cont_out = m_cont.initial_inference(torch.randn(2, 3, 64, 64))
        assert cont_out.policy_logits.shape == (2, 4)
