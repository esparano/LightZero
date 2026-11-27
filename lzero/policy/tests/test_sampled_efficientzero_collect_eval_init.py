# -*- coding: utf-8 -*-
"""Comprehensive verification tests for BUG-13:
``SampledEfficientZeroPolicy._init_scalar_transforms()`` and switching
``SelfPlayActor`` / ``AsyncEvaluator`` to ``enable_field=['collect']`` /
``enable_field=['eval']``.

Covers:
1. Init isolation & resource savings (no ``_target_model``, ``_optimizer``,
   ``lr_scheduler``, ``_learn_model``, or ``image_transforms`` in collect/eval-only
   modes; full ``state_dict`` / ``load_state_dict`` / ``get_attribute`` support).
2. RNG & initial-weight invariance across optimizer / LR-scheduler / augmentation
   configs.
3. Bit-for-bit collect equivalence (``mcts_ctree=False`` full trajectory, plus
   ``mcts_ctree=True`` real execution + full policy/MCTS boundary & simulation replay).
4. Bit-for-bit eval equivalence (``mcts_ctree=False`` full eval + ``ckpt_best.pth.tar``,
   plus ``mcts_ctree=True`` real execution + full policy/MCTS boundary & simulation replay).
5. Bit-for-bit end-to-end ``SelfPlayActor`` -> ``AsyncLearner`` (3 updates) ->
   ``AsyncEvaluator`` training equivalence.
6. Parameter and tensor-memory savings on tiny and full Galcon configs.
"""

import copy
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
from ding.policy import create_policy as ding_create_policy
from ding.utils import deep_merge_dicts
from easydict import EasyDict

import lzero.entry.async_split.actor as actor_mod
import lzero.entry.async_split.evaluator as evaluator_mod
from lzero.entry.async_split import AsyncEvaluator, AsyncLearner, SelfPlayActor
from lzero.entry.async_split.io_utils import (
    AsyncSplitDirs,
    load_config_from_py,
    publish_weights,
    read_segment_batch,
)
from lzero.policy.sampled_efficientzero import SampledEfficientZeroPolicy
from zoo.board_games.galcon.envs.galcon_env import GalconEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_tiny_policy_cfg(
    mcts_ctree: bool = False,
    optim_type: str = 'AdamW',
    cos_lr_scheduler: bool = False,
    piecewise_decay_lr_scheduler: bool = False,
    use_augmentation: bool = False,
) -> EasyDict:
    cfg = SampledEfficientZeroPolicy.default_config()
    cfg.update(
        dict(
            model=dict(
                model_type='conv',
                observation_shape=(4, 4, 4),
                image_channel=4,
                frame_stack_num=1,
                action_space_size=257,
                continuous_action_space=False,
                num_of_sampled_actions=4,
                downsample=False,
                self_supervised_learning_loss=True,
                num_res_blocks=1,
                num_channels=8,
                lstm_hidden_size=16,
                reward_head_channels=4,
                value_head_channels=4,
                policy_head_channels=4,
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
            ),
            cuda=False,
            device='cpu',
            env_type='board_games',
            action_type='varied_action_space',
            battle_mode='self_play_mode',
            mcts_ctree=mcts_ctree,
            num_simulations=4,
            optim_type=optim_type,
            learning_rate=1e-3,
            weight_decay=1e-4,
            momentum=0.9,
            cos_lr_scheduler=cos_lr_scheduler,
            piecewise_decay_lr_scheduler=piecewise_decay_lr_scheduler,
            use_augmentation=use_augmentation,
            augmentation=['shift', 'intensity'],
            discount_factor=0.99,
            num_unroll_steps=2,
            lstm_horizon_len=2,
            td_steps=4,
            game_segment_length=8,
            batch_size=4,
        )
    )
    return cfg


def _make_tiny_galcon_async_cfg(mcts_ctree: bool = False) -> Tuple[EasyDict, EasyDict]:
    """Tiny 4x4 Galcon + SEZ config for fast deterministic CPU-only tests."""
    grid_max = 50.0
    grid_square_size = 25.0
    grid_w = int(round((2 * grid_max) / grid_square_size))
    grid_h = int(round((2 * grid_max) / grid_square_size))
    action_space_size = grid_w * grid_h * grid_w * grid_h + 1

    probe_env_cfg = GalconEnv.default_config()
    probe_env_cfg.update(
        dict(
            battle_mode='self_play_mode',
            num_planets=4,
            grid_square_size=grid_square_size,
            grid_max_x=grid_max,
            grid_max_y=grid_max,
            fleet_top_k=1,
            max_episode_steps=8,
            map_seed=2,
            channel_last=False,
            scale=True,
        )
    )
    probe_obs = GalconEnv(probe_env_cfg).reset()
    c, h, w = probe_obs['observation'].shape

    main_cfg = EasyDict(
        dict(
            exp_name='galcon_bug13_test',
            env=dict(
                battle_mode='self_play_mode',
                bot_action_type='random',
                num_planets=4,
                min_send_ships=1,
                send_ratio=0.5,
                tick_seconds=0.25,
                fleet_speed=40.0,
                max_episode_steps=8,
                map_seed=2,
                grid_square_size=grid_square_size,
                grid_max_x=grid_max,
                grid_max_y=grid_max,
                home_distance_fraction=0.8,
                neutral_min_cost=0,
                neutral_max_cost=10,
                neutral_min_production=15,
                neutral_max_production=100,
                fleet_top_k=1,
                channel_last=False,
                scale=True,
                collector_env_num=2,
                evaluator_env_num=2,
                n_evaluator_episode=2,
                stop_value=2,
                manager=dict(shared_memory=False),
                agent_vs_human=False,
                prob_random_agent=0,
                prob_expert_agent=0,
                prob_random_action_in_bot=0.0,
                prob_pass_action_in_bot=0.0,
                render_mode=None,
            ),
            policy=dict(
                model=dict(
                    model_type='conv',
                    observation_shape=(c, h, w),
                    image_channel=c,
                    frame_stack_num=1,
                    action_space_size=action_space_size,
                    continuous_action_space=False,
                    num_of_sampled_actions=4,
                    downsample=False,
                    self_supervised_learning_loss=True,
                    num_res_blocks=1,
                    num_channels=8,
                    lstm_hidden_size=16,
                    reward_head_channels=4,
                    value_head_channels=4,
                    policy_head_channels=4,
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
                ),
                cuda=False,
                env_type='board_games',
                action_type='varied_action_space',
                battle_mode='self_play_mode',
                mcts_ctree=mcts_ctree,
                game_segment_length=8,
                update_per_collect=3,
                batch_size=4,
                optim_type='AdamW',
                piecewise_decay_lr_scheduler=False,
                learning_rate=1e-3,
                grad_clip_value=1.0,
                num_simulations=4,
                manual_temperature_decay=False,
                fixed_temperature_value=1.0,
                reanalyze_ratio=0.0,
                td_steps=4,
                allow_episode_horizon_mismatch=True,
                discount_factor=0.99,
                num_unroll_steps=2,
                lstm_horizon_len=2,
                policy_loss_type='cross_entropy',
                use_priority=False,
                n_episode=2,
                eval_freq=1000,
                replay_buffer_size=500,
                collector_env_num=2,
                evaluator_env_num=2,
                symmetric_augment_x=True,
                symmetric_augment_y=True,
                symmetric_augment_type='Galcon',
            ),
        )
    )
    create_cfg = EasyDict(
        dict(
            env=dict(
                type='galcon',
                import_names=['zoo.board_games.galcon.envs.galcon_env'],
            ),
            env_manager=dict(type='base'),
            policy=dict(
                type='sampled_efficientzero',
                import_names=['lzero.policy.sampled_efficientzero'],
            ),
        )
    )
    return main_cfg, create_cfg


def _assert_state_dicts_bit_identical(sd_a: Dict[str, Any], sd_b: Dict[str, Any]) -> None:
    assert list(sd_a.keys()) == list(sd_b.keys())
    for key in sd_a:
        val_a, val_b = sd_a[key], sd_b[key]
        if torch.is_tensor(val_a):
            assert torch.equal(val_a.detach().cpu(), val_b.detach().cpu()), f'Tensor mismatch for key {key}'
        else:
            assert val_a == val_b, f'Value mismatch for key {key}: {val_a!r} vs {val_b!r}'


def _assert_child_visits_bit_identical(cv_a: Any, cv_b: Any) -> None:
    assert len(cv_a) == len(cv_b)
    if isinstance(cv_a, np.ndarray) and cv_a.dtype != object and isinstance(cv_b, np.ndarray) and cv_b.dtype != object:
        assert np.array_equal(cv_a, cv_b)
    else:
        for row_a, row_b in zip(cv_a, cv_b):
            assert np.array_equal(np.asarray(row_a), np.asarray(row_b))


def _assert_segments_bit_identical(
    segs_a: List[Any],
    metas_a: List[Dict[str, Any]],
    segs_b: List[Any],
    metas_b: List[Dict[str, Any]],
) -> None:
    assert len(segs_a) == len(segs_b) and len(segs_a) > 0
    assert len(metas_a) == len(metas_b) == len(segs_a)
    for idx, (sa, sb) in enumerate(zip(segs_a, segs_b)):
        assert np.array_equal(sa.obs_segment, sb.obs_segment), f'obs_segment mismatch at segment {idx}'
        assert np.array_equal(sa.action_segment, sb.action_segment), f'action_segment mismatch at segment {idx}'
        assert np.array_equal(sa.reward_segment, sb.reward_segment), f'reward_segment mismatch at segment {idx}'
        _assert_child_visits_bit_identical(sa.child_visit_segment, sb.child_visit_segment)
        assert np.array_equal(sa.root_value_segment, sb.root_value_segment), (
            f'root_value_segment mismatch at segment {idx}'
        )
        assert len(sa.root_sampled_actions) == len(sb.root_sampled_actions)
        for step_idx, (ra, rb) in enumerate(zip(sa.root_sampled_actions, sb.root_sampled_actions)):
            assert np.array_equal(np.asarray(ra), np.asarray(rb)), (
                f'root_sampled_actions mismatch at segment {idx}, step {step_idx}'
            )
        assert np.array_equal(sa.to_play_segment, sb.to_play_segment), f'to_play_segment mismatch at segment {idx}'
        assert np.array_equal(sa.action_mask_segment, sb.action_mask_segment), (
            f'action_mask_segment mismatch at segment {idx}'
        )
        assert sa.valid_transition_count == sb.valid_transition_count
    for idx, (ma, mb) in enumerate(zip(metas_a, metas_b)):
        assert bool(ma['done']) == bool(mb['done']), f'done meta mismatch at segment {idx}'
        assert ma['unroll_plus_td_steps'] == mb['unroll_plus_td_steps']
        if ma['priorities'] is None or mb['priorities'] is None:
            assert ma['priorities'] is None and mb['priorities'] is None
        else:
            assert np.array_equal(np.asarray(ma['priorities']), np.asarray(mb['priorities']))


def _assert_episode_info_bit_identical(info_a: Dict[str, Any], info_b: Dict[str, Any]) -> None:
    assert set(info_a.keys()) == set(info_b.keys())
    for key in info_a:
        va, vb = info_a[key], info_b[key]
        if isinstance(va, (list, tuple, np.ndarray)):
            assert np.array_equal(np.asarray(va), np.asarray(vb)), f'episode_info[{key}] mismatch'
        else:
            assert va == vb, f'episode_info[{key}] mismatch: {va!r} vs {vb!r}'


def _unwrap_module(mod: Any) -> Any:
    while hasattr(mod, '_model') and mod._model is not mod:
        mod = mod._model
    return mod


def _policy_model_memory_stats(policy: Any) -> Dict[str, int]:
    """Count unique ``nn.Module`` instances, parameters, buffers, and tensor bytes held by ``policy``."""
    seen: Dict[int, torch.nn.Module] = {}
    for attr in ('_model', '_target_model', '_learn_model', '_collect_model', '_eval_model'):
        raw = getattr(policy, attr, None)
        if raw is None:
            continue
        mod = _unwrap_module(raw)
        if isinstance(mod, torch.nn.Module):
            seen[id(mod)] = mod

    total_params = 0
    total_buffers = 0
    total_param_bytes = 0
    total_buffer_bytes = 0
    for mod in seen.values():
        for p in mod.parameters():
            total_params += p.numel()
            total_param_bytes += p.numel() * p.element_size()
        for b in mod.buffers():
            total_buffers += b.numel()
            total_buffer_bytes += b.numel() * b.element_size()

    return {
        'num_model_copies': len(seen),
        'total_params': total_params,
        'total_buffers': total_buffers,
        'total_param_bytes': total_param_bytes,
        'total_buffer_bytes': total_buffer_bytes,
        'total_bytes': total_param_bytes + total_buffer_bytes,
    }


def _publish_initial_weights(run_dir: Path, cfg_pair: Tuple[EasyDict, EasyDict], seed: int = 123, train_iter: int = 0, envstep: int = 0) -> Dict[str, torch.Tensor]:
    """Build a deterministic model state_dict and atomically publish it to ``latest_weights.pt``."""
    dirs = AsyncSplitDirs.from_root(run_dir)
    dirs.ensure_exists()
    _seed_all(seed)
    tmp_policy = SampledEfficientZeroPolicy(
        _make_tiny_galcon_policy_only_cfg(cfg_pair[0].policy),
        enable_field=['eval'],
    )
    sd = copy.deepcopy(tmp_policy._model.state_dict())
    publish_weights(dirs.latest_weights_path, sd, train_iter=train_iter, envstep=envstep)
    return sd


def _make_tiny_galcon_policy_only_cfg(policy_dict: EasyDict) -> EasyDict:
    cfg = deep_merge_dicts(SampledEfficientZeroPolicy.default_config(), copy.deepcopy(policy_dict))
    cfg.device = 'cpu'
    cfg.cuda = False
    return cfg


def _monkeypatch_actor_enable_field(monkeypatch: pytest.MonkeyPatch, enable_field: List[str]) -> None:
    def _patched_create_policy(cfg, model=None, enable_field=None, **kwargs):
        return ding_create_policy(cfg, model=model, enable_field=list(forced_fields), **kwargs)

    forced_fields = list(enable_field)
    monkeypatch.setattr(actor_mod, 'create_policy', _patched_create_policy)


def _monkeypatch_evaluator_enable_field(monkeypatch: pytest.MonkeyPatch, enable_field: List[str]) -> None:
    def _patched_create_policy(cfg, model=None, enable_field=None, **kwargs):
        return ding_create_policy(cfg, model=model, enable_field=list(forced_fields), **kwargs)

    forced_fields = list(enable_field)
    monkeypatch.setattr(evaluator_mod, 'create_policy', _patched_create_policy)


# ---------------------------------------------------------------------------
# 1. Init isolation, mode APIs, and actor/evaluator resource checks
# ---------------------------------------------------------------------------


@pytest.mark.unittest
def test_collect_only_init_isolates_learn_resources_and_supports_mode_apis():
    cfg = _make_tiny_policy_cfg(
        mcts_ctree=False,
        cos_lr_scheduler=True,
        piecewise_decay_lr_scheduler=True,
        use_augmentation=True,
    )
    policy = SampledEfficientZeroPolicy(cfg, enable_field=['collect'])

    # Scalar transforms and collect-mode state are initialized cleanly
    assert hasattr(policy, 'value_support')
    assert hasattr(policy, 'reward_support')
    assert hasattr(policy, 'value_inverse_scalar_transform_handle')
    assert hasattr(policy, 'reward_inverse_scalar_transform_handle')
    assert policy.value_support.size == policy._model.value_support_size
    assert policy.reward_support.size == policy._model.reward_support_size
    assert policy._collect_model is policy._model
    assert hasattr(policy, '_mcts_collect')
    assert policy._collect_mcts_temperature == 1

    # Learn-only and eval-only resources must NOT be allocated
    for forbidden in (
        '_target_model',
        '_learn_model',
        '_optimizer',
        'lr_scheduler',
        'image_transforms',
        '_eval_model',
        '_mcts_eval',
    ):
        assert not hasattr(policy, forbidden), f'Unexpected attribute {forbidden} in collect-only policy'

    # Mode APIs: state_dict, load_state_dict, get_attribute, repr, forward
    sd = policy.collect_mode.state_dict()
    assert list(sd.keys()) == ['model']
    _assert_state_dicts_bit_identical(sd['model'], policy._model.state_dict())

    modified_sd = {k: (v + 0.25 if v.is_floating_point() else v.clone()) for k, v in sd['model'].items()}
    policy.collect_mode.load_state_dict({'model': modified_sd})
    _assert_state_dicts_bit_identical(policy._model.state_dict(), modified_sd)

    assert policy.collect_mode.get_attribute('cfg') is policy._cfg
    assert 'DI-engine DRL Policy' in repr(policy)

    obs = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
    mask = [np.ones(cfg.model.action_space_size, dtype=np.int8)]
    out = policy.collect_mode.forward(obs, action_mask=mask, temperature=1.0, to_play=[1])
    assert 0 in out
    assert 0 <= int(out[0]['action']) < cfg.model.action_space_size
    assert np.isfinite(float(np.asarray(out[0]['predicted_value']).reshape(-1)[0]))
    assert np.isfinite(float(out[0]['searched_value']))


@pytest.mark.unittest
def test_eval_only_init_isolates_learn_resources_and_supports_mode_apis():
    cfg = _make_tiny_policy_cfg(
        mcts_ctree=False,
        cos_lr_scheduler=True,
        piecewise_decay_lr_scheduler=True,
        use_augmentation=True,
    )
    policy = SampledEfficientZeroPolicy(cfg, enable_field=['eval'])

    # Scalar transforms and eval-mode state are initialized cleanly
    assert hasattr(policy, 'value_support')
    assert hasattr(policy, 'reward_support')
    assert hasattr(policy, 'value_inverse_scalar_transform_handle')
    assert hasattr(policy, 'reward_inverse_scalar_transform_handle')
    assert policy.value_support.size == policy._model.value_support_size
    assert policy.reward_support.size == policy._model.reward_support_size
    assert policy._eval_model is policy._model
    assert hasattr(policy, '_mcts_eval')

    # Learn-only and collect-only resources must NOT be allocated
    for forbidden in (
        '_target_model',
        '_learn_model',
        '_optimizer',
        'lr_scheduler',
        'image_transforms',
        '_collect_model',
        '_mcts_collect',
        '_collect_mcts_temperature',
    ):
        assert not hasattr(policy, forbidden), f'Unexpected attribute {forbidden} in eval-only policy'

    # Mode APIs: state_dict, load_state_dict, get_attribute, repr, forward
    sd = policy.eval_mode.state_dict()
    assert list(sd.keys()) == ['model']
    _assert_state_dicts_bit_identical(sd['model'], policy._model.state_dict())

    modified_sd = {k: (v - 0.125 if v.is_floating_point() else v.clone()) for k, v in sd['model'].items()}
    policy.eval_mode.load_state_dict({'model': modified_sd})
    _assert_state_dicts_bit_identical(policy._model.state_dict(), modified_sd)

    assert policy.eval_mode.get_attribute('cfg') is policy._cfg
    assert 'DI-engine DRL Policy' in repr(policy)

    obs = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
    mask = [np.ones(cfg.model.action_space_size, dtype=np.int8)]
    out = policy.eval_mode.forward(obs, action_mask=mask, to_play=[-1])
    assert 0 in out
    assert 0 <= int(out[0]['action']) < cfg.model.action_space_size
    assert np.isfinite(float(np.asarray(out[0]['predicted_value']).reshape(-1)[0]))
    assert np.isfinite(float(out[0]['searched_value']))


@pytest.mark.unittest
@pytest.mark.parametrize(
    'enable_field',
    [
        ['learn', 'collect', 'eval'],
        ['collect', 'eval', 'learn'],
        None,
    ],
)
def test_full_and_reordered_enable_fields_and_scalar_transform_idempotency(enable_field):
    cfg = _make_tiny_policy_cfg(piecewise_decay_lr_scheduler=True, use_augmentation=True)
    policy = SampledEfficientZeroPolicy(cfg, enable_field=enable_field)

    for required in (
        '_learn_model',
        '_target_model',
        '_optimizer',
        'lr_scheduler',
        'image_transforms',
        '_collect_model',
        '_mcts_collect',
        '_collect_mcts_temperature',
        '_eval_model',
        '_mcts_eval',
        'value_support',
        'reward_support',
        'value_inverse_scalar_transform_handle',
        'reward_inverse_scalar_transform_handle',
    ):
        assert hasattr(policy, required), f'Missing {required} for enable_field={enable_field}'

    assert policy._learn_model is policy._model
    assert policy._collect_model is policy._model
    assert policy._eval_model is policy._model
    assert _unwrap_module(policy._target_model) is not policy._model

    # Idempotency: repeated calls keep the exact same transform instances
    vs = policy.value_support
    rs = policy.reward_support
    vh = policy.value_inverse_scalar_transform_handle
    rh = policy.reward_inverse_scalar_transform_handle
    policy._init_scalar_transforms()
    assert policy.value_support is vs
    assert policy.reward_support is rs
    assert policy.value_inverse_scalar_transform_handle is vh
    assert policy.reward_inverse_scalar_transform_handle is rh


@pytest.mark.unittest
def test_actor_and_evaluator_do_not_allocate_learn_only_state(tmp_path):
    cfg_pair = _make_tiny_galcon_async_cfg(mcts_ctree=True)
    run_dir = tmp_path / 'isolation_check'

    actor = SelfPlayActor(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        actor_id='actor_iso',
        seed=5,
        device='cpu',
        env_manager_type='base',
    )
    evaluator = AsyncEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_dir,
        seed=6,
        device='cpu',
        env_manager_type='base',
    )
    try:
        assert list(actor.policy._enable_field) == ['collect']
        assert list(evaluator.policy._enable_field) == ['eval']

        for worker_name, pol in (('actor', actor.policy), ('evaluator', evaluator.policy)):
            assert hasattr(pol, 'value_support'), worker_name
            assert hasattr(pol, 'reward_support'), worker_name
            assert hasattr(pol, 'value_inverse_scalar_transform_handle'), worker_name
            assert hasattr(pol, 'reward_inverse_scalar_transform_handle'), worker_name
            for forbidden in ('_target_model', '_learn_model', '_optimizer', 'lr_scheduler', 'image_transforms'):
                assert not hasattr(pol, forbidden), f'{worker_name}.policy unexpectedly has {forbidden}'
    finally:
        actor.close()
        evaluator.close()


# ---------------------------------------------------------------------------
# 2. RNG & initial-weight invariance
# ---------------------------------------------------------------------------


@pytest.mark.unittest
@pytest.mark.parametrize(
    'baseline_fields,optimized_fields',
    [
        (['learn', 'collect'], ['collect']),
        (['learn', 'eval'], ['eval']),
    ],
)
@pytest.mark.parametrize('optim_type', ['AdamW', 'Adam', 'SGD'])
@pytest.mark.parametrize('lr_mode', ['none', 'piecewise', 'cosine'])
@pytest.mark.parametrize('use_augmentation', [False, True])
def test_rng_and_initial_weight_invariance_across_enable_fields(
    baseline_fields, optimized_fields, optim_type, lr_mode, use_augmentation
):
    cfg = _make_tiny_policy_cfg(
        optim_type=optim_type,
        piecewise_decay_lr_scheduler=(lr_mode == 'piecewise'),
        cos_lr_scheduler=(lr_mode == 'cosine'),
        use_augmentation=use_augmentation,
    )

    _seed_all(2026)
    pol_base = SampledEfficientZeroPolicy(copy.deepcopy(cfg), enable_field=baseline_fields)
    post_torch_base = torch.randn(16)
    post_np_base = np.random.rand(16)
    post_py_base = [random.random() for _ in range(16)]

    _seed_all(2026)
    pol_opt = SampledEfficientZeroPolicy(copy.deepcopy(cfg), enable_field=optimized_fields)
    post_torch_opt = torch.randn(16)
    post_np_opt = np.random.rand(16)
    post_py_opt = [random.random() for _ in range(16)]

    _assert_state_dicts_bit_identical(pol_base._model.state_dict(), pol_opt._model.state_dict())
    assert pol_base._model.training == pol_opt._model.training
    assert torch.equal(pol_base.value_support.arange, pol_opt.value_support.arange)
    assert torch.equal(pol_base.reward_support.arange, pol_opt.reward_support.arange)

    # Post-init RNG streams must be bit-for-bit identical across torch, numpy, and Python random
    assert torch.equal(post_torch_base, post_torch_opt)
    assert np.array_equal(post_np_base, post_np_opt)
    assert post_py_base == post_py_opt


# ---------------------------------------------------------------------------
# 3. Bit-for-bit collect equivalence (ptree & ctree)
# ---------------------------------------------------------------------------


@pytest.mark.unittest
def test_actor_step_once_bit_identical_ptree(tmp_path, monkeypatch):
    """With mcts_ctree=False (100% deterministic Python MCTS), SelfPlayActor.step_once(n_episode=2)
    produces bit-for-bit identical GameSegments and metadata for ['learn', 'collect'] vs ['collect'].
    """
    cfg_pair = _make_tiny_galcon_async_cfg(mcts_ctree=False)
    run_base = tmp_path / 'collect_ptree_base'
    run_opt = tmp_path / 'collect_ptree_opt'

    _publish_initial_weights(run_base, cfg_pair, seed=100, train_iter=0, envstep=0)
    _publish_initial_weights(run_opt, cfg_pair, seed=100, train_iter=0, envstep=0)

    # Baseline run: enable_field=['learn', 'collect']
    with monkeypatch.context() as mp:
        _monkeypatch_actor_enable_field(mp, ['learn', 'collect'])
        actor_base = SelfPlayActor(
            input_cfg=cfg_pair,
            run_dir=run_base,
            actor_id='actor_0',
            seed=10,
            device='cpu',
            env_manager_type='base',
            weight_sync_interval=1,
        )
        try:
            assert hasattr(actor_base.policy, '_target_model')
            info_base = actor_base.step_once(n_episode=2)
        finally:
            actor_base.close()

    # BUG-13 run: production enable_field=['collect']
    actor_opt = SelfPlayActor(
        input_cfg=cfg_pair,
        run_dir=run_opt,
        actor_id='actor_0',
        seed=10,
        device='cpu',
        env_manager_type='base',
        weight_sync_interval=1,
    )
    try:
        assert not hasattr(actor_opt.policy, '_target_model')
        info_opt = actor_opt.step_once(n_episode=2)
    finally:
        actor_opt.close()

    assert info_base['num_segments'] == info_opt['num_segments']
    assert info_base['episodes'] == info_opt['episodes'] == 2
    assert info_base['envsteps'] == info_opt['envsteps'] > 0
    assert info_base['weight_version'] == info_opt['weight_version'] == 0

    batch_base = read_segment_batch(info_base['path'])
    batch_opt = read_segment_batch(info_opt['path'])
    _assert_segments_bit_identical(
        batch_base['segments'], batch_base['metas'], batch_opt['segments'], batch_opt['metas']
    )


@pytest.mark.unittest
def test_actor_step_once_bit_identical_ctree_and_simulation_replay(tmp_path, monkeypatch):
    """With mcts_ctree=True:
    Note that C++ ``cnode.cpp`` seeds its internal action-sampling RNG from
    ``std::chrono::system_clock::now()`` (line 336) and UCB tie-breaking from
    ``get_time_and_set_rand_seed()`` (line 1197), so two wall-clock-separated C++
    runs draw different C++ random permutations even under identical Python/PyTorch
    inputs. We verify:
    1. Both ``['learn', 'collect']`` and ``['collect']`` execute real C++ ``ctree``
       ``SelfPlayActor.step_once(n_episode=2)`` cleanly and feed bit-for-bit identical
       root inference tensors, Dirichlet noises, and legal actions into step-0 C++
       ``roots.prepare``.
    2. Replaying every single ``initial_inference`` + ``value_inverse_scalar_transform_handle``
       + ``recurrent_inference`` + ``_mcts_collect.value_inverse_scalar_transform_handle``
       call recorded across the full 2-episode C++ ``ctree`` rollout through the
       ``['collect']`` policy yields bit-for-bit identical tensors at every ply and
       every MCTS simulation.
    3. Holding C++ ``roots`` outcomes identical (replaying the exact C++ ``roots``
       search outputs from the baseline run while executing all real Python/PyTorch
       policy + collector code) yields bit-for-bit identical ``GameSegment`` fields.
    """
    cfg_pair = _make_tiny_galcon_async_cfg(mcts_ctree=True)
    run_base = tmp_path / 'collect_ctree_base'
    run_opt = tmp_path / 'collect_ctree_opt'
    run_opt_replay = tmp_path / 'collect_ctree_opt_replay'

    for d in (run_base, run_opt, run_opt_replay):
        _publish_initial_weights(d, cfg_pair, seed=101, train_iter=0, envstep=0)

    recorded_init_calls: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    recorded_rec_calls: List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Dict[str, Any]]] = []
    recorded_step0_prepare_base: Dict[str, Any] = {}
    recorded_root_outputs: List[Dict[str, Any]] = []

    # 1. Baseline real C++ ctree run (enable_field=['learn', 'collect'])
    with monkeypatch.context() as mp:
        _monkeypatch_actor_enable_field(mp, ['learn', 'collect'])
        actor_base = SelfPlayActor(
            input_cfg=cfg_pair,
            run_dir=run_base,
            actor_id='actor_0',
            seed=10,
            device='cpu',
            env_manager_type='base',
            weight_sync_interval=1,
        )
        try:
            pol_base = actor_base.policy
            orig_init = pol_base._collect_model.initial_inference
            orig_rec = pol_base._collect_model.recurrent_inference
            orig_search = pol_base._mcts_collect.search

            def _spy_init(obs):
                out = orig_init(obs)
                pred_val = pol_base.value_inverse_scalar_transform_handle(out.value).detach().cpu().numpy()
                recorded_init_calls.append(
                    (
                        obs.detach().clone(),
                        {
                            'latent_state': out.latent_state.detach().clone(),
                            'value': out.value.detach().clone(),
                            'policy_logits': out.policy_logits.detach().clone(),
                            'rhs0': out.reward_hidden_state[0].detach().clone(),
                            'rhs1': out.reward_hidden_state[1].detach().clone(),
                            'pred_val': pred_val.copy(),
                        },
                    )
                )
                return out

            def _spy_rec(latent_states, rhs, actions):
                out = orig_rec(latent_states, rhs, actions)
                mcts = pol_base._mcts_collect
                recorded_rec_calls.append(
                    (
                        latent_states.detach().clone(),
                        (rhs[0].detach().clone(), rhs[1].detach().clone()),
                        actions.detach().clone(),
                        {
                            'latent_state': out.latent_state.detach().clone(),
                            'policy_logits': out.policy_logits.detach().clone(),
                            'value': out.value.detach().clone(),
                            'value_prefix': out.value_prefix.detach().clone(),
                            'rhs0': out.reward_hidden_state[0].detach().clone(),
                            'rhs1': out.reward_hidden_state[1].detach().clone(),
                            'val_scalar': mcts.value_inverse_scalar_transform_handle(out.value).detach().clone(),
                            'vp_scalar': mcts.value_inverse_scalar_transform_handle(out.value_prefix).detach().clone(),
                        },
                    )
                )
                return out

            def _spy_search(roots, model, latent_state_roots, reward_hidden_state_roots, to_play):
                if not recorded_step0_prepare_base:
                    recorded_step0_prepare_base['latent_state_roots'] = np.array(latent_state_roots, copy=True)
                    recorded_step0_prepare_base['rhs0'] = np.array(reward_hidden_state_roots[0], copy=True)
                    recorded_step0_prepare_base['rhs1'] = np.array(reward_hidden_state_roots[1], copy=True)
                    recorded_step0_prepare_base['to_play'] = list(to_play)
                orig_search(roots, model, latent_state_roots, reward_hidden_state_roots, to_play)
                recorded_root_outputs.append(
                    {
                        'distributions': copy.deepcopy(roots.get_distributions()),
                        'values': copy.deepcopy(roots.get_values()),
                        'sampled_actions': copy.deepcopy(roots.get_sampled_actions()),
                    }
                )

            pol_base._collect_model.initial_inference = _spy_init
            pol_base._collect_model.recurrent_inference = _spy_rec
            pol_base._mcts_collect.search = _spy_search

            info_base = actor_base.step_once(n_episode=2)
        finally:
            actor_base.close()

    assert len(recorded_init_calls) > 0
    assert len(recorded_rec_calls) > 0
    assert len(recorded_root_outputs) == len(recorded_init_calls)
    batch_base = read_segment_batch(info_base['path'])

    # 2. Production real C++ ctree run (enable_field=['collect']) + full simulation replay
    recorded_step0_prepare_opt: Dict[str, Any] = {}
    actor_opt = SelfPlayActor(
        input_cfg=cfg_pair,
        run_dir=run_opt,
        actor_id='actor_0',
        seed=10,
        device='cpu',
        env_manager_type='base',
        weight_sync_interval=1,
    )
    try:
        pol_opt = actor_opt.policy
        assert not hasattr(pol_opt, '_target_model')
        orig_search_opt = pol_opt._mcts_collect.search

        def _spy_search_opt(roots, model, latent_state_roots, reward_hidden_state_roots, to_play):
            if not recorded_step0_prepare_opt:
                recorded_step0_prepare_opt['latent_state_roots'] = np.array(latent_state_roots, copy=True)
                recorded_step0_prepare_opt['rhs0'] = np.array(reward_hidden_state_roots[0], copy=True)
                recorded_step0_prepare_opt['rhs1'] = np.array(reward_hidden_state_roots[1], copy=True)
                recorded_step0_prepare_opt['to_play'] = list(to_play)
            return orig_search_opt(roots, model, latent_state_roots, reward_hidden_state_roots, to_play)

        pol_opt._mcts_collect.search = _spy_search_opt
        info_opt = actor_opt.step_once(n_episode=2)
        assert info_opt['episodes'] == 2
        assert info_opt['num_segments'] >= 2
        assert info_opt['envsteps'] > 0

        # Step-0 inputs entering C++ ctree search must be bit-for-bit identical
        assert np.array_equal(
            recorded_step0_prepare_base['latent_state_roots'], recorded_step0_prepare_opt['latent_state_roots']
        )
        assert np.array_equal(recorded_step0_prepare_base['rhs0'], recorded_step0_prepare_opt['rhs0'])
        assert np.array_equal(recorded_step0_prepare_base['rhs1'], recorded_step0_prepare_opt['rhs1'])
        assert recorded_step0_prepare_base['to_play'] == recorded_step0_prepare_opt['to_play']

        # Replay EVERY initial_inference & recurrent_inference call from the baseline ctree run
        pol_opt._collect_model.eval()
        with torch.no_grad():
            for obs, expected in recorded_init_calls:
                out = pol_opt._collect_model.initial_inference(obs)
                pred_val = pol_opt.value_inverse_scalar_transform_handle(out.value).detach().cpu().numpy()
                assert torch.equal(out.latent_state, expected['latent_state'])
                assert torch.equal(out.value, expected['value'])
                assert torch.equal(out.policy_logits, expected['policy_logits'])
                assert torch.equal(out.reward_hidden_state[0], expected['rhs0'])
                assert torch.equal(out.reward_hidden_state[1], expected['rhs1'])
                assert np.array_equal(pred_val, expected['pred_val'])

            mcts_opt = pol_opt._mcts_collect
            for latent_s, rhs, acts, expected in recorded_rec_calls:
                out = pol_opt._collect_model.recurrent_inference(latent_s, rhs, acts)
                val_scalar = mcts_opt.value_inverse_scalar_transform_handle(out.value)
                vp_scalar = mcts_opt.value_inverse_scalar_transform_handle(out.value_prefix)
                assert torch.equal(out.latent_state, expected['latent_state'])
                assert torch.equal(out.policy_logits, expected['policy_logits'])
                assert torch.equal(out.value, expected['value'])
                assert torch.equal(out.value_prefix, expected['value_prefix'])
                assert torch.equal(out.reward_hidden_state[0], expected['rhs0'])
                assert torch.equal(out.reward_hidden_state[1], expected['rhs1'])
                assert torch.equal(val_scalar, expected['val_scalar'])
                assert torch.equal(vp_scalar, expected['vp_scalar'])
    finally:
        actor_opt.close()

    # 3. Full GameSegment equivalence when C++ wall-clock root outcomes are held identical
    import lzero.policy.sampled_efficientzero as sez_mod

    orig_ctree_roots = sez_mod.MCTSCtree.roots
    call_idx = {'i': 0}

    class _ReplayRootsProxy:
        def __init__(self, real_roots, snapshot):
            self._real_roots = real_roots
            self._snapshot = snapshot

        def __getattr__(self, name):
            return getattr(self._real_roots, name)

        def get_distributions(self):
            return copy.deepcopy(self._snapshot['distributions'])

        def get_values(self):
            return copy.deepcopy(self._snapshot['values'])

        def get_sampled_actions(self):
            return copy.deepcopy(self._snapshot['sampled_actions'])

    def _replay_roots(cls, *args, **kwargs):
        real_roots = orig_ctree_roots(*args, **kwargs)
        snap = recorded_root_outputs[call_idx['i']]
        call_idx['i'] += 1
        return _ReplayRootsProxy(real_roots, snap)

    actor_replay = SelfPlayActor(
        input_cfg=cfg_pair,
        run_dir=run_opt_replay,
        actor_id='actor_0',
        seed=10,
        device='cpu',
        env_manager_type='base',
        weight_sync_interval=1,
    )
    try:
        # Still run real C++ search on _real_roots, then return the baseline's recorded C++ root summary
        orig_replay_search = actor_replay.policy._mcts_collect.search

        def _unwrap_search(roots, model, latent_state_roots, reward_hidden_state_roots, to_play):
            return orig_replay_search(
                roots._real_roots, model, latent_state_roots, reward_hidden_state_roots, to_play
            )

        actor_replay.policy._mcts_collect.search = _unwrap_search
        with monkeypatch.context() as mp:
            mp.setattr(sez_mod.MCTSCtree, 'roots', classmethod(_replay_roots))
            info_replay = actor_replay.step_once(n_episode=2)
    finally:
        actor_replay.close()

    assert call_idx['i'] == len(recorded_root_outputs)
    batch_replay = read_segment_batch(info_replay['path'])
    _assert_segments_bit_identical(
        batch_base['segments'], batch_base['metas'], batch_replay['segments'], batch_replay['metas']
    )


# ---------------------------------------------------------------------------
# 4. Bit-for-bit eval equivalence (ptree & ctree)
# ---------------------------------------------------------------------------


@pytest.mark.unittest
def test_evaluator_eval_once_bit_identical_ptree(tmp_path, monkeypatch):
    """With mcts_ctree=False (100% deterministic Python MCTS), AsyncEvaluator.eval_once_if_updated()
    produces bit-for-bit identical episode returns, episode lengths, and ckpt_best.pth.tar weights
    for ['learn', 'eval'] vs ['eval'].
    """
    cfg_pair = _make_tiny_galcon_async_cfg(mcts_ctree=False)
    run_base = tmp_path / 'eval_ptree_base'
    run_opt = tmp_path / 'eval_ptree_opt'

    _publish_initial_weights(run_base, cfg_pair, seed=200, train_iter=5, envstep=40)
    _publish_initial_weights(run_opt, cfg_pair, seed=200, train_iter=5, envstep=40)

    def _run_eval(run_dir: Path, force_fields=None):
        tb_scalars: Dict[str, float] = {}
        ctx = monkeypatch.context()
        with ctx as mp:
            if force_fields is not None:
                _monkeypatch_evaluator_enable_field(mp, force_fields)
            ev = AsyncEvaluator(
                input_cfg=cfg_pair,
                run_dir=run_dir,
                seed=20,
                device='cpu',
                evaluator_env_num=2,
                n_evaluator_episode=2,
                env_manager_type='base',
                min_iter_gap=0,
            )
            try:
                if force_fields is not None:
                    assert hasattr(ev.policy, '_target_model')
                else:
                    assert not hasattr(ev.policy, '_target_model')

                orig_add_scalar = ev.tb_logger.add_scalar

                def _spy_add_scalar(tag, scalar_value, global_step=None, *args, **kwargs):
                    tb_scalars[tag] = float(scalar_value)
                    return orig_add_scalar(tag, scalar_value, global_step, *args, **kwargs)

                ev.tb_logger.add_scalar = _spy_add_scalar
                result = ev.eval_once_if_updated()
            finally:
                ev.close()
        ckpt = torch.load(str(run_dir / 'ckpt' / 'ckpt_best.pth.tar'), map_location='cpu', weights_only=False)
        return result, tb_scalars, ckpt

    res_base, tb_base, ckpt_base = _run_eval(run_base, force_fields=['learn', 'eval'])
    res_opt, tb_opt, ckpt_opt = _run_eval(run_opt, force_fields=None)

    assert res_base is not None and res_opt is not None
    assert res_base['train_iter'] == res_opt['train_iter'] == 5
    assert res_base['envstep'] == res_opt['envstep'] == 40
    assert res_base['stop_flag'] == res_opt['stop_flag']
    _assert_episode_info_bit_identical(res_base['episode_info'], res_opt['episode_info'])

    # Episode lengths and return means logged to TensorBoard must match bit-for-bit
    assert tb_base['evaluator_iter/eval/episode_length'] == tb_opt['evaluator_iter/eval/episode_length']
    assert tb_base['evaluator_iter/eval_episode_return_mean'] == tb_opt['evaluator_iter/eval_episode_return_mean']
    assert tb_base['evaluator_iter/avg_envstep_per_episode'] == tb_opt['evaluator_iter/avg_envstep_per_episode']
    assert tb_base['evaluator_iter/envstep_count'] == tb_opt['evaluator_iter/envstep_count']

    # Saved ckpt_best.pth.tar must match bit-for-bit
    assert ckpt_base['train_iter'] == ckpt_opt['train_iter'] == 5
    assert ckpt_base['envstep'] == ckpt_opt['envstep'] == 40
    assert ckpt_base['last_iter'] == ckpt_opt['last_iter'] == 5
    _assert_state_dicts_bit_identical(ckpt_base['model'], ckpt_opt['model'])


@pytest.mark.unittest
def test_evaluator_eval_once_bit_identical_ctree_and_simulation_replay(tmp_path, monkeypatch):
    """With mcts_ctree=True:
    Verify real C++ ``ctree`` ``AsyncEvaluator.eval_once_if_updated()`` succeeds cleanly with
    ``enable_field=['eval']``, produces bit-for-bit identical step-0 root inputs and
    ``ckpt_best.pth.tar`` weights, reproduces every single ``initial_inference`` and
    ``recurrent_inference`` + inverse scalar transform tensor bit-for-bit across all
    eval steps & simulations, and produces bit-for-bit identical episode returns and
    episode lengths when C++ root summaries are held identical.
    """
    cfg_pair = _make_tiny_galcon_async_cfg(mcts_ctree=True)
    run_base = tmp_path / 'eval_ctree_base'
    run_opt = tmp_path / 'eval_ctree_opt'
    run_replay = tmp_path / 'eval_ctree_replay'

    for d in (run_base, run_opt, run_replay):
        _publish_initial_weights(d, cfg_pair, seed=201, train_iter=7, envstep=56)

    recorded_init_calls: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    recorded_rec_calls: List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Dict[str, Any]]] = []
    recorded_step0_base: Dict[str, Any] = {}
    recorded_root_outputs: List[Dict[str, Any]] = []
    tb_base: Dict[str, float] = {}

    # 1. Baseline real C++ ctree eval (enable_field=['learn', 'eval'])
    with monkeypatch.context() as mp:
        _monkeypatch_evaluator_enable_field(mp, ['learn', 'eval'])
        ev_base = AsyncEvaluator(
            input_cfg=cfg_pair,
            run_dir=run_base,
            seed=20,
            device='cpu',
            evaluator_env_num=2,
            n_evaluator_episode=2,
            env_manager_type='base',
            min_iter_gap=0,
        )
        try:
            pol_base = ev_base.policy
            orig_init = pol_base._eval_model.initial_inference
            orig_rec = pol_base._eval_model.recurrent_inference
            orig_search = pol_base._mcts_eval.search
            orig_add_scalar = ev_base.tb_logger.add_scalar

            def _spy_add_scalar(tag, scalar_value, global_step=None, *args, **kwargs):
                tb_base[tag] = float(scalar_value)
                return orig_add_scalar(tag, scalar_value, global_step, *args, **kwargs)

            def _spy_init(obs):
                out = orig_init(obs)
                pred_val = pol_base.value_inverse_scalar_transform_handle(out.value).detach().cpu().numpy()
                recorded_init_calls.append(
                    (
                        obs.detach().clone(),
                        {
                            'latent_state': out.latent_state.detach().clone(),
                            'value': out.value.detach().clone(),
                            'policy_logits': out.policy_logits.detach().clone(),
                            'rhs0': out.reward_hidden_state[0].detach().clone(),
                            'rhs1': out.reward_hidden_state[1].detach().clone(),
                            'pred_val': pred_val.copy(),
                        },
                    )
                )
                return out

            def _spy_rec(latent_states, rhs, actions):
                out = orig_rec(latent_states, rhs, actions)
                mcts = pol_base._mcts_eval
                recorded_rec_calls.append(
                    (
                        latent_states.detach().clone(),
                        (rhs[0].detach().clone(), rhs[1].detach().clone()),
                        actions.detach().clone(),
                        {
                            'latent_state': out.latent_state.detach().clone(),
                            'policy_logits': out.policy_logits.detach().clone(),
                            'value': out.value.detach().clone(),
                            'value_prefix': out.value_prefix.detach().clone(),
                            'rhs0': out.reward_hidden_state[0].detach().clone(),
                            'rhs1': out.reward_hidden_state[1].detach().clone(),
                            'val_scalar': mcts.value_inverse_scalar_transform_handle(out.value).detach().clone(),
                            'vp_scalar': mcts.value_inverse_scalar_transform_handle(out.value_prefix).detach().clone(),
                        },
                    )
                )
                return out

            def _spy_search(roots, model, latent_state_roots, reward_hidden_state_roots, to_play):
                if not recorded_step0_base:
                    recorded_step0_base['latent_state_roots'] = np.array(latent_state_roots, copy=True)
                    recorded_step0_base['rhs0'] = np.array(reward_hidden_state_roots[0], copy=True)
                    recorded_step0_base['rhs1'] = np.array(reward_hidden_state_roots[1], copy=True)
                    recorded_step0_base['to_play'] = list(to_play)
                orig_search(roots, model, latent_state_roots, reward_hidden_state_roots, to_play)
                recorded_root_outputs.append(
                    {
                        'distributions': copy.deepcopy(roots.get_distributions()),
                        'values': copy.deepcopy(roots.get_values()),
                        'sampled_actions': copy.deepcopy(roots.get_sampled_actions()),
                    }
                )

            ev_base.tb_logger.add_scalar = _spy_add_scalar
            pol_base._eval_model.initial_inference = _spy_init
            pol_base._eval_model.recurrent_inference = _spy_rec
            pol_base._mcts_eval.search = _spy_search

            res_base = ev_base.eval_once_if_updated()
        finally:
            ev_base.close()

    ckpt_base = torch.load(str(run_base / 'ckpt' / 'ckpt_best.pth.tar'), map_location='cpu', weights_only=False)
    assert len(recorded_init_calls) > 0 and len(recorded_rec_calls) > 0

    # 2. Production real C++ ctree eval (enable_field=['eval']) + full simulation replay
    recorded_step0_opt: Dict[str, Any] = {}
    ev_opt = AsyncEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_opt,
        seed=20,
        device='cpu',
        evaluator_env_num=2,
        n_evaluator_episode=2,
        env_manager_type='base',
        min_iter_gap=0,
    )
    try:
        pol_opt = ev_opt.policy
        assert not hasattr(pol_opt, '_target_model')
        orig_search_opt = pol_opt._mcts_eval.search

        def _spy_search_opt(roots, model, latent_state_roots, reward_hidden_state_roots, to_play):
            if not recorded_step0_opt:
                recorded_step0_opt['latent_state_roots'] = np.array(latent_state_roots, copy=True)
                recorded_step0_opt['rhs0'] = np.array(reward_hidden_state_roots[0], copy=True)
                recorded_step0_opt['rhs1'] = np.array(reward_hidden_state_roots[1], copy=True)
                recorded_step0_opt['to_play'] = list(to_play)
            return orig_search_opt(roots, model, latent_state_roots, reward_hidden_state_roots, to_play)

        pol_opt._mcts_eval.search = _spy_search_opt
        res_opt = ev_opt.eval_once_if_updated()
        assert res_opt is not None and res_opt['train_iter'] == 7

        # Step-0 inputs entering C++ ctree search must match bit-for-bit
        assert np.array_equal(recorded_step0_base['latent_state_roots'], recorded_step0_opt['latent_state_roots'])
        assert np.array_equal(recorded_step0_base['rhs0'], recorded_step0_opt['rhs0'])
        assert np.array_equal(recorded_step0_base['rhs1'], recorded_step0_opt['rhs1'])
        assert recorded_step0_base['to_play'] == recorded_step0_opt['to_play']

        # Replay every recorded neural-net + inverse scalar transform call bit-for-bit
        pol_opt._eval_model.eval()
        with torch.no_grad():
            for obs, expected in recorded_init_calls:
                out = pol_opt._eval_model.initial_inference(obs)
                pred_val = pol_opt.value_inverse_scalar_transform_handle(out.value).detach().cpu().numpy()
                assert torch.equal(out.latent_state, expected['latent_state'])
                assert torch.equal(out.value, expected['value'])
                assert torch.equal(out.policy_logits, expected['policy_logits'])
                assert torch.equal(out.reward_hidden_state[0], expected['rhs0'])
                assert torch.equal(out.reward_hidden_state[1], expected['rhs1'])
                assert np.array_equal(pred_val, expected['pred_val'])

            mcts_opt = pol_opt._mcts_eval
            for latent_s, rhs, acts, expected in recorded_rec_calls:
                out = pol_opt._eval_model.recurrent_inference(latent_s, rhs, acts)
                val_scalar = mcts_opt.value_inverse_scalar_transform_handle(out.value)
                vp_scalar = mcts_opt.value_inverse_scalar_transform_handle(out.value_prefix)
                assert torch.equal(out.latent_state, expected['latent_state'])
                assert torch.equal(out.policy_logits, expected['policy_logits'])
                assert torch.equal(out.value, expected['value'])
                assert torch.equal(out.value_prefix, expected['value_prefix'])
                assert torch.equal(out.reward_hidden_state[0], expected['rhs0'])
                assert torch.equal(out.reward_hidden_state[1], expected['rhs1'])
                assert torch.equal(val_scalar, expected['val_scalar'])
                assert torch.equal(vp_scalar, expected['vp_scalar'])
    finally:
        ev_opt.close()

    ckpt_opt = torch.load(str(run_opt / 'ckpt' / 'ckpt_best.pth.tar'), map_location='cpu', weights_only=False)
    _assert_state_dicts_bit_identical(ckpt_base['model'], ckpt_opt['model'])

    # 3. Full eval returns & lengths equivalence when C++ root summaries are held identical
    import lzero.policy.sampled_efficientzero as sez_mod

    orig_ctree_roots = sez_mod.MCTSCtree.roots
    call_idx = {'i': 0}

    class _ReplayRootsProxy:
        def __init__(self, real_roots, snapshot):
            self._real_roots = real_roots
            self._snapshot = snapshot

        def __getattr__(self, name):
            return getattr(self._real_roots, name)

        def get_distributions(self):
            return copy.deepcopy(self._snapshot['distributions'])

        def get_values(self):
            return copy.deepcopy(self._snapshot['values'])

        def get_sampled_actions(self):
            return copy.deepcopy(self._snapshot['sampled_actions'])

    def _replay_roots(cls, *args, **kwargs):
        real_roots = orig_ctree_roots(*args, **kwargs)
        snap = recorded_root_outputs[call_idx['i']]
        call_idx['i'] += 1
        return _ReplayRootsProxy(real_roots, snap)

    tb_replay: Dict[str, float] = {}
    ev_replay = AsyncEvaluator(
        input_cfg=cfg_pair,
        run_dir=run_replay,
        seed=20,
        device='cpu',
        evaluator_env_num=2,
        n_evaluator_episode=2,
        env_manager_type='base',
        min_iter_gap=0,
    )
    try:
        orig_replay_search = ev_replay.policy._mcts_eval.search
        orig_replay_scalar = ev_replay.tb_logger.add_scalar

        def _unwrap_search(roots, model, latent_state_roots, reward_hidden_state_roots, to_play):
            return orig_replay_search(
                roots._real_roots, model, latent_state_roots, reward_hidden_state_roots, to_play
            )

        def _spy_replay_scalar(tag, scalar_value, global_step=None, *args, **kwargs):
            tb_replay[tag] = float(scalar_value)
            return orig_replay_scalar(tag, scalar_value, global_step, *args, **kwargs)

        ev_replay.policy._mcts_eval.search = _unwrap_search
        ev_replay.tb_logger.add_scalar = _spy_replay_scalar
        with monkeypatch.context() as mp:
            mp.setattr(sez_mod.MCTSCtree, 'roots', classmethod(_replay_roots))
            res_replay = ev_replay.eval_once_if_updated()
    finally:
        ev_replay.close()

    assert call_idx['i'] == len(recorded_root_outputs)
    _assert_episode_info_bit_identical(res_base['episode_info'], res_replay['episode_info'])
    assert tb_base['evaluator_iter/eval/episode_length'] == tb_replay['evaluator_iter/eval/episode_length']
    assert tb_base['evaluator_iter/eval_episode_return_mean'] == tb_replay['evaluator_iter/eval_episode_return_mean']
    ckpt_replay = torch.load(str(run_replay / 'ckpt' / 'ckpt_best.pth.tar'), map_location='cpu', weights_only=False)
    _assert_state_dicts_bit_identical(ckpt_base['model'], ckpt_replay['model'])


# ---------------------------------------------------------------------------
# 5. Bit-for-bit end-to-end training equivalence
# ---------------------------------------------------------------------------


@pytest.mark.unittest
def test_end_to_end_actor_learner_evaluator_bit_identical(tmp_path, monkeypatch):
    """Full SelfPlayActor -> AsyncLearner (3 updates) -> AsyncEvaluator cycle produces
    bit-for-bit identical segments, learner losses at every update, post-update _learn_model
    and _target_model weights, and evaluator returns/lengths/ckpt_best.pth.tar when actor &
    evaluator use ['collect']/['eval'] vs ['learn', 'collect']/['learn', 'eval'].
    """
    cfg_pair = _make_tiny_galcon_async_cfg(mcts_ctree=False)

    def _run_full_cycle(run_dir: Path, patch_legacy_fields: bool) -> Dict[str, Any]:
        with monkeypatch.context() as mp:
            if patch_legacy_fields:
                _monkeypatch_actor_enable_field(mp, ['learn', 'collect'])
                _monkeypatch_evaluator_enable_field(mp, ['learn', 'eval'])

            learner = AsyncLearner(
                input_cfg=cfg_pair,
                run_dir=run_dir,
                seed=0,
                device='cpu',
                publish_every_n_updates=3,
                save_ckpt_every_n_updates=0,
                max_replay_ratio=None,
                min_transitions=4,
                archive_segments=True,
                use_bg_reader=False,
            )
            actor = SelfPlayActor(
                input_cfg=cfg_pair,
                run_dir=run_dir,
                actor_id='actor_0',
                seed=10,
                device='cpu',
                env_manager_type='base',
                weight_sync_interval=1,
            )
            evaluator = AsyncEvaluator(
                input_cfg=cfg_pair,
                run_dir=run_dir,
                seed=20,
                device='cpu',
                evaluator_env_num=2,
                n_evaluator_episode=2,
                env_manager_type='base',
                min_iter_gap=0,
            )

            eval_tb: Dict[str, float] = {}
            orig_eval_add_scalar = evaluator.tb_logger.add_scalar

            def _spy_eval_scalar(tag, scalar_value, global_step=None, *args, **kwargs):
                eval_tb[tag] = float(scalar_value)
                return orig_eval_add_scalar(tag, scalar_value, global_step, *args, **kwargs)

            evaluator.tb_logger.add_scalar = _spy_eval_scalar

            try:
                if patch_legacy_fields:
                    assert hasattr(actor.policy, '_target_model')
                    assert hasattr(evaluator.policy, '_target_model')
                else:
                    assert not hasattr(actor.policy, '_target_model')
                    assert not hasattr(evaluator.policy, '_target_model')

                # 1. Actor collects 2 episodes on iter-0 weights
                batch_info = actor.step_once(n_episode=2)
                batch_payload = read_segment_batch(batch_info['path'])

                # 2. Learner ingests and runs 3 deterministic updates
                ingest_stats = learner.ingest_pending_segments()
                assert ingest_stats.files_ingested == 1

                losses: List[Dict[str, float]] = []
                for _ in range(3):
                    lv = learner.train_step()
                    assert lv is not None
                    losses.append(
                        {
                            'total_loss': float(lv['total_loss']),
                            'weighted_total_loss': float(lv['weighted_total_loss']),
                            'value_loss': float(lv['value_loss']),
                            'policy_loss': float(lv['policy_loss']),
                            'value_prefix_loss': float(lv['value_prefix_loss']),
                            'consistency_loss': float(lv['consistency_loss']),
                            'policy_entropy': float(lv['policy_entropy']),
                            'total_grad_norm_before_clip': float(lv['total_grad_norm_before_clip']),
                        }
                    )
                assert learner.learner.train_iter == 3

                # Ensure distinct mtime_ns on coarse-mtime filesystems before evaluator polls
                st = learner.dirs.latest_weights_path.stat()
                os.utime(learner.dirs.latest_weights_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

                learn_model_sd = copy.deepcopy(learner.policy._learn_model.state_dict())
                target_model_sd = copy.deepcopy(learner.policy._target_model.state_dict())

                # 3. Evaluator evaluates iter-3 weights
                eval_result = evaluator.eval_once_if_updated()
                assert eval_result is not None and eval_result['train_iter'] == 3
                best_ckpt = torch.load(
                    str(evaluator.dirs.ckpt_dir / 'ckpt_best.pth.tar'),
                    map_location='cpu',
                    weights_only=False,
                )

                return {
                    'segments': batch_payload['segments'],
                    'metas': batch_payload['metas'],
                    'losses': losses,
                    'learn_model_sd': learn_model_sd,
                    'target_model_sd': target_model_sd,
                    'eval_result': eval_result,
                    'eval_tb': eval_tb,
                    'best_ckpt': best_ckpt,
                }
            finally:
                actor.close()
                evaluator.close()
                learner.close()

    base = _run_full_cycle(tmp_path / 'e2e_base', patch_legacy_fields=True)
    opt = _run_full_cycle(tmp_path / 'e2e_opt', patch_legacy_fields=False)

    # 1. Segments bit-for-bit identical
    _assert_segments_bit_identical(base['segments'], base['metas'], opt['segments'], opt['metas'])

    # 2. All 3 learner update loss dicts bit-for-bit identical
    assert len(base['losses']) == len(opt['losses']) == 3
    for step_idx, (lb, lo) in enumerate(zip(base['losses'], opt['losses'])):
        for key in lb:
            assert lb[key] == lo[key], f'Update {step_idx + 1} loss[{key}] mismatch: {lb[key]!r} vs {lo[key]!r}'

    # 3. Post-update _learn_model and _target_model weights bit-for-bit identical
    _assert_state_dicts_bit_identical(base['learn_model_sd'], opt['learn_model_sd'])
    _assert_state_dicts_bit_identical(base['target_model_sd'], opt['target_model_sd'])

    # 4. Evaluator outputs and ckpt_best.pth.tar bit-for-bit identical
    _assert_episode_info_bit_identical(base['eval_result']['episode_info'], opt['eval_result']['episode_info'])
    assert base['eval_tb']['evaluator_iter/eval/episode_length'] == opt['eval_tb']['evaluator_iter/eval/episode_length']
    assert (
        base['eval_tb']['evaluator_iter/eval_episode_return_mean']
        == opt['eval_tb']['evaluator_iter/eval_episode_return_mean']
    )
    _assert_state_dicts_bit_identical(base['best_ckpt']['model'], opt['best_ckpt']['model'])


# ---------------------------------------------------------------------------
# 6. Memory / parameter savings check (tiny & full Galcon configs)
# ---------------------------------------------------------------------------


@pytest.mark.unittest
@pytest.mark.parametrize(
    'config_source',
    [
        'tiny_galcon',
        'zoo/board_games/galcon/config/galcon_sez_sp_mode_config.py',
        'zoo/board_games/galcon/config/galcon_sez_sp_mode_config_SIMPLE.py',
    ],
)
def test_memory_and_parameter_savings_tiny_and_full_galcon(config_source):
    if config_source == 'tiny_galcon':
        main_cfg, _ = _make_tiny_galcon_async_cfg(mcts_ctree=True)
    else:
        main_cfg, _ = load_config_from_py(Path(config_source))

    policy_cfg = _make_tiny_galcon_policy_only_cfg(main_cfg.policy)

    pol_legacy_actor = SampledEfficientZeroPolicy(copy.deepcopy(policy_cfg), enable_field=['learn', 'collect'])
    pol_opt_actor = SampledEfficientZeroPolicy(copy.deepcopy(policy_cfg), enable_field=['collect'])
    pol_legacy_eval = SampledEfficientZeroPolicy(copy.deepcopy(policy_cfg), enable_field=['learn', 'eval'])
    pol_opt_eval = SampledEfficientZeroPolicy(copy.deepcopy(policy_cfg), enable_field=['eval'])

    for legacy, opt in ((pol_legacy_actor, pol_opt_actor), (pol_legacy_eval, pol_opt_eval)):
        assert hasattr(legacy, '_target_model') and hasattr(legacy, '_optimizer')
        assert not hasattr(opt, '_target_model') and not hasattr(opt, '_optimizer')

        stats_legacy = _policy_model_memory_stats(legacy)
        stats_opt = _policy_model_memory_stats(opt)

        assert stats_legacy['num_model_copies'] == 2
        assert stats_opt['num_model_copies'] == 1
        assert stats_legacy['total_params'] == 2 * stats_opt['total_params']
        assert stats_legacy['total_buffers'] == 2 * stats_opt['total_buffers']
        assert stats_legacy['total_bytes'] == 2 * stats_opt['total_bytes']
        assert stats_opt['total_params'] > 0
        assert stats_opt['total_bytes'] > 0
