import copy
import pytest
import torch

from lzero.entry.utils import load_pretrained_checkpoint
from lzero.policy.sampled_efficientzero import SampledEfficientZeroPolicy


@pytest.mark.unittest
def test_load_pretrained_checkpoint_supports_full_model_only_and_raw_state_dict(tmp_path):
    """BUG-12: eval_muzero / train_muzero must load full checkpoints ({model, target_model, optimizer}),
    model-only checkpoints ({model, train_iter} from AsyncEvaluator / latest_weights.pt),
    and raw state_dicts cleanly into both _learn_model and _target_model.
    """
    cfg = SampledEfficientZeroPolicy.default_config()
    cfg.update(
        dict(
            cuda=False,
            device='cpu',
            env_type='board_games',
            action_type='varied_action_space',
            battle_mode='self_play_mode',
            mcts_ctree=False,
            model=dict(
                model_type='conv',
                observation_shape=(4, 2, 2),
                image_channel=4,
                frame_stack_num=1,
                action_space_size=17,
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
            ),
        )
    )

    torch.manual_seed(0)
    src_policy = SampledEfficientZeroPolicy(cfg, enable_field=['learn', 'collect', 'eval'])
    with torch.no_grad():
        for p in src_policy._model.parameters():
            p.add_(0.1234)
    src_sd = copy.deepcopy(src_policy._model.state_dict())
    # Give target_model distinct weights in the full checkpoint to verify both branches
    with torch.no_grad():
        for p in src_policy._target_model.parameters():
            p.add_(0.5678)
    src_target_sd = copy.deepcopy(src_policy._target_model.state_dict())

    full_path = tmp_path / 'full_ckpt.pth.tar'
    model_only_path = tmp_path / 'ckpt_best.pth.tar'
    raw_sd_path = tmp_path / 'raw_sd.pt'

    torch.save(src_policy.learn_mode.state_dict(), full_path)
    torch.save({'model': src_sd, 'train_iter': 42, 'envstep': 100}, model_only_path)
    torch.save(src_sd, raw_sd_path)

    # Full checkpoint restores _learn_model and _target_model separately
    torch.manual_seed(99)
    dst_full = SampledEfficientZeroPolicy(cfg, enable_field=['learn', 'collect', 'eval'])
    load_pretrained_checkpoint(dst_full, str(full_path), device='cpu')
    for k, v in src_sd.items():
        assert torch.allclose(dst_full._learn_model.state_dict()[k], v)
    for k, v in src_target_sd.items():
        assert torch.allclose(dst_full._target_model.state_dict()[k], v)

    # Model-only checkpoint and raw state_dict load into both _learn_model and _target_model
    for ckpt_path in (model_only_path, raw_sd_path):
        torch.manual_seed(99)
        dst_policy = SampledEfficientZeroPolicy(cfg, enable_field=['learn', 'collect', 'eval'])
        load_pretrained_checkpoint(dst_policy, str(ckpt_path), device='cpu')
        for k, v in src_sd.items():
            assert torch.allclose(dst_policy._learn_model.state_dict()[k], v)
            assert torch.allclose(dst_policy._target_model.state_dict()[k], v)
