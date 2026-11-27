import numpy as np
import pytest
from easydict import EasyDict

from ding.envs import BaseEnvTimestep
from lzero.worker.muzero_evaluator import MuZeroEvaluator, balanced_episode_targets


@pytest.mark.parametrize(
    'env_num,n_episode,expected',
    [
        (3, 3, [1, 1, 1]),
        (3, 5, [2, 2, 1]),
        (3, 8, [3, 3, 2]),
    ],
)
def test_balanced_episode_targets_match_vector_eval_monitor(env_num, n_episode, expected):
    targets = balanced_episode_targets(env_num, n_episode)

    np.testing.assert_array_equal(targets, expected)
    assert targets.sum() == n_episode


def test_balanced_episode_targets_reject_too_few_episodes():
    with pytest.raises(ValueError, match='at least env_num'):
        balanced_episode_targets(env_num=3, n_episode=2)


class _StubSingleStepEnvManager:
    """Minimal env manager returning scripted episode returns on a single step per episode."""

    def __init__(self, scripted_returns):
        self._returns = list(scripted_returns)
        self._idx = 0
        self.env_num = 1
        self.action_space = None
        self.closed = False
        self._obs = None

    def _make_obs(self):
        return {
            'observation': np.zeros((4,), dtype=np.float32),
            'action_mask': np.ones((2,), dtype=np.int8),
            'to_play': -1,
            'timestep': 0,
        }

    def launch(self):
        pass

    def close(self):
        self.closed = True

    def reset(self):
        self._obs = {0: self._make_obs()}
        return self._obs

    @property
    def ready_obs(self):
        return self._obs

    def step(self, actions):
        ret = float(self._returns[self._idx])
        self._idx += 1
        obs = self._make_obs()
        self._obs = {0: obs}
        ts = BaseEnvTimestep(
            obs=obs,
            reward=np.array(ret, dtype=np.float32),
            done=True,
            info={'eval_episode_return': np.array(ret, dtype=np.float32)},
        )
        return {0: ts}


class _StubEvalPolicy:
    def reset(self, *args, **kwargs):
        pass

    def get_attribute(self, name):
        if name == 'cfg':
            return EasyDict(dict(type='sampled_efficientzero'))
        return None

    def forward(self, data, action_mask, to_play, ready_env_id=None, **kwargs):
        return {
            env_id: {
                'action': 0,
                'visit_count_distributions': [1, 0],
                'searched_value': 0.0,
                'predicted_value': 0.0,
                'visit_count_distribution_entropy': 0.0,
            }
            for env_id in ready_env_id
        }


class _NoopTBLogger:
    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def test_muzero_evaluator_saves_ckpt_best_on_improvement_and_tie(tmp_path):
    """BUG-11: MuZeroEvaluator.eval previously saved WM_ckpt_best.pth.tar first and mutated
    self._max_episode_return before checking ckpt_best.pth.tar, leaving ckpt_best.pth.tar dead.
    It should save only ckpt_best.pth.tar on improvement or tie (>=) and nothing on regression.
    """
    env = _StubSingleStepEnvManager(scripted_returns=[0.2, 0.2, 0.8, 0.5])
    policy_cfg = EasyDict(
        dict(
            device='cpu',
            game_segment_length=4,
            num_unroll_steps=1,
            td_steps=1,
            discount_factor=1.0,
            gray_scale=False,
            transform2string=False,
            sampled_algo=False,
            gumbel_algo=False,
            use_ture_chance_label_in_chance_encoder=False,
            use_wandb=False,
            model=dict(
                model_type='mlp',
                frame_stack_num=1,
                observation_shape=4,
                action_space_size=2,
            ),
        )
    )
    evaluator = MuZeroEvaluator(
        eval_freq=1,
        n_evaluator_episode=1,
        stop_value=10.0,
        env=env,
        policy=_StubEvalPolicy(),
        tb_logger=_NoopTBLogger(),
        exp_name=str(tmp_path / 'eval_exp'),
        policy_config=policy_cfg,
    )

    saved = []

    def _record_ckpt(name):
        saved.append(name)

    # 1. First eval (0.2 > -inf): saves ckpt_best.pth.tar
    evaluator.eval(save_ckpt_fn=_record_ckpt, train_iter=1, envstep=10)
    assert saved == ['ckpt_best.pth.tar']

    # 2. Tie (0.2 == 0.2): overwrites ckpt_best.pth.tar with newer weights
    saved.clear()
    evaluator.eval(save_ckpt_fn=_record_ckpt, train_iter=2, envstep=20)
    assert saved == ['ckpt_best.pth.tar']

    # 3. Strict improvement (0.8 > 0.2): saves ckpt_best.pth.tar
    saved.clear()
    evaluator.eval(save_ckpt_fn=_record_ckpt, train_iter=3, envstep=30)
    assert saved == ['ckpt_best.pth.tar']

    # 4. Regression (0.5 < 0.8): saves nothing
    saved.clear()
    evaluator.eval(save_ckpt_fn=_record_ckpt, train_iter=4, envstep=40)
    assert saved == []
