"""Decoupled async evaluator worker (no Ray).

Watches ``weights/latest_weights.pt``, evaluates updated policies using
LightZero's ``MuZeroEvaluator`` (with ``GalconEnv`` automatically placed in
``eval_mode`` via ``get_vec_env_setting``), logs standard TensorBoard metrics,
and atomically saves ``ckpt_best.pth.tar`` on improvement or tie.
"""

import copy
import math
import os
import signal
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ditk import logging
from easydict import EasyDict
from tensorboardX import SummaryWriter

from lzero.worker import MuZeroEvaluator as Evaluator

from .io_utils import (
    AsyncSplitDirs,
    WeightWatcher,
    atomic_json_save,
    atomic_torch_save,
    cpu_state_dict,
    read_json,
)
from .tb_utils import inspect_tb_max_steps


def _ding_exp_name(root_dir: Path) -> str:
    return os.path.relpath(str(root_dir), os.getcwd())


class AsyncEvaluator:
    """Standalone evaluator worker polling ``latest_weights.pt``."""

    def __init__(
        self,
        input_cfg: Tuple[dict, dict],
        run_dir: os.PathLike | str,
        seed: int = 0,
        device: str = 'cpu',
        evaluator_env_num: Optional[int] = None,
        n_evaluator_episode: Optional[int] = None,
        bot_action_type: Optional[str] = None,
        num_simulations: Optional[int] = None,
        env_manager_type: Optional[str] = None,
        min_iter_gap: int = 0,
        poll_interval: float = 1.0,
        resume: bool = False,
    ) -> None:
        self.dirs = AsyncSplitDirs.from_root(run_dir)
        self.dirs.ensure_exists()
        self.min_iter_gap = max(0, int(min_iter_gap))
        self.poll_interval = float(poll_interval)
        self.resume = bool(resume)
        self._stop_requested = False
        self._closed = False
        self._last_eval_iter: Optional[int] = None
        self._num_evals = 0

        cfg = EasyDict(copy.deepcopy(input_cfg[0]))
        create_cfg = EasyDict(copy.deepcopy(input_cfg[1]))

        if device == 'auto':
            use_cuda = bool(cfg.policy.get('cuda', False) and torch.cuda.is_available())
        else:
            use_cuda = (device == 'cuda' and torch.cuda.is_available())
        cfg.policy.cuda = use_cuda
        cfg.policy.device = 'cuda' if use_cuda else 'cpu'

        if evaluator_env_num is not None:
            cfg.env.evaluator_env_num = int(evaluator_env_num)
            cfg.policy.evaluator_env_num = int(evaluator_env_num)
        if n_evaluator_episode is not None:
            cfg.env.n_evaluator_episode = int(n_evaluator_episode)
        elif evaluator_env_num is not None and cfg.env.n_evaluator_episode < int(evaluator_env_num):
            cfg.env.n_evaluator_episode = int(evaluator_env_num)
        if bot_action_type is not None:
            cfg.env.bot_action_type = str(bot_action_type)
        if num_simulations is not None:
            cfg.policy.num_simulations = int(num_simulations)
        if env_manager_type is not None:
            create_cfg.env_manager.type = str(env_manager_type)

        cfg.exp_name = _ding_exp_name(self.dirs.root_dir)
        self.cfg = compile_config(
            cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=False
        )
        self.cfg.exp_name = _ding_exp_name(self.dirs.root_dir)
        self.create_cfg = create_cfg
        set_pkg_seed(self.cfg.seed, use_cuda=self.cfg.policy.cuda)

        env_fn, _, evaluator_env_cfg = get_vec_env_setting(self.cfg.env)
        self.evaluator_env = create_env_manager(
            self.cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg]
        )
        self.evaluator_env.seed(self.cfg.seed, dynamic_seed=False)

        self.policy = create_policy(self.cfg.policy, model=None, enable_field=['eval'])
        # Sub-second suffix prevents tensorboardX (1s timestamp resolution) from truncating
        # a prior session's event file on fast restarts before _restore_evaluator_state reads it.
        self.tb_logger = SummaryWriter(
            str(self.dirs.log_dir / 'evaluator'),
            filename_suffix=f'.{time.time_ns() % 1_000_000_000:09d}.{os.getpid()}',
        )
        self.evaluator = Evaluator(
            eval_freq=self.cfg.policy.eval_freq,
            n_evaluator_episode=self.cfg.env.n_evaluator_episode,
            stop_value=self.cfg.env.stop_value,
            env=self.evaluator_env,
            policy=self.policy.eval_mode,
            tb_logger=self.tb_logger,
            exp_name=self.cfg.exp_name,
            policy_config=self.cfg.policy,
        )
        self.weight_watcher = WeightWatcher(self.dirs.latest_weights_path)
        if self.resume:
            self._restore_evaluator_state()

    def _restore_evaluator_state(self) -> None:
        try:
            state = read_json(self.dirs.evaluator_state_path)
        except Exception as exc:
            logging.warning(
                f'[AsyncEvaluator] Ignoring unreadable {self.dirs.evaluator_state_path.name}: {exc}'
            )
            state = None
        if state is not None:
            if state.get('last_eval_iter') is not None:
                self._last_eval_iter = int(state['last_eval_iter'])
            if state.get('num_evals') is not None:
                self._num_evals = int(state['num_evals'])
            max_ret = state.get('max_episode_return')
            if max_ret is not None and math.isfinite(float(max_ret)):
                self.evaluator._max_episode_return = float(max_ret)

        if self._last_eval_iter is None:
            tb_stats = inspect_tb_max_steps(self.dirs.log_dir / 'evaluator')
            if tb_stats['max_iter'] is not None:
                self._last_eval_iter = int(tb_stats['max_iter'])
                logging.info(
                    f'[AsyncEvaluator] Inferred last_eval_iter={self._last_eval_iter} '
                    'from existing TensorBoard logs (evaluator_state.json absent or unset).'
                )

        if state is not None or self._last_eval_iter is not None:
            logging.info(
                f'[AsyncEvaluator] Resumed evaluator state: '
                f'last_eval_iter={self._last_eval_iter}, '
                f'max_episode_return={self.evaluator._max_episode_return}, '
                f'num_evals={self._num_evals}'
            )

    def _save_evaluator_state(self) -> None:
        max_ret = float(self.evaluator._max_episode_return)
        payload: Dict[str, Any] = {
            'version': 1,
            'last_eval_iter': self._last_eval_iter,
            'max_episode_return': max_ret if math.isfinite(max_ret) else None,
            'num_evals': int(self._num_evals),
            'timestamp': time.time(),
        }
        try:
            atomic_json_save(payload, self.dirs.evaluator_state_path)
        except Exception as exc:
            logging.warning(
                f'[AsyncEvaluator] Failed to save {self.dirs.evaluator_state_path.name}: {exc}'
            )

    def _make_save_ckpt_fn(self, train_iter: int, envstep: int):
        def _save_ckpt(ckpt_name: str) -> None:
            payload = {
                'model': cpu_state_dict(self.policy._model.state_dict()),
                'train_iter': int(train_iter),
                'envstep': int(envstep),
                'last_iter': int(train_iter),
            }
            target = self.dirs.ckpt_dir / ckpt_name
            atomic_torch_save(payload, target)
            logging.info(
                f'[AsyncEvaluator] Saved best checkpoint {target} '
                f'(train_iter={train_iter}, envstep={envstep})'
            )

        return _save_ckpt

    def eval_once_if_updated(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """Poll ``latest_weights.pt`` and run evaluation if a new eligible weight version exists."""
        meta = self.weight_watcher.poll_and_load(
            self.policy._model, device=self.cfg.policy.device, force=force
        )
        if meta is None:
            return None

        train_iter = int(meta.get('train_iter', 0))
        envstep = int(meta.get('envstep', 0))

        if (
            not force
            and self._last_eval_iter is not None
            and (
                train_iter <= self._last_eval_iter
                or (train_iter - self._last_eval_iter) < self.min_iter_gap
            )
        ):
            return None

        stop_flag, episode_info = self.evaluator.eval(
            save_ckpt_fn=self._make_save_ckpt_fn(train_iter, envstep),
            train_iter=train_iter,
            envstep=envstep,
        )
        self._last_eval_iter = train_iter
        self._num_evals += 1
        self._save_evaluator_state()
        return {
            'train_iter': train_iter,
            'envstep': envstep,
            'stop_flag': bool(stop_flag),
            'episode_info': episode_info,
        }

    def request_stop(self, *_args: Any) -> None:
        self._stop_requested = True

    def run(
        self,
        max_evals: Optional[int] = None,
        max_idle_seconds: Optional[float] = None,
        install_signal_handlers: bool = True,
    ) -> Dict[str, Any]:
        """Poll for updated weights and run evaluations until stopped or limits are reached."""
        if install_signal_handlers:
            signal.signal(signal.SIGINT, self.request_stop)
            signal.signal(signal.SIGTERM, self.request_stop)

        last_eval_time = time.time()
        last_result: Optional[Dict[str, Any]] = None
        try:
            while not self._stop_requested:
                result = self.eval_once_if_updated()
                if result is not None:
                    last_result = result
                    last_eval_time = time.time()
                    if result['stop_flag']:
                        logging.info('[AsyncEvaluator] Stop value reached; exiting.')
                        break
                    if max_evals is not None and self._num_evals >= max_evals:
                        break
                else:
                    if max_idle_seconds is not None and (time.time() - last_eval_time) >= max_idle_seconds:
                        logging.info('[AsyncEvaluator] Max idle time reached; exiting.')
                        break
                    time.sleep(self.poll_interval)
        finally:
            self.close()

        return {
            'num_evals': self._num_evals,
            'last_eval_iter': self._last_eval_iter,
            'last_result': last_result,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.evaluator.close()
        except Exception:
            pass
