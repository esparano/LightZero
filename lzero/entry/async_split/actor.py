"""Decoupled self-play actor worker (no Ray).

Runs ``MuZeroCollector`` + ``SampledEfficientZeroPolicy`` (or other MuZero-family
policy) in an independent process, periodically polling ``latest_weights.pt``
and atomically writing completed ``GameSegment`` batches to ``segments/incoming/``.
"""

import copy
import os
import signal
import time
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed
from ditk import logging
from easydict import EasyDict
from tensorboardX import SummaryWriter

from lzero.policy import visit_count_temperature
from lzero.worker import MuZeroCollector as Collector

from .io_utils import AsyncSplitDirs, WeightWatcher, write_segment_batch


def _ding_exp_name(root_dir) -> str:
    """Return a relative path from cwd so DI-engine's ``f'./{exp_name}/...'`` stays inside ``root_dir``."""
    return os.path.relpath(str(root_dir), os.getcwd())


class SelfPlayActor:
    """Standalone self-play data generation worker.

    Can run as N independent processes (locally or on remote CPU machines
    sharing ``run_dir`` over a shared filesystem).
    """

    def __init__(
        self,
        input_cfg: Tuple[dict, dict],
        run_dir: os.PathLike | str,
        actor_id: str = 'actor_0',
        seed: int = 0,
        device: str = 'cpu',
        collector_env_num: Optional[int] = None,
        n_episode: Optional[int] = None,
        num_simulations: Optional[int] = None,
        env_manager_type: Optional[str] = None,
        weight_sync_interval: int = 1,
        wait_for_initial_weights: bool = False,
        poll_interval: float = 0.5,
        enable_tb: bool = False,
    ) -> None:
        self.dirs = AsyncSplitDirs.from_root(run_dir)
        self.dirs.ensure_exists()
        self.actor_id = str(actor_id)
        self.weight_sync_interval = max(1, int(weight_sync_interval))
        self.wait_for_initial_weights = bool(wait_for_initial_weights)
        self.poll_interval = float(poll_interval)
        self._stop_requested = False
        self._closed = False
        self._seq = 0
        self._total_episodes_written = 0
        self._total_envsteps_written = 0

        cfg = EasyDict(copy.deepcopy(input_cfg[0]))
        create_cfg = EasyDict(copy.deepcopy(input_cfg[1]))

        if device == 'auto':
            use_cuda = bool(cfg.policy.get('cuda', False) and torch.cuda.is_available())
        else:
            use_cuda = (device == 'cuda' and torch.cuda.is_available())
        cfg.policy.cuda = use_cuda
        cfg.policy.device = 'cuda' if use_cuda else 'cpu'

        if collector_env_num is not None:
            cfg.env.collector_env_num = int(collector_env_num)
            cfg.policy.collector_env_num = int(collector_env_num)
        if n_episode is not None:
            cfg.policy.n_episode = int(n_episode)
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

        env_fn, collector_env_cfg, _ = get_vec_env_setting(self.cfg.env)
        self.collector_env = create_env_manager(
            self.cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg]
        )
        self.collector_env.seed(self.cfg.seed)

        self.policy = create_policy(self.cfg.policy, model=None, enable_field=['collect'])
        self.tb_logger = (
            SummaryWriter(str(self.dirs.log_dir / f'actor_{self.actor_id}')) if enable_tb else None
        )
        self.collector = Collector(
            env=self.collector_env,
            policy=self.policy.collect_mode,
            tb_logger=self.tb_logger,
            exp_name=self.cfg.exp_name,
            instance_name=f'collector_{self.actor_id}',
            policy_config=self.cfg.policy,
        )
        self.weight_watcher = WeightWatcher(self.dirs.latest_weights_path)
        self.current_weight_iter = 0

        if self.cfg.policy.eps.eps_greedy_exploration_in_collect:
            self._epsilon_greedy_fn = get_epsilon_greedy_fn(
                start=self.cfg.policy.eps.start,
                end=self.cfg.policy.eps.end,
                decay=self.cfg.policy.eps.decay,
                type_=self.cfg.policy.eps.type,
            )
        else:
            self._epsilon_greedy_fn = None

    def sync_weights(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """Poll ``latest_weights.pt`` and reload the actor policy model if updated."""
        meta = self.weight_watcher.poll_and_load(
            self.policy._model, device=self.cfg.policy.device, force=force
        )
        if meta is not None:
            self.current_weight_iter = int(meta.get('train_iter', 0))
            logging.info(
                f'[Actor:{self.actor_id}] Reloaded weights: train_iter={self.current_weight_iter}, '
                f'envstep={meta.get("envstep", 0)}'
            )
        return meta

    def _build_collect_kwargs(self) -> Dict[str, Any]:
        policy_cfg = self.cfg.policy
        collect_kwargs: Dict[str, Any] = {
            'temperature': visit_count_temperature(
                policy_cfg.manual_temperature_decay,
                policy_cfg.fixed_temperature_value,
                policy_cfg.threshold_training_steps_for_final_temperature,
                trained_steps=self.current_weight_iter,
            )
        }
        if self._epsilon_greedy_fn is not None:
            collect_kwargs['epsilon'] = self._epsilon_greedy_fn(self.collector.envstep)
        else:
            collect_kwargs['epsilon'] = 0.0
        return collect_kwargs

    def step_once(self, n_episode: Optional[int] = None) -> Dict[str, Any]:
        """Sync weights (if due), collect one batch of episodes, and atomically write segments."""
        if self._seq % self.weight_sync_interval == 0:
            self.sync_weights()

        prev_envstep = self.collector.envstep
        collect_kwargs = self._build_collect_kwargs()
        new_data = self.collector.collect(
            n_episode=n_episode,
            train_iter=self.current_weight_iter,
            policy_kwargs=collect_kwargs,
        )
        segments, metas = new_data
        batch_envsteps = int(self.collector.envstep - prev_envstep)
        batch_episodes = sum(1 for m in metas if bool(m.get('done', False)))

        self._seq += 1
        out_path = write_segment_batch(
            incoming_dir=self.dirs.incoming_dir,
            segments=segments,
            metas=metas,
            actor_id=self.actor_id,
            seq=self._seq,
            envsteps=batch_envsteps,
            episodes=batch_episodes,
            weight_version=self.current_weight_iter,
        )
        self._total_episodes_written += batch_episodes
        self._total_envsteps_written += batch_envsteps

        return {
            'path': out_path,
            'seq': self._seq,
            'num_segments': len(segments),
            'envsteps': batch_envsteps,
            'episodes': batch_episodes,
            'weight_version': self.current_weight_iter,
        }

    def request_stop(self, *_args: Any) -> None:
        self._stop_requested = True

    def run(
        self,
        max_batches: Optional[int] = None,
        max_episodes: Optional[int] = None,
        install_signal_handlers: bool = True,
    ) -> Dict[str, Any]:
        """Run the self-play actor loop until stopped or limits are reached."""
        if install_signal_handlers:
            signal.signal(signal.SIGINT, self.request_stop)
            signal.signal(signal.SIGTERM, self.request_stop)

        if self.wait_for_initial_weights:
            logging.info(
                f'[Actor:{self.actor_id}] Waiting for initial weights at {self.dirs.latest_weights_path}...'
            )
            while not self._stop_requested and self.sync_weights() is None:
                time.sleep(self.poll_interval)

        batches = 0
        try:
            while not self._stop_requested:
                info = self.step_once()
                batches += 1
                logging.info(
                    f'[Actor:{self.actor_id}] Batch {info["seq"]}: '
                    f'segments={info["num_segments"]}, episodes={info["episodes"]}, '
                    f'envsteps={info["envsteps"]}, weight_iter={info["weight_version"]}'
                )
                if max_batches is not None and batches >= max_batches:
                    break
                if max_episodes is not None and self._total_episodes_written >= max_episodes:
                    break
        finally:
            self.close()

        return {
            'actor_id': self.actor_id,
            'batches': batches,
            'episodes': self._total_episodes_written,
            'envsteps': self._total_envsteps_written,
            'last_weight_iter': self.current_weight_iter,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.collector.close()
        except Exception:
            pass
