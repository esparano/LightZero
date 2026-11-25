"""Unified CLI for the decoupled actor-learner-evaluator split (no Ray).

Usage examples:
    python3 -m lzero.entry.async_split learner --config zoo/board_games/galcon/config/galcon_sez_sp_mode_config_SIMPLE.py --run-dir data_sez/galcon_async_run1
    python3 -m lzero.entry.async_split actor --config ... --run-dir data_sez/galcon_async_run1 --actor-id actor_0 --device cpu
    python3 -m lzero.entry.async_split evaluator --config ... --run-dir data_sez/galcon_async_run1 --device cpu --n-episode 20
    python3 -m lzero.entry.async_split launch --config ... --run-dir data_sez/galcon_async_run1 --num-actors 2
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ditk import logging

from .actor import SelfPlayActor
from .evaluator import AsyncEvaluator
from .io_utils import AsyncSplitDirs, load_config_from_py
from .learner import AsyncLearner

DEFAULT_GALCON_CONFIG = 'zoo/board_games/galcon/config/galcon_sez_sp_mode_config_SIMPLE.py'
_THREAD_ENV_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')


def _resolve_run_dir(run_dir: Optional[str], cfg) -> str:
    return run_dir if run_dir else str(cfg.exp_name)


def _compute_default_worker_threads(
    num_actors: int,
    include_evaluator: bool = True,
    cpu_count: Optional[int] = None,
) -> int:
    """Choose a conservative per-worker BLAS/OpenMP thread cap so multi-actor runs
    do not oversubscribe CPU cores when OMP_NUM_THREADS is unset.
    """
    cpus = max(1, int(cpu_count if cpu_count is not None else (os.cpu_count() or 1)))
    total_workers = max(1, max(0, int(num_actors)) + (1 if include_evaluator else 0) + 1)
    return max(1, min(4, cpus // total_workers))


def _build_worker_env(
    num_actors: int,
    include_evaluator: bool = True,
    worker_threads: Optional[int] = None,
    base_env: Optional[Mapping[str, str]] = None,
    cpu_count: Optional[int] = None,
) -> Dict[str, str]:
    """Build subprocess env with sensible default OpenMP/BLAS thread caps.

    - If ``worker_threads`` is explicitly provided (> 0), it overrides all thread vars.
    - Otherwise, any variable already present in ``base_env`` is preserved, and missing
      ones default to ``_compute_default_worker_threads(...)``.
    """
    env = dict(os.environ if base_env is None else base_env)
    if worker_threads is not None:
        if int(worker_threads) <= 0:
            raise ValueError(f'worker_threads must be >= 1, got {worker_threads}')
        val = str(int(worker_threads))
        for var in _THREAD_ENV_VARS:
            env[var] = val
    else:
        default_val = str(
            _compute_default_worker_threads(
                num_actors=num_actors,
                include_evaluator=include_evaluator,
                cpu_count=cpu_count,
            )
        )
        for var in _THREAD_ENV_VARS:
            env.setdefault(var, default_val)
    return env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m lzero.entry.async_split',
        description='Lightweight decoupled actor-learner-evaluator split for LightZero (no Ray).',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # --- actor ---
    p_actor = sub.add_parser('actor', help='Run a standalone self-play actor worker.')
    p_actor.add_argument('--config', default=DEFAULT_GALCON_CONFIG, help='Path to LightZero Python config.')
    p_actor.add_argument('--run-dir', default=None, help='Shared run directory (defaults to cfg.exp_name).')
    p_actor.add_argument('--actor-id', default='actor_0', help='Unique actor identifier.')
    p_actor.add_argument('--seed', type=int, default=0, help='Random seed for this actor.')
    p_actor.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'auto'], help='Inference device.')
    p_actor.add_argument('--env-num', type=int, default=None, help='Override collector_env_num.')
    p_actor.add_argument('--n-episode', type=int, default=None, help='Override episodes per collect batch.')
    p_actor.add_argument('--num-simulations', type=int, default=None, help='Override MCTS num_simulations.')
    p_actor.add_argument('--env-manager', default=None, choices=['base', 'subprocess'], help='Override env_manager type.')
    p_actor.add_argument('--weight-sync-interval', type=int, default=1, help='Sync weights every N batches.')
    p_actor.add_argument('--wait-for-initial-weights', action='store_true', help='Wait for learner weights before collecting.')
    p_actor.add_argument('--poll-interval', type=float, default=0.5, help='Polling interval (seconds) when waiting.')
    p_actor.add_argument('--max-batches', type=int, default=None, help='Stop after N collect batches.')
    p_actor.add_argument('--max-episodes', type=int, default=None, help='Stop after N completed episodes.')
    p_actor.add_argument('--enable-tb', action='store_true', help='Enable per-actor TensorBoard logging.')

    # --- learner ---
    p_learner = sub.add_parser('learner', help='Run the standalone replay-buffer + training learner worker.')
    p_learner.add_argument('--config', default=DEFAULT_GALCON_CONFIG, help='Path to LightZero Python config.')
    p_learner.add_argument('--run-dir', default=None, help='Shared run directory (defaults to cfg.exp_name).')
    p_learner.add_argument('--seed', type=int, default=0, help='Random seed.')
    p_learner.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'auto'], help='Training device.')
    p_learner.add_argument('--model-path', default=None, help='Optional pretrained checkpoint path.')
    p_learner.add_argument('--publish-every', type=int, default=25, help='Publish latest_weights.pt every N updates.')
    p_learner.add_argument('--save-ckpt-every', type=int, default=1000, help='Save iteration checkpoint every N updates (0 disables).')
    p_learner.add_argument(
        '--max-replay-ratio',
        type=float,
        default=0.5,
        help='Max gradient updates per raw ingested transition (<= 0 disables throttling).',
    )
    p_learner.add_argument('--min-transitions', type=int, default=None, help='Minimum sampleable transitions before training.')
    p_learner.add_argument('--poll-interval', type=float, default=0.25, help='Polling interval (seconds) when idle.')
    p_learner.add_argument('--max-train-iter', type=int, default=int(1e10), help='Stop after N learner updates.')
    p_learner.add_argument('--max-env-step', type=int, default=int(1e10), help='Stop after N ingested env steps.')
    p_learner.add_argument('--max-idle-seconds', type=float, default=None, help='Optional idle timeout before exiting.')
    p_learner.add_argument('--archive-segments', action='store_true', help='Move consumed segments to segments/processed/ instead of deleting.')
    p_learner.add_argument('--no-bg-reader', action='store_true', help='Disable background disk reader thread.')
    p_learner.add_argument(
        '--prefetch-workers',
        type=int,
        default=0,
        help='Number of background replay-buffer batch prefetch worker threads (0 disables).',
    )
    p_learner.add_argument(
        '--prefetch-queue-size',
        type=int,
        default=None,
        help='Max queued prefetched batches (defaults to max(2, prefetch_workers * 2)).',
    )

    # --- evaluator ---
    p_eval = sub.add_parser('evaluator', help='Run the standalone async evaluator worker.')
    p_eval.add_argument('--config', default=DEFAULT_GALCON_CONFIG, help='Path to LightZero Python config.')
    p_eval.add_argument('--run-dir', default=None, help='Shared run directory (defaults to cfg.exp_name).')
    p_eval.add_argument('--seed', type=int, default=0, help='Random seed.')
    p_eval.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'auto'], help='Evaluation device.')
    p_eval.add_argument('--env-num', type=int, default=None, help='Override evaluator_env_num.')
    p_eval.add_argument('--n-episode', type=int, default=None, help='Override n_evaluator_episode (e.g. 20).')
    p_eval.add_argument('--bot-action-type', default=None, help='Override opponent bot type (e.g. random, pass, fixed).')
    p_eval.add_argument('--num-simulations', type=int, default=None, help='Override MCTS num_simulations.')
    p_eval.add_argument('--env-manager', default=None, choices=['base', 'subprocess'], help='Override env_manager type.')
    p_eval.add_argument('--min-iter-gap', type=int, default=0, help='Minimum learner iterations between evaluations.')
    p_eval.add_argument('--poll-interval', type=float, default=2.0, help='Polling interval (seconds) for new weights.')
    p_eval.add_argument('--max-evals', type=int, default=None, help='Stop after N evaluations.')
    p_eval.add_argument('--max-idle-seconds', type=float, default=None, help='Optional idle timeout before exiting.')

    # --- launch ---
    p_launch = sub.add_parser('launch', help='Launch local learner + N actors + evaluator as OS subprocesses.')
    p_launch.add_argument('--config', default=DEFAULT_GALCON_CONFIG, help='Path to LightZero Python config.')
    p_launch.add_argument('--run-dir', default=None, help='Shared run directory (defaults to cfg.exp_name).')
    p_launch.add_argument('--seed', type=int, default=0, help='Base seed (actors use seed + 100 + i).')
    p_launch.add_argument('--num-actors', type=int, default=2, help='Number of self-play actor processes.')
    p_launch.add_argument('--actor-device', default='cpu', choices=['cpu', 'cuda', 'auto'])
    p_launch.add_argument('--actor-env-num', type=int, default=None)
    p_launch.add_argument('--actor-n-episode', type=int, default=None)
    p_launch.add_argument('--learner-device', default='auto', choices=['cpu', 'cuda', 'auto'])
    p_launch.add_argument('--model-path', default=None)
    p_launch.add_argument('--publish-every', type=int, default=25)
    p_launch.add_argument('--save-ckpt-every', type=int, default=1000)
    p_launch.add_argument('--max-replay-ratio', type=float, default=0.5)
    p_launch.add_argument('--min-transitions', type=int, default=None)
    p_launch.add_argument('--max-train-iter', type=int, default=int(1e10))
    p_launch.add_argument('--eval-device', default='cpu', choices=['cpu', 'cuda', 'auto'])
    p_launch.add_argument('--eval-env-num', type=int, default=None)
    p_launch.add_argument('--eval-n-episode', type=int, default=None)
    p_launch.add_argument('--eval-bot', default=None)
    p_launch.add_argument('--eval-min-iter-gap', type=int, default=500)
    p_launch.add_argument('--no-evaluator', action='store_true')
    p_launch.add_argument('--env-manager', default=None, choices=['base', 'subprocess'])
    p_launch.add_argument(
        '--prefetch-workers',
        type=int,
        default=0,
        help='Number of background replay-buffer batch prefetch worker threads on learner (0 disables).',
    )
    p_launch.add_argument(
        '--prefetch-queue-size',
        type=int,
        default=None,
        help='Max queued prefetched batches on learner (defaults to max(2, prefetch_workers * 2)).',
    )
    p_launch.add_argument(
        '--worker-threads',
        type=int,
        default=None,
        help=(
            'Per-worker OMP/MKL/OPENBLAS thread cap. When omitted, respects existing '
            'OMP/MKL/OPENBLAS env vars and defaults unset ones to min(4, max(1, cpus // total_workers)).'
        ),
    )

    return parser


def _run_launch(args: argparse.Namespace) -> int:
    main_cfg, _ = load_config_from_py(args.config)
    run_dir = _resolve_run_dir(args.run_dir, main_cfg)
    dirs = AsyncSplitDirs.from_root(run_dir)
    dirs.ensure_exists()
    proc_log_dir = dirs.log_dir / 'procs'
    proc_log_dir.mkdir(parents=True, exist_ok=True)

    worker_env = _build_worker_env(
        num_actors=int(args.num_actors),
        include_evaluator=not bool(args.no_evaluator),
        worker_threads=args.worker_threads,
    )
    logging.info(
        f'[Launcher] Worker thread env: '
        f'OMP_NUM_THREADS={worker_env["OMP_NUM_THREADS"]}, '
        f'MKL_NUM_THREADS={worker_env["MKL_NUM_THREADS"]}, '
        f'OPENBLAS_NUM_THREADS={worker_env["OPENBLAS_NUM_THREADS"]}'
    )

    base_cmd = [sys.executable, '-m', 'lzero.entry.async_split']
    procs: List[Tuple[str, subprocess.Popen, object]] = []

    def _spawn(name: str, cmd: List[str]) -> subprocess.Popen:
        log_file = open(proc_log_dir / f'{name}.log', 'w', encoding='utf-8')
        logging.info(f'[Launcher] Starting {name}: {" ".join(cmd)}')
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=worker_env)
        procs.append((name, proc, log_file))
        return proc

    learner_cmd = base_cmd + [
        'learner',
        '--config', args.config,
        '--run-dir', str(dirs.root_dir),
        '--seed', str(args.seed),
        '--device', args.learner_device,
        '--publish-every', str(args.publish_every),
        '--save-ckpt-every', str(args.save_ckpt_every),
        '--max-replay-ratio', str(args.max_replay_ratio),
        '--max-train-iter', str(args.max_train_iter),
    ]
    if args.model_path:
        learner_cmd += ['--model-path', args.model_path]
    if args.min_transitions is not None:
        learner_cmd += ['--min-transitions', str(args.min_transitions)]
    if args.prefetch_workers and int(args.prefetch_workers) > 0:
        learner_cmd += ['--prefetch-workers', str(args.prefetch_workers)]
    if args.prefetch_queue_size is not None:
        learner_cmd += ['--prefetch-queue-size', str(args.prefetch_queue_size)]

    learner_proc = _spawn('learner', learner_cmd)

    for i in range(max(0, int(args.num_actors))):
        actor_cmd = base_cmd + [
            'actor',
            '--config', args.config,
            '--run-dir', str(dirs.root_dir),
            '--actor-id', f'actor_{i}',
            '--seed', str(args.seed + 100 + i),
            '--device', args.actor_device,
            '--wait-for-initial-weights',
        ]
        if args.actor_env_num is not None:
            actor_cmd += ['--env-num', str(args.actor_env_num)]
        if args.actor_n_episode is not None:
            actor_cmd += ['--n-episode', str(args.actor_n_episode)]
        if args.env_manager is not None:
            actor_cmd += ['--env-manager', args.env_manager]
        _spawn(f'actor_{i}', actor_cmd)

    if not args.no_evaluator:
        eval_cmd = base_cmd + [
            'evaluator',
            '--config', args.config,
            '--run-dir', str(dirs.root_dir),
            '--seed', str(args.seed + 1000),
            '--device', args.eval_device,
            '--min-iter-gap', str(args.eval_min_iter_gap),
        ]
        if args.eval_env_num is not None:
            eval_cmd += ['--env-num', str(args.eval_env_num)]
        if args.eval_n_episode is not None:
            eval_cmd += ['--n-episode', str(args.eval_n_episode)]
        if args.eval_bot is not None:
            eval_cmd += ['--bot-action-type', args.eval_bot]
        if args.env_manager is not None:
            eval_cmd += ['--env-manager', args.env_manager]
        _spawn('evaluator', eval_cmd)

    stop_requested = False

    def _handle_signal(*_a):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    exit_code = 0
    try:
        while not stop_requested:
            rc = learner_proc.poll()
            if rc is not None:
                exit_code = rc
                logging.info(f'[Launcher] Learner exited with code {rc}; shutting down workers.')
                break
            time.sleep(0.5)
    finally:
        for name, proc, _ in procs:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
        deadline = time.time() + 10.0
        for name, proc, log_file in procs:
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logging.warning(f'[Launcher] Killing unresponsive worker {name}')
                proc.kill()
            finally:
                log_file.close()

    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'launch':
        return _run_launch(args)

    main_cfg, create_cfg = load_config_from_py(args.config)
    run_dir = _resolve_run_dir(args.run_dir, main_cfg)

    if args.command == 'actor':
        actor = SelfPlayActor(
            input_cfg=(main_cfg, create_cfg),
            run_dir=run_dir,
            actor_id=args.actor_id,
            seed=args.seed,
            device=args.device,
            collector_env_num=args.env_num,
            n_episode=args.n_episode,
            num_simulations=args.num_simulations,
            env_manager_type=args.env_manager,
            weight_sync_interval=args.weight_sync_interval,
            wait_for_initial_weights=args.wait_for_initial_weights,
            poll_interval=args.poll_interval,
            enable_tb=args.enable_tb,
        )
        actor.run(max_batches=args.max_batches, max_episodes=args.max_episodes)
        return 0

    if args.command == 'learner':
        max_rr = args.max_replay_ratio if args.max_replay_ratio and args.max_replay_ratio > 0 else None
        learner = AsyncLearner(
            input_cfg=(main_cfg, create_cfg),
            run_dir=run_dir,
            seed=args.seed,
            device=args.device,
            model_path=args.model_path,
            publish_every_n_updates=args.publish_every,
            save_ckpt_every_n_updates=args.save_ckpt_every,
            max_replay_ratio=max_rr,
            min_transitions=args.min_transitions,
            poll_interval=args.poll_interval,
            archive_segments=args.archive_segments,
            use_bg_reader=not args.no_bg_reader,
            prefetch_workers=args.prefetch_workers,
            prefetch_queue_size=args.prefetch_queue_size,
        )
        learner.run(
            max_train_iter=args.max_train_iter,
            max_env_step=args.max_env_step,
            max_idle_seconds=args.max_idle_seconds,
        )
        return 0

    if args.command == 'evaluator':
        evaluator = AsyncEvaluator(
            input_cfg=(main_cfg, create_cfg),
            run_dir=run_dir,
            seed=args.seed,
            device=args.device,
            evaluator_env_num=args.env_num,
            n_evaluator_episode=args.n_episode,
            bot_action_type=args.bot_action_type,
            num_simulations=args.num_simulations,
            env_manager_type=args.env_manager,
            min_iter_gap=args.min_iter_gap,
            poll_interval=args.poll_interval,
        )
        evaluator.run(max_evals=args.max_evals, max_idle_seconds=args.max_idle_seconds)
        return 0

    parser.error(f'Unknown command: {args.command}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
