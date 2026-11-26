"""Live Elo checkpoint-ladder evaluator for decoupled async training.

Watches ``weights/latest_weights.pt`` (or evaluates an already-loaded model when
invoked by ``AsyncEvaluator`` in ``eval_mode='both'``), runs paired head-to-head
matches against selected rungs of a persistent checkpoint ladder + anchor bot
(``bot:random`` pinned to 0.0 Elo by default), refits joint Bradley-Terry MLE
ratings, promotes strong checkpoints into the ladder (pruning crowded interior
rungs beyond ``max_ladder_size`` while keeping full match history), logs Elo
metrics to TensorBoard, and atomically saves ``ckpt/elo_ckpt_best.pth.tar``.
"""

from __future__ import annotations

import copy
import math
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from ding.config import compile_config
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ditk import logging
from easydict import EasyDict
from tensorboardX import SummaryWriter

from lzero.worker.elo_rating import EloLadder, PlayerEntry
from lzero.worker.head_to_head import (
    EnvBotMatchPlayer,
    HeadToHeadRunner,
    MatchPlayer,
    PolicyMatchPlayer,
)

from .io_utils import (
    AsyncSplitDirs,
    WeightWatcher,
    atomic_json_save,
    atomic_torch_save,
    cpu_state_dict,
    load_model_state_dict,
    read_json,
)


def _ding_exp_name(root_dir: Path) -> str:
    return os.path.relpath(str(root_dir), os.getcwd())


def _parse_anchor_bots(anchor_bots: Optional[Sequence[str] | str]) -> List[str]:
    if anchor_bots is None:
        return ['random']
    if isinstance(anchor_bots, str):
        items = [s.strip() for s in anchor_bots.split(',') if s.strip()]
        return items if items else ['random']
    items = [str(s).strip() for s in anchor_bots if str(s).strip()]
    return items if items else ['random']


class AsyncEloEvaluator:
    """Standalone live Elo checkpoint-ladder evaluator polling ``latest_weights.pt``."""

    def __init__(
        self,
        input_cfg: Tuple[dict, dict],
        run_dir: os.PathLike | str,
        seed: int = 0,
        device: str = 'cpu',
        evaluator_env_num: Optional[int] = None,
        games_per_opponent: int = 10,
        opponents_per_eval: int = 4,
        max_ladder_size: int = 10,
        promotion_win_rate: float = 0.55,
        promotion_elo_margin: float = 50.0,
        periodic_promote_every: int = 0,
        anchor_bots: Optional[Sequence[str] | str] = ('random',),
        prior_draws: float = 0.1,
        num_simulations: Optional[int] = None,
        min_iter_gap: int = 0,
        poll_interval: float = 2.0,
        delete_pruned_rungs: bool = True,
        resume: bool = False,
    ) -> None:
        self.dirs = AsyncSplitDirs.from_root(run_dir)
        self.dirs.ensure_exists()
        self.dirs.elo_rungs_dir.mkdir(parents=True, exist_ok=True)

        self.seed = int(seed)
        self.games_per_opponent = max(2, int(games_per_opponent))
        self.min_iter_gap = max(0, int(min_iter_gap))
        self.poll_interval = float(poll_interval)
        self.delete_pruned_rungs = bool(delete_pruned_rungs)
        self.resume = bool(resume)
        self._stop_requested = False
        self._closed = False
        self._last_eval_iter: Optional[int] = None
        self._num_evals: int = 0
        self.runner: Optional[HeadToHeadRunner] = None
        self.tb_logger: Optional[SummaryWriter] = None

        try:
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
            if num_simulations is not None:
                cfg.policy.num_simulations = int(num_simulations)

            cfg.exp_name = _ding_exp_name(self.dirs.root_dir)
            self.cfg = compile_config(
                cfg, seed=self.seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=False
            )
            self.cfg.exp_name = _ding_exp_name(self.dirs.root_dir)
            self.create_cfg = create_cfg
            set_pkg_seed(self.cfg.seed, use_cuda=self.cfg.policy.cuda)

            env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(self.cfg.env)
            base_env_cfg = copy.deepcopy(
                collector_env_cfg[0] if collector_env_cfg else evaluator_env_cfg[0]
            )
            base_env_cfg.battle_mode = 'self_play_mode'
            base_env_cfg.render_mode = None
            env_num = int(
                evaluator_env_num
                if evaluator_env_num is not None
                else self.cfg.env.get('evaluator_env_num', 4)
            )
            self.runner = HeadToHeadRunner(env_fn=env_fn, env_cfg=base_env_cfg, env_num=env_num)

            # SampledEfficientZeroPolicy initializes value_inverse_scalar_transform_handle in _init_learn.
            self.candidate_policy = create_policy(
                copy.deepcopy(self.cfg.policy), model=None, enable_field=['learn', 'eval']
            )
            self.opponent_policy = create_policy(
                copy.deepcopy(self.cfg.policy), model=None, enable_field=['learn', 'eval']
            )
            self._bot_players: Dict[str, EnvBotMatchPlayer] = {}

            parsed_anchors = _parse_anchor_bots(anchor_bots)
            self.ladder = EloLadder(
                anchor_bots=parsed_anchors,
                anchor_elo=0.0,
                max_ladder_size=max_ladder_size,
                opponents_per_eval=opponents_per_eval,
                promotion_win_rate=promotion_win_rate,
                promotion_elo_margin=promotion_elo_margin,
                periodic_promote_every=periodic_promote_every,
                prior_draws=prior_draws,
            )

            self.tb_logger = SummaryWriter(str(self.dirs.log_dir / 'elo_evaluator'))
            self.weight_watcher = WeightWatcher(self.dirs.latest_weights_path)
            if self.resume:
                self._restore_state()
        except Exception:
            self.close()
            raise

    def _restore_state(self) -> None:
        try:
            state = read_json(self.dirs.elo_state_path)
        except Exception as exc:
            logging.warning(
                f'[AsyncEloEvaluator] Ignoring unreadable {self.dirs.elo_state_path.name}: {exc}'
            )
            return
        if state is None:
            return
        if state.get('last_eval_iter') is not None:
            self._last_eval_iter = int(state['last_eval_iter'])
        if state.get('num_evals') is not None:
            self._num_evals = int(state['num_evals'])
        if isinstance(state.get('ladder'), dict):
            prev_cfg = self.ladder
            self.ladder = EloLadder.from_dict(state['ladder'])
            # Apply current CLI/constructor ladder knobs while preserving history & ratings
            self.ladder.max_ladder_size = prev_cfg.max_ladder_size
            self.ladder.opponents_per_eval = prev_cfg.opponents_per_eval
            self.ladder.promotion_win_rate = prev_cfg.promotion_win_rate
            self.ladder.promotion_elo_margin = prev_cfg.promotion_elo_margin
            self.ladder.periodic_promote_every = prev_cfg.periodic_promote_every
            self.ladder.prior_draws = prev_cfg.prior_draws
            for bot_type in prev_cfg.anchor_bot_types:
                pid = f'bot:{bot_type}'
                if pid not in self.ladder.players:
                    self.ladder.register_player(
                        PlayerEntry(
                            player_id=pid,
                            kind='bot',
                            bot_type=bot_type,
                            is_anchor=(pid == self.ladder.anchor_id),
                            active=True,
                        )
                    )
                    if bot_type not in self.ladder.anchor_bot_types:
                        self.ladder.anchor_bot_types.append(bot_type)
            # Validate active checkpoint rung files still exist on disk
            for player in self.ladder.players.values():
                if player.kind == 'checkpoint' and player.active:
                    if not player.ckpt_path or not Path(player.ckpt_path).is_file():
                        logging.warning(
                            f'[AsyncEloEvaluator] Rung checkpoint missing for {player.player_id} '
                            f'({player.ckpt_path}); marking inactive.'
                        )
                        player.active = False
            self.ladder.fit_ratings()
        logging.info(
            f'[AsyncEloEvaluator] Resumed Elo ladder state: '
            f'last_eval_iter={self._last_eval_iter}, num_evals={self._num_evals}, '
            f'active_rungs={len(self.ladder.active_rungs())}, '
            f'best_elo={self.ladder.best_elo if math.isfinite(self.ladder.best_elo) else None}'
        )

    def _save_state(self) -> None:
        payload: Dict[str, Any] = {
            'version': 1,
            'last_eval_iter': self._last_eval_iter,
            'num_evals': int(self._num_evals),
            'timestamp': time.time(),
            'ladder': self.ladder.to_dict(),
        }
        try:
            atomic_json_save(payload, self.dirs.elo_state_path)
        except Exception as exc:
            logging.warning(
                f'[AsyncEloEvaluator] Failed to save {self.dirs.elo_state_path.name}: {exc}'
            )

    def _save_checkpoint(
        self,
        target_path: Path,
        train_iter: int,
        envstep: int,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        payload: Dict[str, Any] = {
            'model': cpu_state_dict(self.candidate_policy._model.state_dict()),
            'train_iter': int(train_iter),
            'envstep': int(envstep),
            'last_iter': int(train_iter),
        }
        if extra_meta:
            payload.update(extra_meta)
        return atomic_torch_save(payload, target_path)

    def _build_opponent_player(self, entry: PlayerEntry) -> MatchPlayer:
        if entry.kind == 'bot':
            bot_type = entry.bot_type or 'random'
            if entry.player_id not in self._bot_players:
                self._bot_players[entry.player_id] = EnvBotMatchPlayer(
                    bot_type=bot_type, player_id=entry.player_id
                )
            return self._bot_players[entry.player_id]

        if not entry.ckpt_path or not Path(entry.ckpt_path).is_file():
            raise FileNotFoundError(
                f'Rung checkpoint not found for {entry.player_id}: {entry.ckpt_path}'
            )
        state_dict, _ = load_model_state_dict(entry.ckpt_path, device=self.cfg.policy.device)
        self.opponent_policy._model.load_state_dict(state_dict)
        return PolicyMatchPlayer(
            player_id=entry.player_id,
            policy=self.opponent_policy,
            policy_config=self.cfg.policy,
        )

    def _unique_candidate_id(self, train_iter: int) -> str:
        base_id = f'iter_{int(train_iter)}'
        if base_id not in self.ladder.players:
            return base_id
        suffix = 2
        while f'{base_id}_r{suffix}' in self.ladder.players:
            suffix += 1
        return f'{base_id}_r{suffix}'

    def evaluate_loaded_candidate(
        self,
        train_iter: int,
        envstep: int,
        model_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate the candidate model against selected rungs of the Elo ladder.

        If ``model_state_dict`` is provided, it is loaded into ``self.candidate_policy._model``
        before evaluation; otherwise the currently loaded weights are evaluated.
        """
        t_start = time.time()
        if model_state_dict is not None:
            self.candidate_policy._model.load_state_dict(model_state_dict)

        candidate_id = self._unique_candidate_id(train_iter)
        ckpt_rungs_before = self.ladder.active_checkpoint_rungs()
        champion_before = ckpt_rungs_before[-1] if ckpt_rungs_before else None
        champion_id_before = champion_before.player_id if champion_before is not None else None

        opponents = self.ladder.select_opponents(candidate_id=candidate_id)
        cand_entry = PlayerEntry(
            player_id=candidate_id,
            kind='checkpoint',
            train_iter=int(train_iter),
            envstep=int(envstep),
            is_anchor=False,
            active=False,
        )
        self.ladder.register_player(cand_entry)

        cand_player = PolicyMatchPlayer(
            player_id=candidate_id,
            policy=self.candidate_policy,
            policy_config=self.cfg.policy,
        )

        match_results: List[Dict[str, Any]] = []
        scores_by_opp: Dict[str, float] = {}
        total_wins = 0
        total_draws = 0
        total_losses = 0
        total_p1_wins = 0
        total_steps = 0
        total_games = 0
        score_vs_champion: Optional[float] = None
        score_vs_anchor: Optional[float] = None

        for opp_idx, opp in enumerate(opponents):
            opp_player = self._build_opponent_player(opp)
            match_seed = (
                (
                    self.seed * 1_000_003
                    + int(train_iter) * 97
                    + self._num_evals * 10_007
                    + opp_idx * 1_009
                )
                & 0x7FFFFFFF
            )
            summary = self.runner.play_match(
                player_a=cand_player,
                player_b=opp_player,
                num_games=self.games_per_opponent,
                seed=match_seed,
            )
            self.ladder.record_match(
                player_a=candidate_id,
                player_b=opp.player_id,
                wins_a=summary.wins_a,
                draws=summary.draws,
                losses_a=summary.losses_a,
                metadata={
                    'train_iter': int(train_iter),
                    'envstep': int(envstep),
                    'p1_wins': summary.p1_wins,
                    'p2_wins': summary.p2_wins,
                    'avg_episode_length': summary.avg_episode_length,
                },
            )
            total_wins += summary.wins_a
            total_draws += summary.draws
            total_losses += summary.losses_a
            total_p1_wins += summary.p1_wins
            total_steps += summary.total_steps
            total_games += summary.total_games
            scores_by_opp[opp.player_id] = summary.win_rate_a

            if opp.player_id == champion_id_before:
                score_vs_champion = summary.win_rate_a
            if opp.player_id == self.ladder.anchor_id:
                score_vs_anchor = summary.win_rate_a

            opp_elo_before = self.ladder.get_elo(opp.player_id)
            match_results.append(
                {
                    'opponent_id': opp.player_id,
                    'opponent_elo_before': float(opp_elo_before) if math.isfinite(opp_elo_before) else None,
                    'wins': summary.wins_a,
                    'draws': summary.draws,
                    'losses': summary.losses_a,
                    'win_rate': summary.win_rate_a,
                    'avg_episode_length': summary.avg_episode_length,
                    'elapsed_sec': summary.elapsed_sec,
                }
            )

        # Refit global Bradley-Terry MLE across the full match graph
        # (also recalibrates ladder.best_elo to the updated scale for ladder.best_player_id).
        self.ladder.fit_ratings()
        cand_est = self.ladder.estimates[candidate_id]
        self.ladder.last_candidate_elo = cand_est.elo

        # Identify true post-refit champion among existing active rungs so score_vs_champion
        # and champ_elo always refer to the exact same rung.
        existing_active_rungs = [
            p for p in self.ladder.active_checkpoint_rungs() if p.player_id != candidate_id
        ]
        true_champ_entry = existing_active_rungs[-1] if existing_active_rungs else None
        true_champ_id = true_champ_entry.player_id if true_champ_entry is not None else None
        score_vs_true_champ = scores_by_opp.get(true_champ_id) if true_champ_id is not None else None
        if score_vs_true_champ is not None:
            score_vs_champion = score_vs_true_champ

        promoted, promo_reason = self.ladder.should_promote(
            candidate_id=candidate_id,
            score_vs_champion=score_vs_true_champ,
            champion_id=true_champ_id,
        )
        pruned_ids: List[str] = []
        if promoted:
            rung_path = self.dirs.elo_rungs_dir / f'{candidate_id}.pth.tar'
            self._save_checkpoint(
                rung_path,
                train_iter=train_iter,
                envstep=envstep,
                extra_meta={
                    'elo': float(cand_est.elo) if math.isfinite(cand_est.elo) else None,
                    'elo_std': float(cand_est.elo_std) if math.isfinite(cand_est.elo_std) else None,
                },
            )
            cand_entry.ckpt_path = str(rung_path)
            cand_entry.active = True
            self.ladder.evals_since_promotion = 0

            pruned = self.ladder.prune_if_needed(protected_ids={candidate_id})
            for victim in pruned:
                pruned_ids.append(victim.player_id)
                if self.delete_pruned_rungs and victim.ckpt_path:
                    try:
                        p = Path(victim.ckpt_path)
                        if p.is_file() and p.parent.resolve() == self.dirs.elo_rungs_dir.resolve():
                            p.unlink()
                    except OSError as exc:
                        logging.warning(
                            f'[AsyncEloEvaluator] Could not remove pruned rung {victim.ckpt_path}: {exc}'
                        )
        else:
            self.ladder.evals_since_promotion += 1

        # Resolve the true best checkpoint among (current best, top existing active rung, candidate)
        # so elo_ckpt_best.pth.tar is written at most once and state only updates after disk save.
        existing_active_after = [
            p for p in self.ladder.active_checkpoint_rungs() if p.player_id != candidate_id
        ]
        top_rung = existing_active_after[-1] if existing_active_after else None
        top_rung_est = self.ladder.estimates.get(top_rung.player_id) if top_rung is not None else None
        top_rung_elo = (
            float(top_rung_est.elo)
            if top_rung_est is not None and math.isfinite(top_rung_est.elo)
            else float('-inf')
        )
        current_best_elo = (
            float(self.ladder.best_elo)
            if self.ladder.best_player_id is not None and math.isfinite(self.ladder.best_elo)
            else float('-inf')
        )
        best_path = self.dirs.ckpt_dir / 'elo_ckpt_best.pth.tar'

        is_new_best = math.isfinite(cand_est.elo) and cand_est.elo > max(current_best_elo, top_rung_elo)
        if is_new_best:
            self._save_checkpoint(
                best_path,
                train_iter=train_iter,
                envstep=envstep,
                extra_meta={
                    'elo': float(cand_est.elo),
                    'elo_std': float(cand_est.elo_std) if math.isfinite(cand_est.elo_std) else None,
                },
            )
            self.ladder.best_elo = float(cand_est.elo)
            self.ladder.best_player_id = candidate_id
            logging.info(
                f'[AsyncEloEvaluator] Saved new best Elo checkpoint {best_path} '
                f'(elo={cand_est.elo:.1f} ± {cand_est.elo_std:.1f}, train_iter={train_iter})'
            )
        elif (
            top_rung is not None
            and top_rung_est is not None
            and top_rung_elo > current_best_elo
            and top_rung.ckpt_path
            and Path(top_rung.ckpt_path).is_file()
        ):
            try:
                rung_sd, rung_meta = load_model_state_dict(top_rung.ckpt_path, device='cpu')
                best_payload: Dict[str, Any] = {
                    'model': cpu_state_dict(rung_sd),
                    'train_iter': int(
                        top_rung.train_iter
                        if top_rung.train_iter is not None
                        else rung_meta.get('train_iter', 0)
                    ),
                    'envstep': int(
                        top_rung.envstep
                        if top_rung.envstep is not None
                        else rung_meta.get('envstep', 0)
                    ),
                    'last_iter': int(
                        top_rung.train_iter
                        if top_rung.train_iter is not None
                        else rung_meta.get('train_iter', 0)
                    ),
                    'elo': float(top_rung_est.elo),
                    'elo_std': (
                        float(top_rung_est.elo_std)
                        if math.isfinite(top_rung_est.elo_std)
                        else None
                    ),
                }
                atomic_torch_save(best_payload, best_path)
                self.ladder.best_elo = float(top_rung_est.elo)
                self.ladder.best_player_id = top_rung.player_id
                logging.info(
                    f'[AsyncEloEvaluator] Retroactively promoted active rung {top_rung.player_id} '
                    f'to {best_path} (elo={top_rung_est.elo:.1f})'
                )
            except Exception as exc:
                logging.warning(
                    f'[AsyncEloEvaluator] Failed to copy retroactively best rung '
                    f'{top_rung.player_id}: {exc}'
                )

        elapsed_total = time.time() - t_start
        overall_win_rate = (total_wins + 0.5 * total_draws) / total_games if total_games > 0 else 0.5
        avg_ep_len = float(total_steps) / total_games if total_games > 0 else 0.0
        p1_win_rate = (total_p1_wins + 0.5 * total_draws) / total_games if total_games > 0 else 0.5

        champion_after = self.ladder.get_champion()
        champion_elo_after = (
            self.ladder.get_elo(champion_after.player_id) if champion_after is not None else 0.0
        )

        # Log metrics to TensorBoard
        scalars: Dict[str, float] = {
            'rating': float(cand_est.elo),
            'rating_std': float(cand_est.elo_std) if math.isfinite(cand_est.elo_std) else 0.0,
            'best_rating': float(self.ladder.best_elo),
            'champion_rating': float(champion_elo_after),
            'win_rate_overall': float(overall_win_rate),
            'ladder_size': float(len(self.ladder.active_rungs())),
            'checkpoint_rungs': float(len(self.ladder.active_checkpoint_rungs())),
            'promoted': 1.0 if promoted else 0.0,
            'eval_time_sec': float(elapsed_total),
            'avg_episode_length': float(avg_ep_len),
            'p1_win_rate': float(p1_win_rate),
            'games_played': float(total_games),
        }
        if score_vs_champion is not None:
            scalars['win_rate_vs_champion'] = float(score_vs_champion)
        if score_vs_anchor is not None:
            scalars['win_rate_vs_anchor'] = float(score_vs_anchor)

        for tag, val in scalars.items():
            if not math.isfinite(val):
                continue
            self.tb_logger.add_scalar(f'elo_iter/{tag}', val, int(train_iter))
            self.tb_logger.add_scalar(f'elo_step/{tag}', val, int(envstep))
        self.tb_logger.flush()

        eval_record: Dict[str, Any] = {
            'candidate_id': candidate_id,
            'train_iter': int(train_iter),
            'envstep': int(envstep),
            'elo': float(cand_est.elo) if math.isfinite(cand_est.elo) else None,
            'elo_std': float(cand_est.elo_std) if math.isfinite(cand_est.elo_std) else None,
            'best_elo': float(self.ladder.best_elo) if math.isfinite(self.ladder.best_elo) else None,
            'promoted': bool(promoted),
            'promotion_reason': promo_reason,
            'pruned': pruned_ids,
            'is_new_best': bool(is_new_best),
            'win_rate_overall': float(overall_win_rate),
            'win_rate_vs_champion': score_vs_champion,
            'win_rate_vs_anchor': score_vs_anchor,
            'games_played': int(total_games),
            'elapsed_sec': float(elapsed_total),
            'matches': match_results,
        }
        self.ladder.eval_history.append(eval_record)

        self._last_eval_iter = int(train_iter)
        self._num_evals += 1
        self._save_state()

        logging.info(
            f'[AsyncEloEvaluator] {candidate_id} (envstep={envstep}): '
            f'Elo={cand_est.elo:.1f} ± {cand_est.elo_std:.1f} '
            f'(best={self.ladder.best_elo:.1f}), overall={overall_win_rate:.1%} '
            f'({total_wins}W-{total_draws}D-{total_losses}L), '
            f'promoted={promoted} ({promo_reason}), '
            f'active_rungs={len(self.ladder.active_rungs())}, '
            f'time={elapsed_total:.1f}s\n'
            f'{self.ladder.format_leaderboard(active_only=True)}'
        )
        return eval_record

    def eval_once_if_updated(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """Poll ``latest_weights.pt`` and run Elo ladder evaluation if updated."""
        meta = self.weight_watcher.poll_and_load(
            self.candidate_policy._model, device=self.cfg.policy.device, force=force
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

        return self.evaluate_loaded_candidate(train_iter=train_iter, envstep=envstep)

    def request_stop(self, *_args: Any) -> None:
        self._stop_requested = True

    def run(
        self,
        max_evals: Optional[int] = None,
        max_idle_seconds: Optional[float] = None,
        install_signal_handlers: bool = True,
    ) -> Dict[str, Any]:
        """Poll for updated weights and run Elo evaluations until stopped."""
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
                    if max_evals is not None and self._num_evals >= max_evals:
                        break
                else:
                    if max_idle_seconds is not None and (time.time() - last_eval_time) >= max_idle_seconds:
                        logging.info('[AsyncEloEvaluator] Max idle time reached; exiting.')
                        break
                    time.sleep(self.poll_interval)
        finally:
            self.close()

        return {
            'num_evals': self._num_evals,
            'last_eval_iter': self._last_eval_iter,
            'best_elo': self.ladder.best_elo if math.isfinite(self.ladder.best_elo) else None,
            'last_result': last_result,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.runner is not None:
            try:
                self.runner.close()
            except Exception:
                pass
        if self.tb_logger is not None:
            try:
                self.tb_logger.flush()
                self.tb_logger.close()
            except Exception:
                pass
