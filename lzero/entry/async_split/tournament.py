"""Standalone offline Elo tournament runner for saved LightZero checkpoints.

Supports rating checkpoints from one or multiple training runs (even across
different model configs sharing the same environment) against anchor bots
(``bot:random`` pinned to 0.0 Elo by default) using either:
- ``mode='ladder'`` (default): Round 1 connects chronological neighbors + anchor
  into a chain; subsequent Swiss-style rounds pair nearest-Elo unplayed pairs.
- ``mode='round_robin'``: plays every unordered pair once.

State is saved atomically to ``<output_dir>/tournament_state.json`` after every
match so interrupted tournaments can be resumed cleanly with ``--resume``.
"""

from __future__ import annotations

import copy
import glob
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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

from .elo_evaluator import _parse_anchor_bots
from .io_utils import (
    atomic_json_save,
    load_config_from_py,
    load_model_state_dict,
    parse_checkpoint_iter,
    read_json,
)


@dataclass
class CheckpointSpec:
    """Parsed specification for a checkpoint participant in a tournament."""

    player_id: str
    ckpt_path: Path
    config_path: Optional[Path] = None
    train_iter: Optional[int] = None


def _clean_ckpt_stem(path: Path) -> str:
    name = path.name
    for suffix in ('.pth.tar', '.pt', '.pth', '.ckpt'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def parse_checkpoint_spec(spec: str, default_prefix: Optional[str] = None) -> List[CheckpointSpec]:
    """Parse ``path``, ``label=path``, or ``label=path@config_path`` (with glob support).

    Safely handles file paths containing ``=`` or ``@`` (e.g. ``lr=1e-4/iter_100.pth.tar``)
    by checking whether the raw string already resolves to an existing file and requiring
    labels to contain no path separators.
    """
    raw = str(spec).strip()
    if not raw:
        return []

    label: Optional[str] = None
    config_path: Optional[Path] = None

    # Only interpret `label=...` if raw is not already an existing path and label has no path separators
    if '=' in raw and not Path(raw).is_file():
        left, right = raw.split('=', 1)
        left_s = left.strip()
        if left_s and '/' not in left_s and '\\' not in left_s:
            label = left_s
            raw = right.strip()

    # Only interpret `...@config_path` if raw is not already an existing path
    if '@' in raw and not Path(raw).is_file():
        ckpt_part, cfg_part = raw.rsplit('@', 1)
        ckpt_s = ckpt_part.strip()
        cfg_s = cfg_part.strip()
        if ckpt_s and cfg_s:
            raw = ckpt_s
            config_path = Path(cfg_s).resolve()

    # Expand globs when wildcards are present
    if any(ch in raw for ch in ('*', '?', '[')):
        matched = sorted(Path(p) for p in glob.glob(raw) if Path(p).is_file())
        if not matched:
            raise FileNotFoundError(f'No checkpoints matched pattern: {raw}')
        # Sort numerically if iteration checkpoints
        matched.sort(key=lambda p: (parse_checkpoint_iter(p) is None, parse_checkpoint_iter(p) or 0, p.name))
    else:
        p = Path(raw)
        if not p.is_file():
            raise FileNotFoundError(f'Checkpoint file not found: {p}')
        matched = [p]

    results: List[CheckpointSpec] = []
    for path in matched:
        resolved = path.resolve()
        it = parse_checkpoint_iter(resolved)
        if label is not None and len(matched) == 1:
            pid = label
        else:
            stem = f'iter_{it}' if it is not None else _clean_ckpt_stem(resolved)
            prefix = label if label is not None else default_prefix
            pid = f'{prefix}:{stem}' if prefix else stem
        results.append(
            CheckpointSpec(
                player_id=pid,
                ckpt_path=resolved,
                config_path=config_path,
                train_iter=it,
            )
        )
    return results


def discover_checkpoints_in_dir(
    ckpt_dir: os.PathLike | str,
    prefix: Optional[str] = None,
    every_n: int = 1,
    max_checkpoints: Optional[int] = None,
    config_path: Optional[Path] = None,
) -> List[CheckpointSpec]:
    """Discover ``iteration_<N>.pth.tar`` checkpoints in ``ckpt_dir`` in numeric order."""
    directory = Path(ckpt_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f'Checkpoint directory not found: {directory}')

    found: List[Tuple[int, Path]] = []
    for entry in directory.iterdir():
        if not entry.is_file() or entry.name.startswith('.'):
            continue
        it = parse_checkpoint_iter(entry)
        if it is not None:
            found.append((it, entry))
    found.sort(key=lambda pair: pair[0])

    if not found:
        return []

    step = max(1, int(every_n))
    if step > 1 and len(found) > 2:
        selected = found[::step]
        if selected[-1][0] != found[-1][0]:
            selected.append(found[-1])
        found = selected

    if max_checkpoints is not None and int(max_checkpoints) > 0 and len(found) > int(max_checkpoints):
        cap = int(max_checkpoints)
        if cap == 1:
            found = [found[-1]]
        else:
            # Evenly spaced indices always including first and last
            indices = sorted({round(i * (len(found) - 1) / (cap - 1)) for i in range(cap)})
            found = [found[idx] for idx in indices]

    specs: List[CheckpointSpec] = []
    for it, path in found:
        pid = f'{prefix}:iter_{it}' if prefix else f'iter_{it}'
        specs.append(
            CheckpointSpec(
                player_id=pid,
                ckpt_path=path.resolve(),
                config_path=config_path,
                train_iter=it,
            )
        )
    return specs


def _deduplicate_specs(specs: Sequence[CheckpointSpec]) -> List[CheckpointSpec]:
    used_ids: Set[str] = set()
    out: List[CheckpointSpec] = []
    for spec in specs:
        pid = spec.player_id
        if pid in used_ids:
            suffix = 2
            while f'{pid}_{suffix}' in used_ids:
                suffix += 1
            pid = f'{pid}_{suffix}'
        used_ids.add(pid)
        out.append(
            CheckpointSpec(
                player_id=pid,
                ckpt_path=spec.ckpt_path,
                config_path=spec.config_path,
                train_iter=spec.train_iter,
            )
        )
    return out


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _played_pair_counts(ladder: EloLadder) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for m in ladder.matches:
        if m.total_games <= 0 or m.player_a == m.player_b:
            continue
        key = _pair_key(m.player_a, m.player_b)
        counts[key] = counts.get(key, 0) + 1
    return counts


def plan_round_robin_pairs(
    player_ids: Sequence[str],
    played_counts: Dict[Tuple[str, str], int],
) -> List[Tuple[str, str]]:
    """Return all unordered pairs not yet played."""
    pairs: List[Tuple[str, str]] = []
    for i in range(len(player_ids)):
        for j in range(i + 1, len(player_ids)):
            a, b = player_ids[i], player_ids[j]
            if played_counts.get(_pair_key(a, b), 0) == 0:
                pairs.append((a, b))
    return pairs


def plan_ladder_initial_pairs(
    ordered_player_ids: Sequence[str],
    anchor_id: str,
    played_counts: Dict[Tuple[str, str], int],
) -> List[Tuple[str, str]]:
    """Build a connected initial chain pairing adjacent players (and anchor -> first checkpoint)."""
    ids = list(ordered_player_ids)
    if anchor_id in ids:
        ids.remove(anchor_id)
        ids.insert(0, anchor_id)

    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for i in range(len(ids) - 1):
        a, b = ids[i], ids[i + 1]
        key = _pair_key(a, b)
        if key not in seen and played_counts.get(key, 0) == 0:
            pairs.append((a, b))
            seen.add(key)
    return pairs


def plan_swiss_round_pairs(
    ladder: EloLadder,
    player_ids: Sequence[str],
    played_counts: Dict[Tuple[str, str], int],
) -> List[Tuple[str, str]]:
    """Pair players with closest current Elo who have not yet played each other."""
    sorted_ids = sorted(player_ids, key=lambda pid: (-ladder.get_elo(pid), pid))
    unpaired = list(sorted_ids)
    pairs: List[Tuple[str, str]] = []

    while len(unpaired) >= 2:
        p = unpaired.pop(0)
        p_elo = ladder.get_elo(p)
        # Pick the unplayed opponent with the closest current Elo
        best_idx: Optional[int] = None
        best_diff: Optional[float] = None
        for idx, q in enumerate(unpaired):
            if played_counts.get(_pair_key(p, q), 0) > 0:
                continue
            diff = abs(p_elo - ladder.get_elo(q))
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = idx
        if best_idx is not None:
            q = unpaired.pop(best_idx)
            pairs.append((p, q))
    return pairs


class OfflineTournament:
    """Runs a resumable Bradley-Terry Elo tournament over saved checkpoints + anchors."""

    def __init__(
        self,
        default_input_cfg: Tuple[dict, dict],
        checkpoints: Sequence[CheckpointSpec],
        output_dir: os.PathLike | str,
        anchor_bots: Optional[Sequence[str] | str] = ('random',),
        games_per_pair: int = 10,
        mode: str = 'ladder',
        num_rounds: int = 6,
        seed: int = 0,
        device: str = 'cpu',
        env_num: Optional[int] = None,
        num_simulations: Optional[int] = None,
        prior_draws: float = 0.1,
        resume: bool = False,
        write_tb: bool = False,
    ) -> None:
        if mode not in {'ladder', 'round_robin'}:
            raise ValueError(f"Tournament mode must be 'ladder' or 'round_robin', got {mode!r}")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / 'tournament_state.json'
        self.mode = str(mode)
        self.num_rounds = max(1, int(num_rounds))
        self.games_per_pair = max(2, int(games_per_pair))
        self.seed = int(seed)
        self.device_arg = str(device)
        self.env_num_override = int(env_num) if env_num is not None else None
        self.num_simulations = num_simulations
        self.prior_draws = float(prior_draws)
        self.resume = bool(resume)
        self.write_tb = bool(write_tb)
        self._closed = False
        self._chain_completed = False
        self._completed_swiss_rounds = 0
        self.runner: Optional[HeadToHeadRunner] = None
        self.tb_logger: Optional[SummaryWriter] = None

        try:
            self._default_cfg, self._default_create_cfg = self._prepare_cfg(
                default_input_cfg, seed=self.seed
            )
            set_pkg_seed(self._default_cfg.seed, use_cuda=self._default_cfg.policy.cuda)

            env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(self._default_cfg.env)
            base_env_cfg = copy.deepcopy(
                collector_env_cfg[0] if collector_env_cfg else evaluator_env_cfg[0]
            )
            base_env_cfg.battle_mode = 'self_play_mode'
            base_env_cfg.render_mode = None
            eff_env_num = int(
                self.env_num_override
                if self.env_num_override is not None
                else self._default_cfg.env.get('evaluator_env_num', 4)
            )
            self.runner = HeadToHeadRunner(env_fn=env_fn, env_cfg=base_env_cfg, env_num=eff_env_num)

            # Two reusable policy slots per config key (slot 0 = player A, slot 1 = player B)
            self._cfg_cache: Dict[str, Tuple[EasyDict, EasyDict]] = {
                '__default__': (self._default_cfg, self._default_create_cfg)
            }
            self._policy_slots: Dict[Tuple[str, int], Tuple[Any, EasyDict]] = {}
            self._bot_players: Dict[str, EnvBotMatchPlayer] = {}

            parsed_anchors = _parse_anchor_bots(anchor_bots)
            self.ladder = EloLadder(
                anchor_bots=parsed_anchors,
                anchor_elo=0.0,
                max_ladder_size=max(10, len(checkpoints) + len(parsed_anchors)),
                prior_draws=self.prior_draws,
            )

            if self.resume and self.state_path.is_file():
                self._restore_state(requested_anchors=parsed_anchors)

            reserved_bot_ids = {
                pid for pid, entry in self.ladder.players.items() if entry.kind == 'bot'
            }
            self._ordered_player_ids: List[str] = [f'bot:{b}' for b in self.ladder.anchor_bot_types]
            deduped = _deduplicate_specs(checkpoints)
            for spec in deduped:
                if spec.player_id in reserved_bot_ids:
                    raise ValueError(
                        f"Checkpoint label {spec.player_id!r} collides with reserved bot player ID. "
                        "Use a different label (e.g. 'ckpt_random=path/to/ckpt.pth.tar')."
                    )
                if spec.player_id not in self.ladder.players:
                    self.ladder.register_player(
                        PlayerEntry(
                            player_id=spec.player_id,
                            kind='checkpoint',
                            ckpt_path=str(spec.ckpt_path),
                            config_path=str(spec.config_path) if spec.config_path is not None else None,
                            train_iter=spec.train_iter,
                            is_anchor=False,
                            active=True,
                        )
                    )
                else:
                    # Refresh paths on resume in case files moved or specs were re-passed
                    entry = self.ladder.players[spec.player_id]
                    entry.ckpt_path = str(spec.ckpt_path)
                    if spec.config_path is not None:
                        entry.config_path = str(spec.config_path)
                if spec.player_id not in self._ordered_player_ids:
                    self._ordered_player_ids.append(spec.player_id)

            # Include any players loaded from resumed state
            for pid in self.ladder.players:
                if pid not in self._ordered_player_ids:
                    self._ordered_player_ids.append(pid)

            self.ladder.fit_ratings()
            if self.write_tb:
                self.tb_logger = SummaryWriter(str(self.output_dir / 'tb'))
        except Exception:
            self.close()
            raise

    def _prepare_cfg(self, input_cfg: Tuple[dict, dict], seed: int) -> Tuple[EasyDict, EasyDict]:
        cfg = EasyDict(copy.deepcopy(input_cfg[0]))
        create_cfg = EasyDict(copy.deepcopy(input_cfg[1]))
        if self.device_arg == 'auto':
            use_cuda = bool(cfg.policy.get('cuda', False) and torch.cuda.is_available())
        else:
            use_cuda = (self.device_arg == 'cuda' and torch.cuda.is_available())
        cfg.policy.cuda = use_cuda
        cfg.policy.device = 'cuda' if use_cuda else 'cpu'
        if self.env_num_override is not None:
            cfg.env.evaluator_env_num = self.env_num_override
            cfg.policy.evaluator_env_num = self.env_num_override
        if self.num_simulations is not None:
            cfg.policy.num_simulations = int(self.num_simulations)
        cfg.exp_name = os.path.relpath(str(self.output_dir), os.getcwd())
        compiled = compile_config(
            cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=False
        )
        compiled.exp_name = cfg.exp_name
        return compiled, create_cfg

    def _get_policy_slot(self, config_path: Optional[str], slot: int) -> Tuple[Any, EasyDict]:
        key = str(Path(config_path).resolve()) if config_path else '__default__'
        if key not in self._cfg_cache:
            raw_cfg = load_config_from_py(key)
            self._cfg_cache[key] = self._prepare_cfg(raw_cfg, seed=self.seed)
        if (key, slot) not in self._policy_slots:
            compiled_cfg, _ = self._cfg_cache[key]
            policy = create_policy(
                copy.deepcopy(compiled_cfg.policy), model=None, enable_field=['learn', 'eval']
            )
            self._policy_slots[(key, slot)] = (policy, compiled_cfg.policy)
        return self._policy_slots[(key, slot)]

    def _build_player(self, player_id: str, slot: int) -> MatchPlayer:
        entry = self.ladder.players[player_id]
        if entry.kind == 'bot':
            bot_type = entry.bot_type or 'random'
            if player_id not in self._bot_players:
                self._bot_players[player_id] = EnvBotMatchPlayer(
                    bot_type=bot_type, player_id=player_id
                )
            return self._bot_players[player_id]

        if not entry.ckpt_path or not Path(entry.ckpt_path).is_file():
            raise FileNotFoundError(f'Checkpoint missing for {player_id}: {entry.ckpt_path}')
        policy, policy_cfg = self._get_policy_slot(entry.config_path, slot=slot)
        state_dict, _ = load_model_state_dict(entry.ckpt_path, device=policy_cfg.device)
        policy._model.load_state_dict(state_dict)
        return PolicyMatchPlayer(
            player_id=player_id,
            policy=policy,
            policy_config=policy_cfg,
        )

    def _restore_state(self, requested_anchors: Sequence[str] = ()) -> None:
        state = read_json(self.state_path)
        if state is not None and isinstance(state.get('ladder'), dict):
            self.ladder = EloLadder.from_dict(state['ladder'])
            self.ladder.prior_draws = self.prior_draws
            for bot_type in requested_anchors:
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
            self._chain_completed = bool(state.get('chain_completed', False))
            self._completed_swiss_rounds = int(state.get('completed_swiss_rounds', 0))
            logging.info(
                f'[Tournament] Resumed state from {self.state_path} '
                f'({len(self.ladder.matches)} prior match(es), '
                f'chain_completed={self._chain_completed}, '
                f'completed_swiss_rounds={self._completed_swiss_rounds})'
            )

    def _save_state(self) -> None:
        payload = {
            'version': 1,
            'mode': self.mode,
            'games_per_pair': self.games_per_pair,
            'num_rounds': self.num_rounds,
            'chain_completed': bool(self._chain_completed),
            'completed_swiss_rounds': int(self._completed_swiss_rounds),
            'timestamp': time.time(),
            'ladder': self.ladder.to_dict(),
        }
        atomic_json_save(payload, self.state_path)

    def _run_pair(self, player_a_id: str, player_b_id: str, match_index: int) -> Dict[str, Any]:
        pa = self._build_player(player_a_id, slot=0)
        pb = self._build_player(player_b_id, slot=1)
        match_seed = (self.seed * 1_000_003 + match_index * 7_919) & 0x7FFFFFFF
        summary = self.runner.play_match(
            player_a=pa,
            player_b=pb,
            num_games=self.games_per_pair,
            seed=match_seed,
        )
        self.ladder.record_match(
            player_a=player_a_id,
            player_b=player_b_id,
            wins_a=summary.wins_a,
            draws=summary.draws,
            losses_a=summary.losses_a,
            metadata={
                'match_index': int(match_index),
                'p1_wins': summary.p1_wins,
                'p2_wins': summary.p2_wins,
                'avg_episode_length': summary.avg_episode_length,
            },
        )
        self.ladder.fit_ratings()
        self._save_state()
        logging.info(
            f'[Tournament] Match #{match_index}: {player_a_id} vs {player_b_id} -> '
            f'{summary.wins_a}W-{summary.draws}D-{summary.losses_a}L '
            f'({summary.win_rate_a:.1%}, {summary.elapsed_sec:.1f}s)'
        )
        return {
            'player_a': player_a_id,
            'player_b': player_b_id,
            'wins_a': summary.wins_a,
            'draws': summary.draws,
            'losses_a': summary.losses_a,
            'win_rate_a': summary.win_rate_a,
            'elapsed_sec': summary.elapsed_sec,
        }

    def run(self) -> Dict[str, Any]:
        """Execute the tournament, fit final Bradley-Terry ratings, and save state."""
        t0 = time.time()
        matches_played_now = 0
        match_index = len(self.ladder.matches)

        try:
            if len(self._ordered_player_ids) < 2:
                logging.warning('[Tournament] Fewer than 2 participants; nothing to play.')
            elif self.mode == 'round_robin':
                played = _played_pair_counts(self.ladder)
                pairs = plan_round_robin_pairs(self._ordered_player_ids, played)
                for a, b in pairs:
                    match_index += 1
                    self._run_pair(a, b, match_index=match_index)
                    matches_played_now += 1
            else:
                # Round 1: chronological connectivity chain (skipped if already completed on resume)
                if not self._chain_completed:
                    played = _played_pair_counts(self.ladder)
                    chain_pairs = plan_ladder_initial_pairs(
                        self._ordered_player_ids, self.ladder.anchor_id, played
                    )
                    for a, b in chain_pairs:
                        match_index += 1
                        self._run_pair(a, b, match_index=match_index)
                        matches_played_now += 1
                    self._chain_completed = True
                    self._save_state()

                # Rounds 2..num_rounds: adaptive Swiss nearest-Elo pairing (resumes only remaining rounds)
                target_swiss_rounds = max(0, self.num_rounds - 1)
                while self._completed_swiss_rounds < target_swiss_rounds:
                    played = _played_pair_counts(self.ladder)
                    swiss_pairs = plan_swiss_round_pairs(
                        self.ladder, self._ordered_player_ids, played
                    )
                    if not swiss_pairs:
                        self._completed_swiss_rounds = target_swiss_rounds
                        self._save_state()
                        break
                    for a, b in swiss_pairs:
                        match_index += 1
                        self._run_pair(a, b, match_index=match_index)
                        matches_played_now += 1
                    self._completed_swiss_rounds += 1
                    self._save_state()

            self.ladder.fit_ratings()
            self._save_state()

            if self.tb_logger is not None:
                ckpt_entries = [
                    p
                    for p in self.ladder.players.values()
                    if p.kind == 'checkpoint' and p.train_iter is not None
                ]
                ckpt_entries.sort(key=lambda p: (p.player_id.split(':', 1)[0] if ':' in p.player_id else '', int(p.train_iter)))
                for p in ckpt_entries:
                    est = self.ladder.estimates.get(p.player_id)
                    if est is None or not math.isfinite(est.elo):
                        continue
                    if ':' in p.player_id:
                        run_prefix = p.player_id.split(':', 1)[0]
                        rating_tag = f'tournament_iter/{run_prefix}/rating'
                        std_tag = f'tournament_iter/{run_prefix}/rating_std'
                    else:
                        rating_tag = 'tournament_iter/rating'
                        std_tag = 'tournament_iter/rating_std'
                    self.tb_logger.add_scalar(rating_tag, est.elo, int(p.train_iter))
                    if math.isfinite(est.elo_std):
                        self.tb_logger.add_scalar(std_tag, est.elo_std, int(p.train_iter))
                self.tb_logger.flush()

            table = self.ladder.format_leaderboard(active_only=False)
            logging.info(f'[Tournament] Final Elo Leaderboard:\n{table}')
            return {
                'matches_played': matches_played_now,
                'total_matches': len(self.ladder.matches),
                'elapsed_sec': time.time() - t0,
                'state_path': str(self.state_path),
                'leaderboard': [e.to_dict() for e in self.ladder.leaderboard(active_only=False)],
                'leaderboard_text': table,
            }
        finally:
            self.close()

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
