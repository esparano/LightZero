"""Bradley-Terry MLE Elo rating solver and strength-gated checkpoint ladder.

This module is pure Python + NumPy (no PyTorch or environment dependencies) so it
can be used by both live async evaluation workers and standalone offline
tournaments.

Mathematical model
------------------
Each player ``i`` has an Elo rating ``R_i`` (with ``gamma_i = (ln(10) / 400) * R_i``).
The Bradley-Terry expected score of player ``i`` against player ``j`` is:

    P(i beats j) = sigmoid(gamma_i - gamma_j) = 1 / (1 + 10 ** ((R_j - R_i) / 400))

Given accumulated match counts ``(wins, draws, losses)`` across pairs of players,
with score ``s_ij = wins_ij + 0.5 * draws_ij`` over ``n_ij = wins_ij + draws_ij + losses_ij``
games, we minimize the regularized negative log-likelihood:

    L(gamma) = -sum_{i < j, n_ij > 0} [
        (s_ij + eps / 2) * log(sigmoid(gamma_i - gamma_j))
        + (s_ji + eps / 2) * log(sigmoid(gamma_j - gamma_i))
    ] + 0.5 * l2_reg * sum_{k != anchor} (gamma_k - gamma_anchor) ** 2

subject to ``R_anchor = anchor_elo`` (default ``0.0`` for ``bot:random``).

A small per-edge virtual draw prior ``eps = prior_draws`` (default ``0.1``, i.e.
0.05 virtual wins each side on played pairs) keeps 100%/0% sweeps finite without
shrinking high-Elo players toward the anchor, while ``l2_reg`` (default ``1e-4``)
ensures a strictly positive-definite Hessian even if a player has no games yet.
Asymptotic standard errors are derived directly from the inverse Hessian
(Laplace approximation) at the optimum.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

ELO_SCALE = 400.0 / math.log(10.0)
INV_ELO_SCALE = math.log(10.0) / 400.0


def expected_score(elo_a: float, elo_b: float) -> float:
    """Return Bradley-Terry / Elo expected score of ``a`` against ``b``."""
    diff = (float(elo_a) - float(elo_b)) * INV_ELO_SCALE
    if diff >= 0.0:
        z = math.exp(-diff)
        return 1.0 / (1.0 + z)
    z = math.exp(diff)
    return z / (1.0 + z)


@dataclass
class MatchRecord:
    """Aggregated result of a head-to-head match between two players."""

    player_a: str
    player_b: str
    wins_a: int
    draws: int
    losses_a: int
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_games(self) -> int:
        return int(self.wins_a) + int(self.draws) + int(self.losses_a)

    @property
    def score_a(self) -> float:
        return float(self.wins_a) + 0.5 * float(self.draws)

    @property
    def score_b(self) -> float:
        return float(self.losses_a) + 0.5 * float(self.draws)

    @property
    def win_rate_a(self) -> float:
        n = self.total_games
        return self.score_a / n if n > 0 else 0.5

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'player_a': self.player_a,
            'player_b': self.player_b,
            'wins_a': int(self.wins_a),
            'draws': int(self.draws),
            'losses_a': int(self.losses_a),
            'timestamp': float(self.timestamp),
        }
        if self.metadata:
            out['metadata'] = dict(self.metadata)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MatchRecord':
        return cls(
            player_a=str(data['player_a']),
            player_b=str(data['player_b']),
            wins_a=int(data['wins_a']),
            draws=int(data['draws']),
            losses_a=int(data['losses_a']),
            timestamp=float(data.get('timestamp', 0.0)),
            metadata=dict(data.get('metadata', {})),
        )


@dataclass
class PlayerEntry:
    """Metadata for a player registered in the Elo ladder or tournament."""

    player_id: str
    kind: str  # 'bot' or 'checkpoint'
    ckpt_path: Optional[str] = None
    config_path: Optional[str] = None
    bot_type: Optional[str] = None
    train_iter: Optional[int] = None
    envstep: Optional[int] = None
    is_anchor: bool = False
    active: bool = True
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'player_id': self.player_id,
            'kind': self.kind,
            'ckpt_path': self.ckpt_path,
            'config_path': self.config_path,
            'bot_type': self.bot_type,
            'train_iter': self.train_iter,
            'envstep': self.envstep,
            'is_anchor': bool(self.is_anchor),
            'active': bool(self.active),
            'created_at': float(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlayerEntry':
        return cls(
            player_id=str(data['player_id']),
            kind=str(data['kind']),
            ckpt_path=data.get('ckpt_path'),
            config_path=data.get('config_path'),
            bot_type=data.get('bot_type'),
            train_iter=int(data['train_iter']) if data.get('train_iter') is not None else None,
            envstep=int(data['envstep']) if data.get('envstep') is not None else None,
            is_anchor=bool(data.get('is_anchor', False)),
            active=bool(data.get('active', True)),
            created_at=float(data.get('created_at', 0.0)),
        )


@dataclass
class EloEstimate:
    """Fitted Bradley-Terry rating and summary statistics for one player."""

    player_id: str
    elo: float
    elo_std: float
    games: int
    wins: int
    draws: int
    losses: int

    @property
    def score_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'player_id': self.player_id,
            'elo': float(self.elo) if math.isfinite(self.elo) else None,
            'elo_std': float(self.elo_std) if math.isfinite(self.elo_std) else None,
            'games': int(self.games),
            'wins': int(self.wins),
            'draws': int(self.draws),
            'losses': int(self.losses),
        }


def fit_bradley_terry_elo(
    player_ids: Sequence[str],
    matches: Sequence[MatchRecord],
    anchor_id: str,
    anchor_elo: float = 0.0,
    prior_draws: float = 0.1,
    l2_reg: float = 1e-4,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> Dict[str, EloEstimate]:
    """Fit joint Bradley-Terry MLE Elo ratings and Laplace standard errors.

    Arguments:
        player_ids: Ordered collection of player identifiers to rate.
        matches: Match records accumulated across the pool.
        anchor_id: Player ID whose Elo is pinned to ``anchor_elo``.
        anchor_elo: Fixed reference Elo rating for ``anchor_id`` (default 0.0).
        prior_draws: Total virtual draws added per *played* pair ``(i, j)``
            (split equally ``prior_draws / 2`` to each side). Keeps 100%/0%
            sweeps finite without pulling high-Elo players toward the anchor.
        l2_reg: Small quadratic regularization in log-strength space to ensure
            strict convexity even for players with 0 games or disconnected pairs.
        max_iter: Maximum damped Newton-Raphson iterations.
        tol: Convergence tolerance on the L-infinity norm of the gradient.
    """
    unique_ids: List[str] = []
    seen: Set[str] = set()
    for pid in player_ids:
        if pid not in seen:
            unique_ids.append(pid)
            seen.add(pid)
    for m in matches:
        for pid in (m.player_a, m.player_b):
            if pid not in seen:
                unique_ids.append(pid)
                seen.add(pid)
    if anchor_id not in seen:
        unique_ids.insert(0, anchor_id)
        seen.add(anchor_id)

    n_players = len(unique_ids)
    idx_of = {pid: idx for idx, pid in enumerate(unique_ids)}
    anchor_idx = idx_of[anchor_id]

    wins = np.zeros(n_players, dtype=np.int64)
    draws = np.zeros(n_players, dtype=np.int64)
    losses = np.zeros(n_players, dtype=np.int64)

    # Aggregate undirected pair counts: pair_scores[(i, j)] = [s_ij, s_ji, n_ij] for i < j
    pair_totals: Dict[Tuple[int, int], List[float]] = {}
    for m in matches:
        if m.total_games <= 0 or m.player_a == m.player_b:
            continue
        ia = idx_of[m.player_a]
        ib = idx_of[m.player_b]
        wins[ia] += int(m.wins_a)
        draws[ia] += int(m.draws)
        losses[ia] += int(m.losses_a)
        wins[ib] += int(m.losses_a)
        draws[ib] += int(m.draws)
        losses[ib] += int(m.wins_a)

        if ia < ib:
            key = (ia, ib)
            sa, sb = m.score_a, m.score_b
        else:
            key = (ib, ia)
            sa, sb = m.score_b, m.score_a

        entry = pair_totals.get(key)
        if entry is None:
            pair_totals[key] = [sa, sb, float(m.total_games)]
        else:
            entry[0] += sa
            entry[1] += sb
            entry[2] += float(m.total_games)

    if n_players == 1:
        pid = unique_ids[0]
        return {
            pid: EloEstimate(
                player_id=pid,
                elo=float(anchor_elo),
                elo_std=0.0,
                games=int(wins[0] + draws[0] + losses[0]),
                wins=int(wins[0]),
                draws=int(draws[0]),
                losses=int(losses[0]),
            )
        }

    # Pre-pack edges for vectorized evaluation
    edges_i = np.array([k[0] for k in pair_totals], dtype=np.int64)
    edges_j = np.array([k[1] for k in pair_totals], dtype=np.int64)
    half_prior = 0.5 * max(0.0, float(prior_draws))
    eff_s_ij = np.array([v[0] + half_prior for v in pair_totals.values()], dtype=np.float64)
    eff_s_ji = np.array([v[1] + half_prior for v in pair_totals.values()], dtype=np.float64)
    eff_n_ij = eff_s_ij + eff_s_ji

    # BFS from anchor to determine which players are identifiable relative to anchor when l2_reg == 0
    adj: List[List[int]] = [[] for _ in range(n_players)]
    for ia, ib in pair_totals:
        adj[ia].append(ib)
        adj[ib].append(ia)
    anchor_connected: Set[int] = {anchor_idx}
    queue = [anchor_idx]
    while queue:
        cur = queue.pop()
        for nxt in adj[cur]:
            if nxt not in anchor_connected:
                anchor_connected.add(nxt)
                queue.append(nxt)

    reg = max(0.0, float(l2_reg))
    if reg == 0.0:
        free_indices = np.array(
            [i for i in range(n_players) if i != anchor_idx and i in anchor_connected],
            dtype=np.int64,
        )
    else:
        free_indices = np.array([i for i in range(n_players) if i != anchor_idx], dtype=np.int64)

    gamma = np.zeros(n_players, dtype=np.float64)

    def _objective(g: np.ndarray) -> float:
        val = 0.5 * reg * float(np.dot(g[free_indices], g[free_indices]))
        if edges_i.size > 0:
            diff = g[edges_i] - g[edges_j]
            # -log(sigmoid(x)) = logaddexp(0, -x), -log(sigmoid(-x)) = logaddexp(0, x)
            val += float(
                np.sum(eff_s_ij * np.logaddexp(0.0, -diff) + eff_s_ji * np.logaddexp(0.0, diff))
            )
        return val

    def _grad_and_hess(g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grad = np.zeros(n_players, dtype=np.float64)
        hess = np.zeros((n_players, n_players), dtype=np.float64)
        grad[free_indices] += reg * g[free_indices]
        hess[free_indices, free_indices] += reg

        if edges_i.size > 0:
            diff = g[edges_i] - g[edges_j]
            # Numerically stable sigmoid
            pos = diff >= 0.0
            p_ij = np.empty_like(diff)
            p_ij[pos] = 1.0 / (1.0 + np.exp(-diff[pos]))
            exp_neg = np.exp(diff[~pos])
            p_ij[~pos] = exp_neg / (1.0 + exp_neg)

            residual = eff_n_ij * p_ij - eff_s_ij
            weight = eff_n_ij * p_ij * (1.0 - p_ij)

            np.add.at(grad, edges_i, residual)
            np.add.at(grad, edges_j, -residual)
            np.add.at(hess, (edges_i, edges_i), weight)
            np.add.at(hess, (edges_j, edges_j), weight)
            np.add.at(hess, (edges_i, edges_j), -weight)
            np.add.at(hess, (edges_j, edges_i), -weight)

        return grad[free_indices], hess[np.ix_(free_indices, free_indices)]

    stds = np.full(n_players, float('inf'), dtype=np.float64)
    stds[anchor_idx] = 0.0

    if free_indices.size > 0:
        cur_obj = _objective(gamma)
        for _ in range(max_iter):
            grad_free, hess_free = _grad_and_hess(gamma)
            if float(np.max(np.abs(grad_free))) < tol:
                break

            try:
                step = np.linalg.solve(hess_free, -grad_free)
            except np.linalg.LinAlgError:
                damped = hess_free + 1e-8 * np.eye(len(free_indices), dtype=np.float64)
                step = np.linalg.solve(damped, -grad_free)

            slope = float(np.dot(grad_free, step))
            if not np.all(np.isfinite(step)) or not math.isfinite(slope) or slope >= 0.0:
                step = -grad_free
                slope = float(np.dot(grad_free, step))

            alpha = 1.0
            accepted = False
            for _ls in range(30):
                cand = gamma.copy()
                cand[free_indices] += alpha * step
                cand_obj = _objective(cand)
                if math.isfinite(cand_obj) and cand_obj <= cur_obj + 1e-4 * alpha * slope:
                    gamma = cand
                    cur_obj = cand_obj
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break

        # Final Hessian at optimum for Laplace standard errors
        _, hess_free = _grad_and_hess(gamma)
        try:
            cov_free = np.linalg.inv(hess_free)
            diag_var = np.maximum(0.0, np.diag(cov_free))
            stds[free_indices] = ELO_SCALE * np.sqrt(diag_var)
        except np.linalg.LinAlgError:
            # Partial inversion: keep valid finite stds for non-degenerate players
            # while leaving degenerate/sweeping players at +inf.
            nz_mask = np.diag(hess_free) > 1e-12
            if np.any(nz_mask):
                sub_hess = hess_free[np.ix_(nz_mask, nz_mask)]
                try:
                    sub_cov = np.linalg.inv(sub_hess)
                    sub_var = np.maximum(0.0, np.diag(sub_cov))
                    stds[free_indices[nz_mask]] = ELO_SCALE * np.sqrt(sub_var)
                except np.linalg.LinAlgError:
                    pass

    elos = float(anchor_elo) + ELO_SCALE * gamma

    estimates: Dict[str, EloEstimate] = {}
    for idx, pid in enumerate(unique_ids):
        g_count = int(wins[idx] + draws[idx] + losses[idx])
        estimates[pid] = EloEstimate(
            player_id=pid,
            elo=float(elos[idx]),
            elo_std=float(stds[idx]),
            games=g_count,
            wins=int(wins[idx]),
            draws=int(draws[idx]),
            losses=int(losses[idx]),
        )
    return estimates


class EloLadder:
    """Persistent checkpoint ladder with Bradley-Terry ratings, opponent
    selection, hybrid strength-gated promotion, and crowded-middle pruning.
    """

    def __init__(
        self,
        anchor_bots: Sequence[str] = ('random',),
        anchor_elo: float = 0.0,
        max_ladder_size: int = 10,
        opponents_per_eval: int = 4,
        promotion_win_rate: float = 0.55,
        promotion_elo_margin: float = 50.0,
        periodic_promote_every: int = 0,
        prior_draws: float = 0.1,
        l2_reg: float = 1e-4,
    ) -> None:
        bots = [str(b).strip() for b in anchor_bots if str(b).strip()]
        if not bots:
            bots = ['random']
        self.anchor_bot_types: List[str] = bots
        self.anchor_id: str = f'bot:{bots[0]}'
        self.anchor_elo: float = float(anchor_elo)
        self.max_ladder_size: int = max(2, int(max_ladder_size))
        self.opponents_per_eval: int = max(1, int(opponents_per_eval))
        self.promotion_win_rate: float = float(promotion_win_rate)
        self.promotion_elo_margin: float = float(promotion_elo_margin)
        self.periodic_promote_every: int = max(0, int(periodic_promote_every))
        self.prior_draws: float = float(prior_draws)
        self.l2_reg: float = float(l2_reg)

        self.players: Dict[str, PlayerEntry] = {}
        for idx, bot_type in enumerate(self.anchor_bot_types):
            pid = f'bot:{bot_type}'
            self.players[pid] = PlayerEntry(
                player_id=pid,
                kind='bot',
                bot_type=bot_type,
                is_anchor=(idx == 0),
                active=True,
            )

        self.matches: List[MatchRecord] = []
        self.estimates: Dict[str, EloEstimate] = {}
        self.last_candidate_elo: Optional[float] = None
        self.best_elo: float = float('-inf')
        self.best_player_id: Optional[str] = None
        self.evals_since_promotion: int = 0
        self.eval_history: List[Dict[str, Any]] = []
        self.fit_ratings()

    def register_player(self, entry: PlayerEntry) -> PlayerEntry:
        self.players[entry.player_id] = entry
        if entry.player_id not in self.estimates:
            self.estimates[entry.player_id] = EloEstimate(
                player_id=entry.player_id,
                elo=self.anchor_elo,
                elo_std=0.0 if entry.player_id == self.anchor_id else float('inf'),
                games=0,
                wins=0,
                draws=0,
                losses=0,
            )
        return entry

    def active_rungs(self) -> List[PlayerEntry]:
        """Return all active ladder players (anchors + active checkpoint rungs) sorted by Elo."""
        rungs = [p for p in self.players.values() if p.active]
        rungs.sort(key=lambda p: self.get_elo(p.player_id))
        return rungs

    def active_checkpoint_rungs(self) -> List[PlayerEntry]:
        """Return active checkpoint rungs (excluding bots) sorted by Elo."""
        rungs = [p for p in self.players.values() if p.active and p.kind == 'checkpoint']
        rungs.sort(key=lambda p: self.get_elo(p.player_id))
        return rungs

    def get_elo(self, player_id: str) -> float:
        est = self.estimates.get(player_id)
        return float(est.elo) if est is not None else self.anchor_elo

    def get_champion(self) -> Optional[PlayerEntry]:
        """Return the highest-Elo active checkpoint rung, or highest active rung if none."""
        ckpt_rungs = self.active_checkpoint_rungs()
        if ckpt_rungs:
            return ckpt_rungs[-1]
        rungs = self.active_rungs()
        return rungs[-1] if rungs else None

    def record_match(
        self,
        player_a: str,
        player_b: str,
        wins_a: int,
        draws: int,
        losses_a: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MatchRecord:
        rec = MatchRecord(
            player_a=player_a,
            player_b=player_b,
            wins_a=int(wins_a),
            draws=int(draws),
            losses_a=int(losses_a),
            metadata=dict(metadata) if metadata else {},
        )
        self.matches.append(rec)
        return rec

    def fit_ratings(self) -> Dict[str, EloEstimate]:
        self.estimates = fit_bradley_terry_elo(
            player_ids=list(self.players.keys()),
            matches=self.matches,
            anchor_id=self.anchor_id,
            anchor_elo=self.anchor_elo,
            prior_draws=self.prior_draws,
            l2_reg=self.l2_reg,
        )
        # Keep best_elo calibrated to the latest joint Bradley-Terry scale so an early
        # 10-0 sweep cannot leave a stale inflated best_elo after more matches arrive.
        if self.best_player_id is not None and self.best_player_id in self.estimates:
            self.best_elo = float(self.estimates[self.best_player_id].elo)
        return self.estimates

    def select_opponents(
        self,
        candidate_id: Optional[str] = None,
        max_opponents: Optional[int] = None,
    ) -> List[PlayerEntry]:
        """Choose informative ladder opponents for evaluating a new candidate.

        Strategy:
        - If ``len(active_rungs) <= K``, play all of them.
        - Otherwise prioritize:
          1. Current checkpoint champion (``get_champion()``) — tests for new highs / promotion
             even if a bot currently outranks all checkpoints.
          2. Rungs closest in Elo to ``last_candidate_elo`` (or champion if first eval) —
             maximizes Bradley-Terry Fisher information and handles regressions accurately.
          3. One calibration rung (~250 Elo below the reference, or the anchor bot if
             reference Elo is still within 600 Elo of anchor) — maintains strong graph
             connectivity across the ladder.
        """
        k = self.opponents_per_eval if max_opponents is None else max(1, int(max_opponents))
        pool = [p for p in self.active_rungs() if p.player_id != candidate_id]
        if len(pool) <= k:
            return pool

        chosen: List[PlayerEntry] = []
        chosen_ids: Set[str] = set()

        def _add(entry: PlayerEntry) -> None:
            if entry.player_id not in chosen_ids and len(chosen) < k:
                chosen.append(entry)
                chosen_ids.add(entry.player_id)

        # 1. Always include current checkpoint champion (or highest active rung if none)
        champ = self.get_champion()
        if champ is not None and champ.player_id != candidate_id and any(p.player_id == champ.player_id for p in pool):
            champion = champ
        else:
            champion = pool[-1]
        _add(champion)

        ref_elo = (
            float(self.last_candidate_elo)
            if self.last_candidate_elo is not None
            else self.get_elo(champion.player_id)
        )

        # 2. Reserve 1 slot (if k >= 3) for a lower/calibration rung
        reserve_calibration = k >= 3
        neighbor_limit = (k - 1) if reserve_calibration else k

        remaining = [p for p in pool if p.player_id not in chosen_ids]
        remaining.sort(key=lambda p: (abs(self.get_elo(p.player_id) - ref_elo), -self.get_elo(p.player_id)))
        for p in remaining:
            if len(chosen) >= neighbor_limit:
                break
            _add(p)

        # 3. Calibration / anchor rung
        if len(chosen) < k:
            rest = [p for p in pool if p.player_id not in chosen_ids]
            if rest:
                anchor_entry = next((p for p in rest if p.player_id == self.anchor_id), None)
                if anchor_entry is not None and abs(ref_elo - self.anchor_elo) <= 600.0:
                    _add(anchor_entry)
                else:
                    target_cal_elo = ref_elo - 250.0
                    rest.sort(key=lambda p: abs(self.get_elo(p.player_id) - target_cal_elo))
                    _add(rest[0])

        # Fill any leftover slots by proximity to ref_elo
        for p in remaining:
            if len(chosen) >= k:
                break
            _add(p)

        chosen.sort(key=lambda p: self.get_elo(p.player_id))
        return chosen

    def should_promote(
        self,
        candidate_id: str,
        score_vs_champion: Optional[float],
        champion_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Decide whether ``candidate_id`` should be promoted to the active ladder.

        Uses the **post-refit** Bradley-Terry Elo of the existing champion so
        retroactive rating revisions (e.g., after an early 10-0 sweep gets
        calibrated by later games) never block stronger checkpoints from promoting.

        Hybrid strength-gated rule:
        1. Always promote the very first network checkpoint so the ladder has a
           neural-network seed rung.
        2. Promote if candidate's fitted Elo exceeds ``champion_elo + promotion_elo_margin``
           (and did not lose head-to-head to the champion if played).
        3. Promote if candidate scored ``>= promotion_win_rate`` head-to-head against
           the current champion and has strictly higher fitted Elo than the champion.
        4. Optional periodic fallback if ``periodic_promote_every > 0`` and
           ``evals_since_promotion + 1 >= periodic_promote_every``.
        """
        existing_ckpts = [
            p for p in self.active_checkpoint_rungs() if p.player_id != candidate_id
        ]
        if not existing_ckpts:
            return True, 'first_checkpoint_seed'

        cand_elo = self.get_elo(candidate_id)
        # Always compare against the true post-refit highest active checkpoint Elo
        # (even if a different rung retroactively overtook champion_id after fit_ratings).
        champ_elo = max(self.get_elo(p.player_id) for p in existing_ckpts)

        if (
            math.isfinite(cand_elo)
            and cand_elo >= champ_elo + self.promotion_elo_margin
            and (score_vs_champion is None or score_vs_champion >= 0.5)
        ):
            return True, f'elo_margin (+{cand_elo - champ_elo:.1f} >= +{self.promotion_elo_margin:.1f})'

        if (
            math.isfinite(cand_elo)
            and score_vs_champion is not None
            and score_vs_champion >= self.promotion_win_rate
            and cand_elo > champ_elo
        ):
            return True, f'beat_champion ({score_vs_champion:.1%} >= {self.promotion_win_rate:.1%})'

        if (
            self.periodic_promote_every > 0
            and (self.evals_since_promotion + 1) >= self.periodic_promote_every
        ):
            return True, f'periodic_snapshot (every {self.periodic_promote_every} evals)'

        return False, 'below_promotion_threshold'

    def prune_if_needed(self, protected_ids: Optional[Set[str]] = None) -> List[PlayerEntry]:
        """Prune crowded interior checkpoint rungs if active checkpoint count
        exceeds ``max_ladder_size``.

        Always preserves:
        - All bot players (anchors).
        - The weakest active checkpoint rung (keeps bridge to ``bot:random``).
        - The strongest active checkpoint rung (current champion).
        - Any explicit ``protected_ids`` (e.g., a freshly promoted rung).

        Among eligible interior rungs ``1 .. m-2`` sorted by Elo, removes the
        rung whose surrounding span ``Elo[i+1] - Elo[i-1]`` is smallest, keeping
        an evenly spaced ladder of skill tiers. Historical match records for
        pruned rungs are retained in ``self.matches`` so Bradley-Terry graph
        connectivity is never broken.
        """
        protected = set(protected_ids) if protected_ids else set()
        pruned: List[PlayerEntry] = []
        while True:
            ckpts = self.active_checkpoint_rungs()
            if len(ckpts) <= self.max_ladder_size or len(ckpts) <= 2:
                break
            elos = [self.get_elo(p.player_id) for p in ckpts]
            # Pick eligible interior index with smallest surrounding Elo span
            best_idx: Optional[int] = None
            best_span = float('inf')
            for i in range(1, len(ckpts) - 1):
                if ckpts[i].player_id in protected:
                    continue
                span = elos[i + 1] - elos[i - 1]
                if span < best_span:
                    best_span = span
                    best_idx = i
            if best_idx is None:
                # Fallback if all interior rungs are protected
                for i in range(1, len(ckpts) - 1):
                    span = elos[i + 1] - elos[i - 1]
                    if span < best_span:
                        best_span = span
                        best_idx = i
            if best_idx is None:
                break
            victim = ckpts[best_idx]
            victim.active = False
            pruned.append(victim)
        return pruned

    def leaderboard(self, active_only: bool = False) -> List[EloEstimate]:
        """Return ``EloEstimate`` entries sorted from highest to lowest Elo."""
        rows: List[EloEstimate] = []
        for pid, est in self.estimates.items():
            entry = self.players.get(pid)
            if active_only and entry is not None and not entry.active:
                continue
            rows.append(est)
        rows.sort(key=lambda e: (-e.elo, e.player_id))
        return rows

    def format_leaderboard(self, active_only: bool = False) -> str:
        """Render a clean ASCII leaderboard table for console/log output."""
        rows = self.leaderboard(active_only=active_only)
        lines = [
            f"{'Rank':>4}  {'Player':<28}  {'Elo':>8}  {'±Std':>6}  {'Games':>5}  {'W-D-L':<13}  {'Score%':>6}  {'Status'}",
            '-' * 88,
        ]
        for rank, est in enumerate(rows, start=1):
            entry = self.players.get(est.player_id)
            if entry is None:
                status = ''
            elif entry.is_anchor:
                status = 'anchor'
            elif entry.kind == 'bot':
                status = 'bot'
            elif entry.active:
                status = 'ladder'
            else:
                status = 'archived'
            wdl = f'{est.wins}-{est.draws}-{est.losses}'
            std_str = f'{est.elo_std:6.1f}' if math.isfinite(est.elo_std) else '   inf'
            lines.append(
                f'{rank:>4}  {est.player_id:<28}  {est.elo:>8.1f}  {std_str}  '
                f'{est.games:>5}  {wdl:<13}  {100.0 * est.score_rate:>5.1f}%  {status}'
            )
        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': 1,
            'anchor_bot_types': list(self.anchor_bot_types),
            'anchor_id': self.anchor_id,
            'anchor_elo': float(self.anchor_elo),
            'max_ladder_size': int(self.max_ladder_size),
            'opponents_per_eval': int(self.opponents_per_eval),
            'promotion_win_rate': float(self.promotion_win_rate),
            'promotion_elo_margin': float(self.promotion_elo_margin),
            'periodic_promote_every': int(self.periodic_promote_every),
            'prior_draws': float(self.prior_draws),
            'l2_reg': float(self.l2_reg),
            'last_candidate_elo': (
                float(self.last_candidate_elo)
                if self.last_candidate_elo is not None and math.isfinite(self.last_candidate_elo)
                else None
            ),
            'best_elo': float(self.best_elo) if math.isfinite(self.best_elo) else None,
            'best_player_id': self.best_player_id,
            'evals_since_promotion': int(self.evals_since_promotion),
            'players': {pid: p.to_dict() for pid, p in self.players.items()},
            'matches': [m.to_dict() for m in self.matches],
            'estimates': {pid: e.to_dict() for pid, e in self.estimates.items()},
            'eval_history': list(self.eval_history),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EloLadder':
        ladder = cls(
            anchor_bots=data.get('anchor_bot_types', ['random']),
            anchor_elo=float(data.get('anchor_elo', 0.0)),
            max_ladder_size=int(data.get('max_ladder_size', 10)),
            opponents_per_eval=int(data.get('opponents_per_eval', 4)),
            promotion_win_rate=float(data.get('promotion_win_rate', 0.55)),
            promotion_elo_margin=float(data.get('promotion_elo_margin', 50.0)),
            periodic_promote_every=int(data.get('periodic_promote_every', 0)),
            prior_draws=float(data.get('prior_draws', 0.1)),
            l2_reg=float(data.get('l2_reg', 1e-4)),
        )
        ladder.anchor_id = str(data.get('anchor_id', ladder.anchor_id))
        ladder.players = {
            pid: PlayerEntry.from_dict(pdict)
            for pid, pdict in data.get('players', {}).items()
        }
        ladder.matches = [MatchRecord.from_dict(m) for m in data.get('matches', [])]
        ladder.last_candidate_elo = (
            float(data['last_candidate_elo']) if data.get('last_candidate_elo') is not None else None
        )
        best_elo = data.get('best_elo')
        ladder.best_elo = float(best_elo) if best_elo is not None else float('-inf')
        ladder.best_player_id = data.get('best_player_id')
        ladder.evals_since_promotion = int(data.get('evals_since_promotion', 0))
        ladder.eval_history = list(data.get('eval_history', []))
        ladder.fit_ratings()
        return ladder
