"""Pure MCTS (zero neural network) bot powered by the calibrated C++17 Galcon simulator.

Uses heuristic-guided PUCT search with FastGalconCppSim cloning/rollouts and native C++
sim_eval_heuristic leaf evaluation (24-tick future-production & fleet-arrival projection).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from zoo.board_games.galcon.envs.cpp_sim import (
    DEFAULT_PHYSICS_SUBSTEPS,
    FastGalconCppSim,
    GalconSim,
    HAS_FAST_SIM_CPP,
    RedirOrder,
    SendOrder,
    SimFleet,
    SimPlanet,
)
from zoo.board_games.galcon.envs.cpp_sim.sim import evaluate_galcon_state

# Internal search action: None (pass), a single Order, or a tuple of RedirOrders (cell redirect).
SearchAction = Union[None, SendOrder, RedirOrder, Tuple[RedirOrder, ...]]


class _MCTSNode:
    __slots__ = ("player", "prior", "visits", "value_sum", "children", "expanded")

    def __init__(self, player: int, prior: float = 1.0) -> None:
        self.player: int = player
        self.prior: float = prior
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[SearchAction, "_MCTSNode"] = {}
        self.expanded: bool = False

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


def _score_and_prune_actions(
    sim_snap: GalconSim,
    player: int,
    *,
    send_percent: int = 50,
    min_send_ships: float = 1.0,
    allow_redirect: bool = False,
    max_actions: int = 14,
    prior_temp: float = 1.5,
) -> List[Tuple[SearchAction, float]]:
    """Generate top-K heuristic candidate actions and normalized softmax priors (no NN)."""
    opp = 2 if player == 1 else 1
    planets = list(sim_snap.planets.values())
    if not planets:
        return [(None, 1.0)]

    # Precompute incoming friendly/enemy ships per planet
    inc_my: Dict[int, float] = {}
    inc_en: Dict[int, float] = {}
    for fl in sim_snap.fleets.values():
        if fl.owner == player:
            inc_my[fl.target] = inc_my.get(fl.target, 0.0) + fl.ships
        elif fl.owner == opp:
            inc_en[fl.target] = inc_en.get(fl.target, 0.0) + fl.ships

    my_planets = [p for p in planets if p.owner == player]
    en_planets = [p for p in planets if p.owner == opp]
    if not my_planets and not allow_redirect:
        return [(None, 1.0)]

    # Enemy centroid (for front-feeding heuristics)
    if en_planets:
        en_cx = sum(p.x for p in en_planets) / len(en_planets)
        en_cy = sum(p.y for p in en_planets) / len(en_planets)
    else:
        en_cx, en_cy = 0.0, 0.0

    speed = max(sim_snap.fleet_speed, 1e-6)
    scored: List[Tuple[SearchAction, float]] = []

    # Consider sends from owned planets with enough ships
    eligible_sources = [p for p in my_planets if p.ships >= min_send_ships]
    eligible_sources.sort(key=lambda p: p.ships, reverse=True)
    # Cap source branching on very large maps to keep expansion fast
    if len(eligible_sources) > 8:
        eligible_sources = eligible_sources[:8]

    for src in eligible_sources:
        send_ships = min(src.ships, float(int(math.floor(src.ships * (send_percent / 100.0) + 0.5))))
        if send_ships < 1.0 - 1e-9:
            continue
        # Penalize draining a planet currently under immediate enemy attack
        src_threat = inc_en.get(src.id, 0.0) - (src.ships - send_ships + inc_my.get(src.id, 0.0))
        drain_penalty = 2.0 if src_threat > 0.0 else 0.0
        src_en_dist = math.hypot(src.x - en_cx, src.y - en_cy) if en_planets else 0.0

        for dst in planets:
            if dst.id == src.id:
                continue
            raw_d = math.hypot(dst.x - src.x, dst.y - src.y)
            travel_d = max(0.0, raw_d - src.radius - dst.radius)
            eta = travel_d / speed

            dst_inc_my = inc_my.get(dst.id, 0.0)
            dst_inc_en = inc_en.get(dst.id, 0.0)

            if dst.owner == 0:
                # Neutral expansion: value ROI if we can capture (including friendly incoming)
                eff_cost = dst.ships + dst_inc_en - dst_inc_my
                if eff_cost <= 0.0:
                    # Already captured by incoming friendly fleets
                    continue
                if send_ships > eff_cost:
                    roi = (dst.production + 10.0) / (eff_cost + 8.0 + 4.5 * eta)
                    score = 2.2 * roi + 0.6 - drain_penalty
                elif send_ships > 0.6 * eff_cost and dst_inc_my > 0.0:
                    roi = (dst.production + 5.0) / (eff_cost + 12.0 + 5.0 * eta)
                    score = 1.1 * roi - drain_penalty
                else:
                    # Weak chip on neutral: low prior
                    score = -1.2 - 0.02 * eff_cost - drain_penalty
            elif dst.owner == opp:
                # Enemy attack: account for production during flight
                est_def = dst.ships + (dst.production / 50.0) * eta + dst_inc_en - dst_inc_my
                if send_ships > est_def:
                    val = (dst.production + 20.0) / (max(1.0, est_def) + 8.0 + 4.0 * eta)
                    score = 2.6 * val + 0.9 - drain_penalty
                else:
                    # Pressure / attrition attack on close planet
                    val = (dst.production + 10.0) / (max(1.0, est_def) + 20.0 + 6.0 * eta)
                    score = 0.8 * val - 0.4 - drain_penalty
            else:
                # Friendly reinforcement or front feeding
                threat = dst_inc_en - (dst.ships + dst_inc_my + (dst.production / 50.0) * eta)
                if threat > 0.0:
                    # Urgent defense!
                    save_bonus = 2.0 if send_ships >= threat else 1.0
                    score = save_bonus + (dst.production / 40.0) / (1.0 + 0.5 * eta) - drain_penalty
                elif en_planets:
                    dst_en_dist = math.hypot(dst.x - en_cx, dst.y - en_cy)
                    if dst_en_dist + 15.0 < src_en_dist and src.ships >= 10.0:
                        score = 0.35 + (src_en_dist - dst_en_dist) / (120.0 + 20.0 * eta) - drain_penalty
                    else:
                        continue
                else:
                    continue

            scored.append((SendOrder(src_id=src.id, dst_id=dst.id, percent=send_percent), score))

    # Optional fleet redirects
    if allow_redirect:
        my_fleets = [fl for fl in sim_snap.fleets.values() if fl.owner == player and fl.ships >= 1.0]
        my_fleets.sort(key=lambda f: f.ships, reverse=True)
        for fl in my_fleets[:4]:
            for dst in planets:
                if dst.id == fl.target:
                    continue
                eta = max(0.0, math.hypot(dst.x - fl.x, dst.y - fl.y) - dst.radius) / speed
                if dst.owner == opp:
                    est_def = dst.ships + (dst.production / 50.0) * eta
                    if fl.ships > est_def:
                        sc = 1.8 * (dst.production + 15.0) / (max(1.0, est_def) + 10.0 + 4.0 * eta)
                        scored.append((RedirOrder(fleet_id=fl.id, dst_id=dst.id), sc))
                elif dst.owner == 0 and fl.ships > dst.ships:
                    sc = 1.5 * (dst.production + 10.0) / (dst.ships + 10.0 + 4.0 * eta)
                    scored.append((RedirOrder(fleet_id=fl.id, dst_id=dst.id), sc))
                elif dst.owner == player:
                    threat = inc_en.get(dst.id, 0.0) - (dst.ships + inc_my.get(dst.id, 0.0))
                    if threat > 0.0:
                        scored.append((RedirOrder(fleet_id=fl.id, dst_id=dst.id), 1.6 / (1.0 + 0.5 * eta)))

    # Keep top (max_actions - 1) non-pass actions + always include Pass
    scored.sort(key=lambda item: item[1], reverse=True)
    top = scored[: max(1, max_actions - 1)]
    # Pass baseline sits moderately high so MCTS saves ships when no strong attack/expand exists
    pass_score = 0.45
    candidates: List[Tuple[SearchAction, float]] = [(None, pass_score)] + top

    inv_t = 1.0 / max(prior_temp, 1e-3)
    max_s = max(s for _, s in candidates)
    exps = [math.exp((s - max_s) * inv_t) for _, s in candidates]
    total_e = sum(exps) or 1.0
    return [(act, e / total_e) for (act, _), e in zip(candidates, exps)]


class GalconMCTSSearch:
    """Standalone zero-NN PUCT MCTS engine backed by FastGalconCppSim (with Python fallback)."""

    def __init__(
        self,
        *,
        num_simulations: int = 64,
        c_puct: float = 0.5,
        max_actions: int = 10,
        max_depth: int = 8,
        eval_horizon: int = 40,
        rollout_ticks: int = 4,
        prior_temp: float = 1.5,
        send_percent: int = 50,
        min_send_ships: float = 1.0,
        allow_redirect: bool = False,
        alternating_turns: bool = True,
    ) -> None:
        self.num_simulations = max(1, int(num_simulations))
        self.c_puct = float(c_puct)
        self.max_actions = max(2, int(max_actions))
        self.max_depth = max(1, int(max_depth))
        self.eval_horizon = max(4, int(eval_horizon))
        self.rollout_ticks = max(0, int(rollout_ticks))
        self.prior_temp = float(prior_temp)
        self.send_percent = max(1, min(100, int(send_percent)))
        self.min_send_ships = float(min_send_ships)
        self.allow_redirect = bool(allow_redirect)
        self.alternating_turns = bool(alternating_turns)

    @staticmethod
    def _to_order_dict(player: int, action: SearchAction) -> Dict[int, List[Any]]:
        if action is None:
            return {}
        if isinstance(action, tuple):
            return {player: list(action)}
        return {player: [action]}

    def _expand_node(self, node: _MCTSNode, sim_snap: GalconSim) -> None:
        priors = _score_and_prune_actions(
            sim_snap,
            node.player,
            send_percent=self.send_percent,
            min_send_ships=self.min_send_ships,
            allow_redirect=self.allow_redirect,
            max_actions=self.max_actions,
            prior_temp=self.prior_temp,
        )
        opp = 2 if node.player == 1 else 1
        next_player = opp if self.alternating_turns else node.player
        for act, p in priors:
            node.children[act] = _MCTSNode(player=next_player, prior=p)
        node.expanded = True

    def _select_child(self, node: _MCTSNode) -> Tuple[SearchAction, _MCTSNode]:
        sqrt_n = math.sqrt(float(node.visits) + 1e-8)
        c_puct = self.c_puct
        best_score = -1e18
        best_act: SearchAction = None
        best_child: Optional[_MCTSNode] = None
        for act, child in node.children.items():
            # child.q_value is from child.player's perspective; negate in alternating two-player zero-sum search
            q = (-child.q_value) if self.alternating_turns else child.q_value
            u = c_puct * child.prior * sqrt_n / (1.0 + child.visits)
            sc = q + u
            if sc > best_score:
                best_score = sc
                best_act = act
                best_child = child
        assert best_child is not None
        return best_act, best_child

    def search(
        self,
        root_sim: GalconSim,
        player: int,
        *,
        allowed_actions: Optional[Sequence[Tuple[SearchAction, float]]] = None,
    ) -> Tuple[SearchAction, Dict[SearchAction, Tuple[int, float]]]:
        """Run PUCT MCTS from ``root_sim`` for ``player`` and return (best_action, visit_stats)."""
        root = _MCTSNode(player=player, prior=1.0)
        if allowed_actions is not None:
            opp = 2 if player == 1 else 1
            next_p = opp if self.alternating_turns else player
            for act, p in allowed_actions:
                root.children[act] = _MCTSNode(player=next_p, prior=p)
            root.expanded = True
        else:
            self._expand_node(root, root_sim)

        if len(root.children) <= 1:
            only_act = next(iter(root.children.keys()), None)
            return only_act, {only_act: (1, 0.0)}

        use_fast_cpp = HAS_FAST_SIM_CPP and len(root_sim.planets) <= 64 and len(root_sim.fleets) <= 512
        root_cpp: Optional[FastGalconCppSim] = None
        scratch_cpp: Optional[FastGalconCppSim] = None
        if use_fast_cpp:
            try:
                root_cpp = FastGalconCppSim.from_sim(root_sim)
                scratch_cpp = root_cpp.clone()
            except ValueError:
                use_fast_cpp = False
                root_cpp = None
                scratch_cpp = None

        try:
            for _ in range(self.num_simulations):
                node = root
                path: List[_MCTSNode] = [node]
                depth = 0

                if use_fast_cpp and root_cpp is not None and scratch_cpp is not None:
                    scratch_cpp.copy_from(root_cpp)
                    while node.expanded and node.children and depth < self.max_depth and not scratch_cpp.done:
                        act, child = self._select_child(node)
                        scratch_cpp.step(self._to_order_dict(node.player, act))
                        node = child
                        path.append(node)
                        depth += 1

                    if not scratch_cpp.done and not node.expanded and depth < self.max_depth:
                        self._expand_node(node, scratch_cpp.to_summary_sim(use_cpp=False))

                    if scratch_cpp.done:
                        w = scratch_cpp.winner()
                        leaf_val = 1.0 if w == node.player else (0.0 if w == 0 else -1.0)
                    else:
                        if self.rollout_ticks > 0:
                            scratch_cpp.step_n_ticks(self.rollout_ticks)
                        if scratch_cpp.done:
                            w = scratch_cpp.winner()
                            leaf_val = 1.0 if w == node.player else (0.0 if w == 0 else -1.0)
                        else:
                            leaf_val = float(scratch_cpp.evaluate_heuristic(node.player, horizon=self.eval_horizon))
                else:
                    sim_copy = root_sim.clone(use_cpp=False)
                    while node.expanded and node.children and depth < self.max_depth and not sim_copy.done:
                        act, child = self._select_child(node)
                        sim_copy.step(self._to_order_dict(node.player, act))
                        node = child
                        path.append(node)
                        depth += 1

                    if not sim_copy.done and not node.expanded and depth < self.max_depth:
                        self._expand_node(node, sim_copy)

                    if sim_copy.done:
                        w = sim_copy.winner()
                        leaf_val = 1.0 if w == node.player else (0.0 if w == 0 else -1.0)
                    else:
                        for _r in range(self.rollout_ticks):
                            if sim_copy.done:
                                break
                            sim_copy.step({})
                        if sim_copy.done:
                            w = sim_copy.winner()
                            leaf_val = 1.0 if w == node.player else (0.0 if w == 0 else -1.0)
                        else:
                            leaf_val = float(evaluate_galcon_state(sim_copy, node.player))

                # Backpropagate value up the search path (negating each ply for alternating two-player zero-sum)
                v = leaf_val
                for n in reversed(path):
                    n.visits += 1
                    n.value_sum += v
                    if self.alternating_turns:
                        v = -v
        finally:
            if scratch_cpp is not None:
                scratch_cpp.close()
            if root_cpp is not None:
                root_cpp.close()

        # Select root action with maximum visits (tie-break by Q value)
        stats: Dict[SearchAction, Tuple[int, float]] = {}
        best_act: SearchAction = None
        best_key = (-1, -1e18)
        for act, child in root.children.items():
            q_root = (-child.q_value) if self.alternating_turns else child.q_value
            stats[act] = (child.visits, q_root)
            key = (child.visits, q_root)
            if key > best_key:
                best_key = key
                best_act = act
        return best_act, stats


class GalconMCTSBot:
    """LightZero GalconEnv bot wrapper using pure C++ MCTS (bot_action_type='mcts')."""

    def __init__(
        self,
        env,
        *,
        num_simulations: Optional[int] = None,
        c_puct: Optional[float] = None,
        max_actions: Optional[int] = None,
        max_depth: Optional[int] = None,
        eval_horizon: Optional[int] = None,
        rollout_ticks: Optional[int] = None,
        prior_temp: Optional[float] = None,
    ) -> None:
        self.env = env
        cfg = getattr(env, "cfg", {})
        resolved_sims = int(num_simulations if num_simulations is not None else cfg.get("mcts_num_simulations", 64))
        default_rollout = 8 if resolved_sims <= 16 else 4
        self.search_engine = GalconMCTSSearch(
            num_simulations=resolved_sims,
            c_puct=float(c_puct if c_puct is not None else cfg.get("mcts_c_puct", 0.5)),
            max_actions=int(max_actions if max_actions is not None else cfg.get("mcts_max_actions", 10)),
            max_depth=int(max_depth if max_depth is not None else cfg.get("mcts_max_depth", 8)),
            eval_horizon=int(eval_horizon if eval_horizon is not None else cfg.get("mcts_eval_horizon", 40)),
            rollout_ticks=int(
                rollout_ticks if rollout_ticks is not None else cfg.get("mcts_rollout_ticks", default_rollout)
            ),
            prior_temp=float(prior_temp if prior_temp is not None else cfg.get("mcts_prior_temp", 1.5)),
            send_percent=max(1, min(100, int(round(float(env.send_ratio) * 100.0)))),
            min_send_ships=float(env.min_send_ships),
            allow_redirect=bool(getattr(env, "enable_fleet_redirection", False)),
            alternating_turns=True,
        )

    def _build_sim_from_env(self) -> GalconSim:
        env = self.env
        planets = [
            SimPlanet(
                id=int(p.id),
                owner=int(p.owner),
                ships=float(p.ships),
                x=float(p.x),
                y=float(p.y),
                production=float(p.production),
                radius=float(p.radius),
            )
            for p in env.planets
        ]
        fleets = [
            SimFleet(
                id=int(f.id),
                owner=int(f.owner),
                ships=float(f.ships),
                x=float(f.x),
                y=float(f.y),
                source=int(f.source),
                target=int(f.target),
                radius=float(f.radius),
                num_units=int(getattr(f, "num_units", 1)),
                units=getattr(f, "units", None),
                unit_headings=getattr(f, "unit_headings", None),
            )
            for f in env.fleets
        ]
        remaining_ticks = max(2, int(env.max_episode_steps) - int(env._step_count))
        return GalconSim(
            planets,
            fleets,
            width=2.0 * float(env.grid_max_x),
            height=2.0 * float(env.grid_max_y),
            fleet_speed=float(env.fleet_speed),
            tick_dt=float(env.tick_seconds),
            max_ticks=remaining_ticks + self.search_engine.max_depth + self.search_engine.rollout_ticks + 4,
            tick=0,
            next_fleet_id=int(env._next_fleet_id),
            allow_redirect=bool(getattr(env, "enable_fleet_redirection", False)),
            physics_substeps=int(getattr(env, "physics_substeps", DEFAULT_PHYSICS_SUBSTEPS)),
            use_cpp=True,
        )

    def _encode_search_action(self, action: SearchAction) -> int:
        env = self.env
        if action is None:
            return int(env.pass_action)
        planet_by_id = {int(p.id): p for p in env.planets}
        if isinstance(action, SendOrder):
            src = planet_by_id.get(action.src_id)
            dst = planet_by_id.get(action.dst_id)
            if src is None or dst is None:
                return int(env.pass_action)
            sgx, sgy = env._world_to_grid(src.x, src.y)
            tgx, tgy = env._world_to_grid(dst.x, dst.y)
            return int(env.encode_action(sgx, sgy, tgx, tgy, is_redirect=False))
        if isinstance(action, tuple) and action:
            action = action[0]
        if isinstance(action, RedirOrder):
            fl = next((f for f in env.fleets if f.id == action.fleet_id), None)
            dst = planet_by_id.get(action.dst_id)
            if fl is None or dst is None:
                return int(env.pass_action)
            sgx, sgy = env._world_to_grid(fl.x, fl.y)
            tgx, tgy = env._world_to_grid(dst.x, dst.y)
            return int(env.encode_action(sgx, sgy, tgx, tgy, is_redirect=True))
        return int(env.pass_action)

    def _expand_cell_redirect(self, action: SearchAction, player: int) -> SearchAction:
        """Group a single-fleet RedirOrder with all friendly fleets in its GalconEnv grid cell."""
        if not isinstance(action, RedirOrder):
            return action
        env = self.env
        fl = next((f for f in env.fleets if f.id == action.fleet_id), None)
        if fl is None:
            return action
        cell = env._world_to_grid(fl.x, fl.y)
        grouped = tuple(
            RedirOrder(fleet_id=int(f.id), dst_id=action.dst_id)
            for f in env.fleets
            if f.owner == player and f.target != action.dst_id and env._world_to_grid(f.x, f.y) == cell
        )
        return grouped if len(grouped) > 1 else (grouped[0] if grouped else action)

    def get_action(self) -> int:
        env = self.env
        legal = set(env.legal_actions)
        if len(legal) <= 1:
            return int(env.pass_action)

        root_sim = self._build_sim_from_env()
        player = int(env._current_player)

        # Sync search settings with any dynamic env tweaks
        self.search_engine.send_percent = max(1, min(100, int(round(float(env.send_ratio) * 100.0))))
        self.search_engine.min_send_ships = float(env.min_send_ships)
        self.search_engine.allow_redirect = bool(getattr(env, "enable_fleet_redirection", False))

        # Pre-filter root actions to those that map to strictly legal GalconEnv actions
        raw_priors = _score_and_prune_actions(
            root_sim,
            player,
            send_percent=self.search_engine.send_percent,
            min_send_ships=self.search_engine.min_send_ships,
            allow_redirect=self.search_engine.allow_redirect,
            max_actions=self.search_engine.max_actions,
            prior_temp=self.search_engine.prior_temp,
        )
        filtered: List[Tuple[SearchAction, float]] = []
        seen_env_acts = set()
        for s_act, prob in raw_priors:
            env_act = self._encode_search_action(s_act)
            if env_act in legal and env_act not in seen_env_acts:
                seen_env_acts.add(env_act)
                filtered.append((self._expand_cell_redirect(s_act, player), prob))
        if not filtered:
            return int(env.pass_action)

        tot_p = sum(p for _, p in filtered) or 1.0
        norm_priors = [(a, p / tot_p) for a, p in filtered]

        best_search_act, _ = self.search_engine.search(root_sim, player, allowed_actions=norm_priors)
        chosen = self._encode_search_action(best_search_act)
        return chosen if chosen in legal else int(env.pass_action)
