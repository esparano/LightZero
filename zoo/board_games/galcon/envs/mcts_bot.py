"""LightZero ``GalconEnv`` wrapper around ``galcon_bot_arena.bots.mcts.GalconMCTSSearch``."""

from __future__ import annotations

from typing import List, Optional, Tuple

from galcon_bot_arena.bots.mcts import (
    GalconMCTSSearch,
    SearchAction,
    _score_and_prune_actions,
)
from galcon_bot_arena.sim import (
    DEFAULT_PHYSICS_SUBSTEPS,
    GalconSim,
    RedirOrder,
    SendOrder,
    SimFleet,
    SimPlanet,
)


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
        resolved_sims = int(num_simulations if num_simulations is not None else cfg.get("mcts_num_simulations", 16))
        self.search_engine = GalconMCTSSearch(
            num_simulations=resolved_sims,
            c_puct=float(c_puct if c_puct is not None else cfg.get("mcts_c_puct", 0.5)),
            max_actions=int(max_actions if max_actions is not None else cfg.get("mcts_max_actions", 10)),
            max_depth=int(max_depth if max_depth is not None else cfg.get("mcts_max_depth", 8)),
            eval_horizon=int(eval_horizon if eval_horizon is not None else cfg.get("mcts_eval_horizon", 40)),
            rollout_ticks=int(rollout_ticks if rollout_ticks is not None else cfg.get("mcts_rollout_ticks", 0)),
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

        self.search_engine.send_percent = max(1, min(100, int(round(float(env.send_ratio) * 100.0))))
        self.search_engine.min_send_ships = float(env.min_send_ships)
        self.search_engine.allow_redirect = bool(getattr(env, "enable_fleet_redirection", False))

        raw_priors = _score_and_prune_actions(
            root_sim,
            player,
            send_percents=self.search_engine.send_percents,
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
