"""Paired head-to-head match runner for two-player zero-sum LightZero board games.

Runs batched parallel games in ``battle_mode='self_play_mode'`` between any two
players (LightZero MCTS policies or built-in rule bots such as Galcon's
``random``, ``fixed``, and ``pass`` bots).

Key properties:
- **Duplicate-bridge side swapping:** Every pair of games uses the exact same
  map seed (when ``map_seed is None``) with Player 1 and Player 2 swapped so
  both map variance and first-player advantage cancel out.
- **Preserves LightZero ``to_play`` invariants:** Passes each env's real
  ``to_play in {1, 2}`` to MCTS policies so minimax negation alternates at
  every ply.
- **Full-batch MCTS inference:** Schedules ``A=P1`` games together and ``A=P2``
  games together so all active environments batch into one forward pass per ply.
"""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from ding.torch_utils import to_ndarray
from easydict import EasyDict

from lzero.mcts.utils import prepare_observation


@dataclass
class MatchSummary:
    """Detailed outcome of a head-to-head match between Player A and Player B."""

    player_a: str
    player_b: str
    wins_a: int
    draws: int
    losses_a: int
    p1_wins: int
    p2_wins: int
    total_steps: int
    elapsed_sec: float
    map_seeds: List[int]

    @property
    def total_games(self) -> int:
        return int(self.wins_a + self.draws + self.losses_a)

    @property
    def score_a(self) -> float:
        return float(self.wins_a) + 0.5 * float(self.draws)

    @property
    def win_rate_a(self) -> float:
        n = self.total_games
        return self.score_a / n if n > 0 else 0.5

    @property
    def avg_episode_length(self) -> float:
        n = self.total_games
        return float(self.total_steps) / n if n > 0 else 0.0

    @property
    def p1_win_rate(self) -> float:
        n = self.total_games
        return (float(self.p1_wins) + 0.5 * float(self.draws)) / n if n > 0 else 0.5


class MatchPlayer(ABC):
    """Abstract player interface for ``HeadToHeadRunner``."""

    def __init__(self, player_id: str, frame_stack_num: int = 1) -> None:
        self.player_id = str(player_id)
        self.frame_stack_num = max(1, int(frame_stack_num))

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        """Reset any per-environment internal state (e.g., recurrent/KV caches)."""

    @abstractmethod
    def select_actions(
        self,
        envs: Dict[int, Any],
        env_ids: List[int],
        obs_stacks: List[List[np.ndarray]],
        action_masks: List[np.ndarray],
        to_plays: List[int],
        timesteps: List[int],
    ) -> Dict[int, int]:
        """Choose one legal action per ``env_id`` in ``env_ids``."""


class PolicyMatchPlayer(MatchPlayer):
    """Wraps a LightZero policy (using ``policy.eval_mode``) for head-to-head matches."""

    def __init__(
        self,
        player_id: str,
        policy: Any,
        policy_config: EasyDict,
        task_id: Optional[int] = None,
    ) -> None:
        frame_stack = int(policy_config.model.get('frame_stack_num', 1))
        super().__init__(player_id=player_id, frame_stack_num=frame_stack)
        self.policy = policy
        self.eval_mode = policy.eval_mode if hasattr(policy, 'eval_mode') else policy
        self.policy_config = policy_config
        self.task_id = task_id

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        if not hasattr(self.eval_mode, 'reset'):
            return
        if env_ids is None:
            if self.task_id is None:
                self.eval_mode.reset()
            else:
                self.eval_mode.reset(task_id=self.task_id)
        else:
            try:
                if self.task_id is None:
                    self.eval_mode.reset(list(env_ids))
                else:
                    self.eval_mode.reset(list(env_ids), task_id=self.task_id)
            except TypeError:
                self.eval_mode.reset()

    def select_actions(
        self,
        envs: Dict[int, Any],
        env_ids: List[int],
        obs_stacks: List[List[np.ndarray]],
        action_masks: List[np.ndarray],
        to_plays: List[int],
        timesteps: List[int],
    ) -> Dict[int, int]:
        if not env_ids:
            return {}
        # Build (B, frame_stack_num, ...) array matching MuZeroEvaluator / GameSegment.get_obs()
        stack_obs = to_ndarray(obs_stacks)
        stack_obs = prepare_observation(stack_obs, self.policy_config.model.model_type)
        stack_tensor = torch.from_numpy(stack_obs).to(self.policy_config.device).float()

        to_play_arr = [np.asarray(tp, dtype=np.int64) for tp in to_plays]
        timestep_arr = [np.asarray(ts, dtype=np.int64) for ts in timesteps]

        with torch.no_grad():
            if self.task_id is None:
                policy_output = self.eval_mode.forward(
                    stack_tensor,
                    action_masks,
                    to_play_arr,
                    ready_env_id=env_ids,
                    timestep=timestep_arr,
                )
            else:
                policy_output = self.eval_mode.forward(
                    stack_tensor,
                    action_masks,
                    to_play_arr,
                    ready_env_id=env_ids,
                    timestep=timestep_arr,
                    task_id=self.task_id,
                )

        actions: Dict[int, int] = {}
        for env_id in env_ids:
            act = policy_output[env_id]['action']
            if isinstance(act, np.ndarray):
                act = int(act.reshape(-1)[0])
            elif torch.is_tensor(act):
                act = int(act.reshape(-1)[0].item())
            else:
                act = int(act)
            actions[env_id] = act
        return actions


def _build_env_bot(env: Any, bot_type: str) -> Optional[Any]:
    """Instantiate a built-in rule bot bound to ``env`` when supported.

    On Galcon, supports ``'random'``, ``'fixed'``, ``'pass'``, ``'mcts'`` (tournament-tuned
    64-simulation pure C++ MCTS), and parameterized simulation budgets such as
    ``'mcts:16'``, ``'mcts:32'``, ``'mcts:64'``, ``'mcts:128'`` (or ``'mcts64'``).
    """
    bot_key = str(bot_type).strip().lower()
    env_cfg = getattr(env, 'cfg', None)
    env_id = str(env_cfg.get('env_id', '')).lower() if isinstance(env_cfg, dict) else ''
    if env_id == 'galcon' or (hasattr(env, 'pass_action') and hasattr(env, 'planets')):
        from zoo.board_games.galcon.envs.rule_bot import (
            GalconFixedPolicyBot,
            GalconMCTSBot,
            GalconPassBot,
            GalconRandomBot,
        )

        if bot_key == 'random':
            return GalconRandomBot(env)
        if bot_key == 'fixed':
            return GalconFixedPolicyBot(env)
        if bot_key == 'pass':
            return GalconPassBot(env)
        if bot_key == 'mcts':
            return GalconMCTSBot(env)
        if bot_key.startswith('mcts:') or (bot_key.startswith('mcts') and bot_key[4:].isdigit()):
            sims_str = bot_key.split(':', 1)[1] if ':' in bot_key else bot_key[4:]
            if not sims_str.isdigit() or int(sims_str) < 1:
                raise ValueError(f'Invalid Galcon MCTS bot_type: {bot_type!r} (expected mcts or mcts:<sims>)')
            return GalconMCTSBot(env, num_simulations=int(sims_str))
        raise ValueError(f'Unsupported Galcon bot_type: {bot_type!r}')

    if bot_key == 'random':
        return None  # Fallback to uniform random over action_mask
    if hasattr(env, 'bot_action'):
        return None
    raise ValueError(f'Unsupported bot_type {bot_type!r} for environment {type(env).__name__}')


class EnvBotMatchPlayer(MatchPlayer):
    """Built-in rule-bot player (e.g. Galcon ``random``, ``fixed``, ``pass``, ``mcts``, ``mcts:128``).

    On Galcon, invokes canonical rule bots directly (independent of
    ``prob_random_action_in_bot`` / ``prob_pass_action_in_bot``) so Elo anchors
    stay consistent across configs.
    """

    def __init__(self, bot_type: str, player_id: Optional[str] = None) -> None:
        self.bot_type = str(bot_type).strip().lower()
        super().__init__(player_id=player_id or f'bot:{self.bot_type}', frame_stack_num=1)

    def select_actions(
        self,
        envs: Dict[int, Any],
        env_ids: List[int],
        obs_stacks: List[List[np.ndarray]],
        action_masks: List[np.ndarray],
        to_plays: List[int],
        timesteps: List[int],
    ) -> Dict[int, int]:
        actions: Dict[int, int] = {}
        for idx, env_id in enumerate(env_ids):
            env = envs[env_id]
            bot = _build_env_bot(env, self.bot_type)
            if bot is not None:
                actions[env_id] = int(bot.get_action())
            elif self.bot_type == 'random':
                mask = np.asarray(action_masks[idx])
                legal = np.flatnonzero(mask == 1)
                if legal.size == 0:
                    actions[env_id] = 0
                else:
                    actions[env_id] = int(np.random.choice(legal))
            else:
                env_bot_type = getattr(env, 'bot_action_type', None)
                if (
                    env_bot_type is not None
                    and self.bot_type not in {'default', 'bot', str(env_bot_type).strip().lower()}
                ):
                    raise ValueError(
                        f'EnvBotMatchPlayer requested bot_type={self.bot_type!r}, but '
                        f'{type(env).__name__} is configured with bot_action_type={env_bot_type!r}.'
                    )
                actions[env_id] = int(env.bot_action())
        return actions


def _extract_winner_player(timestep: Any, acting_player: int) -> int:
    """Return winning player id in ``{1, 2}`` or ``-1`` for draw.

    Follows LightZero board-game ``self_play_mode`` invariants:
    1. Prefer explicit ``info['winner']`` in ``{1, 2, -1}`` when provided (e.g. Galcon).
    2. Otherwise prefer ``timestep.reward``, which is always from the **acting player's**
       perspective in ``self_play_mode`` (``> 0`` means ``acting_player`` won).
    3. Only if ``timestep.reward`` is unavailable, fall back to ``info['eval_episode_return']``,
       which is from **Player 1's** perspective in standard LightZero board games.
    """
    info = timestep.info if isinstance(timestep.info, dict) else {}
    if 'winner' in info and info['winner'] is not None:
        w = int(info['winner'])
        if w in (1, 2, -1):
            return w

    if timestep.reward is not None:
        rew = float(np.asarray(timestep.reward).reshape(-1)[0])
        if rew > 0.0:
            return int(acting_player)
        if rew < 0.0:
            return 2 if int(acting_player) == 1 else 1
        return -1

    if 'eval_episode_return' in info and info['eval_episode_return'] is not None:
        ret = float(np.asarray(info['eval_episode_return']).reshape(-1)[0])
        if ret > 0.0:
            return 1
        if ret < 0.0:
            return 2
        return -1

    return -1


def _reset_env_with_pair_seed(env: Any, pair_seed: int, fixed_map_seed: Optional[int]) -> Dict[str, Any]:
    """Reset ``env`` using ``pair_seed`` when maps are randomized, preserving
    ``fixed_map_seed`` when the config pins a single map.

    For ``GalconEnv``, setting ``env.map_seed = pair_seed`` directly ensures
    both halves of a paired game receive the exact same map without resetting
    global ``np.random`` (keeping bot/MCTS stochasticity independent).
    """
    if hasattr(env, 'map_seed'):
        if fixed_map_seed is None:
            env.map_seed = int(pair_seed)
        else:
            env.map_seed = int(fixed_map_seed)
        return env.reset()

    if hasattr(env, 'seed'):
        try:
            env.seed(int(pair_seed), dynamic_seed=False)
        except TypeError:
            env.seed(int(pair_seed))
    return env.reset()


class HeadToHeadRunner:
    """Runs batched paired head-to-head matches in ``self_play_mode``."""

    def __init__(
        self,
        env_fn: Callable[..., Any],
        env_cfg: EasyDict,
        env_num: int = 4,
    ) -> None:
        self.env_fn = env_fn
        cfg = EasyDict(copy.deepcopy(env_cfg))
        cfg.battle_mode = 'self_play_mode'
        cfg.render_mode = None
        if 'prob_random_agent' in cfg:
            cfg.prob_random_agent = 0
        if 'prob_expert_agent' in cfg:
            cfg.prob_expert_agent = 0
        self.env_cfg = cfg
        self.fixed_map_seed = cfg.get('map_seed', None)
        self.env_num = max(1, int(env_num))
        self._envs: Dict[int, Any] = {
            i: self.env_fn(copy.deepcopy(self.env_cfg)) for i in range(self.env_num)
        }

    def close(self) -> None:
        for env in self._envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._envs.clear()

    def play_match(
        self,
        player_a: MatchPlayer,
        player_b: MatchPlayer,
        num_games: int = 10,
        seed: int = 0,
    ) -> MatchSummary:
        """Play ``num_games`` paired games between ``player_a`` and ``player_b``.

        Games are scheduled as paired waves:
        - First ``num_pairs`` games: ``player_a`` plays Player 1 on seeds ``s_0 .. s_{p-1}``.
        - Next ``num_pairs`` games: ``player_a`` plays Player 2 on the exact same seeds ``s_0 .. s_{p-1}``.
        - If ``num_games`` is odd, 1 extra game is appended with Player 1/2 chosen uniformly from ``rng``.
        """
        n_games = max(1, int(num_games))
        rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
        num_pairs = n_games // 2
        pair_seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(num_pairs)]

        schedule: List[Tuple[int, int]] = []
        # Wave 1: A plays Player 1
        for s in pair_seeds:
            schedule.append((s, 1))
        # Wave 2: A plays Player 2 (identical map seeds)
        for s in pair_seeds:
            schedule.append((s, 2))
        if n_games % 2 == 1:
            odd_a_player = int(rng.randint(1, 3))
            schedule.append((int(rng.randint(0, 2**31 - 1)), odd_a_player))

        max_stack = max(player_a.frame_stack_num, player_b.frame_stack_num)
        player_a.reset()
        player_b.reset()

        t0 = time.time()
        wins_a = 0
        draws = 0
        losses_a = 0
        p1_wins = 0
        p2_wins = 0
        total_steps = 0
        played_map_seeds: List[int] = []

        next_idx = 0
        active_slots: Dict[int, Dict[str, Any]] = {}

        def _launch_slot(env_id: int) -> None:
            nonlocal next_idx
            if next_idx >= len(schedule):
                return
            pair_seed, a_player_id = schedule[next_idx]
            next_idx += 1
            env = self._envs[env_id]
            init_obs = _reset_env_with_pair_seed(env, pair_seed, self.fixed_map_seed)
            actual_seed = int(getattr(env, '_current_map_seed', pair_seed))
            played_map_seeds.append(actual_seed)

            frame = to_ndarray(init_obs['observation'])
            buf: Deque[np.ndarray] = deque([frame for _ in range(max_stack)], maxlen=max_stack)
            active_slots[env_id] = {
                'a_player_id': a_player_id,
                'obs_buf': buf,
                'action_mask': to_ndarray(init_obs['action_mask']),
                'to_play': int(np.asarray(init_obs['to_play']).reshape(-1)[0]),
                'timestep': int(np.asarray(init_obs.get('timestep', 0)).reshape(-1)[0]),
            }
            player_a.reset([env_id])
            player_b.reset([env_id])

        for env_id in range(min(self.env_num, len(schedule))):
            _launch_slot(env_id)

        while active_slots:
            a_env_ids: List[int] = []
            b_env_ids: List[int] = []

            for env_id in sorted(active_slots.keys()):
                slot = active_slots[env_id]
                tp = slot['to_play']
                if tp not in (1, 2):
                    raise ValueError(
                        f'HeadToHeadRunner requires self_play_mode with to_play in {{1, 2}}, got {tp}'
                    )
                if tp == slot['a_player_id']:
                    a_env_ids.append(env_id)
                else:
                    b_env_ids.append(env_id)

            chosen_actions: Dict[int, int] = {}
            if a_env_ids:
                a_stacks = [
                    list(active_slots[eid]['obs_buf'])[-player_a.frame_stack_num:] for eid in a_env_ids
                ]
                a_masks = [active_slots[eid]['action_mask'] for eid in a_env_ids]
                a_tps = [active_slots[eid]['to_play'] for eid in a_env_ids]
                a_ts = [active_slots[eid]['timestep'] for eid in a_env_ids]
                chosen_actions.update(
                    player_a.select_actions(self._envs, a_env_ids, a_stacks, a_masks, a_tps, a_ts)
                )
            if b_env_ids:
                b_stacks = [
                    list(active_slots[eid]['obs_buf'])[-player_b.frame_stack_num:] for eid in b_env_ids
                ]
                b_masks = [active_slots[eid]['action_mask'] for eid in b_env_ids]
                b_tps = [active_slots[eid]['to_play'] for eid in b_env_ids]
                b_ts = [active_slots[eid]['timestep'] for eid in b_env_ids]
                chosen_actions.update(
                    player_b.select_actions(self._envs, b_env_ids, b_stacks, b_masks, b_tps, b_ts)
                )

            finished_env_ids: List[int] = []
            for env_id, action in chosen_actions.items():
                slot = active_slots[env_id]
                acting_player = slot['to_play']
                timestep = self._envs[env_id].step(int(action))
                total_steps += 1

                if timestep.done:
                    winner_player = _extract_winner_player(timestep, acting_player)
                    if winner_player == -1:
                        draws += 1
                    else:
                        if winner_player == 1:
                            p1_wins += 1
                        elif winner_player == 2:
                            p2_wins += 1
                        if winner_player == slot['a_player_id']:
                            wins_a += 1
                        else:
                            losses_a += 1
                    finished_env_ids.append(env_id)
                else:
                    next_obs = timestep.obs
                    slot['obs_buf'].append(to_ndarray(next_obs['observation']))
                    slot['action_mask'] = to_ndarray(next_obs['action_mask'])
                    slot['to_play'] = int(np.asarray(next_obs['to_play']).reshape(-1)[0])
                    slot['timestep'] = int(
                        np.asarray(next_obs.get('timestep', slot['timestep'] + 1)).reshape(-1)[0]
                    )

            for env_id in finished_env_ids:
                del active_slots[env_id]
                _launch_slot(env_id)

        elapsed = time.time() - t0
        return MatchSummary(
            player_a=player_a.player_id,
            player_b=player_b.player_id,
            wins_a=wins_a,
            draws=draws,
            losses_a=losses_a,
            p1_wins=p1_wins,
            p2_wins=p2_wins,
            total_steps=total_steps,
            elapsed_sec=elapsed,
            map_seeds=played_map_seeds,
        )
