import random
import numpy as np


class GalconPassBot:
    """
    Fixed policy bot for Galcon which always passes
    """

    def __init__(self, env) -> None:
        self.env = env

    def get_action(self) -> int:
        return int(self.env.pass_action)


class GalconRandomBot:
    """
    Random policy bot for Galcon. Takes a random non-pass action.
    Uses a fast native C++ picker on GalconEnv's live SimState when available.
    """

    def __init__(self, env) -> None:
        self.env = env

    def get_action(self) -> int:
        picker = getattr(self.env, '_pick_cpp_bot_action', None)
        if picker is not None:
            act = picker(1, random.getrandbits(63))
            if act >= 0:
                return act
        non_pass_actions = [action for action in self.env.legal_actions if action != self.env.pass_action]
        if len(non_pass_actions) > 0:
            return int(np.random.choice(non_pass_actions))
        return int(self.env.pass_action)


class GalconFixedPolicyBot:
    """
    Deterministic fixed policy bot for Galcon.

    Phase 2 behavior:
        Return the first legal non-pass action if available; otherwise pass.
    Uses a fast native C++ picker on GalconEnv's live SimState when available.
    """

    def __init__(self, env) -> None:
        self.env = env

    def get_action(self) -> int:
        picker = getattr(self.env, '_pick_cpp_bot_action', None)
        if picker is not None:
            act = picker(0)
            if act >= 0:
                return act
        for action in self.env.legal_actions:
            if action != self.env.pass_action:
                return int(action)
        return int(self.env.pass_action)


from zoo.board_games.galcon.envs.cpp_bots import (  # noqa: E402, F401
    CPP_BOT_KINDS,
    CPP_BOT_TYPES,
    GalconCppBot,
    HAS_CPP_BOTS,
)
from zoo.board_games.galcon.envs.mcts_bot import GalconMCTSBot, GalconMCTSSearch  # noqa: E402, F401


def build_galcon_bot(env, bot_type: str):
    """Instantiate any built-in Galcon bot bound to ``env``.

    Supports:
    - Rule / search bots: ``'random'``, ``'fixed'``, ``'pass'``, ``'mcts'``, ``'mcts:<sims>'``
    - Custom C++ bot ports: ``'waffle'``, ``'waffleopt'``, ``'sparky'``, ``'sparkyn'``,
      ``'greedy'``, ``'expand'``, ``'pressure'``, ``'arcade'``, ``'counter'``
    """
    key = str(bot_type).strip().lower()
    if key == 'random':
        return GalconRandomBot(env)
    if key == 'fixed':
        return GalconFixedPolicyBot(env)
    if key == 'pass':
        return GalconPassBot(env)
    if key == 'mcts':
        return GalconMCTSBot(env)
    if key.startswith('mcts:') or (key.startswith('mcts') and key[4:].isdigit()):
        sims_str = key.split(':', 1)[1] if ':' in key else key[4:]
        if not sims_str.isdigit() or int(sims_str) < 1:
            raise ValueError(f'Invalid Galcon MCTS bot_type: {bot_type!r} (expected mcts or mcts:<sims>)')
        return GalconMCTSBot(env, num_simulations=int(sims_str))
    if key in CPP_BOT_KINDS:
        return GalconCppBot(env, bot_type=key)
    raise ValueError(f'Unsupported Galcon bot_action_type: {bot_type!r}')
