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
    """

    def __init__(self, env) -> None:
        self.env = env

    def get_action(self) -> int:
        non_pass_actions = [action for action in self.env.legal_actions if action != self.env.pass_action]
        if len(non_pass_actions) > 0:
            return int(np.random.choice(non_pass_actions))
        return int(self.env.pass_action)


class GalconFixedPolicyBot:
    """
    Deterministic fixed policy bot for Galcon.

    Phase 2 behavior:
        Return the first legal non-pass action if available; otherwise pass.
    """

    def __init__(self, env) -> None:
        self.env = env

    def get_action(self) -> int:
        for action in self.env.legal_actions:
            if action != self.env.pass_action:
                return int(action)
        return int(self.env.pass_action)