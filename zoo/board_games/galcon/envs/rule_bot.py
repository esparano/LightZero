import numpy as np


class GalconRandomBot:
    """
    Random policy bot for Galcon.
    """

    def __init__(self, env) -> None:
        self.env = env

    def get_action(self) -> int:
        return int(np.random.choice(self.env.legal_actions))


class GalconFixedPolicyBot:
    """
    Deterministic fixed policy bot for Galcon.

    Phase 1 behavior:
        Return the first legal action.
    """

    def __init__(self, env) -> None:
        self.env = env

    def get_action(self) -> int:
        # TODO: improve this policy to something like the "classic" bot.
        return int(self.env.legal_actions[0])