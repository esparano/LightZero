"""Re-export shim delegating native C++ Galcon benchmark bots to ``galcon_bot_arena.bots.cpp``."""

from galcon_bot_arena.bots.cpp import CPP_BOT_KINDS, CPP_BOT_TYPES, GalconCppBot, HAS_CPP_BOTS  # noqa: F401

__all__ = [
    "CPP_BOT_KINDS",
    "CPP_BOT_TYPES",
    "GalconCppBot",
    "HAS_CPP_BOTS",
]
