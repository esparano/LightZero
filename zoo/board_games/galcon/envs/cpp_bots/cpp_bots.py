"""Auto-compiled C++17 wrapper for custom Galcon bots (Waffle, WaffleOpt, Sparky, etc.).

Auto-compiles ``bots_bridge.cpp`` into a SHA-256 hash-keyed shared library in the OS temp
directory (matching ``zoo/board_games/galcon/envs/cpp_sim/fast_sim_cpp.py``).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

_COMPILE_FLAGS = ("-O3", "-march=native", "-ffp-contract=off", "-shared", "-fPIC", "-std=c++17")
_C_VP = ctypes.c_void_p
_C_I32 = ctypes.c_int32
_C_U32 = ctypes.c_uint32
_C_F32 = ctypes.c_float

CPP_BOT_KINDS: Dict[str, int] = {
    "waffle": 0,
    "waffleopt": 1,
    "sparky": 2,
    "sparkyn": 3,
    "greedy": 4,
    "expand": 5,
    "pressure": 6,
    "arcade": 7,
    "counter": 8,
}

CPP_BOT_TYPES: Tuple[str, ...] = tuple(CPP_BOT_KINDS.keys())

_MAX_OUT_EVENTS = 64


def _load_or_build_lib() -> Optional[ctypes.CDLL]:
    if sys.platform == "win32":
        return None
    gpp = shutil.which("g++")
    if not gpp:
        return None
    base_dir = Path(__file__).resolve().parent
    src_files = [
        base_dir / "bots_bridge.cpp",
        base_dir / "bots.hpp",
        base_dir / "galcon.hpp",
        base_dir / "galcon_engine.hpp",
    ]
    if not all(p.is_file() for p in src_files):
        return None

    hasher = hashlib.sha256()
    try:
        for p in src_files:
            hasher.update(p.read_bytes())
    except OSError:
        return None
    hasher.update(" ".join(_COMPILE_FLAGS).encode("utf-8"))
    h = hasher.hexdigest()[:16]

    cache_dir = Path(tempfile.gettempdir()) / "galcon_cpp_bots_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    so_path = cache_dir / f"galcon_cpp_bots_{h}.so"

    if not so_path.is_file():
        pid = os.getpid()
        so_tmp = cache_dir / f"galcon_cpp_bots_{h}_{pid}.so"
        try:
            subprocess.run(
                [
                    gpp,
                    *_COMPILE_FLAGS,
                    f"-I{base_dir}",
                    str(base_dir / "bots_bridge.cpp"),
                    "-lm",
                    "-o",
                    str(so_tmp),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.replace(so_tmp, so_path)
        except Exception:
            try:
                so_tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    try:
        lib = ctypes.CDLL(str(so_path))
        lib.bot_create.argtypes = [_C_I32]
        lib.bot_create.restype = _C_VP
        lib.bot_reset.argtypes = [_C_VP]
        lib.bot_reset.restype = None
        lib.bot_destroy.argtypes = [_C_VP]
        lib.bot_destroy.restype = None
        lib.bot_decide.argtypes = [
            _C_VP,   # handle
            _C_I32,  # me
            _C_F32,  # game_time
            _C_U32,  # seed
            _C_U32,  # tick
            _C_I32,  # allow_redirect
            _C_I32,  # n_planets
            _C_VP,   # p_x
            _C_VP,   # p_y
            _C_VP,   # p_prod
            _C_VP,   # p_ships
            _C_VP,   # p_owner
            _C_I32,  # n_fleets
            _C_VP,   # f_id
            _C_VP,   # f_x
            _C_VP,   # f_y
            _C_VP,   # f_ships
            _C_VP,   # f_target_idx
            _C_VP,   # f_owner
            _C_I32,  # max_out
            _C_VP,   # out_kind
            _C_VP,   # out_who
            _C_VP,   # out_tgt
            _C_VP,   # out_pct
            _C_VP,   # out_n
        ]
        lib.bot_decide.restype = _C_I32
        return lib
    except Exception:
        return None


_LIB: Optional[ctypes.CDLL] = _load_or_build_lib()
HAS_CPP_BOTS: bool = _LIB is not None


class _BotScratch:
    """Reusable per-thread ctypes scratch buffers for zero-alloc C++ bot calls."""

    def __init__(self) -> None:
        self.p_cap = 0
        self.f_cap = 0
        self._alloc_planets(64)
        self._alloc_fleets(256)
        self.out_kind = (_C_I32 * _MAX_OUT_EVENTS)()
        self.out_who = (_C_I32 * _MAX_OUT_EVENTS)()
        self.out_tgt = (_C_I32 * _MAX_OUT_EVENTS)()
        self.out_pct = (_C_I32 * _MAX_OUT_EVENTS)()
        self.out_n = (_C_I32 * _MAX_OUT_EVENTS)()
        self.ptr_out_kind = ctypes.addressof(self.out_kind)
        self.ptr_out_who = ctypes.addressof(self.out_who)
        self.ptr_out_tgt = ctypes.addressof(self.out_tgt)
        self.ptr_out_pct = ctypes.addressof(self.out_pct)
        self.ptr_out_n = ctypes.addressof(self.out_n)

    def _alloc_planets(self, cap: int) -> None:
        self.p_cap = cap
        self.p_x = (_C_F32 * cap)()
        self.p_y = (_C_F32 * cap)()
        self.p_prod = (_C_F32 * cap)()
        self.p_ships = (_C_F32 * cap)()
        self.p_owner = (_C_I32 * cap)()
        self.ptr_p_x = ctypes.addressof(self.p_x)
        self.ptr_p_y = ctypes.addressof(self.p_y)
        self.ptr_p_prod = ctypes.addressof(self.p_prod)
        self.ptr_p_ships = ctypes.addressof(self.p_ships)
        self.ptr_p_owner = ctypes.addressof(self.p_owner)

    def _alloc_fleets(self, cap: int) -> None:
        self.f_cap = cap
        self.f_id = (_C_I32 * cap)()
        self.f_x = (_C_F32 * cap)()
        self.f_y = (_C_F32 * cap)()
        self.f_ships = (_C_F32 * cap)()
        self.f_tgt = (_C_I32 * cap)()
        self.f_owner = (_C_I32 * cap)()
        self.ptr_f_id = ctypes.addressof(self.f_id)
        self.ptr_f_x = ctypes.addressof(self.f_x)
        self.ptr_f_y = ctypes.addressof(self.f_y)
        self.ptr_f_ships = ctypes.addressof(self.f_ships)
        self.ptr_f_tgt = ctypes.addressof(self.f_tgt)
        self.ptr_f_owner = ctypes.addressof(self.f_owner)

    def ensure_planets(self, n: int) -> None:
        if n > self.p_cap:
            self._alloc_planets(max(self.p_cap * 2, n))

    def ensure_fleets(self, n: int) -> None:
        if n > self.f_cap:
            self._alloc_fleets(max(self.f_cap * 2, n))


_TLS = threading.local()


def _get_scratch() -> _BotScratch:
    sc = getattr(_TLS, "scratch", None)
    if sc is None:
        sc = _BotScratch()
        _TLS.scratch = sc
    return sc


class GalconCppBot:
    """LightZero ``GalconEnv`` adapter for custom C++ Galcon bots.

    Supported ``bot_type`` values:
    ``'waffle'``, ``'waffleopt'``, ``'sparky'``, ``'sparkyn'``, ``'greedy'``,
    ``'expand'``, ``'pressure'``, ``'arcade'``, ``'counter'``.

    Maintains per-player C++ ``BotMemory`` across ticks of an episode and automatically
    resets when a new episode begins on ``env`` (detected via ``env._step_count``).
    Translates the C++ bot's emitted send/redirect events into the highest-ship legal
    ``GalconEnv`` discrete action so benchmarks play under identical rules to agents.
    """

    def __init__(self, env, bot_type: str = "waffle", seed: int = 1) -> None:
        key = str(bot_type).strip().lower()
        if key not in CPP_BOT_KINDS:
            raise ValueError(
                f"Unsupported C++ Galcon bot_type: {bot_type!r} "
                f"(expected one of {sorted(CPP_BOT_KINDS)})"
            )
        if not HAS_CPP_BOTS or _LIB is None:
            raise RuntimeError(
                f"C++ Galcon bot {bot_type!r} requires compiled bots_bridge library "
                "(HAS_CPP_BOTS is False; g++ C++17 compiler required on Linux/WSL)."
            )
        self.env = env
        self.bot_type = key
        self.kind = CPP_BOT_KINDS[key]
        self.seed = int(seed) & 0xFFFFFFFF
        self._lib = _LIB
        # Separate C++ BotMemory per Galcon player seat (1 and 2) so the same bot
        # instance can safely play both seats in bot-vs-bot self-play loops.
        self._handles: Dict[int, int] = {}
        self._last_step: Dict[int, int] = {}

    def _get_handle(self, player: int, step_count: int) -> int:
        h = self._handles.get(player)
        if h is None:
            h = int(self._lib.bot_create(self.kind))
            if not h:
                raise RuntimeError("bot_create returned null handle")
            self._handles[player] = h
            self._last_step[player] = step_count
            return h
        prev = self._last_step.get(player, -1)
        if step_count <= prev:
            self._lib.bot_reset(h)
        self._last_step[player] = step_count
        return h

    def reset(self) -> None:
        """Reset all per-seat C++ bot memory states."""
        for h in self._handles.values():
            if h:
                self._lib.bot_reset(h)
        self._last_step.clear()

    def close(self) -> None:
        handles = self._handles
        if not handles:
            return
        self._handles = {}
        self._last_step.clear()
        lib = self._lib
        if lib is not None:
            for h in handles.values():
                if h:
                    lib.bot_destroy(h)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def get_action(self) -> int:
        env = self.env
        planets = env.planets
        n_planets = len(planets)
        if n_planets <= 0:
            return int(env.pass_action)

        player = int(env._current_player)
        if player not in (env.PLAYER_1, env.PLAYER_2):
            return int(env.pass_action)
        me = 0 if player == env.PLAYER_1 else 1
        step_count = int(env._step_count)
        handle = self._get_handle(player, step_count)

        sc = _get_scratch()
        sc.ensure_planets(n_planets)
        p_x = sc.p_x
        p_y = sc.p_y
        p_prod = sc.p_prod
        p_ships = sc.p_ships
        p_owner = sc.p_owner

        pid_to_idx: Dict[int, int] = {}
        for i, p in enumerate(planets):
            pid_to_idx[p.id] = i
            p_x[i] = float(p.x)
            p_y[i] = float(p.y)
            p_prod[i] = float(p.production)
            p_ships[i] = float(p.ships)
            if p.owner == env.PLAYER_1:
                p_owner[i] = 0
            elif p.owner == env.PLAYER_2:
                p_owner[i] = 1
            else:
                p_owner[i] = -1

        fleets = env.fleets
        n_fleets_raw = len(fleets)
        sc.ensure_fleets(max(1, n_fleets_raw))
        f_id = sc.f_id
        f_x = sc.f_x
        f_y = sc.f_y
        f_ships = sc.f_ships
        f_tgt = sc.f_tgt
        f_owner = sc.f_owner

        fid_to_fleet = {}
        nf = 0
        for fl in fleets:
            tidx = pid_to_idx.get(fl.target, -1)
            if tidx < 0 or fl.ships <= 0.0:
                continue
            f_id[nf] = int(fl.id)
            f_x[nf] = float(fl.x)
            f_y[nf] = float(fl.y)
            f_ships[nf] = float(fl.ships)
            f_tgt[nf] = tidx
            f_owner[nf] = 0 if fl.owner == env.PLAYER_1 else (1 if fl.owner == env.PLAYER_2 else -1)
            fid_to_fleet[int(fl.id)] = fl
            nf += 1

        game_time = float(step_count) * float(env.tick_seconds)
        turn_idx = step_count // 2
        env_seed = int(getattr(env, "_seed", self.seed) or self.seed) & 0xFFFFFFFF
        allow_redir = 1 if env.enable_fleet_redirection else 0

        n_events = int(
            self._lib.bot_decide(
                handle,
                me,
                game_time,
                env_seed,
                turn_idx,
                allow_redir,
                n_planets,
                sc.ptr_p_x,
                sc.ptr_p_y,
                sc.ptr_p_prod,
                sc.ptr_p_ships,
                sc.ptr_p_owner,
                nf,
                sc.ptr_f_id,
                sc.ptr_f_x,
                sc.ptr_f_y,
                sc.ptr_f_ships,
                sc.ptr_f_tgt,
                sc.ptr_f_owner,
                _MAX_OUT_EVENTS,
                sc.ptr_out_kind,
                sc.ptr_out_who,
                sc.ptr_out_tgt,
                sc.ptr_out_pct,
                sc.ptr_out_n,
            )
        )
        if n_events <= 0:
            return int(env.pass_action)

        best_action: Optional[int] = None
        best_weight: float = -1.0
        out_kind = sc.out_kind
        out_who = sc.out_who
        out_tgt = sc.out_tgt
        out_n = sc.out_n

        for i in range(n_events):
            ev_kind = int(out_kind[i])
            ev_who = int(out_who[i])
            ev_tgt = int(out_tgt[i])
            if ev_tgt < 0 or ev_tgt >= n_planets:
                continue
            tgt_p = planets[ev_tgt]
            tx, ty = env._world_to_grid(tgt_p.x, tgt_p.y)
            if env._planet_at_cell(tx, ty) is None:
                continue

            if ev_kind == 0:
                if ev_who < 0 or ev_who >= n_planets or ev_who == ev_tgt:
                    continue
                src_p = planets[ev_who]
                if src_p.owner != player or not env._can_send_from_planet(src_p):
                    continue
                sx, sy = env._world_to_grid(src_p.x, src_p.y)
                cell_src = env._planet_at_cell(sx, sy, owner=player)
                if cell_src is None or cell_src.id == tgt_p.id:
                    continue
                act = int(env.encode_action(sx, sy, tx, ty, is_redirect=False))
                weight = float(out_n[i]) if out_n[i] > 0 else float(src_p.ships)
                if weight > best_weight:
                    best_weight = weight
                    best_action = act
            elif ev_kind == 1 and env.enable_fleet_redirection:
                fl = fid_to_fleet.get(ev_who)
                if fl is None or fl.owner != player or fl.target == tgt_p.id:
                    continue
                sx, sy = env._world_to_grid(fl.x, fl.y)
                act = int(env.encode_action(sx, sy, tx, ty, is_redirect=True))
                weight = float(fl.ships)
                if weight > best_weight:
                    best_weight = weight
                    best_action = act

        return best_action if best_action is not None else int(env.pass_action)
