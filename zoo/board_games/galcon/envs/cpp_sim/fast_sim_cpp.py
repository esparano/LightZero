"""Auto-compiled C++17 Galcon physics accelerator and standalone POD ``SimState`` wrapper.

Compiles ``galcon/sim_core.cpp`` via::

    g++ -O3 -march=native -ffp-contract=off -shared -fPIC -std=c++17

into a SHA-256 hash-keyed shared library in the OS temp directory (shared across worker
processes via atomic ``os.replace``). Using ``-ffp-contract=off`` and ``std::sqrt(dx*dx+dy*dy)``
guarantees bit-exact IEEE-754 float64 parity with ``GalconSim._step_physics_python()``.

On Windows (``win32``) or when ``g++`` is unavailable, gracefully sets
``HAS_FAST_SIM_CPP = False`` so ``GalconSim`` transparently falls back to pure Python.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from .sim import Fleet, GalconSim, Order, Planet

_PRODUCTION_DIVISOR = 50.0
_SHIP_RADIUS = 6.0
_SHIP_SEP_DIST = 13.5
_SHIP_SEP_STIFFNESS = 1.0
_SHIP_SEP_MAX_DISP = 6.0
_SPAWN_AREA_RADIUS = 6.0
_SPAWN_SETTLE_ITERS = 0
_SPAWN_SETTLE_TOL = 0.05
_ANTIPODE_ESCAPE_TANGENT = 0.05
_SHIP_TURN_RATE = 4.5
_SHIP_TURN_MIN_SPEED_FACTOR = 1.0
_SEP_MODE_PER_SHIP_CAP = 1
_GOLDEN_ANGLE = 2.399963229728653
_SPAWN_SPIRAL_STEP = 0.05
_EPS_DIST2 = 1e-12

_SIM_MAX_PLANETS = 64
_SIM_MAX_FLEETS = 512
_SIM_MAX_UNITS = 4096
_SIM_SEARCH_MAX_FLEETS = 400
_SIM_SEARCH_MAX_UNITS = 3200

_COMPILE_FLAGS = ("-O3", "-march=native", "-ffp-contract=off", "-shared", "-fPIC", "-std=c++17")
_C_VP = ctypes.c_void_p
_C_I32 = ctypes.c_int32
_C_F64 = ctypes.c_double


class COrder(ctypes.Structure):
    """C ABI order record matching ``struct COrder`` in ``sim_core.cpp``."""

    _fields_ = [
        ("kind", ctypes.c_int32),
        ("player", ctypes.c_int32),
        ("a", ctypes.c_int32),
        ("b", ctypes.c_int32),
        ("percent", ctypes.c_int32),
    ]


class CSimStateView(ctypes.Structure):
    """Zero-copy C ABI view into a live ``SimState``'s contiguous planet & fleet buffers."""

    _fields_ = [
        ("tick", ctypes.c_int32),
        ("max_ticks", ctypes.c_int32),
        ("done", ctypes.c_int32),
        ("winner", ctypes.c_int32),
        ("n_planets", ctypes.c_int32),
        ("n_fleets", ctypes.c_int32),
        ("tick_dt", ctypes.c_double),
        ("fleet_speed", ctypes.c_double),
        ("prod_divisor", ctypes.c_double),
        ("p_id", ctypes.c_void_p),
        ("p_owner", ctypes.c_void_p),
        ("p_ships", ctypes.c_void_p),
        ("p_x", ctypes.c_void_p),
        ("p_y", ctypes.c_void_p),
        ("p_prod", ctypes.c_void_p),
        ("p_radius", ctypes.c_void_p),
        ("fl_target_idx", ctypes.c_void_p),
        ("fl_owner", ctypes.c_void_p),
        ("fl_ships", ctypes.c_void_p),
        ("fl_cx", ctypes.c_void_p),
        ("fl_cy", ctypes.c_void_p),
    ]


def _materialize_fleet_units_local(cx: float, cy: float, count: int) -> List[List[float]]:
    k = max(1, int(count))
    cx_f = float(cx)
    cy_f = float(cy)
    if k == 1:
        return [[cx_f, cy_f]]
    units: List[List[float]] = []
    for j in range(k):
        rj = _SPAWN_SPIRAL_STEP * math.sqrt(float(j))
        ang = _GOLDEN_ANGLE * float(j)
        units.append([cx_f + rj * math.cos(ang), cy_f + rj * math.sin(ang)])
    return units


def _materialize_unit_headings_local(
    units: Sequence[Sequence[float]],
    target_planet: Optional["Planet"],
) -> List[List[float]]:
    if target_planet is None:
        return [[1.0, 0.0] for _ in units]
    tx = float(target_planet.x)
    ty = float(target_planet.y)
    headings: List[List[float]] = []
    for u in units:
        dx = tx - float(u[0])
        dy = ty - float(u[1])
        d2 = dx * dx + dy * dy
        if d2 > _EPS_DIST2:
            d = math.sqrt(d2)
            headings.append([dx / d, dy / d])
        else:
            headings.append([1.0, 0.0])
    return headings


def _load_or_build_lib() -> Optional[ctypes.CDLL]:
    if sys.platform == "win32":
        return None
    gpp = shutil.which("g++")
    if not gpp:
        return None
    cpp_path = Path(__file__).resolve().parent / "sim_core.cpp"
    if not cpp_path.is_file():
        return None

    try:
        src_bytes = cpp_path.read_bytes()
    except OSError:
        return None

    flags_key = " ".join(_COMPILE_FLAGS).encode("utf-8")
    h = hashlib.sha256(src_bytes + flags_key).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "galcon_fast_sim_cpp_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    so_path = cache_dir / f"galcon_sim_core_{h}.so"

    if not so_path.is_file():
        pid = os.getpid()
        so_tmp = cache_dir / f"galcon_sim_core_{h}_{pid}.so"
        try:
            subprocess.run(
                [gpp, *_COMPILE_FLAGS, str(cpp_path), "-lm", "-o", str(so_tmp)],
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

        lib.step_physics_cpp.argtypes = [
            _C_I32, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
            _C_I32, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
            _C_I32, _C_VP, _C_VP, _C_VP, _C_VP,
            _C_F64, _C_F64, _C_I32, _C_F64, _C_F64, _C_F64, _C_I32, _C_F64, _C_F64, _C_F64, _C_F64,
        ]
        lib.step_physics_cpp.restype = _C_I32

        lib.sim_create.argtypes = [
            _C_I32, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
            _C_F64, _C_F64, _C_F64, _C_F64, _C_I32, _C_I32, _C_F64,
            _C_F64, _C_F64, _C_I32, _C_I32, _C_F64, _C_F64, _C_F64,
        ]
        lib.sim_create.restype = _C_VP

        lib.sim_set_meta.argtypes = [_C_VP, _C_I32, _C_I32, _C_I32, _C_I32]
        lib.sim_set_meta.restype = None

        lib.sim_load_fleets.argtypes = [
            _C_VP, _C_I32, _C_VP, _C_VP, _C_VP, _C_VP,
            _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
            _C_I32, _C_VP, _C_VP, _C_VP, _C_VP,
        ]
        lib.sim_load_fleets.restype = _C_I32

        lib.sim_destroy.argtypes = [_C_VP]
        lib.sim_destroy.restype = None

        lib.sim_clone.argtypes = [_C_VP]
        lib.sim_clone.restype = _C_VP

        lib.sim_copy_into.argtypes = [_C_VP, _C_VP]
        lib.sim_copy_into.restype = None

        lib.sim_send_order.argtypes = [_C_VP, _C_I32, _C_I32, _C_I32, _C_I32]
        lib.sim_send_order.restype = _C_I32

        lib.sim_redir_order.argtypes = [_C_VP, _C_I32, _C_I32, _C_I32]
        lib.sim_redir_order.restype = None

        lib.sim_step_with_orders.argtypes = [_C_VP, _C_I32, _C_VP]
        lib.sim_step_with_orders.restype = None

        lib.sim_step_ticks.argtypes = [_C_VP, _C_I32]
        lib.sim_step_ticks.restype = _C_I32

        lib.sim_get_meta.argtypes = [_C_VP, _C_VP]
        lib.sim_get_meta.restype = None

        lib.sim_get_config.argtypes = [_C_VP, _C_VP, _C_VP]
        lib.sim_get_config.restype = None

        lib.sim_read_planets.argtypes = [_C_VP, _C_VP, _C_VP]
        lib.sim_read_planets.restype = None

        lib.sim_read_planets_full.argtypes = [
            _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
        ]
        lib.sim_read_planets_full.restype = None

        lib.sim_read_fleets.argtypes = [
            _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
        ]
        lib.sim_read_fleets.restype = None

        lib.sim_read_units.argtypes = [_C_VP, _C_VP, _C_VP]
        lib.sim_read_units.restype = None

        lib.sim_read_unit_headings.argtypes = [_C_VP, _C_VP, _C_VP]
        lib.sim_read_unit_headings.restype = None

        lib.sim_get_state_view.argtypes = [_C_VP, _C_VP]
        lib.sim_get_state_view.restype = None

        lib.sim_eval_heuristic_from_po.argtypes = [_C_VP, _C_I32, _C_VP, _C_I32]
        lib.sim_eval_heuristic_from_po.restype = _C_F64

        lib.sim_eval_heuristic.argtypes = [_C_VP, _C_I32, _C_I32]
        lib.sim_eval_heuristic.restype = _C_F64

        return lib
    except Exception:
        return None


_LIB: Optional[ctypes.CDLL] = _load_or_build_lib()
_STEP_PHYSICS_FN = _LIB.step_physics_cpp if _LIB is not None else None
HAS_FAST_SIM_CPP: bool = _LIB is not None


class _PhysicsScratch:
    """Reusable per-thread ctypes scratch buffers (dynamically resized on overflow)."""

    def __init__(self) -> None:
        self.p_cap = 0
        self.fl_cap = 0
        self.u_cap = 0
        self.ord_cap = 0
        self._alloc_planets(_SIM_MAX_PLANETS)
        self._alloc_fleets(_SIM_MAX_FLEETS)
        self._alloc_units(_SIM_MAX_UNITS)
        self._alloc_orders(64)
        self.cfg_f64 = (ctypes.c_double * 10)()
        self.cfg_i32 = (ctypes.c_int32 * 4)()
        self.ptr_cfg_f64 = ctypes.addressof(self.cfg_f64)
        self.ptr_cfg_i32 = ctypes.addressof(self.cfg_i32)

    def _alloc_planets(self, cap: int) -> None:
        self.p_cap = cap
        self.p_id = (ctypes.c_int32 * cap)()
        self.p_owner = (ctypes.c_int32 * cap)()
        self.p_ships = (ctypes.c_double * cap)()
        self.p_x = (ctypes.c_double * cap)()
        self.p_y = (ctypes.c_double * cap)()
        self.p_prod = (ctypes.c_double * cap)()
        self.p_radius = (ctypes.c_double * cap)()
        self.ptr_p_id = ctypes.addressof(self.p_id)
        self.ptr_p_owner = ctypes.addressof(self.p_owner)
        self.ptr_p_ships = ctypes.addressof(self.p_ships)
        self.ptr_p_x = ctypes.addressof(self.p_x)
        self.ptr_p_y = ctypes.addressof(self.p_y)
        self.ptr_p_prod = ctypes.addressof(self.p_prod)
        self.ptr_p_radius = ctypes.addressof(self.p_radius)

    def _alloc_fleets(self, cap: int) -> None:
        self.fl_cap = cap
        self.fl_id = (ctypes.c_int32 * cap)()
        self.fl_owner = (ctypes.c_int32 * cap)()
        self.fl_source = (ctypes.c_int32 * cap)()
        self.fl_tgt = (ctypes.c_int32 * cap)()
        self.fl_spu = (ctypes.c_double * cap)()
        self.fl_start = (ctypes.c_int32 * cap)()
        self.fl_cnt = (ctypes.c_int32 * cap)()
        self.fl_ships = (ctypes.c_double * cap)()
        self.fl_cx = (ctypes.c_double * cap)()
        self.fl_cy = (ctypes.c_double * cap)()
        self.fl_rad = (ctypes.c_double * cap)()
        self.ptr_fl_id = ctypes.addressof(self.fl_id)
        self.ptr_fl_owner = ctypes.addressof(self.fl_owner)
        self.ptr_fl_source = ctypes.addressof(self.fl_source)
        self.ptr_fl_tgt = ctypes.addressof(self.fl_tgt)
        self.ptr_fl_spu = ctypes.addressof(self.fl_spu)
        self.ptr_fl_start = ctypes.addressof(self.fl_start)
        self.ptr_fl_cnt = ctypes.addressof(self.fl_cnt)
        self.ptr_fl_ships = ctypes.addressof(self.fl_ships)
        self.ptr_fl_cx = ctypes.addressof(self.fl_cx)
        self.ptr_fl_cy = ctypes.addressof(self.fl_cy)
        self.ptr_fl_rad = ctypes.addressof(self.fl_rad)

    def _alloc_units(self, cap: int) -> None:
        self.u_cap = cap
        self.ux = (ctypes.c_double * cap)()
        self.uy = (ctypes.c_double * cap)()
        self.uhx = (ctypes.c_double * cap)()
        self.uhy = (ctypes.c_double * cap)()
        self.ptr_ux = ctypes.addressof(self.ux)
        self.ptr_uy = ctypes.addressof(self.uy)
        self.ptr_uhx = ctypes.addressof(self.uhx)
        self.ptr_uhy = ctypes.addressof(self.uhy)

    def _alloc_orders(self, cap: int) -> None:
        self.ord_cap = cap
        self.orders = (COrder * cap)()
        self.ptr_orders = ctypes.addressof(self.orders)

    def ensure_planets(self, n: int) -> None:
        if n > self.p_cap:
            self._alloc_planets(max(self.p_cap * 2, n))

    def ensure_fleets(self, n: int) -> None:
        if n > self.fl_cap:
            self._alloc_fleets(max(self.fl_cap * 2, n))

    def ensure_units(self, n: int) -> None:
        if n > self.u_cap:
            self._alloc_units(max(self.u_cap * 2, n))

    def ensure_orders(self, n: int) -> None:
        if n > self.ord_cap:
            self._alloc_orders(max(self.ord_cap * 2, n))


_TLS = threading.local()


def _get_scratch() -> _PhysicsScratch:
    sc = getattr(_TLS, "scratch", None)
    if sc is None:
        sc = _PhysicsScratch()
        _TLS.scratch = sc
    return sc


def step_physics_cpp(sim: "GalconSim") -> None:
    """Drop-in C++ physics accelerator for ``GalconSim.step()`` with bit-exact Python parity.

    Packs ``sim.planets`` (sorted by ``id``) and ``sim.fleets`` (insertion order) into reusable
    ctypes scratch buffers (dynamically resized if larger than the static fast-path), runs the
    C++17 20 Hz substep kernel, and writes updated planet and surviving fleet state back in place.
    """
    fn = _STEP_PHYSICS_FN
    if fn is None:
        sim._step_physics_python()
        return

    sim_planets = sim.planets
    n_planets = len(sim_planets)
    if n_planets <= 0:
        return

    planet_ids = sorted(sim_planets.keys())
    planets = [sim_planets[pid] for pid in planet_ids]

    sc = _get_scratch()
    sc.ensure_planets(n_planets)
    p_owner = sc.p_owner
    p_ships = sc.p_ships
    p_x = sc.p_x
    p_y = sc.p_y
    p_prod = sc.p_prod
    p_radius = sc.p_radius

    for i, p in enumerate(planets):
        p_owner[i] = p.owner
        p_ships[i] = p.ships
        p_x[i] = p.x
        p_y[i] = p.y
        p_prod[i] = p.production
        p_radius[i] = p.radius

    sim_fleets = sim.fleets
    n_fleets = len(sim_fleets)
    substeps = max(1, int(sim.physics_substeps))

    if n_fleets == 0:
        fn(
            n_planets,
            sc.ptr_p_owner,
            sc.ptr_p_ships,
            sc.ptr_p_x,
            sc.ptr_p_y,
            sc.ptr_p_prod,
            sc.ptr_p_radius,
            0,
            sc.ptr_fl_owner,
            sc.ptr_fl_tgt,
            sc.ptr_fl_spu,
            sc.ptr_fl_start,
            sc.ptr_fl_cnt,
            sc.ptr_fl_ships,
            sc.ptr_fl_cx,
            sc.ptr_fl_cy,
            sc.ptr_fl_rad,
            0,
            sc.ptr_ux,
            sc.ptr_uy,
            sc.ptr_uhx,
            sc.ptr_uhy,
            sim.tick_dt,
            sim.fleet_speed,
            substeps,
            _SHIP_RADIUS,
            _SHIP_SEP_STIFFNESS,
            _SHIP_SEP_MAX_DISP,
            _SEP_MODE_PER_SHIP_CAP,
            _PRODUCTION_DIVISOR,
            _ANTIPODE_ESCAPE_TANGENT,
            _SHIP_TURN_RATE,
            _SHIP_TURN_MIN_SPEED_FACTOR,
        )
        for i, p in enumerate(planets):
            p.owner = p_owner[i]
            p.ships = p_ships[i]
        return

    fleets = list(sim_fleets.values())
    total_units = 0
    for fl in fleets:
        units = fl.units
        if not units:
            units = _materialize_fleet_units_local(fl.x, fl.y, fl.num_units)
            fl.units = units
        m = len(units)
        fl.num_units = m
        if fl.unit_headings is None or len(fl.unit_headings) != m:
            fl.unit_headings = _materialize_unit_headings_local(units, sim_planets.get(fl.target))
        total_units += m

    sc.ensure_fleets(n_fleets)
    sc.ensure_units(total_units)

    id_to_idx = {pid: i for i, pid in enumerate(planet_ids)}
    fl_owner = sc.fl_owner
    fl_tgt = sc.fl_tgt
    fl_spu = sc.fl_spu
    fl_start = sc.fl_start
    fl_cnt = sc.fl_cnt
    ux = sc.ux
    uy = sc.uy
    uhx = sc.uhx
    uhy = sc.uhy

    pos = 0
    for fi, fl in enumerate(fleets):
        units = fl.units
        headings = fl.unit_headings
        m = len(units)
        fl_owner[fi] = fl.owner
        fl_tgt[fi] = id_to_idx.get(fl.target, -1)
        fl_spu[fi] = fl.ships / (m if m > 1 else 1)
        fl_start[fi] = pos
        fl_cnt[fi] = m
        for u, h in zip(units, headings):
            ux[pos] = u[0]
            uy[pos] = u[1]
            uhx[pos] = h[0]
            uhy[pos] = h[1]
            pos += 1

    fn(
        n_planets,
        sc.ptr_p_owner,
        sc.ptr_p_ships,
        sc.ptr_p_x,
        sc.ptr_p_y,
        sc.ptr_p_prod,
        sc.ptr_p_radius,
        n_fleets,
        sc.ptr_fl_owner,
        sc.ptr_fl_tgt,
        sc.ptr_fl_spu,
        sc.ptr_fl_start,
        sc.ptr_fl_cnt,
        sc.ptr_fl_ships,
        sc.ptr_fl_cx,
        sc.ptr_fl_cy,
        sc.ptr_fl_rad,
        total_units,
        sc.ptr_ux,
        sc.ptr_uy,
        sc.ptr_uhx,
        sc.ptr_uhy,
        sim.tick_dt,
        sim.fleet_speed,
        substeps,
        _SHIP_RADIUS,
        _SHIP_SEP_STIFFNESS,
        _SHIP_SEP_MAX_DISP,
        _SEP_MODE_PER_SHIP_CAP,
        _PRODUCTION_DIVISOR,
        _ANTIPODE_ESCAPE_TANGENT,
        _SHIP_TURN_RATE,
        _SHIP_TURN_MIN_SPEED_FACTOR,
    )

    for i, p in enumerate(planets):
        p.owner = p_owner[i]
        p.ships = p_ships[i]

    fl_ships = sc.fl_ships
    fl_cx = sc.fl_cx
    fl_cy = sc.fl_cy
    fl_rad = sc.fl_rad
    for fi, fl in enumerate(fleets):
        kept = fl_cnt[fi]
        if kept <= 0:
            del sim_fleets[fl.id]
        else:
            st = fl_start[fi]
            end = st + kept
            fl.units = list(map(list, zip(ux[st:end], uy[st:end])))
            fl.unit_headings = list(map(list, zip(uhx[st:end], uhy[st:end])))
            fl.num_units = kept
            fl.ships = fl_ships[fi]
            fl.x = fl_cx[fi]
            fl.y = fl_cy[fi]
            fl.radius = fl_rad[fi]


_ORDER_TYPES: Optional[tuple] = None
_SIM_TYPES: Optional[tuple] = None


def _get_order_types() -> tuple:
    global _ORDER_TYPES
    if _ORDER_TYPES is None:
        from .sim import RedirOrder, SendOrder

        _ORDER_TYPES = (SendOrder, RedirOrder)
    return _ORDER_TYPES


def _get_sim_types() -> tuple:
    global _SIM_TYPES
    if _SIM_TYPES is None:
        from .sim import Fleet, GalconSim, Planet

        _SIM_TYPES = (Fleet, GalconSim, Planet)
    return _SIM_TYPES


class FastGalconCppSim:
    """High-throughput Python wrapper around an opaque C++17 POD ``SimState*`` handle.

    Designed for ultra-fast tree search / multi-tick rollouts (zero-allocation ``copy_from``
    and native C++ ``step_n_ticks``) while maintaining bit-exact parity with ``GalconSim``.
    """

    __slots__ = ("_handle", "_lib", "_meta", "_ptr_meta", "_state_view", "_ptr_state_view")

    def __init__(self, _handle: int) -> None:
        if not HAS_FAST_SIM_CPP or _LIB is None:
            raise RuntimeError("FastGalconCppSim requires compiled C++ sim_core library (HAS_FAST_SIM_CPP is False)")
        if not _handle:
            raise RuntimeError("Invalid null C++ SimState handle")
        self._handle: Optional[int] = int(_handle)
        self._lib: ctypes.CDLL = _LIB
        self._meta = (ctypes.c_int32 * 7)()
        self._ptr_meta = ctypes.addressof(self._meta)
        self._state_view = CSimStateView()
        self._ptr_state_view = ctypes.addressof(self._state_view)

    def close(self) -> None:
        handle = self._handle
        if handle is not None:
            self._handle = None
            lib = self._lib
            if lib is not None:
                lib.sim_destroy(handle)

    def __enter__(self) -> "FastGalconCppSim":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def can_represent(
        cls,
        sim: "GalconSim",
        *,
        max_fleets: int = _SIM_SEARCH_MAX_FLEETS,
        max_units: int = _SIM_SEARCH_MAX_UNITS,
    ) -> bool:
        """Return ``True`` iff ``sim`` fits safely inside the native C++ ``SimState`` limits.

        Leaves default headroom (``400 / 512`` fleets, ``3200 / 4096`` units) so multi-ply search
        rollouts spawning new fleets never hit fixed POD capacity limits, and respects
        ``sim.use_cpp`` so pure-Python test sims cleanly fall back to ``sim.clone()``.
        """
        if not HAS_FAST_SIM_CPP or _LIB is None:
            return False
        if not getattr(sim, "use_cpp", True):
            return False
        n_planets = len(sim.planets)
        if n_planets <= 0 or n_planets > _SIM_MAX_PLANETS:
            return False
        fl_lim = min(int(max_fleets), _SIM_MAX_FLEETS)
        if len(sim.fleets) > fl_lim:
            return False
        u_lim = min(int(max_units), _SIM_MAX_UNITS)
        total_units = 0
        for fl in sim.fleets.values():
            total_units += len(fl.units) if fl.units else max(1, int(fl.num_units))
            if total_units > u_lim:
                return False
        return True

    def _require_handle(self) -> int:
        handle = self._handle
        if handle is None:
            raise RuntimeError("FastGalconCppSim handle has already been destroyed")
        return handle

    def _read_meta(self) -> ctypes.Array:
        self._lib.sim_get_meta(self._require_handle(), self._ptr_meta)
        return self._meta

    @classmethod
    def from_sim(cls, sim: "GalconSim") -> "FastGalconCppSim":
        """Create an independent native C++ ``SimState`` initialized from ``sim``."""
        if not HAS_FAST_SIM_CPP or _LIB is None:
            raise RuntimeError("FastGalconCppSim is unavailable (HAS_FAST_SIM_CPP is False)")

        planet_ids = sorted(sim.planets.keys())
        planets = [sim.planets[pid] for pid in planet_ids]
        n_planets = len(planets)
        if n_planets > _SIM_MAX_PLANETS:
            raise ValueError(f"FastGalconCppSim supports at most {_SIM_MAX_PLANETS} planets (got {n_planets})")

        fleets = list(sim.fleets.values())
        n_fleets = len(fleets)
        if n_fleets > _SIM_MAX_FLEETS:
            raise ValueError(f"FastGalconCppSim supports at most {_SIM_MAX_FLEETS} fleets (got {n_fleets})")

        unit_lists: List[List[List[float]]] = []
        heading_lists: List[List[List[float]]] = []
        total_units = 0
        for fl in fleets:
            units = fl.units if fl.units else _materialize_fleet_units_local(fl.x, fl.y, fl.num_units)
            if fl.unit_headings is not None and len(fl.unit_headings) == len(units):
                headings = fl.unit_headings
            else:
                headings = _materialize_unit_headings_local(units, sim.planets.get(fl.target))
            unit_lists.append(units)
            heading_lists.append(headings)
            total_units += len(units)
        if total_units > _SIM_MAX_UNITS:
            raise ValueError(f"FastGalconCppSim supports at most {_SIM_MAX_UNITS} physical units (got {total_units})")

        sc = _get_scratch()
        sc.ensure_planets(max(1, n_planets))
        p_id = sc.p_id
        p_owner = sc.p_owner
        p_ships = sc.p_ships
        p_x = sc.p_x
        p_y = sc.p_y
        p_prod = sc.p_prod
        p_radius = sc.p_radius

        for i, p in enumerate(planets):
            p_id[i] = p.id
            p_owner[i] = p.owner
            p_ships[i] = p.ships
            p_x[i] = p.x
            p_y[i] = p.y
            p_prod[i] = p.production
            p_radius[i] = p.radius

        handle = _LIB.sim_create(
            n_planets,
            sc.ptr_p_id,
            sc.ptr_p_owner,
            sc.ptr_p_ships,
            sc.ptr_p_x,
            sc.ptr_p_y,
            sc.ptr_p_prod,
            sc.ptr_p_radius,
            float(sim.width),
            float(sim.height),
            float(sim.fleet_speed),
            float(sim.tick_dt),
            int(sim.max_ticks),
            max(1, int(sim.physics_substeps)),
            _SHIP_RADIUS,
            _SHIP_SEP_STIFFNESS,
            _SHIP_SEP_MAX_DISP,
            _SEP_MODE_PER_SHIP_CAP,
            1 if sim.allow_redirect else 0,
            _ANTIPODE_ESCAPE_TANGENT,
            _SHIP_TURN_RATE,
            _SHIP_TURN_MIN_SPEED_FACTOR,
        )
        if not handle:
            raise RuntimeError("sim_create returned null handle")

        try:
            _LIB.sim_set_meta(
                handle,
                int(sim.tick),
                int(sim.next_fleet_id),
                1 if sim.done else 0,
                int(sim.winner()),
            )
            if n_fleets > 0:
                sc.ensure_fleets(n_fleets)
                sc.ensure_units(max(1, total_units))
                id_to_idx = {pid: i for i, pid in enumerate(planet_ids)}
                fl_id = sc.fl_id
                fl_owner = sc.fl_owner
                fl_source = sc.fl_source
                fl_tgt = sc.fl_tgt
                fl_ships = sc.fl_ships
                fl_cx = sc.fl_cx
                fl_cy = sc.fl_cy
                fl_rad = sc.fl_rad
                fl_start = sc.fl_start
                fl_cnt = sc.fl_cnt
                ux = sc.ux
                uy = sc.uy
                uhx = sc.uhx
                uhy = sc.uhy

                pos = 0
                for fi, (fl, units, headings) in enumerate(zip(fleets, unit_lists, heading_lists)):
                    m = len(units)
                    fl_id[fi] = fl.id
                    fl_owner[fi] = fl.owner
                    fl_source[fi] = fl.source
                    fl_tgt[fi] = id_to_idx.get(fl.target, -1)
                    fl_ships[fi] = fl.ships
                    fl_cx[fi] = fl.x
                    fl_cy[fi] = fl.y
                    fl_rad[fi] = fl.radius
                    fl_start[fi] = pos
                    fl_cnt[fi] = m
                    for u, h in zip(units, headings):
                        ux[pos] = u[0]
                        uy[pos] = u[1]
                        uhx[pos] = h[0]
                        uhy[pos] = h[1]
                        pos += 1

                rc = _LIB.sim_load_fleets(
                    handle,
                    n_fleets,
                    sc.ptr_fl_id,
                    sc.ptr_fl_owner,
                    sc.ptr_fl_source,
                    sc.ptr_fl_tgt,
                    sc.ptr_fl_ships,
                    sc.ptr_fl_cx,
                    sc.ptr_fl_cy,
                    sc.ptr_fl_rad,
                    sc.ptr_fl_start,
                    sc.ptr_fl_cnt,
                    total_units,
                    sc.ptr_ux,
                    sc.ptr_uy,
                    sc.ptr_uhx,
                    sc.ptr_uhy,
                )
                if rc != 0:
                    raise RuntimeError("sim_load_fleets failed")
        except Exception:
            _LIB.sim_destroy(handle)
            raise

        return cls(handle)

    def clone(self) -> "FastGalconCppSim":
        """Allocate a fresh independent C++ copy of this simulation state."""
        new_handle = self._lib.sim_clone(self._require_handle())
        if not new_handle:
            raise RuntimeError("sim_clone returned null handle")
        return FastGalconCppSim(new_handle)

    def copy_from(self, other: "FastGalconCppSim") -> None:
        """Zero-allocation overwrite of ``self`` with ``other``'s full simulation state."""
        self._lib.sim_copy_into(other._require_handle(), self._require_handle())

    def step(self, orders_by_player: Dict[int, Sequence["Order"]]) -> None:
        """Execute simultaneous player orders and advance one tick in native C++."""
        handle = self._require_handle()
        if not orders_by_player:
            self._lib.sim_step_with_orders(handle, 0, 0)
            return

        SendOrder, RedirOrder = _get_order_types()
        sc = _get_scratch()
        n_orders = 0
        for pid in sorted(orders_by_player.keys()):
            for order in orders_by_player[pid]:
                if isinstance(order, SendOrder):
                    sc.ensure_orders(n_orders + 1)
                    co = sc.orders[n_orders]
                    co.kind = 0
                    co.player = pid
                    co.a = order.src_id
                    co.b = order.dst_id
                    co.percent = order.percent
                    n_orders += 1
                elif isinstance(order, RedirOrder):
                    sc.ensure_orders(n_orders + 1)
                    co = sc.orders[n_orders]
                    co.kind = 1
                    co.player = pid
                    co.a = order.fleet_id
                    co.b = order.dst_id
                    co.percent = 0
                    n_orders += 1

        if n_orders == 0:
            self._lib.sim_step_with_orders(handle, 0, 0)
        else:
            self._lib.sim_step_with_orders(handle, n_orders, sc.ptr_orders)

    def step_n_ticks(self, n: int) -> None:
        """Advance up to ``n`` order-free ticks in a tight native C++ loop (stops early if done)."""
        n_i = int(n)
        if n_i > 0:
            self._lib.sim_step_ticks(self._require_handle(), n_i)

    def get_state_view(self) -> CSimStateView:
        """Populate and return a zero-copy ``CSimStateView`` pointing into live C++ arrays."""
        self._lib.sim_get_state_view(self._require_handle(), self._ptr_state_view)
        return self._state_view

    def evaluate_heuristic(self, player: int, horizon: int = 24) -> float:
        """Evaluate tactical heuristic value in ``[-1, +1]`` natively in C++ (matches ``evaluate_galcon_state``)."""
        return float(self._lib.sim_eval_heuristic(self._require_handle(), int(player), int(horizon)))

    def evaluate_heuristic_from_po(self, player: int, po_all_ptr: int, horizon: int = 24) -> float:
        """Evaluate tactical heuristic value natively in C++ reusing an already-projected ``po_all`` buffer."""
        return float(
            self._lib.sim_eval_heuristic_from_po(self._require_handle(), int(player), int(po_all_ptr), int(horizon))
        )

    def read_planets_light(self) -> Tuple[List[int], List["Planet"]]:
        """Read only the planet list (skipping fleets and ``GalconSim`` wrapper) for fast order decoding."""
        _, _, Planet = _get_sim_types()
        handle = self._require_handle()
        meta = self._read_meta()
        np_ = int(meta[3])
        if np_ <= 0:
            return [], []
        sc = _get_scratch()
        sc.ensure_planets(np_)
        self._lib.sim_read_planets_full(
            handle,
            sc.ptr_p_id,
            sc.ptr_p_owner,
            sc.ptr_p_ships,
            sc.ptr_p_x,
            sc.ptr_p_y,
            sc.ptr_p_prod,
            sc.ptr_p_radius,
        )
        p_id = sc.p_id
        p_owner = sc.p_owner
        p_ships = sc.p_ships
        p_x = sc.p_x
        p_y = sc.p_y
        p_prod = sc.p_prod
        p_radius = sc.p_radius
        planet_ids: List[int] = []
        planets: List["Planet"] = []
        for i in range(np_):
            pid = p_id[i]
            planet_ids.append(pid)
            planets.append(Planet(pid, p_owner[i], p_ships[i], p_x[i], p_y[i], p_prod[i], p_radius[i]))
        return planet_ids, planets

    def to_summary_sim(self, *, use_cpp: Optional[bool] = None) -> "GalconSim":
        """Materialize a lightweight read-only ``GalconSim`` snapshot without copying per-unit arrays.

        Reads planet state and fleet summaries (``id, owner, ships, x, y, source, target, radius,
        num_units``) directly from C++ into a ``GalconSim`` instance (leaving ``units`` and
        ``unit_headings`` as ``None``), suitable for feature extraction, heuristic evaluation,
        and order decoding inside search rollouts.
        """
        Fleet, GalconSim, Planet = _get_sim_types()

        handle = self._require_handle()
        meta = self._read_meta()
        tick = int(meta[0])
        done = bool(meta[1])
        winner = int(meta[2])
        np_ = int(meta[3])
        nf = int(meta[4])
        next_fid = int(meta[6])

        sc = _get_scratch()
        self._lib.sim_get_config(handle, sc.ptr_cfg_f64, sc.ptr_cfg_i32)
        cfg_f64 = sc.cfg_f64
        cfg_i32 = sc.cfg_i32

        planets_dict: Dict[int, Planet] = {}
        idx_to_pid: List[int] = []
        if np_ > 0:
            sc.ensure_planets(np_)
            self._lib.sim_read_planets_full(
                handle,
                sc.ptr_p_id,
                sc.ptr_p_owner,
                sc.ptr_p_ships,
                sc.ptr_p_x,
                sc.ptr_p_y,
                sc.ptr_p_prod,
                sc.ptr_p_radius,
            )
            p_id = sc.p_id
            p_owner = sc.p_owner
            p_ships = sc.p_ships
            p_x = sc.p_x
            p_y = sc.p_y
            p_prod = sc.p_prod
            p_radius = sc.p_radius
            for i in range(np_):
                pid = p_id[i]
                idx_to_pid.append(pid)
                planets_dict[pid] = Planet(
                    pid,
                    p_owner[i],
                    p_ships[i],
                    p_x[i],
                    p_y[i],
                    p_prod[i],
                    p_radius[i],
                )

        fleets_dict: Dict[int, Fleet] = {}
        if nf > 0:
            sc.ensure_fleets(nf)
            self._lib.sim_read_fleets(
                handle,
                sc.ptr_fl_id,
                sc.ptr_fl_owner,
                sc.ptr_fl_source,
                sc.ptr_fl_tgt,
                sc.ptr_fl_start,
                sc.ptr_fl_cnt,
                sc.ptr_fl_ships,
                sc.ptr_fl_cx,
                sc.ptr_fl_cy,
                sc.ptr_fl_rad,
            )
            fl_id = sc.fl_id
            fl_owner = sc.fl_owner
            fl_source = sc.fl_source
            fl_tgt = sc.fl_tgt
            fl_cnt = sc.fl_cnt
            fl_ships = sc.fl_ships
            fl_cx = sc.fl_cx
            fl_cy = sc.fl_cy
            fl_rad = sc.fl_rad
            for fi in range(nf):
                fid = fl_id[fi]
                tgt_idx = fl_tgt[fi]
                tgt_pid = idx_to_pid[tgt_idx] if 0 <= tgt_idx < np_ else -1
                fleets_dict[fid] = Fleet(
                    fid,
                    fl_owner[fi],
                    fl_ships[fi],
                    fl_cx[fi],
                    fl_cy[fi],
                    fl_source[fi],
                    tgt_pid,
                    fl_rad[fi],
                    fl_cnt[fi],
                    None,
                    None,
                )

        sim = object.__new__(GalconSim)
        sim.planets = planets_dict
        sim.fleets = fleets_dict
        sim.width = float(cfg_f64[0])
        sim.height = float(cfg_f64[1])
        sim.fleet_speed = float(cfg_f64[2])
        sim.tick_dt = float(cfg_f64[3])
        sim.max_ticks = int(cfg_i32[0])
        sim.tick = tick
        sim.next_fleet_id = next_fid
        sim.allow_redirect = bool(cfg_i32[3])
        sim.physics_substeps = max(1, int(cfg_i32[1]))
        sim.use_cpp = bool(HAS_FAST_SIM_CPP) if use_cpp is None else bool(use_cpp and HAS_FAST_SIM_CPP)
        sim.done = done
        sim._winner = winner
        return sim

    def to_sim(self, *, use_cpp: Optional[bool] = None) -> "GalconSim":
        """Materialize the current C++ state back into a Python ``GalconSim``."""
        Fleet, GalconSim, Planet = _get_sim_types()

        handle = self._require_handle()
        meta = self._read_meta()
        tick = int(meta[0])
        done = bool(meta[1])
        winner = int(meta[2])
        np_ = int(meta[3])
        nf = int(meta[4])
        nu = int(meta[5])
        next_fid = int(meta[6])

        sc = _get_scratch()
        self._lib.sim_get_config(handle, sc.ptr_cfg_f64, sc.ptr_cfg_i32)
        cfg_f64 = sc.cfg_f64
        cfg_i32 = sc.cfg_i32
        planets: List[Planet] = []
        idx_to_pid: List[int] = []
        if np_ > 0:
            sc.ensure_planets(np_)
            self._lib.sim_read_planets_full(
                handle,
                sc.ptr_p_id,
                sc.ptr_p_owner,
                sc.ptr_p_ships,
                sc.ptr_p_x,
                sc.ptr_p_y,
                sc.ptr_p_prod,
                sc.ptr_p_radius,
            )
            for i in range(np_):
                pid = sc.p_id[i]
                idx_to_pid.append(pid)
                planets.append(
                    Planet(
                        id=pid,
                        owner=sc.p_owner[i],
                        ships=sc.p_ships[i],
                        x=sc.p_x[i],
                        y=sc.p_y[i],
                        production=sc.p_prod[i],
                        radius=sc.p_radius[i],
                    )
                )

        fleets: List[Fleet] = []
        if nf > 0:
            sc.ensure_fleets(nf)
            self._lib.sim_read_fleets(
                handle,
                sc.ptr_fl_id,
                sc.ptr_fl_owner,
                sc.ptr_fl_source,
                sc.ptr_fl_tgt,
                sc.ptr_fl_start,
                sc.ptr_fl_cnt,
                sc.ptr_fl_ships,
                sc.ptr_fl_cx,
                sc.ptr_fl_cy,
                sc.ptr_fl_rad,
            )
            if nu > 0:
                sc.ensure_units(nu)
                self._lib.sim_read_units(handle, sc.ptr_ux, sc.ptr_uy)
                self._lib.sim_read_unit_headings(handle, sc.ptr_uhx, sc.ptr_uhy)
            ux = sc.ux
            uy = sc.uy
            uhx = sc.uhx
            uhy = sc.uhy
            for fi in range(nf):
                tgt_idx = sc.fl_tgt[fi]
                tgt_pid = idx_to_pid[tgt_idx] if 0 <= tgt_idx < np_ else -1
                st = sc.fl_start[fi]
                cnt = sc.fl_cnt[fi]
                units = list(map(list, zip(ux[st : st + cnt], uy[st : st + cnt]))) if cnt > 0 else []
                unit_headings = list(map(list, zip(uhx[st : st + cnt], uhy[st : st + cnt]))) if cnt > 0 else []
                fleets.append(
                    Fleet(
                        id=sc.fl_id[fi],
                        owner=sc.fl_owner[fi],
                        ships=sc.fl_ships[fi],
                        x=sc.fl_cx[fi],
                        y=sc.fl_cy[fi],
                        source=sc.fl_source[fi],
                        target=tgt_pid,
                        radius=sc.fl_rad[fi],
                        num_units=cnt,
                        units=units,
                        unit_headings=unit_headings,
                    )
                )

        sim = GalconSim(
            planets,
            fleets,
            width=cfg_f64[0],
            height=cfg_f64[1],
            fleet_speed=cfg_f64[2],
            tick_dt=cfg_f64[3],
            max_ticks=cfg_i32[0],
            tick=tick,
            next_fleet_id=next_fid,
            allow_redirect=bool(cfg_i32[3]),
            physics_substeps=cfg_i32[1],
            use_cpp=use_cpp,
        )
        sim.done = done
        sim._winner = winner
        return sim

    @property
    def tick(self) -> int:
        return int(self._read_meta()[0])

    @property
    def done(self) -> bool:
        return bool(self._read_meta()[1])

    def winner(self) -> int:
        return int(self._read_meta()[2])

    @property
    def num_planets(self) -> int:
        return int(self._read_meta()[3])

    @property
    def num_fleets(self) -> int:
        return int(self._read_meta()[4])

    @property
    def num_units(self) -> int:
        return int(self._read_meta()[5])


__all__ = [
    "COrder",
    "FastGalconCppSim",
    "HAS_FAST_SIM_CPP",
    "step_physics_cpp",
]
