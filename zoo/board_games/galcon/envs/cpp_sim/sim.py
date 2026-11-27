"""Fast continuous-coordinate Python Galcon simulator matching Phil Hassey's ``gbots`` physics.

Physics constants match real Galcon / ``gbots`` and ``esparano/LightZero``:
- Production rate: ``production / 50.0`` ships per second (``prod=100`` -> 2.0 ships/sec).
- Constant fleet speed: ``40.0`` units/sec.
- Default tick step: ``0.25`` seconds (4 ticks/sec, 240 ticks = 60s classic round).
- Actions: ``SendOrder(src_id, dst_id, percent)`` (1..100%) and ``RedirOrder(fleet_id, dst_id)``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .fast_sim_cpp import HAS_FAST_SIM_CPP, step_physics_cpp as _step_physics_cpp

PRODUCTION_DIVISOR = 50.0  # production=100 => 2.0 ships/sec (120 ships/min)
DEFAULT_FLEET_SPEED = 40.0
DEFAULT_TICK_DT = 0.25
DEFAULT_MAX_TICKS = 240  # 60 seconds at 4 ticks/sec
SHIP_RADIUS = 6.0
PLANET_SETTLE_PADDING = 8.0 + math.sqrt(0.5)  # Real G2 planets_settle per-planet padding (~8.707)
PLANET_SETTLE_DELTA = 2.0 * PLANET_SETTLE_PADDING - 2.0 * SHIP_RADIUS  # 4 + sqrt(2) ~= 5.4142
MIN_PLANET_GAP = 2.0 * SHIP_RADIUS + PLANET_SETTLE_DELTA  # 16 + sqrt(2) ~= 17.4142
DEFAULT_MAP_WIDTH = 480.0
DEFAULT_MAP_HEIGHT = 320.0
DEFAULT_NUM_PLAYERS = 2
DEFAULT_NUM_PLANETS = 22
NEUTRAL_SHIPS_MIN = 0
NEUTRAL_SHIPS_MAX = 45
NEUTRAL_PROD_MIN = 15
NEUTRAL_PROD_MAX = 100
TIE_EPS = 1e-9
ARRIVAL_EPS = 1e-9
ITEMS_MAX = 4096
ITEMS_RESERVED = 16
BASE_ITEM_USERS = 3  # neutral + P1 + P2
DEFAULT_PHYSICS_SUBSTEPS = 5  # 20 Hz physics at tick_dt=0.25s (sub_dt=0.05s, sub_step_dist=2.0u)
SHIP_SEP_DIST = 13.5  # ship-to-ship separation buffer (calibrated against real Galcon 2 congestion/landing spread)
SHIP_SEP_STIFFNESS = 1.0
SHIP_SEP_MAX_DISP = 6.0  # per-ship displacement cap (20 Hz substep separation)
SPAWN_AREA_RADIUS = 6.0  # base spawn disc radius; multi-unit fleets scale as SPAWN_AREA_RADIUS * sqrt(parts)
SPAWN_SETTLE_ITERS = 0  # real G2 settles ship overlaps gradually across ticks (verified via 60 Hz spawn probe)
SPAWN_SETTLE_TOL = 0.05  # early-exit tolerance if launch-tick settling is enabled
ANTIPODE_DOT_THRESHOLD = -0.995
ANTIPODE_ESCAPE_TANGENT = 0.0  # Replaced by G2 stuck-unit jitter rule (unit_move @0x453ac0)
SHIP_TURN_RATE = math.pi * 40.0 / 32.0  # Real G2 (unit_move @0x453ac0): 3.92699 rad/s (225 deg/s)
TURN_ESCALATE_PER_SEC = math.pi * 40.0 / 32.0  # Escalation per second after >1.0s sustained turning
SHIP_TURN_MIN_SPEED_FACTOR = 1.0  # real G2 ships maintain full 40 u/s speed through turns
STUCK_DIST_DIVISOR = 15.0  # Real G2 (unit_move @0x453ac0): |x-px|+|y-py| < travel / 15
STUCK_JITTER_MOD = 48  # Real G2 (unit_move @0x453ac0): (uid + age) % 48 == 0
_GOLDEN_ANGLE = 2.399963229728653  # pi * (3 - sqrt(5))
_SPAWN_SPIRAL_STEP = 0.05
_EPS_DIST2 = 1e-12


@dataclass
class Planet:
    id: int
    owner: int  # 0 = neutral, 1..K = player IDs
    ships: float
    x: float
    y: float
    production: float
    radius: float


@dataclass
class Fleet:
    id: int
    owner: int
    ships: float
    x: float
    y: float
    source: int
    target: int
    radius: float = SHIP_RADIUS
    num_units: int = 1
    units: Optional[List[List[float]]] = None
    unit_headings: Optional[List[List[float]]] = None

    @property
    def ships_per_unit(self) -> float:
        return self.ships / max(1, self.num_units)


def _spawn_hash_u01(seed: int) -> float:
    """Deterministic uint32 hash -> ``[0, 1)`` matching C++ ``spawn_hash_u01`` bit-for-bit."""
    x = seed & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return float(x) / 4294967296.0


def _materialize_fleet_units(cx: float, cy: float, count: int) -> List[List[float]]:
    """Build deterministic golden-angle micro-spiral unit coordinates centered at ``(cx, cy)``."""
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


def _materialize_unit_headings(
    units: Sequence[Sequence[float]],
    target_planet: Optional[Planet],
) -> List[List[float]]:
    """Build per-unit heading vectors pointing from each unit toward ``target_planet`` (fallback ``[1.0, 0.0]``)."""
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


def physical_ship_count(
    ships: float,
    total_planet_ships: float,
    item_count: int,
    *,
    items_max: int = ITEMS_MAX,
    reserved: int = ITEMS_RESERVED,
) -> int:
    """Compute physical ship count for a launched fleet (Galcon 2 ``galcon_fleet.cpp`` / Galcon 3 ``galcon/fleet.go``).

    Matches the exact world item-table throttling formula:
    - ``items_max`` defaults to ``ITEMS_MAX = 4096`` and ``reserved`` defaults to ``ITEMS_RESERVED = 16``.
    - ``item_count`` is total occupied item-table slots: ``users + planets + fleets + physical_ship_units``.
    - ``total_planet_ships`` is the sum of non-negative ships on **planets only** (flying fleets excluded),
      counted **before** the fleet leaves its source planet.
    - Returns ``0`` if ``ships < 1.0 - TIE_EPS``; otherwise returns at least ``1`` (soft cap when table is full).
    """
    if ships < 1.0 - TIE_EPS:
        return 0
    avail = items_max - int(item_count) - int(reserved)
    if avail <= 0:
        avail = 0
    avail //= 2
    parts = int(ships)
    parts = int(min(parts, math.sqrt(17.0 * ships)))
    if total_planet_ships > 0.0:
        parts = int(min(parts, avail * ships / total_planet_ships))
    return max(1, parts)


@dataclass(frozen=True)
class SendOrder:
    src_id: int
    dst_id: int
    percent: int  # integer in [1, 100], matching gbots `/SEND PERCENT SOURCE TARGET`


@dataclass(frozen=True)
class RedirOrder:
    fleet_id: int
    dst_id: int  # matching gbots `/REDIR FLEET_ID TARGET`


Order = Union[SendOrder, RedirOrder]


def planet_radius_from_production(production: float) -> float:
    """Exact Galcon 2 planet radius from production (15..100 -> 12.0..24.0, clamped below 15).

    Note: Galcon 3 uses ``(production + 50.0) * 16.0 / 100.0``; we stick with the
    Galcon 2 formula ``max(12.0, (production * 12.0 / 5.0 + 168.0) / 17.0)`` for now.
    """
    return max(12.0, (float(production) * 12.0 / 5.0 + 168.0) / 17.0)


def _axis_bounds(radius: float, span: float) -> Tuple[float, float]:
    """Legal symmetric ``[lo, hi]`` planet-center bounds along one map axis."""
    margin = float(radius) + PLANET_SETTLE_PADDING
    if span - 2.0 * margin >= 0.0:
        return margin, span - margin
    fallback = float(radius) + 2.0
    if span - 2.0 * fallback >= 0.0:
        return fallback, span - fallback
    mid = span * 0.5
    return mid, mid


def _clone_planets(planets: Sequence[Planet]) -> List[Planet]:
    return [
        Planet(p.id, p.owner, float(p.ships), float(p.x), float(p.y), float(p.production), float(p.radius))
        for p in planets
    ]


def _planets_clearance_ok(
    planets: Sequence[Planet],
    min_gap: float = MIN_PLANET_GAP,
    tol: float = 1e-6,
) -> bool:
    n = len(planets)
    for i in range(n):
        pi = planets[i]
        for j in range(i + 1, n):
            pj = planets[j]
            if math.hypot(pi.x - pj.x, pi.y - pj.y) < pi.radius + pj.radius + min_gap - tol:
                return False
    return True


def settle_planets(
    planets: Sequence[Planet],
    width: float,
    height: float,
    min_gap: float = MIN_PLANET_GAP,
    symmetric: bool = True,
    max_iters: int = 30,
) -> bool:
    """Iteratively push overlapping / too-close planets apart to enforce ``min_gap`` clearance.

    Matches Galcon 2's ``planets_settle`` and ``origin/galcon-test2``'s relaxation loop:
    - Keeps every planet within ``[r + SHIP_RADIUS, span - r - SHIP_RADIUS]`` (falling back to
      ``[r + 2.0, span - r - 2.0]`` only if the map is too small for the full margin).
    - When ``symmetric=True``, preserves exact 180-degree rotational symmetry
      (``x_a + x_b == width``, ``y_a + y_b == height``) for symmetric pairs and pins any odd
      center planet at ``(width / 2, height / 2)``.

    Mutates ``planets`` in place and returns ``True`` if all pairs satisfy clearance.
    """
    n = len(planets)
    if n == 0:
        return True
    width = float(width)
    height = float(height)
    cx, cy = width * 0.5, height * 0.5
    min_gap = float(min_gap)

    pairs: List[Tuple[int, int]] = []
    center_indices: List[int] = []
    free_indices: List[int] = []

    if symmetric:
        used = [False] * n
        for i in range(0, n - 1, 2):
            pa, pb = planets[i], planets[i + 1]
            if (
                abs((pa.x + pb.x) - width) <= 1e-3
                and abs((pa.y + pb.y) - height) <= 1e-3
                and abs(pa.radius - pb.radius) <= 1e-3
            ):
                pairs.append((i, i + 1))
                used[i] = used[i + 1] = True
        for i in range(n):
            if used[i]:
                continue
            pa = planets[i]
            for j in range(i + 1, n):
                if used[j]:
                    continue
                pb = planets[j]
                if (
                    abs((pa.x + pb.x) - width) <= 1e-3
                    and abs((pa.y + pb.y) - height) <= 1e-3
                    and abs(pa.radius - pb.radius) <= 1e-3
                ):
                    pairs.append((i, j))
                    used[i] = used[j] = True
                    break
        for i in range(n):
            if used[i]:
                continue
            p = planets[i]
            if abs(p.x - cx) <= 1e-3 and abs(p.y - cy) <= 1e-3:
                center_indices.append(i)
            else:
                free_indices.append(i)
    else:
        free_indices = list(range(n))

    center_set = set(center_indices)

    # Snap bounds and exact 180-degree symmetry before iterating
    for i in center_indices:
        planets[i].x = cx
        planets[i].y = cy
    for a, b in pairs:
        pa, pb = planets[a], planets[b]
        lo_x, hi_x = _axis_bounds(pa.radius, width)
        lo_y, hi_y = _axis_bounds(pa.radius, height)
        pa.x = max(lo_x, min(hi_x, float(pa.x)))
        pa.y = max(lo_y, min(hi_y, float(pa.y)))
        pb.x = width - pa.x
        pb.y = height - pa.y
    for i in free_indices:
        pi = planets[i]
        lo_x, hi_x = _axis_bounds(pi.radius, width)
        lo_y, hi_y = _axis_bounds(pi.radius, height)
        pi.x = max(lo_x, min(hi_x, float(pi.x)))
        pi.y = max(lo_y, min(hi_y, float(pi.y)))

    eps_tol = 1e-7
    nudge_eps = 1e-3
    max_step = max(min_gap, 0.25 * min(width, height))

    def _repulsion_for(idx: int, it: int) -> Tuple[float, float, bool]:
        pi = planets[idx]
        dx_total = 0.0
        dy_total = 0.0
        hit = False
        for j in range(n):
            if j == idx:
                continue
            pj = planets[j]
            dx = pi.x - pj.x
            dy = pi.y - pj.y
            dist = math.hypot(dx, dy)
            min_dist = pi.radius + pj.radius + min_gap
            if dist < min_dist - eps_tol:
                overlap = min_dist - dist
                if dist < 1e-6:
                    angle = idx * 2.399963229728653 + j * 0.7319 + it * 1.13
                    ux, uy = math.cos(angle), math.sin(angle)
                else:
                    ux = dx / dist
                    uy = dy / dist
                frac = 1.0 if j in center_set else 0.5
                step = frac * overlap + nudge_eps
                dx_total += ux * step
                dy_total += uy * step
                hit = True
        mag = math.hypot(dx_total, dy_total)
        if mag > max_step:
            scale = max_step / mag
            dx_total *= scale
            dy_total *= scale
        return dx_total, dy_total, hit

    for it in range(max(1, int(max_iters))):
        moved = False
        pair_disp: List[Tuple[float, float]] = []
        free_disp: List[Tuple[float, float]] = []

        for a, b in pairs:
            dxa, dya, ha = _repulsion_for(a, it)
            if free_indices:
                dxb, dyb, hb = _repulsion_for(b, it)
                dx = 0.5 * (dxa - dxb)
                dy = 0.5 * (dya - dyb)
                if ha or hb:
                    moved = True
            else:
                dx, dy = dxa, dya
                if ha:
                    moved = True
            pair_disp.append((dx, dy))

        for idx in free_indices:
            dx, dy, hit = _repulsion_for(idx, it)
            if hit:
                moved = True
            free_disp.append((dx, dy))

        if not moved:
            return _planets_clearance_ok(planets, min_gap=min_gap)

        for (a, b), (dx, dy) in zip(pairs, pair_disp):
            if dx == 0.0 and dy == 0.0:
                continue
            pa, pb = planets[a], planets[b]
            lo_x, hi_x = _axis_bounds(pa.radius, width)
            lo_y, hi_y = _axis_bounds(pa.radius, height)
            pa.x = max(lo_x, min(hi_x, pa.x + dx))
            pa.y = max(lo_y, min(hi_y, pa.y + dy))
            pb.x = width - pa.x
            pb.y = height - pa.y

        for idx, (dx, dy) in zip(free_indices, free_disp):
            if dx == 0.0 and dy == 0.0:
                continue
            pi = planets[idx]
            lo_x, hi_x = _axis_bounds(pi.radius, width)
            lo_y, hi_y = _axis_bounds(pi.radius, height)
            pi.x = max(lo_x, min(hi_x, pi.x + dx))
            pi.y = max(lo_y, min(hi_y, pi.y + dy))

    return _planets_clearance_ok(planets, min_gap=min_gap)


def symmetric_planet_count(num_planets: int, num_players: int = DEFAULT_NUM_PLAYERS) -> int:
    """Round ``num_planets`` down modulo ``num_players`` (at least ``num_players``) for fair symmetric maps."""
    n_players = max(1, int(num_players))
    req = int(num_planets)
    return max(n_players, req - (req % n_players))


def generate_galcon_map(
    *,
    seed: int = 42,
    num_planets: int = DEFAULT_NUM_PLANETS,
    num_players: int = DEFAULT_NUM_PLAYERS,
    width: float = DEFAULT_MAP_WIDTH,
    height: float = DEFAULT_MAP_HEIGHT,
    home_ships: float = 100.0,
    home_prod: float = 100.0,
    symmetric: bool = True,
) -> List[Planet]:
    """Generate a balanced 2P Galcon map (180-degree rotationally symmetric by default).

    When ``symmetric=True``, extra remainder planets modulo ``num_players`` are dropped so
    symmetric 1v1 maps always have strictly even planet counts with no central odd planet at
    ``(width / 2, height / 2)``.
    """
    rng = random.Random(seed)
    width = float(width)
    height = float(height)
    cx, cy = width * 0.5, height * 0.5
    min_gap = MIN_PLANET_GAP
    home_r = planet_radius_from_production(home_prod)

    # Place P1 and P2 homes symmetrically on an outer ellipse within legal map bounds
    h_min_x, _ = _axis_bounds(home_r, width)
    h_min_y, _ = _axis_bounds(home_r, height)
    rx_max = max(0.0, cx - h_min_x)
    ry_max = max(0.0, cy - h_min_y)
    min_home_center_dist = home_r + 0.5 * min_gap

    hx, hy = cx, cy
    placed_home = False
    for _ in range(64):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        rx = rx_max * rng.uniform(0.70, 0.90)
        ry = ry_max * rng.uniform(0.70, 0.90)
        dx = rx * math.cos(angle)
        dy = ry * math.sin(angle)
        d_center = math.hypot(dx, dy)
        if 1e-6 < d_center < min_home_center_dist:
            scale = min_home_center_dist / d_center
            cand_dx = dx * scale
            cand_dy = dy * scale
            if abs(cand_dx) <= rx_max + 1e-9 and abs(cand_dy) <= ry_max + 1e-9:
                dx, dy = cand_dx, cand_dy
                d_center = math.hypot(dx, dy)
        if d_center >= min_home_center_dist - 1e-9:
            hx = cx + max(-rx_max, min(rx_max, dx))
            hy = cy + max(-ry_max, min(ry_max, dy))
            placed_home = True
            break
    if not placed_home:
        if rx_max >= ry_max:
            hx, hy = cx + rx_max, cy
        else:
            hx, hy = cx, cy + ry_max

    home1 = Planet(id=1, owner=1, ships=float(home_ships), x=hx, y=hy, production=float(home_prod), radius=home_r)
    home2 = Planet(
        id=2,
        owner=2,
        ships=float(home_ships),
        x=width - hx,
        y=height - hy,
        production=float(home_prod),
        radius=home_r,
    )

    if symmetric:
        n_players = max(1, int(num_players))
        effective_planets = symmetric_planet_count(num_planets, n_players)
        # 2-player 180-degree rotational symmetry places planets strictly in mirror pairs with no center planet.
        if effective_planets % 2 == 1 and effective_planets - n_players >= 2:
            effective_planets -= n_players
        target_neutrals = max(0, effective_planets - 2)
        pairs = target_neutrals // 2
        pair_planets: List[Planet] = [home1, home2]

        for _ in range(pairs):
            prod = float(rng.randint(NEUTRAL_PROD_MIN, NEUTRAL_PROD_MAX))
            r = planet_radius_from_production(prod)
            ships = float(rng.randint(NEUTRAL_SHIPS_MIN, NEUTRAL_SHIPS_MAX))
            lo_x, hi_x = _axis_bounds(r, width)
            lo_y, hi_y = _axis_bounds(r, height)
            placed = False
            best_margin = -1e18
            best_xy: Optional[Tuple[float, float]] = None

            for _attempt in range(160):
                x = rng.uniform(lo_x, hi_x)
                y = rng.uniform(lo_y, hi_y)
                mx, my = width - x, height - y
                min_clear = math.hypot(x - mx, y - my) - (2.0 * r + min_gap)
                if min_clear > best_margin or min_clear >= 0.0:
                    for p in pair_planets:
                        c1 = math.hypot(x - p.x, y - p.y) - (r + p.radius + min_gap)
                        if c1 < min_clear:
                            min_clear = c1
                            if min_clear < best_margin and min_clear < 0.0:
                                break
                        c2 = math.hypot(mx - p.x, my - p.y) - (r + p.radius + min_gap)
                        if c2 < min_clear:
                            min_clear = c2
                            if min_clear < best_margin and min_clear < 0.0:
                                break
                    if min_clear >= 0.0:
                        pair_planets.append(Planet(id=0, owner=0, ships=ships, x=x, y=y, production=prod, radius=r))
                        pair_planets.append(Planet(id=0, owner=0, ships=ships, x=mx, y=my, production=prod, radius=r))
                        placed = True
                        break
                    if min_clear > best_margin:
                        best_margin = min_clear
                        best_xy = (x, y)

            if not placed and best_xy is not None:
                bx, by = best_xy
                trial = _clone_planets(pair_planets)
                trial.append(Planet(id=0, owner=0, ships=ships, x=bx, y=by, production=prod, radius=r))
                trial.append(
                    Planet(id=0, owner=0, ships=ships, x=width - bx, y=height - by, production=prod, radius=r)
                )
                if settle_planets(trial, width, height, min_gap=min_gap, symmetric=True, max_iters=30):
                    pair_planets = trial
                    placed = True
            if not placed:
                continue

        planets: List[Planet] = list(pair_planets)
    else:
        target_neutrals = max(2, int(num_planets) - 2)
        planets = [home1, home2]
        for _ in range(target_neutrals):
            prod = float(rng.randint(NEUTRAL_PROD_MIN, NEUTRAL_PROD_MAX))
            r = planet_radius_from_production(prod)
            ships = float(rng.randint(NEUTRAL_SHIPS_MIN, NEUTRAL_SHIPS_MAX))
            lo_x, hi_x = _axis_bounds(r, width)
            lo_y, hi_y = _axis_bounds(r, height)
            placed = False
            best_margin = -1e18
            best_xy = None

            for _attempt in range(160):
                x = rng.uniform(lo_x, hi_x)
                y = rng.uniform(lo_y, hi_y)
                min_clear = 1e18
                for p in planets:
                    c1 = math.hypot(x - p.x, y - p.y) - (r + p.radius + min_gap)
                    if c1 < min_clear:
                        min_clear = c1
                        if min_clear < best_margin and min_clear < 0.0:
                            break
                if min_clear >= 0.0:
                    planets.append(Planet(id=0, owner=0, ships=ships, x=x, y=y, production=prod, radius=r))
                    placed = True
                    break
                if min_clear > best_margin:
                    best_margin = min_clear
                    best_xy = (x, y)

            if not placed and best_xy is not None:
                bx, by = best_xy
                trial = _clone_planets(planets)
                trial.append(Planet(id=0, owner=0, ships=ships, x=bx, y=by, production=prod, radius=r))
                if settle_planets(trial, width, height, min_gap=min_gap, symmetric=False, max_iters=30):
                    planets = trial
                    placed = True
            if not placed:
                continue

    settle_planets(planets, width, height, min_gap=min_gap, symmetric=symmetric, max_iters=30)
    while len(planets) > 2 and not _planets_clearance_ok(planets, min_gap=min_gap):
        if symmetric:
            if len(planets) >= 4:
                planets.pop()
                planets.pop()
            else:
                break
        else:
            planets.pop()
        settle_planets(planets, width, height, min_gap=min_gap, symmetric=symmetric, max_iters=30)

    for idx, p in enumerate(planets, start=1):
        p.id = idx

    return planets


class GalconSim:
    """Fast, cloneable Galcon simulation engine with send & mid-flight redirect support."""

    def __init__(
        self,
        planets: Sequence[Planet],
        fleets: Sequence[Fleet] = (),
        *,
        width: float = DEFAULT_MAP_WIDTH,
        height: float = DEFAULT_MAP_HEIGHT,
        fleet_speed: float = DEFAULT_FLEET_SPEED,
        tick_dt: float = DEFAULT_TICK_DT,
        max_ticks: int = DEFAULT_MAX_TICKS,
        tick: int = 0,
        next_fleet_id: int = 1000,
        allow_redirect: bool = True,
        physics_substeps: int = DEFAULT_PHYSICS_SUBSTEPS,
        use_cpp: Optional[bool] = None,
    ) -> None:
        self.planets: Dict[int, Planet] = {
            p.id: Planet(p.id, p.owner, float(p.ships), float(p.x), float(p.y), float(p.production), float(p.radius))
            for p in planets
        }
        self.fleets: Dict[int, Fleet] = {}
        for f in fleets:
            fx = float(f.x)
            fy = float(f.y)
            if f.units:
                units = [[float(u[0]), float(u[1])] for u in f.units]
                num_units = len(units)
            else:
                num_units = max(1, int(f.num_units))
                units = _materialize_fleet_units(fx, fy, num_units)
            if f.unit_headings is not None and len(f.unit_headings) == num_units:
                unit_headings = [[float(h[0]), float(h[1])] for h in f.unit_headings]
            else:
                unit_headings = _materialize_unit_headings(units, self.planets.get(f.target))
            self.fleets[f.id] = Fleet(
                f.id,
                f.owner,
                float(f.ships),
                fx,
                fy,
                f.source,
                f.target,
                float(f.radius),
                num_units,
                units,
                unit_headings,
            )
        self.width = float(width)
        self.height = float(height)
        self.fleet_speed = float(fleet_speed)
        self.tick_dt = float(tick_dt)
        self.max_ticks = int(max_ticks)
        self.tick = int(tick)
        self.next_fleet_id = int(next_fleet_id)
        self.allow_redirect = bool(allow_redirect)
        self.physics_substeps = max(1, int(physics_substeps))
        self.use_cpp = bool(HAS_FAST_SIM_CPP) if use_cpp is None else bool(use_cpp and HAS_FAST_SIM_CPP)
        self.done = False
        self._winner = 0

    @classmethod
    def create(
        cls,
        *,
        seed: int = 42,
        num_planets: int = DEFAULT_NUM_PLANETS,
        num_players: int = DEFAULT_NUM_PLAYERS,
        width: float = DEFAULT_MAP_WIDTH,
        height: float = DEFAULT_MAP_HEIGHT,
        fleet_speed: float = DEFAULT_FLEET_SPEED,
        tick_dt: float = DEFAULT_TICK_DT,
        max_ticks: int = DEFAULT_MAX_TICKS,
        symmetric: bool = True,
        allow_redirect: bool = True,
        physics_substeps: int = DEFAULT_PHYSICS_SUBSTEPS,
        use_cpp: Optional[bool] = None,
    ) -> "GalconSim":
        planets = generate_galcon_map(
            seed=seed,
            num_planets=num_planets,
            num_players=num_players,
            width=width,
            height=height,
            symmetric=symmetric,
        )
        return cls(
            planets,
            width=width,
            height=height,
            fleet_speed=fleet_speed,
            tick_dt=tick_dt,
            max_ticks=max_ticks,
            allow_redirect=allow_redirect,
            physics_substeps=physics_substeps,
            use_cpp=use_cpp,
        )

    def clone(self, use_cpp: Optional[bool] = None) -> "GalconSim":
        sim = GalconSim(
            list(self.planets.values()),
            list(self.fleets.values()),
            width=self.width,
            height=self.height,
            fleet_speed=self.fleet_speed,
            tick_dt=self.tick_dt,
            max_ticks=self.max_ticks,
            tick=self.tick,
            next_fleet_id=self.next_fleet_id,
            allow_redirect=self.allow_redirect,
            physics_substeps=self.physics_substeps,
            use_cpp=self.use_cpp if use_cpp is None else use_cpp,
        )
        sim.done = self.done
        sim._winner = self._winner
        return sim

    def item_count(self) -> int:
        """Current world item-table occupancy: users + planets + fleets + physical ship units."""
        return (
            BASE_ITEM_USERS
            + len(self.planets)
            + len(self.fleets)
            + sum(len(f.units) if f.units is not None else max(1, f.num_units) for f in self.fleets.values())
        )

    def eta_ticks(self, x1: float, y1: float, dst: Planet) -> int:
        """Exact unobstructed arrival ticks from ``(x1, y1)`` to ``dst`` touch-to-land boundary."""
        dx = dst.x - x1
        dy = dst.y - y1
        dist = max(0.0, math.sqrt(dx * dx + dy * dy) - dst.radius - SHIP_RADIUS)
        step_dist = self.fleet_speed * self.tick_dt
        return max(1, int(math.ceil((dist - ARRIVAL_EPS) / max(1e-6, step_dist))))

    def planet_eta_ticks(self, src: Planet, dst: Planet) -> int:
        """Exact unobstructed arrival ticks for a fleet launched from ``src`` edge to ``dst`` edge."""
        dx = dst.x - src.x
        dy = dst.y - src.y
        dist = max(0.0, math.sqrt(dx * dx + dy * dy) - src.radius - dst.radius - 2.0 * SHIP_RADIUS)
        step_dist = self.fleet_speed * self.tick_dt
        return max(1, int(math.ceil((dist - ARRIVAL_EPS) / max(1e-6, step_dist))))

    def _step_physics_python(self) -> None:
        """Advance production, physical ship separation, movement, planet pushout, and gradual combat."""
        planet_ids = sorted(self.planets.keys())
        planets = [self.planets[pid] for pid in planet_ids]
        n_planets = len(planets)
        if n_planets <= 0:
            return

        substeps = max(1, int(self.physics_substeps))
        sub_dt = self.tick_dt / float(substeps)
        sub_step_dist = self.fleet_speed * sub_dt
        ship_radius = SHIP_RADIUS
        min_sep = SHIP_SEP_DIST
        min_sep2 = min_sep * min_sep
        sep_k = SHIP_SEP_STIFFNESS
        half_k = 0.5 * sep_k
        max_sep_push = SHIP_SEP_MAX_DISP
        cap2 = max_sep_push * max_sep_push
        esc_eps = ANTIPODE_ESCAPE_TANGENT
        antipode_dot = ANTIPODE_DOT_THRESHOLD
        eps_dist2 = _EPS_DIST2
        golden_angle = _GOLDEN_ANGLE
        tie_eps = TIE_EPS

        _sqrt = math.sqrt
        _cos = math.cos
        _sin = math.sin

        turn_rate = SHIP_TURN_RATE
        min_speed_factor = SHIP_TURN_MIN_SPEED_FACTOR
        max_turn = turn_rate * sub_dt
        instant_turn = turn_rate <= 0.0 or max_turn >= math.pi
        if instant_turn:
            cos_max = -2.0
            sin_max = 0.0
        else:
            cos_max = _cos(max_turn)
            sin_max = _sin(max_turn)

        p_x = [p.x for p in planets]
        p_y = [p.y for p in planets]
        p_min_r = [p.radius + ship_radius for p in planets]
        p_min_r2 = [mr * mr for mr in p_min_r]
        p_land_r = [mr + ARRIVAL_EPS for mr in p_min_r]
        p_land_r2_guard = [(lr + 1e-6) * (lr + 1e-6) for lr in p_land_r]
        p_prod_sub = [(p.production / PRODUCTION_DIVISOR) * sub_dt for p in planets]
        max_mr = max(p_min_r)

        fleets = list(self.fleets.values())
        n_fleets = len(fleets)

        if n_fleets == 0:
            for _s in range(substeps):
                for i in range(n_planets):
                    p = planets[i]
                    if p.owner > 0:
                        p.ships += p_prod_sub[i]
            return

        id_to_idx = {p.id: i for i, p in enumerate(planets)}
        fl_ux: List[List[float]] = []
        fl_uy: List[List[float]] = []
        fl_uhx: List[List[float]] = []
        fl_uhy: List[List[float]] = []
        fl_upx: List[List[float]] = []
        fl_upy: List[List[float]] = []
        fl_uage: List[List[int]] = []
        fl_uuid: List[List[int]] = []
        fl_utr: List[List[float]] = []
        fl_utc: List[List[float]] = []
        fl_alive: List[List[bool]] = []
        fl_alive_cnt: List[int] = []
        fl_tgt_idx: List[int] = []
        fl_spu: List[float] = []
        fl_owner: List[int] = []

        next_uid = int(getattr(self, "_next_uid", 0))
        for fl in fleets:
            if not fl.units:
                fl.units = _materialize_fleet_units(fl.x, fl.y, fl.num_units)
            m = len(fl.units)
            fl.num_units = m
            if fl.unit_headings is None or len(fl.unit_headings) != m:
                fl.unit_headings = _materialize_unit_headings(fl.units, self.planets.get(fl.target))
            ux_l = [u[0] for u in fl.units]
            uy_l = [u[1] for u in fl.units]
            fl_ux.append(ux_l)
            fl_uy.append(uy_l)
            fl_uhx.append([h[0] for h in fl.unit_headings])
            fl_uhy.append([h[1] for h in fl.unit_headings])
            f_px = getattr(fl, "_u_px", None)
            f_py = getattr(fl, "_u_py", None)
            f_age = getattr(fl, "_u_age", None)
            f_uid = getattr(fl, "_u_uid", None)
            f_tr = getattr(fl, "_u_tr", None)
            f_tc = getattr(fl, "_u_tc", None)
            if (
                f_px is not None
                and f_py is not None
                and f_age is not None
                and f_uid is not None
                and f_tr is not None
                and f_tc is not None
                and len(f_px) == m
            ):
                fl_upx.append(list(f_px))
                fl_upy.append(list(f_py))
                fl_uage.append(list(f_age))
                fl_uuid.append(list(f_uid))
                fl_utr.append(list(f_tr))
                fl_utc.append(list(f_tc))
            else:
                fl_upx.append(list(ux_l))
                fl_upy.append(list(uy_l))
                fl_uage.append([0] * m)
                uids = list(range(next_uid, next_uid + m))
                next_uid += m
                fl_uuid.append(uids)
                fl_utr.append([turn_rate] * m)
                fl_utc.append([0.0] * m)
            fl_alive.append([True] * m)
            fl_alive_cnt.append(m)
            fl_tgt_idx.append(id_to_idx.get(fl.target, -1))
            fl_spu.append(fl.ships_per_unit)
            fl_owner.append(fl.owner)
        self._next_uid = next_uid

        owner_tgt_fleets: Dict[Tuple[int, int], List[int]] = {}
        for fi, ti in enumerate(fl_tgt_idx):
            if 0 <= ti < n_planets:
                owner_tgt_fleets.setdefault((fl_owner[fi], ti), []).append(fi)

        for _s in range(substeps):
            # 1. Production on owned planets
            for i in range(n_planets):
                p = planets[i]
                if p.owner > 0:
                    p.ships += p_prod_sub[i]

            # 2. Same-owner same-target ship-ship circle separation (Gauss-Seidel + per-ship substep cap)
            if sep_k > 0.0:
                for grp in owner_tgt_fleets.values():
                    n_grp = len(grp)
                    if n_grp == 0:
                        continue
                    if n_grp == 1:
                        fi = grp[0]
                        alive_c = fl_alive_cnt[fi]
                        if alive_c <= 1:
                            continue
                        ux = fl_ux[fi]
                        uy = fl_uy[fi]
                        m = len(ux)
                        all_alive = alive_c == m
                        if max_sep_push > 0.0:
                            ox = ux[:]
                            oy = uy[:]
                        if all_alive:
                            for a in range(m):
                                ax = ux[a]
                                ay = uy[a]
                                for b in range(a + 1, m):
                                    dx = ux[b] - ax
                                    dy = uy[b] - ay
                                    d2 = dx * dx + dy * dy
                                    if d2 < min_sep2:
                                        if d2 > eps_dist2:
                                            dist = _sqrt(d2)
                                            nx = dx / dist
                                            ny = dy / dist
                                        else:
                                            dist = 0.0
                                            ang = golden_angle * float(a * 31 + b + 1)
                                            nx = _cos(ang)
                                            ny = _sin(ang)
                                        push = half_k * (min_sep - dist)
                                        px_d = nx * push
                                        py_d = ny * push
                                        ax -= px_d
                                        ay -= py_d
                                        ux[b] += px_d
                                        uy[b] += py_d
                                ux[a] = ax
                                uy[a] = ay
                            if max_sep_push > 0.0:
                                for u in range(m):
                                    dx = ux[u] - ox[u]
                                    dy = uy[u] - oy[u]
                                    d2 = dx * dx + dy * dy
                                    if d2 > cap2:
                                        sc = max_sep_push / _sqrt(d2)
                                        ux[u] = ox[u] + dx * sc
                                        uy[u] = oy[u] + dy * sc
                        else:
                            alive = fl_alive[fi]
                            for a in range(m):
                                if not alive[a]:
                                    continue
                                ax = ux[a]
                                ay = uy[a]
                                for b in range(a + 1, m):
                                    if not alive[b]:
                                        continue
                                    dx = ux[b] - ax
                                    dy = uy[b] - ay
                                    d2 = dx * dx + dy * dy
                                    if d2 < min_sep2:
                                        if d2 > eps_dist2:
                                            dist = _sqrt(d2)
                                            nx = dx / dist
                                            ny = dy / dist
                                        else:
                                            dist = 0.0
                                            ang = golden_angle * float(a * 31 + b + 1)
                                            nx = _cos(ang)
                                            ny = _sin(ang)
                                        push = half_k * (min_sep - dist)
                                        px_d = nx * push
                                        py_d = ny * push
                                        ax -= px_d
                                        ay -= py_d
                                        ux[b] += px_d
                                        uy[b] += py_d
                                ux[a] = ax
                                uy[a] = ay
                            if max_sep_push > 0.0:
                                for u in range(m):
                                    if not alive[u]:
                                        continue
                                    dx = ux[u] - ox[u]
                                    dy = uy[u] - oy[u]
                                    d2 = dx * dx + dy * dy
                                    if d2 > cap2:
                                        sc = max_sep_push / _sqrt(d2)
                                        ux[u] = ox[u] + dx * sc
                                        uy[u] = oy[u] + dy * sc
                    else:
                        gx: List[float] = []
                        gy: List[float] = []
                        g_ref: List[Tuple[int, int]] = []
                        for fi in grp:
                            alive_c = fl_alive_cnt[fi]
                            if alive_c <= 0:
                                continue
                            ux = fl_ux[fi]
                            uy = fl_uy[fi]
                            m_f = len(ux)
                            if alive_c == m_f:
                                for u in range(m_f):
                                    gx.append(ux[u])
                                    gy.append(uy[u])
                                    g_ref.append((fi, u))
                            else:
                                alive = fl_alive[fi]
                                for u in range(m_f):
                                    if alive[u]:
                                        gx.append(ux[u])
                                        gy.append(uy[u])
                                        g_ref.append((fi, u))
                        m_g = len(gx)
                        if m_g <= 1:
                            continue
                        if max_sep_push > 0.0:
                            ox = gx[:]
                            oy = gy[:]
                        for a in range(m_g):
                            ax = gx[a]
                            ay = gy[a]
                            for b in range(a + 1, m_g):
                                dx = gx[b] - ax
                                dy = gy[b] - ay
                                d2 = dx * dx + dy * dy
                                if d2 < min_sep2:
                                    if d2 > eps_dist2:
                                        dist = _sqrt(d2)
                                        nx = dx / dist
                                        ny = dy / dist
                                    else:
                                        dist = 0.0
                                        ang = golden_angle * float(a * 31 + b + 1)
                                        nx = _cos(ang)
                                        ny = _sin(ang)
                                    push = half_k * (min_sep - dist)
                                    px_d = nx * push
                                    py_d = ny * push
                                    ax -= px_d
                                    ay -= py_d
                                    gx[b] += px_d
                                    gy[b] += py_d
                            gx[a] = ax
                            gy[a] = ay
                        if max_sep_push > 0.0:
                            for idx_g in range(m_g):
                                dx = gx[idx_g] - ox[idx_g]
                                dy = gy[idx_g] - oy[idx_g]
                                d2 = dx * dx + dy * dy
                                if d2 > cap2:
                                    sc = max_sep_push / _sqrt(d2)
                                    gx[idx_g] = ox[idx_g] + dx * sc
                                    gy[idx_g] = oy[idx_g] + dy * sc
                        for idx_g in range(m_g):
                            fi_r, u_r = g_ref[idx_g]
                            fl_ux[fi_r][u_r] = gx[idx_g]
                            fl_uy[fi_r][u_r] = gy[idx_g]

            # 3. Move toward target + 4. Planet collider pushout + 5a. Touch-to-land
            arrivals: Dict[int, Dict[int, float]] = {}
            for fi in range(n_fleets):
                alive_c = fl_alive_cnt[fi]
                if alive_c <= 0:
                    continue
                tgt = fl_tgt_idx[fi]
                if tgt < 0:
                    fl_alive_cnt[fi] = 0
                    continue
                dst = planets[tgt]
                tx = p_x[tgt]
                ty = p_y[tgt]
                ux = fl_ux[fi]
                uy = fl_uy[fi]
                uhx = fl_uhx[fi]
                uhy = fl_uhy[fi]
                upx = fl_upx[fi]
                upy = fl_upy[fi]
                uage = fl_uage[fi]
                uuid = fl_uuid[fi]
                utr = fl_utr[fi]
                utc = fl_utc[fi]
                alive = fl_alive[fi]
                m = len(ux)
                all_alive = alive_c == m

                # 3. G2 stuck-unit jitter + move toward target with heading turn limit + track exact fleet AABB
                min_x = 1e300
                max_x = -1e300
                min_y = 1e300
                max_y = -1e300
                stuck_thresh = sub_step_dist / STUCK_DIST_DIVISOR
                for u in range(m):
                    if not all_alive and not alive[u]:
                        continue
                    x = ux[u]
                    y = uy[u]
                    age_u = uage[u]
                    if age_u > 0:
                        if (abs(x - upx[u]) + abs(y - upy[u]) < stuck_thresh) and ((uuid[u] + age_u) % STUCK_JITTER_MOD == 0):
                            hseed = ((uuid[u] * 1000003) & 0xFFFFFFFF) ^ ((age_u * 2654435761) & 0xFFFFFFFF)
                            x += (_spawn_hash_u01((hseed + 1) & 0xFFFFFFFF) * 2.0 - 1.0) * sub_step_dist
                            y += (_spawn_hash_u01((hseed + 2) & 0xFFFFFFFF) * 2.0 - 1.0) * sub_step_dist
                    upx[u] = x
                    upy[u] = y
                    dx = tx - x
                    dy = ty - y
                    dist = _sqrt(dx * dx + dy * dy)
                    if dist > 0.0:
                        des_x = dx / dist
                        des_y = dy / dist
                        hx = uhx[u]
                        hy = uhy[u]
                        dot = hx * des_x + hy * des_y
                        u_tr = utr[u]
                        if u_tr == turn_rate:
                            u_cos = cos_max
                            u_sin = sin_max
                            u_instant = instant_turn
                        else:
                            u_max_turn = u_tr * sub_dt
                            u_instant = u_tr <= 0.0 or u_max_turn >= math.pi
                            if u_instant:
                                u_cos = -2.0
                                u_sin = 0.0
                            else:
                                u_cos = _cos(u_max_turn)
                                u_sin = _sin(u_max_turn)
                        if u_instant or dot >= u_cos:
                            hx = des_x
                            hy = des_y
                            utc[u] = 0.0
                            if dist <= sub_step_dist:
                                x = tx
                                y = ty
                            else:
                                x += hx * sub_step_dist
                                y += hy * sub_step_dist
                        else:
                            cross = hx * des_y - hy * des_x
                            turn_sign = 1.0 if cross >= 0.0 else -1.0
                            s_rot = turn_sign * u_sin
                            nhx = hx * u_cos - hy * s_rot
                            nhy = hx * s_rot + hy * u_cos
                            hn = _sqrt(nhx * nhx + nhy * nhy)
                            if hn > 0.0:
                                hx = nhx / hn
                                hy = nhy / hn
                            else:
                                hx = des_x
                                hy = des_y
                            tc = utc[u] + sub_dt
                            utc[u] = tc
                            if tc > 1.0:
                                utr[u] = u_tr + sub_dt * TURN_ESCALATE_PER_SEC
                            align = hx * des_x + hy * des_y
                            speed_scale = min_speed_factor + (1.0 - min_speed_factor) * (
                                align if align > 0.0 else 0.0
                            )
                            step = sub_step_dist * speed_scale
                            x += hx * step
                            y += hy * step
                        uhx[u] = hx
                        uhy[u] = hy
                        ux[u] = x
                        uy[u] = y
                    uage[u] = age_u + 1
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y

                # Zero-margin AABB broadphase for planet obstacle colliders
                any_exact_hit = False
                for pi in range(n_planets):
                    if pi == tgt:
                        continue
                    mr = p_min_r[pi]
                    px_i = p_x[pi]
                    py_i = p_y[pi]
                    if px_i + mr >= min_x and px_i - mr <= max_x and py_i + mr >= min_y and py_i - mr <= max_y:
                        any_exact_hit = True
                        break

                # 4. Planet collider radial pushout (padded candidate list preserves exactness)
                if any_exact_hit:
                    near_planets: List[int] = []
                    pad_min_x = min_x - max_mr
                    pad_max_x = max_x + max_mr
                    pad_min_y = min_y - max_mr
                    pad_max_y = max_y + max_mr
                    for pi in range(n_planets):
                        if pi == tgt:
                            continue
                        mr = p_min_r[pi]
                        px_i = p_x[pi]
                        py_i = p_y[pi]
                        if (
                            px_i + mr >= pad_min_x
                            and px_i - mr <= pad_max_x
                            and py_i + mr >= pad_min_y
                            and py_i - mr <= pad_max_y
                        ):
                            near_planets.append(pi)
                    for u in range(m):
                        if not all_alive and not alive[u]:
                            continue
                        x = ux[u]
                        y = uy[u]
                        for pi in near_planets:
                            px_i = p_x[pi]
                            py_i = p_y[pi]
                            dx = x - px_i
                            dy = y - py_i
                            d2 = dx * dx + dy * dy
                            if d2 < p_min_r2[pi]:
                                mr = p_min_r[pi]
                                if d2 > eps_dist2:
                                    dist = _sqrt(d2)
                                    nx = dx / dist
                                    ny = dy / dist
                                    if esc_eps > 0.0:
                                        tdx = tx - px_i
                                        tdy = ty - py_i
                                        td = _sqrt(tdx * tdx + tdy * tdy)
                                        if td > 1e-6:
                                            tnx = tdx / td
                                            tny = tdy / td
                                            if nx * tnx + ny * tny < antipode_dot:
                                                cross = tnx * ny - tny * nx
                                                if abs(cross) > 1e-9:
                                                    sign = 1.0 if cross > 0.0 else -1.0
                                                else:
                                                    sign = 1.0 if (u & 1) == 0 else -1.0
                                                nx += (-tny) * sign * esc_eps
                                                ny += tnx * sign * esc_eps
                                                nn = _sqrt(nx * nx + ny * ny)
                                                nx /= nn
                                                ny /= nn
                                    x = px_i + nx * mr
                                    y = py_i + ny * mr
                                else:
                                    tdx = tx - px_i
                                    tdy = ty - py_i
                                    td2 = tdx * tdx + tdy * tdy
                                    if td2 > eps_dist2:
                                        td = _sqrt(td2)
                                        x = px_i + (tdx / td) * mr
                                        y = py_i + (tdy / td) * mr
                                    else:
                                        x = px_i + mr
                                        y = py_i
                        ux[u] = x
                        uy[u] = y

                # 5a. Touch-to-land check
                lr = p_land_r[tgt]
                lr2_g = p_land_r2_guard[tgt]
                spu = fl_spu[fi]
                own = fl_owner[fi]
                alive_left = alive_c
                for u in range(m):
                    if not all_alive and not alive[u]:
                        continue
                    dx = tx - ux[u]
                    dy = ty - uy[u]
                    d2 = dx * dx + dy * dy
                    if d2 <= lr2_g and _sqrt(d2) <= lr:
                        alive[u] = False
                        alive_left -= 1
                        forces = arrivals.get(tgt)
                        if forces is None:
                            forces = {}
                            if dst.ships > 0.0:
                                forces[dst.owner] = dst.ships
                            arrivals[tgt] = forces
                        forces[own] = forces.get(own, 0.0) + spu
                fl_alive_cnt[fi] = alive_left

            # 5b. Resolve gradual combat on planets with arrivals
            for pi, forces in arrivals.items():
                p = planets[pi]
                n_forces = len(forces)
                if n_forces == 1:
                    only_o, only_v = next(iter(forces.items()))
                    p.owner = only_o
                    p.ships = only_v
                elif n_forces >= 2:
                    top_v = -1.0
                    sec_v = -1.0
                    top_o = p.owner
                    for o, v in forces.items():
                        if v > top_v:
                            sec_v = top_v
                            top_v = v
                            top_o = o
                        elif v > sec_v:
                            sec_v = v
                    if abs(top_v - sec_v) <= tie_eps:
                        p.ships = 0.0
                    else:
                        p.owner = top_o
                        p.ships = top_v - sec_v

        # 6. Compact surviving units & update fleet summaries
        for fi in range(n_fleets):
            fl = fleets[fi]
            alive_c = fl_alive_cnt[fi]
            if alive_c <= 0:
                del self.fleets[fl.id]
                continue
            ux = fl_ux[fi]
            uy = fl_uy[fi]
            uhx = fl_uhx[fi]
            uhy = fl_uhy[fi]
            upx = fl_upx[fi]
            upy = fl_upy[fi]
            uage = fl_uage[fi]
            uuid = fl_uuid[fi]
            utr = fl_utr[fi]
            utc = fl_utc[fi]
            m = len(ux)
            if alive_c == m:
                new_ux = ux
                new_uy = uy
                new_uhx = uhx
                new_uhy = uhy
                new_upx = upx
                new_upy = upy
                new_uage = uage
                new_uuid = uuid
                new_utr = utr
                new_utc = utc
            else:
                alive = fl_alive[fi]
                new_ux = [ux[u] for u in range(m) if alive[u]]
                new_uy = [uy[u] for u in range(m) if alive[u]]
                new_uhx = [uhx[u] for u in range(m) if alive[u]]
                new_uhy = [uhy[u] for u in range(m) if alive[u]]
                new_upx = [upx[u] for u in range(m) if alive[u]]
                new_upy = [upy[u] for u in range(m) if alive[u]]
                new_uage = [uage[u] for u in range(m) if alive[u]]
                new_uuid = [uuid[u] for u in range(m) if alive[u]]
                new_utr = [utr[u] for u in range(m) if alive[u]]
                new_utc = [utc[u] for u in range(m) if alive[u]]
            kept = len(new_ux)
            sum_x = 0.0
            sum_y = 0.0
            min_x = 1e300
            max_x = -1e300
            min_y = 1e300
            max_y = -1e300
            for u in range(kept):
                xu = new_ux[u]
                yu = new_uy[u]
                sum_x += xu
                sum_y += yu
                if xu < min_x:
                    min_x = xu
                if xu > max_x:
                    max_x = xu
                if yu < min_y:
                    min_y = yu
                if yu > max_y:
                    max_y = yu
            fl.units = [[new_ux[u], new_uy[u]] for u in range(kept)]
            fl.unit_headings = [[new_uhx[u], new_uhy[u]] for u in range(kept)]
            if hasattr(fl, "__dict__"):
                fl._u_px = new_upx
                fl._u_py = new_upy
                fl._u_age = new_uage
                fl._u_uid = new_uuid
                fl._u_tr = new_utr
                fl._u_tc = new_utc
            fl.num_units = kept
            fl.ships = float(kept) * fl_spu[fi]
            fl.x = sum_x / float(kept)
            fl.y = sum_y / float(kept)
            fl.radius = (((max_x - min_x) + (max_y - min_y)) / 5.0) if kept > 1 else 0.0

    def _spawn_and_settle_flock(
        self,
        fid: int,
        parts: int,
        sx: float,
        sy: float,
        dst: Planet,
        hx0: float,
        hy0: float,
    ) -> Tuple[List[List[float]], List[List[float]], float, float, float]:
        """Spawn ``parts > 1`` units in a deterministic 12u disc and rapidly settle overlaps & planet clearance."""
        _sqrt = math.sqrt
        _cos = math.cos
        _sin = math.sin
        two_pi = 2.0 * math.pi
        seed_base = (fid * 1000003) & 0xFFFFFFFF
        ux = [0.0] * parts
        uy = [0.0] * parts
        spawn_disc_r = SPAWN_AREA_RADIUS * _sqrt(float(parts))
        for j in range(parts):
            u1 = _spawn_hash_u01((seed_base + j * 2 + 1) & 0xFFFFFFFF)
            u2 = _spawn_hash_u01((seed_base + j * 2 + 2) & 0xFFFFFFFF)
            r = spawn_disc_r * _sqrt(u1)
            ang = two_pi * u2
            ux[j] = sx + r * _cos(ang)
            uy[j] = sy + r * _sin(ang)

        tx = dst.x
        ty = dst.y
        ship_radius = SHIP_RADIUS
        min_sep = SHIP_SEP_DIST
        min_sep2 = min_sep * min_sep
        sep_k = SHIP_SEP_STIFFNESS
        half_k = 0.5 * sep_k
        max_sep_push = SHIP_SEP_MAX_DISP
        cap2 = max_sep_push * max_sep_push
        eps_dist2 = _EPS_DIST2
        golden_angle = _GOLDEN_ANGLE

        non_target_planets: List[Tuple[float, float, float, float]] = []
        for pid_p in sorted(self.planets.keys()):
            if pid_p == dst.id:
                continue
            pl = self.planets[pid_p]
            mr = pl.radius + ship_radius
            non_target_planets.append((pl.x, pl.y, mr, mr * mr))

        for _it in range(SPAWN_SETTLE_ITERS):
            max_ov = 0.0
            if sep_k > 0.0:
                if max_sep_push > 0.0:
                    ox = ux[:]
                    oy = uy[:]
                for a in range(parts):
                    ax = ux[a]
                    ay = uy[a]
                    for b in range(a + 1, parts):
                        ddx = ux[b] - ax
                        ddy = uy[b] - ay
                        d2 = ddx * ddx + ddy * ddy
                        if d2 < min_sep2:
                            if d2 > eps_dist2:
                                dist = _sqrt(d2)
                                nx = ddx / dist
                                ny = ddy / dist
                            else:
                                dist = 0.0
                                ang = golden_angle * float(a * 31 + b + 1)
                                nx = _cos(ang)
                                ny = _sin(ang)
                            ov = min_sep - dist
                            if ov > max_ov:
                                max_ov = ov
                            push = half_k * ov
                            px_d = nx * push
                            py_d = ny * push
                            ax -= px_d
                            ay -= py_d
                            ux[b] += px_d
                            uy[b] += py_d
                    ux[a] = ax
                    uy[a] = ay
                if max_sep_push > 0.0:
                    for u in range(parts):
                        ddx = ux[u] - ox[u]
                        ddy = uy[u] - oy[u]
                        d2 = ddx * ddx + ddy * ddy
                        if d2 > cap2:
                            sc = max_sep_push / _sqrt(d2)
                            ux[u] = ox[u] + ddx * sc
                            uy[u] = oy[u] + ddy * sc

            max_pp = 0.0
            for u in range(parts):
                x = ux[u]
                y = uy[u]
                for px_i, py_i, mr, mr2 in non_target_planets:
                    ddx = x - px_i
                    ddy = y - py_i
                    d2 = ddx * ddx + ddy * ddy
                    if d2 < mr2:
                        if d2 > eps_dist2:
                            dist = _sqrt(d2)
                            pp = mr - dist
                            if pp > max_pp:
                                max_pp = pp
                            x = px_i + (ddx / dist) * mr
                            y = py_i + (ddy / dist) * mr
                        else:
                            if mr > max_pp:
                                max_pp = mr
                            tdx = tx - px_i
                            tdy = ty - py_i
                            td2 = tdx * tdx + tdy * tdy
                            if td2 > eps_dist2:
                                td = _sqrt(td2)
                                x = px_i + (tdx / td) * mr
                                y = py_i + (tdy / td) * mr
                            else:
                                x = px_i + mr
                                y = py_i
                ux[u] = x
                uy[u] = y

            if max_ov <= SPAWN_SETTLE_TOL and max_pp <= SPAWN_SETTLE_TOL:
                break

        sum_x = 0.0
        sum_y = 0.0
        units: List[List[float]] = []
        unit_headings: List[List[float]] = []
        for j in range(parts):
            xj = ux[j]
            yj = uy[j]
            units.append([xj, yj])
            sum_x += xj
            sum_y += yj
            hdx = tx - xj
            hdy = ty - yj
            hd2 = hdx * hdx + hdy * hdy
            if hd2 > eps_dist2:
                hd = _sqrt(hd2)
                unit_headings.append([hdx / hd, hdy / hd])
            else:
                unit_headings.append([hx0, hy0])
        cx = sum_x / float(parts)
        cy = sum_y / float(parts)
        min_x = min(ux)
        max_x = max(ux)
        min_y = min(uy)
        max_y = max(uy)
        fl_radius = (((max_x - min_x) + (max_y - min_y)) / 5.0) if parts > 1 else 0.0
        return units, unit_headings, cx, cy, fl_radius

    def step(self, orders_by_player: Dict[int, Sequence[Order]]) -> None:
        """Advance the Galcon world by one tick (orders -> 20 Hz substeps of production, movement, collisions, combat)."""
        if self.done:
            return

        # 1. Execute player orders simultaneously (deterministic by player ID)
        for pid in sorted(orders_by_player.keys()):
            for order in orders_by_player[pid]:
                if isinstance(order, SendOrder):
                    src = self.planets.get(order.src_id)
                    dst = self.planets.get(order.dst_id)
                    if src is None or dst is None or src.id == dst.id or src.owner != pid:
                        continue
                    if src.ships < 1.0 - TIE_EPS:
                        continue
                    pct = max(1, min(100, int(order.percent)))
                    # Exact Galcon 2 send formula (verified via send_rounding_edges + prod_and_rounding):
                    # min(src.ships, floor(src.ships * pct / 100 + 0.5))
                    send_ships = min(src.ships, float(int(math.floor(src.ships * (pct / 100.0) + 0.5))))
                    if send_ships < 1.0 - TIE_EPS:
                        continue
                    total_planet_ships = sum(max(0.0, p.ships) for p in self.planets.values())
                    parts = physical_ship_count(send_ships, total_planet_ships, self.item_count())
                    src.ships = max(0.0, src.ships - send_ships)
                    dx = dst.x - src.x
                    dy = dst.y - src.y
                    raw_d = math.sqrt(dx * dx + dy * dy)
                    d = raw_d if raw_d > 1e-6 else 1e-6
                    spawn_r = src.radius + SHIP_RADIUS
                    sx = src.x + (dx / d) * spawn_r
                    sy = src.y + (dy / d) * spawn_r
                    hx0 = (dx / d) if raw_d > 1e-6 else 1.0
                    hy0 = (dy / d) if raw_d > 1e-6 else 0.0
                    fid = self.next_fleet_id
                    self.next_fleet_id += 1
                    if parts == 1:
                        units = [[sx, sy]]
                        unit_headings = [[hx0, hy0]]
                        cx, cy, fl_radius = sx, sy, 0.0
                    else:
                        units, unit_headings, cx, cy, fl_radius = self._spawn_and_settle_flock(
                            fid, parts, sx, sy, dst, hx0, hy0
                        )
                    self.fleets[fid] = Fleet(
                        id=fid,
                        owner=pid,
                        ships=send_ships,
                        x=cx,
                        y=cy,
                        source=src.id,
                        target=dst.id,
                        radius=fl_radius,
                        num_units=parts,
                        units=units,
                        unit_headings=unit_headings,
                    )
                elif isinstance(order, RedirOrder) and self.allow_redirect:
                    fl = self.fleets.get(order.fleet_id)
                    dst = self.planets.get(order.dst_id)
                    if fl is not None and dst is not None and fl.owner == pid and fl.target != dst.id:
                        fl.target = dst.id

        # 2. Advance physical ship & planet dynamics across physics_substeps
        if self.use_cpp and HAS_FAST_SIM_CPP:
            _step_physics_cpp(self)
        else:
            self._step_physics_python()

        self.tick += 1
        self._check_terminal()

    def total_ships(self, player: int) -> float:
        s = sum(p.ships for p in self.planets.values() if p.owner == player)
        s += sum(f.ships for f in self.fleets.values() if f.owner == player)
        return s

    def total_production(self, player: int) -> float:
        return sum(p.production for p in self.planets.values() if p.owner == player)

    def _check_terminal(self) -> None:
        alive = []
        for pid in (1, 2):
            has_p = any(p.owner == pid for p in self.planets.values())
            has_f = any(f.owner == pid and f.ships > 0 for f in self.fleets.values())
            if has_p or has_f:
                alive.append(pid)
        if len(alive) == 1:
            self.done = True
            self._winner = alive[0]
            return
        if len(alive) == 0:
            self.done = True
            self._winner = 0
            return
        if self.tick >= self.max_ticks:
            self.done = True
            s1, s2 = self.total_ships(1), self.total_ships(2)
            p1, p2 = self.total_production(1), self.total_production(2)
            score1 = s1 + 12.0 * (p1 / PRODUCTION_DIVISOR)
            score2 = s2 + 12.0 * (p2 / PRODUCTION_DIVISOR)
            if abs(score1 - score2) < 1e-3:
                self._winner = 0
            else:
                self._winner = 1 if score1 > score2 else 2

    def winner(self) -> int:
        return self._winner


def project_planet_future(
    sim: GalconSim,
    planet_id: int,
    player: int,
    horizon: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a single planet's exact future ownership/ship trajectory over ``horizon`` ticks.

    Returns ``(my_arr, en_arr, proj_owner, proj_ships)`` each of shape ``(horizon,)``.
    ``proj_owner`` is ``+1`` (mine), ``-1`` (enemy), ``0`` (neutral).
    """
    p = sim.planets[planet_id]
    my_arr = np.zeros(horizon, dtype=np.float32)
    en_arr = np.zeros(horizon, dtype=np.float32)
    for fl in sim.fleets.values():
        if fl.target != planet_id:
            continue
        eta = sim.eta_ticks(fl.x, fl.y, p)
        idx = eta - 1
        if 0 <= idx < horizon:
            if fl.owner == player:
                my_arr[idx] += float(fl.ships)
            else:
                en_arr[idx] += float(fl.ships)

    proj_owner = np.zeros(horizon, dtype=np.float32)
    proj_ships = np.zeros(horizon, dtype=np.float32)

    cur_owner = 1 if p.owner == player else (0 if p.owner == 0 else -1)
    cur_ships = float(p.ships)
    growth_per_tick = (p.production / PRODUCTION_DIVISOR) * sim.tick_dt

    for t in range(horizon):
        if cur_owner != 0:
            cur_ships += growth_per_tick
        ma = float(my_arr[t])
        ea = float(en_arr[t])
        if ma > 0.0 or ea > 0.0:
            # Resolve 3-way (neutral / mine / enemy)
            f_my = ma + (cur_ships if cur_owner == 1 else 0.0)
            f_en = ea + (cur_ships if cur_owner == -1 else 0.0)
            f_nu = cur_ships if cur_owner == 0 else 0.0
            cands = [(f_my, 1), (f_en, -1), (f_nu, 0)]
            cands.sort(key=lambda kv: -kv[0])
            if abs(cands[0][0] - cands[1][0]) <= TIE_EPS:
                cur_ships = 0.0
            else:
                cur_owner = cands[0][1]
                cur_ships = cands[0][0] - cands[1][0]
        proj_owner[t] = float(cur_owner)
        proj_ships[t] = float(cur_ships)

    return my_arr, en_arr, proj_owner, proj_ships


def project_all_planet_futures(
    sim: GalconSim,
    planets: Sequence[Planet],
    player: int,
    horizon: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized exact future ownership/ship trajectories for all planets in ``(N, horizon)`` arrays.

    Fast-paths planets with zero incoming fleets in pure NumPy and runs exact 3-way combat only
    on planets that actually have incoming fleets. Matches ``project_planet_future`` bit-for-bit.
    """
    n = len(planets)
    my_arrs = np.zeros((n, horizon), dtype=np.float32)
    en_arrs = np.zeros((n, horizon), dtype=np.float32)
    if n == 0:
        z = np.zeros((0, horizon), dtype=np.float32)
        return my_arrs, en_arrs, z, z.copy()

    id_to_idx = {p.id: i for i, p in enumerate(planets)}
    has_inc = np.zeros(n, dtype=bool)
    step_dist = max(1e-6, sim.fleet_speed * sim.tick_dt)
    for fl in sim.fleets.values():
        idx_p = id_to_idx.get(fl.target)
        if idx_p is None:
            continue
        dst = planets[idx_p]
        dx = dst.x - fl.x
        dy = dst.y - fl.y
        dist = max(0.0, math.sqrt(dx * dx + dy * dy) - dst.radius - SHIP_RADIUS)
        eta = max(1, int(math.ceil((dist - ARRIVAL_EPS) / step_dist)))
        t_i = eta - 1
        if 0 <= t_i < horizon:
            has_inc[idx_p] = True
            if fl.owner == player:
                my_arrs[idx_p, t_i] += float(fl.ships)
            else:
                en_arrs[idx_p, t_i] += float(fl.ships)

    owners_i = np.array(
        [1.0 if p.owner == player else (0.0 if p.owner == 0 else -1.0) for p in planets],
        dtype=np.float64,
    )
    ships_0 = np.array([float(p.ships) for p in planets], dtype=np.float64)
    growth = np.array([(p.production / PRODUCTION_DIVISOR) * sim.tick_dt for p in planets], dtype=np.float64)
    t_steps = np.arange(1, horizon + 1, dtype=np.float64)[None, :]  # (1, H)

    # Exact fast-path for all planets (exact wherever has_inc is False)
    proj_owner = np.broadcast_to(owners_i[:, None], (n, horizon)).astype(np.float32).copy()
    proj_ships = (ships_0[:, None] + np.where(owners_i[:, None] != 0.0, growth[:, None] * t_steps, 0.0)).astype(
        np.float32
    )

    # Exact tick-by-tick 3-way combat only for planets with incoming fleets
    for i in np.flatnonzero(has_inc):
        cur_owner = int(owners_i[i])
        cur_ships = float(ships_0[i])
        g_i = float(growth[i])
        ma_row = my_arrs[i]
        ea_row = en_arrs[i]
        po_row = proj_owner[i]
        ps_row = proj_ships[i]
        for t in range(horizon):
            if cur_owner != 0:
                cur_ships += g_i
            ma = float(ma_row[t])
            ea = float(ea_row[t])
            if ma > 0.0 or ea > 0.0:
                f_my = ma + (cur_ships if cur_owner == 1 else 0.0)
                f_en = ea + (cur_ships if cur_owner == -1 else 0.0)
                f_nu = cur_ships if cur_owner == 0 else 0.0
                if f_my >= f_en and f_my >= f_nu:
                    top_v, top_o = f_my, 1
                    sec_v = f_en if f_en >= f_nu else f_nu
                elif f_en >= f_my and f_en >= f_nu:
                    top_v, top_o = f_en, -1
                    sec_v = f_my if f_my >= f_nu else f_nu
                else:
                    top_v, top_o = f_nu, 0
                    sec_v = f_my if f_my >= f_en else f_en
                if abs(top_v - sec_v) <= TIE_EPS:
                    cur_ships = 0.0
                else:
                    cur_owner = top_o
                    cur_ships = top_v - sec_v
            po_row[t] = float(cur_owner)
            ps_row[t] = float(cur_ships)

    return my_arrs, en_arrs, proj_owner, proj_ships


def evaluate_galcon_state(sim: GalconSim, player: int, horizon: int = 24) -> float:
    """Fast tactical value estimate in ``[-1, +1]`` (matches C++ ``sim_eval_heuristic``)."""
    if sim.done:
        w = sim.winner()
        return 1.0 if w == player else (0.0 if w == 0 else -1.0)
    H = horizon if (0 < horizon <= 64) else 24
    opp = 2 if player == 1 else 1
    my_ships = sim.total_ships(player)
    en_ships = sim.total_ships(opp)
    my_prod = sim.total_production(player)
    en_prod = sim.total_production(opp)
    ship_adv = (my_ships - en_ships) / max(30.0, my_ships + en_ships)
    prod_adv = (my_prod - en_prod) / max(40.0, my_prod + en_prod)
    # Also value future projected ownership & post-landing ships after in-flight fleets land
    planets = list(sim.planets.values())
    _, _, po_all, ps_all = project_all_planet_futures(sim, planets, player, horizon=H)
    prods = np.array([p.production for p in planets], dtype=np.float64)
    po_end = po_all[:, -1]
    ps_end = ps_all[:, -1].astype(np.float64)
    fut_my_prod = float(prods[po_end > 0].sum())
    fut_en_prod = float(prods[po_end < 0].sum())
    fut_my_ships = float(ps_end[po_end > 0].sum())
    fut_en_ships = float(ps_end[po_end < 0].sum())
    # Include fleets still in flight beyond horizon H
    id_to_planet = {p.id: p for p in planets}
    step_dist = max(1e-6, sim.fleet_speed * sim.tick_dt)
    for fl in sim.fleets.values():
        dst = id_to_planet.get(fl.target)
        in_horizon = False
        if dst is not None:
            dist = max(0.0, math.hypot(dst.x - fl.x, dst.y - fl.y) - dst.radius - SHIP_RADIUS)
            eta = max(1, int(math.ceil((dist - ARRIVAL_EPS) / step_dist)))
            in_horizon = (0 <= (eta - 1) < H)
        if not in_horizon:
            if fl.owner == player:
                fut_my_ships += float(fl.ships)
            elif fl.owner == opp:
                fut_en_ships += float(fl.ships)
    fut_ship_adv = (fut_my_ships - fut_en_ships) / max(30.0, fut_my_ships + fut_en_ships)
    fut_prod_adv = (fut_my_prod - fut_en_prod) / max(40.0, fut_my_prod + fut_en_prod)
    raw = 0.45 * ship_adv + 0.90 * fut_ship_adv + 0.50 * prod_adv + 1.10 * fut_prod_adv
    return float(math.tanh(raw))
