// galcon.hpp - headless replica of the Galcon 2 simulation, C++ port.
//
// Why this exists: every learning result in this workspace has been starved
// by environment throughput. The in-engine snapshot/search collector manages
// 0.22 states/sec; the training rig wants thousands. This owns the forward
// model so search and self-play become possible at all.
//
// Provenance. Rules are either transcribed from galcon2.exe (see
// research/engine-rules.md and galcon_engine.hpp, which carry the originating
// addresses) or measured by black-box probing through the mod API. Each
// constant below is tagged. Nothing here is guessed.
//
// Fidelity standard. The engine is NOT deterministic: unit spawn positions
// are rand()-sampled, and two runs of the same seed produce zero identical
// games. Exact agreement is therefore unattainable and is the wrong target.
// The replica is judged against sim/noisefloor.py, which measures how much
// the engine disagrees with itself. It currently sits at roughly half that.
//
// Deliberate omissions:
//   - in-transit unit-vs-unit combat (`fleet_crash`). Real, decoded, but a
//     private-room toggle that is almost always off, and it changes the
//     game's strategy wholesale.
// Fleet redirect IS implemented -- see redirect(), transcribed from
// g2_redirect @0x443880. An earlier revision of this comment listed it as not
// yet decoded; that was stale, and both ported ladder bots depend on it
// heavily (Waffle's own W-REDIRECT-1 ablation lost 4-36 without redirection).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

// The literally-transcribed engine primitives and their constants live here;
// this file builds the simulation on top of them rather than restating them.
#include "galcon_engine.hpp"

namespace galcon {

// ---------------------------------------------------------------- constants

constexpr int   kNeutral        = -1;

// EXACT (probe, 60s baseline). ships/sec = production / 50.
constexpr float kProductionRate = 0.02f;

// EXACT (unit_init @0x454074): speed = 40.0f * owner.fleet_v_factor, written
// once and never touched. Independent of ship count, confirmed at source.
constexpr float kSpeed          = 40.0f;

// EXACT (unit_init @0x454109): every unit's collide_r.
constexpr float kUnitRadius     = 6.0f;

// kArena and kUnitK come from galcon_engine.hpp (_fleet_init_units @0x4420e0).

// EXACT (unit_init): six unrolled unit_move calls at dt=0.025 before the unit
// enters sys_loop => 6 * 0.025 * 40 = 6.0 units of free travel at zero
// simulated time.
constexpr float kFreeDistance   = 6.0f;

// EXACT (_fleet_init @0x442330): units are placed at rejection-sampled
// uniform points in a disc of radius sqrt(nunits * 36pi / pi) = 6*sqrt(n).
// Area-conserving at the unit collision radius: n discs of radius 6 have
// total area 36pi*n. Depends on nothing but the unit count.
inline float spawn_disc_radius(int units) {
    return kUnitRadius * std::sqrt((float)units);
}

// FITTED, and the only fitted number left in the motion model. Same-owner
// same-target units push each other apart every frame, so the swarm stretches
// along its axis in flight: measured along-track extent 30.7 at launch, 54.8
// at arrival for an 11-unit fleet. Calibrated to land that fleet over 1.372s.
constexpr float kFlightElongation = 1.637f;

// MEASURED, one route. Blocked 640-unit route took 17.8s against 14.4s clear.
// Units deflect around planets purely positionally (collide_do pushes the
// unit out; the planet never moves).
constexpr float kObstruction    = 1.236f;

// Planet radius formula, process-wide. 0 = Galcon 2 (the default; every G2
// tool and parity check), 1 = Galcon 3's galcon.ProductionToRadius
// (Galcon Playtest mods/galcon: 8 + 0.16 * production, 8..24 for 0..100).
inline int g_radius_g3 = 0;
// EXACT (G2): (production + 70) * 12 / 85, verified 15..200, max delta 1.8e-06.
inline float planet_radius(float production) {
    return g_radius_g3 ? 8.0f + production * 0.16f : (production + 70.0f) * 12.0f / 85.0f;
}

// EXACT: round-half-up, 40/40 probe rows. 37 ships at 50% releases 19, not 18.
inline int send_size(float ships, int percent) {
    return (int)std::floor(ships * (float)percent / 100.0f + 0.5f);
}

// Delegates to the transcription in galcon_engine.hpp so there is exactly one
// copy of the rule.
inline int units_for_fleet(float ships, float total_ships, int items_in_use) {
    return fleet_init_units(ships, total_ships, items_in_use);
}

// -------------------------------------------------------------------- state

struct Planet {
    float x, y;
    float production;
    float ships;
    int32_t owner;

    float radius() const { return planet_radius(production); }
    bool  neutral() const { return owner == kNeutral; }
};

// One in-flight unit, carrying real position and heading because redirect
// makes them observable: a redirected unit keeps flying at full speed along
// its old heading while it turns.
//
// Fleets are not stored as objects -- every rule that matters (arrival,
// combat, capture) applies per unit -- but each unit remembers which fleet
// issued it, because g2_redirect retargets by fleet.
// "locked" is our term, not the engine's. unit_move snaps the heading exactly
// onto the bearing whenever the remaining angular error is smaller than one
// turn step, which on undisturbed flight is every frame. A unit flying at a
// stationary target therefore reaches a state where heading *is* the bearing
// and stays there. In that state the whole update collapses to
// `dist -= speed*dt`: no atan2, no cos/sin, no position write. Position is
// reconstructed on demand (see unit_position) and only actually needed when a
// redirect breaks the lock.
struct Unit {
    float   x, y;         // authoritative only while turning; else stale
    float   dist;         // locked: distance still to run to the landing point
    float   dirx, diry;   // locked: unit vector toward the target
    float   heading;      // radians, wrapped to (-pi, pi]
    float   turn_rate;    // EXACT: init pi/32, escalates, never reset
    float   turn_clock;   // accumulates dt*40 while turning, reset on snap
    float   speed;
    float   ships;
    int32_t target;
    int32_t owner;
    int32_t fleet;
    uint8_t locked;
    // Physics mode only (State::phys): position at the start of the previous
    // move (unit_move's stuck test), frames alive (Item +0x8) and a per-unit
    // id for the jitter schedule. dirx/diry then hold the heading as a unit
    // vector and x/y are always authoritative.
    float   px = 0, py = 0;
    int32_t age = 0;
    int32_t uid = 0;
};

// EXACT (unit_move @0x453ac0). Heading turns toward the bearing at
// turn_rate * (dt * 40) per frame; turn_rate starts at pi/32, so the base
// angular rate is (pi/32)*40 = 3.927 rad/s = 225 deg/s, and a full reversal
// takes about 0.8s. After a sustained second of turning (turn_clock > 40) the
// rate escalates by dt*pi/32 per second and is never reset.
//
// There is no acceleration: `speed` is written once in unit_init and never
// touched again. Direction changes cost time; speed changes do not exist.
constexpr float kPi          = 3.14159265358979f;
constexpr float kTwoPi       = 6.28318530717959f;
constexpr float kTurnRate0   = kPi / 32.0f;
constexpr float kTurnTick    = 40.0f;          // D = dt * 40
constexpr float kTurnEscalate = kPi / 1280.0f;  // f64 0x616670

// --------------------------------------------------------------- bot memory
//
// Bot memory is part of the POSITION, not ambient state. snapshot_replay.lua
// has to reconstruct "users, planets, fleets, and bot memories" to replay a
// position faithfully (research/network-architectures.md); the same is true
// here. A rollout forked from a mid-game position must continue the plan the
// bot actually had -- a stateful bot handed a blank memory restarts its
// opening, which is not a legal continuation of that position and silently
// corrupts every label rolled out from it.
//
// State is forked by value (`State c = s;`), so anything that must survive a
// fork has to live inside it. A static side table cannot work by construction:
// two forks of one position are indistinguishable to any key derived from the
// position, yet must diverge independently and may be evaluated interleaved.
//
// The layout below is owned by bots.hpp -- the simulation never reads it. It
// lives here because State does. It is deliberately a fixed-size value type:
// no pointers and no per-slot heap, so a copy stays one memcpy, and `bots` is
// empty until a stateful bot actually runs, so positions driven by stateless
// policies pay nothing at all.
//
// Capacities are measured, not guessed (1800 games, 10..65 planet maps):
// peak queue 27/31/45 and peak tunnel-request list 17/48/97 at 23/48/65
// planets. The defaults below cover every map up to 64 planets with headroom;
// the 65-planet large map needs -DGALCON_MAX_MEM_PLANETS=80. Overflow would
// silently drop an opening lane, so bots.hpp refuses to run rather than
// truncate when a map exceeds the compiled capacity.
//
// Amounts are double, matching Lua's only number type. Float storage was
// tried: it costs 4% of clone throughput less but flips 2 games in 2000
// against the validated build, so it is not worth it.

#ifndef GALCON_MAX_MEM_PLANETS
#define GALCON_MAX_MEM_PLANETS 64
#endif
constexpr int kMaxMemPlanets = GALCON_MAX_MEM_PLANETS;
constexpr int kMaxMemReq     = kMaxMemPlanets * 2;
constexpr int kMaxMemQueue   = kMaxMemPlanets;
constexpr int kMemWords      = (kMaxMemPlanets + 63) / 64;
// Planet indices and counts in the bots' memory: 8-bit as validated, widened
// only when the compiled capacity needs it (GALCON_MAX_MEM_PLANETS > 127, e.g.
// the 200-planet transfer build), so default builds keep the same layout.
using MemIdx   = std::conditional_t<(kMaxMemPlanets > 127), int16_t, int8_t>;
using MemCount = std::conditional_t<(kMaxMemReq > 255), uint16_t, uint8_t>;

struct BotMemory {
    // -- waffle: memory.init / homes / eteam / expand. Written once on the
    //    first call and read-only afterwards.
    uint8_t w_init = 0, w_homes = 0, w_have_expand = 0;
    MemCount w_nexpand = 0;
    int16_t w_home = -1, w_ehome = -1, w_eteam = -1;
    MemIdx  w_expand[kMaxMemPlanets] = {};   // admission order matters

    // -- sparky: MEM.phase / home / enemy / is_target / tgt / front / worth /
    //    treq / q. Rewritten on every call, which is why forking matters.
    int8_t   s_phase = 0;
    uint8_t  s_first = 0, s_worth_valid = 0;
    int16_t  s_home = -1, s_ehome = -1, s_enemy = -1;
    double   s_horizon = 0.0;
    uint64_t s_is_target[kMemWords] = {};
    uint64_t s_front_set[kMemWords] = {};    // tri-state: nil / false / true
    uint64_t s_front_val[kMemWords] = {};
    MemIdx   s_tgt[kMaxMemPlanets];          // -1 == nil
    double   s_worth[kMaxMemPlanets] = {};
    struct Req { MemIdx f, o; double a; };
    struct Act { int32_t f; double a; MemIdx o; uint8_t r; };
    MemCount s_nreq = 0;
    Req      s_req[kMaxMemReq] = {};
    MemCount s_nq = 0, s_q0 = 0;             // FIFO as a ring buffer
    Act      s_q[kMaxMemQueue] = {};

    BotMemory() { for (int i = 0; i < kMaxMemPlanets; ++i) s_tgt[i] = -1; }
};

struct State {
    std::vector<Planet> planets;
    std::vector<Unit>   units;
    float t = 0.0f;
    int32_t next_fleet = 0;
    // Per-player bot memory; empty until a stateful bot runs. See BotMemory.
    std::vector<BotMemory> bots;
    // 1 = per-unit physics transcribed from the engine (see step_phys): units
    // are pushed out of planets and apart from same-owner same-target units
    // every frame. 0 = the old analytic model (flat obstruction slowdown,
    // fitted elongation), kept for the older tools.
    uint8_t phys = 0;
    uint32_t rng = 0x9e3779b9u;
    int32_t next_uid = 0;

    float total_ships() const {
        float s = 0.0f;
        for (const Planet& p : planets) s += p.ships;
        for (const Unit& u : units) s += u.ships;
        return s;
    }

    // Units are Items too, so a galaxy full of fleets shrinks the budget for
    // the next one. Three users, one Item per planet, one per unit, plus one
    // aggregate Item per fleet (approximated by counting distinct targets is
    // not worth it; the term only binds in very large galaxies).
    int items_in_use() const {
        return 3 + (int)planets.size() + (int)units.size();
    }
};

// ----------------------------------------------------------------- dynamics

// Does any other planet's disc intersect the straight route? Bodies hugging
// either endpoint are brushed past, not detoured around.
inline bool blocked(const State& s, int source, int target) {
    const Planet& a = s.planets[source];
    const Planet& b = s.planets[target];
    float dx = b.x - a.x, dy = b.y - a.y;
    float len = std::sqrt(dx * dx + dy * dy);
    if (len <= 0.0f) return false;
    float ux = dx / len, uy = dy / len;
    float ra = a.radius(), rb = b.radius();
    for (size_t i = 0; i < s.planets.size(); ++i) {
        if ((int)i == source || (int)i == target) continue;
        const Planet& p = s.planets[i];
        float px = p.x - a.x, py = p.y - a.y;
        float proj = px * ux + py * uy;
        if (proj <= ra || proj >= len - rb) continue;
        float perp = std::fabs(-px * uy + py * ux);
        if (perp < p.radius()) return true;
    }
    return false;
}

// Deterministic quasi-uniform points in a disc (Vogel spiral).
//
// The engine rejection-samples rand() here. We deliberately do not reproduce
// that: matching would require matching the exact count and order of every
// rand() call in the frame, and any divergence desynchronises everything
// downstream. For a search simulator a low-variance stand-in with the same
// distribution is strictly better than a fragile attempt at bit-matching.
inline void disc_point(int i, int n, float radius, float& ox, float& oy) {
    if (n <= 1) { ox = oy = 0.0f; return; }
    const float golden = 3.14159265358979f * (3.0f - std::sqrt(5.0f));
    float r = radius * std::sqrt(((float)i + 0.5f) / (float)n);
    float a = (float)i * golden;
    ox = r * std::cos(a);
    oy = r * std::sin(a);
}

// EXACT (unit_logic @0x4532d0), applied per unit:
//   friendly -> add; hostile -> subtract; capture iff strictly negative.
//   An exact tie leaves the defender holding an empty planet.
inline void land(State& s, int target, int owner, float ships) {
    Planet& p = s.planets[target];
    if (p.owner == owner) { p.ships += ships; return; }
    p.ships -= ships;
    if (p.ships < 0.0f) { p.owner = owner; p.ships = -p.ships; }
}

// Deterministic per-state rand() stand-in (0..32767), so forks replay
// identically. The engine draws from the C runtime's rand() at the same points.
inline int phys_rand(State& s) {
    s.rng = s.rng * 1103515245u + 12345u;
    return (int)((s.rng >> 16) & 0x7fff);
}

// Optional per-thread recorder of issued commands. Null (the default) costs
// one predictable branch per send/redirect -- neither is on the hot path. The
// RL env (../rl/galcon_env.cpp) points it at a buffer while a teacher bot runs
// on a fork, which turns any scripted bot into a labeller without touching its
// code. kind 0 = send(who=source planet, pct), kind 1 = redirect(who=fleet).
// n = ships launched by a send (0 for redirects); appended for the native
// Sparky port, whose percents are fractional (see send_exact).
struct ActionEvent { int32_t kind, who, tgt, pct, owner; int32_t n = 0; };
inline thread_local std::vector<ActionEvent>* g_action_log = nullptr;

// Create a fleet with an exact ship count (mirrors g2.new_fleet).
// Returns the fleet id, for redirect().
inline int32_t launch(State& s, int source, int target, float ships,
                      int owner) {
    float total = s.total_ships();
    int count = units_for_fleet(ships, total, s.items_in_use());
    float per_unit = ships / (float)count;

    const Planet& a = s.planets[source];
    const Planet& b = s.planets[target];
    float dx = b.x - a.x, dy = b.y - a.y;
    float centre = std::sqrt(dx * dx + dy * dy);
    if (centre <= 0.0f) centre = 1.0f;
    float ux = dx / centre, uy = dy / centre;

    // fleet_init @0x4425e0: launch point is src.collide_r along the bearing.
    // Probed: for planets collide_r == planet_r exactly, so radius() is right.
    float lx = a.x + ux * a.radius();
    float ly = a.y + uy * a.radius();

    // Deflection around a blocking planet is modelled as an effective speed
    // reduction rather than a simulated detour: same arrival time, and it
    // composes with real positions. Measured 24% on one 640-unit route.
    float speed = blocked(s, source, target) ? kSpeed / kObstruction : kSpeed;

    float radius = spawn_disc_radius(count);
    int32_t fleet = s.next_fleet++;
    s.units.reserve(s.units.size() + (size_t)count);

    if (s.phys) {
        // _fleet_init: rejection-sampled uniform points in the disc of radius
        // 6*sqrt(n) about the launch point; unit_init: heading = bearing, then
        // six unit_move(0.025) calls with no collision in between, i.e. 6.0
        // units straight at the target. Full speed; obstruction is physics.
        for (int i = 0; i < count; ++i) {
            float ox, oy;
            do {
                ox = (phys_rand(s) / 32767.0f * 2.0f - 1.0f);
                oy = (phys_rand(s) / 32767.0f * 2.0f - 1.0f);
            } while (ox * ox + oy * oy > 1.0f);
            float sx = lx + ox * radius, sy = ly + oy * radius;
            float ddx = b.x - sx, ddy = b.y - sy;
            float d = std::sqrt(ddx * ddx + ddy * ddy);
            if (d < 1e-6f) { d = 1e-6f; ddx = d; ddy = 0.0f; }
            float cx = ddx / d, cy = ddy / d;
            Unit u{sx + cx * kFreeDistance, sy + cy * kFreeDistance, 0.0f, cx, cy, 0.0f,
                   kTurnRate0, 0.0f, kSpeed, per_unit, (int32_t)target, (int32_t)owner,
                   fleet, 0};
            u.px = u.x; u.py = u.y; u.age = 0; u.uid = s.next_uid++;
            s.units.push_back(u);
        }
        return fleet;
    }

    for (int i = 0; i < count; ++i) {
        float ox, oy;
        disc_point(i, count, radius, ox, oy);
        // Stretch only the along-track component: the swarm ends up this
        // elongated by arrival (see kFlightElongation), so front-load it.
        float along = (ox * ux + oy * uy) * kFlightElongation;
        float sx = lx + along * ux + (ox - (ox * ux + oy * uy) * ux);
        float sy = ly + along * uy + (oy - (ox * ux + oy * uy) * uy);
        // unit_init: initial heading is the exact bearing, then six free
        // unit_move steps at dt=0.025 before the unit enters sys_loop.
        // Born locked (unit_init sets the heading to the exact bearing), so
        // the direction vector is all we need: cos(atan2(dy,dx)) == dx/d and
        // sin(atan2(dy,dx)) == dy/d. One sqrt replaces atan2+cos+sin per
        // unit, which profiling showed was the single hottest thing in the
        // simulator. `heading` is filled in lazily by redirect(), the only
        // caller that can ever need it.
        float ddx = b.x - sx, ddy = b.y - sy;
        float d = std::sqrt(ddx * ddx + ddy * ddy);
        // A spawn point landing exactly on the target would make the
        // direction NaN, and a NaN dist never satisfies the arrival test, so
        // the unit would fly forever and the pool would grow without bound.
        if (d < 1e-6f) { d = 1e-6f; ddx = d; ddy = 0.0f; }
        float cx = ddx / d, cy = ddy / d;
        // The six free pre-steps run straight toward the target, so they cost
        // exactly kFreeDistance of range and leave the direction unchanged.
        float dist = d - kFreeDistance - (b.radius() + kUnitRadius);
        if (dist < 0.0f) dist = 0.0f;
        s.units.push_back(Unit{sx + cx * kFreeDistance, sy + cy * kFreeDistance,
                               dist, cx, cy, 0.0f, kTurnRate0, 0.0f, speed,
                               per_unit, (int32_t)target, (int32_t)owner,
                               fleet, 1});
    }
    return fleet;
}

// Reconstruct a unit's position. Free for a turning unit; two multiplies for
// a locked one, whose x/y are deliberately left stale.
inline void unit_position(const State& s, const Unit& u, float& x, float& y) {
    if (!u.locked) { x = u.x; y = u.y; return; }
    const Planet& t = s.planets[u.target];
    float k = t.radius() + kUnitRadius + u.dist;
    x = t.x - u.dirx * k;
    y = t.y - u.diry * k;
}

// g2_redirect @0x443880: every unit of the fleet is retargeted in place.
// Position, heading, turn_rate, turn_clock and ships are all untouched, so
// the bearing jumps discontinuously while the heading does not -- the unit
// keeps flying at full speed along its old course while it turns onto the
// new one.
inline void redirect(State& s, int32_t fleet, int new_target) {
    if (g_action_log)
        for (const Unit& u : s.units)
            if (u.fleet == fleet) {
                g_action_log->push_back(ActionEvent{1, fleet, new_target, 0, u.owner});
                break;
            }
    if (s.phys) {
        // heading (dirx, diry) is kept; only the target changes
        for (Unit& u : s.units)
            if (u.fleet == fleet) u.target = (int32_t)new_target;
        return;
    }
    for (Unit& u : s.units)
        if (u.fleet == fleet) {
            // Materialise position and heading before dropping the lock:
            // from here the unit must be integrated properly while it turns.
            // Both are kept lazy precisely so this is the only place that
            // pays for them.
            unit_position(s, u, u.x, u.y);
            if (u.locked) u.heading = std::atan2(u.diry, u.dirx);
            u.target = (int32_t)new_target;
            u.locked = 0;
        }
}

// Issue one command. The bot harness snaps percent to a 5% step in 5..100
// before it reaches the engine, so we do the same.
inline bool send(State& s, int source, int target, int percent) {
    percent = (int)std::lround(percent / 5.0) * 5;
    if (percent < 5) percent = 5;
    if (percent > 100) percent = 100;
    Planet& src = s.planets[source];
    int n = send_size(src.ships, percent);
    if (n <= 0) return false;
    src.ships -= (float)n;
    if (g_action_log)
        g_action_log->push_back(ActionEvent{0, source, target, percent, src.owner, n});
    launch(s, source, target, (float)n, src.owner);
    return true;
}

// Launch exactly n ships from source (n <= 0: nothing). The common tail of
// send_exact; pct_log is only what the action log records as the percent.
inline int send_count(State& s, int source, int target, int n, int pct_log) {
    if (n <= 0) return 0;
    Planet& src = s.planets[source];
    src.ships -= (float)n;
    if (g_action_log)
        g_action_log->push_back(ActionEvent{0, source, target, pct_log, src.owner, n});
    launch(s, source, target, (float)n, src.owner);
    return n;
}

// The native Galcon 2 API's fleet_send(percent, to): the raw (fractional)
// percent, no 5% snap, n = floor(ships * percent / 100 + 0.5) round half up
// (research/engine-rules.md), computed in float like send_size. NaN or
// non-positive percents launch nothing; above 100 is 100. Returns n.
inline int send_exact(State& s, int source, int target, double percent) {
    if (!(percent > 0.0)) return 0;
    if (percent > 100.0) percent = 100.0;
    float ships = s.planets[source].ships;
    int n = (int)std::floor(ships * (float)percent / 100.0f + 0.5f);
    return send_count(s, source, target, n, (int)std::lround(percent));
}

inline void step_phys(State& s, float dt);

inline void step(State& s, float dt) {
    if (s.phys) { step_phys(s, dt); return; }
    for (Planet& p : s.planets)
        if (!p.neutral()) p.ships += p.production * kProductionRate * dt;

    if (!s.units.empty()) {
        size_t w = 0;
        for (size_t r = 0; r < s.units.size(); ++r) {
            Unit u = s.units[r];
            const Planet& t = s.planets[u.target];
            float travel = u.speed * dt;

            if (u.locked) {
                // The whole update. Heading equals the bearing and the target
                // does not move, so remaining distance falls by exactly
                // speed*dt. No trig, no sqrt, no position write.
                u.dist -= travel;
                if (u.dist <= 0.0f) { land(s, u.target, u.owner, u.ships); continue; }
                s.units[w++] = u;
                continue;
            } else {
                // unit_move @0x453ac0, transcribed.
                float T = std::atan2(t.y - u.y, t.x - u.x);
                float H = u.heading;
                if (T - H > kPi) H += kTwoPi;
                if (H - T > kPi) T += kTwoPi;
                float D = dt * kTurnTick;
                float R = u.turn_rate;
                if (T > H) { H += R * D; u.turn_clock += D; }
                if (H > T) { H -= R * D; u.turn_clock += D; }
                if (R * D > std::fabs(T - H)) {
                    H = T; u.turn_clock = 0.0f; u.locked = 1;
                }
                if (u.turn_clock > kTurnTick) u.turn_rate = R + D * kTurnEscalate;
                if (H > kPi) H -= kTwoPi;
                if (H < -kPi) H += kTwoPi;
                u.heading = H;
                u.x += std::cos(H) * travel;
                u.y += std::sin(H) * travel;
            }

            // unit_loop @0x453e50: arrived when (u.r + t.r)^2 > dist^2.
            float ax = t.x - u.x, ay = t.y - u.y;
            float rr = kUnitRadius + t.radius();
            float d2 = ax * ax + ay * ay;
            if (rr * rr > d2) { land(s, u.target, u.owner, u.ships); continue; }
            if (u.locked) {
                // Just snapped onto the bearing: cache the route so every
                // later step is a single subtract.
                float d = std::sqrt(d2);
                u.dirx = ax / d;
                u.diry = ay / d;
                u.dist = d - rr;
            }
            s.units[w++] = u;
        }
        s.units.resize(w);
    }
    s.t += dt;
}

// --------------------------------------------------------- physics mode
//
// sys_loop @0x442e40 order: every unit runs unit_loop (unit_move, arrival
// test, unit_logic), then collide_loop resolves contacts for the frame.
//
// unit_move @0x453ac0, physics parts:
//   stuck test: if |x-px|+|y-py| < dt*speed/15 and (id+age) % 48 == 0, nudge
//   x and y each by (rand()%200-100) * dt*speed/100 (constants 15 @0x616664,
//   100 @0x616630); then px,py := x,y; heading turns toward the bearing at
//   turn_rate*dt*40 per frame, snapping when within one step. age is the
//   per-item frame counter at Item +0x8, incremented by sys_loop.
//
// collide_do @0x43db40 (narrow phase), for every overlapping pair
// (d^2 <= (ra+rb)^2):
//   a moves iff (a.mask & b.key) || a.peer == b.key, likewise b; the
//   moving bodies are placed exactly ra+rb apart about the mass-weighted
//   centre, a := c - n*R*mb/M, b := c + n*R*ma/M, n = (b-a)/d. Planets:
//   key 1, mask 0, mass pi*r^2 (planet_init) -> immovable in practice. Units:
//   mask 1, key = peer = ((owner<<12)|target)<<8, mass 1 -> a unit is pushed
//   out of every planet and apart from units with the same owner AND target;
//   all other unit pairs pass through each other. Below d = 0.001 the axis is
//   random. Mid-flight combat (unit_logic's partner branch) needs
//   fleet_crash != 0, which the classic mode leaves at 0, so it is omitted.
//
// The broad phase (collide_split_*) is a kd split down to buckets of <= 32;
// it only changes the order pairs are resolved in. Pairs are resolved
// sequentially with positions updated in place, as the engine does.
// Tunables for fitting other engine builds (Galcon 3); the defaults are the
// Galcon 2 engine's values, so nothing changes unless a caller sets them.
//   g_planet_pad : unit-planet contact distance beyond the planet radius
//   g_unit_sep   : same-fleet unit separation distance (2 * collide_r)
//   g_stuck_mod  : stuck-jitter period ((uid + age) % g_stuck_mod == 0)
//   g_stuck_amp  : stuck-jitter amplitude multiplier
inline float g_planet_pad = kUnitRadius;
inline float g_unit_sep = 2.0f * kUnitRadius;
inline int g_stuck_mod = 48;
inline float g_stuck_amp = 1.0f;

inline void phys_push_planets(State& s, Unit& u) {
    for (const Planet& p : s.planets) {
        float rs = p.radius() + g_planet_pad;
        float dx = u.x - p.x, dy = u.y - p.y;
        if (std::fabs(dx) > rs || std::fabs(dy) > rs) continue;
        float d2 = dx * dx + dy * dy;
        if (d2 > rs * rs) continue;
        float d = std::sqrt(d2);
        if (d < 0.001f) {
            dx = (phys_rand(s) / 32767.0f * 200.0f - 100.0f) / 100.0f;
            dy = (phys_rand(s) / 32767.0f * 200.0f - 100.0f) / 100.0f;
            d = 1.0f;
        }
        float mp = kPi * p.radius() * p.radius(), M = mp + 1.0f;
        float cx = (p.x * mp + u.x) / M, cy = (p.y * mp + u.y) / M;
        u.x = cx + dx * rs * mp / (d * M);
        u.y = cy + dy * rs * mp / (d * M);
    }
}

inline void step_phys(State& s, float dt) {
    for (Planet& p : s.planets)
        if (!p.neutral()) p.ships += p.production * kProductionRate * dt;

    const float D = dt * kTurnTick;
    size_t w = 0;
    for (size_t r = 0; r < s.units.size(); ++r) {
        Unit u = s.units[r];
        const Planet& t = s.planets[u.target];
        float travel = u.speed * dt;
        if (u.age > 0 && std::fabs(u.x - u.px) + std::fabs(u.y - u.py) < travel / 15.0f
            && (u.uid + u.age) % g_stuck_mod == 0) {
            u.x += (float)(phys_rand(s) % 200 - 100) * travel * g_stuck_amp / 100.0f;
            u.y += (float)(phys_rand(s) % 200 - 100) * travel * g_stuck_amp / 100.0f;
        }
        u.px = u.x; u.py = u.y;
        // heading as a unit vector (dirx, diry); turn toward the bearing
        float bx = t.x - u.x, by = t.y - u.y;
        float bd = std::sqrt(bx * bx + by * by);
        if (bd > 1e-6f) {
            bx /= bd; by /= bd;
            float stp = u.turn_rate * D;
            float dot = u.dirx * bx + u.diry * by;
            float cross = u.dirx * by - u.diry * bx;
            // snap when the angular error is under one step: cos(err) > cos(step)
            if (dot > std::cos(stp)) {
                u.dirx = bx; u.diry = by; u.turn_clock = 0.0f;
            } else {
                float a = cross >= 0 ? stp : -stp;
                float c = std::cos(a), sn = std::sin(a);
                float hx = u.dirx * c - u.diry * sn, hy = u.dirx * sn + u.diry * c;
                float hn = 1.0f / std::sqrt(hx * hx + hy * hy);
                u.dirx = hx * hn; u.diry = hy * hn;
                u.turn_clock += D;
                if (u.turn_clock > kTurnTick) u.turn_rate += D * kTurnEscalate;
            }
        }
        u.x += u.dirx * travel;
        u.y += u.diry * travel;
        float ax = t.x - u.x, ay = t.y - u.y;
        float rr = kUnitRadius + t.radius();
        if (rr * rr > ax * ax + ay * ay) { land(s, u.target, u.owner, u.ships); continue; }
        ++u.age;
        s.units[w++] = u;
    }
    s.units.resize(w);

    // collide_loop: planets push units out; same owner+target units separate.
    const size_t n = s.units.size();
    for (size_t i = 0; i < n; ++i) phys_push_planets(s, s.units[i]);
    if (n > 1) {
        // A plain local, not thread_local: a thread_local vector in the
        // -static MinGW DLL corrupted the heap once the env's worker threads
        // exited (0xc0000374).
        std::vector<uint32_t> ord(n);
        for (size_t i = 0; i < n; ++i) ord[i] = (uint32_t)i;
        std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) {
            const Unit& A = s.units[a]; const Unit& B = s.units[b];
            if (A.owner != B.owner) return A.owner < B.owner;
            if (A.target != B.target) return A.target < B.target;
            return A.x < B.x;
        });
        const float R = g_unit_sep, R2 = R * R;
        for (size_t i = 0; i < n; ++i) {
            Unit& a = s.units[ord[i]];
            for (size_t j = i + 1; j < n; ++j) {
                Unit& b = s.units[ord[j]];
                if (b.owner != a.owner || b.target != a.target) break;
                float dx = b.x - a.x;
                if (dx > R + 4.0f) break;   // slack: earlier pushes reorder x a little
                float dy = b.y - a.y;
                if (std::fabs(dy) > R) continue;
                float d2 = dx * dx + dy * dy;
                if (d2 > R2) continue;
                float d = std::sqrt(d2);
                if (d < 0.001f) {
                    dx = (phys_rand(s) / 32767.0f * 200.0f - 100.0f) / 100.0f;
                    dy = (phys_rand(s) / 32767.0f * 200.0f - 100.0f) / 100.0f;
                    d = 1.0f;
                }
                float cx = (a.x + b.x) * 0.5f, cy = (a.y + b.y) * 0.5f;
                float k = R * 0.5f / d;
                a.x = cx - dx * k; a.y = cy - dy * k;
                b.x = cx + dx * k; b.y = cy + dy * k;
            }
        }
    }
    s.t += dt;
}

// Sole owner of every non-neutral planet and every unit, or kNeutral-1 if the
// game is still contested.
constexpr int32_t kContested = -2;

inline int32_t winner(const State& s) {
    int32_t who = kContested;
    for (const Planet& p : s.planets) {
        if (p.neutral()) continue;
        if (who == kContested) who = p.owner;
        else if (who != p.owner) return kContested;
    }
    for (const Unit& u : s.units) {
        if (who == kContested) who = u.owner;
        else if (who != u.owner) return kContested;
    }
    return who;
}

}  // namespace galcon
