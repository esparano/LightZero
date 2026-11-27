// bots.hpp - faithful C++ ports of data/bots/waffle.lua and data/bots/sparky.lua
//
// These are transliterations, not reimplementations. Every threshold, magic
// constant, tie-break and statement order follows the Lua line for line,
// including the parts the research notes call vestigial or buggy. Where the
// original is ambiguous the choice is marked with a "PORT:" comment; those are
// the fidelity risks and they are all listed in one place at the bottom of this
// header.
//
// Reference sources:
//   ../waffle.lua           the bot
//   ../sparky.lua           the bot (already a 4 Hz FIFO port of Sparky v1.6)
//   ../botlib.lua           _bots_data() snapshot shape, _bots_apply() emission
//   ../research/waffle.md   source-level model
//
// ---------------------------------------------------------------------------
// PORT DECISION 1: sb_stats() is dropped, deliberately
//
// Both bots read the sandbox instruction budget and cut scans short when it
// runs low (`if sb_stats() > 60 then break end` and friends). Every one of
// those guards is treated as ALWAYS FALSE here, so every scan runs to
// completion.
//
// This is a design decision, not an approximation we are stuck with. The
// instruction budget is an online crash-safety limit imposed by the live
// sandbox, not part of Waffle's or Sparky's strategy: it exists so a bot
// cannot stall the server, and where it bites it makes the bot play *worse*
// than its author intended by truncating a scan mid-way. The replica has no
// such limit to protect, so there is nothing to reproduce. The consequence is
// that this Waffle considers every candidate target and every relay planet
// where the live one sometimes stops early, which very likely makes it
// slightly STRONGER than ladder Waffle on crowded positions. That is accepted
// and preferred: the port is meant to be a strong rollout policy, and running
// the author's algorithm to completion is a better rendering of the intent
// than reproducing a server-imposed cutoff.
//
// One place the budget is worth emulating anyway, because there it changes
// tempo rather than search depth: sparky.lua slices its *init* across calls
// behind sb_stats guards, and its own header notes "sparky's opening lands a
// couple ticks late". Running init to completion would hand the port a free
// half-second head start on move one. kSparkyInitDelayTicks charges one call
// for phase 1 and one for phase 2; set it to 0 to measure the sensitivity
// (measured: worth about 2 points of the Waffle-Sparky matchup).
//
// ---------------------------------------------------------------------------
// PORT DECISION 2: bot memory travels inside State
//
// Both bots keep `memory` across calls. That memory is part of the position --
// see the long comment on BotMemory in galcon.hpp -- so it lives in
// State::bots and is forked with the position. A rollout started from a
// mid-game state therefore continues the plan the bot really had rather than
// restarting its opening, and two forks of one position diverge independently.
//
// The only bot-side thing that is NOT in State is Sparky's closest-planet
// lists (`MEM.cp`). Those are a pure function of planet positions, which the
// simulation never mutates, so they are identical for every fork and are
// memoised in a small map-keyed cache. That is a cache, not state: nothing
// ever writes to it after construction, so no fork can observe another's
// writes. Everything mutable is in State.
//
// ---------------------------------------------------------------------------
// PORT DECISION 3: pairs() ordering
//
// Lua's pairs() is unordered and the harness snapshot table is keyed by item id
// (integers for planets, sync-id strings for fleets). Everywhere iteration
// order can affect the outcome this port uses:
//   planets  -> planet index ascending
//   fleets   -> fleet id ascending (= launch order)
//   teams    -> owner id ascending
// and table.sort (unstable in Lua) is replaced by std::stable_sort, so ties
// keep that order. Each such site carries a "PORT:" comment.
//
// ---------------------------------------------------------------------------
// PORT DECISION 4: doubles
//
// Lua numbers are doubles. Every bot-side computation here is double, with the
// float state promoted on read, so thresholds land on the same side of the
// comparison as they do in the original.

#pragma once

#include "galcon.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <vector>

namespace galbots {

using galcon::State;
using galcon::Planet;
using galcon::Unit;

// Charge one decision call to sparky's phase-1 init and one to phase-2, which
// is what its sb_stats slicing costs in the live harness. 0 disables.
// -DSPARKY_INIT_DELAY_TICKS=0 rebuilds the no-delay variant for sensitivity
// measurement.
#ifndef SPARKY_INIT_DELAY_TICKS
#define SPARKY_INIT_DELAY_TICKS 2
#endif
inline constexpr int kSparkyInitDelayTicks = SPARKY_INIT_DELAY_TICKS;

inline const double kHuge = std::numeric_limits<double>::infinity();

inline double dst(double ax, double ay, double bx, double by) {
    double dx = bx - ax, dy = by - ay;
    return std::sqrt(dx * dx + dy * dy);
}

// botlib.lua _bots_apply: percent = max(5, min(100, round(percent/5)*5)),
// round(n) = floor(n+0.5). The engine snaps again; this makes the value the
// engine sees identical to the one the live harness would have sent.
inline int snap_percent(double p) {
    if (std::isnan(p)) return 5;
    double r = std::floor(p / 5.0 + 0.5) * 5.0;
    if (!(r > 5.0)) return 5;
    if (r > 100.0) return 100;
    return (int)r;
}

// ---------------------------------------------------------------------------
// fleet view: the Lua bots see aggregate fleet Items, the replica stores loose
// units. engine-rules.md: "The aggregate fleet Item reports the centroid of its
// units (verified exactly)" and fleet_ships is the sum. Rebuild that here.
// ---------------------------------------------------------------------------
struct FleetView {
    int32_t id;
    double  x, y;
    double  ships;
    int     target;
    int     owner;
};

inline void build_fleets(const State& s, std::vector<FleetView>& out) {
    out.clear();
    if (s.units.empty()) return;
    std::map<int32_t, size_t> idx;      // PORT: fleet id ascending
    std::vector<int> count;
    for (const Unit& u : s.units) {
        size_t k;
        auto it = idx.find(u.fleet);
        if (it == idx.end()) {
            k = out.size();
            idx.emplace(u.fleet, k);
            out.push_back(FleetView{u.fleet, 0.0, 0.0, 0.0, (int)u.target,
                                    (int)u.owner});
            count.push_back(0);
        } else {
            k = it->second;
        }
        float px, py;
        galcon::unit_position(s, u, px, py);
        out[k].x += px;
        out[k].y += py;
        out[k].ships += u.ships;
        out[k].target = (int)u.target;   // shared by every unit of the fleet
        count[k]++;
    }
    for (size_t i = 0; i < out.size(); ++i) {
        out[i].x /= (double)count[i];
        out[i].y /= (double)count[i];
    }
    std::sort(out.begin(), out.end(),
              [](const FleetView& a, const FleetView& b) { return a.id < b.id; });
}

// ---------------------------------------------------------------------------
// per-position memory (see PORT DECISION 2). The struct itself is
// galcon::BotMemory, declared next to State because it is part of the
// position; this is only the accessor.
// ---------------------------------------------------------------------------
using galcon::BotMemory;
using galcon::kMaxMemPlanets;
using galcon::kMaxMemReq;
using galcon::kMaxMemQueue;
using galcon::MemIdx;
using galcon::MemCount;

inline BotMemory& bot_memory(State& s, int me) {
    if ((int)s.bots.size() <= me) s.bots.resize((size_t)me + 1);
    return s.bots[me];
}

// tri-state / bitset helpers for the packed per-planet fields
inline bool bit_get(const uint64_t* w, int i) {
    return ((w[i >> 6] >> (i & 63)) & 1ull) != 0;
}
inline void bit_put(uint64_t* w, int i, bool v) {
    if (v) w[i >> 6] |= (1ull << (i & 63));
    else   w[i >> 6] &= ~(1ull << (i & 63));
}

// A map larger than the compiled capacity would silently truncate the packed
// per-planet arrays, so refuse loudly instead. Raise it with
// -DGALCON_MAX_MEM_PLANETS=N.
inline bool memory_fits(int np) {
    if (np <= kMaxMemPlanets) return true;
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::fprintf(stderr,
                     "[bots.hpp] map has %d planets but GALCON_MAX_MEM_PLANETS "
                     "is %d; waffle/sparky will idle. Rebuild with a larger "
                     "-DGALCON_MAX_MEM_PLANETS.\n", np, kMaxMemPlanets);
    }
    return false;
}

inline uint64_t map_fingerprint(const State& s) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint32_t v) { h ^= (uint64_t)v; h *= 1099511628211ull; };
    mix((uint32_t)s.planets.size());
    for (const Planet& p : s.planets) {
        uint32_t a, b, c;
        std::memcpy(&a, &p.x, 4);
        std::memcpy(&b, &p.y, 4);
        std::memcpy(&c, &p.production, 4);
        mix(a); mix(b); mix(c);
    }
    return h;
}

// Sparky's MEM.cp: for each planet, every other planet sorted by distance.
// Pure geometry -- the simulation never moves a planet -- so this is identical
// for every fork of every position on a given map and is memoised rather than
// stored. Nothing writes to it after construction, so sharing it between forks
// cannot leak state; that is the entire difference between a cache and the
// memory that had to move into State.
inline const std::vector<std::vector<int> >& closest_lists(const State& s) {
    struct Slot { uint64_t fp; int np; std::vector<std::vector<int> > cp; };
    // thread_local: the RL env runs games on a thread pool, and a shared
    // 4-slot cache would be a data race (one thread reassigning a slot's
    // vectors while another reads them). Per-thread it stays a pure cache.
    thread_local static Slot cache[4] = {};
    thread_local static int  next = 0;
    const int NP = (int)s.planets.size();
    uint64_t fp = map_fingerprint(s);
    for (int i = 0; i < 4; ++i)
        if (cache[i].np == NP && cache[i].fp == fp && !cache[i].cp.empty())
            return cache[i].cp;
    Slot& c = cache[next];
    next = (next + 1) & 3;
    c.fp = fp; c.np = NP;
    c.cp.assign((size_t)NP, std::vector<int>());
    std::vector<double> ds((size_t)NP, 0.0);
    for (int n = 0; n < NP; ++n) {
        std::vector<int> cp;
        cp.reserve((size_t)NP);
        for (int m = 0; m < NP; ++m)
            if (m != n) {
                cp.push_back(m);
                ds[m] = dst(s.planets[n].x, s.planets[n].y,
                            s.planets[m].x, s.planets[m].y);
            }
        // PORT: unstable Lua sort -> stable, ties keep index order.
        std::stable_sort(cp.begin(), cp.end(),
                         [&](int a, int b) { return ds[a] < ds[b]; });
        c.cp[(size_t)n] = cp;
    }
    return c.cp;
}

// ===========================================================================
//                                  WAFFLE
//              transliteration of bots_waffle() in ../waffle.lua
// ===========================================================================
inline void act_waffle(State& s, int me, uint32_t /*seed*/, uint32_t /*tick*/) {
    // waffle.lua is fully deterministic -- no random() anywhere -- so the seed
    // is deliberately unused.
    const int NP = (int)s.planets.size();
    if (NP == 0 || !memory_fits(NP)) return;
    BotMemory& M = bot_memory(s, me);

    const bool FIRST = !M.w_init;
    M.w_init = 1;

    const int uteam = me;      // one user per team in the replica

    auto PX = [&](int i) { return (double)s.planets[i].x; };
    auto PY = [&](int i) { return (double)s.planets[i].y; };
    auto PR = [&](int i) { return (double)s.planets[i].radius(); };
    auto D  = [&](int a, int b) { return dst(PX(a), PY(a), PX(b), PY(b)); };

    // -- waffle.lua:23-36  memory.homes / memory.eteam, computed once ------
    if (!M.w_homes) {
        double bestprod[8];
        int    besti[8];
        for (int i = 0; i < 8; ++i) { bestprod[i] = 0.0; besti[i] = -1; }
        for (int i = 0; i < NP; ++i) {          // PORT: index ascending
            const Planet& p = s.planets[i];
            if (p.neutral()) continue;
            int o = p.owner;
            if (o < 0 || o > 7) continue;
            if (besti[o] < 0 || bestprod[o] < (double)p.production) {
                bestprod[o] = p.production;
                besti[o] = i;
            }
        }
        M.w_home = (int16_t)besti[uteam & 7];
        // "for k,v in pairs(homes) do if k~=uteam then eteam=k end" keeps the
        // LAST non-own team; unambiguous in 1v1. PORT: team id ascending.
        for (int k = 0; k < 8; ++k)
            if (k != uteam && besti[k] >= 0) M.w_eteam = (int16_t)k;
        M.w_ehome = (int16_t)((M.w_eteam >= 0) ? besti[M.w_eteam] : -1);
        M.w_homes = 1;
    }
    const int home = M.w_home, ehome = M.w_ehome;

    // -- waffle.lua:38-88  world classification ---------------------------
    std::vector<FleetView> fleets;
    build_fleets(s, fleets);

    double ships = 0, total = 0, tprod = 0, myprod = 0;
    int    ecount = 0;
    double enemies[8], eprod[8];
    bool   ehas[8], ephas[8];
    for (int i = 0; i < 8; ++i) {
        enemies[i] = 0; eprod[i] = 0; ehas[i] = false; ephas[i] = false;
    }

    std::vector<int> lneutral, lmyplanets, lmyteam, leplanets, lothers;
    std::vector<int> lmyfleets, lefleets;   // indices into `fleets`

    for (int i = 0; i < NP; ++i) {          // PORT: index ascending
        const Planet& p = s.planets[i];
        double sh = p.ships;
        if (!p.neutral()) {
            int o = p.owner;
            if (o == uteam) {
                ships += sh;
            } else if (o >= 0 && o < 8) {
                if (!ehas[o]) { ehas[o] = true; enemies[o] = sh; ++ecount; }
                else            enemies[o] += sh;
            }
            total += sh;
        }
        if (p.neutral()) {
            lneutral.push_back(i);
            lothers.push_back(i);
        } else {
            tprod += p.production;
            if (p.owner == uteam) {
                lmyteam.push_back(i);
                myprod += p.production;
                lmyplanets.push_back(i);
            } else {
                int o = p.owner;
                if (o >= 0 && o < 8) {
                    if (!ephas[o]) { ephas[o] = true; eprod[o] = p.production; }
                    else             eprod[o] += p.production;
                }
                lothers.push_back(i);
                leplanets.push_back(i);
            }
        }
    }
    for (size_t fi = 0; fi < fleets.size(); ++fi) {   // PORT: fleet id ascending
        const FleetView& f = fleets[fi];
        int o = f.owner;
        if (o == uteam) ships += f.ships;
        else if (o >= 0 && o < 8) {
            if (!ehas[o]) { ehas[o] = true; enemies[o] = f.ships; ++ecount; }
            else            enemies[o] += f.ships;
        }
        total += f.ships;
        if (o == uteam) lmyfleets.push_back((int)fi);
        else            lefleets.push_back((int)fi);
    }

    // -- waffle.lua:89-98  virtual survival credit ------------------------
    if (ecount == 1 && ehome >= 0 && tprod >= myprod * 2) {
        double dmin = kHuge;
        int closest = home;
        for (size_t ai = 0; ai < lmyplanets.size(); ++ai)
            for (size_t bi = 0; bi < leplanets.size(); ++bi) {
                double d = D(lmyplanets[ai], leplanets[bi]);
                if (dmin > d) { dmin = d; closest = lmyplanets[ai]; }
            }
        // Lua would error here if `closest` were nil; it cannot be unless the
        // bot owns nothing at all, in which case we simply skip the credit.
        if (closest >= 0)
            ships += dmin * (double)s.planets[closest].production / 2000.0;
    }

    const double control = ships / total;

    // -- waffle.lua:100-126  coalition heuristics (inert in 1v1) ----------
    int big = -1;
    for (int e = 0; e < 8; ++e) {           // PORT: team id ascending
        if (!ehas[e]) continue;
        double v = enemies[e] / total;
        enemies[e] = v;
        if (v > .5) big = e;
        else if (v > .4) { if (big >= 0) big = -1; else big = e; }
    }
    double bigv = 0; int bigp = -1;
    for (int e = 0; e < 8; ++e) {
        if (!ephas[e]) continue;
        double n = eprod[e] / tprod;
        if (n > bigv) { bigv = n; bigp = e; }
    }
    double ships0 = 0, ships1 = 0;
    int big0 = -1, big1 = -1;
    for (int e = 0; e < 8; ++e) {
        if (!ehas[e]) continue;
        double v = enemies[e];
        if (v > ships0) { ships1 = ships0; ships0 = v; big1 = big0; big0 = e; }
        else if (v > ships1) { ships1 = v; big1 = e; }
    }
    (void)big0;
    // `hide,big = big` -- unbalanced assignment, so this only clears `big`.
    if ((big >= 0 && (big1 >= 0 && control * 2 < ships1 && ships0 < .6)) ||
        (bigp >= 0 && big != bigp))
        big = -1;
    const double control2 = control / (1.0 - ships0);

    // -- waffle.lua:127-202  first-call opening portfolio -----------------
    if (FIRST && home >= 0 && !lneutral.empty() && ecount == 1) {
        // path(): note that t and ft are rebound INSIDE the loop, so later
        // iterations compare against the newly chosen intermediate.
        auto pathf = [&](int f, int t, const std::vector<int>& set) -> int {
            double ft = D(f, t);
            for (size_t i = 0; i < set.size(); ++i) {
                int p = set[i];
                if (p != f && p != t) {
                    double pt = D(p, t);
                    if (pt < ft) {
                        double fp = D(f, p);
                        if (fp < ft && (pt + fp - PR(p) * 2.0) < ft) {
                            t = p; ft = fp;
                        }
                    }
                }
            }
            return t;
        };
        auto recovertime = [&](int a, int b) -> double {
            double r = (double)s.planets[b].ships - (double)s.planets[a].ships;
            double base = D(a, b) / 20.0 + (double)s.planets[b].ships * 50.0 /
                                               (double)s.planets[b].production;
            if (r < 0) return base;
            return base + r * 50.0 / (double)s.planets[a].production;
        };
        auto pathlength = [&](int f, int t, const std::vector<int>& set) -> double {
            double d = 0;
            for (int i = 0; i < 20; ++i) {
                if (f == t) break;
                int F = pathf(f, t, set);
                d += D(f, F);
                f = F;
            }
            return d;
        };

        double eeval = kHuge;
        int enemy = -1;
        for (size_t i = 0; i < leplanets.size(); ++i) {
            double n = D(leplanets[i], home);
            if (eeval > n) { eeval = n; enemy = leplanets[i]; }
        }
        if (enemy < 0) {
            double e2 = -kHuge;
            for (size_t i = 0; i < lothers.size(); ++i) {
                double n = D(lothers[i], home);
                if (n > e2) { e2 = n; enemy = lothers[i]; }
            }
        }

        std::vector<int> p = lneutral;
        std::vector<double> rt(NP, 0.0);
        for (size_t i = 0; i < p.size(); ++i) rt[p[i]] = recovertime(home, p[i]);
        // PORT: Lua table.sort is unstable; stable_sort keeps index order on
        // equal recovery times.
        std::stable_sort(p.begin(), p.end(),
                         [&](int a, int b) { return rt[a] < rt[b]; });

        std::vector<int> expand;
        double c = s.planets[home].ships;
        double cinit = c, excess = 0;
        double benefit = (double)s.planets[home].production * D(enemy, home) / 2000.0;
        double prod = s.planets[home].production;
        for (size_t i = 0; i < p.size(); ++i) {
            int v = p[i];
            if (D(enemy, v) < D(home, v)) break;   // territorial frontier stop
            double sc = std::max(1.0, (double)s.planets[v].ships + 1.0);
            double ben = ((D(enemy, v) - pathlength(home, v, expand)) / 40.0) *
                             (double)s.planets[v].production / 50.0 - sc;
            double liability = sc - std::max(c, 0.0);
            if (c < sc) ben -= (excess + liability) *
                               (double)s.planets[v].production / prod;
            if (benefit + ben + c > cinit) {
                if (c < sc) excess += liability;
                benefit += ben;
                c -= sc;
                if (c > 0) prod += (double)s.planets[v].production;
                expand.push_back(v);
            }
        }
        // memory.expand, in admission order (order reaches GetTarget's
        // first-wins tie-break, so it is preserved rather than sorted).
        M.w_nexpand = (MemCount)std::min((size_t)kMaxMemPlanets, expand.size());
        for (int i = 0; i < (int)M.w_nexpand; ++i)
            M.w_expand[i] = (MemIdx)expand[(size_t)i];
        M.w_have_expand = 1;
        return;                                    // first call spends nothing
    }

    // -- waffle.lua:203-208  asset priority queue -------------------------
    struct Asset { bool planet; int idx; double ships; };
    std::vector<Asset> stuff;
    // PORT: mystuff is filled during one pairs(G) pass, so planets and fleets
    // are interleaved in an unspecified order; here planets (index ascending)
    // come first, then fleets (id ascending), then a stable sort by ships.
    for (size_t i = 0; i < lmyplanets.size(); ++i)
        stuff.push_back(Asset{true, lmyplanets[i],
                              (double)s.planets[lmyplanets[i]].ships});
    for (size_t i = 0; i < lmyfleets.size(); ++i)
        stuff.push_back(Asset{false, lmyfleets[i], fleets[lmyfleets[i]].ships});
    std::stable_sort(stuff.begin(), stuff.end(),
                     [](const Asset& a, const Asset& b) { return a.ships > b.ships; });

    std::vector<int> targets = leplanets;

    // mutable copies of the snapshot fields the bot writes to
    std::vector<double> pships(NP);
    for (int i = 0; i < NP; ++i) pships[i] = s.planets[i].ships;
    std::vector<int> ftarget(fleets.size());
    for (size_t i = 0; i < fleets.size(); ++i) ftarget[i] = fleets[i].target;

    // -- waffle.lua:209-225  FFA fleet recall (never fires in 1v1) --------
    bool recall = false;
    if (ecount > 1 && control2 < 1.0 / (double)ecount && big < 0) {
        for (size_t ai = 0; ai < stuff.size(); ++ai) {
            if (stuff[ai].planet) continue;
            int fi = stuff[ai].idx;
            if (s.planets[ftarget[fi]].owner == uteam) continue;
            double dmin = kHuge; int closest = -1;
            for (size_t pi = 0; pi < lmyplanets.size(); ++pi) {
                double d = dst(PX(lmyplanets[pi]), PY(lmyplanets[pi]),
                               fleets[fi].x, fleets[fi].y);
                if (dmin > d) { dmin = d; closest = lmyplanets[pi]; }
            }
            if (closest >= 0) {
                if (fleets[fi].target != closest)
                    galcon::redirect(s, fleets[fi].id, closest);
                return;
            }
        }
        recall = true;
    }

    const bool defend_global =
        (ecount == 1 && myprod / tprod > .5 && control < .8) || recall;

    // -- waffle.lua:237-289  strategic target set -------------------------
    double bigsize = -1, smallsize = -1;
    int biggest = -1, smallest = -1;
    double bigprod = 0, smallprod = 0;
    for (int e = 0; e < 8; ++e) {
        if (!ehas[e]) continue;
        double v = enemies[e];
        if (v > bigsize) {
            smallsize = bigsize; bigsize = v * total;
            smallest = biggest;  biggest = e;
            smallprod = bigprod; bigprod = ephas[e] ? eprod[e] : 0.0;
        } else {
            smallest = e; smallsize = v * total;
            smallprod = ephas[e] ? eprod[e] : 0.0;
        }
    }
    (void)smallsize; (void)smallest; (void)bigprod; (void)smallprod;

    // `targets = data.eplanets` ALIASES the list, so the expansion neutrals
    // appended below also land in data.eplanets. The only later read of
    // data.eplanets is the late-expansion `enemy` scan, so the alias is
    // reproduced by scanning `targets` there when this branch was taken.
    bool targets_alias_eplanets = false;
    if (ecount < 2 || control > .5) {
        targets = leplanets;
        targets_alias_eplanets = true;
    } else if (big >= 0) {
        targets.clear();
        for (size_t i = 0; i < leplanets.size(); ++i)
            if (s.planets[leplanets[i]].owner == big) targets.push_back(leplanets[i]);
    } else if (ecount == 2) {
        if (ships > bigsize) {
            targets = leplanets;
            targets_alias_eplanets = true;
        } else {
            targets.clear();
            for (size_t i = 0; i < leplanets.size(); ++i)
                if (s.planets[leplanets[i]].owner == biggest)
                    targets.push_back(leplanets[i]);
        }
    } else {
        targets.clear();
    }

    // -- waffle.lua:290-298  unfinished opening portfolio stays admissible -
    if (M.w_have_expand) {
        for (int i = 0; i < (int)M.w_nexpand; ++i) {
            int v = M.w_expand[i];
            if (v >= 0 && v < NP && s.planets[v].neutral()) targets.push_back(v);
        }
    }

    // -- waffle.lua:348-381  majority-preserving late expansion -----------
    if (control > .5 && control < .8) {
        double lships = ships, ltotal = total;
        std::vector<int> list = lneutral;   // PORT: index ascending
        int targ;
        do {
            targ = -1;
            // `enemy` below is recomputed but only used in a comparison that
            // `d = -production` makes unconditional (research/waffle.md notes
            // this is vestigial). Reproduced verbatim anyway.
            double eeval = kHuge;
            int enemy = -1;
            const std::vector<int>& scan =
                targets_alias_eplanets ? targets : leplanets;
            for (size_t i = 0; i < scan.size(); ++i) {
                double n = D(scan[i], home);
                if (eeval > n) { eeval = n; enemy = scan[i]; }
            }
            if (enemy < 0) {
                double e2 = -kHuge;
                for (size_t i = 0; i < lothers.size(); ++i) {
                    double n = D(lothers[i], home);
                    if (n > e2) { e2 = n; enemy = lothers[i]; }
                }
            }
            double bestd = kHuge;
            for (size_t i = 0; i < list.size(); ++i) {
                int v = list[i];
                if (lships - (double)s.planets[v].ships * 1.5 > ltotal / 2) {
                    double d = -(double)s.planets[v].production;
                    if (bestd > d && (enemy < 0 || d < D(enemy, v))) {
                        bestd = d; targ = v;
                    }
                }
            }
            if (targ >= 0) {
                targets.push_back(targ);
                lships -= (double)s.planets[targ].ships;
                ltotal -= (double)s.planets[targ].ships;
                for (size_t i = 0; i < list.size(); ++i)
                    if (list[i] == targ) {           // swap-remove, as in Lua
                        list[i] = list.back();
                        list.pop_back();
                        break;
                    }
            }
        } while (targ >= 0);
    }

    // -- waffle.lua:382-396  friendly-planet tunnelling -------------------
    // fplanet is the source planet index, or -1 when the source is a fleet
    // (`p ~= f` can never be true for a fleet source).
    auto tunnel = [&](double fx, double fy, int fplanet, int t) -> int {
        if (t < 0) return -1;
        double ft = dst(fx, fy, PX(t), PY(t));
        for (size_t i = 0; i < lmyteam.size(); ++i) {   // PORT: index ascending
            int p = lmyteam[i];
            double fp = dst(fx, fy, PX(p), PY(p));
            if (fp < ft) {
                double pt = D(p, t);
                if (p != fplanet && pt < ft && (pt + fp - PR(p) * 2.0) < ft) {
                    t = p; ft = fp;
                }
            }
        }
        return t;
    };

    // -- waffle.lua:397-417  tactical target choice -----------------------
    auto GetTarget = [&](double fx, double fy, int self_fleet) -> int {
        if (recall) return -1;
        double teval = kHuge;
        int t = -1;
        for (size_t i = 0; i < targets.size(); ++i) {
            int v = targets[i];
            const Planet& P = s.planets[v];
            double sc = P.ships;
            double d = dst(fx, fy, (double)P.x, (double)P.y);
            if (P.neutral()) {
                // eventual reservation: every friendly fleet already committed
                // to this neutral is subtracted, regardless of arrival order.
                for (size_t k = 0; k < lmyfleets.size(); ++k) {
                    int xi = lmyfleets[k];
                    if (xi != self_fleet && ftarget[xi] == v)
                        sc -= std::floor(fleets[xi].ships);
                }
            } else {
                sc += std::ceil((double)P.production * d / 2000.0);
            }
            if (std::floor(sc) >= 0) {
                double n = d / 10.0 + sc - (double)P.production / 4.0;
                if (P.neutral()) n = d / 10.0;
                if (teval > n) { teval = n; t = v; }
            }
        }
        return t;
    };

    // -- waffle.lua:418-458  threat ledger --------------------------------
    std::vector<std::vector<int> > efmap(NP);   // planet -> enemy fleet indices
    std::vector<char> imminent(NP, 0);
    for (size_t i = 0; i < lefleets.size(); ++i) {
        int fi = lefleets[i];
        int t = fleets[fi].target;
        if (t < 0 || t >= NP) continue;
        efmap[t].push_back(fi);
        if (dst(fleets[fi].x, fleets[fi].y, PX(t), PY(t)) < 100.0) imminent[t] = 1;
    }

    std::vector<double> danger(NP, 0.0), help(NP, 0.0);
    std::vector<char> has_danger(NP, 0);
    for (size_t pi = 0; pi < lmyplanets.size(); ++pi) {  // PORT: index ascending
        int i = lmyplanets[pi];
        double available = std::floor((double)s.planets[i].ships);
        double dmin = kHuge;
        // a hostile fleet aimed at a neutral within 100 of this planet is
        // charged against this planet too (one fleet can be charged twice)
        std::vector<int> set = efmap[i];
        for (size_t k = 0; k < lneutral.size(); ++k) {
            int n = lneutral[k];
            if (!efmap[n].empty() && D(n, i) < 100.0)
                for (size_t q = 0; q < efmap[n].size(); ++q)
                    set.push_back(efmap[n][q]);
        }
        std::sort(set.begin(), set.end());                 // Lua set semantics:
        set.erase(std::unique(set.begin(), set.end()), set.end());
        for (size_t k = 0; k < set.size(); ++k) {
            int fi = set[k];
            available -= std::ceil(fleets[fi].ships);
            double d = dst(fleets[fi].x, fleets[fi].y, PX(i), PY(i)) - PR(i);
            if (d < dmin) dmin = d;
        }
        if (dmin == kHuge) dmin = 0;
        double h = 0;
        for (size_t k = 0; k < lmyfleets.size(); ++k)
            if (ftarget[lmyfleets[k]] == i) h += fleets[lmyfleets[k]].ships;
        help[i] = h;
        danger[i] = std::floor(available +
                               (double)s.planets[i].production * dmin / 2000.0);
        has_danger[i] = 1;
    }

    // -- waffle.lua:465-539  one-command scheduler ------------------------
    std::vector<std::pair<bool, int> > selected;   // (is_planet, index)
    int maintarget = -1;
    bool have_percent = false;
    double percent = 0;

    auto avail_of = [&](int v) {
        return (has_danger[v] ? danger[v] : pships[v]) + help[v];
    };

    for (size_t ai = 0; ai < stuff.size(); ++ai) {
        const Asset& A = stuff[ai];
        if (A.planet) {
            int fi = A.idx;
            double available = has_danger[fi] ? danger[fi] : pships[fi];
            if (!(available > 0)) continue;
            double fx = PX(fi), fy = PY(fi);
            int t0 = GetTarget(fx, fy, -1);
            int t  = tunnel(fx, fy, fi, t0);
            double low = 0; int t2 = -1;
            for (size_t k = 0; k < lmyplanets.size(); ++k) {
                int v = lmyplanets[k];
                double d = avail_of(v);
                if (d < low) { low = d; t2 = v; }
            }
            bool def = defend_global;
            if (t2 >= 0 && imminent[t2]) def = true;
            if (t >= 0) {
                if (t2 >= 0 && def && !s.planets[t].neutral() &&
                    dst(PX(t), PY(t), fx, fy) >
                        dst(PX(t2), PY(t2), fx, fy) * 1.5 && t2 != fi)
                    t = t2;
            } else if (t2 >= 0 && def && t2 != fi) {
                t = t2;
            }
            if (t < 0) continue;
            if (maintarget < 0) maintarget = t;
            if (maintarget != t && t0 >= 0) {
                if (dst(fx, fy, PX(t0), PY(t0)) + 10.0 >
                    dst(fx, fy, PX(maintarget), PY(maintarget)) +
                        D(maintarget, t0) - PR(maintarget) * 2.0)
                    t = maintarget;
            }
            if (maintarget != t) continue;
            double fs = pships[fi];
            double a = std::floor(available * 20.0 / fs) * 5.0;
            if (s.planets[t].neutral())
                a = std::min(a, std::ceil(((double)s.planets[t].ships + 1.0) *
                                          20.0 / fs) * 10.0);
            if (!have_percent) { percent = a; have_percent = true; }
            if (percent <= a) {
                // waffle.lua:498 - snapshot-only bookkeeping with what looks
                // like a units mistake (subtracts ~`a` ships, not `a` percent).
                // Kept because `help` reads the mutated value below.
                double ns = fs - std::floor(std::ceil(a * 20.0 / fs) * fs / 20.0 + 0.5);
                if (!std::isfinite(ns)) ns = 0.0;   // PORT: fs==0 -> inf*0 nan
                pships[fi] = ns;
                selected.push_back(std::make_pair(true, fi));
                if (t == t2) help[t] += pships[fi];
            }
        } else {
            int fi = A.idx;
            double fx = fleets[fi].x, fy = fleets[fi].y;
            int t = tunnel(fx, fy, -1, GetTarget(fx, fy, fi));
            double low = 0; int t2 = -1;
            int targ = ftarget[fi];
            for (size_t k = 0; k < lmyplanets.size(); ++k) {
                int v = lmyplanets[k];
                double d = avail_of(v);
                if (v == targ) d -= fleets[fi].ships;
                if (d < low) { low = d; t2 = v; }
            }
            bool def = defend_global;
            if (t2 >= 0 && imminent[t2]) def = true;
            if (t >= 0) {
                if (t2 >= 0 && def && !s.planets[t].neutral() &&
                    dst(PX(t), PY(t), fx, fy) >
                        dst(PX(t2), PY(t2), fx, fy) * 1.5)
                    t = t2;
            } else if (t2 >= 0 && def) {
                t = t2;
            }
            if (t < 0 || ftarget[fi] == t) continue;
            if (maintarget < 0) maintarget = t;
            if (maintarget == t) {
                ftarget[fi] = t;
                selected.push_back(std::make_pair(false, fi));
                if (t == t2) help[t] += fleets[fi].ships;
            }
        }
    }

    if (maintarget < 0 || selected.empty()) return;

    // botlib.lua _bots_apply: one shared percentage, planet sources send,
    // fleet sources redirect (and only if their live target differs).
    int pct = snap_percent(have_percent ? percent : 50.0);
    for (size_t i = 0; i < selected.size(); ++i) {
        if (selected[i].first) {
            galcon::send(s, selected[i].second, maintarget, pct);
        } else {
            int fi = selected[i].second;
            if (fleets[fi].target != maintarget)
                galcon::redirect(s, fleets[fi].id, maintarget);
        }
    }
}

// ===========================================================================
//                            WAFFLE (optimized)
//   transliteration of bots_waffle3z_optimized() (esparano/GalconModVault),
//   with OPTS = its default `optimized` table (params.options is never set).
//
// This is act_waffle with the diff between that file's body and waffle.lua's
// applied. Everything not marked "WOPT:" is identical to act_waffle above,
// including its PORT decisions (sb_stats guards dropped, deterministic
// orders). Each optimized-specific change carries a "WOPT:" comment.
// ===========================================================================
namespace wopt {
constexpr double behind_in_prod_threshold_factor            = 1.355;
constexpr double behind_in_prod_ships_increase_factor       = 0.977;
constexpr double behind_in_prod_nearest_enemy_prod_exponent = 1.142;
constexpr double tunneling_path_dist_discount               = -0.425;
constexpr double recover_time_dist_factor                   = 1.286;
constexpr double recover_time_target_prod_factor            = 0.6185;
constexpr double recover_time_source_prod_factor            = 0.914;
constexpr double benefit_factor                             = 0.9995;
constexpr double benefit_excess_factor                      = 0.908;
constexpr double benefit_liability_factor                   = 0.96;
constexpr double expand_target_prod_exponent                = 0.92;
constexpr double expand_total_prod_exponent                 = 1.076;
constexpr double fleet_imminent_threshold_factor            = 1.42;
constexpr double control_threshold_medium                   = 0.9704;
constexpr double control_threshold_high                     = 0.9948;
constexpr double with_control_ships_factor                  = 0.971;
constexpr double control_maintain_threshold_factor          = 0.988;
constexpr double fleet_radius_dist_factor                   = 1.019;
constexpr double fleet_target_prod_factor                   = 0.922;
constexpr double fleet_target_dist_factor                   = 0.958;
constexpr double fleet_neutral_target_dist_factor           = 0.626;
constexpr double available_ships_target_factor_1            = 0.9425;
constexpr double fleet_target_factor_1                      = 1.043;
constexpr double fleet_target_factor_2                      = 0.9995;
constexpr double defend_factor_1                            = 1.004;
constexpr double defend_factor_2                            = 1.202;
constexpr double danger_strength                            = 0.9545;
constexpr double help_strength                              = 0.9995;
constexpr double target_defense_strength                    = 0.9785;
constexpr double main_target_distance_factor                = 1.1015;
}  // namespace wopt

inline void act_waffle_opt(State& s, int me, uint32_t /*seed*/, uint32_t /*tick*/) {
    // deterministic, like waffle.lua; the seed is unused.
    const int NP = (int)s.planets.size();
    if (NP == 0 || !memory_fits(NP)) return;
    BotMemory& M = bot_memory(s, me);

    const bool FIRST = !M.w_init;
    M.w_init = 1;

    const int uteam = me;

    auto PX = [&](int i) { return (double)s.planets[i].x; };
    auto PY = [&](int i) { return (double)s.planets[i].y; };
    auto PR = [&](int i) { return (double)s.planets[i].radius(); };
    auto D  = [&](int a, int b) { return dst(PX(a), PY(a), PX(b), PY(b)); };

    // -- homes / eteam ----------------------------------------------------
    // WOPT: opt.lua recomputes homes when `homes[uteam].owner ~= USER`. But
    // memory.homes holds the snapshot tables of the call that computed it
    // (botlib copies items fresh every call and never updates them), so that
    // owner field is frozen at USER and the recompute can never fire in the
    // live harness. Reproduced as compute-once, exactly like act_waffle. (It
    // would be inert anyway: after FIRST, `home` only feeds the late-expansion
    // `enemy`, whose comparison against -production is unconditional, and the
    // dead attackclosest.)
    if (!M.w_homes) {
        double bestprod[8];
        int    besti[8];
        for (int i = 0; i < 8; ++i) { bestprod[i] = 0.0; besti[i] = -1; }
        for (int i = 0; i < NP; ++i) {          // PORT: index ascending
            const Planet& p = s.planets[i];
            if (p.neutral()) continue;
            int o = p.owner;
            if (o < 0 || o > 7) continue;
            if (besti[o] < 0 || bestprod[o] < (double)p.production) {
                bestprod[o] = p.production;
                besti[o] = i;
            }
        }
        M.w_home = (int16_t)besti[uteam & 7];
        for (int k = 0; k < 8; ++k)
            if (k != uteam && besti[k] >= 0) M.w_eteam = (int16_t)k;
        M.w_ehome = (int16_t)((M.w_eteam >= 0) ? besti[M.w_eteam] : -1);
        M.w_homes = 1;
    }
    const int home = M.w_home, ehome = M.w_ehome;

    // -- world classification (unchanged) ---------------------------------
    std::vector<FleetView> fleets;
    build_fleets(s, fleets);

    double ships = 0, total = 0, tprod = 0, myprod = 0;
    int    ecount = 0;
    double enemies[8], eprod[8];
    bool   ehas[8], ephas[8];
    for (int i = 0; i < 8; ++i) {
        enemies[i] = 0; eprod[i] = 0; ehas[i] = false; ephas[i] = false;
    }

    std::vector<int> lneutral, lmyplanets, lmyteam, leplanets, lothers;
    std::vector<int> lmyfleets, lefleets;

    for (int i = 0; i < NP; ++i) {          // PORT: index ascending
        const Planet& p = s.planets[i];
        double sh = p.ships;
        if (!p.neutral()) {
            int o = p.owner;
            if (o == uteam) {
                ships += sh;
            } else if (o >= 0 && o < 8) {
                if (!ehas[o]) { ehas[o] = true; enemies[o] = sh; ++ecount; }
                else            enemies[o] += sh;
            }
            total += sh;
        }
        if (p.neutral()) {
            lneutral.push_back(i);
            lothers.push_back(i);
        } else {
            tprod += p.production;
            if (p.owner == uteam) {
                lmyteam.push_back(i);
                myprod += p.production;
                lmyplanets.push_back(i);
            } else {
                int o = p.owner;
                if (o >= 0 && o < 8) {
                    if (!ephas[o]) { ephas[o] = true; eprod[o] = p.production; }
                    else             eprod[o] += p.production;
                }
                lothers.push_back(i);
                leplanets.push_back(i);
            }
        }
    }
    for (size_t fi = 0; fi < fleets.size(); ++fi) {   // PORT: fleet id ascending
        const FleetView& f = fleets[fi];
        int o = f.owner;
        if (o == uteam) ships += f.ships;
        else if (o >= 0 && o < 8) {
            if (!ehas[o]) { ehas[o] = true; enemies[o] = f.ships; ++ecount; }
            else            enemies[o] += f.ships;
        }
        total += f.ships;
        if (o == uteam) lmyfleets.push_back((int)fi);
        else            lefleets.push_back((int)fi);
    }

    // -- virtual survival credit ------------------------------------------
    // WOPT: threshold x behind_in_prod_threshold_factor; credit uses
    // pow(prod, exponent) x behind_in_prod_ships_increase_factor.
    if (ecount == 1 && ehome >= 0 &&
        tprod >= myprod * 2 * wopt::behind_in_prod_threshold_factor) {
        double dmin = kHuge;
        int closest = home;
        for (size_t ai = 0; ai < lmyplanets.size(); ++ai)
            for (size_t bi = 0; bi < leplanets.size(); ++bi) {
                double d = D(lmyplanets[ai], leplanets[bi]);
                if (dmin > d) { dmin = d; closest = lmyplanets[ai]; }
            }
        if (closest >= 0)
            ships += dmin *
                     std::pow((double)s.planets[closest].production,
                              wopt::behind_in_prod_nearest_enemy_prod_exponent) /
                     2000.0 * wopt::behind_in_prod_ships_increase_factor;
    }

    const double control = ships / total;

    // -- coalition heuristics (unchanged) ---------------------------------
    int big = -1;
    for (int e = 0; e < 8; ++e) {           // PORT: team id ascending
        if (!ehas[e]) continue;
        double v = enemies[e] / total;
        enemies[e] = v;
        if (v > .5) big = e;
        else if (v > .4) { if (big >= 0) big = -1; else big = e; }
    }
    double bigv = 0; int bigp = -1;
    for (int e = 0; e < 8; ++e) {
        if (!ephas[e]) continue;
        double n = eprod[e] / tprod;
        if (n > bigv) { bigv = n; bigp = e; }
    }
    double ships0 = 0, ships1 = 0;
    int big0 = -1, big1 = -1;
    for (int e = 0; e < 8; ++e) {
        if (!ehas[e]) continue;
        double v = enemies[e];
        if (v > ships0) { ships1 = ships0; ships0 = v; big1 = big0; big0 = e; }
        else if (v > ships1) { ships1 = v; big1 = e; }
    }
    if ((big >= 0 && (big1 >= 0 && control * 2 < ships1 && ships0 < .6)) ||
        (bigp >= 0 && big != bigp))
        big = -1;
    const double control2 = control / (1.0 - ships0);

    // -- first-call opening portfolio -------------------------------------
    if (FIRST && home >= 0 && !lneutral.empty() && ecount == 1) {
        auto pathf = [&](int f, int t, const std::vector<int>& set) -> int {
            double ft = D(f, t);
            for (size_t i = 0; i < set.size(); ++i) {
                int p = set[i];
                if (p != f && p != t) {
                    double pt = D(p, t);
                    if (pt < ft) {
                        double fp = D(f, p);
                        // WOPT: - tunneling_path_dist_discount
                        if (fp < ft && (pt + fp - PR(p) * 2.0 -
                                        wopt::tunneling_path_dist_discount) < ft) {
                            t = p; ft = fp;
                        }
                    }
                }
            }
            return t;
        };
        // WOPT: each term scaled by its recover_time_* factor. (The new
        // `else return HUGE` for a non-planet source is unreachable: a = home.)
        auto recovertime = [&](int a, int b) -> double {
            double r = (double)s.planets[b].ships - (double)s.planets[a].ships;
            double base = D(a, b) / 20.0 * wopt::recover_time_dist_factor +
                          (double)s.planets[b].ships * 50.0 /
                              (double)s.planets[b].production *
                              wopt::recover_time_target_prod_factor;
            if (r < 0) return base;
            return base + r * 50.0 / (double)s.planets[a].production *
                              wopt::recover_time_source_prod_factor;
        };
        auto pathlength = [&](int f, int t, const std::vector<int>& set) -> double {
            double d = 0;
            for (int i = 0; i < 20; ++i) {
                if (f == t) break;
                int F = pathf(f, t, set);
                d += D(f, F);
                f = F;
            }
            return d;
        };

        double eeval = kHuge;
        int enemy = -1;
        for (size_t i = 0; i < leplanets.size(); ++i) {
            double n = D(leplanets[i], home);
            if (eeval > n) { eeval = n; enemy = leplanets[i]; }
        }
        if (enemy < 0) {
            double e2 = -kHuge;
            for (size_t i = 0; i < lothers.size(); ++i) {
                double n = D(lothers[i], home);
                if (n > e2) { e2 = n; enemy = lothers[i]; }
            }
        }

        std::vector<int> p = lneutral;
        std::vector<double> rt(NP, 0.0);
        for (size_t i = 0; i < p.size(); ++i) rt[p[i]] = recovertime(home, p[i]);
        std::stable_sort(p.begin(), p.end(),     // PORT: stable, index order
                         [&](int a, int b) { return rt[a] < rt[b]; });

        std::vector<int> expand;
        double c = s.planets[home].ships;
        double cinit = c, excess = 0;
        // WOPT: x benefit_factor
        double benefit = (double)s.planets[home].production * D(enemy, home) /
                         2000.0 * wopt::benefit_factor;
        double prod = s.planets[home].production;
        for (size_t i = 0; i < p.size(); ++i) {
            int v = p[i];
            if (D(enemy, v) < D(home, v)) break;
            double sc = std::max(1.0, (double)s.planets[v].ships + 1.0);
            double ben = ((D(enemy, v) - pathlength(home, v, expand)) / 40.0) *
                             (double)s.planets[v].production / 50.0 - sc;
            double liability = sc - std::max(c, 0.0);
            // WOPT: weighted excess/liability, production ratio as pow()s
            if (c < sc)
                ben -= (excess * wopt::benefit_excess_factor +
                        liability * wopt::benefit_liability_factor) *
                       std::pow((double)s.planets[v].production,
                                wopt::expand_target_prod_exponent) /
                       std::pow(prod, wopt::expand_total_prod_exponent);
            if (benefit + ben + c > cinit) {
                if (c < sc) excess += liability;
                benefit += ben;
                c -= sc;
                if (c > 0) prod += (double)s.planets[v].production;
                expand.push_back(v);
            }
        }
        M.w_nexpand = (MemCount)std::min((size_t)kMaxMemPlanets, expand.size());
        for (int i = 0; i < (int)M.w_nexpand; ++i)
            M.w_expand[i] = (MemIdx)expand[(size_t)i];
        M.w_have_expand = 1;
        // WOPT: no `return` here -- the optimized bot also acts on its first
        // call. And `sort(p, ...)` sorted data.neutral IN PLACE, so the rest
        // of this call sees the neutrals in recovery-time order (it reaches
        // the late-expansion list's first-wins tie-break and swap-remove).
        lneutral = p;
    }

    // -- asset priority queue (unchanged) ---------------------------------
    struct Asset { bool planet; int idx; double ships; };
    std::vector<Asset> stuff;
    for (size_t i = 0; i < lmyplanets.size(); ++i)
        stuff.push_back(Asset{true, lmyplanets[i],
                              (double)s.planets[lmyplanets[i]].ships});
    for (size_t i = 0; i < lmyfleets.size(); ++i)
        stuff.push_back(Asset{false, lmyfleets[i], fleets[lmyfleets[i]].ships});
    std::stable_sort(stuff.begin(), stuff.end(),
                     [](const Asset& a, const Asset& b) { return a.ships > b.ships; });

    // WOPT: `targets` starts as a copy of data.eplanets, not an alias.
    std::vector<int> targets = leplanets;

    std::vector<double> pships(NP);
    for (int i = 0; i < NP; ++i) pships[i] = s.planets[i].ships;
    std::vector<int> ftarget(fleets.size());
    for (size_t i = 0; i < fleets.size(); ++i) ftarget[i] = fleets[i].target;

    // -- FFA fleet recall (unchanged; never fires in 1v1) -----------------
    bool recall = false;
    if (ecount > 1 && control2 < 1.0 / (double)ecount && big < 0) {
        for (size_t ai = 0; ai < stuff.size(); ++ai) {
            if (stuff[ai].planet) continue;
            int fi = stuff[ai].idx;
            if (s.planets[ftarget[fi]].owner == uteam) continue;
            double dmin = kHuge; int closest = -1;
            for (size_t pi = 0; pi < lmyplanets.size(); ++pi) {
                double d = dst(PX(lmyplanets[pi]), PY(lmyplanets[pi]),
                               fleets[fi].x, fleets[fi].y);
                if (dmin > d) { dmin = d; closest = lmyplanets[pi]; }
            }
            if (closest >= 0) {
                if (fleets[fi].target != closest)
                    galcon::redirect(s, fleets[fi].id, closest);
                return;
            }
        }
        recall = true;
    }

    const bool defend_global =
        (ecount == 1 && myprod / tprod > .5 && control < .8) || recall;

    // -- strategic target set ---------------------------------------------
    double bigsize = -1, smallsize = -1;
    int biggest = -1, smallest = -1;
    double bigprod = 0, smallprod = 0;
    for (int e = 0; e < 8; ++e) {
        if (!ehas[e]) continue;
        double v = enemies[e];
        if (v > bigsize) {
            smallsize = bigsize; bigsize = v * total;
            smallest = biggest;  biggest = e;
            smallprod = bigprod; bigprod = ephas[e] ? eprod[e] : 0.0;
        } else {
            smallest = e; smallsize = v * total;
            smallprod = ephas[e] ? eprod[e] : 0.0;
        }
    }
    (void)smallsize; (void)smallest; (void)bigprod; (void)smallprod;

    // Only the FFA ecount==2 && ships>bigsize branch still ALIASES
    // data.eplanets (see act_waffle); the alias is honoured where data.eplanets
    // is read again (late-expansion `enemy` scan).
    bool targets_alias_eplanets = false;
    bool default_ffa = false;   // WOPT: `default` FFA behaviour mode
    if (ecount < 2 || control > .5) {
        targets = leplanets;    // WOPT: element copy, no alias
    } else if (big >= 0) {
        targets.clear();
        for (size_t i = 0; i < leplanets.size(); ++i)
            if (s.planets[leplanets[i]].owner == big) targets.push_back(leplanets[i]);
    } else {
        // WOPT: FFA `else` restructure: ecount==2 / ecount>2 sub-branches,
        // then default = true and every neutral is admissible.
        if (ecount == 2) {
            if (ships > bigsize) {
                targets = leplanets;
                targets_alias_eplanets = true;
            } else {
                targets.clear();
                for (size_t i = 0; i < leplanets.size(); ++i)
                    if (s.planets[leplanets[i]].owner == biggest)
                        targets.push_back(leplanets[i]);
            }
        } else {
            // WOPT: `control > (enemies[p.owner] or 0)`. `enemies` is keyed
            // by TEAM id but indexed with p.owner, a USER id in the live
            // snapshot, so the lookup is always nil -> 0. Faithful to live:
            // every enemy planet passes when control > 0.
            targets.clear();
            for (size_t i = 0; i < leplanets.size(); ++i)
                if (control > 0.0) targets.push_back(leplanets[i]);
        }
        default_ffa = true;
        for (size_t i = 0; i < lneutral.size(); ++i) targets.push_back(lneutral[i]);
    }

    // -- unfinished opening portfolio stays admissible (unchanged) --------
    if (M.w_have_expand) {
        for (int i = 0; i < (int)M.w_nexpand; ++i) {
            int v = M.w_expand[i];
            if (v >= 0 && v < NP && s.planets[v].neutral()) targets.push_back(v);
        }
    }

    // -- majority-preserving late expansion -------------------------------
    // WOPT: band and admission thresholds scaled by the control_* factors.
    if (control > .5 * wopt::control_threshold_medium &&
        control < .8 * wopt::control_threshold_high) {
        double lships = ships, ltotal = total;
        std::vector<int> list = lneutral;
        int targ;
        do {
            targ = -1;
            double eeval = kHuge;
            int enemy = -1;
            const std::vector<int>& scan =
                targets_alias_eplanets ? targets : leplanets;
            for (size_t i = 0; i < scan.size(); ++i) {
                double n = D(scan[i], home);
                if (eeval > n) { eeval = n; enemy = scan[i]; }
            }
            if (enemy < 0) {
                double e2 = -kHuge;
                for (size_t i = 0; i < lothers.size(); ++i) {
                    double n = D(lothers[i], home);
                    if (n > e2) { e2 = n; enemy = lothers[i]; }
                }
            }
            double bestd = kHuge;
            for (size_t i = 0; i < list.size(); ++i) {
                int v = list[i];
                if (lships - (double)s.planets[v].ships * 1.5 *
                                 wopt::with_control_ships_factor >
                    ltotal / 2 * wopt::control_maintain_threshold_factor) {
                    double d = -(double)s.planets[v].production;
                    if (bestd > d && (enemy < 0 || d < D(enemy, v))) {
                        bestd = d; targ = v;
                    }
                }
            }
            if (targ >= 0) {
                targets.push_back(targ);
                lships -= (double)s.planets[targ].ships;
                ltotal -= (double)s.planets[targ].ships;
                for (size_t i = 0; i < list.size(); ++i)
                    if (list[i] == targ) {
                        list[i] = list.back();
                        list.pop_back();
                        break;
                    }
            }
        } while (targ >= 0);
    }

    // -- friendly-planet tunnelling ---------------------------------------
    // WOPT: new tunnel(): among own-team planets p ~= f whose detour
    // fp + pt - 2r - discount beats the direct distance, take the one NEAREST
    // the source (no fp<ft / pt<ft gates, t is not rebound mid-scan).
    auto tunnel = [&](double fx, double fy, int fplanet, int t) -> int {
        if (t < 0) return -1;
        double ft = dst(fx, fy, PX(t), PY(t));
        double closest = kHuge;
        int final_t = t;
        for (size_t i = 0; i < lmyteam.size(); ++i) {   // PORT: index ascending
            int p = lmyteam[i];
            double fp = dst(fx, fy, PX(p), PY(p));
            if (p != fplanet && fp < closest) {
                double pt = D(p, t);
                if (fp + pt - PR(p) * 2.0 - wopt::tunneling_path_dist_discount < ft) {
                    final_t = p; closest = fp;
                }
            }
        }
        return final_t;
    };

    // -- threat ledger, part 1 --------------------------------------------
    // planet -> sorted set of enemy fleet indices (Lua set efleetmap[t])
    std::vector<std::vector<int> > efmap(NP);
    std::vector<char> imminent(NP, 0);
    for (size_t i = 0; i < lefleets.size(); ++i) {
        int fi = lefleets[i];
        int t = fleets[fi].target;
        if (t < 0 || t >= NP) continue;
        efmap[t].push_back(fi);   // lefleets ascending -> stays sorted
        // WOPT: imminence radius x fleet_imminent_threshold_factor
        if (dst(fleets[fi].x, fleets[fi].y, PX(t), PY(t)) <
            100.0 * wopt::fleet_imminent_threshold_factor)
            imminent[t] = 1;
    }
    // WOPT: `enemyincoming` is accumulated (adding sync-id strings) but never
    // read; omitted. `attackclosest` is computed and then forced false; only
    // its false branch is ported.

    // -- tactical target choice -------------------------------------------
    // WOPT: rewritten GetTarget. Own-fleet reservations count only fleets
    // closer to the neutral than the source; a `defense`/`newtotal`/
    // `newcontrol` admission test gates targets in `default` (FFA) mode; and
    // the score is s - prod/4*k1 + dist/10*k2 (neutrals: dist/100*k3).
    auto GetTarget = [&](double fx, double fy, int self_fleet) -> int {
        if (recall) return -1;
        double teval = kHuge;
        int t = -1;
        for (size_t i = 0; i < targets.size(); ++i) {
            int v = targets[i];
            const Planet& P = s.planets[v];
            double sc = P.ships;
            double d = dst(fx, fy, (double)P.x, (double)P.y);
            if (P.neutral()) {
                for (size_t k = 0; k < lmyfleets.size(); ++k) {
                    int xi = lmyfleets[k];
                    if (xi != self_fleet && ftarget[xi] == v &&
                        dst(fleets[xi].x, fleets[xi].y, (double)P.x, (double)P.y) < d)
                        sc -= std::floor(fleets[xi].ships);
                }
            } else {
                sc += std::ceil((double)P.production * d / 2000.0);
            }
            // `(enemies[v.owner] or 0)/control`: owner is a user id, enemies
            // is keyed by team -> always 0, so the multiplier is max(1,0) = 1.
            double defense = (double)P.ships * 1.0 * wopt::target_defense_strength;
            double newtotal = total + myprod * d / 2000.0;
            for (int e = 0; e < 8; ++e)            // PORT: team id ascending
                if (ephas[e])
                    newtotal += eprod[e] * d / 2000.0 *
                                wopt::available_ships_target_factor_1;
            double newcontrol = (ships - defense) /
                                (newtotal - (double)P.ships * (P.neutral() ? 1.0 : 2.0));
            bool admit = std::floor(sc) >= 0;
            if (admit && default_ffa) {
                double ep = (big0 >= 0 && ephas[big0]) ? eprod[big0] : 0.0;
                admit = newcontrol / (1.0 - (ships0 * total - ep * d / 2000.0) / newtotal) >
                        1.0 / (double)ecount;
            }
            if (admit) {
                double n = sc - (double)P.production / 4.0 * wopt::fleet_target_prod_factor +
                           d / 10.0 * wopt::fleet_target_dist_factor;
                if (P.neutral()) n = d / 100.0 * wopt::fleet_neutral_target_dist_factor;
                if (teval > n) { teval = n; t = v; }
            }
        }
        return t;
    };

    // -- threat ledger, part 2 --------------------------------------------
    auto set_add = [](std::vector<int>& v, int x) {
        auto it = std::lower_bound(v.begin(), v.end(), x);
        if (it == v.end() || *it != x) v.insert(it, x);
    };
    // WOPT: an enemy fleet aimed at an own planet is re-charged to another
    // own planet within 40*fleet_target_factor_1 that is closer to the fleet.
    // PORT: (p, p2) both in index ascending order; sequential, so a fleet can
    // hop more than once.
    for (size_t a = 0; a < lmyplanets.size(); ++a) {
        int p = lmyplanets[a];
        for (size_t b = 0; b < lmyplanets.size(); ++b) {
            int p2 = lmyplanets[b];
            if (p != p2 && !efmap[p2].empty() &&
                D(p, p2) < 40.0 * wopt::fleet_target_factor_1) {
                std::vector<int> src = efmap[p2];
                for (size_t k = 0; k < src.size(); ++k) {
                    int fi = src[k];
                    if (dst(PX(p), PY(p), fleets[fi].x, fleets[fi].y) <
                        dst(PX(p2), PY(p2), fleets[fi].x, fleets[fi].y)) {
                        set_add(efmap[p], fi);
                        efmap[p2].erase(std::find(efmap[p2].begin(), efmap[p2].end(), fi));
                    }
                }
            }
        }
    }

    std::vector<double> danger(NP, 0.0), help(NP, 0.0);
    std::vector<char> has_danger(NP, 0);
    for (size_t pi = 0; pi < lmyplanets.size(); ++pi) {  // PORT: index ascending
        int i = lmyplanets[pi];
        double available = std::floor((double)s.planets[i].ships);
        double dmin = kHuge;
        // (opt.lua merges the neutrals' sets into efleetmap[f] itself; no
        // later read sees the difference, so a local union is equivalent)
        std::vector<int> set = efmap[i];
        for (size_t k = 0; k < lneutral.size(); ++k) {
            int n = lneutral[k];
            // WOPT: radius x fleet_target_factor_2
            if (!efmap[n].empty() && D(n, i) < 100.0 * wopt::fleet_target_factor_2)
                for (size_t q = 0; q < efmap[n].size(); ++q)
                    set.push_back(efmap[n][q]);
        }
        std::sort(set.begin(), set.end());
        set.erase(std::unique(set.begin(), set.end()), set.end());
        for (size_t k = 0; k < set.size(); ++k) {
            int fi = set[k];
            available -= std::ceil(fleets[fi].ships);
            // WOPT: radius x fleet_radius_dist_factor
            double d = dst(fleets[fi].x, fleets[fi].y, PX(i), PY(i)) -
                       PR(i) * wopt::fleet_radius_dist_factor;
            if (d < dmin) dmin = d;
        }
        if (dmin == kHuge) dmin = 0;
        double h = 0;
        for (size_t k = 0; k < lmyfleets.size(); ++k)
            if (ftarget[lmyfleets[k]] == i) h += fleets[lmyfleets[k]].ships;
        help[i] = h;
        danger[i] = std::floor(available +
                               (double)s.planets[i].production * dmin / 2000.0);
        has_danger[i] = 1;
    }

    // -- one-command scheduler --------------------------------------------
    std::vector<std::pair<bool, int> > selected;
    int maintarget = -1;
    bool have_percent = false;
    double percent = 0;

    for (size_t ai = 0; ai < stuff.size(); ++ai) {
        const Asset& A = stuff[ai];
        if (A.planet) {
            int fi = A.idx;
            double available = has_danger[fi] ? danger[fi] : pships[fi];
            if (!(available > 0)) continue;
            double fx = PX(fi), fy = PY(fi);
            int t0 = GetTarget(fx, fy, -1);
            int t  = tunnel(fx, fy, fi, t0);
            double low = 0; int t2 = -1;
            for (size_t k = 0; k < lmyplanets.size(); ++k) {
                int v = lmyplanets[k];
                // WOPT: danger/help weighted by danger_strength/help_strength
                double d = (has_danger[v] ? danger[v] : pships[v]) * wopt::danger_strength +
                           help[v] * wopt::help_strength;
                if (d < low) { low = d; t2 = v; }
            }
            bool def = defend_global;
            if (t2 >= 0 && imminent[t2]) def = true;
            // WOPT: defend rule. Switch to t2 also when t2 is near the target
            // (distance(t2,t)*defend_factor_2 < d); the no-target fallback no
            // longer excludes t2 == f (a self-send, which the engine allows).
            if (t2 >= 0 && def) {
                if (t >= 0) {
                    double d = dst(PX(t), PY(t), fx, fy);
                    if (!s.planets[t].neutral() &&
                        (d > dst(PX(t2), PY(t2), fx, fy) * 1.5 * wopt::defend_factor_1 ||
                         D(t2, t) * wopt::defend_factor_2 < d) &&
                        t2 != fi)
                        t = t2;
                } else {
                    t = t2;
                }
            }
            if (t < 0) continue;
            if (maintarget < 0) maintarget = t;
            if (maintarget != t && t0 >= 0) {
                // WOPT: slack 10 x main_target_distance_factor
                if (dst(fx, fy, PX(t0), PY(t0)) + 10.0 * wopt::main_target_distance_factor >
                    dst(fx, fy, PX(maintarget), PY(maintarget)) +
                        D(maintarget, t0) - PR(maintarget) * 2.0)
                    t = maintarget;
            }
            if (maintarget != t) continue;
            double fs = pships[fi];
            double a = std::floor(available * 20.0 / fs) * 5.0;
            if (s.planets[t].neutral())
                a = std::min(a, std::ceil(((double)s.planets[t].ships + 1.0) *
                                          20.0 / fs) * 10.0);
            if (!have_percent) { percent = a; have_percent = true; }
            if (percent <= a) {
                double ns = fs - std::floor(std::ceil(a * 20.0 / fs) * fs / 20.0 + 0.5);
                if (!std::isfinite(ns)) ns = 0.0;   // PORT: fs==0 -> inf*0 nan
                pships[fi] = ns;
                selected.push_back(std::make_pair(true, fi));
                if (t == t2) help[t] += pships[fi];
            }
        } else {
            int fi = A.idx;
            double fx = fleets[fi].x, fy = fleets[fi].y;
            int t = tunnel(fx, fy, -1, GetTarget(fx, fy, fi));
            double low = 0; int t2 = -1;
            int targ = ftarget[fi];
            for (size_t k = 0; k < lmyplanets.size(); ++k) {
                int v = lmyplanets[k];
                double d = (has_danger[v] ? danger[v] : pships[v]) + help[v];
                if (v == targ) d -= fleets[fi].ships;
                if (d < low) { low = d; t2 = v; }
            }
            bool def = defend_global;
            if (t2 >= 0 && imminent[t2]) def = true;
            // WOPT: fleet defend rule: plain 1.5, plus distance(t2,t) < d.
            if (t2 >= 0 && def) {
                if (t >= 0) {
                    double d = dst(PX(t), PY(t), fx, fy);
                    if (!s.planets[t].neutral() &&
                        (d > dst(PX(t2), PY(t2), fx, fy) * 1.5 || D(t2, t) < d))
                        t = t2;
                } else {
                    t = t2;
                }
            }
            if (t < 0 || ftarget[fi] == t) continue;
            if (maintarget < 0) maintarget = t;
            if (maintarget == t) {
                ftarget[fi] = t;
                selected.push_back(std::make_pair(false, fi));
                if (t == t2) help[t] += fleets[fi].ships;
            }
        }
    }

    if (maintarget < 0 || selected.empty()) return;

    int pct = snap_percent(have_percent ? percent : 50.0);
    for (size_t i = 0; i < selected.size(); ++i) {
        if (selected[i].first) {
            galcon::send(s, selected[i].second, maintarget, pct);
        } else {
            int fi = selected[i].second;
            if (fleets[fi].target != maintarget)
                galcon::redirect(s, fleets[fi].id, maintarget);
        }
    }
}

// ===========================================================================
//                                  SPARKY
//              transliteration of bots_sparky() in ../sparky.lua
// ===========================================================================
inline void act_sparky(State& s, int me, uint32_t /*seed*/, uint32_t /*tick*/) {
    // sparky.lua is fully deterministic; the seed is deliberately unused.
    const int NP = (int)s.planets.size();
    if (NP == 0 || !memory_fits(NP)) return;
    BotMemory& M = bot_memory(s, me);
    // MEM.cp: pure geometry, memoised outside the position (PORT DECISION 2).
    const std::vector<std::vector<int> >& CP = closest_lists(s);
    // packed accessors for the per-planet fields
    auto is_target  = [&](int n) { return bit_get(M.s_is_target, n); };
    auto set_target = [&](int n) { bit_put(M.s_is_target, n, true); };
    auto front_is_false = [&](int n) {
        return bit_get(M.s_front_set, n) && !bit_get(M.s_front_val, n);
    };
    auto set_front = [&](int n, int v) {          // -1 nil, 0 false, 1 true
        bit_put(M.s_front_set, n, v >= 0);
        bit_put(M.s_front_val, n, v == 1);
    };

    auto PX = [&](int i) { return (double)s.planets[i].x; };
    auto PY = [&](int i) { return (double)s.planets[i].y; };
    auto D  = [&](int a, int b) { return dst(PX(a), PY(a), PX(b), PY(b)); };

    std::vector<FleetView> fleets;
    build_fleets(s, fleets);

    // ---- one-time identification -------------------------------------
    if (M.s_phase == 0) {
        // MEM.pl is the planet id list sorted ascending; here that is 0..NP-1.
        for (int i = 0; i < NP; ++i) {
            const Planet& p = s.planets[i];
            if (p.owner == me) M.s_home = (int16_t)i;
            if (!p.neutral() && p.owner != me) {
                M.s_enemy = (int16_t)p.owner; M.s_ehome = (int16_t)i;
            }
        }
        M.s_phase = 1;   // arrays are already cleared by BotMemory's ctor
    }
    const int EN = M.s_enemy;
    if (M.s_home < 0 || M.s_ehome < 0 || EN < 0) return;   // nothing to plan on

    // ---- phase 1: closest-planet lists --------------------------------
    if (M.s_phase == 1) {
        // the closest-planet lists are already built (closest_lists above);
        // this phase exists only to spend the call the live bot spends on it
        M.s_phase = 2;
        if (kSparkyInitDelayTicks > 0) return;   // see PORT DECISION 1
    }

    auto future_ships = [&](int n, double time) {
        double t = s.planets[n].ships;
        if (!s.planets[n].neutral()) t += time * (double)s.planets[n].production * 0.02;
        return t;
    };
    auto amount_to_send = [&](int to, int from) {
        double d = D(to, from);
        double t = future_ships(to, d * 0.025) + 1.5;
        if (!s.planets[to].neutral())
            t += 0.00012 * (double)s.planets[to].production * d;
        return t;
    };
    auto worth_initially = [&](int n, double time, int home) {
        const Planet& p = s.planets[n];
        return (-(double)p.ships +
                (time - D(n, home) * 0.025) * (double)p.production * 0.02) /
                   ((double)p.ships + 0.1) > 0.3 &&
               (double)p.production / ((double)p.ships + 0.1) > 3;
    };
    auto worth_ever = [&](int n) {
        const Planet& p = s.planets[n];
        return (double)p.production / ((double)p.ships + 0.1) > 6 &&
               (double)p.ships < 7;
    };

    // recursive opening route split (original generate_tunnel_request).
    // Recursion strictly shortens both legs, so it terminates; the depth cap is
    // a safety valve only.
    std::function<void(int, int, double, int)> tunnel_request =
        [&](int from_n, int to_n, double amount, int depth) {
            if (depth > 64) return;
            const std::vector<int>& cp = CP[from_n];
            for (size_t i = 0; i < cp.size(); ++i) {
                int m = cp[i];
                if (m != to_n) {
                    if (is_target(m)) {
                        double d1 = D(from_n, m), d2 = D(m, to_n), d3 = D(from_n, to_n);
                        if (d2 < d3 && d1 < d3 && d1 + d2 < d3 * 1.25) {
                            tunnel_request(from_n, m, amount, depth + 1);
                            tunnel_request(m, to_n, amount, depth + 1);
                            return;
                        }
                    }
                } else {
                    // capacity is measured with headroom; a drop here would be
                    // a lost opening lane, so it must never happen in practice
                    if (M.s_nreq >= kMaxMemReq) return;
                    BotMemory::Req& r = M.s_req[M.s_nreq++];
                    r.f = (MemIdx)from_n; r.o = (MemIdx)to_n; r.a = amount;
                    return;
                }
            }
        };
    std::function<void(int, int, int)> target_tunnel =
        [&](int from_n, int to_n, int depth) {
            if (depth > 64) return;
            const std::vector<int>& cp = CP[from_n];
            for (size_t i = 0; i < cp.size(); ++i) {
                int m = cp[i];
                if (m != to_n) {
                    if (is_target(m)) {
                        double d1 = D(from_n, m), d2 = D(m, to_n), d3 = D(from_n, to_n);
                        if (d2 < d3 && d1 < d3 && d1 + d2 < d3 * 1.25) {
                            target_tunnel(from_n, m, depth + 1);
                            target_tunnel(m, to_n, depth + 1);
                            return;
                        }
                    }
                } else {
                    M.s_tgt[from_n] = (MemIdx)to_n;
                    return;
                }
            }
        };

    // ---- phase 2: horizon, worths, initial expansion tunnels ----------
    if (M.s_phase == 2) {
        M.s_horizon = D(M.s_home, M.s_ehome) * 0.025;
        for (int n = 0; n < NP; ++n) {
            double w = 1;
            for (int m = 0; m < NP; ++m) {
                if (m == n) continue;
                double v = -(double)s.planets[m].ships +
                           (double)s.planets[m].production *
                               (M.s_horizon - D(m, n) * 0.025);
                if (v > 0) w += v;
            }
            M.s_worth[n] = w;
        }
        M.s_worth_valid = 1;
        double h_to_e = D(M.s_home, M.s_ehome);
        const std::vector<int>& hcp = CP[M.s_home];
        for (size_t i = 0; i < hcp.size(); ++i) {
            int n = hcp[i];
            if (!s.planets[n].neutral()) continue;
            double h_to_n = D(M.s_home, n);
            if ((h_to_n < h_to_e && worth_ever(n)) ||
                worth_initially(n, M.s_horizon * 1.15, M.s_home)) {
                tunnel_request(M.s_home, n, amount_to_send(n, M.s_home), 0);
                set_target(n);
            }
        }
        M.s_first = 1;
        M.s_phase = 3;
        if (kSparkyInitDelayTicks > 1) return;   // see PORT DECISION 1
    }

    // ---- action queue -------------------------------------------------
    std::vector<double> left(NP, 0.0);
    std::vector<char> left_set(NP, 0);
    auto avail = [&](int n) -> double {
        if (!left_set[n]) { left[n] = s.planets[n].ships; left_set[n] = 1; }
        return left[n];
    };
    // MEM.q, a ring buffer inside the position. Amounts keep Lua's double
    // precision: storing them as float was measured to flip 2 games in 2000
    // (a 1e-7 perturbation occasionally crosses a 5%-snap boundary and the
    // game diverges from there), and the double form reproduces the validated
    // pre-refactor build bit for bit at a cost of 4% of clone throughput.
    auto q_push = [&](int32_t f, int o, double a, bool r) {
        if (M.s_nq >= kMaxMemQueue) return;      // measured never to happen
        BotMemory::Act& e = M.s_q[(M.s_q0 + M.s_nq) % kMaxMemQueue];
        e.f = f; e.a = a; e.o = (MemIdx)o; e.r = r ? 1 : 0;
        ++M.s_nq;
    };
    auto plan_send = [&](int f, int o, double amount) -> double {
        double av = avail(f);
        double sent = amount;
        if (av < sent) sent = av;
        if (sent <= 0) return 0.0;
        q_push(f, o, sent, false);
        left[f] = av - sent;
        return sent;
    };
    auto plan_redirect = [&](int32_t fleet_id, int o) {
        q_push((int32_t)fleet_id, o, 0.0, true);
    };

    std::map<int32_t, size_t> fleet_by_id;
    for (size_t i = 0; i < fleets.size(); ++i) fleet_by_id[fleets[i].id] = i;

    // emit one queued action against the live snapshot, dropping stale ones
    auto emit = [&]() {
        while (M.s_nq > 0) {
            BotMemory::Act a = M.s_q[M.s_q0];
            M.s_q0 = (MemCount)((M.s_q0 + 1) % kMaxMemQueue);
            --M.s_nq;
            if (a.r) {
                auto it = fleet_by_id.find((int32_t)a.f);
                if (it != fleet_by_id.end()) {
                    const FleetView& f = fleets[it->second];
                    if (f.owner == me && f.target != a.o) {
                        galcon::redirect(s, f.id, a.o);
                        return;
                    }
                }
            } else if (a.f >= 0 && a.f < NP && s.planets[a.f].owner == me &&
                       s.planets[a.f].ships > 0) {
                double perc = a.a / (double)s.planets[a.f].ships * 100.0;
                if (perc > 100) perc = 100;
                galcon::send(s, a.f, a.o, snap_percent(perc));
                return;
            }
        }
    };

    // ---- phase 3: drain the queue; re-plan only when it is empty -------
    if (M.s_nq > 0) { emit(); return; }

    // 1. refresh target set, reset per-turn net ships / under-attack
    std::vector<double> net(NP, 0.0);
    std::vector<char> ua(NP, 0);
    for (int n = 0; n < NP; ++n) {
        const Planet& p = s.planets[n];
        if (p.owner == me || p.owner == EN) set_target(n);
        net[n] = p.ships;
        ua[n] = 0;
    }

    // 2. classify front planets, set forward targets
    for (int n = 0; n < NP; ++n) {
        if (!is_target(n)) continue;
        if (s.planets[n].owner != EN) {
            const std::vector<int>& cp = CP[n];
            int ce = -1;
            for (size_t j = 0; j < cp.size(); ++j)
                if (s.planets[cp[j]].owner == EN) { ce = cp[j]; break; }
            if (ce < 0) continue;                 // front[n]/tgt[n] persist
            int target = -1;
            const std::vector<int>& ecp = CP[ce];
            for (size_t k = 0; k < ecp.size(); ++k) {
                int m = ecp[k];
                if (!is_target(m)) continue;
                double s_to_t = D(n, m), s_to_c = D(n, ce), t_to_c = D(ce, m);
                if (s_to_t < s_to_c && s_to_t + t_to_c < s_to_c * 1.25) {
                    target = m; break;
                }
            }
            if (target == n || target < 0) {
                set_front(n, 1);
            } else {
                set_front(n, 0);
                target_tunnel(n, target, 0);
            }
        } else {
            set_front(n, -1);
            M.s_tgt[n] = -1;
        }
    }

    // 3. feed outstanding expansion tunnels
    {
        int ti = 0;
        while (ti < (int)M.s_nreq) {
            BotMemory::Req& rq = M.s_req[ti];
            if (s.planets[rq.f].owner == me && net[rq.f] > 0) {
                double amt = rq.a;
                if (amt > net[rq.f]) amt = net[rq.f];
                double sent = plan_send(rq.f, rq.o, amt);
                rq.a = rq.a - sent;
                net[rq.f] -= sent;
            }
            if (rq.a <= 0) {
                for (int j = ti; j + 1 < (int)M.s_nreq; ++j)
                    M.s_req[j] = M.s_req[j + 1];
                --M.s_nreq;
            } else {
                ++ti;
            }
        }
    }

    // 4. first turn only: split the rest of home among forwarding planets
    if (M.s_first) {
        M.s_first = 0;
        std::vector<int> fp;
        double tw = 0;
        for (int n = 0; n < NP; ++n)
            if (is_target(n) && M.s_tgt[n] >= 0) {
                fp.push_back(n);
                tw += M.s_worth_valid ? (double)M.s_worth[n] : 1.0;
            }
        for (size_t i = 0; i < fp.size(); ++i)
            // reads the live (already reduced) home pool like the original
            plan_send(M.s_home, fp[i],
                      avail(M.s_home) *
                          (M.s_worth_valid ? (double)M.s_worth[fp[i]] : 1.0) / tw);
        M.s_worth_valid = 0;
    }

    // (sb_stats > 70 -> return emit() : never taken, see PORT DECISION 1)

    // 5. account inbound fleets into net ships; flag planets under attack
    struct Inc { int f; double d; };
    std::vector<std::vector<Inc> > inc(NP);
    for (size_t fi = 0; fi < fleets.size(); ++fi) {   // PORT: fleet id ascending
        const FleetView& f = fleets[fi];
        int tn = f.target;
        if (tn < 0 || tn >= NP) continue;
        const Planet& td = s.planets[tn];
        double d = dst(f.x, f.y, (double)td.x, (double)td.y);
        Inc e; e.f = (int)fi; e.d = d;
        inc[tn].push_back(e);
        int sign = (td.owner == f.owner) ? 1 : -1;
        if (td.owner == me && sign == -1) {
            int cf = -1; double cd = kHuge;
            for (int m = 0; m < NP; ++m)
                if (is_target(m) && s.planets[m].owner == me) {
                    double dd = dst(f.x, f.y, PX(m), PY(m));
                    if (dd < cd) { cd = dd; cf = m; }
                }
            if (cf >= 0) {
                double rsv = f.ships - (cd * 0.025 - 3) *
                                           (double)s.planets[cf].production * 0.02 + 1;
                if (rsv < 0) rsv = 0;
                net[cf] -= rsv * 0.3;
            }
            double rsv = f.ships - (d * 0.025 - 3) * (double)td.production * 0.02 + 1;
            if (rsv < 0) rsv = 0;
            net[tn] -= rsv * 0.7;
            ua[tn] = 1;
        } else {
            net[tn] += sign * f.ships;
        }
    }

    // 6. abandon contested expensive neutrals
    for (int n = 0; n < NP; ++n) {
        if (!is_target(n)) continue;
        const Planet& p = s.planets[n];
        if (!p.neutral() || !(p.ships > 6)) continue;
        std::vector<Inc>& L = inc[n];
        if (L.size() < 2) continue;
        std::stable_sort(L.begin(), L.end(),
                         [](const Inc& x, const Inc& y) { return x.d < y.d; });
        const FleetView& f1 = fleets[L[0].f];
        const FleetView& f2 = fleets[L[1].f];
        if (f1.owner != me || f2.owner != EN) continue;
        double t1 = L[0].d * 0.025, t2 = L[1].d * 0.025;
        if ((-(double)p.ships + f1.ships) + (t2 - t1) * (double)p.production * 0.02
                < f2.ships ||
            (t2 - t1 - 2) * (double)p.production * 0.02 < (double)p.ships) {
            int ce = -1;
            const std::vector<int>& cp = CP[n];
            for (size_t j = 0; j < cp.size(); ++j)
                if (s.planets[cp[j]].owner == EN) { ce = cp[j]; break; }
            if (ce >= 0) {
                plan_redirect(f1.id, ce);
                net[n] += f1.ships;
                net[ce] -= f1.ships;
            }
        }
    }

    // 7. planet-to-planet support for planets in the red
    for (int sn = 0; sn < NP; ++sn) {
        if (!is_target(sn)) continue;
        if (s.planets[sn].owner != me) continue;
        if (!(net[sn] < 0 || (ua[sn] && net[sn] < 2))) continue;
        for (int rep = 0; rep < 4; ++rep) {
            const std::vector<int>& cp = CP[sn];
            for (size_t j = 0; j < cp.size(); ++j) {
                int hn = cp[j];
                // NOTE the original's `else break`: the scan stops at the first
                // neighbour failing the guard, not merely skipping it.
                if (D(sn, hn) < 250 && net[sn] < 0 && net[hn] > 0) {
                    if (s.planets[hn].owner == me) {
                        double need = -net[sn];
                        if (need > net[hn] * 0.3333 + 1) need = net[hn] * 0.3333 + 1;
                        if (!(ua[hn] && net[hn] - need < 2)) {
                            double sent = plan_send(hn, sn, need);
                            net[hn] -= sent;
                            net[sn] += sent;
                        }
                    }
                } else {
                    break;
                }
            }
        }
    }

    // (sb_stats > 70 -> return emit() : never taken, see PORT DECISION 1)

    // 8. offensive / efficiency fleet redirects
    auto planet_worth = [&](int to, double fx, double fy) {
        return -dst(PX(to), PY(to), fx, fy) * 0.09091 -
               2.2 * (double)s.planets[to].ships +
               (double)s.planets[to].production * 0.2;
    };
    for (size_t fi = 0; fi < fleets.size(); ++fi) {
        const FleetView& f = fleets[fi];
        if (f.owner != me) continue;
        int tn = f.target;
        if (tn < 0 || tn >= NP) continue;
        if (s.planets[tn].owner == EN) {
            if (net[tn] > 0 || net[tn] + f.ships < 0) {
                double tworth = planet_worth(tn, f.x, f.y) - 1;
                int best = tn;
                const std::vector<int>& cp = CP[tn];
                for (size_t j = 0; j < cp.size(); ++j) {
                    int m = cp[j];
                    if (dst(PX(m), PY(m), f.x, f.y) < 300 &&
                        s.planets[m].owner == EN) {
                        double w = planet_worth(m, f.x, f.y);
                        if (w > tworth && net[m] < net[tn]) { tworth = w; best = m; }
                    }
                }
                if (best != tn) {
                    plan_redirect(f.id, best);
                    net[tn] += f.ships;
                    // the original compares an owner id with a planet id, which
                    // can never be equal in the harness; kept dead on purpose.
                    net[best] += -1.0 * f.ships;
                }
            }
        } else if (s.planets[tn].owner == me) {
            if (net[tn] - f.ships > 0 && M.s_tgt[tn] >= 0) {
                int fw = M.s_tgt[tn];
                double d1 = dst(f.x, f.y, PX(tn), PY(tn));
                double d2 = D(tn, fw);
                double d3 = dst(f.x, f.y, PX(fw), PY(fw));
                if (!(d2 < d3 && d1 < d3 && d1 + d2 < d3 * 1.25)) {
                    plan_redirect(f.id, fw);
                    net[fw] -= f.ships;
                    net[tn] -= f.ships;
                }
            }
        }
    }

    // 9. spend what's left: forward from rear planets, attack from the front
    for (int sn = 0; sn < NP; ++sn) {
        if (!is_target(sn)) continue;
        if (s.planets[sn].owner != me) continue;
        if (front_is_false(sn)) {
            if (net[sn] > 0 && M.s_tgt[sn] >= 0) {
                plan_send(sn, M.s_tgt[sn], net[sn]);
                net[sn] = 0;
            }
        } else {
            const std::vector<int>& cp = CP[sn];
            for (size_t j = 0; j < cp.size(); ++j) {
                int en = cp[j];
                if (s.planets[en].owner != EN) continue;
                M.s_tgt[sn] = (MemIdx)en;
                double need = amount_to_send(en, sn);
                if (need > net[sn]) {
                    if (net[sn] > 0) { plan_send(sn, en, net[sn]); net[sn] = 0; }
                    break;
                } else {
                    double d = D(en, sn);
                    double time = d / 40 + 1.5 +
                                  0.00012 * (double)s.planets[en].production * d;
                    double amt = net[en] + time * (double)s.planets[en].production / 50.0;
                    if (amt < 0) amt = 0;
                    double sent = plan_send(sn, en, amt);
                    net[sn] -= sent;
                    net[en] -= sent;
                }
            }
        }
    }

    emit();
}

// ===========================================================================
//                       SPARKY NATIVE (v1.6, unrestricted)
//              transliteration of sparkyn_loop() in ../sparkyn.lua
// ===========================================================================
// The original Sparky v1.6 as it runs natively in Galcon 2: every action is
// executed the moment it is decided (many fleet_send / fleet_redirect calls
// per turn, exact fractional percents), so every later decision in the same
// turn sees the engine state after the earlier ones -- the source planet's
// ships drop at once and the new fleet shows up in later fleet searches.
// STANDALONE constants only (sparkyn.lua's SPARKYN_ONLINE / ONLINE_* /
// PORT_SEND_BUG branches are all false here).
//
// Conventions are act_sparky's (PORT DECISIONS 1-4: no sb_stats, planets
// index-ascending, fleets id-ascending, stable sorts, doubles); the
// differences from act_sparky are the native ones:
//
// PORT N1 (memory). A seat runs one bot, so this reuses BotMemory's sparky
//   fields, which hold exactly Sparky v1.6's state (is_target, front, target,
//   worth, tunnel requests, home/enemy, horizon). s_phase == kNativePhase marks
//   them as initialised by this function; the FIFO fields (s_q*) are unused.
//   BotMemory's layout is unchanged, so forks cost what they did.
// PORT N2 (init). No phases and no init delay: the native bot does its whole
//   setup and its first main loop in the first call (it has no sb_stats).
//   home / enemy home are the FIRST planet the user / enemy owns (sparkyn.lua
//   `user_planets[1]`); the enemy is the owner of the LAST non-neutral planet
//   not ours (its planet-ownership loop). Worth sums run in closest-planet
//   order, as the Lua `pairs(planet_data.closest_planets)` does.
// PORT N3 (send size). send_exact (bot side) follows sparkyn.lua: if the
//   planet has fewer ships than asked it sends 100% (galcon::send_exact,
//   n = floor(ships+0.5) in float) and books the pre-send garrison as sent;
//   else it sends `ships/ships_value*100` percent and books `ships`. That
//   partial send launches n = floor(ships + 0.5): the value of the engine
//   formula floor(sv*(ships/sv*100)/100 + 0.5) in real arithmetic. In Galcon
//   2's float path the product can land one ulp under an exact .5 (common: a
//   neutral garrison + the 1.5 pad), where the outcome hangs on unmeasured
//   engine float internals; the real-arithmetic value is deterministic and
//   reproducible in VEX's 32-bit floats.
// PORT N4 (new fleets). A fleet launched during the call is seen, for the
//   rest of the call, at its nominal position src + u*(radius(src) + 6)
//   (launch point plus the six free unit_move steps, u the unit bearing to the
//   target), not at the rand()-sampled centroid of its units: the native item
//   position at creation is undocumented, and the nominal point is
//   deterministic, so the VEX port can reproduce it. Fleets from earlier calls
//   are seen as usual (centroid).
// PORT N5 (dead code kept). sparkyn.lua's `planet_to_attack.owner_n ==
//   fleet.fleet_target` compares a user id with a planet id: never equal.
//   An unreachable nil `target` in the front loop (the source itself always
//   qualifies) is treated as front, like act_sparky.
// Cadence: native runs every 0.2 s; the RL env calls every 0.25 s (kCadence).
inline constexpr int8_t kNativePhase = 100;

inline void act_sparky_native(State& s, int me, uint32_t /*seed*/, uint32_t /*tick*/) {
    const int NP = (int)s.planets.size();
    if (NP == 0 || !memory_fits(NP)) return;
    BotMemory& M = bot_memory(s, me);
    const std::vector<std::vector<int> >& CP = closest_lists(s);
    auto is_target  = [&](int n) { return bit_get(M.s_is_target, n); };
    auto set_target = [&](int n) { bit_put(M.s_is_target, n, true); };
    auto front_is_false = [&](int n) {
        return bit_get(M.s_front_set, n) && !bit_get(M.s_front_val, n);
    };
    auto set_front = [&](int n, int v) {          // -1 nil, 0 false, 1 true
        bit_put(M.s_front_set, n, v >= 0);
        bit_put(M.s_front_val, n, v == 1);
    };
    auto PX = [&](int i) { return (double)s.planets[i].x; };
    auto PY = [&](int i) { return (double)s.planets[i].y; };
    auto D  = [&](int a, int b) { return dst(PX(a), PY(a), PX(b), PY(b)); };
    auto SH = [&](int i) { return (double)s.planets[i].ships; };

    auto future_ships = [&](int n, double time) {
        double t = SH(n);
        if (!s.planets[n].neutral()) t += time * (double)s.planets[n].production * 0.02;
        return t;
    };
    auto amount_to_send = [&](int to, int from) {
        double d = D(to, from);
        double t = future_ships(to, d * 0.025) + 1.5;
        if (!s.planets[to].neutral())
            t += 0.00012 * (double)s.planets[to].production * d;
        return t;
    };
    auto worth_initially = [&](int n, double time, int home) {
        const Planet& p = s.planets[n];
        return (-(double)p.ships +
                (time - D(n, home) * 0.025) * (double)p.production * 0.02) /
                   ((double)p.ships + 0.1) > 0.3 &&
               (double)p.production / ((double)p.ships + 0.1) > 3;
    };
    auto worth_ever = [&](int n) {
        const Planet& p = s.planets[n];
        return (double)p.production / ((double)p.ships + 0.1) > 6 &&
               (double)p.ships < 7;
    };
    std::function<void(int, int, double, int)> tunnel_request =
        [&](int from_n, int to_n, double amount, int depth) {
            if (depth > 64) return;
            const std::vector<int>& cp = CP[from_n];
            for (size_t i = 0; i < cp.size(); ++i) {
                int m = cp[i];
                if (m != to_n) {
                    if (is_target(m)) {
                        double d1 = D(from_n, m), d2 = D(m, to_n), d3 = D(from_n, to_n);
                        if (d2 < d3 && d1 < d3 && d1 + d2 < d3 * 1.25) {
                            tunnel_request(from_n, m, amount, depth + 1);
                            tunnel_request(m, to_n, amount, depth + 1);
                            return;
                        }
                    }
                } else {
                    if (M.s_nreq >= kMaxMemReq) return;
                    BotMemory::Req& r = M.s_req[M.s_nreq++];
                    r.f = (MemIdx)from_n; r.o = (MemIdx)to_n; r.a = amount;
                    return;
                }
            }
        };
    std::function<void(int, int, int)> target_tunnel =
        [&](int from_n, int to_n, int depth) {
            if (depth > 64) return;
            const std::vector<int>& cp = CP[from_n];
            for (size_t i = 0; i < cp.size(); ++i) {
                int m = cp[i];
                if (m != to_n) {
                    if (is_target(m)) {
                        double d1 = D(from_n, m), d2 = D(m, to_n), d3 = D(from_n, to_n);
                        if (d2 < d3 && d1 < d3 && d1 + d2 < d3 * 1.25) {
                            target_tunnel(from_n, m, depth + 1);
                            target_tunnel(m, to_n, depth + 1);
                            return;
                        }
                    }
                } else {
                    M.s_tgt[from_n] = (MemIdx)to_n;
                    return;
                }
            }
        };

    // ---- first call: identification, horizon, worths, opening tunnels ---
    if (M.s_phase != kNativePhase) {
        int en = -1, home = -1, ehome = -1;
        for (int i = 0; i < NP; ++i)
            if (!s.planets[i].neutral() && s.planets[i].owner != me) en = s.planets[i].owner;
        if (en < 0) return;                       // the Lua would error; retry
        for (int i = 0; i < NP && home < 0; ++i) if (s.planets[i].owner == me) home = i;
        for (int i = 0; i < NP && ehome < 0; ++i) if (s.planets[i].owner == en) ehome = i;
        if (home < 0 || ehome < 0) return;
        M = BotMemory();
        M.s_phase = kNativePhase;
        M.s_home = (int16_t)home; M.s_ehome = (int16_t)ehome; M.s_enemy = (int16_t)en;
        M.s_horizon = D(home, ehome) * 0.025;
        for (int n = 0; n < NP; ++n) {
            double w = 1;
            const std::vector<int>& cp = CP[n];
            for (size_t j = 0; j < cp.size(); ++j) {
                int m = cp[j];
                double v = -SH(m) + (double)s.planets[m].production *
                                        (M.s_horizon - D(m, n) * 0.025);
                if (v > 0) w += v;
            }
            M.s_worth[n] = w;
        }
        double h_to_e = D(home, ehome);
        const std::vector<int>& hcp = CP[home];
        for (size_t i = 0; i < hcp.size(); ++i) {
            int n = hcp[i];
            if (!s.planets[n].neutral()) continue;
            double h_to_n = D(home, n);
            if ((h_to_n < h_to_e && worth_ever(n)) ||
                worth_initially(n, M.s_horizon * 1.15, home)) {
                tunnel_request(home, n, amount_to_send(n, home), 0);
                set_target(n);
            }
        }
        M.s_first = 1;
    }
    const int EN = M.s_enemy;

    // ---- immediate execution (PORT N3, N4) --------------------------------
    struct Born { int32_t id; double x, y; };
    std::vector<Born> born;
    auto fleet_send = [&](int from, int to, bool all, double ships) {
        int32_t id = s.next_fleet;
        int n;
        if (all) n = galcon::send_exact(s, from, to, 100.0);
        else     n = galcon::send_count(s, from, to, (int)std::floor(ships + 0.5),
                                        (int)std::lround(ships / SH(from) * 100.0));
        if (n <= 0) return;
        double dx = PX(to) - PX(from), dy = PY(to) - PY(from);
        double c = std::sqrt(dx * dx + dy * dy);
        double k = (double)s.planets[from].radius() + 6.0;
        Born b; b.id = id; b.x = PX(from); b.y = PY(from);
        if (c > 0) { b.x += dx / c * k; b.y += dy / c * k; }
        born.push_back(b);
    };
    // sparkyn.lua bot.send_exact: returns what the caller books as sent
    auto send_exact = [&](int from, int to, double ships) -> double {
        double sv = SH(from);
        if (sv < ships) { fleet_send(from, to, true, 0.0); return sv; }
        if (!(sv > 0)) return ships;               // 0/0 percent: nothing launches
        fleet_send(from, to, false, ships);
        return ships;
    };
    std::vector<FleetView> fleets;
    auto search_fleets = [&]() {                    // g2.search("fleet"), live
        build_fleets(s, fleets);
        for (FleetView& f : fleets)
            for (const Born& b : born)
                if (b.id == f.id) { f.x = b.x; f.y = b.y; }
    };

    // 1. refresh target set, reset per-turn net ships / under-attack
    std::vector<double> net(NP, 0.0);
    std::vector<char> ua(NP, 0);
    for (int n = 0; n < NP; ++n) {
        const Planet& p = s.planets[n];
        if (p.owner == me || p.owner == EN) set_target(n);
        net[n] = p.ships;
        ua[n] = 0;
    }

    // 2. classify front planets, set forward targets
    for (int n = 0; n < NP; ++n) {
        if (!is_target(n)) continue;
        if (s.planets[n].owner != EN) {
            const std::vector<int>& cp = CP[n];
            int ce = -1;
            for (size_t j = 0; j < cp.size(); ++j)
                if (s.planets[cp[j]].owner == EN) { ce = cp[j]; break; }
            if (ce < 0) continue;                 // front[n]/tgt[n] persist
            int target = -1;
            const std::vector<int>& ecp = CP[ce];
            for (size_t k = 0; k < ecp.size(); ++k) {
                int m = ecp[k];
                if (!is_target(m)) continue;
                double s_to_t = D(n, m), s_to_c = D(n, ce), t_to_c = D(ce, m);
                if (s_to_t < s_to_c && s_to_t + t_to_c < s_to_c * 1.25) {
                    target = m; break;
                }
            }
            if (target == n || target < 0) {
                set_front(n, 1);
            } else {
                set_front(n, 0);
                target_tunnel(n, target, 0);
            }
        } else {
            set_front(n, -1);
            M.s_tgt[n] = -1;
        }
    }

    // 3. feed outstanding expansion tunnels (sends execute now)
    {
        int ti = 0;
        while (ti < (int)M.s_nreq) {
            BotMemory::Req& rq = M.s_req[ti];
            bool drop = false;
            if (s.planets[rq.f].owner == me && net[rq.f] > 0) {
                double amt = rq.a;
                if (amt > net[rq.f]) amt = net[rq.f];
                double sent = send_exact(rq.f, rq.o, amt);
                rq.a = rq.a - sent;
                net[rq.f] -= sent;
                drop = rq.a <= 0;
            }
            if (drop) {
                for (int j = ti; j + 1 < (int)M.s_nreq; ++j)
                    M.s_req[j] = M.s_req[j + 1];
                --M.s_nreq;
            } else {
                ++ti;
            }
        }
    }

    // 4. first turn only: split home among the planets that have a target
    if (M.s_first) {
        M.s_first = 0;
        std::vector<int> fp;
        double tw = 0;
        for (int n = 0; n < NP; ++n)
            if (is_target(n) && M.s_tgt[n] >= 0) {
                fp.push_back(n);
                tw += M.s_worth[n];
            }
        for (size_t i = 0; i < fp.size(); ++i)
            // home.ships_value is live: each send shrinks the next share
            send_exact(M.s_home, fp[i], SH(M.s_home) * M.s_worth[fp[i]] / tw);
    }

    // 5. account inbound fleets into net ships; flag planets under attack.
    //    Per target planet (index), its fleets id-ascending, like the Lua's
    //    g2.search("fleet target:"..planet) inside pairs(bot.planets).
    search_fleets();
    struct Inc { int f; double d; };
    std::vector<std::vector<Inc> > inc(NP);
    for (int tn = 0; tn < NP; ++tn) {
        const Planet& td = s.planets[tn];
        for (size_t fi = 0; fi < fleets.size(); ++fi) {
            const FleetView& f = fleets[fi];
            if (f.target != tn) continue;
            double d = dst(f.x, f.y, (double)td.x, (double)td.y);
            Inc e; e.f = (int)fi; e.d = d;
            inc[tn].push_back(e);
            int sign = (td.owner == f.owner) ? 1 : -1;
            if (td.owner == me && sign == -1) {
                int cf = -1; double cd = kHuge;
                for (int m = 0; m < NP; ++m)
                    if (is_target(m) && s.planets[m].owner == me) {
                        double dd = dst(f.x, f.y, PX(m), PY(m));
                        if (dd < cd) { cd = dd; cf = m; }
                    }
                if (cf >= 0) {
                    double rsv = f.ships - (cd * 0.025 - 3) *
                                               (double)s.planets[cf].production * 0.02 + 1;
                    if (rsv < 0) rsv = 0;
                    net[cf] -= rsv * 0.3;
                }
                double rsv = f.ships - (d * 0.025 - 3) * (double)td.production * 0.02 + 1;
                if (rsv < 0) rsv = 0;
                net[tn] -= rsv * 0.7;
                ua[tn] = 1;
            } else {
                net[tn] += sign * f.ships;
            }
        }
    }
    for (int n = 0; n < NP; ++n)
        std::stable_sort(inc[n].begin(), inc[n].end(),
                         [](const Inc& x, const Inc& y) { return x.d < y.d; });

    // 6. abandon contested expensive neutrals (redirect executes now)
    for (int n = 0; n < NP; ++n) {
        if (!is_target(n)) continue;
        const Planet& p = s.planets[n];
        if (!p.neutral() || !(p.ships > 6)) continue;
        const std::vector<Inc>& L = inc[n];
        if (L.size() < 2) continue;
        const FleetView& f1 = fleets[L[0].f];
        const FleetView& f2 = fleets[L[1].f];
        if (f1.owner != me || f2.owner != EN) continue;
        double t1 = L[0].d * 0.025, t2 = L[1].d * 0.025;
        if ((-(double)p.ships + f1.ships) + (t2 - t1) * (double)p.production * 0.02
                < f2.ships ||
            (t2 - t1 - 2) * (double)p.production * 0.02 < (double)p.ships) {
            int ce = -1;
            const std::vector<int>& cp = CP[n];
            for (size_t j = 0; j < cp.size(); ++j)
                if (s.planets[cp[j]].owner == EN) { ce = cp[j]; break; }
            if (ce >= 0) {
                galcon::redirect(s, f1.id, ce);
                net[n] += f1.ships;
                net[ce] -= f1.ships;
            }
        }
    }

    // 7. planet-to-planet support for planets in the red
    for (int sn = 0; sn < NP; ++sn) {
        if (!is_target(sn)) continue;
        if (s.planets[sn].owner != me) continue;
        if (!(net[sn] < 0 || (ua[sn] && net[sn] < 2))) continue;
        for (int rep = 0; rep < 4; ++rep) {
            const std::vector<int>& cp = CP[sn];
            for (size_t j = 0; j < cp.size(); ++j) {
                int hn = cp[j];
                if (D(sn, hn) < 250 && net[sn] < 0 && net[hn] > 0) {
                    if (s.planets[hn].owner == me) {
                        double need = -net[sn];
                        if (need > net[hn] * 0.3333 + 1) need = net[hn] * 0.3333 + 1;
                        if (!(ua[hn] && net[hn] - need < 2)) {
                            double sent = send_exact(hn, sn, need);
                            net[hn] -= sent;
                            net[sn] += sent;
                        }
                    }
                } else {
                    break;
                }
            }
        }
    }

    // 8. offensive / efficiency fleet redirects over the live fleet list
    search_fleets();
    auto planet_worth = [&](int to, double fx, double fy) {
        return -dst(PX(to), PY(to), fx, fy) * 0.09091 - 2.2 * SH(to) +
               (double)s.planets[to].production * 0.2;
    };
    for (size_t fi = 0; fi < fleets.size(); ++fi) {
        const FleetView& f = fleets[fi];
        if (f.owner != me) continue;
        int tn = f.target;
        if (tn < 0 || tn >= NP) continue;
        if (s.planets[tn].owner == EN) {
            if (net[tn] > 0 || net[tn] + f.ships < 0) {
                double tworth = planet_worth(tn, f.x, f.y) - 1;
                int best = tn;
                const std::vector<int>& cp = CP[tn];
                for (size_t j = 0; j < cp.size(); ++j) {
                    int m = cp[j];
                    if (dst(PX(m), PY(m), f.x, f.y) < 300 &&
                        s.planets[m].owner == EN) {
                        double w = planet_worth(m, f.x, f.y);
                        if (w > tworth && net[m] < net[tn]) { tworth = w; best = m; }
                    }
                }
                if (best != tn) {
                    galcon::redirect(s, f.id, best);
                    net[tn] += f.ships;
                    net[best] += -1.0 * f.ships;     // PORT N5: sign is always -1
                }
            }
        } else if (s.planets[tn].owner == me) {
            if (net[tn] - f.ships > 0 && M.s_tgt[tn] >= 0) {
                int fw = M.s_tgt[tn];
                double d1 = dst(f.x, f.y, PX(tn), PY(tn));
                double d2 = D(tn, fw);
                double d3 = dst(f.x, f.y, PX(fw), PY(fw));
                if (!(d2 < d3 && d1 < d3 && d1 + d2 < d3 * 1.25)) {
                    galcon::redirect(s, f.id, fw);
                    net[fw] -= f.ships;
                    net[tn] -= f.ships;
                }
            }
        }
    }

    // 9. spend what's left: forward from rear planets, attack from the front
    for (int sn = 0; sn < NP; ++sn) {
        if (!is_target(sn)) continue;
        if (s.planets[sn].owner != me) continue;
        if (front_is_false(sn)) {
            if (net[sn] > 0 && M.s_tgt[sn] >= 0) {
                send_exact(sn, M.s_tgt[sn], net[sn]);
                net[sn] = 0;
            }
        } else {
            const std::vector<int>& cp = CP[sn];
            for (size_t j = 0; j < cp.size(); ++j) {
                int en = cp[j];
                if (s.planets[en].owner != EN) continue;
                M.s_tgt[sn] = (MemIdx)en;
                double need = amount_to_send(en, sn);
                if (need > net[sn]) {
                    if (net[sn] > 0) { send_exact(sn, en, net[sn]); net[sn] = 0; }
                    break;
                } else {
                    double d = D(en, sn);
                    double time = d / 40 + 1.5 +
                                  0.00012 * (double)s.planets[en].production * d;
                    double amt = net[en] + time * (double)s.planets[en].production / 50.0;
                    if (amt < 0) amt = 0;
                    double sent = send_exact(sn, en, amt);
                    net[sn] -= sent;
                    net[en] -= sent;
                }
            }
        }
    }
}

}  // namespace galbots

// ===========================================================================
// FIDELITY RISK LIST -- every place this port still has to guess.
//
// Two former entries have been resolved and are recorded here so they are not
// re-raised:
//
//   RESOLVED - sb_stats(). Dropping every scan cutoff is now a deliberate
//   design decision, not a risk: the instruction budget is an online
//   crash-safety limit with nothing to protect in the replica, and running the
//   author's algorithm to completion is the better rendering of intent even
//   though it likely makes this Waffle a shade stronger than the ladder one.
//   See PORT DECISION 1.
//
//   RESOLVED - forking. Bot memory now lives in State and travels with the
//   position, so a rollout forked mid-game continues the plan the bot really
//   had. Verified: clone-then-step is identical to step over 375 mid-game
//   positions; 333 sibling-fork triples and 250 interleaved fork pairs show no
//   contamination; 524 forks show no opening restart while a blanked-memory
//   control restarts 524/524. See PORT DECISION 2.
//
// Remaining, in rough order of how much they could matter:
//
// 1. Sparky's init pacing is emulated as a flat two-call delay
//    (kSparkyInitDelayTicks); the real sb_stats cost depends on planet count.
//    Measured sensitivity: about 2 points of the Waffle-Sparky matchup.
// 2. Fleet aggregation. Lua sees one aggregate fleet Item per send; the
//    replica stores loose units, rebuilt here as centroid + summed ships per
//    fleet id, which research/engine-rules.md says is exactly what the engine
//    reports. But kFlightElongation spreads a fleet over roughly
//    +/-6*sqrt(n)*1.637 of track, so on short routes the centroid is a poor
//    stand-in for "where the fleet is". Waffle's 100-unit `imminent` radius
//    and its `distance(fleet, planet) - r` threat distance both read that
//    centroid, so its threat ledger is distorted on short lanes.
// 3. pairs() order -> index / fleet-id ascending; table.sort -> stable_sort.
//    Affects ties only, but it makes this port strictly more deterministic
//    than the original, which cannot reproduce Lua's tie-break variance.
// 4. Memory capacity is a compile-time bound (GALCON_MAX_MEM_PLANETS, default
//    64). Measured peaks leave large headroom up to 48 planets; a map beyond
//    the bound makes both bots idle with a one-shot warning rather than
//    silently truncate.
// 5. Waffle line 498's snapshot mutation produces nan when the source planet
//    holds exactly 0 ships but positive danger; replaced with 0.
// 6. Nothing here has been checked against the Lua at DECISION level. The
//    strongest available next test is to extend difftest_capture.lua to record
//    each bot's chosen {from, to, percent} per 4 Hz call and diff it against
//    these functions on the same state.
// ===========================================================================
