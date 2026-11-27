// Auto-compiled C++17 bridge exposing custom Galcon C++ bot ports to LightZero.
//
// Compile:
//   g++ -O3 -march=native -ffp-contract=off -shared -fPIC -std=c++17 bots_bridge.cpp -lm -o bots_bridge.so
//
// Wraps waffle3z's faithful C++ ports of classic Galcon ladder bots (Waffle,
// WaffleOpt, Sparky 4 Hz, Native Sparky v1.6) plus C++ arena heuristics
// (greedy, expand, pressure, arcade, counter) behind a stateful C ABI session
// so BotMemory persists cleanly across ticks of an episode.

#define SPARKY_INIT_DELAY_TICKS 0
#include "bots.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using namespace galcon;

enum BotKind : int32_t {
    BOT_WAFFLE = 0,
    BOT_WAFFLE_OPT = 1,
    BOT_SPARKY = 2,
    BOT_SPARKY_NATIVE = 3,
    BOT_GREEDY = 4,
    BOT_EXPAND = 5,
    BOT_PRESSURE = 6,
    BOT_ARCADE = 7,
    BOT_COUNTER = 8,
};

struct BotSession {
    int32_t kind = BOT_WAFFLE;
    BotMemory mem;
};

inline uint32_t hash32(uint32_t a, uint32_t b, uint32_t c) {
    uint64_t x = (uint64_t)a * 0x9E3779B97F4A7C15ull
               ^ (uint64_t)b * 0xBF58476D1CE4E5B9ull
               ^ (uint64_t)c * 0x94D049BB133111EBull;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    return (uint32_t)(x >> 16);
}

static void collect_planets(const State& s, int me, std::vector<int>& mine,
                            std::vector<int>& other, float min_ships) {
    mine.clear();
    other.clear();
    for (size_t i = 0; i < s.planets.size(); ++i) {
        if (s.planets[i].owner == me) {
            if (s.planets[i].ships > min_ships) mine.push_back((int)i);
        } else {
            other.push_back((int)i);
        }
    }
}

static void act_greedy(State& s, int me, uint32_t seed, uint32_t tick) {
    std::vector<int> mine, other;
    collect_planets(s, me, mine, other, 10.0f);
    if (mine.empty() || other.empty()) return;
    int src = mine[0];
    for (int i : mine) if (s.planets[i].ships > s.planets[src].ships) src = i;
    int best = other[0];
    float bc = 1e18f;
    for (int t : other) {
        float c = s.planets[t].ships +
                  std::hypot(s.planets[t].x - s.planets[src].x,
                             s.planets[t].y - s.planets[src].y) / 10.0f;
        if (c < bc) { bc = c; best = t; }
    }
    uint32_t h = hash32(seed, tick, (uint32_t)me);
    if ((h & 0xff) < 26) best = other[(h >> 8) % other.size()];
    send(s, src, best, 50);
}

static void act_expand(State& s, int me, uint32_t seed, uint32_t tick) {
    std::vector<int> mine, other;
    collect_planets(s, me, mine, other, 15.0f);
    if (mine.empty() || other.empty()) return;
    int src = mine[0];
    for (int i : mine) if (s.planets[i].ships > s.planets[src].ships) src = i;
    int best = -1;
    float bv = -1e18f;
    for (int t : other) {
        float d = std::hypot(s.planets[t].x - s.planets[src].x,
                             s.planets[t].y - s.planets[src].y);
        float v = s.planets[t].production / (s.planets[t].ships + 1.0f + d / 20.0f);
        if (v > bv) { bv = v; best = t; }
    }
    if (best < 0) return;
    uint32_t h = hash32(seed, tick, (uint32_t)me + 64);
    if ((h & 0xff) < 26) best = other[(h >> 8) % other.size()];
    send(s, src, best, 75);
}

static void act_pressure(State& s, int me, uint32_t seed, uint32_t tick) {
    std::vector<int> mine, other;
    collect_planets(s, me, mine, other, 20.0f);
    if (mine.empty() || other.empty()) return;
    int src = mine[0];
    for (int i : mine) if (s.planets[i].ships > s.planets[src].ships) src = i;
    int best = -1;
    float bc = 1e18f;
    for (int t : other) {
        float c = s.planets[t].ships +
                  std::hypot(s.planets[t].x - s.planets[src].x,
                             s.planets[t].y - s.planets[src].y) / 10.0f;
        if (!s.planets[t].neutral()) c *= 0.5f;
        if (c < bc) { bc = c; best = t; }
    }
    if (best < 0) return;
    uint32_t h = hash32(seed, tick, (uint32_t)me + 128);
    if ((h & 0xff) < 26) best = other[(h >> 8) % other.size()];
    send(s, src, best, 75);
}

static void act_arcade(State& s, int me, uint32_t /*seed*/, uint32_t tick) {
    if (tick % 6 != 0) return;
    int src = -1;
    for (size_t i = 0; i < s.planets.size(); ++i) {
        if (s.planets[i].owner == me && (src < 0 || s.planets[i].ships > s.planets[src].ships)) {
            src = (int)i;
        }
    }
    if (src < 0 || s.planets[src].ships < 2.0f) return;
    int n = (int)std::floor(s.planets[src].ships / 2.0f);
    int best = -1;
    float bd = 1e18f;
    for (size_t i = 0; i < s.planets.size(); ++i) {
        const Planet& t = s.planets[i];
        if (t.owner == me || t.ships >= (float)n) continue;
        float d = std::hypot(t.x - s.planets[src].x, t.y - s.planets[src].y);
        if (d < bd) { bd = d; best = (int)i; }
    }
    if (best < 0) return;
    send_count(s, src, best, n, 50);
}

static void act_counter(State& s, int me, uint32_t seed, uint32_t tick) {
    constexpr float kCounterUntil = 20.0f;
    const int n = (int)s.planets.size();
    std::vector<galbots::FleetView> fv;
    galbots::build_fleets(s, fv);
    std::vector<float> inc((size_t)n, 0.0f), eta((size_t)n, 1e9f), mine_inc((size_t)n, 0.0f);
    for (const auto& f : fv) {
        if (f.target < 0 || f.target >= n) continue;
        const Planet& t = s.planets[(size_t)f.target];
        if (f.owner == me) { mine_inc[(size_t)f.target] += (float)f.ships; continue; }
        if (t.owner != me) continue;
        inc[(size_t)f.target] += (float)f.ships;
        float d = std::max(0.0f, (float)std::hypot(f.x - t.x, f.y - t.y) - t.radius());
        eta[(size_t)f.target] = std::min(eta[(size_t)f.target], d / kSpeed);
    }
    bool threatened = false;
    for (int i = 0; i < n; ++i) {
        const Planet& p = s.planets[(size_t)i];
        if (s.t > kCounterUntil || p.owner != me || inc[(size_t)i] < 40.0f) continue;
        float hold = p.ships + mine_inc[(size_t)i] + p.production * kProductionRate * eta[(size_t)i];
        if (inc[(size_t)i] > hold) threatened = true;
    }
    if (threatened) {
        float have = 0.0f;
        for (const Planet& p : s.planets) if (p.owner == me) have += std::floor(p.ships);
        int best = -1;
        float bp = -1.0f;
        for (int q = 0; q < n; ++q) {
            const Planet& t = s.planets[(size_t)q];
            if (t.owner == me || t.neutral()) continue;
            float tmax = 0.0f;
            for (const Planet& p : s.planets) {
                if (p.owner == me && p.ships >= 1.0f) {
                    tmax = std::max(tmax, std::hypot(p.x - t.x, p.y - t.y) / kSpeed);
                }
            }
            float enemy_inc = 0.0f;
            for (const auto& f : fv) {
                if (f.target == q && f.owner == t.owner) enemy_inc += (float)f.ships;
            }
            float need = t.ships + enemy_inc + t.production * kProductionRate * tmax + 2.0f - mine_inc[(size_t)q];
            if (have * 0.95f > need && t.production > bp) { bp = t.production; best = q; }
        }
        if (best >= 0) {
            for (int i = 0; i < n; ++i) {
                if (s.planets[(size_t)i].owner == me && s.planets[(size_t)i].ships >= 2.0f) {
                    send(s, i, best, 100);
                }
            }
            return;
        }
    }
    galbots::act_waffle(s, me, seed, tick);
}

static void dispatch_bot(int32_t kind, State& s, int me, uint32_t seed, uint32_t tick) {
    switch (kind) {
        case BOT_WAFFLE:        galbots::act_waffle(s, me, seed, tick); break;
        case BOT_WAFFLE_OPT:    galbots::act_waffle_opt(s, me, seed, tick); break;
        case BOT_SPARKY:        galbots::act_sparky(s, me, seed, tick); break;
        case BOT_SPARKY_NATIVE: galbots::act_sparky_native(s, me, seed, tick); break;
        case BOT_GREEDY:        act_greedy(s, me, seed, tick); break;
        case BOT_EXPAND:        act_expand(s, me, seed, tick); break;
        case BOT_PRESSURE:      act_pressure(s, me, seed, tick); break;
        case BOT_ARCADE:        act_arcade(s, me, seed, tick); break;
        case BOT_COUNTER:       act_counter(s, me, seed, tick); break;
        default: break;
    }
}

}  // namespace

extern "C" {

void* bot_create(int32_t kind) {
    auto* sess = new BotSession();
    sess->kind = kind;
    return sess;
}

void bot_reset(void* handle) {
    if (!handle) return;
    auto* sess = static_cast<BotSession*>(handle);
    sess->mem = BotMemory();
}

void bot_destroy(void* handle) {
    if (!handle) return;
    delete static_cast<BotSession*>(handle);
}

// Run the bot on a snapshot of the current board and return emitted ActionEvents.
//
// Planet owners: -1 = neutral, 0 = player 1, 1 = player 2.
// Fleet target_idx: 0..n_planets-1.
// Returns the number of events written (capped at max_out).
int32_t bot_decide(
    void* handle,
    int32_t me,
    float game_time,
    uint32_t seed,
    uint32_t tick,
    int32_t allow_redirect,
    int32_t n_planets,
    const float* p_x,
    const float* p_y,
    const float* p_prod,
    const float* p_ships,
    const int32_t* p_owner,
    int32_t n_fleets,
    const int32_t* f_id,
    const float* f_x,
    const float* f_y,
    const float* f_ships,
    const int32_t* f_target_idx,
    const int32_t* f_owner,
    int32_t max_out,
    int32_t* out_kind,
    int32_t* out_who,
    int32_t* out_tgt,
    int32_t* out_pct,
    int32_t* out_n
) {
    if (!handle || n_planets <= 0 || max_out <= 0) return 0;
    auto* sess = static_cast<BotSession*>(handle);

    State s;
    s.t = game_time;
    s.phys = 0;
    s.planets.resize((size_t)n_planets);
    for (int32_t i = 0; i < n_planets; ++i) {
        s.planets[(size_t)i] = Planet{p_x[i], p_y[i], p_prod[i], p_ships[i], p_owner[i]};
    }

    // One lightweight proxy Unit per aggregate fleet: build_fleets() reads
    // unit_position(s, u) -> (u.x, u.y) when locked==0, u.ships, u.target, u.owner, u.fleet.
    s.units.reserve((size_t)std::max(0, n_fleets));
    int32_t max_fid = 0;
    for (int32_t i = 0; i < n_fleets; ++i) {
        if (f_target_idx[i] < 0 || f_target_idx[i] >= n_planets) continue;
        if (f_ships[i] <= 0.0f) continue;
        Unit u{};
        u.x = f_x[i];
        u.y = f_y[i];
        u.ships = f_ships[i];
        u.target = f_target_idx[i];
        u.owner = f_owner[i];
        u.fleet = f_id[i];
        u.locked = 0;
        u.speed = kSpeed;
        u.turn_rate = kTurnRate0;
        s.units.push_back(u);
        if (f_id[i] >= max_fid) max_fid = f_id[i] + 1;
    }
    s.next_fleet = max_fid;

    // Load persistent BotMemory for this seat.
    s.bots.resize((size_t)std::max(2, me + 1));
    s.bots[(size_t)me] = sess->mem;

    std::vector<ActionEvent> events;
    events.reserve(16);
    g_action_log = &events;
    dispatch_bot(sess->kind, s, me, seed, tick);

    // If Sparky's 4 Hz FIFO queue popped only a redirect while redirects are
    // disabled in the env, allow a few extra queue pops to reach the next send.
    if (sess->kind == BOT_SPARKY && !allow_redirect) {
        int extra = 0;
        while (extra < 8 && s.bots[(size_t)me].s_nq > 0) {
            bool has_send = false;
            for (const auto& ev : events) {
                if (ev.kind == 0 && ev.who != ev.tgt && ev.n > 0) {
                    has_send = true;
                    break;
                }
            }
            if (has_send) break;
            events.clear();
            dispatch_bot(sess->kind, s, me, seed, tick);
            ++extra;
        }
    }
    g_action_log = nullptr;

    // Persist updated BotMemory back into the session.
    sess->mem = s.bots[(size_t)me];

    int32_t count = 0;
    for (const auto& ev : events) {
        if (count >= max_out) break;
        out_kind[count] = ev.kind;
        out_who[count] = ev.who;
        out_tgt[count] = ev.tgt;
        out_pct[count] = ev.pct;
        out_n[count] = ev.n;
        ++count;
    }
    return count;
}

}  // extern "C"
