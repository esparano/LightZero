// Auto-compiled C++17 Galcon physical-ship 20 Hz physics engine & standalone POD SimState.
// Compile:
//   g++ -O3 -march=native -ffp-contract=off -shared -fPIC -std=c++17 sim_core.cpp -lm -o sim_core.so
//
// Important:
//   - Compiled with -ffp-contract=off (no FMA contraction) and uses std::sqrt(dx*dx + dy*dy)
//     to guarantee bit-exact IEEE-754 float64 parity with GalconSim._step_physics_python().
//   - Uses static thread_local scratch buffers for <=64 planets / <=512 fleets / <=4096 units
//     and automatically falls back to std::vector scratch buffers for arbitrarily large worlds.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr double PRODUCTION_DIVISOR = 50.0;
constexpr double DEFAULT_SHIP_RADIUS = 6.0;
constexpr double DEFAULT_SHIP_SEP_DIST = 13.5;
constexpr double TIE_EPS = 1e-9;
constexpr double ARRIVAL_EPS = 1e-9;
constexpr double GOLDEN_ANGLE = 2.399963229728653; // pi * (3 - sqrt(5))
constexpr double SPAWN_AREA_RADIUS = 6.0;
constexpr int32_t SPAWN_SETTLE_ITERS = 0;
constexpr double SPAWN_SETTLE_TOL = 0.05;
constexpr double EPS_DIST2 = 1e-12;
constexpr double ANTIPODE_DOT_THRESHOLD = -0.995;
constexpr double PI = 3.14159265358979323846;
constexpr double DEFAULT_SHIP_TURN_RATE = 4.5;
constexpr double DEFAULT_SHIP_TURN_MIN_SPEED_FACTOR = 1.0;

constexpr int32_t ITEMS_MAX = 4096;
constexpr int32_t ITEMS_RESERVED = 16;
constexpr int32_t BASE_ITEM_USERS = 3;

// Max distinct owners fighting on a single planet in one substep.
constexpr int32_t MAX_ARR_OWNERS = 16;

// Limits for fast POD SimState & thread-local scratch (~167 KB).
constexpr int32_t SIM_MAX_PLANETS = 64;
constexpr int32_t SIM_MAX_FLEETS = 512;
constexpr int32_t SIM_MAX_UNITS = 4096;

struct ArrivalBucket {
    int32_t owners[MAX_ARR_OWNERS];
    double ships[MAX_ARR_OWNERS];
    int32_t count;
};

struct PhysicsScratch {
    uint8_t u_alive[SIM_MAX_UNITS];
    int32_t fl_alive_cnt[SIM_MAX_FLEETS];
    double tmp_x[SIM_MAX_UNITS];
    double tmp_y[SIM_MAX_UNITS];
    ArrivalBucket arr_buckets[SIM_MAX_PLANETS];
    int32_t dirty_planets[SIM_MAX_PLANETS];
    int32_t near_planets[SIM_MAX_PLANETS];
    double p_min_r[SIM_MAX_PLANETS];
    double p_min_r2[SIM_MAX_PLANETS];
    double p_land_r[SIM_MAX_PLANETS];
    double p_land_r2_guard[SIM_MAX_PLANETS];
    double p_prod_sub[SIM_MAX_PLANETS];
    int32_t tgt_head[SIM_MAX_PLANETS];
    int32_t grp_head[SIM_MAX_FLEETS];
    int32_t grp_tail[SIM_MAX_FLEETS];
    int32_t grp_cnt[SIM_MAX_FLEETS];
    int32_t grp_owner[SIM_MAX_FLEETS];
    int32_t grp_next[SIM_MAX_FLEETS];
    int32_t fl_next[SIM_MAX_FLEETS];
    int32_t grp_uids[SIM_MAX_UNITS];
    double grp_x[SIM_MAX_UNITS];
    double grp_y[SIM_MAX_UNITS];
};

static inline PhysicsScratch& get_physics_scratch() {
    static thread_local PhysicsScratch s;
    return s;
}

static inline double dist2d(double dx, double dy) {
    return std::sqrt(dx * dx + dy * dy);
}

static inline double spawn_hash_u01(uint32_t seed) {
    uint32_t x = seed;
    x ^= (x >> 16);
    x *= 0x7FEB352Du;
    x ^= (x >> 15);
    x *= 0x846CA68Bu;
    x ^= (x >> 16);
    return static_cast<double>(x) / 4294967296.0;
}

// Exact Galcon 2 / Galcon 3 physical_ship_count formula matching galcon/sim.py.
static inline int32_t physical_ship_count_cpp(
    double ships,
    double total_planet_ships,
    int32_t item_count,
    int32_t items_max = ITEMS_MAX,
    int32_t reserved = ITEMS_RESERVED
) {
    if (ships < 1.0 - TIE_EPS) {
        return 0;
    }
    int32_t avail = items_max - item_count - reserved;
    if (avail <= 0) {
        avail = 0;
    }
    avail /= 2;

    double parts_d = std::floor(ships);
    const double sqrt_cap = std::sqrt(17.0 * ships);
    if (sqrt_cap < parts_d) {
        parts_d = sqrt_cap;
    }
    parts_d = std::floor(parts_d);

    if (total_planet_ships > 0.0) {
        const double avail_cap = ((double)avail * ships) / total_planet_ships;
        if (avail_cap < parts_d) {
            parts_d = avail_cap;
        }
        parts_d = std::floor(parts_d);
    }

    if (parts_d > 2000000000.0) {
        parts_d = 2000000000.0;
    }
    const int32_t parts = (int32_t)parts_d;
    return parts < 1 ? 1 : parts;
}

template <bool CheckAlive, bool PairCap, bool TrackMaxOv = false>
static inline double separate_fleet_gs_core(
    int32_t start,
    int32_t end,
    double* __restrict__ ux,
    double* __restrict__ uy,
    const uint8_t* __restrict__ u_alive,
    double min_sep,
    double min_sep2,
    double half_k,
    double max_sep_push
) {
    double max_ov = 0.0;
    for (int32_t a = start; a < end; ++a) {
        if constexpr (CheckAlive) {
            if (!u_alive[a]) continue;
        }
        double ax = ux[a];
        double ay = uy[a];
        for (int32_t b = a + 1; b < end; ++b) {
            if constexpr (CheckAlive) {
                if (!u_alive[b]) continue;
            }
            const double dx = ux[b] - ax;
            const double dy = uy[b] - ay;
            const double d2 = dx * dx + dy * dy;
            if (d2 < min_sep2) {
                double nx, ny, dist;
                if (d2 > EPS_DIST2) {
                    dist = std::sqrt(d2);
                    nx = dx / dist;
                    ny = dy / dist;
                } else {
                    dist = 0.0;
                    const double ang = GOLDEN_ANGLE * (double)((a - start) * 31 + (b - start) + 1);
                    nx = std::cos(ang);
                    ny = std::sin(ang);
                }
                const double ov = min_sep - dist;
                if constexpr (TrackMaxOv) {
                    if (ov > max_ov) max_ov = ov;
                }
                double push = half_k * ov;
                if constexpr (PairCap) {
                    if (push > max_sep_push) push = max_sep_push;
                }
                const double px_d = nx * push;
                const double py_d = ny * push;
                ax -= px_d;
                ay -= py_d;
                ux[b] += px_d;
                uy[b] += py_d;
            }
        }
        ux[a] = ax;
        uy[a] = ay;
    }
    return max_ov;
}

template <bool CheckAlive, bool PairCap>
static inline void separate_fleet_gs(
    int32_t start,
    int32_t end,
    double* __restrict__ ux,
    double* __restrict__ uy,
    const uint8_t* __restrict__ u_alive,
    double min_sep,
    double min_sep2,
    double half_k,
    double max_sep_push
) {
    separate_fleet_gs_core<CheckAlive, PairCap, false>(
        start, end, ux, uy, u_alive, min_sep, min_sep2, half_k, max_sep_push
    );
}

// Bit-exact fleet-slice dynamic AABB Gauss-Seidel separation for multi-fleet target groups.
// Each fleet slice [sl_start[c], sl_end[c]) maintains a conservative enclosing AABB updated
// whenever any unit inside that slice moves. Any unit `a` whose distance to slice `cb`'s AABB
// is >= min_sep skips all units in `cb` in 4 comparisons while preserving exact (a, b) order.
template <bool PairCap>
static void separate_multi_fleet_gs(
    int32_t n_slices,
    const int32_t* __restrict__ sl_start,
    const int32_t* __restrict__ sl_end,
    double* __restrict__ sl_x0,
    double* __restrict__ sl_x1,
    double* __restrict__ sl_y0,
    double* __restrict__ sl_y1,
    double* __restrict__ ux,
    double* __restrict__ uy,
    double min_sep,
    double min_sep2,
    double half_k,
    double max_sep_push
) {
    for (int32_t ca = 0; ca < n_slices; ++ca) {
        const int32_t sa = sl_start[ca];
        const int32_t ea = sl_end[ca];
        double ax0 = sl_x0[ca], ax1 = sl_x1[ca], ay0 = sl_y0[ca], ay1 = sl_y1[ca];
        for (int32_t a = sa; a < ea; ++a) {
            double ax = ux[a];
            double ay = uy[a];
            // 1. Remainder of own fleet slice
            for (int32_t b = a + 1; b < ea; ++b) {
                const double dx = ux[b] - ax;
                const double dy = uy[b] - ay;
                const double d2 = dx * dx + dy * dy;
                if (d2 < min_sep2) {
                    double nx, ny, dist;
                    if (d2 > EPS_DIST2) {
                        dist = std::sqrt(d2);
                        nx = dx / dist;
                        ny = dy / dist;
                    } else {
                        dist = 0.0;
                        const double ang = GOLDEN_ANGLE * (double)(a * 31 + b + 1);
                        nx = std::cos(ang);
                        ny = std::sin(ang);
                    }
                    double push = half_k * (min_sep - dist);
                    if constexpr (PairCap) {
                        if (push > max_sep_push) push = max_sep_push;
                    }
                    const double px_d = nx * push;
                    const double py_d = ny * push;
                    ax -= px_d;
                    ay -= py_d;
                    const double nbx = ux[b] + px_d;
                    const double nby = uy[b] + py_d;
                    ux[b] = nbx;
                    uy[b] = nby;
                    if (nbx < ax0) ax0 = nbx;
                    if (nbx > ax1) ax1 = nbx;
                    if (nby < ay0) ay0 = nby;
                    if (nby > ay1) ay1 = nby;
                }
            }
            // 2. Subsequent fleet slices screened by dynamic AABB
            for (int32_t cb = ca + 1; cb < n_slices; ++cb) {
                if (ax + min_sep <= sl_x0[cb] || ax - min_sep >= sl_x1[cb] ||
                    ay + min_sep <= sl_y0[cb] || ay - min_sep >= sl_y1[cb]) {
                    continue;
                }
                const int32_t sb = sl_start[cb];
                const int32_t eb = sl_end[cb];
                double cx0 = sl_x0[cb], cx1 = sl_x1[cb], cy0 = sl_y0[cb], cy1 = sl_y1[cb];
                for (int32_t b = sb; b < eb; ++b) {
                    const double dx = ux[b] - ax;
                    const double dy = uy[b] - ay;
                    const double d2 = dx * dx + dy * dy;
                    if (d2 < min_sep2) {
                        double nx, ny, dist;
                        if (d2 > EPS_DIST2) {
                            dist = std::sqrt(d2);
                            nx = dx / dist;
                            ny = dy / dist;
                        } else {
                            dist = 0.0;
                            const double ang = GOLDEN_ANGLE * (double)(a * 31 + b + 1);
                            nx = std::cos(ang);
                            ny = std::sin(ang);
                        }
                        double push = half_k * (min_sep - dist);
                        if constexpr (PairCap) {
                            if (push > max_sep_push) push = max_sep_push;
                        }
                        const double px_d = nx * push;
                        const double py_d = ny * push;
                        ax -= px_d;
                        ay -= py_d;
                        const double nbx = ux[b] + px_d;
                        const double nby = uy[b] + py_d;
                        ux[b] = nbx;
                        uy[b] = nby;
                        if (nbx < cx0) cx0 = nbx;
                        if (nbx > cx1) cx1 = nbx;
                        if (nby < cy0) cy0 = nby;
                        if (nby > cy1) cy1 = nby;
                    }
                }
                sl_x0[cb] = cx0;
                sl_x1[cb] = cx1;
                sl_y0[cb] = cy0;
                sl_y1[cb] = cy1;
            }
            ux[a] = ax;
            uy[a] = ay;
        }
        sl_x0[ca] = ax0;
        sl_x1[ca] = ax1;
        sl_y0[ca] = ay0;
        sl_y1[ca] = ay1;
    }
}

static void separate_multi_fleet_jacobi(
    int32_t n_slices,
    const int32_t* __restrict__ sl_start,
    const int32_t* __restrict__ sl_end,
    const double* __restrict__ sl_x0,
    const double* __restrict__ sl_x1,
    const double* __restrict__ sl_y0,
    const double* __restrict__ sl_y1,
    const double* __restrict__ ux,
    const double* __restrict__ uy,
    double* __restrict__ tmp_x,
    double* __restrict__ tmp_y,
    double min_sep,
    double min_sep2,
    double half_k
) {
    for (int32_t ca = 0; ca < n_slices; ++ca) {
        const int32_t sa = sl_start[ca];
        const int32_t ea = sl_end[ca];
        for (int32_t a = sa; a < ea; ++a) {
            const double ax = ux[a];
            const double ay = uy[a];
            for (int32_t b = a + 1; b < ea; ++b) {
                const double dx = ux[b] - ax;
                const double dy = uy[b] - ay;
                const double d2 = dx * dx + dy * dy;
                if (d2 < min_sep2) {
                    double nx, ny, dist;
                    if (d2 > EPS_DIST2) {
                        dist = std::sqrt(d2);
                        nx = dx / dist;
                        ny = dy / dist;
                    } else {
                        dist = 0.0;
                        const double ang = GOLDEN_ANGLE * (double)(a * 31 + b + 1);
                        nx = std::cos(ang);
                        ny = std::sin(ang);
                    }
                    const double push = half_k * (min_sep - dist);
                    tmp_x[a] -= nx * push;
                    tmp_y[a] -= ny * push;
                    tmp_x[b] += nx * push;
                    tmp_y[b] += ny * push;
                }
            }
            for (int32_t cb = ca + 1; cb < n_slices; ++cb) {
                if (ax + min_sep <= sl_x0[cb] || ax - min_sep >= sl_x1[cb] ||
                    ay + min_sep <= sl_y0[cb] || ay - min_sep >= sl_y1[cb]) {
                    continue;
                }
                const int32_t sb = sl_start[cb];
                const int32_t eb = sl_end[cb];
                for (int32_t b = sb; b < eb; ++b) {
                    const double dx = ux[b] - ax;
                    const double dy = uy[b] - ay;
                    const double d2 = dx * dx + dy * dy;
                    if (d2 < min_sep2) {
                        double nx, ny, dist;
                        if (d2 > EPS_DIST2) {
                            dist = std::sqrt(d2);
                            nx = dx / dist;
                            ny = dy / dist;
                        } else {
                            dist = 0.0;
                            const double ang = GOLDEN_ANGLE * (double)(a * 31 + b + 1);
                            nx = std::cos(ang);
                            ny = std::sin(ang);
                        }
                        const double push = half_k * (min_sep - dist);
                        tmp_x[a] -= nx * push;
                        tmp_y[a] -= ny * push;
                        tmp_x[b] += nx * push;
                        tmp_y[b] += ny * push;
                    }
                }
            }
        }
    }
}

template <bool CheckAlive>
static inline void separate_fleet_jacobi_scalar(
    int32_t start,
    int32_t end,
    const double* __restrict__ ux,
    const double* __restrict__ uy,
    const uint8_t* __restrict__ u_alive,
    double* __restrict__ tmp_x,
    double* __restrict__ tmp_y,
    double min_sep,
    double min_sep2,
    double half_k
) {
    for (int32_t a = start; a < end; ++a) {
        if constexpr (CheckAlive) {
            if (!u_alive[a]) continue;
        }
        const double ax = ux[a];
        const double ay = uy[a];
        for (int32_t b = a + 1; b < end; ++b) {
            if constexpr (CheckAlive) {
                if (!u_alive[b]) continue;
            }
            const double dx = ux[b] - ax;
            const double dy = uy[b] - ay;
            const double d2 = dx * dx + dy * dy;
            if (d2 < min_sep2) {
                double nx, ny, dist;
                if (d2 > EPS_DIST2) {
                    dist = std::sqrt(d2);
                    nx = dx / dist;
                    ny = dy / dist;
                } else {
                    dist = 0.0;
                    const double ang = GOLDEN_ANGLE * (double)((a - start) * 31 + (b - start) + 1);
                    nx = std::cos(ang);
                    ny = std::sin(ang);
                }
                const double push = half_k * (min_sep - dist);
                tmp_x[a] -= nx * push;
                tmp_y[a] -= ny * push;
                tmp_x[b] += nx * push;
                tmp_y[b] += ny * push;
            }
        }
    }
}

// Core physics step operating on flat SoA arrays.
// Shared identically by both step_physics_cpp() and SimState::step_one_tick().
//
// sep_mode:
//   0 = Gauss-Seidel pairwise pushout (optional per-pair cap if max_sep_push > 0)
//   1 = Gauss-Seidel pairwise pushout + per-ship substep total displacement cap (default)
//   2 = Jacobi pairwise pushout (optional per-ship cap)
//
// obstacle_escape_eps:
//   0.0 = pure radial pushout
//   >0  = tiny deterministic tangential nudge only when a unit is near-antipodal
//         to target on an obstacle planet (dot(radial, to_target) < -0.995)
static int32_t run_physics_substeps_core(
    int32_t n_planets,
    int32_t* __restrict__ p_owner,
    double* __restrict__ p_ships,
    const double* __restrict__ p_x,
    const double* __restrict__ p_y,
    const double* __restrict__ p_prod,
    const double* __restrict__ p_radius,
    int32_t n_fleets,
    const int32_t* __restrict__ fl_owner,
    const int32_t* __restrict__ fl_target_idx,
    double* __restrict__ fl_spu,
    int32_t* __restrict__ fl_unit_start,
    int32_t* __restrict__ fl_unit_count,
    double* __restrict__ fl_out_ships,
    double* __restrict__ fl_out_cx,
    double* __restrict__ fl_out_cy,
    double* __restrict__ fl_out_radius,
    int32_t total_units,
    double* __restrict__ ux,
    double* __restrict__ uy,
    double* __restrict__ uhx,
    double* __restrict__ uhy,
    double tick_dt,
    double fleet_speed,
    int32_t substeps,
    double ship_radius,
    double sep_stiffness,
    double max_sep_push,
    int32_t sep_mode,
    double prod_divisor,
    double obstacle_escape_eps,
    double turn_rate,
    double turn_min_speed_factor,
    // Scratch buffers
    uint8_t* __restrict__ u_alive,
    int32_t* __restrict__ fl_alive_cnt,
    double* __restrict__ tmp_x,
    double* __restrict__ tmp_y,
    ArrivalBucket* __restrict__ arr_buckets,
    int32_t* __restrict__ dirty_planets,
    int32_t* __restrict__ near_planets,
    double* __restrict__ p_min_r,
    double* __restrict__ p_min_r2,
    double* __restrict__ p_land_r,
    double* __restrict__ p_land_r2_guard,
    double* __restrict__ p_prod_sub,
    int32_t* __restrict__ tgt_head,
    int32_t* __restrict__ grp_head,
    int32_t* __restrict__ grp_tail,
    int32_t* __restrict__ grp_cnt,
    int32_t* __restrict__ grp_owner,
    int32_t* __restrict__ grp_next,
    int32_t* __restrict__ fl_next,
    int32_t* __restrict__ grp_uids,
    double* __restrict__ grp_x,
    double* __restrict__ grp_y,
    int32_t n_ticks,
    bool (*on_tick_end)(void*),
    void* tick_ctx
) {
    if (n_planets <= 0 || n_ticks <= 0) return 0;
    if (substeps < 1) substeps = 1;

    const double sub_dt = tick_dt / (double)substeps;

    // Fast path when no fleets exist: only planet production runs across substeps.
    if (n_fleets <= 0) {
        for (int32_t p = 0; p < n_planets; ++p) {
            p_prod_sub[p] = (p_prod[p] / prod_divisor) * sub_dt;
        }
        for (int32_t t = 0; t < n_ticks; ++t) {
            for (int32_t s = 0; s < substeps; ++s) {
                for (int32_t p = 0; p < n_planets; ++p) {
                    if (p_owner[p] > 0) {
                        p_ships[p] += p_prod_sub[p];
                    }
                }
            }
            if (on_tick_end && on_tick_end(tick_ctx)) {
                break;
            }
        }
        return 0;
    }

    const double sub_step_dist = fleet_speed * sub_dt;
    const double min_sep = DEFAULT_SHIP_SEP_DIST;
    const double min_sep2 = min_sep * min_sep;
    const double half_k = 0.5 * sep_stiffness;
    const double max_turn = turn_rate * sub_dt;
    const bool instant_turn = (turn_rate <= 0.0) || (max_turn >= PI);
    const double cos_max = instant_turn ? -2.0 : std::cos(max_turn);
    const double sin_max = instant_turn ? 0.0 : std::sin(max_turn);

    double max_mr = 0.0;
    for (int32_t p = 0; p < n_planets; ++p) {
        const double mr = p_radius[p] + ship_radius;
        if (mr > max_mr) max_mr = mr;
        p_min_r[p] = mr;
        p_min_r2[p] = mr * mr;
        const double lr = mr + ARRIVAL_EPS;
        p_land_r[p] = lr;
        const double lrg = lr + 1e-6;
        p_land_r2_guard[p] = lrg * lrg;
        p_prod_sub[p] = (p_prod[p] / prod_divisor) * sub_dt;
        arr_buckets[p].count = 0;
    }

    if (total_units > 0) {
        std::memset(u_alive, 1, (size_t)total_units);
    }
    int32_t active_fleets = 0;
    for (int32_t f = 0; f < n_fleets; ++f) {
        fl_alive_cnt[f] = fl_unit_count[f];
        if (fl_unit_count[f] > 0) ++active_fleets;
    }

    // Group fleets by (owner, target planet index) preserving ascending fleet index order.
    // Real Galcon 2 (collide_do @0x43db40) keys unit separation by ((owner << 12) | target) << 8,
    // so opposing-owner fleets converging on the same planet pass through each other.
    for (int32_t p = 0; p < n_planets; ++p) {
        tgt_head[p] = -1;
    }
    int32_t n_groups = 0;
    for (int32_t f = 0; f < n_fleets; ++f) {
        fl_next[f] = -1;
        const int32_t tgt = fl_target_idx[f];
        if (tgt < 0 || tgt >= n_planets) continue;
        const int32_t own = fl_owner[f];
        int32_t g = tgt_head[tgt];
        while (g >= 0 && grp_owner[g] != own) {
            g = grp_next[g];
        }
        if (g < 0) {
            g = n_groups++;
            grp_owner[g] = own;
            grp_head[g] = f;
            grp_tail[g] = f;
            grp_cnt[g] = 1;
            grp_next[g] = tgt_head[tgt];
            tgt_head[tgt] = g;
        } else {
            fl_next[grp_tail[g]] = f;
            grp_tail[g] = f;
            grp_cnt[g] += 1;
        }
    }

    for (int32_t t = 0; t < n_ticks; ++t) {
        if (active_fleets <= 0) {
            for (int32_t s = 0; s < substeps; ++s) {
                for (int32_t p = 0; p < n_planets; ++p) {
                    if (p_owner[p] > 0) {
                        p_ships[p] += p_prod_sub[p];
                    }
                }
            }
            if (on_tick_end && on_tick_end(tick_ctx)) {
                break;
            }
            continue;
        }

        for (int32_t s = 0; s < substeps; ++s) {
        // 1. Production on owned planets
        for (int32_t p = 0; p < n_planets; ++p) {
            if (p_owner[p] > 0) {
                p_ships[p] += p_prod_sub[p];
            }
        }

        // 2. Same-owner same-target ship-ship circle separation
        if (sep_stiffness > 0.0) {
            const bool ship_cap = (sep_mode == 1 && max_sep_push > 0.0);
            const bool pair_cap = (sep_mode == 0 && max_sep_push > 0.0);
            const double cap2 = (max_sep_push > 0.0) ? (max_sep_push * max_sep_push) : 0.0;

            for (int32_t g = 0; g < n_groups; ++g) {
                const int32_t nf_grp = grp_cnt[g];
                if (nf_grp <= 0) continue;

                if (nf_grp == 1) {
                    const int32_t f = grp_head[g];
                    const int32_t alive_c = fl_alive_cnt[f];
                    if (alive_c <= 1) continue;
                    const int32_t start = fl_unit_start[f];
                    const int32_t cnt = fl_unit_count[f];
                    const int32_t end = start + cnt;
                    const bool all_alive = (alive_c == cnt);

                    if (sep_mode == 2) {
                        for (int32_t u = start; u < end; ++u) {
                            tmp_x[u] = 0.0;
                            tmp_y[u] = 0.0;
                        }
                        if (all_alive) {
                            separate_fleet_jacobi_scalar<false>(start, end, ux, uy, u_alive, tmp_x, tmp_y, min_sep, min_sep2, half_k);
                        } else {
                            separate_fleet_jacobi_scalar<true>(start, end, ux, uy, u_alive, tmp_x, tmp_y, min_sep, min_sep2, half_k);
                        }
                        for (int32_t u = start; u < end; ++u) {
                            if (!all_alive && !u_alive[u]) continue;
                            double dx = tmp_x[u];
                            double dy = tmp_y[u];
                            if (cap2 > 0.0) {
                                const double d2 = dx * dx + dy * dy;
                                if (d2 > cap2) {
                                    const double sc = max_sep_push / std::sqrt(d2);
                                    dx *= sc;
                                    dy *= sc;
                                }
                            }
                            ux[u] += dx;
                            uy[u] += dy;
                        }
                    } else {
                        if (ship_cap) {
                            std::memcpy(tmp_x + start, ux + start, (size_t)cnt * sizeof(double));
                            std::memcpy(tmp_y + start, uy + start, (size_t)cnt * sizeof(double));
                        }

                        if (all_alive) {
                            if (pair_cap) {
                                separate_fleet_gs<false, true>(start, end, ux, uy, u_alive, min_sep, min_sep2, half_k, max_sep_push);
                            } else {
                                separate_fleet_gs<false, false>(start, end, ux, uy, u_alive, min_sep, min_sep2, half_k, max_sep_push);
                            }
                        } else {
                            if (pair_cap) {
                                separate_fleet_gs<true, true>(start, end, ux, uy, u_alive, min_sep, min_sep2, half_k, max_sep_push);
                            } else {
                                separate_fleet_gs<true, false>(start, end, ux, uy, u_alive, min_sep, min_sep2, half_k, max_sep_push);
                            }
                        }

                        if (ship_cap) {
                            for (int32_t u = start; u < end; ++u) {
                                if (!all_alive && !u_alive[u]) continue;
                                const double dx = ux[u] - tmp_x[u];
                                const double dy = uy[u] - tmp_y[u];
                                const double d2 = dx * dx + dy * dy;
                                if (d2 > cap2) {
                                    const double sc = max_sep_push / std::sqrt(d2);
                                    ux[u] = tmp_x[u] + dx * sc;
                                    uy[u] = tmp_y[u] + dy * sc;
                                }
                            }
                        }
                    }
                } else {
                    // Gather alive units per fleet slice and record each slice's enclosing AABB.
                    constexpr int32_t MAX_SLICES = 64;
                    int32_t sl_start[MAX_SLICES], sl_end[MAX_SLICES];
                    double sl_x0[MAX_SLICES], sl_x1[MAX_SLICES], sl_y0[MAX_SLICES], sl_y1[MAX_SLICES];
                    int32_t n_slices = 0;
                    int32_t m = 0;
                    for (int32_t f = grp_head[g]; f >= 0; f = fl_next[f]) {
                        const int32_t alive_c = fl_alive_cnt[f];
                        if (alive_c <= 0) continue;
                        const int32_t start = fl_unit_start[f];
                        const int32_t cnt = fl_unit_count[f];
                        const int32_t end = start + cnt;
                        const int32_t s0 = m;
                        double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;
                        if (alive_c == cnt) {
                            for (int32_t u = start; u < end; ++u) {
                                const double xu = ux[u];
                                const double yu = uy[u];
                                grp_uids[m] = u;
                                grp_x[m] = xu;
                                grp_y[m] = yu;
                                if (xu < x0) x0 = xu;
                                if (xu > x1) x1 = xu;
                                if (yu < y0) y0 = yu;
                                if (yu > y1) y1 = yu;
                                ++m;
                            }
                        } else {
                            for (int32_t u = start; u < end; ++u) {
                                if (u_alive[u]) {
                                    const double xu = ux[u];
                                    const double yu = uy[u];
                                    grp_uids[m] = u;
                                    grp_x[m] = xu;
                                    grp_y[m] = yu;
                                    if (xu < x0) x0 = xu;
                                    if (xu > x1) x1 = xu;
                                    if (yu < y0) y0 = yu;
                                    if (yu > y1) y1 = yu;
                                    ++m;
                                }
                            }
                        }
                        if (n_slices < MAX_SLICES) {
                            sl_start[n_slices] = s0;
                            sl_end[n_slices] = m;
                            sl_x0[n_slices] = x0;
                            sl_x1[n_slices] = x1;
                            sl_y0[n_slices] = y0;
                            sl_y1[n_slices] = y1;
                            ++n_slices;
                        } else {
                            // Overflow: fold tail into last slice conservatively
                            const int32_t last = MAX_SLICES - 1;
                            sl_end[last] = m;
                            if (x0 < sl_x0[last]) sl_x0[last] = x0;
                            if (x1 > sl_x1[last]) sl_x1[last] = x1;
                            if (y0 < sl_y0[last]) sl_y0[last] = y0;
                            if (y1 > sl_y1[last]) sl_y1[last] = y1;
                        }
                    }
                    if (m <= 1) continue;

                    if (sep_mode == 2) {
                        for (int32_t u = 0; u < m; ++u) {
                            tmp_x[u] = 0.0;
                            tmp_y[u] = 0.0;
                        }
                        if (n_slices >= 2) {
                            separate_multi_fleet_jacobi(
                                n_slices, sl_start, sl_end, sl_x0, sl_x1, sl_y0, sl_y1,
                                grp_x, grp_y, tmp_x, tmp_y, min_sep, min_sep2, half_k
                            );
                        } else {
                            separate_fleet_jacobi_scalar<false>(0, m, grp_x, grp_y, u_alive, tmp_x, tmp_y, min_sep, min_sep2, half_k);
                        }
                        for (int32_t u = 0; u < m; ++u) {
                            double dx = tmp_x[u];
                            double dy = tmp_y[u];
                            if (cap2 > 0.0) {
                                const double d2 = dx * dx + dy * dy;
                                if (d2 > cap2) {
                                    const double sc = max_sep_push / std::sqrt(d2);
                                    dx *= sc;
                                    dy *= sc;
                                }
                            }
                            grp_x[u] += dx;
                            grp_y[u] += dy;
                        }
                    } else {
                        if (ship_cap) {
                            std::memcpy(tmp_x, grp_x, (size_t)m * sizeof(double));
                            std::memcpy(tmp_y, grp_y, (size_t)m * sizeof(double));
                        }

                        if (n_slices >= 2) {
                            if (pair_cap) {
                                separate_multi_fleet_gs<true>(
                                    n_slices, sl_start, sl_end, sl_x0, sl_x1, sl_y0, sl_y1,
                                    grp_x, grp_y, min_sep, min_sep2, half_k, max_sep_push
                                );
                            } else {
                                separate_multi_fleet_gs<false>(
                                    n_slices, sl_start, sl_end, sl_x0, sl_x1, sl_y0, sl_y1,
                                    grp_x, grp_y, min_sep, min_sep2, half_k, max_sep_push
                                );
                            }
                        } else {
                            if (pair_cap) {
                                separate_fleet_gs<false, true>(0, m, grp_x, grp_y, u_alive, min_sep, min_sep2, half_k, max_sep_push);
                            } else {
                                separate_fleet_gs<false, false>(0, m, grp_x, grp_y, u_alive, min_sep, min_sep2, half_k, max_sep_push);
                            }
                        }

                        if (ship_cap) {
                            for (int32_t u = 0; u < m; ++u) {
                                const double dx = grp_x[u] - tmp_x[u];
                                const double dy = grp_y[u] - tmp_y[u];
                                const double d2 = dx * dx + dy * dy;
                                if (d2 > cap2) {
                                    const double sc = max_sep_push / std::sqrt(d2);
                                    grp_x[u] = tmp_x[u] + dx * sc;
                                    grp_y[u] = tmp_y[u] + dy * sc;
                                }
                            }
                        }
                    }

                    for (int32_t u = 0; u < m; ++u) {
                        const int32_t uid = grp_uids[u];
                        ux[uid] = grp_x[u];
                        uy[uid] = grp_y[u];
                    }
                }
            }
        }

        // 3. Move toward target + 4. Planet collider pushout + 5a. Touch-to-land
        int32_t n_dirty = 0;

        for (int32_t f = 0; f < n_fleets; ++f) {
            const int32_t alive_c = fl_alive_cnt[f];
            if (alive_c <= 0) continue;
            const int32_t tgt = fl_target_idx[f];
            if (tgt < 0 || tgt >= n_planets) {
                fl_alive_cnt[f] = 0;
                continue;
            }
            const double tx = p_x[tgt];
            const double ty = p_y[tgt];
            const int32_t start = fl_unit_start[f];
            const int32_t cnt = fl_unit_count[f];
            const int32_t end = start + cnt;
            const bool all_alive = (alive_c == cnt);

            // Step 3: Move toward target with heading turn limit + track exact fleet AABB
            double min_x = 1e300, max_x = -1e300, min_y = 1e300, max_y = -1e300;
            for (int32_t u = start; u < end; ++u) {
                if (!all_alive && !u_alive[u]) continue;
                double x = ux[u];
                double y = uy[u];
                const double dx = tx - x;
                const double dy = ty - y;
                const double dist = dist2d(dx, dy);
                if (dist > 0.0) {
                    const double des_x = dx / dist;
                    const double des_y = dy / dist;
                    double hx = uhx[u];
                    double hy = uhy[u];
                    const double dot = hx * des_x + hy * des_y;
                    if (instant_turn || dot >= cos_max) {
                        hx = des_x;
                        hy = des_y;
                        if (dist <= sub_step_dist) {
                            x = tx;
                            y = ty;
                        } else {
                            x += hx * sub_step_dist;
                            y += hy * sub_step_dist;
                        }
                    } else {
                        const double cross = hx * des_y - hy * des_x;
                        const double turn_sign = (cross >= 0.0) ? 1.0 : -1.0;
                        const double s_rot = turn_sign * sin_max;
                        const double nhx = hx * cos_max - hy * s_rot;
                        const double nhy = hx * s_rot + hy * cos_max;
                        const double hn = dist2d(nhx, nhy);
                        if (hn > 0.0) {
                            hx = nhx / hn;
                            hy = nhy / hn;
                        } else {
                            hx = des_x;
                            hy = des_y;
                        }
                        const double align = hx * des_x + hy * des_y;
                        const double speed_scale =
                            turn_min_speed_factor + (1.0 - turn_min_speed_factor) * (align > 0.0 ? align : 0.0);
                        const double step = sub_step_dist * speed_scale;
                        x += hx * step;
                        y += hy * step;
                    }
                    uhx[u] = hx;
                    uhy[u] = hy;
                    ux[u] = x;
                    uy[u] = y;
                }
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }

            // Exact zero-margin broadphase: check if any planet p != tgt overlaps the fleet AABB
            bool any_exact_hit = false;
            for (int32_t p = 0; p < n_planets; ++p) {
                if (p == tgt) continue;
                const double mr = p_min_r[p];
                if (p_x[p] + mr >= min_x && p_x[p] - mr <= max_x &&
                    p_y[p] + mr >= min_y && p_y[p] - mr <= max_y) {
                    any_exact_hit = true;
                    break;
                }
            }

            // Step 4: Planet collider radial pushout (padded candidate list preserves exactness)
            if (any_exact_hit) {
                int32_t n_near = 0;
                const double pad_min_x = min_x - max_mr;
                const double pad_max_x = max_x + max_mr;
                const double pad_min_y = min_y - max_mr;
                const double pad_max_y = max_y + max_mr;
                for (int32_t p = 0; p < n_planets; ++p) {
                    if (p == tgt) continue;
                    const double mr = p_min_r[p];
                    if (p_x[p] + mr >= pad_min_x && p_x[p] - mr <= pad_max_x &&
                        p_y[p] + mr >= pad_min_y && p_y[p] - mr <= pad_max_y) {
                        near_planets[n_near++] = p;
                    }
                }
                for (int32_t u = start; u < end; ++u) {
                    if (!all_alive && !u_alive[u]) continue;
                    double x = ux[u];
                    double y = uy[u];
                    for (int32_t ni = 0; ni < n_near; ++ni) {
                        const int32_t p = near_planets[ni];
                        const double dx = x - p_x[p];
                        const double dy = y - p_y[p];
                        const double d2 = dx * dx + dy * dy;
                        if (d2 < p_min_r2[p]) {
                            const double mr = p_min_r[p];
                            if (d2 > EPS_DIST2) {
                                const double dist = std::sqrt(d2);
                                double nx = dx / dist;
                                double ny = dy / dist;
                                if (obstacle_escape_eps > 0.0) {
                                    const double tdx = tx - p_x[p];
                                    const double tdy = ty - p_y[p];
                                    const double td = std::sqrt(tdx * tdx + tdy * tdy);
                                    if (td > 1e-6) {
                                        const double tnx = tdx / td;
                                        const double tny = tdy / td;
                                        const double dot = nx * tnx + ny * tny;
                                        if (dot < ANTIPODE_DOT_THRESHOLD) {
                                            const double cross = tnx * ny - tny * nx;
                                            const double sign = (std::fabs(cross) > 1e-9)
                                                ? (cross > 0.0 ? 1.0 : -1.0)
                                                : ((((u - start) & 1) == 0) ? 1.0 : -1.0);
                                            nx += (-tny) * sign * obstacle_escape_eps;
                                            ny += (tnx) * sign * obstacle_escape_eps;
                                            const double nn = std::sqrt(nx * nx + ny * ny);
                                            nx /= nn;
                                            ny /= nn;
                                        }
                                    }
                                }
                                x = p_x[p] + nx * mr;
                                y = p_y[p] + ny * mr;
                            } else {
                                const double tdx = tx - p_x[p];
                                const double tdy = ty - p_y[p];
                                const double td2 = tdx * tdx + tdy * tdy;
                                if (td2 > EPS_DIST2) {
                                    const double td = std::sqrt(td2);
                                    x = p_x[p] + (tdx / td) * mr;
                                    y = p_y[p] + (tdy / td) * mr;
                                } else {
                                    x = p_x[p] + mr;
                                    y = p_y[p];
                                }
                            }
                        }
                    }
                    ux[u] = x;
                    uy[u] = y;
                    if (x < min_x) min_x = x;
                    if (x > max_x) max_x = x;
                    if (y < min_y) min_y = y;
                    if (y > max_y) max_y = y;
                }
            }

            // Step 5a: Touch-to-land check (exact fleet-AABB early-out skips mid-flight fleets)
            const double land_r = p_land_r[tgt];
            const double land_r_g = land_r + 1e-6;
            if (max_x >= tx - land_r_g && min_x <= tx + land_r_g &&
                max_y >= ty - land_r_g && min_y <= ty + land_r_g) {
                const double land_r2_g = p_land_r2_guard[tgt];
                const int32_t owner = fl_owner[f];
                const double spu = fl_spu[f];
                int32_t alive_left = alive_c;

                for (int32_t u = start; u < end; ++u) {
                    if (!all_alive && !u_alive[u]) continue;
                    const double dx = tx - ux[u];
                    const double dy = ty - uy[u];
                    const double d2 = dx * dx + dy * dy;
                    if (d2 <= land_r2_g && std::sqrt(d2) <= land_r) {
                        u_alive[u] = 0;
                        --alive_left;

                        ArrivalBucket& b = arr_buckets[tgt];
                        if (b.count == 0) {
                            dirty_planets[n_dirty++] = tgt;
                            if (p_ships[tgt] > 0.0) {
                                b.owners[0] = p_owner[tgt];
                                b.ships[0] = p_ships[tgt];
                                b.count = 1;
                            }
                        }
                        bool found = false;
                        for (int32_t k = 0; k < b.count; ++k) {
                            if (b.owners[k] == owner) {
                                b.ships[k] += spu;
                                found = true;
                                break;
                            }
                        }
                        if (!found && b.count < MAX_ARR_OWNERS) {
                            b.owners[b.count] = owner;
                            b.ships[b.count] = spu;
                            b.count++;
                        }
                    }
                }
                fl_alive_cnt[f] = alive_left;
            }
        }

        // Step 5b: Resolve gradual combat on planets with arrivals
        for (int32_t di = 0; di < n_dirty; ++di) {
            const int32_t p = dirty_planets[di];
            ArrivalBucket& b = arr_buckets[p];
            const int32_t cnt = b.count;
            if (cnt == 1) {
                p_owner[p] = b.owners[0];
                p_ships[p] = b.ships[0];
            } else if (cnt >= 2) {
                double top_v = -1.0;
                double sec_v = -1.0;
                int32_t top_o = p_owner[p];
                for (int32_t k = 0; k < cnt; ++k) {
                    const double v = b.ships[k];
                    const int32_t o = b.owners[k];
                    if (v > top_v) {
                        sec_v = top_v;
                        top_v = v;
                        top_o = o;
                    } else if (v > sec_v) {
                        sec_v = v;
                    }
                }
                if (std::fabs(top_v - sec_v) <= TIE_EPS) {
                    p_ships[p] = 0.0;
                } else {
                    p_owner[p] = top_o;
                    p_ships[p] = top_v - sec_v;
                }
            }
            b.count = 0;
        }
        } // end substeps loop

        if (n_ticks > 1) {
            int32_t next_active = 0;
            for (int32_t f = 0; f < n_fleets; ++f) {
                const int32_t alive_c = fl_alive_cnt[f];
                if (alive_c > 0) {
                    fl_out_ships[f] = (double)alive_c * fl_spu[f];
                    ++next_active;
                } else {
                    fl_out_ships[f] = 0.0;
                }
            }
            active_fleets = next_active;
            if (on_tick_end && on_tick_end(tick_ctx)) {
                break;
            }
            if (t + 1 < n_ticks && active_fleets > 0) {
                for (int32_t f = 0; f < n_fleets; ++f) {
                    const int32_t old_count = fl_unit_count[f];
                    if (old_count <= 0) continue;
                    const int32_t alive_c = fl_alive_cnt[f];
                    if (alive_c <= 0) {
                        fl_unit_count[f] = 0;
                    } else {
                        fl_spu[f] = fl_out_ships[f] / (double)(alive_c > 1 ? alive_c : 1);
                        if (alive_c < old_count) {
                            const int32_t st_u = fl_unit_start[f];
                            int32_t wp = st_u;
                            for (int32_t u = st_u; u < st_u + old_count; ++u) {
                                if (u_alive[u]) {
                                    if (wp != u) {
                                        ux[wp] = ux[u];
                                        uy[wp] = uy[u];
                                        uhx[wp] = uhx[u];
                                        uhy[wp] = uhy[u];
                                    }
                                    u_alive[wp] = 1;
                                    ++wp;
                                }
                            }
                            fl_unit_count[f] = alive_c;
                        }
                    }
                }
            }
        }
    } // end ticks loop

    // 6. Compact alive units in-place & compute fleet summaries
    int32_t write_pos = 0;
    for (int32_t f = 0; f < n_fleets; ++f) {
        const int32_t old_start = fl_unit_start[f];
        const int32_t old_count = fl_unit_count[f];
        const int32_t new_start = write_pos;
        double sum_x = 0.0;
        double sum_y = 0.0;
        int32_t kept = 0;

        if (fl_alive_cnt[f] == old_count && old_count > 0) {
            if (write_pos != old_start) {
                std::memmove(ux + write_pos, ux + old_start, (size_t)old_count * sizeof(double));
                std::memmove(uy + write_pos, uy + old_start, (size_t)old_count * sizeof(double));
                std::memmove(uhx + write_pos, uhx + old_start, (size_t)old_count * sizeof(double));
                std::memmove(uhy + write_pos, uhy + old_start, (size_t)old_count * sizeof(double));
            }
            for (int32_t u = new_start; u < new_start + old_count; ++u) {
                sum_x += ux[u];
                sum_y += uy[u];
            }
            kept = old_count;
            write_pos += old_count;
        } else if (fl_alive_cnt[f] > 0) {
            for (int32_t u = old_start; u < old_start + old_count; ++u) {
                if (u_alive[u]) {
                    const double x = ux[u];
                    const double y = uy[u];
                    ux[write_pos] = x;
                    uy[write_pos] = y;
                    uhx[write_pos] = uhx[u];
                    uhy[write_pos] = uhy[u];
                    sum_x += x;
                    sum_y += y;
                    ++write_pos;
                    ++kept;
                }
            }
        }

        fl_unit_start[f] = new_start;
        fl_unit_count[f] = kept;
        if (kept > 0) {
            const double cx = sum_x / (double)kept;
            const double cy = sum_y / (double)kept;
            double max_d2 = 0.0;
            for (int32_t u = new_start; u < new_start + kept; ++u) {
                const double dx = ux[u] - cx;
                const double dy = uy[u] - cy;
                const double d2 = dx * dx + dy * dy;
                if (d2 > max_d2) max_d2 = d2;
            }
            fl_out_ships[f] = (double)kept * fl_spu[f];
            fl_out_cx[f] = cx;
            fl_out_cy[f] = cy;
            fl_out_radius[f] = std::sqrt(max_d2) + ship_radius;
        } else {
            fl_out_ships[f] = 0.0;
            fl_out_cx[f] = 0.0;
            fl_out_cy[f] = 0.0;
            fl_out_radius[f] = ship_radius;
        }
    }

    return write_pos;
}

int32_t run_physics_substeps(
    int32_t n_planets,
    int32_t* __restrict__ p_owner,
    double* __restrict__ p_ships,
    const double* __restrict__ p_x,
    const double* __restrict__ p_y,
    const double* __restrict__ p_prod,
    const double* __restrict__ p_radius,
    int32_t n_fleets,
    const int32_t* __restrict__ fl_owner,
    const int32_t* __restrict__ fl_target_idx,
    const double* __restrict__ fl_spu,
    int32_t* __restrict__ fl_unit_start,
    int32_t* __restrict__ fl_unit_count,
    double* __restrict__ fl_out_ships,
    double* __restrict__ fl_out_cx,
    double* __restrict__ fl_out_cy,
    double* __restrict__ fl_out_radius,
    int32_t total_units,
    double* __restrict__ ux,
    double* __restrict__ uy,
    double* __restrict__ uhx,
    double* __restrict__ uhy,
    double tick_dt,
    double fleet_speed,
    int32_t substeps,
    double ship_radius,
    double sep_stiffness,
    double max_sep_push,
    int32_t sep_mode,
    double prod_divisor,
    double obstacle_escape_eps,
    double turn_rate,
    double turn_min_speed_factor,
    // Scratch buffers
    uint8_t* __restrict__ u_alive,
    int32_t* __restrict__ fl_alive_cnt,
    double* __restrict__ tmp_x,
    double* __restrict__ tmp_y,
    ArrivalBucket* __restrict__ arr_buckets,
    int32_t* __restrict__ dirty_planets,
    int32_t* __restrict__ near_planets,
    double* __restrict__ p_min_r,
    double* __restrict__ p_min_r2,
    double* __restrict__ p_land_r,
    double* __restrict__ p_land_r2_guard,
    double* __restrict__ p_prod_sub,
    int32_t* __restrict__ tgt_head,
    int32_t* __restrict__ grp_head,
    int32_t* __restrict__ grp_tail,
    int32_t* __restrict__ grp_cnt,
    int32_t* __restrict__ grp_owner,
    int32_t* __restrict__ grp_next,
    int32_t* __restrict__ fl_next,
    int32_t* __restrict__ grp_uids,
    double* __restrict__ grp_x,
    double* __restrict__ grp_y
) {
    return run_physics_substeps_core(
        n_planets, p_owner, p_ships, p_x, p_y, p_prod, p_radius,
        n_fleets, fl_owner, fl_target_idx, const_cast<double*>(fl_spu), fl_unit_start, fl_unit_count,
        fl_out_ships, fl_out_cx, fl_out_cy, fl_out_radius,
        total_units, ux, uy, uhx, uhy,
        tick_dt, fleet_speed, substeps, ship_radius,
        sep_stiffness, max_sep_push, sep_mode, prod_divisor, obstacle_escape_eps,
        turn_rate, turn_min_speed_factor,
        u_alive, fl_alive_cnt, tmp_x, tmp_y,
        arr_buckets, dirty_planets, near_planets,
        p_min_r, p_min_r2, p_land_r, p_land_r2_guard, p_prod_sub,
        tgt_head, grp_head, grp_tail, grp_cnt, grp_owner, grp_next, fl_next, grp_uids, grp_x, grp_y,
        1,
        nullptr,
        nullptr
    );
}

} // namespace

// ============================================================================
// Standalone POD SimState (inline SoA buffers for ultra-fast clone & rollouts)
// ============================================================================

struct COrder {
    int32_t kind;    // 0 = SendOrder, 1 = RedirOrder
    int32_t player;  // player ID
    int32_t a;       // Send: src_id, Redir: fleet_id
    int32_t b;       // Send: dst_id, Redir: dst_id
    int32_t percent; // Send: 1..100
};

struct SimState {
    // Config
    double width;
    double height;
    double fleet_speed;
    double tick_dt;
    double ship_radius;
    double sep_stiffness;
    double max_sep_push;
    double prod_divisor;
    double obstacle_escape_eps;
    double turn_rate;
    double turn_min_speed_factor;
    int32_t max_ticks;
    int32_t substeps;
    int32_t sep_mode;
    int32_t allow_redirect;
    int32_t items_max;
    int32_t items_reserved;
    int32_t base_item_users;

    // Dynamic scalar state
    int32_t tick;
    int32_t next_fleet_id;
    int32_t done;
    int32_t winner;

    // Planets SoA (sorted by planet id)
    int32_t n_planets;
    int32_t p_id[SIM_MAX_PLANETS];
    int32_t p_owner[SIM_MAX_PLANETS];
    double p_ships[SIM_MAX_PLANETS];
    double p_x[SIM_MAX_PLANETS];
    double p_y[SIM_MAX_PLANETS];
    double p_prod[SIM_MAX_PLANETS];
    double p_radius[SIM_MAX_PLANETS];

    // Fleets SoA (insertion order, matching Python dict iteration order)
    int32_t n_fleets;
    int32_t fl_id[SIM_MAX_FLEETS];
    int32_t fl_owner[SIM_MAX_FLEETS];
    int32_t fl_source_id[SIM_MAX_FLEETS];
    int32_t fl_target_idx[SIM_MAX_FLEETS];
    double fl_spu[SIM_MAX_FLEETS];
    int32_t fl_unit_start[SIM_MAX_FLEETS];
    int32_t fl_unit_count[SIM_MAX_FLEETS];
    double fl_ships[SIM_MAX_FLEETS];
    double fl_cx[SIM_MAX_FLEETS];
    double fl_cy[SIM_MAX_FLEETS];
    double fl_radius[SIM_MAX_FLEETS];

    // Physical ship units SoA
    int32_t n_units;
    double ux[SIM_MAX_UNITS];
    double uy[SIM_MAX_UNITS];
    double uhx[SIM_MAX_UNITS];
    double uhy[SIM_MAX_UNITS];

    int32_t find_planet_idx(int32_t planet_id) const {
        for (int32_t i = 0; i < n_planets; ++i) {
            if (p_id[i] == planet_id) return i;
        }
        return -1;
    }

    int32_t item_count() const {
        return base_item_users + n_planets + n_fleets + n_units;
    }

    double total_ships(int32_t player) const {
        double s = 0.0;
        for (int32_t i = 0; i < n_planets; ++i) {
            if (p_owner[i] == player) s += p_ships[i];
        }
        for (int32_t f = 0; f < n_fleets; ++f) {
            if (fl_owner[f] == player) s += fl_ships[f];
        }
        return s;
    }

    double total_production(int32_t player) const {
        double s = 0.0;
        for (int32_t i = 0; i < n_planets; ++i) {
            if (p_owner[i] == player) s += p_prod[i];
        }
        return s;
    }

    void check_terminal() {
        bool alive1 = false, alive2 = false;
        for (int32_t i = 0; i < n_planets; ++i) {
            if (p_owner[i] == 1) alive1 = true;
            else if (p_owner[i] == 2) alive2 = true;
        }
        for (int32_t f = 0; f < n_fleets; ++f) {
            if (fl_ships[f] > 0.0) {
                if (fl_owner[f] == 1) alive1 = true;
                else if (fl_owner[f] == 2) alive2 = true;
            }
        }
        if (alive1 && !alive2) {
            done = 1;
            winner = 1;
            return;
        }
        if (alive2 && !alive1) {
            done = 1;
            winner = 2;
            return;
        }
        if (!alive1 && !alive2) {
            done = 1;
            winner = 0;
            return;
        }
        if (tick >= max_ticks) {
            done = 1;
            const double s1 = total_ships(1);
            const double s2 = total_ships(2);
            const double pr1 = total_production(1);
            const double pr2 = total_production(2);
            const double sc1 = s1 + 12.0 * (pr1 / prod_divisor);
            const double sc2 = s2 + 12.0 * (pr2 / prod_divisor);
            if (std::fabs(sc1 - sc2) < 1e-3) {
                winner = 0;
            } else {
                winner = (sc1 > sc2) ? 1 : 2;
            }
        }
    }

    int32_t apply_send(int32_t pid, int32_t src_id, int32_t dst_id, int32_t percent) {
        if (src_id == dst_id) return -1;
        const int32_t si = find_planet_idx(src_id);
        const int32_t di = find_planet_idx(dst_id);
        if (si < 0 || di < 0 || p_owner[si] != pid) return -1;
        if (p_ships[si] < 1.0 - TIE_EPS) return -1;

        const int32_t pct = percent < 1 ? 1 : (percent > 100 ? 100 : percent);
        // Exact Galcon 2 send formula (verified via send_rounding_edges + prod_and_rounding):
        // min(p_ships, floor(p_ships * pct / 100 + 0.5))
        const double rounded = std::floor(p_ships[si] * ((double)pct / 100.0) + 0.5);
        const double send_ships = (rounded < p_ships[si]) ? rounded : p_ships[si];
        if (send_ships < 1.0 - TIE_EPS) return -1;

        double total_planet_ships = 0.0;
        for (int32_t i = 0; i < n_planets; ++i) {
            if (p_ships[i] > 0.0) {
                total_planet_ships += p_ships[i];
            }
        }
        const int32_t parts = physical_ship_count_cpp(
            send_ships, total_planet_ships, item_count(), items_max, items_reserved
        );
        if (parts <= 0) return -1;
        if (n_fleets >= SIM_MAX_FLEETS || n_units + parts > SIM_MAX_UNITS) return -1;

        const double rem = p_ships[si] - send_ships;
        p_ships[si] = rem > 0.0 ? rem : 0.0;

        const double dx = p_x[di] - p_x[si];
        const double dy = p_y[di] - p_y[si];
        const double raw_d = dist2d(dx, dy);
        const double d = (raw_d > 1e-6) ? raw_d : 1e-6;
        const double spawn_r = p_radius[si] + ship_radius;
        const double sx = p_x[si] + (dx / d) * spawn_r;
        const double sy = p_y[si] + (dy / d) * spawn_r;
        const double hx0 = (raw_d > 1e-6) ? (dx / d) : 1.0;
        const double hy0 = (raw_d > 1e-6) ? (dy / d) : 0.0;

        const int32_t f = n_fleets++;
        const int32_t fid = next_fleet_id++;
        const int32_t u_start = n_units;
        fl_id[f] = fid;
        fl_owner[f] = pid;
        fl_source_id[f] = src_id;
        fl_target_idx[f] = di;
        fl_spu[f] = send_ships / (double)parts;
        fl_unit_start[f] = u_start;
        fl_unit_count[f] = parts;
        fl_ships[f] = send_ships;

        if (parts == 1) {
            ux[u_start] = sx;
            uy[u_start] = sy;
            uhx[u_start] = hx0;
            uhy[u_start] = hy0;
            fl_cx[f] = sx;
            fl_cy[f] = sy;
            fl_radius[f] = ship_radius;
        } else {
            double* const fux = ux + u_start;
            double* const fuy = uy + u_start;
            const double two_pi = 2.0 * PI;
            const uint32_t seed_base = static_cast<uint32_t>(fid) * 1000003u;
            const double spawn_disc_r = SPAWN_AREA_RADIUS * std::sqrt(static_cast<double>(parts));
            for (int32_t j = 0; j < parts; ++j) {
                const uint32_t uj = static_cast<uint32_t>(j);
                const double u1 = spawn_hash_u01(seed_base + uj * 2u + 1u);
                const double u2 = spawn_hash_u01(seed_base + uj * 2u + 2u);
                const double r = spawn_disc_r * std::sqrt(u1);
                const double ang = two_pi * u2;
                fux[j] = sx + r * std::cos(ang);
                fuy[j] = sy + r * std::sin(ang);
            }

            const double tx = p_x[di];
            const double ty = p_y[di];
            const double min_sep = DEFAULT_SHIP_SEP_DIST;
            const double min_sep2 = min_sep * min_sep;
            const double half_k = 0.5 * sep_stiffness;
            const double cap2 = max_sep_push * max_sep_push;

            static thread_local double s_ox[SIM_MAX_UNITS];
            static thread_local double s_oy[SIM_MAX_UNITS];
            static thread_local double s_p_mr[SIM_MAX_PLANETS];
            static thread_local double s_p_mr2[SIM_MAX_PLANETS];
            static thread_local int32_t s_near_p[SIM_MAX_PLANETS];
            double max_mr = 0.0;
            for (int32_t pi = 0; pi < n_planets; ++pi) {
                const double mr = p_radius[pi] + ship_radius;
                if (mr > max_mr) max_mr = mr;
                s_p_mr[pi] = mr;
                s_p_mr2[pi] = mr * mr;
            }

            for (int32_t it = 0; it < SPAWN_SETTLE_ITERS; ++it) {
                double max_ov = 0.0;
                if (sep_stiffness > 0.0) {
                    if (max_sep_push > 0.0) {
                        std::memcpy(s_ox, fux, (size_t)parts * sizeof(double));
                        std::memcpy(s_oy, fuy, (size_t)parts * sizeof(double));
                    }
                    max_ov = separate_fleet_gs_core<false, false, true>(
                        0, parts, fux, fuy, nullptr, min_sep, min_sep2, half_k, max_sep_push
                    );
                    if (max_sep_push > 0.0) {
                        for (int32_t u = 0; u < parts; ++u) {
                            const double ddx = fux[u] - s_ox[u];
                            const double ddy = fuy[u] - s_oy[u];
                            const double d2 = ddx * ddx + ddy * ddy;
                            if (d2 > cap2) {
                                const double sc = max_sep_push / std::sqrt(d2);
                                fux[u] = s_ox[u] + ddx * sc;
                                fuy[u] = s_oy[u] + ddy * sc;
                            }
                        }
                    }
                }

                // Exact padded planet broadphase around spawned unit AABB
                double min_x = fux[0], max_x = fux[0], min_y = fuy[0], max_y = fuy[0];
                for (int32_t u = 1; u < parts; ++u) {
                    const double xu = fux[u];
                    const double yu = fuy[u];
                    if (xu < min_x) min_x = xu;
                    if (xu > max_x) max_x = xu;
                    if (yu < min_y) min_y = yu;
                    if (yu > max_y) max_y = yu;
                }
                bool any_exact_hit = false;
                for (int32_t pi = 0; pi < n_planets; ++pi) {
                    if (pi == di) continue;
                    const double mr = s_p_mr[pi];
                    if (p_x[pi] + mr >= min_x && p_x[pi] - mr <= max_x &&
                        p_y[pi] + mr >= min_y && p_y[pi] - mr <= max_y) {
                        any_exact_hit = true;
                        break;
                    }
                }

                double max_pp = 0.0;
                if (any_exact_hit) {
                    int32_t n_near = 0;
                    const double pad_min_x = min_x - max_mr;
                    const double pad_max_x = max_x + max_mr;
                    const double pad_min_y = min_y - max_mr;
                    const double pad_max_y = max_y + max_mr;
                    for (int32_t pi = 0; pi < n_planets; ++pi) {
                        if (pi == di) continue;
                        const double mr = s_p_mr[pi];
                        if (p_x[pi] + mr >= pad_min_x && p_x[pi] - mr <= pad_max_x &&
                            p_y[pi] + mr >= pad_min_y && p_y[pi] - mr <= pad_max_y) {
                            s_near_p[n_near++] = pi;
                        }
                    }
                    for (int32_t u = 0; u < parts; ++u) {
                        double x = fux[u];
                        double y = fuy[u];
                        for (int32_t ni = 0; ni < n_near; ++ni) {
                            const int32_t pi = s_near_p[ni];
                            const double px_i = p_x[pi];
                            const double py_i = p_y[pi];
                            const double ddx = x - px_i;
                            const double ddy = y - py_i;
                            const double d2 = ddx * ddx + ddy * ddy;
                            if (d2 < s_p_mr2[pi]) {
                                const double mr = s_p_mr[pi];
                                if (d2 > EPS_DIST2) {
                                    const double dist = std::sqrt(d2);
                                    const double pp = mr - dist;
                                    if (pp > max_pp) max_pp = pp;
                                    x = px_i + (ddx / dist) * mr;
                                    y = py_i + (ddy / dist) * mr;
                                } else {
                                    if (mr > max_pp) max_pp = mr;
                                    const double tdx = tx - px_i;
                                    const double tdy = ty - py_i;
                                    const double td2 = tdx * tdx + tdy * tdy;
                                    if (td2 > EPS_DIST2) {
                                        const double td = std::sqrt(td2);
                                        x = px_i + (tdx / td) * mr;
                                        y = py_i + (tdy / td) * mr;
                                    } else {
                                        x = px_i + mr;
                                        y = py_i;
                                    }
                                }
                            }
                        }
                        fux[u] = x;
                        fuy[u] = y;
                    }
                }

                if (max_ov <= SPAWN_SETTLE_TOL && max_pp <= SPAWN_SETTLE_TOL) {
                    break;
                }
            }

            double sum_x = 0.0;
            double sum_y = 0.0;
            for (int32_t j = 0; j < parts; ++j) {
                const double xj = fux[j];
                const double yj = fuy[j];
                sum_x += xj;
                sum_y += yj;
                const double hdx = tx - xj;
                const double hdy = ty - yj;
                const double hd2 = hdx * hdx + hdy * hdy;
                if (hd2 > EPS_DIST2) {
                    const double hd = std::sqrt(hd2);
                    uhx[u_start + j] = hdx / hd;
                    uhy[u_start + j] = hdy / hd;
                } else {
                    uhx[u_start + j] = hx0;
                    uhy[u_start + j] = hy0;
                }
            }
            const double cx = sum_x / (double)parts;
            const double cy = sum_y / (double)parts;
            double max_d2 = 0.0;
            for (int32_t j = 0; j < parts; ++j) {
                const double ddx = fux[j] - cx;
                const double ddy = fuy[j] - cy;
                const double dd2 = ddx * ddx + ddy * ddy;
                if (dd2 > max_d2) max_d2 = dd2;
            }
            fl_cx[f] = cx;
            fl_cy[f] = cy;
            fl_radius[f] = std::sqrt(max_d2) + ship_radius;
        }
        n_units += parts;
        return fid;
    }

    void apply_redir(int32_t pid, int32_t fleet_id, int32_t dst_id) {
        if (!allow_redirect) return;
        const int32_t di = find_planet_idx(dst_id);
        if (di < 0) return;
        for (int32_t f = 0; f < n_fleets; ++f) {
            if (fl_id[f] == fleet_id && fl_owner[f] == pid && fl_target_idx[f] != di) {
                fl_target_idx[f] = di;
                return;
            }
        }
    }

    void step_one_tick() {
        if (done) return;

        // Recompute ships_per_unit identically to Python's Fleet.ships_per_unit
        // (fl.ships / max(1, fl.num_units)) for bit-exact parity across partial landings.
        for (int32_t f = 0; f < n_fleets; ++f) {
            const int32_t c = fl_unit_count[f];
            fl_spu[f] = fl_ships[f] / (double)(c > 1 ? c : 1);
        }

        PhysicsScratch& s = get_physics_scratch();

        n_units = run_physics_substeps(
            n_planets,
            p_owner, p_ships, p_x, p_y, p_prod, p_radius,
            n_fleets,
            fl_owner, fl_target_idx, fl_spu, fl_unit_start, fl_unit_count,
            fl_ships, fl_cx, fl_cy, fl_radius,
            n_units, ux, uy, uhx, uhy,
            tick_dt, fleet_speed, substeps, ship_radius,
            sep_stiffness, max_sep_push, sep_mode, prod_divisor, obstacle_escape_eps,
            turn_rate, turn_min_speed_factor,
            s.u_alive, s.fl_alive_cnt, s.tmp_x, s.tmp_y,
            s.arr_buckets, s.dirty_planets, s.near_planets,
            s.p_min_r, s.p_min_r2, s.p_land_r, s.p_land_r2_guard, s.p_prod_sub,
            s.tgt_head, s.grp_head, s.grp_tail, s.grp_cnt, s.grp_owner, s.grp_next, s.fl_next, s.grp_uids, s.grp_x, s.grp_y
        );

        int32_t wf = 0;
        for (int32_t f = 0; f < n_fleets; ++f) {
            if (fl_unit_count[f] > 0) {
                if (wf != f) {
                    fl_id[wf] = fl_id[f];
                    fl_owner[wf] = fl_owner[f];
                    fl_source_id[wf] = fl_source_id[f];
                    fl_target_idx[wf] = fl_target_idx[f];
                    fl_spu[wf] = fl_spu[f];
                    fl_unit_start[wf] = fl_unit_start[f];
                    fl_unit_count[wf] = fl_unit_count[f];
                    fl_ships[wf] = fl_ships[f];
                    fl_cx[wf] = fl_cx[f];
                    fl_cy[wf] = fl_cy[f];
                    fl_radius[wf] = fl_radius[f];
                }
                ++wf;
            }
        }
        n_fleets = wf;

        tick += 1;
        check_terminal();
    }

    struct MultiTickCtx {
        SimState* self;
        int32_t stepped;
    };

    static bool on_multi_tick_end(void* raw_ctx) {
        MultiTickCtx* ctx = static_cast<MultiTickCtx*>(raw_ctx);
        SimState* st = ctx->self;
        st->tick += 1;
        ctx->stepped += 1;
        st->check_terminal();
        return st->done != 0;
    }

    int32_t step_n_ticks_fast(int32_t n_ticks) {
        if (done || n_ticks <= 0) return 0;
        if (n_ticks == 1) {
            step_one_tick();
            return 1;
        }

        for (int32_t f = 0; f < n_fleets; ++f) {
            const int32_t c = fl_unit_count[f];
            fl_spu[f] = fl_ships[f] / (double)(c > 1 ? c : 1);
        }

        PhysicsScratch& s = get_physics_scratch();
        MultiTickCtx ctx{this, 0};
        n_units = run_physics_substeps_core(
            n_planets,
            p_owner, p_ships, p_x, p_y, p_prod, p_radius,
            n_fleets,
            fl_owner, fl_target_idx, fl_spu, fl_unit_start, fl_unit_count,
            fl_ships, fl_cx, fl_cy, fl_radius,
            n_units, ux, uy, uhx, uhy,
            tick_dt, fleet_speed, substeps, ship_radius,
            sep_stiffness, max_sep_push, sep_mode, prod_divisor, obstacle_escape_eps,
            turn_rate, turn_min_speed_factor,
            s.u_alive, s.fl_alive_cnt, s.tmp_x, s.tmp_y,
            s.arr_buckets, s.dirty_planets, s.near_planets,
            s.p_min_r, s.p_min_r2, s.p_land_r, s.p_land_r2_guard, s.p_prod_sub,
            s.tgt_head, s.grp_head, s.grp_tail, s.grp_cnt, s.grp_owner, s.grp_next, s.fl_next, s.grp_uids, s.grp_x, s.grp_y,
            n_ticks,
            &SimState::on_multi_tick_end,
            &ctx
        );

        int32_t wf = 0;
        for (int32_t f = 0; f < n_fleets; ++f) {
            if (fl_unit_count[f] > 0) {
                if (wf != f) {
                    fl_id[wf] = fl_id[f];
                    fl_owner[wf] = fl_owner[f];
                    fl_source_id[wf] = fl_source_id[f];
                    fl_target_idx[wf] = fl_target_idx[f];
                    fl_spu[wf] = fl_spu[f];
                    fl_unit_start[wf] = fl_unit_start[f];
                    fl_unit_count[wf] = fl_unit_count[f];
                    fl_ships[wf] = fl_ships[f];
                    fl_cx[wf] = fl_cx[f];
                    fl_cy[wf] = fl_cy[f];
                    fl_radius[wf] = fl_radius[f];
                }
                ++wf;
            }
        }
        n_fleets = wf;
        return ctx.stepped;
    }
};

static inline void copy_sim_state_fast(const SimState* src, SimState* dst) {
    dst->width = src->width;
    dst->height = src->height;
    dst->fleet_speed = src->fleet_speed;
    dst->tick_dt = src->tick_dt;
    dst->ship_radius = src->ship_radius;
    dst->sep_stiffness = src->sep_stiffness;
    dst->max_sep_push = src->max_sep_push;
    dst->prod_divisor = src->prod_divisor;
    dst->obstacle_escape_eps = src->obstacle_escape_eps;
    dst->turn_rate = src->turn_rate;
    dst->turn_min_speed_factor = src->turn_min_speed_factor;
    dst->max_ticks = src->max_ticks;
    dst->substeps = src->substeps;
    dst->sep_mode = src->sep_mode;
    dst->allow_redirect = src->allow_redirect;
    dst->items_max = src->items_max;
    dst->items_reserved = src->items_reserved;
    dst->base_item_users = src->base_item_users;
    dst->tick = src->tick;
    dst->next_fleet_id = src->next_fleet_id;
    dst->done = src->done;
    dst->winner = src->winner;

    const int32_t np = src->n_planets;
    dst->n_planets = np;
    if (np > 0) {
        std::memcpy(dst->p_id, src->p_id, (size_t)np * sizeof(int32_t));
        std::memcpy(dst->p_owner, src->p_owner, (size_t)np * sizeof(int32_t));
        std::memcpy(dst->p_ships, src->p_ships, (size_t)np * sizeof(double));
        std::memcpy(dst->p_x, src->p_x, (size_t)np * sizeof(double));
        std::memcpy(dst->p_y, src->p_y, (size_t)np * sizeof(double));
        std::memcpy(dst->p_prod, src->p_prod, (size_t)np * sizeof(double));
        std::memcpy(dst->p_radius, src->p_radius, (size_t)np * sizeof(double));
    }

    const int32_t nf = src->n_fleets;
    dst->n_fleets = nf;
    if (nf > 0) {
        std::memcpy(dst->fl_id, src->fl_id, (size_t)nf * sizeof(int32_t));
        std::memcpy(dst->fl_owner, src->fl_owner, (size_t)nf * sizeof(int32_t));
        std::memcpy(dst->fl_source_id, src->fl_source_id, (size_t)nf * sizeof(int32_t));
        std::memcpy(dst->fl_target_idx, src->fl_target_idx, (size_t)nf * sizeof(int32_t));
        std::memcpy(dst->fl_spu, src->fl_spu, (size_t)nf * sizeof(double));
        std::memcpy(dst->fl_unit_start, src->fl_unit_start, (size_t)nf * sizeof(int32_t));
        std::memcpy(dst->fl_unit_count, src->fl_unit_count, (size_t)nf * sizeof(int32_t));
        std::memcpy(dst->fl_ships, src->fl_ships, (size_t)nf * sizeof(double));
        std::memcpy(dst->fl_cx, src->fl_cx, (size_t)nf * sizeof(double));
        std::memcpy(dst->fl_cy, src->fl_cy, (size_t)nf * sizeof(double));
        std::memcpy(dst->fl_radius, src->fl_radius, (size_t)nf * sizeof(double));
    }

    const int32_t nu = src->n_units;
    dst->n_units = nu;
    if (nu > 0) {
        std::memcpy(dst->ux, src->ux, (size_t)nu * sizeof(double));
        std::memcpy(dst->uy, src->uy, (size_t)nu * sizeof(double));
        std::memcpy(dst->uhx, src->uhx, (size_t)nu * sizeof(double));
        std::memcpy(dst->uhy, src->uhy, (size_t)nu * sizeof(double));
    }
}

// ============================================================================
// C ABI Exports
// ============================================================================

extern "C" {

int32_t step_physics_cpp(
    int32_t n_planets,
    int32_t* p_owner,
    double* p_ships,
    const double* p_x,
    const double* p_y,
    const double* p_prod,
    const double* p_radius,
    int32_t n_fleets,
    const int32_t* fl_owner,
    const int32_t* fl_target_idx,
    const double* fl_spu,
    int32_t* fl_unit_start,
    int32_t* fl_unit_count,
    double* fl_out_ships,
    double* fl_out_cx,
    double* fl_out_cy,
    double* fl_out_radius,
    int32_t total_units,
    double* ux,
    double* uy,
    double* uhx,
    double* uhy,
    double tick_dt,
    double fleet_speed,
    int32_t substeps,
    double ship_radius,
    double sep_stiffness,
    double max_sep_push,
    int32_t sep_mode,
    double prod_divisor,
    double obstacle_escape_eps,
    double turn_rate,
    double turn_min_speed_factor
) {
    if (n_planets <= 0) return 0;

    if (n_planets <= SIM_MAX_PLANETS && n_fleets <= SIM_MAX_FLEETS && total_units <= SIM_MAX_UNITS) {
        PhysicsScratch& s = get_physics_scratch();

        return run_physics_substeps(
            n_planets, p_owner, p_ships, p_x, p_y, p_prod, p_radius,
            n_fleets, fl_owner, fl_target_idx, fl_spu, fl_unit_start, fl_unit_count,
            fl_out_ships, fl_out_cx, fl_out_cy, fl_out_radius,
            total_units, ux, uy, uhx, uhy,
            tick_dt, fleet_speed, substeps, ship_radius,
            sep_stiffness, max_sep_push, sep_mode, prod_divisor, obstacle_escape_eps,
            turn_rate, turn_min_speed_factor,
            s.u_alive, s.fl_alive_cnt, s.tmp_x, s.tmp_y,
            s.arr_buckets, s.dirty_planets, s.near_planets,
            s.p_min_r, s.p_min_r2, s.p_land_r, s.p_land_r2_guard, s.p_prod_sub,
            s.tgt_head, s.grp_head, s.grp_tail, s.grp_cnt, s.grp_owner, s.grp_next, s.fl_next, s.grp_uids, s.grp_x, s.grp_y
        );
    }

    const size_t sz_p = (size_t)n_planets;
    const size_t sz_f = (size_t)(n_fleets > 0 ? n_fleets : 0);
    const size_t sz_u = (size_t)(total_units > 0 ? total_units : 0);

    std::vector<uint8_t> u_alive(sz_u);
    std::vector<int32_t> fl_alive_cnt(sz_f);
    std::vector<double> tmp_x(sz_u);
    std::vector<double> tmp_y(sz_u);
    std::vector<ArrivalBucket> arr_buckets(sz_p);
    std::vector<int32_t> dirty_planets(sz_p);
    std::vector<int32_t> near_planets(sz_p);
    std::vector<double> p_min_r(sz_p);
    std::vector<double> p_min_r2(sz_p);
    std::vector<double> p_land_r(sz_p);
    std::vector<double> p_land_r2_guard(sz_p);
    std::vector<double> p_prod_sub(sz_p);
    std::vector<int32_t> tgt_head(sz_p);
    std::vector<int32_t> grp_head(sz_f);
    std::vector<int32_t> grp_tail(sz_f);
    std::vector<int32_t> grp_cnt(sz_f);
    std::vector<int32_t> grp_owner(sz_f);
    std::vector<int32_t> grp_next(sz_f);
    std::vector<int32_t> fl_next(sz_f);
    std::vector<int32_t> grp_uids(sz_u);
    std::vector<double> grp_x(sz_u);
    std::vector<double> grp_y(sz_u);

    return run_physics_substeps(
        n_planets, p_owner, p_ships, p_x, p_y, p_prod, p_radius,
        n_fleets, fl_owner, fl_target_idx, fl_spu, fl_unit_start, fl_unit_count,
        fl_out_ships, fl_out_cx, fl_out_cy, fl_out_radius,
        total_units, ux, uy, uhx, uhy,
        tick_dt, fleet_speed, substeps, ship_radius,
        sep_stiffness, max_sep_push, sep_mode, prod_divisor, obstacle_escape_eps,
        turn_rate, turn_min_speed_factor,
        u_alive.data(), fl_alive_cnt.data(), tmp_x.data(), tmp_y.data(),
        arr_buckets.data(), dirty_planets.data(), near_planets.data(),
        p_min_r.data(), p_min_r2.data(), p_land_r.data(), p_land_r2_guard.data(), p_prod_sub.data(),
        tgt_head.data(), grp_head.data(), grp_tail.data(), grp_cnt.data(), grp_owner.data(), grp_next.data(), fl_next.data(), grp_uids.data(), grp_x.data(), grp_y.data()
    );
}

SimState* sim_create(
    int32_t n_planets,
    const int32_t* p_id,
    const int32_t* p_owner,
    const double* p_ships,
    const double* p_x,
    const double* p_y,
    const double* p_prod,
    const double* p_radius,
    double width,
    double height,
    double fleet_speed,
    double tick_dt,
    int32_t max_ticks,
    int32_t substeps,
    double ship_radius,
    double sep_stiffness,
    double max_sep_push,
    int32_t sep_mode,
    int32_t allow_redirect,
    double obstacle_escape_eps,
    double turn_rate,
    double turn_min_speed_factor
) {
    if (n_planets < 0 || n_planets > SIM_MAX_PLANETS) return nullptr;
    SimState* st = new SimState();
    st->width = width;
    st->height = height;
    st->fleet_speed = fleet_speed;
    st->tick_dt = tick_dt;
    st->ship_radius = ship_radius;
    st->sep_stiffness = sep_stiffness;
    st->max_sep_push = max_sep_push;
    st->prod_divisor = PRODUCTION_DIVISOR;
    st->obstacle_escape_eps = obstacle_escape_eps;
    st->turn_rate = turn_rate;
    st->turn_min_speed_factor = turn_min_speed_factor;
    st->max_ticks = max_ticks;
    st->substeps = substeps < 1 ? 1 : substeps;
    st->sep_mode = sep_mode;
    st->allow_redirect = allow_redirect;
    st->items_max = ITEMS_MAX;
    st->items_reserved = ITEMS_RESERVED;
    st->base_item_users = BASE_ITEM_USERS;
    st->tick = 0;
    st->next_fleet_id = 1000;
    st->done = 0;
    st->winner = 0;

    st->n_planets = n_planets;
    for (int32_t i = 0; i < n_planets; ++i) {
        st->p_id[i] = p_id[i];
        st->p_owner[i] = p_owner[i];
        st->p_ships[i] = p_ships[i];
        st->p_x[i] = p_x[i];
        st->p_y[i] = p_y[i];
        st->p_prod[i] = p_prod[i];
        st->p_radius[i] = p_radius[i];
    }
    st->n_fleets = 0;
    st->n_units = 0;
    return st;
}

void sim_set_meta(SimState* st, int32_t tick, int32_t next_fleet_id, int32_t done, int32_t winner) {
    if (!st) return;
    st->tick = tick;
    st->next_fleet_id = next_fleet_id;
    st->done = done;
    st->winner = winner;
}

int32_t sim_load_fleets(
    SimState* st,
    int32_t n_fleets,
    const int32_t* fl_id,
    const int32_t* fl_owner,
    const int32_t* fl_source_id,
    const int32_t* fl_target_idx,
    const double* fl_ships,
    const double* fl_cx,
    const double* fl_cy,
    const double* fl_radius,
    const int32_t* fl_unit_start,
    const int32_t* fl_unit_count,
    int32_t n_units,
    const double* ux,
    const double* uy,
    const double* uhx,
    const double* uhy
) {
    if (!st) return -1;
    if (n_fleets < 0 || n_fleets > SIM_MAX_FLEETS) return -1;
    if (n_units < 0 || n_units > SIM_MAX_UNITS) return -1;
    st->n_fleets = n_fleets;
    st->n_units = n_units;
    for (int32_t f = 0; f < n_fleets; ++f) {
        st->fl_id[f] = fl_id[f];
        st->fl_owner[f] = fl_owner[f];
        st->fl_source_id[f] = fl_source_id[f];
        st->fl_target_idx[f] = fl_target_idx[f];
        st->fl_ships[f] = fl_ships[f];
        st->fl_cx[f] = fl_cx[f];
        st->fl_cy[f] = fl_cy[f];
        st->fl_radius[f] = fl_radius[f];
        st->fl_unit_start[f] = fl_unit_start[f];
        const int32_t cnt = fl_unit_count[f];
        st->fl_unit_count[f] = cnt;
        st->fl_spu[f] = fl_ships[f] / (double)(cnt > 1 ? cnt : 1);
    }
    if (n_units > 0) {
        std::memcpy(st->ux, ux, (size_t)n_units * sizeof(double));
        std::memcpy(st->uy, uy, (size_t)n_units * sizeof(double));
        if (uhx && uhy) {
            std::memcpy(st->uhx, uhx, (size_t)n_units * sizeof(double));
            std::memcpy(st->uhy, uhy, (size_t)n_units * sizeof(double));
        } else {
            for (int32_t f = 0; f < n_fleets; ++f) {
                const int32_t tgt = st->fl_target_idx[f];
                const int32_t start = st->fl_unit_start[f];
                const int32_t end = start + st->fl_unit_count[f];
                if (tgt >= 0 && tgt < st->n_planets) {
                    const double tx = st->p_x[tgt];
                    const double ty = st->p_y[tgt];
                    for (int32_t u = start; u < end; ++u) {
                        const double dx = tx - st->ux[u];
                        const double dy = ty - st->uy[u];
                        const double d2 = dx * dx + dy * dy;
                        if (d2 > EPS_DIST2) {
                            const double d = std::sqrt(d2);
                            st->uhx[u] = dx / d;
                            st->uhy[u] = dy / d;
                        } else {
                            st->uhx[u] = 1.0;
                            st->uhy[u] = 0.0;
                        }
                    }
                } else {
                    for (int32_t u = start; u < end; ++u) {
                        st->uhx[u] = 1.0;
                        st->uhy[u] = 0.0;
                    }
                }
            }
        }
    }
    return 0;
}

void sim_destroy(SimState* st) {
    delete st;
}

SimState* sim_clone(const SimState* src) {
    if (!src) return nullptr;
    SimState* dst = new SimState();
    copy_sim_state_fast(src, dst);
    return dst;
}

void sim_copy_into(const SimState* src, SimState* dst) {
    if (!src || !dst) return;
    copy_sim_state_fast(src, dst);
}

int32_t sim_send_order(SimState* st, int32_t player, int32_t src_id, int32_t dst_id, int32_t percent) {
    if (!st || st->done) return -1;
    return st->apply_send(player, src_id, dst_id, percent);
}

void sim_redir_order(SimState* st, int32_t player, int32_t fleet_id, int32_t dst_id) {
    if (!st || st->done) return;
    st->apply_redir(player, fleet_id, dst_id);
}

void sim_step_with_orders(SimState* st, int32_t n_orders, const COrder* orders) {
    if (!st || st->done) return;
    for (int32_t i = 0; i < n_orders; ++i) {
        const COrder& o = orders[i];
        if (o.kind == 0) {
            st->apply_send(o.player, o.a, o.b, o.percent);
        } else if (o.kind == 1) {
            st->apply_redir(o.player, o.a, o.b);
        }
    }
    st->step_one_tick();
}

int32_t sim_step_ticks(SimState* st, int32_t n_ticks) {
    if (!st) return 0;
    return st->step_n_ticks_fast(n_ticks);
}

void sim_get_meta(const SimState* st, int32_t* out_meta) {
    if (!st || !out_meta) return;
    out_meta[0] = st->tick;
    out_meta[1] = st->done;
    out_meta[2] = st->winner;
    out_meta[3] = st->n_planets;
    out_meta[4] = st->n_fleets;
    out_meta[5] = st->n_units;
    out_meta[6] = st->next_fleet_id;
}

void sim_get_config(const SimState* st, double* out_f64, int32_t* out_i32) {
    if (!st) return;
    if (out_f64) {
        out_f64[0] = st->width;
        out_f64[1] = st->height;
        out_f64[2] = st->fleet_speed;
        out_f64[3] = st->tick_dt;
        out_f64[4] = st->ship_radius;
        out_f64[5] = st->sep_stiffness;
        out_f64[6] = st->max_sep_push;
        out_f64[7] = st->obstacle_escape_eps;
        out_f64[8] = st->turn_rate;
        out_f64[9] = st->turn_min_speed_factor;
    }
    if (out_i32) {
        out_i32[0] = st->max_ticks;
        out_i32[1] = st->substeps;
        out_i32[2] = st->sep_mode;
        out_i32[3] = st->allow_redirect;
    }
}

void sim_read_planets(const SimState* st, int32_t* out_owner, double* out_ships) {
    if (!st || st->n_planets <= 0) return;
    const int32_t np = st->n_planets;
    std::memcpy(out_owner, st->p_owner, (size_t)np * sizeof(int32_t));
    std::memcpy(out_ships, st->p_ships, (size_t)np * sizeof(double));
}

void sim_read_planets_full(
    const SimState* st,
    int32_t* out_id,
    int32_t* out_owner,
    double* out_ships,
    double* out_x,
    double* out_y,
    double* out_prod,
    double* out_radius
) {
    if (!st || st->n_planets <= 0) return;
    const int32_t np = st->n_planets;
    std::memcpy(out_id, st->p_id, (size_t)np * sizeof(int32_t));
    std::memcpy(out_owner, st->p_owner, (size_t)np * sizeof(int32_t));
    std::memcpy(out_ships, st->p_ships, (size_t)np * sizeof(double));
    std::memcpy(out_x, st->p_x, (size_t)np * sizeof(double));
    std::memcpy(out_y, st->p_y, (size_t)np * sizeof(double));
    std::memcpy(out_prod, st->p_prod, (size_t)np * sizeof(double));
    std::memcpy(out_radius, st->p_radius, (size_t)np * sizeof(double));
}

void sim_read_fleets(
    const SimState* st,
    int32_t* out_id,
    int32_t* out_owner,
    int32_t* out_source_id,
    int32_t* out_target_idx,
    int32_t* out_unit_start,
    int32_t* out_unit_count,
    double* out_ships,
    double* out_cx,
    double* out_cy,
    double* out_radius
) {
    if (!st || st->n_fleets <= 0) return;
    const int32_t nf = st->n_fleets;
    std::memcpy(out_id, st->fl_id, (size_t)nf * sizeof(int32_t));
    std::memcpy(out_owner, st->fl_owner, (size_t)nf * sizeof(int32_t));
    if (out_source_id) {
        std::memcpy(out_source_id, st->fl_source_id, (size_t)nf * sizeof(int32_t));
    }
    std::memcpy(out_target_idx, st->fl_target_idx, (size_t)nf * sizeof(int32_t));
    std::memcpy(out_unit_start, st->fl_unit_start, (size_t)nf * sizeof(int32_t));
    std::memcpy(out_unit_count, st->fl_unit_count, (size_t)nf * sizeof(int32_t));
    std::memcpy(out_ships, st->fl_ships, (size_t)nf * sizeof(double));
    std::memcpy(out_cx, st->fl_cx, (size_t)nf * sizeof(double));
    std::memcpy(out_cy, st->fl_cy, (size_t)nf * sizeof(double));
    std::memcpy(out_radius, st->fl_radius, (size_t)nf * sizeof(double));
}

void sim_read_units(const SimState* st, double* out_ux, double* out_uy) {
    if (!st || st->n_units <= 0) return;
    const int32_t nu = st->n_units;
    std::memcpy(out_ux, st->ux, (size_t)nu * sizeof(double));
    std::memcpy(out_uy, st->uy, (size_t)nu * sizeof(double));
}

void sim_read_unit_headings(const SimState* st, double* out_uhx, double* out_uhy) {
    if (!st || st->n_units <= 0) return;
    const int32_t nu = st->n_units;
    std::memcpy(out_uhx, st->uhx, (size_t)nu * sizeof(double));
    std::memcpy(out_uhy, st->uhy, (size_t)nu * sizeof(double));
}

struct SimStateView {
    int32_t tick;
    int32_t max_ticks;
    int32_t done;
    int32_t winner;
    int32_t n_planets;
    int32_t n_fleets;
    double tick_dt;
    double fleet_speed;
    double prod_divisor;
    const int32_t* p_id;
    const int32_t* p_owner;
    const double* p_ships;
    const double* p_x;
    const double* p_y;
    const double* p_prod;
    const double* p_radius;
    const int32_t* fl_target_idx;
    const int32_t* fl_owner;
    const double* fl_ships;
    const double* fl_cx;
    const double* fl_cy;
};

void sim_get_state_view(const SimState* st, SimStateView* out) {
    if (!st || !out) return;
    out->tick = st->tick;
    out->max_ticks = st->max_ticks;
    out->done = st->done;
    out->winner = st->winner;
    out->n_planets = st->n_planets;
    out->n_fleets = st->n_fleets;
    out->tick_dt = st->tick_dt;
    out->fleet_speed = st->fleet_speed;
    out->prod_divisor = st->prod_divisor;
    out->p_id = st->p_id;
    out->p_owner = st->p_owner;
    out->p_ships = st->p_ships;
    out->p_x = st->p_x;
    out->p_y = st->p_y;
    out->p_prod = st->p_prod;
    out->p_radius = st->p_radius;
    out->fl_target_idx = st->fl_target_idx;
    out->fl_owner = st->fl_owner;
    out->fl_ships = st->fl_ships;
    out->fl_cx = st->fl_cx;
    out->fl_cy = st->fl_cy;
}

double sim_eval_heuristic_from_po(const SimState* st, int32_t player, const float* po_all, int32_t horizon) {
    if (!st) return 0.0;
    if (st->done) {
        return (st->winner == player) ? 1.0 : ((st->winner == 0) ? 0.0 : -1.0);
    }
    const int32_t opp = (player == 1) ? 2 : 1;
    double my_ships = 0.0, en_ships = 0.0;
    double my_prod = 0.0, en_prod = 0.0;
    double fut_my_prod = 0.0, fut_en_prod = 0.0;
    const int32_t np = st->n_planets;
    const int32_t last_col = (horizon > 0) ? (horizon - 1) : 0;
    for (int32_t i = 0; i < np; ++i) {
        const int32_t o = st->p_owner[i];
        const double pr = st->p_prod[i];
        if (o == player) {
            my_ships += st->p_ships[i];
            my_prod += pr;
        } else if (o == opp) {
            en_ships += st->p_ships[i];
            en_prod += pr;
        }
        if (po_all && horizon > 0) {
            const float po = po_all[i * horizon + last_col];
            if (po > 0.0f) fut_my_prod += pr;
            else if (po < 0.0f) fut_en_prod += pr;
        }
    }
    const int32_t nf = st->n_fleets;
    for (int32_t f = 0; f < nf; ++f) {
        const int32_t o = st->fl_owner[f];
        if (o == player) my_ships += st->fl_ships[f];
        else if (o == opp) en_ships += st->fl_ships[f];
    }
    const double ship_denom = (my_ships + en_ships > 30.0) ? (my_ships + en_ships) : 30.0;
    const double prod_denom = (my_prod + en_prod > 40.0) ? (my_prod + en_prod) : 40.0;
    const double fut_denom = (fut_my_prod + fut_en_prod > 40.0) ? (fut_my_prod + fut_en_prod) : 40.0;
    const double ship_adv = (my_ships - en_ships) / ship_denom;
    const double prod_adv = (my_prod - en_prod) / prod_denom;
    const double fut_prod_adv = (fut_my_prod - fut_en_prod) / fut_denom;
    const double raw = 1.15 * ship_adv + 0.70 * prod_adv + 1.10 * fut_prod_adv;
    return std::tanh(raw);
}

double sim_eval_heuristic(const SimState* st, int32_t player, int32_t horizon) {
    if (!st) return 0.0;
    if (st->done) {
        return (st->winner == player) ? 1.0 : ((st->winner == 0) ? 0.0 : -1.0);
    }
    const int32_t np = st->n_planets;
    if (np <= 0) return 0.0;
    const int32_t H = (horizon > 0 && horizon <= 64) ? horizon : 24;
    const int32_t opp = (player == 1) ? 2 : 1;
    const double step_dist_raw = st->fleet_speed * st->tick_dt;
    const double step_dist = (step_dist_raw > 1e-6) ? step_dist_raw : 1e-6;

    double my_ships = 0.0, en_ships = 0.0;
    double my_prod = 0.0, en_prod = 0.0;
    for (int32_t i = 0; i < np; ++i) {
        const int32_t o = st->p_owner[i];
        if (o == player) {
            my_ships += st->p_ships[i];
            my_prod += st->p_prod[i];
        } else if (o == opp) {
            en_ships += st->p_ships[i];
            en_prod += st->p_prod[i];
        }
    }

    static thread_local float s_my_arr[SIM_MAX_PLANETS * 64];
    static thread_local float s_en_arr[SIM_MAX_PLANETS * 64];
    uint8_t has_inc[SIM_MAX_PLANETS];
    std::memset(has_inc, 0, (size_t)np);
    std::memset(s_my_arr, 0, (size_t)(np * H) * sizeof(float));
    std::memset(s_en_arr, 0, (size_t)(np * H) * sizeof(float));

    double fut_my_ships = 0.0, fut_en_ships = 0.0;
    const int32_t nf = st->n_fleets;
    for (int32_t f = 0; f < nf; ++f) {
        const int32_t fo = st->fl_owner[f];
        const double fs = st->fl_ships[f];
        if (fo == player) my_ships += fs;
        else if (fo == opp) en_ships += fs;
        const int32_t idx_p = st->fl_target_idx[f];
        if (idx_p < 0 || idx_p >= np) {
            if (fo == player) fut_my_ships += fs;
            else if (fo == opp) fut_en_ships += fs;
            continue;
        }
        const double dx = st->p_x[idx_p] - st->fl_cx[f];
        const double dy = st->p_y[idx_p] - st->fl_cy[f];
        const double raw_d = std::sqrt(dx * dx + dy * dy) - st->p_radius[idx_p] - st->ship_radius;
        const double dist = (raw_d > 0.0) ? raw_d : 0.0;
        int32_t eta = static_cast<int32_t>(std::ceil((dist - ARRIVAL_EPS) / step_dist));
        if (eta < 1) eta = 1;
        const int32_t t_i = eta - 1;
        if (t_i >= 0 && t_i < H) {
            has_inc[idx_p] = 1;
            if (fo == player) s_my_arr[idx_p * H + t_i] += static_cast<float>(fs);
            else s_en_arr[idx_p * H + t_i] += static_cast<float>(fs);
        } else {
            if (fo == player) fut_my_ships += fs;
            else if (fo == opp) fut_en_ships += fs;
        }
    }

    double fut_my_prod = 0.0, fut_en_prod = 0.0;
    for (int32_t i = 0; i < np; ++i) {
        int32_t cur_owner = (st->p_owner[i] == player) ? 1 : ((st->p_owner[i] == 0) ? 0 : -1);
        double cur_ships = st->p_ships[i];
        const double g_i = (st->p_prod[i] / st->prod_divisor) * st->tick_dt;
        if (has_inc[i]) {
            const int32_t base = i * H;
            for (int32_t t = 0; t < H; ++t) {
                if (cur_owner != 0) cur_ships += g_i;
                const double ma = static_cast<double>(s_my_arr[base + t]);
                const double ea = static_cast<double>(s_en_arr[base + t]);
                if (ma > 0.0 || ea > 0.0) {
                    const double f_my = ma + ((cur_owner == 1) ? cur_ships : 0.0);
                    const double f_en = ea + ((cur_owner == -1) ? cur_ships : 0.0);
                    const double f_nu = (cur_owner == 0) ? cur_ships : 0.0;
                    double top_v, sec_v;
                    int32_t top_o;
                    if (f_my >= f_en && f_my >= f_nu) {
                        top_v = f_my; top_o = 1;
                        sec_v = (f_en >= f_nu) ? f_en : f_nu;
                    } else if (f_en >= f_my && f_en >= f_nu) {
                        top_v = f_en; top_o = -1;
                        sec_v = (f_my >= f_nu) ? f_my : f_nu;
                    } else {
                        top_v = f_nu; top_o = 0;
                        sec_v = (f_my >= f_en) ? f_my : f_en;
                    }
                    if (std::fabs(top_v - sec_v) <= TIE_EPS) {
                        cur_ships = 0.0;
                    } else {
                        cur_owner = top_o;
                        cur_ships = top_v - sec_v;
                    }
                }
            }
        } else if (cur_owner != 0) {
            cur_ships += g_i * static_cast<double>(H);
        }
        if (cur_owner > 0) {
            fut_my_prod += st->p_prod[i];
            fut_my_ships += cur_ships;
        } else if (cur_owner < 0) {
            fut_en_prod += st->p_prod[i];
            fut_en_ships += cur_ships;
        }
    }

    const double ship_denom = (my_ships + en_ships > 30.0) ? (my_ships + en_ships) : 30.0;
    const double fut_ship_denom = (fut_my_ships + fut_en_ships > 30.0) ? (fut_my_ships + fut_en_ships) : 30.0;
    const double prod_denom = (my_prod + en_prod > 40.0) ? (my_prod + en_prod) : 40.0;
    const double fut_denom = (fut_my_prod + fut_en_prod > 40.0) ? (fut_my_prod + fut_en_prod) : 40.0;
    const double ship_adv = (my_ships - en_ships) / ship_denom;
    const double fut_ship_adv = (fut_my_ships - fut_en_ships) / fut_ship_denom;
    const double prod_adv = (my_prod - en_prod) / prod_denom;
    const double fut_prod_adv = (fut_my_prod - fut_en_prod) / fut_denom;
    const double raw = 0.45 * ship_adv + 0.90 * fut_ship_adv + 0.50 * prod_adv + 1.10 * fut_prod_adv;
    return std::tanh(raw);
}

int32_t sim_update_planets(
    SimState* st,
    int32_t n_planets,
    const int32_t* p_id,
    const int32_t* p_owner,
    const double* p_ships,
    const double* p_x,
    const double* p_y,
    const double* p_prod,
    const double* p_radius,
    double fleet_speed,
    double tick_dt,
    int32_t substeps,
    int32_t clear_fleets
) {
    if (!st || n_planets < 0 || n_planets > SIM_MAX_PLANETS) return -1;
    st->fleet_speed = fleet_speed;
    st->tick_dt = tick_dt;
    st->substeps = substeps < 1 ? 1 : substeps;
    st->n_planets = n_planets;
    for (int32_t i = 0; i < n_planets; ++i) {
        st->p_id[i] = p_id[i];
        st->p_owner[i] = p_owner[i];
        st->p_ships[i] = p_ships[i];
        st->p_x[i] = p_x[i];
        st->p_y[i] = p_y[i];
        st->p_prod[i] = p_prod[i];
        st->p_radius[i] = p_radius[i];
    }
    if (clear_fleets) {
        st->n_fleets = 0;
        st->n_units = 0;
        st->tick = 0;
        st->done = 0;
        st->winner = 0;
    }
    return 0;
}

int32_t sim_spawn_fleet_exact(
    SimState* st,
    int32_t owner,
    int32_t src_idx,
    int32_t dst_idx,
    double ships_to_send,
    int32_t fleet_id,
    double* out_cx,
    double* out_cy,
    double* out_radius,
    int32_t* out_num_units
) {
    if (!st) return -1;
    if (src_idx < 0 || src_idx >= st->n_planets || dst_idx < 0 || dst_idx >= st->n_planets) return -1;

    double total_planet_ships = 0.0;
    for (int32_t i = 0; i < st->n_planets; ++i) {
        if (st->p_ships[i] > 0.0) {
            total_planet_ships += st->p_ships[i];
        }
    }
    const int32_t parts = physical_ship_count_cpp(
        ships_to_send, total_planet_ships, st->item_count(), st->items_max, st->items_reserved
    );
    if (parts <= 0) return 1;
    if (st->n_fleets >= SIM_MAX_FLEETS || st->n_units + parts > SIM_MAX_UNITS) return -1;

    const double rem = st->p_ships[src_idx] - ships_to_send;
    st->p_ships[src_idx] = rem > 0.0 ? rem : 0.0;

    const double dx = st->p_x[dst_idx] - st->p_x[src_idx];
    const double dy = st->p_y[dst_idx] - st->p_y[src_idx];
    const double raw_d = std::hypot(dx, dy);
    const double d = (raw_d > 1e-6) ? raw_d : 1e-6;
    const double hx0 = (raw_d > 1e-6) ? (dx / d) : 1.0;
    const double hy0 = (raw_d > 1e-6) ? (dy / d) : 0.0;
    const double spawn_r = st->p_radius[src_idx] + st->ship_radius;
    const double sx = st->p_x[src_idx] + hx0 * spawn_r;
    const double sy = st->p_y[src_idx] + hy0 * spawn_r;

    const int32_t f = st->n_fleets++;
    const int32_t u_start = st->n_units;
    st->fl_id[f] = fleet_id;
    st->fl_owner[f] = owner;
    st->fl_source_id[f] = st->p_id[src_idx];
    st->fl_target_idx[f] = dst_idx;
    st->fl_spu[f] = ships_to_send / (double)parts;
    st->fl_unit_start[f] = u_start;
    st->fl_unit_count[f] = parts;
    st->fl_ships[f] = ships_to_send;

    if (parts == 1) {
        st->ux[u_start] = sx;
        st->uy[u_start] = sy;
        st->uhx[u_start] = hx0;
        st->uhy[u_start] = hy0;
        st->fl_cx[f] = sx;
        st->fl_cy[f] = sy;
        st->fl_radius[f] = st->ship_radius;
    } else {
        double* const fux = st->ux + u_start;
        double* const fuy = st->uy + u_start;
        double* const fuhx = st->uhx + u_start;
        double* const fuhy = st->uhy + u_start;
        const double two_pi = 2.0 * PI;
        const uint32_t seed_base = static_cast<uint32_t>(fleet_id) * 1000003u;
        const double spawn_disc_r = SPAWN_AREA_RADIUS * std::sqrt(static_cast<double>(parts));
        const double tx = st->p_x[dst_idx];
        const double ty = st->p_y[dst_idx];
        double sum_x = 0.0;
        double sum_y = 0.0;
        for (int32_t j = 0; j < parts; ++j) {
            const uint32_t uj = static_cast<uint32_t>(j);
            const double u1 = spawn_hash_u01(seed_base + uj * 2u + 1u);
            const double u2 = spawn_hash_u01(seed_base + uj * 2u + 2u);
            const double r = spawn_disc_r * std::sqrt(u1);
            const double ang = two_pi * u2;
            const double xj = sx + r * std::cos(ang);
            const double yj = sy + r * std::sin(ang);
            fux[j] = xj;
            fuy[j] = yj;
            sum_x += xj;
            sum_y += yj;
            const double hdx = tx - xj;
            const double hdy = ty - yj;
            const double hd2 = hdx * hdx + hdy * hdy;
            if (hd2 > EPS_DIST2) {
                const double hd = std::sqrt(hd2);
                fuhx[j] = hdx / hd;
                fuhy[j] = hdy / hd;
            } else {
                fuhx[j] = hx0;
                fuhy[j] = hy0;
            }
        }
        const double cx = sum_x / (double)parts;
        const double cy = sum_y / (double)parts;
        double max_d2 = 0.0;
        for (int32_t j = 0; j < parts; ++j) {
            const double ddx = fux[j] - cx;
            const double ddy = fuy[j] - cy;
            const double dd2 = ddx * ddx + ddy * ddy;
            if (dd2 > max_d2) max_d2 = dd2;
        }
        st->fl_cx[f] = cx;
        st->fl_cy[f] = cy;
        st->fl_radius[f] = std::sqrt(max_d2) + st->ship_radius;
    }

    st->n_units += parts;
    if (out_cx) *out_cx = st->fl_cx[f];
    if (out_cy) *out_cy = st->fl_cy[f];
    if (out_radius) *out_radius = st->fl_radius[f];
    if (out_num_units) *out_num_units = parts;
    return 0;
}

int32_t sim_set_fleet_target_by_id(SimState* st, int32_t fleet_id, int32_t dst_idx) {
    if (!st || dst_idx < 0 || dst_idx >= st->n_planets) return 0;
    for (int32_t f = 0; f < st->n_fleets; ++f) {
        if (st->fl_id[f] == fleet_id) {
            st->fl_target_idx[f] = dst_idx;
            return 1;
        }
    }
    return 0;
}

int32_t sim_step_physics_and_read(
    SimState* st,
    int32_t* out_p_owner,
    double* out_p_ships,
    int32_t* out_fl_id,
    int32_t* out_fl_cnt,
    double* out_fl_ships,
    double* out_fl_cx,
    double* out_fl_cy,
    double* out_fl_rad
) {
    if (!st || st->n_planets <= 0) return 0;

    for (int32_t f = 0; f < st->n_fleets; ++f) {
        const int32_t c = st->fl_unit_count[f];
        st->fl_spu[f] = st->fl_ships[f] / (double)(c > 1 ? c : 1);
    }

    PhysicsScratch& s = get_physics_scratch();
    st->n_units = run_physics_substeps(
        st->n_planets,
        st->p_owner, st->p_ships, st->p_x, st->p_y, st->p_prod, st->p_radius,
        st->n_fleets,
        st->fl_owner, st->fl_target_idx, st->fl_spu, st->fl_unit_start, st->fl_unit_count,
        st->fl_ships, st->fl_cx, st->fl_cy, st->fl_radius,
        st->n_units, st->ux, st->uy, st->uhx, st->uhy,
        st->tick_dt, st->fleet_speed, st->substeps, st->ship_radius,
        st->sep_stiffness, st->max_sep_push, st->sep_mode, st->prod_divisor, st->obstacle_escape_eps,
        st->turn_rate, st->turn_min_speed_factor,
        s.u_alive, s.fl_alive_cnt, s.tmp_x, s.tmp_y,
        s.arr_buckets, s.dirty_planets, s.near_planets,
        s.p_min_r, s.p_min_r2, s.p_land_r, s.p_land_r2_guard, s.p_prod_sub,
        s.tgt_head, s.grp_head, s.grp_tail, s.grp_cnt, s.grp_owner, s.grp_next, s.fl_next, s.grp_uids, s.grp_x, s.grp_y
    );

    int32_t wf = 0;
    for (int32_t f = 0; f < st->n_fleets; ++f) {
        if (st->fl_unit_count[f] > 0) {
            if (wf != f) {
                st->fl_id[wf] = st->fl_id[f];
                st->fl_owner[wf] = st->fl_owner[f];
                st->fl_source_id[wf] = st->fl_source_id[f];
                st->fl_target_idx[wf] = st->fl_target_idx[f];
                st->fl_spu[wf] = st->fl_spu[f];
                st->fl_unit_start[wf] = st->fl_unit_start[f];
                st->fl_unit_count[wf] = st->fl_unit_count[f];
                st->fl_ships[wf] = st->fl_ships[f];
                st->fl_cx[wf] = st->fl_cx[f];
                st->fl_cy[wf] = st->fl_cy[f];
                st->fl_radius[wf] = st->fl_radius[f];
            }
            ++wf;
        }
    }
    st->n_fleets = wf;

    const int32_t np = st->n_planets;
    std::memcpy(out_p_owner, st->p_owner, (size_t)np * sizeof(int32_t));
    std::memcpy(out_p_ships, st->p_ships, (size_t)np * sizeof(double));
    if (wf > 0) {
        std::memcpy(out_fl_id, st->fl_id, (size_t)wf * sizeof(int32_t));
        std::memcpy(out_fl_cnt, st->fl_unit_count, (size_t)wf * sizeof(int32_t));
        std::memcpy(out_fl_ships, st->fl_ships, (size_t)wf * sizeof(double));
        std::memcpy(out_fl_cx, st->fl_cx, (size_t)wf * sizeof(double));
        std::memcpy(out_fl_cy, st->fl_cy, (size_t)wf * sizeof(double));
        std::memcpy(out_fl_rad, st->fl_radius, (size_t)wf * sizeof(double));
    }
    return wf;
}

int32_t sim_read_fleet_units_by_id(
    const SimState* st,
    int32_t fleet_id,
    double* out_ux,
    double* out_uy,
    double* out_uhx,
    double* out_uhy,
    int32_t max_units
) {
    if (!st) return -1;
    for (int32_t f = 0; f < st->n_fleets; ++f) {
        if (st->fl_id[f] == fleet_id) {
            const int32_t cnt = st->fl_unit_count[f];
            if (cnt > max_units) return -1;
            const int32_t start = st->fl_unit_start[f];
            if (cnt > 0) {
                std::memcpy(out_ux, st->ux + start, (size_t)cnt * sizeof(double));
                std::memcpy(out_uy, st->uy + start, (size_t)cnt * sizeof(double));
                std::memcpy(out_uhx, st->uhx + start, (size_t)cnt * sizeof(double));
                std::memcpy(out_uhy, st->uhy + start, (size_t)cnt * sizeof(double));
            }
            return cnt;
        }
    }
    return -1;
}

static inline float clamp01_f32(double v) {
    if (v <= 0.0) return 0.0f;
    if (v >= 1.0) return 1.0f;
    return static_cast<float>(v);
}

int32_t sim_encode_obs_and_mask(
    const SimState* st,
    int32_t cur_player,
    int32_t grid_width,
    int32_t grid_height,
    double grid_max_x,
    double grid_max_y,
    double grid_square_size,
    int32_t fleet_top_k,
    double max_expected_ships,
    double max_expected_production,
    double max_fleet_radius,
    double max_expected_eta,
    double send_ratio,
    double min_send_ships,
    int32_t enable_fleet_redirection,
    int32_t n_buckets,
    const double* bucket_lowers,
    const double* bucket_uppers,
    float* out_obs,
    int8_t* out_mask
) {
    constexpr int32_t MAX_GRID_CELLS = 1024;
    constexpr int32_t MAX_BUCKETS = 16;
    if (!st) return -1;
    const int32_t grid_cells = grid_width * grid_height;
    if (grid_cells <= 0 || grid_cells > MAX_GRID_CELLS) return -1;
    if (n_buckets < 0 || n_buckets > MAX_BUCKETS) return -1;

    const int32_t np = st->n_planets;
    const int32_t nf = st->n_fleets;
    const int32_t pair_action_count = grid_cells * grid_cells;
    const int32_t total_actions = enable_fleet_redirection ? (2 * pair_action_count + 1) : (pair_action_count + 1);

    const int32_t fleet_slot_per_side = fleet_top_k + 1;
    const int32_t fleet_ch_cnt = 2 * fleet_slot_per_side * 5;
    const int32_t total_ch = 8 + fleet_ch_cnt + n_buckets * 4;

    if (out_obs) {
        std::memset(out_obs, 0, static_cast<size_t>(total_ch) * static_cast<size_t>(grid_cells) * sizeof(float));
    }
    if (out_mask) {
        std::memset(out_mask, 0, static_cast<size_t>(total_actions));
        out_mask[0] = 1;
    }

    const int32_t w_m1 = grid_width - 1;
    const int32_t h_m1 = grid_height - 1;
    const double span_x = 2.0 * grid_max_x;
    const double span_y = 2.0 * grid_max_y;
    const double speed = (st->fleet_speed > 1e-6) ? st->fleet_speed : 1e-6;
    const double max_rad = (max_fleet_radius > 1e-6) ? max_fleet_radius : 1e-6;

    // Per-planet grid coords & per-cell target/friendly lookup
    int32_t p_gx[SIM_MAX_PLANETS];
    int32_t p_gy[SIM_MAX_PLANETS];
    int32_t cell_tgt_id[MAX_GRID_CELLS];
    int32_t cell_fr_id[MAX_GRID_CELLS];
    uint8_t cell_can_send[MAX_GRID_CELLS];
    int32_t tgt_cells[SIM_MAX_PLANETS];
    int32_t src_cells[SIM_MAX_PLANETS];
    int32_t n_tgt = 0;
    int32_t n_src = 0;

    if (out_mask) {
        for (int32_t c = 0; c < grid_cells; ++c) {
            cell_tgt_id[c] = -1;
            cell_fr_id[c] = -1;
            cell_can_send[c] = 0;
        }
    }

    for (int32_t i = 0; i < np; ++i) {
        const double px = st->p_x[i];
        const double py = st->p_y[i];
        int32_t gx = static_cast<int32_t>(std::floor((px + grid_max_x) / grid_square_size));
        if (gx < 0) gx = 0;
        else if (gx > w_m1) gx = w_m1;
        int32_t gy = static_cast<int32_t>(std::floor((py + grid_max_y) / grid_square_size));
        if (gy < 0) gy = 0;
        else if (gy > h_m1) gy = h_m1;
        p_gx[i] = gx;
        p_gy[i] = gy;
        const int32_t cell = gy * grid_width + gx;

        if (out_obs) {
            const float lx = clamp01_f32((px - (-grid_max_x + static_cast<double>(gx) * grid_square_size)) / grid_square_size);
            const float ly = clamp01_f32((py - (-grid_max_y + static_cast<double>(gy) * grid_square_size)) / grid_square_size);
            const float sv = clamp01_f32(st->p_ships[i] / max_expected_ships);
            const float pv = clamp01_f32(st->p_prod[i] / max_expected_production);
            out_obs[0 * grid_cells + cell] = lx;
            out_obs[1 * grid_cells + cell] = ly;
            const int32_t own = st->p_owner[i];
            if (own == cur_player) {
                out_obs[2 * grid_cells + cell] = sv;
                out_obs[5 * grid_cells + cell] = pv;
            } else if (own == 0) {
                out_obs[4 * grid_cells + cell] = sv;
                out_obs[7 * grid_cells + cell] = pv;
            } else {
                out_obs[3 * grid_cells + cell] = sv;
                out_obs[6 * grid_cells + cell] = pv;
            }
        }

        if (out_mask) {
            const int32_t pid = st->p_id[i];
            if (cell_tgt_id[cell] < 0) {
                cell_tgt_id[cell] = pid;
                tgt_cells[n_tgt++] = cell;
            }
            if (st->p_owner[i] == cur_player) {
                if (cell_fr_id[cell] < 0) {
                    cell_fr_id[cell] = pid;
                }
                const double ships = st->p_ships[i];
                if (ships > 0.0) {
                    const double scaled = ships * send_ratio;
                    const double to_send = std::floor(ships < scaled ? ships : scaled);
                    if (to_send >= min_send_ships && to_send > 0.0) {
                        if (!cell_can_send[cell]) {
                            cell_can_send[cell] = 1;
                            src_cells[n_src++] = cell;
                        }
                    }
                }
            }
        }
    }

    if (out_mask) {
        for (int32_t si = 0; si < n_src; ++si) {
            const int32_t sc = src_cells[si];
            const int32_t src_id = cell_fr_id[sc];
            const int32_t row_base = sc * grid_cells + 1;
            for (int32_t ti = 0; ti < n_tgt; ++ti) {
                const int32_t tc = tgt_cells[ti];
                if (src_id != cell_tgt_id[tc]) {
                    out_mask[row_base + tc] = 1;
                }
            }
        }
    }

    if (nf <= 0) {
        return 0;
    }

    // Single pass over fleets:
    //  - compute ETA once
    //  - bucket into friendly/enemy per-cell linked lists (preserving insertion order)
    //  - accumulate redirect mask per-cell target status
    //  - accumulate landing schedule per target planet & bucket
    int32_t fr_head[MAX_GRID_CELLS];
    int32_t fr_tail[MAX_GRID_CELLS];
    int32_t en_head[MAX_GRID_CELLS];
    int32_t en_tail[MAX_GRID_CELLS];
    int32_t fr_occ_cells[SIM_MAX_FLEETS];
    int32_t en_occ_cells[SIM_MAX_FLEETS];
    int32_t n_fr_occ = 0;
    int32_t n_en_occ = 0;
    int32_t fl_next[SIM_MAX_FLEETS];
    double fl_eta[SIM_MAX_FLEETS];

    int32_t redir_cell_tgt[MAX_GRID_CELLS];
    int32_t redir_cells[SIM_MAX_FLEETS];
    int32_t n_redir_cells = 0;

    float land_fr[SIM_MAX_PLANETS * MAX_BUCKETS];
    float land_en[SIM_MAX_PLANETS * MAX_BUCKETS];
    uint8_t tgt_has_land[SIM_MAX_PLANETS];

    if (out_obs) {
        for (int32_t c = 0; c < grid_cells; ++c) {
            fr_head[c] = -1;
            en_head[c] = -1;
        }
        std::memset(tgt_has_land, 0, static_cast<size_t>(np));
    }
    if (out_mask && enable_fleet_redirection) {
        for (int32_t c = 0; c < grid_cells; ++c) {
            redir_cell_tgt[c] = -2; // sentinel: no friendly fleet in cell
        }
    }

    const bool default_3_buckets = (
        n_buckets == 3
        && bucket_lowers[0] == 0.0 && bucket_uppers[0] == 1.0
        && bucket_lowers[1] == 1.0 && bucket_uppers[1] == 2.0
        && bucket_lowers[2] == 2.0 && std::isinf(bucket_uppers[2])
    );

    for (int32_t f = 0; f < nf; ++f) {
        int32_t gx = static_cast<int32_t>(std::floor((st->fl_cx[f] + grid_max_x) / grid_square_size));
        if (gx < 0) gx = 0;
        else if (gx > w_m1) gx = w_m1;
        int32_t gy = static_cast<int32_t>(std::floor((st->fl_cy[f] + grid_max_y) / grid_square_size));
        if (gy < 0) gy = 0;
        else if (gy > h_m1) gy = h_m1;
        const int32_t cell = gy * grid_width + gx;
        const int32_t tidx = st->fl_target_idx[f];
        const bool is_fr = (st->fl_owner[f] == cur_player);

        if (out_mask && enable_fleet_redirection && is_fr && tidx >= 0 && tidx < np) {
            const int32_t tid = st->p_id[tidx];
            const int32_t prev = redir_cell_tgt[cell];
            if (prev == -2) {
                redir_cell_tgt[cell] = tid;
                redir_cells[n_redir_cells++] = cell;
            } else if (prev != -1 && prev != tid) {
                redir_cell_tgt[cell] = -1;
            }
        }

        if (out_obs) {
            double eta = 0.0;
            if (tidx >= 0 && tidx < np) {
                const double dist_edge = std::hypot(st->fl_cx[f] - st->p_x[tidx], st->fl_cy[f] - st->p_y[tidx]) - st->p_radius[tidx];
                if (dist_edge > 0.0) {
                    eta = dist_edge / speed;
                }
            }
            fl_eta[f] = eta;
            fl_next[f] = -1;

            if (is_fr) {
                if (fr_head[cell] < 0) {
                    fr_head[cell] = f;
                    fr_tail[cell] = f;
                    fr_occ_cells[n_fr_occ++] = cell;
                } else {
                    fl_next[fr_tail[cell]] = f;
                    fr_tail[cell] = f;
                }
            } else {
                if (en_head[cell] < 0) {
                    en_head[cell] = f;
                    en_tail[cell] = f;
                    en_occ_cells[n_en_occ++] = cell;
                } else {
                    fl_next[en_tail[cell]] = f;
                    en_tail[cell] = f;
                }
            }

            if (tidx >= 0 && tidx < np && n_buckets > 0) {
                if (!tgt_has_land[tidx]) {
                    tgt_has_land[tidx] = 1;
                    const int32_t base = tidx * n_buckets;
                    for (int32_t b = 0; b < n_buckets; ++b) {
                        land_fr[base + b] = 0.0f;
                        land_en[base + b] = 0.0f;
                    }
                }
                int32_t b_idx = n_buckets - 1;
                if (default_3_buckets) {
                    b_idx = (eta < 1.0) ? 0 : ((eta < 2.0) ? 1 : 2);
                } else {
                    for (int32_t b = 0; b < n_buckets; ++b) {
                        if (eta >= bucket_lowers[b] && eta < bucket_uppers[b]) {
                            b_idx = b;
                            break;
                        }
                    }
                }
                const int32_t idx_lb = tidx * n_buckets + b_idx;
                if (is_fr) {
                    land_fr[idx_lb] += static_cast<float>(st->fl_ships[f]);
                } else {
                    land_en[idx_lb] += static_cast<float>(st->fl_ships[f]);
                }
            }
        }
    }

    if (out_mask && enable_fleet_redirection && n_redir_cells > 0) {
        const int32_t redir_offset = pair_action_count + 1;
        for (int32_t ri = 0; ri < n_redir_cells; ++ri) {
            const int32_t sc = redir_cells[ri];
            const int32_t only_tgt = redir_cell_tgt[sc];
            const int32_t row_base = sc * grid_cells + redir_offset;
            if (only_tgt == -1) {
                for (int32_t ti = 0; ti < n_tgt; ++ti) {
                    out_mask[row_base + tgt_cells[ti]] = 1;
                }
            } else {
                for (int32_t ti = 0; ti < n_tgt; ++ti) {
                    const int32_t tc = tgt_cells[ti];
                    if (cell_tgt_id[tc] != only_tgt) {
                        out_mask[row_base + tc] = 1;
                    }
                }
            }
        }
    }

    if (!out_obs) {
        return 0;
    }

    auto encode_side_cells = [&](const int32_t* occ_cells, int32_t n_occ, const int32_t* heads, int32_t base_ch) {
        int32_t tmp_fl[SIM_MAX_FLEETS];
        for (int32_t ci = 0; ci < n_occ; ++ci) {
            const int32_t cell = occ_cells[ci];
            int32_t cnt = 0;
            for (int32_t f = heads[cell]; f >= 0; f = fl_next[f]) {
                tmp_fl[cnt++] = f;
            }
            // Stable descending insertion sort by fl_ships (preserves insertion order on ties)
            for (int32_t a = 1; a < cnt; ++a) {
                const int32_t key_f = tmp_fl[a];
                const double key_s = st->fl_ships[key_f];
                int32_t b = a - 1;
                while (b >= 0 && st->fl_ships[tmp_fl[b]] < key_s) {
                    tmp_fl[b + 1] = tmp_fl[b];
                    --b;
                }
                tmp_fl[b + 1] = key_f;
            }
            const int32_t n_top = (cnt > fleet_top_k) ? fleet_top_k : cnt;
            for (int32_t s = 0; s < n_top; ++s) {
                const int32_t f = tmp_fl[s];
                const int32_t tidx = st->fl_target_idx[f];
                const double tx = (tidx >= 0 && tidx < np) ? st->p_x[tidx] : 0.0;
                const double ty = (tidx >= 0 && tidx < np) ? st->p_y[tidx] : 0.0;
                const int32_t ch = base_ch + s * 5;
                out_obs[(ch + 0) * grid_cells + cell] = clamp01_f32((tx + grid_max_x) / span_x);
                out_obs[(ch + 1) * grid_cells + cell] = clamp01_f32((ty + grid_max_y) / span_y);
                out_obs[(ch + 2) * grid_cells + cell] = clamp01_f32(st->fl_radius[f] / max_rad);
                out_obs[(ch + 3) * grid_cells + cell] = clamp01_f32(st->fl_ships[f] / max_expected_ships);
                out_obs[(ch + 4) * grid_cells + cell] = clamp01_f32(fl_eta[f] / max_expected_eta);
            }
            if (cnt > fleet_top_k) {
                double tot_s = 0.0;
                for (int32_t s = fleet_top_k; s < cnt; ++s) {
                    tot_s += st->fl_ships[tmp_fl[s]];
                }
                if (tot_s > 0.0) {
                    double w_tx = 0.0, w_ty = 0.0, w_rad = 0.0, w_eta = 0.0;
                    for (int32_t s = fleet_top_k; s < cnt; ++s) {
                        const int32_t f = tmp_fl[s];
                        const int32_t tidx = st->fl_target_idx[f];
                        const double tx = (tidx >= 0 && tidx < np) ? st->p_x[tidx] : 0.0;
                        const double ty = (tidx >= 0 && tidx < np) ? st->p_y[tidx] : 0.0;
                        const double w = st->fl_ships[f] / tot_s;
                        w_tx += tx * w;
                        w_ty += ty * w;
                        w_rad += st->fl_radius[f] * w;
                        w_eta += fl_eta[f] * w;
                    }
                    const int32_t ch = base_ch + fleet_top_k * 5;
                    out_obs[(ch + 0) * grid_cells + cell] = clamp01_f32((w_tx + grid_max_x) / span_x);
                    out_obs[(ch + 1) * grid_cells + cell] = clamp01_f32((w_ty + grid_max_y) / span_y);
                    out_obs[(ch + 2) * grid_cells + cell] = clamp01_f32(w_rad / max_rad);
                    out_obs[(ch + 3) * grid_cells + cell] = clamp01_f32(tot_s / max_expected_ships);
                    out_obs[(ch + 4) * grid_cells + cell] = clamp01_f32(w_eta / max_expected_eta);
                }
            }
        }
    };

    encode_side_cells(fr_occ_cells, n_fr_occ, fr_head, 8);
    encode_side_cells(en_occ_cells, n_en_occ, en_head, 8 + fleet_slot_per_side * 5);

    const int32_t land_base_ch = 8 + fleet_ch_cnt;
    for (int32_t i = 0; i < np; ++i) {
        if (!tgt_has_land[i]) continue;
        const int32_t cell = p_gy[i] * grid_width + p_gx[i];
        const int32_t base = i * n_buckets;
        for (int32_t b = 0; b < n_buckets; ++b) {
            const double fr = static_cast<double>(land_fr[base + b]);
            const double en = static_cast<double>(land_en[base + b]);
            if (fr == 0.0 && en == 0.0) continue;
            const int32_t ch = land_base_ch + b * 4;
            if (fr > 0.0) out_obs[(ch + 0) * grid_cells + cell] = clamp01_f32(fr / max_expected_ships);
            if (en > 0.0) out_obs[(ch + 1) * grid_cells + cell] = clamp01_f32(en / max_expected_ships);
            const double diff = fr - en;
            if (diff > 0.0) {
                out_obs[(ch + 2) * grid_cells + cell] = clamp01_f32(diff / max_expected_ships);
            } else if (diff < 0.0) {
                out_obs[(ch + 3) * grid_cells + cell] = clamp01_f32((-diff) / max_expected_ships);
            }
        }
    }

    return 0;
}

} // extern "C"
