// galcon_engine.hpp - functions transcribed directly from galcon2.exe.
//
// These are not approximations fitted to observations. Each one is a literal
// reading of the disassembly, with the originating address recorded. The
// binary is pei-i386 with an intact MinGW symbol table, so every function
// below was located by name rather than by search.
//
// Decoded so far:
//   gc_do_move_apart   0x442a00  (448 bytes)   COMPLETE
//   gc_do_collide      0x442bc0  (640 bytes)   COMPLETE
//   _fleet_init_units  0x4420e0  (592 bytes)   COMPLETE
//   unit_logic         0x4532d0  (2032 bytes)  NOT YET DECODED  <-- the blocker
//   collide_loop       0x43e690  (368 bytes)   NOT YET DECODED
//   collide_do         0x43db40  (864 bytes)   NOT YET DECODED
//
// FLOAT SEMANTICS WARNING
// The original is 32-bit x87 code: intermediates are evaluated at 80-bit
// extended precision and only narrowed when stored back to a float field.
// A modern x86-64 build uses SSE2 and will *not* reproduce this bit-exactly.
// If bit-exactness turns out to matter, either build 32-bit with
// -mfpmath=387, or promote the intermediates here to long double. Every
// value does get narrowed to float on store, which bounds the divergence,
// so start without it and let the differential test decide.

#pragma once

#include <cmath>
#include <cstdlib>

namespace galcon {

// Item field offsets recovered from the disassembly. sizeof(Item) == 640,
// which the engine prints at startup ("sizeof(galcon.Item) = 640 b") and
// which matches the 0x280 stride in _fleet_init_units' item loop.
struct Item {
    float x;       // +0x44
    float y;       // +0x48
    float r;       // +0x4c   collision radius
    float mass;    // +0x50
    float vx;      // +0x234
    float vy;      // +0x238
};

// The engine's degenerate-distance epsilon, at 0x615f08 (a double).
constexpr double kEpsilon = 0.001;

// The random unit offset used when two bodies are exactly coincident:
// (rand()/32767.0 * 200.0 - 100.0) / 100.0, i.e. uniform on [-1, 1].
// Constants at 0x615f10, 0x615f14, 0x615f18.
inline float random_axis() {
    return (float)((std::rand() / 32767.0f) * 200.0f - 100.0f) / 100.0f;
}

// ---------------------------------------------------------------------------
// gc_do_move_apart(Galcon&, Item* a, Item* b, int pin_a)   @ 0x442a00
//
// Separates two overlapping bodies to exactly touching, preserving their
// centre of mass. `pin_a` substitutes mass_a := 1 and mass_b := 0, which
// collapses the centre of mass onto `a` and leaves `a` untouched while `b`
// is pushed out to the contact distance. That is the immovable-planet case.
// ---------------------------------------------------------------------------
inline void do_move_apart(Item* a, Item* b, int pin_a) {
    float dx = b->x - a->x;
    float dy = b->y - a->y;
    float rsum = a->r + b->r;
    float dist = std::sqrt(dx * dx + dy * dy);

    if (dist > rsum) return;            // not overlapping: nothing to do

    if (dist < kEpsilon) {              // coincident: pick a random direction
        dx = random_axis();
        dy = random_axis();
        dist = 1.0f;
    }

    float mA, mB, M;
    if (pin_a) { mA = 1.0f; mB = 0.0f; M = 1.0f; }
    else       { mA = a->mass; mB = b->mass; M = mA + mB; }

    float cx = (a->x * mA + b->x * mB) / M;   // centre of mass
    float cy = (a->y * mA + b->y * mB) / M;

    a->x = cx - dx * rsum * mB / (dist * M);
    a->y = cy - dy * rsum * mB / (dist * M);
    b->x = cx + dx * rsum * mA / (dist * M);
    b->y = cy + dy * rsum * mA / (dist * M);
}

// ---------------------------------------------------------------------------
// gc_do_collide(Galcon&, Item* a, Item* b)                 @ 0x442bc0
//
// Standard 2D elastic impulse along the contact normal, followed by the body
// of gc_do_move_apart inlined with pin_a == 0. Note the impulse is applied
// unconditionally, before the overlap test that guards the separation.
// ---------------------------------------------------------------------------
inline void do_collide(Item* a, Item* b) {
    // --- impulse -----------------------------------------------------------
    float dx = a->x - b->x;                 // note: a - b here, b - a below
    float dy = a->y - b->y;
    float dist = std::sqrt(dx * dx + dy * dy);
    float nx = dx, ny = dy;
    if (dist != 0.0f) { nx = dx / dist; ny = dy / dist; }

    float va_n = a->vx * nx + a->vy * ny;
    float vb_n = b->vx * nx + b->vy * ny;
    float j = 2.0f * (va_n - vb_n) / (a->mass + b->mass);

    a->vx -= j * b->mass * nx;
    a->vy -= j * b->mass * ny;
    b->vx += j * a->mass * nx;
    b->vy += j * a->mass * ny;

    // --- separation (gc_do_move_apart inlined, pin_a == 0) ------------------
    do_move_apart(a, b, 0);
}

// ---------------------------------------------------------------------------
// _fleet_init_units(Galcon&, float ships) -> int            @ 0x4420e0
//
// How many units a fleet is split into. All three truncations are toward
// zero: the code sets the x87 rounding-control field to 11 (`mov ah,0xc`)
// before each fistp.
//
// `total_ships` is summed by a loop over the item array with a type and
// bounds filter, so it is the ships visible in play, not a running tally.
//
// Reproduces all 17 measured observations. The empirical sqrt-fit constant
// 4.1 was sqrt(17) = 4.1231.
// ---------------------------------------------------------------------------
constexpr int kArena = 4080;      // `mov eax,0xff0`
constexpr float kUnitK = 17.0f;   // float32 at 0x615ee8

inline int fleet_init_units(float ships, float total_ships, int items_in_use) {
    if (ships <= 0.0f) return 1;
    int budget = (kArena - items_in_use) / 2;
    if (budget < 0) budget = 0;

    int n = (int)std::sqrt(kUnitK * ships);      // trunc toward zero
    if ((int)ships < n) n = (int)ships;          // never <1 ship per unit
    if (total_ships > 0.0f) {
        int by_share = (int)(budget * ships / total_ships);
        if (by_share < n) n = by_share;
    }
    return n < 1 ? 1 : n;
}

}  // namespace galcon
