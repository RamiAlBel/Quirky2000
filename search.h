#pragma once
#include "position.h"
#include <vector>
#include <cstdint>

constexpr int MAX_PLY = 128;
constexpr int VALUE_MATE = 32000, VALUE_INF = 32001, VALUE_NONE = 32002;
constexpr int MATE_BOUND = VALUE_MATE - MAX_PLY;

struct Limits {
    int depth = 0;
    int64_t movetime = 0;
    uint64_t nodes = 0;
    bool infinite = false;
};

namespace search {
void init();
void set_hash_mb(size_t mb);
void clear_hash();
void set_threads(int n);
void set_multipv(int n);
int threads();
int multipv();
// `history` = zobrist keys of every position in the game so far, ending with root's.
void start(const Position& root, const std::vector<uint64_t>& history, const Limits& lim);
void stop();
void wait();
// Synchronous fixed-depth search used by `bench`; returns nodes searched.
uint64_t bench_one(const Position& root, int depth, bool quiet);
}

void out_line(const char* fmt, ...);
