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
    bool ponder = false;     // "go ponder": search without time checks until ponderhit/stop
    bool fromClock = false;  // movetime was derived from wtime/btime (time management may override it)
    int64_t time = -1, inc = 0;
    int64_t oppTime = -1;    // opponent's clock (flag mode)
};

namespace search {
void init();
void set_hash_mb(size_t mb);
void clear_hash();
void set_threads(int n);
void set_multipv(int n);
int set_syzygy(const std::string& path);
bool set_small_net(const std::string& path);  // 128-wide LNN2/3/4 net for UseSmallNet  // returns the largest piece count available (0 = none)
int threads();
int multipv();
// `history` = zobrist keys of every position in the game so far, ending with root's.
void start(const Position& root, const std::vector<uint64_t>& history, const Limits& lim);
void stop();
void ponderhit();
void wait();
// Synchronous fixed-depth search used by `bench`; returns nodes searched.
bool cuckoo_check(const Position& root, const std::vector<uint64_t>& history);  // debug
uint64_t bench_one(const Position& root, int depth, bool quiet);
// Synchronous silent search limited to `nodes` (datagen): best move and its score (side to move's view).
void search_nodes(const Position& root, const std::vector<uint64_t>& history, uint64_t nodes, Move& best, int& score);
}

void out_line(const char* fmt, ...);
