#include "position.h"
#include <cstdio>
#include <chrono>

static uint64_t perft(const Position& pos, int depth) {
    MoveList list;
    generate(pos, list, GEN_ALL);
    uint64_t n = 0;
    for (int i = 0; i < list.size; i++) {
        Move m = list.moves[i];
        if (!pos.is_legal(m)) continue;
        if (depth == 1) { n++; continue; }
        Position child = pos;
        child.do_move(m);
        n += perft(child, depth - 1);
    }
    return n;
}

// Also verifies the incrementally-updated zobrist key against a from-scratch
// rebuild at every node -- a mismatch here would silently poison the TT later.
static uint64_t key_errors = 0;
static void key_check(const Position& pos, int depth) {
    Position fresh;
    fresh.set_fen(pos.fen());
    if (fresh.key != pos.key) key_errors++;
    if (depth == 0) return;
    MoveList list;
    generate_legal(pos, list);
    for (int i = 0; i < list.size; i++) {
        Position child = pos;
        child.do_move(list.moves[i]);
        key_check(child, depth - 1);
    }
}

int main() {
    bb::init();
    struct Case { const char* fen; int depth; uint64_t expected; };
    const Case cases[] = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609ULL},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324ULL},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603ULL},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690ULL},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083ULL},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292ULL},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194ULL},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594ULL},
    };
    bool allOk = true;
    uint64_t totalNodes = 0;
    double totalSec = 0;
    for (auto& c : cases) {
        Position pos;
        pos.set_fen(c.fen);
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t n = perft(pos, c.depth);
        double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        bool ok = n == c.expected;
        allOk &= ok;
        totalNodes += n; totalSec += sec;
        printf("%-4s perft(%d) = %11llu expected %11llu  %6.2fs  %6.1f Mnps  %s\n", ok ? "OK" : "FAIL",
               c.depth, (unsigned long long)n, (unsigned long long)c.expected, sec, n / sec / 1e6, c.fen);
    }
    printf("total: %.1f Mnps\n", totalNodes / totalSec / 1e6);
    for (auto& c : cases) { Position pos; pos.set_fen(c.fen); key_check(pos, 3); }
    printf("zobrist incremental-vs-scratch mismatches over depth-3 trees: %llu\n", (unsigned long long)key_errors);
    allOk &= key_errors == 0;
    printf(allOk ? "ALL OK\n" : "FAILURES\n");
    return allOk ? 0 : 1;
}
