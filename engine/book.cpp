// Polyglot opening book: key computation, file loading and weighted-random probing.
#include "book.h"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "polyglot_random.inc"

namespace book {

struct Entry { uint64_t key; uint16_t move, weight; };
static std::vector<Entry> entries;

static uint64_t be(const unsigned char* p, int n) {
    uint64_t v = 0;
    for (int i = 0; i < n; i++) v = (v << 8) | p[i];
    return v;
}

bool open(const std::string& path) {
    entries.clear();
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    unsigned char buf[16];
    while (fread(buf, 1, 16, f) == 16)
        entries.push_back({be(buf, 8), (uint16_t)be(buf + 8, 2), (uint16_t)be(buf + 10, 2)});
    fclose(f);
    return !entries.empty();
}

bool loaded() { return !entries.empty(); }

uint64_t key(const Position& pos) {
    uint64_t k = 0;
    for (int s = 0; s < 64; s++) {
        int pc = pos.board[s];
        if (pc == NO_PIECE) continue;
        int kind = 2 * type_of(pc) + (color_of(pc) == WHITE ? 1 : 0);  // bp=0, wp=1, bn=2, ...
        k ^= PolyRandom[64 * kind + s];
    }
    for (int i = 0; i < 4; i++)
        if (pos.castling & (1 << i)) k ^= PolyRandom[768 + i];  // K Q k q, same bit order as ours
    if (pos.ep >= 0) k ^= PolyRandom[772 + (pos.ep & 7)];       // only set when a capture is possible
    if (pos.stm == WHITE) k ^= PolyRandom[780];
    return k;
}

// polyglot move -> legal move (castling is encoded as king takes own rook)
static Move decode(const Position& pos, uint16_t pm) {
    int to = pm & 63, from = (pm >> 6) & 63, pr = (pm >> 12) & 7;  // pr: 1 N, 2 B, 3 R, 4 Q
    static const int promoType[5] = {-1, KNIGHT, BISHOP, ROOK, QUEEN};
    MoveList legal;
    generate_legal(pos, legal);
    for (int i = 0; i < legal.size; i++) {
        Move m = legal.moves[i];
        int mto = to_sq(m);
        if (flags_of(m) == MF_KCASTLE) mto = to_sq(m) + 1;
        else if (flags_of(m) == MF_QCASTLE) mto = to_sq(m) - 2;
        if (from_sq(m) == from && mto == to && (is_promo(m) ? promo_type(m) == promoType[pr] : pr == 0)) return m;
    }
    return 0;
}

Move probe(const Position& pos, bool best) {
    uint64_t k = key(pos);
    auto lo = std::lower_bound(entries.begin(), entries.end(), k, [](const Entry& e, uint64_t v) { return e.key < v; });
    if (best) {  // highest weight; ties -> first in file
        auto top = entries.end();
        for (auto it = lo; it != entries.end() && it->key == k; ++it)
            if (it->weight && (top == entries.end() || it->weight > top->weight)) top = it;
        return top == entries.end() ? 0 : decode(pos, top->move);
    }
    uint32_t total = 0;
    for (auto it = lo; it != entries.end() && it->key == k; ++it) total += it->weight;
    if (!total) return 0;
    static std::mt19937 rng(std::random_device{}());
    uint32_t r = std::uniform_int_distribution<uint32_t>(0, total - 1)(rng);
    for (auto it = lo; it != entries.end() && it->key == k; ++it) {
        if (r < it->weight) return decode(pos, it->move);
        r -= it->weight;
    }
    return 0;
}

}  // namespace book
