// LNN6 (Round H): net trained with bullet (exp2/bullet/bullet/examples/quirky), converted by exp2/bullet/lnn6.py.
// Included by nnue.cpp when NNUE_LNN6 is defined (main net only; nnue_small keeps the old code).
//
// Inputs per perspective (indices identical to examples/quirky/inputs.rs):
//   PSQ     16 king buckets x 768, board oriented to the perspective and mirrored so its king is on files a-d
//   threats (attacker P..Q, from, attacked piece, to) for every occupied square a piece attacks, 59808 rows
// FT int16 (1.0 = 255) -> clamp [0,255] -> pairwise product >> 9 (uint8, 1.0 ~ 127) -> L1 int8 (per-output scale)
// -> 32 per output bucket -> (y, y^2) clamped -> L2 32 -> 1, float. Output bucket (pieces - 2) / 4.
//
// Threats change non-locally, so update() diffs them: attackers on changed squares lose/gain all their threats,
// and every other attacker whose attack set (parent occupancy) touches a changed square is recomputed.
#include "nnue.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

namespace nnue {

static constexpr int NPSQ = 16 * 768, NTHR = 59808, NOB = 8, L2 = 32, L3 = 32, MAXF = 512;
static constexpr float CP_PER_UNIT = 400.0f;
// x_u8 * w_i8 -> real: x = a*b/512 with a,b at 255, w at S1[b][o] (per output, 127/max|w|)
static constexpr float X_DEQ = 512.0f / 65025.0f;
alignas(32) static float S1[NOB][L2], L1Deq[NOB][L2];

#ifdef THR16
typedef int16_t thr_t;  // threat rows widened at load: no sign-extend in the update loop (2x memory)
#else
typedef int8_t thr_t;
#endif
// PSQ + THR are 86 MB. Default: private 2MB-aligned buffers (THP on Linux). -DNNUE_MMAP maps a v2 file (64-byte
// aligned blocks) read-only so concurrent processes share it; measured no faster with 8 concurrent benches.
static const int16_t* PSQ = nullptr;  // [NPSQ][NNUE_ACC]
static const thr_t* THR = nullptr;    // [NTHR][NNUE_ACC]
static int16_t* PsqBuf = nullptr;     // fread path
static thr_t* ThrBuf = nullptr;
alignas(64) static int16_t FTB[NNUE_ACC];
alignas(64) static int8_t W1[NOB][NNUE_ACC / 4][L2][4];
alignas(32) static float B1[NOB][L2], W2[NOB][2 * L2][L3], B2[NOB][L3], W3[NOB][L3], B3[NOB];
static int Dual = 1, NetGen = 0;
static bool Threats = true;  // false: all threat rows are zero (net trained without threats), skip their bookkeeping
static bool Loaded = false;

// ---- feature indexing ----
static const int BucketLayout[32] = {
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  8,  9,  9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
};

struct View { int xf, bucket; };  // square transform sq ^ xf (orientation ^ mirror)
static inline View view_of(const Position& pos, int persp) {
    int k = pos.king_sq(persp) ^ (persp == BLACK ? 56 : 0), f = k & 7;
    return {(persp == BLACK ? 56 : 0) ^ (f > 3 ? 7 : 0), BucketLayout[(k >> 3) * 4 + (f > 3 ? 7 - f : f)]};
}
static inline int cache_slot(const View& v) { return v.bucket * 2 + (v.xf & 7 ? 1 : 0); }
static inline int psq_feat(const View& v, int persp, int pc, int sq) {
    return v.bucket * 768 + 384 * (color_of(pc) != persp) + 64 * type_of(pc) + (sq ^ v.xf);
}

struct ThrData { Bitboard att[64]; int idx[64], tgt[12], count; };
static ThrData TD[4];  // knight, bishop, rook, queen
static int PawnMap[12], Toff[5], Half;

static void make_targets(int* tgt, std::initializer_list<int> valid) {
    for (int i = 0; i < 12; i++) tgt[i] = -1;
    int n = (int)valid.size(), i = 0;
    for (int v : valid) { tgt[v] = i; tgt[v + 6] = i + n; i++; }
}
static void init_threats() {
    for (int k = 0; k < 4; k++) {
        ThrData& t = TD[k];
        t.count = 0;
        for (int s = 0; s < 64; s++) {
            t.att[s] = k == 0 ? bb::KnightAttacks[s] : k == 1 ? bb::bishop_attacks(s, 0)
                     : k == 2 ? bb::rook_attacks(s, 0) : bb::queen_attacks(s, 0);
            t.idx[s] = t.count;
            t.count += popcount(t.att[s]);
        }
    }
    make_targets(TD[0].tgt, {PAWN, KNIGHT, BISHOP, ROOK, QUEEN});
    make_targets(TD[1].tgt, {PAWN, KNIGHT, BISHOP, ROOK});
    make_targets(TD[2].tgt, {PAWN, KNIGHT, BISHOP, ROOK});
    make_targets(TD[3].tgt, {PAWN, KNIGHT, BISHOP, ROOK, QUEEN});
    make_targets(PawnMap, {KNIGHT, ROOK});
    const int ntgt[4] = {10, 8, 8, 10};
    Toff[0] = 4 * 84;
    for (int k = 0; k < 4; k++) Toff[k + 1] = Toff[k] + ntgt[k] * TD[k].count;
    Half = Toff[4];
}
// src/dest already transformed to the perspective; target = type + 6 if not the perspective's piece
static inline int map_single(int pt, int src, int dest, int target) {
    if (pt == PAWN) {
        if (PawnMap[target] < 0) return -1;
        int id = std::abs(dest - src) == (dest > src ? 7 : 9) ? 0 : 1;
        return PawnMap[target] * 84 + ((src >> 3) - 1) * 14 + 2 * (src & 7) + id - 1;
    }
    const ThrData& t = TD[pt - 1];
    if (t.tgt[target] < 0 || (dest > src && target % 6 == pt)) return -1;
    return Toff[pt - 1] + t.tgt[target] * t.count + t.idx[src] + popcount(t.att[src] & ((1ULL << dest) - 1));
}
// threat row (0..NTHR) or -1
static inline int thr_feat(const View& v, int persp, int pc, int src, int victim, int dest) {
    int i = map_single(type_of(pc), src ^ v.xf, dest ^ v.xf, type_of(victim) + 6 * (color_of(victim) != persp));
    return i < 0 ? -1 : i + Half * (color_of(pc) != persp);
}
static inline Bitboard piece_att(int pc, int s, Bitboard occ) {
    switch (type_of(pc)) {
        case PAWN: return bb::PawnAttacks[color_of(pc)][s];
        case KNIGHT: return bb::KnightAttacks[s];
        case BISHOP: return bb::bishop_attacks(s, occ);
        case ROOK: return bb::rook_attacks(s, occ);
        case QUEEN: return bb::queen_attacks(s, occ);
        default: return 0;
    }
}
static inline Bitboard non_kings(const Position& pos) {
    return pos.occupied & ~(pos.pieces[make_piece(WHITE, KING)] | pos.pieces[make_piece(BLACK, KING)]);
}

void features(const Position& pos, int persp, std::vector<int>& out) {
    out.clear();
    View v = view_of(pos, persp);
    for (Bitboard b = pos.occupied; b;) { int s = pop_lsb(b); out.push_back(psq_feat(v, persp, pos.board[s], s)); }
    for (Bitboard a = non_kings(pos); a;) {
        int s = pop_lsb(a), pc = pos.board[s];
        for (Bitboard t = piece_att(pc, s, pos.occupied) & pos.occupied; t;) {
            int d = pop_lsb(t), f = thr_feat(v, persp, pc, s, pos.board[d], d);
            if (f >= 0) out.push_back(NPSQ + f);
        }
    }
    std::sort(out.begin(), out.end());
}

// ---- accumulator arithmetic ----
static inline __m256i prow(int f, int j) { return _mm256_load_si256((const __m256i*)(PSQ + (size_t)f * NNUE_ACC + j)); }
static inline __m256i trow(int f, int j) {
#ifdef THR16
    return _mm256_load_si256((const __m256i*)(THR + (size_t)f * NNUE_ACC + j));
#else
    return _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(THR + (size_t)f * NNUE_ACC + j)));
#endif
}
#ifdef NNUE_PROF
#include <x86intrin.h>
static uint64_t PrN[6], PrC[6], PrRows;  // update (incl. refreshes), refresh, eval, apply
static struct PrPrint { ~PrPrint() {
    const char* n[6] = {"update", "refresh", "eval", "apply", "gen", "map"};
    fprintf(stderr, "prof rows/apply %.2f\n", (double)PrRows / PrN[3]);
    for (int i = 0; i < 6; i++) fprintf(stderr, "prof %-8s %10llu calls %8.1f cyc/call %6.2f Gcyc\n", n[i], (unsigned long long)PrN[i], PrN[i] ? (double)PrC[i] / PrN[i] : 0.0, PrC[i] / 1e9);
} } PrPrinter;
struct PrT { int i; uint64_t t0; PrT(int i) : i(i), t0(__rdtsc()) {} ~PrT() { PrN[i]++; PrC[i] += __rdtsc() - t0; } };
#define PROF(i) PrT prof_(i)
#else
#define PROF(i)
#endif
struct Delta {
    int ps[32], pa[32], ts[MAXF], ta[MAXF];
    int nps = 0, npa = 0, nts = 0, nta = 0;
};
#ifdef THR_STATS
uint64_t StatUpd = 0, StatRows = 0;
static struct StatPrint { ~StatPrint() { fprintf(stderr, "apply calls %llu, threat rows/call %.2f\n", (unsigned long long)StatUpd, (double)StatRows / StatUpd); } } StatPrinter;
#endif
// dst = src - ps + pa - ts + ta (rows), and - xs + xa (whole vectors) if given
static void apply(const int16_t* src, int16_t* dst, const Delta& d, const int16_t* xs = nullptr, const int16_t* xa = nullptr) {
    PROF(3);
#ifdef NNUE_PROF
    PrRows += d.nps + d.npa + d.nts + d.nta;
#endif
#ifdef NO_APPLY
    if (src != dst) memcpy(dst, src, NNUE_ACC * 2);
    return;
#endif
#ifdef THR_STATS
    StatUpd++, StatRows += d.nts + d.nta;
#endif
#ifdef THR_PREFETCH
    for (int i = 0; i < d.nts; i++)
        for (int j = 0; j < NNUE_ACC; j += 64) _mm_prefetch((const char*)(THR + (size_t)d.ts[i] * NNUE_ACC + j), _MM_HINT_T0);
    for (int i = 0; i < d.nta; i++)
        for (int j = 0; j < NNUE_ACC; j += 64) _mm_prefetch((const char*)(THR + (size_t)d.ta[i] * NNUE_ACC + j), _MM_HINT_T0);
#endif
    for (int j = 0; j < NNUE_ACC; j += 128) {
        __m256i r[8];
        for (int k = 0; k < 8; k++) r[k] = _mm256_load_si256((const __m256i*)(src + j) + k);
        if (xs)
            for (int k = 0; k < 8; k++)
                r[k] = _mm256_add_epi16(_mm256_sub_epi16(r[k], _mm256_load_si256((const __m256i*)(xs + j) + k)),
                                        _mm256_load_si256((const __m256i*)(xa + j) + k));
        for (int i = 0; i < d.nps; i++)
            for (int k = 0; k < 8; k++) r[k] = _mm256_sub_epi16(r[k], prow(d.ps[i], j + 16 * k));
        for (int i = 0; i < d.npa; i++)
            for (int k = 0; k < 8; k++) r[k] = _mm256_add_epi16(r[k], prow(d.pa[i], j + 16 * k));
        for (int i = 0; i < d.nts; i++)
            for (int k = 0; k < 8; k++) r[k] = _mm256_sub_epi16(r[k], trow(d.ts[i], j + 16 * k));
        for (int i = 0; i < d.nta; i++)
            for (int k = 0; k < 8; k++) r[k] = _mm256_add_epi16(r[k], trow(d.ta[i], j + 16 * k));
        for (int k = 0; k < 8; k++) _mm256_store_si256((__m256i*)(dst + j) + k, r[k]);
    }
}

// per-thread refresh cache of the PSQ part, one slot per (king bucket, mirror)
struct alignas(64) FinnyEntry { int16_t acc[NNUE_ACC]; Bitboard pcs[12]; };
static thread_local FinnyEntry* Finny = nullptr;  // [2][32]
static thread_local int FinnyGen = -1;

// bring the cache entry of view v up to date with pos's pieces; returns its PSQ-only accumulator
static const int16_t* finny_sync(const Position& pos, int persp, const View& v) {
    if (!Finny) Finny = (FinnyEntry*)aligned_alloc64(sizeof(FinnyEntry) * 64);
    if (FinnyGen != NetGen) {
        for (int i = 0; i < 64; i++) {
            memcpy(Finny[i].acc, FTB, sizeof(FTB));
            memset(Finny[i].pcs, 0, sizeof(Finny[i].pcs));
        }
        FinnyGen = NetGen;
    }
    FinnyEntry& e = Finny[persp * 32 + cache_slot(v)];
    Delta d;
    for (int pc = 0; pc < 12; pc++) {
        for (Bitboard b = e.pcs[pc] & ~pos.pieces[pc]; b;) d.ps[d.nps++] = psq_feat(v, persp, pc, pop_lsb(b));
        for (Bitboard b = pos.pieces[pc] & ~e.pcs[pc]; b;) d.pa[d.npa++] = psq_feat(v, persp, pc, pop_lsb(b));
        e.pcs[pc] = pos.pieces[pc];
    }
    if (d.nps + d.npa) apply(e.acc, e.acc, d);
    return e.acc;
}

void refresh(const Position& pos, Accumulator& acc, int persp) {
    PROF(1);
    View v = view_of(pos, persp);
    const int16_t* base = finny_sync(pos, persp, v);
    static thread_local Delta d;
    d.nps = d.npa = d.nts = d.nta = 0;
    for (Bitboard a = Threats ? non_kings(pos) : 0; a;) {
        int s = pop_lsb(a), pc = pos.board[s];
        for (Bitboard t = piece_att(pc, s, pos.occupied) & pos.occupied; t;) {
            int dq = pop_lsb(t), f = thr_feat(v, persp, pc, s, pos.board[dq], dq);
            if (f >= 0 && d.nta < MAXF) d.ta[d.nta++] = f;
        }
    }
    apply(base, acc.v[persp], d);
    memset(acc.psqt[persp], 0, sizeof(acc.psqt[persp]));
}

void refresh_all(const Position& pos, Accumulator& acc) {
    refresh(pos, acc, WHITE);
    refresh(pos, acc, BLACK);
}

void update(const Position& parent, Move, const Position& child, const Accumulator& in, Accumulator& out) {
    PROF(0);
    // squares whose content changed
    Bitboard chg = 0;
    for (int h = 0; h < 2; h++) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(parent.board + 32 * h));
        __m256i b = _mm256_loadu_si256((const __m256i*)(child.board + 32 * h));
        chg |= (Bitboard)(uint32_t)~_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b)) << (32 * h);
    }
    // threat deltas, absolute: (attacker, from, victim, to)
    struct T { uint8_t pc, s, v, d; };
    T sub[MAXF], add[MAXF];
    int ns = 0, na = 0;
#ifdef NNUE_PROF
    { PrT prof_g(4);
#endif
    if (Threats) {
        const Bitboard occP = parent.occupied, occC = child.occupied;
        for (Bitboard c = chg; c;) {
            int s = pop_lsb(c), pp = parent.board[s], cp = child.board[s];
            if (pp != NO_PIECE && type_of(pp) != KING)
                for (Bitboard t = piece_att(pp, s, occP) & occP; t && ns < MAXF;) { int d = pop_lsb(t); sub[ns++] = {(uint8_t)pp, (uint8_t)s, parent.board[d], (uint8_t)d}; }
            if (cp != NO_PIECE && type_of(cp) != KING)
                for (Bitboard t = piece_att(cp, s, occC) & occC; t && na < MAXF;) { int d = pop_lsb(t); add[na++] = {(uint8_t)cp, (uint8_t)s, child.board[d], (uint8_t)d}; }
        }
        // unchanged attackers that reach a changed square (as slider ray or target) with the parent occupancy
        const Bitboard* P = parent.pieces;
        const Bitboard diag = P[make_piece(WHITE, BISHOP)] | P[make_piece(BLACK, BISHOP)] | P[make_piece(WHITE, QUEEN)] | P[make_piece(BLACK, QUEEN)];
        const Bitboard orth = P[make_piece(WHITE, ROOK)] | P[make_piece(BLACK, ROOK)] | P[make_piece(WHITE, QUEEN)] | P[make_piece(BLACK, QUEEN)];
        const Bitboard knights = P[make_piece(WHITE, KNIGHT)] | P[make_piece(BLACK, KNIGHT)];
        Bitboard S = 0;
        for (Bitboard c = chg; c;) {
            int x = pop_lsb(c);
            S |= (bb::PawnAttacks[BLACK][x] & P[make_piece(WHITE, PAWN)]) | (bb::PawnAttacks[WHITE][x] & P[make_piece(BLACK, PAWN)])
               | (bb::KnightAttacks[x] & knights) | (bb::bishop_attacks(x, occP) & diag) | (bb::rook_attacks(x, occP) & orth);
        }
        for (S &= ~chg; S;) {
            int s = pop_lsb(S), pc = parent.board[s];
            Bitboard aP = piece_att(pc, s, occP) & occP, aC = piece_att(pc, s, occC) & occC;
            for (Bitboard t = aP & ~aC; t && ns < MAXF;) { int d = pop_lsb(t); sub[ns++] = {(uint8_t)pc, (uint8_t)s, parent.board[d], (uint8_t)d}; }
            for (Bitboard t = aC & ~aP; t && na < MAXF;) { int d = pop_lsb(t); add[na++] = {(uint8_t)pc, (uint8_t)s, child.board[d], (uint8_t)d}; }
            for (Bitboard t = aP & aC & chg; t && ns < MAXF && na < MAXF;) {
                int d = pop_lsb(t);
                sub[ns++] = {(uint8_t)pc, (uint8_t)s, parent.board[d], (uint8_t)d};
                add[na++] = {(uint8_t)pc, (uint8_t)s, child.board[d], (uint8_t)d};
            }
        }
    }

#ifdef NNUE_PROF
    }
#endif
    for (int persp = 0; persp < 2; persp++) {
        View v = view_of(child, persp), pv = view_of(parent, persp);
        if (v.xf != pv.xf) { refresh(child, out, persp); continue; }
        // king bucket changed, same mirror: threat indices don't depend on the bucket, so swap the PSQ part
        // (out = in - psq(parent, old bucket) + psq(child, new bucket)) and keep the threat deltas incremental
        const bool swap = v.bucket != pv.bucket;
        Delta d;
#ifdef NNUE_PROF
        { PrT prof_m(5);
#endif
        for (Bitboard c = swap ? 0 : chg; c;) {
            int s = pop_lsb(c);
            if (parent.board[s] != NO_PIECE) d.ps[d.nps++] = psq_feat(v, persp, parent.board[s], s);
            if (child.board[s] != NO_PIECE) d.pa[d.npa++] = psq_feat(v, persp, child.board[s], s);
        }
        for (int i = 0; i < ns; i++) { int f = thr_feat(v, persp, sub[i].pc, sub[i].s, sub[i].v, sub[i].d); if (f >= 0) d.ts[d.nts++] = f; }
        for (int i = 0; i < na; i++) { int f = thr_feat(v, persp, add[i].pc, add[i].s, add[i].v, add[i].d); if (f >= 0) d.ta[d.nta++] = f; }
#ifdef NNUE_PROF
        }
#endif
        if (swap) {
            const int16_t* po = finny_sync(parent, persp, pv);
            apply(in.v[persp], out.v[persp], d, po, finny_sync(child, persp, v));
        } else
            apply(in.v[persp], out.v[persp], d);
        memset(out.psqt[persp], 0, sizeof(out.psqt[persp]));
    }
}

// ---- load ----
#if defined(__linux__)
#include <sys/mman.h>
static void* large_alloc(size_t n) {
    void* p = nullptr;
    if (posix_memalign(&p, 1 << 21, n)) return nullptr;
    madvise(p, n, MADV_HUGEPAGE);
    return p;
}
#else
static void* large_alloc(size_t n) { return aligned_alloc64(n); }
#endif
#if defined(__linux__) && defined(NNUE_MMAP) && !defined(THR16)
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
static void* MapBase = nullptr;
static size_t MapLen = 0;
static const uint8_t* map_file(const std::string& path, size_t need) {  // nullptr -> use fread
    if (MapBase) { munmap(MapBase, MapLen); MapBase = nullptr; }
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat st;
    void* p = fstat(fd, &st) == 0 && (size_t)st.st_size >= need ? mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
    close(fd);
    if (p == MAP_FAILED) return nullptr;
    MapBase = p, MapLen = st.st_size;
    return (const uint8_t*)p;
}
#else
static const uint8_t* map_file(const std::string&, size_t) { return nullptr; }
#endif
bool load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    int h[8];
    bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, "LNN6", 4) == 0 && fread(h, 4, 8, f) == 8 &&
              (h[0] == 1 || h[0] == 2) && h[1] == NNUE_ACC && h[2] == NPSQ && h[3] == NTHR && h[4] == L2 && h[5] == L3 && h[6] == NOB;
    if (!ok) {
        fclose(f);
        fprintf(stderr, "nnue: %s is not an LNN6 net of width %d (build with -DNNUE_LNN6 -DACC_WIDTH=<width>)\n",
                path.c_str(), NNUE_ACC);
        return false;
    }
    init_threats();
    if (2 * Half != NTHR) { fclose(f); fprintf(stderr, "nnue: threat table size %d != %d\n", 2 * Half, NTHR); return false; }
    Dual = h[7];
    int in2 = Dual ? 2 * L2 : L2;
    const size_t hdr = h[0] == 2 ? 64 : 36, psqB = (size_t)NPSQ * NNUE_ACC * 2, thrB = (size_t)NTHR * NNUE_ACC;
    const size_t psqOff = hdr + 2 * NNUE_ACC, thrOff = psqOff + psqB;
    std::vector<float> w2((size_t)NOB * in2 * L3);
    ok = fseek(f, (long)hdr, SEEK_SET) == 0 && fread(FTB, 2, NNUE_ACC, f) == (size_t)NNUE_ACC;
    const uint8_t* m = ok && h[0] == 2 ? map_file(path, thrOff + thrB) : nullptr;
    if (m) {
        PSQ = (const int16_t*)(m + psqOff), THR = (const thr_t*)(m + thrOff);
        ok = fseek(f, (long)(thrOff + thrB), SEEK_SET) == 0;
    } else if (ok) {
        if (!PsqBuf) PsqBuf = (int16_t*)large_alloc(psqB);
        if (!ThrBuf) ThrBuf = (thr_t*)large_alloc(thrB * sizeof(thr_t));
        ok = fread(PsqBuf, 1, psqB, f) == psqB && fread(ThrBuf, 1, thrB, f) == thrB;
#ifdef THR16  // the int8 rows were read into the front half, widen in place from the back
        for (size_t i = thrB; ok && i-- > 0;) ThrBuf[i] = ((const int8_t*)ThrBuf)[i];
#endif
        PSQ = PsqBuf, THR = ThrBuf;
    }
    ok = ok && fread(W1, 1, sizeof(W1), f) == sizeof(W1) &&
         fread(S1, 4, NOB * L2, f) == (size_t)NOB * L2 &&
         fread(B1, 4, NOB * L2, f) == (size_t)NOB * L2 &&
         fread(w2.data(), 4, w2.size(), f) == w2.size() &&
         fread(B2, 4, NOB * L3, f) == (size_t)NOB * L3 &&
         fread(W3, 4, NOB * L3, f) == (size_t)NOB * L3 &&
         fread(B3, 4, NOB, f) == (size_t)NOB;
    fclose(f);
    if (!ok) return false;
    Threats = false;
    for (size_t i = 0; i < thrB && !Threats; i += 64) {
        const thr_t* r = THR + i;
        for (int k = 0; k < 64; k++) Threats |= r[k] != 0;
    }
    for (int b = 0; b < NOB; b++)
        for (int o = 0; o < L2; o++) L1Deq[b][o] = X_DEQ / S1[b][o];
    memset(W2, 0, sizeof(W2));
    for (int b = 0; b < NOB; b++)
        for (int i = 0; i < in2; i++)
            for (int o = 0; o < L3; o++) W2[b][i][o] = w2[((size_t)b * in2 + i) * L3 + o];
    NetGen++;
    Loaded = true;
    return true;
}

// ---- evaluation ----
static inline float hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
static inline int out_bucket(int pieces) { return std::min(NOB - 1, std::max(0, (pieces - 2) / 4)); }

// L1 outputs (pre-activation) -> network output
static inline float head(const float* y, int b) {
    alignas(32) float hid[2 * L2];
    const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    for (int k = 0; k < L2; k += 8) {
        __m256 v = _mm256_load_ps(y + k);
        _mm256_store_ps(hid + k, _mm256_min_ps(_mm256_max_ps(v, zero), one));
        _mm256_store_ps(hid + L2 + k, _mm256_min_ps(_mm256_mul_ps(v, v), one));
    }
    int in2 = Dual ? 2 * L2 : L2;
    __m256 acc[4];
    for (int k = 0; k < 4; k++) acc[k] = _mm256_load_ps(&B2[b][8 * k]);
    for (int i = 0; i < in2; i++) {
        if (hid[i] == 0.0f) continue;
        __m256 s = _mm256_set1_ps(hid[i]);
        for (int k = 0; k < 4; k++) acc[k] = _mm256_fmadd_ps(s, _mm256_load_ps(&W2[b][i][8 * k]), acc[k]);
    }
    __m256 o = _mm256_setzero_ps();
    for (int k = 0; k < 4; k++)
        o = _mm256_fmadd_ps(_mm256_min_ps(_mm256_max_ps(acc[k], zero), one), _mm256_load_ps(&W3[b][8 * k]), o);
    return hsum_ps(o) + B3[b];
}

double evaluate_quant_raw(const Accumulator& acc, int stm, int pieces) {
    constexpr int H = NNUE_ACC / 2;
    alignas(32) uint8_t x[NNUE_ACC];
    const __m256i zero = _mm256_setzero_si256(), qmax = _mm256_set1_epi16(255);
    for (int half = 0; half < 2; half++) {
        const int16_t* a = acc.v[half ? stm ^ 1 : stm];
        for (int j = 0; j < H; j += 32) {
            __m256i a0 = _mm256_load_si256((const __m256i*)(a + j)), a1 = _mm256_load_si256((const __m256i*)(a + j + 16));
            __m256i b0 = _mm256_load_si256((const __m256i*)(a + H + j)), b1 = _mm256_load_si256((const __m256i*)(a + H + j + 16));
            a0 = _mm256_slli_epi16(_mm256_min_epi16(_mm256_max_epi16(a0, zero), qmax), 7);
            a1 = _mm256_slli_epi16(_mm256_min_epi16(_mm256_max_epi16(a1, zero), qmax), 7);
            b0 = _mm256_min_epi16(_mm256_max_epi16(b0, zero), qmax);
            b1 = _mm256_min_epi16(_mm256_max_epi16(b1, zero), qmax);
            __m256i p = _mm256_packus_epi16(_mm256_mulhi_epi16(a0, b0), _mm256_mulhi_epi16(a1, b1));
            _mm256_store_si256((__m256i*)(x + half * H + j), _mm256_permute4x64_epi64(p, 0xD8));
        }
    }
    const int b = out_bucket(pieces);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i s[4] = {zero, zero, zero, zero};
    const int32_t* xi = (const int32_t*)x;
    for (int c = 0; c < NNUE_ACC / 4; c += 8) {
        unsigned m = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_load_si256((const __m256i*)(xi + c)), zero)));
        while (m) {
            int k = c + (int)_tzcnt_u32(m);
            m &= m - 1;
            __m256i xv = _mm256_set1_epi32(xi[k]);
            const __m256i* w = (const __m256i*)W1[b][k];
            for (int q = 0; q < 4; q++)
                s[q] = _mm256_add_epi32(s[q], _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_load_si256(w + q)), ones));
        }
    }
    alignas(32) float y[L2];
    for (int q = 0; q < 4; q++)
        _mm256_store_ps(y + 8 * q, _mm256_fmadd_ps(_mm256_cvtepi32_ps(s[q]), _mm256_load_ps(&L1Deq[b][8 * q]), _mm256_load_ps(&B1[b][8 * q])));
    return head(y, b);
}

double evaluate_float_raw(const Accumulator& acc, int stm, int pieces) {
    constexpr int H = NNUE_ACC / 2;
    float x[NNUE_ACC];
    for (int half = 0; half < 2; half++) {
        const int16_t* a = acc.v[half ? stm ^ 1 : stm];
        for (int j = 0; j < H; j++) {
            float u = std::min(1.0f, std::max(0.0f, a[j] / 255.0f)), v = std::min(1.0f, std::max(0.0f, a[H + j] / 255.0f));
            x[half * H + j] = u * v;
        }
    }
    const int b = out_bucket(pieces);
    alignas(32) float y[L2];
    for (int o = 0; o < L2; o++) {
        double t = B1[b][o];
        for (int i = 0; i < NNUE_ACC; i++) t += (double)x[i] * W1[b][i / 4][o][i % 4] / S1[b][o];
        y[o] = (float)t;
    }
    return head(y, b);
}

int evaluate(const Accumulator& acc, int stm, int pieces) {
    PROF(2);
    return (int)std::lround(evaluate_quant_raw(acc, stm, pieces) * CP_PER_UNIT);
}

bool load_sigma(const std::string&) { return false; }
bool sigma_loaded() { return false; }
int evaluate_sigma(const Accumulator& acc, int stm, int pieces, int& sigma) {
    sigma = -1;
    return evaluate(acc, stm, pieces);
}

}  // namespace nnue
