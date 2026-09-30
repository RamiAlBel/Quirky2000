#include "nnue.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

extern int UseFinny;  // search.cpp tunable

namespace NNUE_NS {

static ft_t* FT = nullptr;                            // [rows][NNUE_ACC]
alignas(64) static int16_t FTB[NNUE_ACC];              // feature bias (all zero for LNN1)
static float FtScale = 0;

// ---- feature sets ----
// 0 HalfKP: king square x 10 non-king planes x 64 (kings are not features)
// 1 HKB (LNN4): king bucket x 11 planes (own P..Q, opp P..Q, both kings) x 64, board mirrored so the
//   perspective's king is on files a-d; the accumulator is refreshed when the king changes bucket or side
// 2 THR: HKB x (piece attacked by the opponent or not): 22 planes. The 24 piece-code x attacked
//   bitboards of parent and child are diffed on every update (moves change attack status non-locally)
static int FeatSet = 0, PsqtOn = 0, NB_psqt = 1;
static int KB[64], NKB = 8;
// weight sets (LNN5): one FT copy per set, chosen per perspective; 0 none, 1 phase2 (pieces >= 17 -> 0),
// 2 phase4 ((pieces-1)*4/32), 3 phase3 (>= 25 / 17..24 / <= 16), 4 colour of the perspective
static int SetSel = 0;
static const int SetCount[5] = {1, 2, 4, 3, 2};
static inline int set_of(const Position& pos, int persp) {
    if (SetSel == 0) return 0;
    if (SetSel == 4) return persp;
    int n = popcount(pos.occupied);
    if (SetSel == 1) return n >= 17 ? 0 : 1;
    if (SetSel == 2) return (n - 1) * 4 / 32;
    return n >= 25 ? 0 : n >= 17 ? 1 : 2;
}
static int32_t* PSQT = nullptr;  // [rows][NB_psqt]
static float PsqtScale = 0;

struct View { int base, flip, key; };  // key: refresh-cache slot (0..63)
static inline View view_of(const Position& pos, int persp) {
    int k = pos.king_sq(persp) ^ (persp == BLACK ? 56 : 0);
    if (FeatSet == 0) return {k * 640, persp == BLACK ? 56 : 0, k};
    int mirror = (k & 7) >= 4 ? 7 : 0;
    int b = KB[k ^ mirror], s = set_of(pos, persp);
    return {(s * NKB + b) * (FeatSet == 2 ? 1408 : 704), (persp == BLACK ? 56 : 0) ^ mirror, s * 64 + b * 2 + (mirror ? 1 : 0)};
}
static Bitboard attacks_by(const Position& p, int c) {
    Bitboard a = 0, occ = p.occupied, b;
    b = p.pcs(c, PAWN);   while (b) a |= bb::PawnAttacks[c][pop_lsb(b)];
    b = p.pcs(c, KNIGHT); while (b) a |= bb::KnightAttacks[pop_lsb(b)];
    b = p.pcs(c, BISHOP) | p.pcs(c, QUEEN); while (b) a |= bb::bishop_attacks(pop_lsb(b), occ);
    b = p.pcs(c, ROOK) | p.pcs(c, QUEEN);   while (b) a |= bb::rook_attacks(pop_lsb(b), occ);
    return a | bb::KingAttacks[p.king_sq(c)];
}
// THR feature sets: set[2*code + attacked] = squares of piece code with that attack status
static void threat_sets(const Position& p, Bitboard set[24]) {
    Bitboard att[2] = {attacks_by(p, WHITE), attacks_by(p, BLACK)};
    for (int k = 0; k < 12; k++) {
        Bitboard a = att[color_of(k) ^ 1];
        set[2 * k] = p.pieces[k] & ~a;
        set[2 * k + 1] = p.pieces[k] & a;
    }
}
// feature of set index i (THR: code*2+att; otherwise the piece code) on square sq
static inline int feat_set(const View& v, int persp, int i, int sq) {
    int code = FeatSet == 2 ? i >> 1 : i, rel = color_of(code) == persp ? 0 : 1, t = type_of(code);
    int plane = t == KING ? 10 : rel * 5 + t;
    if (FeatSet == 2) plane = plane * 2 + (i & 1);
    return v.base + plane * 64 + (sq ^ v.flip);
}
static inline int n_sets() { return FeatSet == 2 ? 24 : 12; }
static inline void get_sets(const Position& p, Bitboard set[24]) {
    if (FeatSet == 2) threat_sets(p, set);
    else for (int k = 0; k < 12; k++) set[k] = p.pieces[k];
}
static inline int feat(const View& v, int persp, int pc, int sq) {
    int rel = color_of(pc) == persp ? 0 : 1, t = type_of(pc);
    int plane = t == KING ? 10 : rel * 5 + t;
    return v.base + plane * 64 + (sq ^ v.flip);
}
static inline bool is_feature_piece(int pc) { return FeatSet == 1 || type_of(pc) != KING; }
static inline __m256i row16(int idx, int j) {
#if defined(NNUE_LNN2) && !defined(FT_INT8)
    return _mm256_load_si256((const __m256i*)(FT + (size_t)idx * NNUE_ACC + j));
#else
    return _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(FT + (size_t)idx * NNUE_ACC + j)));
#endif
}
static inline void psqt_apply(int32_t* dst, const int* add, int na, const int* sub, int ns) {
    for (int k = 0; k < na; k++) for (int b = 0; b < NB_psqt; b++) dst[b] += PSQT[(size_t)add[k] * NB_psqt + b];
    for (int k = 0; k < ns; k++) for (int b = 0; b < NB_psqt; b++) dst[b] -= PSQT[(size_t)sub[k] * NB_psqt + b];
}

void refresh(const Position& pos, Accumulator& acc, int persp) {
    View vw = view_of(pos, persp);
    int idx[32], n = 0;
    if (FeatSet == 2) {
        Bitboard set[24];
        threat_sets(pos, set);
        for (int i = 0; i < 24; i++) for (Bitboard b = set[i]; b;) idx[n++] = feat_set(vw, persp, i, pop_lsb(b));
    } else {
        Bitboard b = FeatSet == 1 ? pos.occupied : pos.occupied & ~(pos.pcs(WHITE, KING) | pos.pcs(BLACK, KING));
        while (b) { int s = pop_lsb(b); idx[n++] = feat(vw, persp, pos.board[s], s); }
    }
    // 4 registers x 16 lanes = 64 lanes per pass keeps sums in registers across all pieces
    for (int j = 0; j < NNUE_ACC; j += 64) {
        const __m256i* bias = (const __m256i*)(FTB + j);
        __m256i a0 = _mm256_load_si256(bias), a1 = _mm256_load_si256(bias + 1),
                a2 = _mm256_load_si256(bias + 2), a3 = _mm256_load_si256(bias + 3);
        for (int k = 0; k < n; k++) {
            a0 = _mm256_add_epi16(a0, row16(idx[k], j));
            a1 = _mm256_add_epi16(a1, row16(idx[k], j + 16));
            a2 = _mm256_add_epi16(a2, row16(idx[k], j + 32));
            a3 = _mm256_add_epi16(a3, row16(idx[k], j + 48));
        }
        __m256i* out = (__m256i*)(acc.v[persp] + j);
        _mm256_store_si256(out, a0); _mm256_store_si256(out + 1, a1);
        _mm256_store_si256(out + 2, a2); _mm256_store_si256(out + 3, a3);
    }
    memset(acc.psqt[persp], 0, sizeof(acc.psqt[persp]));
    if (PsqtOn) psqt_apply(acc.psqt[persp], idx, n, nullptr, 0);
}

// Refresh cache ("Finny table"): per thread, per perspective and king square, the accumulator and piece
// sets it was last refreshed with. A king-move refresh then only applies the pieces that changed since.
// Integer add/sub wrap mod 2^16, so the result is bit-identical to a full refresh.
static int NetGen = 0;  // bumped on load: invalidates every thread's cache
struct alignas(64) FinnyEntry {
    int16_t acc[NNUE_ACC];
    int32_t psqt[8];
    Bitboard pcs[24];  // feature sets (piece codes; THR: code x attacked)
};
static thread_local FinnyEntry Finny[2][4 * 64];  // [persp][set * 64 + bucket * 2 + mirrored]
static thread_local int FinnyGen = -1;

static void refresh_cached(const Position& pos, Accumulator& acc, int persp) {
    if (FinnyGen != NetGen) {
        for (auto& side : Finny)
            for (auto& e : side) {
                memcpy(e.acc, FTB, sizeof(e.acc));
                memset(e.psqt, 0, sizeof(e.psqt));
                memset(e.pcs, 0, sizeof(e.pcs));
            }
        FinnyGen = NetGen;
    }
    View vw = view_of(pos, persp);
    FinnyEntry& e = Finny[persp][vw.key];
    int add[32], sub[32], na = 0, ns = 0;
    Bitboard set[24];
    get_sets(pos, set);
    for (int i = 0; i < n_sets(); i++) {
        if (FeatSet == 0 && type_of(i) == KING) continue;
        Bitboard gone = e.pcs[i] & ~set[i], come = set[i] & ~e.pcs[i];
        while (gone && ns < 32) sub[ns++] = feat_set(vw, persp, i, pop_lsb(gone));
        while (come && na < 32) add[na++] = feat_set(vw, persp, i, pop_lsb(come));
        e.pcs[i] = set[i];
    }
    if (PsqtOn) psqt_apply(e.psqt, add, na, sub, ns);
    memcpy(acc.psqt[persp], e.psqt, sizeof(e.psqt));
    for (int j = 0; j < NNUE_ACC; j += 64) {
        __m256i* c = (__m256i*)(e.acc + j);
        __m256i a0 = _mm256_load_si256(c), a1 = _mm256_load_si256(c + 1),
                a2 = _mm256_load_si256(c + 2), a3 = _mm256_load_si256(c + 3);
        for (int k = 0; k < ns; k++) {
            a0 = _mm256_sub_epi16(a0, row16(sub[k], j));
            a1 = _mm256_sub_epi16(a1, row16(sub[k], j + 16));
            a2 = _mm256_sub_epi16(a2, row16(sub[k], j + 32));
            a3 = _mm256_sub_epi16(a3, row16(sub[k], j + 48));
        }
        for (int k = 0; k < na; k++) {
            a0 = _mm256_add_epi16(a0, row16(add[k], j));
            a1 = _mm256_add_epi16(a1, row16(add[k], j + 16));
            a2 = _mm256_add_epi16(a2, row16(add[k], j + 32));
            a3 = _mm256_add_epi16(a3, row16(add[k], j + 48));
        }
        _mm256_store_si256(c, a0); _mm256_store_si256(c + 1, a1);
        _mm256_store_si256(c + 2, a2); _mm256_store_si256(c + 3, a3);
        __m256i* out = (__m256i*)(acc.v[persp] + j);
        _mm256_store_si256(out, a0); _mm256_store_si256(out + 1, a1);
        _mm256_store_si256(out + 2, a2); _mm256_store_si256(out + 3, a3);
    }
}

void refresh_all(const Position& pos, Accumulator& acc) {
    refresh(pos, acc, WHITE);
    refresh(pos, acc, BLACK);
}

void update(const Position& parent, Move m, const Position& child, const Accumulator& in, Accumulator& out) {
    int us = parent.stm, them = us ^ 1;
    int from = from_sq(m), to = to_sq(m), fl = flags_of(m);
    int pc = parent.board[from], pt = type_of(pc);
    int placed = is_promo(m) ? make_piece(us, promo_type(m)) : pc;
    int capPc = NO_PIECE, capSq = -1;
    if (fl == MF_EP) { capPc = make_piece(them, PAWN); capSq = to ^ 8; }
    else if (fl & 4) { capPc = parent.board[to]; capSq = to; }
    int rook = make_piece(us, ROOK), rFrom = -1, rTo = -1;
    if (fl == MF_KCASTLE) { rFrom = to + 1; rTo = to - 1; }
    else if (fl == MF_QCASTLE) { rFrom = to - 2; rTo = to + 1; }

    Bitboard ps[24], cs[24];
    if (FeatSet == 2) { threat_sets(parent, ps); threat_sets(child, cs); }
    for (int persp = 0; persp < 2; persp++) {
        View vw = view_of(child, persp);
        if ((pt == KING && persp == us) || (SetSel >= 1 && SetSel <= 3 && capPc != NO_PIECE)) {  // bucket/set may change
            View pv = view_of(parent, persp);
            if (FeatSet == 0 || pv.base != vw.base || pv.flip != vw.flip) {
                if (UseFinny) refresh_cached(child, out, persp);
                else refresh(child, out, persp);
                continue;
            }
        }
        int add[32], sub[32], na = 0, ns = 0;
        if (FeatSet == 2) {
            for (int i = 0; i < 24; i++) {
                for (Bitboard b = ps[i] & ~cs[i]; b && ns < 32;) sub[ns++] = feat_set(vw, persp, i, pop_lsb(b));
                for (Bitboard b = cs[i] & ~ps[i]; b && na < 32;) add[na++] = feat_set(vw, persp, i, pop_lsb(b));
            }
        } else {
            if (is_feature_piece(pc)) { sub[ns++] = feat(vw, persp, pc, from); add[na++] = feat(vw, persp, placed, to); }
            if (capPc != NO_PIECE) sub[ns++] = feat(vw, persp, capPc, capSq);
            if (rFrom >= 0) { sub[ns++] = feat(vw, persp, rook, rFrom); add[na++] = feat(vw, persp, rook, rTo); }
        }
        const __m256i* src = (const __m256i*)in.v[persp];
        __m256i* dst = (__m256i*)out.v[persp];
        for (int c = 0; c < NNUE_ACC / 16; c++) {
            __m256i v = _mm256_load_si256(src + c);
            for (int k = 0; k < ns; k++) v = _mm256_sub_epi16(v, row16(sub[k], c * 16));
            for (int k = 0; k < na; k++) v = _mm256_add_epi16(v, row16(add[k], c * 16));
            _mm256_store_si256(dst + c, v);
        }
        memcpy(out.psqt[persp], in.psqt[persp], sizeof(out.psqt[persp]));
        if (PsqtOn) psqt_apply(out.psqt[persp], add, na, sub, ns);
    }
}

static inline int hsum_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}
static inline float hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

#ifndef NNUE_LNN2
// ---- LNN1: fixed 256 -> 8 -> 32 -> 1 head ----

alignas(64) static int8_t L1W[NNUE_H1][2 * NNUE_ACC];  // quantized
static float L1Wf[NNUE_H1][2 * NNUE_ACC];              // float reference
static float L1B[NNUE_H1], L1Dequant;
alignas(32) static float L2WT[NNUE_H1][NNUE_H2];        // transposed: [input][output]
alignas(32) static float L2B[NNUE_H2], L3W[NNUE_H2];
static float L3B;
static int16_t RequantMul;                             // uint8 = mulhrs(clamp(acc,0,..), RequantMul)

// Network output o predicts tanh(pawns/4) as tanh(o), so centipawns = 400*o.
static constexpr float CP_PER_UNIT = 400.0f;

bool load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    int acc, rows, h1, h2;
    float scale;
#ifdef NNUE_LNN2
    const char* want = "LNN2";
#else
    const char* want = "LNN1";
#endif
    bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, want, 4) == 0 &&
              fread(&acc, 4, 1, f) == 1 && fread(&rows, 4, 1, f) == 1 && fread(&h1, 4, 1, f) == 1 &&
              fread(&h2, 4, 1, f) == 1 && fread(&scale, 4, 1, f) == 1 &&
              acc == NNUE_ACC && h1 == NNUE_H1 && h2 == NNUE_H2 && rows == 40961;
    if (!ok) {
        fclose(f);
        fprintf(stderr, "nnue: %s is not a %s net of width %d (build with -DNNUE_LNN2 -DACC_WIDTH=<width> for LNN2)\n",
                path.c_str(), want, NNUE_ACC);
        return false;
    }
    FtScale = scale;
    NetGen++;
    if (FT) aligned_free64(FT);
    FT = (ft_t*)aligned_alloc64((size_t)rows * NNUE_ACC * sizeof(ft_t));
    std::vector<float> l1w(h1 * 2 * acc), l2w(h2 * h1);
#ifdef NNUE_LNN2
    ok = fread(FTB, 2, acc, f) == (size_t)acc;
#else
    memset(FTB, 0, sizeof(FTB));
#endif
    ok = ok && fread(FT, sizeof(ft_t), (size_t)rows * acc, f) == (size_t)rows * acc &&
         fread(l1w.data(), 4, l1w.size(), f) == l1w.size() &&
         fread(L1B, 4, h1, f) == (size_t)h1 &&
         fread(l2w.data(), 4, l2w.size(), f) == l2w.size() &&
         fread(L2B, 4, h2, f) == (size_t)h2 &&
         fread(L3W, 4, h2, f) == (size_t)h2 &&
         fread(&L3B, 4, 1, f) == 1;
    fclose(f);
    if (!ok) return false;

    float maxAbs = 0;
    for (float w : l1w) maxAbs = std::fmax(maxAbs, std::fabs(w));
    float s1 = 127.0f / maxAbs;  // widest int8 range the weights allow
    for (int o = 0; o < h1; o++)
        for (int i = 0; i < 2 * acc; i++) {
            float w = l1w[o * 2 * acc + i];
            L1Wf[o][i] = w;
            L1W[o][i] = (int8_t)std::lround(w * s1);
        }
    L1Dequant = 1.0f / (127.0f * s1);
    for (int o = 0; o < h2; o++)
        for (int i = 0; i < h1; i++) L2WT[i][o] = l2w[o * h1 + i];
    // acc_int * scale in [0,1]  ->  uint8 in [0,127]:  (v * M + 2^14) >> 15
    RequantMul = (int16_t)std::lround(FtScale * 127.0f * 32768.0f);
    return true;
}

// Shared tail: 8 hidden activations (already clipped) -> 32 -> 1
static inline float tail(const float* h1) {
    __m256 acc[4];
    for (int k = 0; k < 4; k++) acc[k] = _mm256_load_ps(L2B + 8 * k);
    for (int i = 0; i < NNUE_H1; i++) {
        __m256 s = _mm256_set1_ps(h1[i]);
        for (int k = 0; k < 4; k++) acc[k] = _mm256_fmadd_ps(s, _mm256_load_ps(&L2WT[i][8 * k]), acc[k]);
    }
    const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    __m256 o = _mm256_setzero_ps();
    for (int k = 0; k < 4; k++) {
        __m256 h2 = _mm256_min_ps(_mm256_max_ps(acc[k], zero), one);
        o = _mm256_fmadd_ps(h2, _mm256_load_ps(L3W + 8 * k), o);
    }
    return hsum_ps(o) + L3B;
}

double evaluate_quant_raw(const Accumulator& acc, int stm, int) {
    alignas(32) uint8_t x[2 * NNUE_ACC];
    const __m256i zero = _mm256_setzero_si256(), mul = _mm256_set1_epi16(RequantMul), cap = _mm256_set1_epi8(127);
    for (int half = 0; half < 2; half++) {
        const __m256i* a = (const __m256i*)acc.v[half == 0 ? stm : stm ^ 1];
        for (int c = 0; c < NNUE_ACC / 32; c++) {
            __m256i lo = _mm256_mulhrs_epi16(_mm256_max_epi16(_mm256_load_si256(a + 2 * c), zero), mul);
            __m256i hi = _mm256_mulhrs_epi16(_mm256_max_epi16(_mm256_load_si256(a + 2 * c + 1), zero), mul);
            __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi16(lo, hi), 0xD8);
            _mm256_store_si256((__m256i*)(x + half * NNUE_ACC + c * 32), _mm256_min_epu8(p, cap));
        }
    }
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i sum[NNUE_H1];
    for (int o = 0; o < NNUE_H1; o++) sum[o] = _mm256_setzero_si256();
    for (int i = 0; i < 2 * NNUE_ACC; i += 32) {
        __m256i xv = _mm256_load_si256((const __m256i*)(x + i));
        for (int o = 0; o < NNUE_H1; o++) {
            __m256i p = _mm256_maddubs_epi16(xv, _mm256_load_si256((const __m256i*)&L1W[o][i]));
            sum[o] = _mm256_add_epi32(sum[o], _mm256_madd_epi16(p, ones));
        }
    }
    alignas(32) float h1[NNUE_H1];
    for (int o = 0; o < NNUE_H1; o++) {
        float v = hsum_epi32(sum[o]) * L1Dequant + L1B[o];
        h1[o] = v < 0 ? 0 : (v > 1 ? 1 : v);
    }
    return tail(h1);
}

double evaluate_float_raw(const Accumulator& acc, int stm, int) {
    float x[2 * NNUE_ACC];
    for (int half = 0; half < 2; half++) {
        const int16_t* a = acc.v[half == 0 ? stm : stm ^ 1];
        for (int j = 0; j < NNUE_ACC; j++) {
            float v = a[j] * FtScale;
            x[half * NNUE_ACC + j] = v < 0 ? 0 : (v > 1 ? 1 : v);
        }
    }
    alignas(32) float h1[NNUE_H1];
    for (int o = 0; o < NNUE_H1; o++) {
        double s = L1B[o];
        for (int i = 0; i < 2 * NNUE_ACC; i++) s += (double)x[i] * L1Wf[o][i];
        h1[o] = (float)(s < 0 ? 0 : (s > 1 ? 1 : s));
    }
    return tail(h1);
}

int evaluate(const Accumulator& acc, int stm, int pieces) {
    return (int)std::lround(evaluate_quant_raw(acc, stm, pieces) * CP_PER_UNIT);
}

#else
#include "nnue_head.inc"
#endif

}  // namespace NNUE_NS
