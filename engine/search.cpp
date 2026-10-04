#include "search.h"
#include "nnue.h"
#include "nnue_small.h"
#include "tune.h"
#include "tbprobe.h"
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <climits>
#include <algorithm>

std::vector<Tunable>& tunables() { static std::vector<Tunable> t; return t; }
bool set_tunable(const std::string& name, int value) {
    for (auto& t : tunables())
        if (name == t.name) { *t.v = std::clamp(value, t.lo, t.hi); return true; }
    return false;
}

// ---- experiment switches (0 = behaviour before the experiments) ----
TUNE(UseFinny, 0, 0, 1);    // king-move refresh from a per-thread accumulator cache (nnue.cpp)
TUNE(UseLazyAcc, 0, 0, 1);  // update accumulators only when a node is actually evaluated
TUNE(UseCorrHist, 0, 0, 1);  // pawn-structure correction history on the static eval
TUNE(CorrDiv, 256, 64, 1024);
TUNE(UseContHist, 0, 0, 1);  // 1- and 2-ply continuation history (ordering + LMR)
TUNE(UseCaptHist, 0, 0, 1);  // capture history (ordering)
TUNE(UseSingular, 0, 0, 1);  // singular extensions + multicut + negative extension
TUNE(SingDepth, 8, 4, 12);
TUNE(SingMargin, 2, 1, 6);
TUNE(UseProbCut, 0, 0, 1);
TUNE(ProbCutMargin, 200, 50, 400);
TUNE(UseRazor, 0, 0, 1);
TUNE(RazorMargin, 400, 100, 1000);
TUNE(RazorMul, 250, 50, 500);
TUNE(SyzygyProbeDepth, 1, 1, 10);  // Syzygy is active when SyzygyPath is set (UCI string option)
TUNE(UseBook, 0, 0, 1);      // play from the Polyglot book set with BookFile (main.cpp)
TUNE(BookDepth, 20, 0, 255);  // plies from the game start
TUNE(BookBest, 0, 0, 1);      // 1 = always the highest-weight book move (Cerebellum/BrainFish style), 0 = weighted random
TUNE(UseSmallNet, 0, 0, 1);            // 128-wide net (SmallNetFile) when material is lopsided
TUNE(SmallNetThreshold, 1000, -1, 3000);  // |material balance| in cp above which the small net is used
TUNE(UseTM, 0, 0, 1);        // soft/hard time limits + best-move stability
TUNE(TmSoftDiv, 30, 10, 60);
TUNE(TmIncPct, 75, 0, 100);
TUNE(TmHardMul, 4, 2, 8);
// pre-existing search constants (defaults = old hard-coded values), for SPSA
TUNE(RfpDepth, 9, 4, 14);
TUNE(RfpMargin, 85, 40, 150);
TUNE(RfpImp, 25, 0, 60);
TUNE(NmpBase, 3, 2, 5);
TUNE(NmpDiv, 3, 2, 6);
TUNE(NmpEvalDiv, 200, 80, 400);
TUNE(LmpBase, 3, 1, 8);
TUNE(FutBase, 100, 30, 250);
TUNE(FutMul, 110, 40, 250);
TUNE(SeeQuiet, 25, 5, 80);
TUNE(SeeNoisy, 100, 30, 200);
TUNE(HistDiv, 6000, 2000, 16000);
TUNE(LmrBase, 75, 0, 150);
TUNE(LmrDiv, 225, 150, 400);
TUNE(AspDelta, 18, 5, 60);
TUNE(QsFut, 200, 50, 400);
// ---- round E (on top of bundle D) ----
TUNE(UseCorr2, 0, 0, 1);    // correction history v2: weighted average (not a sum), pawn + per-colour non-pawn keys
TUNE(Corr2W, 256, 64, 1024); // update denominator: weight min(depth+1,16)/Corr2W
TUNE(Corr2P, 100, 0, 200);   // % weight of the pawn table
TUNE(Corr2N, 50, 0, 200);    // % weight of each non-pawn table
TUNE(UseTtPv, 0, 0, 1);      // remember "was on the PV" in the TT; reduce such nodes less
TUNE(UseHistPrune, 0, 0, 1); // prune quiets with very bad history at low depth
TUNE(HpDepth, 4, 1, 8);
TUNE(HpMul, 2000, 500, 8000);
TUNE(UseCont4, 0, 0, 1);     // continuation history also for the move 4 plies back
TUNE(UseDblExt, 0, 0, 1);    // double singular extension (non-PV, bounded per line)
TUNE(DblMargin, 20, 0, 100);
TUNE(DblLimit, 6, 1, 16);
TUNE(UseNodeTM, 0, 0, 1);    // soft limit scaled by the share of root nodes spent on the best move
TUNE(NtmBase, 150, 100, 250);
TUNE(NtmMul, 135, 50, 250);
TUNE(UseRfpBlend, 0, 0, 1);  // reverse futility returns (eval+beta)/2 instead of eval
TUNE(UseLmrTtCap, 0, 0, 1);  // quiets reduced one more ply when the TT move is a capture
TUNE(UsePawnHist, 0, 0, 1);  // pawn history: quiet score by (pawn structure, piece, to)
TUNE(UsePriorBonus, 0, 0, 1); // node fails low -> the parent's quiet move that led here gets a history bonus
TUNE(UseAspFH, 0, 0, 1);     // aspiration fail high -> re-search one ply shallower (cumulative)
TUNE(UseCapFut, 0, 0, 1);    // futility pruning for captures in the main search
TUNE(CapFutBase, 200, 50, 500);
TUNE(CapFutMul, 150, 50, 400);
TUNE(UseDeeper, 0, 0, 1);    // LMR re-search one ply deeper / shallower depending on how much it surprised
TUNE(DeeperMargin, 40, 10, 150);
// ---- uncertainty-driven search (sigma head, SigmaFile) ----
// ---- flag mode: opponent low on time and well behind us on the clock ----
TUNE(UseFlag, 0, 0, 1);       // move faster (keep the clock lead) and avoid draws while the opponent may flag
TUNE(FlagOppMs, 20000, 1000, 120000);
TUNE(FlagRatio, 200, 100, 800);   // our time >= FlagRatio% of theirs
TUNE(FlagTimePct, 60, 20, 100);   // soft/hard limits scaled by this
TUNE(FlagContempt, 30, 0, 200);   // draw = -FlagContempt cp for us
TUNE(FlagNoInc, 0, 0, 1);         // 1: flag mode only when neither side has an increment (1+0, 3+0, ...)
TUNE(UseSigma, 0, 0, 1);     // RFP and futility margins scaled by the node's predicted eval error
TUNE(SigRef, 100, 20, 400);  // sigma (cp) at which margins are unchanged
TUNE(SigMix, 50, 0, 100);    // % of the margin that scales with sigma/SigRef
TUNE(UseSigLmr, 0, 0, 1);    // reduce more in calm nodes (sigma < SigLo), less in sharp ones (sigma > SigHi)
TUNE(SigLo, 40, 5, 200);
TUNE(SigHi, 200, 50, 800);
// ---- round H (Coda-style search ideas, on top of version G) ----
TUNE(UseThreatHist, 0, 0, 1); // main history also indexed by "from attacked" x "to attacked" (by the opponent)
TUNE(UseCont6, 0, 0, 1);      // continuation history at plies 1, 2, 4 and 6
TUNE(Cont1W, 100, 0, 200);    // % weights of the continuation tables in the quiet score (UseCont6)
TUNE(Cont2W, 100, 0, 200);
TUNE(Cont4W, 50, 0, 200);
TUNE(Cont6W, 50, 0, 200);
TUNE(PawnHistW, 100, 0, 200); // % weight of pawn history in the quiet score (UsePawnHist, incremental pawn key)
TUNE(UseCorr5, 0, 0, 1);      // correction history from pawn, white/black non-pawn, continuation and transition keys
TUNE(C5W, 256, 64, 1024);     // update denominator
TUNE(C5P, 100, 0, 200);       // % weights of the five tables
TUNE(C5N, 50, 0, 200);
TUNE(C5C, 50, 0, 200);
TUNE(C5T, 50, 0, 200);
TUNE(UseCuckoo, 0, 0, 1);     // upcoming-repetition detection: raise alpha to the draw score if a cycle is one move away
TUNE(UseHindsight, 0, 0, 1);  // adjust depth by how the parent's reduced move turned out
TUNE(HsRed, 3, 1, 6);         // parent reduction >= HsRed and both sides not worsening -> depth + 1
TUNE(HsWorsen, 2, -50, 50);
TUNE(HsMargin, 200, 50, 400); // parent reduced and both static evals sum above this -> depth - 1
TUNE(UseFhBlend, 0, 0, 1);    // non-PV fail high: return (best * depth + beta) / (depth + 1)
TUNE(UseTtBlend, 0, 0, 1);    // non-PV TT cutoff above beta: same blend with the TT depth
TUNE(UseQsBlend, 0, 0, 1);    // qsearch fail high: (best + beta) / 2
TUNE(UseHistNorm, 0, 0, 1);   // LMR: scale the quiet score back to the weight of the G tables (main + 2 cont = 300%)

static std::mutex outMutex;
static bool quietOutput = false;
void out_line(const char* fmt, ...) {
    std::lock_guard<std::mutex> lk(outMutex);
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fputc('\n', stdout);
    fflush(stdout);
}

// ======================= transposition table =======================
// 4 x 16-byte entries per 64-byte cluster. Lockless across threads via the
// classic "store key ^ data" trick: a torn write from a racing thread makes the
// xor check fail, so it reads as a miss instead of as corrupted data.
enum : int { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };
struct TTEntry { std::atomic<uint64_t> key; std::atomic<uint64_t> data; };
struct alignas(64) Cluster { TTEntry e[4]; };
struct TTData { Move move; int score, eval, depth, bound; bool pv; };

static Cluster* ttTable = nullptr;
static uint64_t ttClusters = 0;
static int ttGen = 0;

static inline uint64_t tt_pack(Move m, int score, int eval, int depth, int bound, bool pv) {
    return (uint64_t)m | ((uint64_t)(uint16_t)(int16_t)score << 16) | ((uint64_t)(uint16_t)(int16_t)eval << 32) |
           ((uint64_t)((depth & 0x7F) | (pv << 7)) << 48) | ((uint64_t)(bound | (ttGen << 2)) << 56);
}
static inline int tt_bound(uint64_t d) { return (int)((d >> 56) & 3); }
static inline int tt_depth(uint64_t d) { return (int)((d >> 48) & 0x7F); }
static inline bool tt_pv(uint64_t d) { return (d >> 55) & 1; }
static inline int tt_gen(uint64_t d) { return (int)(d >> 58); }
#ifdef _MSC_VER
static inline Cluster& tt_cluster(uint64_t key) { return ttTable[__umulh(key, ttClusters)]; }
#else
static inline Cluster& tt_cluster(uint64_t key) { return ttTable[(uint64_t)(((unsigned __int128)key * ttClusters) >> 64)]; }
#endif

// start loading the child's TT cluster into cache right after the move is made, so the probe at the start of the
// child node does not stall on a cache miss (-DNO_TT_PREFETCH turns it off for A/B tests)
static inline void tt_prefetch(uint64_t key) {
#ifndef NO_TT_PREFETCH
    _mm_prefetch((const char*)&tt_cluster(key), _MM_HINT_T0);
#endif
}

static bool tt_probe(uint64_t key, TTData& out) {
    Cluster& c = tt_cluster(key);
    for (auto& e : c.e) {
        uint64_t d = e.data.load(std::memory_order_relaxed);
        uint64_t k = e.key.load(std::memory_order_relaxed);
        if ((k ^ d) == key && tt_bound(d) != BOUND_NONE) {
            out.move = (Move)(d & 0xFFFF);
            out.score = (int16_t)(d >> 16);
            out.eval = (int16_t)(d >> 32);
            out.depth = tt_depth(d);
            out.bound = tt_bound(d);
            out.pv = tt_pv(d);
            return true;
        }
    }
    return false;
}

static void tt_store(uint64_t key, Move move, int score, int eval, int depth, int bound, bool pv = false) {
    Cluster& c = tt_cluster(key);
    TTEntry* rep = nullptr;
    uint64_t old = 0;
    bool same = false;
    int bestRank = INT_MAX;
    for (auto& e : c.e) {
        uint64_t d = e.data.load(std::memory_order_relaxed);
        uint64_t k = e.key.load(std::memory_order_relaxed);
        if ((k ^ d) == key || tt_bound(d) == BOUND_NONE) { rep = &e; old = d; same = (k ^ d) == key && tt_bound(d) != BOUND_NONE; break; }
        int rank = tt_depth(d) - 8 * ((ttGen - tt_gen(d)) & 63);
        if (rank < bestRank) { bestRank = rank; rep = &e; old = d; }
    }
    if (same) {
        if (!move) move = (Move)(old & 0xFFFF);
        if (bound != BOUND_EXACT && depth + 4 < tt_depth(old) && tt_gen(old) == ttGen) return;
    }
    uint64_t d = tt_pack(move, score, eval, depth, bound, pv);
    rep->data.store(d, std::memory_order_relaxed);
    rep->key.store(key ^ d, std::memory_order_relaxed);
}

static int hashfull() {
    int n = 0;
    for (int i = 0; i < 250 && i < (int)ttClusters; i++)
        for (auto& e : ttTable[i].e) {
            uint64_t d = e.data.load(std::memory_order_relaxed);
            if (tt_bound(d) != BOUND_NONE && tt_gen(d) == ttGen) n++;
        }
    return n;
}

static inline int value_to_tt(int v, int ply) { return v >= MATE_BOUND ? v + ply : v <= -MATE_BOUND ? v - ply : v; }
static inline int value_from_tt(int v, int ply) {
    if (v == VALUE_NONE) return VALUE_NONE;
    return v >= MATE_BOUND ? v - ply : v <= -MATE_BOUND ? v + ply : v;
}

// ======================= search state =======================
struct RootMove {
    Move move = 0;
    int score = -VALUE_INF, prevScore = -VALUE_INF, selDepth = 0;
    std::vector<Move> pv;
    uint64_t nodes = 0;  // main worker: nodes spent below this root move
};

struct Worker {
    int id = 0;
    Position pos[MAX_PLY + 4];
    Accumulator acc[MAX_PLY + 4];
    bool accOk[MAX_PLY + 4];  // lazy mode: acc[ply] is up to date
    nnue_small::Accumulator accS[MAX_PLY + 4];  // small net, always lazy
    bool accSOk[MAX_PLY + 4];
    std::vector<uint64_t> keys;
    int gameLen = 0;
    int pfn[MAX_PLY + 4];  // plies since last null move (bounds repetition scans)
    Move killers[MAX_PLY + 4][2];
    int hist[2][64][64];
    Move counter[12][64];
    int16_t corr[2][16384];
    int16_t corrP[2][16384], corrN[2][2][16384];  // v2: [stm][pawn key], [stm][colour][non-pawn key]
    int dext[MAX_PLY + 4];
    int sigma[MAX_PLY + 4];
    bool sigOk[MAX_PLY + 4];
    int16_t phist[1024][12][64];  // [pawn key][piece][to]  // double extensions on the current line
    int16_t cont[12][64][12][64];  // [prev piece][prev to][piece][to]
    int thist[2][2][2][64][64];    // [stm][from attacked][to attacked][from][to] (UseThreatHist)
    Bitboard thr[MAX_PLY + 4];     // squares attacked by the side not to move (UseThreatHist)
    int redIn[MAX_PLY + 4];        // reduction applied to the move that led to this ply (UseHindsight)
    int16_t c5P[2][16384], c5W[2][16384], c5B[2][16384], c5T[2][16384];  // [stm][key bits] (UseCorr5)
    int16_t c5C[12][64][12][64];   // [piece][to] two plies back x [piece][to] one ply back
    int capt[12][64][6];           // [piece][to][victim type; 5 = none (promotion)]
    Move excluded[MAX_PLY + 4];
    int staticEval[MAX_PLY + 4];
    Move currentMove[MAX_PLY + 4];
    int movedPiece[MAX_PLY + 4];
    Move pv[MAX_PLY + 4][MAX_PLY + 4];
    int pvLen[MAX_PLY + 4];
    std::atomic<uint64_t> nodes{0};
    int selDepth = 0, rootDepth = 0, pvIdx = 0, multiPV = 1, completedDepth = 0;
    std::vector<RootMove> rootMoves;
    uint32_t tick = 0;
};

static std::vector<Worker*> workers;
static std::atomic<bool> stopFlag{false};
static Limits limits;
static std::chrono::steady_clock::time_point startTime;
static std::thread mainThread;
static int optThreads = 1, optMultiPV = 1;
static std::atomic<bool> pondering{false};
static int64_t softMs = 0;
static int drawScore = 0;  // contempt, from the root side's view (flag mode)
static inline int draw_value(int ply) { return (ply & 1) ? drawScore : -drawScore; }
static int LMR[64][64];

static int64_t elapsed_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
}
static uint64_t total_nodes() {
    uint64_t n = 0;
    for (auto* w : workers) n += w->nodes.load(std::memory_order_relaxed);
    return n;
}
static void check_time() {
    if (pondering.load(std::memory_order_relaxed)) return;
    if (limits.movetime && elapsed_ms() >= limits.movetime) stopFlag = true;
    if (limits.nodes && total_nodes() >= limits.nodes) stopFlag = true;
}

// ======================= helpers =======================
static const int SeeVal[6] = {100, 320, 330, 500, 950, 0};

static bool see_ge(const Position& pos, Move m, int threshold) {
    if (flags_of(m) == MF_EP || is_castle(m) || is_promo(m)) return 0 >= threshold;
    int from = from_sq(m), to = to_sq(m);
    int swap = (pos.board[to] != NO_PIECE ? SeeVal[type_of(pos.board[to])] : 0) - threshold;
    if (swap < 0) return false;
    swap = SeeVal[type_of(pos.board[from])] - swap;
    if (swap <= 0) return true;
    Bitboard occ = pos.occupied ^ sqbb(from) ^ sqbb(to);
    int stm = color_of(pos.board[from]);
    Bitboard attackers = pos.attackers_to(to, occ);
    Bitboard diag = pos.pcs(WHITE, BISHOP) | pos.pcs(BLACK, BISHOP) | pos.pcs(WHITE, QUEEN) | pos.pcs(BLACK, QUEEN);
    Bitboard orth = pos.pcs(WHITE, ROOK) | pos.pcs(BLACK, ROOK) | pos.pcs(WHITE, QUEEN) | pos.pcs(BLACK, QUEEN);
    int res = 1;
    while (true) {
        stm ^= 1;
        attackers &= occ;
        Bitboard stmAtt = attackers & pos.byColor[stm];
        if (!stmAtt) break;
        res ^= 1;
        Bitboard b;
        if ((b = stmAtt & pos.pcs(stm, PAWN))) {
            if ((swap = SeeVal[PAWN] - swap) < res) break;
            occ ^= b & (~b + 1);
            attackers |= bb::bishop_attacks(to, occ) & diag;
        } else if ((b = stmAtt & pos.pcs(stm, KNIGHT))) {
            if ((swap = SeeVal[KNIGHT] - swap) < res) break;
            occ ^= b & (~b + 1);
        } else if ((b = stmAtt & pos.pcs(stm, BISHOP))) {
            if ((swap = SeeVal[BISHOP] - swap) < res) break;
            occ ^= b & (~b + 1);
            attackers |= bb::bishop_attacks(to, occ) & diag;
        } else if ((b = stmAtt & pos.pcs(stm, ROOK))) {
            if ((swap = SeeVal[ROOK] - swap) < res) break;
            occ ^= b & (~b + 1);
            attackers |= bb::rook_attacks(to, occ) & orth;
        } else if ((b = stmAtt & pos.pcs(stm, QUEEN))) {
            if ((swap = SeeVal[QUEEN] - swap) < res) break;
            occ ^= b & (~b + 1);
            attackers |= (bb::bishop_attacks(to, occ) & diag) | (bb::rook_attacks(to, occ) & orth);
        } else {
            return (attackers & ~pos.byColor[stm]) ? (res ^ 1) != 0 : res != 0;
        }
    }
    return res != 0;
}

static bool is_draw(const Worker& w, int ply) {
    const Position& p = w.pos[ply];
    if (p.halfmove >= 100) return true;
    int idx = w.gameLen - 1 + ply;
    int lim = std::min(std::min(p.halfmove, w.pfn[ply]), idx);
    for (int i = 4; i <= lim; i += 2)
        if (w.keys[idx - i] == p.key) return true;
    if (!(p.pcs(WHITE, PAWN) | p.pcs(BLACK, PAWN) | p.pcs(WHITE, ROOK) | p.pcs(BLACK, ROOK) |
          p.pcs(WHITE, QUEEN) | p.pcs(BLACK, QUEEN)) && popcount(p.occupied) <= 3)
        return true;
    return false;
}

// lazy mode: bring acc[ply] up to date from the nearest computed ancestor (null move = copy)
static void ensure_acc(Worker& w, int ply) {
    if (w.accOk[ply]) return;
    ensure_acc(w, ply - 1);
    Move m = w.currentMove[ply - 1];
    if (m) nnue::update(w.pos[ply - 1], m, w.pos[ply], w.acc[ply - 1], w.acc[ply]);
    else w.acc[ply] = w.acc[ply - 1];
    w.accOk[ply] = true;
}

static bool smallLoaded = false;
static void ensure_accS(Worker& w, int ply) {
    if (w.accSOk[ply]) return;
    if (ply == 0) nnue_small::refresh_all(w.pos[0], w.accS[0]);
    else {
        ensure_accS(w, ply - 1);
        Move m = w.currentMove[ply - 1];
        if (m) nnue_small::update(w.pos[ply - 1], m, w.pos[ply], w.accS[ply - 1], w.accS[ply]);
        else w.accS[ply] = w.accS[ply - 1];
    }
    w.accSOk[ply] = true;
}
static inline int material(const Position& p) {  // white minus black, simple piece values
    static const int V[5] = {100, 320, 330, 500, 950};
    int m = 0;
    for (int t = 0; t < 5; t++) m += V[t] * (popcount(p.pcs(WHITE, t)) - popcount(p.pcs(BLACK, t)));
    return m;
}

static inline int evaluate(Worker& w, int ply, bool withSigma = false) {
    const Position& p = w.pos[ply];
    int v;
    if (UseSmallNet && smallLoaded && std::abs(material(p)) > SmallNetThreshold) {
        ensure_accS(w, ply);
        v = nnue_small::evaluate(w.accS[ply], p.stm, popcount(p.occupied));
    } else if (withSigma) {
        ensure_acc(w, ply);
        v = nnue::evaluate_sigma(w.acc[ply], p.stm, popcount(p.occupied), w.sigma[ply]);
        w.sigOk[ply] = true;
    } else {
        ensure_acc(w, ply);
        v = nnue::evaluate(w.acc[ply], p.stm, popcount(p.occupied));
    }
    v = v * (200 - p.halfmove) / 200;
    return std::clamp(v, -MATE_BOUND + 1, MATE_BOUND - 1);
}

// sigma of the node at ply (computed with its eval, or on demand when the eval came from the TT)
static inline int node_sigma(Worker& w, int ply) {
    if (!w.sigOk[ply]) {
        ensure_acc(w, ply);
        const Position& p = w.pos[ply];
        nnue::evaluate_sigma(w.acc[ply], p.stm, popcount(p.occupied), w.sigma[ply]);
        w.sigOk[ply] = true;
    }
    return w.sigma[ply];
}
// margin multiplier in percent: 100 at sigma == SigRef
static inline int sig_pct(Worker& w, int ply) {
    if (!UseSigma || !nnue::sigma_loaded()) return 100;
    return std::clamp(100 - SigMix + SigMix * node_sigma(w, ply) / SigRef, 50, 200);
}

static inline bool is_quiet(Move m) { return !is_capture(m) && !is_promo(m); }

template <typename T>
static inline void update_hist(T& h, int bonus) { h += bonus - h * std::abs(bonus) / 16384; }

static inline uint32_t pawn_idx(const Position& p) {
    uint64_t k = p.pcs(WHITE, PAWN) * 0x9E3779B97F4A7C15ULL ^ (p.pcs(BLACK, PAWN) + 0x632BE59BD9B4E019ULL) * 0xC2B2AE3D27D4EB4FULL;
    return (uint32_t)(k >> 50);  // 14 bits
}
static inline uint32_t np_idx(const Position& p, int c) {
    uint64_t k = 0x9E3779B97F4A7C15ULL;
    for (int t = KNIGHT; t <= KING; t++) k = (k ^ p.pcs(c, t)) * 0xC2B2AE3D27D4EB4FULL + t;
    return (uint32_t)(k >> 50);
}
static inline int corr2_total(const Worker& w, const Position& p) {
    int s = w.corrP[p.stm][pawn_idx(p)] * Corr2P + (w.corrN[p.stm][WHITE][np_idx(p, WHITE)] + w.corrN[p.stm][BLACK][np_idx(p, BLACK)]) * Corr2N;
    return s / (16 * 100);
}
static inline void corr2_update(Worker& w, const Position& p, int diff, int depth) {
    int wt = std::min(depth + 1, 16), t = std::clamp(diff * 16, -8192, 8192);
    auto upd = [&](int16_t& c) { c = (int16_t)std::clamp((c * (Corr2W - wt) + t * wt) / Corr2W, -4096, 4096); };
    upd(w.corrP[p.stm][pawn_idx(p)]);
    upd(w.corrN[p.stm][WHITE][np_idx(p, WHITE)]);
    upd(w.corrN[p.stm][BLACK][np_idx(p, BLACK)]);
}
// corr holds the average (search result - static eval) of this pawn structure in 1/16 cp
static inline int corr5_total(const Worker& cw, const Position& p, int ply);
static inline int corrected(const Worker& w, const Position& p, int raw, int ply) {
    if (UseCorr5) return std::clamp(raw + corr5_total(w, p, ply), -MATE_BOUND + 1, MATE_BOUND - 1);
    if (UseCorr2) return std::clamp(raw + corr2_total(w, p), -MATE_BOUND + 1, MATE_BOUND - 1);
    if (!UseCorrHist) return raw;
    return std::clamp(raw + w.corr[p.stm][pawn_idx(p)] / 16, -MATE_BOUND + 1, MATE_BOUND - 1);
}
static inline int cont_idx_ok(const Worker& w, int ply, int back) {
    return ply >= back && w.currentMove[ply - back] && w.movedPiece[ply - back] < 12;
}
static inline int phist_idx(const Position& p) { return (int)(p.pawnKey >> 54); }  // 10 bits
// squares attacked by colour c
static inline Bitboard attacks_by(const Position& p, int c) {
    constexpr Bitboard FA = 0x0101010101010101ULL, FH = FA << 7;
    Bitboard pw = p.pcs(c, PAWN), a;
    a = c == WHITE ? ((pw << 7) & ~FH) | ((pw << 9) & ~FA) : ((pw >> 9) & ~FH) | ((pw >> 7) & ~FA);
    for (Bitboard b = p.pcs(c, KNIGHT); b;) a |= bb::KnightAttacks[pop_lsb(b)];
    for (Bitboard b = p.pcs(c, BISHOP) | p.pcs(c, QUEEN); b;) a |= bb::bishop_attacks(pop_lsb(b), p.occupied);
    for (Bitboard b = p.pcs(c, ROOK) | p.pcs(c, QUEEN); b;) a |= bb::rook_attacks(pop_lsb(b), p.occupied);
    return a | bb::KingAttacks[p.king_sq(c)];
}
// main history entry of a quiet move; thr = squares the opponent attacks (UseThreatHist)
static inline int& main_hist(Worker& w, int c, Bitboard thr, Move m) {
    int f = from_sq(m), t = to_sq(m);
    if (UseThreatHist) return w.thist[c][(thr >> f) & 1][(thr >> t) & 1][f][t];
    return w.hist[c][f][t];
}
static const int contBacks[4] = {1, 2, 4, 6};
static inline int n_cont() { return UseCont6 ? 4 : 2 + UseCont4; }
static inline int cont_w(int i) {
    if (!UseCont6) return 100;
    return i == 0 ? Cont1W : i == 1 ? Cont2W : i == 2 ? Cont4W : Cont6W;
}
static inline int hist_wsum() {  // total % weight of the tables summed in quiet_score
    int t = 100 + (UsePawnHist ? PawnHistW : 0);
    if (UseContHist) for (int i = 0; i < n_cont(); i++) t += cont_w(i);
    return std::max(t, 1);
}
static inline int quiet_score(const Worker& w, const Position& pos, int ply, Move m) {
    int s = main_hist(const_cast<Worker&>(w), pos.stm, w.thr[ply], m);
    if (UsePawnHist) s += w.phist[phist_idx(pos)][pos.board[from_sq(m)]][to_sq(m)] * PawnHistW / 100;
    if (UseContHist) {
        int pc = pos.board[from_sq(m)], to = to_sq(m);
        for (int i = 0; i < n_cont(); i++) {
            int back = contBacks[i];
            if (cont_idx_ok(w, ply, back))
                s += w.cont[w.movedPiece[ply - back]][to_sq(w.currentMove[ply - back])][pc][to] * cont_w(i) / 100;
        }
    }
    return s;
}
// ---- correction history v5 ----
static inline uint32_t key14(uint64_t k) { return (uint32_t)((k * 0x9E3779B97F4A7C15ULL) >> 50); }
static inline uint64_t trans_key(const Worker& w, int ply) {  // what the last move changed (0 at the game start)
    int idx = w.gameLen - 1 + ply;
    return idx >= 1 ? w.keys[idx] ^ w.keys[idx - 1] : 0;
}
static inline int16_t* c5_cont(Worker& w, int ply) {
    if (!cont_idx_ok(w, ply, 1) || !cont_idx_ok(w, ply, 2)) return nullptr;
    return &w.c5C[w.movedPiece[ply - 2]][to_sq(w.currentMove[ply - 2])][w.movedPiece[ply - 1]][to_sq(w.currentMove[ply - 1])];
}
static inline int corr5_total(const Worker& cw, const Position& p, int ply) {
    Worker& w = const_cast<Worker&>(cw);
    int c = p.stm;
    int s = w.c5P[c][key14(p.pawnKey)] * C5P + (w.c5W[c][key14(p.npKey[WHITE])] + w.c5B[c][key14(p.npKey[BLACK])]) * C5N +
            w.c5T[c][key14(trans_key(w, ply))] * C5T;
    if (int16_t* e = c5_cont(w, ply)) s += *e * C5C;
    return s / (16 * 100);
}
static inline void corr5_update(Worker& w, const Position& p, int ply, int diff, int depth) {
    int wt = std::min(depth + 1, 16), t = std::clamp(diff * 16, -8192, 8192), c = p.stm;
    auto upd = [&](int16_t& e) { e = (int16_t)std::clamp((e * (C5W - wt) + t * wt) / C5W, -4096, 4096); };
    upd(w.c5P[c][key14(p.pawnKey)]);
    upd(w.c5W[c][key14(p.npKey[WHITE])]);
    upd(w.c5B[c][key14(p.npKey[BLACK])]);
    upd(w.c5T[c][key14(trans_key(w, ply))]);
    if (int16_t* e = c5_cont(w, ply)) upd(*e);
}
// ---- cuckoo tables: Zobrist difference of every reversible (non-pawn) move on an empty board ----
static uint64_t cuckooKey[8192];
static Move cuckooMove[8192];
static inline int ck1(uint64_t k) { return (int)(k & 0x1FFF); }
static inline int ck2(uint64_t k) { return (int)((k >> 16) & 0x1FFF); }
static void init_cuckoo() {
    std::memset(cuckooKey, 0, sizeof(cuckooKey));
    std::memset(cuckooMove, 0, sizeof(cuckooMove));
    int count = 0;
    for (int pc = 0; pc < 12; pc++) {
        int t = type_of(pc);
        if (t == PAWN) continue;
        for (int a = 0; a < 64; a++)
            for (int b = a + 1; b < 64; b++) {
                Bitboard att = t == KNIGHT ? bb::KnightAttacks[a] : t == BISHOP ? bb::bishop_attacks(a, 0)
                             : t == ROOK ? bb::rook_attacks(a, 0) : t == QUEEN ? bb::queen_attacks(a, 0) : bb::KingAttacks[a];
                if (!(att & sqbb(b))) continue;
                uint64_t key = zob::Piece[pc][a] ^ zob::Piece[pc][b] ^ zob::Side;
                Move mv = make_move(a, b, 0);
                int i = ck1(key);
                while (true) {  // cuckoo insertion
                    std::swap(cuckooKey[i], key);
                    std::swap(cuckooMove[i], mv);
                    if (!mv) break;
                    i = i == ck1(key) ? ck2(key) : ck1(key);
                }
                count++;
            }
    }
    if (count != 3668) out_line("info string cuckoo table size %d (expected 3668)", count);
}
// some legal reversible move of the side to move reaches a position already on the line (any repetition = draw here)
static bool upcoming_repetition(const Worker& w, int ply) {
    const Position& p = w.pos[ply];
    int idx = w.gameLen - 1 + ply;
    int end = std::min(std::min(p.halfmove, w.pfn[ply]), idx);
    if (end < 3) return false;
    for (int i = 3; i <= end; i += 2) {
        uint64_t mk = p.key ^ w.keys[idx - i];
        int j = ck1(mk);
        if (cuckooKey[j] != mk) { j = ck2(mk); if (cuckooKey[j] != mk) continue; }
        Move mv = cuckooMove[j];
        int a = from_sq(mv), b = to_sq(mv);
        if (bb::Between[a][b] & p.occupied) continue;
        if (ply > i) return true;  // the repeated position is inside the search tree
        // before the root: the piece must belong to the side to move (otherwise it's the opponent's move)
        int sq = p.board[a] != NO_PIECE ? a : b;
        if (p.board[sq] != NO_PIECE && color_of(p.board[sq]) == p.stm) return true;
    }
    return false;
}
// ---- Syzygy (Fathom) ----
static constexpr int TB_WIN_V = MATE_BOUND - 1;  // below mate scores, above any eval
static std::atomic<uint64_t> tbHits{0};
static inline bool tb_ok(const Position& p) {
    return TB_LARGEST && !p.castling && popcount(p.occupied) <= (int)TB_LARGEST;
}
static inline unsigned tb_args_wdl(const Position& p) {
    return tb_probe_wdl(p.byColor[WHITE], p.byColor[BLACK], p.pcs(WHITE, KING) | p.pcs(BLACK, KING),
                        p.pcs(WHITE, QUEEN) | p.pcs(BLACK, QUEEN), p.pcs(WHITE, ROOK) | p.pcs(BLACK, ROOK),
                        p.pcs(WHITE, BISHOP) | p.pcs(BLACK, BISHOP), p.pcs(WHITE, KNIGHT) | p.pcs(BLACK, KNIGHT),
                        p.pcs(WHITE, PAWN) | p.pcs(BLACK, PAWN), 0, 0, p.ep < 0 ? 0 : p.ep, p.stm == WHITE);
}
// root: the DTZ-optimal move that keeps the tablebase result (0 if not a tablebase position)
static Move tb_root_move(const Position& p) {
    if (!tb_ok(p)) return 0;
    unsigned r = tb_probe_root(p.byColor[WHITE], p.byColor[BLACK], p.pcs(WHITE, KING) | p.pcs(BLACK, KING),
                               p.pcs(WHITE, QUEEN) | p.pcs(BLACK, QUEEN), p.pcs(WHITE, ROOK) | p.pcs(BLACK, ROOK),
                               p.pcs(WHITE, BISHOP) | p.pcs(BLACK, BISHOP), p.pcs(WHITE, KNIGHT) | p.pcs(BLACK, KNIGHT),
                               p.pcs(WHITE, PAWN) | p.pcs(BLACK, PAWN), p.halfmove, 0, p.ep < 0 ? 0 : p.ep,
                               p.stm == WHITE, nullptr);
    if (r == TB_RESULT_FAILED || r == TB_RESULT_CHECKMATE || r == TB_RESULT_STALEMATE) return 0;
    static const int promoMap[5] = {-1, QUEEN, ROOK, BISHOP, KNIGHT};
    int from = TB_GET_FROM(r), to = TB_GET_TO(r), pr = promoMap[TB_GET_PROMOTES(r)];
    MoveList legal;
    generate_legal(p, legal);
    for (int i = 0; i < legal.size; i++) {
        Move m = legal.moves[i];
        if (from_sq(m) == from && to_sq(m) == to && (is_promo(m) ? promo_type(m) == pr : pr < 0)) return m;
    }
    return 0;
}

static inline int victim_of(const Position& pos, Move m) {
    if (flags_of(m) == MF_EP) return PAWN;
    return pos.board[to_sq(m)] != NO_PIECE ? type_of(pos.board[to_sq(m)]) : 5;
}

// Stage-free picker: generate once, score, lazily selection-sort.
struct Picker {
    MoveList list;
    int scores[256];
    int idx = 0;
    Move ttMove;

    void init_main(const Worker& w, const Position& pos, int ply, Move tt) {
        ttMove = tt;
        generate(pos, list, GEN_ALL);
        Move k0 = w.killers[ply][0], k1 = w.killers[ply][1], cm = 0;
        if (ply > 0 && w.currentMove[ply - 1]) cm = w.counter[w.movedPiece[ply - 1]][to_sq(w.currentMove[ply - 1])];
        for (int i = 0; i < list.size; i++) {
            Move m = list.moves[i];
            int s;
            if (m == tt) s = 30000000;
            else if (!is_quiet(m)) {
                int victim = flags_of(m) == MF_EP ? PAWN : (pos.board[to_sq(m)] != NO_PIECE ? type_of(pos.board[to_sq(m)]) : -1);
                int mvv = (victim >= 0 ? SeeVal[victim] : 0) + (is_promo(m) && promo_type(m) == QUEEN ? 950 : 0);
                int lva = type_of(pos.board[from_sq(m)]);
                bool good = is_promo(m) ? promo_type(m) == QUEEN : see_ge(pos, m, 0);
                s = (good ? 20000000 : -10000000) + mvv * 16 - lva;
                if (UseCaptHist) s += w.capt[pos.board[from_sq(m)]][to_sq(m)][victim_of(pos, m)] / 32;
            } else if (m == k0) s = 10000000;
            else if (m == k1) s = 9000000;
            else if (m == cm) s = 8000000;
            else s = quiet_score(w, pos, ply, m);
            scores[i] = s;
        }
    }
    void init_q(const Worker& w, const Position& pos, Move tt, bool inCheck) {
        ttMove = tt;
        generate(pos, list, inCheck ? GEN_ALL : GEN_NOISY);
        int us = pos.stm;
        for (int i = 0; i < list.size; i++) {
            Move m = list.moves[i];
            int s;
            if (m == tt) s = 30000000;
            else if (!is_quiet(m)) {
                int victim = flags_of(m) == MF_EP ? PAWN : (pos.board[to_sq(m)] != NO_PIECE ? type_of(pos.board[to_sq(m)]) : -1);
                int mvv = (victim >= 0 ? SeeVal[victim] : 0) + (is_promo(m) && promo_type(m) == QUEEN ? 950 : 0);
                s = 20000000 + mvv * 16 - type_of(pos.board[from_sq(m)]);
            } else s = UseThreatHist ? main_hist(const_cast<Worker&>(w), us, attacks_by(pos, us ^ 1), m) : w.hist[us][from_sq(m)][to_sq(m)];
            scores[i] = s;
        }
    }
    Move next(bool skipQuiets) {
        while (idx < list.size) {
            int best = idx;
            for (int j = idx + 1; j < list.size; j++)
                if (scores[j] > scores[best]) best = j;
            std::swap(list.moves[idx], list.moves[best]);
            std::swap(scores[idx], scores[best]);
            Move m = list.moves[idx++];
            if (skipQuiets && is_quiet(m) && m != ttMove) continue;
            return m;
        }
        return 0;
    }
};

static inline void make_child(Worker& w, int ply, Move m) {
    Position& child = w.pos[ply + 1];
    child = w.pos[ply];
    child.do_move(m);
    tt_prefetch(child.key);
    if (UseLazyAcc) w.accOk[ply + 1] = false;
    else nnue::update(w.pos[ply], m, child, w.acc[ply], w.acc[ply + 1]), w.accOk[ply + 1] = true;
    w.accSOk[ply + 1] = false;
    w.currentMove[ply] = m;
    w.movedPiece[ply] = w.pos[ply].board[from_sq(m)];
    w.keys[w.gameLen + ply] = child.key;
    w.pfn[ply + 1] = w.pfn[ply] + 1;
    w.dext[ply + 1] = w.dext[ply];
    w.redIn[ply + 1] = 0;
    w.nodes.fetch_add(1, std::memory_order_relaxed);
}

// ======================= quiescence =======================
template <bool PV>
static int qsearch(Worker& w, int alpha, int beta, int ply) {
    const Position& pos = w.pos[ply];
    if (PV) { w.pvLen[ply] = 0; if (ply + 1 > w.selDepth) w.selDepth = ply + 1; }
    if (w.id == 0 && (++w.tick & 2047) == 0) check_time();
    if (stopFlag.load(std::memory_order_relaxed)) return 0;
    if (is_draw(w, ply)) return draw_value(ply);
    if (UseCuckoo && alpha < draw_value(ply) && upcoming_repetition(w, ply)) {
        alpha = draw_value(ply);
        if (alpha >= beta) return alpha;
    }
    bool inCheck = pos.checkers != 0;
    if (ply >= MAX_PLY - 1) return inCheck ? 0 : evaluate(w, ply);

    TTData tt;
    bool ttHit = tt_probe(pos.key, tt);
    int ttValue = ttHit ? value_from_tt(tt.score, ply) : VALUE_NONE;
    if (!PV && ttHit && ttValue != VALUE_NONE && (tt.bound & (ttValue >= beta ? BOUND_LOWER : BOUND_UPPER)))
        return ttValue;

    int bestValue, rawEval = VALUE_NONE, futilityBase = -VALUE_INF;
    if (inCheck) bestValue = -VALUE_INF;
    else {
        rawEval = (ttHit && tt.eval != VALUE_NONE) ? tt.eval : evaluate(w, ply);
        bestValue = corrected(w, pos, rawEval, ply);
        if (ttHit && ttValue != VALUE_NONE && (tt.bound & (ttValue > bestValue ? BOUND_LOWER : BOUND_UPPER)))
            bestValue = ttValue;
        if (bestValue >= beta) {
            if (UseQsBlend && std::abs(bestValue) < MATE_BOUND && std::abs(beta) < MATE_BOUND) bestValue = (bestValue + beta) / 2;
            if (!ttHit) tt_store(pos.key, 0, value_to_tt(bestValue, ply), rawEval, 0, BOUND_LOWER);
            return bestValue;
        }
        if (bestValue > alpha) alpha = bestValue;
        futilityBase = bestValue + QsFut;
    }

    Picker mp;
    mp.init_q(w, pos, ttHit ? tt.move : 0, inCheck);
    Move bestMove = 0;
    int moveCount = 0;
    Move m;
    while ((m = mp.next(false))) {
        if (!pos.is_legal(m)) continue;
        moveCount++;
        if (!inCheck) {
            if (!is_promo(m)) {
                int victim = flags_of(m) == MF_EP ? PAWN : type_of(pos.board[to_sq(m)]);
                int fv = futilityBase + SeeVal[victim];
                if (fv <= alpha) { bestValue = std::max(bestValue, fv); continue; }
            }
            if (!see_ge(pos, m, 0)) continue;
        }
        make_child(w, ply, m);
        int v = -qsearch<PV>(w, -beta, -alpha, ply + 1);
        if (stopFlag.load(std::memory_order_relaxed)) return 0;
        if (v > bestValue) {
            bestValue = v;
            if (v > alpha) {
                bestMove = m;
                if (PV) {
                    w.pv[ply][0] = m;
                    std::memcpy(&w.pv[ply][1], w.pv[ply + 1], sizeof(Move) * w.pvLen[ply + 1]);
                    w.pvLen[ply] = w.pvLen[ply + 1] + 1;
                }
                if (v >= beta) break;
                alpha = v;
            }
        }
    }
    if (inCheck && moveCount == 0) return -VALUE_MATE + ply;
    if (UseQsBlend && bestValue >= beta && std::abs(bestValue) < MATE_BOUND && std::abs(beta) < MATE_BOUND)
        bestValue = (bestValue + beta) / 2;
    tt_store(pos.key, bestMove, value_to_tt(bestValue, ply), rawEval, 0, bestValue >= beta ? BOUND_LOWER : BOUND_UPPER);
    return bestValue;
}

// ======================= main search =======================
template <bool PV>
static int negamax(Worker& w, int alpha, int beta, int depth, int ply, bool cutNode) {
    if (depth <= 0) return qsearch<PV>(w, alpha, beta, ply);
    const Position& pos = w.pos[ply];
    const bool root = ply == 0;
    if (PV) { w.pvLen[ply] = 0; if (ply + 1 > w.selDepth) w.selDepth = ply + 1; }
    if (w.id == 0 && (++w.tick & 2047) == 0) check_time();

    if (!root) {
        if (stopFlag.load(std::memory_order_relaxed)) return 0;
        if (is_draw(w, ply)) return draw_value(ply);
        if (UseCuckoo && alpha < draw_value(ply) && upcoming_repetition(w, ply)) {
            alpha = draw_value(ply);
            if (alpha >= beta) return alpha;
        }
        if (ply >= MAX_PLY - 1) return pos.checkers ? 0 : evaluate(w, ply);
        alpha = std::max(-VALUE_MATE + ply, alpha);
        beta = std::min(VALUE_MATE - ply - 1, beta);
        if (alpha >= beta) return alpha;
    }

    const bool inCheck = pos.checkers != 0;
    const int us = pos.stm, alphaOrig = alpha;
    TTData tt;
    bool ttHit = tt_probe(pos.key, tt);
    int ttValue = ttHit ? value_from_tt(tt.score, ply) : VALUE_NONE;
    Move ttMove = root ? w.rootMoves[w.pvIdx].move : (ttHit ? tt.move : 0);
    const Move excl = w.excluded[ply];
    const bool ttPv = UseTtPv && (PV || (ttHit && tt.pv));

    if (!PV && !excl && ttHit && tt.depth >= depth && ttValue != VALUE_NONE &&
        (tt.bound & (ttValue >= beta ? BOUND_LOWER : BOUND_UPPER))) {
        if (UseTtBlend && ttValue >= beta && std::abs(ttValue) < MATE_BOUND && std::abs(beta) < MATE_BOUND)
            return (ttValue * tt.depth + beta) / (tt.depth + 1);
        return ttValue;
    }

    // Syzygy WDL: exact result for <= TB_LARGEST pieces right after a capture/pawn move
    if (!root && !excl && pos.halfmove == 0 && depth >= SyzygyProbeDepth && tb_ok(pos)) {
        unsigned r = tb_args_wdl(pos);
        if (r != TB_RESULT_FAILED) {
            tbHits.fetch_add(1, std::memory_order_relaxed);
            int v = r == TB_WIN ? TB_WIN_V - ply : r == TB_LOSS ? -TB_WIN_V + ply : 0;  // cursed/blessed = draw
            int b = r == TB_WIN ? BOUND_LOWER : r == TB_LOSS ? BOUND_UPPER : BOUND_EXACT;
            if (b == BOUND_EXACT || (b == BOUND_LOWER ? v >= beta : v <= alpha)) {
                tt_store(pos.key, 0, v, VALUE_NONE, std::min(MAX_PLY - 1, depth + 6), b);
                return v;
            }
        }
    }

    int eval, rawEval;
    w.sigOk[ply] = false;
    if (inCheck) {
        rawEval = eval = w.staticEval[ply] = VALUE_NONE;
    } else {
        rawEval = (ttHit && tt.eval != VALUE_NONE) ? tt.eval : evaluate(w, ply, (UseSigma || UseSigLmr) && nnue::sigma_loaded());
        eval = w.staticEval[ply] = corrected(w, pos, rawEval, ply);
        if (ttHit && ttValue != VALUE_NONE && (tt.bound & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttValue;
    }
    bool improving = false;
    if (!inCheck && ply >= 2) {
        if (w.staticEval[ply - 2] != VALUE_NONE) improving = w.staticEval[ply] > w.staticEval[ply - 2];
        else if (ply >= 4 && w.staticEval[ply - 4] != VALUE_NONE) improving = w.staticEval[ply] > w.staticEval[ply - 4];
        else improving = true;
    }
    w.killers[ply + 1][0] = w.killers[ply + 1][1] = 0;
    if (UseThreatHist) w.thr[ply] = attacks_by(pos, us ^ 1);

    // hindsight: the parent reduced the move to here; judge the reduction by both static evals
    if (UseHindsight && !root && !inCheck && !excl && w.redIn[ply] > 0 && w.staticEval[ply - 1] != VALUE_NONE) {
        int sum = w.staticEval[ply] + w.staticEval[ply - 1];
        if (w.redIn[ply] >= HsRed && sum <= HsWorsen) depth++;
        else if (depth >= 2 && sum > HsMargin) depth--;
    }

    if (!PV && !inCheck && !excl) {
        // razoring: hopeless at low depth -> verify with quiescence only
        if (UseRazor && depth <= 3 && eval < alpha - RazorMargin - RazorMul * depth * depth) {
            int v = qsearch<false>(w, alpha, alpha + 1, ply);
            if (v <= alpha) return v;
        }
        // reverse futility: static eval is so far above beta that a shallow search won't bring it back
        if (depth < RfpDepth && eval - (RfpMargin - RfpImp * improving) * depth * sig_pct(w, ply) / 100 >= beta && eval < MATE_BOUND && beta > -MATE_BOUND)
            return UseRfpBlend ? (eval + beta) / 2 : eval;
        // null move: if passing still beats beta, a real move almost surely does too
        if (depth >= 3 && eval >= beta && w.staticEval[ply] >= beta - 20 * depth + 180 &&
            w.currentMove[ply - 1] != 0 && pos.has_non_pawn(us) && beta > -MATE_BOUND) {
            int R = NmpBase + depth / NmpDiv + std::min((eval - beta) / NmpEvalDiv, 3);
            Position& child = w.pos[ply + 1];
            child = pos;
            child.do_null();
            tt_prefetch(child.key);
            if (UseLazyAcc) w.accOk[ply + 1] = false;
            else w.acc[ply + 1] = w.acc[ply], w.accOk[ply + 1] = true;
            w.accSOk[ply + 1] = false;
            w.currentMove[ply] = 0;
            w.movedPiece[ply] = NO_PIECE;
            w.keys[w.gameLen + ply] = child.key;
            w.pfn[ply + 1] = 0;
            w.dext[ply + 1] = w.dext[ply];
            w.nodes.fetch_add(1, std::memory_order_relaxed);
            int v = -negamax<false>(w, -beta, -beta + 1, depth - R, ply + 1, !cutNode);
            if (stopFlag.load(std::memory_order_relaxed)) return 0;
            if (v >= beta) return v >= MATE_BOUND ? beta : v;
        }
        // ProbCut: a good capture that beats beta by a margin at reduced depth almost surely beats beta
        if (UseProbCut && depth >= 5 && std::abs(beta) < MATE_BOUND) {
            int pcBeta = beta + ProbCutMargin;
            if (!(ttHit && tt.depth >= depth - 3 && ttValue != VALUE_NONE && ttValue < pcBeta)) {
                Picker pc;
                pc.init_q(w, pos, ttMove && !is_quiet(ttMove) ? ttMove : 0, false);
                Move m;
                while ((m = pc.next(false))) {
                    if (is_quiet(m) || !pos.is_legal(m) || !see_ge(pos, m, pcBeta - w.staticEval[ply])) continue;
                    make_child(w, ply, m);
                    int v = -qsearch<false>(w, -pcBeta, -pcBeta + 1, ply + 1);
                    if (v >= pcBeta) v = -negamax<false>(w, -pcBeta, -pcBeta + 1, depth - 4, ply + 1, !cutNode);
                    if (stopFlag.load(std::memory_order_relaxed)) return 0;
                    if (v >= pcBeta) {
                        tt_store(pos.key, m, value_to_tt(v, ply), rawEval, depth - 3, BOUND_LOWER);
                        return v;
                    }
                }
            }
        }
    }

    // internal iterative reduction: no TT move means ordering is poor here, so spend less
    if (!root && !excl && depth >= 4 && !ttMove) depth--;

    Picker mp;
    mp.init_main(w, pos, ply, ttMove);
    int bestValue = -VALUE_INF, moveCount = 0;
    Move bestMove = 0;
    Move quiets[64], caps[32];
    int nq = 0, nc = 0;
    bool skipQuiets = false;
    Move m;

    while ((m = mp.next(skipQuiets))) {
        if (m == excl || !pos.is_legal(m)) continue;
        if (root) {
            bool inRange = false;
            for (size_t i = w.pvIdx; i < w.rootMoves.size(); i++)
                if (w.rootMoves[i].move == m) { inRange = true; break; }
            if (!inRange) continue;
        }
        moveCount++;
        const bool quiet = is_quiet(m);
        const int h = quiet ? quiet_score(w, pos, ply, m) : 0;
        int newDepth = depth - 1;
        const int r0 = LMR[std::min(depth, 63)][std::min(moveCount, 63)];

        if (!root && bestValue > -MATE_BOUND && pos.has_non_pawn(us)) {
            int lmrDepth = std::max(newDepth - r0, 0);
            if (quiet) {
                if (depth <= 8 && moveCount >= (LmpBase + depth * depth) / (2 - improving)) { skipQuiets = true; continue; }
                if (!inCheck && lmrDepth < 7 && w.staticEval[ply] + (FutBase + FutMul * lmrDepth) * sig_pct(w, ply) / 100 <= alpha) continue;
                if (UseHistPrune && lmrDepth < HpDepth && h < -HpMul * depth) continue;
                if (lmrDepth < 8 && !see_ge(pos, m, -SeeQuiet * lmrDepth * lmrDepth)) continue;
            } else {
                if (UseCapFut && !inCheck && lmrDepth < 7 && !is_promo(m) &&
                    w.staticEval[ply] + CapFutBase + CapFutMul * lmrDepth + SeeVal[victim_of(pos, m) == 5 ? 0 : victim_of(pos, m)] <= alpha)
                    continue;
                if (depth < 7 && !see_ge(pos, m, -SeeNoisy * depth)) continue;
            }
        }

        // singular extension: if every other move fails well below the TT score, the TT move is forced
        int ext = 0;
        if (UseSingular && !root && m == ttMove && !excl && depth >= SingDepth && ttHit && tt.depth >= depth - 3 &&
            (tt.bound & BOUND_LOWER) && ttValue != VALUE_NONE && std::abs(ttValue) < MATE_BOUND && ply < 2 * w.rootDepth) {
            int sBeta = ttValue - SingMargin * depth;
            w.excluded[ply] = m;
            int sv = negamax<false>(w, sBeta - 1, sBeta, (depth - 1) / 2, ply, cutNode);
            w.excluded[ply] = 0;
            if (stopFlag.load(std::memory_order_relaxed)) return 0;
            if (sv < sBeta) ext = UseDblExt && !PV && sv < sBeta - DblMargin && w.dext[ply] < DblLimit ? 2 : 1;
            else if (sBeta >= beta) return sBeta;  // multicut: several moves beat beta
            else if (ttValue >= beta) ext = -1;
        }
        newDepth += ext;

        const uint64_t nodesBefore = root ? w.nodes.load(std::memory_order_relaxed) : 0;
        make_child(w, ply, m);
        if (ext >= 2) w.dext[ply + 1]++;
        const bool givesCheck = w.pos[ply + 1].checkers != 0;
        if (givesCheck && ply < 2 * w.rootDepth && ext <= 0) newDepth++;

        int v = 0;
        if (depth >= 2 && moveCount > 1 + root && (quiet || !PV)) {
            int r = r0;
            if (PV) r--;
            if (cutNode) r++;
            if (!improving) r++;
            if (m == w.killers[ply][0] || m == w.killers[ply][1]) r--;
            if (givesCheck) r--;
            if (!quiet) r--;
            if (ttPv && !PV) r--;
            if (UseLmrTtCap && quiet && ttMove && !is_quiet(ttMove)) r++;
            if (UseSigLmr && !inCheck && nnue::sigma_loaded()) {
                int sg = node_sigma(w, ply);
                r += (sg < SigLo) - (sg > SigHi);
            }
            r -= (UseHistNorm ? h * 300 / hist_wsum() : h) / HistDiv;
            int d = std::clamp(newDepth - r, 1, newDepth);
            w.redIn[ply + 1] = newDepth - d;
            v = -negamax<false>(w, -(alpha + 1), -alpha, d, ply + 1, true);
            w.redIn[ply + 1] = 0;
            if (v > alpha && d < newDepth) {
                if (UseDeeper) newDepth += (v > bestValue + DeeperMargin + 2 * newDepth) - (v < bestValue + newDepth);
                if (d < newDepth) v = -negamax<false>(w, -(alpha + 1), -alpha, newDepth, ply + 1, !cutNode);
            }
        } else if (!PV || moveCount > 1) {
            v = -negamax<false>(w, -(alpha + 1), -alpha, newDepth, ply + 1, !cutNode);
        }
        if (PV && (moveCount == 1 || v > alpha))
            v = -negamax<true>(w, -beta, -alpha, newDepth, ply + 1, false);

        if (stopFlag.load(std::memory_order_relaxed)) return 0;

        if (root) {
            for (auto& rm : w.rootMoves) {
                if (rm.move != m) continue;
                rm.nodes += w.nodes.load(std::memory_order_relaxed) - nodesBefore;
                if (moveCount == 1 || v > alpha) {
                    rm.score = v;
                    rm.selDepth = w.selDepth;
                    rm.pv.assign(1, m);
                    rm.pv.insert(rm.pv.end(), w.pv[1], w.pv[1] + w.pvLen[1]);
                } else rm.score = -VALUE_INF;
                break;
            }
        }

        if (v > bestValue) {
            bestValue = v;
            if (v > alpha) {
                bestMove = m;
                if (PV) {
                    w.pv[ply][0] = m;
                    std::memcpy(&w.pv[ply][1], w.pv[ply + 1], sizeof(Move) * w.pvLen[ply + 1]);
                    w.pvLen[ply] = w.pvLen[ply + 1] + 1;
                }
                if (v >= beta) break;
                alpha = v;
            }
        }
        if (m != bestMove && quiet && nq < 64) quiets[nq++] = m;
        if (m != bestMove && !quiet && nc < 32) caps[nc++] = m;
    }

    if (moveCount == 0) return excl ? alpha : inCheck ? -VALUE_MATE + ply : 0;

    if (bestValue >= beta) {
        int bonus = std::min(16 * depth * depth + 32 * depth, 1200);
        if (is_quiet(bestMove)) {
            if (w.killers[ply][0] != bestMove) { w.killers[ply][1] = w.killers[ply][0]; w.killers[ply][0] = bestMove; }
            if (ply > 0 && w.currentMove[ply - 1])
                w.counter[w.movedPiece[ply - 1]][to_sq(w.currentMove[ply - 1])] = bestMove;
            update_hist(main_hist(w, us, w.thr[ply], bestMove), bonus);
            for (int i = 0; i < nq; i++) update_hist(main_hist(w, us, w.thr[ply], quiets[i]), -bonus);
            if (UsePawnHist) {
                auto& ph = w.phist[phist_idx(pos)];
                update_hist(ph[pos.board[from_sq(bestMove)]][to_sq(bestMove)], bonus);
                for (int i = 0; i < nq; i++) update_hist(ph[pos.board[from_sq(quiets[i])]][to_sq(quiets[i])], -bonus);
            }
            if (UseContHist)
                for (int bi = 0; bi < n_cont(); bi++) {
                    int back = contBacks[bi];
                    if (!cont_idx_ok(w, ply, back)) continue;
                    auto& tab = w.cont[w.movedPiece[ply - back]][to_sq(w.currentMove[ply - back])];
                    update_hist(tab[pos.board[from_sq(bestMove)]][to_sq(bestMove)], bonus);
                    for (int i = 0; i < nq; i++) update_hist(tab[pos.board[from_sq(quiets[i])]][to_sq(quiets[i])], -bonus);
                }
        } else if (UseCaptHist) {
            update_hist(w.capt[pos.board[from_sq(bestMove)]][to_sq(bestMove)][victim_of(pos, bestMove)], bonus);
        }
        if (UseCaptHist)
            for (int i = 0; i < nc; i++) update_hist(w.capt[pos.board[from_sq(caps[i])]][to_sq(caps[i])][victim_of(pos, caps[i])], -bonus);
    } else if (UsePriorBonus && !excl && bestValue <= alphaOrig && ply >= 1 && w.currentMove[ply - 1] &&
               is_quiet(w.currentMove[ply - 1]) && w.movedPiece[ply - 1] < 12) {
        // every reply failed low: the opponent's last quiet move was good, reward it from their side
        int bonus = std::min(16 * depth * depth + 32 * depth, 1200);
        Move pm = w.currentMove[ply - 1];
        update_hist(main_hist(w, us ^ 1, w.thr[ply - 1], pm), bonus);
        if (UseContHist && cont_idx_ok(w, ply - 1, 1))
            update_hist(w.cont[w.movedPiece[ply - 2]][to_sq(w.currentMove[ply - 2])][w.movedPiece[ply - 1]][to_sq(pm)], bonus);
    }

    if (UseFhBlend && !PV && !excl && bestValue >= beta && depth >= 2 && std::abs(bestValue) < MATE_BOUND && std::abs(beta) < MATE_BOUND)
        bestValue = (bestValue * depth + beta) / (depth + 1);
    int bound = bestValue >= beta ? BOUND_LOWER : (PV && bestMove ? BOUND_EXACT : BOUND_UPPER);
    // correction history: learn how far the static eval of this pawn structure is off
    if (UseCorrHist && !inCheck && !excl && (!bestMove || is_quiet(bestMove)) && std::abs(bestValue) < TB_WIN_V - MAX_PLY &&
        !(bound == BOUND_LOWER && bestValue <= w.staticEval[ply]) && !(bound == BOUND_UPPER && bestValue >= w.staticEval[ply])) {
        int16_t& c = w.corr[us][pawn_idx(pos)];
        int nv = c + (bestValue - w.staticEval[ply]) * 16 * std::min(depth + 1, 16) / CorrDiv;
        c = (int16_t)std::clamp(nv, -2048, 2048);
    }
    if (UseCorr2 && !inCheck && !excl && (!bestMove || is_quiet(bestMove)) && std::abs(bestValue) < TB_WIN_V - MAX_PLY &&
        !(bound == BOUND_LOWER && bestValue <= w.staticEval[ply]) && !(bound == BOUND_UPPER && bestValue >= w.staticEval[ply]))
        corr2_update(w, pos, bestValue - w.staticEval[ply], depth);
    if (UseCorr5 && !inCheck && !excl && (!bestMove || is_quiet(bestMove)) && std::abs(bestValue) < TB_WIN_V - MAX_PLY &&
        !(bound == BOUND_LOWER && bestValue <= w.staticEval[ply]) && !(bound == BOUND_UPPER && bestValue >= w.staticEval[ply]))
        corr5_update(w, pos, ply, bestValue - w.staticEval[ply], depth);
    if (!(root && w.pvIdx) && !excl)
        tt_store(pos.key, bestMove, value_to_tt(bestValue, ply), rawEval, depth, bound, ttPv);
    return bestValue;
}

// ======================= driver =======================
static void report(const Worker& w, int depth) {
    if (quietOutput) return;
    uint64_t nodes = total_nodes();
    int64_t ms = std::max<int64_t>(1, elapsed_ms());
    int hf = hashfull();
    for (int i = 0; i < w.multiPV; i++) {
        const RootMove& rm = w.rootMoves[i];
        int v = rm.score != -VALUE_INF ? rm.score : rm.prevScore;
        if (v == -VALUE_INF) continue;
        char sc[32];
        if (std::abs(v) >= MATE_BOUND) {
            int mateIn = v > 0 ? (VALUE_MATE - v + 1) / 2 : -(VALUE_MATE + v) / 2;
            snprintf(sc, sizeof sc, "mate %d", mateIn);
        } else snprintf(sc, sizeof sc, "cp %d", v);
        std::string pv;
        for (Move m : rm.pv) { pv += move_to_uci(m); pv += ' '; }
        out_line("info depth %d seldepth %d multipv %d score %s nodes %llu nps %llu hashfull %d tbhits %llu time %lld pv %s",
                 depth, rm.selDepth, i + 1, sc, (unsigned long long)nodes,
                 (unsigned long long)(nodes * 1000 / ms), hf, (unsigned long long)tbHits.load(), (long long)ms, pv.c_str());
    }
}

static void id_loop(Worker& w) {
    const bool mainW = w.id == 0;
    Move lastBest = 0;
    int stable = 0;
    w.multiPV = mainW ? std::min<int>(optMultiPV, (int)w.rootMoves.size()) : 1;
    int maxDepth = limits.depth > 0 ? std::min(limits.depth, MAX_PLY - 8) : MAX_PLY - 8;
    for (int depth = 1; depth <= maxDepth && !stopFlag; depth++) {
        // helper threads skip some depths so they aren't all duplicating the main thread's tree
        if (!mainW && depth > 2 && ((depth + w.id) % 3 == 0)) continue;
        w.rootDepth = depth;
        for (auto& rm : w.rootMoves) rm.prevScore = rm.score;
        for (w.pvIdx = 0; w.pvIdx < w.multiPV && !stopFlag; w.pvIdx++) {
            w.selDepth = 0;
            int prev = w.rootMoves[w.pvIdx].prevScore;
            int delta = AspDelta, alpha = -VALUE_INF, beta = VALUE_INF, fh = 0;
            if (depth >= 4 && prev != -VALUE_INF && std::abs(prev) < MATE_BOUND) {
                alpha = std::max(prev - delta, -VALUE_INF);
                beta = std::min(prev + delta, VALUE_INF);
            }
            while (true) {
                int score = negamax<true>(w, alpha, beta, std::max(1, depth - fh), 0, false);
                std::stable_sort(w.rootMoves.begin() + w.pvIdx, w.rootMoves.end(),
                                 [](const RootMove& a, const RootMove& b) { return a.score > b.score; });
                if (stopFlag) break;
                if (score <= alpha) { beta = (alpha + beta) / 2; alpha = std::max(score - delta, -VALUE_INF); fh = 0; }
                else if (score >= beta) { beta = std::min(score + delta, VALUE_INF); if (UseAspFH && std::abs(score) < MATE_BOUND) fh++; }
                else break;
                delta += delta / 2;
                if (delta > 800) { alpha = -VALUE_INF; beta = VALUE_INF; }
            }
            std::stable_sort(w.rootMoves.begin(), w.rootMoves.begin() + w.pvIdx + 1,
                             [](const RootMove& a, const RootMove& b) { return a.score > b.score; });
        }
        if (stopFlag) break;
        w.completedDepth = depth;
        if (mainW) report(w, depth);
        // soft limit: don't start another iteration once past it; stretched while the best move is unstable
        if (mainW && softMs && !pondering.load(std::memory_order_relaxed)) {
            Move bm = w.rootMoves[0].move;
            stable = bm == lastBest ? stable + 1 : 0;
            lastBest = bm;
            double f = 1.3 - 0.05 * std::min(stable, 8);
            if (UseNodeTM && depth >= 6) {
                double frac = (double)w.rootMoves[0].nodes / std::max<uint64_t>(1, w.nodes.load(std::memory_order_relaxed));
                f *= (NtmBase / 100.0 - frac) * NtmMul / 100.0;
            }
            if (elapsed_ms() > softMs * f) stopFlag = true;
        }
    }
}

static void clear_worker(Worker& w) {
    std::memset(w.hist, 0, sizeof(w.hist));
    std::memset(w.counter, 0, sizeof(w.counter));
    std::memset(w.corr, 0, sizeof(w.corr));
    std::memset(w.corrP, 0, sizeof(w.corrP));
    std::memset(w.corrN, 0, sizeof(w.corrN));
    std::memset(w.phist, 0, sizeof(w.phist));
    std::memset(w.cont, 0, sizeof(w.cont));
    std::memset(w.capt, 0, sizeof(w.capt));
    std::memset(w.thist, 0, sizeof(w.thist));
    std::memset(w.thr, 0, sizeof(w.thr));
    std::memset(w.redIn, 0, sizeof(w.redIn));
    std::memset(w.c5P, 0, sizeof(w.c5P));
    std::memset(w.c5W, 0, sizeof(w.c5W));
    std::memset(w.c5B, 0, sizeof(w.c5B));
    std::memset(w.c5T, 0, sizeof(w.c5T));
    std::memset(w.c5C, 0, sizeof(w.c5C));
    std::memset(w.excluded, 0, sizeof(w.excluded));
}

namespace search {

void init() {
    init_cuckoo();
    set_hash_mb(256);
    set_threads(1);
}

void set_hash_mb(size_t mb) {
    wait();
    if (ttTable) aligned_free64(ttTable);
    ttClusters = std::max<uint64_t>(1, (uint64_t)mb * 1024 * 1024 / sizeof(Cluster));
    ttTable = (Cluster*)aligned_alloc64(ttClusters * sizeof(Cluster));
    clear_hash();
}

void clear_hash() {
    std::memset((void*)ttTable, 0, ttClusters * sizeof(Cluster));
    for (auto* w : workers) clear_worker(*w);
}

void set_threads(int n) {
    wait();
    for (auto* w : workers) delete w;
    workers.clear();
    optThreads = std::max(1, n);
    for (int i = 0; i < optThreads; i++) {
        Worker* w = new Worker();
        w->id = i;
        clear_worker(*w);
        workers.push_back(w);
    }
}
void set_multipv(int n) { optMultiPV = std::max(1, n); }
bool set_small_net(const std::string& path) {
    wait();
    smallLoaded = nnue_small::load(path);
    return smallLoaded;
}
int set_syzygy(const std::string& path) {
    wait();
    tb_init(path.c_str());
    return (int)TB_LARGEST;
}
int threads() { return optThreads; }
int multipv() { return optMultiPV; }

void stop() { stopFlag = true; }
// the opponent played the expected move: the ponder search becomes a normal timed search whose clock
// started at "go ponder", so time spent pondering counts as already thought (answers fast, saves clock)
void ponderhit() { pondering = false; }
void wait() { if (mainThread.joinable()) mainThread.join(); }

void start(const Position& root, const std::vector<uint64_t>& history, const Limits& lim) {
    stop();
    wait();
    stopFlag = false;
    limits = lim;
    startTime = std::chrono::steady_clock::now();
    pondering = lim.ponder;
    for (int d = 0; d < 64; d++)
        for (int m = 0; m < 64; m++)
            LMR[d][m] = (d == 0 || m == 0) ? 0 : (int)(LmrBase / 100.0 + std::log(d) * std::log(m) / (LmrDiv / 100.0));
    softMs = 0;
    if (UseTM && lim.fromClock) {
        int64_t soft = lim.time / TmSoftDiv + lim.inc * TmIncPct / 100;
        int64_t hard = std::max<int64_t>(10, std::min<int64_t>(soft * TmHardMul, lim.time - 50));
        softMs = std::max<int64_t>(5, std::min(soft, hard));
        limits.movetime = hard;
    }
    drawScore = 0;
    if (UseFlag && lim.fromClock && lim.oppTime >= 0 && lim.oppTime < FlagOppMs && lim.time * 100 >= lim.oppTime * FlagRatio
        && !(FlagNoInc && (lim.inc > 0 || lim.oppInc > 0))) {
        drawScore = FlagContempt;
        if (!quietOutput) out_line("info string flag mode: draw = -%d cp (clock %lld vs %lld ms)", (int)FlagContempt, (long long)lim.time, (long long)lim.oppTime);
        if (softMs) { softMs = std::max<int64_t>(5, softMs * FlagTimePct / 100); limits.movetime = std::max<int64_t>(10, limits.movetime * FlagTimePct / 100); }
    }
    ttGen = (ttGen + 1) & 63;

    MoveList legal;
    generate_legal(root, legal);
    if (Move tbm = tb_root_move(root)) { legal.size = 1; legal.moves[0] = tbm; }
    tbHits = 0;
    for (auto* w : workers) {
        w->pos[0] = root;
        nnue::refresh_all(root, w->acc[0]);
        w->accOk[0] = true;
        w->accSOk[0] = false;
        w->keys = history;
        w->gameLen = (int)history.size();
        w->keys.resize(w->gameLen + MAX_PLY + 8);
        w->pfn[0] = root.halfmove;
        w->nodes = 0;
        w->completedDepth = 0;
        w->currentMove[0] = 0;
        w->dext[0] = 0;
        std::memset(w->killers, 0, sizeof(w->killers));
        for (auto& row : w->hist) for (auto& col : row) for (int& v : col) v /= 2;
        if (UseThreatHist) for (int& v : *reinterpret_cast<int(*)[sizeof(w->thist) / sizeof(int)]>(&w->thist)) v /= 2;
        w->redIn[0] = 0;
        w->rootMoves.clear();
        for (int i = 0; i < legal.size; i++) { RootMove rm; rm.move = legal.moves[i]; w->rootMoves.push_back(rm); }
    }

    mainThread = std::thread([]() {
        Worker& mw = *workers[0];
        if (!quietOutput) out_line("info string multipv_count %d", std::min<int>(optMultiPV, (int)mw.rootMoves.size()));
        if (mw.rootMoves.empty()) {
            if (!quietOutput) out_line("info depth 0 score %s", mw.pos[0].checkers ? "mate 0" : "cp 0");
            while (limits.infinite && !stopFlag) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if (!quietOutput) out_line("bestmove 0000");
            return;
        }
        std::vector<std::thread> helpers;
        for (size_t i = 1; i < workers.size(); i++) helpers.emplace_back([i]() { id_loop(*workers[i]); });
        id_loop(mw);
        while ((limits.infinite || pondering) && !stopFlag) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        stopFlag = true;
        for (auto& t : helpers) t.join();
        const RootMove& best = mw.rootMoves[0];
        if (!quietOutput) {
            if (best.pv.size() > 1)
                out_line("bestmove %s ponder %s", move_to_uci(best.move).c_str(), move_to_uci(best.pv[1]).c_str());
            else
                out_line("bestmove %s", move_to_uci(best.move).c_str());
        }
    });
}

void search_nodes(const Position& root, const std::vector<uint64_t>& history, uint64_t nodes, Move& best, int& score) {
    quietOutput = true;
    Limits lim;
    lim.nodes = nodes;
    start(root, history, lim);
    wait();
    quietOutput = false;
    const RootMove& rm = workers[0]->rootMoves.empty() ? RootMove() : workers[0]->rootMoves[0];
    best = rm.move;
    score = rm.score != -VALUE_INF ? rm.score : rm.prevScore;
}

// debug: does the side to move at `root` have a reversible move back into the game history?
bool cuckoo_check(const Position& root, const std::vector<uint64_t>& history) {
    auto w = std::make_unique<Worker>();
    w->pos[0] = root;
    w->keys = history;
    w->gameLen = (int)history.size();
    w->keys.resize(w->gameLen + MAX_PLY + 8);
    w->pfn[0] = root.halfmove;
    return upcoming_repetition(*w, 0);
}
uint64_t bench_one(const Position& root, int depth, bool quiet) {
    quietOutput = quiet;
    Limits lim;
    lim.depth = depth;
    start(root, std::vector<uint64_t>{root.key}, lim);
    wait();
    quietOutput = false;
    return total_nodes();
}

}  // namespace search
