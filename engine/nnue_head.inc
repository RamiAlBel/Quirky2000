// ---- LNN2 / LNN3: head read from the net file (included by nnue.cpp inside namespace nnue) ----
//   ft act  0 crelu  clamp(a,0,1)            -> uint8 [0,127] per accumulator lane
//           1 screlu clamp(a,0,1)^2          -> uint8 [0,127]
//           2 pair   clamp(a_j)*clamp(a_j+ACC/2), j < ACC/2  (L1 input halves)
//   L1:     uint8 x int8 (per-row scale) -> int32, one weight set per output bucket; h1 multiple of 4
//           (the float layers after it run on h1 padded to 8 with zero weights)
//   hidden  0 crelu, 1 screlu, 2 dual ([crelu, crelu^2]); L2 optional (h2 = 0); all float FMA
// LNN2 files load as h1 16, h2 32, 1 bucket, crelu/crelu, cp scale 400.
constexpr int MAX_H1 = 32, MAX_H2 = 64, MAX_NB = 8, MAX_L1IN = 2 * NNUE_ACC;

static int H1, H1p, H2, NB, FtAct, HidAct, L1In, Mult;  // H1p = H1 rounded up to 8 (float part is padded)
static float CpScale;
alignas(64) static int8_t L1W[MAX_NB][MAX_H1][MAX_L1IN];
// sparse-L1 layout of the same weights: [bucket][input group of 4][H1p outputs][4 bytes], so one non-zero
// group of 4 inputs times one 32-byte load covers 8 outputs (Stockfish-style "skip the zero inputs")
alignas(64) static int8_t L1WS[MAX_NB * (MAX_L1IN / 4) * MAX_H1 * 4];
alignas(16) static uint16_t NzLut[256][8];  // bit mask -> positions of its set bits
static std::vector<float> L1Wf;                          // float reference [nb][h1][l1in]
static float L1Deq[MAX_NB][MAX_H1], L1B[MAX_NB][MAX_H1];
alignas(32) static float L2WT[MAX_NB][2 * MAX_H1][MAX_H2];  // transposed: [input][output]
alignas(32) static float L2B[MAX_NB][MAX_H2];
alignas(32) static float OutW[MAX_NB][2 * MAX_H2];
static float OutB[MAX_NB];
static int16_t RequantMul;  // crelu:        uint8 = mulhrs(max(acc,0), RequantMul)
static int16_t ClampQ, SqK;  // screlu/pair: uint8 = mulhrs(mulhrs(u<<5, w<<5), SqK), u,w = clamp(acc,0,ClampQ)

bool load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    int acc = 0, rows = 0, h1 = 0, h2 = 0, nb = 1, fa = 0, ha = 0, fs = 0, nkb = 32, psqt = 0, sets = 0;
    float scale = 0, cps = 400.0f, psqtScale = 0;
    bool ok = fread(magic, 1, 4, f) == 4;
    bool v5 = ok && memcmp(magic, "LNN5", 4) == 0;  // LNN4 + weight-set selector (training/nnue4.py SETS)
    bool v4 = ok && (memcmp(magic, "LNN4", 4) == 0 || v5);
    bool v3 = ok && (memcmp(magic, "LNN3", 4) == 0 || v4), v2 = ok && memcmp(magic, "LNN2", 4) == 0;
    ok = (v2 || v3) && fread(&acc, 4, 1, f) == 1 && fread(&rows, 4, 1, f) == 1 && fread(&h1, 4, 1, f) == 1 &&
         fread(&h2, 4, 1, f) == 1;
    if (ok && v3) ok = fread(&nb, 4, 1, f) == 1 && fread(&fa, 4, 1, f) == 1 && fread(&ha, 4, 1, f) == 1;
    if (ok && v4) ok = fread(&fs, 4, 1, f) == 1 && fread(&nkb, 4, 1, f) == 1 && fread(&psqt, 4, 1, f) == 1;
    if (ok && v5) ok = fread(&sets, 4, 1, f) == 1 && sets >= 0 && sets <= 4 && fs >= 1;
    ok = ok && fread(&scale, 4, 1, f) == 1;
    if (ok && v3) ok = fread(&cps, 4, 1, f) == 1;
    if (ok && v4) ok = fread(&psqtScale, 4, 1, f) == 1;
    int clampQ = scale > 0 ? (int)std::lround(1.0 / scale) : 0;
    int nfeat = fs == 0 ? 40960 : nkb * (fs == 2 ? 1408 : 704) * SetCount[sets];
    ok = ok && (fs == 0 || ((fs == 1 || fs == 2) && (nkb == 8 || nkb == 16 || nkb == 32))) && rows == nfeat + 1 &&
         acc == NNUE_ACC && h1 > 0 && h1 % 4 == 0 && h1 <= MAX_H1 && h2 >= 0 &&
         h2 % 8 == 0 && h2 <= MAX_H2 && nb >= 1 && nb <= MAX_NB && fa >= 0 && fa <= 2 && ha >= 0 && ha <= 2 &&
         (fa == 0 || clampQ * 32 <= 32767);
    if (!ok) {
        fclose(f);
        fprintf(stderr, "nnue: %s is not an LNN2/3/4 net of width %d this engine supports\n", path.c_str(), NNUE_ACC);
        return false;
    }
    int l1in = fa == 2 ? acc : 2 * acc, mult = ha == 2 ? 2 : 1, outIn = (h2 ? h2 : h1) * mult;
    std::vector<float> l1w((size_t)nb * h1 * l1in), l1b(nb * h1), l2w((size_t)nb * h2 * h1 * mult), l2b(nb * h2),
        ow(nb * outIn), ob(nb);
    if (FT) aligned_free64(FT);
    FT = (ft_t*)aligned_alloc64((size_t)rows * NNUE_ACC * sizeof(ft_t));
    if (PSQT) { free(PSQT); PSQT = nullptr; }
    std::vector<int32_t> ps(psqt ? (size_t)rows * nb : 0);
    // FT weights are int16 in every file; an FT_INT8 build narrows them (the net must fit, see --ft8)
    std::vector<int16_t> ft16((size_t)rows * acc);
    ok = fread(FTB, 2, acc, f) == (size_t)acc && fread(ft16.data(), 2, ft16.size(), f) == ft16.size() &&
         fread(ps.data(), 4, ps.size(), f) == ps.size() &&
         fread(l1w.data(), 4, l1w.size(), f) == l1w.size() && fread(l1b.data(), 4, l1b.size(), f) == l1b.size() &&
         fread(l2w.data(), 4, l2w.size(), f) == l2w.size() && fread(l2b.data(), 4, l2b.size(), f) == l2b.size() &&
         fread(ow.data(), 4, ow.size(), f) == ow.size() && fread(ob.data(), 4, ob.size(), f) == ob.size();
    fclose(f);
    if (!ok) return false;

    for (size_t i = 0; i < ft16.size(); i++) {
#ifdef FT_INT8
        if (ft16[i] < -127 || ft16[i] > 127) {
            fprintf(stderr, "nnue: %s has FT weights outside int8 (train with --ft8)\n", path.c_str());
            return false;
        }
#endif
        FT[i] = (ft_t)ft16[i];
    }
    FtScale = scale; CpScale = cps;
    FeatSet = fs; PsqtOn = psqt; NB_psqt = nb; PsqtScale = psqtScale; SetSel = sets; NKB = nkb;
    if (fs >= 1)
        for (int sq = 0; sq < 64; sq++) {
            int r = sq >> 3, fl = std::min(sq & 7, 7 - (sq & 7));
            KB[sq] = nkb == 32 ? r * 4 + fl : nkb == 16 ? std::min(r, 3) * 4 + fl : std::min(r, 1) * 4 + fl;
        }
    if (psqt) {
        PSQT = (int32_t*)malloc(ps.size() * 4);
        memcpy(PSQT, ps.data(), ps.size() * 4);
    }
    NetGen++;
    H1 = h1; H1p = (h1 + 7) / 8 * 8; H2 = h2; NB = nb; FtAct = fa; HidAct = ha; L1In = l1in; Mult = mult;
    RequantMul = (int16_t)std::min(32767L, std::lround(FtScale * 127.0f * 32768.0f));
    ClampQ = (int16_t)clampQ;
    // exact multiplier 127*scale^2*2^20 is ~142; the rounding error is undone in L1Deq
    double sqkExact = 127.0 * (double)scale * scale * 1048576.0;
    SqK = (int16_t)std::lround(sqkExact);
    double actCorr = fa == 0 ? 1.0 : sqkExact / SqK;
    L1Wf = l1w;
    memset(L1W, 0, sizeof(L1W));
    memset(L2WT, 0, sizeof(L2WT));
    memset(OutW, 0, sizeof(OutW));
    for (int b = 0; b < nb; b++) {
        for (int o = 0; o < h1; o++) {
            const float* w = &l1w[((size_t)b * h1 + o) * l1in];
            float maxAbs = 1e-12f;
            for (int i = 0; i < l1in; i++) maxAbs = std::fmax(maxAbs, std::fabs(w[i]));
            float s = 127.0f / maxAbs;  // widest int8 range this row allows
            for (int i = 0; i < l1in; i++) L1W[b][o][i] = (int8_t)std::lround(w[i] * s);
            L1Deq[b][o] = (float)(actCorr / (127.0 * s));
            L1B[b][o] = l1b[b * h1 + o];
        }
        auto pad = [&](int i) { return i < h1 ? i : H1p + (i - h1); };
        for (int o = 0; o < h2; o++) {
            for (int i = 0; i < h1 * mult; i++) L2WT[b][pad(i)][o] = l2w[((size_t)b * h2 + o) * h1 * mult + i];
            L2B[b][o] = l2b[b * h2 + o];
        }
        for (int i = 0; i < outIn; i++) OutW[b][h2 ? i : pad(i)] = ow[b * outIn + i];
        OutB[b] = ob[b];
    }
    memset(L1WS, 0, sizeof(L1WS));
    for (int b = 0; b < nb; b++)
        for (int o = 0; o < h1; o++)
            for (int i = 0; i < l1in; i++)
                L1WS[(((size_t)b * (l1in / 4) + i / 4) * H1p + o) * 4 + i % 4] = L1W[b][o][i];
    for (int m = 0; m < 256; m++)
        for (int k = 0, n = 0; k < 8; k++)
            if (m >> k & 1) NzLut[m][n++] = (uint16_t)k;
    return true;
}

static inline float psqt_term(const Accumulator& acc, int stm, int b) {
    return PsqtOn ? (acc.psqt[stm][b] - acc.psqt[stm ^ 1][b]) * PsqtScale * 0.5f : 0.0f;
}

static inline int bucket(int pieces) {
    int b = (pieces - 1) * NB / 32;
    return b < 0 ? 0 : (b >= NB ? NB - 1 : b);
}

// hidden activation of n (multiple of 8) values; dual writes crelu to out[0,n) and crelu^2 to out[n,2n)
static inline void hid_act(const float* in, int n, float* out) {
    const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    for (int i = 0; i < n; i += 8) {
        __m256 v = _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(in + i), zero), one);
        if (HidAct == 1) v = _mm256_mul_ps(v, v);
        _mm256_store_ps(out + i, v);
        if (HidAct == 2) _mm256_store_ps(out + n + i, _mm256_mul_ps(v, v));
    }
}

static inline float dot8(const float* a, const float* w, int n) {
    __m256 s = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) s = _mm256_fmadd_ps(_mm256_load_ps(a + i), _mm256_load_ps(w + i), s);
    return hsum_ps(s);
}

// L1 pre-activations (h1 floats) -> network output
static float tail(const float* l1out, int b) {
    alignas(32) float a1[2 * MAX_H1], z[MAX_H2], a2[2 * MAX_H2];
    hid_act(l1out, H1p, a1);
    if (!H2) return dot8(a1, OutW[b], H1p * Mult) + OutB[b];
    __m256 acc[MAX_H2 / 8];
    const int K = H2 / 8, n1 = H1p * Mult;
    for (int k = 0; k < K; k++) acc[k] = _mm256_load_ps(L2B[b] + 8 * k);
    for (int i = 0; i < n1; i++) {
        __m256 s = _mm256_set1_ps(a1[i]);
        for (int k = 0; k < K; k++) acc[k] = _mm256_fmadd_ps(s, _mm256_load_ps(&L2WT[b][i][8 * k]), acc[k]);
    }
    for (int k = 0; k < K; k++) _mm256_store_ps(z + 8 * k, acc[k]);
    hid_act(z, H2, a2);
    return dot8(a2, OutW[b], H2 * Mult) + OutB[b];
}

static inline __m256i sq_lanes(const int16_t* a, const int16_t* p, __m256i zero, __m256i cq, __m256i k) {
    __m256i u = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i*)a), zero), cq);
    __m256i w = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i*)p), zero), cq);
    return _mm256_mulhrs_epi16(_mm256_mulhrs_epi16(_mm256_slli_epi16(u, 5), _mm256_slli_epi16(w, 5)), k);
}

template <int N>
static inline void l1_rows(const uint8_t* x, int b, int o0, float* l1out) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i sum[N];
    for (int o = 0; o < N; o++) sum[o] = _mm256_setzero_si256();
    for (int i = 0; i < L1In; i += 32) {
        __m256i xv = _mm256_load_si256((const __m256i*)(x + i));
        for (int o = 0; o < N; o++) {
            __m256i p = _mm256_maddubs_epi16(xv, _mm256_load_si256((const __m256i*)&L1W[b][o0 + o][i]));
            sum[o] = _mm256_add_epi32(sum[o], _mm256_madd_epi16(p, ones));
        }
    }
    for (int o = 0; o < N; o++) l1out[o0 + o] = hsum_epi32(sum[o]) * L1Deq[b][o0 + o] + L1B[b][o0 + o];
}

#ifdef L1_STATS
static uint64_t StatEvals = 0, StatNz = 0, StatNzB = 0, StatNz2 = 0, StatIdx[MAX_L1IN];
// printed at exit; l1_idx.txt = fraction of evals in which each L1 input byte is non-zero
static struct StatPrint {
    ~StatPrint() {
        if (!StatEvals) return;
        fprintf(stderr, "L1 non-zero groups of 4: %.1f of %d (%.1f%%), bytes %.1f%%, pairs %.1f%%\n",
                (double)StatNz / StatEvals, L1In / 4, 100.0 * StatNz / StatEvals / (L1In / 4),
                100.0 * StatNzB / StatEvals / L1In, 100.0 * StatNz2 / StatEvals / (L1In / 2));
        if (FILE* f = fopen("l1_idx.txt", "w")) {
            for (int i = 0; i < L1In; i++) fprintf(f, "%.5f\n", (double)StatIdx[i] / StatEvals);
            fclose(f);
        }
    }
} StatPrinter;
#endif

// -DSPARSE_L1: same result as l1_rows over all H1 outputs (identical maddubs pairs, so identical saturation), but only for the
// groups of 4 inputs that are not all zero.
static inline void l1_sparse(const uint8_t* x, int b, float* l1out) {
    const int G = L1In / 4, K = H1p / 8;
    const int32_t* x32 = (const int32_t*)x;
    alignas(16) uint16_t nz[MAX_L1IN / 4 + 8];
    int n = 0;
    const __m256i zero = _mm256_setzero_si256();
    for (int i = 0; i < G; i += 8) {
        __m256i v = _mm256_load_si256((const __m256i*)(x32 + i));
        unsigned m = ~_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, zero))) & 0xFF;
        _mm_storeu_si128((__m128i*)(nz + n), _mm_add_epi16(_mm_set1_epi16((short)i), _mm_load_si128((const __m128i*)NzLut[m])));
        n += _mm_popcnt_u32(m);
    }
#ifdef L1_STATS
    StatEvals++; StatNz += n;
    for (int i = 0; i < L1In; i++) StatNzB += x[i] != 0, StatIdx[i] += x[i] != 0;
    for (int i = 0; i < L1In; i += 2) StatNz2 += (x[i] | x[i + 1]) != 0;
#endif
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i sum[MAX_H1 / 8];
    for (int k = 0; k < K; k++) sum[k] = zero;
    const int8_t* W = L1WS + (size_t)b * G * H1p * 4;
    if (K == 1) {  // 8 outputs: 4 independent sums, one 32-byte weight load per non-zero group
        __m256i s0 = zero, s1 = zero, s2 = zero, s3 = zero;
        const __m256i* w = (const __m256i*)W;
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi32(x32[nz[j]]), _mm256_load_si256(w + nz[j])), ones));
            s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi32(x32[nz[j + 1]]), _mm256_load_si256(w + nz[j + 1])), ones));
            s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi32(x32[nz[j + 2]]), _mm256_load_si256(w + nz[j + 2])), ones));
            s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi32(x32[nz[j + 3]]), _mm256_load_si256(w + nz[j + 3])), ones));
        }
        for (; j < n; j++)
            s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi32(x32[nz[j]]), _mm256_load_si256(w + nz[j])), ones));
        sum[0] = _mm256_add_epi32(_mm256_add_epi32(s0, s1), _mm256_add_epi32(s2, s3));
    } else for (int j = 0; j < n; j++) {
        const __m256i xv = _mm256_set1_epi32(x32[nz[j]]);
        const __m256i* w = (const __m256i*)(W + (size_t)nz[j] * H1p * 4);
        for (int k = 0; k < K; k++)
            sum[k] = _mm256_add_epi32(sum[k], _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_load_si256(w + k)), ones));
    }
    alignas(32) int32_t s[MAX_H1];
    for (int k = 0; k < K; k++) _mm256_store_si256((__m256i*)(s + 8 * k), sum[k]);
    for (int o = 0; o < H1; o++) l1out[o] = s[o] * L1Deq[b][o] + L1B[b][o];
}

// L1 input: requantized FT activations of both perspectives (stm first), L1In uint8 values
static void build_x(const Accumulator& acc, int stm, uint8_t* x) {
    const __m256i zero = _mm256_setzero_si256(), cap = _mm256_set1_epi8(127);
    const int half_n = L1In / 2;
    for (int half = 0; half < 2; half++) {
        const int16_t* a = acc.v[half == 0 ? stm : stm ^ 1];
        uint8_t* xo = x + half * half_n;
        if (FtAct == 0) {
            const __m256i mul = _mm256_set1_epi16(RequantMul);
            for (int c = 0; c < NNUE_ACC / 32; c++) {
                __m256i lo = _mm256_mulhrs_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i*)(a + 32 * c)), zero), mul);
                __m256i hi = _mm256_mulhrs_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i*)(a + 32 * c + 16)), zero), mul);
                __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi16(lo, hi), 0xD8);
                _mm256_store_si256((__m256i*)(xo + c * 32), _mm256_min_epu8(p, cap));
            }
        } else {
            const __m256i cq = _mm256_set1_epi16(ClampQ), k = _mm256_set1_epi16(SqK);
            const int16_t* pb = FtAct == 2 ? a + NNUE_ACC / 2 : a;
            for (int c = 0; c < half_n / 32; c++) {
                __m256i lo = sq_lanes(a + 32 * c, pb + 32 * c, zero, cq, k);
                __m256i hi = sq_lanes(a + 32 * c + 16, pb + 32 * c + 16, zero, cq, k);
                __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi16(lo, hi), 0xD8);
                _mm256_store_si256((__m256i*)(xo + c * 32), _mm256_min_epu8(p, cap));
            }
        }
    }
}

double evaluate_quant_raw(const Accumulator& acc, int stm, int pieces) {
    alignas(32) uint8_t x[MAX_L1IN];
    build_x(acc, stm, x);
    const int b = bucket(pieces);
    alignas(32) float l1out[MAX_H1] = {};
#ifndef SPARSE_L1
    int o0 = 0;
    for (; o0 + 8 <= H1; o0 += 8) l1_rows<8>(x, b, o0, l1out);
    if (o0 < H1) l1_rows<4>(x, b, o0, l1out);
#else
    l1_sparse(x, b, l1out);
#endif
    return tail(l1out, b) + psqt_term(acc, stm, b);
}

double evaluate_float_raw(const Accumulator& acc, int stm, int pieces) {
    std::vector<float> x(L1In);
    const int half_n = L1In / 2;
    for (int half = 0; half < 2; half++) {
        const int16_t* a = acc.v[half == 0 ? stm : stm ^ 1];
        for (int j = 0; j < half_n; j++) {
            float u = std::fmin(std::fmax(a[j] * FtScale, 0.0f), 1.0f);
            if (FtAct == 1) u = u * u;
            if (FtAct == 2) u *= std::fmin(std::fmax(a[j + NNUE_ACC / 2] * FtScale, 0.0f), 1.0f);
            x[half * half_n + j] = u;
        }
    }
    const int b = bucket(pieces);
    alignas(32) float l1out[MAX_H1] = {};
    for (int o = 0; o < H1; o++) {
        const float* w = &L1Wf[((size_t)b * H1 + o) * L1In];
        double s = L1B[b][o];
        for (int i = 0; i < L1In; i++) s += (double)x[i] * w[i];
        l1out[o] = (float)s;
    }
    return tail(l1out, b) + psqt_term(acc, stm, b);
}

int evaluate(const Accumulator& acc, int stm, int pieces) {
    return (int)std::lround(evaluate_quant_raw(acc, stm, pieces) * CpScale);
}

// ---- uncertainty head (SIG1, training/sigma.py): expected |static eval - deep eval| in cp ----
//   L1 on the same uint8 input as the eval head (int8 rows, per-row scale), clamp, dual,
//   out = w . [h, h^2, a, a^2] + b with a = min(|eval|/400, 4); sigma = 100 * softplus(out)
constexpr int SIG_H1 = 8;
static int SigNB = 0;
alignas(64) static int8_t SL1W[MAX_NB][SIG_H1][MAX_L1IN];
static float SL1Deq[MAX_NB][SIG_H1], SL1B[MAX_NB][SIG_H1], SOutW[MAX_NB][2 * SIG_H1 + 2], SOutB[MAX_NB];

bool load_sigma(const std::string& path) {
    SigNB = 0;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    int nb = 0, h1 = 0, l1in = 0;
    bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, "SIG1", 4) == 0 && fread(&nb, 4, 1, f) == 1 &&
              fread(&h1, 4, 1, f) == 1 && fread(&l1in, 4, 1, f) == 1 && nb >= 1 && nb <= MAX_NB && h1 == SIG_H1 &&
              l1in == L1In;
    std::vector<float> w1((size_t)nb * h1 * l1in), b1(nb * h1), wo(nb * (2 * h1 + 2)), bo(nb);
    ok = ok && fread(w1.data(), 4, w1.size(), f) == w1.size() && fread(b1.data(), 4, b1.size(), f) == b1.size() &&
         fread(wo.data(), 4, wo.size(), f) == wo.size() && fread(bo.data(), 4, bo.size(), f) == bo.size();
    fclose(f);
    if (!ok) { fprintf(stderr, "nnue: %s is not a SIG1 file for this net\n", path.c_str()); return false; }
    // FtAct pair/screlu inputs carry the same activation correction as the eval head
    double actCorr = FtAct == 0 ? 1.0 : (127.0 * (double)FtScale * FtScale * 1048576.0) / SqK;
    for (int b = 0; b < nb; b++) {
        for (int o = 0; o < h1; o++) {
            const float* w = &w1[((size_t)b * h1 + o) * l1in];
            float maxAbs = 1e-12f;
            for (int i = 0; i < l1in; i++) maxAbs = std::fmax(maxAbs, std::fabs(w[i]));
            float sc = 127.0f / maxAbs;
            for (int i = 0; i < l1in; i++) SL1W[b][o][i] = (int8_t)std::lround(w[i] * sc);
            SL1Deq[b][o] = (float)(actCorr / (127.0 * sc));
            SL1B[b][o] = b1[b * h1 + o];
        }
        for (int i = 0; i < 2 * h1 + 2; i++) SOutW[b][i] = wo[b * (2 * h1 + 2) + i];
        SOutB[b] = bo[b];
    }
    SigNB = nb;
    return true;
}
bool sigma_loaded() { return SigNB > 0; }

static int sigma_from_x(const uint8_t* x, int pieces, int evalCp) {
    int b = (pieces - 1) * SigNB / 32;
    b = b < 0 ? 0 : (b >= SigNB ? SigNB - 1 : b);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i sum[SIG_H1];
    for (int o = 0; o < SIG_H1; o++) sum[o] = _mm256_setzero_si256();
    for (int i = 0; i < L1In; i += 32) {
        __m256i xv = _mm256_load_si256((const __m256i*)(x + i));
        for (int o = 0; o < SIG_H1; o++) {
            __m256i p = _mm256_maddubs_epi16(xv, _mm256_load_si256((const __m256i*)&SL1W[b][o][i]));
            sum[o] = _mm256_add_epi32(sum[o], _mm256_madd_epi16(p, ones));
        }
    }
    float a = std::fmin(std::fabs((float)evalCp) / 400.0f, 4.0f);
    float z = SOutB[b] + SOutW[b][2 * SIG_H1] * a + SOutW[b][2 * SIG_H1 + 1] * a * a;
    for (int o = 0; o < SIG_H1; o++) {
        float h = std::fmin(std::fmax(hsum_epi32(sum[o]) * SL1Deq[b][o] + SL1B[b][o], 0.0f), 1.0f);
        z += SOutW[b][o] * h + SOutW[b][SIG_H1 + o] * h * h;
    }
    float sp = z > 20.0f ? z : std::log1p(std::exp(z));
    return (int)std::lround(100.0f * sp);
}

int evaluate_sigma(const Accumulator& acc, int stm, int pieces, int& sigma) {
    alignas(32) uint8_t x[MAX_L1IN];
    build_x(acc, stm, x);
    const int b = bucket(pieces);
    alignas(32) float l1out[MAX_H1] = {};
    int o0 = 0;
    for (; o0 + 8 <= H1; o0 += 8) l1_rows<8>(x, b, o0, l1out);
    if (o0 < H1) l1_rows<4>(x, b, o0, l1out);
    int v = (int)std::lround((tail(l1out, b) + psqt_term(acc, stm, b)) * CpScale);
    sigma = sigma_from_x(x, pieces, v);
    return v;
}
