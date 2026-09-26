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
    int acc = 0, rows = 0, h1 = 0, h2 = 0, nb = 1, fa = 0, ha = 0, fs = 0, nkb = 32, psqt = 0;
    float scale = 0, cps = 400.0f, psqtScale = 0;
    bool ok = fread(magic, 1, 4, f) == 4;
    bool v4 = ok && memcmp(magic, "LNN4", 4) == 0;
    bool v3 = ok && (memcmp(magic, "LNN3", 4) == 0 || v4), v2 = ok && memcmp(magic, "LNN2", 4) == 0;
    ok = (v2 || v3) && fread(&acc, 4, 1, f) == 1 && fread(&rows, 4, 1, f) == 1 && fread(&h1, 4, 1, f) == 1 &&
         fread(&h2, 4, 1, f) == 1;
    if (ok && v3) ok = fread(&nb, 4, 1, f) == 1 && fread(&fa, 4, 1, f) == 1 && fread(&ha, 4, 1, f) == 1;
    if (ok && v4) ok = fread(&fs, 4, 1, f) == 1 && fread(&nkb, 4, 1, f) == 1 && fread(&psqt, 4, 1, f) == 1;
    ok = ok && fread(&scale, 4, 1, f) == 1;
    if (ok && v3) ok = fread(&cps, 4, 1, f) == 1;
    if (ok && v4) ok = fread(&psqtScale, 4, 1, f) == 1;
    int clampQ = scale > 0 ? (int)std::lround(1.0 / scale) : 0;
    int nfeat = fs == 0 ? 40960 : nkb * 704;
    ok = ok && (fs == 0 || (fs == 1 && (nkb == 8 || nkb == 16 || nkb == 32))) && rows == nfeat + 1 &&
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
    FeatSet = fs; PsqtOn = psqt; NB_psqt = nb; PsqtScale = psqtScale;
    if (fs == 1)
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

double evaluate_quant_raw(const Accumulator& acc, int stm, int pieces) {
    alignas(32) uint8_t x[MAX_L1IN];
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
    const int b = bucket(pieces);
    alignas(32) float l1out[MAX_H1] = {};
    int o0 = 0;
    for (; o0 + 8 <= H1; o0 += 8) l1_rows<8>(x, b, o0, l1out);
    if (o0 < H1) l1_rows<4>(x, b, o0, l1out);
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
