// Quantized HalfKP(40960) -> 2xACC -> H1 -> 32 -> 1 inference, Stockfish-style.
// Net format is chosen at compile time:
//   default:     LNN1, ACC 256, H1 8,  int8 feature weights (the original lite.nnue)
//   -DNNUE_LNN2: ACC = -DACC_WIDTH (256/512/1024/2048), int16 feature weights + bias; loads
//                LNN2 (H1 16, H2 32, clipped ReLU) or LNN3 nets (head sizes, activations and
//                piece-count output buckets read from the file, see training/head_lib.py)
//   feature transformer: int16 accumulator, updated incrementally
//   L1 input:            clipped accumulator requantized to uint8 [0,127]
//   L1:                  uint8 x int8 via _mm256_maddubs_epi16 / _mm256_madd_epi16 -> int32
//   L2, L3:              tiny (8->32->1), done in float FMA
#pragma once
#include "position.h"
// The same code is compiled twice: as namespace nnue (main net, width ACC_WIDTH) and, via nnue_small.cpp,
// as namespace nnue_small (width 128, used for clearly decided positions when UseSmallNet is on).
#ifndef NNUE_NS
#define NNUE_NS nnue
#endif

#ifdef NNUE_LNN2
#ifndef ACC_WIDTH
#define ACC_WIDTH 256
#endif
constexpr int NNUE_ACC = ACC_WIDTH;
#ifdef FT_INT8
typedef int8_t ft_t;  // -DFT_INT8: nets trained with --ft8 (all FT weights within +-127 at scale 1/127)
#else
typedef int16_t ft_t;
#endif
#else
constexpr int NNUE_ACC = 256, NNUE_H1 = 8, NNUE_H2 = 32;
typedef int8_t ft_t;
#endif
static_assert(NNUE_ACC % 64 == 0, "accumulator width must be a multiple of 64");

namespace NNUE_NS {
struct alignas(64) Accumulator {
    int16_t v[2][NNUE_ACC];
    int32_t psqt[2][8];  // LNN4 material shortcut sums per output bucket (zero otherwise)
};

bool load(const std::string& path);
void refresh(const Position& pos, Accumulator& acc, int persp);
void refresh_all(const Position& pos, Accumulator& acc);
// parent/in = state before m; child/out = state after
void update(const Position& parent, Move m, const Position& child, const Accumulator& in, Accumulator& out);
// pieces = pieces on the board incl. kings (selects the output bucket of LNN3 nets)
int evaluate(const Accumulator& acc, int stm, int pieces);             // centipawns, quantized SIMD path
double evaluate_float_raw(const Accumulator& acc, int stm, int pieces); // raw network output, float reference
double evaluate_quant_raw(const Accumulator& acc, int stm, int pieces); // raw network output, quantized path
#ifdef NNUE_LNN2
bool load_sigma(const std::string& path);  // SIG1 uncertainty head for the loaded net (training/sigma.py)
bool sigma_loaded();
int evaluate_sigma(const Accumulator& acc, int stm, int pieces, int& sigma);  // eval cp + sigma cp in one pass
#endif
}
#ifndef NNUE_SMALL_TU
using nnue::Accumulator;
#endif
