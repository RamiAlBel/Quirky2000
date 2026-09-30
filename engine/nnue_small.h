// Interface of the 128-wide second net (nnue_small.cpp); layout must match nnue.h built with ACC_WIDTH 128
#pragma once
#include "position.h"
namespace nnue_small {
constexpr int ACC = 128;
struct alignas(64) Accumulator {
    int16_t v[2][ACC];
    int32_t psqt[2][8];
};
bool load(const std::string& path);
void refresh_all(const Position& pos, Accumulator& acc);
void update(const Position& parent, Move m, const Position& child, const Accumulator& in, Accumulator& out);
int evaluate(const Accumulator& acc, int stm, int pieces);
}
