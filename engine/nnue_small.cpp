// Second copy of the NNUE code: namespace nnue_small, accumulator width 128 (see nnue.h / nnue_small.h)
#define NNUE_SMALL_TU
#define NNUE_NS nnue_small
#undef ACC_WIDTH
#define ACC_WIDTH 128
#include "nnue.cpp"
