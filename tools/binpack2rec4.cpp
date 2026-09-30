// Convert a Stockfish .binpack (e.g. linrock's Leela test80 data) into REC4 records (see training/extract_v4.py).
// Filters like extract_v4: ply >= 8, side to move not in check, move not a capture/promotion, valid score.
// cp = binpack score (side to move) * SCALE, clamped +-1500; result = game result from the side to move's view.
// usage: binpack2rec4 in.binpack out.bin [scale=1.0] [keep_prob=1.0] [max_out=0 (all)]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <random>
#include "lib/binpack.h"

#pragma pack(push, 1)
struct Rec4 { uint16_t pc[32]; int16_t cp; uint8_t stm, ply; int8_t result; uint8_t pad; };
#pragma pack(pop)
static_assert(sizeof(Rec4) == 70, "REC4 must be 70 bytes");

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s in.binpack out.bin [scale] [keep_prob] [max_out]\n", argv[0]); return 1; }
    const double scale = argc > 3 ? atof(argv[3]) : 1.0, keep = argc > 4 ? atof(argv[4]) : 1.0;
    const uint64_t max_out = argc > 5 ? strtoull(argv[5], nullptr, 10) : 0;
    binpack::CompressedTrainingDataEntryReader reader(argv[1]);
    FILE* out = fopen(argv[2], "wb");
    if (!out) { perror("out"); return 1; }
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> U(0, 1);
    uint64_t seen = 0, kept = 0, none = 0, filt = 0;
    int last_ply = -1, last_score = 32002;
    std::vector<Rec4> buf; buf.reserve(1 << 16);
    while (reader.hasNext()) {
        const auto e = reader.next();
        ++seen;
        // same placeholder-zero heuristic as nnue-pytorch's loader
        const bool ph0 = e.ply > last_ply && last_score != 32002 && std::abs(last_score) > 100 && e.result != 0 && e.score == 0;
        last_ply = e.ply;
        if (e.score == 32002 || ph0) { ++none; continue; }
        last_score = e.score;
        if (e.ply < 8 || e.isCapturingMove() || e.move.type == chess::MoveType::Promotion || e.pos.isCheck()) { ++filt; continue; }
        if (keep < 1.0 && U(rng) >= keep) continue;
        Rec4 r; std::fill(std::begin(r.pc), std::end(r.pc), 0xFFFF);
        int n = 0;
        for (int c = 0; c < 2; ++c)
            for (int pt = 0; pt < 6; ++pt) {
                const auto p = chess::Piece(chess::PieceType(pt), chess::Color(c));
                for (chess::Square sq : e.pos.piecesBB(p))
                    if (n < 32) r.pc[n++] = uint16_t((c * 6 + pt) * 64 + int(chess::ordinal(sq)));
            }
        const double cp = std::clamp(std::round(e.score * scale), -1500.0, 1500.0);
        r.cp = int16_t(cp);
        r.stm = e.pos.sideToMove() == chess::Color::White ? 1 : 0;
        r.ply = uint8_t(std::min<int>(e.ply, 255));
        r.result = int8_t(e.result > 0 ? 1 : e.result < 0 ? -1 : 0);
        r.pad = 0;
        buf.push_back(r); ++kept;
        if (buf.size() == buf.capacity()) { fwrite(buf.data(), sizeof(Rec4), buf.size(), out); buf.clear(); }
        if (seen % 100000000 == 0) fprintf(stderr, "seen %llu kept %llu\n", (unsigned long long)seen, (unsigned long long)kept);
        if (max_out && kept >= max_out) break;
    }
    fwrite(buf.data(), sizeof(Rec4), buf.size(), out); fclose(out);
    printf("{\"seen\": %llu, \"kept\": %llu, \"no_score\": %llu, \"filtered\": %llu, \"scale\": %g, \"keep_prob\": %g}\n",
           (unsigned long long)seen, (unsigned long long)kept, (unsigned long long)none, (unsigned long long)filt, scale, keep);
}
