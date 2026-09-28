// UCI front end. Works with the local web bridge (server.js) and with any
// standard UCI GUI (CuteChess, Arena, ...).
#include "position.h"
#include "nnue.h"
#include "search.h"
#include "tune.h"
#include "book.h"

extern int UseBook, BookDepth;  // search.cpp tunables
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#include <climits>
#endif

static const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static std::string exe_dir() {
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
#else
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf));
    if (n < 0) n = 0;
#endif
    std::string p(buf, n);
    size_t k = p.find_last_of("\\/");
    return k == std::string::npos ? "." : p.substr(0, k);
}

static void set_position(std::istringstream& is, Position& pos, std::vector<uint64_t>& hist) {
    std::string tok, fen;
    is >> tok;
    if (tok == "startpos") { fen = START_FEN; is >> tok; }
    else if (tok == "fen") {
        while (is >> tok && tok != "moves") fen += tok + " ";
    } else return;
    Position candidate;
    candidate.set_fen(fen);
    if (!candidate.is_valid()) {
        out_line("info string invalid position rejected: %s", fen.c_str());
        return;
    }
    pos = candidate;
    hist.assign(1, pos.key);
    if (tok == "moves") {
        while (is >> tok) {
            Move m = parse_uci_move(pos, tok);
            if (!m) break;
            pos.do_move(m);
            hist.push_back(pos.key);
        }
    }
}

static uint64_t perft(const Position& pos, int depth) {
    MoveList list;
    generate(pos, list, GEN_ALL);
    uint64_t n = 0;
    for (int i = 0; i < list.size; i++) {
        if (!pos.is_legal(list.moves[i])) continue;
        if (depth == 1) { n++; continue; }
        Position c = pos;
        c.do_move(list.moves[i]);
        n += perft(c, depth - 1);
    }
    return n;
}

static void bench(int depth) {
    const char* fens[] = {
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "r2qk2r/ppp2ppp/2np1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 4 7",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
    };
    uint64_t total = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (const char* f : fens) {
        Position p;
        p.set_fen(f);
        auto s = std::chrono::steady_clock::now();
        uint64_t n = search::bench_one(p, depth, true);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count();
        total += n;
        printf("depth %d  nodes %11llu  %8.0f ms  %s\n", depth, (unsigned long long)n, ms, f);
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("bench: %llu nodes  %.2f s  %.0f nps  (threads %d)\n", (unsigned long long)total, sec, total / sec, search::threads());
    fflush(stdout);
}

// ---- self-play data generation (REC4 records, same format as training/extract_v4.py) ----
// datagen OUT GAMES NODES SEED: games from the start position + 8 random plies, NODES per move with the current
// options; keeps positions with ply >= 16, side to move not in check, best move quiet, |score| < 2000.
// Game end: mate/stalemate/draw rules, adjudication at |score| >= 2500 for 4 plies (win) or ply 400 (draw).
#pragma pack(push, 1)
struct Rec4 { uint16_t pc[32]; int16_t cp; uint8_t stm, ply; int8_t result; uint8_t pad; };
#pragma pack(pop)
static_assert(sizeof(Rec4) == 70, "REC4");
static void datagen(const std::string& out, int games, uint64_t nodes, uint64_t seed) {
    FILE* f = fopen(out.c_str(), "ab");
    if (!f) { printf("cannot open %s\n", out.c_str()); return; }
    uint64_t rng = seed * 0x9E3779B97F4A7C15ULL + 1;
    auto rnd = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    long kept = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < games; g++) {
        Position p;
        p.set_fen(START_FEN);
        std::vector<uint64_t> hist{p.key};
        bool ok = true;
        for (int i = 0; i < 8 && ok; i++) {
            MoveList l;
            generate_legal(p, l);
            if (!l.size) ok = false;
            else { p.do_move(l.moves[rnd() % l.size]); hist.push_back(p.key); }
        }
        MoveList l0;
        generate_legal(p, l0);
        if (!ok || !l0.size) continue;
        search::clear_hash();
        struct Pend { Rec4 r; int stmWhite; };
        std::vector<Pend> pend;
        int resultWhite = 0, streakW = 0, streakB = 0;
        for (int ply = 8;; ply++) {
            MoveList l;
            generate_legal(p, l);
            if (!l.size) { resultWhite = p.checkers ? (p.stm == WHITE ? -1 : 1) : 0; break; }
            if (p.halfmove >= 100 || ply >= 400) break;
            int rep = 0;
            for (size_t i = 0; i + 1 < hist.size(); i++) rep += hist[i] == p.key;
            if (rep >= 2) break;
            Move best; int score;
            search::search_nodes(p, hist, nodes, best, score);
            if (!best) break;
            int sw = p.stm == WHITE ? score : -score;
            streakW = sw >= 2500 ? streakW + 1 : 0;
            streakB = sw <= -2500 ? streakB + 1 : 0;
            if (streakW >= 4) { resultWhite = 1; break; }
            if (streakB >= 4) { resultWhite = -1; break; }
            if (ply >= 16 && !p.checkers && !is_capture(best) && !is_promo(best) && std::abs(score) < 2000) {
                Pend e{};
                int n = 0;
                for (int sq = 0; sq < 64; sq++)
                    if (p.board[sq] != NO_PIECE && n < 32) e.r.pc[n++] = (uint16_t)(p.board[sq] * 64 + sq);
                for (; n < 32; n++) e.r.pc[n] = 0xFFFF;
                e.r.cp = (int16_t)std::max(-1500, std::min(1500, score));
                e.r.stm = p.stm == WHITE;
                e.r.ply = (uint8_t)std::min(ply, 255);
                e.stmWhite = p.stm == WHITE;
                pend.push_back(e);
            }
            p.do_move(best);
            hist.push_back(p.key);
        }
        for (auto& e : pend) { e.r.result = (int8_t)(e.stmWhite ? resultWhite : -resultWhite); fwrite(&e.r, sizeof(Rec4), 1, f); }
        kept += (long)pend.size();
        if ((g + 1) % 20 == 0) {
            fflush(f);
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("datagen: %d games, %ld positions, %.1f games/min\n", g + 1, kept, (g + 1) / s * 60);
            fflush(stdout);
        }
    }
    fclose(f);
}

int main(int argc, char** argv) {
    bb::init();
    // default net next to the exe; the EvalFile option can load another one
    std::string netPath = exe_dir() + "/lite.nnue";
    bool netLoaded = nnue::load(netPath);
    if (!netLoaded) {
        printf("info string could not load network %s (set EvalFile)\n", netPath.c_str());
        fflush(stdout);
    }
    search::init();
    Position pos;
    pos.set_fen(START_FEN);
    std::vector<uint64_t> hist{pos.key};

    // allow one-shot commands from argv, e.g. `nnue_engine.exe bench 14`
    std::vector<std::string> queued;
    if (argc > 1) {
        std::string cmd;
        for (int i = 1; i < argc; i++) { cmd += argv[i]; cmd += ' '; }
        queued.push_back(cmd);
        queued.push_back("quit");
    }

    std::string line;
    size_t qi = 0;
    while (true) {
        if (qi < queued.size()) line = queued[qi++];
        else if (!std::getline(std::cin, line)) line = "quit";
        if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF)
            line.erase(0, 3);  // PowerShell 5.1 prefixes piped stdin with a UTF-8 BOM
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream is(line);
        std::string tok;
        is >> tok;
        if (tok == "uci") {
            out_line("id name LiteNNUE-native");
            out_line("id author ralbe + Claude");
            out_line("option name Hash type spin default 256 min 1 max 16384");
            out_line("option name Threads type spin default 1 min 1 max 64");
            out_line("option name MultiPV type spin default 1 min 1 max 16");
            out_line("option name Ponder type check default false");
            out_line("option name SyzygyPath type string default <empty>");
            out_line("option name BookFile type string default <empty>");
            out_line("option name SmallNetFile type string default <empty>");
            out_line("option name EvalFile type string default lite.nnue");
            out_line("option name SigmaFile type string default <empty>");
            for (auto& t : tunables())
                out_line("option name %s type spin default %d min %d max %d", t.name, t.def, t.lo, t.hi);
            out_line("uciok");
        } else if (tok == "isready") {
            out_line("readyok");
        } else if (tok == "setoption") {
            std::string name, value, t;
            is >> t;  // "name"
            while (is >> t && t != "value") name += (name.empty() ? "" : " ") + t;
            std::getline(is, value);  // rest of line: file paths may contain spaces
            value.erase(0, value.find_first_not_of(' '));
            while (!value.empty() && value.back() == ' ') value.pop_back();
            if (name == "EvalFile") {
                search::stop(); search::wait();
                std::string path = value.find_first_of("/\\") == std::string::npos ? exe_dir() + "/" + value : value;
                netLoaded = nnue::load(path);
                out_line(netLoaded ? "info string loaded network %s" : "info string ERROR could not load network %s", path.c_str());
            } else if (name == "SigmaFile") {
                search::stop(); search::wait();
                out_line(nnue::load_sigma(value) ? "info string sigma head %s loaded" : "info string ERROR could not load sigma head %s", value.c_str());
            } else if (name == "SmallNetFile") {
                out_line(search::set_small_net(value) ? "info string small net %s loaded" : "info string ERROR could not load small net %s", value.c_str());
            } else if (name == "BookFile") {
                out_line(book::open(value) ? "info string book %s loaded" : "info string ERROR could not load book %s", value.c_str());
            } else if (name == "SyzygyPath") {
                int n = search::set_syzygy(value);
                out_line("info string syzygy: %d-piece tables from %s", n, value.c_str());
            } else if (name == "Hash") search::set_hash_mb(std::stoul(value));
            else if (name == "Threads") search::set_threads(std::stoi(value));
            else if (name == "MultiPV") search::set_multipv(std::stoi(value));
            else {
                char* end = nullptr;
                long v = std::strtol(value.c_str(), &end, 10);
                if (end == value.c_str() || !set_tunable(name, (int)v)) out_line("info string ignored option %s", name.c_str());
            }
        } else if (tok == "ucinewgame") {
            search::stop(); search::wait();
            search::clear_hash();
        } else if (tok == "position") {
            search::stop(); search::wait();
            set_position(is, pos, hist);
        } else if (tok == "go") {
            if (!netLoaded) {
                out_line("info string ERROR no network loaded");
                return 1;
            }
            Limits lim;
            int64_t wtime = -1, btime = -1, winc = 0, binc = 0;
            while (is >> tok) {
                if (tok == "infinite") lim.infinite = true;
                else if (tok == "ponder") lim.ponder = true;
                else if (tok == "depth") is >> lim.depth;
                else if (tok == "movetime") is >> lim.movetime;
                else if (tok == "nodes") is >> lim.nodes;
                else if (tok == "wtime") is >> wtime;
                else if (tok == "btime") is >> btime;
                else if (tok == "winc") is >> winc;
                else if (tok == "binc") is >> binc;
            }
            int64_t myTime = pos.stm == WHITE ? wtime : btime, myInc = pos.stm == WHITE ? winc : binc;
            if (myTime >= 0 && !lim.movetime) {
                lim.movetime = std::max<int64_t>(10, std::min(myTime / 25 + myInc * 3 / 4, myTime - 50));
                lim.fromClock = true;
                lim.time = myTime;
                lim.inc = myInc;
            }
            // own book: answer instantly while in book (hist holds one key per position since the start)
            if (UseBook && book::loaded() && (int)hist.size() - 1 < BookDepth && !lim.ponder) {
                if (Move bm = book::probe(pos)) {
                    out_line("bestmove %s", move_to_uci(bm).c_str());
                    continue;
                }
            }
            search::start(pos, hist, lim);
        } else if (tok == "ponderhit") {
            search::ponderhit();
        } else if (tok == "stop") {
            search::stop(); search::wait();
        } else if (tok == "quit") {
            search::stop(); search::wait();
            break;
        } else if (tok == "polykey") {
            printf("%016llx\n", (unsigned long long)book::key(pos));
            fflush(stdout);
        } else if (tok == "d") {
            printf("%s\n", pos.fen().c_str());
            fflush(stdout);
        } else if (tok == "eval") {
            Accumulator acc;
            nnue::refresh_all(pos, acc);
            double q = nnue::evaluate_quant_raw(acc, pos.stm, popcount(pos.occupied)), f = nnue::evaluate_float_raw(acc, pos.stm, popcount(pos.occupied));
            printf("raw float %.6f  raw quant %.6f  |diff| %.6f  tanh(float) %.6f  cp(quant) %d\n",
                   f, q, std::fabs(f - q), std::tanh(f), nnue::evaluate(acc, pos.stm, popcount(pos.occupied)));
            fflush(stdout);
        } else if (tok == "acccheck") {
            // walk a depth-N tree applying incremental updates, compare against full refresh at every node
            int depth = 4;
            is >> depth;
            uint64_t nodes = 0, bad = 0;
            std::vector<Accumulator> stack(depth + 1);
            nnue::refresh_all(pos, stack[0]);
            struct Walk {
                static void go(const Position& p, int d, int ply, std::vector<Accumulator>& st, uint64_t& n, uint64_t& b) {
                    Accumulator fresh;
                    nnue::refresh_all(p, fresh);
                    n++;
                    if (std::memcmp(&fresh, &st[ply], sizeof(Accumulator)) != 0) b++;
                    if (d == 0) return;
                    MoveList l;
                    generate_legal(p, l);
                    for (int i = 0; i < l.size; i++) {
                        Position c = p;
                        c.do_move(l.moves[i]);
                        nnue::update(p, l.moves[i], c, st[ply], st[ply + 1]);
                        go(c, d - 1, ply + 1, st, n, b);
                    }
                }
            };
            Walk::go(pos, depth, 0, stack, nodes, bad);
            printf("acccheck depth %d: %llu nodes, %llu mismatches\n", depth, (unsigned long long)nodes, (unsigned long long)bad);
            fflush(stdout);
        } else if (tok == "evalbench") {
            Accumulator acc;
            nnue::refresh_all(pos, acc);
            const int N = 5000000;
            volatile int sink = 0;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N; i++) sink += nnue::evaluate(acc, i & 1, 32);
            double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / N;
            printf("quantized evaluate(): %.1f ns/call\n", ns);
            fflush(stdout);
        } else if (tok == "sig") {  // static eval and uncertainty of the current position (cross-check vs sigma.py)
            Accumulator a;
            nnue::refresh_all(pos, a);
            int sg = -1, v = nnue::sigma_loaded() ? nnue::evaluate_sigma(a, pos.stm, popcount(pos.occupied), sg)
                                                   : nnue::evaluate(a, pos.stm, popcount(pos.occupied));
            printf("eval %d sigma %d\n", v, sg);
            fflush(stdout);
        } else if (tok == "datagen") {
            std::string out = "datagen.bin";
            int games = 100; uint64_t nodes = 5000, seed = 1;
            is >> out >> games >> nodes >> seed;
            datagen(out, games, nodes, seed);
        } else if (tok == "bench") {
            int depth = 13;
            is >> depth;
            bench(depth);
        } else if (tok == "perft") {
            int depth = 5;
            is >> depth;
            auto t0 = std::chrono::steady_clock::now();
            uint64_t n = perft(pos, depth);
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("perft %d: %llu  (%.2fs, %.1f Mnps)\n", depth, (unsigned long long)n, s, n / s / 1e6);
            fflush(stdout);
        }
    }
    return 0;
}
