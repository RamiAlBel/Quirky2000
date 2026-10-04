// Bitboard board representation + legal move generation.
// Square indexing a1=0 .. h8=63 (rank*8+file), the same indexing the NNUE uses.
// Sliding attacks use PEXT (BMI2, fast on Zen 3) when USE_PEXT is defined,
// otherwise classic "fancy" magic bitboards -- both index the same tables.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <sstream>
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _WIN32  // MSVC and MinGW
#include <malloc.h>
inline void* aligned_alloc64(size_t n) { return _aligned_malloc(n, 64); }
inline void aligned_free64(void* p) { _aligned_free(p); }
#else
#include <cstdlib>
inline void* aligned_alloc64(size_t n) { void* p = nullptr; return posix_memalign(&p, 64, n) ? nullptr : p; }
inline void aligned_free64(void* p) { free(p); }
#endif

typedef uint64_t Bitboard;
typedef uint16_t Move;

enum Color { WHITE = 0, BLACK = 1 };
enum PieceType { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING };
constexpr int NO_PIECE = 12;
inline int make_piece(int c, int pt) { return c * 6 + pt; }
inline int type_of(int p) { return p % 6; }
inline int color_of(int p) { return p / 6; }

inline int lsb(Bitboard b) { return (int)_tzcnt_u64(b); }
inline int popcount(Bitboard b) { return (int)_mm_popcnt_u64(b); }
inline int pop_lsb(Bitboard& b) { int s = lsb(b); b &= b - 1; return s; }
inline Bitboard sqbb(int s) { return 1ULL << s; }
inline bool more_than_one(Bitboard b) { return (b & (b - 1)) != 0; }

// ---- move encoding: from | to<<6 | flags<<12 ----
enum MoveFlag {
    MF_QUIET = 0, MF_DOUBLE = 1, MF_KCASTLE = 2, MF_QCASTLE = 3,
    MF_CAPTURE = 4, MF_EP = 5,
    MF_PROMO = 8,            // 8..11 = promo to N,B,R,Q
    MF_PROMO_CAP = 12,       // 12..15 = capture-promo to N,B,R,Q
};
inline Move make_move(int from, int to, int flags) { return (Move)(from | (to << 6) | (flags << 12)); }
inline int from_sq(Move m) { return m & 63; }
inline int to_sq(Move m) { return (m >> 6) & 63; }
inline int flags_of(Move m) { return m >> 12; }
inline bool is_capture(Move m) { return (flags_of(m) & 4) != 0; }
inline bool is_promo(Move m) { return (flags_of(m) & 8) != 0; }
inline int promo_type(Move m) { return (flags_of(m) & 3) + KNIGHT; }
inline bool is_castle(Move m) { int f = flags_of(m); return f == MF_KCASTLE || f == MF_QCASTLE; }

inline std::string move_to_uci(Move m) {
    if (!m) return "0000";
    std::string s;
    s += char('a' + (from_sq(m) & 7)); s += char('1' + (from_sq(m) >> 3));
    s += char('a' + (to_sq(m) & 7));   s += char('1' + (to_sq(m) >> 3));
    if (is_promo(m)) s += "nbrq"[promo_type(m) - KNIGHT];
    return s;
}

// ---- attack tables ----
namespace bb {
extern Bitboard PawnAttacks[2][64], KnightAttacks[64], KingAttacks[64];
extern Bitboard Between[64][64], Line[64][64];

struct SlideEntry { Bitboard mask; Bitboard magic; Bitboard* attacks; int shift; };
extern SlideEntry RookE[64], BishopE[64];

inline unsigned slide_index(const SlideEntry& e, Bitboard occ) {
#ifdef USE_PEXT
    return (unsigned)_pext_u64(occ, e.mask);
#else
    return (unsigned)(((occ & e.mask) * e.magic) >> e.shift);
#endif
}
inline Bitboard rook_attacks(int s, Bitboard occ) { return RookE[s].attacks[slide_index(RookE[s], occ)]; }
inline Bitboard bishop_attacks(int s, Bitboard occ) { return BishopE[s].attacks[slide_index(BishopE[s], occ)]; }
inline Bitboard queen_attacks(int s, Bitboard occ) { return rook_attacks(s, occ) | bishop_attacks(s, occ); }

void init();
}

namespace zob {
extern uint64_t Piece[12][64], Castle[16], Ep[8], Side;
}

struct MoveList {
    Move moves[256];
    int size = 0;
    void add(Move m) { moves[size++] = m; }
};

struct Position {
    Bitboard pieces[12];
    Bitboard byColor[2];
    Bitboard occupied;
    uint64_t key;
    uint64_t pawnKey, npKey[2];  // Zobrist of pawns / of each colour's non-pawns (incl. king), for correction history
    Bitboard checkers, pinned;   // relative to side to move
    uint8_t board[64];
    int stm, castling, ep;       // ep = square or -1 (only set when a capture is actually possible)
    int halfmove, fullmove;

    void set_fen(const std::string& fen);
    bool is_valid() const;
    std::string fen() const;

    Bitboard pcs(int c, int pt) const { return pieces[make_piece(c, pt)]; }
    int king_sq(int c) const { return lsb(pieces[make_piece(c, KING)]); }
    bool has_non_pawn(int c) const {
        return (pcs(c, KNIGHT) | pcs(c, BISHOP) | pcs(c, ROOK) | pcs(c, QUEEN)) != 0;
    }

    Bitboard attackers_to(int s, Bitboard occ) const {
        return (bb::PawnAttacks[BLACK][s] & pieces[make_piece(WHITE, PAWN)])
             | (bb::PawnAttacks[WHITE][s] & pieces[make_piece(BLACK, PAWN)])
             | (bb::KnightAttacks[s] & (pieces[make_piece(WHITE, KNIGHT)] | pieces[make_piece(BLACK, KNIGHT)]))
             | (bb::KingAttacks[s] & (pieces[make_piece(WHITE, KING)] | pieces[make_piece(BLACK, KING)]))
             | (bb::bishop_attacks(s, occ) & (pieces[make_piece(WHITE, BISHOP)] | pieces[make_piece(BLACK, BISHOP)] |
                                              pieces[make_piece(WHITE, QUEEN)] | pieces[make_piece(BLACK, QUEEN)]))
             | (bb::rook_attacks(s, occ) & (pieces[make_piece(WHITE, ROOK)] | pieces[make_piece(BLACK, ROOK)] |
                                            pieces[make_piece(WHITE, QUEEN)] | pieces[make_piece(BLACK, QUEEN)]));
    }
    bool attacked_by(int s, int c, Bitboard occ) const { return (attackers_to(s, occ) & byColor[c]) != 0; }

    void put_piece(int p, int s) {
        pieces[p] |= sqbb(s); byColor[color_of(p)] |= sqbb(s); occupied |= sqbb(s);
        board[s] = (uint8_t)p; key ^= zob::Piece[p][s];
        if (type_of(p) == PAWN) pawnKey ^= zob::Piece[p][s]; else npKey[color_of(p)] ^= zob::Piece[p][s];
    }
    void remove_piece(int s) {
        int p = board[s];
        pieces[p] ^= sqbb(s); byColor[color_of(p)] ^= sqbb(s); occupied ^= sqbb(s);
        board[s] = NO_PIECE; key ^= zob::Piece[p][s];
        if (type_of(p) == PAWN) pawnKey ^= zob::Piece[p][s]; else npKey[color_of(p)] ^= zob::Piece[p][s];
    }

    void compute_check_info();
    void do_move(Move m);
    void do_null();
    bool is_legal(Move m) const;
};

enum GenType { GEN_ALL, GEN_NOISY };
void generate(const Position& pos, MoveList& list, GenType type);
void generate_legal(const Position& pos, MoveList& list);
Move parse_uci_move(const Position& pos, const std::string& s);
