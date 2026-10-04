#include "position.h"
#include <cctype>

namespace bb {
Bitboard PawnAttacks[2][64], KnightAttacks[64], KingAttacks[64];
Bitboard Between[64][64], Line[64][64];
SlideEntry RookE[64], BishopE[64];
static Bitboard RookTable[0x19000];
static Bitboard BishopTable[0x1480];

static const int RookDirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
static const int BishopDirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

static Bitboard sliding_attack(const int dirs[4][2], int s, Bitboard occ) {
    Bitboard att = 0;
    int r0 = s >> 3, f0 = s & 7;
    for (int d = 0; d < 4; d++) {
        int r = r0 + dirs[d][0], f = f0 + dirs[d][1];
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            Bitboard b = sqbb(r * 8 + f);
            att |= b;
            if (occ & b) break;
            r += dirs[d][0]; f += dirs[d][1];
        }
    }
    return att;
}

struct PRNG {
    uint64_t s;
    uint64_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 2685821657736338717ULL; }
    uint64_t sparse() { return next() & next() & next(); }
};

static void init_slider(SlideEntry* E, Bitboard* table, const int dirs[4][2]) {
    static Bitboard occs[4096], refs[4096];
    static int epoch[4096];
    int cnt = 0, size = 0;
    PRNG rng{0x9E3779B97F4A7C15ULL};
    const Bitboard Rank1 = 0xFFULL, Rank8 = 0xFFULL << 56;
    const Bitboard FileA = 0x0101010101010101ULL, FileH = FileA << 7;
    for (int s = 0; s < 64; s++) {
        Bitboard rankBB = 0xFFULL << (8 * (s >> 3)), fileBB = FileA << (s & 7);
        Bitboard edges = ((Rank1 | Rank8) & ~rankBB) | ((FileA | FileH) & ~fileBB);
        SlideEntry& e = E[s];
        e.mask = sliding_attack(dirs, s, 0) & ~edges;
        e.shift = 64 - popcount(e.mask);
        e.attacks = (s == 0) ? table : E[s - 1].attacks + size;
        Bitboard b = 0;
        size = 0;
        do {
            occs[size] = b;
            refs[size] = sliding_attack(dirs, s, b);
#ifdef USE_PEXT
            e.attacks[_pext_u64(b, e.mask)] = refs[size];
#endif
            size++;
            b = (b - e.mask) & e.mask;
        } while (b);
#ifndef USE_PEXT
        for (int i = 0; i < size;) {
            for (e.magic = 0; popcount((e.magic * e.mask) >> 56) < 6;) e.magic = rng.sparse();
            for (++cnt, i = 0; i < size; ++i) {
                unsigned idx = slide_index(e, occs[i]);
                if (epoch[idx] < cnt) { epoch[idx] = cnt; e.attacks[idx] = refs[i]; }
                else if (e.attacks[idx] != refs[i]) break;
            }
        }
#endif
    }
}

void init() {
    for (int s = 0; s < 64; s++) {
        int r = s >> 3, f = s & 7;
        auto set = [&](Bitboard& t, int dr, int df) {
            int rr = r + dr, ff = f + df;
            if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) t |= sqbb(rr * 8 + ff);
        };
        PawnAttacks[WHITE][s] = 0; PawnAttacks[BLACK][s] = 0; KnightAttacks[s] = 0; KingAttacks[s] = 0;
        set(PawnAttacks[WHITE][s], 1, -1); set(PawnAttacks[WHITE][s], 1, 1);
        set(PawnAttacks[BLACK][s], -1, -1); set(PawnAttacks[BLACK][s], -1, 1);
        const int kn[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
        for (auto& d : kn) set(KnightAttacks[s], d[0], d[1]);
        for (int dr = -1; dr <= 1; dr++) for (int df = -1; df <= 1; df++) if (dr || df) set(KingAttacks[s], dr, df);
    }
    init_slider(RookE, RookTable, RookDirs);
    init_slider(BishopE, BishopTable, BishopDirs);
    for (int a = 0; a < 64; a++) for (int b2 = 0; b2 < 64; b2++) {
        Between[a][b2] = Line[a][b2] = 0;
        if (a == b2) continue;
        if (bishop_attacks(a, 0) & sqbb(b2)) {
            Line[a][b2] = (bishop_attacks(a, 0) & bishop_attacks(b2, 0)) | sqbb(a) | sqbb(b2);
            Between[a][b2] = bishop_attacks(a, sqbb(b2)) & bishop_attacks(b2, sqbb(a));
        } else if (rook_attacks(a, 0) & sqbb(b2)) {
            Line[a][b2] = (rook_attacks(a, 0) & rook_attacks(b2, 0)) | sqbb(a) | sqbb(b2);
            Between[a][b2] = rook_attacks(a, sqbb(b2)) & rook_attacks(b2, sqbb(a));
        }
    }
    // zobrist
    PRNG rng{1070372ULL};
    for (int p = 0; p < 12; p++) for (int s = 0; s < 64; s++) zob::Piece[p][s] = rng.next();
    for (int i = 0; i < 16; i++) zob::Castle[i] = rng.next();
    for (int i = 0; i < 8; i++) zob::Ep[i] = rng.next();
    zob::Side = rng.next();
}
}  // namespace bb

namespace zob {
uint64_t Piece[12][64], Castle[16], Ep[8], Side;
}

static int CastleMask[64];
static struct CastleMaskInit {
    CastleMaskInit() {
        for (int i = 0; i < 64; i++) CastleMask[i] = 15;
        CastleMask[4] = 15 & ~3;   // e1
        CastleMask[7] = 15 & ~1;   // h1
        CastleMask[0] = 15 & ~2;   // a1
        CastleMask[60] = 15 & ~12; // e8
        CastleMask[63] = 15 & ~4;  // h8
        CastleMask[56] = 15 & ~8;  // a8
    }
} castleMaskInit;

static int piece_from_char(char c) {
    int color = isupper((unsigned char)c) ? WHITE : BLACK;
    switch (tolower((unsigned char)c)) {
        case 'p': return make_piece(color, PAWN);
        case 'n': return make_piece(color, KNIGHT);
        case 'b': return make_piece(color, BISHOP);
        case 'r': return make_piece(color, ROOK);
        case 'q': return make_piece(color, QUEEN);
        case 'k': return make_piece(color, KING);
    }
    return NO_PIECE;
}

bool Position::is_valid() const {
    if (popcount(pcs(WHITE, KING)) != 1 || popcount(pcs(BLACK, KING)) != 1) return false;
    if ((pcs(WHITE, PAWN) | pcs(BLACK, PAWN)) & 0xFF000000000000FFULL) return false;
    return !attacked_by(king_sq(stm ^ 1), stm, occupied);  // side not to move can't be in check
}

void Position::set_fen(const std::string& fen) {
    std::memset(pieces, 0, sizeof(pieces));
    byColor[0] = byColor[1] = occupied = 0;
    key = 0; pawnKey = 0; npKey[0] = npKey[1] = 0;
    for (int i = 0; i < 64; i++) board[i] = NO_PIECE;
    std::istringstream is(fen);
    std::string bpart, stmPart, castlePart, epPart;
    is >> bpart >> stmPart >> castlePart >> epPart;
    if (!(is >> halfmove)) halfmove = 0;
    if (!(is >> fullmove)) fullmove = 1;
    int r = 7, f = 0;
    for (char c : bpart) {
        if (c == '/') { r--; f = 0; }
        else if (isdigit((unsigned char)c)) f += c - '0';
        else { put_piece(piece_from_char(c), r * 8 + f); f++; }
    }
    stm = (stmPart == "b") ? BLACK : WHITE;
    if (stm == BLACK) key ^= zob::Side;
    castling = 0;
    for (char c : castlePart) {
        if (c == 'K') castling |= 1; else if (c == 'Q') castling |= 2;
        else if (c == 'k') castling |= 4; else if (c == 'q') castling |= 8;
    }
    key ^= zob::Castle[castling];
    ep = -1;
    if (epPart.size() == 2 && epPart != "-") {
        int s = (epPart[1] - '1') * 8 + (epPart[0] - 'a');
        if (s >= 0 && s < 64 && (bb::PawnAttacks[stm ^ 1][s] & pcs(stm, PAWN))) { ep = s; key ^= zob::Ep[s & 7]; }
    }
    if (popcount(pcs(WHITE, KING)) == 1 && popcount(pcs(BLACK, KING)) == 1) compute_check_info();
}

std::string Position::fen() const {
    std::string s;
    const char* pc = "PNBRQKpnbrqk";
    for (int r = 7; r >= 0; r--) {
        int empty = 0;
        for (int f = 0; f < 8; f++) {
            int p = board[r * 8 + f];
            if (p == NO_PIECE) { empty++; continue; }
            if (empty) { s += char('0' + empty); empty = 0; }
            s += pc[p];
        }
        if (empty) s += char('0' + empty);
        if (r) s += '/';
    }
    s += stm == WHITE ? " w " : " b ";
    std::string c;
    if (castling & 1) c += 'K'; if (castling & 2) c += 'Q';
    if (castling & 4) c += 'k'; if (castling & 8) c += 'q';
    s += c.empty() ? "-" : c;
    s += ' ';
    if (ep >= 0) { s += char('a' + (ep & 7)); s += char('1' + (ep >> 3)); } else s += '-';
    s += " " + std::to_string(halfmove) + " " + std::to_string(fullmove);
    return s;
}

void Position::compute_check_info() {
    int us = stm, them = stm ^ 1;
    int ksq = king_sq(us);
    checkers = attackers_to(ksq, occupied) & byColor[them];
    pinned = 0;
    Bitboard snipers = (bb::rook_attacks(ksq, 0) & (pcs(them, ROOK) | pcs(them, QUEEN)))
                     | (bb::bishop_attacks(ksq, 0) & (pcs(them, BISHOP) | pcs(them, QUEEN)));
    while (snipers) {
        int s = pop_lsb(snipers);
        Bitboard b = bb::Between[ksq][s] & occupied;
        if (b && !more_than_one(b) && (b & byColor[us])) pinned |= b;
    }
}

void Position::do_move(Move m) {
    int us = stm, them = us ^ 1;
    int from = from_sq(m), to = to_sq(m), fl = flags_of(m);
    int pc = board[from], pt = type_of(pc);

    if (ep >= 0) { key ^= zob::Ep[ep & 7]; ep = -1; }
    key ^= zob::Castle[castling];
    halfmove++;

    if (fl == MF_EP) { remove_piece(to ^ 8); halfmove = 0; }
    else if (fl & 4) { remove_piece(to); halfmove = 0; }

    remove_piece(from);
    put_piece(is_promo(m) ? make_piece(us, promo_type(m)) : pc, to);

    if (pt == PAWN) {
        halfmove = 0;
        if (fl == MF_DOUBLE) {
            int e = to ^ 8;
            if (bb::PawnAttacks[us][e] & pcs(them, PAWN)) { ep = e; key ^= zob::Ep[e & 7]; }
        }
    } else if (fl == MF_KCASTLE) {
        int rfrom = to + 1, rto = to - 1;
        int rook = board[rfrom]; remove_piece(rfrom); put_piece(rook, rto);
    } else if (fl == MF_QCASTLE) {
        int rfrom = to - 2, rto = to + 1;
        int rook = board[rfrom]; remove_piece(rfrom); put_piece(rook, rto);
    }

    castling &= CastleMask[from] & CastleMask[to];
    key ^= zob::Castle[castling];
    stm = them;
    key ^= zob::Side;
    if (us == BLACK) fullmove++;
    compute_check_info();
}

void Position::do_null() {
    if (ep >= 0) { key ^= zob::Ep[ep & 7]; ep = -1; }
    stm ^= 1;
    key ^= zob::Side;
    halfmove++;
    compute_check_info();
}

bool Position::is_legal(Move m) const {
    int us = stm, them = us ^ 1;
    int from = from_sq(m), to = to_sq(m);
    int ksq = king_sq(us);
    if (flags_of(m) == MF_EP) {
        int capsq = to ^ 8;
        if (checkers & ~sqbb(capsq) & (pcs(them, PAWN) | pcs(them, KNIGHT))) return false;
        Bitboard occ = (occupied ^ sqbb(from) ^ sqbb(capsq)) | sqbb(to);
        return !(bb::bishop_attacks(ksq, occ) & (pcs(them, BISHOP) | pcs(them, QUEEN)))
            && !(bb::rook_attacks(ksq, occ) & (pcs(them, ROOK) | pcs(them, QUEEN)));
    }
    if (type_of(board[from]) == KING) {
        if (is_castle(m)) return true;  // generated only when fully legal
        return !(attackers_to(to, occupied ^ sqbb(from)) & byColor[them]);
    }
    if (checkers) {
        if (more_than_one(checkers)) return false;
        int c = lsb(checkers);
        if (!((bb::Between[ksq][c] | sqbb(c)) & sqbb(to))) return false;
    }
    return !(pinned & sqbb(from)) || (bb::Line[ksq][from] & sqbb(to));
}

static const Bitboard FILE_A = 0x0101010101010101ULL, FILE_H = FILE_A << 7;
static const Bitboard RANK_2 = 0xFFULL << 8, RANK_3 = 0xFFULL << 16, RANK_6 = 0xFFULL << 40, RANK_7 = 0xFFULL << 48;

static inline void add_promos(MoveList& list, int from, int to, bool cap, bool queenOnly) {
    int base = cap ? MF_PROMO_CAP : MF_PROMO;
    list.add(make_move(from, to, base + 3));
    if (!queenOnly) {
        list.add(make_move(from, to, base + 0));
        list.add(make_move(from, to, base + 1));
        list.add(make_move(from, to, base + 2));
    }
}

void generate(const Position& pos, MoveList& list, GenType type) {
    list.size = 0;
    int us = pos.stm, them = us ^ 1;
    Bitboard occ = pos.occupied, ours = pos.byColor[us], empty = ~occ;
    Bitboard theirs = pos.byColor[them] & ~pos.pcs(them, KING);  // never generate king captures
    Bitboard targets = (type == GEN_ALL) ? (~ours & ~pos.pcs(them, KING)) : theirs;
    bool noisyOnly = type == GEN_NOISY;

    // pawns
    Bitboard pawns = pos.pcs(us, PAWN);
    Bitboard rank7 = us == WHITE ? RANK_7 : RANK_2, rank3 = us == WHITE ? RANK_3 : RANK_6;
    Bitboard nonPromo = pawns & ~rank7, promo = pawns & rank7;
    int up = us == WHITE ? 8 : -8;
    auto shiftUp = [&](Bitboard b) { return us == WHITE ? b << 8 : b >> 8; };

    if (!noisyOnly) {
        Bitboard single = shiftUp(nonPromo) & empty;
        Bitboard dbl = shiftUp(single & rank3) & empty;
        while (single) { int to = pop_lsb(single); list.add(make_move(to - up, to, MF_QUIET)); }
        while (dbl) { int to = pop_lsb(dbl); list.add(make_move(to - 2 * up, to, MF_DOUBLE)); }
    }
    // captures: "west" = toward file a, "east" = toward file h
    Bitboard capW, capE;
    int dW, dE;
    if (us == WHITE) { capW = (nonPromo & ~FILE_A) << 7; dW = 7; capE = (nonPromo & ~FILE_H) << 9; dE = 9; }
    else { capW = (nonPromo & ~FILE_A) >> 9; dW = -9; capE = (nonPromo & ~FILE_H) >> 7; dE = -7; }
    capW &= theirs; capE &= theirs;
    while (capW) { int to = pop_lsb(capW); list.add(make_move(to - dW, to, MF_CAPTURE)); }
    while (capE) { int to = pop_lsb(capE); list.add(make_move(to - dE, to, MF_CAPTURE)); }

    if (promo) {
        Bitboard push = shiftUp(promo) & empty;
        Bitboard pW, pE;
        if (us == WHITE) { pW = ((promo & ~FILE_A) << 7) & theirs; pE = ((promo & ~FILE_H) << 9) & theirs; }
        else { pW = ((promo & ~FILE_A) >> 9) & theirs; pE = ((promo & ~FILE_H) >> 7) & theirs; }
        while (push) { int to = pop_lsb(push); add_promos(list, to - up, to, false, noisyOnly); }
        while (pW) { int to = pop_lsb(pW); add_promos(list, to - dW, to, true, noisyOnly); }
        while (pE) { int to = pop_lsb(pE); add_promos(list, to - dE, to, true, noisyOnly); }
    }
    if (pos.ep >= 0) {
        Bitboard att = bb::PawnAttacks[them][pos.ep] & nonPromo;
        while (att) { int from = pop_lsb(att); list.add(make_move(from, pos.ep, MF_EP)); }
    }

    auto addPiece = [&](int from, Bitboard att) {
        att &= targets;
        while (att) {
            int to = pop_lsb(att);
            list.add(make_move(from, to, pos.board[to] != NO_PIECE ? MF_CAPTURE : MF_QUIET));
        }
    };
    Bitboard b = pos.pcs(us, KNIGHT);
    while (b) { int s = pop_lsb(b); addPiece(s, bb::KnightAttacks[s]); }
    b = pos.pcs(us, BISHOP);
    while (b) { int s = pop_lsb(b); addPiece(s, bb::bishop_attacks(s, occ)); }
    b = pos.pcs(us, ROOK);
    while (b) { int s = pop_lsb(b); addPiece(s, bb::rook_attacks(s, occ)); }
    b = pos.pcs(us, QUEEN);
    while (b) { int s = pop_lsb(b); addPiece(s, bb::queen_attacks(s, occ)); }
    int ksq = pos.king_sq(us);
    addPiece(ksq, bb::KingAttacks[ksq]);

    if (!noisyOnly && !pos.checkers) {
        if (us == WHITE) {
            if ((pos.castling & 1) && !(occ & (sqbb(5) | sqbb(6))) &&
                !pos.attacked_by(5, them, occ) && !pos.attacked_by(6, them, occ))
                list.add(make_move(4, 6, MF_KCASTLE));
            if ((pos.castling & 2) && !(occ & (sqbb(1) | sqbb(2) | sqbb(3))) &&
                !pos.attacked_by(3, them, occ) && !pos.attacked_by(2, them, occ))
                list.add(make_move(4, 2, MF_QCASTLE));
        } else {
            if ((pos.castling & 4) && !(occ & (sqbb(61) | sqbb(62))) &&
                !pos.attacked_by(61, them, occ) && !pos.attacked_by(62, them, occ))
                list.add(make_move(60, 62, MF_KCASTLE));
            if ((pos.castling & 8) && !(occ & (sqbb(57) | sqbb(58) | sqbb(59))) &&
                !pos.attacked_by(59, them, occ) && !pos.attacked_by(58, them, occ))
                list.add(make_move(60, 58, MF_QCASTLE));
        }
    }
}

void generate_legal(const Position& pos, MoveList& list) {
    MoveList pseudo;
    generate(pos, pseudo, GEN_ALL);
    list.size = 0;
    for (int i = 0; i < pseudo.size; i++)
        if (pos.is_legal(pseudo.moves[i])) list.add(pseudo.moves[i]);
}

Move parse_uci_move(const Position& pos, const std::string& s) {
    MoveList list;
    generate_legal(pos, list);
    for (int i = 0; i < list.size; i++)
        if (move_to_uci(list.moves[i]) == s) return list.moves[i];
    return 0;
}
