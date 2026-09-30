"""Vectorised bitboard attacks for batches of positions (numpy uint64), a1 = bit 0."""
import numpy as np

U = np.uint64
FILE_A, FILE_H = U(0x0101010101010101), U(0x8080808080808080)
NOT_A, NOT_H = ~FILE_A, ~FILE_H
NOT_AB, NOT_GH = ~(FILE_A | (FILE_A << U(1))), ~(FILE_H | (FILE_H >> U(1)))
ALL = U(0xFFFFFFFFFFFFFFFF)
# (shift, mask applied after the shift) for the 8 ray directions
DIRS = {"N": (8, ALL), "S": (-8, ALL), "E": (1, NOT_A), "W": (-1, NOT_H),
        "NE": (9, NOT_A), "NW": (7, NOT_H), "SE": (-7, NOT_A), "SW": (-9, NOT_H)}


def sh(b, s):
    return (b << U(s)) if s > 0 else (b >> U(-s))


def slide(pieces, empty, dirs):
    att = np.zeros_like(pieces)
    for d in dirs:
        s, m = DIRS[d]
        g = pieces
        for _ in range(7):
            g = sh(g, s) & m
            att |= g
            g &= empty
    return att


def attacked_by(bb, color):
    """bb: dict code -> (B,) uint64 for codes 0..11 (color*6 + type); returns squares attacked by `color`"""
    o = color * 6
    occ = np.zeros_like(bb[0])
    for k in range(12):
        occ |= bb[k]
    empty = ~occ
    p, n, b, r, q, k = (bb[o + i] for i in range(6))
    if color == 0:
        a = ((p & NOT_A) << U(7)) | ((p & NOT_H) << U(9))
    else:
        a = ((p & NOT_A) >> U(9)) | ((p & NOT_H) >> U(7))
    a |= ((n & NOT_H) << U(17)) | ((n & NOT_A) << U(15)) | ((n & NOT_GH) << U(10)) | ((n & NOT_AB) << U(6))
    a |= ((n & NOT_A) >> U(17)) | ((n & NOT_H) >> U(15)) | ((n & NOT_AB) >> U(10)) | ((n & NOT_GH) >> U(6))
    a |= sh(k, 8) | sh(k, -8) | (sh(k, 1) & NOT_A) | (sh(k, -1) & NOT_H) | (sh(k, 9) & NOT_A) | (sh(k, 7) & NOT_H) \
        | (sh(k, -7) & NOT_A) | (sh(k, -9) & NOT_H)
    a |= slide(b | q, empty, ["NE", "NW", "SE", "SW"])
    a |= slide(r | q, empty, ["N", "S", "E", "W"])
    return a


def attacked_flags(pc):
    """pc (B,32) uint16 piece codes -> (B,32) bool: the piece in that slot is attacked by the other colour"""
    pc = pc.astype(np.int64)
    valid = pc != 0xFFFF
    code, sq = np.where(valid, pc >> 6, 0), pc & 63
    bit = np.where(valid, np.left_shift(U(1), sq.astype(np.uint64)), U(0))
    bb = {k: np.bitwise_or.reduce(np.where(code == k, bit, U(0)), axis=1) for k in range(12)}
    att = [attacked_by(bb, 0), attacked_by(bb, 1)]
    opp_att = np.where(code // 6 == 0, att[1][:, None], att[0][:, None])
    return valid & ((opp_att & bit) != 0)
