"""Build a FEN string from per-square predictions, applying chess-rule constraints."""
from __future__ import annotations

import chess


PIECES = "PNBRQKpnbrqk"


def symbols_to_fen(symbols: list[str]) -> str:
    """`symbols` is 64-long, display order (rank 8 .. rank 1, file a .. h)."""
    rows = []
    for r in range(8):
        row = symbols[r * 8 : (r + 1) * 8]
        s = ""
        run = 0
        for sq in row:
            if sq == "." or sq == "empty":
                run += 1
            else:
                if run:
                    s += str(run)
                    run = 0
                s += sq
        if run:
            s += str(run)
        rows.append(s)
    placement = "/".join(rows)
    return f"{placement} w - - 0 1"


def constrain_predictions(probs_per_sq: list[list[float]], classes: list[str]) -> list[str]:
    """Greedy constraint enforcement.

    Hard rules applied (relative to display-order squares = rank8→rank1, a→h):
      * exactly one white king and one black king
      * no pawn on rank 1 or rank 8
      * at most 8 pawns per side
      * at most 16 pieces per side total

    For each square take the argmax over allowed classes given current counts.
    Squares are processed in descending order of their max probability so that
    the most confident predictions claim "rare" classes first.
    """
    from .classify_squares import dirname_to_symbol

    sym_per_class = [dirname_to_symbol(c) for c in classes]

    # Index helpers.
    def is_pawn(sym: str) -> bool:
        return sym in ("P", "p")

    def color_of(sym: str) -> str | None:
        if sym == "." or sym == "empty":
            return None
        return "w" if sym.isupper() else "b"

    n_squares = len(probs_per_sq)
    order = sorted(range(n_squares), key=lambda i: -max(probs_per_sq[i]))

    placed: list[str | None] = [None] * n_squares
    counts: dict[str, int] = {p: 0 for p in PIECES}
    color_total = {"w": 0, "b": 0}

    LIMITS = {
        "K": 1, "k": 1,
        "P": 8, "p": 8,
        "Q": 9, "q": 9,
        "R": 10, "r": 10,
        "B": 10, "b": 10,
        "N": 10, "n": 10,
    }

    for idx in order:
        rank = 8 - (idx // 8)  # rank 8 ... 1
        ranked = sorted(
            range(len(classes)),
            key=lambda k: -probs_per_sq[idx][k],
        )
        for k in ranked:
            sym = sym_per_class[k]
            if sym == "." or sym == "empty":
                placed[idx] = "."
                break
            # rule: pawn cannot be on rank 1 or 8
            if is_pawn(sym) and rank in (1, 8):
                continue
            col = color_of(sym)
            if counts[sym] + 1 > LIMITS.get(sym, 16):
                continue
            if col and color_total[col] + 1 > 16:
                continue
            placed[idx] = sym
            counts[sym] += 1
            if col:
                color_total[col] += 1
            break
        if placed[idx] is None:
            placed[idx] = "."

    # Fix kings: each side must have exactly one. If missing, force the highest-prob
    # empty / wrong-class square to become the king.
    for king_sym in ("K", "k"):
        if counts[king_sym] == 1:
            continue
        if king_sym in classes:
            ki = classes.index([c for c in classes if dirname_to_symbol(c) == king_sym][0])
        else:
            continue
        if counts[king_sym] == 0:
            best = max(
                (i for i in range(n_squares) if placed[i] != king_sym),
                key=lambda i: probs_per_sq[i][ki],
                default=None,
            )
            if best is not None:
                old = placed[best]
                placed[best] = king_sym
                if old and old not in (".", "empty"):
                    counts[old] -= 1
                    c = color_of(old)
                    if c:
                        color_total[c] -= 1
                counts[king_sym] += 1
                color_total["w" if king_sym == "K" else "b"] += 1

    return [s if s else "." for s in placed]


def board_from_fen(fen: str) -> chess.Board | None:
    try:
        return chess.Board(fen)
    except (ValueError, chess.InvalidMoveError):
        return None
