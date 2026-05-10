"""Sample chess positions for synthetic data.

Two strategies are mixed:
1. Realistic positions reached by self-playing random legal moves from the start.
2. Random "puzzle-like" positions: place a small random selection of pieces.

The mix gives the classifier exposure to dense (opening) and sparse (endgame /
puzzle) boards, which is what the target book shows.
"""
from __future__ import annotations

import random

import chess


PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def random_selfplay(min_plies: int = 0, max_plies: int = 80) -> chess.Board:
    board = chess.Board()
    n = random.randint(min_plies, max_plies)
    for _ in range(n):
        if board.is_game_over():
            break
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(random.choice(moves))
    return board


def random_puzzle_position() -> chess.Board:
    """Sparse-to-medium random position with kings + a handful of other pieces."""
    board = chess.Board.empty()
    squares = list(chess.SQUARES)
    random.shuffle(squares)
    it = iter(squares)

    # Kings: never adjacent.
    while True:
        wk = next(it)
        bk = next(it)
        if chess.square_distance(wk, bk) >= 2:
            board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            break

    n_white = random.randint(0, 12)
    n_black = random.randint(0, 12)

    def add(color: chess.Color, n: int) -> None:
        added = 0
        for sq in list(it):
            if added >= n:
                return
            pt = random.choice(PIECE_TYPES)
            # Skip pawns on rank 1/8.
            rank = chess.square_rank(sq)
            if pt == chess.PAWN and rank in (0, 7):
                continue
            board.set_piece_at(sq, chess.Piece(pt, color))
            added += 1

    add(chess.WHITE, n_white)
    add(chess.BLACK, n_black)
    return board


def sample_board() -> chess.Board:
    """Mix of self-play and puzzle positions."""
    if random.random() < 0.5:
        return random_selfplay()
    return random_puzzle_position()
