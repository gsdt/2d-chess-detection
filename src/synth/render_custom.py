"""Render a chess board SVG using an arbitrary Lichess piece set.

Replaces python-chess's hardcoded cburnett with composition over any
12-piece SVG set under data/piece_sets/<style>/. Each piece SVG has a
45x45 viewBox; we tile 8x8 squares of 45 units and embed each piece's
inner SVG inside a translated <g>.
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass

import cairosvg
import chess
from PIL import Image

from .palettes import PALETTES


SQUARE = 45  # canonical Lichess piece-SVG viewBox unit
BOARD = 8 * SQUARE  # 360


@dataclass
class CustomRenderConfig:
    size: int = 384
    coords: bool = True
    flipped: bool = False
    palette_idx: int | None = None
    border: bool = True


def _coords_text(file_idx: int, rank_idx: int, flipped: bool) -> tuple[str, str]:
    """Return (file_label, rank_label) for the bottom-left of square at (file, rank)."""
    files = "abcdefgh"
    return (files[file_idx], str(rank_idx + 1))


def build_board_svg(
    board: chess.Board,
    piece_set: dict[str, str],
    cfg: CustomRenderConfig,
    palette: dict[str, str],
) -> str:
    margin = 17 if (cfg.coords or cfg.border) else 0
    total = BOARD + margin * 2

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {total} {total}" width="{cfg.size}" height="{cfg.size}">'
    )

    # Outer margin / border background.
    if margin > 0:
        parts.append(
            f'<rect x="0" y="0" width="{total}" height="{total}" fill="{palette["margin"]}"/>'
        )

    # Squares (display orientation: rank 8 at top, file a at left; flipped reverses both).
    for r in range(8):
        for c in range(8):
            display_rank = 7 - r if not cfg.flipped else r
            display_file = c if not cfg.flipped else 7 - c
            light = (display_file + display_rank) % 2 == 1
            x = margin + c * SQUARE
            y = margin + r * SQUARE
            color = palette["light"] if light else palette["dark"]
            parts.append(
                f'<rect x="{x}" y="{y}" width="{SQUARE}" height="{SQUARE}" fill="{color}"/>'
            )

    # Coordinate labels.
    if cfg.coords and margin > 0:
        files = "abcdefgh"
        # Bottom row file labels and left column rank labels.
        for c in range(8):
            display_file = c if not cfg.flipped else 7 - c
            tx = margin + c * SQUARE + SQUARE - 4
            ty = total - 4
            parts.append(
                f'<text x="{tx}" y="{ty}" font-family="Helvetica,Arial,sans-serif" '
                f'font-size="10" text-anchor="end" fill="{palette["coord"]}">'
                f"{files[display_file]}</text>"
            )
        for r in range(8):
            display_rank = 7 - r if not cfg.flipped else r
            tx = 4
            ty = margin + r * SQUARE + 12
            parts.append(
                f'<text x="{tx}" y="{ty}" font-family="Helvetica,Arial,sans-serif" '
                f'font-size="10" fill="{palette["coord"]}">'
                f"{display_rank + 1}</text>"
            )

    # Pieces.
    for r in range(8):
        for c in range(8):
            display_rank = 7 - r if not cfg.flipped else r
            display_file = c if not cfg.flipped else 7 - c
            sq = chess.square(display_file, display_rank)
            piece = board.piece_at(sq)
            if piece is None:
                continue
            sym = piece.symbol()
            inner = piece_set.get(sym)
            if not inner:
                continue
            x = margin + c * SQUARE
            y = margin + r * SQUARE
            parts.append(f'<g transform="translate({x},{y})">{inner}</g>')

    parts.append("</svg>")
    return "".join(parts)


def render_board_image_custom(
    board: chess.Board,
    piece_set: dict[str, str],
    cfg: CustomRenderConfig | None = None,
) -> Image.Image:
    cfg = cfg or CustomRenderConfig()
    palette = PALETTES[cfg.palette_idx if cfg.palette_idx is not None else random.randrange(len(PALETTES))]
    svg = build_board_svg(board, piece_set, cfg, palette)
    png = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=cfg.size,
        output_height=cfg.size,
    )
    return Image.open(io.BytesIO(png)).convert("RGB")


def get_inner_board_box(cfg: CustomRenderConfig) -> tuple[int, int, int, int]:
    """Inner 8x8 board area in pixel coords for a render of cfg.size."""
    margin = 17 if (cfg.coords or cfg.border) else 0
    total = BOARD + margin * 2
    unit = cfg.size / total
    m = int(round(margin * unit))
    inner = int(round(BOARD * unit))
    return (m, m, m + inner, m + inner)
