"""Render a single 2D chess board PNG from a chess.Board with style variation.

Wraps python-chess `chess.svg.board()` and renders to PNG via cairosvg, then
applies augmentation (paper texture, JPEG, blur, noise, slight rotation).
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass

import cairosvg
import chess
import chess.svg
import numpy as np
from PIL import Image, ImageFilter

from .palettes import PALETTES


PIECES = "PNBRQKpnbrqk"
LABEL_TO_IDX = {"empty": 0, **{p: i + 1 for i, p in enumerate(PIECES)}}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
NUM_CLASSES = 13


@dataclass
class RenderConfig:
    size: int = 384
    coords: bool = True
    flipped: bool = False
    palette_idx: int | None = None  # None → random
    border: bool = True


def _build_css(palette: dict[str, str]) -> str:
    return (
        f".square.light {{ fill: {palette['light']}; }} "
        f".square.dark {{ fill: {palette['dark']}; }} "
        f".margin {{ fill: {palette['margin']}; }} "
        f".coord {{ fill: {palette['coord']}; }} "
    )


def render_board_image(board: chess.Board, cfg: RenderConfig | None = None) -> Image.Image:
    cfg = cfg or RenderConfig()
    palette = PALETTES[cfg.palette_idx if cfg.palette_idx is not None else random.randrange(len(PALETTES))]
    svg = chess.svg.board(
        board,
        size=cfg.size,
        coordinates=cfg.coords,
        flipped=cfg.flipped,
        borders=cfg.border,
        style=_build_css(palette),
    )
    png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=cfg.size, output_height=cfg.size)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return img


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def _paper_overlay(img: Image.Image, strength: float = 0.15) -> Image.Image:
    arr = np.asarray(img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    noise = np.random.normal(0.5, 0.15, (h // 4, w // 4)).astype(np.float32)
    noise = np.clip(noise, 0, 1)
    noise = np.array(Image.fromarray((noise * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    tint = np.array([random.uniform(0.92, 1.0), random.uniform(0.88, 1.0), random.uniform(0.78, 0.95)])
    paper = noise[..., None] * tint[None, None, :]
    out = arr * (1 - strength) + paper * strength
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _jpeg(img: Image.Image, q_lo: int = 35, q_hi: int = 92) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(q_lo, q_hi))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _blur(img: Image.Image) -> Image.Image:
    r = random.uniform(0, 1.2)
    if r < 0.1:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def _gauss_noise(img: Image.Image, sigma: float | None = None) -> Image.Image:
    s = sigma if sigma is not None else random.uniform(0, 6)
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.normal(0, s, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _rotate(img: Image.Image, max_deg: float = 1.5) -> Image.Image:
    angle = random.uniform(-max_deg, max_deg)
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255), expand=False)


def _contrast(img: Image.Image) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    c = random.uniform(0.85, 1.15)
    b = random.uniform(-12, 12)
    arr = np.clip(arr * c + b, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def augment(img: Image.Image, intensity: float = 1.0) -> Image.Image:
    """Apply a stochastic chain of augmentations approximating book-print artifacts."""
    out = img
    if random.random() < 0.7 * intensity:
        out = _contrast(out)
    if random.random() < 0.6 * intensity:
        out = _paper_overlay(out, strength=random.uniform(0.05, 0.25) * intensity)
    if random.random() < 0.4 * intensity:
        out = _blur(out)
    if random.random() < 0.4 * intensity:
        out = _gauss_noise(out)
    if random.random() < 0.3 * intensity:
        out = _rotate(out)
    if random.random() < 0.6 * intensity:
        out = _jpeg(out)
    return out


# ---------------------------------------------------------------------------
# Square crops
# ---------------------------------------------------------------------------

def square_labels(board: chess.Board) -> list[int]:
    """Return 64 labels for squares in row-major order with rank 8 first (display order)."""
    labels = []
    for rank in range(7, -1, -1):
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            labels.append(LABEL_TO_IDX[piece.symbol()] if piece else LABEL_TO_IDX["empty"])
    return labels


def crop_squares(img: Image.Image, board_box: tuple[int, int, int, int]) -> list[Image.Image]:
    """Slice the inner 8x8 board area into 64 PIL crops in display order (rank 8 → 1)."""
    x0, y0, x1, y1 = board_box
    w = (x1 - x0) / 8
    h = (y1 - y0) / 8
    crops = []
    for r in range(8):
        for c in range(8):
            sx0 = int(round(x0 + c * w))
            sy0 = int(round(y0 + r * h))
            sx1 = int(round(x0 + (c + 1) * w))
            sy1 = int(round(y0 + (r + 1) * h))
            crops.append(img.crop((sx0, sy0, sx1, sy1)))
    return crops


def get_inner_board_box(cfg: RenderConfig) -> tuple[int, int, int, int]:
    """Reproduce python-chess SVG layout: 8x8 grid of 45-unit squares.

    With borders+coords the viewBox is 394x394 with a 17-unit margin on each side.
    Without borders the viewBox is 360x360 with no margin.
    """
    if cfg.coords or cfg.border:
        unit = cfg.size / 394.0
        m = int(round(17 * unit))
    else:
        unit = cfg.size / 360.0
        m = 0
    inner = int(round(360 * unit))
    return (m, m, m + inner, m + inner)
