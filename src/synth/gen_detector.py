"""Generate a YOLO detection dataset of synthetic chess-book pages.

Each page composites 1..12 rendered boards onto a paper-textured background, with
a small amount of fake "text" (rectangles and noise lines) and outputs YOLO-format
bounding-box labels for each board (class 0 = "board").

Layout:
  data/detector/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt
    data.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from .lichess_pieces import available_styles, load_piece_set
from .render_board import RenderConfig, augment, render_board_image
from .positions import sample_board

_PIECE_SETS: list[dict[str, str]] = []  # populated by generate()


PAGE_SIZES = [(900, 1270), (1100, 1500), (820, 1160)]  # roughly book pages


def _paper_bg(size: tuple[int, int]) -> Image.Image:
    w, h = size
    base_tone = random.choice(
        [
            (245, 240, 220),
            (250, 245, 230),
            (255, 252, 240),
            (240, 235, 215),
            (230, 226, 205),
            (255, 255, 255),
        ]
    )
    arr = np.full((h, w, 3), base_tone, dtype=np.float32)
    noise = np.random.normal(0, 6, (h, w, 1))
    arr += noise
    # Add a couple of large soft splotches to mimic uneven aging.
    for _ in range(random.randint(0, 3)):
        cx, cy = random.randint(0, w), random.randint(0, h)
        rad = random.randint(min(w, h) // 6, min(w, h) // 2)
        yy, xx = np.ogrid[:h, :w]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = np.exp(-d2 / (2 * rad * rad))[..., None]
        delta = random.uniform(-12, 6)
        arr += mask * delta
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _draw_fake_text(img: Image.Image) -> None:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    n_lines = random.randint(0, 6)
    for _ in range(n_lines):
        x0 = random.randint(0, w // 3)
        y = random.randint(0, h - 5)
        x1 = x0 + random.randint(80, w // 2)
        thickness = random.randint(2, 5)
        gray = random.randint(40, 110)
        draw.line([(x0, y), (x1, y)], fill=(gray, gray, gray), width=thickness)


def _try_place(boxes: list[tuple[int, int, int, int]], box: tuple[int, int, int, int], pad: int = 8) -> bool:
    bx0, by0, bx1, by1 = box
    for x0, y0, x1, y1 in boxes:
        if not (bx1 + pad <= x0 or x1 + pad <= bx0 or by1 + pad <= y0 or y1 + pad <= by0):
            return False
    return True


def _grid_layout(page_w: int, page_h: int) -> list[tuple[int, int, int, int]]:
    """Return candidate cell boxes for a roughly grid layout (1x1 .. 4x3)."""
    cols = random.choice([1, 2, 3, 3, 3, 4])
    rows = random.choice([1, 2, 2, 3, 3, 4])
    margin_x = random.randint(20, 60)
    margin_y = random.randint(40, 100)
    cell_w = (page_w - margin_x * 2) // cols
    cell_h = (page_h - margin_y * 2) // rows
    cells = []
    for r in range(rows):
        for c in range(cols):
            x0 = margin_x + c * cell_w
            y0 = margin_y + r * cell_h
            cells.append((x0, y0, x0 + cell_w, y0 + cell_h))
    return cells


def _render_one_board(target_size: int) -> Image.Image:
    cfg = RenderConfig(
        size=target_size,
        coords=random.random() < 0.7,
        flipped=random.random() < 0.2,
        border=random.random() < 0.85,
    )
    board = sample_board()
    piece_set = random.choice(_PIECE_SETS) if _PIECE_SETS else None
    img = render_board_image(board, cfg, piece_set=piece_set)
    img = augment(img, intensity=0.6)  # light, full-page aug applied later
    return img


def make_page(min_boards: int = 1, max_boards: int = 12) -> tuple[Image.Image, list[tuple[float, float, float, float]]]:
    page_w, page_h = random.choice(PAGE_SIZES)
    page = _paper_bg((page_w, page_h))
    _draw_fake_text(page)

    cells = _grid_layout(page_w, page_h)
    random.shuffle(cells)
    n_boards = random.randint(min_boards, min(max_boards, len(cells)))
    placed: list[tuple[int, int, int, int]] = []
    bboxes_norm: list[tuple[float, float, float, float]] = []

    for cell in cells[:n_boards]:
        cx0, cy0, cx1, cy1 = cell
        cw, ch = cx1 - cx0, cy1 - cy0
        size = max(120, min(cw, ch) - random.randint(8, 30))
        board_img = _render_one_board(size)
        # Random offset within cell.
        ox = cx0 + random.randint(0, max(0, cw - size))
        oy = cy0 + random.randint(0, max(0, ch - size))
        bx = (ox, oy, ox + size, oy + size)
        if not _try_place(placed, bx):
            continue
        page.paste(board_img, (ox, oy))
        placed.append(bx)
        cx = (ox + size / 2) / page_w
        cy = (oy + size / 2) / page_h
        bw = size / page_w
        bh = size / page_h
        bboxes_norm.append((cx, cy, bw, bh))

    # Page-level aug: slight blur, JPEG compression.
    if random.random() < 0.4:
        page = page.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    return page, bboxes_norm


def write_yaml(root: Path) -> None:
    yaml = (
        f"path: {root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: board\n"
    )
    (root / "data.yaml").write_text(yaml)


def generate(
    out_dir: Path,
    n_pages: int,
    val_frac: float = 0.1,
    seed: int = 0,
    piece_sets_dir: Path | None = None,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    global _PIECE_SETS
    _PIECE_SETS = []
    if piece_sets_dir is not None:
        styles = available_styles(piece_sets_dir)
        print(f"loaded {len(styles)} piece styles from {piece_sets_dir}: {styles}")
        _PIECE_SETS = [load_piece_set(piece_sets_dir, s) for s in styles]
    if not _PIECE_SETS:
        print("no piece sets found — falling back to python-chess cburnett only")

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n_pages), desc="pages"):
        page, bboxes = make_page()
        split = "val" if random.random() < val_frac else "train"
        name = f"page_{i:06d}"
        page.save(out_dir / "images" / split / f"{name}.jpg", quality=random.randint(70, 95))
        with (out_dir / "labels" / split / f"{name}.txt").open("w") as f:
            for cx, cy, bw, bh in bboxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    write_yaml(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/detector"))
    ap.add_argument("--pages", type=int, default=400)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--piece-sets", type=Path, default=Path("data/piece_sets"))
    args = ap.parse_args()
    generate(args.out, args.pages, args.val_frac, args.seed, piece_sets_dir=args.piece_sets)


if __name__ == "__main__":
    main()
