"""Generate the per-square piece-classifier dataset.

For each synthetic board:
  1. Render the board PNG with a random palette / coord setting.
  2. Apply book-style augmentation.
  3. Slice into 64 square crops.
  4. Save crops into class folders.

Layout:
  data/classifier/
    train/{empty,P,N,B,R,Q,K,p,n,b,r,q,k}/*.png
    val/...
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from .render_board import (
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    RenderConfig,
    augment,
    crop_squares,
    get_inner_board_box,
    render_board_image,
    square_labels,
)
from .positions import sample_board


CLASS_DIRS = ["empty"] + list("PNBRQKpnbrqk")


def _safe_dirname(label: str) -> str:
    # "P" / "p" collide on case-insensitive filesystems (macOS default).
    if label == "empty":
        return "empty"
    color = "w" if label.isupper() else "b"
    return f"{color}{label.upper()}"


def ensure_dirs(root: Path) -> None:
    for split in ("train", "val"):
        for label in CLASS_DIRS:
            (root / split / _safe_dirname(label)).mkdir(parents=True, exist_ok=True)


def generate(out_dir: Path, n_boards: int, val_frac: float = 0.1, seed: int = 0) -> None:
    random.seed(seed)
    ensure_dirs(out_dir)

    sizes = [256, 320, 384, 448]

    for board_idx in tqdm(range(n_boards), desc="boards"):
        size = random.choice(sizes)
        cfg = RenderConfig(
            size=size,
            coords=random.random() < 0.7,
            flipped=random.random() < 0.3,
            border=random.random() < 0.8,
        )
        board = sample_board()
        img = render_board_image(board, cfg)
        img = augment(img)
        box = get_inner_board_box(cfg)
        crops = crop_squares(img, box)
        labels = square_labels(board)
        if cfg.flipped:
            # square_labels assumes display-order rank8→rank1, file a→h.
            # When flipped, rank1 is on top and file h is on left, so reverse all 64.
            labels = list(reversed(labels))

        split = "val" if random.random() < val_frac else "train"

        for sq_idx, (crop, lab_idx) in enumerate(zip(crops, labels)):
            label = IDX_TO_LABEL[lab_idx]
            dest = out_dir / split / _safe_dirname(label) / f"b{board_idx:06d}_s{sq_idx:02d}.png"
            crop.save(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/classifier"))
    ap.add_argument("--boards", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.out, args.boards, args.val_frac, args.seed)


if __name__ == "__main__":
    main()
