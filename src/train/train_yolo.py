"""Train YOLO11m on the synthetic chess-page detector dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/detector/data.yaml"))
    ap.add_argument("--weights", type=str, default="yolo11m.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", type=str, default="board_detector")
    ap.add_argument(
        "--project",
        type=str,
        default=str(Path("data/checkpoints/yolo_runs").resolve()),
        help="absolute path; ultralytics will save to <project>/<name>/",
    )
    ap.add_argument("--device", type=str, default=None)  # e.g. "0" / "cpu" / "mps"
    args = ap.parse_args()

    model = YOLO(args.weights)
    Path(args.project).mkdir(parents=True, exist_ok=True)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        exist_ok=True,
        # Mild aug — boards are visually dense and large flips/rotations would
        # hurt rather than help, since we know they're upright on a page.
        degrees=0.0,
        translate=0.05,
        scale=0.2,
        fliplr=0.0,
        mosaic=0.5,
        cls=1.0,
        box=7.5,
        patience=10,
        verbose=True,
    )


if __name__ == "__main__":
    main()
