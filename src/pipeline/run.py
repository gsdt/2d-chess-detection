"""End-to-end: PDF -> per-page detection -> per-board classification -> FEN.

Outputs:
  data/output/<pdf_stem>/page_<i>_overlay.jpg     # YOLO bbox + index overlay
  data/output/<pdf_stem>/page_<i>_board_<j>.jpg   # board crop
  data/output/<pdf_stem>/page_<i>_board_<j>.png   # python-chess render of predicted FEN
  data/output/<pdf_stem>/results.json             # all FENs + bboxes + confidences
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cairosvg
import chess
import chess.svg
from PIL import Image, ImageDraw

from .board_detect import BoardDetector
from .classify_squares import PieceClassifier
from .fen import constrain_predictions, symbols_to_fen
from .pdf_to_pages import render_pdf
from .refine_corners import refine_to_inner_board


def _draw_overlay(page: Image.Image, boxes: list[tuple[int, int, int, int, float]]) -> Image.Image:
    out = page.copy()
    draw = ImageDraw.Draw(out)
    for i, (x0, y0, x1, y1, conf) in enumerate(boxes):
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=4)
        draw.text((x0 + 4, y0 + 4), f"#{i} {conf:.2f}", fill=(255, 0, 0))
    return out


def _render_fen_png(fen: str, size: int = 256) -> Image.Image:
    placement = fen.split()[0]
    try:
        board = chess.Board(placement + " w - - 0 1")
    except ValueError:
        board = chess.Board.empty()
        # Best-effort: parse only valid-looking pieces.
        ranks = placement.split("/")
        for r_idx, row in enumerate(ranks[:8]):
            file = 0
            for ch in row:
                if ch.isdigit():
                    file += int(ch)
                elif ch in "PNBRQKpnbrqk" and file < 8:
                    sq = chess.square(file, 7 - r_idx)
                    board.set_piece_at(sq, chess.Piece.from_symbol(ch))
                    file += 1
    svg = chess.svg.board(board, size=size)
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=size)
    return Image.open(io.BytesIO(png)).convert("RGB")


def run(args: argparse.Namespace) -> None:
    out_root = args.out / args.pdf.stem
    out_root.mkdir(parents=True, exist_ok=True)

    detector = BoardDetector(args.detector_weights, conf=args.conf, iou=args.iou, imgsz=args.det_imgsz)
    classifier = PieceClassifier(args.classifier_weights, device=args.device)

    page_indices = list(range(*args.pages)) if args.pages else None
    results: list[dict] = []

    for page_idx, page in render_pdf(args.pdf, scale=args.scale, page_indices=page_indices):
        boxes = detector.detect(page)
        boxes.sort(key=lambda b: (b[1] // 50, b[0]))  # reading order: rows then cols

        overlay = _draw_overlay(page, boxes)
        overlay_path = out_root / f"page_{page_idx:03d}_overlay.jpg"
        overlay.save(overlay_path, quality=85)

        page_entry = {"page": page_idx, "boards": []}
        for j, (x0, y0, x1, y1, conf) in enumerate(boxes):
            inner = refine_to_inner_board(page, (x0, y0, x1, y1))
            board_crop = page.crop(inner)
            board_path = out_root / f"page_{page_idx:03d}_board_{j:02d}.jpg"
            board_crop.save(board_path, quality=92)

            symbols, probs = classifier.classify_squares(board_crop)
            constrained = constrain_predictions(probs, classifier.classes)
            fen = symbols_to_fen(constrained)
            raw_fen = symbols_to_fen(symbols)

            try:
                rendered = _render_fen_png(fen)
                rendered.save(out_root / f"page_{page_idx:03d}_board_{j:02d}_pred.png")
            except Exception as e:  # noqa: BLE001
                print(f"  render fail page {page_idx} board {j}: {e}")

            page_entry["boards"].append(
                {
                    "index": j,
                    "yolo_bbox": [x0, y0, x1, y1],
                    "refined_bbox": list(inner),
                    "yolo_conf": conf,
                    "fen": fen,
                    "raw_fen": raw_fen,
                }
            )
            print(f"page {page_idx} board {j}: {fen}")
        results.append(page_entry)

    (out_root / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_root / 'results.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/output"))
    ap.add_argument("--detector-weights", type=Path, required=True)
    ap.add_argument("--classifier-weights", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--det-imgsz", type=int, default=1280)
    ap.add_argument("--pages", type=int, nargs=2, default=None, help="start end (exclusive)")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
