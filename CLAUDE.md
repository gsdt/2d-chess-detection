# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Extract chess positions (FENs) from chess-book PDFs. The repo trains everything from **synthetic data** — there is no annotated dataset. Two models are trained back-to-back: a YOLO11m board detector (finds boards on a page) and an EfficientNet-V2-S piece classifier (13-way: 12 pieces + empty). Inference runs PDF → page images → YOLO bboxes → corner refinement → 8×8 square classification → constraint-checked FEN.

`docs/plan.md` is the original design doc (in Vietnamese) and reflects the intended approach.

## Pipeline orchestration

`run_all.sh` is the source of truth for the end-to-end flow. It activates `.venv` and runs five steps that each correspond to one Python module:

1. `python -m src.synth.gen_detector --pages N --out data/detector` — synthesises YOLO-format page images (1–12 boards composited onto paper-textured backgrounds with fake text). Writes `data/detector/{images,labels}/{train,val}` and `data.yaml`.
2. `python -m src.synth.gen_classifier --boards N --out data/classifier` — renders boards and slices each into 64 square crops, saved into class folders. Important: folder names are case-safe (`wP`, `bP`, …, `empty`) because macOS HFS+/APFS is case-insensitive by default. The mapping back to FEN symbols (`P` upper = white, `p` lower = black) lives in `src/train/train_classifier.py:DIRNAME_TO_LABEL` and `src/pipeline/classify_squares.py:dirname_to_symbol`.
3. `python -m src.train.train_yolo --data data/detector/data.yaml ...` — fine-tunes `yolo11m.pt`. Saves to `data/checkpoints/yolo_runs/<name>/weights/best.pt`. Augmentation is intentionally mild (`fliplr=0`, `degrees=0`) because boards on book pages are upright.
4. `python -m src.train.train_classifier --epochs N --img-size 96` — trains EfficientNet-V2-S to `data/checkpoints/classifier.pt`. Saves a checkpoint dict containing `{model, classes, img_size, val_acc}`; `classes` is the alphabetically-sorted folder list and is the canonical index→class mapping at inference.
5. `python -m src.pipeline.run --pdf <path> --detector-weights ... --classifier-weights ... --out data/output` — full inference. Per page: YOLO detect → `refine_to_inner_board` (Hough lines) → 64-square classify → `constrain_predictions` (rule-based filter) → FEN. Writes overlays, board crops, predicted-board renders, and `results.json`.

Tunable env vars when invoking `run_all.sh`: `PDF`, `DET_PAGES`, `CLF_BOARDS`, `YOLO_EPOCHS`, `CLF_EPOCHS`, `YOLO_IMGSZ`, `YOLO_BATCH`.

## Module map

- `src/synth/render_board.py` — single source of truth for board rendering. `render_board_image` (python-chess + cairosvg + custom CSS palette), `augment` (paper texture, JPEG, blur, noise, slight rotation), `crop_squares` / `square_labels` / `get_inner_board_box`. **Both** dataset generators and the classifier-time crop logic must agree on geometry; if you change `RenderConfig` or the inner-board box, update generators and inference together.
- `src/synth/palettes.py`, `src/synth/positions.py` — colour palettes (light/dark/margin/coord) and FEN-position sampling for synthetic boards.
- `src/pipeline/classify_squares.py` — at inference time the board crop is divided by 8 evenly (no inner-margin handling). If `refine_to_inner_board` ever returns a box that includes the coord margin, classification will misalign.
- `src/pipeline/fen.py` — `constrain_predictions` is a greedy rule filter, not an ILP. It enforces: ≤8 pawns/side, ≤16 pieces/side, no pawn on rank 1/8, exactly 1 king/side. Squares are processed in descending confidence so confident predictions claim rare classes first. Kings are repaired in a second pass.
- `src/pipeline/refine_corners.py` — Hough-based corner refinement; returns the original bbox unchanged when refinement fails (and therefore inference must be tolerant of either a coord-margin-included or coord-stripped crop).

## Data flow gotchas

- **Display order is rank 8 → rank 1, file a → h.** This convention is fixed in `square_labels`, `classify_squares`, and `symbols_to_fen`. When `cfg.flipped=True` the generator reverses all 64 labels (see `gen_classifier.py`); inference does **not** detect orientation, so `flipped` boards produce mirrored FENs. Orientation detection is listed as future work in `docs/plan.md`.
- The classifier checkpoint stores its own `classes` list and `img_size`; the inference path trusts these and rebuilds the model with `pretrained=False`. Don't change `build_model` in a way that breaks `load_state_dict` for older checkpoints.
- `data/`, `runs/`, `*.pt`, `*.log` are gitignored. The `yolo11m.pt` at the repo root is the pretrained YOLO seed and is also gitignored despite being committed-looking in `ls`.

## Environment

- `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Apple-Silicon hosts: training uses `mps` automatically when `torch.backends.mps.is_available()`; YOLO accepts `--device mps` explicitly.
- No tests, linter, or formatter are configured. There is no CI.
