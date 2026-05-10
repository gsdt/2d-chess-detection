#!/usr/bin/env bash
# End-to-end pipeline: synth data → train → infer on Step 2 Extra.pdf.
# Uses MPS on Apple Silicon when available.
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

PDF="${PDF:-/Users/gsdt/Downloads/Step 2 Extra.pdf}"

DET_PAGES=${DET_PAGES:-300}
CLF_BOARDS=${CLF_BOARDS:-800}
YOLO_EPOCHS=${YOLO_EPOCHS:-15}
CLF_EPOCHS=${CLF_EPOCHS:-6}
YOLO_IMGSZ=${YOLO_IMGSZ:-960}
YOLO_BATCH=${YOLO_BATCH:-8}

echo "==> 1. generate synthetic detector dataset ($DET_PAGES pages)"
python -m src.synth.gen_detector --pages "$DET_PAGES" --out data/detector

echo "==> 2. generate synthetic classifier dataset ($CLF_BOARDS boards)"
python -m src.synth.gen_classifier --boards "$CLF_BOARDS" --out data/classifier

echo "==> 3. train YOLO11m board detector ($YOLO_EPOCHS epochs, imgsz=$YOLO_IMGSZ)"
python -m src.train.train_yolo \
    --data data/detector/data.yaml \
    --epochs "$YOLO_EPOCHS" --imgsz "$YOLO_IMGSZ" --batch "$YOLO_BATCH" \
    --name board_detector

echo "==> 4. train piece classifier ($CLF_EPOCHS epochs)"
python -m src.train.train_classifier --epochs "$CLF_EPOCHS" --img-size 96

echo "==> 5. run pipeline on $PDF"
python -m src.pipeline.run \
    --pdf "$PDF" \
    --detector-weights data/checkpoints/yolo_runs/board_detector/weights/best.pt \
    --classifier-weights data/checkpoints/classifier.pt \
    --out data/output

echo "done. results in data/output/$(basename "${PDF%.*}")/"
