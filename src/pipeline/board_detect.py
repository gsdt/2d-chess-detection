"""Detect chess-board bounding boxes on a page image using a trained YOLO model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


class BoardDetector:
    def __init__(self, weights: Path, conf: float = 0.25, iou: float = 0.5, imgsz: int = 960):
        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def detect(self, page: Image.Image) -> list[tuple[int, int, int, int, float]]:
        """Return list of (x0, y0, x1, y1, conf) board boxes in pixel coords."""
        arr = np.asarray(page)
        results = self.model.predict(
            arr, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False
        )
        out: list[tuple[int, int, int, int, float]] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for (x0, y0, x1, y1), c in zip(xyxy, confs):
                out.append((int(x0), int(y0), int(x1), int(y1), float(c)))
        return out
