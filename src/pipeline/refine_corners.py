"""Refine a YOLO board bbox to pixel-perfect inner-board corners.

Strategy:
  1. Take the YOLO bbox crop (with a small margin).
  2. Convert to grayscale and Canny-edge it.
  3. Apply Hough-line transform → cluster into near-vertical / near-horizontal lines.
  4. Take outermost lines on each side as the inner-board borders.
  5. Fall back to the YOLO bbox if Hough fails.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _hough_borders(crop_gray: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = crop_gray.shape
    edges = cv2.Canny(crop_gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=int(min(h, w) * 0.4))
    if lines is None:
        return None
    horiz_y: list[float] = []
    vert_x: list[float] = []
    for rho, theta in lines[:, 0]:
        # near-horizontal: theta close to pi/2; near-vertical: theta close to 0 or pi
        if abs(theta - np.pi / 2) < np.deg2rad(8):
            y = rho / np.sin(theta) if np.sin(theta) != 0 else None
            if y is not None and 0 <= y <= h:
                horiz_y.append(y)
        elif theta < np.deg2rad(8) or theta > np.pi - np.deg2rad(8):
            x = rho / np.cos(theta) if np.cos(theta) != 0 else None
            if x is not None and 0 <= x <= w:
                vert_x.append(x)
    if len(horiz_y) < 2 or len(vert_x) < 2:
        return None
    y0 = int(min(horiz_y))
    y1 = int(max(horiz_y))
    x0 = int(min(vert_x))
    x1 = int(max(vert_x))
    if (y1 - y0) < 0.5 * h or (x1 - x0) < 0.5 * w:
        return None
    return (x0, y0, x1, y1)


def refine_to_inner_board(page: Image.Image, bbox: tuple[int, int, int, int], pad: float = 0.06) -> tuple[int, int, int, int]:
    """Given a YOLO bbox on the page, return refined inner-board bbox in page coords.

    The refinement extends the bbox by `pad` (fraction) and runs Hough lines inside
    that crop. Falls back to the original bbox if refinement fails.
    """
    x0, y0, x1, y1 = bbox
    w, h = page.size
    bw, bh = x1 - x0, y1 - y0
    px = int(bw * pad)
    py = int(bh * pad)
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(w, x1 + px), min(h, y1 + py)
    crop = np.asarray(page.crop((cx0, cy0, cx1, cy1)).convert("L"))
    border = _hough_borders(crop)
    if border is None:
        return bbox
    rx0, ry0, rx1, ry1 = border
    return (cx0 + rx0, cy0 + ry0, cx0 + rx1, cy0 + ry1)
