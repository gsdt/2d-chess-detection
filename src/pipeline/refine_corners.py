"""Refine a YOLO bbox to the 4 actual corners of the (possibly tilted) board.

The detector is trained to put an axis-aligned bbox around the tilted board, so
the bbox contains some off-board paper. We need the inner 4 corners to
perspective-rectify the board into a clean upright square before slicing into
the 8x8 grid.

Strategy:
  1. Take the YOLO bbox crop with a small padding margin.
  2. Threshold + Canny to expose the board outline.
  3. Find contours, keep the largest closed quadrilateral that covers most of
     the crop area.
  4. Order its corners (TL, TR, BR, BL) and translate back to page coords.
  5. Fall back to the axis-aligned bbox if no quad is found.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return the 4 points reordered as [TL, TR, BR, BL]."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array(
        [
            pts[np.argmin(s)],   # TL  (small x + small y)
            pts[np.argmin(d)],   # TR  (large x - small y → small (y - x))
            pts[np.argmax(s)],   # BR  (large x + large y)
            pts[np.argmax(d)],   # BL  (small x - large y → large (y - x))
        ],
        dtype=np.float32,
    )


def _find_quad(crop_gray: np.ndarray, min_area_frac: float = 0.25) -> np.ndarray | None:
    h, w = crop_gray.shape
    blurred = cv2.GaussianBlur(crop_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    min_area = min_area_frac * w * h

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        # Fit the convex hull, then try to simplify to 4 vertices.
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        for eps_factor in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
            approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return _order_corners(approx.reshape(4, 2))
        # Fall back: minimum-area rotated rectangle around the contour. This is
        # robust to extra ridges in the convex hull and works well for the
        # genuinely-rectangular chess board.
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        if cv2.contourArea(box) >= min_area:
            return _order_corners(box)
    return None


def find_board_corners(
    page: Image.Image,
    bbox: tuple[int, int, int, int],
    pad: float = 0.06,
) -> np.ndarray:
    """Return the 4 board corners in page-pixel coords, ordered TL,TR,BR,BL.

    Falls back to the axis-aligned bbox corners if quad detection fails.
    """
    x0, y0, x1, y1 = bbox
    w, h = page.size
    bw, bh = x1 - x0, y1 - y0
    px = int(bw * pad)
    py = int(bh * pad)
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(w, x1 + px), min(h, y1 + py)
    crop_gray = np.asarray(page.crop((cx0, cy0, cx1, cy1)).convert("L"))
    quad = _find_quad(crop_gray)
    if quad is None:
        return np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            dtype=np.float32,
        )
    quad[:, 0] += cx0
    quad[:, 1] += cy0
    return quad


def rectify_board(
    page: Image.Image,
    corners: np.ndarray,
    out_size: int = 384,
) -> Image.Image:
    """Warp the quadrilateral defined by `corners` (TL,TR,BR,BL) to an upright square."""
    src = corners.astype(np.float32)
    dst = np.array(
        [[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    page_arr = np.asarray(page.convert("RGB"))
    warped = cv2.warpPerspective(page_arr, M, (out_size, out_size), flags=cv2.INTER_CUBIC)
    return Image.fromarray(warped)


# Back-compat alias retained for callers expecting a bbox.
def refine_to_inner_board(
    page: Image.Image,
    bbox: tuple[int, int, int, int],
    pad: float = 0.06,
) -> tuple[int, int, int, int]:
    corners = find_board_corners(page, bbox, pad=pad)
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    return int(x0), int(y0), int(x1), int(y1)
