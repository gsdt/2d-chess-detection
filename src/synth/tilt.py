"""Rotation + perspective skew utilities for synthetic board augmentation.

Given a square board PIL image, return a transformed image plus the 4 corner
positions (in the new image's pixel space) so the detector dataset can record
where the inner 8x8 lives even after the board is tilted on the page.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image


def _rotation_matrix(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def random_tilt(
    board_img: Image.Image,
    max_rot: float = 12.0,
    max_skew: float = 0.06,
) -> tuple[Image.Image, np.ndarray]:
    """Apply random rotation + slight perspective skew to a square board.

    Returns:
        (out_img, corners) where `out_img` is RGBA (transparent outside the
        board quad so it can be alpha-pasted onto any background) and
        `corners` is a (4,2) float array of [TL, TR, BR, BL] in the OUTPUT
        image's pixel coordinates.
    """
    if board_img.mode != "RGBA":
        board_img = board_img.convert("RGBA")
    w, h = board_img.size
    src_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)

    # 1) random perspective skew: nudge each corner by ±max_skew * size
    skew_amp = max_skew * max(w, h)
    skewed = src_corners + np.random.uniform(-skew_amp, skew_amp, size=src_corners.shape)

    # 2) rotate around the centroid
    angle = random.uniform(-max_rot, max_rot)
    R = _rotation_matrix(angle)
    centroid = skewed.mean(axis=0)
    rotated = (skewed - centroid) @ R.T + centroid

    # 3) translate so all corners are >=0 and compute output canvas size
    mn = rotated.min(axis=0)
    mx = rotated.max(axis=0)
    rotated -= mn
    out_w = int(math.ceil(mx[0] - mn[0]))
    out_h = int(math.ceil(mx[1] - mn[1]))

    # PIL's Image.transform with PERSPECTIVE wants the inverse mapping
    # (output → input). Compute it from the 4-point correspondence.
    src = src_corners.astype(np.float32)
    dst = rotated.astype(np.float32)
    coeffs = _perspective_coeffs(dst, src)

    out = board_img.transform(
        (out_w, out_h),
        Image.PERSPECTIVE,
        coeffs,
        resample=Image.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )
    return out, rotated.astype(np.float32)


def _perspective_coeffs(src: np.ndarray, dst: np.ndarray) -> tuple[float, ...]:
    """Compute the 8 coefficients PIL needs for an output→input perspective map."""
    A = []
    B = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y])
        A.append([0, 0, 0, X, Y, 1, -y * X, -y * Y])
        B.extend([x, y])
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    coeffs = np.linalg.solve(A, B)
    return tuple(coeffs.tolist())


def axis_aligned_bbox(corners: np.ndarray) -> tuple[int, int, int, int]:
    """Return the (x0,y0,x1,y1) axis-aligned bbox enclosing 4 corners."""
    x0 = int(math.floor(corners[:, 0].min()))
    y0 = int(math.floor(corners[:, 1].min()))
    x1 = int(math.ceil(corners[:, 0].max()))
    y1 = int(math.ceil(corners[:, 1].max()))
    return x0, y0, x1, y1
