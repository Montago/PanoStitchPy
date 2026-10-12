"""Sea horizons as a levelling cue.

Over open water the horizon is a long, perfectly straight edge at a known elevation (just below level by the
dip angle, sqrt(2h/R)). Where the metadata pose puts the horizon in an image, the strongest edge in a narrow
band around the predicted line is fitted with a straight line; if it is long, straight and sharp (water, not
trees or hills), points on it are handed to the bundle adjustment, which keeps them at that elevation. This
levels images that have few distant feature matches, e.g. panoramas shot from low altitude where nearby
ground shows parallax.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Camera

EARTH_RADIUS_M = 6371000.0


@dataclass
class HorizonLine:
    image: int
    pts: np.ndarray        # (N, 2) full-resolution pixel coordinates on the horizon
    elevation_deg: float   # where the horizon really is (negative: below level)


def dip_deg(altitude_m: float) -> float:
    return -math.degrees(math.acos(EARTH_RADIUS_M / (EARTH_RADIUS_M + max(altitude_m, 0.0))))


def _predicted_row(cam: Camera, xs: np.ndarray, elevation: float) -> np.ndarray | None:
    """Pixel row of the given elevation in each column xs, from the metadata pose (None if not in view)."""
    # rays along the image columns at the image centre row, turned to the target elevation
    pts = np.stack([xs, np.full_like(xs, cam.cy)], axis=1).astype(float)
    rays = cam.rays_world(pts)
    az = np.arctan2(rays[:, 0], rays[:, 2])
    e = math.radians(elevation)
    world = np.stack([np.sin(az) * math.cos(e), np.full_like(az, math.sin(e)), np.cos(az) * math.cos(e)], axis=1)
    c = world @ cam.R.T
    if (c[:, 2] <= 0.1).any():
        return None
    _, v = cam.project(c[:, 0], c[:, 1], c[:, 2])
    return v


def detect(cam: Camera, gray: np.ndarray, scale: float, index: int, altitude_m: float,
           band_deg: float = 2.5, min_coverage: float = 0.7, max_resid_px: float = 1.0,
           min_contrast: float = 6.0) -> HorizonLine | None:
    """Look for a sea horizon in one image. gray: the image (uint8) at `scale` of full resolution."""
    elev = dip_deg(altitude_m)
    xs_full = np.linspace(0.05, 0.95, 60) * (cam.width - 1)
    rows = _predicted_row(cam, xs_full, elev)
    if rows is None or rows.min() < 0.08 * cam.height or rows.max() > 0.92 * cam.height:
        return None
    band = band_deg / math.degrees(1.0 / cam.focal)   # half band in full-res pixels
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    h = gray.shape[0]
    found_x, found_y, strength = [], [], []
    for x, r in zip(xs_full, rows):
        xi = int(round(x * scale))
        y0, y1 = int(max(1, (r - band) * scale)), int(min(h - 2, (r + band) * scale))
        if y1 - y0 < 5:
            continue
        col = np.abs(gy[y0:y1, max(0, xi - 2):xi + 3].mean(axis=1))
        k = int(np.argmax(col))
        if 0 < k < len(col) - 1:   # sub-pixel peak
            a, b, c = col[k - 1], col[k], col[k + 1]
            k = k + 0.5 * (a - c) / max(a - 2 * b + c, -1e9) if (a - 2 * b + c) < 0 else k
        found_x.append(x)
        found_y.append((y0 + k + 0.5) / scale - 0.5)
        strength.append(col.max())
    if len(found_x) < 0.5 * len(xs_full):
        return None
    fx, fy, st = np.array(found_x), np.array(found_y), np.array(strength)
    # robust straight-line fit in the image (the horizon is a near-straight line close to the centre;
    # remaining lens distortion is small at these image positions)
    best = None
    rng = np.random.default_rng(index)
    for _ in range(200):
        i, j = rng.choice(len(fx), 2, replace=False)
        if fx[i] == fx[j]:
            continue
        m = (fy[j] - fy[i]) / (fx[j] - fx[i])
        resid = np.abs(fy - (fy[i] + m * (fx - fx[i])))
        inl = resid < max_resid_px / scale
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < min_coverage * len(xs_full):
        return None
    A = np.stack([fx[best], np.ones(best.sum())], axis=1)
    m, c = np.linalg.lstsq(A, fy[best], rcond=None)[0]
    resid = fy[best] - (m * fx[best] + c)
    if np.sqrt(np.mean(resid ** 2)) > max_resid_px / scale or np.median(st[best]) < min_contrast:
        return None
    span = fx[best].max() - fx[best].min()
    if span < 0.6 * cam.width:
        return None
    px = np.linspace(fx[best].min(), fx[best].max(), 24)
    return HorizonLine(index, np.stack([px, m * px + c], axis=1), elev)


def horizon_gray(path, full_width: int) -> tuple[np.ndarray, float]:
    """The image in grey at about half resolution, and its scale relative to full resolution."""
    from .imageio import imread
    from .raw import is_raw, load_normalised

    if is_raw(path):
        gray = cv2.cvtColor(load_normalised(path, half=True), cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.resize(imread(path, cv2.IMREAD_GRAYSCALE), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return gray, gray.shape[1] / full_width
