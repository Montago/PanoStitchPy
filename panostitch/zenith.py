"""Fill the uncovered sky around the zenith (the drone's gimbal cannot point straight up).

The cap above `cap_lat` is reprojected to a polar (azimuthal equidistant) view looking straight up, where the hole
is a roughly circular blob. The hole is filled with a smooth membrane (harmonic) interpolation of the surrounding
sky, solved coarse-to-fine, and blended back into the equirectangular image with a soft edge.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def _cap_maps(size: int, cap_colat: float, W: int, H: int):
    """Polar image (size x size, zenith in the centre, radius = cap_colat radians) -> equirect sample coords."""
    c = (size - 1) / 2.0
    v, u = np.mgrid[0:size, 0:size].astype(np.float32)
    dx, dy = (u - c) / c, (v - c) / c
    rho = np.sqrt(dx * dx + dy * dy) * cap_colat
    az = np.arctan2(dx, -dy)  # up in the polar image = equirect column 0.5 * W (centre heading)
    map_x = ((az / (2 * math.pi) + 0.5) * W - 0.5).astype(np.float32)
    map_y = (rho / math.pi * H - 0.5).astype(np.float32)
    inside = rho <= cap_colat
    return map_x, map_y, inside


def _diffuse(img: np.ndarray, known: np.ndarray, iters: int) -> np.ndarray:
    out = img.copy()
    for _ in range(iters):
        blurred = cv2.blur(out, (3, 3))
        out = np.where(known[..., None], img, blurred)
    return out


def _membrane_fill(img: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Harmonic fill of the unknown pixels, coarse-to-fine so it converges quickly."""
    h, w = known.shape
    if min(h, w) <= 16:
        mean = img[known].mean(axis=0) if known.any() else np.zeros(3, np.float32)
        start = np.where(known[..., None], img, mean.astype(np.float32))
        return _diffuse(start, known, 200)
    small_w = cv2.resize(known.astype(np.float32), (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    small_img = cv2.resize(img * known[..., None], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    small_known = small_w > 0.5
    small_img = np.where(small_known[..., None], small_img / np.maximum(small_w, 1e-6)[..., None], 0)
    coarse = _membrane_fill(small_img.astype(np.float32), small_known)
    start = np.where(known[..., None], img, cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR))
    return _diffuse(start, known, 40)


def zenith_hole(covered: np.ndarray, cap_lat_deg: float = 45.0) -> np.ndarray | None:
    """Mask of the uncovered area around the zenith if it is the normal gap above the drone (stays above
    cap_lat_deg), otherwise None: no hole, or a hole reaching further down (e.g. images missing because the
    drone stopped early) which should stay black. Other uncovered areas are never part of the mask."""
    H, W = covered.shape
    cap_rows = int(math.ceil((90.0 - cap_lat_deg) / 180.0 * H))
    if covered[0].all():
        return None
    _, labels = cv2.connectedComponents((~covered).astype(np.uint8), connectivity=8)
    top = np.unique(labels[0][~covered[0]])
    # the equirect wraps around horizontally, so components touching both side edges are the same hole
    region = np.isin(labels, top)
    if region[cap_rows:].any():
        return None
    return region


def fill_zenith(pano: np.ndarray, covered: np.ndarray, region: np.ndarray, cap_lat_deg: float = 45.0,
                size: int = 1024, feather_px: float = 24.0) -> np.ndarray:
    """Return a copy of `pano` with `region` (from zenith_hole) filled with smooth sky."""
    H, W = covered.shape
    cap_rows = int(math.ceil((90.0 - cap_lat_deg) / 180.0 * H))
    region = region[:cap_rows]
    # pixels right at the edge of the coverage are contaminated by the black hole (blending, JPEG), so treat a
    # thin margin around the hole as unknown too
    margin = max(3, W // 1000)
    region = cv2.dilate(region.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0
    other_black = ~covered[:cap_rows] & ~region  # e.g. missing images: unknown for the solve, but stays black
    covered = np.vstack([covered[:cap_rows] & ~region, covered[cap_rows:]])

    cap_colat = math.radians(90.0 - cap_lat_deg)
    map_x, map_y, _ = _cap_maps(size, cap_colat, W, H)
    cap = cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP).astype(np.float32)
    cap_cov = cv2.remap(covered.astype(np.uint8) * 255, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_WRAP)
    # the square polar image extends past the disc; everything covered there is a valid boundary value too
    known = cap_cov > 250
    filled = _membrane_fill(np.where(known[..., None], cap, 0).astype(np.float32), known)
    # very light smoothing of the filled area only (diffusion can leave faint radial structure)
    filled = np.where(known[..., None], filled, cv2.GaussianBlur(filled, (0, 0), 3))

    # back to equirect for the cap rows: (lon, lat) -> polar image coordinates
    c = (size - 1) / 2.0
    cols = (np.arange(W, dtype=np.float32) + 0.5) / W * 2 * math.pi - math.pi
    rows = (np.arange(cap_rows, dtype=np.float32) + 0.5) / H * math.pi  # colatitude
    rho = rows[:, None] / cap_colat
    back_x = (c + np.sin(cols)[None, :] * rho * c).astype(np.float32)
    back_y = (c - np.cos(cols)[None, :] * rho * c).astype(np.float32)
    fill_eq = cv2.remap(filled, back_x, back_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # soft transition: inside the hole use the fill, fade into the original over feather_px
    dist = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(dist / feather_px, 0.0, 1.0)  # 0 in the hole, 1 well away from it
    alpha[other_black] = 1.0
    alpha = alpha[..., None]
    top = pano[:cap_rows].astype(np.float32)
    out = pano.copy()
    out[:cap_rows] = np.clip(alpha * top + (1 - alpha) * fill_eq, 0, 255).astype(np.uint8)
    return out
