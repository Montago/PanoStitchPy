"""Panorama types and output layout: which kind of panorama a set of shots is, which canvas (projection,
orientation, extent) to render it on, and how to crop away the uncovered borders.

DJI panorama modes (Air 3). The drone starts from the direction/tilt the pilot points at and shoots the
other images around that first one:
    sphere    33 images, 8 columns x 4 rows + nadir    -> full 360 x 180 equirectangular
    180       21 images, 7 columns x 3 rows            -> ~250 x 100 degrees, equirectangular, cropped
    wide       9 images, 3 x 3                         -> ~130 x 110 degrees, equirectangular, cropped
    vertical   3 images, 1 column                      -> tall image, transverse projection, cropped
"""
from __future__ import annotations

import math

import numpy as np

from .camera import Camera
from .projection import Canvas, build_maps

PANO_TYPES = {
    "sphere": "360° kugle",
    "180": "180° panorama",
    "wide": "Vidvinkel",
    "vertical": "Lodret",
}
OUTPUT_SUFFIX = {"sphere": "360", "180": "180", "wide": "vidvinkel", "vertical": "lodret"}


def _columns(yaws: list[float], tol: float = 6.0) -> list[float]:
    """Cluster headings into shooting columns (circular), returning the column headings."""
    cols: list[float] = []
    for y in sorted(v % 360 for v in yaws):
        if cols and abs((y - cols[-1] + 180) % 360 - 180) < tol:
            continue
        cols.append(y)
    if len(cols) > 1 and abs((cols[0] - cols[-1] + 180) % 360 - 180) < tol:
        cols.pop()
    return cols


def _largest_gap(headings: list[float]) -> tuple[float, float]:
    """Largest circular gap between sorted headings: (gap size, heading where the gap starts)."""
    hs = sorted(h % 360 for h in headings)
    gaps = [((hs[(i + 1) % len(hs)] - hs[i]) % 360 or 360, hs[i]) for i in range(len(hs))]
    return max(gaps)


def detect_type(yaws: list[float], hfov_deg: float = 72.0) -> str:
    """Panorama type from the gimbal headings of the shots."""
    cols = _columns(yaws)
    if len(cols) <= 1:
        return "vertical"
    gap, _ = _largest_gap(cols)
    if gap < hfov_deg:          # columns go all the way round
        return "sphere"
    span = 360 - gap + hfov_deg
    return "180" if span >= 180 else "wide"


def coverage_estimate_deg(yaws: list[float], hfov_deg: float = 72.0) -> float:
    """Rough horizontal coverage from the shooting columns (for labels)."""
    cols = _columns(yaws)
    if len(cols) == 1:
        return hfov_deg
    gap, _ = _largest_gap(cols)
    return 360.0 if gap < hfov_deg else 360 - gap + hfov_deg


def _coverage(cams: list[Camera], canvas: Canvas) -> np.ndarray:
    rays = canvas.rays()
    cov = np.zeros((canvas.height, canvas.width), bool)
    for cam in cams:
        cov |= build_maps(cam, rays)[2]
    return cov


def plan_canvas(cams: list[Camera], pano_type: str, deg_per_px: float, sphere_width: int,
                margin_deg: float = 0.5) -> Canvas:
    """The output canvas for the aligned cameras. deg_per_px sets the resolution of partial panoramas;
    a sphere is always sphere_width wide (which must suit the multi-band blender)."""
    yaws = [c.euler()[0] for c in cams]
    if pano_type == "sphere":
        return Canvas.full_sphere(sphere_width, center_yaw=yaws[0])  # centred on the start direction

    # centre: middle of the arc of camera headings (the complement of the largest gap between them)
    gap, gap_start = _largest_gap(yaws)
    center_yaw = 0.0 if gap >= 360 else (gap_start + gap + (360 - gap) / 2 + 180) % 360 - 180
    if len(_columns(yaws)) <= 1:
        center_yaw = math.degrees(math.atan2(np.mean(np.sin(np.radians(yaws))), np.mean(np.cos(np.radians(yaws)))))

    if pano_type == "vertical":
        pitches = [c.euler()[1] for c in cams]
        frame = Canvas("equirect", 360, 720, center_yaw, (max(pitches) + min(pitches)) / 2, True)
    else:
        frame = Canvas("equirect", 720, 360, center_yaw)
    cov = _coverage(cams, frame)
    if frame.portrait:
        cov = np.rot90(cov, k=-1)  # back to the unturned (landscape) layout used by lons()/lats()
    lon = np.degrees(frame.lons())
    lat = np.degrees(frame.lats())
    # near the poles of the frame a small area spans many longitudes, so measure the width away from them
    mid = np.abs(lat) <= 70
    cols = np.where(cov[mid].any(axis=0))[0]
    rows = np.where(cov.any(axis=1))[0]
    if len(cols) == 0 or len(rows) == 0:
        raise ValueError("Billederne dækker ikke noget område – er vinklerne i metadataen gyldige?")
    step = 360.0 / 720
    lon_left = max(-180.0, lon[cols.min()] - step / 2 - margin_deg)
    lon_right = min(180.0, lon[cols.max()] + step / 2 + margin_deg)
    lat_top = min(90.0, lat[rows.min()] + step / 2 + margin_deg)
    lat_bottom = max(-90.0, lat[rows.max()] - step / 2 - margin_deg)
    return Canvas.for_region("equirect", deg_per_px, center_yaw, frame.center_pitch, frame.portrait,
                             lon_left, lon_right, lat_top, lat_bottom)


def largest_filled_rect(mask: np.ndarray, max_side: int = 1200) -> tuple[int, int, int, int]:
    """Largest axis-aligned rectangle (x0, y0, x1, y1) that is completely True, found on a downscaled copy and
    shrunk conservatively when scaled back."""
    import cv2

    H, W = mask.shape
    f = min(1.0, max_side / max(H, W))
    small = cv2.resize(mask.astype(np.float32), (max(1, int(W * f)), max(1, int(H * f))),
                       interpolation=cv2.INTER_AREA) >= 0.999
    h, w = small.shape
    heights = np.zeros(w, int)
    best = (0, 0, 0, 0, 0)  # area, x0, y0, x1, y1
    for y in range(h):
        heights = np.where(small[y], heights + 1, 0)
        stack: list[int] = []
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            while stack and heights[stack[-1]] >= cur:
                top = stack.pop()
                height = heights[top]
                left = stack[-1] + 1 if stack else 0
                area = height * (x - left)
                if area > best[0]:
                    best = (area, left, y - height + 1, x, y + 1)
            stack.append(x)
    _, x0, y0, x1, y1 = best
    if best[0] == 0:
        return 0, 0, W, H
    # scale back, staying inside (one small pixel of safety on each side)
    sx, sy = W / w, H / h
    return (min(W, int(math.ceil((x0 + 1) * sx))) if x0 > 0 else 0,
            min(H, int(math.ceil((y0 + 1) * sy))) if y0 > 0 else 0,
            max(0, int(math.floor((x1 - 1) * sx))) if x1 < w else W,
            max(0, int(math.floor((y1 - 1) * sy))) if y1 < h else H)
