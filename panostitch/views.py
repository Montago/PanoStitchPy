"""Views of an equirectangular 360 panorama: normal (rectilinear) camera views and stereographic views such as
"tiny planet" (looking straight down) and "inverted" (looking straight up, sky in the middle).

A view is described by its mode, heading (yaw), pitch and field of view. Output pixels are given in *view plane*
coordinates: (0, 0) is the centre of the view and +-1 is half the shorter side of the window, so a view looks the
same in a window and in an exported image of any shape (the export simply covers a different part of the plane).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ViewMode:
    label: str
    stereographic: bool
    fov: float               # default field of view (degrees across the shorter side)
    pitch: float             # default pitch
    fov_range: tuple[float, float]


VIEW_MODES = {
    "normal": ViewMode("Normal", False, 75.0, 0.0, (20.0, 120.0)),
    "planet": ViewMode("Tiny planet", True, 250.0, -90.0, (60.0, 330.0)),
    "inverted": ViewMode("Inverted", True, 250.0, 90.0, (60.0, 330.0)),
}


def view_lonlat(mode: str, yaw: float, pitch: float, fov: float, xs: np.ndarray, ys: np.ndarray):
    """Longitude and latitude (radians, relative to the panorama centre) seen at view plane points.

    xs: (W,) view plane x of each column (right positive), ys: (H,) view plane y of each row (down positive).
    """
    x = xs.astype(np.float32)[None, :]
    y = -ys.astype(np.float32)[:, None]          # up positive
    if VIEW_MODES[mode].stereographic:
        # stereographic: distance from the centre r = 2 tan(theta / 2), scaled so r = 1 is fov / 2
        k = math.tan(math.radians(fov) / 4)
        rho = np.sqrt(x * x + y * y)
        theta = 2.0 * np.arctan(rho * k)
        s = np.where(rho > 1e-9, np.sin(theta) / np.maximum(rho, 1e-9), 0.0)
        cx, cy, cz = x * s, y * s, np.cos(theta)
    else:
        t = math.tan(math.radians(min(fov, 170.0)) / 2)
        cx, cy, cz = x * t, y * t, np.ones_like(x * y)
    # tilt (about the camera x axis), then pan (about the vertical axis); world: x east, y up, z forward/north
    p, yw = math.radians(pitch), math.radians(yaw)
    y2 = cy * math.cos(p) + cz * math.sin(p)
    z2 = -cy * math.sin(p) + cz * math.cos(p)
    x3 = cx * math.cos(yw) + z2 * math.sin(yw)
    z3 = -cx * math.sin(yw) + z2 * math.cos(yw)
    lon = np.arctan2(x3, z3)
    lat = np.arctan2(y2, np.sqrt(x3 * x3 + z3 * z3))
    return lon, lat


def render_view(pano: np.ndarray, mode: str, yaw: float, pitch: float, fov: float,
                xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample the equirectangular `pano` (any channel order) for the given view plane grid."""
    lon, lat = view_lonlat(mode, yaw, pitch, fov, xs, ys)
    H, W = pano.shape[:2]
    mx = ((lon * (0.5 / math.pi) + 0.5) * W - 0.5).astype(np.float32)
    my = ((0.5 - lat * (1.0 / math.pi)) * H - 0.5).astype(np.float32)
    return cv2.remap(pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def frame_rect(win_w: float, win_h: float, aspect: float | None) -> tuple[float, float, float, float]:
    """The largest centred rectangle (x, y, w, h) of the given aspect (w / h) inside the window; the whole
    window when aspect is None."""
    if aspect is None:
        return 0.0, 0.0, float(win_w), float(win_h)
    if win_w / win_h > aspect:
        h = float(win_h)
        w = h * aspect
    else:
        w = float(win_w)
        h = w / aspect
    return (win_w - w) / 2, (win_h - h) / 2, w, h


def plane_coords(win_w: float, win_h: float, rect: tuple[float, float, float, float], out_w: int, out_h: int):
    """View plane coordinates (xs, ys) of an out_w x out_h pixel grid covering `rect` of the window."""
    half = min(win_w, win_h) / 2
    x0, y0, w, h = rect
    xs = (x0 + (np.arange(out_w) + 0.5) / out_w * w - win_w / 2) / half
    ys = (y0 + (np.arange(out_h) + 0.5) / out_h * h - win_h / 2) / half
    return xs, ys


def render_export(pano: np.ndarray, mode: str, yaw: float, pitch: float, fov: float, win_w: float, win_h: float,
                  aspect: float | None, long_side: int, supersample: int | None = None,
                  chunk_rows: int = 512) -> np.ndarray:
    """The part of the view inside frame_rect(aspect) at long_side pixels, rendered from the full-resolution
    panorama in row chunks (bounded memory), supersampled for smooth edges."""
    rect = frame_rect(win_w, win_h, aspect)
    a = rect[2] / rect[3]
    out_w, out_h = (long_side, max(1, round(long_side / a))) if a >= 1 else (max(1, round(long_side * a)), long_side)
    ss = supersample or (2 if long_side <= 4096 else 1)
    xs, ys = plane_coords(win_w, win_h, rect, out_w * ss, out_h * ss)
    out = np.empty((out_h, out_w) + pano.shape[2:], pano.dtype)
    step = chunk_rows * ss
    for r0 in range(0, out_h * ss, step):
        part = render_view(pano, mode, yaw, pitch, fov, xs, ys[r0:r0 + step])
        if ss > 1:
            part = cv2.resize(part, (out_w, part.shape[0] // ss), interpolation=cv2.INTER_AREA)
        out[r0 // ss:r0 // ss + part.shape[0]] = part
    return out
