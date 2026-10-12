"""Keep the sun from a single image and keep lens-flare ghosts out of the panorama.

The sun's glare looks different in every image (it depends on where the sun is in the frame and on the
exposure), so a seam through the glow cuts it into mismatched pieces. The sun's direction is predicted from
GPS position and capture time, refined to the brightest saturated spot nearby, and a disc around it is
reserved for the image that shows the sun farthest from its edges (as far as that image reaches).

Flare ghosts land in a different place in every image (on the line from the sun through the image centre),
so wherever another image covers the same spot, the seams avoid the image that has a ghost there.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Camera
from .metadata import ShotInfo
from .projection import build_maps


def sun_direction(lat_deg: float, lon_deg: float, when: dt.datetime) -> np.ndarray:
    """Unit vector towards the sun in the world frame (x = east, y = up, z = north); `when` must be
    timezone-aware. Low-precision solar ephemeris (~0.1 degree), no refraction."""
    n = when.timestamp() / 86400.0 + 2440587.5 - 2451545.0       # days since J2000
    mean_lon = math.radians((280.460 + 0.9856474 * n) % 360)
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    ecl_lon = mean_lon + math.radians(1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.atan2(math.cos(eps) * math.sin(ecl_lon), math.cos(ecl_lon))
    dec = math.asin(math.sin(eps) * math.sin(ecl_lon))
    gmst = math.radians((280.46061837 + 360.98564736629 * n) % 360)
    h = gmst + math.radians(lon_deg) - ra                         # hour angle
    lat = math.radians(lat_deg)
    east = -math.cos(dec) * math.sin(h)
    north = math.cos(lat) * math.sin(dec) - math.sin(lat) * math.cos(dec) * math.cos(h)
    up = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(h)
    return np.array([east, up, north])


def predicted_direction(shots: list[ShotInfo]) -> np.ndarray | None:
    """Sun direction at the middle of the shoot, or None without GPS or a capture time with time zone."""
    good = [s for s in shots if s.taken is not None and s.taken.tzinfo is not None
            and not math.isnan(s.latitude) and not math.isnan(s.longitude)]
    if not good:
        return None
    times = sorted(s.taken for s in good)
    mid = times[0] + (times[-1] - times[0]) / 2
    return sun_direction(float(np.mean([s.latitude for s in good])), float(np.mean([s.longitude for s in good])),
                         mid)


def edge_margin_deg(cam: Camera, direction: np.ndarray, limit: float = 30.0, step: float = 0.5) -> float:
    """Largest angle around `direction` that is still inside the image (0 if the direction is outside)."""
    d = direction / np.linalg.norm(direction)
    a = np.cross(d, [0.0, 1.0, 0.0] if abs(d[1]) < 0.9 else [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    phi = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    for r in np.arange(0.0, limit + step / 2, step):
        t = math.radians(r)
        pts = (math.cos(t) * d + math.sin(t) * (np.outer(np.cos(phi), a) + np.outer(np.sin(phi), b))) @ cam.R.T
        if (pts[:, 2] <= 0.05).any():
            return max(0.0, r - step)
        u, v = cam.project(pts[:, 0], pts[:, 1], pts[:, 2])
        if (u < 0).any() or (u > cam.width - 1).any() or (v < 0).any() or (v > cam.height - 1).any():
            return max(0.0, r - step)
    return limit


@dataclass
class SunGuard:
    direction: np.ndarray   # world unit vector
    image: int              # index of the image that keeps the sun
    radius_deg: float       # how far around the sun that image is used (as far as it reaches)

    def disc(self, rays) -> np.ndarray:
        """Boolean mask of the canvas pixels (given by their world rays) within radius_deg of the sun."""
        x, y, z = rays
        d = self.direction.astype(np.float32)
        return d[0] * x + d[1] * y + d[2] * z > np.float32(math.cos(math.radians(self.radius_deg)))

    def keep_mask(self, rays, valid_keep: np.ndarray, deg_per_px: float, edge_deg: float = 1.5) -> np.ndarray:
        """The part of the disc reserved for the sun image: what it covers, minus a strip along its edges so
        the seams (and the blending across them) stay inside it."""
        k = max(1, int(round(edge_deg / deg_per_px)))
        inner = cv2.erode(valid_keep.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
        return self.disc(rays) & inner

    def flare_band(self, cam: Camera, rays, width: float = 0.05, beyond: float = 1.3,
                   outside: float = 0.25) -> np.ndarray | None:
        """Canvas mask of where lens-flare ghosts of the sun can show in this image, or None.

        Ghosts lie on the line from the sun through the image centre, mostly around the sun's mirror point; the
        band runs from the sun to `beyond` times the mirror distance, `width` of the image diagonal to each side.
        Images with the sun up to `outside` of their size beyond the frame are included, since they flare too.
        """
        d = cam.R @ self.direction
        if d[2] <= 0.05:
            return None
        su, sv = (float(a[0]) for a in cam.project(d[0:1], d[1:2], d[2:3]))
        if not (-outside * cam.width <= su <= (1 + outside) * cam.width
                and -outside * cam.height <= sv <= (1 + outside) * cam.height):
            return None
        mx, my, valid = build_maps(cam, rays)
        ex, ey = cam.cx + beyond * (cam.cx - su), cam.cy + beyond * (cam.cy - sv)
        dx, dy = ex - su, ey - sv
        t = np.clip(((mx - su) * dx + (my - sv) * dy) / max(dx * dx + dy * dy, 1e-6), 0.0, 1.0)
        dist2 = (mx - su - t * dx) ** 2 + (my - sv - t * dy) ** 2
        return valid & (dist2 < (width * math.hypot(cam.width, cam.height)) ** 2)

    def describe(self, cams: list[Camera]) -> str:
        x, y, z = self.direction
        az = math.degrees(math.atan2(x, z)) % 360
        el = math.degrees(math.asin(max(-1.0, min(1.0, y))))
        return (f"Sun at {az:.0f}° / {el:.0f}° elevation: kept from {cams[self.image].meta.name} "
                f"up to {self.radius_deg:.0f}° around it")


def find_sun(cams: list[Camera], warped: list[np.ndarray], valid: list[np.ndarray], rays, deg_per_px: float,
             search_el: float = 5.0, search_az: float = 40.0, radius_deg: float = 20.0,
             min_margin_deg: float = 4.0, saturation: list[float] | None = None) -> SunGuard | None:
    """Locate a visible sun on a (low-resolution) canvas and pick the image that should keep it.

    warped/valid: the images on that canvas (uint8 BGR, or linear float for raw images with their clip levels
    in `saturation`) and their coverage, rays: the canvas' world rays.
    The search window is wide in azimuth because the drone's heading can be off from true north by 10-25 degrees,
    while elevation is reliable (the alignment levels the horizon).
    Returns None when the time/position is unknown, the sun is outside the panorama, below the horizon or
    hidden (no saturated spot near the predicted direction), or no image shows it away from its edges.
    """
    pred = predicted_direction([c.meta for c in cams])
    if pred is None or pred[1] < -0.02:
        return None
    x, y, z = rays
    el_pred = math.asin(pred[1])
    horiz = np.hypot(pred[0], pred[2])
    cos_daz = (x * pred[0] + z * pred[2]) / np.maximum(np.hypot(x, z) * horiz, 1e-6)
    near = (np.abs(np.arcsin(np.clip(y, -1, 1)) - el_pred) < math.radians(search_el)) & \
        ((cos_daz > math.cos(math.radians(search_az))) | (pred[1] > math.sin(math.radians(80))))
    if not near.any():
        return None

    # fraction of the overlapping images that are saturated, smoothed over the size of a sun with its halo
    sat = np.zeros(near.shape, np.float32)
    cnt = np.zeros(near.shape, np.float32)
    for k, (w, v) in enumerate(zip(warped, valid)):
        m = v & near
        if m.any():
            # raw: only the tiny disc itself clips, and at this resolution it is averaged with its glow
            clipped = 250 if w.dtype == np.uint8 else 0.5 * (saturation[k] if saturation else 1.0)
            sat += m & (w.min(axis=2) >= clipped)
            cnt += m
    sigma = max(1.0, 1.5 / deg_per_px)
    sat = cv2.GaussianBlur(sat, (0, 0), sigma)
    cnt = cv2.GaussianBlur(cnt, (0, 0), sigma)
    frac = np.where(near & (cnt > 0.5), sat / np.maximum(cnt, 1e-6), 0.0)
    r, c = np.unravel_index(int(np.argmax(frac)), frac.shape)
    if frac[r, c] < 0.6:
        return None
    direction = np.array([x[r, c], y[r, c], z[r, c]], float)
    direction /= np.linalg.norm(direction)

    margins = [edge_margin_deg(cam, direction) for cam in cams]
    best = int(np.argmax(margins))
    if margins[best] < min_margin_deg:
        return None
    return SunGuard(direction, best, radius_deg)
