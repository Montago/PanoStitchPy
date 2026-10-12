"""Output projections: output pixel -> world ray -> source pixel."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .camera import Camera


@dataclass(frozen=True)
class Canvas:
    """The output image: a projection, its orientation, and the part of the sphere it shows.

    The image is computed in a *local* frame looking at (center_yaw, center_pitch):
      kind 'equirect'    : columns linear in local longitude, rows linear in local latitude
      kind 'cylindrical' : columns linear in local longitude, rows linear in tan(local latitude)
      kind 'rectilinear' : a flat image plane (like one big camera): straight lines stay straight. lon/lat
                           extents are the angles along the centre row/column; only for spans well below 180
    Normal frames have center_pitch 0 and an upright local frame, so the horizon is a straight line.
    Portrait frames (vertical panoramas) lay the local equator along the vertical great circle through
    (center_yaw, center_pitch) and turn the result 90 degrees, so the image is tall with the sky at the top;
    this keeps a panorama that passes straight down (nadir) free of the usual pole distortion.
    lon_left/lon_right/lat_top/lat_bottom are the local extents in degrees (before turning).
    """
    kind: str
    width: int                 # final image size (after turning a portrait canvas)
    height: int
    center_yaw: float = 0.0
    center_pitch: float = 0.0
    portrait: bool = False
    lon_left: float = -180.0
    lon_right: float = 180.0
    lat_top: float = 90.0
    lat_bottom: float = -90.0

    @staticmethod
    def full_sphere(width: int, center_yaw: float = 0.0) -> "Canvas":
        return Canvas("equirect", width, width // 2, center_yaw)

    @staticmethod
    def for_region(kind: str, deg_per_px: float, center_yaw: float, center_pitch: float, portrait: bool,
                   lon_left: float, lon_right: float, lat_top: float, lat_bottom: float,
                   width: int | None = None) -> "Canvas":
        """Canvas with square pixels on the local equator, deg_per_px degrees per pixel (or a given width)."""
        if kind == "cylindrical":
            lat_top, lat_bottom = min(lat_top, 80.0), max(lat_bottom, -80.0)
        span = lon_right - lon_left
        if kind == "rectilinear":   # deg_per_px holds in the image centre
            t = lambda a: math.tan(math.radians(a))
            land_w = width or max(16, int(round((t(lon_right) - t(lon_left)) / t(deg_per_px))))
            land_h = max(16, int(round(land_w * (t(lat_top) - t(lat_bottom)) / (t(lon_right) - t(lon_left)))))
            return Canvas(kind, land_w, land_h, center_yaw, center_pitch, False, lon_left, lon_right, lat_top,
                          lat_bottom)
        land_w = width if width is not None and not portrait else max(16, int(round(span / deg_per_px)))
        if kind == "cylindrical":
            v_span = math.tan(math.radians(lat_top)) - math.tan(math.radians(lat_bottom))
            land_h = int(round(land_w * v_span / math.radians(span)))
        else:
            land_h = int(round(land_w * (lat_top - lat_bottom) / span))
        land_h = max(16, land_h)
        w, h = (land_h, land_w) if portrait else (land_w, land_h)
        return Canvas(kind, w, h, center_yaw, center_pitch, portrait, lon_left, lon_right, lat_top, lat_bottom)

    @property
    def lon_span(self) -> float:
        return self.lon_right - self.lon_left

    @property
    def wraps(self) -> bool:
        return not self.portrait and self.lon_span >= 360.0 - 1e-6

    @property
    def is_full_sphere(self) -> bool:
        return self.wraps and self.kind == "equirect" and self.lat_top >= 90 and self.lat_bottom <= -90

    @property
    def deg_per_px(self) -> float:
        return self.lon_span / (self.height if self.portrait else self.width)

    def with_width(self, width: int) -> "Canvas":
        return replace(self, width=width, height=max(8, int(round(width * self.height / self.width))))

    def frame(self) -> np.ndarray:
        """3x3 matrix whose columns are the local x (right), y (up), z (forward) axes in world coordinates."""
        yaw, pitch = math.radians(self.center_yaw), math.radians(self.center_pitch)
        right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
        if not self.portrait:
            forward = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
            return np.stack([right, np.array([0.0, 1.0, 0.0]), forward], axis=1)
        forward = np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
        up_tangent = np.array([-math.sin(pitch) * math.sin(yaw), math.cos(pitch), -math.sin(pitch) * math.cos(yaw)])
        # local x runs towards the sky along the vertical arc, local y towards the viewer's left; after turning the
        # image 90 degrees counter-clockwise the sky is at the top and left stays left
        return np.stack([up_tangent, -right, forward], axis=1)

    def _land_size(self) -> tuple[int, int]:
        return (self.height, self.width) if self.portrait else (self.width, self.height)

    def lons(self) -> np.ndarray:
        """Local longitude (radians) of each column of the (unturned) image."""
        w, _ = self._land_size()
        t = (np.arange(w, dtype=np.float32) + 0.5) / w
        if self.kind == "rectilinear":
            left, right = math.tan(math.radians(self.lon_left)), math.tan(math.radians(self.lon_right))
            return np.arctan(left + t * (right - left)).astype(np.float32)
        return np.radians(self.lon_left + t * self.lon_span).astype(np.float32)

    def lats(self) -> np.ndarray:
        """Local latitude (radians) of each row of the (unturned) image."""
        _, h = self._land_size()
        t = (np.arange(h, dtype=np.float32) + 0.5) / h
        if self.kind in ("cylindrical", "rectilinear"):
            top, bottom = math.tan(math.radians(self.lat_top)), math.tan(math.radians(self.lat_bottom))
            return np.arctan(top - t * (top - bottom)).astype(np.float32)
        return np.radians(self.lat_top - t * (self.lat_top - self.lat_bottom)).astype(np.float32)

    def rays(self):
        """World-space unit rays (x, y, z), each (height, width) float32."""
        lon, lat = self.lons(), self.lats()
        if self.kind == "rectilinear":
            px, py = np.tan(lon)[None, :], np.tan(lat)[:, None]
            inv = 1.0 / np.sqrt(px * px + py * py + 1.0)
            lx, ly, lz = px * inv, py * inv, np.broadcast_to(inv, (len(lat), len(lon)))
        else:
            cos_lat = np.cos(lat)[:, None]
            lx = cos_lat * np.sin(lon)[None, :]
            ly = np.broadcast_to(np.sin(lat)[:, None], lx.shape)
            lz = cos_lat * np.cos(lon)[None, :]
        F = self.frame().astype(np.float32)
        out = []
        for r in range(3):
            a = F[r, 0] * lx + F[r, 1] * ly + F[r, 2] * lz
            out.append(np.ascontiguousarray(np.rot90(a) if self.portrait else a, dtype=np.float32))
        return tuple(out)

    def gpano_crop(self, x0: int, y0: int, x1: int, y1: int) -> dict | None:
        """GPano 'cropped area' fields for the pixel rectangle of a normal equirect canvas, else None."""
        if self.kind != "equirect" or self.portrait:
            return None
        full_w = int(round(360.0 / self.deg_per_px))
        full_h = full_w // 2
        lon_left = self.lon_left + x0 * self.deg_per_px          # relative to center_yaw
        lat_top = self.lat_top - y0 * (self.lat_top - self.lat_bottom) / self.height
        return {"full_w": full_w, "full_h": full_h, "crop_w": x1 - x0, "crop_h": y1 - y0,
                "left": int(round((lon_left + 180.0) / 360.0 * full_w)) % full_w,
                "top": max(0, int(round((90.0 - lat_top) / 180.0 * full_h))), "heading": self.center_yaw}


def equirect_rays(width: int, height: int, center_yaw_deg: float = 0.0):
    """World-space unit rays (x, y, z), each (height, width) float32.

    Column 0 is center_yaw - 180 deg, the middle column is center_yaw. Row 0 is straight up.
    """
    lon = ((np.arange(width, dtype=np.float32) + 0.5) / width * 2.0 - 1.0) * math.pi
    lon += math.radians(center_yaw_deg)
    lat = (0.5 - (np.arange(height, dtype=np.float32) + 0.5) / height) * math.pi
    cos_lat = np.cos(lat)[:, None]
    x = cos_lat * np.sin(lon)[None, :]
    y = np.broadcast_to(np.sin(lat)[:, None], (height, width))
    z = cos_lat * np.cos(lon)[None, :]
    return x.astype(np.float32), np.ascontiguousarray(y, dtype=np.float32), z.astype(np.float32)


def world_to_equirect(direction: np.ndarray, width: int, height: int, center_yaw_deg: float = 0.0):
    """Pixel position (u, v) of a world direction in the equirectangular image."""
    x, y, z = direction / np.linalg.norm(direction)
    lon = math.atan2(x, z) - math.radians(center_yaw_deg)
    lon = (lon + math.pi) % (2 * math.pi) - math.pi
    lat = math.asin(max(-1.0, min(1.0, y)))
    return (lon / math.pi + 1.0) / 2.0 * width, (0.5 - lat / math.pi) * height


def _reference_to_camera(x, y, z, t, h):
    """Same as camera.reference_to_camera, on separate float32 component arrays (saves memory)."""
    below = y < -1e-4
    s = h / np.maximum(-y, np.float32(1e-4))
    px, py, pz = x * s - t[0], y * s - t[1], z * s - t[2]
    inv = 1.0 / np.sqrt(px * px + py * py + pz * pz)
    return np.where(below, px * inv, x), np.where(below, py * inv, y), np.where(below, pz * inv, z)


def build_maps(cam: Camera, rays, extend: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remap tables (map_x, map_y) into the source image plus a validity mask.

    With extend=True, rays somewhat outside the image still map to the nearest edge pixel (replicated border)
    so that pyramid blending does not pull black into the image near its edges; the mask is unchanged.
    """
    x, y, z = rays
    if cam.position is not None:
        x, y, z = _reference_to_camera(x, y, z, cam.position.astype(np.float32), np.float32(cam.ground_h))
    R = cam.R.astype(np.float32)
    zc = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z
    xc = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z
    yc = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z
    # The distortion polynomial folds back for rays far outside the field of view, so reject anything
    # beyond the (undistorted) image corner before projecting.
    in_front = zc > 0.05
    tan2 = (xc * xc + yc * yc) / np.where(in_front, zc * zc, 1.0)
    max_tan2 = cam.max_tan2()
    u, v = cam.project(xc, yc, zc)
    valid = in_front & (tan2 < max_tan2 * 1.01) & (u >= 0) & (u <= cam.width - 1) & (v >= 0) & (v <= cam.height - 1)
    if extend:
        ext = in_front & (tan2 < max_tan2 * 2.0)
        map_x = np.where(ext, np.clip(u, 0, cam.width - 1), -1.0).astype(np.float32)
        map_y = np.where(ext, np.clip(v, 0, cam.height - 1), -1.0).astype(np.float32)
    else:
        map_x = np.where(valid, u, -1.0).astype(np.float32)
        map_y = np.where(valid, v, -1.0).astype(np.float32)
    return map_x, map_y, valid


def feather_weight(cam: Camera, map_x: np.ndarray, map_y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Weight that is 1 at the image centre and falls linearly to 0 at the edges."""
    wx = 1.0 - np.abs(map_x / (cam.width - 1) * 2.0 - 1.0)
    wy = 1.0 - np.abs(map_y / (cam.height - 1) * 2.0 - 1.0)
    return np.where(valid, np.clip(wx, 0, 1) * np.clip(wy, 0, 1), 0.0).astype(np.float32)
