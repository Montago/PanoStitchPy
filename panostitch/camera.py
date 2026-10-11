"""Pinhole camera model with rotation-only pose.

World frame: x = east, y = up, z = north.
Camera frame (OpenCV convention): x = right, y = down, z = forward (optical axis).
Yaw is a compass heading (0 = north, +90 = east), pitch is positive upwards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .metadata import ShotInfo

_FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)


def focal_px_from_35mm(focal_35mm: float, width: int, height: int) -> float:
    """35 mm-equivalent focal length is defined against the full-frame diagonal."""
    return focal_35mm * math.hypot(width, height) / _FULL_FRAME_DIAGONAL_MM


def _level_axes(yaw: float):
    """Right, down and forward of a level camera with heading `yaw` (radians), in world coordinates."""
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    down = np.array([0.0, -1.0, 0.0])
    forward = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    return right, down, forward


def rotation_world_to_cam(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """3x3 matrix R such that ray_cam = R @ ray_world (rows: camera right, down, forward in world coordinates).

    DJI gimbal convention, verified against bundle adjustment on the 9-image wide panorama (whose side columns
    have roll +-9.9): heading first, then roll about the (level) viewing direction, then pitch about the rolled
    right axis. Positive roll tips the right side of the image down, positive pitch looks up.
    """
    yaw, pitch, roll = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    r0, d0, f0 = _level_axes(yaw)
    right = math.cos(roll) * r0 + math.sin(roll) * d0
    down_rolled = -math.sin(roll) * r0 + math.cos(roll) * d0
    forward = math.cos(pitch) * f0 - math.sin(pitch) * down_rolled
    down = math.sin(pitch) * f0 + math.cos(pitch) * down_rolled
    return np.stack([right, down, forward])


def euler_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of rotation_world_to_cam: (yaw, pitch, roll) in degrees."""
    right, forward = R[0], R[2]
    roll = math.asin(max(-1.0, min(1.0, -right[1])))
    yaw = math.atan2(-right[2], right[0])
    r0, d0, f0 = _level_axes(yaw)
    down_rolled = -math.sin(roll) * r0 + math.cos(roll) * d0
    pitch = math.atan2(-np.dot(forward, down_rolled), np.dot(forward, f0))
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def unproject_normalized(pts: np.ndarray, focal: float, cx: float, cy: float,
                         k1: float = 0.0, k2: float = 0.0, iters: int = 6) -> np.ndarray:
    """Pixel coordinates (N, 2) -> unit rays (N, 3) in the camera frame, inverting radial distortion."""
    xd = (pts[:, 0] - cx) / focal
    yd = (pts[:, 1] - cy) / focal
    x, y = xd, yd
    if k1 or k2:
        for _ in range(iters):
            r2 = x * x + y * y
            s = 1.0 + k1 * r2 + k2 * r2 * r2
            x, y = xd / s, yd / s
    rays = np.stack([x, y, np.ones_like(x)], axis=1)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def ground_parallax(dirs: np.ndarray, position: np.ndarray, ground_h: float) -> np.ndarray:
    """Convert unit rays seen from `position` into directions seen from the reference point (origin).

    Rays below the horizon are assumed to hit a flat ground plane ground_h metres below the origin; rays above
    the horizon are at infinity and unchanged. dirs: (..., 3); position: (3,) or broadcastable to dirs.
    """
    dy = dirs[..., 1]
    below = dy < -1e-4
    dist = (ground_h + position[..., 1]) / np.maximum(-dy, 1e-4)
    pts = position + dirs * dist[..., None]
    pts /= np.linalg.norm(pts, axis=-1, keepdims=True)
    return np.where(below[..., None], pts, dirs)


def reference_to_camera(dirs: np.ndarray, position: np.ndarray, ground_h: float) -> np.ndarray:
    """Inverse of ground_parallax: directions from the origin -> unit rays from the camera at `position`."""
    dy = dirs[..., 1]
    below = dy < -1e-4
    dist = ground_h / np.maximum(-dy, 1e-4)
    pts = dirs * dist[..., None] - position
    pts /= np.linalg.norm(pts, axis=-1, keepdims=True)
    return np.where(below[..., None], pts, dirs)


def save_cameras(path, cams: list["Camera"]) -> None:
    import json

    out = []
    for c in cams:
        yaw, pitch, roll = c.euler()
        out.append({"file": c.meta.name, "yaw": yaw, "pitch": pitch, "roll": roll, "R": c.R.tolist(),
                    "focal": c.focal, "cx": c.cx, "cy": c.cy, "k1": c.k1, "k2": c.k2,
                    "position": c.position.tolist() if c.position is not None else None, "ground_h": c.ground_h})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


def load_cameras(path, shots: list[ShotInfo]) -> list["Camera"]:
    """Cameras written by step 2 (cameras.json), matched to the shots by file name."""
    import json

    by_name = {s.name: s for s in shots}
    cams = []
    for d in json.loads(open(path, encoding="utf-8").read()):
        s = by_name[d["file"]]
        pos = np.array(d["position"]) if d.get("position") is not None else None
        cams.append(Camera(s.width, s.height, d["focal"], d["cx"], d["cy"], np.array(d["R"]),
                           d["k1"], d["k2"], meta=s, position=pos, ground_h=d.get("ground_h", 0.0)))
    return cams


@dataclass
class Camera:
    width: int
    height: int
    focal: float  # pixels
    cx: float
    cy: float
    R: np.ndarray  # world -> camera
    k1: float = 0.0
    k2: float = 0.0
    gain: float = 1.0
    meta: ShotInfo | None = field(default=None, repr=False)
    position: np.ndarray | None = None  # metres, world frame, relative to the panorama reference point
    ground_h: float = 0.0               # ground plane distance below the reference point (parallax model)

    @classmethod
    def from_shot(cls, shot: ShotInfo) -> "Camera":
        return cls(
            width=shot.width,
            height=shot.height,
            focal=focal_px_from_35mm(shot.focal_35mm, shot.width, shot.height),
            cx=(shot.width - 1) / 2.0,
            cy=(shot.height - 1) / 2.0,
            R=rotation_world_to_cam(shot.gimbal_yaw, shot.gimbal_pitch, shot.gimbal_roll),
            meta=shot,
        )

    @property
    def forward_world(self) -> np.ndarray:
        return self.R[2]

    def fov_deg(self) -> tuple[float, float]:
        return (
            math.degrees(2 * math.atan(self.width / 2 / self.focal)),
            math.degrees(2 * math.atan(self.height / 2 / self.focal)),
        )

    def scaled(self, s: float) -> "Camera":
        """Same camera for a source image resized by factor s."""
        return Camera(round(self.width * s), round(self.height * s), self.focal * s, (self.cx + 0.5) * s - 0.5,
                      (self.cy + 0.5) * s - 0.5, self.R, self.k1, self.k2, self.gain, self.meta,
                      self.position, self.ground_h)

    def max_tan2(self) -> float:
        """Squared tangent of the angle between the optical axis and the farthest image corner."""
        corners = np.array([[0, 0], [self.width - 1, 0], [0, self.height - 1], [self.width - 1, self.height - 1]], float)
        rays = unproject_normalized(corners, self.focal, self.cx, self.cy, self.k1, self.k2)
        return float(np.max((rays[:, 0] ** 2 + rays[:, 1] ** 2) / rays[:, 2] ** 2))

    def euler(self) -> tuple[float, float, float]:
        return euler_from_rotation(self.R)

    def rays_world(self, pts: np.ndarray) -> np.ndarray:
        """Pixel coordinates (N, 2) -> unit world rays (N, 3), as seen from the reference point."""
        rays = unproject_normalized(pts, self.focal, self.cx, self.cy, self.k1, self.k2) @ self.R
        if self.position is not None:
            rays = ground_parallax(rays, self.position, self.ground_h)
        return rays

    def project(self, xc: np.ndarray, yc: np.ndarray, zc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project camera-frame rays to pixel coordinates (rays with zc <= 0 must be masked by caller)."""
        inv_z = 1.0 / np.where(zc > 1e-6, zc, 1e-6)
        xn = xc * inv_z
        yn = yc * inv_z
        if self.k1 or self.k2:
            r2 = xn * xn + yn * yn
            radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
            xn = xn * radial
            yn = yn * radial
        return self.focal * xn + self.cx, self.focal * yn + self.cy
