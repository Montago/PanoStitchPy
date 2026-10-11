"""Read DJI drone metadata (XMP drone-dji namespace + standard EXIF) from JPEGs."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_XMP_ATTR = re.compile(r'([\w-]+):(\w+)="([^"]*)"')

# EXIF tag ids (Exif sub-IFD)
_EXIF_IFD = 0x8769
_EXPOSURE_TIME = 0x829A
_F_NUMBER = 0x829D
_ISO = 0x8827
_FOCAL_LENGTH = 0x920A
_FOCAL_35MM = 0xA405
_ORIENTATION = 0x0112


@dataclass
class ShotInfo:
    path: Path
    width: int
    height: int
    orientation: int
    focal_mm: float
    focal_35mm: float
    exposure_time: float
    iso: float
    f_number: float
    gimbal_yaw: float
    gimbal_pitch: float
    gimbal_roll: float
    flight_yaw: float
    flight_pitch: float
    flight_roll: float
    relative_altitude: float
    model: str

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def exposure(self) -> float:
        """Relative sensor exposure (proportional to recorded value for a fixed scene radiance)."""
        return self.exposure_time * self.iso / (self.f_number ** 2)


def read_xmp(path: Path, max_bytes: int = 512 * 1024) -> dict[str, str]:
    """Return XMP attributes as {'namespace:Name': value}. DJI puts XMP near the start of the file."""
    with open(path, "rb") as fh:
        data = fh.read(max_bytes)
    start = data.find(b"<x:xmpmeta")
    end = data.find(b"</x:xmpmeta>")
    if start < 0 or end < 0:
        return {}
    xmp = data[start:end].decode("utf-8", "replace")
    return {f"{ns}:{key}": val for ns, key, val in _XMP_ATTR.findall(xmp)}


def read_shot(path: Path) -> ShotInfo:
    xmp = read_xmp(path)

    def dji(key: str, default: float = 0.0) -> float:
        val = xmp.get(f"drone-dji:{key}")
        return float(val) if val not in (None, "") else default

    with Image.open(path) as im:
        width, height = im.size
        exif = im.getexif()
        orientation = int(exif.get(_ORIENTATION, 1))
        sub = exif.get_ifd(_EXIF_IFD)

    def num(tag: int, default: float = 0.0) -> float:
        val = sub.get(tag)
        return float(val) if val is not None else default

    return ShotInfo(
        path=path,
        width=width,
        height=height,
        orientation=orientation,
        focal_mm=num(_FOCAL_LENGTH),
        focal_35mm=num(_FOCAL_35MM),
        exposure_time=num(_EXPOSURE_TIME),
        iso=num(_ISO, 100.0),
        f_number=num(_F_NUMBER, 1.0),
        gimbal_yaw=dji("GimbalYawDegree"),
        gimbal_pitch=dji("GimbalPitchDegree"),
        gimbal_roll=dji("GimbalRollDegree"),
        flight_yaw=dji("FlightYawDegree"),
        flight_pitch=dji("FlightPitchDegree"),
        flight_roll=dji("FlightRollDegree"),
        relative_altitude=dji("RelativeAltitude"),
        model=xmp.get("tiff:Model", ""),
    )


def positions_world(shots: list[ShotInfo]) -> np.ndarray:
    """Camera positions (n, 3) in metres in the panorama world frame (x = east, y = up, z = north),
    relative to their mean. Uses the DJI GPS fields."""
    geo = []
    for s in shots:
        x = read_xmp(s.path)
        geo.append((float(x["drone-dji:GpsLongitude"]), float(x["drone-dji:GpsLatitude"]),
                    float(x["drone-dji:AbsoluteAltitude"])))
    geo = np.array(geo)
    cos_lat = math.cos(math.radians(geo[:, 1].mean()))
    east = (geo[:, 0] - geo[0, 0]) * 111320.0 * cos_lat
    north = (geo[:, 1] - geo[0, 1]) * 110540.0
    pos = np.stack([east, geo[:, 2] - geo[0, 2], north], axis=1)
    return pos - pos.mean(axis=0)


def read_folder(folder: Path) -> list[ShotInfo]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    return [read_shot(p) for p in files]
