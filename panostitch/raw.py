"""DJI DNG (raw) images: demosaic with LibRaw (rawpy) into linear RGB and apply the lens corrections DJI stores in
the DNG but LibRaw leaves out (OpcodeList3: GainMap = vignetting, WarpRectilinear = distortion and lateral
chromatic aberration), so a decoded frame lines up with the JPEG the drone would have made.

Values are linear, 1.0 = sensor saturation. Unlike the JPEGs nothing is clipped about 2 stops below that,
which is the point: the sun, its glow and bright sky keep their detail.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

RAW_SUFFIXES = (".dng",)

# The drone's own rendering ("look"), fitted on a DNG + JPEG pair from a DJI Air 3 (single shot 0220,
# rms error 7 levels): JPEG = tone(MATRIX @ linear camera RGB). MATRIX rows sum to 1 (neutrals stay neutral);
# the tone curve lifts the mid-tones far more than sRGB does and has a soft shoulder.
DJI_MATRIX = np.array([[0.8254, 0.3341, -0.1595],
                       [-0.0505, 1.1667, -0.1162],
                       [0.0180, -0.1961, 1.1781]], np.float32)   # RGB
_TONE_X = np.array([0.00400, 0.00503, 0.00634, 0.00798, 0.01004, 0.01264, 0.01591, 0.02002, 0.02520, 0.03172,
                    0.03992, 0.05025, 0.06325, 0.07961, 0.10020, 0.12612, 0.15874, 0.19980, 0.25149, 0.31654,
                    0.39842, 0.50148, 0.63121, 0.79449, 1.00000])
_TONE_Y = np.array([0.0220, 0.0277, 0.0348, 0.0438, 0.0539, 0.0692, 0.0883, 0.1116, 0.1395, 0.1726, 0.2074,
                    0.2431, 0.2835, 0.3346, 0.4026, 0.4798, 0.5547, 0.6247, 0.6954, 0.7666, 0.8292, 0.8772,
                    0.9102, 0.9623, 1.0000])

_OPCODE_LISTS = (51008, 51009, 51022)   # OpcodeList1/2/3
_GAIN_MAP, _WARP_RECTILINEAR = 9, 1


def is_raw(path: Path | str) -> bool:
    return Path(path).suffix.lower() in RAW_SUFFIXES


@dataclass
class GainMap:
    area: tuple[int, int, int, int]   # top, left, bottom, right (pixels of the full image)
    plane: int
    planes: int
    origin: tuple[float, float]       # (v, h), relative to the image size
    spacing: tuple[float, float]
    gains: np.ndarray                 # (points_v, points_h, map_planes) float32


@dataclass
class Warp:
    coeffs: np.ndarray                # (planes, 6): kr0..kr3, kt0, kt1
    center: tuple[float, float]       # (x, y), relative to the image size


@dataclass
class Opcodes:
    gain_maps: list[GainMap]
    warp: Warp | None


def _parse_opcodes(data: bytes) -> tuple[list[GainMap], Warp | None]:
    gain_maps, warp = [], None
    count = struct.unpack(">I", data[:4])[0]
    pos = 4
    for _ in range(count):
        opid, _version, _flags, size = struct.unpack(">IIII", data[pos:pos + 16])
        p = data[pos + 16:pos + 16 + size]
        pos += 16 + size
        if opid == _GAIN_MAP:
            top, left, bottom, right, plane, planes, _rp, _cp, nv, nh = struct.unpack(">10I", p[:40])
            sv, sh, ov, oh = struct.unpack(">4d", p[40:72])
            mplanes = struct.unpack(">I", p[72:76])[0]
            gains = np.frombuffer(p[76:76 + 4 * nv * nh * mplanes], ">f4").astype(np.float32)
            gain_maps.append(GainMap((top, left, bottom, right), plane, planes, (ov, oh), (sv, sh),
                                     gains.reshape(nv, nh, mplanes)))
        elif opid == _WARP_RECTILINEAR:
            n = struct.unpack(">I", p[:4])[0]
            vals = struct.unpack(f">{6 * n + 2}d", p[4:4 + 8 * (6 * n + 2)])
            warp = Warp(np.array(vals[:6 * n]).reshape(n, 6), (vals[-2], vals[-1]))
    return gain_maps, warp


def read_opcodes(path: Path | str) -> Opcodes:
    """The opcode lists of the raw image's IFD (the largest SubIFD)."""
    from PIL import Image, TiffImagePlugin

    gain_maps, warp = [], None
    with Image.open(path) as im:
        subs = im.tag_v2.get(330) or ()
        subs = subs if isinstance(subs, (tuple, list)) else (subs,)
        best = None
        for off in subs:
            im.fp.seek(off)
            ifd = TiffImagePlugin.ImageFileDirectory_v2(prefix=im.tag_v2.prefix)
            ifd.load(im.fp)
            if best is None or ifd.get(256, 0) > best.get(256, 0):
                best = ifd
        for tag in _OPCODE_LISTS:
            if best is not None and tag in best:
                data = best[tag]
                g, w = _parse_opcodes(bytes(data) if not isinstance(data, bytes) else data)
                gain_maps += g
                warp = w or warp
    return Opcodes(gain_maps, warp)


def raw_size(path: Path | str) -> tuple[int, int]:
    """(width, height) of the decoded image."""
    import rawpy

    with rawpy.imread(str(path)) as r:
        return r.sizes.width, r.sizes.height


def camera_wb(path: Path | str) -> np.ndarray:
    """As-shot white balance multipliers (R, G, B), green = 1."""
    import rawpy

    with rawpy.imread(str(path)) as r:
        wb = np.array(r.camera_whitebalance[:3], float)
    return wb / wb[1]


def _apply_gain_map(img: np.ndarray, gm: GainMap, full_h: int, full_w: int) -> None:
    """In place. img is BGR, possibly a scaled-down version of the full image."""
    h, w = img.shape[:2]
    sy, sx = h / full_h, w / full_w
    top, left, bottom, right = gm.area
    y0, y1 = int(round(top * sy)), int(round(min(bottom, full_h) * sy))
    x0, x1 = int(round(left * sx)), int(round(min(right, full_w) * sx))
    ys = ((np.arange(y0, y1, dtype=np.float32) + 0.5) / h - gm.origin[0]) / gm.spacing[0]
    xs = ((np.arange(x0, x1, dtype=np.float32) + 0.5) / w - gm.origin[1]) / gm.spacing[1]
    mx, my = np.meshgrid(xs, ys)
    for p in range(gm.planes):
        mp = min(p, gm.gains.shape[2] - 1)
        g = cv2.remap(np.ascontiguousarray(gm.gains[:, :, mp]), mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
        c = 2 - (gm.plane + p)   # DNG planes are R, G, B; img is BGR
        if 0 <= c < 3:
            img[y0:y1, x0:x1, c] *= g


def _apply_warp(img: np.ndarray, warp: Warp) -> np.ndarray:
    """DNG WarpRectilinear: for every output pixel, the radial/tangential polynomial gives where to sample."""
    h, w = img.shape[:2]
    cx, cy = warp.center[0] * w, warp.center[1] * h
    m = max(math.hypot(x - cx, y - cy) for x in (0, w) for y in (0, h))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = (xx + 0.5 - cx) / m, (yy + 0.5 - cy) / m
    r2 = dx * dx + dy * dy
    out = np.empty_like(img)
    for c in range(3):
        kr0, kr1, kr2, kr3, kt0, kt1 = warp.coeffs[min(2 - c, len(warp.coeffs) - 1)]
        f = kr0 + r2 * (kr1 + r2 * (kr2 + r2 * kr3))
        sx = cx + m * (f * dx + kt0 * 2 * dx * dy + kt1 * (r2 + 2 * dx * dx)) - 0.5
        sy = cy + m * (f * dy + kt1 * 2 * dx * dy + kt0 * (r2 + 2 * dy * dy)) - 0.5
        out[..., c] = cv2.remap(img[..., c], sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    return out


def decode(path: Path | str, wb: np.ndarray | None = None, half: bool = False,
           opcodes: Opcodes | None = None) -> np.ndarray:
    """Linear BGR float32 (1.0 = sensor saturation) with DJI's vignetting and distortion corrections applied.

    wb: white balance multipliers (R, G, B); None = as shot. half: LibRaw's fast half-size decode (no
    demosaicing), for features and previews."""
    import rawpy

    with rawpy.imread(str(path)) as r:
        full_w, full_h = r.sizes.width, r.sizes.height
        kw = dict(use_camera_wb=True) if wb is None else dict(user_wb=[float(wb[0]), float(wb[1]), float(wb[2]),
                                                                         float(wb[1])])
        rgb = r.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, half_size=half, user_flip=0,
                            output_color=rawpy.ColorSpace.sRGB, **kw)
    img = rgb[..., ::-1].astype(np.float32) * np.float32(1 / 65535)
    ops = opcodes if opcodes is not None else read_opcodes(path)
    for gm in ops.gain_maps:
        _apply_gain_map(img, gm, full_h, full_w)
    if ops.warp is not None:
        img = _apply_warp(img, ops.warp)
    return img


def to_dji_colour(img: np.ndarray) -> np.ndarray:
    """Linear camera RGB (BGR order) -> linear RGB with the drone's colour rendering (still linear)."""
    m = DJI_MATRIX[::-1, ::-1]   # same matrix for BGR order
    return np.einsum("...c,dc->...d", img, m).astype(np.float32)


def tone(lin: np.ndarray) -> np.ndarray:
    """The drone's tone curve: linear 0-1 -> display 0-1 (values outside 0-1 are clipped)."""
    x = np.clip(lin, 0.0, 1.0)
    lx = np.log2(np.maximum(x, 1e-6)).astype(np.float32)
    y = np.interp(lx, np.log2(_TONE_X).astype(np.float32), _TONE_Y.astype(np.float32)).astype(np.float32)
    toe = x < _TONE_X[0]
    y[toe] = (x[toe] * (_TONE_Y[0] / _TONE_X[0])).astype(np.float32)
    return y


def load_linear(path: Path | str, wb: np.ndarray | None = None, half: bool = False) -> np.ndarray:
    """decode() plus the drone's colour rendering: linear BGR, 1.0 = sensor saturation."""
    return to_dji_colour(decode(path, wb=wb, half=half))


def load_display(path: Path | str, half: bool = True) -> np.ndarray:
    """The frame roughly as the drone would have saved it as a JPEG (uint8 BGR), for features and previews."""
    return (tone(load_linear(path, half=half)) * 255 + 0.5).astype(np.uint8)


def load_normalised(path: Path | str, half: bool = True) -> np.ndarray:
    """The frame as uint8 BGR, brightened so its median sits at mid-grey: the raw frames are often exposed
    far below the drone's JPEGs, and 8-bit dark images lose the texture SIFT needs."""
    lin = load_linear(path, half=half)
    lum = lin @ np.array([0.0722, 0.7152, 0.2126], np.float32)
    lin *= np.float32(0.18 / max(float(np.median(lum)), 1e-4))
    return (tone(lin) * 255 + 0.5).astype(np.uint8)
