"""Render a set of posed cameras into equirectangular previews (feather blend, EXIF-exposure blend, seams, coverage)."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .imageio import imread
from .projection import build_maps, equirect_rays, feather_weight, world_to_equirect

_JPEG = [cv2.IMWRITE_JPEG_QUALITY, 92]


def srgb_to_linear_lut() -> np.ndarray:
    c = np.arange(256, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def tonemap(linear: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Simple global Reinhard: median luminance -> 0.18, brightest 0.5 % -> white."""
    lum = 0.2126 * linear[..., 2] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 0]
    vals = lum[mask]
    scale = 0.18 / max(float(np.median(vals)), 1e-6)
    white = max(float(np.percentile(vals, 99.5)) * scale, 1.0)
    x = linear * scale
    return x * (1.0 + x / (white * white)) / (1.0 + x)


def render_all(cams: list[Camera], out_dir: Path, prefix: str, width: int = 6000,
               center_yaw: float = 0.0, verbose: bool = True) -> dict:
    W, H = width, width // 2
    out_dir.mkdir(parents=True, exist_ok=True)
    rays = equirect_rays(W, H, center_yaw)
    lut = srgb_to_linear_lut()
    exposures = np.array([c.meta.exposure for c in cams])
    ref_exposure = float(np.exp(np.mean(np.log(exposures))))

    acc = np.zeros((H, W, 3), np.float32)
    acc_lin = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W), np.float32)
    best_w = np.zeros((H, W), np.float32)
    seams = np.zeros((H, W, 3), np.uint8)
    coverage = np.zeros((H, W), np.uint8)

    t0 = time.time()
    for cam in cams:
        img = imread(cam.meta.path)
        map_x, map_y, valid = build_maps(cam, rays)
        warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        w = feather_weight(cam, map_x, map_y, valid)

        acc += warped.astype(np.float32) * w[..., None]
        acc_lin += lut[warped] * (w * (ref_exposure / cam.meta.exposure))[..., None]
        wsum += w
        coverage += valid.astype(np.uint8)
        better = w > best_w
        best_w[better] = w[better]
        seams[better] = warped[better]
    if verbose:
        print(f"  rendered {len(cams)} images in {time.time() - t0:.1f}s")

    covered = wsum > 0
    denom = np.where(covered, wsum, 1.0)[..., None]

    feather = np.clip(acc / denom, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{prefix}_feather.jpg"), feather, _JPEG)

    exif_img = (linear_to_srgb(tonemap(acc_lin / denom, covered)) * 255).astype(np.uint8)
    exif_img[~covered] = 0
    cv2.imwrite(str(out_dir / f"{prefix}_feather_exif.jpg"), exif_img, _JPEG)

    font_scale = W / 3000
    for i, cam in enumerate(cams):
        u, v = world_to_equirect(cam.forward_world, W, H, center_yaw)
        v = min(max(v, 60 * font_scale), H - 20 * font_scale)
        label = str(i + 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, int(3 * font_scale))
        org = (int(u - tw / 2), int(v + th / 2))
        cv2.putText(seams, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), int(8 * font_scale))
        cv2.putText(seams, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), int(3 * font_scale))
    cv2.imwrite(str(out_dir / f"{prefix}_seams.jpg"), seams, _JPEG)

    cov_vis = cv2.applyColorMap((np.minimum(coverage, 6) * 42).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cov_vis[coverage == 0] = 0
    cv2.imwrite(str(out_dir / f"{prefix}_coverage.png"), cov_vis)

    lat = (0.5 - (np.arange(H) + 0.5) / H) * 180.0
    hole_rows = np.where((coverage[: H // 2] == 0).any(axis=1))[0]
    return {
        "uncovered_pct": 100 * float((~covered).mean()),
        "zenith_hole_lat": float(lat[hole_rows.max()]) if hole_rows.size else None,
        "exposure_stops": float(np.log2(exposures.max() / exposures.min())),
    }
