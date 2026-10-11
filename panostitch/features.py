"""SIFT features and pairwise matching, restricted to image pairs that overlap according to the metadata."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .imageio import imread


@dataclass
class Features:
    pts: np.ndarray   # (N, 2) float32, full-resolution pixel coordinates
    desc: np.ndarray  # (N, 128) float32


@dataclass
class PairMatches:
    i: int
    j: int
    pts_i: np.ndarray  # (M, 2)
    pts_j: np.ndarray  # (M, 2)
    n_ratio: int       # after Lowe ratio test
    n_prior: int       # after metadata-consistency filter
    n_inliers: int     # after RANSAC


def _cache_key(path: Path) -> str:
    """DJI reuses file names (PANO_0001.JPG ...) in every panorama folder, so key on the full path and the file's
    size/mtime as well, not just the name."""
    st = path.stat()
    digest = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{digest}"


def detect(path: Path, scale: float = 0.5, nfeatures: int = 8000, cache_dir: Path | None = None) -> Features:
    cache = cache_dir / f"{_cache_key(path)}_sift_{scale}_{nfeatures}.npz" if cache_dir else None
    if cache and cache.exists():
        data = np.load(cache)
        return Features(data["pts"], data["desc"])

    gray = imread(path, cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # CLAHE lifts texture in dark ground shots and bright sky shots alike
    small = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(small)
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.02)
    kps, desc = sift.detectAndCompute(small, None)
    pts = (np.array([k.pt for k in kps], np.float32).reshape(-1, 2) + 0.5) / scale - 0.5
    desc = desc if desc is not None else np.zeros((0, 128), np.float32)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, pts=pts, desc=desc)
    return Features(pts, desc)


def candidate_pairs(cams: list[Camera], max_angle_deg: float = 65.0) -> list[tuple[int, int]]:
    """Pairs whose optical axes are close enough (per metadata) that the images can overlap."""
    cos_max = math.cos(math.radians(max_angle_deg))
    fwd = np.stack([c.forward_world for c in cams])
    dots = fwd @ fwd.T
    return [(i, j) for i in range(len(cams)) for j in range(i + 1, len(cams)) if dots[i, j] > cos_max]


_flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))


def match_pair(cams: list[Camera], feats: list[Features], i: int, j: int, ratio: float = 0.75,
               prior_deg: float = 8.0, ransac_px: float = 8.0, min_inliers: int = 20,
               max_matches: int = 400, rng: np.random.Generator | None = None) -> PairMatches | None:
    fi, fj = feats[i], feats[j]
    if len(fi.desc) < 2 or len(fj.desc) < 2:
        return None
    knn = _flann.knnMatch(fi.desc, fj.desc, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if len(good) < min_inliers:
        return None
    pi = fi.pts[[m.queryIdx for m in good]]
    pj = fj.pts[[m.trainIdx for m in good]]
    n_ratio = len(good)

    # Drop matches that disagree with the metadata pose by more than prior_deg
    cos_ang = np.sum(cams[i].rays_world(pi) * cams[j].rays_world(pj), axis=1)
    keep = cos_ang > math.cos(math.radians(prior_deg))
    pi, pj = pi[keep], pj[keep]
    n_prior = len(pi)
    if n_prior < min_inliers:
        return None

    # Rotation-only cameras are related by a homography; lens distortion is absorbed by the loose threshold
    _, mask = cv2.findHomography(pi, pj, cv2.RANSAC, ransac_px, maxIters=5000, confidence=0.999)
    if mask is None:
        return None
    inl = mask.ravel().astype(bool)
    pi, pj = pi[inl], pj[inl]
    n_inliers = len(pi)
    if n_inliers < min_inliers:
        return None

    if n_inliers > max_matches:
        rng = rng or np.random.default_rng(0)
        sel = rng.choice(n_inliers, max_matches, replace=False)
        pi, pj = pi[sel], pj[sel]
    return PairMatches(i, j, pi, pj, n_ratio, n_prior, n_inliers)
