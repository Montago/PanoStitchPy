"""Gain compensation, seam finding and wrap-aware multi-band blending for equirectangular panoramas."""
from __future__ import annotations

import math

import cv2
import numpy as np

# ---------------------------------------------------------------------------------------------------------
# Gain compensation (Brown & Lowe): per-channel gains that make overlaps agree while staying close to 1,
# so the camera's own per-image rendering (its "look") is kept as much as possible.
# ---------------------------------------------------------------------------------------------------------


def solve_gains(means: np.ndarray, counts: np.ndarray, sigma_n: float = 0.04, sigma_g: float = 0.5) -> np.ndarray:
    """means[i, j, c]: mean of image i (channel c, linear 0-1) over its overlap with j; counts[i, j]: overlap size.

    Returns gains (n, 3) minimising sum N_ij [(g_i m_ij - g_j m_ji)^2 / sigma_n^2 + (1 - g_i)^2 / sigma_g^2].
    """
    n = counts.shape[0]
    gains = np.ones((n, 3))
    for c in range(3):
        A = np.zeros((n, n))
        b = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j or counts[i, j] == 0:
                    continue
                N = counts[i, j]
                mi, mj = means[i, j, c], means[j, i, c]
                A[i, i] += N * (mi * mi / sigma_n ** 2 + 1 / sigma_g ** 2)
                A[i, j] -= N * mi * mj / sigma_n ** 2
                b[i] += N / sigma_g ** 2
        for i in range(n):
            if A[i, i] == 0:  # no overlaps at all
                A[i, i], b[i] = 1.0, 1.0
        gains[:, c] = np.linalg.solve(A, b)
    return gains


def overlap_stats(warped: list[np.ndarray], valid: list[np.ndarray], clip_lo: float = 0.003, clip_hi: float = 0.93,
                  saturation: list[float] | None = None):
    """Pairwise overlap means/counts from low-res warped linear images, ignoring clipped and black pixels.

    clip_lo/clip_hi are relative to each image's saturation level (1.0 for JPEGs)."""
    n = len(warped)
    sat = saturation or [1.0] * n
    usable = [v & (w.max(axis=2) < clip_hi * s) & (w.min(axis=2) > clip_lo * s)
              for w, v, s in zip(warped, valid, sat)]
    means = np.zeros((n, n, 3))
    counts = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            both = usable[i] & usable[j]
            k = int(both.sum())
            if k < 50:
                continue
            counts[i, j] = counts[j, i] = k
            means[i, j] = warped[i][both].mean(axis=0)
            means[j, i] = warped[j][both].mean(axis=0)
    return means, counts


# ---------------------------------------------------------------------------------------------------------
# Seams
# ---------------------------------------------------------------------------------------------------------


def find_seams(warped: list[np.ndarray], valid: list[np.ndarray], weight: list[np.ndarray],
               method: str = "graphcut", keep: tuple[int, np.ndarray] | None = None,
               avoid: list[tuple[int, np.ndarray]] = ()) -> np.ndarray:
    """Label image (H, W) int16 at the resolution of the inputs: index of the image used per pixel, -1 = none.

    'graphcut' routes seams through areas where the overlapping images agree (colour + gradient cost),
    'voronoi' simply takes the image whose centre is closest. Images are cropped to their bounding box so the
    graph cut only works on actual overlaps. keep = (image, mask) reserves the masked area for that image
    (which must cover it); the seams then run around it. avoid = [(image, mask), ...] takes the masked area
    away from that image wherever another image still covers it.
    """
    n = len(warped)
    valid = list(valid)
    if keep is not None:
        k_img, k_mask = keep
        valid = [v if i == k_img else v & ~k_mask for i, v in enumerate(valid)]
    for i, mask in avoid:
        others = np.zeros_like(valid[i])
        for j, v in enumerate(valid):
            if j != i:
                others |= v
        valid[i] = valid[i] & ~(mask & others)
    weight = [np.where(v, w, 0) for v, w in zip(valid, weight)]
    H, W = valid[0].shape
    best_w = np.zeros((H, W), np.float32)
    best_i = np.full((H, W), -1, np.int16)
    for i in range(n):
        better = weight[i] > best_w
        best_w[better] = weight[i][better]
        best_i[better] = i
    if method == "voronoi":
        return best_i

    corners, imgs, masks, boxes = [], [], [], []
    for i in range(n):
        ys, xs = np.nonzero(valid[i])
        if len(xs) == 0:
            boxes.append(None)
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        boxes.append((x0, y0, x1, y1))
        corners.append((int(x0), int(y0)))
        imgs.append(cv2.UMat(np.ascontiguousarray(warped[i][y0:y1, x0:x1], dtype=np.float32)))
        masks.append(cv2.UMat((valid[i][y0:y1, x0:x1] * 255).astype(np.uint8)))

    finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    masks = finder.find(imgs, corners, masks)

    labels = np.full((H, W), -1, np.int16)
    k = 0
    for i in range(n):
        if boxes[i] is None:
            continue
        x0, y0, x1, y1 = boxes[i]
        m = masks[k].get() > 0
        k += 1
        region = labels[y0:y1, x0:x1]
        region[m & (region < 0)] = i
    # anything covered but left unlabeled by the cut falls back to the closest image centre
    holes = (labels < 0) & (best_i >= 0)
    labels[holes] = best_i[holes]
    return labels


# ---------------------------------------------------------------------------------------------------------
# Multi-band blending with horizontal wrap-around (lon -180 == +180)
# ---------------------------------------------------------------------------------------------------------

_PAD = 8


def _wrap(x: np.ndarray, p: int) -> np.ndarray:
    return cv2.copyMakeBorder(x, 0, 0, p, p, cv2.BORDER_WRAP)


def pyr_down(x: np.ndarray, wrap: bool = True) -> np.ndarray:
    if not wrap:
        return cv2.pyrDown(x)
    w = x.shape[1]
    y = cv2.pyrDown(_wrap(x, _PAD))
    return y[:, _PAD // 2: _PAD // 2 + w // 2]


def pyr_up(x: np.ndarray, shape: tuple[int, int], wrap: bool = True) -> np.ndarray:
    h, w = shape
    if not wrap:
        return cv2.pyrUp(x, dstsize=(w, h))
    p = _PAD // 2
    xp = _wrap(x, p)
    y = cv2.pyrUp(xp, dstsize=(xp.shape[1] * 2, h))
    return y[:, 2 * p: 2 * p + w]


def max_levels(height: int, width: int, wanted: int, wrap: bool) -> int:
    """Fewer pyramid levels for small images (coarsest level at least ~4 px) and, when wrapping, as many as the
    width can be halved exactly."""
    levels = max(1, min(wanted, int(math.log2(max(4, min(height, width)) / 4))))
    if wrap:
        while levels > 1 and width % (1 << levels):
            levels -= 1
    return levels


class MultiBandBlender:
    def __init__(self, height: int, width: int, levels: int, wrap: bool = True):
        if wrap and width % (1 << levels):
            raise ValueError(f"width {width} must be divisible by 2^{levels} for wrap-around blending")
        self.levels = levels
        self.wrap = wrap
        self.shapes = [(height, width)]
        for _ in range(levels):
            h, w = self.shapes[-1]
            self.shapes.append(((h + 1) // 2, w // 2 if wrap else (w + 1) // 2))
        self.acc = [np.zeros((h, w, 3), np.float32) for h, w in self.shapes]
        self.wacc = [np.zeros((h, w), np.float32) for h, w in self.shapes]
        self.free: list[np.ndarray] | None = None  # 1 - reserved area, per level

    def reserve(self, area: np.ndarray) -> None:
        """Keep an area (H, W, values in [0, 1]) for one image at every level: images fed with reserved=False
        get no weight there. Plain multi-band blending would still mix the other images' coarse levels into an
        area that is smaller than the coarsest scale (e.g. the sun's broad glow)."""
        area = area.astype(np.float32)
        self.free = [1 - cv2.resize(area, (w, h), interpolation=cv2.INTER_AREA) for h, w in self.shapes]

    def feed(self, img: np.ndarray, mask: np.ndarray, reserved: bool = True) -> None:
        """img: float32 (H, W, 3), mask: float32 (H, W) in [0, 1]. reserved=False keeps this image out of the
        area given to reserve()."""
        g = img
        m = mask
        free = None if reserved else self.free
        for k in range(self.levels):
            g_next = pyr_down(g, self.wrap)
            lap = g - pyr_up(g_next, self.shapes[k], self.wrap)
            mk = m if free is None else m * free[k]
            self.acc[k] += lap * mk[..., None]
            self.wacc[k] += mk
            g = g_next
            m = pyr_down(m, self.wrap)
        if free is not None:
            m = m * free[-1]
        self.acc[-1] += g * m[..., None]
        self.wacc[-1] += m

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        out = None
        for k in range(self.levels, -1, -1):
            w = self.wacc[k]
            level = self.acc[k] / np.maximum(w, 1e-5)[..., None]
            out = level if out is None else pyr_up(out, self.shapes[k], self.wrap) + level
        return out, self.wacc[0] > 1e-3
