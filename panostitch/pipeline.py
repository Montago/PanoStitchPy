"""The complete stitching pipeline as library calls with progress reporting and cancellation.

    align()  : SIFT matching + bundle adjustment (step 2)
    blend()  : gains, seams, multi-band blending, zenith fill (step 3)
    stitch() : align + blend + write a 360 JPEG with GPano metadata

`progress(fraction, message)` receives 0..1 for the call it was passed to; `cancelled()` is polled between
images and raises Cancelled when it returns True. Progress and log messages are user-facing (they end up in the GUI).
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .blend import MultiBandBlender, find_seams, overlap_stats, solve_gains
from .bundle import BAResult, bundle_adjust
from .camera import Camera
from .features import candidate_pairs, detect, match_pair
from .horizon import detect as detect_horizon, horizon_gray
from .blend import max_levels
from .imageio import imread
from .layout import OUTPUT_SUFFIX, PANO_TYPES, coverage_estimate_deg, detect_type, largest_filled_rect, plan_canvas
from .metadata import ShotInfo, image_files, positions_world, read_folder, read_xmp
from .raw import camera_wb, is_raw, load_linear, tone as raw_tone
from .projection import Canvas, build_maps, feather_weight
from .render import linear_to_srgb, srgb_to_linear_lut
from .sun import find_sun
from .xmp import full_sphere_gpano, write_pano_jpeg
from .zenith import fill_zenith, zenith_hole

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]
LogFn = Callable[[str], None]


class Cancelled(Exception):
    pass


def _noop(*_args) -> None:
    pass


def _never() -> bool:
    return False


def _sub(progress: ProgressFn, a: float, b: float) -> ProgressFn:
    return lambda f, msg: progress(a + (b - a) * f, msg)


@dataclass
class Options:
    width: int = 8192              # width of a full 360 sphere (partial panoramas get the same detail per degree);
                                   # must be divisible by 2^levels
    pano_type: str = "auto"        # auto | sphere | 180 | wide | vertical (see layout.PANO_TYPES)
    crop: bool = True              # partial panoramas: crop away uncovered (black) borders
    parallax: str = "fit"          # off | gps | fit
    position_sigma: float = 0.3    # metres, GPS prior when parallax == "fit"
    max_correction: float = 6.0    # degrees; larger corrections are treated as bad matches
    horizon: bool = True           # level the panorama on sea horizons found in the images
    max_matches: int = 200         # per image pair; bundle adjustment time is roughly proportional
    gain: bool = True
    anchor: str = "ground"         # ground | all | none: which images keep their as-shot exposure
    gain_strength: float = 0.6     # how much of the exposure difference to even out (1 = all); the rest is left to
                                   # the multi-band blending, so images shot towards the sun are not pushed into white
    seams: str = "graphcut"        # graphcut | voronoi
    seam_width: int = 1536
    sun: bool = True               # keep the sun's glare from a single image and steer seams around flare ghosts
    raw_key: float = 0.0           # DNG: log-average display luminance to expose for; 0 = keep the ground at the
                                   # brightness the drone intended (DJI's own panoramas range 0.07-0.21)
    levels: int = 9
    center_yaw: float | None = None  # sphere: compass heading at the image centre (None = start direction)
    fill_zenith: bool = True
    jpeg_quality: int = 93
    cache_dir: Path | None = None  # SIFT feature cache


# ---------------------------------------------------------------------------------------------------------
# Finding panoramas
# ---------------------------------------------------------------------------------------------------------

@dataclass
class PanoramaSet:
    folder: Path
    count: int
    model: str
    taken: dt.datetime | None
    pano_type: str               # detected from the shooting pattern
    coverage_deg: float          # rough horizontal coverage

    @property
    def label(self) -> str:
        when = self.taken.strftime("%d-%m-%Y %H:%M") if self.taken else "unknown time"
        kind = PANO_TYPES[self.pano_type]
        if self.pano_type in ("180", "wide"):
            kind += f" (approx. {self.coverage_deg:.0f}°)"
        return f"{self.folder.name} — {kind} · {self.count} images · {self.model or 'unknown camera'} · {when}"

    def default_output_name(self, pano_type: str | None = None) -> str:
        stamp = self.taken.strftime("%Y-%m-%d_%H%M_") if self.taken else ""
        return f"{stamp}{self.folder.name}_{OUTPUT_SUFFIX[pano_type or self.pano_type]}.jpg"


def find_panorama_sets(root: Path, max_depth: int = 3, min_images: int = 2) -> list[PanoramaSet]:
    """Folders under `root` (or root itself) that contain a DJI panorama: several JPEGs or DNGs with drone metadata."""
    found = []

    def visit(folder: Path, depth: int):
        files = image_files(folder)
        if len(files) >= min_images:
            xmp = read_xmp(files[0])
            if "drone-dji:GimbalYawDegree" in xmp:
                taken = None
                if xmp.get("xmp:CreateDate"):
                    try:
                        taken = dt.datetime.fromisoformat(xmp["xmp:CreateDate"])
                    except ValueError:
                        pass
                yaws = []
                for f in files:
                    try:
                        yaws.append(float(read_xmp(f, max_bytes=64 * 1024).get("drone-dji:GimbalYawDegree", "nan")))
                    except ValueError:
                        pass
                yaws = [y for y in yaws if not math.isnan(y)]
                found.append(PanoramaSet(folder, len(files), xmp.get("drone-dji:ProductName", ""), taken,
                                         detect_type(yaws), coverage_estimate_deg(yaws)))
                return
        if depth < max_depth:
            try:
                subdirs = sorted(p for p in folder.iterdir() if p.is_dir())
            except OSError:
                return
            for sub in subdirs:
                visit(sub, depth + 1)

    visit(root, 0)
    return found


# ---------------------------------------------------------------------------------------------------------
# Step 2: alignment
# ---------------------------------------------------------------------------------------------------------

@dataclass
class AlignResult:
    shots: list[ShotInfo]
    meta_cams: list[Camera]
    cams: list[Camera]
    ba: BAResult
    linked: np.ndarray            # number of matched pairs per image
    n_pairs: int
    n_matched: int
    ignored: list[int] = field(default_factory=list)  # images whose matches were dropped as implausible


def align(folder: Path, opt: Options, progress: ProgressFn = _noop, cancelled: CancelFn = _never,
          log: LogFn = _noop) -> AlignResult:
    shots = read_folder(folder)
    if len(shots) < 2:
        raise ValueError(f"Found only {len(shots)} image(s) in {folder}")
    if all(s.focal_35mm == 0 for s in shots):
        raise ValueError("The images have no focal length in EXIF – are they the original images from the drone?")
    cams = [Camera.from_shot(s) for s in shots]
    n = len(cams)

    feats = []
    for k, c in enumerate(cams):
        if cancelled():
            raise Cancelled
        progress(0.25 * k / n, f"Finding features in image {k + 1} of {n}")
        feats.append(detect(c.meta.path, cache_dir=opt.cache_dir))

    pairs = candidate_pairs(cams)
    rng = np.random.default_rng(0)
    matches = []
    for k, (i, j) in enumerate(pairs):
        if cancelled():
            raise Cancelled
        if k % 5 == 0:
            progress(0.25 + 0.25 * k / len(pairs), f"Matching overlapping images ({k + 1} of {len(pairs)})")
        m = match_pair(cams, feats, i, j, max_matches=opt.max_matches, rng=rng)
        if m is not None:
            matches.append(m)
    log(f"{len(matches)} of {len(pairs)} image pairs matched")
    linked = np.zeros(n, int)
    for m in matches:
        linked[m.i] += 1
        linked[m.j] += 1

    positions, ground_h = None, 0.0
    if opt.parallax != "off":
        positions = positions_world(shots)
        ground_h = float(np.mean([s.relative_altitude for s in shots]))
        if ground_h <= 5:
            log("Flight altitude unknown or very low – parallax correction disabled")
            positions = None
    altitude = float(np.mean([s.relative_altitude for s in shots]))
    if 0 < altitude <= 5:
        # Close to the ground the nearby ground shows parallax (the lens is a few cm from the gimbal's pivot),
        # which tilts the rotations away from the horizon. Matches count by their distance (flat ground at
        # the flight altitude): distant content (>= 30 m: the horizon, far shores, clouds) fully, the ground
        # a few metres away only a little, so it still holds neighbouring images together.
        n_far = 0
        weighted = []
        for m in matches:
            dy = cams[m.i].rays_world(m.pts_i.astype(float))[:, 1]
            dist = np.where(dy < -1e-3, altitude / np.maximum(-dy, 1e-3), np.inf)
            w = np.clip(dist / 30.0, 0.0, 1.0) ** 2
            n_far += int((w > 0.5).sum())
            weighted.append(replace(m, weight=w))
        matches = weighted
        log(f"Low flight ({altitude:.1f} m): nearby ground counts less for the angles "
            f"({n_far} distant matches)")

    # sea horizons in view level the images that see them
    horizons = []
    if opt.horizon:
        for k, c in enumerate(cams):
            if abs(c.meta.gimbal_pitch) <= 35:
                gray, scale = horizon_gray(c.meta.path, c.width)
                h = detect_horizon(c, gray, scale, k, altitude)
                if h is not None:
                    horizons.append(h)
        if horizons:
            log(f"Sea horizon found in {len(horizons)} image(s) – used to level the panorama")

    neighbors = candidate_pairs(cams, max_angle_deg=50.0)
    ignored: list[int] = []
    for attempt in range(3):
        if cancelled():
            raise Cancelled
        progress(0.5 + 0.15 * attempt, "Optimising camera angles (may take a few minutes)")
        ba = bundle_adjust(cams, matches, neighbors=neighbors, positions=positions, ground_h=ground_h,
                           horizons=[h for h in horizons if h.image not in ignored],
                           fit_positions=opt.parallax == "fit" and positions is not None,
                           position_sigma_m=opt.position_sigma, verbose=False)
        bad = np.where(np.abs(ba.omega_deg).max(axis=1) > opt.max_correction)[0]
        if not len(bad):
            break
        for k in bad:
            log(f"Ignoring unreliable matches for {cams[k].meta.name} ({np.abs(ba.omega_deg[k]).max():.1f}°)")
        ignored += [int(k) for k in bad]
        matches = [m for m in matches if m.i not in bad and m.j not in bad]

    after = np.array(list(ba.pair_rms_after.values()))
    log(f"Alignment done: median error between images {np.median(after):.1f} px")
    progress(1.0, "Alignment done")
    return AlignResult(shots, cams, ba.cams, ba, linked, len(pairs), len(matches), ignored)


# ---------------------------------------------------------------------------------------------------------
# Step 3: blending
# ---------------------------------------------------------------------------------------------------------

_LIN = srgb_to_linear_lut()


def _srgb_ext(lin: np.ndarray) -> np.ndarray:
    """sRGB encoding (0-1) continued above 1, where values stand for light brighter than white."""
    lin = np.maximum(lin, 0.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def _srgb_ext_inverse(v: np.ndarray) -> np.ndarray:
    v = np.maximum(v, 0.0)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


# how "clipped" an 8-bit value is: 0 up to 235, rising smoothly to 1 at 255
_t = np.clip((np.arange(256, dtype=np.float32) - 235.0) / 20.0, 0.0, 1.0)
_CLIPPED = _t * _t * (3 - 2 * _t)
del _t


def _push_clipped(out: np.ndarray, img_u8: np.ndarray, white: float) -> None:
    """Pixels near 255 in all three channels -> towards neutral `white` (in place, extended sRGB)."""
    _push_weight(out, _CLIPPED[img_u8.min(axis=2)], white)


def _push_weight(out: np.ndarray, clipped: np.ndarray, white: float) -> None:
    if clipped.any():  # smooth, so JPEG noise in a near-white sky does not turn into speckles
        clipped = cv2.GaussianBlur(clipped, (0, 0), 2.0)
    idx = np.nonzero(clipped > 1e-3)
    if idx[0].size:
        top = np.float32(_srgb_ext(np.float32(white)) * 255)
        out[idx] += clipped[idx][:, None] * (top - out[idx])


def apply_gain(img_u8: np.ndarray, gain: np.ndarray, white: float | None = None,
               saturation: float | None = None) -> np.ndarray:
    """Exposure change in linear light (keeps the camera's contrast/colour), returned as float32 sRGB where
    0-255 is the normal range and values above 255 are brighter than white; compress_highlights() maps those
    back once everything is blended. Nothing is clipped here, so a brightened image keeps the detail in its
    highlights.

    A clipped pixel only says "at least this bright", so pixels that are near 255 in all three channels are
    pushed towards neutral `white` (the largest gain of the panorama): clipped clouds and the sun's core stay
    at the top instead of turning grey. Coloured highlights (only some channels high) are left alone.
    The gain is constant per channel, so the exposure change is a 256-entry lookup table per channel.

    A float image (a decoded DNG) is already linear; `saturation` is the value at which it is clipped."""
    if img_u8.dtype != np.uint8:
        out = (_srgb_ext(img_u8 * gain.astype(np.float32)[None, None, :]) * 255).astype(np.float32)
        if white is not None:
            t = np.clip((img_u8.min(axis=2) / np.float32(saturation or 1.0) - 0.92) / 0.08, 0, 1)
            _push_weight(out, (t * t * (3 - 2 * t)).astype(np.float32), white)
        return out
    out = np.empty(img_u8.shape, np.float32)
    for c in range(3):
        lut = (_srgb_ext(_LIN * np.float32(gain[c])) * 255).astype(np.float32)
        out[..., c] = lut[img_u8[..., c]]
    if white is not None:
        _push_clipped(out, img_u8, white)
    return out


def _roll_off(y: np.ndarray, knee: float, white: float) -> np.ndarray:
    """Identity up to the knee, then a smooth curve (slope 1 at the knee) that reaches 1 at `white`."""
    w = (white - knee) / (1 - knee)
    e = np.maximum(y - knee, 0) / (1 - knee)
    return np.where(y > knee, knee + (1 - knee) * e * (1 + e / (w * w)) / (1 + e), y)


def _auto_exposure(lum: np.ndarray, tone, key: float, knee: float, max_white: float) -> float:
    """Exposure factor that gives the panorama a log-average display luminance of `key`, taking the highlight
    roll-off and the tone curve into account (lum: a sample of linear luminances)."""
    lum = lum[lum > 0]

    def display_key(g: float) -> float:
        y = lum * g
        white = min(max(float(np.percentile(y, 99.9)), 1.02), max_white)
        disp = tone(_roll_off(y, knee, white))
        return float(np.exp(np.mean(np.log(np.maximum(_srgb_ext_inverse(disp), 1e-4)))))

    lo, hi = -8.0, 8.0   # stops
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if display_key(2.0 ** mid) < key else (lo, mid)
    return 2.0 ** ((lo + hi) / 2)


def compress_highlights(pano: np.ndarray, covered: np.ndarray, knee: float = 0.7, max_white: float = 4.0,
                        fade: float = 0.8, rows: int = 256, tone=None, key: float | None = None) -> np.ndarray:
    """Blended float sRGB (values above 255 = brighter than white) -> uint8.

    The brightest 0.1 % of the panorama (by luminance, typically the sun) becomes white; luminance above the
    knee (linear) is rolled off towards it with all three channels scaled alike, so a bright sunset sky keeps
    its colour. Only the very brightest part (rolled-off luminance above `fade`) loses its colour towards
    white, like the sun and its inner halo do in the camera's own images. A panorama without such highlights
    is just clipped. `tone` maps the result (linear 0-1) to display values 0-1; default sRGB.

    key: if given, the overall exposure is chosen so the log-average display luminance becomes `key` (raw
    images: the drone exposes them dark to protect the highlights and brightens its JPEGs per scene)."""
    tone = tone or linear_to_srgb
    lum_w = np.array([0.0722, 0.7152, 0.2126], np.float32)  # BGR
    sample = _srgb_ext_inverse(pano[::4, ::4][covered[::4, ::4]] / 255.0) @ lum_w
    gain = 1.0
    if key is not None and sample.size:
        gain = _auto_exposure(sample, tone, key, knee, max_white)
    white = float(np.percentile(sample * gain, 99.9)) if sample.size else 1.0
    white = min(white, max_white)
    if white <= 1.02 and tone is linear_to_srgb and gain == 1.0:
        return np.clip(pano, 0, 255).astype(np.uint8)
    white = max(white, 1.02)
    out = np.empty(pano.shape, np.uint8)
    for y0 in range(0, pano.shape[0], rows):
        lin = _srgb_ext_inverse(pano[y0:y0 + rows] / np.float32(255)).astype(np.float32)
        if gain != 1.0:
            lin *= np.float32(gain)
        lum = lin @ lum_w
        new_lum = _roll_off(lum, knee, white)
        lin *= (new_lum / np.maximum(lum, 1e-6))[..., None]
        t = np.clip((new_lum - fade) / (1 - fade), 0, 1)
        t = (t * t * (3 - 2 * t))[..., None]
        lin += t * (new_lum[..., None] - lin)
        out[y0:y0 + rows] = np.rint(np.clip(tone(lin), 0, 1) * 255).astype(np.uint8)
    return out


def _source_scale(cam: Camera, target_width: int, oversample: float = 1.25) -> float:
    return min(1.0, target_width / (2 * math.pi * cam.focal) * oversample)


class _Frames:
    """Loads the source images for blending. JPEGs come as uint8. DNGs come as linear float32, developed with
    one white balance for the whole panorama and scaled by their EXIF exposure towards that of the ground
    images (the drone exposes every frame differently). Like the gains for JPEGs, only `strength` of the
    difference is evened out: the drone's own panoramas keep the ground about a stop brighter than a fully
    matched exposure would. `saturation[k]` is the value at which frame k is clipped."""

    def __init__(self, cams: list[Camera], strength: float = 1.0):
        self.raw = [is_raw(c.meta.path) for c in cams]
        self.scale = np.ones(len(cams))
        self.wb = None
        if any(self.raw):
            self.wb = np.median([camera_wb(c.meta.path) for c, r in zip(cams, self.raw) if r], axis=0)
            exps = np.array([c.meta.sensor_exposure for c in cams])
            intended = np.array([2.0 ** c.meta.baseline_exposure for c in cams])
            ground = ([e for e, c, r in zip(exps, cams, self.raw) if r and c.meta.gimbal_pitch < -10]
                      or [e for e, r in zip(exps, self.raw) if r])
            ref = float(np.exp(np.mean(np.log(ground))))
            # between the drone's own rendering of each frame (strength 0) and fully matched exposure (1)
            matched = ref / np.maximum(exps, 1e-12)
            scale = matched ** strength * intended ** (1 - strength)
            # keep the ground frames at the brightness the drone intended for them (like anchor="ground")
            g = ([k for k, (c, r) in enumerate(zip(cams, self.raw)) if r and c.meta.gimbal_pitch < -10]
                 or [k for k, r in enumerate(self.raw) if r])
            scale *= np.exp(np.mean(np.log(intended[g])) - np.mean(np.log(scale[g])))
            self.scale = np.where(self.raw, scale, 1.0)
        self.saturation = [float(s) for s in self.scale]

    @property
    def any_raw(self) -> bool:
        return any(self.raw)

    def load(self, k: int, cam: Camera, target_width: int) -> tuple[np.ndarray, Camera]:
        """The image pre-shrunk to roughly the resolution it gets in the panorama (avoids aliasing)."""
        s = _source_scale(cam, target_width)
        if self.raw[k]:
            img = load_linear(cam.meta.path, wb=self.wb, half=s <= 0.5)
            img *= np.float32(self.scale[k])
        else:
            img = imread(cam.meta.path)
        if s < 1.0:
            sc = cam.scaled(s)
            return cv2.resize(img, (sc.width, sc.height), interpolation=cv2.INTER_AREA), sc
        if img.shape[1] != cam.width:
            img = cv2.resize(img, (cam.width, cam.height), interpolation=cv2.INTER_LINEAR)
        return img, cam


@dataclass
class BlendResult:
    canvas: Canvas
    pano: np.ndarray        # uint8 BGR, (H, W, 3)
    covered: np.ndarray     # bool, pixels covered by at least one image
    labels: np.ndarray      # int16, image index per pixel (-1 = none)
    gains: np.ndarray       # (n, 3)


def blend(cams: list[Camera], opt: Options, canvas: Canvas, progress: ProgressFn = _noop,
          cancelled: CancelFn = _never, log: LogFn = _noop) -> BlendResult:
    n = len(cams)
    sphere_equiv = 360.0 / canvas.deg_per_px  # source images are shrunk to the detail level of the output

    # --- low resolution: gains + seams
    seam_canvas = canvas.with_width(max(64, int(round(canvas.width * opt.seam_width / sphere_equiv))))
    rays = seam_canvas.rays()
    frames = _Frames(cams, opt.gain_strength if opt.gain else 0.0)
    if frames.any_raw:
        log("Developing the raw (DNG) images with one white balance, matched by their exposure")
    warped_lo, valid_lo, weight_lo = [], [], []
    for k, cam in enumerate(cams):
        if cancelled():
            raise Cancelled
        progress(0.12 * k / n, f"Preparing image {k + 1} of {n}")
        img, sc = frames.load(k, cam, opt.seam_width)
        mx, my, valid = build_maps(sc, rays)
        warped_lo.append(cv2.remap(img, mx, my, cv2.INTER_LINEAR))
        valid_lo.append(valid)
        weight_lo.append(feather_weight(sc, mx, my, valid))

    # the sun's glare differs between images: one image keeps it, the seams steer around flare ghosts, and
    # neither counts for exposure matching
    sun = (find_sun(cams, warped_lo, valid_lo, rays, 360.0 / opt.seam_width, saturation=frames.saturation)
           if opt.sun else None)
    keep, avoid, sun_area = None, [], None
    gain_valid = valid_lo
    if sun is not None:
        sun_area = sun.keep_mask(rays, valid_lo[sun.image], 360.0 / opt.seam_width)
        keep = (sun.image, sun_area)
        for k, cam in enumerate(cams):
            band = sun.flare_band(cam, rays)
            if band is not None:
                avoid.append((k, band & ~sun_area))
        log(sun.describe(cams) + (f"; avoiding lens flare in {len(avoid)} images" if avoid else ""))
        bands = dict(avoid)
        gain_valid = [v & ~sun_area & ~bands.get(k, np.False_) for k, v in enumerate(valid_lo)]

    gains = np.ones((n, 3))
    if opt.gain:
        gains = solve_gains(*overlap_stats([w if w.dtype != np.uint8 else _LIN[w] for w in warped_lo], gain_valid,
                                           saturation=frames.saturation))
        if opt.anchor != "none":
            ref = [k for k, c in enumerate(cams) if opt.anchor == "all" or c.meta.gimbal_pitch < -10] \
                or list(range(n))
            gains /= np.exp(np.log(gains[ref]).mean(axis=0))
        if not frames.any_raw:  # raw frames: the look is set in _Frames, what is left here are measurement errors
            gains = gains ** opt.gain_strength
        stops = np.log2(gains @ np.array([0.114, 0.587, 0.299]))
        log(f"Exposure balanced: {stops.min():+.2f} to {stops.max():+.2f} stops")
    # what a clipped pixel is pushed up to: the brightest level any frame can show
    white = max(float(max(g.max() * s for g, s in zip(gains, frames.saturation))), 1.0)
    for k in range(n):
        warped_lo[k] = apply_gain(warped_lo[k], gains[k], white, frames.saturation[k])

    progress(0.12, "Finding the best seams between the images")
    labels_lo = find_seams(warped_lo, valid_lo, weight_lo, opt.seams, keep, avoid)
    del warped_lo, valid_lo, weight_lo, keep, avoid, gain_valid

    # --- full resolution: make every label point to an image that is valid there
    W, H = canvas.width, canvas.height
    rays = canvas.rays()
    labels = cv2.resize(labels_lo, (W, H), interpolation=cv2.INTER_NEAREST)
    best_w = np.zeros((H, W), np.float32)
    best_i = np.full((H, W), -1, np.int16)
    for k, cam in enumerate(cams):
        if cancelled():
            raise Cancelled
        progress(0.2 + 0.2 * k / n, f"Placing seams at full resolution ({k + 1} of {n})")
        sc = cam.scaled(_source_scale(cam, sphere_equiv))
        mx, my, valid = build_maps(sc, rays)
        w = feather_weight(sc, mx, my, valid)
        labels[(labels == k) & ~valid] = -2
        better = w > best_w
        best_w[better] = w[better]
        best_i[better] = k
    fix = (labels == -2) | ((labels == -1) & (best_i >= 0))
    labels[fix] = best_i[fix]
    del best_w, best_i

    # --- multi-band blend (wrapping around at +-180 degrees when the canvas goes all the way round)
    blender = MultiBandBlender(H, W, max_levels(H, W, opt.levels, canvas.wraps), wrap=canvas.wraps)
    if sun_area is not None:  # the sun image's glow is broad: keep the other images out of it at every scale
        blender.reserve(cv2.resize(sun_area.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR))
    for k, cam in enumerate(cams):
        if cancelled():
            raise Cancelled
        progress(0.4 + 0.5 * k / n, f"Blending image {k + 1} of {n}")
        mask = (labels == k).astype(np.float32)
        if not mask.any():
            continue
        img, sc = frames.load(k, cam, sphere_equiv)
        mx, my, _ = build_maps(sc, rays, extend=True)
        warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        del img
        blender.feed(apply_gain(warped, gains[k], white, frames.saturation[k]), mask,
                     reserved=sun is not None and k == sun.image)
    del rays
    progress(0.9, "Assembling the panorama")
    pano, covered = blender.result()
    del blender
    if frames.any_raw:
        pano = compress_highlights(pano, covered, tone=raw_tone, key=opt.raw_key if opt.raw_key > 0 else None)
    else:
        pano = compress_highlights(pano, covered)
    pano[~covered] = 0

    if canvas.is_full_sphere:
        lat = canvas.lats()
        coverage = float((covered * np.cos(lat)[:, None]).sum() / (np.cos(lat).sum() * W))
        log(f"The images cover {coverage * 100:.0f} % of the sphere")
        if opt.fill_zenith and not covered[0].all():
            region = zenith_hole(covered)
            if region is not None:
                progress(0.95, "Filling in the sky above the drone")
                pano = fill_zenith(pano, covered, region)
            else:
                log("The gap in the sky is larger than usual (are images missing?) – it is left black")
    progress(1.0, "Blending done")
    return BlendResult(canvas, pano, covered, labels, gains)


# ---------------------------------------------------------------------------------------------------------
# Everything
# ---------------------------------------------------------------------------------------------------------

def finish(result: BlendResult, opt: Options) -> tuple[np.ndarray, dict | None]:
    """Crop a partial panorama to its fully covered area; returns the image and its GPano metadata (or None for
    projections that 360 viewers do not understand, e.g. vertical panoramas)."""
    canvas, pano, covered = result.canvas, result.pano, result.covered
    H, W = covered.shape
    if canvas.is_full_sphere:
        return pano, full_sphere_gpano(W, H, canvas.center_yaw)
    x0, y0, x1, y1 = 0, 0, W, H
    if opt.crop:
        if canvas.wraps:  # a 360 band: keep the full width, crop the rows
            full_rows = np.where(covered.all(axis=1))[0]
            if len(full_rows):
                y0, y1 = int(full_rows.min()), int(full_rows.max()) + 1
        else:
            x0, y0, x1, y1 = largest_filled_rect(covered)
    return pano[y0:y1, x0:x1].copy(), canvas.gpano_crop(x0, y0, x1, y1)


def choose_canvas(aligned: AlignResult, opt: Options, log: LogFn = _noop) -> tuple[str, Canvas]:
    pano_type = opt.pano_type
    if pano_type == "auto":
        pano_type = detect_type([s.gimbal_yaw for s in aligned.shots])
        log(f"Panorama type detected: {PANO_TYPES[pano_type]}")
    canvas = plan_canvas(aligned.cams, pano_type, 360.0 / opt.width, opt.width)
    if canvas.is_full_sphere and opt.center_yaw is not None:
        canvas = Canvas.full_sphere(opt.width, opt.center_yaw)
    if not canvas.is_full_sphere:
        log(f"Extent: {canvas.lon_span:.0f}° × {canvas.lat_top - canvas.lat_bottom:.0f}°"
            f"{' (vertical)' if canvas.portrait else ''}")
    return pano_type, canvas


def stitch(folder: Path, out_path: Path, opt: Options, progress: ProgressFn = _noop, cancelled: CancelFn = _never,
           log: LogFn = _noop) -> Path:
    log(f"Stitching {folder}")
    aligned = align(folder, opt, _sub(progress, 0.0, 0.45), cancelled, log)
    _, canvas = choose_canvas(aligned, opt, log)
    result = blend(aligned.cams, opt, canvas, _sub(progress, 0.45, 0.97), cancelled, log)
    progress(0.98, "Cropping and saving")
    pano, gpano = finish(result, opt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_pano_jpeg(out_path, pano, gpano, quality=opt.jpeg_quality)
    log(f"Saved: {out_path} ({pano.shape[1]} × {pano.shape[0]})")
    progress(1.0, "Done")
    return out_path
