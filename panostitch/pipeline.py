"""The complete stitching pipeline as library calls with progress reporting and cancellation.

    align()  : SIFT matching + bundle adjustment (step 2)
    blend()  : gains, seams, multi-band blending, zenith fill (step 3)
    stitch() : align + blend + write a 360 JPEG with GPano metadata

`progress(fraction, message)` receives 0..1 for the call it was passed to; `cancelled()` is polled between
images and raises Cancelled when it returns True. User-facing messages are in Danish (they end up in the GUI).
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .blend import MultiBandBlender, find_seams, overlap_stats, solve_gains
from .bundle import BAResult, bundle_adjust
from .camera import Camera
from .features import candidate_pairs, detect, match_pair
from .blend import max_levels
from .imageio import imread
from .layout import OUTPUT_SUFFIX, PANO_TYPES, coverage_estimate_deg, detect_type, largest_filled_rect, plan_canvas
from .metadata import ShotInfo, positions_world, read_folder, read_xmp
from .projection import Canvas, build_maps, feather_weight
from .render import linear_to_srgb, srgb_to_linear_lut
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
    max_matches: int = 200         # per image pair; bundle adjustment time is roughly proportional
    gain: bool = True
    anchor: str = "ground"         # ground | all | none: which images keep their as-shot exposure
    seams: str = "graphcut"        # graphcut | voronoi
    seam_width: int = 1536
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
        when = self.taken.strftime("%d-%m-%Y %H:%M") if self.taken else "ukendt tidspunkt"
        kind = PANO_TYPES[self.pano_type]
        if self.pano_type in ("180", "wide"):
            kind += f" (ca. {self.coverage_deg:.0f}°)"
        return f"{self.folder.name} — {kind} · {self.count} billeder · {self.model or 'ukendt kamera'} · {when}"

    def default_output_name(self, pano_type: str | None = None) -> str:
        stamp = self.taken.strftime("%Y-%m-%d_%H%M_") if self.taken else ""
        return f"{stamp}{self.folder.name}_{OUTPUT_SUFFIX[pano_type or self.pano_type]}.jpg"


def _jpegs(folder: Path) -> list[Path]:
    try:
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
    except OSError:
        return []


def find_panorama_sets(root: Path, max_depth: int = 3, min_images: int = 2) -> list[PanoramaSet]:
    """Folders under `root` (or root itself) that contain a DJI panorama: several JPEGs with drone metadata."""
    found = []

    def visit(folder: Path, depth: int):
        files = _jpegs(folder)
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
        raise ValueError(f"Fandt kun {len(shots)} billede(r) i {folder}")
    if all(s.focal_35mm == 0 for s in shots):
        raise ValueError("Billederne mangler brændvidde i EXIF – er det originalbillederne fra dronen?")
    cams = [Camera.from_shot(s) for s in shots]
    n = len(cams)

    feats = []
    for k, c in enumerate(cams):
        if cancelled():
            raise Cancelled
        progress(0.25 * k / n, f"Finder kendetegn i billede {k + 1} af {n}")
        feats.append(detect(c.meta.path, cache_dir=opt.cache_dir))

    pairs = candidate_pairs(cams)
    rng = np.random.default_rng(0)
    matches = []
    for k, (i, j) in enumerate(pairs):
        if cancelled():
            raise Cancelled
        if k % 5 == 0:
            progress(0.25 + 0.25 * k / len(pairs), f"Matcher overlappende billeder ({k + 1} af {len(pairs)})")
        m = match_pair(cams, feats, i, j, max_matches=opt.max_matches, rng=rng)
        if m is not None:
            matches.append(m)
    log(f"{len(matches)} af {len(pairs)} billedpar matchet")
    linked = np.zeros(n, int)
    for m in matches:
        linked[m.i] += 1
        linked[m.j] += 1

    positions, ground_h = None, 0.0
    if opt.parallax != "off":
        positions = positions_world(shots)
        ground_h = float(np.mean([s.relative_altitude for s in shots]))
        if ground_h <= 5:
            log("Flyvehøjden er ukendt eller meget lav – slår parallakse-korrektion fra")
            positions = None

    neighbors = candidate_pairs(cams, max_angle_deg=50.0)
    ignored: list[int] = []
    for attempt in range(3):
        if cancelled():
            raise Cancelled
        progress(0.5 + 0.15 * attempt, "Optimerer kameravinkler (kan tage et par minutter)")
        ba = bundle_adjust(cams, matches, neighbors=neighbors, positions=positions, ground_h=ground_h,
                           fit_positions=opt.parallax == "fit" and positions is not None,
                           position_sigma_m=opt.position_sigma, verbose=False)
        bad = np.where(np.abs(ba.omega_deg).max(axis=1) > opt.max_correction)[0]
        if not len(bad):
            break
        for k in bad:
            log(f"Ignorerer usikre matches for {cams[k].meta.name} ({np.abs(ba.omega_deg[k]).max():.1f}°)")
        ignored += [int(k) for k in bad]
        matches = [m for m in matches if m.i not in bad and m.j not in bad]

    after = np.array(list(ba.pair_rms_after.values()))
    log(f"Justering færdig: median fejl mellem billeder {np.median(after):.1f} px")
    progress(1.0, "Justering færdig")
    return AlignResult(shots, cams, ba.cams, ba, linked, len(pairs), len(matches), ignored)


# ---------------------------------------------------------------------------------------------------------
# Step 3: blending
# ---------------------------------------------------------------------------------------------------------

_LIN = srgb_to_linear_lut()


def _soft_clip(x: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Identity below the knee, smooth roll-off towards 1 above it (avoids hard clipping when brightening)."""
    over = x > knee
    if over.any():
        x = x.copy()
        x[over] = knee + (1 - knee) * np.tanh((x[over] - knee) / (1 - knee))
    return x


def apply_gain(img_u8: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Exposure change in linear light, returned as float32 sRGB 0-255 (keeps the camera's contrast/colour).

    The input is 8-bit and the gain is constant per channel, so this is a 256-entry lookup table per channel."""
    out = np.empty(img_u8.shape, np.float32)
    for c in range(3):
        lin = _LIN * np.float32(gain[c])
        if gain[c] > 1.0:
            lin = _soft_clip(lin)
        lut = (linear_to_srgb(lin) * 255).astype(np.float32)
        out[..., c] = lut[img_u8[..., c]]
    return out


def _source_scale(cam: Camera, target_width: int, oversample: float = 1.25) -> float:
    return min(1.0, target_width / (2 * math.pi * cam.focal) * oversample)


def _load_scaled(cam: Camera, target_width: int) -> tuple[np.ndarray, Camera]:
    """The image pre-shrunk to roughly the resolution it gets in the panorama (avoids aliasing)."""
    img = imread(cam.meta.path)
    s = _source_scale(cam, target_width)
    if s < 1.0:
        sc = cam.scaled(s)
        return cv2.resize(img, (sc.width, sc.height), interpolation=cv2.INTER_AREA), sc
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
    warped_lo, valid_lo, weight_lo = [], [], []
    for k, cam in enumerate(cams):
        if cancelled():
            raise Cancelled
        progress(0.12 * k / n, f"Forbereder billede {k + 1} af {n}")
        img, sc = _load_scaled(cam, opt.seam_width)
        mx, my, valid = build_maps(sc, rays)
        warped_lo.append(cv2.remap(img, mx, my, cv2.INTER_LINEAR))
        valid_lo.append(valid)
        weight_lo.append(feather_weight(sc, mx, my, valid))

    gains = np.ones((n, 3))
    if opt.gain:
        gains = solve_gains(*overlap_stats([_LIN[w] for w in warped_lo], valid_lo))
        if opt.anchor != "none":
            ref = [k for k, c in enumerate(cams) if opt.anchor == "all" or c.meta.gimbal_pitch < -10] \
                or list(range(n))
            gains /= np.exp(np.log(gains[ref]).mean(axis=0))
        stops = np.log2(gains @ np.array([0.114, 0.587, 0.299]))
        log(f"Eksponering udlignet: {stops.min():+.2f} til {stops.max():+.2f} stop")
    for k in range(n):
        warped_lo[k] = apply_gain(warped_lo[k], gains[k])

    progress(0.12, "Finder de bedste sømme mellem billederne")
    labels_lo = find_seams(warped_lo, valid_lo, weight_lo, opt.seams)
    del warped_lo, valid_lo, weight_lo

    # --- full resolution: make every label point to an image that is valid there
    W, H = canvas.width, canvas.height
    rays = canvas.rays()
    labels = cv2.resize(labels_lo, (W, H), interpolation=cv2.INTER_NEAREST)
    best_w = np.zeros((H, W), np.float32)
    best_i = np.full((H, W), -1, np.int16)
    for k, cam in enumerate(cams):
        if cancelled():
            raise Cancelled
        progress(0.2 + 0.2 * k / n, f"Lægger sømme i fuld opløsning ({k + 1} af {n})")
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
    for k, cam in enumerate(cams):
        if cancelled():
            raise Cancelled
        progress(0.4 + 0.5 * k / n, f"Blender billede {k + 1} af {n}")
        mask = (labels == k).astype(np.float32)
        if not mask.any():
            continue
        img, sc = _load_scaled(cam, sphere_equiv)
        mx, my, _ = build_maps(sc, rays, extend=True)
        warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        blender.feed(apply_gain(warped, gains[k]), mask)
    del rays
    progress(0.9, "Samler panoramaet")
    pano, covered = blender.result()
    del blender
    pano = np.clip(pano, 0, 255).astype(np.uint8)
    pano[~covered] = 0

    if canvas.is_full_sphere:
        lat = canvas.lats()
        coverage = float((covered * np.cos(lat)[:, None]).sum() / (np.cos(lat).sum() * W))
        log(f"Billederne dækker {coverage * 100:.0f} % af kuglen")
        if opt.fill_zenith and not covered[0].all():
            region = zenith_hole(covered)
            if region is not None:
                progress(0.95, "Udfylder himlen over dronen")
                pano = fill_zenith(pano, covered, region)
            else:
                log("Hullet i himlen er større end normalt (mangler der billeder?) – det efterlades sort")
    progress(1.0, "Blending færdig")
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
        log(f"Panoramatype genkendt: {PANO_TYPES[pano_type]}")
    canvas = plan_canvas(aligned.cams, pano_type, 360.0 / opt.width, opt.width)
    if canvas.is_full_sphere and opt.center_yaw is not None:
        canvas = Canvas.full_sphere(opt.width, opt.center_yaw)
    if not canvas.is_full_sphere:
        log(f"Udsnit: {canvas.lon_span:.0f}° × {canvas.lat_top - canvas.lat_bottom:.0f}°"
            f"{' (lodret)' if canvas.portrait else ''}")
    return pano_type, canvas


def stitch(folder: Path, out_path: Path, opt: Options, progress: ProgressFn = _noop, cancelled: CancelFn = _never,
           log: LogFn = _noop) -> Path:
    log(f"Stitcher {folder}")
    aligned = align(folder, opt, _sub(progress, 0.0, 0.45), cancelled, log)
    _, canvas = choose_canvas(aligned, opt, log)
    result = blend(aligned.cams, opt, canvas, _sub(progress, 0.45, 0.97), cancelled, log)
    progress(0.98, "Beskærer og gemmer")
    pano, gpano = finish(result, opt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_pano_jpeg(out_path, pano, gpano, quality=opt.jpeg_quality)
    log(f"Gemt: {out_path} ({pano.shape[1]} × {pano.shape[0]})")
    progress(1.0, "Færdig")
    return out_path
