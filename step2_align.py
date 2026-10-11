"""Step 2: refine every image's rotation (yaw, pitch AND roll) from image content.

Pipeline: SIFT on half-res images -> match only pairs that overlap per metadata -> drop matches inconsistent
with the metadata pose -> RANSAC homography -> bundle adjustment with the metadata as weak prior
(optionally also estimating the drone's position per shot, see --parallax).

Outputs (in --out): cameras.json, step2_*.jpg previews (same set as step 1), and a per-image correction table.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from panostitch.camera import save_cameras
from panostitch.pipeline import Options, align
from panostitch.render import render_all


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=Path("output/step2"))
    ap.add_argument("--width", type=int, default=6000, help="preview width")
    ap.add_argument("--center-yaw", type=float, default=0.0)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--parallax", choices=["off", "gps", "fit"], default="fit",
                    help="model the drone moving between shots (flat ground RelativeAltitude below): 'gps' uses the "
                         "GPS positions as-is, 'fit' refines them from the images with GPS as prior")
    ap.add_argument("--position-sigma", type=float, default=0.3, help="GPS prior sigma in metres for --parallax fit")
    ap.add_argument("--max-correction", type=float, default=6.0,
                    help="largest plausible rotation correction (deg) relative to the DJI metadata")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    opt = Options(parallax=args.parallax, position_sigma=args.position_sigma, max_correction=args.max_correction,
                  center_yaw=args.center_yaw, cache_dir=args.out.parent / "cache")
    t0 = time.time()
    res = align(args.folder, opt, log=print)
    ba, shots, cams = res.ba, res.shots, res.meta_cams
    print(f"BA done ({time.time() - t0:.1f}s): focal {cams[0].focal:.1f} -> {ba.cams[0].focal:.1f}px, "
          f"k1 {ba.cams[0].k1:+.4f}, k2 {ba.cams[0].k2:+.4f}, {ba.n_outliers_removed} outlier matches removed")
    before = np.array(list(ba.pair_rms_before.values()))
    after = np.array(list(ba.pair_rms_after.values()))
    print(f"Pair RMS: before median {np.median(before):.1f}px (max {before.max():.1f}), "
          f"after median {np.median(after):.2f}px (max {after.max():.2f})")
    groups = {"ground (< -50)": [], "low (-50..-10)": [], "horizon/sky (> -10)": []}
    for (i, j), rms in ba.pair_rms_after.items():
        p = (shots[i].gimbal_pitch + shots[j].gimbal_pitch) / 2
        groups["ground (< -50)" if p < -50 else "low (-50..-10)" if p < -10 else "horizon/sky (> -10)"].append(rms)
    print("Pair RMS after, by pitch: " + ", ".join(f"{k} {np.median(v):.2f}px" for k, v in groups.items() if v))
    if args.parallax == "fit" and ba.cams[0].position is not None:
        fitted = np.stack([c.position for c in ba.cams])
        print(f"Fitted drift up to {np.linalg.norm(fitted[:, [0, 2]], axis=1).max():.2f} m")

    print(f"\n{'#':>3} {'file':14} {'pitch':>6} {'dYaw':>6} {'dPitch':>6} {'Roll':>6}  {'FlightRoll':>10} {'links':>5}")
    rolls, flight_rolls, order = [], [], []
    for k, (c0, c1) in enumerate(zip(cams, ba.cams)):
        y0, p0, _ = c0.euler()
        y1, p1, r1 = c1.euler()
        nadir = abs(p0) > 80  # yaw/roll are not separable straight down
        if not nadir:
            rolls.append(r1)
            flight_rolls.append(c0.meta.flight_roll)
            order.append(k)
        roll_s = "  n/a " if nadir else f"{r1:+6.2f}"
        yaw_s = "  n/a " if nadir else f"{wrap180(y1 - y0):+6.2f}"
        note = "  (matches ignored)" if k in res.ignored else ""
        print(f"{k + 1:3d} {c0.meta.name:14} {p0:+6.1f} {yaw_s} {p1 - p0:+6.2f} {roll_s}"
              f"  {c0.meta.flight_roll:+10.1f} {res.linked[k]:5d}{note}")

    rolls = np.array(rolls)
    print(f"\nRoll correction: mean {rolls.mean():+.2f}, std {rolls.std():.2f}, max |roll| {np.abs(rolls).max():.2f} deg")
    print(f"Correlation roll vs FlightRoll: {np.corrcoef(rolls, flight_rolls)[0, 1]:+.2f}")
    print(f"Correlation roll vs shot order: {np.corrcoef(rolls, order)[0, 1]:+.2f}")

    save_cameras(args.out / "cameras.json", ba.cams)
    if not args.no_render:
        stats = render_all(ba.cams, args.out, "step2", args.width, args.center_yaw)
        print(f"Rendered -> {args.out} (uncovered {stats['uncovered_pct']:.1f} %)")
    print(f"Total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
