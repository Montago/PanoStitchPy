"""Step 1: place every image using only the DJI gimbal angles and blend into a 360x180 equirectangular.

No feature matching or optimisation: the output shows how accurate the drone metadata is on its own.

Outputs (in --out):
  step1_feather.jpg       linear feather blend of the JPEGs as shot (per-image auto exposure visible)
  step1_feather_exif.jpg  same, but each image normalised by its EXIF exposure (shutter * ISO), then tonemapped
  step1_seams.jpg         hard seams (each pixel from the image whose centre is closest) + image numbers;
                          misalignment shows up as broken lines across seams
  step1_coverage.png      number of images covering each pixel (black = hole)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from panostitch.camera import Camera
from panostitch.metadata import read_folder
from panostitch.render import render_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=Path("output/step1"))
    ap.add_argument("--width", type=int, default=6000, help="equirect width (height = width / 2)")
    ap.add_argument("--center-yaw", type=float, default=0.0, help="compass heading at image centre (0 = north)")
    args = ap.parse_args()

    cams = [Camera.from_shot(s) for s in read_folder(args.folder)]
    hfov, vfov = cams[0].fov_deg()
    print(f"{len(cams)} images, {cams[0].width}x{cams[0].height}, f={cams[0].focal:.1f}px, FOV {hfov:.1f} x {vfov:.1f} deg")

    stats = render_all(cams, args.out, "step1", args.width, args.center_yaw)
    print(f"Uncovered pixels: {stats['uncovered_pct']:.2f} %")
    if stats["zenith_hole_lat"] is not None:
        print(f"Zenith hole reaches down to latitude {stats['zenith_hole_lat']:+.1f} deg")
    print(f"EXIF exposure spread: {stats['exposure_stops']:.1f} stops")


if __name__ == "__main__":
    main()
