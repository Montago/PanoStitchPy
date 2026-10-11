"""Step 3: gain compensation, seam finding, multi-band blending (and zenith fill / cropping, depending on the
panorama type) of the cameras from step 2.

The camera's own JPEG rendering is kept (no re-exposure / tonemapping): gains only nudge the images so their
overlaps agree, and multi-band blending spreads the remaining large-scale brightness differences smoothly.

Outputs (in --out):
  step3_pano.jpg      final equirectangular panorama, with GPano 360 metadata
  step3_seams.jpg     panorama with the seam lines drawn in
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from panostitch.camera import load_cameras
from panostitch.layout import PANO_TYPES, detect_type, plan_canvas
from panostitch.metadata import read_folder
from panostitch.pipeline import Options, blend, finish
from panostitch.projection import Canvas
from panostitch.xmp import write_pano_jpeg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--cameras", type=Path, default=Path("output/step2/cameras.json"))
    ap.add_argument("--out", type=Path, default=Path("output/step3"))
    ap.add_argument("--width", type=int, default=8192,
                    help="width of a full 360 sphere (multiple of 2^levels); partial panoramas get the same detail")
    ap.add_argument("--type", choices=list(PANO_TYPES), default=None, help="panorama type (default: detect)")
    ap.add_argument("--no-crop", action="store_true", help="keep the black borders of partial panoramas")
    ap.add_argument("--seam-width", type=int, default=1536, help="resolution used for gains and seam finding")
    ap.add_argument("--seams", choices=["graphcut", "voronoi"], default="graphcut")
    ap.add_argument("--levels", type=int, default=9, help="multi-band levels")
    ap.add_argument("--no-gain", action="store_true")
    ap.add_argument("--anchor", choices=["ground", "all", "none"], default="ground",
                    help="which images keep their as-shot exposure on average (ground = pitch below -10 deg)")
    ap.add_argument("--no-zenith", action="store_true", help="leave the uncovered sky above the drone black")
    ap.add_argument("--center-yaw", type=float, default=None,
                    help="360 sphere: compass heading at the image centre (default: the start direction)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    shots = read_folder(args.folder)
    cams = load_cameras(args.cameras, shots)
    opt = Options(width=args.width, seam_width=args.seam_width, seams=args.seams, levels=args.levels,
                  gain=not args.no_gain, anchor=args.anchor, fill_zenith=not args.no_zenith,
                  center_yaw=args.center_yaw, crop=not args.no_crop)
    pano_type = args.type or detect_type([s.gimbal_yaw for s in shots])
    canvas = plan_canvas(cams, pano_type, 360.0 / args.width, args.width)
    if canvas.is_full_sphere and args.center_yaw is not None:
        canvas = Canvas.full_sphere(args.width, args.center_yaw)
    print(f"{PANO_TYPES[pano_type]}: canvas {canvas.width} x {canvas.height}, {canvas.lon_span:.0f} x "
          f"{canvas.lat_top - canvas.lat_bottom:.0f} deg{' (portrait)' if canvas.portrait else ''}")
    t0 = time.time()
    last = [-1]

    def progress(f: float, msg: str) -> None:
        if int(f * 10) != last[0]:
            last[0] = int(f * 10)
            print(f"  {f * 100:3.0f}%  {msg}  ({time.time() - t0:.0f}s)")

    res = blend(cams, opt, canvas, progress, log=print)
    pano, gpano = finish(res, opt)
    write_pano_jpeg(args.out / "step3_pano.jpg", pano, gpano)
    print(f"Saved {pano.shape[1]} x {pano.shape[0]}{' with GPano metadata' if gpano else ''}")

    labels = res.labels
    edges = np.zeros(labels.shape, bool)
    edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edges[1:, :] |= labels[1:, :] != labels[:-1, :]
    edges = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    seams_img = res.pano.copy()
    seams_img[edges & res.covered] = (0, 255, 255)
    cv2.imwrite(str(args.out / "step3_seams.jpg"), seams_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Done in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
