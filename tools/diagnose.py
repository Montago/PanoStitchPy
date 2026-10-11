"""Diagnose remaining misalignment after step 2.

Prints the drone's GPS drift during the panorama and the per-pair residuals, and writes
output/diag/residual_map.jpg: every feature match drawn on the panorama, coloured by its error after BA
(green < 2 px, yellow 2-5 px, red > 5 px at full resolution).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panostitch.camera import Camera  # noqa: E402
from panostitch.features import candidate_pairs, detect, match_pair  # noqa: E402
from panostitch.metadata import read_folder, read_xmp  # noqa: E402
from panostitch.projection import world_to_equirect  # noqa: E402


def gps_drift(shots):
    pos = []
    for s in shots:
        x = read_xmp(s.path)
        lat, lon = float(x["drone-dji:GpsLatitude"]), float(x["drone-dji:GpsLongitude"])
        pos.append((lon, lat, float(x["drone-dji:AbsoluteAltitude"])))
    pos = np.array(pos)
    lat0 = math.radians(pos[0, 1])
    enu = np.stack([(pos[:, 0] - pos[0, 0]) * 111320 * math.cos(lat0),
                    (pos[:, 1] - pos[0, 1]) * 110540,
                    pos[:, 2] - pos[0, 2]], axis=1)
    return enu - enu.mean(axis=0)


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "input/001_0174")
    cam_json = Path(sys.argv[2] if len(sys.argv) > 2 else "output/step2/cameras.json")
    out = Path("output/diag")
    out.mkdir(parents=True, exist_ok=True)

    shots = read_folder(folder)
    enu = gps_drift(shots)
    print("GPS position relative to mean (m):")
    for k, (s, p) in enumerate(zip(shots, enu)):
        print(f"  {k + 1:3d} {s.name}  E{p[0]:+6.2f}  N{p[1]:+6.2f}  Up{p[2]:+6.2f}")
    horiz = np.linalg.norm(enu[:, :2], axis=1)
    print(f"Max horizontal distance from mean: {horiz.max():.2f} m, vertical range {np.ptp(enu[:, 2]):.2f} m")
    alt = shots[0].relative_altitude
    print(f"At {alt:.0f} m altitude, 1 m of drift = {math.degrees(math.atan(1 / alt)):.2f} deg parallax straight down")

    meta_cams = [Camera.from_shot(s) for s in shots]
    solved = json.loads(cam_json.read_text())
    cams = []
    for c, d in zip(meta_cams, solved):
        cams.append(Camera(c.width, c.height, d["focal"], d["cx"], d["cy"], np.array(d["R"]), d["k1"], d["k2"], meta=c.meta))

    feats = [detect(c.meta.path, cache_dir=Path("output/cache")) for c in cams]
    rng = np.random.default_rng(0)
    matches = [m for i, j in candidate_pairs(meta_cams)
               if (m := match_pair(meta_cams, feats, i, j, rng=rng)) is not None]

    W, H = 3000, 1500
    pano = cv2.imread("output/step2/step2_feather_exif.jpg")
    pano = (cv2.resize(pano, (W, H), interpolation=cv2.INTER_AREA) * 0.45).astype(np.uint8)
    rows = []
    for m in matches:
        wi, wj = cams[m.i].rays_world(m.pts_i), cams[m.j].rays_world(m.pts_j)
        err_px = np.arccos(np.clip(np.sum(wi * wj, axis=1), -1, 1)) * cams[0].focal
        pitch = (shots[m.i].gimbal_pitch + shots[m.j].gimbal_pitch) / 2
        rows.append((np.median(err_px), np.percentile(err_px, 90), m.i, m.j, pitch, len(err_px)))
        for w, e in zip((wi + wj) / 2, err_px):
            u, v = world_to_equirect(w, W, H)
            color = (0, 200, 0) if e < 2 else (0, 220, 255) if e < 5 else (0, 0, 255)
            cv2.circle(pano, (int(u), int(v)), 2, color, -1)
    cv2.imwrite(str(out / "residual_map.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 90])

    rows.sort(reverse=True)
    print("\nWorst pairs after BA (full-res px):")
    for med, p90, i, j, pitch, n in rows[:12]:
        print(f"  {i + 1:2d}-{j + 1:2d}  median {med:5.2f}  p90 {p90:5.2f}  (avg pitch {pitch:+5.1f}, {n} matches)")
    by_row = {}
    for med, p90, i, j, pitch, n in rows:
        by_row.setdefault(round(pitch / 5) * 5, []).append(med)
    print("\nMedian pair error by average pitch of the pair:")
    for p in sorted(by_row):
        print(f"  pitch {p:+4.0f}: {np.median(by_row[p]):5.2f} px  ({len(by_row[p])} pairs)")
    print(f"\nWrote {out / 'residual_map.jpg'}")


if __name__ == "__main__":
    main()
