"""Print the DJI shooting pattern of a panorama folder: angles, exposure, timing, lens.

usage: python tools/inspect_set.py <folder>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panostitch.camera import Camera  # noqa: E402
from panostitch.metadata import read_folder, read_xmp  # noqa: E402

folder = Path(sys.argv[1])
shots = read_folder(folder)
cam = Camera.from_shot(shots[0])
hfov, vfov = cam.fov_deg()
print(f"{len(shots)} images, {shots[0].model}, {shots[0].width}x{shots[0].height}, "
      f"{shots[0].focal_mm} mm ({shots[0].focal_35mm:.0f} mm eq.), FOV ~{hfov:.0f} x {vfov:.0f} deg")
print(f"{'#':>3} {'file':16} {'time':8} {'gYaw':>7} {'gPitch':>7} {'gRoll':>6} {'fYaw':>7} {'exp':>7} {'ISO':>5} {'relAlt':>7}")
for k, s in enumerate(shots):
    t = read_xmp(s.path).get("xmp:CreateDate", "")[11:19]
    print(f"{k + 1:3d} {s.name:16} {t:8} {s.gimbal_yaw:+7.1f} {s.gimbal_pitch:+7.1f} {s.gimbal_roll:+6.1f} "
          f"{s.flight_yaw:+7.1f} 1/{1 / max(s.exposure_time, 1e-9):<5.0f} {s.iso:5.0f} {s.relative_altitude:7.1f}")
yaws = sorted({round(s.gimbal_yaw) for s in shots})
pitches = sorted({round(s.gimbal_pitch) for s in shots})
print("distinct yaws:", yaws)
print("distinct pitches:", pitches)
