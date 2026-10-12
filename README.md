# PanoStitch

Stitches the original single images from DJI drone panorama modes into one panorama, with a simple Windows
desktop app and a built-in 360° viewer. Developed and tested with the DJI Air 3.

| Type | Images | Result |
|---|---|---|
| 360° sphere | 33 | Full 360×180° equirectangular image with GPano metadata; the sky above the drone is filled in |
| 180° panorama | 21 | Approx. 270×100°, cropped equirectangular with GPano "cropped area" metadata |
| Wide angle | 9 | Approx. 130×110°, cropped equirectangular with GPano "cropped area" metadata |
| Vertical | 3 | Tall image in a transverse projection (can reach straight down below the drone) |

The type is detected automatically from the angles of the images (or chosen in the app). The drone takes the
first image in the direction and tilt the pilot points at, and the others around it.

360°, 180° and wide-angle results carry GPano metadata, so Facebook, Google Photos, Kuula, VR headsets etc.
show them as (partial) 360° panoramas.

## Download

The [GitHub releases](https://github.com/Montago/PanoStitchPy/releases) have:
- `PanoStitch-<version>-win64.zip` – standalone Windows build, no Python needed: unzip and run `PanoStitch.exe`.
- `PanoStitch-<version>-portable.zip` – the program for an installed Python (small download): unzip, run
  `Installer.bat` once, then `PanoStitch.bat`.

## Running from source

Requires Python 3.11+ (developed with 3.13).

```
python -m pip install -r requirements.txt
python app/main.py
```

On Windows you can also double-click `Installer.bat` once and then `PanoStitch.bat`.

### Using the app
1. Choose a panorama folder – or the whole `PANORAMA` folder from the drone's SD card to find all panoramas.
   You can also drag a folder onto the window. Use the drone's original single images
   (e.g. `PANORAMA/001_0174/PANO_0001.JPG …`). Panoramas shot in **DNG (raw)** work too and keep far more
   detail in bright areas such as the sun and the sky around it.
2. Choose type, quality and exposure. The defaults are good.
3. Press **Stitch**. The result is saved as `<date>_<folder>_<type>.jpg` (by default in `Pictures\PanoStitch`).
4. Look at the result in the built-in 360° viewer (drag to look around, scroll to zoom).
5. Switch the view to **Tiny planet** (the ground curved into a small planet) or **Inverted** (the sky in the
   middle, the ground around it). Drag sideways to turn, up/down to tilt towards a normal view, scroll to make
   the planet smaller or bigger. **Save view…** writes what is inside the frame (window shape, 1:1, 4:5, 16:9
   or 9:16) at up to 8192 px, rendered from the full-resolution panorama.

If images are missing (e.g. the drone stopped early), those areas are left black.

### Command line
```
python step2_align.py <folder> --out output/step2
python step3_blend.py <folder> --cameras output/step2/cameras.json --out output/step3
```
`step1_metadata_only.py` places the images from the DJI metadata alone (useful to see how accurate it is).
Run any script with `--help` for its options.

### Building the exe
```
python tools/build_release.py
```
Builds `dist/PanoStitch/PanoStitch.exe` with PyInstaller (from `PanoStitch.spec`), adds the third-party
licenses and a short README, and zips it as `dist/PanoStitch-<version>-win64.zip`. It also writes
`dist/PanoStitch-<version>-portable.zip` with the sources, batch files and README for an installed Python.
`--selftest <panorama folder>` stitches a panorama with the built exe as a check.

## How it works

1. **Metadata** – the gimbal yaw/pitch/roll, GPS position and EXIF focal length of every image give a first
   estimate of where it points (typically 1–3° off).
2. **Alignment** – SIFT features are matched between overlapping images only. Bundle adjustment then refines
   the rotation of every image plus a shared focal length and lens distortion, with the metadata as a weak
   prior. With `--parallax fit` (default) it also estimates how far the drone drifted between shots (up to
   ~1 m in wind), assuming flat ground `RelativeAltitude` below; this halves the error on the ground close to
   the drone. Images without usable matches (plain sky) follow their neighbours, and an image that gets an
   implausibly large correction (usually from matches on moving clouds) has its matches ignored.
   A **sea horizon** in view (a long, perfectly straight edge near where the metadata puts the horizon) is
   used to level the images. When the drone flies very low (≤ 5 m), matches on the nearby ground count less,
   because the ground shows parallax there and would tilt the horizon.
3. **Blending** – exposure is balanced in linear light (keeping the drone's own look on the ground by
   default), seams are found with a graph cut so they run around cars and buildings, and multi-band blending
   hides the transitions. For a 360° sphere the blending wraps around at ±180°. Only 60 % of the exposure
   differences are evened out (`--gain-strength`); the multi-band blending smooths the rest, so the images the
   drone shot darker towards the sun are not pushed into white. Nothing is clipped while blending: highlights
   brighter than white are rolled off once at the end, keeping the colour of a bright sunset sky.
   **The sun** is found from the GPS position and capture time (refined to the bright spot in the images) and
   kept from the one image that shows it farthest from its edges, up to 20° around it and at every blending
   scale, so its graded glow is not cut into mismatched pieces. Lens-flare ghosts, which land on the line from the sun through the image centre and
   so in a different place in every image, are avoided by the seams wherever another image covers the spot.
   `step3_blend.py --no-sun` turns this off.
   **Raw (DNG) panoramas** are decoded with LibRaw and get the lens corrections DJI stores in each DNG
   (vignetting, distortion and colour fringing), one white balance for all frames, and an exposure match from
   the shutter speed and `BaselineExposure` (the Air 3 records ISO above 100 there instead of in the data).
   The ground keeps the brightness the drone intended for it, and the panorama is rendered with a tone curve
   and colour matrix measured from a DNG + JPEG pair of the drone, so it looks like the drone's own JPEGs –
   without their clipped highlights. `step3_blend.py --raw-key 0.2` exposes for a fixed overall brightness
   instead.
4. **Finishing** – for a 360° sphere the normal gap above the drone is filled with smooth sky (only if it is
   the normal gap; larger holes from missing images stay black). Partial panoramas are cropped to the largest
   fully covered rectangle. 180° panoramas are equirectangular (like the drone's); **wide-angle** panoramas use
   a level rectilinear projection like the drone does – straight lines stay straight – limited to about
   100° × 80° around what the images cover.

### Project layout
```
panostitch/
  metadata.py      DJI XMP (gimbal angles, GPS) and EXIF (focal length, exposure)
  raw.py           DNG decoding (rawpy/LibRaw), DJI's lens corrections, the drone's tone curve and colours
  camera.py        pinhole camera, DJI angle convention, ground-plane parallax model
  features.py      SIFT and matching of overlapping pairs (metadata filter + RANSAC)
  bundle.py        bundle adjustment
  layout.py        panorama types, choosing the output canvas, cropping
  horizon.py       finding sea horizons to level the panorama
  projection.py    output canvases (equirectangular, transverse) and remap tables
  blend.py         gain compensation, graph-cut seams, wrap-around multi-band blending
  sun.py           sun position, keeping the sun from one image, steering seams around lens flare
  zenith.py        filling the sky above the drone
  xmp.py           GPano metadata
  views.py         normal, tiny planet and inverted (stereographic) views of a 360° panorama, and their export
  pipeline.py      the whole pipeline as library calls (progress + cancel), used by the app and the scripts
  imageio.py       image I/O that works with non-ASCII Windows paths
  render.py        simple previews for the step scripts
app/               the desktop app (PySide6): window, 360° viewer, background worker
tools/             diagnostics, release build, icon generator
```

### Conventions
- World frame: x = east, y = up, z = north. Yaw is the compass heading (0 = north, +90 = east), pitch is
  positive upwards. Camera frame (OpenCV): x = right, y = down, z = forward.
- DJI gimbal angles are applied as yaw → roll → pitch; positive roll tips the right side of the image down
  (verified on the wide-angle panorama, whose side columns have a roll of ±9.9°).
- A 360° sphere is centred on the start direction (the first image) unless `--center-yaw` is given; the
  compass heading is stored in `GPano:PoseHeadingDegrees`.
- Partial panoramas are centred on the covered area. Vertical panoramas use a transverse frame, so the poles
  of the projection lie to the sides instead of straight below the drone.

## License

MIT, see [LICENSE](LICENSE). The Windows build bundles third-party components under their own licenses
(Qt/PySide6 under LGPL-3.0, OpenCV under Apache-2.0, NumPy/SciPy under BSD, Pillow under MIT-CMU, Python under
the PSF license); their license texts are included in the release zip.
