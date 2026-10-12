# Context for new sessions

Working notes so a new (Claude) session can pick up without re-deriving everything. The user-facing
documentation is [README.md](README.md); this file covers decisions, measurements, gotchas and the workflow.

## The project in short
- PanoStitch stitches the original single images from **DJI drone panorama modes** (developed with a DJI Air 3)
  into one panorama, with a PySide6 desktop app, a 360° viewer with tiny planet / inverted views, and a
  standalone Windows build. Released as v0.1.0 and v0.2.0 on GitHub (Montago/PanoStitchPy), MIT licensed.
- The repo folder is `C:\Git\PanoStichPy` (note the spelling *Stich*; the GitHub repo is *PanoStitchPy*).
- Conversation with the user is in **Danish**; all code, comments, UI text and `.md` files are **English**.

## Environment (Windows 11)
- Python 3.13 at `%LOCALAPPDATA%\Programs\Python\Python313\python.exe` (packages installed globally, no venv):
  opencv-python 5.0, numpy 2.5, scipy 1.18, pillow 12, PySide6-Essentials 6.12, pyinstaller 6.22.
- The `python` on PATH may be shadowed in an already-open shell; calling the full path above always works.
- GitHub CLI: `gh` installed with winget (user scope), logged in as Montago. In shells opened before the
  install, use the full path `%LOCALAPPDATA%\Microsoft\WinGet\Packages\GitHub.cli_*\bin\gh.exe`.
- Machine: 16 GB RAM. "High" quality (12288 px sphere) works but is slow (~10 min).

## Local data (git-ignored)
`input/` holds copies of real panoramas (the photos contain **GPS positions** – never commit them):

| Folder | Type | Notes / last measured result |
|---|---|---|
| `input/001_0174` | 360° sphere, 33 images | 8 columns × 4 rows (pitch −60/−25/+10/+45) + nadir; ~65 m altitude; median pair error 1.5 px |
| `input/003_0188` | 180°, 21 images | 7 columns × 3 rows (30° apart); ~268×98° after alignment; 1.8 px |
| `input/003_0197` | Wide angle, 9 images | 3×3; side columns have **gimbal roll ±9.9°**; 1.9 px |
| `input/003_0198` | Vertical, 3 images | one column, pitch −66/−31/+3, reaches past straight down; 1.8 px |
| `input/003_0209` | 360°, 33 **DNG** | mostly sea + sun glitter: only 44 of 143 pairs match, 2.9 px – a hard scene, not a DNG problem |
| `input/003_0218` | 360°, 33 **DNG** | park + coast, sun at ~238°/17° in image 2; 105 of 143 pairs, 1.3 px. **Main raw test set** |
| `input/003_0213` | 180°, 21 **DNG** | 106 of 110 pairs, 1.5 px |
| `input/003_0216` | Wide, 9 **DNG** | low over a beach (strong parallax, sky-only top row): 19 of 34 pairs, 4.0 px |
| `input/003_0211` | Vertical, 3 **DNG** | |
| `input/single_0220` | DNG + JPEG of one shot | used to fit the drone's tone curve and colour matrix (`raw.DJI_MATRIX`, `_TONE_*`) |
| `input/003_0194` | 360° sphere, 33 images | **sun test set**: sun at 161°/29° in images 2, 3, 32, 33 (clearest in 3); flare ghost in 33; 1.4 px. DJI's own result: `Z:\DJI Air 3\2026-10-04\DJI_20261004122504_0194_D.JPG` |

`output/` holds experiment results and caches; `dist/` the release build; `build/` PyInstaller's work dir.

## Key technical decisions (with the reasons)
- **DJI angle convention** (`camera.rotation_world_to_cam`): yaw → roll → pitch, positive roll = right side of
  the image down. Found by fitting against bundle adjustment on the wide set; the naive yaw → pitch → roll
  order gave up to 8° errors there. 360° spheres always have roll 0, so they never showed it.
- `GimbalRollDegree` in the sphere set is always 0, but the real per-image roll errors are up to ~3°
  (largest in the +45° sky row). Bundle adjustment fixes yaw, pitch and roll per image.
- **Parallax** (`--parallax fit`, default): the drone drifts up to ~1 m during a sphere. Camera positions are
  estimated with GPS as a 0.3 m prior and flat ground at `RelativeAltitude`. Using GPS positions *as-is* was
  worse on the ground; fitting them halved the ground error (3.6 → 1.7 px).
- **Plausibility check**: an image whose correction exceeds 6° (usually from matches on moving clouds in the
  sky row) gets its matches dropped and follows its neighbours + metadata. Image 25 of the sphere set
  triggered this (14–19° bogus roll).
- Features: SIFT on half-resolution images with CLAHE, only pairs that overlap per metadata, matches
  pre-filtered against the metadata pose (8°), RANSAC homography, max 200 matches per pair (400 was twice
  as slow with no gain). Focal length converges to ~2950 px (DJI's EXIF 35 mm equivalent suggests 2796).
- **Exposure**: per-channel gains solved in *linear* light; the default anchor keeps the ground images at the
  drone's own exposure. The user prefers the drone's look (contrast/colour of the JPEGs) over a re-tonemapped
  "physical" exposure, which looked dull. Gains are damped to `gain**0.6` (`gain_strength`): with full gains
  the image shot towards the sun in 001_0174 got +1.2 stops and the whole sky around the sun clipped (DJI's own
  result keeps it near the drone's exposure). Blending runs in *extended* sRGB (values > 255 allowed, no
  clipping); pixels near 255 in all channels are pushed (smoothly, to avoid JPEG-noise speckles) to neutral
  `max gain` so clipped clouds stay white; `compress_highlights` then rolls luminance above 0.7 (linear) off
  towards the 99.9th percentile and fades only the near-white part to neutral.
  Tried and rejected: per-channel roll-off (washes out the sunset colours), keeping the sun image's own
  exposure near the sun with an angular gain ramp (dark ring + white blob in 001_0174 – would need real local
  tone mapping), and the old soft clip (`tanh` above 0.8 per channel) that turned the 001_0174 sun area into a
  flat 230 grey plate once the multi-band coarse levels mixed in darker neighbours.
- **Seams**: OpenCV graph cut (`COST_COLOR_GRAD`) at 1536 px (sphere-equivalent width), then multi-band
  blending with 9 levels that wraps around at ±180° for full spheres.
- **Sun** (`sun.py`): before, the graph cut put image 33 (sun behind a cloud) over the sun, so the sun vanished
  into a glowing blob. Now the sun direction is predicted from GPS + `xmp:CreateDate` (needs a time zone),
  refined to the most saturated spot within ±5° elevation / ±40° azimuth – **DJI's heading is off true north by
  ~10° (003_0194) to ~24° (001_0174)**, elevation is reliable. The image with the largest sun-to-edge margin
  keeps everything within 20° of the sun that it covers (minus 1.5° along its edges) – in the graph cut **and**
  at every multi-band level (`MultiBandBlender.reserve`), otherwise the coarse levels (up to ~22°) mix the
  neighbours into the broad glow and only a small, hard-edged disc survives (the user spotted this). The glare
  region is left out of the gain statistics. Flare ghosts: a band (±5 % of the diagonal) from the sun through the image centre
  to 1.3× the mirror point is taken away from each image that has the sun in or near its frame, wherever another
  image covers it – this also swaps in neighbours without the veiling glare on the ground. `--no-sun` in step 3.
- **Raw / DNG** (`raw.py`, `pipeline._Frames`): the drone's stitched JPGs for the DNG sets 0209–0219 are in
  `Z:\DJI Air 3\2026-10-12` (same numbers). Findings:
  - LibRaw does not apply DJI's OpcodeList3: **GainMap** (vignetting, corners ×5 ≈ 2.3 stops) and
    **WarpRectilinear** (distortion + lateral CA per plane). `raw.decode` applies both; afterwards the DNG
    matches the drone's JPEG within 2–4 px over 8064 px, and the residual vignetting measured from overlaps is
    under 5 %.
  - **ISO above 100 is not in the raw data** – the sensor stays at base ISO and the gain is stored as
    `BaselineExposure` (ISO 160 → 0.69, ISO 190 → 0.95). Normalising by shutter × ISO left neighbours up to 2×
    apart (visible steps on the ground); `ShotInfo.sensor_exposure` divides by 2^BaselineExposure and brings
    them within ±15 %.
  - The drone exposes the raw frames **dark** (highlight protection); the intended brightness of each frame is
    2^BaselineExposure. Default: the ground frames keep their intended brightness (like anchor="ground"); this
    matched DJI's panoramas better on sea scenes (0209/0213) than a fixed key. `raw_key` > 0 instead picks the
    overall exposure so the log-average display luminance is that value (`compress_highlights(key=...)`; DJI's
    own panoramas: 0.07 for sea scenes … 0.21 for land).
  - Look: `JPEG = tone(DJI_MATRIX @ linear)`, fitted on single shot 0220 (rms 7 levels); our RAW development of
    that shot is visually very close to the drone's JPEG, only a bit noisier (no noise reduction yet).
  - `gain_strength` for raw: frames are scaled between the drone's intended rendering (2^BaselineExposure,
    strength 0) and fully matched exposure (1); the remaining solved gains are applied undamped (they are
    measurement errors, not look) – so `gain_strength` hardly matters for raw. Our sky stays brighter than
    DJI's (they seem to tone-map locally).
  - Only the sun's disc clips in raw, so `find_sun` uses half the clip level as "bright" for raw frames.
  - SIFT runs on a brightness-normalised half-size decode (`raw.load_normalised`): the dark 8-bit renderings
    lost texture.
- **Horizon / low flight**: 0216 was shot at **0.7 m** over a beach; the sand 1–3 m away has parallax (lens
  offset from the gimbal pivot) and BA tilted the middle images to fit it – a 4–5 px step in the sea horizon
  (DJI's own 0216 has a broken horizon too). Two fixes: (1) below 5 m altitude, matches are weighted by
  distance to flat ground, `w = min(1, d / 30 m)^2` (`PairMatches.weight`); dropping near matches entirely
  left image 5 under-constrained (0.8° tilt). (2) `horizon.py`: in images with |pitch| ≤ 35 where the metadata
  puts the horizon in view, the strongest edge in a ±2.5° band is fitted with a line (RANSAC, ≥ 70 % of
  columns, ≤ 1 px rms, ≥ 60 % of the width); its 24 points are kept at the dip elevation in BA
  (`horizon_weight` 5). Result: horizon within 1 px over the whole width. Detected in 0216 (3 images) and
  0188 (2), not in land scenes. The reported median pair error rises (6.7 px in 0216) because the down-weighted
  sand matches no longer have to fit – that is expected.
- **Zenith fill** (`zenith.py`): only the uncovered component touching the top row, and only if it stays above
  45° latitude; larger holes (missing images) stay black, as the user wants.
- **Output canvas** (`layout.py`, `projection.Canvas`): sphere = full equirect centred on the start direction
  (first image), heading stored in `GPano:PoseHeadingDegrees`; 180° = equirect cropped to the largest
  fully covered rectangle, with GPano cropped-area fields; **wide = level rectilinear** (`kind="rectilinear"`,
  pitch 0 so verticals stay vertical), window of at most 100° × 80° centred on the rows/columns that are at
  least half covered, no GPano (shown in the app's "Flat image" tab); vertical = transverse frame turned to
  portrait, no GPano. Measured DJI's projections by matching their panoramas against our aligned frames
  (fit of world ray -> DJI pixel): wide 0216 = rectilinear (9 px vs 60–70 px for the others), ~98° × 78°;
  180° 0213 = equirect/cylindrical/Mercator equally good (5 px), so we keep equirect. Type detection uses the shooting pattern (number of yaw columns and their spread), not fixed
  angles, because the drone starts from wherever the pilot points.
- **Views** (`views.py`): normal = rectilinear, tiny planet / inverted = stereographic looking down / up;
  view-plane coordinates make the window and the export frame consistent. Export renders from the
  full-resolution panorama in row chunks, 2× supersampled up to 4096 px (~3–4 s).

## Workflow
- Quick test without the GUI: `python app/main.py --selftest <folder> <out.jpg>` (low resolution, log in
  `<out.jpg>.log`) – also works on the built exe.
- Step scripts for experiments: `step1_metadata_only.py`, `step2_align.py` (prints per-image corrections and
  pair errors), `step3_blend.py`; `tools/diagnose.py` draws a residual map; `tools/compare_crops.py` makes
  side-by-side crops; `tools/inspect_set.py` prints a folder's shooting pattern.
- GUI screenshots without showing a window: real Windows platform plus `Qt.WA_DontShowOnScreen`, then
  `widget.grab()`. (`QT_QPA_PLATFORM=offscreen` renders text as boxes because it has no fonts.)
- **Release**: bump `panostitch/__init__.py` `__version__`, run
  `python tools/build_release.py --selftest input/003_0198`, commit, `git tag -a vX.Y.Z`, push with
  `--follow-tags`, then `gh release create vX.Y.Z dist/PanoStitch-X.Y.Z-win64.zip
  dist/PanoStitch-X.Y.Z-portable.zip --notes-file ...`. Release notes are English.
- Commit messages end with a `Co-Authored-By: Claude …` line. The user asks explicitly for commits/pushes.

## Gotchas
- `cv2.imread/imwrite` fail on non-ASCII Windows paths (æøå) – always use `panostitch.imageio`.
- In PowerShell, `git push` and PyInstaller write progress to stderr, so the tool reports exit code 255/1
  although they succeeded – check the output text. `robocopy` returns 1 for "files copied".
- A safety hook blocks `Remove-Item` when the path comes from a variable; use `-LiteralPath "<full path>"`.
- The Python `.gitignore` template ignores `*.spec`; `!PanoStitch.spec` re-includes the build recipe.
- PyInstaller warns `Hidden import "scipy.special._cdflib" not found` – harmless.
- The PySide6 wheel does not ship the LGPL text; `packaging/licenses/` has the FSF LGPL-3.0 and GPL-3.0 texts
  that the build copies into `THIRD_PARTY_LICENSES/Qt`. Keep the onedir build (Qt DLLs replaceable).
- Equirect rendering of *partial* panoramas: never treat "covered in every column" as a 360° band near the
  poles – corners of downward images span many longitudes there (this bug once made the wide set 360° wide).

## Ideas not done yet
- Raw: light noise reduction (the DNG development is noisier than the drone's JPEGs in the sky), and maybe a
  scene-adaptive shadow lift like the drone's JPEG engine (DJI's ground is a bit brighter than ours).
- A "free" panorama type for hand-placed shots (DJI Air 3S feature); can't be tested without such a drone.
- Ignore sky matches for geometry in strong wind; finer seams (`--seam-width 3072`) to avoid cutting small
  moving objects.
- Full native resolution (~18,500 px sphere) needs tiled blending to fit in memory.
- Choice of cylindrical projection for 180° panoramas (`Canvas` supports `kind="cylindrical"`).
