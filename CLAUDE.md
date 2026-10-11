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
- **Exposure**: per-channel gains solved in *linear* light, applied with a soft highlight roll-off; the
  default anchor keeps the ground images at the drone's own exposure. The user prefers the drone's look
  (contrast/colour of the JPEGs) over a re-tonemapped "physical" exposure, which looked dull.
- **Seams**: OpenCV graph cut (`COST_COLOR_GRAD`) at 1536 px (sphere-equivalent width), then multi-band
  blending with 9 levels that wraps around at ±180° for full spheres.
- **Zenith fill** (`zenith.py`): only the uncovered component touching the top row, and only if it stays above
  45° latitude; larger holes (missing images) stay black, as the user wants.
- **Output canvas** (`layout.py`, `projection.Canvas`): sphere = full equirect centred on the start direction
  (first image), heading stored in `GPano:PoseHeadingDegrees`; 180°/wide = equirect cropped to the largest
  fully covered rectangle, with GPano cropped-area fields; vertical = transverse frame turned to portrait,
  no GPano. Type detection uses the shooting pattern (number of yaw columns and their spread), not fixed
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
- A "free" panorama type for hand-placed shots (DJI Air 3S feature); can't be tested without such a drone.
- Ignore sky matches for geometry in strong wind; finer seams (`--seam-width 3072`) to avoid cutting small
  moving objects.
- Full native resolution (~18,500 px sphere) needs tiled blending to fit in memory.
- Choice of cylindrical projection for partial panoramas (`Canvas` already supports `kind="cylindrical"`).
