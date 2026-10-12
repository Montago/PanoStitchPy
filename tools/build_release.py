"""Build the Windows release: PanoStitch.exe (PyInstaller), third-party licenses, README, zips.

usage: python tools/build_release.py [--skip-build] [--selftest <panorama folder>]

Output in dist/:
  PanoStitch/                        the standalone program folder
  PanoStitch-<version>-win64.zip     that folder zipped (no Python needed)
  PanoStitch-<version>-portable.zip  the source program for an installed Python (Installer.bat + PanoStitch.bat)
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from panostitch import __version__  # noqa: E402

DIST = ROOT / "dist"
APP_DIR = DIST / "PanoStitch"

# distribution name, what it is, license, where the source is
COMPONENTS = [
    ("PySide6-Essentials", "Qt for Python (PySide6) and the Qt 6 libraries", "LGPL-3.0",
     "https://code.qt.io/cgit/pyside/pyside-setup.git/ - Qt: https://download.qt.io/official_releases/qt/"),
    ("shiboken6", "Shiboken6 (Qt for Python runtime)", "LGPL-3.0",
     "https://code.qt.io/cgit/pyside/pyside-setup.git/"),
    ("opencv-python", "OpenCV, including the bundled FFmpeg build", "Apache-2.0 (third-party parts: see LICENSE-3RD-PARTY.txt)",
     "https://github.com/opencv/opencv-python"),
    ("numpy", "NumPy", "BSD-3-Clause and others", "https://github.com/numpy/numpy"),
    ("scipy", "SciPy, including the bundled OpenBLAS and GCC runtime", "BSD-3-Clause and others",
     "https://github.com/scipy/scipy"),
    ("pillow", "Pillow", "MIT-CMU", "https://github.com/python-pillow/Pillow"),
    ("rawpy", "rawpy and the bundled LibRaw (raw/DNG decoding)", "rawpy: MIT; LibRaw: LGPL-2.1 or CDDL-1.0",
     "https://github.com/letmaik/rawpy - LibRaw: https://github.com/LibRaw/LibRaw"),
]

LGPL_NOTE = """\
Qt and Qt for Python (PySide6, shiboken6) are used under the GNU Lesser General Public License v3
(THIRD_PARTY_LICENSES/Qt/LGPL-3.0.txt, which builds on THIRD_PARTY_LICENSES/Qt/GPL-3.0.txt). They are not
modified. The Qt libraries are separate files in _internal/PySide6 and _internal/shiboken6 and can be replaced
with compatible versions of your own. The corresponding source code is available from the links above.
LibRaw (used through rawpy) is used under the GNU Lesser General Public License v2.1 (see
THIRD_PARTY_LICENSES/rawpy/LICENSE.LibRaw); it is the separate, replaceable file _internal/rawpy/raw_r.dll.
"""


def license_files(dist: str) -> list[tuple[str, Path]]:
    """(flat file name, location) of every license/notice file installed by a distribution."""
    keys = ("LICENSE", "COPYING", "NOTICE")
    return [(str(f).replace("/", "__"), Path(f.locate())) for f in (md.files(dist) or [])
            if any(k in f.name.upper() for k in keys)]


def collect_licenses(target: Path) -> None:
    out = target / "THIRD_PARTY_LICENSES"
    if out.exists():
        shutil.rmtree(out)
    lines = [f"PanoStitch {__version__} - third-party components", "=" * 50, ""]

    py_license = Path(sys.base_prefix) / "LICENSE.txt"
    (out / "Python").mkdir(parents=True)
    shutil.copy2(py_license, out / "Python" / "LICENSE.txt")
    lines += [f"Python {sys.version.split()[0]}", "  License: Python Software Foundation License (PSF-2.0)",
              "  Source:  https://www.python.org/downloads/source/", "  Files:   THIRD_PARTY_LICENSES/Python", ""]

    for dist, what, lic, source in COMPONENTS:
        folder = out / dist
        folder.mkdir(parents=True)
        for name, path in license_files(dist):
            shutil.copy2(path, folder / name)
        lines += [f"{what} ({dist} {md.version(dist)})", f"  License: {lic}", f"  Source:  {source}",
                  f"  Files:   THIRD_PARTY_LICENSES/{dist}", ""]

    (out / "Qt").mkdir()
    for name in ("LGPL-3.0.txt", "GPL-3.0.txt"):
        shutil.copy2(ROOT / "packaging" / "licenses" / name, out / "Qt" / name)
    lines += [LGPL_NOTE]

    lines += ["PyInstaller bootloader (used to build PanoStitch.exe)",
              "  License: GPL-2.0-or-later with the PyInstaller exception, which allows distributing programs",
              "           built with it under any license.", "  Source:  https://github.com/pyinstaller/pyinstaller",
              ""]
    (target / "THIRD_PARTY_NOTICES.txt").write_text("\n".join(lines), encoding="utf-8")


PORTABLE_FILES = ["PanoStitch.bat", "Installer.bat", "requirements.txt"]
PORTABLE_DIRS = ["app", "panostitch"]


def build_portable_zip() -> Path:
    """Zip of the program sources for running on an installed Python (no build step involved)."""
    zip_path = DIST / f"PanoStitch-{__version__}-portable.zip"
    top = Path("PanoStitch-portable")
    readme = (ROOT / "packaging" / "README-portable.txt").read_text(encoding="utf-8").replace("{version}", __version__)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for d in PORTABLE_DIRS:
            for f in sorted((ROOT / d).rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts:
                    zf.write(f, top / f.relative_to(ROOT))
        for name in PORTABLE_FILES:
            text = (ROOT / name).read_text(encoding="utf-8")
            if name.endswith(".bat"):  # cmd.exe needs Windows line endings
                text = text.replace("\r\n", "\n").replace("\n", "\r\n")
            zf.writestr(str(top / name), text)
        zf.writestr(str(top / "README.txt"), readme.replace("\r\n", "\n").replace("\n", "\r\n"))
        zf.write(ROOT / "LICENSE", top / "LICENSE.txt")
    return zip_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-build", action="store_true", help="reuse dist/PanoStitch, only redo licenses and zips")
    ap.add_argument("--selftest", type=Path, help="stitch this panorama folder with the built exe as a test")
    args = ap.parse_args()

    if not args.skip_build:
        subprocess.run([sys.executable, "-m", "PyInstaller", str(ROOT / "PanoStitch.spec"), "--noconfirm",
                        "--distpath", str(DIST), "--workpath", str(ROOT / "build"), "--log-level", "WARN"],
                       check=True, cwd=ROOT)

    collect_licenses(APP_DIR)
    readme = (ROOT / "packaging" / "README.txt").read_text(encoding="utf-8").replace("{version}", __version__)
    (APP_DIR / "README.txt").write_text(readme, encoding="utf-8")
    shutil.copy2(ROOT / "LICENSE", APP_DIR / "LICENSE.txt")

    if args.selftest:
        out = DIST / "selftest.jpg"
        result = subprocess.run([str(APP_DIR / "PanoStitch.exe"), "--selftest", str(args.selftest), str(out)])
        log = out.with_suffix(".jpg.log").read_text(encoding="utf-8")
        print(log)
        if result.returncode != 0:
            sys.exit("Self-test FAILED")

    zip_path = DIST / f"PanoStitch-{__version__}-win64.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in sorted(APP_DIR.rglob("*")):
            if f.is_file():
                zf.write(f, Path("PanoStitch") / f.relative_to(APP_DIR))
    print(f"{zip_path} ({zip_path.stat().st_size / 2**20:.0f} MB)")
    portable = build_portable_zip()
    print(f"{portable} ({portable.stat().st_size / 2**10:.0f} KB)")


if __name__ == "__main__":
    main()
