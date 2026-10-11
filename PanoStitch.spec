# PyInstaller build of the PanoStitch desktop app (one folder: PanoStitch.exe + _internal).
#
#   python -m PyInstaller PanoStitch.spec --noconfirm --distpath <output folder>
#
# Test the result without a window:  PanoStitch.exe --selftest <panorama folder> <out.jpg>  (log in <out.jpg>.log)
from pathlib import Path

ROOT = Path(SPECPATH)

a = Analysis(
    [str(ROOT / "app" / "main.py")],
    pathex=[str(ROOT)],
    datas=[(str(ROOT / "app" / "icons"), "app/icons")],
    hiddenimports=[],
    excludes=["tkinter", "matplotlib", "IPython", "PyInstaller"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PanoStitch",
    console=False,
    icon=str(ROOT / "app" / "icons" / "app.ico"),
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PanoStitch", upx=False)
