from pathlib import Path

# Works both from source and inside a PyInstaller bundle (where the package lives in _internal/app).
ICON_DIR = Path(__file__).resolve().parent / "icons"
