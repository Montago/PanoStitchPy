"""Entry point for the PanoStitch desktop app.

    PanoStitch [folder]                      open the window (optionally with a folder preloaded)
    PanoStitch --selftest <folder> <out.jpg> stitch at low resolution without a window and exit (0 = ok);
                                             the log goes to <out.jpg>.log. Used to test the packaged .exe.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication

# allow `python app/main.py` as well as `python -m app.main`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import ICON_DIR  # noqa: E402
from app.main_window import MainWindow  # noqa: E402

ACCENT = QColor("#2f7fe0")


def apply_theme(app: QApplication) -> None:
    """Fusion follows the Windows light/dark mode, but also its accent colour, which can be a pale yellow or
    orange that makes ticked checkboxes hard to see. Use a fixed blue accent with white on top instead."""
    pal = app.palette()
    for group in (QPalette.Active, QPalette.Inactive):
        pal.setColor(group, QPalette.Highlight, ACCENT)
        pal.setColor(group, QPalette.Accent, ACCENT)
        pal.setColor(group, QPalette.HighlightedText, QColor(Qt.white))
    pal.setColor(QPalette.Disabled, QPalette.Accent, ACCENT.darker(160))
    pal.setColor(QPalette.Disabled, QPalette.Highlight, ACCENT.darker(160))
    app.setPalette(pal)

    # Draw checkboxes ourselves (also the ones in the panorama list): a clearly visible frame when empty, solid
    # blue with a white tick when checked - readable in both light and dark mode.
    check = (ICON_DIR / "check.png").as_posix()
    accent = ACCENT.name()
    app.setStyleSheet(f"""
        QCheckBox::indicator, QListView::indicator {{
            width: 15px; height: 15px; border: 1px solid #8c8c96; border-radius: 3px; background: palette(base);
        }}
        QCheckBox::indicator:hover, QListView::indicator:hover {{ border-color: {accent}; }}
        QCheckBox::indicator:checked, QListView::indicator:checked {{
            background: {accent}; border-color: {accent}; image: url("{check}");
        }}
        QCheckBox::indicator:disabled, QListView::indicator:disabled {{ border-color: #5a5a62; }}
        QCheckBox::indicator:checked:disabled, QListView::indicator:checked:disabled {{
            background: #4a5a70; border-color: #4a5a70;
        }}
    """)


def selftest(folder: Path, out: Path) -> int:
    """Run the whole pipeline at low resolution without any window (no console in the .exe, so log to a file)."""
    import time
    import traceback

    from panostitch.pipeline import Options, stitch

    log_path = out.with_suffix(out.suffix + ".log")
    with open(log_path, "w", encoding="utf-8") as log:
        def write(msg: str) -> None:
            log.write(msg + "\n")
            log.flush()

        t0 = time.time()
        try:
            stitch(folder, out, Options(width=1024, levels=5, seam_width=512), log=write)
            write(f"SELFTEST OK {time.time() - t0:.0f}s")
            return 0
        except Exception:
            write(traceback.format_exc())
            write("SELFTEST FAILED")
            return 1


def main() -> int:
    if len(sys.argv) >= 4 and sys.argv[1] == "--selftest":
        return selftest(Path(sys.argv[2]), Path(sys.argv[3]))

    if sys.platform == "win32":
        # own taskbar identity, so Windows shows the PanoStitch icon instead of Python's
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PanoStichPy.PanoStitch")
    app = QApplication(sys.argv)
    app.setApplicationName("PanoStitch")
    app.setOrganizationName("PanoStichPy")
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(str(ICON_DIR / "app.png")))
    apply_theme(app)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
        win.load_folder(Path(sys.argv[1]))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
