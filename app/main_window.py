"""Main window: choose panorama folders, pick a few simple settings, stitch, and look at the result in 360."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSettings, QStandardPaths, Qt, QThread
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QSizePolicy, QSplitter, QTabWidget, QToolButton,
                               QVBoxLayout, QWidget)

from panostitch import __version__
from panostitch.imageio import imread
from panostitch.metadata import read_xmp
from panostitch.pipeline import Options, PanoramaSet, find_panorama_sets
from panostitch.xmp import read_gpano

from .viewer import PanoViewer
from .worker import Job, StitchWorker

QUALITY = [
    ("Fast  –  4096 × 2048", 4096),
    ("Standard  –  8192 × 4096", 8192),
    ("High  –  12288 × 6144  (slow, uses a lot of memory)", 12288),
]
LIGHT = [
    ("Keep the drone's exposure on the ground", "ground"),
    ("Balanced between sky and ground", "all"),
]
TYPES = [
    ("Automatic (detected from the images)", "auto"),
    ("360° sphere  –  33 images", "sphere"),
    ("180° panorama  –  21 images", "180"),
    ("Wide angle  –  9 images", "wide"),
    ("Vertical  –  3 images", "vertical"),
]
VIEW_MAX_WIDTH = 8192  # larger panoramas are shrunk for the on-screen viewer only

ROLE_SET = Qt.UserRole
ROLE_OUTPUT = Qt.UserRole + 1


class FlatView(QLabel):
    """The whole equirectangular image, scaled to fit."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 100)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background: #18181c; color: #a0a0aa;")
        self._pixmap: QPixmap | None = None
        self.setText("No panorama yet")

    def set_image(self, bgr: np.ndarray | None) -> None:
        if bgr is None:
            self._pixmap = None
            self.setText("No panorama yet")
            return
        h, w = bgr.shape[:2]
        f = min(1.0, 4096 / max(h, w))
        small = bgr if f == 1.0 else cv2.resize(bgr, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]
        self._pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, _event) -> None:
        self._rescale()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PanoStitch {__version__} – DJI drone panoramas")
        self.resize(1400, 820)
        self.setAcceptDrops(True)
        self.settings = QSettings("PanoStichPy", "PanoStitch")
        self._thread: QThread | None = None
        self._worker: StitchWorker | None = None
        self._job_items: list[QListWidgetItem] = []
        self._last_output: Path | None = None

        splitter = QSplitter()
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_preview())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 980])
        self.setCentralWidget(splitter)

        last = self.settings.value("input_folder", "")
        if last and Path(last).is_dir():
            self.load_folder(Path(last))

    # ------------------------------------------------------------------ layout
    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(360)
        layout = QVBoxLayout(panel)

        # 1. images
        box1 = QGroupBox("1. Choose images")
        v1 = QVBoxLayout(box1)
        hint = QLabel("Choose the folder with a panorama from the drone – or the whole <b>PANORAMA</b> folder on "
                      "the SD card to find all panoramas. You can also drag a folder onto the window.")
        hint.setWordWrap(True)
        v1.addWidget(hint)
        row = QHBoxLayout()
        self.btn_folder = QPushButton("Choose folder…")
        self.btn_folder.clicked.connect(self.choose_folder)
        row.addWidget(self.btn_folder)
        self.lbl_folder = QLabel("")
        self.lbl_folder.setStyleSheet("color: gray;")
        self.lbl_folder.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(self.lbl_folder, 1)
        v1.addLayout(row)
        self.list_sets = QListWidget()
        self.list_sets.setMinimumHeight(140)
        self.list_sets.itemDoubleClicked.connect(self._open_item_result)
        self.list_sets.itemChanged.connect(lambda _item: self._update_buttons())
        v1.addWidget(self.list_sets)
        layout.addWidget(box1)

        # 2. settings
        box2 = QGroupBox("2. Settings")
        form = QFormLayout(box2)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.cmb_type = QComboBox()
        self.cmb_type.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_type.setMinimumContentsLength(18)
        for label, _t in TYPES:
            self.cmb_type.addItem(label)
        self.cmb_type.setToolTip("The kind of panorama the drone took. 'Automatic' detects it from the angles\n"
                                 "of the images (the type is shown next to each panorama in the list).")
        self.cmb_type.setCurrentIndex(int(self.settings.value("type", 0)))
        self.cmb_type.currentIndexChanged.connect(lambda _i: self._refresh_outputs())
        form.addRow("Type:", self.cmb_type)
        self.cmb_quality = QComboBox()
        # don't let the long entries widen the combo box (and squeeze the labels)
        self.cmb_quality.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_quality.setMinimumContentsLength(18)
        for label, _w in QUALITY:
            self.cmb_quality.addItem(label)
        self.cmb_quality.setCurrentIndex(int(self.settings.value("quality", 1)))
        form.addRow("Quality:", self.cmb_quality)
        self.cmb_light = QComboBox()
        self.cmb_light.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_light.setMinimumContentsLength(18)
        for label, _a in LIGHT:
            self.cmb_light.addItem(label)
        self.cmb_light.setCurrentIndex(int(self.settings.value("light", 0)))
        form.addRow("Exposure:", self.cmb_light)
        self.chk_parallax = QCheckBox("Correct for the drone drifting while shooting")
        self.chk_parallax.setChecked(self.settings.value("parallax", "true") == "true")
        self.chk_parallax.setToolTip("Makes the ground close to the drone sharper. Uses GPS and the images to work out\n"
                                     "how far the drone drifted in the wind between shots.")
        form.addRow(self.chk_parallax)
        self.chk_zenith = QCheckBox("Fill in the gap in the sky above the drone")
        self.chk_zenith.setChecked(self.settings.value("zenith", "true") == "true")
        self.chk_zenith.setToolTip("The drone's camera cannot point straight up. The gap is filled with smooth sky (360° only).")
        form.addRow(self.chk_zenith)
        out_row = QHBoxLayout()
        default_out = Path(QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)) / "PanoStitch"
        self.txt_output = QLineEdit(self.settings.value("output_folder", str(default_out)))
        self.txt_output.editingFinished.connect(self._refresh_outputs)
        out_row.addWidget(self.txt_output, 1)
        btn_out = QToolButton()
        btn_out.setText("…")
        btn_out.clicked.connect(self.choose_output)
        out_row.addWidget(btn_out)
        form.addRow("Save to:", out_row)
        layout.addWidget(box2)

        # 3. run
        box3 = QGroupBox("3. Stitch")
        v3 = QVBoxLayout(box3)
        row = QHBoxLayout()
        self.btn_run = QPushButton("Stitch panorama")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setStyleSheet("QPushButton { font-weight: bold; font-size: 14px; }")
        self.btn_run.clicked.connect(self.start)
        row.addWidget(self.btn_run, 1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        row.addWidget(self.btn_cancel)
        v3.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        v3.addWidget(self.progress)
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        v3.addWidget(self.lbl_status)
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(2000)
        self.txt_log.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        v3.addWidget(self.txt_log, 1)
        layout.addWidget(box3, 1)
        return panel

    def _build_preview(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        row = QHBoxLayout()
        btn_open = QPushButton("Open panorama…")
        btn_open.clicked.connect(self.open_panorama)
        row.addWidget(btn_open)
        self.btn_show = QPushButton("Show in Explorer")
        self.btn_show.setEnabled(False)
        self.btn_show.clicked.connect(self.show_in_explorer)
        row.addWidget(self.btn_show)
        self.lbl_viewing = QLabel("")
        self.lbl_viewing.setStyleSheet("color: gray;")
        row.addWidget(self.lbl_viewing, 1)
        v.addLayout(row)
        self.tabs = QTabWidget()
        self.viewer = PanoViewer()
        self.flat = FlatView()
        self.tabs.addTab(self.viewer, "360° view")
        self.tabs.addTab(self.flat, "Flat image")
        v.addWidget(self.tabs, 1)
        return panel

    # ------------------------------------------------------------------ folders
    def choose_folder(self) -> None:
        start = self.settings.value("input_folder", "")
        folder = QFileDialog.getExistingDirectory(self, "Choose panorama folder", start)
        if folder:
            self.load_folder(Path(folder))

    def load_folder(self, folder: Path) -> None:
        self.settings.setValue("input_folder", str(folder))
        self.lbl_folder.setText(str(folder))
        self.list_sets.clear()
        self.lbl_status.setText("Looking for panoramas…")
        self.repaint()
        sets = find_panorama_sets(folder)
        for s in sets:
            item = QListWidgetItem(s.label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setData(ROLE_SET, s)
            item.setCheckState(Qt.Unchecked if self._output_path(s).exists() else Qt.Checked)
            self.list_sets.addItem(item)
        self._refresh_outputs()
        if not sets:
            self.lbl_status.setText("No DJI panoramas found in the folder. Choose the folder with the original "
                                    "images (e.g. PANORAMA/001_0174).")
        else:
            self.lbl_status.setText(f"Found {len(sets)} panorama{'s' if len(sets) != 1 else ''}")
        self._update_buttons()

    def choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Save panoramas to", self.txt_output.text())
        if folder:
            self.txt_output.setText(folder)
            self._refresh_outputs()

    def _chosen_type(self, s: PanoramaSet) -> str:
        t = TYPES[self.cmb_type.currentIndex()][1]
        return s.pano_type if t == "auto" else t

    def _output_path(self, s: PanoramaSet) -> Path:
        return Path(self.txt_output.text()) / s.default_output_name(self._chosen_type(s))

    def _refresh_outputs(self) -> None:
        """Mark panoramas whose output for the current type/folder already exists."""
        for i in range(self.list_sets.count()):
            item = self.list_sets.item(i)
            s: PanoramaSet = item.data(ROLE_SET)
            out = self._output_path(s)
            if out.exists():
                item.setData(ROLE_OUTPUT, str(out))
                item.setText("✓ " + s.label)
                item.setToolTip(f"Already stitched: {out}\nDouble-click to view it")
            else:
                item.setData(ROLE_OUTPUT, None)
                item.setText(s.label)
                item.setToolTip("")

    def _checked_items(self) -> list[QListWidgetItem]:
        return [self.list_sets.item(i) for i in range(self.list_sets.count())
                if self.list_sets.item(i).checkState() == Qt.Checked]

    def _update_buttons(self) -> None:
        running = self._thread is not None
        n = len(self._checked_items())
        self.btn_run.setEnabled(not running and n > 0)
        self.btn_run.setText("Stitch panorama" if n <= 1 else f"Stitch {n} panoramas")
        self.btn_cancel.setEnabled(running)
        for w in (self.btn_folder, self.list_sets, self.cmb_type, self.cmb_quality, self.cmb_light,
                  self.chk_parallax, self.chk_zenith, self.txt_output):
            w.setEnabled(not running)

    # drag & drop of a folder
    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if e.mimeData().hasUrls() and any(Path(u.toLocalFile()).is_dir() for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent) -> None:
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.is_dir() and self._thread is None:
                self.load_folder(p)
                break

    # ------------------------------------------------------------------ running
    def _options(self) -> Options:
        return Options(width=QUALITY[self.cmb_quality.currentIndex()][1],
                       pano_type=TYPES[self.cmb_type.currentIndex()][1],
                       anchor=LIGHT[self.cmb_light.currentIndex()][1],
                       parallax="fit" if self.chk_parallax.isChecked() else "off",
                       fill_zenith=self.chk_zenith.isChecked())

    def start(self) -> None:
        items = self._checked_items()
        if not items:
            return
        out_dir = Path(self.txt_output.text())
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "PanoStitch", f"Cannot create the folder\n{out_dir}\n\n{e}")
            return
        for key, value in (("type", self.cmb_type.currentIndex()),
                           ("quality", self.cmb_quality.currentIndex()), ("light", self.cmb_light.currentIndex()),
                           ("parallax", str(self.chk_parallax.isChecked()).lower()),
                           ("zenith", str(self.chk_zenith.isChecked()).lower()), ("output_folder", str(out_dir))):
            self.settings.setValue(key, value)

        self._job_items = items
        jobs = [Job(it.data(ROLE_SET).folder, self._output_path(it.data(ROLE_SET))) for it in items]
        self.txt_log.clear()
        self.progress.setValue(0)
        self._thread = QThread(self)
        self._worker = StitchWorker(jobs, self._options())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self.txt_log.appendPlainText)
        self._worker.job_started.connect(self._on_job_started)
        self._worker.job_done.connect(self._on_job_done)
        self._worker.job_failed.connect(self._on_job_failed)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()
        self._update_buttons()

    def cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.lbl_status.setText("Cancelling – waiting for the current step to finish…")
            self.btn_cancel.setEnabled(False)

    def _on_progress(self, frac: float, msg: str) -> None:
        self.progress.setValue(int(frac * 1000))
        self.lbl_status.setText(msg)

    def _on_job_started(self, idx: int) -> None:
        item = self._job_items[idx]
        self.list_sets.setCurrentItem(item)
        item.setText("⏳ " + item.data(ROLE_SET).label)

    def _on_job_done(self, idx: int, path: str) -> None:
        item = self._job_items[idx]
        item.setData(ROLE_OUTPUT, path)
        item.setText("✓ " + item.data(ROLE_SET).label)
        item.setToolTip(f"{path}\nDouble-click to view it")
        item.setCheckState(Qt.Unchecked)
        self.show_panorama(Path(path))

    def _on_job_failed(self, idx: int, message: str) -> None:
        item = self._job_items[idx]
        item.setText("✗ " + item.data(ROLE_SET).label)
        item.setToolTip(message)
        self.txt_log.appendPlainText(f"ERROR: {message}")

    def _on_finished(self, cancelled: bool) -> None:
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None
        failed = sum(1 for it in self._job_items if it.text().startswith("✗"))
        if cancelled:
            self.lbl_status.setText("Cancelled")
        elif failed:
            self.lbl_status.setText(f"Done – {failed} panorama{'s' if failed != 1 else ''} failed (see the log)")
        else:
            self.progress.setValue(1000)
            self.lbl_status.setText("Done")
        for it in self._job_items:
            if it.text().startswith("⏳"):
                it.setText(it.data(ROLE_SET).label)
        self._update_buttons()

    # ------------------------------------------------------------------ viewing
    def open_panorama(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open panorama", self.txt_output.text(),
                                              "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
        if path:
            self.show_panorama(Path(path))

    def _open_item_result(self, item: QListWidgetItem) -> None:
        out = item.data(ROLE_OUTPUT)
        if out and Path(out).exists():
            self.show_panorama(Path(out))

    def show_panorama(self, path: Path) -> None:
        try:
            img = imread(path)
        except IOError as e:
            QMessageBox.warning(self, "PanoStitch", str(e))
            return
        h, w = img.shape[:2]
        sphere = self._as_sphere(img, read_gpano(read_xmp(path)))
        self.viewer.set_image(sphere)
        self.flat.set_image(img)
        self.tabs.setTabEnabled(0, sphere is not None)
        self.tabs.setTabToolTip(0, "" if sphere is not None else
                                "This image is not a 360° panorama (e.g. a vertical panorama) – see 'Flat image'")
        if sphere is None:
            self.tabs.setCurrentIndex(1)
        self._last_output = path
        self.btn_show.setEnabled(True)
        self.lbl_viewing.setText(f"{path.name}  ({w} × {h})")

    @staticmethod
    def _as_sphere(img: np.ndarray, gpano: dict | None) -> np.ndarray | None:
        """A full equirectangular image for the 360 viewer: the image itself, a cropped panorama placed on an
        otherwise black sphere using its GPano metadata, or None if it cannot be shown in 360."""
        h, w = img.shape[:2]
        if gpano is None:
            if abs(w - 2 * h) > 2:
                return None
            gpano = {"full_w": w, "full_h": h, "crop_w": w, "crop_h": h, "left": 0, "top": 0}
        f = min(1.0, VIEW_MAX_WIDTH / gpano["full_w"])
        if gpano["crop_w"] == gpano["full_w"] and gpano["crop_h"] == gpano["full_h"]:
            return img if f == 1.0 else cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        full_w, full_h = int(gpano["full_w"] * f), int(gpano["full_h"] * f)
        cw, ch = max(1, int(gpano["crop_w"] * f)), max(1, int(gpano["crop_h"] * f))
        crop = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
        sphere = np.zeros((full_h, full_w, 3), np.uint8)
        top = min(int(gpano["top"] * f), full_h - ch)
        cols = (int(gpano["left"] * f) + np.arange(cw)) % full_w  # may wrap around the back
        sphere[top:top + ch, cols] = crop
        return sphere

    def show_in_explorer(self) -> None:
        if self._last_output and self._last_output.exists():
            if os.name == "nt":
                subprocess.Popen(["explorer", "/select,", str(self._last_output)])
            else:
                subprocess.Popen(["xdg-open", str(self._last_output.parent)])

    def closeEvent(self, e) -> None:
        if self._thread is not None:
            if QMessageBox.question(self, "PanoStitch", "Still stitching. Do you want to stop and close?") \
                    != QMessageBox.Yes:
                e.ignore()
                return
            self._worker.cancel()
            self._thread.quit()
            self._thread.wait(3000)
        e.accept()
