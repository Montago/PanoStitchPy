"""Interactive 360 viewer for equirectangular images: drag to look around, scroll to zoom, double-click to reset.

Each frame is a rectilinear (normal camera) view rendered with numpy + cv2.remap, at reduced resolution while
dragging so it stays fluid.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QWheelEvent
from PySide6.QtWidgets import QWidget


class PanoViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.OpenHandCursor)
        self._pano: np.ndarray | None = None   # RGB
        self._yaw, self._pitch, self._fov = 0.0, 0.0, 90.0  # degrees
        self._drag_from: QPointF | None = None
        self._frame: QImage | None = None
        self._grid_key = None
        self._grid = None

    # --- public
    def set_image(self, bgr: np.ndarray | None) -> None:
        self._pano = None if bgr is None else np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.reset_view()

    def has_image(self) -> bool:
        return self._pano is not None

    def reset_view(self) -> None:
        self._yaw, self._pitch, self._fov = 0.0, 0.0, 90.0
        self._render()

    # --- rendering
    def _rays(self, w: int, h: int):
        key = (w, h, self._fov)
        if key != self._grid_key:
            f = (w / 2) / math.tan(math.radians(self._fov) / 2)
            xs = ((np.arange(w, dtype=np.float32) - w / 2 + 0.5) / f)[None, :]
            ys = (-(np.arange(h, dtype=np.float32) - h / 2 + 0.5) / f)[:, None]
            self._grid = (np.broadcast_to(xs, (h, w)), np.broadcast_to(ys, (h, w)))
            self._grid_key = key
        return self._grid

    def _render(self, fast: bool = False) -> None:
        if self._pano is None:
            self._frame = None
            self.update()
            return
        scale = 0.5 if fast else 1.0
        dpr = self.devicePixelRatioF()
        w = max(16, int(self.width() * dpr * scale))
        h = max(16, int(self.height() * dpr * scale))
        x, y = self._rays(w, h)
        p, yw = math.radians(self._pitch), math.radians(self._yaw)
        # tilt (about x), then pan (about the vertical axis); world: x east, y up, z north
        y2 = y * math.cos(p) + math.sin(p)
        z2 = -y * math.sin(p) + math.cos(p)
        x3 = x * math.cos(yw) + z2 * math.sin(yw)
        z3 = -x * math.sin(yw) + z2 * math.cos(yw)
        lon = np.arctan2(x3, z3)
        lat = np.arctan2(y2, np.sqrt(x3 * x3 + z3 * z3))
        H, W = self._pano.shape[:2]
        mx = ((lon * (0.5 / math.pi) + 0.5) * W - 0.5).astype(np.float32)
        my = ((0.5 - lat * (1.0 / math.pi)) * H - 0.5).astype(np.float32)
        view = cv2.remap(self._pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        img = QImage(view.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._frame = img
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 24, 28))
        if self._frame is None:
            painter.setPen(QColor(160, 160, 170))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Intet panorama endnu\n\nStitch et panorama eller åbn et eksisterende 360-billede")
            return
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(self.rect(), self._frame)
        text = (f"Træk for at se rundt · scroll for zoom · dobbeltklik nulstiller      "
                f"retning {self._yaw % 360:3.0f}°   hældning {self._pitch:+3.0f}°   synsfelt {self._fov:.0f}°")
        bar_h = painter.fontMetrics().height() + 10
        bar = self.rect().adjusted(0, self.height() - bar_h, 0, 0)
        painter.fillRect(bar, QColor(0, 0, 0, 130))  # keeps the hint readable on bright sky or sand
        painter.setPen(QColor(235, 235, 240))
        painter.drawText(bar.adjusted(10, 0, -10, 0), Qt.AlignVCenter | Qt.AlignLeft, text)

    def resizeEvent(self, _event) -> None:
        self._render()

    # --- interaction
    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.LeftButton:
            self._drag_from = e.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._drag_from is None or self._pano is None:
            return
        d = e.position() - self._drag_from
        self._drag_from = e.position()
        deg_per_px = self._fov / max(1, self.width())
        self._yaw -= d.x() * deg_per_px
        self._pitch = max(-90.0, min(90.0, self._pitch + d.y() * deg_per_px))
        self._render(fast=True)

    def mouseReleaseEvent(self, _e: QMouseEvent) -> None:
        self._drag_from = None
        self.setCursor(Qt.OpenHandCursor)
        self._render()

    def mouseDoubleClickEvent(self, _e: QMouseEvent) -> None:
        self.reset_view()

    def wheelEvent(self, e: QWheelEvent) -> None:
        steps = e.angleDelta().y() / 120.0
        self._fov = max(20.0, min(120.0, self._fov * (0.9 ** steps)))
        self._render()

    def keyPressEvent(self, e) -> None:
        step = self._fov / 10
        key = e.key()
        if key == Qt.Key_Left:
            self._yaw -= step
        elif key == Qt.Key_Right:
            self._yaw += step
        elif key == Qt.Key_Up:
            self._pitch = min(90.0, self._pitch + step)
        elif key == Qt.Key_Down:
            self._pitch = max(-90.0, self._pitch - step)
        else:
            return super().keyPressEvent(e)
        self._render()
