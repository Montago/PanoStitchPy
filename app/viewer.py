"""Interactive 360 viewer for equirectangular images.

Views: normal camera view, "tiny planet" and "inverted" (see panostitch.views). Drag to look around (in the
planet views: sideways turns the planet, up/down tilts towards a normal view), scroll to zoom, double-click to
reset. An optional export frame shows which part of the view "Save view" will write.
"""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget

from panostitch.views import VIEW_MODES, frame_rect, plane_coords, render_export, render_view


class PanoViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.OpenHandCursor)
        self._pano: np.ndarray | None = None   # RGB, possibly downscaled for speed
        self._mode = "normal"
        self._yaw, self._pitch, self._fov = 0.0, 0.0, VIEW_MODES["normal"].fov
        self._aspect: float | None = None      # export frame aspect (w / h); None = whole window
        self._drag_from: QPointF | None = None
        self._frame: QImage | None = None

    # --- public
    def set_image(self, bgr: np.ndarray | None) -> None:
        self._pano = None if bgr is None else np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.reset_view()

    def has_image(self) -> bool:
        return self._pano is not None

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.reset_view()

    def set_export_aspect(self, aspect: float | None) -> None:
        self._aspect = aspect
        self.update()

    def reset_view(self) -> None:
        m = VIEW_MODES[self._mode]
        # inverted looks up, which puts the start direction at the bottom: turn it round so it is at the top
        self._yaw = 180.0 if self._mode == "inverted" else 0.0
        self._pitch, self._fov = m.pitch, m.fov
        self._render()

    def export(self, pano_bgr: np.ndarray, long_side: int) -> np.ndarray:
        """The current view inside the export frame, rendered from `pano_bgr` (full resolution) at long_side px."""
        return render_export(pano_bgr, self._mode, self._yaw, self._pitch, self._fov, self.width(), self.height(),
                             self._aspect, long_side)

    # --- rendering
    def _render(self, fast: bool = False) -> None:
        if self._pano is None:
            self._frame = None
            self.update()
            return
        scale = (0.5 if fast else 1.0) * self.devicePixelRatioF()
        w, h = max(16, int(self.width() * scale)), max(16, int(self.height() * scale))
        xs, ys = plane_coords(self.width(), self.height(), (0, 0, self.width(), self.height()), w, h)
        view = render_view(self._pano, self._mode, self._yaw, self._pitch, self._fov, xs, ys)
        self._frame = QImage(view.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 24, 28))
        if self._frame is None:
            painter.setPen(QColor(160, 160, 170))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "No panorama yet\n\nStitch a panorama or open an existing 360° image")
            return
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(self.rect(), self._frame)

        if self._aspect is not None:  # dim what is outside the export frame
            x, y, w, h = frame_rect(self.width(), self.height(), self._aspect)
            shade = QColor(0, 0, 0, 120)
            painter.fillRect(QRectF(0, 0, self.width(), y), shade)
            painter.fillRect(QRectF(0, y + h, self.width(), self.height() - y - h), shade)
            painter.fillRect(QRectF(0, y, x, h), shade)
            painter.fillRect(QRectF(x + w, y, self.width() - x - w, h), shade)
            painter.setPen(QPen(QColor(255, 255, 255, 160), 1, Qt.DashLine))
            painter.drawRect(QRectF(x, y, w - 1, h - 1))

        hint = "turn / tilt" if VIEW_MODES[self._mode].stereographic else "look around"
        text = (f"Drag to {hint} · scroll to zoom · double-click to reset      "
                f"heading {self._yaw % 360:3.0f}°   pitch {self._pitch:+3.0f}°   field of view {self._fov:.0f}°")
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
        half = min(self.width(), self.height()) / 2
        if VIEW_MODES[self._mode].stereographic:
            # turn the planet with sideways drags; tilt towards a normal view with vertical drags
            self._yaw -= d.x() / half * 45.0
            self._pitch = max(-90.0, min(90.0, self._pitch + d.y() / half * 45.0))
        else:
            deg_per_px = self._fov / (2 * half)
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
        lo, hi = VIEW_MODES[self._mode].fov_range
        steps = e.angleDelta().y() / 120.0
        self._fov = max(lo, min(hi, self._fov * (0.92 ** steps)))
        self._render()

    def keyPressEvent(self, e) -> None:
        step = 5.0 if VIEW_MODES[self._mode].stereographic else self._fov / 10
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
