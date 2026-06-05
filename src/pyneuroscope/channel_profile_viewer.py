from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .models import ChannelGroup


def channel_rms(data: np.ndarray, scale: float = 1.0) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("data must be a 2D samples-by-channels array")
    if values.shape[0] == 0:
        return np.zeros((values.shape[1],), dtype=np.float64)
    scaled = values * float(scale)
    return np.sqrt(np.mean(np.square(scaled), axis=0))


class ChannelProfileViewer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rms = np.asarray([], dtype=np.float64)
        self._groups: list[ChannelGroup] = []
        self._bad_channels: set[int] = set()
        self._channel_colors: dict[int, str] = {}
        self._unit = "uV"
        self._subtitle = ""
        self.setMinimumWidth(150)
        self.setMinimumHeight(420)

    def set_profile(
        self,
        rms: Sequence[float],
        groups: Sequence[ChannelGroup],
        bad_channels: set[int],
        channel_colors: dict[int, str],
        *,
        unit: str = "uV",
        subtitle: str = "",
    ) -> None:
        self._rms = np.asarray(rms, dtype=np.float64).reshape(-1)
        self._groups = list(groups)
        self._bad_channels = set(bad_channels)
        self._channel_colors = dict(channel_colors)
        self._unit = unit
        self._subtitle = subtitle
        self.update()

    def clear(self) -> None:
        self._rms = np.asarray([], dtype=np.float64)
        self._subtitle = ""
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101216"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        groups = [group for group in self._groups if group.channels]
        if self._rms.size == 0 or not groups:
            painter.setPen(QPen(QColor("#8a9099")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "RMS")
            return

        finite = np.asarray(
            [
                self._rms[channel]
                for group in groups
                for channel in group.channels
                if channel < self._rms.size
            ],
            dtype=np.float64,
        )
        finite = finite[np.isfinite(finite)]
        max_value = float(np.max(finite)) if finite.size else 1.0
        max_value = max(max_value, 1e-12)

        margin_left = 10
        margin_right = 10
        header = 44
        footer = 52
        width = max(1.0, self.width() - margin_left - margin_right)
        height = max(1.0, self.height() - header - footer)
        group_count = max(1, len(groups))
        gap = 4.0
        column_width = max(8.0, (width - gap * (group_count - 1)) / group_count)

        painter.setPen(QPen(QColor("#d6dde8")))
        painter.drawText(
            QRectF(0, 4, self.width(), 18),
            Qt.AlignmentFlag.AlignCenter,
            f"RMS ({self._unit})",
        )
        for group_index, group in enumerate(groups):
            x = margin_left + group_index * (column_width + gap)
            cell_height = max(3.0, height / max(1, len(group.channels)))
            painter.setPen(QPen(QColor("#9aa4b2")))
            painter.drawText(
                QRectF(x, header - 20, column_width, 16),
                Qt.AlignmentFlag.AlignCenter,
                f"G{group_index + 1}",
            )
            for row, channel in enumerate(group.channels):
                y = header + row * cell_height
                rect = QRectF(x, y, column_width, max(1.0, cell_height - 1.0))
                value = float(self._rms[channel]) if channel < self._rms.size else np.nan
                color = QColor("#343a43") if not np.isfinite(value) else self._heat_color(value / max_value)
                if channel in self._bad_channels:
                    color = QColor("#5f6670")
                painter.fillRect(rect, color)
                painter.setPen(QPen(QColor("#15191f"), 1))
                painter.drawRect(rect)
                if cell_height >= 12 and column_width >= 20:
                    painter.setPen(QPen(QColor("#e7edf6") if value / max_value > 0.45 else QColor("#15191f")))
                    painter.drawText(
                        rect.adjusted(1, 0, -1, 0),
                        Qt.AlignmentFlag.AlignCenter,
                        str(channel),
                    )

        self._draw_colorbar(painter, margin_left, self.height() - footer + 8, width, max_value)
        if self._subtitle:
            painter.setPen(QPen(QColor("#9aa4b2")))
            painter.drawText(
                QRectF(2, self.height() - 18, self.width() - 4, 16),
                Qt.AlignmentFlag.AlignCenter,
                self._subtitle,
            )

    def _heat_color(self, normalized: float) -> QColor:
        x = min(1.0, max(0.0, float(normalized)))
        anchors = [
            (0.0, QColor("#1f2a44")),
            (0.35, QColor("#2aa7b8")),
            (0.65, QColor("#f0d35a")),
            (1.0, QColor("#e84a5f")),
        ]
        for index in range(len(anchors) - 1):
            left_pos, left = anchors[index]
            right_pos, right = anchors[index + 1]
            if left_pos <= x <= right_pos:
                span = max(1e-12, right_pos - left_pos)
                t = (x - left_pos) / span
                return QColor(
                    int(left.red() + (right.red() - left.red()) * t),
                    int(left.green() + (right.green() - left.green()) * t),
                    int(left.blue() + (right.blue() - left.blue()) * t),
                )
        return anchors[-1][1]

    def _draw_colorbar(self, painter: QPainter, x: float, y: float, width: float, max_value: float) -> None:
        steps = max(1, int(width))
        for index in range(steps):
            value = index / max(1, steps - 1)
            painter.fillRect(QRectF(x + index, y, 1, 8), self._heat_color(value))
        painter.setPen(QPen(QColor("#7f8793")))
        painter.drawText(QRectF(x, y + 10, width * 0.5, 14), Qt.AlignmentFlag.AlignLeft, "0")
        painter.drawText(
            QRectF(x + width * 0.5, y + 10, width * 0.5, 14),
            Qt.AlignmentFlag.AlignRight,
            f"{max_value:.1f}",
        )
