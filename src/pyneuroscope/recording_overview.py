from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class RecordingOverviewWidget(QWidget):
    _SEGMENT_COLORS = (
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
        "#d9d9d9",
        "#bc80bd",
        "#ccebc5",
        "#ffed6f",
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._segments: list[tuple[str, float, float]] = []
        self._total_duration = 0.0
        self._window_start = 0.0
        self._window_duration = 0.0
        self._click_callback = None
        self.setMinimumHeight(50)
        self.setMaximumHeight(58)

    def clear(self) -> None:
        self._segments = []
        self._total_duration = 0.0
        self._window_start = 0.0
        self._window_duration = 0.0
        self.update()

    def set_epochs(self, segments: Sequence[tuple[str, float, float]], total_duration: float) -> None:
        self._segments = [(str(label), float(start), float(end)) for label, start, end in segments]
        self._total_duration = max(0.0, float(total_duration))
        self.update()

    def set_window(self, start_seconds: float, duration_seconds: float) -> None:
        self._window_start = max(0.0, float(start_seconds))
        self._window_duration = max(0.0, float(duration_seconds))
        self.update()

    def set_click_callback(self, callback) -> None:
        self._click_callback = callback

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101216"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor("#dfe5ee")))
        painter.drawText(8, 14, "Session epochs")

        bar_rect = QRectF(8.0, 20.0, max(40.0, self.width() - 16.0), 20.0)
        painter.setPen(QPen(QColor("#46515f"), 1))
        painter.setBrush(QColor("#151a20"))
        painter.drawRect(bar_rect)

        if not self._segments or self._total_duration <= 0:
            painter.setPen(QPen(QColor("#8a9099")))
            painter.drawText(bar_rect.toRect(), Qt.AlignmentFlag.AlignCenter, "No session loaded")
            return

        label_font = QFont()
        label_font.setPointSize(8)
        painter.setFont(label_font)

        for index, (label, start, end) in enumerate(self._segments):
            left = bar_rect.left() + (start / self._total_duration) * bar_rect.width()
            right = bar_rect.left() + (end / self._total_duration) * bar_rect.width()
            segment_rect = QRectF(left, bar_rect.top(), max(1.0, right - left), bar_rect.height())
            segment_color = QColor(self._SEGMENT_COLORS[index % len(self._SEGMENT_COLORS)])
            segment_color.setAlpha(255)
            painter.setPen(QPen(QColor("#151a20"), 1))
            painter.setBrush(segment_color)
            painter.drawRect(segment_rect)
            if segment_rect.width() >= 16.0:
                text_rect = segment_rect.adjusted(4.0, 1.0, -4.0, -1.0)
                painter.setPen(QPen(QColor("#121417"), 1))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)

        view_start = max(0.0, min(self._window_start, self._total_duration))
        view_end = max(view_start, min(self._total_duration, self._window_start + self._window_duration))
        if view_end <= view_start:
            return

        left = bar_rect.left() + (view_start / self._total_duration) * bar_rect.width()
        right = bar_rect.left() + (view_end / self._total_duration) * bar_rect.width()
        window_rect = QRectF(left, bar_rect.top(), max(2.0, right - left), bar_rect.height())
        if window_rect.left() > bar_rect.left():
            painter.fillRect(QRectF(bar_rect.left(), bar_rect.top(), window_rect.left() - bar_rect.left(), bar_rect.height()), QColor(0, 0, 0, 28))
        if window_rect.right() < bar_rect.right():
            painter.fillRect(QRectF(window_rect.right(), bar_rect.top(), bar_rect.right() - window_rect.right(), bar_rect.height()), QColor(0, 0, 0, 28))
        painter.setPen(QPen(QColor("#f4f7fb"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(window_rect)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._click_callback is None or self._total_duration <= 0:
            return
        timestamp = self._timestamp_at_position(event.position())
        if timestamp is None:
            return
        self._click_callback(timestamp)
        event.accept()

    def _timestamp_at_position(self, position: QPointF) -> float | None:
        bar_rect = QRectF(8.0, 20.0, max(40.0, self.width() - 16.0), 20.0)
        if not bar_rect.contains(position):
            return None
        fraction = (position.x() - bar_rect.left()) / max(1.0, bar_rect.width())
        return max(0.0, min(self._total_duration, fraction * self._total_duration))
