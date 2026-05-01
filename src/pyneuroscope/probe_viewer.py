from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .models import ChannelGroup


class ProbeViewer(QWidget):
    channelDoubleClicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._n_channels = 0
        self._groups: list[ChannelGroup] = []
        self._bad_channels: set[int] = set()
        self._channel_colors: dict[int, str] = {}
        self._dot_hits: dict[int, tuple[float, float, float]] = {}
        self.setMinimumWidth(280)
        self.setMinimumHeight(420)

    def set_probe(
        self,
        n_channels: int,
        groups: list[ChannelGroup],
        bad_channels: set[int],
        channel_colors: dict[int, str],
    ) -> None:
        self._n_channels = n_channels
        self._groups = list(groups)
        self._bad_channels = set(bad_channels)
        self._channel_colors = dict(channel_colors)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101216"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._dot_hits = {}

        if not self._groups:
            painter.setPen(QPen(QColor("#8a9099")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Channel groups")
            return

        margin = 22
        title_height = 24
        label_height = 20
        header = title_height + label_height
        width = max(1, self.width() - 2 * margin)
        height = max(1, self.height() - margin - header)
        group_count = max(1, len(self._groups))
        column_width = width / group_count
        max_rows = max((len(group.channels) for group in self._groups), default=1)
        row_step = height / max(1, max_rows)
        radius = max(4.0, min(10.0, column_width * 0.15, row_step * 0.28))

        painter.setPen(QPen(QColor("#d6dde8")))
        painter.drawText(QRectF(0, 0, self.width(), title_height), Qt.AlignmentFlag.AlignCenter, "Channel Groups")

        for group_index, group in enumerate(self._groups):
            x_center = margin + column_width * (group_index + 0.5)
            painter.setPen(QPen(QColor("#9aa4b2")))
            painter.drawText(
                QRectF(margin + column_width * group_index, title_height, column_width, label_height),
                Qt.AlignmentFlag.AlignCenter,
                f"G{group_index + 1}",
            )
            painter.setPen(QPen(QColor("#2d333d")))
            painter.drawLine(int(x_center), header + 8, int(x_center), self.height() - margin)

            for row, channel in enumerate(group.channels):
                y = header + row_step * (row + 0.5)
                color = QColor("#5f6670") if channel in self._bad_channels else QColor(
                    self._channel_colors.get(channel, "#ff00ff")
                )
                painter.setBrush(color)
                pen = QPen(QColor("#c3ccd8") if channel in self._bad_channels else QColor("#12161c"))
                pen.setWidth(2 if channel in self._bad_channels else 1)
                painter.setPen(pen)
                painter.drawEllipse(QPointF(x_center, y), radius, radius)
                self._dot_hits[channel] = (x_center, y, radius + 5)
                painter.setPen(QPen(QColor("#b8c7da")))
                painter.drawText(
                    QRectF(x_center + radius + 3, y - 8, max(24, column_width / 2), 16),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    str(channel),
                )

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        channel = self._channel_at(pos.x(), pos.y())
        if channel is not None:
            self.channelDoubleClicked.emit(channel)

    def _channel_at(self, x: float, y: float) -> int | None:
        best_channel: int | None = None
        best_distance = float("inf")
        for channel, (cx, cy, radius) in self._dot_hits.items():
            distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if distance <= radius and distance < best_distance:
                best_channel = channel
                best_distance = distance
        return best_channel
