from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .models import ChannelGroup
from .probe_geometry import ProbeSitePosition


class ProbeViewer(QWidget):
    channelDoubleClicked = Signal(int)
    groupClicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._n_channels = 0
        self._groups: list[ChannelGroup] = []
        self._bad_channels: set[int] = set()
        self._channel_colors: dict[int, str] = {}
        self._channel_geometry: dict[int, ProbeSitePosition] = {}
        self._dot_hits: dict[int, tuple[float, float, float]] = {}
        self._visible_groups: set[int] = set()
        self._group_hits: dict[int, QRectF] = {}
        self.setMinimumWidth(280)
        self.setMinimumHeight(420)

    def set_probe(
        self,
        n_channels: int,
        groups: list[ChannelGroup],
        bad_channels: set[int],
        channel_colors: dict[int, str],
        visible_groups: set[int] | None = None,
        channel_geometry: dict[int, ProbeSitePosition] | None = None,
    ) -> None:
        self._n_channels = n_channels
        self._groups = list(groups)
        self._bad_channels = set(bad_channels)
        self._channel_colors = dict(channel_colors)
        self._channel_geometry = dict(channel_geometry or {})
        if visible_groups is None:
            self._visible_groups = set(range(len(self._groups)))
        else:
            self._visible_groups = {index for index in visible_groups if 0 <= index < len(self._groups)}
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101216"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._dot_hits = {}
        self._group_hits = {}

        if not self._groups:
            painter.setPen(QPen(QColor("#8a9099")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Channel groups")
            return

        if self._channel_geometry and any(
            channel in self._channel_geometry
            for group in self._groups
            for channel in group.channels
        ):
            self._paint_geometry(painter)
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
            is_visible = group_index in self._visible_groups
            x_center = margin + column_width * (group_index + 0.5)
            header_rect = QRectF(margin + column_width * group_index, title_height, column_width, label_height)
            self._group_hits[group_index] = header_rect
            painter.setPen(QPen(QColor("#9aa4b2") if is_visible else QColor("#59616d")))
            painter.drawText(
                header_rect,
                Qt.AlignmentFlag.AlignCenter,
                f"G{group_index + 1}",
            )
            painter.setPen(QPen(QColor("#2d333d") if is_visible else QColor("#1f242c")))
            painter.drawLine(int(x_center), header + 8, int(x_center), self.height() - margin)

            for row, channel in enumerate(group.channels):
                y = header + row_step * (row + 0.5)
                color = QColor("#5f6670") if channel in self._bad_channels else QColor(self._channel_colors.get(channel, "#ff00ff"))
                if not is_visible:
                    color.setAlpha(70)
                painter.setBrush(color)
                pen = QPen(QColor("#c3ccd8") if channel in self._bad_channels else QColor("#12161c"))
                if not is_visible:
                    pen.setColor(QColor("#3a4049"))
                pen.setWidth(2 if channel in self._bad_channels else 1)
                painter.setPen(pen)
                painter.drawEllipse(QPointF(x_center, y), radius, radius)
                self._dot_hits[channel] = (x_center, y, radius + 5)
                painter.setPen(QPen(QColor("#b8c7da") if is_visible else QColor("#616977")))
                painter.drawText(
                    QRectF(x_center + radius + 3, y - 8, max(24, column_width / 2), 16),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    str(channel),
                )

    def _paint_geometry(self, painter: QPainter) -> None:
        margin = 22
        title_height = 24
        label_height = 20
        header = title_height + label_height
        draw_rect = QRectF(
            margin,
            header + 6,
            max(1, self.width() - 2 * margin),
            max(1, self.height() - margin - header - 6),
        )
        points: list[ProbeSitePosition] = []
        for group in self._groups:
            points.extend(self._channel_geometry[channel] for channel in group.channels if channel in self._channel_geometry)
        if not points:
            return
        min_x = min(point.x for point in points)
        max_x = max(point.x for point in points)
        min_y = min(point.y for point in points)
        max_y = max(point.y for point in points)
        x_span = max(1.0, max_x - min_x)
        y_span = max(1.0, max_y - min_y)
        x_scale = draw_rect.width() / x_span
        y_scale = draw_rect.height() / y_span
        if not x_scale or x_scale <= 0:
            x_scale = 1.0
        if not y_scale or y_scale <= 0:
            y_scale = 1.0
        uniform_scale = min(x_scale, y_scale)
        blend = 0.55
        x_scale = uniform_scale * blend + x_scale * (1.0 - blend)
        y_scale = uniform_scale * blend + y_scale * (1.0 - blend)
        used_width = x_span * x_scale
        used_height = y_span * y_scale
        x_origin = draw_rect.left()
        y_origin = draw_rect.top()
        if used_width < draw_rect.width():
            x_origin += (draw_rect.width() - used_width) * 0.5
        if used_height < draw_rect.height():
            y_origin += (draw_rect.height() - used_height) * 0.5
        x_spacing = _nearest_spacing(point.x for point in points) * x_scale
        y_spacing = _nearest_spacing(point.y for point in points) * y_scale
        radius = max(3.5, min(9.0, 0.35 * min(x_spacing, y_spacing)))
        show_labels = y_spacing >= 8.0

        painter.setPen(QPen(QColor("#d6dde8")))
        painter.drawText(QRectF(0, 0, self.width(), title_height), Qt.AlignmentFlag.AlignCenter, "Channel Geometry")

        for group_index, group in enumerate(self._groups):
            group_points = [
                self._channel_geometry[channel]
                for channel in group.channels
                if channel in self._channel_geometry
            ]
            if not group_points:
                continue
            is_visible = group_index in self._visible_groups
            center_x = sum(point.x for point in group_points) / len(group_points)
            center_px = x_origin + (center_x - min_x) * x_scale
            header_rect = QRectF(center_px - 18, title_height, 36, label_height)
            self._group_hits[group_index] = header_rect
            painter.setPen(QPen(QColor("#9aa4b2") if is_visible else QColor("#59616d")))
            painter.drawText(header_rect, Qt.AlignmentFlag.AlignCenter, f"G{group_index + 1}")
            painter.setPen(QPen(QColor("#2d333d") if is_visible else QColor("#1f242c")))
            painter.drawLine(QPointF(center_px, draw_rect.top()), QPointF(center_px, draw_rect.bottom()))

            for channel in group.channels:
                position = self._channel_geometry.get(channel)
                if position is None:
                    continue
                x = x_origin + (position.x - min_x) * x_scale
                y = y_origin + (position.y - min_y) * y_scale
                color = QColor("#5f6670") if channel in self._bad_channels else QColor(self._channel_colors.get(channel, "#ff00ff"))
                if not is_visible:
                    color.setAlpha(70)
                painter.setBrush(color)
                pen = QPen(QColor("#c3ccd8") if channel in self._bad_channels else QColor("#12161c"))
                if not is_visible:
                    pen.setColor(QColor("#3a4049"))
                pen.setWidth(2 if channel in self._bad_channels else 1)
                painter.setPen(pen)
                painter.drawEllipse(QPointF(x, y), radius, radius)
                self._dot_hits[channel] = (x, y, radius + 5)
                if show_labels:
                    painter.setPen(QPen(QColor("#b8c7da") if is_visible else QColor("#616977")))
                    label_left = x + radius + 3
                    alignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                    if label_left + 32 > self.width():
                        label_left = x - radius - 35
                        alignment = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    painter.drawText(
                        QRectF(label_left, y - 8, 32, 16),
                        alignment,
                        str(channel),
                    )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        group_index = self._group_at(pos.x(), pos.y())
        if group_index is not None:
            self.groupClicked.emit(group_index)

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

    def _group_at(self, x: float, y: float) -> int | None:
        for group_index, rect in self._group_hits.items():
            if rect.contains(QPointF(x, y)):
                return group_index
        return None


def _nearest_spacing(values) -> float:
    ordered = sorted(set(float(value) for value in values))
    if len(ordered) < 2:
        return 20.0
    return min(
        max(1.0, right - left)
        for left, right in zip(ordered, ordered[1:])
        if right > left
    )
