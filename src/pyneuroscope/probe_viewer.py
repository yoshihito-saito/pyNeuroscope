from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from .models import ChannelGroup
from .probe_geometry import ProbeSitePosition


class ProbeViewer(QWidget):
    channelDoubleClicked = Signal(int)
    groupClicked = Signal(int)
    channelsSelected = Signal(object)

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
        self._physical_sites: list[ProbeSitePosition] = []
        self._active_channels: set[int] | None = None
        self._selected_channels: set[int] | None = None
        self._zoom = 1.0
        self._pan = QPointF()
        self._drag_start = self._drag_current = self._pan_start = None
        self._pan_origin = QPointF()
        self.setMinimumWidth(280)
        self.setMinimumHeight(420)
        self.setToolTip("Wheel: zoom / Left drag: show enclosed channels / Double-click: clear selection / Right drag: pan")

    def set_probe(
        self,
        n_channels: int,
        groups: list[ChannelGroup],
        bad_channels: set[int],
        channel_colors: dict[int, str],
        visible_groups: set[int] | None = None,
        channel_geometry: dict[int, ProbeSitePosition] | None = None,
        *,
        physical_sites: list[ProbeSitePosition] | None = None,
        active_channels: set[int] | None = None,
        selected_channels: set[int] | None = None,
    ) -> None:
        if dict(channel_geometry or {}) != self._channel_geometry or list(physical_sites or []) != self._physical_sites or n_channels != self._n_channels:
            self.reset_view()
        self._n_channels = n_channels
        self._groups = list(groups)
        self._bad_channels = set(bad_channels)
        self._channel_colors = dict(channel_colors)
        self._channel_geometry = dict(channel_geometry or {})
        self._physical_sites = list(physical_sites or [])
        self._active_channels = None if active_channels is None else set(active_channels)
        self._selected_channels = None if selected_channels is None else set(selected_channels)
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
        painter.translate(self._pan)
        painter.scale(self._zoom, self._zoom)

        if not self._groups:
            painter.setPen(QPen(QColor("#8a9099")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Channel groups")
            return

        if self._physical_sites or self._channel_geometry and any(
            channel in self._channel_geometry
            for group in self._groups
            for channel in group.channels
        ):
            self._paint_geometry(painter)
            self._finish_paint(painter)
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
                if not is_visible or self._selected_channels is not None and channel not in self._selected_channels:
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
        self._finish_paint(painter)

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
        points: list[ProbeSitePosition] = list(self._physical_sites)
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
        radius = max(1.8 if self._physical_sites else 3.5, min(9.0, 0.35 * min(x_spacing, y_spacing))) / max(1.0, self._zoom ** 0.5)
        show_labels = y_spacing * self._zoom >= 12.0

        if self._physical_sites:
            painter.setPen(QPen(QColor("#303844"), max(0.8, min(3.0, y_spacing * .3))))
            painter.drawPoints(QPolygonF([
                QPointF(x_origin + (point.x - min_x) * x_scale,
                        y_origin + (max_y - point.y) * y_scale)
                for point in self._physical_sites
            ]))

        painter.setPen(QPen(QColor("#d6dde8")))
        title = "Physical sites / active DAT ch" if self._physical_sites else "Channel Geometry"
        painter.drawText(QRectF(0, 0, self.width(), title_height), Qt.AlignmentFlag.AlignCenter, title)

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
                y = y_origin + ((max_y - position.y) if self._physical_sites else (position.y - min_y)) * y_scale
                color = QColor("#5f6670") if channel in self._bad_channels else QColor(self._channel_colors.get(channel, "#ff00ff"))
                active = self._active_channels is None or channel in self._active_channels
                if not active:
                    color = QColor("#343b45")
                if not is_visible or self._selected_channels is not None and channel not in self._selected_channels:
                    color.setAlpha(70)
                painter.setBrush(color)
                pen = QPen(QColor("#c3ccd8") if channel in self._bad_channels else QColor("#12161c"))
                if not is_visible:
                    pen.setColor(QColor("#3a4049"))
                pen.setWidth(2 if channel in self._bad_channels else 1)
                painter.setPen(pen)
                painter.drawEllipse(QPointF(x, y), radius, radius)
                if active:
                    self._dot_hits[channel] = (x, y, radius + 3 / self._zoom)
                if show_labels and active:
                    painter.save()
                    font = painter.font()
                    font.setPointSizeF(max(0.1, 9.0 / self._zoom))
                    painter.setFont(font)
                    painter.setPen(QPen(QColor("#b8c7da") if is_visible else QColor("#616977")))
                    label_left = x + radius + 3 / self._zoom
                    alignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                    if label_left * self._zoom + self._pan.x() + 32 > self.width():
                        label_left = x - radius - 35 / self._zoom
                        alignment = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    painter.drawText(
                        QRectF(label_left, y - 8 / self._zoom, 32 / self._zoom, 16 / self._zoom),
                        alignment,
                        str(channel),
                    )
                    painter.restore()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_start = event.position()
            self._pan_origin = QPointF(self._pan)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        group_index = self._group_at(pos.x(), pos.y())
        if group_index is not None:
            self.groupClicked.emit(group_index)
        else:
            self._drag_start = self._drag_current = pos

    def _finish_paint(self, painter) -> None:
        self._dot_hits = {ch: (x * self._zoom + self._pan.x(), y * self._zoom + self._pan.y(), r * self._zoom)
                          for ch, (x, y, r) in self._dot_hits.items()}
        self._group_hits = {index: QRectF(rect.topLeft() * self._zoom + self._pan,
                                        rect.bottomRight() * self._zoom + self._pan)
                            for index, rect in self._group_hits.items()}
        painter.resetTransform()
        if self._physical_sites and not self._channel_geometry:
            painter.setPen(QPen(QColor("#aab5c4")))
            painter.drawText(QRectF(8, self.height() - 30, self.width() - 16, 24),
                             Qt.AlignmentFlag.AlignCenter, "Load recording map for active sites")
        if self._drag_start is not None and self._drag_current is not None:
            rect = QRectF(self._drag_start, self._drag_current).normalized()
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#7aa7ff")))
            painter.drawRect(rect)

    def reset_view(self) -> None:
        self._zoom, self._pan = 1.0, QPointF()
        self._drag_start = self._drag_current = None
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if not delta:
            return
        anchor = event.position()
        zoom = max(.25, min(100.0, self._zoom * 1.2 ** (delta / 120.0)))
        self._pan = anchor - (anchor - self._pan) * (zoom / self._zoom)
        self._zoom = zoom
        self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._pan_start is not None:
            self._pan = self._pan_origin + event.position() - self._pan_start
            self.update()
        elif self._drag_start is not None:
            self._drag_current = event.position()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_start = None
        elif event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            rect = QRectF(self._drag_start, event.position()).normalized()
            if rect.width() >= 4 or rect.height() >= 4:
                self.channelsSelected.emit(self.channels_in_rect(rect))
            self._drag_start = self._drag_current = None
            self.update()

    def channels_in_rect(self, rect: QRectF) -> set[int]:
        return {ch for ch, (x, y, _) in self._dot_hits.items() if rect.contains(QPointF(x, y))}

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self._drag_start = self._drag_current = None
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._selected_channels is not None:
            self._selected_channels = None
            self.channelsSelected.emit(None)
            self.update()
            event.accept()
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
