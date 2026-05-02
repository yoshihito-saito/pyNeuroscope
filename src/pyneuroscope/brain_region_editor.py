from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .models import ChannelGroup


class BrainRegionProbeWidget(QWidget):
    def __init__(
        self,
        groups: list[ChannelGroup],
        channel_regions: dict[int, str],
        channel_colors: dict[int, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._groups = list(groups)
        self._channel_regions = dict(channel_regions)
        self._channel_colors = dict(channel_colors or {})
        self._selected_channels: set[int] = set()
        self._dot_hits: dict[int, tuple[float, float, float]] = {}
        self._drag_start: QPointF | None = None
        self._drag_current: QPointF | None = None
        self.setMinimumWidth(640)
        self.setMinimumHeight(520)
        self.setMouseTracking(True)

    @property
    def selected_channels(self) -> set[int]:
        return set(self._selected_channels)

    @property
    def channel_regions(self) -> dict[int, str]:
        return dict(self._channel_regions)

    def set_data(self, groups: list[ChannelGroup], channel_regions: dict[int, str]) -> None:
        self._groups = list(groups)
        self._channel_regions = dict(channel_regions)
        self._selected_channels &= {channel for group in groups for channel in group.channels}
        self.update()

    def assign_region(self, name: str) -> None:
        label = name.strip()
        if not label:
            return
        for channel in self._selected_channels:
            self._channel_regions[channel] = label
        self.update()

    def clear_region(self) -> None:
        for channel in self._selected_channels:
            self._channel_regions.pop(channel, None)
        self.update()

    def clear_selection(self) -> None:
        self._selected_channels.clear()
        self.update()

    def region_names(self) -> list[str]:
        return sorted({label for label in self._channel_regions.values() if label.strip()})

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101216"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._dot_hits = {}

        if not self._groups:
            painter.setPen(QPen(QColor("#8a9099")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No channel groups")
            return

        margin = 22
        title_height = 28
        label_height = 22
        header = title_height + label_height
        width = max(1, self.width() - 2 * margin)
        height = max(1, self.height() - margin - header)
        group_count = max(1, len(self._groups))
        column_width = width / group_count
        max_rows = max((len(group.channels) for group in self._groups), default=1)
        row_step = height / max(1, max_rows)
        radius = max(6.0, min(13.0, column_width * 0.12, row_step * 0.28))

        painter.setPen(QPen(QColor("#d6dde8")))
        painter.drawText(QRectF(0, 0, self.width(), title_height), Qt.AlignmentFlag.AlignCenter, "Brain Regions")

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
                label = self._channel_regions.get(channel, "")
                selected = channel in self._selected_channels
                color = QColor(self._channel_colors.get(channel, "#808080"))
                painter.setBrush(color)
                pen = QPen(QColor("#facc15") if selected else QColor("#c3ccd8"))
                pen.setWidth(3 if selected else 1)
                painter.setPen(pen)
                painter.drawEllipse(QPointF(x_center, y), radius, radius)
                self._dot_hits[channel] = (x_center, y, radius + 8)

                painter.setPen(QPen(QColor("#b8c7da")))
                painter.drawText(
                    QRectF(x_center + radius + 4, y - 8, max(30, column_width / 2), 16),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    str(channel),
                )
                if label:
                    painter.setPen(QPen(QColor("#9bd1ff")))
                    painter.drawText(
                        QRectF(x_center + radius + 34, y - 8, max(40, column_width / 1.8), 16),
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                        label,
                    )

        if self._drag_start is not None and self._drag_current is not None:
            rect = QRectF(self._drag_start, self._drag_current).normalized()
            painter.setPen(QPen(QColor("#facc15"), 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(250, 204, 21, 40))
            painter.drawRect(rect)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        channel = self._channel_at(event.position().x(), event.position().y())
        modifiers = event.modifiers()
        if channel is not None:
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                if channel in self._selected_channels:
                    self._selected_channels.remove(channel)
                else:
                    self._selected_channels.add(channel)
            else:
                self._selected_channels = {channel}
            self._drag_start = None
            self._drag_current = None
            self.update()
            return
        self._drag_start = event.position()
        self._drag_current = event.position()
        if not (modifiers & Qt.KeyboardModifier.ControlModifier):
            self._selected_channels.clear()
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_start is None:
            return
        self._drag_current = event.position()
        self._selected_channels |= self._channels_in_rect(QRectF(self._drag_start, self._drag_current).normalized())
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._drag_start is not None and self._drag_current is not None:
            rect = QRectF(self._drag_start, self._drag_current).normalized()
            self._selected_channels |= self._channels_in_rect(rect)
        self._drag_start = None
        self._drag_current = None
        self.update()

    def _channels_in_rect(self, rect: QRectF) -> set[int]:
        selected: set[int] = set()
        for channel, (cx, cy, radius) in self._dot_hits.items():
            hit_rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
            if rect.intersects(hit_rect) or rect.contains(QPointF(cx, cy)):
                selected.add(channel)
        return selected

    def _channel_at(self, x: float, y: float) -> int | None:
        best_channel: int | None = None
        best_distance = float("inf")
        for channel, (cx, cy, radius) in self._dot_hits.items():
            distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if distance <= radius and distance < best_distance:
                best_channel = channel
                best_distance = distance
        return best_channel


class BrainRegionEditorDialog(QDialog):
    def __init__(
        self,
        groups: list[ChannelGroup],
        channel_regions: dict[int, str] | None = None,
        channel_colors: dict[int, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Probe Editor")
        self.resize(980, 680)
        self.viewer = BrainRegionProbeWidget(groups, channel_regions or {}, channel_colors or {})
        self.region_list = QListWidget()

        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(self.viewer, 1)

        controls = QHBoxLayout()
        assign_button = QPushButton("Assign Region")
        assign_button.clicked.connect(self._assign_region)
        clear_button = QPushButton("Clear Region")
        clear_button.clicked.connect(self._clear_region)
        help_button = QPushButton("Help")
        help_button.clicked.connect(self._show_help)
        controls.addWidget(assign_button)
        controls.addWidget(clear_button)
        controls.addWidget(help_button)
        controls.addStretch(1)
        left.addLayout(controls)

        layout.addLayout(left, 1)

        side = QVBoxLayout()
        side.addWidget(QLabel("Assigned regions"))
        side.addWidget(self.region_list, 1)
        dialog_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        dialog_buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        dialog_buttons.rejected.connect(self.reject)
        side.addWidget(dialog_buttons)
        layout.addLayout(side)
        self._refresh_region_list()

    @property
    def channel_regions(self) -> dict[int, str]:
        return self.viewer.channel_regions

    def _assign_region(self) -> None:
        if not self.viewer.selected_channels:
            return
        value, ok = QInputDialog.getText(
            self,
            "Assign Region",
            f"Region name for {len(self.viewer.selected_channels)} selected channel(s)",
        )
        if not ok:
            return
        self.viewer.assign_region(value)
        self._refresh_region_list()

    def _clear_region(self) -> None:
        if not self.viewer.selected_channels:
            return
        self.viewer.clear_region()
        self._refresh_region_list()

    def _refresh_region_list(self) -> None:
        self.region_list.clear()
        self.region_list.addItems(self.viewer.region_names())

    def _show_help(self) -> None:
        QMessageBox.information(
            self,
            "How to Use Brain Regions",
            "\n".join(
                [
                    "Click a channel to select one channel.",
                    "Ctrl+click to add or remove individual channels.",
                    "Drag with the mouse to select a range of channels.",
                    "Press Assign Region to enter one label for all selected channels.",
                    "Press Clear Region to remove the region label from the selected channels.",
                ]
            ),
        )
