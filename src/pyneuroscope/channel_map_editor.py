from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .models import ChannelGroup


@dataclass
class GroupDesign:
    name: str
    channels_per_group: int = 16
    slots: list[int | None] = field(default_factory=lambda: [None] * 16)

    def resize(self, channels_per_group: int) -> None:
        old_slots = self.slots
        self.channels_per_group = channels_per_group
        self.slots = [
            old_slots[index] if index < len(old_slots) else None
            for index in range(channels_per_group)
        ]

    def group(self, group_index: int) -> ChannelGroup:
        channels = [channel for channel in self.slots if channel is not None]
        name = self.name.strip() or f"group{group_index + 1}"
        return ChannelGroup(name, channels)


def group_designs_from_groups(groups: list[ChannelGroup]) -> list[GroupDesign]:
    if not groups:
        return [GroupDesign("group1")]

    designs: list[GroupDesign] = []
    for index, group in enumerate(groups):
        channels_per_group = max(1, len(group.channels))
        design = GroupDesign(group.name or f"group{index + 1}", channels_per_group)
        design.resize(channels_per_group)
        for slot_index, channel in enumerate(group.channels[:channels_per_group]):
            design.slots[slot_index] = channel
        designs.append(design)
    return designs or [GroupDesign("group1")]


class GroupDesignWidget(QWidget):
    def __init__(
        self,
        design: GroupDesign,
        n_channels: int,
        channel_colors: dict[int, str] | None = None,
        display_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.design = design
        self.n_channels = n_channels
        self.channel_colors = channel_colors or {}
        self.display_name = display_name or design.name
        self._hits: dict[int, tuple[float, float, float]] = {}
        self._zoom = 1.0
        self._pan = QPointF()
        self._pan_start = None
        self._pan_origin = QPointF()
        self.setMinimumHeight(420)
        self.setToolTip("Wheel: zoom / Right or middle drag: pan / Reset view: restore position and zoom / Left click: edit channel")

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101216"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._hits = {}

        margin = 42
        header = 28
        height = max(1, self.height() - margin - header)
        row_step = height / max(1, self.design.channels_per_group)
        x = self.width() * 0.45
        radius = max(0.6, min(14.0, row_step * 0.3, self.width() * 0.045)) / max(1.0, self._zoom ** 0.5)

        painter.setPen(QPen(QColor("#d6dde8")))
        painter.drawText(0, 2, self.width(), header, Qt.AlignmentFlag.AlignCenter, self.display_name)
        painter.setClipRect(QRectF(0, header, self.width(), self.height() - header))
        painter.translate(self._pan)
        painter.scale(self._zoom, self._zoom)
        painter.setPen(QPen(QColor("#2d333d"), 1.0 / self._zoom))
        painter.drawLine(int(x), header, int(x), self.height() - margin)

        for slot in range(self.design.channels_per_group):
            y = header + row_step * (slot + 0.5)
            channel = self.design.slots[slot]
            invalid = channel is not None and (channel < 0 or channel >= self.n_channels)
            color = QColor("#3a414d") if channel is None else QColor(self.channel_colors.get(channel, "#ff00ff"))
            painter.setBrush(color)
            pen = QPen(QColor("#ff3333") if invalid else QColor("#c3ccd8"))
            pen.setWidthF((2 if invalid else 1) / self._zoom)
            painter.setPen(pen)
            painter.drawEllipse(QPointF(x, y), radius, radius)

            if row_step * self._zoom >= 12:
                painter.save()
                # Draw labels in screen coordinates: tiny font sizes under a
                # large transform can crash Windows font rendering.
                painter.resetTransform()
                font = painter.font()
                font.setPointSizeF(9.0)
                painter.setFont(font)
                painter.setPen(QPen(QColor("#d6dde8")))
                painter.drawText(
                    QRectF(x * self._zoom + self._pan.x() + radius * self._zoom + 8,
                           y * self._zoom + self._pan.y() - 8,
                           max(42, self.width() - x - radius - 14), 16),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    "" if channel is None else str(channel),
                )
                painter.restore()
            self._hits[slot] = (x * self._zoom + self._pan.x(), y * self._zoom + self._pan.y(), radius * self._zoom + 6)

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        anchor = anchor if anchor is not None else QPointF(self.width() * 0.45, self.height() * 0.5)
        zoom = max(0.25, min(100.0, self._zoom * factor))
        self._pan = anchor - (anchor - self._pan) * (zoom / self._zoom)
        self._zoom = zoom
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF()
        self._pan_start = None
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta:
            self.zoom_by(1.2 ** (delta / 120.0), event.position())
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._pan_start is not None:
            self._pan = self._pan_origin + event.position() - self._pan_start
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_start = None
            event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_start = event.position()
            self._pan_origin = QPointF(self._pan)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        slot = self._slot_at(event.position().x(), event.position().y())
        if slot is None:
            return
        current = self.design.slots[slot]
        value, ok = QInputDialog.getInt(
            self,
            "Channel ID",
            f"Channel for site {slot + 1}",
            current if current is not None else 0,
            -1,
            max(0, self.n_channels - 1),
        )
        if ok:
            self.design.slots[slot] = None if value < 0 else value
            self.update()

    def _slot_at(self, x: float, y: float) -> int | None:
        best: int | None = None
        best_distance = float("inf")
        for slot, (cx, cy, radius) in self._hits.items():
            distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if distance <= radius and distance < best_distance:
                best = slot
                best_distance = distance
        return best


class GroupTab(QWidget):
    def __init__(
        self,
        design: GroupDesign,
        n_channels: int,
        channel_colors: dict[int, str],
        display_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.design = design
        self.viewer = GroupDesignWidget(design, n_channels, channel_colors, display_name)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.channels_per_group = QSpinBox()
        self.channels_per_group.setRange(1, 4096)
        self.channels_per_group.setValue(design.channels_per_group)
        controls.addWidget(QLabel("channels/group"))
        controls.addWidget(self.channels_per_group)
        controls.addStretch(1)
        self.zoom_out = QPushButton("−")
        self.zoom_out.setToolTip("Zoom out")
        self.zoom_out.clicked.connect(lambda: self.viewer.zoom_by(1 / 1.2))
        self.zoom_in = QPushButton("+")
        self.zoom_in.setToolTip("Zoom in")
        self.zoom_in.clicked.connect(lambda: self.viewer.zoom_by(1.2))
        self.reset_view = QPushButton("Reset view")
        self.reset_view.clicked.connect(self.viewer.reset_view)
        controls.addWidget(self.zoom_out)
        controls.addWidget(self.zoom_in)
        controls.addWidget(self.reset_view)
        layout.addLayout(controls)
        layout.addWidget(self.viewer, 1)
        self.channels_per_group.valueChanged.connect(self._resize_design)

    def _resize_design(self) -> None:
        self.design.resize(self.channels_per_group.value())
        self.viewer.reset_view()


class ChannelMapDialog(QDialog):
    def __init__(
        self,
        n_channels: int,
        designs: list[GroupDesign] | None = None,
        channel_colors: dict[int, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Channel Groups")
        self.resize(760, 680)
        self.n_channels = n_channels
        self.designs = designs or [GroupDesign("group1")]
        self.channel_colors = channel_colors or {}
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(False)
        layout.addWidget(self.tabs, 1)

        buttons = QHBoxLayout()
        add_group = QPushButton("+ Group")
        add_group.clicked.connect(self._add_group)
        remove_group = QPushButton("Remove Group")
        remove_group.clicked.connect(self._remove_current_group)
        help_button = QPushButton("Help")
        help_button.clicked.connect(self._show_help)
        buttons.addWidget(add_group)
        buttons.addWidget(remove_group)
        buttons.addWidget(help_button)
        buttons.addStretch(1)

        dialog_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        dialog_buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        dialog_buttons.rejected.connect(self.reject)
        buttons.addWidget(dialog_buttons)
        layout.addLayout(buttons)
        self._rebuild_tabs()

    def groups(self) -> list[ChannelGroup]:
        return [design.group(index) for index, design in enumerate(self.designs)]

    def _add_group(self) -> None:
        self.designs.append(GroupDesign(f"group{len(self.designs) + 1}"))
        self._rebuild_tabs()
        self.tabs.setCurrentIndex(len(self.designs) - 1)

    def _remove_current_group(self) -> None:
        if len(self.designs) <= 1:
            return
        index = self.tabs.currentIndex()
        if index < 0:
            return
        del self.designs[index]
        for group_index, design in enumerate(self.designs):
            design.name = f"group{group_index + 1}"
        self._rebuild_tabs()
        self.tabs.setCurrentIndex(min(index, len(self.designs) - 1))

    def _rebuild_tabs(self) -> None:
        self.tabs.clear()
        for index, design in enumerate(self.designs):
            display_name = f"G{index + 1}"
            self.tabs.addTab(GroupTab(design, self.n_channels, self.channel_colors, display_name), display_name)

    def _show_help(self) -> None:
        QMessageBox.information(
            self,
            "How to Edit Channel Groups",
            "\n".join(
                [
                    "Each tab is one group/shank.",
                    "Click a site to set its channel number.",
                    "Use the wheel or − / + to zoom; right or middle drag to pan.",
                    "Use Reset view to restore the initial position and zoom.",
                    "Use channels/group to change how many sites are shown in the current group.",
                    "Use + Group or Remove Group to change the number of groups.",
                    "Press Apply when the channel layout matches your probe.",
                ]
            ),
        )
