import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

from pyneuroscope.main_window import MainWindow
from pyneuroscope.models import ChannelGroup


class WheelEvent:
    def __init__(self, delta: int, modifiers=Qt.KeyboardModifier.ControlModifier) -> None:
        self._delta = delta
        self._modifiers = modifiers
        self.accepted = False

    def modifiers(self):
        return self._modifiers

    def angleDelta(self) -> QPoint:  # noqa: N802
        return QPoint(0, self._delta)

    def accept(self) -> None:
        self.accepted = True


def test_ctrl_wheel_changes_time_window_from_viewer_and_scroll_viewport() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window._set_window_start_seconds(2.0)
    window._set_window_duration_seconds(1.0)

    event = WheelEvent(120)
    assert window._handle_trace_wheel(window.viewer, event)
    assert event.accepted
    assert window._window_start_seconds() == 2.0
    assert window._window_duration_seconds() == 0.8

    window.view_mode.setCurrentText("group_columns")
    event = WheelEvent(120)
    assert window._handle_trace_wheel(window.signal_scroll.viewport(), event)
    assert event.accepted
    assert window._window_start_seconds() == 2.0
    assert window._window_duration_seconds() == 0.64


def test_color_modes_use_group_order_and_group_local_palettes() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.n_channels.setValue(6)
    window.groups = [ChannelGroup("g1", [4, 2, 0]), ChannelGroup("g2", [5, 1])]

    window.color_map.setCurrentText("spring")
    window.color_mode.setCurrentText("all")
    window._reset_colors()
    assert window.channel_colors[4] == "#ff00ff"
    assert window.channel_colors[0] == "#ff8080"
    assert window.channel_colors[1] == "#ffff00"
    assert window.channel_colors[3] == "#808080"

    window.color_mode.setCurrentText("group")
    window._reset_colors()
    assert window.channel_colors[4] == "#ff00ff"
    assert window.channel_colors[2] == "#ff8080"
    assert window.channel_colors[0] == "#ffff00"
    assert window.channel_colors[5] == "#ff00ff"
    assert window.channel_colors[1] == "#ffff00"
    assert window.channel_colors[3] == "#808080"


def test_shift_wheel_changes_row_spacing() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    start = window.spacing.value()

    event = WheelEvent(120, Qt.KeyboardModifier.ShiftModifier)
    assert window._handle_trace_wheel(window.viewer, event)
    assert event.accepted
    assert window.spacing.value() > start


def test_region_summary_reports_assigned_counts() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.channel_regions = {0: "CA1", 1: "CA1", 2: "CTX"}

    window._refresh_region_summary()

    assert window.region_summary.text() == "Assigned channels: CA1: 2, CTX: 1"
