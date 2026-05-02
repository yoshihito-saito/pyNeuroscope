import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from pyneuroscope.signal_layout import TraceLayoutItem
from pyneuroscope.signal_viewer import SignalViewer


def test_group_column_x_selection_uses_drag_start_column() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    viewer = SignalViewer()
    viewer.resize(600, 400)
    viewer.set_traces(
        None,
        None,
        [
            TraceLayoutItem(0, 0, 0, 0, "#ffffff", False),
            TraceLayoutItem(1, 1, 0, 1, "#ffffff", False),
        ],
    )
    first, second = viewer._trace_x_regions()
    start = int(second[0] + (second[1] - second[0]) * 0.25)
    end = int(second[0] + (second[1] - second[0]) * 0.75)

    left, right = viewer._x_bounds_for_selection(start, end)

    assert 0.20 < left < 0.30
    assert 0.70 < right < 0.80


def test_group_column_x_selection_can_drag_backward_in_start_column() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    viewer = SignalViewer()
    viewer.resize(600, 400)
    viewer.set_traces(
        None,
        None,
        [
            TraceLayoutItem(0, 0, 0, 0, "#ffffff", False),
            TraceLayoutItem(1, 1, 0, 1, "#ffffff", False),
        ],
    )
    _, second = viewer._trace_x_regions()
    start = int(second[0] + (second[1] - second[0]) * 0.75)
    end = int(second[0] + (second[1] - second[0]) * 0.25)

    left, right = viewer._x_bounds_for_selection(start, end)

    assert 0.20 < left < 0.30
    assert 0.70 < right < 0.80
