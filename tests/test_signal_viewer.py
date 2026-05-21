import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from pyneuroscope.models import SignalSpikeOverlay
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
    _, second = viewer._trace_x_regions()
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


class FakePainter:
    def __init__(self) -> None:
        self.saved = 0
        self.restored = 0

    def setPen(self, pen) -> None:  # noqa: N802
        _ = pen

    def drawText(self, *args) -> None:  # noqa: N802
        _ = args

    def save(self) -> None:
        self.saved += 1

    def setClipRect(self, rect) -> None:  # noqa: N802
        _ = rect

    def drawLine(self, *args) -> None:  # noqa: N802
        _ = args

    def restore(self) -> None:
        self.restored += 1


def test_spike_raster_restores_painter_clip() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    viewer = SignalViewer()
    viewer._spike_overlays = [
        SignalSpikeOverlay(unit_id=1, label="unit 1", times=np.asarray([0.25, 0.75]), color="#33aaff", channel=0)
    ]

    painter = FakePainter()
    viewer._draw_spike_raster(
        painter,
        np.asarray([0.0, 1.0]),
        columns=1,
        margin_left=50.0,
        column_width=200.0,
        label_gutter=38.0,
        trace_width=150.0,
        top=100.0,
        bottom=180.0,
    )

    assert painter.saved == 1
    assert painter.restored == 1
