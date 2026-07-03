import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from pyneuroscope.models import SignalSpikeOverlay
from pyneuroscope.signal_layout import TraceLayoutItem
from pyneuroscope.signal_viewer import SignalViewer, _csd_depths, _normalized_time_fractions


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


def test_geometry_columns_use_compact_spacing_mode() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    viewer = SignalViewer()
    items = [
        TraceLayoutItem(0, 0, 0, 0, "#ffffff", False, x=-10.0),
        TraceLayoutItem(1, 0, 1, 1, "#ffffff", False, x=10.0),
    ]

    viewer.set_traces(None, None, items)
    viewer._show_channel_labels = False

    assert viewer._uses_compact_geometry_columns()
    assert len(viewer._trace_x_regions()) == 2


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


def test_white_background_overlay_color_uses_gentle_dimming() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    viewer = SignalViewer()
    viewer.set_background_mode("white")

    color = viewer._overlay_color("#ffff00")

    assert color.red() == 178
    assert color.green() == 178
    assert color.blue() == 0


def test_csd_image_interpolates_vertically_to_target_height() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    viewer = SignalViewer()
    csd = np.asarray(
        [
            [-1.0, 1.0],
            [0.0, 0.0],
            [1.0, -1.0],
        ]
    )

    image = viewer._csd_image(csd, target_height=24)

    assert image.height() == 24
    assert image.width() == 3


def test_csd_segments_split_at_bad_channels() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    viewer = SignalViewer()
    items = [
        TraceLayoutItem(0, 0, 0, 0, "#ffffff", False),
        TraceLayoutItem(1, 0, 1, 0, "#ffffff", False),
        TraceLayoutItem(2, 0, 2, 0, "#ffffff", False),
        TraceLayoutItem(3, 0, 3, 0, "#ffffff", True),
        TraceLayoutItem(4, 0, 4, 0, "#ffffff", False),
        TraceLayoutItem(5, 0, 5, 0, "#ffffff", False),
        TraceLayoutItem(6, 0, 6, 0, "#ffffff", False),
        TraceLayoutItem(7, 0, 7, 0, "#ffffff", False),
    ]

    segments = viewer._csd_valid_segments(items)

    assert [[item.channel for item in segment] for segment in segments] == [[0, 1, 2], [4, 5, 6, 7]]


def test_csd_segments_drop_runs_shorter_than_three_channels() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    viewer = SignalViewer()
    items = [
        TraceLayoutItem(0, 0, 0, 0, "#ffffff", False),
        TraceLayoutItem(1, 0, 1, 0, "#ffffff", False),
        TraceLayoutItem(2, 0, 2, 0, "#ffffff", True),
        TraceLayoutItem(3, 0, 3, 0, "#ffffff", False),
        TraceLayoutItem(4, 0, 4, 0, "#ffffff", False),
        TraceLayoutItem(5, 0, 5, 0, "#ffffff", False),
    ]

    segments = viewer._csd_valid_segments(items)

    assert [[item.channel for item in segment] for segment in segments] == [[3, 4, 5]]


def test_trace_x_fractions_use_actual_time_spacing() -> None:
    fractions = _normalized_time_fractions(
        np.asarray([0.0, 0.1, 1.0]),
        np.asarray([0.0, 0.1, 1.0]),
    )

    assert fractions.tolist() == [0.0, 0.1, 1.0]


def test_trace_display_keeps_half_sample_per_pixel_for_single_column() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    viewer = SignalViewer()

    assert viewer._trace_display_max_points(300.0, columns=1) == 150


def test_trace_display_keeps_two_samples_per_pixel_for_multi_column() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    viewer = SignalViewer()

    assert viewer._trace_display_max_points(300.0, columns=2) == 600


def test_trace_pen_width_increases_as_time_axis_zooms_in() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    viewer = SignalViewer()
    full_window = np.arange(0.0, 1.0, 1.0 / 20000.0)
    zoomed_window = np.arange(0.0, 0.05, 1.0 / 20000.0)

    assert viewer._trace_pen_width(zoomed_window, 500.0) > viewer._trace_pen_width(full_window, 500.0)


def test_trace_pen_width_responds_to_scale_only_for_multi_column() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    viewer = SignalViewer()
    time = np.arange(0.0, 0.2, 1.0 / 20000.0)

    viewer._vertical_scale = 1.0
    single_base = viewer._trace_pen_width(time, 300.0, columns=1)
    multi_base = viewer._trace_pen_width(time, 300.0, columns=2)
    viewer._vertical_scale = 2.0

    assert viewer._trace_pen_width(time, 300.0, columns=1) == single_base
    assert viewer._trace_pen_width(time, 300.0, columns=2) > multi_base


def test_csd_depths_use_geometry_y_when_strictly_ordered() -> None:
    items = [
        TraceLayoutItem(0, 0, 0, 0, "#ffffff", False, y=-60.0),
        TraceLayoutItem(1, 0, 1, 0, "#ffffff", False, y=-40.0),
        TraceLayoutItem(2, 0, 2, 0, "#ffffff", False, y=-10.0),
    ]

    assert _csd_depths(items) == [-60.0, -40.0, -10.0]


def test_csd_depths_fall_back_without_monotonic_geometry() -> None:
    items = [
        TraceLayoutItem(0, 0, 0, 0, "#ffffff", False, y=-20.0),
        TraceLayoutItem(1, 0, 1, 0, "#ffffff", False, y=-40.0),
        TraceLayoutItem(2, 0, 2, 0, "#ffffff", False, y=-10.0),
    ]

    assert _csd_depths(items) is None


def test_csd_display_data_decimates_to_1250_hz() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    viewer = SignalViewer()
    sampling_rate = 20000.0
    time = np.arange(0.0, 1.0, 1.0 / sampling_rate)
    data = np.column_stack((time, time * 2.0))

    reduced = viewer._csd_display_data(data, time, max_points=2000)

    assert 1200 <= reduced.shape[0] <= 1250


def test_csd_display_data_also_caps_very_long_windows_by_screen_bins() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    viewer = SignalViewer()
    sampling_rate = 1250.0
    time = np.arange(0.0, 10.0, 1.0 / sampling_rate)
    data = np.column_stack((time, time * 2.0))

    reduced = viewer._csd_display_data(data, time, max_points=100)

    assert reduced.shape[0] <= 250
