import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pyneuroscope.bitmap_display import bitmap_amplitudes, signed_peak_bins
from pyneuroscope.color_map import palette_from_name
from pyneuroscope.main_window import MainWindow, ProbeConfig
from pyneuroscope.models import SignalEventOverlay
from pyneuroscope.signal_layout import TraceLayoutItem
from pyneuroscope.signal_viewer import SignalViewer


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def items(channels=(0, 1)):
    return [TraceLayoutItem(ch, 0, row, 0, "#ff00ff", False) for row, ch in enumerate(channels)]


def test_signed_peak_bins_preserve_polarity_and_missing_bins():
    values = np.array([0, -19, 4, 2, 13, -8, np.nan, np.nan, np.nan], dtype=float)
    result = signed_peak_bins(values, 3)
    assert result[:2].tolist() == [-19, 13]
    assert np.isnan(result[2])
    assert np.array_equal(values[:6], [0, -19, 4, 2, 13, -8])


def test_bins_are_evenly_partitioned_in_time_and_include_last_sample():
    values = np.arange(103, dtype=float)
    result = signed_peak_bins(values, 10)
    edges = np.arange(11) * 103 // 10
    assert result.tolist() == (edges[1:] - 1).tolist()


def test_shared_color_limit_keeps_relative_channel_amplitudes_and_removes_dc():
    wave = np.sin(np.linspace(0, 16 * np.pi, 2049))
    data = np.column_stack([wave + 1000, wave * 10 - 200])
    original = data.copy()
    values, centers, limit = bitmap_amplitudes(data, [0, 1], 2049)
    assert centers[0] == pytest.approx(1000, abs=.02)
    assert centers[1] == pytest.approx(-200, abs=.02)
    assert limit > 8
    assert np.max(values[1]) / np.max(values[0]) == pytest.approx(10, rel=.02)
    assert np.array_equal(data, original)


def test_bitmap_cache_colormap_scale_zoom_and_data_invalidation(app, monkeypatch):
    from pyneuroscope import signal_viewer as module
    calls = []
    original = module.bitmap_amplitudes
    def compute(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, "bitmap_amplitudes", compute)
    v = SignalViewer()
    try:
        time = np.linspace(0, 1, 2000)
        data = np.column_stack([np.sin(time * 50), np.cos(time * 20)])
        v.resize(800, 400)
        v.set_display_mode("bitmap")
        v.set_traces(time, data, items())
        v.grab()
        first = v._bitmap_cache
        assert first is not None
        assert v._trace_cache == {} and v._path_cache == {}
        v.grab()
        assert v._bitmap_cache is first
        assert len(calls) == 1
        v.set_bitmap_colormaps("Blues", {1: "Reds"})
        v.grab()
        assert v._bitmap_cache is not first
        assert len(calls) == 2
        v.set_traces(time, data, items(), vertical_scale=2)
        v.grab()
        assert len(calls) == 3
        v._x_range = (.25, .75)
        v.grab()
        assert len(calls) == 4
        v.set_traces(time, data.copy(), items(), vertical_scale=2)
        v.grab()
        assert len(calls) == 5
    finally:
        v.close()


def test_bitmap_missing_samples_transparent_and_existing_palette_exact(app):
    v = SignalViewer()
    try:
        data = np.column_stack([np.zeros(10), np.full(10, np.nan)])
        v.set_display_mode("bitmap")
        v.set_bitmap_colormaps("winter")
        v.set_traces(np.arange(10), data, items())
        v.grab()
        image = v._bitmap_cache[1][0]
        assert image.pixelColor(0, 0).name() == palette_from_name("winter", 256)[128]
        assert image.pixelColor(0, 1).alpha() == 0
        assert np.isnan(data[:, 1]).all()
    finally:
        v.close()


def test_mode_control_before_view_uses_existing_probe_cmaps_and_selection(app):
    w = MainWindow()
    try:
        mode_row = w.display_mode.parentWidget().layout().itemAt(0).layout()
        assert mode_row.indexOf(w.display_mode) < mode_row.indexOf(w.view_mode)
        w.probes = [ProbeConfig(2, cmap="Blues"), ProbeConfig(2, cmap="Reds")]
        w._apply_probe_configs_to_model()
        w._current_time = np.linspace(0, 1, 100)
        w._current_data = np.column_stack([np.sin(w._current_time * 20)] * 4)
        raw = w._current_data.copy()
        w.color_mode.setCurrentText("per probe")
        w._select_probe_channels({0, 3})
        w.display_mode.setCurrentText("bitmap")
        assert w.viewer._display_mode == "bitmap"
        assert w.scale.value() == 1
        assert [i.channel for i in w.viewer._layout_items] == [0, 3]
        assert w.viewer._bitmap_channel_colormaps == {0: "Blues", 1: "Blues", 2: "Reds", 3: "Reds"}
        assert not w.csd_enabled.isEnabled()
        w.viewer.grab()
        assert w.viewer._bitmap_cache is not None
        w.display_mode.setCurrentText("trace")
        assert w.viewer._display_mode == "trace"
        assert w.csd_enabled.isEnabled()
        assert np.array_equal(w._current_data, raw)
    finally:
        w.close()


def test_bitmap_group_columns_and_event_overlay_survive_image(app):
    v = SignalViewer()
    try:
        v.resize(800, 400)
        time = np.linspace(0, 1, 10)
        layout = [TraceLayoutItem(0, 0, 0, 0, "#ffffff", False),
                  TraceLayoutItem(1, 1, 0, 1, "#ffffff", False)]
        v.set_display_mode("bitmap")
        v.set_traces(time, np.zeros((10, 2)), layout)
        v.set_event_overlays([SignalEventOverlay(name="event", color="#ff0000", timestamps=np.empty((0, 2)), peaks=np.array([.5]), show_peaks=True)])
        image = v.grab().toImage()
        assert set(v._bitmap_cache[1]) == {0, 1}
        for left, right in v._trace_x_regions():
            color = image.pixelColor(round((left + right) / 2), 60)
            assert color.red() > 200 and color.green() < 100
        left, right = v._trace_x_regions()[1]
        assert v._x_bounds_for_selection(round(left), round(right)) == pytest.approx((0, 1), abs=.01)
    finally:
        v.close()
