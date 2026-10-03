import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QSpinBox

from pyneuroscope.main_window import MainWindow, ProbeConfig
from pyneuroscope.models import ChannelGroup
from pyneuroscope.probe_geometry import (
    ProbeSitePosition, RecordingChannelMap, load_probe_geometry, selected_site_map,
)
from pyneuroscope.probe_viewer import ProbeViewer


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_map_sorts_each_group_from_top_without_changing_probe_membership(app):
    w = MainWindow()
    try:
        groups = [ChannelGroup("a", [0, 1]), ChannelGroup("b", [3, 2])]
        w.probes = [ProbeConfig(4, groups=groups, probe_type="Neuropixels 1.0"), ProbeConfig(2)]
        w._apply_probe_configs_to_model()
        other = list(w.groups[-1].channels)
        mapping = selected_site_map("0 959 400 401", load_probe_geometry("Neuropixels 1.0"), 4)
        w._set_probe_channel_map(0, mapping)
        assert [g.channels for g in w.probes[0].groups] == [[1, 0], [2, 3]]
        assert [g.channels for g in w.groups] == [[1, 0], [2, 3], other]
        assert w.probes[0].channel_map.site_ids[1] == "s0_front_0959"
        assert [g.name for g in w.probes[0].groups] == ["a", "b"]
        assert [i.channel for i in w.viewer._layout_items] == [1, 0, 2, 3, 4, 5]
    finally:
        w.close()


def test_regular_map_sorts_from_top_and_keeps_unmapped_ids(app):
    w = MainWindow()
    try:
        w._set_probe_channel_map(0, RecordingChannelMap(
            {0: ProbeSitePosition(0, 20), 1: ProbeSitePosition(0, 0)}, {0, 1}))
        assert w.groups[0].channels == [1, 0, 2, 3]
    finally:
        w.close()


def test_double_click_clears_selection_without_marking_bad(app):
    w = MainWindow()
    try:
        w._select_probe_channels({1})
        w.probe_viewer.show()
        w.probe_viewer.grab()
        cx, cy, _ = w.probe_viewer._dot_hits[1]
        QTest.mouseDClick(w.probe_viewer, Qt.MouseButton.LeftButton, pos=QPointF(cx, cy).toPoint())
        app.processEvents()
        assert w.selected_channels is None
        assert w.probe_viewer._selected_channels is None
        assert w.bad_channels == set()
        assert {i.channel for i in w.viewer._layout_items} == {0, 1, 2, 3}
    finally:
        w.close()


def test_rectangle_interior_does_not_cover_channel_dots(app):
    v = ProbeViewer()
    try:
        v.resize(400, 600)
        v.set_probe(4, [ChannelGroup("g", [0, 1, 2, 3])], set(), {1: "#ff00ff"})
        before = v.grab().toImage()
        cx, cy, _ = v._dot_hits[1]
        v._drag_start = QPointF(cx - 12, cy - 12)
        v._drag_current = QPointF(cx + 12, cy + 12)
        after = v.grab().toImage()
        assert before.pixelColor(round(cx), round(cy)) == after.pixelColor(round(cx), round(cy))
    finally:
        v.close()


def test_two_probe_controls_scroll_without_collapsing(app):
    w = MainWindow()
    try:
        w.probes = [ProbeConfig(384, probe_type="Neuropixels 1.0") for _ in range(2)]
        w._refresh_probe_controls()
        w.resize(1280, 700)
        w.show()
        app.processEvents()
        rows = [w.probe_rows_layout.itemAt(i).widget() for i in range(2)]
        assert w.recording_scroll.verticalScrollBar().maximum() > 0
        for row in rows:
            spin = row.findChild(QSpinBox)
            assert spin.height() >= spin.minimumSizeHint().height()
            buttons = row.findChildren(QPushButton)
            assert "Active site IDs" not in [b.text() for b in buttons]
            for button in buttons:
                assert button.height() >= button.minimumSizeHint().height()
            visible_rows = sorted({c.geometry().top() for c in row.children() if hasattr(c, "geometry") and c.isWidgetType()})
            assert len(visible_rows) >= 4
    finally:
        w.close()


def test_ctrl_click_adds_removes_and_clears_with_double_click(app):
    w = MainWindow()
    try:
        v = w.probe_viewer
        v.show()
        v.grab()
        def click(ch):
            x, y, _ = v._dot_hits[ch]
            QTest.mouseClick(v, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier,
                             pos=QPointF(x, y).toPoint())
            app.processEvents()
        click(1)
        assert w.selected_channels == {1}
        click(3)
        assert w.selected_channels == {1, 3}
        assert {i.channel for i in w.viewer._layout_items} == {1, 3}
        click(1)
        assert w.selected_channels == {3}
        click(3)
        assert w.selected_channels == set()
        assert w.viewer._layout_items == []
        QTest.mouseDClick(v, Qt.MouseButton.LeftButton, pos=QPointF(10, 10).toPoint())
        app.processEvents()
        assert w.selected_channels is None
        assert {i.channel for i in w.viewer._layout_items} == {0, 1, 2, 3}
        assert not w.bad_channels
    finally:
        w.close()


def test_ctrl_drag_adds_disjoint_range_and_plain_drag_replaces_after_zoom(app):
    v = ProbeViewer()
    try:
        v.resize(400, 600)
        v.set_probe(4, [ChannelGroup("g", [0, 1, 2, 3])], set(), {},
                    channel_geometry={ch: ProbeSitePosition(0, ch * 20) for ch in range(4)})
        v._zoom = 1.05
        v._pan = QPointF(-10, -15)
        v.grab()
        selected = []
        v.channelsSelected.connect(selected.append)
        def drag(ch, modifiers=Qt.KeyboardModifier.NoModifier):
            x, y, _ = v._dot_hits[ch]
            start, end = QPointF(x - 3, y - 3).toPoint(), QPointF(x + 3, y + 3).toPoint()
            QTest.mousePress(v, Qt.MouseButton.LeftButton, modifiers, pos=start)
            QTest.mouseMove(v, end)
            QTest.mouseRelease(v, Qt.MouseButton.LeftButton, modifiers, pos=end)
            v.grab()
        drag(0)
        assert selected[-1] == {0}
        drag(2, Qt.KeyboardModifier.ControlModifier)
        assert selected[-1] == {0, 2}
        drag(3)
        assert selected[-1] == {3}
    finally:
        v.close()


def test_bitmap_map_uses_white_preserving_bad_selection_and_trace_colors(app):
    w = MainWindow()
    try:
        w.bad_channels = {1}
        w._select_probe_channels({0, 1})
        original = dict(w.channel_colors)
        w.display_mode.setCurrentText("bitmap")
        v = w.probe_viewer
        image = v.grab().toImage()
        def dot(ch):
            x, y, _ = v._dot_hits[ch]
            return image.pixelColor(round(x), round(y)).name()
        assert dot(0) == "#ffffff"
        assert dot(1) == "#5f6670"
        assert dot(2) != "#ffffff"
        assert w.channel_colors == original
        w.display_mode.setCurrentText("trace")
        image = v.grab().toImage()
        assert dot(0) == original[0]
        assert dot(1) == "#5f6670"
        assert w.selected_channels == {0, 1}
    finally:
        w.close()


def test_bitmap_physical_geometry_shows_active_white_and_inactive_gray(app):
    v = ProbeViewer()
    try:
        v.resize(400, 600)
        points = [ProbeSitePosition(0, 0), ProbeSitePosition(0, 20), ProbeSitePosition(0, 40)]
        v.set_probe(3, [ChannelGroup("g", [0, 1, 2])], {2}, {0: "#ff0000", 1: "#00ff00", 2: "#0000ff"},
                    channel_geometry=dict(enumerate(points)), physical_sites=points,
                    active_channels={0, 2}, white_channels=True)
        image = v.grab().toImage()
        for ch, color in [(0, "#ffffff"), (2, "#5f6670")]:
            x, y, _ = v._dot_hits[ch]
            assert image.pixelColor(round(x), round(y)).name() == color
        assert 1 not in v._dot_hits
        # The disconnected site in the middle keeps its inactive fill.
        x0, y0, _ = v._dot_hits[0]
        x2, y2, _ = v._dot_hits[2]
        assert image.pixelColor(round((x0 + x2) / 2), round((y0 + y2) / 2)).name() == "#343b45"
        v._zoom = 20
        v._pan = QPointF(x0, y0) * (1 - v._zoom)
        image = v.grab().toImage()
        x, y, _ = v._dot_hits[0]
        assert image.pixelColor(round(x), round(y)).name() == "#ffffff"
    finally:
        v.close()


def test_active_map_button_only_for_neuropixels_and_navigation_removed(app, monkeypatch):
    w = MainWindow()
    try:
        from pyneuroscope.probe_geometry import neuropixels_specs
        calls = []
        monkeypatch.setattr(w, "_load_probe_channel_map", calls.append)
        for probe_type in ["", "linear", *neuropixels_specs()]:
            w._probe_type_changed(0, probe_type)
            row = w.probe_rows_layout.itemAt(0).widget()
            buttons = {b.text(): b for b in row.findChildren(QPushButton)}
            assert "Update XML" in buttons
            assert ("Load active channel map" in buttons) == (probe_type in neuropixels_specs())
            if probe_type in neuropixels_specs():
                buttons["Load active channel map"].click()
                assert calls[-1] == 0
        right_buttons = [b.text() for b in w.color_mode.parentWidget().findChildren(QPushButton)]
        assert "Show all channels" not in right_buttons and "Fit probe" not in right_buttons
    finally:
        w.close()
