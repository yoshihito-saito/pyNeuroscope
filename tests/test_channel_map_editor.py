import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QFont, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from pyneuroscope.channel_map_editor import GroupDesign, GroupTab, group_designs_from_groups
from pyneuroscope.models import ChannelGroup


def test_group_designs_from_groups_preserves_each_group() -> None:
    designs = group_designs_from_groups(
        [
            ChannelGroup("group1", [0, 2]),
            ChannelGroup("group2", [1, 3, 5]),
        ]
    )

    assert len(designs) == 2
    assert designs[0].name == "group1"
    assert designs[0].slots[:2] == [0, 2]
    assert designs[1].name == "group2"
    assert designs[1].channels_per_group == 3
    assert designs[1].slots[:3] == [1, 3, 5]


def test_group_designs_from_empty_groups_returns_default_group() -> None:
    designs = group_designs_from_groups([])

    assert len(designs) == 1
    assert designs[0].name == "group1"


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_dense_group_can_zoom_pan_edit_and_reset_without_changing_other_slots(app, monkeypatch):
    from pyneuroscope import channel_map_editor as module
    design = GroupDesign("g", 384, list(range(384)))
    tab = GroupTab(design, 384, {}, "G1")
    try:
        tab.setFont(QFont("Arial", 9))
        tab.resize(760, 680)
        tab.show()
        app.processEvents()
        v = tab.viewer
        v.grab()
        original_hits = dict(v._hits)
        x, y, _ = v._hits[190]
        anchor = QPointF(x, y)
        event = QWheelEvent(anchor, QPointF(v.mapToGlobal(anchor.toPoint())), QPoint(), QPoint(0, 1920),
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(v, event)
        assert v._zoom > 15
        v.grab()
        assert v._slot_at(x, y) == 190
        assert v._hits[191][1] - v._hits[190][1] > 12
        assert design.slots == list(range(384))
        QTest.mousePress(v, Qt.MouseButton.RightButton, pos=anchor.toPoint())
        QTest.mouseMove(v, (anchor + QPointF(25, -10)).toPoint())
        QTest.mouseRelease(v, Qt.MouseButton.RightButton, pos=(anchor + QPointF(25, -10)).toPoint())
        v.grab()
        cx, cy, _ = v._hits[190]
        edits = []
        def get_channel(*args):
            edits.append(args[3])
            return 383, True
        monkeypatch.setattr(module.QInputDialog, "getInt", get_channel)
        QTest.mouseClick(v, Qt.MouseButton.LeftButton, pos=QPointF(cx, cy).toPoint())
        assert edits == [190]
        assert design.slots[190] == 383
        assert design.slots[:190] == list(range(190))
        assert design.slots[191:] == list(range(191, 384))
        QTest.mouseClick(tab.reset_view, Qt.MouseButton.LeftButton)
        v.grab()
        assert v._zoom == 1 and v._pan == QPointF()
        assert v._hits == original_hits
        assert design.slots[190] == 383
    finally:
        tab.close()


def test_group_zoom_buttons_and_resize_restore_view(app):
    tab = GroupTab(GroupDesign("g", 4, [0, 1, 2, 3]), 4, {}, "G1")
    try:
        tab.zoom_in.click()
        assert tab.viewer._zoom == pytest.approx(1.2)
        tab.zoom_out.click()
        assert tab.viewer._zoom == pytest.approx(1)
        tab.viewer.zoom_by(10)
        tab.channels_per_group.setValue(8)
        assert tab.viewer._zoom == 1 and tab.viewer._pan == QPointF()
        assert tab.design.slots == [0, 1, 2, 3, None, None, None, None]
    finally:
        tab.close()
