import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from pyneuroscope.brain_region_editor import BrainRegionProbeWidget
from pyneuroscope.models import ChannelGroup


def test_assign_region_updates_all_selected_channels() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    widget = BrainRegionProbeWidget(
        [ChannelGroup("g1", [0, 1, 2]), ChannelGroup("g2", [3, 4, 5])],
        {},
    )
    widget._selected_channels = {1, 4}

    widget.assign_region("CA1")

    assert widget.channel_regions == {1: "CA1", 4: "CA1"}


def test_clear_region_removes_selected_assignments() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    widget = BrainRegionProbeWidget(
        [ChannelGroup("g1", [0, 1, 2])],
        {0: "CTX", 1: "CA1"},
    )
    widget._selected_channels = {1}

    widget.clear_region()

    assert widget.channel_regions == {0: "CTX"}
