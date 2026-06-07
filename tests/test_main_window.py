import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from pathlib import Path
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QLabel, QPushButton
from scipy.io import savemat
from types import SimpleNamespace

from pyneuroscope.channel_profile_viewer import channel_rms
from pyneuroscope.channel_map_editor import group_designs_from_groups
from pyneuroscope.color_map import COLOR_MAP_NAMES, palette_from_name
from pyneuroscope.dat_reader import DatReaderError
from pyneuroscope.main_window import MainWindow
from pyneuroscope.models import ChannelGroup, EventSeries, SpikeUnit, SpikesData
from pyneuroscope.xml_builder import build_neurocode_xml


class WheelEvent:
    def __init__(self, delta: int, modifiers=Qt.KeyboardModifier.ControlModifier, x: int = 0) -> None:
        self._delta = delta
        self._modifiers = modifiers
        self._x = x
        self.accepted = False

    def modifiers(self):
        return self._modifiers

    def angleDelta(self) -> QPoint:  # noqa: N802
        return QPoint(0, self._delta)

    def position(self):
        class _Pos:
            def __init__(self, x: int) -> None:
                self._x = x

            def toPoint(self) -> QPoint:  # noqa: N802
                return QPoint(self._x, 0)

        return _Pos(self._x)

    def accept(self) -> None:
        self.accepted = True


class KeyEvent:
    def __init__(self, key: int, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
        self._key = key
        self._modifiers = modifiers

    def key(self) -> int:
        return self._key

    def modifiers(self):
        return self._modifiers


def test_top_bar_uses_folder_and_single_dat_buttons() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()

    button_labels = [button.text() for button in window.findChildren(QPushButton)]

    assert "Browse Folder" in button_labels
    assert "Open single DAT" in button_labels
    assert "Open" not in button_labels


def test_session_xml_buttons_use_requested_labels() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()

    button_labels = [button.text() for button in window.findChildren(QPushButton)]

    assert "Load Session XML" in button_labels
    assert "Load Session ChannelMap" in button_labels
    assert "Edit Channel Groups" in button_labels
    assert "Save Session XML" in button_labels
    assert "Save XML" not in button_labels


def test_session_xml_and_bad_channels_order() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    layout = window.recording_tab.layout()
    button_by_label = {
        button.text(): button
        for button in window.findChildren(QPushButton)
    }

    def top_level_index(target) -> int:
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item.widget() is target:
                return index
            child_layout = item.layout()
            if child_layout is not None and child_layout.indexOf(target) != -1:
                return index
        return -1

    load_index = top_level_index(button_by_label["Load Session XML"])
    chanmap_index = top_level_index(button_by_label["Load Session ChannelMap"])
    edit_index = top_level_index(button_by_label["Edit Channel Groups"])
    bad_channels_index = top_level_index(window.bad_channels_text)
    save_index = top_level_index(button_by_label["Save Session XML"])

    assert load_index < chanmap_index < edit_index < bad_channels_index < save_index


def test_browse_folder_commits_selected_recording_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    selected = tmp_path / "session"
    selected.mkdir()
    committed_paths: list[str] = []
    monkeypatch.setattr(
        "pyneuroscope.main_window.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(selected),
    )
    monkeypatch.setattr(window, "_dat_path_committed", lambda: committed_paths.append(window.dat_path.text()))

    window._browse_dat()

    assert window.dat_path.text() == str(selected)
    assert committed_paths == [str(selected)]


def test_open_single_dat_commits_selected_dat_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    selected = tmp_path / "amplifier.dat"
    selected.write_bytes(b"\0" * 16)
    committed_paths: list[str] = []
    monkeypatch.setattr(
        "pyneuroscope.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(selected), "DAT files (*.dat)"),
    )
    monkeypatch.setattr(window, "_dat_path_committed", lambda: committed_paths.append(window.dat_path.text()))

    window._browse_dat_file()

    assert window.dat_path.text() == str(selected)
    assert committed_paths == [str(selected)]


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


def test_ctrl_wheel_can_zoom_below_one_millisecond_at_high_sampling_rate() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.sampling_rate.setValue(20000.0)
    window._set_window_duration_seconds(0.001)

    event = WheelEvent(120)
    assert window._handle_trace_wheel(window.viewer, event)

    assert event.accepted
    assert window._window_duration_seconds() == pytest.approx(0.0008)
    assert window.duration_msec.value() == pytest.approx(0.8)


def test_zoom_time_window_clamps_to_one_sample_duration() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.sampling_rate.setValue(20000.0)
    window._set_window_duration_seconds(0.00006)

    window._zoom_time_window(0.8)

    assert window._window_duration_seconds() == pytest.approx(0.00005)
    assert window.duration_msec.value() == pytest.approx(0.05)


def test_csd_controls_default_to_bwr_and_update_viewer() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()

    assert window.csd_enabled.text() == "Current Source Density"
    assert window.csd_enabled.isCheckable()
    assert window.csd_cmap.currentText() == "bwr"
    assert [window.csd_cmap.itemText(i) for i in range(window.csd_cmap.count())] == [
        "bwr",
        "PiYG",
        "PRGn",
        "BrBG",
        "PuOr",
        "RdGy",
        "RdBu",
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
    ]

    window.csd_enabled.setChecked(True)
    window.csd_cmap.setCurrentText("RdBu")

    assert window.viewer._show_csd is True
    assert window.viewer._csd_colormap_name == "RdBu"


def test_channel_rms_uses_no_extra_filtering() -> None:
    data = np.asarray(
        [
            [3.0, 4.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(channel_rms(data, scale=2.0), [np.sqrt(4.5) * 2.0, np.sqrt(8.0) * 2.0])


def test_ch_profile_tab_calculates_current_window_rms(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window._current_data = np.asarray(
        [
            [3.0, 4.0, 0.0, 1.0],
            [0.0, 0.0, 8.0, 1.0],
        ],
        dtype=float,
    )
    monkeypatch.setattr(window, "_channel_profile_scale", lambda: (2.0, "uV"))

    window.channel_tabs.setCurrentWidget(window.channel_profile_viewer)

    assert window.channel_tabs.tabText(window.channel_tabs.indexOf(window.probe_viewer)) == "Ch map"
    assert window.channel_tabs.tabText(window.channel_tabs.indexOf(window.channel_profile_viewer)) == "Ch profile"
    np.testing.assert_allclose(
        window.channel_profile_viewer._rms,
        [np.sqrt(4.5) * 2.0, np.sqrt(8.0) * 2.0, np.sqrt(32.0) * 2.0, 2.0],
    )
    assert window.channel_profile_viewer._unit == "uV"


def test_duplicate_event_names_have_independent_controls(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    first = EventSeries(
        name="ripples",
        path=tmp_path / "session_a.ripples.events.mat",
        timestamps=np.asarray([[0.1, 0.2], [0.4, 0.5]], dtype=float),
        peaks=np.asarray([0.15, 0.45], dtype=float),
    )
    second = EventSeries(
        name="ripples",
        path=tmp_path / "session_b.ripples.events.mat",
        timestamps=np.asarray([[1.1, 1.2]], dtype=float),
        peaks=np.asarray([1.15], dtype=float),
    )
    window.event_series = [first, second]

    window._rebuild_event_controls()
    first_key = window._event_key(first)
    second_key = window._event_key(second)
    window.event_controls[first_key]["show"].setChecked(True)

    assert first_key in window.event_controls
    assert second_key in window.event_controls
    assert len(window.event_controls) == 2
    assert len(window.viewer._event_overlays) == 1
    assert window.viewer._event_overlays[0].timestamps.shape == (2, 2)
    assert "session_a.ripples.events.mat" in window.viewer._event_overlays[0].name


def test_event_name_click_selects_jump_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    first = EventSeries(
        name="ripples",
        path=tmp_path / "ripples.events.mat",
        timestamps=np.asarray([[1.0, 1.2]], dtype=float),
        peaks=None,
    )
    second = EventSeries(
        name="sharpwaves",
        path=tmp_path / "sharpwaves.events.mat",
        timestamps=np.asarray([[9.0, 9.2]], dtype=float),
        peaks=None,
    )
    monkeypatch.setattr(window, "_load_window", lambda silent=True: None)
    window.event_series = [first, second]
    window._rebuild_event_controls()
    first_key = window._event_key(first)
    second_key = window._event_key(second)
    window.event_controls[first_key]["show"].setChecked(True)
    window.event_controls[second_key]["show"].setChecked(True)

    window.event_controls[second_key]["name"].click()
    window.event_id_text.setText("1")
    window._jump_to_event_id()

    assert window._selected_event_key == second_key
    assert "#5a2528" in window.event_controls[second_key]["name"].styleSheet()
    assert "#5a2528" not in window.event_controls[first_key]["name"].styleSheet()
    assert window._window_start_seconds() == pytest.approx(8.6)


def test_event_arrows_jump_to_nearest_event_from_current_window(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    event = EventSeries(
        name="ripples",
        path=Path("ripples.events.mat"),
        timestamps=np.asarray([[1.0, 1.2], [5.0, 5.2], [9.0, 9.2]], dtype=float),
        peaks=None,
    )
    load_calls: list[bool] = []
    monkeypatch.setattr(window, "_load_window", lambda silent=True: load_calls.append(silent))
    window.event_series = [event]
    window._rebuild_event_controls()
    window.left_tabs.setCurrentWidget(window.events_tab)
    window._set_window_start_seconds(3.0)
    window._set_window_duration_seconds(2.0)

    assert window._handle_navigation_key(KeyEvent(Qt.Key.Key_Right))
    assert window.event_id_text.text() == "2"
    assert window._window_start_seconds() == pytest.approx(4.1)

    assert window._handle_navigation_key(KeyEvent(Qt.Key.Key_Left))
    assert window.event_id_text.text() == "1"
    assert window._window_start_seconds() == pytest.approx(0.1)
    assert load_calls


def test_event_arrows_first_snap_to_nearest_then_step(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    event = EventSeries(
        name="ripples",
        path=Path("ripples.events.mat"),
        timestamps=np.asarray([[1.0, 1.2], [5.0, 5.2], [9.0, 9.2]], dtype=float),
        peaks=None,
    )
    monkeypatch.setattr(window, "_load_window", lambda silent=True: None)
    window.event_series = [event]
    window._rebuild_event_controls()
    window.left_tabs.setCurrentWidget(window.events_tab)
    window._set_window_duration_seconds(2.0)

    window._set_window_start_seconds(4.4)
    assert window._handle_navigation_key(KeyEvent(Qt.Key.Key_Right))
    assert window.event_id_text.text() == "2"
    assert window._handle_navigation_key(KeyEvent(Qt.Key.Key_Right))
    assert window.event_id_text.text() == "3"

    window._set_window_start_seconds(3.8)
    assert window._handle_navigation_key(KeyEvent(Qt.Key.Key_Left))
    assert window.event_id_text.text() == "2"
    assert window._handle_navigation_key(KeyEvent(Qt.Key.Key_Left))
    assert window.event_id_text.text() == "1"


def test_event_arrows_keep_stepping_when_first_event_cannot_be_centered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    event = EventSeries(
        name="ripples",
        path=Path("ripples.events.mat"),
        timestamps=np.asarray([[0.04, 0.06], [0.20, 0.22], [0.40, 0.42]], dtype=float),
        peaks=None,
    )
    monkeypatch.setattr(window, "_load_window", lambda silent=True: None)
    window.event_series = [event]
    window._rebuild_event_controls()
    window.left_tabs.setCurrentWidget(window.events_tab)
    window._set_window_duration_seconds(1.0)

    window._jump_to_event_index(event, 0)
    assert window._window_start_seconds() == 0.0
    assert window._handle_navigation_key(KeyEvent(Qt.Key.Key_Right))
    assert window.event_id_text.text() == "2"
    assert window._handle_navigation_key(KeyEvent(Qt.Key.Key_Right))
    assert window.event_id_text.text() == "3"


def test_spike_cmap_colors_follow_display_order() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.groups = [ChannelGroup("upper", [2]), ChannelGroup("lower", [0])]
    window._reset_visible_groups()
    window.spikes_data = SpikesData(
        path=Path("spikes.cellinfo.mat"),
        basename="spikes",
        units=[
            SpikeUnit(10, "loaded first", np.asarray([0.1], dtype=float), channel=0),
            SpikeUnit(20, "displayed first", np.asarray([0.2], dtype=float), channel=2),
        ],
    )

    window.spikes_cmap.setCurrentText("spring")
    window.show_spikes.setChecked(True)
    window._refresh_spike_overlay()

    expected = palette_from_name("spring", 2)
    assert [overlay.unit_id for overlay in window.viewer._spike_overlays] == [20, 10]
    assert [overlay.color for overlay in window.viewer._spike_overlays] == expected


def test_recording_colormap_options_and_initial_channel_colors() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()

    assert [window.color_map.itemText(i) for i in range(window.color_map.count())] == COLOR_MAP_NAMES
    assert window.color_map.currentText() == "summer"
    expected = palette_from_name("summer", window.n_channels.value())
    assert [window.channel_colors[index] for index in range(window.n_channels.value())] == expected


def test_analysis_controls_live_in_analysis_tab() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()

    tab_labels = [window.left_tabs.tabText(index) for index in range(window.left_tabs.count())]
    assert tab_labels == ["Recording", "Spikes", "Events", "Analysis", "State editor"]
    assert window.analysis_tab.isAncestorOf(window.spectrogram_button)
    assert window.analysis_tab.isAncestorOf(window.csd_enabled)
    assert not window.recording_tab.isAncestorOf(window.spectrogram_button)
    assert not window.recording_tab.isAncestorOf(window.csd_enabled)


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

    window.channel_regions = {4: "CA1", 2: "PFC", 5: "CA1", 1: "CA1"}
    window.color_mode.setCurrentText("per region")
    window.region_cmap_controls["CA1"].setCurrentText("spring")
    window.region_cmap_controls["PFC"].setCurrentText("winter")
    window._reset_colors()
    assert window.channel_colors[4] == "#ff00ff"
    assert window.channel_colors[5] == "#ff8080"
    assert window.channel_colors[1] == "#ffff00"
    assert window.channel_colors[2] == "#0000ff"
    assert window.channel_colors[0] == "#808080"


def test_n_channels_change_expands_default_virtual_linear_probe() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()

    window.n_channels.setValue(8)

    assert window._group_source == "default"
    assert window.groups == [ChannelGroup("group1", list(range(8)))]
    assert window.group_designs[0].slots == list(range(8))


def test_n_channels_change_preserves_manual_channel_map() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window._group_source = "manual"
    window.groups = [ChannelGroup("manual", [3, 1])]
    window.group_designs = group_designs_from_groups(window.groups)

    window.n_channels.setValue(8)

    assert window.groups == [ChannelGroup("manual", [3, 1])]


def test_add_probe_creates_second_probe_and_merges_default_groups() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.probes = [window.probes[0].__class__(2)]
    window._refresh_probe_controls()
    window._apply_probe_configs_to_model()

    window._add_probe()

    assert len(window.probes) == 2
    assert window.n_channels.value() == 4
    assert [group.channels for group in window.groups] == [[0, 1], [2, 3]]


def test_remove_probe_button_removes_last_probe_but_keeps_one() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.probes = [window.probes[0].__class__(2)]
    window._refresh_probe_controls()
    window._apply_probe_configs_to_model()

    assert window.remove_probe_button.isEnabled() is False
    window._add_probe()
    assert window.remove_probe_button.isEnabled() is True

    window._remove_probe()

    assert len(window.probes) == 1
    assert window.n_channels.value() == 2
    assert [group.channels for group in window.groups] == [[0, 1]]
    assert window.remove_probe_button.isEnabled() is False

    window._remove_probe()

    assert len(window.probes) == 1


def test_recording_tab_orders_brain_regions_before_filters() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    layout = window.recording_tab.layout()
    brain_regions_index = layout.indexOf(window.brain_regions_section)
    filter_index = layout.indexOf(window.filter_panel)
    streaming_index = layout.indexOf(window.streaming_mode)
    filter_labels = [
        label.text()
        for label in window.filter_panel.findChildren(QLabel)
    ]

    assert brain_regions_index != -1
    assert filter_index != -1
    assert streaming_index != -1
    assert brain_regions_index < filter_index
    assert filter_index < streaming_index
    assert "Filters" in filter_labels


def test_probe_xml_loads_are_offset_and_merged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    first_xml = tmp_path / "probe1.xml"
    second_xml = tmp_path / "probe2.xml"
    first_xml.write_text(
        build_neurocode_xml(
            2,
            20000,
            1250,
            [ChannelGroup("g1", [1, 0])],
            {1},
        ),
        encoding="utf-8",
    )
    second_xml.write_text(
        build_neurocode_xml(
            3,
            20000,
            1250,
            [ChannelGroup("g1", [0, 2])],
            {2},
        ),
        encoding="utf-8",
    )
    selected_paths = iter([str(first_xml), str(second_xml)])
    monkeypatch.setattr(
        "pyneuroscope.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (next(selected_paths), "XML files (*.xml)"),
    )

    window._load_probe_xml(0)
    window._add_probe()
    window._load_probe_xml(1)

    assert window.n_channels.value() == 5
    assert window.total_n_channels_label.text() == "5"
    assert [group.channels for group in window.groups] == [[1, 0], [2, 4]]
    assert window.bad_channels == {1, 4}


def test_load_session_chanmap_button_sets_explicit_geometry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    mat_path = tmp_path / "custom_chanMap.mat"
    savemat(
        mat_path,
        {
            "chanMap0ind": [[1]],
            "xcoords": [[42]],
            "ycoords": [[-12]],
        },
    )
    monkeypatch.setattr(
        "pyneuroscope.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(mat_path), "MAT files (*.mat)"),
    )

    window._load_session_chanmap()

    assert window._chanmap_geometry_path == mat_path
    assert window.probes[0].probe_type == ""
    assert window._probe_channel_geometry()[1].x == 42


def test_recording_discovery_accepts_explicit_dat_file(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    selected = Path("session") / "basename.dat"
    monkeypatch.setattr(Path, "is_file", lambda self: self == selected)

    assert window._resolve_recording_dat_paths(selected) == [selected]


def test_recording_discovery_uses_direct_basename_dat(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    selected = tmp_path / "session"
    selected.mkdir()
    basename_dat = selected / "session.dat"
    basename_dat.write_bytes(b"\0" * 16)

    assert window._resolve_recording_dat_paths(selected) == [basename_dat]


def test_recording_discovery_checks_only_one_folder_level(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    first = tmp_path / "pisco_linear_track_260514_142413"
    second = tmp_path / "pisco_postsleep_260514_144740"
    nested = second / "original_dat"
    first.mkdir()
    second.mkdir()
    nested.mkdir()
    first_dat = first / "amplifier.dat"
    second_dat = second / "amplifier.dat"
    nested_dat = nested / "amplifier.dat"
    first_dat.write_bytes(b"\0" * 16)
    second_dat.write_bytes(b"\0" * 16)
    nested_dat.write_bytes(b"\0" * 16)

    assert window._resolve_recording_dat_paths(tmp_path) == [first_dat, second_dat]


def test_recording_discovery_prefers_open_ephys_continuous_dat_over_basename_dat(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    basename_dat = tmp_path / f"{tmp_path.name}.dat"
    basename_dat.write_bytes(b"\0" * 16)
    first = tmp_path / "2026-05-20_12-00-00" / "Record Node 101" / "experiment1" / "recording1" / "continuous" / "Acquisition_Board-102.acquisition_board"
    second = tmp_path / "2026-05-20_12-30-00" / "Record Node 101" / "experiment1" / "recording1" / "continuous" / "Acquisition_Board-102.acquisition_board"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    first_dat = first / "continuous.dat"
    second_dat = second / "continuous.dat"
    first_dat.write_bytes(b"\0" * 16)
    second_dat.write_bytes(b"\0" * 16)

    assert window._resolve_recording_dat_paths(tmp_path) == [first_dat, second_dat]


def test_adjacent_xml_prefers_selected_folder_basename_xml_for_open_ephys(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    selected = tmp_path / "day17"
    dat_folder = selected / "2026-05-20_12-32-59" / "Record Node 101" / "experiment1" / "recording1" / "continuous" / "Acquisition_Board-102.acquisition_board"
    dat_folder.mkdir(parents=True)
    dat_path = dat_folder / "continuous.dat"
    dat_path.write_bytes(b"\0" * 16)
    base_xml = selected / "day17.xml"
    continuous_xml = dat_folder / "continuous.xml"
    base_xml.write_text("<parameters />", encoding="utf-8")
    continuous_xml.write_text("<parameters />", encoding="utf-8")
    window.dat_path.blockSignals(True)
    window.dat_path.setText(str(selected))
    window.dat_path.blockSignals(False)

    assert window._resolve_adjacent_xml_path(dat_path) == base_xml


def test_dat_path_commit_loads_xml_before_inspecting_dat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    dat_path = tmp_path / "continuous.dat"
    dat_path.write_bytes(b"\0" * 16)
    calls: list[str] = []
    window.dat_path.blockSignals(True)
    window.dat_path.setText(str(tmp_path))
    window.dat_path.blockSignals(False)
    monkeypatch.setattr(window, "_resolve_recording_dat_paths", lambda path: [dat_path])
    monkeypatch.setattr(window, "_load_adjacent_xml_if_present", lambda path: calls.append("xml"))
    monkeypatch.setattr(window, "_recording_dat_infos", lambda: calls.append("inspect") or [SimpleNamespace(duration_seconds=1.0)])
    monkeypatch.setattr(window, "_load_adjacent_anatomical_map_if_present", lambda path: None)
    monkeypatch.setattr(window, "_load_adjacent_spikes_and_events", lambda: None)
    monkeypatch.setattr(window, "_load_window", lambda silent=True: None)

    window._dat_path_committed()

    assert calls[:2] == ["xml", "inspect"]


def test_parent_anatomical_map_is_auto_resolved(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    session = tmp_path / "day"
    subsession = session / "rec_260101_120000"
    subsession.mkdir(parents=True)
    dat_path = subsession / "amplifier.dat"
    dat_path.write_bytes(b"\0" * 16)
    (session / "anatomical_map.csv").write_text("CA1\n", encoding="utf-8")
    window.dat_path.blockSignals(True)
    window.dat_path.setText(str(session))
    window.dat_path.blockSignals(False)
    window.groups = [ChannelGroup("group1", [0])]

    window._load_adjacent_anatomical_map_if_present(dat_path)

    assert window.channel_regions == {0: "CA1"}


def test_screenshot_filename_helpers() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()

    assert window._duration_slug(65.432) == "1min-5sec-432ms"
    assert window._screenshot_path_with_suffix(Path("shot.pdf"), "PNG image (*.png)") == Path("shot.png")
    assert window._screenshot_path_with_suffix(Path("shot"), "PNG image (*.png)") == Path("shot.png")


def test_screenshot_default_directory_uses_recording_path(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.dat_path.blockSignals(True)
    window.dat_path.setText(str(tmp_path))
    window.dat_path.blockSignals(False)

    assert window._default_screenshot_path().parent == tmp_path


def test_spectrogram_button_opens_current_recording_window(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    captured: dict[str, object] = {}

    class FakeSpectrogramDialog:
        def __init__(
            self,
            time,
            data,
            *,
            sampling_rate,
            window_start_seconds,
            window_duration_seconds,
            total_duration_seconds,
            window_loader,
            window_changed_callback,
            streaming,
            parent=None,
        ) -> None:
            captured["time"] = time
            captured["data"] = data
            captured["sampling_rate"] = sampling_rate
            captured["window_start_seconds"] = window_start_seconds
            captured["window_duration_seconds"] = window_duration_seconds
            captured["total_duration_seconds"] = total_duration_seconds
            captured["window_loader"] = window_loader
            captured["window_changed_callback"] = window_changed_callback
            captured["streaming"] = streaming
            captured["parent"] = parent

        def show(self) -> None:
            captured["shown"] = True

        def raise_(self) -> None:
            captured["raised"] = True

        def activateWindow(self) -> None:  # noqa: N802
            captured["activated"] = True

    monkeypatch.setattr("pyneuroscope.main_window.ChannelSpectrogramDialog", FakeSpectrogramDialog)
    window._current_time = np.asarray([0.0, 0.5, 1.0])
    window._current_data = np.asarray([[0, 1], [2, 3], [4, 5]], dtype=float)
    window.sampling_rate.setValue(1000.0)

    window._show_spectrogram_window()

    assert captured["sampling_rate"] == 1000.0
    assert captured["window_start_seconds"] == window._window_start_seconds()
    assert captured["window_duration_seconds"] == window._window_duration_seconds()
    assert callable(captured["window_loader"])
    assert captured["window_changed_callback"] == window._apply_spectrogram_window_to_recording
    assert captured["streaming"] is False
    assert np.asarray(captured["data"]).shape == (3, 2)
    assert captured["shown"] is True
    assert window.spectrogram_button.text() == "Spectrogram"


def test_hidden_probe_group_hides_detected_spikes() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.groups = [ChannelGroup("g1", [0]), ChannelGroup("g2", [1])]
    window.visible_groups = {0}
    window.spikes_data = SpikesData(
        path=Path("spikes.mat"),
        basename="test",
        units=[
            SpikeUnit(1, "u1", np.asarray([0.1]), channel=0),
            SpikeUnit(2, "u2", np.asarray([0.2]), channel=1),
        ],
    )
    window.show_spikes.setChecked(True)

    window._refresh_spike_overlay()

    assert [unit.unit_id for unit in window.viewer._spike_overlays] == [1]


def test_spike_per_region_keeps_probe_group_display_order() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.groups = [ChannelGroup("PFC", [10]), ChannelGroup("CA1", [1])]
    window.visible_groups = {0, 1}
    window.channel_regions = {10: "PFC", 1: "CA1"}
    window.spikes_data = SpikesData(
        path=Path("spikes.mat"),
        basename="test",
        units=[
            SpikeUnit(1, "ca1", np.asarray([0.1]), channel=1),
            SpikeUnit(2, "pfc", np.asarray([0.2]), channel=10),
        ],
    )
    window.show_spikes.setChecked(True)
    window.spikes_per_region.setChecked(True)

    window._refresh_spike_overlay()

    assert [unit.unit_id for unit in window.viewer._spike_overlays] == [2, 1]


def test_shift_wheel_changes_row_spacing() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    start = window.spacing.value()

    event = WheelEvent(120, Qt.KeyboardModifier.ShiftModifier)
    assert window._handle_trace_wheel(window.viewer, event)
    assert event.accepted
    assert window.spacing.value() > start


def test_sleep_viewer_wheel_zooms_time_window_without_modifiers() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.left_tabs.setCurrentIndex(window.left_tabs.indexOf(window.sleep_tab))
    window.sleep_state_data = {
        "idx": {
            "timestamps": np.asarray([0.0, 10.0, 20.0], dtype=float),
            "states": np.asarray([1, 3, 5], dtype=float),
        }
    }
    window._set_window_start_seconds(2.0)
    window._set_window_duration_seconds(10.0)

    plot_left, plot_right = window.sleep_viewer._plot_geometry()
    x_mid = (plot_left + plot_right) // 2
    event = WheelEvent(120, Qt.KeyboardModifier.NoModifier, x=x_mid)
    assert window._handle_trace_wheel(window.sleep_viewer, event)
    assert event.accepted
    assert window._window_duration_seconds() == 8.0
    assert window._window_start_seconds() == 3.0


def test_sleep_transition_row_extracts_transition_timing() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.sleep_viewer.set_data(
        state_timestamps=np.asarray([0.0, 10.0, 20.0, 30.0], dtype=float),
        metric_timestamps=np.asarray([], dtype=float),
        states=np.asarray([1.0, 1.0, 3.0, 5.0], dtype=float),
        sw=np.asarray([], dtype=float),
        emg=np.asarray([], dtype=float),
        thratio=np.asarray([], dtype=float),
        sw_threshold=None,
        emg_threshold=None,
        thratio_threshold=None,
        spec=np.empty((0, 0), dtype=float),
        freqs=np.asarray([], dtype=float),
        spec_timestamps=np.asarray([], dtype=float),
    )

    assert window.sleep_viewer._transition_times(0.0, 30.0) == [20.0, 30.0]
    assert window.sleep_viewer._transition_times(0.0, 25.0) == [20.0]


def test_sleep_state_click_selects_contiguous_episode() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    selected: list[tuple[float, float]] = []
    window.sleep_viewer.set_selection_callback(lambda lo, hi: selected.append((lo, hi)))
    window.sleep_viewer.set_data(
        state_timestamps=np.asarray([0.0, 10.0, 20.0, 30.0], dtype=float),
        metric_timestamps=np.asarray([], dtype=float),
        states=np.asarray([1.0, 3.0, 3.0, 5.0], dtype=float),
        sw=np.asarray([], dtype=float),
        emg=np.asarray([], dtype=float),
        thratio=np.asarray([], dtype=float),
        sw_threshold=None,
        emg_threshold=None,
        thratio_threshold=None,
        spec=np.empty((0, 0), dtype=float),
        freqs=np.asarray([], dtype=float),
        spec_timestamps=np.asarray([], dtype=float),
    )
    window.sleep_viewer.set_window(0.0, 40.0)

    _, rows = window.sleep_viewer._layout_rows()
    nrem_lane = window.sleep_viewer._state_lane_rects(rows[0])[1][1]
    x = window.sleep_viewer._x_at_time(15.0, 0.0, 40.0)

    assert window.sleep_viewer._select_state_episode_at(QPoint(x, nrem_lane.center().y()))
    assert selected[-1] == (10.0, 30.0)


def test_region_summary_reports_assigned_counts() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.channel_regions = {0: "CA1", 1: "CA1", 2: "CTX"}

    window._refresh_region_summary()

    assert window.region_summary.text() == "Assigned channels: CA1: 2, CTX: 1"


def test_arrow_navigation_uses_quarter_window_step() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window._set_window_start_seconds(10.0)
    window._set_window_duration_seconds(8.0)

    assert window._handle_navigation_key(KeyEvent(Qt.Key.Key_Right))
    assert window._window_start_seconds() == 12.0

    assert window._handle_navigation_key(KeyEvent(Qt.Key.Key_Left))
    assert window._window_start_seconds() == 10.0


def test_group_visibility_filters_displayed_groups() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.groups = [ChannelGroup("g1", [0, 1]), ChannelGroup("g2", [2, 3])]
    window._reset_visible_groups()

    window._toggle_group_visibility(1)

    assert window.visible_groups == {0}
    assert [group.channels for group in window._visible_groups()] == [[0, 1]]


def test_sleep_state_duration_uses_full_timestamp_span() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.sleep_state_data = {
        "idx": {
            "timestamps": np.asarray([5.0, 15.0, 35.0], dtype=float),
            "states": np.asarray([1, 3, 5], dtype=float),
        }
    }

    assert window._sleep_state_duration_seconds() == 30.0


def test_reset_sleep_view_window_restores_full_state_span() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.sleep_state_data = {
        "idx": {
            "timestamps": np.asarray([5.0, 15.0, 35.0], dtype=float),
            "states": np.asarray([1, 3, 5], dtype=float),
        }
    }
    window.sleep_selection_range = (10.0, 15.0)
    window.sleep_pending_edit = (10.0, 15.0, 3)

    window._reset_sleep_view_window()

    assert window.sleep_selection_range is None
    assert window.sleep_pending_edit is None
    assert window._window_start_seconds() == 5.0
    assert window._window_duration_seconds() == 30.0


def test_spectrogram_target_count_is_capped_for_lightweight_rendering() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()

    assert window.sleep_viewer._target_point_count(10_000, 1200, 1200.0, kind="spectrogram") == 256
    assert window.sleep_viewer._target_point_count(10_000, 90, 1200.0, kind="spectrogram") == 72


def test_sleep_tab_dat_commit_without_adjacent_state_clears_sleep_context(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    dat_path = Path("recording.dat")
    window.sleep_state_data = {"idx": {"timestamps": np.asarray([0.0, 1.0]), "states": np.asarray([1, 3])}}
    window.sleep_state_path = Path("old.SleepState.states.mat")
    window.left_tabs.setCurrentIndex(window.left_tabs.indexOf(window.sleep_tab))
    window.dat_path.setText(str(dat_path))
    monkeypatch.setattr(window, "_resolve_recording_dat_paths", lambda path: [path])
    monkeypatch.setattr(window, "_recording_dat_infos", lambda: [SimpleNamespace(duration_seconds=1.0)])
    monkeypatch.setattr(window, "_load_adjacent_xml_if_present", lambda path: None)
    monkeypatch.setattr(window, "_load_adjacent_anatomical_map_if_present", lambda path: None)
    monkeypatch.setattr(window, "_refresh_duration", lambda: None)
    monkeypatch.setattr(window, "_sync_time_scroll", lambda value: None)

    window._dat_path_committed()

    assert window.sleep_state_data is None
    assert window.sleep_state_path is None


def test_sleep_tab_exposes_manual_edit_controls() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()

    assert not window.sleep_show_transitions.isChecked()
    assert window.sleep_show_transitions.text() == "State transition timing"
    assert window.sleep_spectrogram_cmap.currentText() == "Viridis"
    assert [window.sleep_spectrogram_cmap.itemText(i) for i in range(window.sleep_spectrogram_cmap.count())] == [
        "Viridis",
        "Magma",
        "Mako",
        "Inferno",
        "Jet",
    ]
    assert not window.sleep_manual_state.isHidden()
    assert not window.sleep_manual_selection.isHidden()
    assert not window.sleep_modify_button.isHidden()
    assert not window.sleep_update_button.isHidden()


def test_update_sleep_state_file_overwrites_selected_state_range(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    saved_payloads: list[tuple[Path, dict]] = []
    episode_calls: list[tuple[Path, str]] = []
    sleep_state_path = Path("recording.SleepState.states.mat")
    window.sleep_state_path = sleep_state_path
    window.sleep_state_data = {
        "idx": {
            "timestamps": np.asarray([[0.0], [10.0], [20.0], [30.0]], dtype=float),
            "states": np.asarray([[1], [1], [5], [5]], dtype=np.uint8),
            "statenames": np.asarray([["WAKE", "", "NREM", "", "REM"]], dtype=object),
        },
        "ints": {
            "WAKEstate": np.asarray([[0.0, 10.0]], dtype=float),
            "NREMstate": np.empty((0, 2), dtype=float),
            "REMstate": np.asarray([[20.0, 30.0]], dtype=float),
        },
        "detectorinfo": {
            "detectionparms": {
                "SleepScoreMetrics": {
                    "thratio": np.asarray([[0.0], [0.0], [2.0], [2.0]], dtype=float),
                    "EMG": np.asarray([[0.0], [0.0], [2.0], [2.0]], dtype=float),
                    "t_clus": np.asarray([[0.0], [10.0], [20.0], [30.0]], dtype=float),
                    "histsandthreshs": {"THthresh": 1.0, "EMGthresh": 1.0},
                }
            }
        },
    }
    monkeypatch.setattr(
        "pyneuroscope.main_window.append_theta_epochs",
        lambda state, parent, basename: (state, parent / f"{basename}.SleepState.states.mat"),
    )
    monkeypatch.setattr(
        "pyneuroscope.main_window.states_to_episodes",
        lambda state, parent, basename: episode_calls.append((parent, basename)),
    )
    monkeypatch.setattr(
        "pyneuroscope.main_window.savemat",
        lambda path, payload, do_compression=True: saved_payloads.append((path, payload)),
    )

    window.sleep_selection_range = (10.0, 20.0)
    window.sleep_manual_state.setCurrentText("NREM")

    window._modify_sleep_state_selection()
    window._update_sleep_state_file()

    updated_states = np.asarray(window.sleep_state_data["idx"]["states"]).reshape(-1)
    assert updated_states.tolist() == [1, 3, 3, 5]
    assert saved_payloads
    assert saved_payloads[-1][0] == sleep_state_path
    assert np.asarray(saved_payloads[-1][1]["SleepState"]["idx"]["states"]).reshape(-1).tolist() == [1, 3, 3, 5]
    assert episode_calls == [(Path("."), "recording")]
