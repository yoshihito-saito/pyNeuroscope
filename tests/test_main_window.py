import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from pathlib import Path
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication
from types import SimpleNamespace

from pyneuroscope.dat_reader import DatReaderError
from pyneuroscope.main_window import MainWindow
from pyneuroscope.models import ChannelGroup, SpikeUnit, SpikesData


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


def test_recording_discovery_rejects_explicit_basename_dat(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    selected = Path("session") / "basename.dat"
    monkeypatch.setattr(Path, "is_file", lambda self: self == selected)

    with pytest.raises(DatReaderError, match="Expected amplifier.dat"):
        window._resolve_recording_dat_paths(selected)


def test_recording_discovery_ignores_basename_dat_in_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    selected = Path("session")
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    monkeypatch.setattr(Path, "exists", lambda self: self == selected)
    monkeypatch.setattr(Path, "is_dir", lambda self: self == selected)
    monkeypatch.setattr(Path, "iterdir", lambda self: [self / "basename.dat"])

    with pytest.raises(DatReaderError, match="No amplifier.dat"):
        window._resolve_recording_dat_paths(selected)


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
