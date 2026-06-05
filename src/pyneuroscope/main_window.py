from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from importlib.resources import files
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
import numpy as np
from scipy.io import loadmat, savemat

from .anatomical_map import AnatomicalMapError, build_anatomical_map_csv, load_anatomical_map_csv
from .brain_region_editor import BrainRegionEditorDialog
from .channel_profile_viewer import ChannelProfileViewer, channel_rms
from .color_map import (
    COLOR_MAP_NAMES,
    ColorMapError,
    color_by_group_channel_index,
    color_by_group_sequence,
    palette_from_name,
)
from .channel_map_editor import ChannelMapDialog, GroupDesign, group_designs_from_groups
from .csd import CSD_COLORMAPS
from .dat_reader import DatReaderError, inspect_dat, read_dat_window
from .event_io import EventLoadError, candidate_analysis_dirs, find_event_files, find_spikes_file, load_event_file, load_spikes_cellinfo
from .models import ChannelGroup, EventSeries, RecordingMetadata, SignalEventOverlay, SignalSpikeOverlay, SpikesData
from .probe_viewer import ProbeViewer
from .recording_overview import RecordingOverviewWidget
from .signal_layout import group_column_layout, single_column_layout
from .sleep_state_edit import append_theta_epochs, idx_to_intervals, states_to_episodes
from .sleep_state_viewer import SPECTROGRAM_COLORMAPS, SleepStateViewer
from .signal_viewer import SignalViewer
from .signal_filters import SignalFilterError, bandpass_filter, common_average_reference
from .spectrogram_viewer import ChannelSpectrogramDialog
from .validation import validate_settings
from .xml_builder import XmlError, build_neurocode_xml, parse_neurosuite_xml


@dataclass
class ProbeConfig:
    n_channels: int
    xml_path: Path | None = None
    groups: list[ChannelGroup] | None = None
    bad_channels: set[int] = field(default_factory=set)


class MainWindow(QMainWindow):
    def __init__(self, initial_path: str | Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("pyNeuroscope")
        self.setWindowIcon(QIcon(str(files("pyneuroscope.resources").joinpath("logo.ico"))))
        self.resize(1280, 820)
        self.groups: list[ChannelGroup] = []
        self.group_designs: list[GroupDesign] = []
        self.probes: list[ProbeConfig] = [ProbeConfig(4)]
        self.bad_channels: set[int] = set()
        self.visible_groups: set[int] = set()
        self.channel_colors: dict[int, str] = {}
        self.channel_regions: dict[int, str] = {}
        self.region_cmap_controls: dict[str, QComboBox] = {}
        self.loaded_metadata = RecordingMetadata()
        self._group_source = "default"
        self._recording_dat_paths: list[Path] = []
        self._recording_epoch_segments: list[tuple[str, float, float]] = []
        self._recording_epoch_boundaries = np.asarray([], dtype=float)
        self.row_spacing = 1.0
        self.record_window_start_seconds = 0.0
        self.record_window_duration_seconds = 1.0
        self.sleep_window_start_seconds = 0.0
        self.sleep_window_duration_seconds = 20.0 * 60.0
        self._active_left_tab_index = 0
        self._updating_time_scroll = False
        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self._stream_latest_window)
        self.sleep_state_data: dict | None = None
        self.sleep_state_path: Path | None = None
        self.sleep_selection_range: tuple[float, float] | None = None
        self.sleep_selection_patches: list = []
        self.sleep_span_selectors: list = []
        self.sleep_pending_edit: tuple[float, float, int] | None = None
        self.spikes_data: SpikesData | None = None
        self.event_series: list[EventSeries] = []
        self.event_controls: dict[str, dict[str, QWidget]] = {}
        self.event_display_names: dict[str, str] = {}
        self._selected_event_key: str | None = None
        self._event_navigation_anchor: tuple[str, int] | None = None
        self._setting_event_navigation_window = False
        self.spike_group_cmaps: dict[str, QComboBox] = {}
        self._build_ui()
        QApplication.instance().installEventFilter(self)
        self._generate_groups()
        if initial_path is not None:
            self.dat_path.setText(str(initial_path))
            QTimer.singleShot(0, self._dat_path_committed)
        self._refresh_all()

    def _build_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self._build_top_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 9)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([330, 1005, 260])
        layout.addWidget(splitter, 1)
        self.setCentralWidget(container)

    def _build_top_bar(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.dat_path = QLineEdit()
        self.dat_path.setMinimumWidth(260)
        self.dat_path.setMaximumWidth(720)
        self.dat_path.setPlaceholderText("Select .dat file, basepath, or session folder")
        browse_folder = QPushButton("Browse Folder")
        browse_folder.clicked.connect(self._browse_dat)
        open_dat = QPushButton("Open single DAT")
        open_dat.clicked.connect(self._browse_dat_file)
        layout.addWidget(browse_folder)
        layout.addWidget(open_dat)
        layout.addWidget(QLabel("Recording path"))
        layout.addWidget(self.dat_path, 1)
        layout.addStretch(2)
        return panel

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(360)
        panel.setMinimumWidth(320)
        layout = QVBoxLayout(panel)
        self.left_tabs = QTabWidget()
        self.left_tabs.setUsesScrollButtons(True)
        self.left_tabs.tabBar().setElideMode(Qt.TextElideMode.ElideNone)
        self.left_tabs.tabBar().setExpanding(False)
        self.left_tabs.tabBar().setStyleSheet("QTabBar::tab { padding: 5px 8px; font-size: 12px; }")
        self.recording_tab = self._build_recording_tab()
        self.spikes_tab = self._build_spikes_tab()
        self.events_tab = self._build_events_tab()
        self.analysis_tab = self._build_analysis_tab()
        self.sleep_tab = self._build_sleep_scoring_tab()
        self.left_tabs.addTab(self.recording_tab, "Recording")
        self.left_tabs.addTab(self.spikes_tab, "Spikes")
        self.left_tabs.addTab(self.events_tab, "Events")
        self.left_tabs.addTab(self.analysis_tab, "Analysis")
        self.left_tabs.addTab(self.sleep_tab, "State editor")
        self.left_tabs.currentChanged.connect(self._left_tab_changed)
        layout.addWidget(self.left_tabs, 1)
        return panel

    def _build_recording_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        form = QFormLayout()

        self.n_channels = QSpinBox()
        self.n_channels.setRange(1, 4096)
        self.n_channels.setValue(4)
        self.sampling_rate = QDoubleSpinBox()
        self.sampling_rate.setRange(1, 1000000)
        self.sampling_rate.setDecimals(3)
        self.sampling_rate.setValue(20000)
        self.lfp_sampling_rate = QDoubleSpinBox()
        self.lfp_sampling_rate.setRange(1, 1000000)
        self.lfp_sampling_rate.setDecimals(3)
        self.lfp_sampling_rate.setValue(1250)
        self.total_n_channels_label = QLabel("4")
        self.duration_label = QLabel("-")
        self.start_minutes = QSpinBox()
        self.start_minutes.setRange(0, 10**7)
        self.start_seconds = QSpinBox()
        self.start_seconds.setRange(0, 59)
        self.start_msec = QDoubleSpinBox()
        self.start_msec.setRange(0.0, 999.999)
        self.start_msec.setDecimals(3)
        self.start_msec.setSingleStep(0.1)
        self.duration_minutes = QSpinBox()
        self.duration_minutes.setRange(0, 10**7)
        self.duration_seconds = QSpinBox()
        self.duration_seconds.setRange(0, 59)
        self.duration_msec = QDoubleSpinBox()
        self.duration_msec.setRange(0.0, 999.999)
        self.duration_msec.setDecimals(3)
        self.duration_msec.setSingleStep(0.1)
        self.duration_seconds.setValue(1)
        self.bad_channels_text = QLineEdit()
        self.bad_channels_text.setPlaceholderText("e.g. 1, 12, 34")
        self.ignore_bad_channels = QCheckBox("Ignore bad ch")
        self.ignore_bad_channels.setChecked(False)
        self.streaming_mode = QCheckBox("Streaming mode")
        self.bandpass_enabled = QCheckBox("Bandpass")
        self.bandpass_low = QDoubleSpinBox()
        self.bandpass_low.setRange(0.1, 1000000)
        self.bandpass_low.setDecimals(1)
        self.bandpass_low.setMinimumWidth(78)
        self.bandpass_low.setValue(500.0)
        self.bandpass_high = QDoubleSpinBox()
        self.bandpass_high.setRange(0.1, 1000000)
        self.bandpass_high.setDecimals(1)
        self.bandpass_high.setMinimumWidth(78)
        self.bandpass_high.setValue(6000.0)
        self.car_enabled = QCheckBox("Common average")
        self.car_mode = QComboBox()
        self.car_mode.addItems(["all", "per group"])

        form.addRow("Total nChannels", self.total_n_channels_label)
        form.addRow("samplingRate", self.sampling_rate)
        form.addRow("lfpSamplingRate", self.lfp_sampling_rate)
        form.addRow("Duration", self.duration_label)


        for widget in [
            self.n_channels,
            self.sampling_rate,
            self.lfp_sampling_rate,
        ]:
            widget.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.n_channels.valueChanged.connect(self._n_channels_changed)
        for widget in [
            self.sampling_rate,
            self.lfp_sampling_rate,
        ]:
            widget.valueChanged.connect(self._refresh_all)
        for widget in [
            self.start_minutes,
            self.start_seconds,
            self.start_msec,
            self.duration_minutes,
            self.duration_seconds,
            self.duration_msec,
            self.bandpass_low,
            self.bandpass_high,
        ]:
            widget.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
            widget.valueChanged.connect(self._time_controls_changed)
            widget.lineEdit().returnPressed.connect(self._apply_window_controls)

        self.dat_path.textChanged.connect(self._dat_path_text_changed)
        self.dat_path.editingFinished.connect(self._dat_path_committed)
        self.bad_channels_text.textChanged.connect(self._bad_channels_changed)
        self.ignore_bad_channels.toggled.connect(self._refresh_viewer_layout)
        self.streaming_mode.toggled.connect(self._set_streaming_enabled)
        self.bandpass_enabled.toggled.connect(self._bandpass_toggled)
        self.bandpass_low.valueChanged.connect(self._reprocess_current_window)
        self.bandpass_high.valueChanged.connect(self._reprocess_current_window)
        self.car_enabled.toggled.connect(self._reprocess_current_window)
        self.car_mode.currentTextChanged.connect(self._reprocess_current_window)
        load_xml_design = QPushButton("Load Session XML")
        load_xml_design.clicked.connect(self._load_xml)
        edit_map = QPushButton("Edit Channel Groups")
        edit_map.clicked.connect(self._edit_channel_map)
        save_xml = QPushButton("Save Session XML")
        save_xml.clicked.connect(self._save_xml)
        self.spectrogram_button = QPushButton("Spectrogram")
        self.spectrogram_button.clicked.connect(self._show_spectrogram_window)
        self.csd_enabled = QPushButton("Current Source Density")
        self.csd_enabled.setCheckable(True)
        self.csd_enabled.setToolTip("Overlay relative current source density on the recording traces")
        self.csd_enabled.toggled.connect(self._refresh_viewer_layout)
        self.csd_cmap = QComboBox()
        self.csd_cmap.addItems(list(CSD_COLORMAPS))
        self.csd_cmap.setCurrentText("bwr")
        self.csd_cmap.currentTextChanged.connect(self._refresh_viewer_layout)
        shortcuts = QPushButton("Keyboard Shortcuts")
        shortcuts.clicked.connect(self._show_shortcuts)

        layout.addLayout(form)
        layout.addWidget(self._build_probe_section())
        layout.addWidget(load_xml_design)
        layout.addWidget(edit_map)
        bad_form = QFormLayout()
        bad_form.addRow("Bad channels", self.bad_channels_text)
        layout.addLayout(bad_form)
        layout.addWidget(save_xml)
        layout.addWidget(self.ignore_bad_channels)
        self.brain_regions_section = self._build_brain_regions_section()
        self.filter_panel = self._build_filter_panel()
        layout.addWidget(self.brain_regions_section)
        layout.addWidget(self.filter_panel)
        layout.addStretch(1)
        layout.addWidget(self.streaming_mode)
        layout.addWidget(shortcuts)
        return panel

    def _build_probe_section(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(4)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Probes")
        title.setStyleSheet("font-weight: 600;")
        add_probe = QPushButton("+")
        add_probe.setFixedWidth(32)
        add_probe.clicked.connect(self._add_probe)
        self.remove_probe_button = QPushButton("-")
        self.remove_probe_button.setFixedWidth(32)
        self.remove_probe_button.clicked.connect(self._remove_probe)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.remove_probe_button)
        header.addWidget(add_probe)
        self.probe_rows = QWidget()
        self.probe_rows_layout = QVBoxLayout(self.probe_rows)
        self.probe_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.probe_rows_layout.setSpacing(6)
        layout.addLayout(header)
        layout.addWidget(self.probe_rows)
        self._refresh_probe_controls()
        return panel

    def _refresh_probe_controls(self) -> None:
        if not hasattr(self, "probe_rows_layout"):
            return
        while self.probe_rows_layout.count():
            item = self.probe_rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, probe in enumerate(self.probes):
            row_panel = QWidget()
            row_layout = QVBoxLayout(row_panel)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(3)
            title = QLabel(f"Probe {index + 1}")
            title.setStyleSheet("font-weight: 600; color: #d6dde8;")
            controls = QHBoxLayout()
            controls.setContentsMargins(0, 0, 0, 0)
            channels = QSpinBox()
            channels.setRange(1, 4096)
            channels.setValue(probe.n_channels)
            channels.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
            channels.valueChanged.connect(lambda value, probe_index=index: self._probe_n_channels_changed(probe_index, value))
            load_xml = QPushButton("Probe XML")
            load_xml.clicked.connect(lambda checked=False, probe_index=index: self._load_probe_xml(probe_index))
            controls.addWidget(QLabel("nChannels"))
            controls.addWidget(channels)
            controls.addWidget(load_xml, 1)
            row_layout.addWidget(title)
            row_layout.addLayout(controls)
            if probe.xml_path is not None:
                loaded = QLabel(probe.xml_path.name)
                loaded.setWordWrap(True)
                row_layout.addWidget(loaded)
            self.probe_rows_layout.addWidget(row_panel)
        self._update_total_n_channels_label()
        if hasattr(self, "remove_probe_button"):
            self.remove_probe_button.setEnabled(len(self.probes) > 1)

    def _add_probe(self) -> None:
        self.probes.append(ProbeConfig(self._default_probe_n_channels()))
        self._refresh_probe_controls()
        self._apply_probe_configs_to_model()

    def _remove_probe(self) -> None:
        if len(self.probes) <= 1:
            self.statusBar().showMessage("At least one probe must remain", 2500)
            return
        self.probes.pop()
        self._refresh_probe_controls()
        self._apply_probe_configs_to_model()

    def _default_probe_n_channels(self) -> int:
        return self.probes[-1].n_channels if self.probes else max(1, self.n_channels.value())

    def _probe_n_channels_changed(self, probe_index: int, value: int) -> None:
        if not (0 <= probe_index < len(self.probes)):
            return
        probe = self.probes[probe_index]
        self.probes[probe_index] = ProbeConfig(int(value))
        if probe.xml_path is not None:
            self.statusBar().showMessage(f"Cleared Probe {probe_index + 1} XML because nChannels was edited", 5000)
        self._apply_probe_configs_to_model()

    def _load_probe_xml(self, probe_index: int) -> None:
        if not (0 <= probe_index < len(self.probes)):
            return
        start = ""
        current = self.probes[probe_index].xml_path
        if current is not None:
            start = str(current.parent)
        elif self.dat_path.text().strip():
            selected = Path(self.dat_path.text().strip())
            start = str(selected if selected.is_dir() else selected.parent)
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load XML for Probe {probe_index + 1}",
            start,
            "XML files (*.xml);;All files (*)",
        )
        if not path:
            return
        xml_path = Path(path)
        try:
            metadata, groups, bad = parse_neurosuite_xml(xml_path)
        except XmlError as exc:
            QMessageBox.critical(self, "XML Error", str(exc))
            return
        self.probes[probe_index] = ProbeConfig(
            n_channels=metadata.n_channels,
            xml_path=xml_path,
            groups=groups,
            bad_channels=bad,
        )
        if probe_index == 0:
            self.sampling_rate.setValue(metadata.sampling_rate)
            if metadata.lfp_sampling_rate > 0:
                self.lfp_sampling_rate.setValue(metadata.lfp_sampling_rate)
        self._refresh_probe_controls()
        self._apply_probe_configs_to_model()
        self.statusBar().showMessage(f"Loaded Probe {probe_index + 1} XML: {xml_path.name}", 5000)

    def _apply_probe_configs_to_model(self) -> None:
        total, groups, bad = self._merged_probe_model()
        self._group_source = "probe"
        self.loaded_metadata = RecordingMetadata(
            n_channels=total,
            sampling_rate=self.sampling_rate.value(),
            lfp_sampling_rate=self.lfp_sampling_rate.value(),
        )
        self._set_total_n_channels(total)
        self.groups = groups
        self.group_designs = group_designs_from_groups(groups)
        self._reset_visible_groups()
        self.bad_channels = bad
        self.bad_channels_text.setText(", ".join(str(ch) for ch in sorted(bad)))
        self.channel_regions = {
            channel: label
            for channel, label in self.channel_regions.items()
            if 0 <= channel < total
        }
        self._reset_colors()
        self._refresh_region_summary()
        self._refresh_all()

    def _merged_probe_model(self) -> tuple[int, list[ChannelGroup], set[int]]:
        total = 0
        merged_groups: list[ChannelGroup] = []
        merged_bad: set[int] = set()
        for probe_index, probe in enumerate(self.probes):
            offset = total
            if probe.groups:
                for group in probe.groups:
                    merged_groups.append(
                        ChannelGroup(
                            f"Probe {probe_index + 1} {group.name}",
                            [channel + offset for channel in group.channels],
                        )
                    )
                merged_bad.update(channel + offset for channel in probe.bad_channels)
            else:
                merged_groups.append(
                    ChannelGroup(
                        f"Probe {probe_index + 1}",
                        list(range(offset, offset + probe.n_channels)),
                    )
                )
            total += probe.n_channels
        return total, merged_groups, merged_bad

    def _set_total_n_channels(self, n_channels: int) -> None:
        self.n_channels.blockSignals(True)
        self.n_channels.setValue(max(1, int(n_channels)))
        self.n_channels.blockSignals(False)
        self._update_total_n_channels_label()

    def _update_total_n_channels_label(self) -> None:
        if hasattr(self, "total_n_channels_label"):
            self.total_n_channels_label.setText(str(sum(probe.n_channels for probe in self.probes)))

    def _build_brain_regions_section(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(4)
        section_label = QLabel("Brain Regions")
        section_label.setStyleSheet("font-weight: 600;")
        open_editor = QPushButton("Add Brain Regions")
        open_editor.clicked.connect(self._edit_brain_regions)
        save_csv = QPushButton("Save anatomical_map.csv")
        save_csv.clicked.connect(self._save_anatomical_map)
        load_csv = QPushButton("Load anatomical_map.csv")
        load_csv.clicked.connect(self._load_anatomical_map)
        self.region_summary = QLabel("No regions assigned")
        self.region_summary.setWordWrap(True)

        layout.addWidget(section_label)
        layout.addWidget(open_editor)
        layout.addWidget(load_csv)
        layout.addWidget(save_csv)
        layout.addWidget(self.region_summary)
        return panel

    def _build_analysis_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        cmap_row = QHBoxLayout()
        cmap_row.setContentsMargins(0, 0, 0, 0)
        cmap_row.addWidget(QLabel("CSD cmap"))
        cmap_row.addWidget(self.csd_cmap, 1)

        layout.addWidget(self.spectrogram_button)
        layout.addWidget(self.csd_enabled)
        layout.addLayout(cmap_row)
        layout.addStretch(1)
        return panel

    def _build_spikes_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        self.spikes_status = QLabel("Spikes: not loaded")
        self.spikes_status.setWordWrap(True)
        self.show_spikes = QCheckBox("Show spikes")
        self.spikes_below = QCheckBox("Below traces")
        self.spikes_show_waveforms = QCheckBox("Show waveform")
        self.spikes_per_region = QCheckBox("Per region")
        self.spikes_per_group = QCheckBox("Per probe group")
        self.spikes_cmap = QComboBox()
        self.spikes_cmap.addItems(COLOR_MAP_NAMES)
        self.spikes_cmap.setCurrentText("plasma")
        for widget in [self.show_spikes, self.spikes_show_waveforms, self.spikes_per_region, self.spikes_per_group]:
            widget.toggled.connect(self._refresh_spike_overlay)
        self.spikes_below.toggled.connect(self._spikes_below_changed)
        self.spikes_per_region.toggled.connect(lambda checked: self._spike_grouping_changed("region", checked))
        self.spikes_per_group.toggled.connect(lambda checked: self._spike_grouping_changed("group", checked))
        self.spikes_cmap.currentTextChanged.connect(self._refresh_spike_overlay)
        self.spike_group_cmap_panel = QWidget()
        self.spike_group_cmap_layout = QFormLayout(self.spike_group_cmap_panel)

        layout.addWidget(self.spikes_status)
        layout.addWidget(self.show_spikes)
        layout.addWidget(self.spikes_below)
        layout.addWidget(self.spikes_show_waveforms)
        layout.addWidget(QLabel("Default unit colormap"))
        layout.addWidget(self.spikes_cmap)
        layout.addWidget(self.spikes_per_region)
        layout.addWidget(self.spikes_per_group)
        layout.addWidget(self.spike_group_cmap_panel)
        layout.addStretch(1)
        return panel

    def _build_events_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        self.events_status = QLabel("Events: not loaded")
        self.events_status.setWordWrap(True)
        self.events_list = QWidget()
        self.events_list_layout = QGridLayout(self.events_list)
        self.events_list_layout.setContentsMargins(0, 0, 0, 0)
        self.events_list_layout.setHorizontalSpacing(6)
        self.events_list_layout.setVerticalSpacing(3)
        self.event_id_text = QLineEdit()
        self.event_id_text.setPlaceholderText("Event ID")
        self.event_id_text.returnPressed.connect(self._jump_to_event_id)
        previous_event = QPushButton("←")
        previous_event.setText("<")
        previous_event.clicked.connect(lambda: self._step_event_id(-1))
        next_event = QPushButton("→")
        next_event.setText(">")
        next_event.clicked.connect(lambda: self._step_event_id(1))
        jump_row = QHBoxLayout()
        jump_row.addWidget(previous_event)
        jump_row.addWidget(self.event_id_text, 1)
        jump_row.addWidget(next_event)

        layout.addWidget(self.events_status)
        layout.addWidget(self.events_list)
        layout.addWidget(QLabel("Jump to event"))
        layout.addLayout(jump_row)
        layout.addStretch(1)
        return panel

    def _build_sleep_scoring_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        self.sleep_manual_state = QComboBox()
        self.sleep_manual_state.addItems(["Wake", "NREM", "REM"])
        self.sleep_manual_selection = QLabel("Selected range: -")
        self.sleep_manual_selection.setWordWrap(True)
        self.sleep_show_transitions = QCheckBox("State transition timing")
        self.sleep_show_transitions.toggled.connect(self._refresh_sleep_plot_window)
        self.sleep_spectrogram_cmap = QComboBox()
        self.sleep_spectrogram_cmap.addItems([name.capitalize() if name != "mako" else "Mako" for name in SPECTROGRAM_COLORMAPS])
        self.sleep_spectrogram_cmap.setCurrentText("Viridis")
        self.sleep_spectrogram_cmap.currentTextChanged.connect(self._refresh_sleep_plot_window)
        self.sleep_modify_button = QPushButton("Modify State")
        self.sleep_modify_button.clicked.connect(self._modify_sleep_state_selection)
        self.sleep_modify_button.setEnabled(False)
        self.sleep_update_button = QPushButton("Update")
        self.sleep_update_button.clicked.connect(self._update_sleep_state_file)
        self.sleep_update_button.setEnabled(False)
        self.sleep_load_button = QPushButton("Load Result Folder")
        self.sleep_load_button.clicked.connect(self._load_sleep_state_clicked)
        self.sleep_status = QLabel("Load an existing SleepState.states.mat file to inspect and edit results.")
        self.sleep_status.setWordWrap(True)
        self.sleep_outputs = QLabel("Outputs: -")
        self.sleep_outputs.setWordWrap(True)
        info = QLabel("This tab loads an existing sleep-state result folder and lets you overwrite state labels.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #b8c7da;")

        layout.addWidget(info)
        layout.addWidget(self.sleep_load_button)
        layout.addWidget(self.sleep_status)
        layout.addWidget(self.sleep_outputs)
        layout.addWidget(QLabel("Spectrogram colormap"))
        layout.addWidget(self.sleep_spectrogram_cmap)
        layout.addWidget(self.sleep_show_transitions)
        layout.addWidget(self.sleep_manual_state)
        layout.addWidget(self.sleep_manual_selection)
        layout.addWidget(self.sleep_modify_button)
        layout.addWidget(self.sleep_update_button)
        layout.addStretch(1)
        return panel

    def _build_filter_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 10, 0, 4)
        layout.setSpacing(4)

        section_label = QLabel("Filters")
        section_label.setStyleSheet("font-weight: 600;")
        bandpass_row = QHBoxLayout()
        bandpass_row.setContentsMargins(0, 0, 0, 0)
        bandpass_row.addWidget(self.bandpass_low)
        bandpass_row.addWidget(QLabel("-"))
        bandpass_row.addWidget(self.bandpass_high)
        bandpass_row.addWidget(QLabel("Hz"))
        bandpass_row.addStretch(1)

        layout.addWidget(section_label)
        layout.addWidget(self.bandpass_enabled)
        layout.addLayout(bandpass_row)
        layout.addWidget(self.car_enabled)
        layout.addWidget(QLabel("CAR mode"))
        layout.addWidget(self.car_mode)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.center_stack = QStackedWidget()
        self.center_stack.addWidget(self._build_signal_center_page())
        self.center_stack.addWidget(self._build_sleep_center_page())
        layout.addWidget(self.center_stack, 1)
        layout.addWidget(self._build_time_bar())
        return panel

    def _build_signal_center_page(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        mode_row = QHBoxLayout()
        self.view_mode = QComboBox()
        self.view_mode.addItems(["single_column", "group_columns"])
        self.view_mode.currentTextChanged.connect(self._view_mode_changed)
        self.signal_background = QComboBox()
        self.signal_background.addItems(["black", "white"])
        self.signal_background.setCurrentText("black")
        self.signal_background.currentTextChanged.connect(self._signal_background_changed)
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.05, 20)
        self.scale.setDecimals(2)
        self.scale.setSingleStep(0.1)
        self.scale.setValue(8.0)
        self.scale.valueChanged.connect(self._refresh_viewer_layout)
        self.spacing = QDoubleSpinBox()
        self.spacing.setRange(0.25, 8.0)
        self.spacing.setDecimals(2)
        self.spacing.setSingleStep(0.1)
        self.spacing.setValue(1.0)
        self.spacing.valueChanged.connect(self._spacing_changed)
        mode_row.addWidget(QLabel("View"))
        mode_row.addWidget(self.view_mode)
        mode_row.addWidget(QLabel("Background"))
        mode_row.addWidget(self.signal_background)
        mode_row.addWidget(QLabel("Scale"))
        mode_row.addWidget(self.scale)
        mode_row.addWidget(QLabel("Spacing"))
        mode_row.addWidget(self.spacing)
        mode_row.addStretch(1)
        self.recording_overview = RecordingOverviewWidget()
        self.recording_overview.set_click_callback(self._jump_to_recording_overview_time)
        self.viewer = SignalViewer()
        self.signal_scroll = QScrollArea()
        self.signal_scroll.setWidget(self.viewer)
        self.signal_scroll.setWidgetResizable(True)
        self.signal_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addLayout(mode_row)
        layout.addWidget(self.recording_overview)
        layout.addWidget(self.signal_scroll, 1)
        return panel

    def _build_sleep_center_page(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.sleep_viewer = SleepStateViewer()
        self.sleep_viewer.set_selection_callback(self._sleep_span_selected)
        self.sleep_viewer.set_reset_view_callback(self._reset_sleep_view_window)
        layout.addWidget(self.sleep_viewer, 1)
        self.sleep_center_layout = layout
        return panel

    def _build_time_bar(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Start"))
        layout.addWidget(self.start_minutes)
        layout.addWidget(QLabel("min"))
        layout.addWidget(self.start_seconds)
        layout.addWidget(QLabel("sec"))
        layout.addWidget(self.start_msec)
        layout.addWidget(QLabel("msec"))
        layout.addSpacing(18)
        layout.addWidget(QLabel("Duration"))
        layout.addWidget(self.duration_minutes)
        layout.addWidget(QLabel("min"))
        layout.addWidget(self.duration_seconds)
        layout.addWidget(QLabel("sec"))
        layout.addWidget(self.duration_msec)
        layout.addWidget(QLabel("msec"))
        layout.addStretch(1)
        self.time_scroll = QScrollBar(Qt.Orientation.Horizontal)
        self.time_scroll.setRange(0, 1_000_000)
        self.time_scroll.setPageStep(50_000)
        self.time_scroll.setSingleStep(5_000)
        self.time_scroll.setTracking(False)
        self.time_scroll.setMinimumWidth(320)
        self.time_scroll.valueChanged.connect(self._time_scroll_changed)
        layout.addWidget(QLabel("Time"))
        layout.addWidget(self.time_scroll, 2)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.probe_viewer = ProbeViewer()
        self.probe_viewer.channelDoubleClicked.connect(self._toggle_bad_channel)
        self.probe_viewer.groupClicked.connect(self._toggle_group_visibility)
        self.channel_profile_viewer = ChannelProfileViewer()
        self.channel_tabs = QTabWidget()
        self.channel_tabs.setUsesScrollButtons(False)
        self.channel_tabs.addTab(self.probe_viewer, "Ch map")
        self.channel_tabs.addTab(self.channel_profile_viewer, "Ch profile")
        self.channel_tabs.currentChanged.connect(self._channel_view_tab_changed)
        self.color_map = QComboBox()
        self.color_map.addItems(COLOR_MAP_NAMES)
        self.color_map.setCurrentText("summer")
        self.color_map.currentTextChanged.connect(self._reset_colors)
        self.color_mode = QComboBox()
        self.color_mode.addItems(["all", "group", "per region"])
        self.color_mode.setCurrentText("all")
        self.color_mode.currentTextChanged.connect(self._color_mode_changed)
        color_row = QHBoxLayout()
        color_row.setContentsMargins(0, 0, 0, 0)
        color_row.addWidget(QLabel("Color map"))
        color_row.addWidget(self.color_map, 1)
        color_mode_row = QHBoxLayout()
        color_mode_row.setContentsMargins(0, 0, 0, 0)
        color_mode_row.addWidget(QLabel("Color mode"))
        color_mode_row.addWidget(self.color_mode, 1)
        self.region_cmap_panel = QWidget()
        self.region_cmap_layout = QFormLayout(self.region_cmap_panel)
        self.region_cmap_panel.setVisible(False)
        layout.addWidget(self.channel_tabs, 1)
        layout.addLayout(color_row)
        layout.addLayout(color_mode_row)
        layout.addWidget(self.region_cmap_panel)
        return panel

    def _browse_dat(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select basepath or session folder", self._recording_dialog_start_path())
        if path:
            self.dat_path.setText(path)
            self._dat_path_committed()

    def _recording_dialog_start_path(self) -> str:
        text = self.dat_path.text().strip()
        if not text:
            return ""
        current = Path(text)
        return str(current.parent if current.is_file() else current)

    def _browse_dat_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select .dat recording file",
            self._recording_dialog_start_path(),
            "DAT files (*.dat);;All files (*)",
        )
        if path:
            self.dat_path.setText(path)
            self._dat_path_committed()

    def _dat_path_text_changed(self) -> None:
        self._refresh_duration()
        if not self._is_sleep_scoring_active():
            self._refresh_viewer_layout()

    def _resolve_recording_dat_paths(self, selected_path: Path) -> list[Path]:
        if selected_path.is_file():
            if selected_path.suffix.lower() != ".dat":
                raise DatReaderError(f"Expected a .dat file, got: {selected_path.name}")
            return [selected_path]
        if not selected_path.exists():
            raise DatReaderError(f"Path does not exist: {selected_path}")
        if not selected_path.is_dir():
            raise DatReaderError(f"Unsupported path: {selected_path}")
        open_ephys_matches = self._find_open_ephys_continuous_dat_paths(selected_path)
        if open_ephys_matches:
            return open_ephys_matches
        direct_basename_dat = selected_path / f"{selected_path.name}.dat"
        if direct_basename_dat.exists():
            return [direct_basename_dat]
        direct_dat = selected_path / "amplifier.dat"
        if direct_dat.exists():
            return [direct_dat]
        try:
            children = list(selected_path.iterdir())
        except OSError as exc:
            raise DatReaderError(f"Could not inspect folder: {selected_path}") from exc
        matches = sorted(
            (
                candidate
                for child in children
                if child.is_dir()
                for candidate in [child / "amplifier.dat", child / f"{child.name}.dat"]
                if candidate.is_file()
            ),
            key=self._recording_session_sort_key,
        )
        if not matches:
            raise DatReaderError(f"No amplifier.dat, basename.dat, or continuous.dat found under: {selected_path}")
        return matches

    def _find_open_ephys_continuous_dat_paths(self, selected_path: Path) -> list[Path]:
        matches: list[Path] = []
        stack: list[tuple[Path, int]] = [(selected_path, 0)]
        skip_names = {
            ".git",
            "__pycache__",
            "analysis",
            "kilosort",
            "kilosort2",
            "kilosort3",
            "phy",
            "original_dat",
        }
        while stack:
            folder, depth = stack.pop()
            candidate = folder / "continuous.dat"
            if candidate.is_file():
                matches.append(candidate)
                continue
            if depth >= 8:
                continue
            try:
                children = sorted(folder.iterdir(), key=lambda path: path.name.lower())
            except OSError:
                continue
            for child in reversed(children):
                if not child.is_dir():
                    continue
                if child.name.lower() in skip_names:
                    continue
                stack.append((child, depth + 1))
        unique: list[Path] = []
        seen: set[str] = set()
        for path in sorted(matches, key=self._recording_session_sort_key):
            key = str(path.resolve()) if path.exists() else str(path)
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    def _recording_session_sort_key(self, path: Path) -> tuple:
        timestamp = self._extract_recording_folder_timestamp(path)
        if timestamp is not None:
            return (0, timestamp, str(path).lower())
        return (1, path.stat().st_mtime, str(path).lower())

    def _extract_recording_folder_timestamp(self, path: Path) -> tuple[int, int, int, int, int, int] | None:
        for candidate in [path.parent.name, *[parent.name for parent in path.parents]]:
            match = re.search(r"(\d{6})_(\d{6})$", candidate)
            if match is None:
                continue
            date_part, time_part = match.groups()
            return (
                int(date_part[0:2]),
                int(date_part[2:4]),
                int(date_part[4:6]),
                int(time_part[0:2]),
                int(time_part[2:4]),
                int(time_part[4:6]),
            )
        return None

    def _active_recording_dat_paths(self) -> list[Path]:
        if self._recording_dat_paths:
            return list(self._recording_dat_paths)
        path = self.dat_path.text().strip()
        if not path:
            return []
        try:
            return self._resolve_recording_dat_paths(Path(path))
        except DatReaderError:
            return []

    def _recording_dat_infos(self):
        infos = []
        for path in self._active_recording_dat_paths():
            infos.append(
                inspect_dat(
                    path,
                    self.n_channels.value(),
                    self.sampling_rate.value(),
                    allow_trailing_bytes=True,
                )
            )
        return infos

    def _update_recording_epoch_metadata(self, infos) -> None:
        segments: list[tuple[str, float, float]] = []
        boundaries: list[float] = []
        offset = 0.0
        for index, info in enumerate(infos, 1):
            start = offset
            end = offset + info.duration_seconds
            segments.append((str(index), start, end))
            if index > 1:
                boundaries.append(start)
            offset = end
        self._recording_epoch_segments = segments
        self._recording_epoch_boundaries = np.asarray(boundaries, dtype=float)
        self._refresh_recording_overview()

    def _refresh_recording_overview(self) -> None:
        if not hasattr(self, "recording_overview"):
            return
        if not self._recording_epoch_segments:
            self.recording_overview.clear()
            return
        total_duration = self._recording_epoch_segments[-1][2]
        self.recording_overview.set_epochs(self._recording_epoch_segments, total_duration)
        self.recording_overview.set_window(self.record_window_start_seconds, self.record_window_duration_seconds)

    def _jump_to_recording_overview_time(self, timestamp_seconds: float) -> None:
        duration = self._current_recording_duration_seconds()
        if duration is None:
            return
        window_duration = self._window_duration_seconds()
        max_start = max(0.0, duration - window_duration)
        start = max(0.0, min(max_start, float(timestamp_seconds) - window_duration * 0.5))
        self._set_window_start_seconds(start)
        if self._is_sleep_scoring_active():
            self._refresh_sleep_plot_window()
        else:
            self._load_window(silent=True)

    def _load_recording_window(self, *, silent: bool = False) -> None:
        result = self._read_recording_window_data(
            self._window_start_seconds(),
            self._window_duration_seconds(),
            silent=silent,
        )
        if result is None:
            return
        self._current_time, self._raw_data = result
        self._current_data = self._process_window_data(self._raw_data)
        self._refresh_viewer_layout()
        self._refresh_spectrogram_window_from_current_data()

    def _read_recording_window_data(
        self,
        start: float,
        duration: float,
        *,
        silent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        infos = self._recording_dat_infos()
        if not infos:
            return None
        self._update_recording_epoch_metadata(infos)
        total_duration = sum(info.duration_seconds for info in infos)
        start = max(0.0, float(start))
        duration = max(self._minimum_window_duration_seconds(), float(duration))
        if start >= total_duration:
            if not silent:
                QMessageBox.critical(self, "DAT Error", "Requested window is outside the recording.")
            return None
        time_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []
        remaining_start = start
        remaining_duration = duration
        offset = 0.0
        for info in infos:
            if remaining_duration <= 0:
                break
            seg_duration = info.duration_seconds
            if remaining_start >= seg_duration:
                remaining_start -= seg_duration
                offset += seg_duration
                continue
            local_start = remaining_start
            local_duration = min(remaining_duration, seg_duration - local_start)
            try:
                window = read_dat_window(
                    info.path,
                    self.n_channels.value(),
                    self.sampling_rate.value(),
                    local_start,
                    local_duration,
                    allow_trailing_bytes=True,
                )
            except (DatReaderError, OSError) as exc:
                if not silent:
                    QMessageBox.critical(self, "DAT Error", str(exc))
                return None
            if window.time_seconds.size:
                time_parts.append(window.time_seconds + offset)
                data_parts.append(window.data)
            remaining_duration -= local_duration
            remaining_start = 0.0
            offset += seg_duration
        if not time_parts or not data_parts:
            return None
        return np.concatenate(time_parts, axis=0), np.concatenate(data_parts, axis=0)

    def _dat_path_committed(self) -> None:
        path = self.dat_path.text().strip()
        if not path:
            return
        try:
            self._recording_dat_paths = self._resolve_recording_dat_paths(Path(path))
            primary_dat_path = self._recording_dat_paths[0]
            loaded_xml = self._load_adjacent_xml_if_present(primary_dat_path)
            if not loaded_xml:
                self._use_default_linear_probe()
            total_duration = sum(info.duration_seconds for info in self._recording_dat_infos())
        except DatReaderError as exc:
            QMessageBox.critical(self, "Recording Path Error", str(exc))
            self._recording_dat_paths = []
            return
        self.record_window_start_seconds = 0.0
        self.record_window_duration_seconds = min(
            1.0,
            max(self._minimum_window_duration_seconds(), total_duration),
        )
        if not self._is_sleep_scoring_active():
            self._apply_window_controls_to_widgets(self.record_window_start_seconds, self.record_window_duration_seconds)
        self._load_adjacent_anatomical_map_if_present(primary_dat_path)
        self._load_adjacent_spikes_and_events()
        if self._is_sleep_scoring_active():
            self._clear_sleep_state_context()
            self.sleep_status.setText("DAT selected. Load an existing SleepState.states.mat file when needed.")
            self.sleep_outputs.setText("Outputs: -")
            self._refresh_duration()
            self._sync_time_scroll(self._current_recording_duration_seconds())
        else:
            self._load_window(silent=True)
        if len(self._recording_dat_paths) > 1:
            self.statusBar().showMessage(f"Loaded {len(self._recording_dat_paths)} sessions in chronological order", 5000)
        else:
            self.statusBar().showMessage(f"Loaded session: {primary_dat_path.parent.name}", 5000)

    def _left_tab_changed(self, index: int) -> None:
        if not hasattr(self, "center_stack"):
            return
        self._store_current_window_controls(self._active_left_tab_index)
        if self._is_sleep_tab_index(index):
            self._apply_window_controls_to_widgets(self.sleep_window_start_seconds, self.sleep_window_duration_seconds)
            self.center_stack.setCurrentIndex(1)
            self._refresh_sleep_plot_window()
        else:
            self._apply_window_controls_to_widgets(self.record_window_start_seconds, self.record_window_duration_seconds)
            self.center_stack.setCurrentIndex(0)
            self._load_window(silent=True)
        self._active_left_tab_index = index
        self._sync_time_scroll(self._current_recording_duration_seconds())

    def _is_sleep_tab_index(self, index: int) -> bool:
        return hasattr(self, "left_tabs") and hasattr(self, "sleep_tab") and index == self.left_tabs.indexOf(self.sleep_tab)

    def _is_events_active(self) -> bool:
        return hasattr(self, "left_tabs") and hasattr(self, "events_tab") and self.left_tabs.currentIndex() == self.left_tabs.indexOf(self.events_tab)

    def _default_sleep_state_path(self) -> Path | None:
        paths = self._active_recording_dat_paths()
        if not paths:
            return None
        dat_path = paths[0]
        return dat_path.parent / f"{dat_path.stem}.SleepState.states.mat"

    def _load_sleep_state_clicked(self) -> None:
        default_path = self._default_sleep_state_path()
        start = str(default_path.parent if default_path is not None else Path.cwd())
        path = QFileDialog.getExistingDirectory(self, "Select state-scoring result folder", start)
        if not path:
            return
        self._load_sleep_state_basepath(Path(path))

    def _clear_sleep_state_context(self) -> None:
        self.sleep_state_data = None
        self.sleep_state_path = None
        self._clear_sleep_plot()

    def _load_sleep_state_basepath(self, basepath: Path) -> None:
        sleep_state_path = self._resolve_sleep_state_path(basepath)
        if sleep_state_path is None:
            QMessageBox.information(
                self,
                "State editor",
                f"Could not find a *.SleepState.states.mat file in:\n{basepath}",
            )
            return
        self._load_sleep_state_path(sleep_state_path)

    def _resolve_sleep_state_path(self, basepath: Path) -> Path | None:
        dat_path_text = self.dat_path.text().strip()
        if dat_path_text:
            candidate = basepath / f"{Path(dat_path_text).stem}.SleepState.states.mat"
            if candidate.exists():
                return candidate
        matches = sorted(basepath.glob("*.SleepState.states.mat"))
        if not matches:
            return None
        return matches[0]

    def _load_sleep_state_path(self, sleep_state_path: Path) -> None:
        basepath = sleep_state_path.parent
        basename = sleep_state_path.name.replace(".SleepState.states.mat", "")
        self._load_sleep_scoring_plot(sleep_state_path)
        self.sleep_status.setText("Loaded existing scoring.")
        output_names = [sleep_state_path.name]
        for related in [
            basepath / f"{basename}.SleepScoreLFP.LFP.mat",
            basepath / f"{basename}.EMGFromLFP.LFP.mat",
            basepath / f"{basename}.SleepStateEpisodes.states.mat",
        ]:
            if related.exists():
                output_names.append(related.name)
        self.sleep_outputs.setText(f"Outputs: {', '.join(output_names)}")

    def _load_xml(self) -> None:
        start = ""
        if self.dat_path.text().strip():
            adjacent = Path(self.dat_path.text().strip()).with_suffix(".xml")
            start = str(adjacent if adjacent.exists() else adjacent.parent)
        path, _ = QFileDialog.getOpenFileName(self, "Load amplifier.xml", start, "XML files (*.xml);;All files (*)")
        if path:
            self._apply_xml_metadata(Path(path))

    def _load_adjacent_xml_if_present(self, dat_path: Path) -> bool:
        xml_path = self._resolve_adjacent_xml_path(dat_path)
        if xml_path is None:
            return False
        if xml_path.exists():
            self._apply_xml_metadata(xml_path, show_status=True)
            return True
        return False

    def _resolve_adjacent_xml_path(self, dat_path: Path) -> Path | None:
        candidates: list[Path] = []
        selected_text = self.dat_path.text().strip()
        if selected_text:
            selected_path = Path(selected_text)
            if selected_path.is_dir():
                candidates.append(selected_path / f"{selected_path.name}.xml")
                candidates.append(selected_path / "amplifier.xml")
            else:
                candidates.append(selected_path.with_suffix(".xml"))
        candidates.append(dat_path.with_suffix(".xml"))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _apply_xml_metadata(self, path: Path, *, show_status: bool = False) -> None:
        try:
            metadata, groups, bad = parse_neurosuite_xml(path)
        except XmlError as exc:
            QMessageBox.critical(self, "XML Error", str(exc))
            return
        self.loaded_metadata = metadata
        self._group_source = "xml"
        for widget in [self.n_channels, self.sampling_rate, self.lfp_sampling_rate]:
            widget.blockSignals(True)
        self.n_channels.setValue(metadata.n_channels)
        self.sampling_rate.setValue(metadata.sampling_rate)
        if metadata.lfp_sampling_rate > 0:
            self.lfp_sampling_rate.setValue(metadata.lfp_sampling_rate)
        for widget in [self.n_channels, self.sampling_rate, self.lfp_sampling_rate]:
            widget.blockSignals(False)
        self.groups = groups
        self.group_designs = group_designs_from_groups(groups)
        self.probes = [
            ProbeConfig(
                n_channels=metadata.n_channels,
                xml_path=path,
                groups=groups,
                bad_channels=bad,
            )
        ]
        self._refresh_probe_controls()
        self._reset_visible_groups()
        self.bad_channels = bad
        self.bad_channels_text.setText(", ".join(str(ch) for ch in sorted(bad)))
        dat_path = self.dat_path.text().strip()
        if dat_path:
            self._load_adjacent_anatomical_map_if_present(Path(dat_path))
        else:
            self.channel_regions = {}
        self._reset_colors()
        self._refresh_region_summary()
        self._refresh_all()
        self._load_window(silent=True)
        if show_status:
            self.statusBar().showMessage(f"Loaded XML metadata: {path.name}", 5000)

    def _edit_brain_regions(self) -> None:
        dialog = BrainRegionEditorDialog(self.groups, self.channel_regions, self.channel_colors, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.channel_regions = dialog.channel_regions
        self._refresh_region_summary()

    def _default_anatomical_map_path(self) -> Path:
        path = self.dat_path.text().strip()
        if path:
            return Path(path).parent / "anatomical_map.csv"
        return Path("anatomical_map.csv")

    def _save_anatomical_map(self) -> None:
        if not self.groups:
            QMessageBox.information(self, "Brain Regions", "Define channel groups before saving an anatomical map.")
            return
        default = str(self._default_anatomical_map_path())
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save anatomical_map.csv",
            default,
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        save_path = Path(path)
        if save_path.exists():
            answer = QMessageBox.question(
                self,
                "Overwrite anatomical_map.csv?",
                f"{save_path.name} already exists.\nOverwrite this CSV file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        save_path.write_text(build_anatomical_map_csv(self.groups, self.channel_regions), encoding="utf-8")
        self.statusBar().showMessage(f"Saved anatomical map: {save_path.name}", 5000)

    def _load_anatomical_map(self) -> None:
        if not self.groups:
            QMessageBox.information(self, "Brain Regions", "Load or define channel groups before loading an anatomical map.")
            return
        default = str(self._default_anatomical_map_path())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load anatomical_map.csv",
            default,
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            self.channel_regions = load_anatomical_map_csv(path, self.groups)
        except AnatomicalMapError as exc:
            QMessageBox.critical(self, "Brain Regions", str(exc))
            return
        self._refresh_region_summary()
        self.statusBar().showMessage(f"Loaded anatomical map: {Path(path).name}", 5000)

    def _load_adjacent_anatomical_map_if_present(self, dat_path: Path) -> None:
        if not self.groups:
            self.channel_regions = {}
            self._refresh_region_summary()
            return
        csv_path = self._resolve_adjacent_anatomical_map_path(dat_path)
        if csv_path is None:
            self.channel_regions = {}
            self._refresh_region_summary()
            return
        try:
            self.channel_regions = load_anatomical_map_csv(csv_path, self.groups)
        except AnatomicalMapError:
            self.channel_regions = {}
        self._refresh_region_summary()

    def _resolve_adjacent_anatomical_map_path(self, dat_path: Path) -> Path | None:
        candidates: list[Path] = []
        selected_text = self.dat_path.text().strip()
        if selected_text:
            selected_path = Path(selected_text)
            if selected_path.is_dir():
                candidates.append(selected_path / "anatomical_map.csv")
            else:
                candidates.append(selected_path.parent / "anatomical_map.csv")
        candidates.extend(
            [
                dat_path.parent / "anatomical_map.csv",
                dat_path.parent.parent / "anatomical_map.csv",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_adjacent_spikes_and_events(self) -> None:
        path_text = self.dat_path.text().strip()
        if not path_text:
            return
        selected_path = Path(path_text)
        dat_paths = self._active_recording_dat_paths()
        basenames = self._analysis_basenames(selected_path, dat_paths)
        base_dirs = candidate_analysis_dirs(selected_path, dat_paths)

        self.spikes_data = None
        spikes_file = find_spikes_file(base_dirs, basenames)
        if spikes_file is not None:
            try:
                self.spikes_data = load_spikes_cellinfo(spikes_file)
                self.spikes_status.setText(f"Spikes: {len(self.spikes_data.units)} units from {spikes_file.name}")
            except EventLoadError as exc:
                self.spikes_status.setText(f"Spikes: {exc}")
        else:
            self.spikes_status.setText("Spikes: not found")

        self.event_series = []
        errors: list[str] = []
        seen_event_keys: set[str] = set()
        for event_file in find_event_files(base_dirs, basenames):
            try:
                event = load_event_file(event_file)
            except EventLoadError as exc:
                errors.append(str(exc))
                continue
            key = self._event_key(event)
            if key in seen_event_keys:
                continue
            seen_event_keys.add(key)
            self.event_series.append(event)
        if self.event_series:
            display_names = self._event_display_names()
            names = ", ".join(display_names[self._event_key(event)] for event in self.event_series)
            self.events_status.setText(f"Events: {names}")
        elif errors:
            self.events_status.setText(f"Events: {errors[0]}")
        else:
            self.events_status.setText("Events: not found")
        self._rebuild_event_controls()
        self._refresh_spike_group_cmap_controls()
        self._refresh_event_overlay()
        self._refresh_spike_overlay()

    def _analysis_basenames(self, selected_path: Path, dat_paths: list[Path]) -> list[str]:
        names: list[str] = []
        if selected_path.name:
            names.append(selected_path.stem if selected_path.is_file() else selected_path.name)
        for dat_path in dat_paths:
            names.append(dat_path.parent.name)
            names.append(dat_path.parent.parent.name)
        result: list[str] = []
        for name in names:
            if name and name not in result:
                result.append(name)
        return result

    def _rebuild_event_controls(self) -> None:
        self._event_navigation_anchor = None
        selected_key = self._selected_event_key
        while self.events_list_layout.count():
            item = self.events_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.event_controls = {}
        self.event_display_names = self._event_display_names()
        headers = ["Name", "Show", "Intervals", "Peaks", "Below"]
        for column, header in enumerate(headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: 600; color: #d6dde8;")
            self.events_list_layout.addWidget(label, 0, column)
        for row, event in enumerate(self.event_series, start=1):
            key = self._event_key(event)
            show = QCheckBox()
            intervals = QCheckBox()
            peaks = QCheckBox()
            below = QCheckBox()
            show.setChecked(False)
            intervals.setChecked(True)
            peaks.setChecked(event.peaks is not None)
            below.setChecked(True)
            if event.peaks is None:
                peaks.setEnabled(False)
            for checkbox in [show, intervals, peaks, below]:
                checkbox.toggled.connect(self._refresh_event_overlay)
            display_name = self.event_display_names.get(key, event.name)
            name = QPushButton(f"{display_name} ({event.timestamps.shape[0]})")
            name.setFlat(True)
            name.clicked.connect(lambda checked=False, event_key=key: self._select_event_for_navigation(event_key))
            self.events_list_layout.addWidget(name, row, 0)
            self.events_list_layout.addWidget(show, row, 1)
            self.events_list_layout.addWidget(intervals, row, 2)
            self.events_list_layout.addWidget(peaks, row, 3)
            self.events_list_layout.addWidget(below, row, 4)
            self.event_controls[key] = {
                "show": show,
                "intervals": intervals,
                "peaks": peaks,
                "below": below,
                "name": name,
            }
        if selected_key not in self.event_controls:
            self._selected_event_key = None
        self._refresh_event_selection_styles()

    def _select_event_for_navigation(self, event_key: str) -> None:
        if event_key not in self.event_controls:
            return
        self._selected_event_key = event_key
        self._event_navigation_anchor = None
        self._refresh_event_selection_styles()

    def _refresh_event_selection_styles(self) -> None:
        for key, controls in self.event_controls.items():
            name = controls.get("name")
            if name is None:
                continue
            selected = key == self._selected_event_key
            name.setStyleSheet(
                "QPushButton {"
                " text-align: left;"
                " padding: 2px 4px;"
                " border-radius: 3px;"
                " border: 1px solid transparent;"
                f" color: {'#ffd8d8' if selected else '#d6dde8'};"
                f" background-color: {'#5a2528' if selected else 'transparent'};"
                f" font-weight: {'600' if selected else '400'};"
                "}"
                "QPushButton:hover {"
                f" background-color: {'#6b2d31' if selected else '#333842'};"
                "}"
            )

    def _event_key(self, event: EventSeries) -> str:
        try:
            return str(event.path.resolve())
        except OSError:
            return str(event.path)

    def _event_display_names(self) -> dict[str, str]:
        counts: dict[str, int] = {}
        for event in self.event_series:
            counts[event.name] = counts.get(event.name, 0) + 1
        display_names: dict[str, str] = {}
        for event in self.event_series:
            key = self._event_key(event)
            if counts.get(event.name, 0) > 1:
                display_names[key] = f"{event.name} [{event.path.name}]"
            else:
                display_names[key] = event.name
        return display_names

    def _spike_grouping_changed(self, mode: str, checked: bool) -> None:
        if checked and mode == "region" and self.spikes_per_group.isChecked():
            self.spikes_per_group.blockSignals(True)
            self.spikes_per_group.setChecked(False)
            self.spikes_per_group.blockSignals(False)
        if checked and mode == "group" and self.spikes_per_region.isChecked():
            self.spikes_per_region.blockSignals(True)
            self.spikes_per_region.setChecked(False)
            self.spikes_per_region.blockSignals(False)
        self._refresh_spike_group_cmap_controls()

    def _spikes_below_changed(self, checked: bool) -> None:
        if checked:
            self.spikes_show_waveforms.blockSignals(True)
            self.spikes_show_waveforms.setChecked(False)
            self.spikes_show_waveforms.setEnabled(False)
            self.spikes_show_waveforms.blockSignals(False)
        else:
            self.spikes_show_waveforms.setEnabled(True)
        self._refresh_spike_overlay()

    def _refresh_spike_group_cmap_controls(self) -> None:
        if not hasattr(self, "spike_group_cmap_layout"):
            return
        while self.spike_group_cmap_layout.count():
            item = self.spike_group_cmap_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.spike_group_cmaps = {}
        groups = self._spike_color_groups()
        if len(groups) <= 1:
            self.spike_group_cmap_panel.setVisible(False)
            self._refresh_spike_overlay()
            return
        self.spike_group_cmap_panel.setVisible(True)
        for label in groups:
            combo = QComboBox()
            combo.addItems(COLOR_MAP_NAMES)
            combo.setCurrentText(self.spikes_cmap.currentText())
            combo.currentTextChanged.connect(self._refresh_spike_overlay)
            self.spike_group_cmap_layout.addRow(label, combo)
            self.spike_group_cmaps[label] = combo
        self._refresh_spike_overlay()

    def _spike_color_groups(self) -> list[str]:
        if self.spikes_data is None:
            return []
        if self.spikes_per_region.isChecked():
            region_order: list[str] = []
            for group in self.groups:
                for channel in group.channels:
                    region = self.channel_regions.get(channel)
                    if not region:
                        continue
                    if any(unit.channel == channel for unit in self.spikes_data.units) and region not in region_order:
                        region_order.append(region)
            if any(
                self._spike_group_label(unit) == "unassigned region"
                for unit in self.spikes_data.units
                if self._spike_unit_is_visible(unit)
            ):
                region_order.append("unassigned region")
            return region_order
        if self.spikes_per_group.isChecked():
            labels: list[str] = []
            for index, group in enumerate(self.groups):
                if any(self._probe_group_for_channel(unit.channel) == index for unit in self.spikes_data.units):
                    labels.append(group.name)
            fallback_labels = [
                self._spike_group_label(unit)
                for unit in self.spikes_data.units
                if self._probe_group_for_channel(unit.channel) is None
            ]
            for label in fallback_labels:
                if label not in labels:
                    labels.append(label)
            return labels
        labels = [self._spike_group_label(unit) for unit in self.spikes_data.units]
        result: list[str] = []
        for label in labels:
            if label not in result:
                result.append(label)
        return result

    def _spike_group_label(self, unit) -> str:
        if self.spikes_per_region.isChecked():
            if unit.channel is not None:
                region = self.channel_regions.get(unit.channel)
                if region:
                    return region
            return "unassigned region"
        if self.spikes_per_group.isChecked():
            group_index = self._probe_group_for_channel(unit.channel)
            if group_index is not None:
                return self.groups[group_index].name
            if unit.group is not None:
                return f"group {unit.group}"
            return "unassigned group"
        return "all units"

    def _probe_group_for_channel(self, channel: int | None) -> int | None:
        if channel is None:
            return None
        for index, group in enumerate(self.groups):
            if channel in group.channels:
                return index
        return None

    def _spike_unit_is_visible(self, unit) -> bool:
        group_index = self._probe_group_for_channel(unit.channel)
        if group_index is None:
            return True
        return group_index in self._effective_visible_group_indices()

    def _refresh_spike_overlay(self) -> None:
        if not hasattr(self, "viewer"):
            return
        overlays: list[SignalSpikeOverlay] = []
        if self.spikes_data is not None:
            unit_colors: dict[int, str] = {}
            groups: dict[str, list] = {}
            ordered_units = [
                unit
                for unit in self._ordered_spike_units_for_display()
                if self._spike_unit_is_visible(unit)
            ]
            for unit in ordered_units:
                groups.setdefault(self._spike_group_label(unit), []).append(unit)
            for group_label, units in groups.items():
                cmap = self.spike_group_cmaps.get(group_label)
                cmap_name = cmap.currentText() if cmap is not None else self.spikes_cmap.currentText()
                colors = palette_from_name(cmap_name, max(1, len(units)))
                for index, unit in enumerate(units):
                    unit_colors[unit.uid] = colors[index % len(colors)]
            for unit in ordered_units:
                overlays.append(
                    SignalSpikeOverlay(
                        unit_id=unit.uid,
                        label=unit.label,
                        times=unit.times,
                        color=unit_colors.get(unit.uid, "#808080"),
                        channel=unit.channel,
                    )
                )
        self.viewer.set_spike_overlays(
            overlays,
            show=self.show_spikes.isChecked() if hasattr(self, "show_spikes") else False,
            below=self.spikes_below.isChecked() if hasattr(self, "spikes_below") else False,
            show_waveforms=(
                self.spikes_show_waveforms.isChecked()
                if hasattr(self, "spikes_show_waveforms") and not self.spikes_below.isChecked()
                else False
            ),
        )

    def _ordered_spike_units_for_display(self) -> list:
        if self.spikes_data is None:
            return []
        original_index = {id(unit): index for index, unit in enumerate(self.spikes_data.units)}

        def sort_key(unit) -> tuple[int, int, int]:
            group_index = self._probe_group_for_channel(unit.channel)
            if group_index is None:
                return (10**9, unit.channel if unit.channel is not None else 10**9, original_index[id(unit)])
            group = self.groups[group_index]
            try:
                channel_order = group.channels.index(unit.channel)
            except ValueError:
                channel_order = 10**9
            return (group_index, channel_order, original_index[id(unit)])

        return sorted(self.spikes_data.units, key=sort_key)

    def _refresh_event_overlay(self) -> None:
        if not hasattr(self, "viewer"):
            return
        overlays: list[SignalEventOverlay] = []
        for index, event in enumerate(self.event_series):
            key = self._event_key(event)
            controls = self.event_controls.get(key)
            if controls is None or not controls["show"].isChecked():
                continue
            overlays.append(
                SignalEventOverlay(
                    name=self.event_display_names.get(key, event.name),
                    color=self._event_overlay_color(),
                    timestamps=event.timestamps,
                    peaks=event.peaks,
                    show_intervals=controls["intervals"].isChecked(),
                    show_peaks=controls["peaks"].isChecked(),
                    below=controls["below"].isChecked(),
                )
            )
        self.viewer.set_event_overlays(overlays)

    def _event_overlay_color(self) -> str:
        background = self.signal_background.currentText() if hasattr(self, "signal_background") else "black"
        return "#20242a" if background == "white" else "#ffffff"

    def _primary_event_for_navigation(self) -> EventSeries | None:
        if self._selected_event_key is not None:
            for event in self.event_series:
                if self._event_key(event) == self._selected_event_key:
                    return event
        for event in self.event_series:
            controls = self.event_controls.get(self._event_key(event))
            if controls is not None and controls["show"].isChecked():
                return event
        return self.event_series[0] if self.event_series else None

    def _jump_to_event_id(self) -> None:
        event = self._primary_event_for_navigation()
        if event is None:
            return
        try:
            event_id = int(self.event_id_text.text().strip())
        except ValueError:
            return
        event_id = max(1, min(event.timestamps.shape[0], event_id))
        self._jump_to_event_index(event, event_id - 1)

    def _step_event_id(self, step: int) -> None:
        event = self._primary_event_for_navigation()
        if event is None:
            return
        centers = self._event_centers(event)
        if centers.size == 0:
            return
        current_index = self._event_navigation_index(event, centers)
        if current_index is None:
            window_center = self._window_start_seconds() + self._window_duration_seconds() * 0.5
            event_index = int(np.argmin(np.abs(centers - window_center)))
        else:
            event_index = current_index + (1 if step > 0 else -1)
            if event_index < 0 or event_index >= centers.size:
                return
        self._jump_to_event_index(event, event_index)

    def _event_navigation_index(self, event: EventSeries, centers: np.ndarray) -> int | None:
        anchor = self._event_navigation_anchor
        if anchor is None:
            return None
        event_key, event_index = anchor
        if event_key != self._event_key(event):
            return None
        if not (0 <= event_index < centers.size):
            return None
        return event_index

    def _jump_to_event_index(self, event: EventSeries, event_index: int) -> None:
        event_index = max(0, min(event.timestamps.shape[0] - 1, int(event_index)))
        self.event_id_text.setText(str(event_index + 1))
        self._selected_event_key = self._event_key(event)
        self._refresh_event_selection_styles()
        start, end = event.timestamps[event_index]
        center = (float(start) + float(end)) * 0.5
        self._setting_event_navigation_window = True
        try:
            self._set_window_start_seconds(max(0.0, center - self._window_duration_seconds() * 0.5))
        finally:
            self._setting_event_navigation_window = False
        self._event_navigation_anchor = (self._event_key(event), event_index)
        self._load_window(silent=True)

    def _event_centers(self, event: EventSeries) -> np.ndarray:
        timestamps = np.asarray(event.timestamps, dtype=float)
        if timestamps.ndim != 2 or timestamps.shape[0] == 0:
            return np.asarray([], dtype=float)
        return np.mean(timestamps[:, :2], axis=1)

    def _refresh_region_summary(self) -> None:
        if not hasattr(self, "region_summary"):
            return
        counts: dict[str, int] = {}
        for label in self.channel_regions.values():
            clean = label.strip()
            if clean:
                counts[clean] = counts.get(clean, 0) + 1
        if not counts:
            self.region_summary.setText("No regions assigned")
            if hasattr(self, "spike_group_cmap_layout"):
                self._refresh_spike_group_cmap_controls()
            if hasattr(self, "region_cmap_layout"):
                self._refresh_region_cmap_controls()
            return
        summary = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
        self.region_summary.setText(f"Assigned channels: {summary}")
        if hasattr(self, "spike_group_cmap_layout"):
            self._refresh_spike_group_cmap_controls()
        if hasattr(self, "region_cmap_layout"):
            self._refresh_region_cmap_controls()

    def _store_current_window_controls(self, tab_index: int | None = None) -> None:
        target = self.left_tabs.currentIndex() if tab_index is None else tab_index
        start = self._window_start_seconds()
        duration = self._window_duration_seconds()
        if self._is_sleep_tab_index(target):
            self.sleep_window_start_seconds = start
            self.sleep_window_duration_seconds = duration
        else:
            self.record_window_start_seconds = start
            self.record_window_duration_seconds = duration

    def _apply_window_controls_to_widgets(self, start_seconds: float, duration_seconds: float) -> None:
        self._set_window_duration_seconds(duration_seconds, persist=False)
        self._set_window_start_seconds(start_seconds, persist=False)

    def _clear_sleep_plot(self) -> None:
        self.sleep_viewer.clear_data()
        self.sleep_selection_range = None
        self.sleep_pending_edit = None
        self.sleep_manual_selection.setText("Selected range: -")
        self.sleep_modify_button.setEnabled(False)
        self.sleep_update_button.setEnabled(False)

    def _load_sleep_scoring_plot(self, sleep_state_path: Path) -> None:
        loaded = loadmat(sleep_state_path, simplify_cells=True)
        sleep_state = loaded.get("SleepState")
        if not isinstance(sleep_state, dict):
            self._clear_sleep_plot()
            return
        self.sleep_state_data = sleep_state
        self.sleep_state_path = sleep_state_path
        self.sleep_selection_range = None
        self.sleep_pending_edit = None
        self.sleep_manual_selection.setText("Selected range: -")
        self.sleep_modify_button.setEnabled(False)
        self.sleep_update_button.setEnabled(False)
        self.sleep_window_start_seconds = 0.0
        self.sleep_window_duration_seconds = self._sleep_state_duration_seconds()
        self._apply_window_controls_to_widgets(self.sleep_window_start_seconds, self.sleep_window_duration_seconds)
        self._sync_time_scroll(self._current_recording_duration_seconds())
        self._refresh_sleep_plot_window()

    def _is_sleep_scoring_active(self) -> bool:
        return hasattr(self, "left_tabs") and self._is_sleep_tab_index(self.left_tabs.currentIndex())

    def _refresh_sleep_plot_window(self) -> None:
        if self.sleep_state_data is None:
            self._clear_sleep_plot()
            return

        idx = self.sleep_state_data.get("idx", {})
        detectorinfo = self.sleep_state_data.get("detectorinfo", {})
        detectionparms = detectorinfo.get("detectionparms", {}) if isinstance(detectorinfo, dict) else {}
        plot_materials = detectorinfo.get("StatePlotMaterials", {}) if isinstance(detectorinfo, dict) else {}
        metrics = detectionparms.get("SleepScoreMetrics", {}) if isinstance(detectionparms, dict) else {}
        hists = metrics.get("histsandthreshs", {}) if isinstance(metrics, dict) else {}
        state_t = np.asarray(idx.get("timestamps", np.asarray([])), dtype=float).reshape(-1)
        states = np.asarray(idx.get("states", np.asarray([])), dtype=float).reshape(-1)
        sw = np.asarray(metrics.get("broadbandSlowWave", np.asarray([])), dtype=float).reshape(-1)
        emg = np.asarray(metrics.get("EMG", np.asarray([])), dtype=float).reshape(-1)
        thratio = np.asarray(metrics.get("thratio", np.asarray([])), dtype=float).reshape(-1)
        metric_t = np.asarray(metrics.get("t_clus", np.asarray([])), dtype=float).reshape(-1)
        sw_thresh = float(hists.get("swthresh")) if isinstance(hists, dict) and hists.get("swthresh") is not None else None
        emg_thresh = float(hists.get("EMGthresh")) if isinstance(hists, dict) and hists.get("EMGthresh") is not None else None
        thratio_thresh = float(hists.get("THthresh")) if isinstance(hists, dict) and hists.get("THthresh") is not None else None
        if isinstance(plot_materials, dict) and plot_materials.get("thFFTspec_raw") is not None:
            th_spec = np.asarray(plot_materials.get("thFFTspec_raw", np.empty((0, 0))), dtype=float)
            th_spec_log_scale = True
        else:
            th_spec = np.asarray(plot_materials.get("thFFTspec", np.empty((0, 0))), dtype=float) if isinstance(plot_materials, dict) else np.empty((0, 0))
            th_spec_log_scale = False
        th_freqs = np.asarray(plot_materials.get("thFFTfreqs", np.asarray([])), dtype=float).reshape(-1) if isinstance(plot_materials, dict) else np.asarray([])
        th_spec_t = np.asarray(plot_materials.get("t_clus", np.asarray([])), dtype=float).reshape(-1) if isinstance(plot_materials, dict) else np.asarray([])
        if metric_t.size == 0:
            metric_t = th_spec_t if th_spec_t.size else state_t

        start = self._window_start_seconds()
        self.sleep_viewer.set_data(
            state_timestamps=state_t,
            metric_timestamps=metric_t,
            states=states,
            sw=sw,
            emg=emg,
            thratio=thratio,
            sw_threshold=sw_thresh,
            emg_threshold=emg_thresh,
            thratio_threshold=thratio_thresh,
            spec=th_spec,
            freqs=th_freqs,
            spec_timestamps=th_spec_t,
            spec_log_scale=th_spec_log_scale,
        )
        self.sleep_viewer.set_spectrogram_colormap(self.sleep_spectrogram_cmap.currentText())
        self.sleep_viewer.set_show_state_transitions(self.sleep_show_transitions.isChecked())
        self.sleep_viewer.set_window(start, self._window_duration_seconds())
        self.sleep_viewer.set_selection(self.sleep_selection_range)

    def _sleep_span_selected(self, xmin: float, xmax: float) -> None:
        lo = float(min(xmin, xmax))
        hi = float(max(xmin, xmax))
        if hi - lo <= 0:
            self.sleep_selection_range = None
            self.sleep_pending_edit = None
            self.sleep_manual_selection.setText("Selected range: -")
            self.sleep_modify_button.setEnabled(False)
            self.sleep_update_button.setEnabled(False)
            self._draw_sleep_selection()
            return
        self.sleep_selection_range = (lo, hi)
        self.sleep_pending_edit = None
        self.sleep_manual_selection.setText(f"Selected range: {lo:.3f}s to {hi:.3f}s")
        self.sleep_modify_button.setEnabled(self.sleep_state_data is not None)
        self.sleep_update_button.setEnabled(False)
        self._draw_sleep_selection()

    def _clear_sleep_selection_artists(self) -> None:
        self.sleep_selection_patches = []

    def _draw_sleep_selection(self) -> None:
        self.sleep_viewer.set_selection(self.sleep_selection_range)

    def _modify_sleep_state_selection(self) -> None:
        if self.sleep_state_data is None or self.sleep_selection_range is None:
            return
        idx = self.sleep_state_data.get("idx", {})
        timestamps = np.asarray(idx.get("timestamps", np.asarray([])), dtype=float).reshape(-1)
        if not timestamps.size:
            return
        lo, hi = self.sleep_selection_range
        state_code = {"Wake": 1, "NREM": 3, "REM": 5}[self.sleep_manual_state.currentText()]
        lo_idx = int(np.searchsorted(timestamps, lo, side="left"))
        hi_idx = int(np.searchsorted(timestamps, hi, side="right")) - 1
        if lo_idx >= timestamps.size or hi_idx < 0 or hi_idx < lo_idx:
            return
        snapped_lo = float(timestamps[lo_idx])
        snapped_hi = float(timestamps[hi_idx])
        self.sleep_selection_range = (snapped_lo, snapped_hi)
        self.sleep_pending_edit = (snapped_lo, snapped_hi, state_code)
        state_name = self.sleep_manual_state.currentText()
        self.sleep_manual_selection.setText(f"Selected range: {snapped_lo:.3f}s to {snapped_hi:.3f}s -> {state_name}")
        self.sleep_update_button.setEnabled(True)
        self.sleep_status.setText("Manual state edit staged. Press Update to overwrite SleepState.states.mat.")
        self._draw_sleep_selection()

    def _update_sleep_state_file(self) -> None:
        if self.sleep_state_data is None or self.sleep_state_path is None or self.sleep_pending_edit is None:
            return
        try:
            idx = self.sleep_state_data.get("idx", {})
            timestamps = np.asarray(idx.get("timestamps", np.asarray([])), dtype=float).reshape(-1)
            states = np.asarray(idx.get("states", np.asarray([])), dtype=np.uint8).reshape(-1)
            if not timestamps.size or not states.size:
                return
            lo, hi, state_code = self.sleep_pending_edit
            mask = (timestamps >= lo) & (timestamps <= hi)
            if not np.any(mask):
                return
            states = states.copy()
            states[mask] = np.uint8(state_code)
            statenames = ["WAKE", "", "NREM", "", "REM"]
            ints = idx_to_intervals(states, timestamps, statenames)
            self.sleep_state_data["idx"]["states"] = states.reshape(-1, 1)
            self.sleep_state_data.setdefault("ints", {})
            self.sleep_state_data["ints"]["WAKEstate"] = np.asarray(
                ints.get("WAKEstate", np.empty((0, 2))), dtype=np.float64
            ).reshape(-1, 2)
            self.sleep_state_data["ints"]["NREMstate"] = np.asarray(
                ints.get("NREMstate", np.empty((0, 2))), dtype=np.float64
            ).reshape(-1, 2)
            self.sleep_state_data["ints"]["REMstate"] = np.asarray(
                ints.get("REMstate", np.empty((0, 2))), dtype=np.float64
            ).reshape(-1, 2)
            basename = self.sleep_state_path.name.replace(".SleepState.states.mat", "")
            self.sleep_state_data, _ = append_theta_epochs(self.sleep_state_data, self.sleep_state_path.parent, basename)
            savemat(self.sleep_state_path, {"SleepState": self.sleep_state_data}, do_compression=True)
            states_to_episodes(self.sleep_state_data, self.sleep_state_path.parent, basename)
        except Exception:
            savemat(self.sleep_state_path, {"SleepState": self.sleep_state_data}, do_compression=True)
        self.sleep_update_button.setEnabled(False)
        self.sleep_pending_edit = None
        self.sleep_status.setText(f"Updated {self.sleep_state_path.name}")
        self._refresh_sleep_plot_window()

    def _generate_groups(self) -> None:
        self._group_source = "default"
        self._initialize_manual_designs()
        self.groups = self._groups_from_group_designs()
        self._reset_visible_groups()
        self._reset_colors()
        self._refresh_all()

    def _initialize_manual_designs(self) -> None:
        if self.group_designs:
            return
        n_channels = self.n_channels.value()
        self.group_designs = [GroupDesign("group1", n_channels, list(range(n_channels)))]

    def _groups_from_group_designs(self) -> list[ChannelGroup]:
        return [design.group(index) for index, design in enumerate(self.group_designs)]

    def _n_channels_changed(self) -> None:
        if self._group_source == "default":
            self.probes = [ProbeConfig(self.n_channels.value())]
            self._refresh_probe_controls()
            self._set_default_linear_probe()
            self._reset_colors()
            self._refresh_region_summary()
        self._refresh_all()

    def _use_default_linear_probe(self) -> None:
        if self._group_source == "manual":
            return
        self._group_source = "default"
        self._set_default_linear_probe()
        self._reset_colors()
        self._refresh_region_summary()

    def _set_default_linear_probe(self) -> None:
        n_channels = self.n_channels.value()
        channels = list(range(n_channels))
        self.loaded_metadata = RecordingMetadata(
            n_channels=n_channels,
            sampling_rate=self.sampling_rate.value(),
            lfp_sampling_rate=self.lfp_sampling_rate.value(),
        )
        self.probes = [ProbeConfig(n_channels)]
        self._refresh_probe_controls()
        self.group_designs = [GroupDesign("group1", n_channels, channels)]
        self.groups = [ChannelGroup("group1", channels)]
        self._reset_visible_groups()
        self.channel_regions = {
            channel: label
            for channel, label in self.channel_regions.items()
            if 0 <= channel < n_channels
        }

    def _edit_channel_map(self) -> None:
        self._initialize_manual_designs()
        dialog = ChannelMapDialog(
            self.n_channels.value(),
            self.group_designs,
            self.channel_colors,
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._group_source = "manual"
        self.group_designs = dialog.designs
        self.groups = dialog.groups()
        self._reset_visible_groups()
        self.channel_regions = {channel: label for channel, label in self.channel_regions.items() if any(channel in group.channels for group in self.groups)}
        self._reset_colors()
        self._refresh_region_summary()
        self._refresh_all()

    def _reset_colors(self) -> None:
        try:
            color_mode = self.color_mode.currentText()
            if color_mode == "all":
                group_channel_count = sum(len(group.channels) for group in self.groups)
                self.channel_colors = color_by_group_sequence(
                    self.n_channels.value(),
                    self.groups,
                    self._color_palette(max(1, group_channel_count)),
                )
            elif color_mode == "per region":
                self.channel_colors = self._color_by_region_cmap()
            else:
                self.channel_colors = self._color_by_group_local_cmap()
        except ColorMapError:
            self.channel_colors = color_by_group_sequence(
                self.n_channels.value(),
                self.groups,
                self._color_palette(self.n_channels.value()),
            )
        self._refresh_viewer_layout()

    def _color_mode_changed(self) -> None:
        self._refresh_region_cmap_controls()
        self._reset_colors()

    def _refresh_region_cmap_controls(self) -> None:
        if not hasattr(self, "region_cmap_layout"):
            return
        while self.region_cmap_layout.count():
            item = self.region_cmap_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.region_cmap_controls = {}
        regions = self._ordered_channel_regions()
        show_panel = self.color_mode.currentText() == "per region" and bool(regions)
        self.region_cmap_panel.setVisible(show_panel)
        if not show_panel:
            return
        header = QLabel("Region colormaps")
        header.setStyleSheet("font-weight: 600;")
        self.region_cmap_layout.addRow(header)
        for region in regions:
            combo = QComboBox()
            combo.addItems(COLOR_MAP_NAMES)
            combo.setCurrentText(self.color_map.currentText())
            combo.currentTextChanged.connect(self._reset_colors)
            self.region_cmap_layout.addRow(region, combo)
            self.region_cmap_controls[region] = combo

    def _color_by_region_cmap(self) -> dict[int, str]:
        colors = {channel: "#808080" for channel in range(self.n_channels.value())}
        regions = self._ordered_channel_regions()
        if not regions:
            return colors
        for region in regions:
            channels = self._channels_for_region(region)
            cmap = self.region_cmap_controls.get(region) if hasattr(self, "region_cmap_controls") else None
            cmap_name = cmap.currentText() if cmap is not None else self.color_map.currentText()
            region_colors = palette_from_name(cmap_name, max(1, len(channels)))
            for index, channel in enumerate(channels):
                if 0 <= channel < self.n_channels.value():
                    colors[channel] = region_colors[index % len(region_colors)]
        return colors

    def _channels_for_region(self, region: str) -> list[int]:
        channels: list[int] = []
        for group in self.groups:
            for channel in group.channels:
                if self.channel_regions.get(channel, "").strip() == region:
                    channels.append(channel)
        return channels

    def _ordered_channel_regions(self) -> list[str]:
        regions: list[str] = []
        for group in self.groups:
            for channel in group.channels:
                region = self.channel_regions.get(channel, "").strip()
                if region and region not in regions:
                    regions.append(region)
        return regions

    def _color_by_group_local_cmap(self) -> dict[int, str]:
        colors = {channel: "#808080" for channel in range(self.n_channels.value())}
        for group in self.groups:
            group_colors = color_by_group_channel_index(
                self.n_channels.value(),
                [group],
                self._color_palette(max(1, len(group.channels))),
            )
            for channel in group.channels:
                colors[channel] = group_colors[channel]
        return colors

    def _color_palette(self, size: int) -> list[str]:
        name = self.color_map.currentText() if hasattr(self, "color_map") else "spring"
        return palette_from_name(name or "spring", size)

    def _bad_channels_changed(self) -> None:
        self.bad_channels = self._parse_bad_channels()
        self._refresh_all()

    def _toggle_group_visibility(self, group_index: int) -> None:
        if not (0 <= group_index < len(self.groups)):
            return
        currently_visible = self._effective_visible_group_indices()
        if group_index in currently_visible and len(currently_visible) <= 1:
            self.statusBar().showMessage("At least one group must remain visible", 2500)
            return
        if group_index in self.visible_groups:
            self.visible_groups.remove(group_index)
            now_visible = False
        else:
            self.visible_groups.add(group_index)
            now_visible = True
        self._refresh_viewer_layout()
        self._refresh_spike_overlay()
        self.statusBar().showMessage(f"Group {group_index + 1} {'shown' if now_visible else 'hidden'}", 2500)

    def _toggle_bad_channel(self, channel: int) -> None:
        if channel in self.bad_channels:
            self.bad_channels.remove(channel)
        else:
            self.bad_channels.add(channel)
        self.bad_channels_text.blockSignals(True)
        self.bad_channels_text.setText(", ".join(str(ch) for ch in sorted(self.bad_channels)))
        self.bad_channels_text.blockSignals(False)
        self._refresh_all()
        self.statusBar().showMessage(
            f"Channel {channel} {'marked bad' if channel in self.bad_channels else 'marked good'}",
            2500,
        )

    def _refresh_all(self) -> None:
        self._refresh_duration()
        self._refresh_viewer_layout()
        self._load_window(silent=True)

    def _reset_visible_groups(self) -> None:
        self.visible_groups = set(range(len(self.groups)))

    def _effective_visible_group_indices(self) -> set[int]:
        all_groups = set(range(len(self.groups)))
        if not all_groups:
            return set()
        active = self.visible_groups & all_groups
        return active or all_groups

    def _sleep_state_duration_seconds(self) -> float:
        if self.sleep_state_data is None:
            return 20.0 * 60.0
        idx = self.sleep_state_data.get("idx", {})
        timestamps = np.asarray(idx.get("timestamps", np.asarray([])), dtype=float).reshape(-1)
        if timestamps.size == 0:
            return 20.0 * 60.0
        return max(1e-3, float(timestamps[-1] - timestamps[0]))

    def _reset_sleep_view_window(self) -> None:
        if self.sleep_state_data is None:
            return
        idx = self.sleep_state_data.get("idx", {})
        timestamps = np.asarray(idx.get("timestamps", np.asarray([])), dtype=float).reshape(-1)
        if timestamps.size == 0:
            return
        self.sleep_selection_range = None
        self.sleep_pending_edit = None
        self.sleep_manual_selection.setText("Selected range: -")
        self.sleep_modify_button.setEnabled(False)
        self.sleep_update_button.setEnabled(False)
        self.sleep_window_start_seconds = float(timestamps[0])
        self.sleep_window_duration_seconds = max(1e-3, float(timestamps[-1] - timestamps[0]))
        self._apply_window_controls_to_widgets(self.sleep_window_start_seconds, self.sleep_window_duration_seconds)
        self._sync_time_scroll(self._current_recording_duration_seconds())
        self._refresh_sleep_plot_window()

    def _refresh_duration(self) -> None:
        if self._is_sleep_scoring_active() and self.sleep_state_data is not None:
            duration = self._current_recording_duration_seconds()
            if duration is not None:
                self.duration_label.setText(self._format_duration(duration))
                self._sync_time_scroll(duration)
                return
        paths = self._active_recording_dat_paths()
        if not paths:
            self.duration_label.setText("-")
            self._recording_epoch_segments = []
            self._recording_epoch_boundaries = np.asarray([], dtype=float)
            self._refresh_recording_overview()
            self._sync_time_scroll(None)
            return
        try:
            infos = self._recording_dat_infos()
            total_duration = sum(info.duration_seconds for info in infos)
        except DatReaderError as exc:
            self.duration_label.setText(str(exc))
            self._recording_epoch_segments = []
            self._recording_epoch_boundaries = np.asarray([], dtype=float)
            self._refresh_recording_overview()
            self._sync_time_scroll(None)
            return
        self._update_recording_epoch_metadata(infos)
        self.duration_label.setText(self._format_duration(total_duration))
        self._sync_time_scroll(total_duration)

    def _load_window(self, *, silent: bool = False) -> None:
        self._load_recording_window(silent=silent)

    def _show_spectrogram_window(self) -> None:
        if self._is_sleep_scoring_active():
            QMessageBox.information(self, "Spectrogram", "Switch to Recording, Spikes, or Events to inspect recording channels.")
            return
        if getattr(self, "_current_data", None) is None:
            self._load_window(silent=False)
        data = getattr(self, "_current_data", None)
        time = getattr(self, "_current_time", None)
        if data is None or time is None or np.asarray(data).ndim != 2 or np.asarray(data).size == 0:
            QMessageBox.information(self, "Spectrogram", "Load a DAT window before opening the spectrogram.")
            return
        dialog = ChannelSpectrogramDialog(
            time,
            data,
            sampling_rate=self.sampling_rate.value(),
            window_start_seconds=self._window_start_seconds(),
            window_duration_seconds=self._window_duration_seconds(),
            total_duration_seconds=self._current_recording_duration_seconds(),
            window_loader=self._load_spectrogram_window_data,
            window_changed_callback=self._apply_spectrogram_window_to_recording,
            streaming=self.streaming_mode.isChecked(),
            parent=self,
        )
        self.spectrogram_window = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _load_spectrogram_window_data(
        self,
        start: float,
        duration: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        result = self._read_recording_window_data(start, duration, silent=True)
        if result is None:
            return None
        time, raw = result
        return time, self._process_window_data(raw)

    def _apply_spectrogram_window_to_recording(self, start: float, duration: float) -> None:
        self._set_window_duration_seconds(float(duration))
        self._set_window_start_seconds(float(start))
        self._load_window(silent=True)

    def _set_spectrogram_streaming_mode(self, enabled: bool) -> None:
        dialog = getattr(self, "spectrogram_window", None)
        if dialog is not None:
            dialog.set_streaming_mode(enabled)

    def _refresh_spectrogram_window_from_current_data(self) -> None:
        dialog = getattr(self, "spectrogram_window", None)
        if dialog is None:
            return
        data = getattr(self, "_current_data", None)
        time = getattr(self, "_current_time", None)
        if data is None or time is None:
            return
        dialog.update_recording_window(
            time,
            data,
            window_start_seconds=self._window_start_seconds(),
            window_duration_seconds=self._window_duration_seconds(),
            streaming=self.streaming_mode.isChecked(),
        )

    def _reprocess_current_window(self) -> None:
        raw = getattr(self, "_raw_data", None)
        if raw is None:
            return
        self._current_data = self._process_window_data(raw)
        self._refresh_viewer_layout()

    def _bandpass_toggled(self, enabled: bool) -> None:
        self.scale.setValue(self._default_scale())
        self._reprocess_current_window()

    def _view_mode_changed(self) -> None:
        self.scale.setValue(self._default_scale())
        self._refresh_viewer_layout()

    def _signal_background_changed(self) -> None:
        self._refresh_event_overlay()
        self._refresh_viewer_layout()

    def _default_scale(self) -> float:
        if self.view_mode.currentText() == "group_columns":
            return 0.8 if self.bandpass_enabled.isChecked() else 1.0
        return 1.0 if self.bandpass_enabled.isChecked() else 8.0

    def _process_window_data(self, data):
        processed = data
        try:
            if self.bandpass_enabled.isChecked():
                processed = bandpass_filter(
                    processed,
                    self.sampling_rate.value(),
                    self.bandpass_low.value(),
                    self.bandpass_high.value(),
                )
            if self.car_enabled.isChecked():
                processed = common_average_reference(
                    processed,
                    self.groups,
                    self._car_mode(),
                    bad_channels=self.bad_channels,
                    local_radius_um=200.0,
                )
        except SignalFilterError as exc:
            self.statusBar().showMessage(str(exc), 5000)
        return processed

    def _car_mode(self) -> str:
        return "all" if self.car_mode.currentText() == "all" else "group"

    def _refresh_viewer_layout(self) -> None:
        time = getattr(self, "_current_time", None)
        data = getattr(self, "_current_data", None)
        if self.view_mode.currentText() == "group_columns":
            layout_groups = self._visible_groups()
            layout = group_column_layout(layout_groups, self.bad_channels, self.channel_colors)
        else:
            layout_groups = self._visible_groups()
            layout = single_column_layout(layout_groups, self.bad_channels, self.channel_colors)
        self.viewer.set_viewport_height(self.signal_scroll.viewport().height())
        self.viewer.set_background_mode(self.signal_background.currentText() if hasattr(self, "signal_background") else "black")
        self.viewer.set_csd_overlay(
            self.csd_enabled.isChecked() if hasattr(self, "csd_enabled") else False,
            self.csd_cmap.currentText() if hasattr(self, "csd_cmap") else "bwr",
        )
        self.viewer.set_traces(
            time,
            data,
            layout,
            vertical_scale=self.scale.value(),
            row_spacing=self.spacing.value(),
            show_channel_labels=self.view_mode.currentText() != "group_columns",
            epoch_boundaries=self._recording_epoch_boundaries,
        )
        self.probe_viewer.set_probe(
            self.n_channels.value(),
            self.groups,
            self.bad_channels,
            self.channel_colors,
            self.visible_groups,
        )
        if self._is_channel_profile_tab_active():
            self._refresh_channel_profile()

    def _channel_view_tab_changed(self, index: int) -> None:
        if self._is_channel_profile_tab_active():
            if getattr(self, "_current_data", None) is None:
                self._load_window(silent=True)
            self._refresh_channel_profile()

    def _refresh_channel_profile_if_visible(self) -> None:
        if self._is_channel_profile_tab_active():
            self._refresh_channel_profile()

    def _is_channel_profile_tab_active(self) -> bool:
        return (
            hasattr(self, "channel_tabs")
            and hasattr(self, "channel_profile_viewer")
            and self.channel_tabs.currentWidget() is self.channel_profile_viewer
        )

    def _refresh_channel_profile(self) -> None:
        data = getattr(self, "_current_data", None)
        if data is None:
            self.channel_profile_viewer.clear()
            return
        scale, unit = self._channel_profile_scale()
        try:
            rms = channel_rms(data, scale)
        except ValueError:
            self.channel_profile_viewer.clear()
            return
        self.channel_profile_viewer.set_profile(
            rms,
            self._visible_groups(),
            self.bad_channels,
            self.channel_colors,
            unit=unit,
            subtitle=self._channel_profile_subtitle(),
        )

    def _channel_profile_scale(self) -> tuple[float, str]:
        if self.loaded_metadata.least_significant_bit is not None:
            return float(self.loaded_metadata.least_significant_bit), "uV"
        n_bits = self.loaded_metadata.n_bits
        voltage_range = self.loaded_metadata.voltage_range
        amplification = self.loaded_metadata.amplification
        if n_bits and voltage_range and amplification:
            scale = float(voltage_range) / float(amplification) / float(2**int(n_bits)) * 1_000_000.0
            if np.isfinite(scale) and scale > 0:
                return scale, "uV"
        return 1.0, "counts"

    def _channel_profile_subtitle(self) -> str:
        start = self._window_start_seconds()
        duration = self._window_duration_seconds()
        return f"{start:.3f}-{start + duration:.3f}s"

    def _visible_groups(self) -> list[ChannelGroup]:
        visible_indices = self._effective_visible_group_indices()
        groups = [self.groups[index] for index in sorted(visible_indices)]
        if not self.ignore_bad_channels.isChecked():
            return groups
        return [ChannelGroup(group.name, [channel for channel in group.channels if channel not in self.bad_channels]) for group in groups]

    def _save_xml(self) -> None:
        result = validate_settings(
            self._metadata(),
            self.groups,
            self.bad_channels,
            self.channel_colors,
            selected_window_start_seconds=self._window_start_seconds(),
            selected_window_duration_seconds=self._window_duration_seconds(),
        )
        if not result.ok:
            QMessageBox.critical(self, "Validation Error", "\n".join(result.errors))
            return
        xml = build_neurocode_xml(
            self.n_channels.value(),
            self.sampling_rate.value(),
            self.lfp_sampling_rate.value(),
            self.groups,
            self.bad_channels,
            metadata=self._metadata(),
            channel_colors=self.channel_colors,
        )
        default = str(Path(self.dat_path.text()).with_suffix(".xml")) if self.dat_path.text() else "amplifier.xml"
        path, _ = QFileDialog.getSaveFileName(self, "Save amplifier.xml", default, "XML files (*.xml);;All files (*)")
        if not path:
            return
        save_path = Path(path)
        save_path.write_text(xml, encoding="utf-8")
        QMessageBox.information(self, "Saved", f"Saved XML to {path}")

    def _window_start_seconds(self) -> float:
        return self.start_minutes.value() * 60.0 + self.start_seconds.value() + self.start_msec.value() / 1000.0

    def _window_duration_seconds(self) -> float:
        duration = (
            self.duration_minutes.value() * 60.0
            + self.duration_seconds.value()
            + self.duration_msec.value() / 1000.0
        )
        return max(self._minimum_window_duration_seconds(), duration)

    def _set_window_duration_seconds(self, value: float, *, persist: bool = True) -> None:
        value = max(self._minimum_window_duration_seconds(), value)
        total_usec = int(round(value * 1_000_000.0))
        minutes, rem = divmod(total_usec, 60_000_000)
        seconds, usec = divmod(rem, 1_000_000)
        msec = usec / 1000.0
        for widget in [self.duration_minutes, self.duration_seconds, self.duration_msec]:
            widget.blockSignals(True)
        self.duration_minutes.setValue(minutes)
        self.duration_seconds.setValue(seconds)
        self.duration_msec.setValue(msec)
        for widget in [self.duration_minutes, self.duration_seconds, self.duration_msec]:
            widget.blockSignals(False)
        if persist:
            self._store_current_window_controls()
        self._refresh_duration()
        if self._is_sleep_scoring_active():
            self._refresh_sleep_plot_window()
        else:
            self._refresh_viewer_layout()
            self._refresh_recording_overview()
        self._update_stream_timer_interval()

    def _set_window_start_seconds(self, value: float, *, persist: bool = True) -> None:
        if hasattr(self, "_event_navigation_anchor") and not getattr(self, "_setting_event_navigation_window", False):
            self._event_navigation_anchor = None
        value = max(0.0, value)
        total_usec = int(round(value * 1_000_000.0))
        minutes, rem = divmod(total_usec, 60_000_000)
        seconds, usec = divmod(rem, 1_000_000)
        msec = usec / 1000.0
        for widget in [self.start_minutes, self.start_seconds, self.start_msec]:
            widget.blockSignals(True)
        self.start_minutes.setValue(minutes)
        self.start_seconds.setValue(seconds)
        self.start_msec.setValue(msec)
        for widget in [self.start_minutes, self.start_seconds, self.start_msec]:
            widget.blockSignals(False)
        if persist:
            self._store_current_window_controls()
        self._refresh_duration()
        if self._is_sleep_scoring_active():
            self._refresh_sleep_plot_window()
        else:
            self._refresh_viewer_layout()
            self._refresh_recording_overview()

    def _time_controls_changed(self) -> None:
        self._store_current_window_controls()
        self._refresh_duration()
        if self._is_sleep_scoring_active():
            self._refresh_sleep_plot_window()
        else:
            self._refresh_viewer_layout()
            self._refresh_recording_overview()
        self._update_stream_timer_interval()

    def _minimum_window_duration_seconds(self) -> float:
        sampling_rate = self.sampling_rate.value() if hasattr(self, "sampling_rate") else 0.0
        if sampling_rate > 0:
            return max(1e-6, 1.0 / float(sampling_rate))
        return 1e-6

    def _apply_window_controls(self) -> None:
        self._store_current_window_controls()
        self._refresh_duration()
        if self._is_sleep_scoring_active():
            self._refresh_sleep_plot_window()
        else:
            self._load_window(silent=False)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._handle_navigation_key(event):
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Wheel and self._handle_trace_wheel(watched, event):
            return True
        if event.type() == QEvent.Type.KeyPress and self._handle_navigation_key(event):
            return True
        return super().eventFilter(watched, event)

    def _handle_trace_wheel(self, watched: QObject, event) -> bool:
        if not isinstance(watched, QWidget):
            return False
        if not self._is_signal_view_wheel_target(watched):
            return False
        delta = event.angleDelta().y()
        if delta == 0:
            return False
        modifiers = event.modifiers()
        if hasattr(self, "sleep_viewer") and (watched is self.sleep_viewer or self.sleep_viewer.isAncestorOf(watched)):
            if not (modifiers & Qt.KeyboardModifier.ControlModifier) and not (modifiers & Qt.KeyboardModifier.ShiftModifier):
                x_pos = int(event.position().toPoint().x()) if hasattr(event, "position") else 0
                anchor_fraction = self.sleep_viewer.x_fraction_at_x(x_pos)
                self._zoom_time_window(0.8 if delta > 0 else 1.25, anchor_fraction=anchor_fraction)
                event.accept()
                return True
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self._zoom_time_window(0.8 if delta > 0 else 1.25)
            event.accept()
            return True
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            factor = 1.1 if delta > 0 else 1 / 1.1
            self.spacing.setValue(
                max(self.spacing.minimum(), min(self.spacing.maximum(), self.spacing.value() * factor))
            )
            event.accept()
            return True
        return False

    def _is_signal_view_wheel_target(self, watched: QWidget) -> bool:
        if hasattr(self, "sleep_viewer") and (watched is self.sleep_viewer or self.sleep_viewer.isAncestorOf(watched)):
            return True
        if watched is self.viewer or self.viewer.isAncestorOf(watched):
            return True
        viewport = self.signal_scroll.viewport()
        return watched is viewport or viewport.isAncestorOf(watched)

    def _handle_navigation_key(self, event) -> bool:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_S:
                self._save_signal_screenshot()
                return True
            if event.key() == Qt.Key.Key_U:
                self.streaming_mode.setChecked(not self.streaming_mode.isChecked())
                return True
            if event.key() == Qt.Key.Key_I:
                self.scale.setValue(min(self.scale.maximum(), self.scale.value() * 1.25))
                return True
            if event.key() == Qt.Key.Key_D:
                self.scale.setValue(max(self.scale.minimum(), self.scale.value() / 1.25))
                return True
            if event.key() == Qt.Key.Key_BracketRight:
                self.spacing.setValue(min(self.spacing.maximum(), self.spacing.value() * 1.15))
                return True
            if event.key() == Qt.Key.Key_BracketLeft:
                self.spacing.setValue(max(self.spacing.minimum(), self.spacing.value() / 1.15))
                return True
        if self._is_events_active():
            if event.key() == Qt.Key.Key_Right:
                self._step_event_id(1)
                return True
            if event.key() == Qt.Key.Key_Left:
                self._step_event_id(-1)
                return True
        if event.key() == Qt.Key.Key_Right:
            self._scroll_time(self._window_duration_seconds() * 0.25)
            return True
        if event.key() == Qt.Key.Key_Left:
            self._scroll_time(-self._window_duration_seconds() * 0.25)
            return True
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_End:
            self._go_to_latest_window()
            return True
        if event.key() == Qt.Key.Key_Down:
            self._scroll_traces(160)
            return True
        if event.key() == Qt.Key.Key_Up:
            self._scroll_traces(-160)
            return True
        return False

    def _scroll_time(self, delta_seconds: float) -> None:
        self._set_window_start_seconds(self._window_start_seconds() + delta_seconds)
        if self._is_sleep_scoring_active():
            self._refresh_sleep_plot_window()
        elif getattr(self, "_current_data", None) is not None and self.dat_path.text().strip():
            self._load_window(silent=True)

    def _zoom_time_window(self, factor: float, *, anchor_fraction: float = 0.0) -> None:
        old_duration = self._window_duration_seconds()
        minimum_duration = self._minimum_window_duration_seconds()
        new_duration = max(minimum_duration, old_duration * factor)
        recording_duration = self._current_recording_duration_seconds()
        if recording_duration is not None:
            new_duration = min(new_duration, max(minimum_duration, recording_duration))
        start = self._window_start_seconds()
        anchor_fraction = max(0.0, min(1.0, float(anchor_fraction)))
        anchor_time = start + old_duration * anchor_fraction
        max_start = max(0.0, (recording_duration or float("inf")) - new_duration)
        new_start = anchor_time - new_duration * anchor_fraction
        self._set_window_duration_seconds(new_duration)
        self._set_window_start_seconds(max(0.0, min(max_start, new_start)))
        if self._is_sleep_scoring_active():
            self._refresh_sleep_plot_window()
        else:
            self._load_window(silent=True)

    def _scroll_traces(self, delta_pixels: int) -> None:
        bar = self.signal_scroll.verticalScrollBar()
        bar.setValue(bar.value() + delta_pixels)

    def _save_signal_screenshot(self) -> None:
        if self._is_sleep_scoring_active():
            QMessageBox.information(self, "Screenshot", "Switch to Recording, Spikes, or Events to save the signal view.")
            return
        viewport = self.signal_scroll.viewport()
        if viewport.width() <= 0 or viewport.height() <= 0:
            QMessageBox.critical(self, "Screenshot", "Signal view is not ready to save.")
            return
        default_path = self._default_screenshot_path()
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save signal screenshot",
            str(default_path),
            "PNG image (*.png)",
        )
        if not path:
            return
        save_path = self._screenshot_path_with_suffix(Path(path), selected_filter)
        try:
            self._save_signal_viewport_png(save_path)
        except OSError as exc:
            QMessageBox.critical(self, "Screenshot Error", str(exc))
            return
        self.statusBar().showMessage(f"Saved screenshot: {save_path.name}", 5000)

    def _save_signal_viewport_png(self, path: Path) -> None:
        pixmap = self.signal_scroll.viewport().grab()
        if not pixmap.save(str(path), "PNG"):
            raise OSError(f"Could not save PNG screenshot: {path}")

    def _default_screenshot_path(self) -> Path:
        start = self._duration_slug(self._window_start_seconds())
        duration = self._duration_slug(self._window_duration_seconds())
        filename = f"screenshot_from-{start}_duration-{duration}.png"
        recording_path = self.dat_path.text().strip()
        if recording_path:
            path = Path(recording_path)
            return (path if path.is_dir() else path.parent) / filename
        return Path.cwd() / filename

    def _screenshot_path_with_suffix(self, path: Path, selected_filter: str) -> Path:
        suffix = path.suffix.lower()
        if suffix == ".png":
            return path
        return path.with_suffix(".png")

    def _duration_slug(self, seconds: float) -> str:
        total_msec = max(0, int(round(float(seconds) * 1000.0)))
        minutes, rem = divmod(total_msec, 60000)
        secs, msec = divmod(rem, 1000)
        return f"{minutes}min-{secs}sec-{msec}ms"

    def _spacing_changed(self) -> None:
        self.row_spacing = self.spacing.value()
        self._refresh_viewer_layout()

    def _time_scroll_changed(self, value: int) -> None:
        if self._updating_time_scroll:
            return
        duration = self._current_recording_duration_seconds()
        if duration is None:
            return
        max_start = max(0.0, duration - self._window_duration_seconds())
        start = max_start * (value / max(1, self.time_scroll.maximum()))
        self._set_window_start_seconds(start)
        if self._is_sleep_scoring_active():
            self._refresh_sleep_plot_window()
        else:
            self._load_window(silent=True)

    def _sync_time_scroll(self, recording_duration_seconds: float | None) -> None:
        if not hasattr(self, "time_scroll"):
            return
        if recording_duration_seconds is None or recording_duration_seconds <= 0:
            self.time_scroll.setEnabled(False)
            return
        max_start = max(0.0, recording_duration_seconds - self._window_duration_seconds())
        self.time_scroll.setEnabled(max_start > 0)
        position = 0 if max_start <= 0 else round(self._window_start_seconds() / max_start * self.time_scroll.maximum())
        page_step = round(self._window_duration_seconds() / recording_duration_seconds * self.time_scroll.maximum())
        self._updating_time_scroll = True
        self.time_scroll.setPageStep(max(1, min(self.time_scroll.maximum(), page_step)))
        self.time_scroll.setValue(max(0, min(self.time_scroll.maximum(), position)))
        self._updating_time_scroll = False

    def _current_recording_duration_seconds(self) -> float | None:
        if self._is_sleep_scoring_active() and self.sleep_state_data is not None:
            timestamps = np.asarray(self.sleep_state_data.get("idx", {}).get("timestamps", np.asarray([])), dtype=float).reshape(-1)
            if timestamps.size:
                return float(timestamps[-1])
        paths = self._active_recording_dat_paths()
        if not paths:
            return None
        try:
            infos = self._recording_dat_infos()
        except DatReaderError:
            return None
        return sum(info.duration_seconds for info in infos)

    def _go_to_latest_window(self) -> None:
        if self._is_sleep_scoring_active():
            duration = self._current_recording_duration_seconds()
            if duration is None:
                return
            start = max(0.0, duration - self._window_duration_seconds())
            self._set_window_start_seconds(start)
            self._refresh_sleep_plot_window()
            return
        paths = self._active_recording_dat_paths()
        if not paths:
            return
        try:
            duration = sum(info.duration_seconds for info in self._recording_dat_infos())
        except DatReaderError as exc:
            QMessageBox.critical(self, "DAT Error", str(exc))
            return
        start = max(0.0, duration - self._window_duration_seconds())
        self._set_window_start_seconds(start)
        self._load_window(silent=False)

    def _set_streaming_enabled(self, enabled: bool) -> None:
        if enabled:
            self._set_spectrogram_streaming_mode(True)
            self._update_stream_timer_interval()
            self._stream_latest_window()
            self.stream_timer.start()
            self.statusBar().showMessage("Streaming mode on", 3000)
        else:
            self.stream_timer.stop()
            self._set_spectrogram_streaming_mode(False)
            self.statusBar().showMessage("Streaming mode off", 3000)

    def _update_stream_timer_interval(self) -> None:
        interval_ms = max(200, int(round(self._window_duration_seconds() * 1000.0)))
        self.stream_timer.setInterval(interval_ms)

    def _stream_latest_window(self) -> None:
        paths = self._active_recording_dat_paths()
        if not paths:
            return
        try:
            duration = sum(info.duration_seconds for info in self._recording_dat_infos())
        except DatReaderError:
            return
        start = max(0.0, duration - self._window_duration_seconds())
        self._set_window_start_seconds(start)
        self._load_window(silent=True)

    def _format_duration(self, seconds: float) -> str:
        total_msec = max(0, int(round(seconds * 1000.0)))
        minutes, rem = divmod(total_msec, 60000)
        secs, msec = divmod(rem, 1000)
        return f"{minutes} min {secs} sec {msec} msec"

    def _show_shortcuts(self) -> None:
        QMessageBox.information(
            self,
            "Keyboard Shortcuts",
            "\n".join(
                [
                    "Left / Right: scroll time by one window",
                    "Up / Down: scroll traces vertically",
                    "Ctrl+U: toggle streaming mode",
                    "Ctrl+S: save signal screenshot as PNG",
                    "Ctrl+End: jump to latest complete window",
                    "Ctrl+I / Ctrl+D: increase / decrease trace scale",
                    "Ctrl+Mouse wheel on traces: zoom time window",
                    "Shift+Mouse wheel on traces: change row spacing",
                    "Ctrl+] / Ctrl+[: increase / decrease row spacing",
                    "Drag on traces: zoom selected X/Y range",
                    "Double-click traces: reset X/Y zoom",
                    "Enter in time fields: load requested window",
                ]
            ),
        )

    def _metadata(self) -> RecordingMetadata:
        paths = self._active_recording_dat_paths()
        path = str(paths[0]) if paths else None
        return RecordingMetadata(
            dat_path=path,
            n_channels=self.n_channels.value(),
            sampling_rate=self.sampling_rate.value(),
            lfp_sampling_rate=self.lfp_sampling_rate.value(),
            dtype=self.loaded_metadata.dtype,
            n_bits=self.loaded_metadata.n_bits,
            voltage_range=self.loaded_metadata.voltage_range,
            amplification=self.loaded_metadata.amplification,
            offset=self.loaded_metadata.offset,
            least_significant_bit=self.loaded_metadata.least_significant_bit,
            duration_seconds=self.loaded_metadata.duration_seconds,
            total_frames=self.loaded_metadata.total_frames,
            file_size_bytes=self.loaded_metadata.file_size_bytes,
        )

    def _parse_bad_channels(self) -> set[int]:
        text = self.bad_channels_text.text().strip()
        if not text:
            return set()
        channels: set[int] = set()
        for token in text.replace(";", ",").split(","):
            token = token.strip()
            if token:
                try:
                    channels.add(int(token))
                except ValueError:
                    pass
        return channels


def run(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    args = sys.argv[1:] if argv is None else argv
    initial_path = next((arg for arg in args if not arg.startswith("-")), None)
    window = MainWindow(initial_path=initial_path)
    window.show()
    return app.exec()
