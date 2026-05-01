from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .color_map import (
    COLOR_MAP_NAMES,
    ColorMapError,
    color_by_group_channel_index,
    color_by_group_sequence,
    palette_from_name,
)
from .channel_map_editor import ChannelMapDialog, GroupDesign, group_designs_from_groups
from .dat_reader import DatReaderError, inspect_dat, read_dat_window
from .models import ChannelGroup, RecordingMetadata
from .probe_viewer import ProbeViewer
from .signal_layout import group_column_layout, single_column_layout
from .signal_viewer import SignalViewer
from .signal_filters import SignalFilterError, bandpass_filter, common_average_reference
from .validation import validate_settings
from .xml_builder import XmlError, build_neurocode_xml, parse_neurosuite_xml


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("pyNeuroscope")
        self.resize(1280, 820)
        self.groups: list[ChannelGroup] = []
        self.group_designs: list[GroupDesign] = []
        self.bad_channels: set[int] = set()
        self.channel_colors: dict[int, str] = {}
        self.loaded_metadata = RecordingMetadata()
        self.row_spacing = 1.0
        self._updating_time_scroll = False
        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self._stream_latest_window)
        self._build_ui()
        QApplication.instance().installEventFilter(self)
        self._generate_groups()
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
        splitter.setSizes([215, 1120, 260])
        layout.addWidget(splitter, 1)
        self.setCentralWidget(container)

    def _build_top_bar(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.dat_path = QLineEdit()
        self.dat_path.setMinimumWidth(260)
        self.dat_path.setMaximumWidth(720)
        self.dat_path.setPlaceholderText("Select amplifier.dat")
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_dat)
        layout.addWidget(browse)
        layout.addWidget(QLabel("DAT file"))
        layout.addWidget(self.dat_path, 1)
        layout.addStretch(2)
        return panel

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(300)
        panel.setMinimumWidth(260)
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
        self.duration_label = QLabel("-")
        self.start_minutes = QSpinBox()
        self.start_minutes.setRange(0, 10**7)
        self.start_seconds = QSpinBox()
        self.start_seconds.setRange(0, 59)
        self.start_msec = QSpinBox()
        self.start_msec.setRange(0, 999)
        self.duration_minutes = QSpinBox()
        self.duration_minutes.setRange(0, 10**7)
        self.duration_seconds = QSpinBox()
        self.duration_seconds.setRange(0, 59)
        self.duration_msec = QSpinBox()
        self.duration_msec.setRange(0, 999)
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

        form.addRow("nChannels", self.n_channels)
        form.addRow("samplingRate", self.sampling_rate)
        form.addRow("lfpSamplingRate", self.lfp_sampling_rate)
        form.addRow("Duration", self.duration_label)


        for widget in [
            self.n_channels,
            self.sampling_rate,
            self.lfp_sampling_rate,
        ]:
            widget.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
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

        self.dat_path.textChanged.connect(self._refresh_all)
        self.dat_path.editingFinished.connect(self._dat_path_committed)
        self.bad_channels_text.textChanged.connect(self._bad_channels_changed)
        self.ignore_bad_channels.toggled.connect(self._refresh_viewer_layout)
        self.streaming_mode.toggled.connect(self._set_streaming_enabled)
        self.bandpass_enabled.toggled.connect(self._bandpass_toggled)
        self.bandpass_low.valueChanged.connect(self._reprocess_current_window)
        self.bandpass_high.valueChanged.connect(self._reprocess_current_window)
        self.car_enabled.toggled.connect(self._reprocess_current_window)
        self.car_mode.currentTextChanged.connect(self._reprocess_current_window)
        save_xml = QPushButton("Save XML")
        save_xml.clicked.connect(self._save_xml)
        edit_map = QPushButton("Edit Channel Groups")
        edit_map.clicked.connect(self._edit_channel_map)
        load_xml_design = QPushButton("Load XML")
        load_xml_design.clicked.connect(self._load_xml)
        shortcuts = QPushButton("Keyboard Shortcuts")
        shortcuts.clicked.connect(self._show_shortcuts)

        layout.addLayout(form)
        layout.addWidget(load_xml_design)
        layout.addWidget(edit_map)
        layout.addWidget(save_xml)
        bad_form = QFormLayout()
        bad_form.addRow("Bad channels", self.bad_channels_text)
        layout.addLayout(bad_form)
        layout.addWidget(self.ignore_bad_channels)
        layout.addWidget(self.streaming_mode)
        layout.addWidget(self._build_filter_panel())
        layout.addStretch(1)
        layout.addWidget(shortcuts)
        return panel

    def _build_filter_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        bandpass_row = QHBoxLayout()
        bandpass_row.setContentsMargins(0, 0, 0, 0)
        bandpass_row.addWidget(self.bandpass_low)
        bandpass_row.addWidget(QLabel("-"))
        bandpass_row.addWidget(self.bandpass_high)
        bandpass_row.addWidget(QLabel("Hz"))
        bandpass_row.addStretch(1)

        layout.addWidget(self.bandpass_enabled)
        layout.addLayout(bandpass_row)
        layout.addWidget(self.car_enabled)
        layout.addWidget(QLabel("CAR mode"))
        layout.addWidget(self.car_mode)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        mode_row = QHBoxLayout()
        self.view_mode = QComboBox()
        self.view_mode.addItems(["single_column", "group_columns"])
        self.view_mode.currentTextChanged.connect(self._view_mode_changed)
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
        mode_row.addWidget(QLabel("Scale"))
        mode_row.addWidget(self.scale)
        mode_row.addWidget(QLabel("Spacing"))
        mode_row.addWidget(self.spacing)
        mode_row.addStretch(1)
        self.viewer = SignalViewer()
        self.signal_scroll = QScrollArea()
        self.signal_scroll.setWidget(self.viewer)
        self.signal_scroll.setWidgetResizable(True)
        self.signal_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addLayout(mode_row)
        layout.addWidget(self.signal_scroll, 1)
        layout.addWidget(self._build_time_bar())
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
        self.color_map = QComboBox()
        self.color_map.addItems(COLOR_MAP_NAMES)
        self.color_map.setCurrentText("spring")
        self.color_map.currentTextChanged.connect(self._reset_colors)
        self.color_mode = QComboBox()
        self.color_mode.addItems(["all", "group"])
        self.color_mode.setCurrentText("all")
        self.color_mode.currentTextChanged.connect(self._reset_colors)
        color_row = QHBoxLayout()
        color_row.setContentsMargins(0, 0, 0, 0)
        color_row.addWidget(QLabel("Color map"))
        color_row.addWidget(self.color_map, 1)
        color_mode_row = QHBoxLayout()
        color_mode_row.setContentsMargins(0, 0, 0, 0)
        color_mode_row.addWidget(QLabel("Color mode"))
        color_mode_row.addWidget(self.color_mode, 1)
        layout.addWidget(self.probe_viewer, 1)
        layout.addLayout(color_row)
        layout.addLayout(color_mode_row)
        return panel

    def _browse_dat(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select amplifier.dat", "", "DAT files (*.dat);;All files (*)")
        if path:
            self.dat_path.setText(path)
            self._dat_path_committed()

    def _dat_path_committed(self) -> None:
        path = self.dat_path.text().strip()
        if not path:
            return
        self._load_adjacent_xml_if_present(Path(path))
        self._load_window(silent=True)

    def _load_xml(self) -> None:
        start = ""
        if self.dat_path.text().strip():
            adjacent = Path(self.dat_path.text().strip()).with_suffix(".xml")
            start = str(adjacent if adjacent.exists() else adjacent.parent)
        path, _ = QFileDialog.getOpenFileName(self, "Load amplifier.xml", start, "XML files (*.xml);;All files (*)")
        if path:
            self._apply_xml_metadata(Path(path))

    def _load_adjacent_xml_if_present(self, dat_path: Path) -> None:
        xml_path = dat_path.with_suffix(".xml")
        if xml_path.exists():
            self._apply_xml_metadata(xml_path, show_status=True)

    def _apply_xml_metadata(self, path: Path, *, show_status: bool = False) -> None:
        try:
            metadata, groups, bad = parse_neurosuite_xml(path)
        except XmlError as exc:
            QMessageBox.critical(self, "XML Error", str(exc))
            return
        self.loaded_metadata = metadata
        self.n_channels.setValue(metadata.n_channels)
        self.sampling_rate.setValue(metadata.sampling_rate)
        if metadata.lfp_sampling_rate > 0:
            self.lfp_sampling_rate.setValue(metadata.lfp_sampling_rate)
        self.groups = groups
        self.group_designs = group_designs_from_groups(groups)
        self.bad_channels = bad
        self.bad_channels_text.setText(", ".join(str(ch) for ch in sorted(bad)))
        self._reset_colors()
        self._refresh_all()
        self._load_window(silent=True)
        if show_status:
            self.statusBar().showMessage(f"Loaded XML metadata: {path.name}", 5000)

    def _generate_groups(self) -> None:
        self._initialize_manual_designs()
        self.groups = self._groups_from_group_designs()
        self._refresh_all()

    def _initialize_manual_designs(self) -> None:
        if self.group_designs:
            return
        n_channels = self.n_channels.value()
        self.group_designs = [GroupDesign("group1", n_channels, list(range(n_channels)))]

    def _groups_from_group_designs(self) -> list[ChannelGroup]:
        return [design.group(index) for index, design in enumerate(self.group_designs)]

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
        self.group_designs = dialog.designs
        self.groups = dialog.groups()
        self._reset_colors()
        self._refresh_all()

    def _reset_colors(self) -> None:
        try:
            if self.color_mode.currentText() == "all":
                group_channel_count = sum(len(group.channels) for group in self.groups)
                self.channel_colors = color_by_group_sequence(
                    self.n_channels.value(),
                    self.groups,
                    self._color_palette(max(1, group_channel_count)),
                )
            else:
                self.channel_colors = self._color_by_group_local_cmap()
        except ColorMapError:
            self.channel_colors = color_by_group_sequence(
                self.n_channels.value(),
                self.groups,
                self._color_palette(self.n_channels.value()),
            )
        self._refresh_viewer_layout()

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

    def _refresh_duration(self) -> None:
        path = self.dat_path.text().strip()
        if not path:
            self.duration_label.setText("-")
            self._sync_time_scroll(None)
            return
        try:
            info = inspect_dat(
                path,
                self.n_channels.value(),
                self.sampling_rate.value(),
                allow_trailing_bytes=True,
            )
        except DatReaderError as exc:
            self.duration_label.setText(str(exc))
            self._sync_time_scroll(None)
            return
        self.duration_label.setText(self._format_duration(info.duration_seconds))
        self._sync_time_scroll(info.duration_seconds)

    def _load_window(self, *, silent: bool = False) -> None:
        if not self.dat_path.text().strip():
            return
        try:
            window = read_dat_window(
                self.dat_path.text().strip(),
                self.n_channels.value(),
                self.sampling_rate.value(),
                self._window_start_seconds(),
                self._window_duration_seconds(),
                allow_trailing_bytes=True,
            )
        except (DatReaderError, OSError) as exc:
            if not silent:
                QMessageBox.critical(self, "DAT Error", str(exc))
            return
        self._current_time = window.time_seconds
        self._raw_data = window.data
        self._current_data = self._process_window_data(window.data)
        self._refresh_viewer_layout()

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
        self.viewer.set_traces(
            time,
            data,
            layout,
            vertical_scale=self.scale.value(),
            row_spacing=self.spacing.value(),
        )
        self.probe_viewer.set_probe(self.n_channels.value(), self.groups, self.bad_channels, self.channel_colors)

    def _visible_groups(self) -> list[ChannelGroup]:
        if not self.ignore_bad_channels.isChecked():
            return self.groups
        return [
            ChannelGroup(group.name, [channel for channel in group.channels if channel not in self.bad_channels])
            for group in self.groups
        ]

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
        if save_path.exists():
            answer = QMessageBox.question(
                self,
                "Overwrite XML?",
                f"{save_path.name} already exists.\nOverwrite this XML file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
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
        return max(0.001, duration)

    def _set_window_duration_seconds(self, value: float) -> None:
        value = max(0.001, value)
        total_msec = int(round(value * 1000.0))
        minutes, rem = divmod(total_msec, 60000)
        seconds, msec = divmod(rem, 1000)
        for widget in [self.duration_minutes, self.duration_seconds, self.duration_msec]:
            widget.blockSignals(True)
        self.duration_minutes.setValue(minutes)
        self.duration_seconds.setValue(seconds)
        self.duration_msec.setValue(msec)
        for widget in [self.duration_minutes, self.duration_seconds, self.duration_msec]:
            widget.blockSignals(False)
        self._refresh_duration()
        self._refresh_viewer_layout()
        self._update_stream_timer_interval()

    def _set_window_start_seconds(self, value: float) -> None:
        value = max(0.0, value)
        total_msec = int(round(value * 1000.0))
        minutes, rem = divmod(total_msec, 60000)
        seconds, msec = divmod(rem, 1000)
        for widget in [self.start_minutes, self.start_seconds, self.start_msec]:
            widget.blockSignals(True)
        self.start_minutes.setValue(minutes)
        self.start_seconds.setValue(seconds)
        self.start_msec.setValue(msec)
        for widget in [self.start_minutes, self.start_seconds, self.start_msec]:
            widget.blockSignals(False)
        self._refresh_duration()
        self._refresh_viewer_layout()

    def _time_controls_changed(self) -> None:
        self._refresh_duration()
        self._refresh_viewer_layout()
        self._update_stream_timer_interval()

    def _apply_window_controls(self) -> None:
        self._refresh_duration()
        self._load_window(silent=False)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._handle_navigation_key(event):
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Wheel and self._handle_time_zoom_wheel(watched, event):
            return True
        if event.type() == QEvent.Type.KeyPress and self._handle_navigation_key(event):
            return True
        return super().eventFilter(watched, event)

    def _handle_time_zoom_wheel(self, watched: QObject, event) -> bool:
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            return False
        if not isinstance(watched, QWidget):
            return False
        if not self._is_signal_view_wheel_target(watched):
            return False
        delta = event.angleDelta().y()
        if delta == 0:
            return False
        self._zoom_time_window(0.8 if delta > 0 else 1.25)
        event.accept()
        return True

    def _is_signal_view_wheel_target(self, watched: QWidget) -> bool:
        if watched is self.viewer or self.viewer.isAncestorOf(watched):
            return True
        viewport = self.signal_scroll.viewport()
        return watched is viewport or viewport.isAncestorOf(watched)

    def _handle_navigation_key(self, event) -> bool:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
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
        if event.key() == Qt.Key.Key_Right:
            self._scroll_time(self._window_duration_seconds())
            return True
        if event.key() == Qt.Key.Key_Left:
            self._scroll_time(-self._window_duration_seconds())
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
        if getattr(self, "_current_data", None) is not None and self.dat_path.text().strip():
            self._load_window(silent=True)

    def _zoom_time_window(self, factor: float) -> None:
        old_duration = self._window_duration_seconds()
        new_duration = max(0.001, old_duration * factor)
        recording_duration = self._current_recording_duration_seconds()
        if recording_duration is not None:
            new_duration = min(new_duration, max(0.001, recording_duration))
        start = self._window_start_seconds()
        max_start = max(0.0, (recording_duration or float("inf")) - new_duration)
        self._set_window_duration_seconds(new_duration)
        self._set_window_start_seconds(max(0.0, min(max_start, start)))
        self._load_window(silent=True)

    def _scroll_traces(self, delta_pixels: int) -> None:
        bar = self.signal_scroll.verticalScrollBar()
        bar.setValue(bar.value() + delta_pixels)

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
        self.time_scroll.setPageStep(max(1, page_step))
        self.time_scroll.setValue(max(0, min(self.time_scroll.maximum(), position)))
        self._updating_time_scroll = False

    def _current_recording_duration_seconds(self) -> float | None:
        path = self.dat_path.text().strip()
        if not path:
            return None
        try:
            info = inspect_dat(
                path,
                self.n_channels.value(),
                self.sampling_rate.value(),
                allow_trailing_bytes=True,
            )
        except DatReaderError:
            return None
        return info.duration_seconds

    def _go_to_latest_window(self) -> None:
        path = self.dat_path.text().strip()
        if not path:
            return
        try:
            info = inspect_dat(
                path,
                self.n_channels.value(),
                self.sampling_rate.value(),
                allow_trailing_bytes=True,
            )
        except DatReaderError as exc:
            QMessageBox.critical(self, "DAT Error", str(exc))
            return
        start = max(0.0, info.duration_seconds - self._window_duration_seconds())
        self._set_window_start_seconds(start)
        self._load_window(silent=False)

    def _set_streaming_enabled(self, enabled: bool) -> None:
        if enabled:
            self._update_stream_timer_interval()
            self._stream_latest_window()
            self.stream_timer.start()
            self.statusBar().showMessage("Streaming mode on", 3000)
        else:
            self.stream_timer.stop()
            self.statusBar().showMessage("Streaming mode off", 3000)

    def _update_stream_timer_interval(self) -> None:
        interval_ms = max(200, int(round(self._window_duration_seconds() * 1000.0)))
        self.stream_timer.setInterval(interval_ms)

    def _stream_latest_window(self) -> None:
        path = self.dat_path.text().strip()
        if not path:
            return
        try:
            info = inspect_dat(
                path,
                self.n_channels.value(),
                self.sampling_rate.value(),
                allow_trailing_bytes=True,
            )
        except DatReaderError:
            return
        start = max(0.0, info.duration_seconds - self._window_duration_seconds())
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
                    "Ctrl+End: jump to latest complete window",
                    "Ctrl+I / Ctrl+D: increase / decrease trace scale",
                    "Ctrl+Mouse wheel on traces: zoom time window",
                    "Ctrl+] / Ctrl+[: increase / decrease row spacing",
                    "Drag on traces: zoom selected X/Y range",
                    "Double-click traces: reset X/Y zoom",
                    "Enter in time fields: load requested window",
                ]
            ),
        )

    def _metadata(self) -> RecordingMetadata:
        path = self.dat_path.text().strip() or None
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


def run() -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
