from __future__ import annotations

from collections.abc import Callable

import numpy as np
from matplotlib import colormaps
from PySide6.QtCore import QEvent, QObject, QRect, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QDialog,
    QScrollBar,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy import signal


MAX_DISPLAY_FREQ_HZ = 300.0
MIN_DISPLAY_FREQ_HZ = 1.0
NEUROSCOPE_SPECTROGRAM_WINDOW_SECONDS = 0.2
NEUROSCOPE_SPECTROGRAM_FREQ_LOW_HZ = 1.0
NEUROSCOPE_SPECTROGRAM_FREQ_HIGH_HZ = 300.0
NEUROSCOPE_SPECTROGRAM_FREQ_STEP_HZ = 2.0
WAVELET_COLORMAPS = ("viridis", "plasma", "inferno", "magma", "cividis")
FREQUENCY_SCALES = ("linear", "log")
SPECTROGRAM_MAX_SAMPLES = 12000
SPECTROGRAM_OVERLAP_FRACTION = 0.95
WAVELET_DISPLAY_LOW_PERCENTILE = 5.0
WAVELET_DISPLAY_HIGH_PERCENTILE = 99.5


class ChannelSpectrogramWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._channel = 0
        self._sampling_rate = 1.0
        self._window_start = 0.0
        self._trace = np.asarray([], dtype=float)
        self._freqs = np.asarray([], dtype=float)
        self._times = np.asarray([], dtype=float)
        self._spec_db = np.empty((0, 0), dtype=float)
        self._display_freq_range = (
            NEUROSCOPE_SPECTROGRAM_FREQ_LOW_HZ,
            NEUROSCOPE_SPECTROGRAM_FREQ_HIGH_HZ,
        )
        self._value_range = (-120.0, -40.0)
        self._image: QImage | None = None
        self._image_key: tuple | None = None
        self._colormap_name = "viridis"
        self._colormap = self._build_colormap(self._colormap_name)
        self._frequency_scale = "linear"
        self._spectrogram_window_seconds = NEUROSCOPE_SPECTROGRAM_WINDOW_SECONDS
        self._freq_step_hz = NEUROSCOPE_SPECTROGRAM_FREQ_STEP_HZ
        self.setMinimumSize(720, 480)

    def set_trace(
        self,
        trace: np.ndarray,
        *,
        channel: int,
        sampling_rate: float,
        window_start_seconds: float,
    ) -> None:
        self._channel = int(channel)
        self._sampling_rate = max(1e-9, float(sampling_rate))
        self._window_start = float(window_start_seconds)
        self._trace = np.asarray(trace, dtype=float).reshape(-1)
        self._compute_spectrogram()
        self.update()

    def set_frequency_range(self, low_hz: float, high_hz: float) -> None:
        nyquist = max(1e-9, self._sampling_rate * 0.5)
        low = max(MIN_DISPLAY_FREQ_HZ, min(float(low_hz), nyquist))
        high = max(MIN_DISPLAY_FREQ_HZ, min(float(high_hz), nyquist, MAX_DISPLAY_FREQ_HZ))
        if high <= low:
            high = min(nyquist, low + 1.0)
            if high <= low:
                low = max(MIN_DISPLAY_FREQ_HZ, high - 1.0)
        if np.isclose(low, self._display_freq_range[0]) and np.isclose(high, self._display_freq_range[1]):
            return
        self._display_freq_range = (low, high)
        self._compute_spectrogram()
        self.update()

    def set_spectrogram_parameters(self, *, window_seconds: float, freq_step_hz: float) -> None:
        window = max(1.0 / max(1e-9, self._sampling_rate), float(window_seconds))
        step = max(1e-9, float(freq_step_hz))
        changed = (
            not np.isclose(window, self._spectrogram_window_seconds)
            or not np.isclose(step, self._freq_step_hz)
        )
        self._spectrogram_window_seconds = window
        self._freq_step_hz = step
        if changed:
            self._compute_spectrogram()
            self.update()

    def set_colormap(self, name: str) -> None:
        clean = name.strip().lower()
        if clean not in WAVELET_COLORMAPS:
            clean = "viridis"
        if clean == self._colormap_name:
            return
        self._colormap_name = clean
        self._colormap = self._build_colormap(clean)
        self._image = None
        self._image_key = None
        self.update()

    def set_frequency_scale(self, scale: str) -> None:
        clean = scale.strip().lower()
        if clean not in FREQUENCY_SCALES:
            clean = "linear"
        if clean == self._frequency_scale:
            return
        self._frequency_scale = clean
        self._image = None
        self._image_key = None
        self.update()

    def _compute_spectrogram(self) -> None:
        self._image = None
        self._image_key = None
        trace = self._trace
        if trace.size < 4 or not np.any(np.isfinite(trace)):
            self._freqs = np.asarray([], dtype=float)
            self._times = np.asarray([], dtype=float)
            self._spec_db = np.empty((0, 0), dtype=float)
            return
        trace = trace - float(np.nanmedian(trace))
        trace, fs = self._resample_for_spectrogram(trace, self._sampling_rate)
        if trace.size < 4:
            self._freqs = np.asarray([], dtype=float)
            self._times = np.asarray([], dtype=float)
            self._spec_db = np.empty((0, 0), dtype=float)
            return
        freqs, times, spec = self._compute_neuroscope_spectrogram(trace, fs)
        self._freqs = freqs
        self._times = self._window_start + times
        self._spec_db = spec
        self._display_freq_range = (
            min(max(self._display_freq_range[0], MIN_DISPLAY_FREQ_HZ), fs * 0.5),
            min(max(self._display_freq_range[1], MIN_DISPLAY_FREQ_HZ), fs * 0.5, MAX_DISPLAY_FREQ_HZ),
        )
        self._value_range = self._compute_display_value_range()

    def _resample_for_spectrogram(self, trace: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, float]:
        fs = max(1e-9, float(sampling_rate))
        target_fs = min(fs, max(MAX_DISPLAY_FREQ_HZ * 4.0, 1000.0))
        down = max(1, int(np.floor(fs / target_fs)))
        if down > 1:
            trace = signal.resample_poly(trace, up=1, down=down)
            fs = fs / down
        if trace.size > SPECTROGRAM_MAX_SAMPLES:
            down = int(np.ceil(trace.size / SPECTROGRAM_MAX_SAMPLES))
            trace = signal.resample_poly(trace, up=1, down=down)
            fs = fs / down
        return np.asarray(trace, dtype=float), fs

    def _compute_neuroscope_spectrogram(
        self,
        trace: np.ndarray,
        sampling_rate: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        window_samples = self._spectrogram_segment_size(trace.size, sampling_rate)
        nfft = self._spectrogram_fft_size(window_samples, sampling_rate)
        noverlap = self._spectrogram_overlap_size(window_samples)
        freqs, times, stft = signal.spectrogram(
            trace * 5.0,
            fs=sampling_rate,
            window="hamming",
            nperseg=window_samples,
            nfft=nfft,
            noverlap=noverlap,
            detrend=False,
            scaling="spectrum",
            mode="complex",
        )
        freq_range = self._spectrogram_frequency_range(sampling_rate)
        if freq_range.size == 0 or stft.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float), np.empty((0, 0), dtype=float)
        magnitude = np.abs(stft)
        interpolated = np.vstack(
            [
                np.interp(freq_range, freqs, magnitude[:, column])
                for column in range(magnitude.shape[1])
            ]
        ).T
        floor = np.finfo(np.float64).tiny
        return freq_range, times, 200.0 * np.log10(np.maximum(interpolated, floor))

    def _spectrogram_frequency_range(self, sampling_rate: float) -> np.ndarray:
        nyquist = max(1e-9, float(sampling_rate) * 0.5)
        low, high = self._display_freq_range
        low = max(MIN_DISPLAY_FREQ_HZ, min(float(low), nyquist))
        high = max(low, min(float(high), nyquist, MAX_DISPLAY_FREQ_HZ))
        step = max(1e-9, float(self._freq_step_hz))
        if high <= low:
            return np.asarray([], dtype=float)
        freqs = np.arange(low, high + step * 0.5, step, dtype=float)
        freqs = freqs[freqs <= high]
        if freqs.size == 0 or not np.isclose(freqs[-1], high):
            freqs = np.append(freqs, high)
        return freqs

    def _spectrogram_segment_size(self, sample_count: int, sampling_rate: float | None = None) -> int:
        sample_count = max(1, int(sample_count))
        fs = self._sampling_rate if sampling_rate is None else max(1e-9, float(sampling_rate))
        target = int(round(fs * self._spectrogram_window_seconds))
        return max(4, min(sample_count, max(4, target)))

    def _spectrogram_fft_size(self, segment_size: int, sampling_rate: float | None = None) -> int:
        fs = self._sampling_rate if sampling_rate is None else max(1e-9, float(sampling_rate))
        frequency_bins = int(np.ceil(fs / max(1e-9, self._freq_step_hz)))
        return max(int(segment_size), frequency_bins)

    def _spectrogram_overlap_size(self, segment_size: int) -> int:
        segment_size = max(1, int(segment_size))
        return min(segment_size - 1, max(0, int(round(segment_size * SPECTROGRAM_OVERLAP_FRACTION))))

    def _compute_display_value_range(self) -> tuple[float, float]:
        data = self._display_spectrogram_db()
        finite = data[np.isfinite(data)]
        if finite.size:
            lo, hi = np.nanpercentile(
                finite,
                [WAVELET_DISPLAY_LOW_PERCENTILE, WAVELET_DISPLAY_HIGH_PERCENTILE],
            )
            if hi <= lo:
                hi = lo + 1.0
            return float(lo), float(hi)
        return -120.0, -40.0

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor("#101216"))
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            plot_rect = self._plot_rect()
            self._draw_frame(painter, plot_rect)
            if self._display_spectrogram_db().size == 0 or self._freqs.size == 0 or self._times.size == 0:
                painter.setPen(QPen(QColor("#8a9099")))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Not enough samples for spectrogram")
                return
            image = self._spectrogram_image(plot_rect)
            painter.drawImage(plot_rect, image)
            self._draw_axes(painter, plot_rect)
        finally:
            painter.end()

    def _plot_rect(self) -> QRect:
        margin_left = 76
        margin_right = 24
        margin_top = 38
        margin_bottom = 58
        return QRect(
            margin_left,
            margin_top,
            max(1, self.width() - margin_left - margin_right),
            max(1, self.height() - margin_top - margin_bottom),
        )

    def _draw_frame(self, painter: QPainter, rect: QRect) -> None:
        painter.setPen(QPen(QColor("#d6dde8")))
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(12, 8, self.width() - 24, 22, Qt.AlignmentFlag.AlignLeft, f"ch {self._channel}")
        painter.setPen(QPen(QColor("#4b515a")))
        painter.drawRect(rect)

    def _spectrogram_image(self, rect: QRect) -> QImage:
        key = (
            int(rect.width()),
            int(rect.height()),
            int(self._spec_db.shape[0]),
            int(self._spec_db.shape[1]),
            float(self._display_freq_range[0]),
            float(self._display_freq_range[1]),
            float(self._value_range[0]),
            float(self._value_range[1]),
            self._colormap_name,
            self._frequency_scale,
        )
        if self._image is not None and self._image_key == key:
            return self._image
        data = self._reduced_spectrogram(rect)
        lo, hi = self._value_range
        scaled = np.clip((data - lo) / max(1e-9, hi - lo), 0.0, 1.0)
        color_index = np.asarray(np.clip(np.rint(np.flipud(scaled) * 255.0), 0, 255), dtype=np.uint8)
        image = QImage(color_index.shape[1], color_index.shape[0], QImage.Format.Format_RGB32)
        for y in range(color_index.shape[0]):
            for x in range(color_index.shape[1]):
                image.setPixelColor(x, y, self._colormap[int(color_index[y, x])])
        self._image = image
        self._image_key = key
        return image

    def _reduced_spectrogram(self, rect: QRect) -> np.ndarray:
        data = self._display_spectrogram_db()
        freqs = self._display_frequencies()
        target_cols = max(1, min(data.shape[1], rect.width()))
        target_rows = max(1, min(data.shape[0], rect.height()))
        data = self._bin_axis(data, target_cols, axis=1)
        if self._frequency_scale == "log":
            return self._resample_frequency_axis_log(data, freqs, target_rows)
        return self._bin_axis(data, target_rows, axis=0)

    def _display_spectrogram_db(self) -> np.ndarray:
        if self._spec_db.size == 0 or self._freqs.size == 0:
            return np.empty((0, 0), dtype=float)
        low, high = self._display_freq_range
        freq_mask = (self._freqs >= low) & (self._freqs <= high)
        if not np.any(freq_mask):
            return np.empty((0, 0), dtype=float)
        return self._spec_db[freq_mask, :]

    def _display_frequencies(self) -> np.ndarray:
        if self._freqs.size == 0:
            return np.asarray([], dtype=float)
        low, high = self._display_freq_range
        return self._freqs[(self._freqs >= low) & (self._freqs <= high)]

    def _resample_frequency_axis_log(
        self,
        data: np.ndarray,
        freqs: np.ndarray,
        target_rows: int,
    ) -> np.ndarray:
        if data.size == 0 or freqs.size == 0:
            return data
        target_rows = max(1, int(target_rows))
        if freqs.size == 1:
            return data
        low = max(1e-9, float(freqs[0]))
        high = max(low, float(freqs[-1]))
        target_freqs = np.geomspace(low, high, target_rows)
        log_freqs = np.log(np.maximum(freqs, MIN_DISPLAY_FREQ_HZ))
        log_targets = np.log(target_freqs)
        columns = [
            np.interp(log_targets, log_freqs, data[:, column])
            for column in range(data.shape[1])
        ]
        return np.column_stack(columns)

    def _bin_axis(self, data: np.ndarray, target_size: int, *, axis: int) -> np.ndarray:
        size = int(data.shape[axis])
        target = max(1, int(target_size))
        if size <= target:
            return data
        step = int(np.ceil(size / target))
        usable = (size // step) * step
        if usable <= 0:
            return data
        if axis == 1:
            reduced = np.nanmean(data[:, :usable].reshape(data.shape[0], usable // step, step), axis=2)
            tail = data[:, usable:]
            if tail.size:
                reduced = np.column_stack((reduced, np.nanmean(tail, axis=1)))
            return reduced
        reduced = np.nanmean(data[:usable, :].reshape(usable // step, step, data.shape[1]), axis=1)
        tail = data[usable:, :]
        if tail.size:
            reduced = np.vstack((reduced, np.nanmean(tail, axis=0, keepdims=True)))
        return reduced

    def _draw_axes(self, painter: QPainter, rect: QRect) -> None:
        painter.setPen(QPen(QColor("#b8c7da")))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        min_freq, max_freq = self._display_freq_range
        for freq in self._frequency_ticks(min_freq, max_freq):
            y = self._y_at_frequency(freq, rect)
            painter.drawLine(rect.left() - 4, y, rect.left(), y)
            painter.drawText(4, y - 8, rect.left() - 12, 16, Qt.AlignmentFlag.AlignRight, self._format_frequency_tick(freq))
        t0 = float(self._times[0]) if self._times.size else self._window_start
        t1 = float(self._times[-1]) if self._times.size else self._window_start
        for timepoint in [t0, (t0 + t1) * 0.5, t1]:
            frac = (timepoint - t0) / max(1e-9, t1 - t0)
            x = rect.left() + int(round(frac * rect.width()))
            painter.drawLine(x, rect.bottom(), x, rect.bottom() + 4)
            painter.drawText(x - 42, rect.bottom() + 10, 84, 18, Qt.AlignmentFlag.AlignCenter, f"{timepoint:.3f}s")
        painter.drawText(4, rect.top() - 26, rect.left() - 12, 18, Qt.AlignmentFlag.AlignRight, "Hz")
        painter.drawText(rect.left(), rect.bottom() + 34, rect.width(), 18, Qt.AlignmentFlag.AlignCenter, "Time")

    def _y_at_frequency(self, freq: float, rect: QRect) -> int:
        low, high = self._display_freq_range
        low = max(MIN_DISPLAY_FREQ_HZ, float(low))
        high = max(low, float(high))
        if high <= low:
            return rect.bottom()
        if self._frequency_scale == "log":
            log_low = max(1e-9, low)
            fraction = (np.log(max(freq, log_low)) - np.log(log_low)) / max(1e-9, np.log(high) - np.log(log_low))
        else:
            fraction = (float(freq) - low) / max(1e-9, high - low)
        return rect.bottom() - int(round(float(fraction) * rect.height()))

    def _frequency_ticks(self, low: float, high: float) -> list[float]:
        low = max(MIN_DISPLAY_FREQ_HZ, float(low))
        high = max(low, float(high))
        if self._frequency_scale == "linear":
            return [low, (low + high) * 0.5, high]
        candidates: list[float] = []
        decade_min = int(np.floor(np.log10(low)))
        decade_max = int(np.ceil(np.log10(high)))
        for decade in range(decade_min, decade_max + 1):
            for base in (1.0, 2.0, 5.0):
                tick = base * (10.0**decade)
                if low <= tick <= high:
                    candidates.append(tick)
        ticks = [low, *candidates, high]
        unique = sorted({round(tick, 6) for tick in ticks})
        if len(unique) > 7:
            keep = [unique[0], unique[-1]]
            middle = unique[1:-1]
            stride = max(1, int(np.ceil(len(middle) / 5)))
            keep[1:1] = middle[::stride][:5]
            unique = sorted(keep)
        return [float(tick) for tick in unique]

    def _format_frequency_tick(self, freq: float) -> str:
        if freq >= 10.0:
            return f"{freq:.0f}"
        return f"{freq:g}"

    def _build_colormap(self, name: str) -> list[QColor]:
        cmap = colormaps[name]
        return [
            QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
            for rgba in cmap(np.linspace(0.0, 1.0, 256))
        ]


class ChannelSpectrogramDialog(QDialog):
    def __init__(
        self,
        time_seconds: np.ndarray,
        data: np.ndarray,
        *,
        sampling_rate: float,
        window_start_seconds: float,
        window_duration_seconds: float | None = None,
        total_duration_seconds: float | None = None,
        window_loader: Callable[[float, float], tuple[np.ndarray, np.ndarray] | None] | None = None,
        window_changed_callback: Callable[[float, float], None] | None = None,
        streaming: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spectrogram")
        self.resize(900, 620)
        self._time_seconds = np.asarray(time_seconds, dtype=float).reshape(-1)
        self._data = np.asarray(data)
        self._sampling_rate = float(sampling_rate)
        self._window_start = float(window_start_seconds)
        self._window_duration = self._infer_window_duration(window_duration_seconds)
        self._total_duration = total_duration_seconds
        self._window_loader = window_loader
        self._window_changed_callback = window_changed_callback
        self._streaming = bool(streaming)
        self._updating_controls = False
        self._updating_time_scroll = False
        self.viewer = ChannelSpectrogramWidget()

        layout = QVBoxLayout(self)
        content_layout = QHBoxLayout()
        sidebar = self._build_sidebar()
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.summary = QLabel(self._summary_text())
        self.summary.setStyleSheet("color: #b8c7da;")
        right_layout.addWidget(self.summary)
        right_layout.addWidget(self.viewer, 1)
        content_layout.addWidget(sidebar, 0)
        content_layout.addWidget(right_panel, 1)
        layout.addLayout(content_layout, 1)
        layout.addWidget(self._build_time_bar())
        self._install_navigation_filters()
        if self._channel_count() > 0:
            self.channel_combo.setCurrentIndex(0)
            self._channel_changed(0)
        self.set_streaming_mode(self._streaming)

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(170)
        panel.setMaximumWidth(220)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        channel_label = QLabel("Channel")
        channel_label.setStyleSheet("font-weight: 600;")
        self.channel_combo = QComboBox()
        for channel in range(self._channel_count()):
            self.channel_combo.addItem(f"ch {channel}", channel)
        self.channel_combo.currentIndexChanged.connect(self._channel_changed)

        window_label = QLabel("Spectrogram")
        window_label.setStyleSheet("font-weight: 600;")
        self.spectrogram_window = QDoubleSpinBox()
        self.spectrogram_window.setRange(0.001, max(0.001, self._window_duration))
        self.spectrogram_window.setDecimals(3)
        self.spectrogram_window.setSingleStep(0.05)
        self.spectrogram_window.setSuffix(" s")
        self.spectrogram_window.setValue(min(NEUROSCOPE_SPECTROGRAM_WINDOW_SECONDS, self.spectrogram_window.maximum()))
        self.spectrogram_window.valueChanged.connect(self._spectrogram_parameters_changed)

        range_label = QLabel("Frequency")
        range_label.setStyleSheet("font-weight: 600;")
        nyquist = max(MIN_DISPLAY_FREQ_HZ, min(MAX_DISPLAY_FREQ_HZ, self._sampling_rate * 0.5))
        self.freq_min = QDoubleSpinBox()
        self.freq_min.setRange(MIN_DISPLAY_FREQ_HZ, nyquist)
        self.freq_min.setDecimals(1)
        self.freq_min.setSuffix(" Hz")
        self.freq_min.setValue(min(NEUROSCOPE_SPECTROGRAM_FREQ_LOW_HZ, nyquist))
        self.freq_max = QDoubleSpinBox()
        self.freq_max.setRange(MIN_DISPLAY_FREQ_HZ, nyquist)
        self.freq_max.setDecimals(1)
        self.freq_max.setSuffix(" Hz")
        self.freq_max.setValue(min(NEUROSCOPE_SPECTROGRAM_FREQ_HIGH_HZ, nyquist))
        self.freq_step = QDoubleSpinBox()
        self.freq_step.setRange(0.1, max(0.1, nyquist))
        self.freq_step.setDecimals(1)
        self.freq_step.setSuffix(" Hz")
        self.freq_step.setValue(NEUROSCOPE_SPECTROGRAM_FREQ_STEP_HZ)
        self.freq_step.valueChanged.connect(self._spectrogram_parameters_changed)
        self.freq_min.valueChanged.connect(self._frequency_range_changed)
        self.freq_max.valueChanged.connect(self._frequency_range_changed)
        range_form = QFormLayout()
        range_form.addRow("Min", self.freq_min)
        range_form.addRow("Max", self.freq_max)
        range_form.addRow("Step", self.freq_step)

        cmap_label = QLabel("Colormap")
        cmap_label.setStyleSheet("font-weight: 600;")
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(list(WAVELET_COLORMAPS))
        self.cmap_combo.setCurrentText("viridis")
        self.cmap_combo.currentTextChanged.connect(self._colormap_changed)

        scale_label = QLabel("Freq scale")
        scale_label.setStyleSheet("font-weight: 600;")
        self.freq_scale_combo = QComboBox()
        self.freq_scale_combo.addItems(["Linear", "Log"])
        self.freq_scale_combo.setCurrentText("Linear")
        self.freq_scale_combo.currentTextChanged.connect(self._frequency_scale_changed)

        layout.addWidget(channel_label)
        layout.addWidget(self.channel_combo)
        layout.addWidget(window_label)
        layout.addWidget(self.spectrogram_window)
        layout.addWidget(range_label)
        layout.addLayout(range_form)
        layout.addWidget(scale_label)
        layout.addWidget(self.freq_scale_combo)
        layout.addWidget(cmap_label)
        layout.addWidget(self.cmap_combo)
        layout.addStretch(1)
        return panel

    def _build_time_bar(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
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
        for widget in [
            self.start_minutes,
            self.start_seconds,
            self.start_msec,
            self.duration_minutes,
            self.duration_seconds,
            self.duration_msec,
        ]:
            widget.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
            widget.valueChanged.connect(self._time_controls_changed)

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
        self._sync_time_controls()
        self._sync_time_scroll()
        return panel

    def _install_navigation_filters(self) -> None:
        for widget in [
            self.viewer,
            self.channel_combo,
            self.start_seconds,
            self.start_minutes,
            self.start_msec,
            self.duration_seconds,
            self.duration_minutes,
            self.duration_msec,
            self.freq_min,
            self.freq_max,
            self.freq_step,
            self.spectrogram_window,
            self.freq_scale_combo,
            self.cmap_combo,
            self.time_scroll,
        ]:
            widget.installEventFilter(self)

    def _channel_count(self) -> int:
        if self._data.ndim != 2:
            return 0
        return int(self._data.shape[1])

    def _infer_window_duration(self, fallback: float | None) -> float:
        if fallback is not None:
            return max(0.001, float(fallback))
        if self._time_seconds.size >= 2:
            return max(0.001, float(self._time_seconds[-1] - self._time_seconds[0]))
        return 1.0

    def _summary_text(self) -> str:
        low = self.freq_min.value() if hasattr(self, "freq_min") else 0.0
        high = self.freq_max.value() if hasattr(self, "freq_max") else MAX_DISPLAY_FREQ_HZ
        mode = "Streaming" if self._streaming else "Window"
        if self._time_seconds.size:
            start = float(self._time_seconds[0])
            stop = float(self._time_seconds[-1])
            return f"{mode}: {start:.3f}-{stop:.3f} s    NeuroScope STFT    Sampling rate: {self._sampling_rate:.3f} Hz    Frequency: {low:.0f}-{high:.0f} Hz"
        return f"{mode}    NeuroScope STFT    Sampling rate: {self._sampling_rate:.3f} Hz    Frequency: {low:.0f}-{high:.0f} Hz"

    def _channel_changed(self, index: int) -> None:
        if index < 0 or index >= self._channel_count():
            return
        self._set_current_channel_trace()

    def _set_current_channel_trace(self) -> None:
        index = self.channel_combo.currentIndex()
        if index < 0 or index >= self._channel_count():
            return
        self.viewer.set_trace(
            self._data[:, index],
            channel=index,
            sampling_rate=self._sampling_rate,
            window_start_seconds=self._window_start,
        )
        self._apply_frequency_range()

    def _frequency_range_changed(self) -> None:
        if self.freq_max.value() <= self.freq_min.value():
            sender = self.sender()
            if sender is self.freq_min:
                self.freq_max.setValue(min(self.freq_max.maximum(), self.freq_min.value() + 1.0))
            else:
                self.freq_min.setValue(max(self.freq_min.minimum(), self.freq_max.value() - 1.0))
        self._apply_frequency_range()
        self.summary.setText(self._summary_text())

    def _apply_frequency_range(self) -> None:
        self.viewer.set_frequency_range(self.freq_min.value(), self.freq_max.value())
        self.viewer.set_spectrogram_parameters(
            window_seconds=self.spectrogram_window.value(),
            freq_step_hz=self.freq_step.value(),
        )

    def _spectrogram_parameters_changed(self) -> None:
        self.viewer.set_spectrogram_parameters(
            window_seconds=self.spectrogram_window.value(),
            freq_step_hz=self.freq_step.value(),
        )
        self.summary.setText(self._summary_text())

    def _colormap_changed(self, name: str) -> None:
        self.viewer.set_colormap(name)

    def _frequency_scale_changed(self, scale: str) -> None:
        self.viewer.set_frequency_scale(scale)

    def _time_controls_changed(self) -> None:
        if self._updating_controls or self._streaming:
            return
        self._load_requested_window(self._window_start_controls_seconds(), self._window_duration_controls_seconds())

    def _window_start_controls_seconds(self) -> float:
        return self.start_minutes.value() * 60.0 + self.start_seconds.value() + self.start_msec.value() / 1000.0

    def _window_duration_controls_seconds(self) -> float:
        duration = (
            self.duration_minutes.value() * 60.0
            + self.duration_seconds.value()
            + self.duration_msec.value() / 1000.0
        )
        return max(0.001, duration)

    def _load_requested_window(self, start: float, duration: float) -> None:
        if self._window_loader is None:
            self._window_start = max(0.0, float(start))
            self._window_duration = max(0.001, float(duration))
            self._sync_time_scroll()
            self.summary.setText(self._summary_text())
            return
        start = max(0.0, float(start))
        duration = max(0.001, float(duration))
        total = self._total_duration
        if total is not None:
            duration = min(duration, max(0.001, float(total)))
            start = min(start, max(0.0, float(total) - duration))
        result = self._window_loader(start, duration)
        if result is None:
            return
        time, data = result
        if self._window_changed_callback is not None:
            self._window_changed_callback(start, duration)
        self.update_recording_window(
            time,
            data,
            window_start_seconds=start,
            window_duration_seconds=duration,
            streaming=False,
        )

    def update_recording_window(
        self,
        time_seconds: np.ndarray,
        data: np.ndarray,
        *,
        window_start_seconds: float,
        window_duration_seconds: float,
        streaming: bool,
    ) -> None:
        self._time_seconds = np.asarray(time_seconds, dtype=float).reshape(-1)
        self._data = np.asarray(data)
        self._window_start = max(0.0, float(window_start_seconds))
        self._window_duration = max(0.001, float(window_duration_seconds))
        self._sync_time_controls()
        self._streaming = bool(streaming)
        self._set_current_channel_trace()
        self.set_streaming_mode(self._streaming)
        self.summary.setText(self._summary_text())

    def _sync_time_controls(self) -> None:
        self._updating_controls = True
        self._set_split_time_controls(
            self._window_start,
            self.start_minutes,
            self.start_seconds,
            self.start_msec,
        )
        self._set_split_time_controls(
            self._window_duration,
            self.duration_minutes,
            self.duration_seconds,
            self.duration_msec,
        )
        self._updating_controls = False
        self._sync_time_scroll()

    def _time_scroll_changed(self, value: int) -> None:
        if self._updating_time_scroll or self._streaming or self._total_duration is None:
            return
        max_start = max(0.0, float(self._total_duration) - self._window_duration)
        start = max_start * (value / max(1, self.time_scroll.maximum()))
        self._load_requested_window(start, self._window_duration)

    def _sync_time_scroll(self) -> None:
        if not hasattr(self, "time_scroll"):
            return
        if self._total_duration is None or self._total_duration <= 0:
            self.time_scroll.setEnabled(False)
            return
        max_start = max(0.0, float(self._total_duration) - self._window_duration)
        self.time_scroll.setEnabled((not self._streaming) and max_start > 0)
        position = 0 if max_start <= 0 else round(self._window_start / max_start * self.time_scroll.maximum())
        page_step = round(self._window_duration / float(self._total_duration) * self.time_scroll.maximum())
        self._updating_time_scroll = True
        self.time_scroll.setPageStep(max(1, page_step))
        self.time_scroll.setValue(max(0, min(self.time_scroll.maximum(), position)))
        self._updating_time_scroll = False

    def _set_split_time_controls(
        self,
        value: float,
        minutes_widget: QSpinBox,
        seconds_widget: QSpinBox,
        msec_widget: QSpinBox,
    ) -> None:
        total_msec = max(0, int(round(float(value) * 1000.0)))
        minutes, rem = divmod(total_msec, 60000)
        seconds, msec = divmod(rem, 1000)
        minutes_widget.setValue(minutes)
        seconds_widget.setValue(seconds)
        msec_widget.setValue(msec)

    def set_streaming_mode(self, enabled: bool) -> None:
        self._streaming = bool(enabled)
        controls_enabled = not self._streaming
        for widget in [
            self.start_minutes,
            self.start_seconds,
            self.start_msec,
            self.duration_minutes,
            self.duration_seconds,
            self.duration_msec,
        ]:
            widget.setEnabled(controls_enabled)
        self._sync_time_scroll()
        self.summary.setText(self._summary_text())

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._handle_navigation_key(event):
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta:
                self._zoom_time_window(0.8 if delta > 0 else 1.25)
                event.accept()
                return
        super().wheelEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.KeyPress and self._handle_navigation_key(event):
            return True
        return super().eventFilter(watched, event)

    def _handle_navigation_key(self, event) -> bool:
        if self._streaming:
            return False
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_End:
            self._go_to_latest_window()
            return True
        if event.key() == Qt.Key.Key_Right:
            self._scroll_time(self._window_duration * 0.25)
            return True
        if event.key() == Qt.Key.Key_Left:
            self._scroll_time(-self._window_duration * 0.25)
            return True
        return False

    def _scroll_time(self, delta_seconds: float) -> None:
        total = self._total_duration
        max_start = float("inf") if total is None else max(0.0, float(total) - self._window_duration)
        start = max(0.0, min(max_start, self._window_start + float(delta_seconds)))
        self._load_requested_window(start, self._window_duration)

    def _zoom_time_window(self, factor: float) -> None:
        if self._streaming:
            return
        old_duration = self._window_duration
        new_duration = max(0.001, old_duration * float(factor))
        if self._total_duration is not None:
            new_duration = min(new_duration, max(0.001, float(self._total_duration)))
        center = self._window_start + old_duration * 0.5
        start = max(0.0, center - new_duration * 0.5)
        if self._total_duration is not None:
            start = min(start, max(0.0, float(self._total_duration) - new_duration))
        self._load_requested_window(start, new_duration)

    def _go_to_latest_window(self) -> None:
        if self._total_duration is None:
            return
        start = max(0.0, float(self._total_duration) - self._window_duration)
        self._load_requested_window(start, self._window_duration)
