import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from pyneuroscope.spectrogram_viewer import (
    NEUROSCOPE_SPECTROGRAM_FREQ_HIGH_HZ,
    NEUROSCOPE_SPECTROGRAM_FREQ_LOW_HZ,
    NEUROSCOPE_SPECTROGRAM_FREQ_STEP_HZ,
    NEUROSCOPE_SPECTROGRAM_WINDOW_SECONDS,
    SPECTROGRAM_OVERLAP_FRACTION,
    WAVELET_COLORMAPS,
    WAVELET_DISPLAY_HIGH_PERCENTILE,
    WAVELET_DISPLAY_LOW_PERCENTILE,
    ChannelSpectrogramDialog,
    ChannelSpectrogramWidget,
)


def test_channel_spectrogram_widget_computes_frequency_time_grid() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    sampling_rate = 1000.0
    time = np.arange(0.0, 2.0, 1.0 / sampling_rate)
    trace = np.sin(2.0 * np.pi * 20.0 * time)
    widget = ChannelSpectrogramWidget()

    widget.set_trace(trace, channel=0, sampling_rate=sampling_rate, window_start_seconds=3.0)

    assert widget._spec_db.size > 0
    assert widget._freqs[0] == NEUROSCOPE_SPECTROGRAM_FREQ_LOW_HZ
    assert widget._freqs[-1] == NEUROSCOPE_SPECTROGRAM_FREQ_HIGH_HZ
    widget.set_frequency_range(10.0, 120.0)
    assert widget._display_freq_range == (10.0, 120.0)
    assert widget._freqs[0] == 10.0
    assert widget._freqs[-1] == 120.0
    assert widget._times[0] >= 3.0
    assert widget._colormap_name == "viridis"
    assert widget._frequency_scale == "linear"


def test_channel_spectrogram_widget_display_range_uses_robust_percentiles() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    widget = ChannelSpectrogramWidget()
    widget._freqs = np.asarray([1.0, 10.0])
    widget._spec_db = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 1000.0],
        ],
        dtype=float,
    )
    widget._display_freq_range = (1.0, 10.0)
    assert widget._compute_display_value_range() == tuple(
        np.percentile(
            widget._spec_db,
            [WAVELET_DISPLAY_LOW_PERCENTILE, WAVELET_DISPLAY_HIGH_PERCENTILE],
        )
    )


def test_channel_spectrogram_widget_uses_neuroscope_stft_settings() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    widget = ChannelSpectrogramWidget()
    sampling_rate = 1000.0
    time = np.arange(0.0, 2.0, 1.0 / sampling_rate)
    trace = np.sin(2.0 * np.pi * 20.0 * time)

    freqs, times, spec = widget._compute_neuroscope_spectrogram(trace, sampling_rate)

    assert freqs.size > 0
    assert times.size > 0
    assert spec.shape == (freqs.size, times.size)
    assert freqs[0] == NEUROSCOPE_SPECTROGRAM_FREQ_LOW_HZ
    assert freqs[1] - freqs[0] == NEUROSCOPE_SPECTROGRAM_FREQ_STEP_HZ
    assert freqs[-1] == NEUROSCOPE_SPECTROGRAM_FREQ_HIGH_HZ
    assert widget._spectrogram_segment_size(trace.size, sampling_rate) == round(
        sampling_rate * NEUROSCOPE_SPECTROGRAM_WINDOW_SECONDS
    )


def test_channel_spectrogram_widget_uses_95_percent_overlap() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    widget = ChannelSpectrogramWidget()

    assert widget._spectrogram_overlap_size(200) == round(200 * SPECTROGRAM_OVERLAP_FRACTION)
    assert widget._spectrogram_fft_size(200, 1000.0) >= 500


def test_channel_spectrogram_widget_resamples_frequency_axis_on_log_scale() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    widget = ChannelSpectrogramWidget()
    freqs = np.asarray([1.0, 10.0, 100.0], dtype=float)
    data = np.asarray([[1.0], [10.0], [100.0]], dtype=float)

    resampled = widget._resample_frequency_axis_log(data, freqs, 3)

    assert np.allclose(resampled.reshape(-1), [1.0, 10.0, 100.0])


class KeyEvent:
    def __init__(self, key: int, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
        self._key = key
        self._modifiers = modifiers

    def key(self) -> int:
        return self._key

    def modifiers(self):
        return self._modifiers


def test_channel_spectrogram_dialog_uses_channel_dropdown_and_frequency_controls() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    sampling_rate = 1000.0
    time = np.arange(0.0, 1.0, 1.0 / sampling_rate)
    data = np.column_stack([np.sin(2.0 * np.pi * (10.0 + channel) * time) for channel in range(32)])

    dialog = ChannelSpectrogramDialog(
        time,
        data,
        sampling_rate=sampling_rate,
        window_start_seconds=0.0,
    )

    assert dialog.channel_combo.count() == 32
    assert dialog.channel_combo.currentIndex() == 0
    assert dialog.viewer._channel == 0
    dialog.channel_combo.setCurrentIndex(1)
    assert dialog.viewer._channel == 1
    assert dialog.freq_min.value() == NEUROSCOPE_SPECTROGRAM_FREQ_LOW_HZ
    assert dialog.freq_max.value() == NEUROSCOPE_SPECTROGRAM_FREQ_HIGH_HZ
    assert dialog.freq_step.value() == NEUROSCOPE_SPECTROGRAM_FREQ_STEP_HZ
    assert dialog.spectrogram_window.value() == NEUROSCOPE_SPECTROGRAM_WINDOW_SECONDS
    assert dialog.freq_scale_combo.currentText() == "Linear"
    assert dialog.viewer._frequency_scale == "linear"
    assert [dialog.cmap_combo.itemText(i) for i in range(dialog.cmap_combo.count())] == list(WAVELET_COLORMAPS)
    assert dialog.cmap_combo.currentText() == "viridis"
    assert "NeuroScope STFT" in dialog.summary.text()
    assert "Frequency: 1-300 Hz" in dialog.summary.text()
    dialog.freq_max.setValue(120.0)
    assert dialog.viewer._display_freq_range == (1.0, 120.0)
    assert "Frequency: 1-120 Hz" in dialog.summary.text()
    dialog.cmap_combo.setCurrentText("plasma")
    assert dialog.viewer._colormap_name == "plasma"
    dialog.freq_scale_combo.setCurrentText("Log")
    assert dialog.viewer._frequency_scale == "log"


def test_channel_spectrogram_dialog_loads_new_window_from_internal_controls() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    sampling_rate = 1000.0
    time = np.arange(0.0, 1.0, 1.0 / sampling_rate)
    data = np.column_stack([np.sin(2.0 * np.pi * 10.0 * time)])
    calls: list[tuple[float, float]] = []
    synced: list[tuple[float, float]] = []

    def loader(start: float, duration: float):
        calls.append((start, duration))
        loaded_time = np.arange(start, start + duration, 1.0 / sampling_rate)
        loaded_data = np.column_stack([np.sin(2.0 * np.pi * 20.0 * loaded_time)])
        return loaded_time, loaded_data

    dialog = ChannelSpectrogramDialog(
        time,
        data,
        sampling_rate=sampling_rate,
        window_start_seconds=0.0,
        window_duration_seconds=1.0,
        total_duration_seconds=10.0,
        window_loader=loader,
        window_changed_callback=lambda start, duration: synced.append((start, duration)),
    )

    dialog.start_seconds.setValue(2)

    assert calls[-1] == (2.0, 1.0)
    assert synced[-1] == (2.0, 1.0)
    assert dialog._window_start == 2.0
    assert dialog.viewer._times[0] >= 2.0

    assert dialog._handle_navigation_key(KeyEvent(Qt.Key.Key_Right))
    assert calls[-1] == (2.25, 1.0)

    dialog.time_scroll.setValue(dialog.time_scroll.maximum() // 2)
    assert calls[-1][0] == 4.5


def test_channel_spectrogram_dialog_streaming_update_disables_time_controls() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    sampling_rate = 1000.0
    time = np.arange(0.0, 1.0, 1.0 / sampling_rate)
    data = np.column_stack([np.sin(2.0 * np.pi * 10.0 * time)])
    dialog = ChannelSpectrogramDialog(
        time,
        data,
        sampling_rate=sampling_rate,
        window_start_seconds=0.0,
        window_duration_seconds=1.0,
        streaming=True,
    )

    assert not dialog.start_seconds.isEnabled()
    assert not dialog.time_scroll.isEnabled()
    assert "Streaming" in dialog.summary.text()

    new_time = np.arange(5.0, 6.0, 1.0 / sampling_rate)
    new_data = np.column_stack([np.sin(2.0 * np.pi * 30.0 * new_time)])
    dialog.update_recording_window(
        new_time,
        new_data,
        window_start_seconds=5.0,
        window_duration_seconds=1.0,
        streaming=True,
    )

    assert dialog.start_seconds.value() == 5
    assert dialog.viewer._times[0] >= 5.0


def test_channel_spectrogram_dialog_places_window_controls_in_bottom_time_bar() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    sampling_rate = 1000.0
    time = np.arange(0.0, 1.0, 1.0 / sampling_rate)
    data = np.column_stack([np.sin(2.0 * np.pi * 10.0 * time)])
    dialog = ChannelSpectrogramDialog(
        time,
        data,
        sampling_rate=sampling_rate,
        window_start_seconds=65.432,
        window_duration_seconds=2.5,
        total_duration_seconds=100.0,
    )

    assert dialog.start_minutes.value() == 1
    assert dialog.start_seconds.value() == 5
    assert dialog.start_msec.value() == 432
    assert dialog.duration_minutes.value() == 0
    assert dialog.duration_seconds.value() == 2
    assert dialog.duration_msec.value() == 500
    assert dialog.time_scroll.isEnabled()
