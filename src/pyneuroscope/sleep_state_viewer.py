from __future__ import annotations

from collections.abc import Callable

from matplotlib import colormaps
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


SPECTROGRAM_COLORMAPS = ("viridis", "magma", "mako", "inferno", "jet")


class SleepStateViewer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state_timestamps = np.asarray([], dtype=float)
        self._metric_timestamps = np.asarray([], dtype=float)
        self._states = np.asarray([], dtype=float)
        self._sw = np.asarray([], dtype=float)
        self._emg = np.asarray([], dtype=float)
        self._thratio = np.asarray([], dtype=float)
        self._sw_threshold: float | None = None
        self._emg_threshold: float | None = None
        self._thratio_threshold: float | None = None
        self._spec = np.empty((0, 0), dtype=float)
        self._spec_log_scale = True
        self._spectrogram_colormap = "viridis"
        self._spectrogram_color_lut = self._build_colormap_lut("viridis")
        self._freqs = np.asarray([], dtype=float)
        self._spec_timestamps = np.asarray([], dtype=float)
        self._psd_cache_key: tuple | None = None
        self._psd_cache_image: QImage | None = None
        self._psd_value_range = (0.0, 1.0)
        self._sw_value_range = (0.0, 1.0)
        self._emg_value_range = (0.0, 1.0)
        self._thratio_value_range = (0.0, 1.0)
        self._window = (0.0, 1.0)
        self._selection: tuple[float, float] | None = None
        self._show_state_transitions = False
        self._on_selection: Callable[[float, float], None] | None = None
        self._on_reset_view: Callable[[], None] | None = None
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        self.setMinimumHeight(360)
        self.setMinimumWidth(520)
        self.setAutoFillBackground(True)

    def set_selection_callback(self, callback: Callable[[float, float], None]) -> None:
        self._on_selection = callback

    def set_reset_view_callback(self, callback: Callable[[], None]) -> None:
        self._on_reset_view = callback

    def set_data(
        self,
        *,
        state_timestamps: np.ndarray,
        metric_timestamps: np.ndarray,
        states: np.ndarray,
        sw: np.ndarray,
        emg: np.ndarray,
        thratio: np.ndarray,
        sw_threshold: float | None,
        emg_threshold: float | None,
        thratio_threshold: float | None,
        spec: np.ndarray,
        freqs: np.ndarray,
        spec_timestamps: np.ndarray,
        spec_log_scale: bool = True,
    ) -> None:
        self._state_timestamps = np.asarray(state_timestamps, dtype=float).reshape(-1)
        self._metric_timestamps = np.asarray(metric_timestamps, dtype=float).reshape(-1)
        self._states = np.asarray(states, dtype=float).reshape(-1)
        self._sw = np.asarray(sw, dtype=float).reshape(-1)
        self._emg = np.asarray(emg, dtype=float).reshape(-1)
        self._thratio = np.asarray(thratio, dtype=float).reshape(-1)
        self._sw_threshold = None if sw_threshold is None else float(sw_threshold)
        self._emg_threshold = None if emg_threshold is None else float(emg_threshold)
        self._thratio_threshold = None if thratio_threshold is None else float(thratio_threshold)
        self._spec = np.asarray(spec, dtype=float)
        self._spec_log_scale = bool(spec_log_scale)
        self._freqs = np.asarray(freqs, dtype=float).reshape(-1)
        self._spec_timestamps = np.asarray(spec_timestamps, dtype=float).reshape(-1)
        self._psd_cache_key = None
        self._psd_cache_image = None
        self._psd_value_range = self._compute_psd_value_range()
        self._sw_value_range = self._compute_value_range(self._sw, self._sw_threshold)
        self._emg_value_range = self._compute_value_range(self._emg, self._emg_threshold)
        self._thratio_value_range = self._compute_value_range(self._thratio, self._thratio_threshold)
        self.update()

    def clear_data(self) -> None:
        self.set_data(
            state_timestamps=np.asarray([], dtype=float),
            metric_timestamps=np.asarray([], dtype=float),
            states=np.asarray([], dtype=float),
            sw=np.asarray([], dtype=float),
            emg=np.asarray([], dtype=float),
            thratio=np.asarray([], dtype=float),
            sw_threshold=None,
            emg_threshold=None,
            thratio_threshold=None,
            spec=np.empty((0, 0), dtype=float),
            freqs=np.asarray([], dtype=float),
            spec_timestamps=np.asarray([], dtype=float),
            spec_log_scale=True,
        )
        self._selection = None

    def set_window(self, start: float, duration: float) -> None:
        self._window = (float(start), max(1e-6, float(duration)))
        self.update()

    def set_selection(self, selection: tuple[float, float] | None) -> None:
        self._selection = selection
        self.update()

    def set_spectrogram_colormap(self, name: str) -> None:
        clean = name.strip().lower()
        if clean not in SPECTROGRAM_COLORMAPS:
            clean = "viridis"
        if clean == self._spectrogram_colormap:
            return
        self._spectrogram_colormap = clean
        self._spectrogram_color_lut = self._build_colormap_lut(clean)
        self._psd_cache_key = None
        self._psd_cache_image = None
        self.update()

    def set_show_state_transitions(self, visible: bool) -> None:
        self._show_state_transitions = bool(visible)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor("#101216"))
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

            labels, rows = self._layout_rows()

            label_font = QFont()
            label_font.setPointSize(8)
            painter.setFont(label_font)
            for label, rect in zip(labels, rows):
                if label:
                    self._draw_rotated_label(painter, rect, label)
                painter.setPen(QPen(QColor("#4b515a")))
                painter.drawRect(rect)

            if self._state_timestamps.size == 0:
                painter.setPen(QPen(QColor("#8a9099")))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load sleep state to preview scoring")
                return

            start, duration = self._window
            stop = start + duration

            if self._show_state_transitions:
                state_row, transition_row, psd_row, sw_row, theta_row, emg_row = rows
                self._draw_state(painter, state_row, start, stop)
                self._draw_transition_row(painter, transition_row, start, stop)
            else:
                state_row, psd_row, sw_row, theta_row, emg_row = rows
                self._draw_state(painter, state_row, start, stop)
            self._draw_psd(painter, psd_row, start, stop)
            self._draw_trace(painter, sw_row, start, stop, self._sw, QColor("#f4f7fb"), self._sw_threshold)
            self._draw_trace(painter, theta_row, start, stop, self._thratio, QColor("#f4f7fb"), self._thratio_threshold)
            self._draw_trace(painter, emg_row, start, stop, self._emg, QColor("#f4f7fb"), self._emg_threshold)
            self._draw_selection(painter, rows, start, stop)
            self._draw_drag(painter, rows)
        finally:
            painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            self._drag_current = self._drag_start
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_start is not None:
            self._drag_current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._drag_start is None:
            return
        start_point = self._drag_start
        end_point = event.position().toPoint()
        self._drag_start = None
        self._drag_current = None
        if abs(end_point.x() - start_point.x()) >= 4:
            lo, hi = sorted((self._time_at_x(start_point.x()), self._time_at_x(end_point.x())))
            self._selection = (lo, hi)
            if self._on_selection is not None:
                self._on_selection(lo, hi)
        else:
            selected = self._select_state_episode_at(end_point)
            if not selected:
                self._selection = None
                if self._on_selection is not None:
                    self._on_selection(0.0, 0.0)
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._on_reset_view is not None:
            self._drag_start = None
            self._drag_current = None
            self._on_reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _layout_rows(self) -> tuple[list[str], list[QRect]]:
        width = max(1, self.width())
        height = max(1, self.height())
        margin_left = 58
        margin_right = 10
        margin_top = 24
        margin_bottom = 24
        label_gutter = 76
        plot_left = margin_left + label_gutter
        plot_right = width - margin_right
        plot_width = max(1, plot_right - plot_left)
        plot_area_height = max(1, height - margin_top - margin_bottom)
        gap = 14
        row_heights = [1.0, 2.0, 1.0, 1.0, 1.0]
        labels = ["", "Spectrogram", "SW", "Theta", "EMG"]
        if self._show_state_transitions:
            row_heights.insert(1, 0.45)
            labels.insert(1, "Transitions")
        total_gap = gap * max(0, len(row_heights) - 1)
        unit = max(1.0, (plot_area_height - total_gap) / sum(row_heights))
        rows: list[QRect] = []
        y = margin_top
        for ratio in row_heights:
            row_height = max(1, int(round(unit * ratio)))
            rows.append(QRect(plot_left, int(y), plot_width, row_height))
            y += row_height + gap
        return labels, rows

    def _plot_geometry(self) -> tuple[int, int]:
        margin_left = 58
        margin_right = 10
        label_gutter = 76
        return margin_left + label_gutter, max(margin_left + label_gutter + 1, self.width() - margin_right)

    def _x_at_time(self, timepoint: float, start: float, stop: float) -> int:
        plot_left, plot_right = self._plot_geometry()
        frac = (float(timepoint) - start) / max(1e-9, stop - start)
        return int(round(plot_left + np.clip(frac, 0.0, 1.0) * (plot_right - plot_left)))

    def _time_at_x(self, x: int) -> float:
        plot_left, plot_right = self._plot_geometry()
        start, duration = self._window
        frac = (float(x) - plot_left) / max(1.0, plot_right - plot_left)
        return start + np.clip(frac, 0.0, 1.0) * duration

    def x_fraction_at_x(self, x: int) -> float:
        plot_left, plot_right = self._plot_geometry()
        return float(np.clip((float(x) - plot_left) / max(1.0, plot_right - plot_left), 0.0, 1.0))

    def _window_mask(self, values: np.ndarray, start: float, stop: float, n: int) -> np.ndarray:
        return (values[:n] >= start) & (values[:n] <= stop)

    def _state_lane_rects(self, rect: QRect) -> list[tuple[int, QRect]]:
        lane_gap = 6
        lane_height = max(10, int((rect.height() - lane_gap * 2) / 3))
        out: list[tuple[int, QRect]] = []
        y = rect.top()
        for state_code in (1, 3, 5):
            out.append((state_code, QRect(rect.left(), y, rect.width(), lane_height)))
            y += lane_height + lane_gap
        return out

    def _select_state_episode_at(self, point: QPoint) -> bool:
        if self._state_timestamps.size == 0 or self._states.size == 0:
            return False
        _, rows = self._layout_rows()
        state_row = rows[0]
        clicked_lane: int | None = None
        for state_code, lane_rect in self._state_lane_rects(state_row):
            if lane_rect.contains(point):
                clicked_lane = state_code
                break
        if clicked_lane is None:
            return False

        n = min(self._state_timestamps.size, self._states.size)
        times = self._state_timestamps[:n]
        states = self._states[:n]
        timepoint = self._time_at_x(point.x())
        index = int(np.searchsorted(times, timepoint, side="right") - 1)
        index = int(np.clip(index, 0, n - 1))
        if int(states[index]) != clicked_lane:
            return False

        left = index
        while left > 0 and int(states[left - 1]) == clicked_lane:
            left -= 1
        right = index
        while right + 1 < n and int(states[right + 1]) == clicked_lane:
            right += 1

        start = float(times[left])
        window_start, window_duration = self._window
        window_stop = window_start + window_duration
        stop = float(times[right + 1]) if right + 1 < n else window_stop
        if stop <= start:
            return False
        self._selection = (start, stop)
        if self._on_selection is not None:
            self._on_selection(start, stop)
        return True

    def _draw_state(self, painter: QPainter, rect: QRect, start: float, stop: float) -> None:
        n = min(self._state_timestamps.size, self._states.size)
        if n == 0:
            return
        mask = self._window_mask(self._state_timestamps, start, stop, n)
        times = self._state_timestamps[:n][mask]
        states = self._states[:n][mask]
        if times.size == 0:
            return
        lanes = [
            ("Awake", 1, QColor("#26547C")),
            ("NREM", 3, QColor("#EF476F")),
            ("REM", 5, QColor("#FFD166")),
        ]
        lane_rects = [
            (label, state_code, color, lane_rect)
            for (label, state_code, color), (_, lane_rect) in zip(lanes, self._state_lane_rects(rect))
        ]

        label_font = QFont()
        label_font.setPointSize(8)
        painter.setFont(label_font)
        for label, state_code, color, lane_rect in lane_rects:
            painter.setPen(QPen(QColor("#b8c7da")))
            painter.drawText(
                10,
                lane_rect.top(),
                rect.left() - 20,
                lane_rect.height(),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            painter.setPen(QPen(QColor("#4b515a")))
            painter.drawRect(lane_rect)

        block_colors = {
            1: QColor("#26547C"),
            3: QColor("#EF476F"),
            5: QColor("#FFD166"),
        }
        compact_transition_mode = self._show_state_transitions
        for _, state_code, _, lane_rect in lane_rects:
            for idx, (timepoint, state) in enumerate(zip(times, states)):
                if int(state) != state_code:
                    continue
                x0 = self._x_at_time(timepoint, start, stop)
                next_time = times[idx + 1] if idx + 1 < times.size else stop
                x1 = self._x_at_time(next_time, start, stop)
                width = max(1, x1 - x0)
                if compact_transition_mode and width < 3:
                    continue
                painter.fillRect(QRect(x0, lane_rect.top() + 1, width, lane_rect.height() - 2), block_colors[state_code])

    def _draw_trace(
        self,
        painter: QPainter,
        rect: QRect,
        start: float,
        stop: float,
        values: np.ndarray,
        color: QColor,
        threshold: float | None,
    ) -> None:
        n = min(self._metric_timestamps.size, values.size)
        if n == 0:
            return
        mask = self._window_mask(self._metric_timestamps, start, stop, n)
        times = self._metric_timestamps[:n][mask]
        data = values[:n][mask]
        finite = np.isfinite(data)
        if times.size == 0 or not np.any(finite):
            return
        times = times[finite]
        data = data[finite]
        max_points = self._target_point_count(data.size, rect.width(), stop - start, kind="trace")
        step = max(1, data.size // max_points)
        times = times[::step]
        data = data[::step]
        if values is self._sw:
            lo, hi = self._sw_value_range
        elif values is self._emg:
            lo, hi = self._emg_value_range
        elif values is self._thratio:
            lo, hi = self._thratio_value_range
        else:
            lo, hi = self._compute_value_range(values[:n], threshold)
        self._draw_value_axis(painter, rect, [hi, (lo + hi) * 0.5, lo], lo, hi, formatter="{:.3g}")
        path = QPainterPath()
        for idx, (timepoint, value) in enumerate(zip(times, data)):
            x = self._x_at_time(timepoint, start, stop)
            frac = (float(value) - lo) / max(1e-9, hi - lo)
            y = int(round(rect.bottom() - np.clip(frac, 0.0, 1.0) * rect.height()))
            if idx == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.setPen(QPen(color, 1))
        painter.drawPath(path)
        if threshold is not None and np.isfinite(threshold):
            thr_frac = (float(threshold) - lo) / max(1e-9, hi - lo)
            y_thr = int(round(rect.bottom() - np.clip(thr_frac, 0.0, 1.0) * rect.height()))
            pen = QPen(QColor("#ff4d4f"))
            pen.setStyle(Qt.PenStyle.SolidLine)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(rect.left(), y_thr, rect.right(), y_thr)

    def _draw_psd(self, painter: QPainter, rect: QRect, start: float, stop: float) -> None:
        if self._spec.size == 0 or self._freqs.size == 0 or self._spec_timestamps.size == 0:
            return
        spec_cols = min(self._spec.shape[1], self._spec_timestamps.size)
        spec_t = self._spec_timestamps[:spec_cols]
        time_mask = (spec_t >= start) & (spec_t <= stop)
        freq_mask = (self._freqs >= 0.0) & (self._freqs <= 30.0)
        if not np.any(time_mask) or not np.any(freq_mask):
            return
        spec = self._spec[:, :spec_cols]
        data = self._spectrogram_display_data(spec[freq_mask][:, time_mask])
        if data.size == 0:
            return
        max_cols = self._target_point_count(data.shape[1], rect.width(), stop - start, kind="spectrogram")
        data, col_step = self._bin_spectrogram_axis(data, max_cols, axis=1)
        max_rows = 128
        data, row_step = self._bin_spectrogram_axis(data, max_rows, axis=0)
        lo, hi = self._psd_value_range
        self._draw_psd_axis(painter, rect)
        scaled = np.clip((data - lo) / max(1e-9, hi - lo), 0.0, 1.0)
        cache_key = (
            int(rect.width()),
            int(rect.height()),
            int(data.shape[0]),
            int(data.shape[1]),
            int(row_step),
            int(col_step),
            float(start),
            float(stop),
            float(lo),
            float(hi),
            self._spectrogram_colormap,
        )
        if self._psd_cache_key != cache_key or self._psd_cache_image is None:
            image = QImage(scaled.shape[1], scaled.shape[0], QImage.Format.Format_RGB32)
            color_index = np.asarray(np.clip(np.rint(np.flipud(scaled) * 255.0), 0, 255), dtype=np.uint8)
            for y in range(color_index.shape[0]):
                for x in range(color_index.shape[1]):
                    image.setPixelColor(x, y, self._spectrogram_color_lut[int(color_index[y, x])])
            self._psd_cache_key = cache_key
            self._psd_cache_image = image
        painter.drawImage(rect, self._psd_cache_image)

    def _build_colormap_lut(self, name: str) -> list[QColor]:
        if name != "mako":
            cmap = colormaps[name]
            return [
                QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
                for rgba in cmap(np.linspace(0.0, 1.0, 256))
            ]
        stops = (
            (0.0, QColor("#0b0405")),
            (0.18, QColor("#262136")),
            (0.36, QColor("#3b4267")),
            (0.55, QColor("#357ba3")),
            (0.76, QColor("#42b7b9")),
            (1.0, QColor("#def5e5")),
        )
        colors: list[QColor] = []
        for value in np.linspace(0.0, 1.0, 256):
            colors.append(self._interpolate_color(float(value), stops))
        return colors

    def _interpolate_color(self, value: float, stops: tuple[tuple[float, QColor], ...]) -> QColor:
        for (lo_v, lo_c), (hi_v, hi_c) in zip(stops[:-1], stops[1:]):
            if value <= hi_v:
                frac = (value - lo_v) / max(1e-9, hi_v - lo_v)
                return QColor(
                    int(lo_c.red() + frac * (hi_c.red() - lo_c.red())),
                    int(lo_c.green() + frac * (hi_c.green() - lo_c.green())),
                    int(lo_c.blue() + frac * (hi_c.blue() - lo_c.blue())),
                )
        return stops[-1][1]

    def _draw_value_axis(
        self,
        painter: QPainter,
        rect: QRect,
        ticks: list[float],
        lo: float,
        hi: float,
        *,
        formatter: str = "{:.3g}",
    ) -> None:
        if hi <= lo:
            return
        font = QFont()
        font.setPointSize(7)
        painter.setFont(font)
        tick_pen = QPen(QColor("#8a9099"))
        text_pen = QPen(QColor("#b8c7da"))
        for value in ticks:
            frac = (float(value) - lo) / max(1e-9, hi - lo)
            y = int(round(rect.bottom() - np.clip(frac, 0.0, 1.0) * rect.height()))
            painter.setPen(tick_pen)
            painter.drawLine(rect.left() - 4, y, rect.left(), y)
            painter.setPen(text_pen)
            painter.drawText(6, y - 8, rect.left() - 18, 16, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, formatter.format(value))

    def _draw_rotated_label(self, painter: QPainter, rect: QRect, label: str) -> None:
        painter.save()
        painter.setPen(QPen(QColor("#b8c7da")))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        label_x = rect.left() - 28
        label_y = rect.center().y()
        painter.translate(label_x, label_y)
        painter.rotate(-90)
        painter.drawText(QRect(-rect.height() // 2, -12, rect.height(), 24), Qt.AlignmentFlag.AlignCenter, label)
        painter.restore()

    def _draw_psd_axis(self, painter: QPainter, rect: QRect) -> None:
        freq_lo = 0.0
        freq_hi = 30.0
        ticks = [30.0, 15.0, 0.0]
        self._draw_value_axis(painter, rect, ticks, freq_lo, freq_hi, formatter="{:.0f}")

    def _transition_times(self, start: float, stop: float) -> list[float]:
        n = min(self._state_timestamps.size, self._states.size)
        if n < 2:
            return []
        transitions: list[float] = []
        times = self._state_timestamps[:n]
        states = self._states[:n]
        for index in range(1, n):
            prev_state = int(states[index - 1])
            curr_state = int(states[index])
            if curr_state != prev_state and start <= float(times[index]) <= stop:
                transitions.append(float(times[index]))
        return transitions

    def _draw_transition_row(self, painter: QPainter, rect: QRect, start: float, stop: float) -> None:
        transitions = self._transition_times(start, stop)
        if not transitions:
            return
        pen = QPen(QColor("#ffffff"))
        pen.setStyle(Qt.PenStyle.SolidLine)
        pen.setWidthF(0.75)
        pen.setCosmetic(True)
        painter.setPen(pen)
        y0 = rect.top() + 2
        y1 = rect.bottom() - 2
        for timepoint in transitions:
            x = self._x_at_time(timepoint, start, stop)
            painter.drawLine(x, y0, x, y1)

    def _compute_value_range(self, values: np.ndarray, threshold: float | None = None) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 1.0
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if threshold is not None and np.isfinite(threshold):
            lo = min(lo, float(threshold))
            hi = max(hi, float(threshold))
        if hi <= lo:
            pad = max(1e-6, abs(lo) * 0.05 + 1e-6)
            lo -= pad
            hi += pad
        return lo, hi

    def _compute_psd_value_range(self) -> tuple[float, float]:
        if self._spec.size == 0 or self._freqs.size == 0:
            return 0.0, 1.0
        freq_mask = (self._freqs >= 0.0) & (self._freqs <= 30.0)
        if not np.any(freq_mask):
            return 0.0, 1.0
        spec = np.asarray(self._spec[freq_mask], dtype=float)
        data = self._spectrogram_display_data(spec)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return 0.0, 1.0
        lo, hi = np.nanpercentile(finite, [2.0, 98.0])
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            hi = lo + 1e-9
        return lo, hi

    def _spectrogram_display_data(self, spec: np.ndarray) -> np.ndarray:
        data = np.asarray(spec, dtype=float)
        if data.size == 0:
            return np.empty(data.shape, dtype=float)
        if not self._spec_log_scale:
            return data
        valid = np.isfinite(data) & (data > 0)
        if not np.any(valid):
            return np.full(data.shape, np.nan, dtype=float)
        floor = max(float(np.nanpercentile(data[valid], 1.0)), np.finfo(np.float64).eps)
        return np.log10(np.maximum(data, floor))

    def _bin_spectrogram_axis(self, data: np.ndarray, target_size: int, *, axis: int) -> tuple[np.ndarray, int]:
        size = int(data.shape[axis])
        target = max(1, int(target_size))
        if size <= target:
            return data, 1
        step = int(np.ceil(size / target))
        usable = (size // step) * step
        if usable <= 0:
            return data, 1
        if axis == 1:
            main = data[:, :usable].reshape(data.shape[0], usable // step, step)
            reduced = np.nanmean(main, axis=2)
            tail = data[:, usable:]
            if tail.size:
                reduced = np.column_stack((reduced, np.nanmean(tail, axis=1)))
            return reduced, step
        main = data[:usable, :].reshape(usable // step, step, data.shape[1])
        reduced = np.nanmean(main, axis=1)
        tail = data[usable:, :]
        if tail.size:
            reduced = np.vstack((reduced, np.nanmean(tail, axis=0, keepdims=True)))
        return reduced, step

    def _target_point_count(self, available: int, width: int, duration: float, *, kind: str = "trace") -> int:
        pixel_target = max(2, int(width))
        if kind == "spectrogram":
            hz_target = max(2, int(max(duration, 1e-6) * 8.0))
            target = max(72, min(pixel_target * 3 // 4, hz_target, 256))
        else:
            hz_target = max(2, int(max(duration, 1e-6) * 10.0))
            target = max(pixel_target // 3, min(pixel_target * 3 // 4, hz_target, 600))
        return max(2, min(int(available), target))

    def _draw_selection(self, painter: QPainter, rows: list[QRect], start: float, stop: float) -> None:
        if self._selection is None:
            return
        lo, hi = self._selection
        x0 = self._x_at_time(lo, start, stop)
        x1 = self._x_at_time(hi, start, stop)
        painter.fillRect(QRect(min(x0, x1), rows[0].top(), abs(x1 - x0), rows[-1].bottom() - rows[0].top()), QColor(244, 185, 66, 45))

    def _draw_drag(self, painter: QPainter, rows: list[QRect]) -> None:
        if self._drag_start is None or self._drag_current is None:
            return
        x0 = self._drag_start.x()
        x1 = self._drag_current.x()
        painter.fillRect(QRect(min(x0, x1), rows[0].top(), abs(x1 - x0), rows[-1].bottom() - rows[0].top()), QColor(120, 160, 220, 55))
