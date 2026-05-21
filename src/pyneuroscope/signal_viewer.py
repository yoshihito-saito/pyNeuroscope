from __future__ import annotations

from typing import Sequence

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from .models import SignalEventOverlay, SignalSpikeOverlay
from .signal_layout import TraceLayoutItem


class SignalViewer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._time_seconds: np.ndarray | None = None
        self._data: np.ndarray | None = None
        self._layout_items: list[TraceLayoutItem] = []
        self._vertical_scale = 1.0
        self._row_spacing = 1.0
        self._viewport_height = 360
        self._x_range = (0.0, 1.0)
        self._visible_rows: tuple[int, int] | None = None
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._show_channel_labels = True
        self._epoch_boundaries = np.asarray([], dtype=np.float64)
        self._spike_overlays: list[SignalSpikeOverlay] = []
        self._show_spikes = False
        self._spikes_below = False
        self._spike_waveforms = False
        self._event_overlays: list[SignalEventOverlay] = []
        self._background_mode = "black"
        self.setMinimumHeight(360)
        self.setMinimumWidth(520)
        self.setAutoFillBackground(True)

    def set_traces(
        self,
        time_seconds: np.ndarray | None,
        data: np.ndarray | None,
        layout_items: Sequence[TraceLayoutItem],
        *,
        vertical_scale: float = 1.0,
        row_spacing: float = 1.0,
        show_channel_labels: bool = True,
        epoch_boundaries: Sequence[float] | None = None,
    ) -> None:
        self._time_seconds = time_seconds
        self._data = data
        self._layout_items = list(layout_items)
        self._vertical_scale = max(0.05, float(vertical_scale))
        self._row_spacing = max(0.25, float(row_spacing))
        self._show_channel_labels = bool(show_channel_labels)
        self._epoch_boundaries = np.asarray(epoch_boundaries if epoch_boundaries is not None else [], dtype=np.float64).reshape(-1)
        self._update_content_height()
        self.update()

    def set_spike_overlays(
        self,
        spikes: Sequence[SignalSpikeOverlay],
        *,
        show: bool = False,
        below: bool = False,
        show_waveforms: bool = False,
    ) -> None:
        self._spike_overlays = list(spikes)
        self._show_spikes = bool(show)
        self._spikes_below = bool(below)
        self._spike_waveforms = bool(show_waveforms)
        self._update_content_height()
        self.update()

    def set_event_overlays(self, events: Sequence[SignalEventOverlay]) -> None:
        self._event_overlays = list(events)
        self._update_content_height()
        self.update()

    def set_background_mode(self, mode: str) -> None:
        self._background_mode = "white" if str(mode).lower() == "white" else "black"
        self.update()

    def set_viewport_height(self, height: int) -> None:
        self._viewport_height = max(1, int(height))
        self._update_content_height()

    def reset_time_zoom(self) -> None:
        self._x_range = (0.0, 1.0)
        self._visible_rows = None
        self._update_content_height()
        self.update()

    def _update_content_height(self) -> None:
        margins = 48 + self._bottom_overlay_height()
        rows = self._display_row_count()
        base_available = max(1, self._viewport_height - margins)
        content_height = margins + base_available * self._row_spacing
        if rows > 0 and self._row_spacing > 1.0:
            content_height = max(content_height, margins + rows * 8 * self._row_spacing)
        self.setMinimumHeight(int(max(self._viewport_height, content_height)))
        self.updateGeometry()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._color("background"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        if self._data is None or self._time_seconds is None or not self._layout_items:
            painter.setPen(QPen(self._color("muted")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Select a DAT file to preview traces")
            return

        data = self._data
        width = max(1, self.width())
        height = max(1, self.height())
        margin_left = 50
        margin_right = 10
        margin_top = 24
        margin_bottom = 24
        label_gutter = 38
        bottom_overlay_height = self._bottom_overlay_height()
        trace_bottom = max(margin_top + 24, height - margin_bottom - bottom_overlay_height)
        visible_items = self._visible_layout_items()
        columns = max(item.column for item in self._layout_items) + 1
        rows_by_column = {
            column: max((item.row for item in visible_items if item.column == column), default=-1) + 1
            for column in range(columns)
        }
        column_width = max(24.0, (width - margin_left - margin_right) / columns)

        painter.setPen(QPen(self._color("grid")))
        for column in range(columns):
            x0 = margin_left + column * column_width
            painter.drawLine(int(x0 + label_gutter), margin_top, int(x0 + label_gutter), height - margin_bottom)

        trace_width = max(8.0, column_width - label_gutter - 8)
        max_points = max(2, int(trace_width))
        start_index, end_index = self._visible_sample_bounds(data.shape[0])
        visible_data = data[start_index:end_index]
        visible_time = self._time_seconds[start_index:end_index]
        step = max(1, visible_data.shape[0] // max_points)
        sampled_data = visible_data[::step]
        x_values = np.linspace(0, 1, sampled_data.shape[0], dtype=np.float64)
        item_geometries: dict[int, tuple[float, float, float, float, float, float]] = {}

        if visible_time.size >= 2:
            self._draw_signal_event_overlays(
                painter,
                visible_time,
                columns,
                margin_left,
                column_width,
                label_gutter,
                trace_width,
                margin_top,
                trace_bottom,
                below=False,
            )

        for item in visible_items:
            if item.channel < 0 or item.channel >= data.shape[1]:
                continue
            rows = max(1, rows_by_column[item.column])
            trace_height = max(1.0, (trace_bottom - margin_top) / rows)
            column_x0 = margin_left + item.column * column_width
            trace_x0 = column_x0 + label_gutter
            y_center = margin_top + (item.row + 0.5) * trace_height
            trace = sampled_data[:, item.channel].astype(np.float64)
            if trace.size == 0:
                continue
            centered = trace - float(np.median(trace))
            peak = float(np.percentile(np.abs(centered), 98)) or 1.0
            normalized = centered / peak
            normalized = normalized * self._vertical_scale
            item_geometries[item.channel] = (trace_x0, trace_width, y_center, trace_height, float(np.median(trace)), peak)

            path = QPainterPath()
            path.moveTo(QPointF(trace_x0, y_center - normalized[0] * trace_height * 0.35))
            for x_norm, y_norm in zip(x_values[1:], normalized[1:]):
                x = trace_x0 + x_norm * trace_width
                y = y_center - y_norm * trace_height * 0.35
                path.lineTo(QPointF(x, y))

            pen = QPen(QColor("#5f6670") if item.is_bad else QColor(item.color))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawPath(path)
            if self._show_channel_labels and trace_height >= 5.0 and label_gutter >= 28:
                font = QFont()
                font.setPointSize(max(6, min(9, int(trace_height * 0.7))))
                painter.setFont(font)
                painter.setPen(QPen(self._color("muted") if item.is_bad else self._color("label")))
                painter.drawText(
                    int(column_x0 + 2),
                    int(y_center - trace_height * 0.45),
                    label_gutter - 6,
                    max(8, int(trace_height * 0.9)),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    f"ch {item.channel}",
                )

        if visible_time.size >= 2 and self._epoch_boundaries.size:
            self._draw_epoch_boundaries(
                painter,
                visible_time,
                columns,
                margin_left,
                column_width,
                label_gutter,
                trace_width,
                margin_top,
                trace_bottom,
            )

        if visible_time.size >= 2 and self._show_spikes and self._spike_overlays:
            if self._spikes_below:
                spike_top, spike_bottom = self._below_spike_bounds(trace_bottom)
                self._draw_spike_raster(
                    painter,
                    visible_time,
                    columns,
                    margin_left,
                    column_width,
                    label_gutter,
                    trace_width,
                    spike_top,
                    spike_bottom,
                )
                self._draw_signal_event_overlays(
                    painter,
                    visible_time,
                    columns,
                    margin_left,
                    column_width,
                    label_gutter,
                    trace_width,
                    spike_top,
                    spike_bottom,
                    below=False,
                    label_events=False,
                )
            else:
                self._draw_spike_trace_overlays(painter, visible_time, visible_data, item_geometries)

        if visible_time.size >= 2 and any(event.below for event in self._event_overlays):
            top, bottom = self._below_event_bounds(trace_bottom)
            self._draw_signal_event_overlays(
                painter,
                visible_time,
                columns,
                margin_left,
                column_width,
                label_gutter,
                trace_width,
                top,
                bottom,
                below=True,
            )

        if self._x_range != (0.0, 1.0):
            painter.setPen(QPen(self._color("text")))
            painter.drawText(8, height - 8, f"zoom {self._x_range[0]:.3f}-{self._x_range[1]:.3f}")

        if self._drag_start is not None and self._drag_current is not None:
            rect = QRect(self._drag_start, self._drag_current).normalized()
            painter.fillRect(rect, QColor(120, 160, 220, 55))
            painter.setPen(QPen(QColor("#7aa7ff")))
            painter.drawRect(rect)

    def _color(self, role: str) -> QColor:
        if self._background_mode == "white":
            colors = {
                "background": "#ffffff",
                "grid": "#d5dbe3",
                "muted": "#697280",
                "label": "#17202b",
                "text": "#111827",
            }
        else:
            colors = {
                "background": "#101216",
                "grid": "#2d333d",
                "muted": "#8a9099",
                "label": "#b8c7da",
                "text": "#d6dde8",
            }
        return QColor(colors.get(role, colors["text"]))

    def _overlay_color(self, color: str) -> QColor:
        qcolor = QColor(color)
        if not qcolor.isValid():
            return self._color("text")
        red, green, blue = qcolor.red(), qcolor.green(), qcolor.blue()
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        if self._background_mode == "white" and luminance > 145:
            factor = 0.48
            return QColor(int(red * factor), int(green * factor), int(blue * factor))
        if self._background_mode == "black" and luminance < 80:
            return QColor(
                min(255, int(red + (255 - red) * 0.35)),
                min(255, int(green + (255 - green) * 0.35)),
                min(255, int(blue + (255 - blue) * 0.35)),
            )
        return qcolor

    def _bottom_overlay_height(self) -> int:
        return self._below_spike_height() + self._below_gap_height() + self._below_event_height()

    def _below_spike_bounds(self, overlay_top: float) -> tuple[float, float]:
        top = overlay_top
        return top, top + self._below_spike_height()

    def _below_event_bounds(self, overlay_top: float) -> tuple[float, float]:
        top = overlay_top + self._below_spike_height() + self._below_gap_height()
        return top, top + self._below_event_height()

    def _below_gap_height(self) -> int:
        if self._below_spike_height() > 0 and self._below_event_height() > 0:
            return self._below_overlay_gap()
        return 0

    def _below_spike_height(self) -> int:
        if not (self._show_spikes and self._spikes_below and self._spike_overlays):
            return 0
        return max(150, min(280, 18 + int(round(len(self._spike_overlays) * 3.6))))

    def _spike_raster_height(self) -> int:
        return self._below_spike_height()

    def _below_overlay_gap(self) -> int:
        return 16

    def _below_event_height(self) -> int:
        count = sum(1 for event in self._event_overlays if event.below)
        return 42 * count

    def _effective_column_count(self) -> int:
        if not self._layout_items:
            return 1
        return max(item.column for item in self._layout_items) + 1

    def _draw_spike_trace_overlays(
        self,
        painter: QPainter,
        visible_time: np.ndarray,
        data: np.ndarray,
        item_geometries: dict[int, tuple[float, float, float, float, float, float]],
    ) -> None:
        t0 = float(visible_time[0])
        t1 = float(visible_time[-1])
        if t1 <= t0:
            return
        for unit in self._spike_overlays:
            if unit.channel is None or unit.channel not in item_geometries or unit.channel >= data.shape[1]:
                continue
            times = _times_in_window(unit.times, t0, t1)
            if times.size == 0:
                continue
            trace_x0, trace_width, y_center, trace_height, median, peak = item_geometries[unit.channel]
            pen = QPen(self._overlay_color(unit.color))
            pen.setWidth(1)
            painter.setPen(pen)
            if self._spike_waveforms:
                waveform_pen = QPen(self._overlay_color(unit.color))
                waveform_pen.setWidth(1)
                painter.setPen(waveform_pen)
                for time in times[:1200]:
                    self._draw_spike_waveform_segment(
                        painter,
                        visible_time,
                        data[:, unit.channel].astype(float),
                        float(time),
                        trace_x0,
                        trace_width,
                        y_center,
                        trace_height,
                        median,
                        peak,
                    )
                painter.setPen(pen)
            trace = data[:, unit.channel].astype(float)
            samples = _samples_at_times(visible_time, trace, times)
            y_values = y_center - ((samples - median) / peak) * self._vertical_scale * trace_height * 0.35
            for time, y_value in zip(times[:4000], y_values[:4000]):
                x = trace_x0 + ((float(time) - t0) / (t1 - t0)) * trace_width
                painter.drawLine(QPointF(x, y_value - 3.75), QPointF(x, y_value + 3.75))

    def _draw_spike_waveform_segment(
        self,
        painter: QPainter,
        visible_time: np.ndarray,
        trace: np.ndarray,
        spike_time: float,
        trace_x0: float,
        trace_width: float,
        y_center: float,
        trace_height: float,
        median: float,
        peak: float,
    ) -> None:
        t0 = float(visible_time[0])
        t1 = float(visible_time[-1])
        lo = spike_time - 0.0008
        hi = spike_time + 0.0008
        left = int(np.searchsorted(visible_time, lo, side="left"))
        right = int(np.searchsorted(visible_time, hi, side="right"))
        if right - left < 2:
            return
        times = visible_time[left:right]
        values = trace[left:right]
        path = QPainterPath()
        for index, (time, value) in enumerate(zip(times, values)):
            x = trace_x0 + ((float(time) - t0) / (t1 - t0)) * trace_width
            y = y_center - ((float(value) - median) / peak) * self._vertical_scale * trace_height * 0.35
            if index == 0:
                path.moveTo(QPointF(x, y))
            else:
                path.lineTo(QPointF(x, y))
        painter.drawPath(path)

    def _draw_spike_raster(
        self,
        painter: QPainter,
        visible_time: np.ndarray,
        columns: int,
        margin_left: float,
        column_width: float,
        label_gutter: float,
        trace_width: float,
        top: float,
        bottom: float,
    ) -> None:
        t0 = float(visible_time[0])
        t1 = float(visible_time[-1])
        if t1 <= t0 or bottom <= top:
            return
        painter.setPen(QPen(self._color("muted")))
        painter.drawText(
            QRect(0, int(top), max(1, int(margin_left)), max(1, int(bottom - top))),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            "spikes",
        )
        painter.save()
        painter.setClipRect(
            QRect(
                int(margin_left),
                int(top),
                max(1, int(columns * column_width)),
                max(1, int(bottom - top)),
            )
        )
        units_by_column = self._spike_units_by_column(columns)
        global_row_height = self._spike_raster_global_row_height(units_by_column, bottom - top)
        for column, units in units_by_column.items():
            if not units:
                continue
            row_height = global_row_height
            column_raster_height = row_height * len(units)
            column_top = top + max(0.0, (bottom - top - column_raster_height) * 0.5)
            lane_left = margin_left + column * column_width + label_gutter
            painter.setPen(QPen(self._color("grid")))
            painter.drawLine(QPointF(lane_left, top), QPointF(lane_left, bottom))
            for row, unit in enumerate(units):
                times = _times_in_window(unit.times, t0, t1)
                if times.size == 0:
                    continue
                pen = QPen(self._overlay_color(unit.color))
                pen.setWidth(1 if row_height < 2.5 else 2)
                painter.setPen(pen)
                y = column_top + (row + 0.5) * row_height
                for time in times[:10000]:
                    fraction = (float(time) - t0) / (t1 - t0)
                    x = lane_left + fraction * trace_width
                    half_height = self._compressed_raster_tick_half_height(row_height)
                    painter.drawLine(QPointF(x, y - half_height), QPointF(x, y + half_height))
        painter.restore()

    def _spike_raster_global_row_height(
        self,
        units_by_column: dict[int, list[SignalSpikeOverlay]],
        height: float,
    ) -> float:
        max_units = max((len(units) for units in units_by_column.values()), default=1)
        return max(1.0, height / max(1, max_units))

    def _compressed_raster_y(self, row: int, count: int, height: float) -> float:
        if count <= 1:
            return height * 0.5
        if height <= count:
            return min(max(0.5, row + 0.5), max(0.5, height - 0.5))
        return (row + 0.5) * (height / count)

    def _compressed_raster_tick_half_height(self, row_height: float) -> float:
        if row_height < 2.5:
            return 0.45
        return row_height * 0.38

    def _spike_units_by_column(self, columns: int) -> dict[int, list[SignalSpikeOverlay]]:
        if columns <= 1:
            return {0: list(self._spike_overlays)}
        channel_to_column = {item.channel: item.column for item in self._layout_items}
        grouped = {column: [] for column in range(columns)}
        for unit in self._spike_overlays:
            column = channel_to_column.get(unit.channel)
            if column is None:
                column = 0
            grouped.setdefault(column, []).append(unit)
        return grouped

    def _draw_signal_event_overlays(
        self,
        painter: QPainter,
        visible_time: np.ndarray,
        columns: int,
        margin_left: float,
        column_width: float,
        label_gutter: float,
        trace_width: float,
        top: float,
        bottom: float,
        *,
        below: bool,
        label_events: bool = True,
    ) -> None:
        t0 = float(visible_time[0])
        t1 = float(visible_time[-1])
        events = [event for event in self._event_overlays if event.below == below]
        if t1 <= t0 or not events:
            return
        lanes = max(1, len(events)) if below else 1
        lane_height = max(10.0, (bottom - top) / lanes)
        for event_index, event in enumerate(events):
            lane_top = top + event_index * lane_height if below else top
            lane_bottom = min(bottom, lane_top + lane_height)
            color = QColor(event.color)
            if below and label_events:
                painter.setPen(QPen(self._color("label")))
                painter.drawText(
                    QRect(0, int(lane_top), max(1, int(margin_left)), max(1, int(lane_bottom - lane_top))),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    event.name,
                )
            painter.save()
            painter.setClipRect(
                QRect(
                    int(margin_left),
                    int(lane_top),
                    max(1, int(columns * column_width)),
                    max(1, int(lane_bottom - lane_top)),
                )
            )
            if event.show_intervals:
                fill = QColor(color)
                fill.setAlpha(96 if below else 70)
                intervals = event.timestamps
                mask = (intervals[:, 1] >= t0) & (intervals[:, 0] <= t1)
                for start, end in intervals[mask][:2000]:
                    left_fraction = max(0.0, (float(start) - t0) / (t1 - t0))
                    right_fraction = min(1.0, (float(end) - t0) / (t1 - t0))
                    if right_fraction < left_fraction:
                        continue
                    for column in range(columns):
                        x0 = margin_left + column * column_width + label_gutter + left_fraction * trace_width
                        x1 = margin_left + column * column_width + label_gutter + right_fraction * trace_width
                        painter.fillRect(
                            QRect(int(x0), int(lane_top), max(1, int(x1 - x0)), max(1, int(lane_bottom - lane_top))),
                            fill,
                        )
            if event.show_peaks and event.peaks is not None:
                peak_color = QColor(color)
                peak_color.setAlpha(240)
                pen = QPen(peak_color)
                pen.setWidth(2 if below else 1)
                painter.setPen(pen)
                peaks = _times_in_window(event.peaks, t0, t1)
                for peak in peaks[:4000]:
                    fraction = (float(peak) - t0) / (t1 - t0)
                    for column in range(columns):
                        x = margin_left + column * column_width + label_gutter + fraction * trace_width
                        painter.drawLine(QPointF(x, lane_top), QPointF(x, lane_bottom))
            painter.restore()

    def _draw_epoch_boundaries(
        self,
        painter: QPainter,
        visible_time: np.ndarray,
        columns: int,
        margin_left: float,
        column_width: float,
        label_gutter: float,
        trace_width: float,
        top: float,
        bottom: float,
    ) -> None:
        start_time = float(visible_time[0])
        end_time = float(visible_time[-1])
        if end_time <= start_time:
            return
        pen = QPen(QColor(255, 255, 255, 255))
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        for boundary in self._epoch_boundaries:
            if boundary <= start_time or boundary >= end_time:
                continue
            fraction = (float(boundary) - start_time) / (end_time - start_time)
            for column in range(columns):
                x = margin_left + column * column_width + label_gutter + fraction * trace_width
                painter.drawLine(QPointF(x, top), QPointF(x, bottom))

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
        end = event.position().toPoint()
        start = self._drag_start
        self._drag_start = None
        self._drag_current = None
        if abs(end.x() - start.x()) >= 8:
            x_bounds = self._x_bounds_for_selection(start.x(), end.x())
            left, right = x_bounds if x_bounds is not None else (0.0, 1.0)
            if right - left > 0.002:
                current_left, current_right = self._x_range
                width = current_right - current_left
                self._x_range = (current_left + left * width, current_left + right * width)
        if abs(end.y() - start.y()) >= 8:
            row_bounds = self._row_bounds_for_selection(start.y(), end.y())
            if row_bounds is not None:
                self._visible_rows = row_bounds
                self._update_content_height()
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_time_zoom()

    def _max_rows(self) -> int:
        if not self._layout_items:
            return 1
        columns = max(item.column for item in self._layout_items) + 1
        return max(
            max((item.row for item in self._layout_items if item.column == column), default=-1) + 1
            for column in range(columns)
        )

    def _display_row_count(self) -> int:
        if self._visible_rows is None:
            return self._max_rows()
        start, end = self._visible_rows
        return max(1, end - start)

    def _visible_layout_items(self) -> list[TraceLayoutItem]:
        if self._visible_rows is None:
            return self._layout_items
        start, end = self._visible_rows
        return [
            TraceLayoutItem(
                channel=item.channel,
                group_index=item.group_index,
                row=item.row - start,
                column=item.column,
                color=item.color,
                is_bad=item.is_bad,
            )
            for item in self._layout_items
            if start <= item.row < end
        ]

    def _visible_sample_bounds(self, n_samples: int) -> tuple[int, int]:
        if n_samples <= 1:
            return 0, n_samples
        left, right = self._x_range
        start = max(0, min(n_samples - 1, int(round(left * (n_samples - 1)))))
        end = max(start + 2, min(n_samples, int(round(right * (n_samples - 1))) + 1))
        return start, end

    def _x_fraction_for_position(self, x_position: int) -> float:
        regions = self._trace_x_regions()
        if not regions:
            return 0.0
        best = min(regions, key=lambda region: min(abs(x_position - region[0]), abs(x_position - region[1])))
        return self._fraction_in_region(x_position, best)

    def _x_bounds_for_selection(self, x1: int, x2: int) -> tuple[float, float] | None:
        regions = self._trace_x_regions()
        if not regions:
            return None
        left_px, right_px = sorted((x1, x2))
        start_region = self._region_containing_x(x1, regions)
        if start_region is not None:
            left = self._fraction_in_region(left_px, start_region)
            right = self._fraction_in_region(right_px, start_region)
            return tuple(sorted((left, right)))

        fractions: list[float] = []
        for region in regions:
            overlap = min(right_px, region[1]) - max(left_px, region[0])
            if overlap >= 4:
                fractions.append(self._fraction_in_region(max(left_px, region[0]), region))
                fractions.append(self._fraction_in_region(min(right_px, region[1]), region))
        if fractions:
            return min(fractions), max(fractions)

        if len(regions) == 1:
            best_region = regions[0]
        else:
            center = (left_px + right_px) / 2.0
            best_region = min(
                regions,
                key=lambda region: 0.0
                if region[0] <= center <= region[1]
                else min(abs(center - region[0]), abs(center - region[1])),
            )
        left = self._fraction_in_region(left_px, best_region)
        right = self._fraction_in_region(right_px, best_region)
        return tuple(sorted((left, right)))

    def _region_containing_x(
        self,
        x_position: float,
        regions: list[tuple[float, float]],
    ) -> tuple[float, float] | None:
        for region in regions:
            if region[0] <= x_position <= region[1]:
                return region
        return None

    def _trace_x_regions(self) -> list[tuple[float, float]]:
        if not self._layout_items:
            return []
        width = max(1, self.width())
        margin_left = 70
        margin_right = 12
        label_gutter = 54
        columns = max(item.column for item in self._layout_items) + 1
        column_width = max(24.0, (width - margin_left - margin_right) / columns)
        trace_width = max(8.0, column_width - label_gutter - 8)
        return [
            (
                margin_left + column * column_width + label_gutter,
                margin_left + column * column_width + label_gutter + trace_width,
            )
            for column in range(columns)
        ]

    def _fraction_in_region(self, x_position: float, region: tuple[float, float]) -> float:
        left, right = region
        width = max(1.0, right - left)
        return max(0.0, min(1.0, (x_position - left) / width))

    def _row_bounds_for_selection(self, y1: int, y2: int) -> tuple[int, int] | None:
        if not self._layout_items:
            return None
        height = max(1, self.height())
        margin_top = 24
        margin_bottom = 24
        rows = self._display_row_count()
        trace_height = max(1.0, (height - margin_top - margin_bottom) / max(1, rows))
        top = max(0.0, min(y1, y2) - margin_top)
        bottom = max(0.0, max(y1, y2) - margin_top)
        selected_start = int(top // trace_height)
        selected_end = int(bottom // trace_height) + 1
        selected_start = max(0, min(rows - 1, selected_start))
        selected_end = max(selected_start + 1, min(rows, selected_end))
        base_start = self._visible_rows[0] if self._visible_rows is not None else 0
        absolute_start = base_start + selected_start
        absolute_end = base_start + selected_end
        if absolute_end - absolute_start >= self._max_rows():
            return None
        return absolute_start, absolute_end


def _times_in_window(times: np.ndarray, start: float, end: float) -> np.ndarray:
    arr = np.asarray(times, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    left = int(np.searchsorted(arr, start, side="left"))
    right = int(np.searchsorted(arr, end, side="right"))
    return arr[left:right]


def _samples_at_times(time: np.ndarray, trace: np.ndarray, times: np.ndarray) -> np.ndarray:
    if time.size == 0 or trace.size == 0 or times.size == 0:
        return np.asarray([], dtype=float)
    indices = np.searchsorted(time, times, side="left")
    indices = np.clip(indices, 0, trace.size - 1)
    previous = np.clip(indices - 1, 0, trace.size - 1)
    choose_previous = np.abs(time[previous] - times) < np.abs(time[indices] - times)
    indices = np.where(choose_previous, previous, indices)
    return trace[indices].astype(float)
