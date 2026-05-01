from __future__ import annotations

from typing import Sequence

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

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
    ) -> None:
        self._time_seconds = time_seconds
        self._data = data
        self._layout_items = list(layout_items)
        self._vertical_scale = max(0.05, float(vertical_scale))
        self._row_spacing = max(0.25, float(row_spacing))
        self._update_content_height()
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
        margins = 48
        rows = self._display_row_count()
        base_available = max(1, self._viewport_height - margins)
        content_height = margins + base_available * self._row_spacing
        if rows > 0 and self._row_spacing > 1.0:
            content_height = max(content_height, margins + rows * 8 * self._row_spacing)
        self.setMinimumHeight(int(max(self._viewport_height, content_height)))
        self.updateGeometry()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101216"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        if self._data is None or self._time_seconds is None or not self._layout_items:
            painter.setPen(QPen(QColor("#8a9099")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Select a DAT file to preview traces")
            return

        data = self._data
        width = max(1, self.width())
        height = max(1, self.height())
        margin_left = 70
        margin_right = 12
        margin_top = 24
        margin_bottom = 24
        label_gutter = 54
        visible_items = self._visible_layout_items()
        columns = max(item.column for item in self._layout_items) + 1
        rows_by_column = {
            column: max((item.row for item in visible_items if item.column == column), default=-1) + 1
            for column in range(columns)
        }
        column_width = max(24.0, (width - margin_left - margin_right) / columns)

        painter.setPen(QPen(QColor("#2d333d")))
        for column in range(columns):
            x0 = margin_left + column * column_width
            painter.drawLine(int(x0 + label_gutter), margin_top, int(x0 + label_gutter), height - margin_bottom)

        trace_width = max(8.0, column_width - label_gutter - 8)
        max_points = max(2, int(trace_width))
        start_index, end_index = self._visible_sample_bounds(data.shape[0])
        visible_data = data[start_index:end_index]
        step = max(1, visible_data.shape[0] // max_points)
        sampled_data = visible_data[::step]
        x_values = np.linspace(0, 1, sampled_data.shape[0], dtype=np.float64)

        for item in visible_items:
            if item.channel < 0 or item.channel >= data.shape[1]:
                continue
            rows = max(1, rows_by_column[item.column])
            trace_height = max(1.0, (height - margin_top - margin_bottom) / rows)
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
            if trace_height >= 5.0 and label_gutter >= 28:
                font = QFont()
                font.setPointSize(max(6, min(9, int(trace_height * 0.7))))
                painter.setFont(font)
                painter.setPen(QPen(QColor("#78818d") if item.is_bad else QColor("#b8c7da")))
                painter.drawText(
                    int(column_x0 + 2),
                    int(y_center - trace_height * 0.45),
                    label_gutter - 6,
                    max(8, int(trace_height * 0.9)),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    f"ch {item.channel}",
                )

        if self._x_range != (0.0, 1.0):
            painter.setPen(QPen(QColor("#d6dde8")))
            painter.drawText(8, height - 8, f"zoom {self._x_range[0]:.3f}-{self._x_range[1]:.3f}")

        if self._drag_start is not None and self._drag_current is not None:
            rect = QRect(self._drag_start, self._drag_current).normalized()
            painter.fillRect(rect, QColor(120, 160, 220, 55))
            painter.setPen(QPen(QColor("#7aa7ff")))
            painter.drawRect(rect)

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
