from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import ChannelGroup
from .probe_geometry import ProbeSitePosition


@dataclass(frozen=True)
class TraceLayoutItem:
    channel: int
    group_index: int
    row: int
    column: int
    color: str
    is_bad: bool
    x: float | None = None
    y: float | None = None
    row_count: int | None = None


@dataclass(frozen=True)
class PhysicalRowGrid:
    top_y: float
    pitch: float
    row_count: int

    def row_for_y(self, y: float) -> int:
        # Physical probes use +y toward the base, shown at the top of the map.
        return max(0, min(self.row_count - 1, round((self.top_y - y) / self.pitch)))


def physical_row_grid(sites: Iterable[ProbeSitePosition], *, max_rows: int = 4096) -> PhysicalRowGrid | None:
    """Use one vertical scale for all contacts on physical probe layouts."""
    y_values = sorted({site.y for site in sites})
    if not y_values:
        return None
    span = y_values[-1] - y_values[0]
    gaps = [upper - lower for lower, upper in zip(y_values, y_values[1:]) if upper > lower]
    pitch = min(gaps) if gaps else 1.0
    if span > 0:
        pitch = max(pitch, span / max(1, max_rows - 1))
    return PhysicalRowGrid(y_values[-1], pitch, min(max_rows, round(span / pitch) + 1))


def single_column_layout(
    groups: list[ChannelGroup],
    bad_channels: set[int],
    channel_colors: dict[int, str],
    *,
    default_color: str = "#1f77b4",
    channel_geometry: dict[int, ProbeSitePosition] | None = None,
) -> list[TraceLayoutItem]:
    _ = channel_geometry
    items: list[TraceLayoutItem] = []
    for group_index, group in enumerate(groups):
        for channel in group.channels:
            items.append(
                TraceLayoutItem(
                    channel=channel,
                    group_index=group_index,
                    row=len(items),
                    column=0,
                    color=channel_colors.get(channel, default_color),
                    is_bad=channel in bad_channels,
                )
            )
    return items


def group_column_layout(
    groups: list[ChannelGroup],
    bad_channels: set[int],
    channel_colors: dict[int, str],
    *,
    default_color: str = "#1f77b4",
    channel_geometry: dict[int, ProbeSitePosition] | None = None,
    physical_row_grids: dict[int, PhysicalRowGrid] | None = None,
) -> list[TraceLayoutItem]:
    if channel_geometry:
        return _geometry_group_column_layout(
            groups, bad_channels, channel_colors, default_color, channel_geometry,
            physical_row_grids or {},
        )

    items: list[TraceLayoutItem] = []
    for group_index, group in enumerate(groups):
        for row, channel in enumerate(group.channels):
            items.append(
                TraceLayoutItem(
                    channel=channel,
                    group_index=group_index,
                    row=row,
                    column=group_index,
                    color=channel_colors.get(channel, default_color),
                    is_bad=channel in bad_channels,
                )
            )
    return items


def _geometry_group_column_layout(
    groups: list[ChannelGroup],
    bad_channels: set[int],
    channel_colors: dict[int, str],
    default_color: str,
    channel_geometry: dict[int, ProbeSitePosition],
    physical_row_grids: dict[int, PhysicalRowGrid],
) -> list[TraceLayoutItem]:
    items: list[TraceLayoutItem] = []
    column_offset = 0
    for group_index, group in enumerate(groups):
        positions = {
            channel: channel_geometry[channel]
            for channel in group.channels
            if channel in channel_geometry
        }
        if not positions:
            for row, channel in enumerate(group.channels):
                items.append(
                    TraceLayoutItem(
                        channel=channel,
                        group_index=group_index,
                        row=row,
                        column=column_offset,
                        color=channel_colors.get(channel, default_color),
                        is_bad=channel in bad_channels,
                    )
                )
            column_offset += 1
            continue

        x_values = _rank_values(position.x for position in positions.values())
        physical_grid = physical_row_grids.get(group_index)
        y_values = _rank_values(position.y for position in positions.values()) if physical_grid is None else {}
        missing_slots = [slot for slot, channel in enumerate(group.channels) if channel not in positions]
        row_count = physical_grid.row_count if physical_grid is not None else len(y_values)
        row_count = max(row_count, max(missing_slots, default=-1) + 1)
        for slot, channel in enumerate(group.channels):
            position = positions.get(channel)
            if position is None:
                row = slot
                column = column_offset
                x = None
                y = None
            else:
                row = physical_grid.row_for_y(position.y) if physical_grid is not None else y_values[position.y]
                column = column_offset + x_values[position.x]
                x = position.x
                y = position.y
            items.append(
                TraceLayoutItem(
                    channel=channel,
                    group_index=group_index,
                    row=row,
                    column=column,
                    color=channel_colors.get(channel, default_color),
                    is_bad=channel in bad_channels,
                    x=x,
                    y=y,
                    row_count=row_count,
                )
            )
        column_offset += max(1, len(x_values))
    return items


def unassigned_channels(n_channels: int, groups: list[ChannelGroup]) -> list[int]:
    assigned = {channel for group in groups for channel in group.channels}
    return [channel for channel in range(n_channels) if channel not in assigned]


def _rank_values(values: Iterable[float]) -> dict[float, int]:
    return {value: index for index, value in enumerate(sorted(set(values)))}
