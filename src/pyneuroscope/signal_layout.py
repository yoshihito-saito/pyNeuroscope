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
    shank: tuple[int, int] | None = None


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
    channel_shanks: dict[int, tuple[int, int]] | None = None,
) -> list[TraceLayoutItem]:
    items: list[TraceLayoutItem] = []
    positions = channel_geometry or {}
    shanks = channel_shanks or {}
    for group_index, group in enumerate(groups):
        channels = group.channels
        if shanks:
            # Show mapped NP2 contacts by probe and shank, from base (+y) to tip.
            # Channels without an NP2 site retain their XML order after them.
            original_order = {channel: index for index, channel in enumerate(channels)}
            channels = sorted(
                channels,
                key=lambda channel: (
                    (0, *shanks[channel], -positions[channel].y, positions[channel].x, channel)
                    if channel in shanks and channel in positions
                    else (1, 0, 0, original_order[channel], 0, channel)
                ),
            )
        for channel in channels:
            position = positions.get(channel) if channel in shanks else None
            items.append(
                TraceLayoutItem(
                    channel=channel,
                    group_index=group_index,
                    row=len(items),
                    column=0,
                    color=channel_colors.get(channel, default_color),
                    is_bad=channel in bad_channels,
                    x=position.x if position is not None else None,
                    y=position.y if position is not None else None,
                    shank=shanks.get(channel),
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
    channel_shanks: dict[int, tuple[int, int]] | None = None,
) -> list[TraceLayoutItem]:
    if channel_geometry:
        return _geometry_group_column_layout(
            groups, bad_channels, channel_colors, default_color, channel_geometry,
            physical_row_grids or {}, channel_shanks or {},
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
    channel_shanks: dict[int, tuple[int, int]],
) -> list[TraceLayoutItem]:
    items: list[TraceLayoutItem] = []
    column_offset = 0
    for group_index, group in enumerate(groups):
        positions = {
            channel: channel_geometry[channel]
            for channel in group.channels
            if channel in channel_geometry
        }
        shanks = sorted({channel_shanks[channel] for channel in group.channels
                         if channel in channel_shanks and channel in positions})
        if shanks:
            for shank_column, shank in enumerate(shanks):
                ordered = sorted(
                    (channel for channel in group.channels if channel_shanks.get(channel) == shank
                     and channel in positions),
                    key=lambda channel: (-positions[channel].y, positions[channel].x, channel),
                )
                for row, channel in enumerate(ordered):
                    position = positions[channel]
                    items.append(TraceLayoutItem(
                        channel=channel,
                        group_index=group_index,
                        row=row,
                        column=column_offset + shank_column,
                        color=channel_colors.get(channel, default_color),
                        is_bad=channel in bad_channels,
                        x=position.x,
                        y=position.y,
                        row_count=len(ordered),
                        shank=shank,
                    ))
            unmatched = [channel for channel in group.channels
                         if channel not in channel_shanks or channel not in positions]
            if unmatched:
                for row, channel in enumerate(unmatched):
                    position = positions.get(channel)
                    items.append(TraceLayoutItem(
                        channel=channel,
                        group_index=group_index,
                        row=row,
                        column=column_offset + len(shanks),
                        color=channel_colors.get(channel, default_color),
                        is_bad=channel in bad_channels,
                        x=position.x if position is not None else None,
                        y=position.y if position is not None else None,
                        row_count=len(unmatched),
                    ))
            column_offset += len(shanks) + bool(unmatched)
            continue
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
