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
) -> list[TraceLayoutItem]:
    if channel_geometry:
        return _geometry_group_column_layout(groups, bad_channels, channel_colors, default_color, channel_geometry)

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
        y_values = _rank_values(position.y for position in positions.values())
        for slot, channel in enumerate(group.channels):
            position = positions.get(channel)
            if position is None:
                row = slot
                column = column_offset
                x = None
                y = None
            else:
                row = y_values[position.y]
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
                )
            )
        column_offset += max(1, len(x_values))
    return items


def unassigned_channels(n_channels: int, groups: list[ChannelGroup]) -> list[int]:
    assigned = {channel for group in groups for channel in group.channels}
    return [channel for channel in range(n_channels) if channel not in assigned]


def _rank_values(values: Iterable[float]) -> dict[float, int]:
    return {value: index for index, value in enumerate(sorted(set(values)))}
