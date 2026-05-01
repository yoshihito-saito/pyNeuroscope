from __future__ import annotations

from dataclasses import dataclass

from .models import ChannelGroup


@dataclass(frozen=True)
class TraceLayoutItem:
    channel: int
    group_index: int
    row: int
    column: int
    color: str
    is_bad: bool


def single_column_layout(
    groups: list[ChannelGroup],
    bad_channels: set[int],
    channel_colors: dict[int, str],
    *,
    default_color: str = "#1f77b4",
) -> list[TraceLayoutItem]:
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
) -> list[TraceLayoutItem]:
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


def unassigned_channels(n_channels: int, groups: list[ChannelGroup]) -> list[int]:
    assigned = {channel for group in groups for channel in group.channels}
    return [channel for channel in range(n_channels) if channel not in assigned]
