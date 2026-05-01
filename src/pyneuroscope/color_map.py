from __future__ import annotations

import re

from .models import ChannelGroup

DEFAULT_PALETTE = [
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


class ColorMapError(ValueError):
    """Raised when a channel color map is invalid."""


def validate_color(color: str) -> str:
    if not isinstance(color, str) or not _HEX_COLOR.match(color):
        raise ColorMapError(f"Invalid color: {color}")
    return color.lower()


def color_by_channel_index(n_channels: int, palette: list[str] | None = None) -> dict[int, str]:
    if n_channels <= 0:
        raise ColorMapError("n_channels must be positive")
    colors = palette or DEFAULT_PALETTE
    return {channel: validate_color(colors[channel % len(colors)]) for channel in range(n_channels)}


def color_by_group(
    n_channels: int,
    groups: list[ChannelGroup],
    palette: list[str] | None = None,
    *,
    unassigned_color: str = "#808080",
) -> dict[int, str]:
    mapping = {channel: validate_color(unassigned_color) for channel in range(n_channels)}
    colors = palette or DEFAULT_PALETTE
    for group_index, group in enumerate(groups):
        color = validate_color(colors[group_index % len(colors)])
        for channel in group.channels:
            if channel < 0 or channel >= n_channels:
                raise ColorMapError(f"Channel {channel} is outside 0..{n_channels - 1}")
            mapping[channel] = color
    return mapping


def apply_channel_overrides(
    base: dict[int, str],
    overrides: dict[int, str],
    n_channels: int,
) -> dict[int, str]:
    result = dict(base)
    for channel, color in overrides.items():
        if channel < 0 or channel >= n_channels:
            raise ColorMapError(f"Channel {channel} is outside 0..{n_channels - 1}")
        result[channel] = validate_color(color)
    return result
