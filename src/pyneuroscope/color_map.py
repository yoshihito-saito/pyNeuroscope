from __future__ import annotations

import re

from .models import ChannelGroup

DEFAULT_PALETTE = [
    "#ff00ff",
    "#ff1ee0",
    "#ff3dc1",
    "#ff5ba3",
    "#ff7a84",
    "#ff9966",
    "#ffb747",
    "#ffd629",
    "#fff40a",
    "#ffff00",
]

COLOR_MAPS = {
    "white": ["#cfd5df"],
    "black": ["#394150"],
    "rainbow": ["#2d5bff", "#00a4ff", "#00d084", "#d8e52d", "#ff9d00", "#ff3d3d", "#b032ff"],
    "Greys": ["#f7f7f7", "#d9d9d9", "#969696", "#525252", "#111111"],
    "Purples": ["#fcfbfd", "#dadaeb", "#9e9ac8", "#6a51a3", "#3f007d"],
    "Blues": ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    "Greens": ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"],
    "Oranges": ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"],
    "Reds": ["#fff5f0", "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"],
    "spring": ["#ff00ff", "#ff40bf", "#ff8080", "#ffbf40", "#ffff00"],
    "summer": ["#008066", "#40a666", "#80cc66", "#bff266", "#ffff66"],
    "autumn": ["#ff0000", "#ff4000", "#ff8000", "#ffbf00", "#ffff00"],
    "winter": ["#0000ff", "#0040df", "#0080bf", "#00bf9f", "#00ff80"],
    "cool": ["#00ffff", "#40bfff", "#8080ff", "#bf40ff", "#ff00ff"],
    "hot": ["#0b0000", "#800000", "#ff0000", "#ffbf00", "#ffffff"],
    "plasma": ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"],
    "gray": ["#202020", "#606060", "#a0a0a0", "#e0e0e0"],
}

COLOR_MAP_NAMES = list(COLOR_MAPS)

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


class ColorMapError(ValueError):
    """Raised when a channel color map is invalid."""


def spring_palette(size: int) -> list[str]:
    if size <= 0:
        raise ColorMapError("palette size must be positive")
    if size == 1:
        return ["#ff00ff"]
    colors: list[str] = []
    for index in range(size):
        t = index / (size - 1)
        red = 255
        green = round(255 * t)
        blue = round(255 * (1.0 - t))
        colors.append(f"#{red:02x}{green:02x}{blue:02x}")
    return colors


def palette_from_name(name: str, size: int) -> list[str]:
    if size <= 0:
        raise ColorMapError("palette size must be positive")
    anchors = COLOR_MAPS.get(name)
    if anchors is None:
        raise ColorMapError(f"Unknown color map: {name}")
    if size == 1:
        return [validate_color(anchors[0])]
    if len(anchors) == 1:
        return [validate_color(anchors[0])] * size

    colors: list[str] = []
    last_anchor = len(anchors) - 1
    for index in range(size):
        position = index / (size - 1) * last_anchor
        left = int(position)
        right = min(last_anchor, left + 1)
        fraction = position - left
        colors.append(_interpolate_hex(anchors[left], anchors[right], fraction))
    return colors


def _interpolate_hex(left: str, right: str, fraction: float) -> str:
    left = validate_color(left)
    right = validate_color(right)
    rgb_left = [int(left[index : index + 2], 16) for index in (1, 3, 5)]
    rgb_right = [int(right[index : index + 2], 16) for index in (1, 3, 5)]
    rgb = [
        round(start + (end - start) * fraction)
        for start, end in zip(rgb_left, rgb_right, strict=True)
    ]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def validate_color(color: str) -> str:
    if not isinstance(color, str) or not _HEX_COLOR.match(color):
        raise ColorMapError(f"Invalid color: {color}")
    return color.lower()


def color_by_channel_index(n_channels: int, palette: list[str] | None = None) -> dict[int, str]:
    if n_channels <= 0:
        raise ColorMapError("n_channels must be positive")
    colors = palette or spring_palette(n_channels)
    return {channel: validate_color(colors[channel % len(colors)]) for channel in range(n_channels)}


def color_by_group(
    n_channels: int,
    groups: list[ChannelGroup],
    palette: list[str] | None = None,
    *,
    unassigned_color: str = "#808080",
) -> dict[int, str]:
    mapping = {channel: validate_color(unassigned_color) for channel in range(n_channels)}
    colors = palette or spring_palette(max(1, len(groups)))
    for group_index, group in enumerate(groups):
        color = validate_color(colors[group_index % len(colors)])
        for channel in group.channels:
            if channel < 0 or channel >= n_channels:
                raise ColorMapError(f"Channel {channel} is outside 0..{n_channels - 1}")
            mapping[channel] = color
    return mapping


def color_by_group_channel_index(
    n_channels: int,
    groups: list[ChannelGroup],
    palette: list[str] | None = None,
    *,
    unassigned_color: str = "#808080",
) -> dict[int, str]:
    mapping = {channel: validate_color(unassigned_color) for channel in range(n_channels)}
    max_group_size = max((len(group.channels) for group in groups), default=1)
    colors = palette or spring_palette(max(1, max_group_size))
    for group in groups:
        for index, channel in enumerate(group.channels):
            if channel < 0 or channel >= n_channels:
                raise ColorMapError(f"Channel {channel} is outside 0..{n_channels - 1}")
            mapping[channel] = validate_color(colors[index % len(colors)])
    return mapping


def color_by_group_sequence(
    n_channels: int,
    groups: list[ChannelGroup],
    palette: list[str] | None = None,
    *,
    unassigned_color: str = "#808080",
) -> dict[int, str]:
    mapping = {channel: validate_color(unassigned_color) for channel in range(n_channels)}
    ordered_channels = [channel for group in groups for channel in group.channels]
    colors = palette or spring_palette(max(1, len(ordered_channels)))
    for index, channel in enumerate(ordered_channels):
        if channel < 0 or channel >= n_channels:
            raise ColorMapError(f"Channel {channel} is outside 0..{n_channels - 1}")
        mapping[channel] = validate_color(colors[index % len(colors)])
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
