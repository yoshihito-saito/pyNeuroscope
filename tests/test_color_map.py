import pytest

from pyneuroscope.color_map import (
    COLOR_MAP_NAMES,
    ColorMapError,
    apply_channel_overrides,
    color_by_channel_index,
    color_by_group,
    color_by_group_channel_index,
    color_by_group_sequence,
    palette_from_name,
    spring_palette,
)
from pyneuroscope.models import ChannelGroup


def test_color_by_channel_index_covers_all_channels() -> None:
    mapping = color_by_channel_index(12)

    assert set(mapping) == set(range(12))
    assert mapping[0].startswith("#")
    assert mapping[0] == "#ff00ff"
    assert mapping[11] == "#ffff00"


def test_spring_palette_runs_from_magenta_to_yellow() -> None:
    assert spring_palette(3) == ["#ff00ff", "#ff8080", "#ffff00"]


def test_palette_from_name_interpolates_common_color_maps() -> None:
    assert palette_from_name("spring", 3) == ["#ff00ff", "#ff8080", "#ffff00"]
    winter = palette_from_name("winter", 4)
    assert len(winter) == 4
    assert winter[0] == "#0000ff"
    assert winter[-1] == "#00ff80"
    assert palette_from_name("cool", 2) == ["#00ffff", "#ff00ff"]
    assert palette_from_name("hot", 2) == ["#0b0000", "#ffffff"]
    assert palette_from_name("plasma", 2) == ["#0d0887", "#f0f921"]


def test_color_map_names_keep_single_rainbow_variant() -> None:
    assert "rainbow" in COLOR_MAP_NAMES
    assert "camp rainbow" not in COLOR_MAP_NAMES


def test_color_by_group_preserves_group_assignment() -> None:
    groups = [ChannelGroup("a", [2, 0]), ChannelGroup("b", [1])]

    mapping = color_by_group(4, groups)

    assert mapping[2] == mapping[0]
    assert mapping[1] != mapping[0]
    assert mapping[3] == "#808080"


def test_color_by_group_channel_index_restarts_palette_per_group() -> None:
    groups = [ChannelGroup("a", [2, 0]), ChannelGroup("b", [1, 3])]

    mapping = color_by_group_channel_index(5, groups, ["#111111", "#222222"])

    assert mapping[2] == "#111111"
    assert mapping[0] == "#222222"
    assert mapping[1] == "#111111"
    assert mapping[3] == "#222222"
    assert mapping[4] == "#808080"


def test_color_by_group_sequence_follows_group_order_not_channel_number() -> None:
    groups = [ChannelGroup("a", [4, 2]), ChannelGroup("b", [0, 3])]

    mapping = color_by_group_sequence(5, groups, ["#111111", "#222222", "#333333", "#444444"])

    assert mapping[4] == "#111111"
    assert mapping[2] == "#222222"
    assert mapping[0] == "#333333"
    assert mapping[3] == "#444444"
    assert mapping[1] == "#808080"


def test_applies_overrides_and_rejects_invalid_channel() -> None:
    base = color_by_channel_index(3)

    assert apply_channel_overrides(base, {1: "#ABCDEF"}, 3)[1] == "#abcdef"
    with pytest.raises(ColorMapError):
        apply_channel_overrides(base, {3: "#abcdef"}, 3)
