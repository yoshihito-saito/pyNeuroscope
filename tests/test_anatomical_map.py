from pathlib import Path

from pyneuroscope.anatomical_map import (
    build_anatomical_map_csv,
    build_anatomical_map_rows,
    load_anatomical_map_csv,
    parse_anatomical_map_csv,
)
from pyneuroscope.models import ChannelGroup


def test_build_anatomical_map_rows_matches_group_layout() -> None:
    groups = [
        ChannelGroup("g1", [0, 2, 4]),
        ChannelGroup("g2", [1, 3]),
    ]
    channel_regions = {
        0: "CA1",
        2: "CA1",
        1: "CTX",
        3: "CTX",
    }

    rows = build_anatomical_map_rows(groups, channel_regions)

    assert rows == [
        ["CA1", "CTX"],
        ["CA1", "CTX"],
        ["", ""],
    ]


def test_parse_anatomical_map_csv_maps_cells_back_to_channels() -> None:
    groups = [
        ChannelGroup("g1", [10, 12, 14]),
        ChannelGroup("g2", [11, 13, 15]),
    ]
    text = "CA1,CTX\nCA1,CTX\n,DG\n"

    channel_regions = parse_anatomical_map_csv(text, groups)

    assert channel_regions == {
        10: "CA1",
        12: "CA1",
        11: "CTX",
        13: "CTX",
        15: "DG",
    }


def test_load_and_build_anatomical_map_csv_round_trip() -> None:
    groups = [
        ChannelGroup("g1", [0, 1]),
        ChannelGroup("g2", [2, 3]),
    ]
    channel_regions = {
        0: "CA1",
        1: "CA1",
        2: "CTX",
    }

    csv_text = build_anatomical_map_csv(groups, channel_regions)
    path = Path("build/test_anatomical_map.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv_text, encoding="utf-8")

    loaded = load_anatomical_map_csv(path, groups)

    assert loaded == channel_regions
