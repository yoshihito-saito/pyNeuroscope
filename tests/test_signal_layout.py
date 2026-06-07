from pyneuroscope.models import ChannelGroup
from pyneuroscope.probe_geometry import ProbeSitePosition
from pyneuroscope.signal_layout import group_column_layout, single_column_layout, unassigned_channels


def test_single_column_layout_preserves_group_and_channel_order() -> None:
    groups = [ChannelGroup("a", [3, 1]), ChannelGroup("b", [2, 0])]

    layout = single_column_layout(groups, {1}, {3: "#111111"})

    assert [item.channel for item in layout] == [3, 1, 2, 0]
    assert [item.row for item in layout] == [0, 1, 2, 3]
    assert {item.column for item in layout} == {0}
    assert layout[1].is_bad is True


def test_single_column_layout_ignores_probe_geometry() -> None:
    groups = [ChannelGroup("a", [0, 1])]
    geometry = {0: ProbeSitePosition(-10, 0), 1: ProbeSitePosition(10, 20)}

    layout = single_column_layout(groups, set(), {}, channel_geometry=geometry)

    assert [item.column for item in layout] == [0, 0]
    assert [item.x for item in layout] == [None, None]


def test_group_column_layout_uses_group_as_column() -> None:
    groups = [ChannelGroup("a", [3, 1]), ChannelGroup("b", [2])]

    layout = group_column_layout(groups, set(), {})

    assert [(item.channel, item.column, item.row) for item in layout] == [
        (3, 0, 0),
        (1, 0, 1),
        (2, 1, 0),
    ]


def test_group_column_layout_expands_probe_geometry_columns() -> None:
    groups = [ChannelGroup("shank1", [10, 11, 12, 13])]
    geometry = {
        10: ProbeSitePosition(-10, 0),
        11: ProbeSitePosition(10, 20),
        12: ProbeSitePosition(-10, 40),
        13: ProbeSitePosition(10, 60),
    }

    layout = group_column_layout(groups, set(), {}, channel_geometry=geometry)

    assert [(item.channel, item.column, item.row) for item in layout] == [
        (10, 0, 0),
        (11, 1, 1),
        (12, 0, 2),
        (13, 1, 3),
    ]
    assert layout[0].x == -10
    assert layout[1].x == 10


def test_unassigned_channels() -> None:
    groups = [ChannelGroup("a", [0, 2])]

    assert unassigned_channels(4, groups) == [1, 3]
