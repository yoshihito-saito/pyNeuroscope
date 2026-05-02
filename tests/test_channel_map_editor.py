from pyneuroscope.channel_map_editor import group_designs_from_groups
from pyneuroscope.models import ChannelGroup


def test_group_designs_from_groups_preserves_each_group() -> None:
    designs = group_designs_from_groups(
        [
            ChannelGroup("group1", [0, 2]),
            ChannelGroup("group2", [1, 3, 5]),
        ]
    )

    assert len(designs) == 2
    assert designs[0].name == "group1"
    assert designs[0].slots[:2] == [0, 2]
    assert designs[1].name == "group2"
    assert designs[1].channels_per_group == 3
    assert designs[1].slots[:3] == [1, 3, 5]


def test_group_designs_from_empty_groups_returns_default_group() -> None:
    designs = group_designs_from_groups([])

    assert len(designs) == 1
    assert designs[0].name == "group1"
