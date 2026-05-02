import numpy as np

from pyneuroscope.models import ChannelGroup
from pyneuroscope.signal_filters import bandpass_filter, common_average_reference


def test_bandpass_filter_preserves_shape() -> None:
    sampling_rate = 20000
    t = np.arange(2000, dtype=np.float32) / sampling_rate
    data = np.column_stack(
        [
            np.sin(2 * np.pi * 1000 * t),
            np.sin(2 * np.pi * 2000 * t),
        ]
    ).astype(np.float32)

    filtered = bandpass_filter(data, sampling_rate, 500, 6000)

    assert filtered.shape == data.shape
    assert filtered.dtype == np.float32


def test_common_average_by_probe_removes_shared_signal() -> None:
    data = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)

    referenced = common_average_reference(data, [], "all")

    assert np.allclose(referenced, [[-1.0, 1.0], [-1.0, 1.0]])


def test_common_average_by_probe_excludes_bad_channels_from_reference() -> None:
    data = np.array([[1.0, 3.0, 100.0]], dtype=np.float32)

    referenced = common_average_reference(data, [], "all", bad_channels={2})

    assert np.allclose(referenced, [[-1.0, 1.0, 100.0]])


def test_common_average_by_shank_keeps_groups_separate() -> None:
    data = np.array([[1.0, 3.0, 10.0, 14.0]], dtype=np.float32)
    groups = [ChannelGroup("a", [0, 1]), ChannelGroup("b", [2, 3])]

    referenced = common_average_reference(data, groups, "group")

    assert np.allclose(referenced, [[-1.0, 1.0, -2.0, 2.0]])


def test_common_average_by_shank_excludes_bad_channels_from_reference() -> None:
    data = np.array([[1.0, 3.0, 100.0, 14.0]], dtype=np.float32)
    groups = [ChannelGroup("a", [0, 1]), ChannelGroup("b", [2, 3])]

    referenced = common_average_reference(data, groups, "group", bad_channels={2})

    assert np.allclose(referenced, [[-1.0, 1.0, 100.0, 0.0]])


def test_local_reference_uses_group_order_neighbors() -> None:
    data = np.array([[1.0, 2.0, 10.0]], dtype=np.float32)
    groups = [ChannelGroup("a", [0, 1, 2])]

    referenced = common_average_reference(data, groups, "local", local_radius_um=20, pitch_um=20)

    assert referenced.shape == data.shape
    assert np.allclose(referenced[0, 0], -0.5)


def test_local_reference_excludes_bad_channels_from_neighbors() -> None:
    data = np.array([[1.0, 2.0, 100.0]], dtype=np.float32)
    groups = [ChannelGroup("a", [0, 1, 2])]

    referenced = common_average_reference(data, groups, "local", bad_channels={2}, local_radius_um=20, pitch_um=20)

    assert np.allclose(referenced[0], [-0.5, 0.5, 100.0])
