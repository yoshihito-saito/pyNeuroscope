from pathlib import Path

import numpy as np

from pyneuroscope.models import ChannelGroup, RecordingMetadata
from pyneuroscope.validation import validate_settings


def test_detects_duplicate_group_channels() -> None:
    result = validate_settings(
        RecordingMetadata(n_channels=3, sampling_rate=20000, lfp_sampling_rate=1250),
        [ChannelGroup("g1", [0, 1]), ChannelGroup("g2", [1, 2])],
    )

    assert not result.ok
    assert any("appears more than once" in error for error in result.errors)


def test_warns_for_unassigned_channels() -> None:
    result = validate_settings(
        RecordingMetadata(n_channels=4, sampling_rate=20000, lfp_sampling_rate=1250),
        [ChannelGroup("g1", [0, 1])],
    )

    assert result.ok
    assert any("not assigned" in warning for warning in result.warnings)


def test_detects_invalid_bad_channel_and_color() -> None:
    result = validate_settings(
        RecordingMetadata(n_channels=2, sampling_rate=20000, lfp_sampling_rate=1250),
        [ChannelGroup("g1", [0, 1])],
        bad_channels={2},
        channel_colors={1: "red"},
    )

    assert not result.ok
    assert any("Bad channel 2" in error for error in result.errors)
    assert any("Invalid color" in error for error in result.errors)


def test_detects_unparseable_xml() -> None:
    result = validate_settings(
        RecordingMetadata(n_channels=2, sampling_rate=20000, lfp_sampling_rate=1250),
        [ChannelGroup("g1", [0, 1])],
        xml_text="<parameters>",
    )

    assert not result.ok
    assert any("XML validation failed" in error for error in result.errors)


def test_lfp_validation_uses_lfp_sampling_rate(tmp_path: Path) -> None:
    path = tmp_path / "basename.lfp"
    np.arange(8, dtype=np.int16).tofile(path)

    result = validate_settings(
        RecordingMetadata(
            dat_path=str(path),
            n_channels=2,
            sampling_rate=1000,
            lfp_sampling_rate=2,
        ),
        [ChannelGroup("g1", [0, 1])],
        selected_window_start_seconds=1.5,
        selected_window_duration_seconds=0.5,
    )

    assert result.ok
    assert not any("outside the recording" in error for error in result.errors)


def test_validation_uses_extra_file_channels_for_raw_frame_width(tmp_path: Path) -> None:
    path = tmp_path / "continuous.dat"
    np.asarray([[1, 2, 100], [3, 4, 101]], dtype=np.int16).tofile(path)

    result = validate_settings(
        RecordingMetadata(
            dat_path=str(path),
            n_channels=2,
            sampling_rate=1,
            lfp_sampling_rate=1,
            file_extra_channels=1,
        ),
        [ChannelGroup("g1", [0, 1])],
        selected_window_start_seconds=1.0,
        selected_window_duration_seconds=1.0,
    )

    assert result.ok
