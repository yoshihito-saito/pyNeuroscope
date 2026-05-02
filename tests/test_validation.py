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
