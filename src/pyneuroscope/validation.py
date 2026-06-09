from __future__ import annotations

from pathlib import Path

from .color_map import ColorMapError, apply_channel_overrides
from .dat_reader import DatReaderError, compute_window_bounds, inspect_dat
from .models import ChannelGroup, RecordingMetadata, ValidationMessage, ValidationResult
from .xml_builder import XmlError, build_neurocode_xml, parse_xml_text


def validate_settings(
    recording: RecordingMetadata,
    groups: list[ChannelGroup],
    bad_channels: set[int] | None = None,
    channel_colors: dict[int, str] | None = None,
    *,
    selected_window_start_seconds: float = 0.0,
    selected_window_duration_seconds: float = 1.0,
    xml_text: str | None = None,
    template_n_channels: int | None = None,
) -> ValidationResult:
    messages: list[ValidationMessage] = []
    bad = bad_channels or set()
    colors = channel_colors or {}

    if recording.n_channels <= 0:
        messages.append(_error("nChannels must be a positive integer"))
    if recording.sampling_rate <= 0:
        messages.append(_error("samplingRate must be positive"))
    if recording.lfp_sampling_rate <= 0:
        messages.append(_error("lfpSamplingRate must be positive"))

    if template_n_channels is not None and recording.n_channels > 0 and template_n_channels != recording.n_channels:
        messages.append(_warning("Probe template channel count differs from nChannels"))

    if recording.dat_path:
        try:
            file_sampling_rate = _recording_file_sampling_rate(recording)
            file_n_channels = _recording_file_n_channels(recording)
            info = inspect_dat(recording.dat_path, file_n_channels, file_sampling_rate, recording.dtype)
            try:
                _, _, clipped = compute_window_bounds(
                    info.total_frames,
                    info.sampling_rate,
                    selected_window_start_seconds,
                    selected_window_duration_seconds,
                    clip=True,
                )
                if clipped:
                    messages.append(_warning("Selected window was clipped to recording duration"))
            except DatReaderError as exc:
                messages.append(_error(str(exc)))
        except DatReaderError as exc:
            messages.append(_error(str(exc)))
    elif recording.dat_path is not None:
        messages.append(_error("DAT file path is empty"))

    messages.extend(_validate_groups(recording.n_channels, groups))
    messages.extend(_validate_channels(recording.n_channels, bad, "Bad channel"))

    try:
        apply_channel_overrides({}, colors, recording.n_channels)
    except ColorMapError as exc:
        messages.append(_error(str(exc)))

    if not any(message.level == "error" for message in messages) and recording.n_channels > 0:
        try:
            generated = build_neurocode_xml(
                recording.n_channels,
                recording.sampling_rate,
                recording.lfp_sampling_rate,
                groups,
                bad,
                metadata=recording,
                channel_colors=colors,
            )
            parse_xml_text(xml_text if xml_text is not None else generated)
        except XmlError as exc:
            messages.append(_error(f"XML validation failed: {exc}"))

    return ValidationResult(messages)


def _recording_file_sampling_rate(recording: RecordingMetadata) -> float:
    path = Path(recording.dat_path) if recording.dat_path else None
    if path is not None and path.suffix.lower() == ".lfp":
        return recording.lfp_sampling_rate
    return recording.sampling_rate


def _recording_file_n_channels(recording: RecordingMetadata) -> int:
    return int(recording.n_channels) + max(0, int(recording.file_extra_channels))


def _validate_groups(n_channels: int, groups: list[ChannelGroup]) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    seen: set[int] = set()
    duplicates: set[int] = set()
    invalid: set[int] = set()
    for group in groups:
        for channel in group.channels:
            if channel < 0 or channel >= n_channels:
                invalid.add(channel)
            if channel in seen:
                duplicates.add(channel)
            seen.add(channel)
    for channel in sorted(invalid):
        messages.append(_error(f"Group channel {channel} is outside 0..{n_channels - 1}"))
    for channel in sorted(duplicates):
        messages.append(_error(f"Group channel {channel} appears more than once"))
    if n_channels > 0:
        missing = set(range(n_channels)) - seen
        if missing:
            messages.append(_warning(f"{len(missing)} channels are not assigned to any group"))
    return messages


def _validate_channels(n_channels: int, channels: set[int], label: str) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    for channel in sorted(channels):
        if channel < 0 or channel >= n_channels:
            messages.append(_error(f"{label} {channel} is outside 0..{n_channels - 1}"))
    return messages


def _error(message: str) -> ValidationMessage:
    return ValidationMessage("error", message)


def _warning(message: str) -> ValidationMessage:
    return ValidationMessage("warning", message)
