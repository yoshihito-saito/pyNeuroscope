from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RecordingMetadata:
    dat_path: str | None = None
    n_channels: int = 0
    sampling_rate: float = 0.0
    lfp_sampling_rate: float = 0.0
    dtype: str = "int16"
    n_bits: int | None = 16
    voltage_range: float | None = 20.0
    amplification: float | None = 1000.0
    offset: float = 0.0
    least_significant_bit: float | None = None
    duration_seconds: float | None = None
    total_frames: int | None = None
    file_size_bytes: int | None = None

    @property
    def path(self) -> Path | None:
        return Path(self.dat_path) if self.dat_path else None


@dataclass(frozen=True)
class ChannelGroup:
    name: str
    channels: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ProbeTemplate:
    vendor: str
    model: str
    n_channels: int | None = None
    groups: list[ChannelGroup] = field(default_factory=list)
    grouping: str | None = None
    draft: bool = False


@dataclass(frozen=True)
class ChannelDisplaySettings:
    channel: int
    color: str
    visible: bool = True
    scale: float = 1.0


@dataclass(frozen=True)
class DatInfo:
    path: Path
    n_channels: int
    sampling_rate: float
    dtype: str
    file_size_bytes: int
    total_frames: int
    duration_seconds: float
    frame_bytes: int


@dataclass(frozen=True)
class DatWindow:
    time_seconds: object
    data: object
    start_frame: int
    frames_read: int
    clipped: bool = False


@dataclass(frozen=True)
class ValidationMessage:
    level: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    messages: list[ValidationMessage]

    @property
    def ok(self) -> bool:
        return not any(message.level == "error" for message in self.messages)

    @property
    def has_warnings(self) -> bool:
        return any(message.level == "warning" for message in self.messages)

    @property
    def errors(self) -> list[str]:
        return [message.message for message in self.messages if message.level == "error"]

    @property
    def warnings(self) -> list[str]:
        return [message.message for message in self.messages if message.level == "warning"]


@dataclass(frozen=True)
class SpikeUnit:
    uid: int
    label: str
    times: np.ndarray
    channel: int | None = None
    group: int | None = None
    region: str | None = None


@dataclass(frozen=True)
class SpikesData:
    path: Path
    basename: str
    units: list[SpikeUnit] = field(default_factory=list)


@dataclass(frozen=True)
class EventSeries:
    name: str
    path: Path
    timestamps: np.ndarray
    peaks: np.ndarray | None = None


@dataclass(frozen=True)
class SignalSpikeOverlay:
    unit_id: int
    label: str
    times: np.ndarray
    color: str
    channel: int | None = None


@dataclass(frozen=True)
class SignalEventOverlay:
    name: str
    color: str
    timestamps: np.ndarray
    peaks: np.ndarray | None = None
    show_intervals: bool = True
    show_peaks: bool = False
    below: bool = False
