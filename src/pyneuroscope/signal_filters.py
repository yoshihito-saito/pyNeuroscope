from __future__ import annotations

import numpy as np
from scipy import signal

from .models import ChannelGroup


class SignalFilterError(ValueError):
    """Raised when signal filter settings are invalid."""


def bandpass_filter(
    data: np.ndarray,
    sampling_rate: float,
    low_hz: float,
    high_hz: float,
    *,
    order: int = 3,
) -> np.ndarray:
    if data.size == 0:
        return data
    if sampling_rate <= 0:
        raise SignalFilterError("sampling_rate must be positive")
    nyquist = sampling_rate / 2.0
    if low_hz <= 0 or high_hz <= 0 or low_hz >= high_hz or high_hz >= nyquist:
        raise SignalFilterError(f"Invalid bandpass range: {low_hz:g}-{high_hz:g} Hz")
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=sampling_rate, output="sos")
    values = data.astype(np.float32, copy=False)
    try:
        return signal.sosfiltfilt(sos, values, axis=0).astype(np.float32, copy=False)
    except ValueError:
        return signal.sosfilt(sos, values, axis=0).astype(np.float32, copy=False)


def common_average_reference(
    data: np.ndarray,
    groups: list[ChannelGroup],
    mode: str,
    *,
    bad_channels: set[int] | None = None,
    local_radius_um: float = 200.0,
    pitch_um: float = 20.0,
) -> np.ndarray:
    if data.size == 0:
        return data
    bad = bad_channels or set()
    values = data.astype(np.float32, copy=True)
    if mode in {"all", "probe"}:
        return _subtract_reference(values, range(values.shape[1]), bad)
    if mode in {"group", "per_group", "shank"}:
        for group in groups:
            _subtract_reference(values, group.channels, bad)
        return values
    if mode == "local":
        return _local_reference(values, groups, bad, local_radius_um=local_radius_um, pitch_um=pitch_um)
    raise SignalFilterError(f"Unknown CAR mode: {mode}")


def _subtract_reference(values: np.ndarray, channels: object, bad_channels: set[int]) -> np.ndarray:
    valid = _valid_channels(values.shape[1], channels, bad_channels)
    if not valid:
        return values
    reference = np.mean(values[:, valid], axis=1, keepdims=True)
    values[:, valid] -= reference
    return values


def _local_reference(
    values: np.ndarray,
    groups: list[ChannelGroup],
    bad_channels: set[int],
    *,
    local_radius_um: float,
    pitch_um: float,
) -> np.ndarray:
    result = values.copy()
    radius_steps = max(0, int(round(local_radius_um / max(1e-9, pitch_um))))
    channel_to_neighbors: dict[int, list[int]] = {}
    for group in groups:
        valid_group = [channel for channel in group.channels if 0 <= channel < values.shape[1]]
        for index, channel in enumerate(valid_group):
            left = max(0, index - radius_steps)
            right = min(len(valid_group), index + radius_steps + 1)
            neighbors = [ch for ch in valid_group[left:right] if ch not in bad_channels]
            channel_to_neighbors[channel] = neighbors
    for channel, neighbors in channel_to_neighbors.items():
        if channel in bad_channels or not neighbors:
            continue
        reference = np.mean(values[:, neighbors], axis=1)
        result[:, channel] = values[:, channel] - reference
    return result


def _valid_channels(n_channels: int, channels: object, bad_channels: set[int]) -> list[int]:
    return [channel for channel in channels if 0 <= channel < n_channels and channel not in bad_channels]
