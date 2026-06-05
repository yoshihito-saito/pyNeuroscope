from __future__ import annotations

from collections.abc import Sequence

import numpy as np

CSD_COLORMAPS = (
    "bwr",
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
)


def standard_1d_csd(
    data: np.ndarray,
    channels: Sequence[int],
    *,
    subtract_channel_mean: bool = True,
) -> np.ndarray:
    """Return relative 1D CSD as the negative second channel difference."""
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("data must be a 2D samples-by-channels array")
    channel_indices = [int(channel) for channel in channels]
    if len(channel_indices) < 3:
        return np.empty((values.shape[0], 0), dtype=np.float64)
    if min(channel_indices) < 0 or max(channel_indices) >= values.shape[1]:
        raise ValueError("channels must be valid data columns")

    traces = values[:, channel_indices]
    if subtract_channel_mean and traces.size:
        traces = traces - np.nanmean(traces, axis=0, keepdims=True)
    return -np.diff(traces, n=2, axis=1)


def robust_csd_limits(csd: np.ndarray, percentile: float = 98.0) -> tuple[float, float]:
    values = np.asarray(csd, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    limit = float(np.nanpercentile(np.abs(finite), percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return -limit, limit
