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
    depths: Sequence[float] | None = None,
    subtract_channel_mean: bool = True,
) -> np.ndarray:
    """Return relative 1D CSD as the negative second depth difference."""
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
    if depths is not None:
        return _nonuniform_second_difference(traces, depths)
    return -np.diff(traces, n=2, axis=1)


def _nonuniform_second_difference(traces: np.ndarray, depths: Sequence[float]) -> np.ndarray:
    depth_values = np.asarray(depths, dtype=np.float64).reshape(-1)
    if depth_values.size != traces.shape[1]:
        raise ValueError("depths must match channels")
    if not np.all(np.isfinite(depth_values)):
        raise ValueError("depths must be finite")
    spacing = np.diff(depth_values)
    if np.any(spacing <= 0):
        raise ValueError("depths must be strictly increasing")
    median_spacing = float(np.nanmedian(spacing))
    if not np.isfinite(median_spacing) or median_spacing <= 0:
        raise ValueError("depths must have positive spacing")
    normalized_depths = depth_values / median_spacing
    left_spacing = normalized_depths[1:-1] - normalized_depths[:-2]
    right_spacing = normalized_depths[2:] - normalized_depths[1:-1]
    total_spacing = normalized_depths[2:] - normalized_depths[:-2]
    left_slope = (traces[:, 1:-1] - traces[:, :-2]) / left_spacing
    right_slope = (traces[:, 2:] - traces[:, 1:-1]) / right_spacing
    return -(2.0 * (right_slope - left_slope) / total_spacing)


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
