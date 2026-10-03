"""Display-only amplitude bins for the channel/time bitmap."""
from __future__ import annotations

import numpy as np

from .trace_display import trace_statistics


def signed_peak_bins(values: np.ndarray, pixels: int) -> np.ndarray:
    """Keep the strongest signed excursion in each equal-duration sample bin."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    pixels = max(1, int(pixels))
    if len(values) <= pixels:
        return np.where(np.isfinite(values), values, np.nan)
    starts = np.arange(pixels, dtype=np.int64) * len(values) // pixels
    finite = np.isfinite(values)
    low = np.minimum.reduceat(np.where(finite, values, np.inf), starts)
    high = np.maximum.reduceat(np.where(finite, values, -np.inf), starts)
    result = np.where(np.abs(low) > np.abs(high), low, high)
    result[~np.isfinite(result)] = np.nan
    return result


def bitmap_amplitudes(data: np.ndarray, channels: list[int], pixels: int):
    """Median-center each channel and use one robust amplitude limit for all."""
    if not len(data):
        return np.empty((len(channels), 0)), {ch: 0.0 for ch in channels}, 1.0
    indices = np.linspace(0, len(data) - 1, min(len(data), 2048), dtype=np.int64)
    sampled = np.asarray(data[np.ix_(indices, channels)], dtype=np.float64)
    centers = {ch: trace_statistics(sampled[:, i])[0] for i, ch in enumerate(channels)}
    centered = sampled - np.asarray([centers[ch] for ch in channels])
    finite = np.abs(centered[np.isfinite(centered)])
    limit = float(np.percentile(finite, 98)) if finite.size else 1.0
    if limit <= 0:
        limit = 1.0
    rows = [signed_peak_bins(data[:, ch].astype(np.float64) - centers[ch], pixels) for ch in channels]
    return np.asarray(rows), centers, limit
