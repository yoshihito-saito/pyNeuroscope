"""Display-only envelopes; raw samples remain available to analysis and overlays."""
from __future__ import annotations

import numpy as np


def peak_envelope_indices(values: np.ndarray, max_points: int) -> np.ndarray:
    """Keep each bin's extrema in time order, endpoints, and finite/gap edges."""
    values = np.asarray(values).reshape(-1)
    count = len(values)
    if count <= max(2, max_points):
        return np.arange(count)
    bins = max(1, int(max_points) // 2)
    step = int(np.ceil(count / bins))
    full = count // step
    blocks = values[:full * step].reshape(full, step)
    finite = np.isfinite(blocks)
    starts = np.arange(full) * step
    minima = starts + np.argmin(np.where(finite, blocks, np.inf), axis=1)
    maxima = starts + np.argmax(np.where(finite, blocks, -np.inf), axis=1)
    indices = [minima, maxima, np.asarray([0, count - 1])]
    tail = values[full * step:]
    if tail.size:
        valid = np.isfinite(tail)
        indices.append(full * step + np.asarray([
            np.argmin(np.where(valid, tail, np.inf)),
            np.argmax(np.where(valid, tail, -np.inf)),
        ]))
    finite = np.isfinite(values)
    edges = np.flatnonzero(finite[1:] != finite[:-1])
    indices.extend([edges, edges + 1])
    return np.unique(np.concatenate(indices))


def trace_statistics(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    center = float(np.median(finite))
    peak = float(np.percentile(np.abs(finite.astype(np.float64) - center), 98)) or 1.0
    return center, peak
