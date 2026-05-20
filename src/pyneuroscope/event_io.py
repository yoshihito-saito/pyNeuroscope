from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

from .models import EventSeries, SpikeUnit, SpikesData


class EventLoadError(ValueError):
    """Raised when spike or event MAT files do not contain expected fields."""


def candidate_analysis_dirs(selected_path: Path, dat_paths: Iterable[Path]) -> list[Path]:
    dirs: list[Path] = []
    if selected_path.exists():
        dirs.append(selected_path if selected_path.is_dir() else selected_path.parent)
    for dat_path in dat_paths:
        dirs.append(dat_path.parent)
        dirs.append(dat_path.parent.parent)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in dirs:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def find_spikes_file(base_dirs: Iterable[Path], basenames: Iterable[str]) -> Path | None:
    names = [name for name in basenames if name]
    for base_dir in base_dirs:
        for name in names:
            candidate = base_dir / f"{name}.spikes.cellinfo.mat"
            if candidate.exists():
                return candidate
        matches = sorted(base_dir.glob("*.spikes.cellinfo.mat"))
        if matches:
            return matches[0]
    return None


def find_event_files(base_dirs: Iterable[Path], basenames: Iterable[str]) -> list[Path]:
    names = [name for name in basenames if name]
    matches: list[Path] = []
    seen: set[str] = set()
    for base_dir in base_dirs:
        candidates: list[Path] = []
        for name in names:
            candidates.extend(base_dir.glob(f"{name}.*.events.mat"))
        candidates.extend(base_dir.glob("*.events.mat"))
        for path in sorted(candidates):
            if _is_ignored_event_file(path):
                continue
            key = str(path.resolve()) if path.exists() else str(path)
            if key not in seen:
                seen.add(key)
                matches.append(path)
    return matches


def load_spikes_cellinfo(path: str | Path) -> SpikesData:
    mat_path = Path(path)
    loaded = loadmat(mat_path, simplify_cells=True)
    spikes = loaded.get("spikes")
    if not isinstance(spikes, dict):
        raise EventLoadError(f"No spikes struct found in {mat_path}")

    times_by_unit = _object_items(spikes.get("times"))
    if not times_by_unit:
        times_by_unit = _object_items(spikes.get("ts"))
        sr = float(spikes.get("sr", 1.0) or 1.0)
        times_by_unit = [np.asarray(times, dtype=float).reshape(-1) / sr for times in times_by_unit]
    if not times_by_unit:
        raise EventLoadError(f"No spike times found in {mat_path}")

    basename = str(spikes.get("basename") or mat_path.name.replace(".spikes.cellinfo.mat", ""))
    uid_values = _scalar_items(spikes.get("UID"), len(times_by_unit), default_start=1)
    clu_values = _scalar_items(spikes.get("cluID"), len(times_by_unit), default_start=1)
    channels = _channel_items(spikes, len(times_by_unit))
    groups = _scalar_items(spikes.get("shankID"), len(times_by_unit), default_start=0)

    units: list[SpikeUnit] = []
    for index, times in enumerate(times_by_unit):
        clean_times = np.asarray(times, dtype=float).reshape(-1)
        clean_times = clean_times[np.isfinite(clean_times)]
        uid = int(uid_values[index]) if index < len(uid_values) else index + 1
        clu = int(clu_values[index]) if index < len(clu_values) else uid
        channel = channels[index] if index < len(channels) else None
        group = int(groups[index]) if index < len(groups) and groups[index] is not None else None
        units.append(
            SpikeUnit(
                uid=uid,
                label=f"UID {uid} / clu {clu}",
                times=np.sort(clean_times),
                channel=channel,
                group=group,
            )
        )
    return SpikesData(path=mat_path, basename=basename, units=units)


def load_event_file(path: str | Path) -> EventSeries:
    mat_path = Path(path)
    loaded = loadmat(mat_path, simplify_cells=True)
    event_name = _event_name_from_path(mat_path)
    if event_name.lower() == "mergepoints":
        raise EventLoadError(f"Ignoring MergePoints event file: {mat_path}")
    event_struct = loaded.get(event_name)
    if not isinstance(event_struct, dict):
        structs = [(key, value) for key, value in loaded.items() if not key.startswith("__") and isinstance(value, dict)]
        if not structs:
            raise EventLoadError(f"No event struct found in {mat_path}")
        event_name, event_struct = structs[0]
        if str(event_name).lower() == "mergepoints":
            raise EventLoadError(f"Ignoring MergePoints event file: {mat_path}")

    timestamps = np.asarray(event_struct.get("timestamps", np.empty((0, 2))), dtype=float)
    if timestamps.ndim == 1:
        timestamps = timestamps.reshape(-1, 1)
    if timestamps.shape[1] == 1:
        timestamps = np.column_stack([timestamps[:, 0], timestamps[:, 0]])
    if timestamps.shape[1] > 2:
        timestamps = timestamps[:, :2]
    timestamps = timestamps[np.all(np.isfinite(timestamps), axis=1)]
    if timestamps.size == 0:
        raise EventLoadError(f"No timestamps found in {mat_path}")

    peaks_value = event_struct.get("peaks")
    peaks = None
    if peaks_value is not None:
        peak_array = np.asarray(peaks_value, dtype=float).reshape(-1)
        peak_array = peak_array[np.isfinite(peak_array)]
        peaks = peak_array if peak_array.size else None

    return EventSeries(name=str(event_name), path=mat_path, timestamps=timestamps, peaks=peaks)


def _is_ignored_event_file(path: Path) -> bool:
    return _event_name_from_path(path).lower() == "mergepoints"


def _event_name_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".events.mat"):
        name = name[: -len(".events.mat")]
    parts = name.split(".")
    return parts[-1] if parts else name


def _object_items(value) -> list[np.ndarray]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        return [np.asarray(arr.item(), dtype=float).reshape(-1)]
    return [np.asarray(item, dtype=float).reshape(-1) for item in arr.reshape(-1)]


def _scalar_items(value, count: int, *, default_start: int) -> list[int | None]:
    if value is None:
        return [default_start + index for index in range(count)]
    arr = np.asarray(value, dtype=object).reshape(-1)
    items: list[int | None] = []
    for item in arr[:count]:
        try:
            scalar = np.asarray(item).reshape(-1)[0]
            if np.isfinite(float(scalar)):
                items.append(int(scalar))
            else:
                items.append(None)
        except (IndexError, TypeError, ValueError):
            items.append(None)
    while len(items) < count:
        items.append(default_start + len(items))
    return items


def _channel_items(spikes: dict, count: int) -> list[int | None]:
    for key, one_based in [("maxWaveformCh1", True), ("phy_maxWaveformCh1", True), ("maxWaveformCh", False)]:
        if key not in spikes:
            continue
        values = _scalar_items(spikes.get(key), count, default_start=0)
        channels: list[int | None] = []
        for value in values:
            if value is None:
                channels.append(None)
            else:
                channels.append(max(0, int(value) - 1 if one_based else int(value)))
        return channels
    return [None] * count
