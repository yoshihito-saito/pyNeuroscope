from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

from .models import ChannelGroup


BUILTIN_PROBE_PATTERNS = (
    "poly2",
    "poly3",
    "poly5",
    "staggered",
    "neuropixel",
    "double_sided",
    "neurogrid",
)


@dataclass(frozen=True)
class ProbeSitePosition:
    x: float
    y: float


@dataclass(frozen=True)
class ProbeGeometry:
    name: str
    units: str
    sites_by_channel: dict[int, ProbeSitePosition]
    sites_by_slot: dict[int, ProbeSitePosition]
    sites_by_group_slot: dict[tuple[int, int], ProbeSitePosition]
    group_pitch_um: float | None = None
    pattern: str | None = None

    def positions_for_groups(
        self,
        groups: Iterable[ChannelGroup],
        *,
        channel_offset: int = 0,
    ) -> dict[int, ProbeSitePosition]:
        group_list = list(groups)
        if self.pattern:
            return positions_for_pattern(self.pattern, group_list, channel_offset=channel_offset)

        positions: dict[int, ProbeSitePosition] = {
            channel + channel_offset: position
            for channel, position in self.sites_by_channel.items()
        }
        for group_index, group in enumerate(group_list):
            group_x_offset = self._group_x_offset(group_index)
            for slot, channel in enumerate(group.channels):
                position = self.sites_by_group_slot.get((group_index, slot))
                if position is None:
                    position = self.sites_by_slot.get(slot)
                    if position is not None:
                        position = ProbeSitePosition(position.x + group_x_offset, position.y)
                if position is not None:
                    positions[channel + channel_offset] = position
        return positions

    def _group_x_offset(self, group_index: int) -> float:
        if self.group_pitch_um is None:
            return 0.0
        return float(group_index) * float(self.group_pitch_um)


class ProbeGeometryError(ValueError):
    """Raised when a probe geometry file cannot be parsed."""


def geometry_search_paths() -> list[Path]:
    paths = [
        Path.cwd() / "probe_geometry",
        Path(__file__).resolve().parents[2] / "probe_geometry",
        Path(__file__).resolve().parent / "probe_geometry",
    ]
    unique: list[Path] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def available_probe_geometries(paths: Iterable[Path] | None = None) -> list[str]:
    names: set[str] = set(BUILTIN_PROBE_PATTERNS)
    for directory in paths or geometry_search_paths():
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            if path.name.startswith("."):
                continue
            names.add(path.stem)
    return sorted(names)


def load_probe_geometry(name: str, paths: Iterable[Path] | None = None) -> ProbeGeometry | None:
    clean = name.strip()
    if not clean:
        return None
    for directory in paths or geometry_search_paths():
        path = directory / f"{clean}.json"
        if path.is_file():
            return parse_probe_geometry(path.read_text(encoding="utf-8"), fallback_name=clean)
    pattern = _canonical_pattern(clean)
    if pattern:
        return ProbeGeometry(
            name=pattern,
            units="um",
            sites_by_channel={},
            sites_by_slot={},
            sites_by_group_slot={},
            pattern=pattern,
        )
    return None


def positions_for_pattern(
    pattern: str,
    groups: Iterable[ChannelGroup],
    *,
    channel_offset: int = 0,
) -> dict[int, ProbeSitePosition]:
    clean = _canonical_pattern(pattern)
    if not clean:
        raise ProbeGeometryError(f"Unknown probe geometry pattern: {pattern}")
    positions: dict[int, ProbeSitePosition] = {}
    for local_idx, group in enumerate(groups):
        x, y = _pattern_xy(clean, len(group.channels), local_idx)
        for channel, x_value, y_value in zip(group.channels, x, y):
            positions[channel + channel_offset] = ProbeSitePosition(float(x_value), float(y_value))
    return positions


def load_chanmap_geometry(path: str | Path) -> dict[int, ProbeSitePosition]:
    mat_path = Path(path)
    loaded = loadmat(mat_path, simplify_cells=True)
    xcoords = _mat_vector(loaded.get("xcoords"), "xcoords")
    ycoords = _mat_vector(loaded.get("ycoords"), "ycoords")
    if xcoords.size != ycoords.size:
        raise ProbeGeometryError(f"{mat_path.name} has mismatched xcoords/ycoords lengths")
    channels = _chanmap_channels(loaded, xcoords.size)
    if channels.size != xcoords.size:
        raise ProbeGeometryError(f"{mat_path.name} has mismatched channel/xcoords lengths")
    return {
        int(channel): ProbeSitePosition(float(x), float(y))
        for channel, x, y in zip(channels, xcoords, ycoords)
        if np.isfinite(float(channel)) and np.isfinite(float(x)) and np.isfinite(float(y))
    }


def find_chanmap_file(base_dirs: Iterable[Path], basenames: Iterable[str]) -> Path | None:
    names = [name for name in basenames if name]
    for base_dir in base_dirs:
        for name in names:
            for filename in [f"{name}.chanMap.mat", f"{name}.ChanMap.mat"]:
                candidate = base_dir / filename
                if candidate.exists():
                    return candidate
        candidate = base_dir / "chanMap.mat"
        if candidate.exists():
            return candidate
        matches = sorted(base_dir.glob("*chanMap*.mat")) + sorted(base_dir.glob("*ChanMap*.mat"))
        if matches:
            return matches[0]
    return None


def parse_probe_geometry(text: str, *, fallback_name: str = "") -> ProbeGeometry:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProbeGeometryError(str(exc)) from exc
    if not isinstance(raw, dict):
        raise ProbeGeometryError("Probe geometry must be a JSON object")
    sites = raw.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ProbeGeometryError("Probe geometry needs a non-empty sites list")

    sites_by_channel: dict[int, ProbeSitePosition] = {}
    sites_by_slot: dict[int, ProbeSitePosition] = {}
    sites_by_group_slot: dict[tuple[int, int], ProbeSitePosition] = {}
    for index, site in enumerate(sites):
        if not isinstance(site, dict):
            raise ProbeGeometryError(f"Site {index} must be an object")
        position = _parse_position(site, index)
        if "channel" in site:
            sites_by_channel[_parse_non_negative_int(site["channel"], f"Site {index} channel")] = position
        elif "slot" in site:
            slot = _parse_non_negative_int(site["slot"], f"Site {index} slot")
            if "group" in site:
                group = _parse_non_negative_int(site["group"], f"Site {index} group")
                sites_by_group_slot[(group, slot)] = position
            else:
                sites_by_slot[slot] = position
        else:
            raise ProbeGeometryError(f"Site {index} needs channel or slot")

    group_pitch = raw.get("group_pitch_um")
    if group_pitch is None:
        group_pitch = raw.get("group_pitch")
    if group_pitch is not None and not isinstance(group_pitch, (int, float)):
        raise ProbeGeometryError("group_pitch_um must be numeric")

    name = str(raw.get("name") or fallback_name).strip() or fallback_name
    return ProbeGeometry(
        name=name,
        units=str(raw.get("units") or "um"),
        sites_by_channel=sites_by_channel,
        sites_by_slot=sites_by_slot,
        sites_by_group_slot=sites_by_group_slot,
        group_pitch_um=float(group_pitch) if group_pitch is not None else None,
    )


def _parse_position(site: dict, index: int) -> ProbeSitePosition:
    try:
        x = float(site["x"])
        y = float(site["y"])
    except KeyError as exc:
        raise ProbeGeometryError(f"Site {index} needs x and y") from exc
    except (TypeError, ValueError) as exc:
        raise ProbeGeometryError(f"Site {index} x and y must be numeric") from exc
    return ProbeSitePosition(x, y)


def _parse_non_negative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ProbeGeometryError(f"{label} must be a non-negative integer")
    return value


def _canonical_pattern(pattern: str) -> str | None:
    clean = pattern.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "chanmap": "chanmap.mat",
        "chanmap_mat": "chanmap.mat",
        "channelmap": "chanmap.mat",
        "channel_map": "chanmap.mat",
        "neuro_pixel": "neuropixel",
        "neuro_pixel_1": "neuropixel",
        "neuropixels": "neuropixel",
        "poly_2": "poly2",
        "poly_3": "poly3",
        "poly_5": "poly5",
    }
    clean = aliases.get(clean, clean)
    if clean == "chanmap.mat":
        return None
    return clean if clean in {name.lower() for name in BUILTIN_PROBE_PATTERNS} else None


def _pattern_xy(pattern: str, n_ch: int, local_idx: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(n_ch, dtype=float)
    y = np.zeros(n_ch, dtype=float)
    shank_id = local_idx + 1

    if pattern == "double_sided":
        pair_idx = local_idx // 2
        is_front = local_idx % 2 == 1
        y = np.arange(1, n_ch + 1, dtype=float) * -20.0
        x[:] = 20.0
        x[::2] = -20.0
        pair_origin = (pair_idx + 1) * 400.0
        intra_pair_offset = 80.0 if is_front else 0.0
        x = x + pair_origin + intra_pair_offset
    elif pattern == "neuropixel":
        x_pat = [20.0, 60.0, 0.0, 40.0]
        x = np.tile(x_pat, (n_ch // 4) + 1)[:n_ch].astype(float)
        y_base = (np.arange(n_ch) // 2) + 1
        y = y_base.astype(float) * -20.0
        x = x + shank_id * 200.0
    elif pattern in {"poly2", "staggered"}:
        x[:] = 20.0
        y = np.arange(1, n_ch + 1, dtype=float) * -20.0
        x[::2] = -20.0
        x = x + shank_id * 200.0
    elif pattern == "poly3":
        ext = n_ch % 3
        poly = (np.arange(1, n_ch - ext + 1)) % 3
        x[:] = 0.0
        x[np.where(poly == 1)[0] + ext] = -18.0
        x[np.where(poly == 2)[0] + ext] = 0.0
        x[np.where(poly == 0)[0] + ext] = 18.0
        x[:ext] = 0.0
        for x_value, y_offset in [(18.0, 0.0), (0.0, -10.0 + ext * 20.0), (-18.0, 0.0)]:
            mask = x == x_value
            y[mask] = np.arange(1, np.sum(mask) + 1, dtype=float) * -20.0 + y_offset
        x = x + shank_id * 200.0
    elif pattern == "poly5":
        ext = n_ch % 5
        poly = (np.arange(1, n_ch - ext + 1)) % 5
        x[:] = np.nan
        x[np.where(poly == 1)[0] + ext] = -36.0
        x[np.where(poly == 2)[0] + ext] = -18.0
        x[np.where(poly == 3)[0] + ext] = 0.0
        x[np.where(poly == 4)[0] + ext] = 18.0
        x[np.where(poly == 0)[0] + ext] = 36.0
        if ext > 0:
            x[:ext] = 18.0 * ((-1.0) ** np.arange(1, ext + 1))
        for x_value, y_offset in [(36.0, 0.0), (18.0, -14.0), (0.0, 0.0), (-18.0, -14.0), (-36.0, 0.0)]:
            mask = x == x_value
            if np.any(mask):
                y[mask] = np.arange(1, np.sum(mask) + 1, dtype=float) * -28.0 + y_offset
        x = x + shank_id * 200.0
    elif pattern == "neurogrid":
        for index in range(n_ch):
            x[index] = n_ch - (index + 1)
            y[index] = -(index + 1) * 30.0
        x = x + shank_id * 30.0
    else:
        raise ProbeGeometryError(f"Unknown probe geometry pattern: {pattern}")
    return x, y


def _mat_vector(value: object, label: str) -> np.ndarray:
    if value is None:
        raise ProbeGeometryError(f"chanMap.mat is missing {label}")
    return np.asarray(value, dtype=float).reshape(-1)


def _chanmap_channels(loaded: dict, count: int) -> np.ndarray:
    if "chanMap0ind" in loaded:
        return np.asarray(loaded["chanMap0ind"], dtype=float).reshape(-1).astype(int)
    if "chanMap" in loaded:
        channels = np.asarray(loaded["chanMap"], dtype=float).reshape(-1).astype(int)
        if channels.size and np.min(channels) >= 1:
            channels = channels - 1
        return channels
    return np.arange(count, dtype=int)
