from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from functools import lru_cache
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
    physical_sites: dict[str, ProbeSitePosition] = field(default_factory=dict)
    note: str = ""

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
    names.update(neuropixels_specs())
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
    if clean in neuropixels_specs():
        return _neuropixels_geometry(clean)
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


@lru_cache(maxsize=3)
def _neuropixels_geometry(name: str) -> ProbeGeometry:
    spec = neuropixels_specs()[name]
    physical = {
        f"s{shank}_front_{site:04d}": ProbeSitePosition(
            float(spec["x_pattern_um"][site % len(spec["x_pattern_um"])] + shank * spec["shank_pitch_um"]),
            float(spec["first_site_y_um"] + (site // 2) * spec["row_pitch_um"]),
        )
        for shank in range(spec["shanks"])
        for site in range(spec["sites_per_shank"])
    }
    return ProbeGeometry(name, "um", {}, {}, {}, physical_sites=physical, note=spec["note"])


@lru_cache(maxsize=1)
def neuropixels_specs() -> dict:
    return json.loads(files("pyneuroscope.resources").joinpath("neuropixels_geometries.json").read_text(encoding="utf-8"))


def selected_site_map(text: str, geometry: ProbeGeometry, n_channels: int) -> RecordingChannelMap:
    """Selected physical IDs in DAT column order, with no hardware reordering."""
    import re
    tokens = [token for token in re.split(r"[\s,;]+", text.strip()) if token]
    if len(tokens) != n_channels:
        raise ProbeGeometryError(f"Enter exactly {n_channels} site IDs in DAT channel order (received {len(tokens)})")
    ids = []
    multi = any(site.startswith("s1_") for site in geometry.physical_sites)
    for token in tokens:
        if re.fullmatch(r"\d+", token) and not multi:
            site = f"s0_front_{int(token):04d}"
        elif re.fullmatch(r"\d+:\d+", token):
            shank, index = map(int, token.split(":"))
            site = f"s{shank}_front_{index:04d}"
        else:
            site = token
        if site not in geometry.physical_sites:
            raise ProbeGeometryError(f"Unknown physical site: {token}")
        ids.append(site)
    if len(ids) != len(set(ids)):
        raise ProbeGeometryError("A physical site cannot occur twice in the DAT channel list")
    return RecordingChannelMap({ch: geometry.physical_sites[site] for ch, site in enumerate(ids)},
                               set(range(n_channels)), dict(enumerate(ids)))


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
    return load_recording_channel_map(path).positions


@dataclass(frozen=True)
class RecordingChannelMap:
    positions: dict[int, ProbeSitePosition]
    active_channels: set[int]
    site_ids: dict[int, str] = field(default_factory=dict)
    probe_type: str = ""
    n_channels: int | None = None

    def __post_init__(self):
        channels = set(self.positions)
        if not self.active_channels.issubset(channels) or not set(self.site_ids).issubset(channels):
            raise ProbeGeometryError("Active channels and site IDs must have recording coordinates")
        ids = list(self.site_ids.values())
        if len(ids) != len(set(ids)):
            raise ProbeGeometryError("Duplicate physical site assignments")
        if self.n_channels is not None and (
            type(self.n_channels) is not int or self.n_channels <= 0
            or any(ch >= self.n_channels for ch in channels)
        ):
            raise ProbeGeometryError("Invalid recording map channel count")


def load_recording_channel_map(path: str | Path) -> RecordingChannelMap:
    """Read recording-column coordinates without compressing or renumbering rows."""
    path = Path(path)
    try:
        if path.suffix.lower() == ".json":
            return parse_recording_channel_map(json.loads(path.read_text(encoding="utf-8-sig")))
        if path.suffix.lower() == ".imro":
            from .neuropixels_maps import parse_imro
            return parse_imro(path.read_text(encoding="utf-8-sig"))
        loaded = loadmat(path, simplify_cells=True)
        coords = loaded.get("chanCoords")
        if coords is not None:
            x = _mat_vector(coords.get("x"), "chanCoords.x")
            y = _mat_vector(coords.get("y"), "chanCoords.y")
            channels = _mat_vector(coords.get("channel", np.arange(1, len(x) + 1)), "channel") - 1
            source = coords
        else:
            x = _mat_vector(loaded.get("xcoords"), "xcoords")
            y = _mat_vector(loaded.get("ycoords"), "ycoords")
            if "chanMap0ind" in loaded:
                channels = _mat_vector(loaded["chanMap0ind"], "chanMap0ind")
            elif "chanMap" in loaded:
                channels = _mat_vector(loaded["chanMap"], "chanMap") - 1
            else:
                channels = np.arange(len(x))
            source = loaded
        if len(x) != len(y) or len(channels) != len(x):
            raise ProbeGeometryError("Channel and coordinate lengths do not match")
        channels = _checked_channel_ids(channels)
        connected = np.asarray(source.get("connected", np.isfinite(x) & np.isfinite(y)), dtype=bool).reshape(-1)
        active = np.asarray(source.get("active", connected), dtype=bool).reshape(-1)
        site_ids = np.asarray(source.get("site_id", [""] * len(x)), dtype=object).reshape(-1)
        if any(len(a) != len(x) for a in [connected, active, site_ids]):
            raise ProbeGeometryError("Channel flags/site IDs and coordinate lengths do not match")
        positions, active_channels, ids = {}, set(), {}
        for channel, cx, cy, linked, used, site in zip(channels, x, y, connected, active, site_ids):
            if not linked:
                if used:
                    raise ProbeGeometryError("An active channel cannot be disconnected")
                continue
            if not np.isfinite([cx, cy]).all():
                raise ProbeGeometryError(f"Connected channel {channel} has non-finite coordinates")
            positions[channel] = ProbeSitePosition(float(cx), float(cy))
            if used:
                active_channels.add(channel)
            if str(site):
                ids[channel] = str(site)
        return RecordingChannelMap(positions, active_channels, ids)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, NotImplementedError) as exc:
        raise ProbeGeometryError(f"{path.name}: {exc}") from exc


def _checked_channel_ids(values) -> list[int]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ProbeGeometryError("Channel IDs must be finite, zero-based non-negative integers")
    channels = [int(value) for value in values]
    if len(channels) != len(set(channels)):
        raise ProbeGeometryError("Duplicate recording channel IDs")
    return channels


def parse_recording_channel_map(raw: dict) -> RecordingChannelMap:
    if not isinstance(raw, dict):
        raise ProbeGeometryError("Recording map must be a JSON object")
    if raw.get("format") == "WILDX NP applied probe receipt":
        from .neuropixels_maps import parse_wildx_metadata
        return parse_wildx_metadata(raw)
    # Atlaxis JSON includes geometry and the selected contact_to_channel table.
    if "geometry" in raw and "channel_map" in raw:
        contacts = {c["contact_id"]: c for c in raw["geometry"]["contacts"]}
        mapping = raw["channel_map"]
        sites = [dict(channel=ch, site_id=site, x=contacts[site]["x_um"], y=contacts[site]["y_um"],
                      active=ch not in mapping.get("skipped", []))
                 for site, ch in mapping["contact_to_channel"].items()]
    else:
        sites = raw["sites"]
    channels = _checked_channel_ids([s["channel"] for s in sites])
    positions, active, ids = {}, set(), {}
    for channel, site in zip(channels, sites):
        if not site.get("connected", True):
            if site.get("active", False):
                raise ProbeGeometryError("An active channel cannot be disconnected")
            continue
        position = _parse_position(site, channel)
        if not np.isfinite([position.x, position.y]).all():
            raise ProbeGeometryError("Channel coordinates must be finite")
        positions[channel] = position
        if site.get("active", True):
            active.add(channel)
        if site.get("site_id"):
            ids[channel] = str(site["site_id"])
    return RecordingChannelMap(positions, active, ids)


def find_chanmap_file(base_dirs: Iterable[Path], basenames: Iterable[str]) -> Path | None:
    names = [name for name in basenames if name]
    for base_dir in base_dirs:
        for name in names:
            for filename in [f"{name}.chanCoords.channelInfo.mat", f"{name}.chanMap.mat", f"{name}.ChanMap.mat"]:
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
    # Continue loading saved legacy configurations without offering this pattern.
    return clean if clean in {name.lower() for name in BUILTIN_PROBE_PATTERNS} | {"neuropixel"} else None


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
