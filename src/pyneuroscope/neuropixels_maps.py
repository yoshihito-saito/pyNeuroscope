"""Probe-local DAT assignments from WILDX receipts and SpikeGLX IMRO files.

IMRO fields follow https://billkarsh.github.io/SpikeGLX/help/imroTables/.
Readout channel IDs are used as DAT column IDs; no sorting by physical site.
"""
from __future__ import annotations

import re

from .probe_geometry import (
    ProbeGeometryError, RecordingChannelMap, load_probe_geometry, neuropixels_specs,
)


READOUT_CHANNELS = 384
NP1 = "Neuropixels 1.0"
NP2_SINGLE = "Neuropixels 2.0 single shank"
NP2_FOUR = "Neuropixels 2.0 four shank"


def _integer(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ProbeGeometryError(f"{name} must be a zero-based non-negative integer")
    return value


def _flag(row: dict, key: str, default: bool) -> bool:
    value = row.get(key, default)
    if type(value) is not bool:
        raise ProbeGeometryError(f"{key} must be true or false")
    return value


def _physical_map(model: str, rows: list[dict]) -> RecordingChannelMap:
    geometry = load_probe_geometry(model)
    positions, active, ids = {}, set(), {}
    channels = [_integer(row["channel"], "Channel ID") for row in rows]
    if len(channels) != READOUT_CHANNELS or set(channels) != set(range(READOUT_CHANNELS)):
        raise ProbeGeometryError("Neuropixels map must contain each output channel 0–383 exactly once")
    for channel, row in zip(channels, rows):
        # Unconnected DAT columns remain in the 384-column layout, even when the
        # device reports an invalid/sentinel physical site for them.
        if not row.get("connected", True):
            continue
        shank = _integer(row.get("shank", 0), "Shank ID")
        site = _integer(row["site"], "Physical site ID")
        site_id = f"s{shank}_front_{site:04d}"
        if site_id not in geometry.physical_sites:
            raise ProbeGeometryError(f"Physical site {shank}:{site} is outside {model}")
        positions[channel] = geometry.physical_sites[site_id]
        ids[channel] = site_id
        if row.get("active", True):
            active.add(channel)
    return RecordingChannelMap(positions, active, ids, model, READOUT_CHANNELS)


def parse_wildx_metadata(raw: dict) -> RecordingChannelMap:
    """Use the applied selection, preserving output_channel order and standby."""
    try:
        if type(raw.get("format_version")) is not int or raw["format_version"] != 1:
            raise ProbeGeometryError("Unsupported WILDX metadata version")
        model = raw["identity"]["model"]
        if model not in neuropixels_specs():
            raise ProbeGeometryError(f"Unsupported WILDX probe model: {model}")
        config = raw["configuration"]
        if config.get("state") != "applied" or config.get("selection_present") is not True:
            raise ProbeGeometryError("Metadata does not contain an applied channel selection")
        channels = config["channels"]
        if not isinstance(channels, list):
            raise ProbeGeometryError("Metadata channels must be a list")
        rows = []
        for row in channels:
            connected = _flag(row, "connected", True)
            standby = _flag(row, "standby", False)
            if model == NP2_FOUR and connected and "shank" not in row:
                raise ProbeGeometryError("Four-shank metadata requires an explicit shank for each connected channel")
            rows.append(dict(channel=row["output_channel"], site=row.get("site"),
                             shank=row.get("shank", 0), connected=connected,
                             active=connected and not standby))
        return _physical_map(model, rows)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ProbeGeometryError(f"Invalid WILDX probe metadata: {exc}") from exc


def _imro_model(identifier: str) -> str:
    # Restrict part numbers to the bundled physical layouts. Other NP1-family
    # variants can have different contact spacing or numbers of sites.
    if identifier in {"0", "NP1000", "NP1001"}:
        return NP1
    if identifier in {"21", "2003", "NP2000", "NP2003", "NP2004", "NP2005", "NP2006"}:
        return NP2_SINGLE
    if identifier in {"24", "2013", "NP2010", "NP2013", "NP2014"}:
        return NP2_FOUR
    raise ProbeGeometryError(f"Unsupported IMRO probe identifier: {identifier}")


def parse_imro(text: str) -> RecordingChannelMap:
    """Read NP1/NP2 IMRO, including numeric and part-number headers."""
    text = text.strip()
    entries = re.findall(r"\(([^()]*)\)", text)
    if len(entries) < 2 or re.sub(r"\([^()]*\)", "", text).strip():
        raise ProbeGeometryError("Invalid IMRO: expected a header and parenthesized channel entries")
    header = [part.strip() for part in entries[0].split(",")]
    if len(header) != 2:
        raise ProbeGeometryError("Unsupported IMRO header: expected (probe identifier,384)")
    model = _imro_model(header[0])
    try:
        count = int(header[1])
    except ValueError as exc:
        raise ProbeGeometryError("Invalid IMRO channel count") from exc
    if count != READOUT_CHANNELS or len(entries) - 1 != count:
        raise ProbeGeometryError("IMRO must declare 384 channels and contain exactly 384 entries")
    expected_fields = {NP1: 6, NP2_SINGLE: 4, NP2_FOUR: 5}[model]
    rows = []
    for entry in entries[1:]:
        tokens = entry.split()
        if len(tokens) != expected_fields or any(not re.fullmatch(r"\d+", t) for t in tokens):
            raise ProbeGeometryError(f"Invalid IMRO row for {model}: {entry}")
        values = list(map(int, tokens))
        channel = values[0]
        if channel >= READOUT_CHANNELS:
            raise ProbeGeometryError("IMRO channel ID must be between 0 and 383")
        if model == NP1:
            bank = values[1]
            if bank not in {0, 1, 2}:
                raise ProbeGeometryError("NP1 IMRO bank must be 0, 1, or 2")
            # NP1 electrode routing: each channel selects its site in a bank.
            site, shank = channel + READOUT_CHANNELS * bank, 0
        elif model == NP2_SINGLE:
            if values[1] not in {1, 2, 4, 8}:
                raise ProbeGeometryError("NP2 IMRO must select one bank per channel; multi-bank masks are unsupported")
            site, shank = values[3], 0
        else:
            if values[2] not in {0, 1, 2, 3}:
                raise ProbeGeometryError("NP2 four-shank IMRO bank must be between 0 and 3")
            site, shank = values[4], values[1]
        rows.append(dict(channel=channel, site=site, shank=shank))
    return _physical_map(model, rows)
