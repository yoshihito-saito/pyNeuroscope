from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Iterable

from .models import ChannelGroup, ProbeTemplate


class ProbeTemplateError(ValueError):
    """Raised when a probe template is malformed."""


def _template_path() -> Path:
    return Path(files("pyneuroscope.resources").joinpath("probe_templates.json"))


def _parse_group(raw: dict, index: int) -> ChannelGroup:
    channels = raw.get("channels")
    if not isinstance(channels, list) or not all(isinstance(ch, int) for ch in channels):
        raise ProbeTemplateError(f"Group {index} has invalid channels")
    return ChannelGroup(name=str(raw.get("name") or f"group{index + 1}"), channels=list(channels))


def parse_templates(raw_templates: Iterable[dict]) -> list[ProbeTemplate]:
    templates: list[ProbeTemplate] = []
    for idx, raw in enumerate(raw_templates):
        vendor = raw.get("vendor")
        model = raw.get("model")
        if not isinstance(vendor, str) or not isinstance(model, str):
            raise ProbeTemplateError(f"Template {idx} needs vendor and model")
        n_channels = raw.get("n_channels")
        if n_channels is not None and (not isinstance(n_channels, int) or n_channels <= 0):
            raise ProbeTemplateError(f"Template {vendor} {model} has invalid n_channels")
        groups = [_parse_group(group, i) for i, group in enumerate(raw.get("groups", []))]
        if not groups:
            raise ProbeTemplateError(f"Template {vendor} {model} needs groups")
        templates.append(
            ProbeTemplate(
                vendor=vendor,
                model=model,
                n_channels=n_channels,
                groups=groups,
                draft=bool(raw.get("draft", False)),
            )
        )
    return templates


def load_builtin_templates() -> list[ProbeTemplate]:
    with _template_path().open("r", encoding="utf-8") as handle:
        return parse_templates(json.load(handle))


def vendors(templates: Iterable[ProbeTemplate]) -> list[str]:
    return sorted({template.vendor for template in templates})


def models_for_vendor(templates: Iterable[ProbeTemplate], vendor: str) -> list[ProbeTemplate]:
    return [template for template in templates if template.vendor == vendor]


def generate_sequential_shank_groups(
    n_channels: int,
    n_probes: int,
    shanks_per_probe: int,
    channels_per_shank: int,
    *,
    start_channel: int = 0,
) -> list[ChannelGroup]:
    _require_positive_channels(n_channels)
    if n_probes <= 0:
        raise ProbeTemplateError("n_probes must be positive")
    if shanks_per_probe <= 0:
        raise ProbeTemplateError("shanks_per_probe must be positive")
    if channels_per_shank <= 0:
        raise ProbeTemplateError("channels_per_shank must be positive")
    if start_channel < 0 or start_channel >= n_channels:
        raise ProbeTemplateError("start_channel must be a valid channel index")

    groups: list[ChannelGroup] = []
    next_channel = start_channel
    for probe_index in range(n_probes):
        for shank_index in range(shanks_per_probe):
            if next_channel >= n_channels:
                return groups
            end_channel = min(n_channels, next_channel + channels_per_shank)
            groups.append(
                ChannelGroup(
                    f"probe{probe_index + 1}_shank{shank_index + 1}",
                    list(range(next_channel, end_channel)),
                )
            )
            next_channel = end_channel
    return groups


def _require_positive_channels(n_channels: int) -> None:
    if n_channels <= 0:
        raise ProbeTemplateError("n_channels must be positive")
