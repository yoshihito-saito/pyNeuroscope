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
        grouping = raw.get("grouping")
        if grouping is not None and grouping not in {"linear", "tetrode", "fixed_shank"}:
            raise ProbeTemplateError(f"Template {vendor} {model} has invalid grouping")
        if not groups and grouping is None:
            raise ProbeTemplateError(f"Template {vendor} {model} needs groups or grouping")
        templates.append(
            ProbeTemplate(
                vendor=vendor,
                model=model,
                n_channels=n_channels,
                groups=groups,
                grouping=grouping,
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


def generate_linear_groups(n_channels: int) -> list[ChannelGroup]:
    _require_positive_channels(n_channels)
    return [ChannelGroup("group1", list(range(n_channels)))]


def generate_tetrode_groups(n_channels: int) -> list[ChannelGroup]:
    return generate_fixed_size_groups(n_channels, 4, prefix="tetrode")


def generate_fixed_size_groups(
    n_channels: int,
    channels_per_group: int,
    *,
    prefix: str = "shank",
) -> list[ChannelGroup]:
    _require_positive_channels(n_channels)
    if channels_per_group <= 0:
        raise ProbeTemplateError("channels_per_group must be positive")
    groups: list[ChannelGroup] = []
    for start in range(0, n_channels, channels_per_group):
        end = min(n_channels, start + channels_per_group)
        groups.append(ChannelGroup(f"{prefix}{len(groups) + 1}", list(range(start, end))))
    return groups


def _require_positive_channels(n_channels: int) -> None:
    if n_channels <= 0:
        raise ProbeTemplateError("n_channels must be positive")
