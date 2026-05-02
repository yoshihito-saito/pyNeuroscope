from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

from .models import ChannelGroup


class AnatomicalMapError(ValueError):
    """Raised when an anatomical map cannot be parsed or generated."""


def build_anatomical_map_rows(
    groups: list[ChannelGroup],
    channel_regions: dict[int, str],
) -> list[list[str]]:
    if not groups:
        return []

    max_rows = max((len(group.channels) for group in groups), default=0)
    rows: list[list[str]] = []
    for row_index in range(max_rows):
        row: list[str] = []
        for group in groups:
            if row_index < len(group.channels):
                label = channel_regions.get(group.channels[row_index], "").strip()
                row.append(label)
            else:
                row.append("")
        rows.append(row)
    return rows


def build_anatomical_map_csv(
    groups: list[ChannelGroup],
    channel_regions: dict[int, str],
) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in build_anatomical_map_rows(groups, channel_regions):
        writer.writerow(row)
    return buffer.getvalue()


def parse_anatomical_map_csv(
    text: str,
    groups: list[ChannelGroup],
) -> dict[int, str]:
    reader = csv.reader(StringIO(text))
    rows = list(reader)
    channel_regions: dict[int, str] = {}

    for row_index, row in enumerate(rows):
        for group_index, value in enumerate(row):
            if group_index >= len(groups):
                continue
            group = groups[group_index]
            if row_index >= len(group.channels):
                continue
            label = value.strip()
            if label:
                channel_regions[group.channels[row_index]] = label
    return channel_regions


def load_anatomical_map_csv(path: str | Path, groups: list[ChannelGroup]) -> dict[int, str]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise AnatomicalMapError(f"Anatomical map not found: {csv_path}")
    return parse_anatomical_map_csv(csv_path.read_text(encoding="utf-8"), groups)

