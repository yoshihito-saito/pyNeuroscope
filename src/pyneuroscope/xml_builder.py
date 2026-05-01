from __future__ import annotations

from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

from .models import ChannelGroup, RecordingMetadata


class XmlError(ValueError):
    """Raised when NeuroSuite XML cannot be parsed or generated."""


def _text(parent: ET.Element, tag: str, value: object) -> ET.Element:
    child = ET.SubElement(parent, tag)
    child.text = _format_value(value)
    return child


def _format_value(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def pretty_xml(root: ET.Element) -> str:
    rough = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def build_neurocode_xml(
    n_channels: int,
    sampling_rate: float,
    lfp_sampling_rate: float,
    groups: list[ChannelGroup],
    bad_channels: set[int] | None = None,
    *,
    metadata: RecordingMetadata | None = None,
    emit_skip_attribute: bool = True,
) -> str:
    if n_channels <= 0:
        raise XmlError("n_channels must be positive")
    if sampling_rate <= 0 or lfp_sampling_rate <= 0:
        raise XmlError("sampling rates must be positive")

    bad = bad_channels or set()
    root = ET.Element("parameters")
    acquisition = ET.SubElement(root, "acquisitionSystem")
    source = metadata or RecordingMetadata(
        n_channels=n_channels,
        sampling_rate=sampling_rate,
        lfp_sampling_rate=lfp_sampling_rate,
    )
    if source.n_bits is not None:
        _text(acquisition, "nBits", source.n_bits)
    _text(acquisition, "nChannels", n_channels)
    _text(acquisition, "samplingRate", sampling_rate)
    if source.voltage_range is not None:
        _text(acquisition, "voltageRange", source.voltage_range)
    if source.amplification is not None:
        _text(acquisition, "amplification", source.amplification)
    if source.offset is not None:
        _text(acquisition, "offset", source.offset)

    field_potentials = ET.SubElement(root, "fieldPotentials")
    _text(field_potentials, "lfpSamplingRate", lfp_sampling_rate)

    anatomical = ET.SubElement(root, "anatomicalDescription")
    channel_groups = ET.SubElement(anatomical, "channelGroups")
    for group in groups:
        group_el = ET.SubElement(channel_groups, "group")
        for channel in group.channels:
            channel_el = ET.SubElement(group_el, "channel")
            if emit_skip_attribute:
                channel_el.set("skip", "1" if channel in bad else "0")
            channel_el.text = str(channel)
    return pretty_xml(root)


def parse_neurosuite_xml(path_or_text: str | Path) -> tuple[RecordingMetadata, list[ChannelGroup], set[int]]:
    root = _root_from_path_or_text(path_or_text)
    if root.tag != "parameters":
        raise XmlError("Root element must be <parameters>")

    acquisition = root.find("acquisitionSystem")
    field_potentials = root.find("fieldPotentials")
    if acquisition is None:
        raise XmlError("Missing acquisitionSystem")

    metadata = RecordingMetadata(
        n_channels=_int_text(acquisition, "nChannels", required=True),
        sampling_rate=_float_text(acquisition, "samplingRate", required=True),
        lfp_sampling_rate=_float_text(field_potentials, "lfpSamplingRate", default=0.0),
        n_bits=_int_text(acquisition, "nBits", default=None),
        voltage_range=_float_text(acquisition, "voltageRange", default=None),
        amplification=_float_text(acquisition, "amplification", default=None),
        offset=_float_text(acquisition, "offset", default=0.0),
    )

    channel_groups_el = root.find("./anatomicalDescription/channelGroups")
    groups: list[ChannelGroup] = []
    bad_channels: set[int] = set()
    if channel_groups_el is not None:
        for index, group_el in enumerate(channel_groups_el.findall("group")):
            channels: list[int] = []
            for channel_el in group_el.findall("channel"):
                if channel_el.text is None:
                    raise XmlError("Channel element has no text")
                channel = int(channel_el.text.strip())
                channels.append(channel)
                if channel_el.get("skip", "0") == "1":
                    bad_channels.add(channel)
            groups.append(ChannelGroup(f"group{index + 1}", channels))

    return metadata, groups, bad_channels


def parse_xml_text(text: str) -> ET.Element:
    try:
        return ET.fromstring(text)
    except ET.ParseError as exc:
        raise XmlError(str(exc)) from exc


def _root_from_path_or_text(path_or_text: str | Path) -> ET.Element:
    text_or_path = str(path_or_text)
    if not isinstance(path_or_text, Path) and text_or_path.lstrip().startswith("<"):
        return parse_xml_text(text_or_path)
    if isinstance(path_or_text, Path) or ("\n" not in text_or_path and Path(text_or_path).exists()):
        try:
            return ET.parse(path_or_text).getroot()
        except ET.ParseError as exc:
            raise XmlError(str(exc)) from exc
    return parse_xml_text(text_or_path)


def _int_text(parent: ET.Element | None, tag: str, *, required: bool = False, default: int | None = 0) -> int | None:
    value = _find_text(parent, tag, required=required)
    return int(value) if value is not None else default


def _float_text(
    parent: ET.Element | None,
    tag: str,
    *,
    required: bool = False,
    default: float | None = 0.0,
) -> float | None:
    value = _find_text(parent, tag, required=required)
    return float(value) if value is not None else default


def _find_text(parent: ET.Element | None, tag: str, *, required: bool) -> str | None:
    if parent is None:
        if required:
            raise XmlError(f"Missing {tag}")
        return None
    child = parent.find(tag)
    if child is None or child.text is None or not child.text.strip():
        if required:
            raise XmlError(f"Missing {tag}")
        return None
    return child.text.strip()
