"""Embedded pyNeuroscope probe maps, alongside standard NeuroSuite XML fields."""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from .xml_builder import XmlError, _root_from_path_or_text

NAMESPACE = "urn:pyneuroscope:probe-settings:1"
ET.register_namespace("pyneuroscope", NAMESPACE)


def _tag(name: str) -> str:
    return f"{{{NAMESPACE}}}{name}"


def embed_probe_settings(xml: str, settings: dict) -> str:
    root = _root_from_path_or_text(xml)
    old = root.find(_tag("probeSettings"))
    if old is not None:
        root.remove(old)
    node = ET.SubElement(root, _tag("probeSettings"),
                         version="1", colorMode=settings.get("color_mode", "all"),
                         colorMap=settings.get("color_map", "summer"))
    for probe in settings["probes"]:
        entry = ET.SubElement(node, _tag("probe"), nChannels=str(probe["n_channels"]),
                              type=probe.get("probe_type", ""), cmap=probe.get("cmap", "spring"))
        if probe.get("has_map", bool(probe.get("sites"))):
            mapping = ET.SubElement(entry, _tag("channelMap"))
            for site in probe.get("sites", []):
                ET.SubElement(mapping, _tag("channel"),
                              id=str(site["channel"]), x=str(site["x"]), y=str(site["y"]),
                              siteId=site.get("site_id", ""),
                              active="true" if site.get("active", True) else "false")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")


def read_probe_settings(path_or_text: str | Path) -> dict | None:
    node = _root_from_path_or_text(path_or_text).find(_tag("probeSettings"))
    if node is None:
        return None
    if node.get("version") != "1":
        raise XmlError("Unsupported embedded probe settings version")
    probes = []
    try:
        for probe in node.findall(_tag("probe")):
            mapping = probe.find(_tag("channelMap"))
            sites = []
            if mapping is not None:
                for site in mapping.findall(_tag("channel")):
                    active = site.get("active", "true")
                    if active not in {"true", "false"}:
                        raise XmlError("Embedded channel active flag must be true or false")
                    sites.append(dict(channel=int(site.attrib["id"]),
                                      x=float(site.attrib["x"]), y=float(site.attrib["y"]),
                                      site_id=site.get("siteId", ""), active=active == "true"))
            probes.append(dict(n_channels=int(probe.attrib["nChannels"]),
                               probe_type=probe.get("type", ""), cmap=probe.get("cmap", "spring"),
                               has_map=mapping is not None, sites=sites))
    except (KeyError, ValueError) as exc:
        raise XmlError(f"Invalid embedded probe settings: {exc}") from exc
    if not probes:
        raise XmlError("Embedded probe settings contain no probes")
    return dict(format="pyneuroscope-probes", version=1,
                color_mode=node.get("colorMode", "all"),
                color_map=node.get("colorMap", "summer"), probes=probes)
