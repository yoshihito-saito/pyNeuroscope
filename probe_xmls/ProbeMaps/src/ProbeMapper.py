"""Passive probe to Intan headstage channel mapping.

This module provides an API for converting Omnetics probe
channel numbering to Intan headstage channel numbering.

Example
-------
Basic usage with a 16-channel single-shank probe::

    from ProbeMapper import ProbeMapper

    mapper = ProbeMapper(
        channel_map=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        chan_per_shank=[16],
        probe_type="neuronexus",
        probe_connector="H16",
        probe_name="A1x16",
        basepath="./output",
    )
    mapper.compute()

With custom Neuroscope parameters::

    from ProbeMapper import ProbeMapper, NeuroscopeParams

    params = NeuroscopeParams(
        sampling_rate=30000,
        lfp_sampling_rate=1250,
        extra_channels=[16, 17, 18],  # ADC/AUX channels
        skip_channels=[5],            # Skip broken channel
    )

    mapper = ProbeMapper(
        channel_map=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        chan_per_shank=[16],
        probe_type="neuronexus",
        probe_connector="H16",
        probe_name="A1x16",
        basepath="./output",
        export_txt=True,
        xml_params=params,
    )
    mapper.compute()

Multi-shank 64-channel probe (4 shanks x 16 channels)::

    mapper = ProbeMapper(
        channel_map=list(range(1, 65)),
        chan_per_shank=[16, 16, 16, 16],
        probe_type="neuronexus",
        probe_connector="H64",
        probe_name="A4x16",
        basepath="./output",
        export_txt=True,
    )
    mapper.compute()

Create a mapper from a previously generated Neuroscope XML::

    mapper = ProbeMapper.from_xml(
        "./output/neuronexus_A4x16_version1.xml",
        probe_type="neuronexus",
        probe_connector="H64",
    )
    mapper.compute()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Mapping, Sequence


@dataclass(frozen=True)
class Layout:
    """Immutable representation of a connector layout."""

    name: str
    pins: Sequence[int]

    @property
    def flipped(self) -> List[int]:
        """Return the reversed layout (useful for flipped headstage orientation)."""

        return list(reversed(self.pins))

    def row_flipped(self, row_size: int = 16) -> List[int]:
        """Return layout with each row of `row_size` channels reversed.
        
        This is useful when the Intan headstage is physically flipped,
        which reverses each row of pins independently.
        """
        pins = list(self.pins)
        result = []
        for i in range(0, len(pins), row_size):
            row = pins[i:i + row_size]
            result.extend(reversed(row))
        return result


@dataclass
class NeuroscopeParams:
    """Parameters for Neuroscope XML export.
    
    Attributes
    ----------
    n_bits : int
        Bit depth of the recording (default 16).
    sampling_rate : int
        Acquisition sampling rate in Hz (default 20000).
    lfp_sampling_rate : int
        LFP downsampled rate in Hz (default 1250).
    voltage_range : int
        Voltage range in mV (default 20).
    amplification : int
        Amplifier gain (default 1000).
    offset : int
        DC offset (default 0).
    n_samples : int
        Spike waveform samples (default 32).
    peak_sample_index : int
        Peak sample index in waveform (default 16).
    screen_gain : float
        Screen gain for display (default 0.2).
    skip_channels : Sequence[int]
        Channel indices to mark as skipped (default empty).
    extra_channels : Sequence[int]
        Additional non-probe channels to include (e.g., ADC, AUX). These are
        appended after probe channels and marked as skipped (default empty).
    channel_color : str
        Default channel color in hex (default "#000000").
    """

    n_bits: int = 16
    sampling_rate: int = 20000
    lfp_sampling_rate: int = 1250
    voltage_range: int = 20
    amplification: int = 1000
    offset: int = 0
    n_samples: int = 32
    peak_sample_index: int = 16
    screen_gain: float = 0.2
    skip_channels: Sequence[int] = field(default_factory=list)
    extra_channels: Sequence[int] = field(default_factory=list)
    channel_color: str = "#0080ff"


# Intan headstage layouts (preamp ordering)
# Add new Intan headstage sizes here: use channel count as key and a Layout with pins
# ordered per the Intan schematic (dorsal→ventral, left→right as needed).
INTAN_LAYOUTS: Mapping[int, Layout] = {
    16: Layout(
        name="intan16",
        pins=[11, 10, 9, 8, 7, 6, 5, 4, 12, 13, 14, 15, 0, 1, 2, 3],
    ),
    32: Layout(
        name="intan32",
        pins=[
            23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8,
            24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7,
        ],
    ),
    64: Layout(
        name="intan64",
        pins=[
            46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16,
            47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 21, 19, 17,
            49, 51, 53, 55, 57, 59, 61, 63, 1, 3, 5, 7, 9, 11, 13, 15,
            48, 50, 52, 54, 56, 58, 60, 62, 0, 2, 4, 6, 8, 10, 12, 14,
        ],
    ),
}


# Omnetics pinouts for  probe families
# Add new probes/connectors here: key is (probe_type, connector), value is Layout with
# Omnetics pin order as printed on the probe datasheet (match the mechanical numbering).
OMNETICS_LAYOUTS: Mapping[tuple[str, str], Layout] = {
    ("neuronexus", "H16"): Layout(
        name="neuronexus_H16",
        pins=[14, 15, 9, 16, 1, 8, 2, 3, 12, 11, 10, 13, 4, 7, 6, 5],
    ),
    ("neuronexus", "H32"): Layout(
        name="neuronexus_H32",
        pins=[
            18, 27, 28, 29, 17, 30, 31, 32, 1, 2, 3, 16, 4, 5, 6, 15,
            20, 21, 22, 23, 19, 24, 25, 26, 7, 8, 9, 14, 10, 11, 12, 13,
        ],
    ),
    ("neuronexus", "H64"): Layout(
        name="neuronexus_H64",
        pins=[
            34, 43, 44, 45, 33, 46, 47, 48, 17, 18, 19, 32, 20, 21, 22, 31,
            42, 41, 40, 35, 39, 38, 37, 36, 29, 28, 27, 26, 30, 25, 24, 23,
            64, 62, 60, 58, 56, 54, 52, 50, 15, 13, 11, 9, 7, 5, 3, 1,
            63, 61, 59, 57, 55, 53, 51, 49, 16, 14, 12, 10, 8, 6, 4, 2,
        ],
    ),
    ("neuronexus", "H64LP"): Layout(
        name="neuronexus_H64LP",
        pins=[
            37, 39, 40, 42, 43, 45, 46, 48, 17, 19, 20, 22, 23, 25, 26, 28,
            36, 38, 35, 41, 34, 44, 33, 47, 18, 32, 21, 31, 24, 30, 27, 29,
            64, 62, 60, 58, 56, 54, 52, 50, 15, 13, 11, 9, 7, 5, 3, 1,
            63, 61, 59, 57, 55, 53, 51, 49, 16, 14, 12, 10, 8, 6, 4, 2,
        ],
    ),
}


class ProbeMapper:
    """Utility for converting Omnetics channel numbering to Intan numbering.

    Parameters
    ----------
    channel_map:
        Desired ordering of probe channels as they appear on the shanks
        (dorsal to ventral, left to right). The length must match the chosen
        connector layout.
    chan_per_shank:
        Number of channels per shank, used for shank-wise outputs.
    probe_type / probe_connector:
        Keys into the predefined OMNETICS_LAYOUTS dictionary.
    probe_name:
        Identifier used in saved file names.
    basepath:
        Directory in which mapping files are written.
    export_txt:
        Save .txt channel mapping files (default False). Useful for manual
        XML creation when you need to copy/paste channel order.
    export_xml:
        Save Neuroscope .xml files (default True).
    xml_params:
        Parameters for Neuroscope XML export. If None, uses defaults.
    version_suffixes:
        Suffixes for the two output files: normal and flipped orientations.
        Default ("version1", "version2").
    flip_intan:
        If True, flip each row of 16 channels in the Intan layout. Use this
        when the headstage is physically mounted in the opposite orientation.
    """

    def __init__(
        self,
        *,
        channel_map: Sequence[int],
        chan_per_shank: Sequence[int],
        probe_type: str,
        probe_connector: str,
        probe_name: str,
        basepath: str | Path | None = None,
        export_txt: bool = False,
        export_xml: bool = True,
        xml_params: NeuroscopeParams | None = None,
        version_suffixes: Sequence[str] = ("version1", "version2"),
        flip_intan: bool = False,
    ) -> None:
        self.channel_map = list(channel_map)
        self.chan_per_shank = list(chan_per_shank)
        self.probe_type = probe_type.lower()
        self.probe_connector = probe_connector.upper()
        self.probe_name = probe_name
        self.basepath = Path(basepath) if basepath is not None else Path.cwd()
        self.export_txt = export_txt
        self.export_xml = export_xml
        self.xml_params = xml_params or NeuroscopeParams()
        self.version_suffixes = list(version_suffixes)
        self.flip_intan = flip_intan

        self.omnetics_layout = self._require_omnetics_layout()
        self.intan_layout = self._require_intan_layout()
        self._validate_lengths()
        self._validate_version_suffixes()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    def from_xml(
        cls,
        xml_path: str | Path,
        *,
        probe_type: str,
        probe_connector: str,
        probe_name: str | None = None,
        basepath: str | Path | None = None,
        export_txt: bool = False,
        export_xml: bool = True,
        version_suffixes: Sequence[str] = ("version1", "version2"),
        flip_intan: bool = False,
    ) -> ProbeMapper:
        """Create a ``ProbeMapper`` instance from a Neuroscope XML file.

        Parameters
        ----------
        xml_path : str or pathlib.Path
            Input Neuroscope XML path.
        probe_type : str
            Probe family key (example: ``"neuronexus"``).
        probe_connector : str
            Connector key (example: ``"H64"``, ``"H32"``, ``"H16"``).
        probe_name : str or None, optional
            Optional output name; inferred from filename when omitted.
        basepath : str or pathlib.Path or None, optional
            Output directory used by :meth:`compute`.
        export_txt : bool, optional
            Write ``.txt`` mapping outputs when :meth:`compute` is called.
        export_xml : bool, optional
            Write ``.xml`` mapping outputs when :meth:`compute` is called.
        version_suffixes : Sequence[str], optional
            Two output suffixes for regular/flipped outputs.
        flip_intan : bool, optional
            Use row-flipped Intan layout recovery for opposite headstage orientation.

        Notes
        -----
        This parser is designed for XML files following the schema generated by
        :meth:`_generate_xml`.
        """
        xml_path = Path(xml_path)
        root = ET.parse(xml_path).getroot()

        probe_type_normalized = probe_type.lower()
        probe_connector_normalized = probe_connector.upper()
        key = (probe_type_normalized, probe_connector_normalized)
        if key not in OMNETICS_LAYOUTS:
            known = ", ".join(sorted({f"{k[0]}-{k[1]}" for k in OMNETICS_LAYOUTS}))
            raise ValueError(f"Unsupported probe/connector: {key}. Known: {known}")

        expected_probe_channels = len(OMNETICS_LAYOUTS[key].pins)
        mapped, chan_per_shank, skip_channels, extra_channels = cls._parse_anatomical_groups(
            root, expected_probe_channels
        )
        xml_params = cls._parse_xml_params(root, skip_channels, extra_channels)

        mapper = cls(
            channel_map=list(range(1, expected_probe_channels + 1)),
            chan_per_shank=chan_per_shank,
            probe_type=probe_type_normalized,
            probe_connector=probe_connector_normalized,
            probe_name=probe_name or cls._infer_probe_name_from_path(xml_path, probe_type_normalized),
            basepath=basepath if basepath is not None else xml_path.parent,
            export_txt=export_txt,
            export_xml=export_xml,
            xml_params=xml_params,
            version_suffixes=version_suffixes,
            flip_intan=flip_intan,
        )

        intan_pins = (
            mapper.intan_layout.row_flipped(row_size=16)
            if flip_intan
            else list(mapper.intan_layout.pins)
        )
        mapper.channel_map = mapper._recover_channel_map(mapped, intan_pins)
        return mapper

    def compute(self) -> None:
        """Compute regular and flipped mappings, optionally write files, print report."""

        # Use row-flipped Intan layout if flip_intan is True
        if self.flip_intan:
            intan_pins = self.intan_layout.row_flipped(row_size=16)
            intan_flipped = list(reversed(intan_pins))
        else:
            intan_pins = self.intan_layout.pins
            intan_flipped = self.intan_layout.flipped

        device_channel_indices = self._map_channel_indices(intan_pins)
        flipped_channel_indices = self._map_channel_indices(intan_flipped)

        version_a, version_b = self.version_suffixes

        if self.export_txt:
            path_a = self._save_txt(device_channel_indices, suffix=version_a)
            path_b = self._save_txt(flipped_channel_indices, suffix=version_b)
            print(f"Saved: {path_a}")
            print(f"Saved: {path_b}")

        if self.export_xml:
            xml_a = self._save_xml(device_channel_indices, suffix=version_a)
            xml_b = self._save_xml(flipped_channel_indices, suffix=version_b)
            print(f"Saved: {xml_a}")
            print(f"Saved: {xml_b}")

        print(self._format_report(device_channel_indices, flipped_channel_indices))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_int(node: ET.Element | None, default: int) -> int:
        if node is None or node.text is None:
            return default
        try:
            return int(node.text.strip())
        except ValueError:
            return default

    @staticmethod
    def _parse_float(node: ET.Element | None, default: float) -> float:
        if node is None or node.text is None:
            return default
        try:
            return float(node.text.strip())
        except ValueError:
            return default

    @staticmethod
    def _infer_probe_name_from_path(xml_path: Path, probe_type: str) -> str:
        stem = xml_path.stem
        prefix = f"{probe_type}_"
        if stem.startswith(prefix):
            candidate = stem[len(prefix):]
            if "_" in candidate:
                probe_name, maybe_suffix = candidate.rsplit("_", 1)
                if maybe_suffix.lower().startswith("version"):
                    return probe_name
            return candidate
        return stem

    @staticmethod
    def _parse_anatomical_groups(
        root: ET.Element,
        expected_probe_channels: int,
    ) -> tuple[List[int], List[int], List[int], List[int]]:
        groups = root.findall("./anatomicalDescription/channelGroups/group")
        if not groups:
            raise ValueError("XML is missing anatomicalDescription/channelGroups/group entries")

        flattened_channels: List[int] = []
        flattened_skip_flags: List[bool] = []
        group_counts: List[int] = []

        for group in groups:
            group_channels = []
            group_skips = []
            for channel_node in group.findall("channel"):
                if channel_node.text is None:
                    continue
                ch = int(channel_node.text.strip())
                skip_attr = channel_node.attrib.get("skip", "0").strip()
                group_channels.append(ch)
                group_skips.append(skip_attr == "1")
            if not group_channels:
                continue
            flattened_channels.extend(group_channels)
            flattened_skip_flags.extend(group_skips)
            group_counts.append(len(group_channels))

        if len(flattened_channels) < expected_probe_channels:
            raise ValueError(
                "XML does not contain enough channels for the selected probe connector "
                f"({len(flattened_channels)} found, {expected_probe_channels} required)"
            )

        mapped = flattened_channels[:expected_probe_channels]
        mapped_skip_flags = flattened_skip_flags[:expected_probe_channels]
        extra_channels = flattened_channels[expected_probe_channels:]

        extra_count = len(flattened_channels) - expected_probe_channels
        chan_per_shank = list(group_counts)
        if extra_count:
            if not chan_per_shank or chan_per_shank[-1] < extra_count:
                raise ValueError("Could not infer shank channel counts from XML anatomical groups")
            chan_per_shank[-1] -= extra_count

        if sum(chan_per_shank) != expected_probe_channels:
            raise ValueError("Inferred chan_per_shank does not match expected probe channel count")

        skip_channels = [ch for ch, skip in zip(mapped, mapped_skip_flags) if skip]
        return mapped, chan_per_shank, skip_channels, extra_channels

    @classmethod
    def _parse_xml_params(
        cls,
        root: ET.Element,
        skip_channels: Sequence[int],
        extra_channels: Sequence[int],
    ) -> NeuroscopeParams:
        params = NeuroscopeParams()
        params.n_bits = cls._parse_int(root.find("./acquisitionSystem/nBits"), params.n_bits)
        params.sampling_rate = cls._parse_int(
            root.find("./acquisitionSystem/samplingRate"), params.sampling_rate
        )
        params.lfp_sampling_rate = cls._parse_int(
            root.find("./fieldPotentials/lfpSamplingRate"), params.lfp_sampling_rate
        )
        params.voltage_range = cls._parse_int(
            root.find("./acquisitionSystem/voltageRange"), params.voltage_range
        )
        params.amplification = cls._parse_int(
            root.find("./acquisitionSystem/amplification"), params.amplification
        )
        params.offset = cls._parse_int(root.find("./acquisitionSystem/offset"), params.offset)
        params.n_samples = cls._parse_int(root.find("./neuroscope/spikes/nSamples"), params.n_samples)
        params.peak_sample_index = cls._parse_int(
            root.find("./neuroscope/spikes/peakSampleIndex"), params.peak_sample_index
        )
        params.screen_gain = cls._parse_float(
            root.find("./neuroscope/miscellaneous/screenGain"), params.screen_gain
        )
        params.skip_channels = list(skip_channels)
        params.extra_channels = list(extra_channels)

        for node in root.findall("./neuroscope/channels/channelColors"):
            color_node = node.find("color")
            if color_node is None or color_node.text is None:
                continue
            color = color_node.text.strip()
            if color and color.lower() != "#ffffff":
                params.channel_color = color
                break

        return params

    def _recover_channel_map(self, mapped: Sequence[int], intan_preamp: Sequence[int]) -> List[int]:
        intan_index = {value: idx for idx, value in enumerate(intan_preamp)}
        if len(intan_index) != len(intan_preamp):
            raise ValueError("Intan layout contains duplicate channel indices")

        channel_map: List[int] = []
        for ch in mapped:
            if ch not in intan_index:
                raise ValueError(f"Channel {ch} not found in selected Intan layout")
            omnetics_idx = intan_index[ch]
            channel_map.append(self.omnetics_layout.pins[omnetics_idx])
        return channel_map

    def _require_omnetics_layout(self) -> Layout:
        key = (self.probe_type, self.probe_connector)
        if key not in OMNETICS_LAYOUTS:
            known = ", ".join(sorted({f"{k[0]}-{k[1]}" for k in OMNETICS_LAYOUTS}))
            raise ValueError(f"Unsupported probe/connector: {key}. Known: {known}")
        return OMNETICS_LAYOUTS[key]

    def _require_intan_layout(self) -> Layout:
        pin_count = len(self.omnetics_layout.pins)
        if pin_count not in INTAN_LAYOUTS:
            known_sizes = ", ".join(str(k) for k in sorted(INTAN_LAYOUTS))
            raise ValueError(f"No Intan layout for {pin_count} channels. Known sizes: {known_sizes}")
        return INTAN_LAYOUTS[pin_count]

    def _validate_lengths(self) -> None:
        if len(self.channel_map) != len(self.omnetics_layout.pins):
            raise ValueError(
                "channel_map length must match Omnetics layout length"
                f" ({len(self.channel_map)} vs {len(self.omnetics_layout.pins)})"
            )
        if sum(self.chan_per_shank) != len(self.channel_map):
            raise ValueError(
                "chan_per_shank must sum to the total number of channels"
            )

    def _validate_version_suffixes(self) -> None:
        if len(self.version_suffixes) != 2 or any(not v for v in self.version_suffixes):
            raise ValueError("version_suffixes must be a pair of non-empty strings")

    def _map_channel_indices(self, intan_preamp: Sequence[int]) -> List[int]:
        probe_index = {value: index for index, value in enumerate(self.omnetics_layout.pins)}
        return [intan_preamp[probe_index[ch]] for ch in self.channel_map]

    def _save_txt(self, mapped: Sequence[int], *, suffix: str) -> Path:
        path = self.basepath / f"{self.probe_type}_{self.probe_name}_{suffix}.txt"
        lines = self._format_lines(mapped)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines))
        return path

    def _format_lines(self, mapped: Sequence[int]) -> List[str]:
        lines: List[str] = []
        start = 0
        for shank_idx, count in enumerate(self.chan_per_shank, start=1):
            end = start + count
            lines.append(f"shank{shank_idx}: {' '.join(str(v) for v in mapped[start:end])}")
            start = end
        return lines

    def _format_report(self, mapped: Sequence[int], flipped: Sequence[int]) -> str:
        version_a, version_b = self.version_suffixes
        report_lines = [f"Contents of {self.probe_type}_{self.probe_name}_{version_a}.txt:"]
        report_lines.extend(self._format_lines(mapped))
        report_lines.append("")
        report_lines.append(f"Contents of {self.probe_type}_{self.probe_name}_{version_b}.txt:")
        report_lines.extend(self._format_lines(flipped))
        return "\n".join(report_lines)

    def _save_xml(self, mapped: Sequence[int], *, suffix: str) -> Path:
        """Generate and save a Neuroscope-compatible XML file."""
        path = self.basepath / f"{self.probe_type}_{self.probe_name}_{suffix}.xml"
        xml_content = self._generate_xml(mapped)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(xml_content, encoding="utf-8")
        return path

    def _generate_xml(self, mapped: Sequence[int]) -> str:
        """Generate Neuroscope XML content for the given channel mapping."""
        p = self.xml_params
        skip_set = set(p.skip_channels)
        extra_channels = list(p.extra_channels)

        # Total channels = probe channels + extra channels
        n_channels = len(mapped) + len(extra_channels)

        # Build anatomical description - one group per shank
        # Extra channels are appended to the LAST shank group (matching Neuroscope format)
        anatomical_groups = []
        spike_groups = []
        start = 0
        num_shanks = len(self.chan_per_shank)
        
        for shank_idx, count in enumerate(self.chan_per_shank):
            end = start + count
            shank_channels = mapped[start:end]
            
            # Anatomical group
            anat_lines = []
            for ch in shank_channels:
                skip = "1" if ch in skip_set else "0"
                anat_lines.append(f"    <channel skip=\"{skip}\">{ch}</channel>")
            
            # Add extra channels to the last shank group (or first if single shank)
            if shank_idx == num_shanks - 1 and extra_channels:
                for ch in extra_channels:
                    anat_lines.append(f"    <channel skip=\"1\">{ch}</channel>")
            
            anatomical_groups.append("   <group>\n" + "\n".join(anat_lines) + "\n   </group>")
            
            # Spike detection group (only non-skipped probe channels, no extras)
            spike_lines = []
            for ch in shank_channels:
                if ch not in skip_set:
                    spike_lines.append(f"     <channel>{ch}</channel>")
            spike_groups.append("   <group>\n    <channels>\n" + "\n".join(spike_lines) + "\n    </channels>\n   </group>")
            
            start = end

        # Build channel colors/offsets for all channels (sorted by channel number)
        all_channels = sorted(set(mapped) | set(extra_channels))
        channel_config_lines = []
        for ch in all_channels:
            # Use different color for extra channels
            is_extra = ch in extra_channels
            color = "#ffffff" if is_extra else p.channel_color
            anatomy_color = p.channel_color  # anatomy/spike colors stay the same
            channel_config_lines.append(f"""   <channelColors>
    <channel>{ch}</channel>
    <color>{color}</color>
    <anatomyColor>{anatomy_color}</anatomyColor>
    <spikeColor>{anatomy_color}</spikeColor>
   </channelColors>
   <channelOffset>
    <channel>{ch}</channel>
    <defaultOffset>0</defaultOffset>
   </channelOffset>""")

        xml = f"""<?xml version='1.0'?>
<parameters version="1.0" creator="neuroscope-2.0.0">
 <acquisitionSystem>
  <nBits>{p.n_bits}</nBits>
  <nChannels>{n_channels}</nChannels>
  <samplingRate>{p.sampling_rate}</samplingRate>
  <voltageRange>{p.voltage_range}</voltageRange>
  <amplification>{p.amplification}</amplification>
  <offset>{p.offset}</offset>
 </acquisitionSystem>
 <fieldPotentials>
  <lfpSamplingRate>{p.lfp_sampling_rate}</lfpSamplingRate>
 </fieldPotentials>
 <anatomicalDescription>
  <channelGroups>
{chr(10).join(anatomical_groups)}
  </channelGroups>
 </anatomicalDescription>
 <spikeDetection>
  <channelGroups>
{chr(10).join(spike_groups)}
  </channelGroups>
 </spikeDetection>
 <neuroscope version="2.0.0">
  <miscellaneous>
   <screenGain>{p.screen_gain}</screenGain>
   <traceBackgroundImage></traceBackgroundImage>
  </miscellaneous>
  <video>
   <rotate>0</rotate>
   <flip>0</flip>
   <videoImage></videoImage>
   <positionsBackground>0</positionsBackground>
  </video>
  <spikes>
   <nSamples>{p.n_samples}</nSamples>
   <peakSampleIndex>{p.peak_sample_index}</peakSampleIndex>
  </spikes>
  <channels>
{chr(10).join(channel_config_lines)}
  </channels>
 </neuroscope>
</parameters>"""
        return xml


__all__ = ["ProbeMapper", "Layout", "NeuroscopeParams", "INTAN_LAYOUTS", "OMNETICS_LAYOUTS"]


