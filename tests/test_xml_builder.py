from pathlib import Path

from pyneuroscope.models import ChannelGroup, RecordingMetadata
from pyneuroscope.xml_builder import build_neurocode_xml, parse_neurosuite_xml


def test_builds_xml_with_skip_attributes_and_order() -> None:
    groups = [ChannelGroup("g1", [3, 1]), ChannelGroup("g2", [0, 2])]

    xml = build_neurocode_xml(4, 20000, 1250, groups, {1})
    metadata, parsed_groups, bad = parse_neurosuite_xml(xml)

    assert metadata.n_channels == 4
    assert [group.channels for group in parsed_groups] == [[3, 1], [0, 2]]
    assert bad == {1}
    assert '<channel skip="1">1</channel>' in xml


def test_parses_classic_channel_without_skip_as_normal() -> None:
    xml = """<parameters>
      <acquisitionSystem><nChannels>2</nChannels><samplingRate>20000</samplingRate></acquisitionSystem>
      <fieldPotentials><lfpSamplingRate>1250</lfpSamplingRate></fieldPotentials>
      <anatomicalDescription><channelGroups><group><channel>1</channel></group></channelGroups></anatomicalDescription>
    </parameters>"""

    _, groups, bad = parse_neurosuite_xml(xml)

    assert groups[0].channels == [1]
    assert bad == set()


def test_parses_demo_xml_when_present() -> None:
    path = Path("demo/test_recording_251224_122225/amplifier.xml")
    if not path.exists():
        return

    metadata, groups, bad = parse_neurosuite_xml(path)

    assert metadata.n_channels == 128
    assert metadata.n_bits == 16
    assert metadata.sampling_rate == 20000
    assert metadata.lfp_sampling_rate == 1250
    assert len(groups) == 8
    assert 34 in bad


def test_includes_optional_acquisition_fields() -> None:
    metadata = RecordingMetadata(n_bits=16, voltage_range=20, amplification=1000, offset=0)

    xml = build_neurocode_xml(2, 20000, 1250, [ChannelGroup("g", [0, 1])], metadata=metadata)

    assert "<voltageRange>20</voltageRange>" in xml
    assert "<amplification>1000</amplification>" in xml


def test_default_metadata_includes_neurocode_acquisition_fields() -> None:
    xml = build_neurocode_xml(2, 20000, 1250, [ChannelGroup("g", [0, 1])])

    assert "<voltageRange>20</voltageRange>" in xml
    assert "<amplification>1000</amplification>" in xml


def test_includes_neuroscope_channel_colors_from_cmap() -> None:
    xml = build_neurocode_xml(
        2,
        20000,
        1250,
        [ChannelGroup("g", [0, 1])],
        channel_colors={0: "#112233", 1: "#aabbcc"},
    )

    assert "<neuroscope>" in xml
    assert "<channel>0</channel>" in xml
    assert "<color>#112233</color>" in xml
    assert "<anatomyColor>#112233</anatomyColor>" in xml
    assert "<spikeColor>#112233</spikeColor>" in xml
    assert "<defaultOffset>0</defaultOffset>" in xml
    assert "<color>#aabbcc</color>" in xml
