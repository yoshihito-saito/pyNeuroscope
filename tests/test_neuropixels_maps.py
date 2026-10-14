import copy
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from pyneuroscope.main_window import MainWindow, ProbeConfig
from pyneuroscope.neuropixels_maps import parse_imro
from pyneuroscope.probe_geometry import (
    ProbeGeometryError, available_probe_geometries, load_probe_geometry,
    load_recording_channel_map, parse_recording_channel_map,
)


def receipt(bank=0):
    # Matches the WILDX v1 schema inspected in the user's recording.
    return dict(format="WILDX NP applied probe receipt", format_version=1,
                identity=dict(model="Neuropixels 1.0"), configuration=dict(
                    state="applied", selection_present=True, channels=[dict(
                        output_channel=ch, site=ch + bank * 384, connected=True,
                        standby=False) for ch in range(384)]))


def imro(identifier="0", rows=None):
    if rows is None:
        rows = [f"{ch} 1 0 500 250 1" for ch in range(384)]
    return f"({identifier},384)" + "".join(f"({row})" for row in rows)


NP2_SINGLE = "Neuropixels 2.0 single shank"
NP2_FOUR = "Neuropixels 2.0 four shank"


def mapped_receipt(model):
    if model == "Neuropixels 1.0":
        return receipt(1)
    raw = receipt()
    raw["identity"]["model"] = "Neuropixels 2.0"
    for row in raw["configuration"]["channels"]:
        ch = row["output_channel"]
        row["site"] = 700 + 383 - ch if model == NP2_SINGLE else 600 + ch % 96
        if model == NP2_FOUR:
            row["shank"] = ch // 96
    raw["configuration"]["channels"][5].update(connected=False, site=None)
    raw["configuration"]["channels"][9]["standby"] = True
    raw["configuration"]["channels"].reverse()
    return raw


def mapped_imro(model, identifier=None):
    if model == "Neuropixels 1.0":
        return imro()
    if model == NP2_SINGLE:
        return imro(identifier or "NP2003", [f"{ch} 4 0 {700+383-ch}" for ch in range(384)])
    return imro(identifier or "NP2013", [f"{ch} {ch//96} 1 0 {600+ch%96}" for ch in range(384)])


@pytest.mark.parametrize("model", [NP2_SINGLE, NP2_FOUR])
@pytest.mark.parametrize("identity", ["generic", "missing"])
def test_manual_np2_type_interprets_generic_or_missing_json_model(model, identity):
    raw = mapped_receipt(model)
    if identity == "missing":
        del raw["identity"]
    mapping = parse_recording_channel_map(raw, probe_type=model)
    geometry = load_probe_geometry(model)
    assert mapping.probe_type == model and mapping.n_channels == 384
    assert mapping.active_channels == set(range(384)) - {5, 9}
    assert 5 not in mapping.positions and 9 in mapping.positions
    assert mapping.positions[0] == geometry.physical_sites[mapping.site_ids[0]]
    assert mapping.site_ids[0] == ("s0_front_1083" if model == NP2_SINGLE else "s0_front_0600")
    if model == NP2_FOUR:
        assert mapping.site_ids[96] == "s1_front_0600"


@pytest.mark.parametrize("model", [NP2_SINGLE, NP2_FOUR])
def test_manual_imro_type_handles_custom_header_but_checks_row_format(model):
    mapping = parse_imro(mapped_imro(model, "NP2-custom"), probe_type=model)
    assert mapping.probe_type == model
    with pytest.raises(ProbeGeometryError):
        parse_imro(mapped_imro(model, "NP2-custom"), probe_type=NP2_FOUR if model == NP2_SINGLE else NP2_SINGLE)


def test_manual_imro_type_rejects_conflicting_recognized_header():
    with pytest.raises(ProbeGeometryError, match="selected probe type"):
        parse_imro(mapped_imro(NP2_FOUR), probe_type=NP2_SINGLE)


@pytest.mark.parametrize("model", ["Neuropixels 1.0", NP2_SINGLE, NP2_FOUR])
@pytest.mark.parametrize("extension", [".imro", ".json"])
@pytest.mark.parametrize("save_kind", ["probe", "session"])
def test_np_manual_import_and_saved_settings_reusable_without_original_map(tmp_path, monkeypatch, model, extension, save_kind):
    from pyneuroscope import main_window as module
    from PySide6.QtWidgets import QComboBox
    app = QApplication.instance() or QApplication([])
    raw_path = tmp_path / ("active" + extension)
    raw_path.write_text(mapped_imro(model) if extension == ".imro" else json.dumps(mapped_receipt(model)))
    saved = tmp_path / "reusable.xml"
    w = MainWindow()
    other = None
    try:
        # Select the geometry using the same combo as the Recording tab.
        combo = w.probe_rows_layout.itemAt(0).widget().findChild(QComboBox)
        combo.setCurrentIndex(combo.findData(model))
        monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a: (str(raw_path), ""))
        w._load_probe_channel_map(0)
        assert w.probes[0].probe_type == model and w.probes[0].n_channels == 384
        assert len(w.probe_viewer._physical_sites) == {"Neuropixels 1.0": 960, NP2_SINGLE: 1280, NP2_FOUR: 5120}[model]
        w.probes[0].cmap = "coolwarm"
        w.color_mode.setCurrentText("per probe")
        w.bad_channels = {10}
        monkeypatch.setattr(module.QFileDialog, "getSaveFileName", lambda *a: (str(saved), ""))
        monkeypatch.setattr(module.QMessageBox, "information", lambda *a: None)
        if save_kind == "probe":
            w._save_probe_xml(0)
        else:
            w._save_xml()
        expected_map = w.probes[0].channel_map
        raw_path.unlink()
        other = MainWindow()
        if save_kind == "probe":
            monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a: (str(saved), ""))
            other._load_probe_xml(0)
        else:
            other._apply_xml_metadata(saved)
        assert other.probes[0].probe_type == model
        assert other.probes[0].cmap == "coolwarm"
        assert other.probes[0].channel_map.site_ids == expected_map.site_ids
        assert other.probes[0].channel_map.positions == expected_map.positions
        assert other.probes[0].channel_map.active_channels == expected_map.active_channels
        assert {ch for g in other.groups for ch in g.channels} == set(range(384))
        assert other.bad_channels == {10}
        assert other.color_mode.currentText() == "per probe"
        assert {i.channel for i in other.viewer._layout_items} == expected_map.active_channels
    finally:
        if other is not None:
            other.close()
        w.close()


def test_metadata_uses_output_ids_not_row_or_site_order(tmp_path):
    raw = receipt(1)
    raw["configuration"]["channels"].reverse()
    path = tmp_path / "neuropixels_probe1_metadata.json"
    path.write_text(json.dumps(raw), encoding="utf-8-sig")
    mapping = load_recording_channel_map(path)
    assert mapping.n_channels == 384
    assert mapping.probe_type == "Neuropixels 1.0"
    assert mapping.site_ids[0] == "s0_front_0384"
    assert mapping.site_ids[383] == "s0_front_0767"
    assert mapping.positions[0] == load_probe_geometry(mapping.probe_type).physical_sites["s0_front_0384"]


@pytest.mark.parametrize("disconnected_site", [None, 65535])
def test_metadata_disconnected_and_standby_keep_dat_column_ids(disconnected_site):
    raw = receipt()
    rows = raw["configuration"]["channels"]
    rows[2].update(connected=False, site=disconnected_site)
    rows[3]["standby"] = True
    mapping = parse_recording_channel_map(raw)
    assert mapping.n_channels == 384
    assert 2 not in mapping.positions
    assert mapping.site_ids[4] == "s0_front_0004"
    assert 3 in mapping.positions and 3 not in mapping.active_channels
    assert len(mapping.active_channels) == 382


def test_metadata_four_shank_requires_explicit_shank():
    raw = receipt()
    raw["identity"]["model"] = "Neuropixels 2.0 four shank"
    with pytest.raises(ProbeGeometryError, match="explicit shank"):
        parse_recording_channel_map(raw)
    for row in raw["configuration"]["channels"]:
        row["shank"] = 3
    mapping = parse_recording_channel_map(raw)
    assert mapping.site_ids[0] == "s3_front_0000"


@pytest.mark.parametrize("change", [
    lambda r: r.update(format_version=2),
    lambda r: r["identity"].update(model="Unknown NP"),
    lambda r: r["configuration"].update(state="requested"),
    lambda r: r["configuration"].update(selection_present=False),
    lambda r: r["configuration"]["channels"].pop(),
    lambda r: r["configuration"]["channels"][0].update(output_channel=1),
    lambda r: r["configuration"]["channels"][0].update(output_channel=0.5),
    lambda r: r["configuration"]["channels"][0].update(site=1),
    lambda r: r["configuration"]["channels"][0].update(site=960),
    lambda r: r["configuration"]["channels"][0].update(standby="false"),
])
def test_metadata_rejects_invalid_assignments(change):
    raw = receipt()
    change(raw)
    with pytest.raises(ProbeGeometryError):
        parse_recording_channel_map(raw)


@pytest.mark.parametrize("identifier", ["0", "NP1000", "NP1001"])
def test_np1_imro_matches_applied_metadata(identifier):
    mapping = parse_imro(imro(identifier, [f"{ch} 1 0 500 250 1" for ch in reversed(range(384))]))
    assert mapping == parse_recording_channel_map(receipt(1))


def test_np1_imro_bank_two_boundary():
    mapping = parse_imro(imro(rows=[f"{ch} {2 if ch < 192 else 1} 0 500 250 1" for ch in range(384)]))
    assert mapping.site_ids[191] == "s0_front_0959"
    assert mapping.site_ids[192] == "s0_front_0576"
    with pytest.raises(ProbeGeometryError):
        parse_imro(imro(rows=[f"{ch} 2 0 500 250 1" for ch in range(384)]))


@pytest.mark.parametrize("identifier", ["21", "2003", "NP2003"])
def test_np2_single_imro_uses_explicit_electrode(identifier):
    mapping = parse_imro(imro(identifier, [f"{ch} 2 0 {767-ch}" for ch in range(384)]))
    assert mapping.probe_type == "Neuropixels 2.0 single shank"
    assert mapping.site_ids[0] == "s0_front_0767"
    assert mapping.site_ids[383] == "s0_front_0384"


@pytest.mark.parametrize("identifier", ["24", "2013", "NP2013", "NP2014"])
def test_np2_four_imro_distinguishes_shanks(identifier):
    mapping = parse_imro(imro(identifier, [f"{ch} {ch//96} 0 0 {ch%96}" for ch in range(384)]))
    assert mapping.probe_type == "Neuropixels 2.0 four shank"
    assert mapping.site_ids[96] == "s1_front_0000"
    assert mapping.positions[96].x - mapping.positions[0].x == 250


@pytest.mark.parametrize("text", [
    imro("NP1110"), imro()[:-1], "junk" + imro(), imro() + "()",
    imro(rows=["0 0 0 500 250 1"] * 384),
    imro(rows=[f"{ch} 3 0 500 250 1" for ch in range(384)]),
    imro("NP2003", [f"{ch} 3 0 {ch}" for ch in range(384)]),
    imro("NP2013", [f"{ch} 4 0 0 {ch}" for ch in range(384)]),
    imro(rows=["0 0 0 500 250 1"]),
])
def test_imro_rejects_malformed_or_unsupported_maps(text):
    with pytest.raises(ProbeGeometryError):
        parse_imro(text)


def test_each_probe_can_choose_metadata_or_imro_and_round_trip(tmp_path, monkeypatch):
    from pyneuroscope import main_window as module
    app = QApplication.instance() or QApplication([])
    first = tmp_path / "neuropixels_probe1_metadata.json"
    second = tmp_path / "probe2.imro"
    first.write_text(json.dumps(receipt(1)), encoding="utf-8")
    second.write_text(imro("NP2013", [f"{ch} 3 0 0 {ch}" for ch in range(384)]), encoding="utf-8")
    chosen = iter([str(first), str(second)])
    filters = []

    def choose(*args):
        filters.append(args[3])
        return next(chosen), ""

    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", choose)
    w = MainWindow()
    other = None
    try:
        w.probes = [ProbeConfig(4, cmap="Blues"), ProbeConfig(4, cmap="Reds")]
        w._apply_probe_configs_to_model()
        rates = (w.sampling_rate.value(), w.lfp_sampling_rate.value())
        w._load_probe_channel_map(0)
        w._load_probe_channel_map(1)
        assert w.n_channels.value() == 768
        assert [p.probe_type for p in w.probes] == ["Neuropixels 1.0", "Neuropixels 2.0 four shank"]
        assert [p.cmap for p in w.probes] == ["Blues", "Reds"]
        assert w.probes[0].channel_map.site_ids[0] == "s0_front_0384"
        assert w.probes[1].channel_map.site_ids[0] == "s3_front_0000"
        assert set(w._probe_scene()[0]) == set(range(768))
        assert len(w._probe_scene()[1]) == 6080
        assert (w.sampling_rate.value(), w.lfp_sampling_rate.value()) == rates
        assert all("*.imro" in f and "*.json" in f for f in filters)

        path = tmp_path / "session.xml"
        monkeypatch.setattr(module.QFileDialog, "getSaveFileName", lambda *a: (str(path), ""))
        monkeypatch.setattr(module.QMessageBox, "information", lambda *a: None)
        w._save_xml()
        other = MainWindow()
        other._apply_xml_metadata(path)
        assert other.n_channels.value() == 768
        assert other.probes[1].channel_map.site_ids == w.probes[1].channel_map.site_ids
        assert other.probes[0].channel_map.active_channels == set(range(384))
    finally:
        if other is not None:
            other.close()
        w.close()


def test_invalid_map_does_not_replace_existing_probe():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    try:
        original = copy.deepcopy(w.probes)
        mapping = parse_imro(imro())
        mapping.positions[0] = type(mapping.positions[0])(9999, 0)
        with pytest.raises(ProbeGeometryError):
            w._set_probe_channel_map(0, mapping)
        assert w.probes == original
    finally:
        w.close()


def test_map_import_preserves_matching_xml_groups_and_bad_channels(tmp_path):
    from pyneuroscope.models import ChannelGroup
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    try:
        path = tmp_path / "probe.xml"
        groups = [ChannelGroup("a", list(range(384)))]
        w.probes = [ProbeConfig(384, xml_path=path, groups=groups, bad_channels={3})]
        w._apply_probe_configs_to_model()
        w._set_probe_channel_map(0, parse_imro(imro()))
        assert w.probes[0].xml_path == path
        assert [g.name for g in w.probes[0].groups] == [g.name for g in groups]
        assert set(w.probes[0].groups[0].channels) == set(groups[0].channels)
        assert w.probes[0].bad_channels == {3}
    finally:
        w.close()


def test_legacy_pattern_is_hidden_but_saved_config_is_loadable():
    assert "neuropixel" not in available_probe_geometries()
    assert load_probe_geometry("neuropixel") is not None
