import json
import os
from shutil import copy2

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QPushButton

from pyneuroscope.main_window import MainWindow
from pyneuroscope.probe_settings import read_probe_settings
from pyneuroscope.xml_builder import build_neurocode_xml, parse_neurosuite_xml

NP1 = "Neuropixels 1.0"
NP2_SINGLE = "Neuropixels 2.0 single shank"
NP2_FOUR = "Neuropixels 2.0 four shank"


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def imro_text(identifier, model):
    if model == NP1:
        rows = [f"{ch} 1 0 500 250 1" for ch in range(384)]
    elif model == NP2_SINGLE:
        rows = [f"{ch} 2 0 {ch + 384}" for ch in range(384)]
    else:
        rows = [f"{ch} 2 1 0 {ch + 384}" for ch in range(384)]
    return f"({identifier},384)" + "".join(f"({row})" for row in rows)


@pytest.mark.parametrize("identifier,model", [
    ("0", NP1), ("NP1000", NP1), ("21", NP2_SINGLE),
    ("NP2003", NP2_SINGLE), ("24", NP2_FOUR), ("NP2013", NP2_FOUR),
])
def test_imro_button_infers_type_without_a_prior_selection(app, tmp_path, monkeypatch, identifier, model):
    from pyneuroscope import main_window as module
    path = tmp_path / "probe.IMRO"
    path.write_text(imro_text(identifier, model))
    calls = []
    def choose(*args):
        calls.append(args)
        return str(path), ""
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", choose)
    w = MainWindow()
    try:
        assert w.probes[0].probe_type == ""
        w.sampling_rate.setValue(25000)
        button = next(b for b in w.probe_rows_layout.itemAt(0).widget().findChildren(QPushButton)
                      if b.text() == "Probe XML or IMRO")
        button.click()
        assert len(calls) == 1
        assert w.probes[0].probe_type == model
        assert w.n_channels.value() == 384
        assert w.sampling_rate.value() == 25000
        assert w.probes[0].xml_path is None
        assert len(w.probes[0].channel_map.active_channels) == 384
        assert {item.channel for item in w.viewer._layout_items} == set(range(384))
        expected_shank = 2 if model == NP2_FOUR else 0
        assert w.probes[0].channel_map.site_ids[10] == f"s{expected_shank}_front_0394"
    finally:
        w.close()


def test_known_imro_header_replaces_previous_type_and_custom_header_uses_manual_type(app, tmp_path, monkeypatch):
    from pyneuroscope import main_window as module
    path = tmp_path / "probe.imro"
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a: (str(path), ""))
    w = MainWindow()
    try:
        w._probe_type_changed(0, NP1)
        path.write_text(imro_text("NP2013", NP2_FOUR))
        w._load_probe_file(0)
        assert w.probes[0].probe_type == NP2_FOUR
        path.write_text(imro_text("custom", NP2_FOUR))
        w._load_probe_file(0)
        assert w.probes[0].probe_type == NP2_FOUR
        assert w.probes[0].channel_map.site_ids[10] == "s2_front_0394"
    finally:
        w.close()


def test_xml_or_imro_xml_path_uses_one_dialog_and_keeps_recording_rates(app, tmp_path, monkeypatch):
    from pyneuroscope import main_window as module
    path = tmp_path / "probe.xml"
    path.write_text("<parameters><acquisitionSystem><nChannels>3</nChannels></acquisitionSystem>"
                    "<anatomicalDescription><channelGroups><group><channel>2</channel><channel>0</channel>"
                    "</group><group><channel>1</channel></group></channelGroups></anatomicalDescription></parameters>")
    calls = []
    def choose(*args):
        calls.append(args)
        return str(path), ""
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", choose)
    w = MainWindow()
    try:
        w.sampling_rate.setValue(25000)
        w._load_probe_file(0)
        assert len(calls) == 1
        assert w.n_channels.value() == 3 and w.sampling_rate.value() == 25000
        assert [group.channels for group in w.groups] == [[2, 0], [1]]
    finally:
        w.close()


@pytest.mark.parametrize("legacy_sidecar", [False, True])
def test_session_reuse_from_single_xml_and_legacy_json(app, tmp_path, monkeypatch, legacy_sidecar):
    from pyneuroscope import main_window as module
    imro = tmp_path / "probe.imro"
    imro.write_text(imro_text("NP2013", NP2_FOUR))
    saved = tmp_path / "session.xml"
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a: (str(imro), ""))
    monkeypatch.setattr(module.QFileDialog, "getSaveFileName", lambda *a: (str(saved), ""))
    monkeypatch.setattr(module.QMessageBox, "information", lambda *a: None)
    w, other = MainWindow(), None
    try:
        # Mix an ordinary probe with NP2; DAT offsets must survive reload.
        w._add_probe()
        w._load_probe_file(1)
        w.probes[1].channel_map.active_channels.remove(5)
        w.probes[1].cmap = "coolwarm"
        w.color_map.setCurrentText("red_white_black")
        w.color_mode.setCurrentText("per probe")
        w.bad_channels = {1, 12}
        original_positions = dict(w.probes[1].channel_map.positions)
        standard_xml = build_neurocode_xml(w.n_channels.value(), w.sampling_rate.value(), w.lfp_sampling_rate.value(),
                                          w.groups, w.bad_channels, channel_colors=w.channel_colors)
        if legacy_sidecar:
            saved.write_text(standard_xml)
            settings = dict(format="pyneuroscope-probes", version=1, color_mode="per probe",
                            probes=[w._probe_settings(probe) for probe in w.probes])
            saved.with_suffix(".probes.json").write_text(json.dumps(settings))
        else:
            w._save_xml()
            assert not saved.with_suffix(".probes.json").exists()
            # Standard NeuroSuite groups/count/rates/bad channels remain intact.
            assert parse_neurosuite_xml(saved) == parse_neurosuite_xml(standard_xml)
            exported = tmp_path / "standalone"
            exported.mkdir()
            copy2(saved, exported / saved.name)
            saved = exported / saved.name
            # An old, stale companion must not override embedded settings.
            saved.with_suffix(".probes.json").write_text("invalid old sidecar")
        imro.unlink()
        other = MainWindow()
        other._apply_xml_metadata(saved)
        assert [probe.n_channels for probe in other.probes] == [4, 384]
        assert other.n_channels.value() == 388
        assert other.probes[1].probe_type == NP2_FOUR
        assert other.probes[1].cmap == "coolwarm"
        assert other.color_mode.currentText() == "per probe"
        if not legacy_sidecar:
            assert other.color_map.currentText() == "red_white_black"
        assert other.probes[1].channel_map.positions == original_positions
        assert 5 not in other.probes[1].channel_map.active_channels
        assert other.bad_channels == {1, 12}
        assert {item.channel for item in other.viewer._layout_items} == set(range(388)) - {9}
        if not legacy_sidecar:
            assert len(read_probe_settings(saved)["probes"]) == 2
    finally:
        if other is not None:
            other.close()
        w.close()
