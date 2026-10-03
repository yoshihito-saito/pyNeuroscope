import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from scipy.io import savemat

from pyneuroscope.color_map import palette_from_name
from pyneuroscope.main_window import MainWindow, ProbeConfig
from pyneuroscope.models import ChannelGroup
from pyneuroscope.probe_geometry import (
    ProbeGeometryError, ProbeSitePosition, load_probe_geometry,
    load_recording_channel_map, selected_site_map,
)
from pyneuroscope.probe_viewer import ProbeViewer
from pyneuroscope.signal_layout import TraceLayoutItem
from pyneuroscope.signal_viewer import SignalViewer
from pyneuroscope.trace_display import peak_envelope_indices
from pyneuroscope.xml_builder import XmlError, parse_neurosuite_xml


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def template_xml(count=4):
    return ('<parameters><acquisitionSystem><nChannels>' + str(count) + '</nChannels></acquisitionSystem>'
            '<anatomicalDescription><channelGroups><group>' +
            ''.join(f'<channel>{ch}</channel>' for ch in reversed(range(count))) +
            '</group></channelGroups></anatomicalDescription></parameters>')


@pytest.mark.parametrize("name,count", [("Neuropixels 1.0",960),
    ("Neuropixels 2.0 single shank",1280), ("Neuropixels 2.0 four shank",5120)])
def test_catalog_has_all_sites_and_no_invented_dat_channels(name, count):
    geometry = load_probe_geometry(name)
    assert len(geometry.physical_sites) == count
    assert geometry.positions_for_groups([ChannelGroup("g", list(range(384)))]) == {}


def test_site_ids_follow_dat_order_instead_of_group_order():
    g = load_probe_geometry("Neuropixels 1.0")
    m = selected_site_map("384, 0, 959, 17", g, 4)
    assert m.positions[0] == g.physical_sites["s0_front_0384"]
    assert m.positions[1] == g.physical_sites["s0_front_0000"]
    assert m.site_ids[2] == "s0_front_0959"
    assert m.active_channels == {0,1,2,3}


def test_four_shank_ids_are_explicit_and_checked():
    g = load_probe_geometry("Neuropixels 2.0 four shank")
    m = selected_site_map("0:1279 3:0", g, 2)
    assert m.positions[1].x - g.physical_sites["s0_front_0000"].x == 750
    for text in ["0 1", "0:1 0:1", "4:0 0:1", "0:0"]:
        with pytest.raises(ProbeGeometryError):
            selected_site_map(text, g, 2)


def test_atlaxis_mat_preserves_disconnected_rows_and_active_flags(tmp_path):
    path = tmp_path / "probe.chanCoords.channelInfo.mat"
    savemat(path, {"chanCoords": {"channel": [1,2,3,4], "x": [-24,np.nan,8,-8],
        "y": [200,np.nan,200,220], "connected": [True,False,True,True],
        "active": [True,False,False,True], "site_id": ["s0_front_0000","","s0_front_0001","s0_front_0002"]}})
    m = load_recording_channel_map(path)
    assert set(m.positions) == {0,2,3}
    assert m.active_channels == {0,3}
    assert m.positions[3].y == 220


@pytest.mark.parametrize("channels", [[0,0], [0,.5], [-1,0], [0,np.nan]])
def test_recording_map_rejects_invalid_channel_ids(tmp_path, channels):
    path = tmp_path / "chanMap.mat"
    savemat(path, {"chanMap0ind": channels, "xcoords": [0,20], "ycoords": [0,20]})
    with pytest.raises(ProbeGeometryError):
        load_recording_channel_map(path)


def test_template_rates_are_missing_and_invalid_rates_still_fail():
    metadata, groups, _ = parse_neurosuite_xml(template_xml())
    assert metadata.sampling_rate == metadata.lfp_sampling_rate == 0
    assert groups[0].channels == [3,2,1,0]
    for value in ["0", "-1", "nan", "oops", "inf"]:
        xml = template_xml().replace('</nChannels>', f'</nChannels><samplingRate>{value}</samplingRate>')
        with pytest.raises(XmlError):
            parse_neurosuite_xml(xml)


def test_xml_without_rates_keeps_gui_and_updates_probe_file(app, tmp_path, monkeypatch):
    from pyneuroscope import main_window as module
    source, output = tmp_path / "probe.xml", tmp_path / "updated.xml"
    source.write_text(template_xml())
    w = MainWindow()
    w.sampling_rate.setValue(30000)
    w.lfp_sampling_rate.setValue(2500)
    monkeypatch.setattr(module.QFileDialog, 'getOpenFileName', lambda *a: (str(source), ''))
    w._load_probe_xml(0)
    assert w.sampling_rate.value() == 30000
    assert w.lfp_sampling_rate.value() == 2500
    assert 'XML has no' in w.xml_settings_note.text()
    monkeypatch.setattr(module.QFileDialog, 'getSaveFileName', lambda *a: (str(output), ''))
    w._save_probe_xml(0)
    metadata, groups, _ = parse_neurosuite_xml(output)
    assert metadata.sampling_rate == 30000
    assert metadata.lfp_sampling_rate == 2500
    assert metadata.amplification is None
    assert groups[0].channels == [3,2,1,0]
    # A settings sidecar with no map must not hide every recording channel.
    monkeypatch.setattr(module.QFileDialog, 'getOpenFileName', lambda *a: (str(output), ''))
    w._load_probe_xml(0)
    assert [ch for g in w._visible_groups() for ch in g.channels] == [3,2,1,0]
    w.close()


def test_session_xml_without_rates_keeps_gui(app, tmp_path):
    path = tmp_path / "probe.xml"
    path.write_text(template_xml())
    w = MainWindow()
    w.sampling_rate.setValue(30000)
    w._apply_xml_metadata(path)
    assert w.sampling_rate.value() == 30000
    assert w.groups[0].channels == [3,2,1,0]
    w.close()


def test_probe_selection_filters_activity_only_and_can_restore(app):
    w = MainWindow()
    before = list(w.groups)
    w._select_probe_channels({1,3})
    assert {i.channel for i in w.viewer._layout_items} == {1,3}
    assert w.groups == before
    assert not w.bad_channels
    w._select_probe_channels(set())
    assert not w.viewer._layout_items
    w._select_probe_channels(None)
    assert {i.channel for i in w.viewer._layout_items} == {0,1,2,3}
    w.close()


def test_probe_colormaps_are_independent_and_survive_controls_refresh(app):
    w = MainWindow()
    w.probes = [ProbeConfig(2, cmap="Blues"), ProbeConfig(2, cmap="Reds")]
    w._apply_probe_configs_to_model()
    w.color_mode.setCurrentText('per probe')
    assert [w.channel_colors[i] for i in [0,1]] == palette_from_name('Blues',2)
    assert [w.channel_colors[i] for i in [2,3]] == palette_from_name('Reds',2)
    w.probe_cmap_controls[1].setCurrentText('Greens')
    w._refresh_probe_controls()
    assert w.probe_cmap_controls[1].currentText() == 'Greens'
    assert w.probes[0].cmap == 'Blues'
    w.close()


def test_np_map_and_cmap_round_trip_session_sidecar(app, tmp_path, monkeypatch):
    from pyneuroscope import main_window as module
    w = MainWindow()
    w.probes = [ProbeConfig(2, probe_type='Neuropixels 1.0', cmap='Blues'),
                ProbeConfig(2, probe_type='Neuropixels 2.0 four shank', cmap='Reds')]
    w._apply_probe_configs_to_model()
    w._set_probe_channel_map(0, selected_site_map('384 959',load_probe_geometry(w.probes[0].probe_type),2))
    w._set_probe_channel_map(1, selected_site_map('3:0 1:512',load_probe_geometry(w.probes[1].probe_type),2))
    w.color_mode.setCurrentText('per probe')
    w._select_probe_channels({0})
    path = tmp_path / 'session.xml'
    monkeypatch.setattr(module.QFileDialog, 'getSaveFileName', lambda *a: (str(path), ''))
    monkeypatch.setattr(module.QMessageBox, 'information', lambda *a: None)
    w._save_xml()
    other = MainWindow()
    other._apply_xml_metadata(path)
    assert [p.probe_type for p in other.probes] == [p.probe_type for p in w.probes]
    assert [p.cmap for p in other.probes] == ['Blues','Reds']
    assert other.probes[1].channel_map.site_ids[0] == 's3_front_0000'
    assert other.selected_channels is None
    assert len(other.probe_viewer._physical_sites) == 6080
    assert {i.channel for i in other.viewer._layout_items} == {0,1,2,3}
    other.close()
    w.close()


def test_wheel_and_rectangle_select_dat_ids_after_zoom(app):
    v = ProbeViewer()
    v.resize(400,600)
    v.set_probe(4,[ChannelGroup('g',[3,1,2,0])],set(),{},channel_geometry={
        ch: ProbeSitePosition(0,ch*20) for ch in range(4)})
    v.show()
    app.processEvents()
    anchor = QPointF(*v._dot_hits[1][:2])
    old = v._zoom
    event = SimpleNamespace(angleDelta=lambda: SimpleNamespace(y=lambda:120), position=lambda:anchor, accept=lambda:None)
    v.wheelEvent(event)
    app.processEvents()
    assert v._zoom > old
    assert np.allclose(v._dot_hits[1][:2], [anchor.x(),anchor.y()])
    cx,cy,_ = v._dot_hits[1]
    selected = []
    v.channelsSelected.connect(selected.append)
    QTest.mousePress(v, Qt.MouseButton.LeftButton, pos=QPointF(cx-10,cy-10).toPoint())
    QTest.mouseMove(v, QPointF(cx+10,cy+10).toPoint())
    QTest.mouseRelease(v, Qt.MouseButton.LeftButton, pos=QPointF(cx+10,cy+10).toPoint())
    assert selected == [{1}]
    v.reset_view()
    assert v._zoom == 1
    v.close()


def test_minmax_display_preserves_single_sample_spikes_endpoints_and_nan_gap():
    values = np.zeros(1003)
    values[31],values[99],values[-1] = 123,-97,84
    values[400:440] = np.nan
    indices = peak_envelope_indices(values,100)
    assert {0,31,99,1002,399,400,439,440}.issubset(indices)
    assert len(indices) < 110
    assert np.all(np.diff(indices) > 0)


def test_render_cache_reuses_paths_and_invalidates_new_data(app):
    v = SignalViewer()
    v.resize(600,400)
    data = np.zeros((10000,2), dtype=np.int16)
    data[77,0] = 100
    time = np.arange(len(data))/30000
    layout = [TraceLayoutItem(ch,0,ch,0,'#ffffff',False) for ch in range(2)]
    v.set_traces(time,data,layout)
    v.show()
    app.processEvents()
    v.grab()
    path = v._path_cache[0][1]
    cached = v._trace_cache[0]
    assert np.max(cached[1]) == 100
    v.set_traces(time,data,layout)
    v.grab()
    assert v._path_cache[0][1] is path
    v.set_traces(time,data.copy(),layout)
    assert not v._path_cache
    v.grab()
    assert v._path_cache[0][1] is not path
    layer = v._trace_image_cache[1]
    v.set_background_mode('white')
    v.grab()
    assert v._trace_image_cache[1] is not layer
    cached = v._trace_cache[0]
    v.set_traces(time, v._data, layout, vertical_scale=2)
    v.grab()
    assert v._trace_cache[0] is cached
    v.close()


def test_np_rejects_unregistered_coordinates_instead_of_inventing_sites(app):
    from pyneuroscope.probe_geometry import RecordingChannelMap
    w = MainWindow()
    w.probes[0].probe_type = 'Neuropixels 1.0'
    with pytest.raises(ProbeGeometryError, match='DAT order'):
        w._set_probe_channel_map(0,RecordingChannelMap({0:ProbeSitePosition(0,0)},{0}))
    w.close()


def test_inactive_map_channels_are_distinct_from_bad_channels(app):
    from pyneuroscope.probe_geometry import RecordingChannelMap
    w = MainWindow()
    g = load_probe_geometry('Neuropixels 1.0')
    w.probes[0].probe_type = g.name
    m = selected_site_map('0 1 2 3',g,4)
    w._set_probe_channel_map(0,RecordingChannelMap(m.positions,{0,2,3},m.site_ids))
    w.bad_channels = {2}
    w._refresh_viewer_layout()
    assert {item.channel for item in w.viewer._layout_items} == {0,2,3}
    assert w.probe_viewer._active_channels == {0,2,3}
    w.ignore_bad_channels.setChecked(True)
    assert {item.channel for item in w.viewer._layout_items} == {0,3}
    assert w.probe_viewer._active_channels == {0,2,3}
    assert not w._spike_unit_is_visible(SimpleNamespace(channel=1))
    w.close()
