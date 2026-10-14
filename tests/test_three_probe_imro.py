import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QPushButton

from pyneuroscope.color_map import palette_from_name
from pyneuroscope.main_window import MainWindow
from pyneuroscope.probe_geometry import load_recording_channel_map
from pyneuroscope.probe_settings import read_probe_settings


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def write_imro(path, shank):
    path.write_text("(NP2013,384)" + "".join(f"({ch} {shank} 1 0 {ch + 384})" for ch in range(384)))


def test_three_probes_imro_only_keep_all_dat_columns_and_scroll_to_last(app, tmp_path, monkeypatch):
    from pyneuroscope import main_window as module
    w = MainWindow()
    assert w.scale.value() == 1.0
    w.spacing.setValue(1.0)
    other = None
    try:
        paths = [tmp_path / f"probe{i + 1}.imro" for i in range(3)]
        for i, path in enumerate(paths):
            write_imro(path, i)
        for i, path in enumerate(paths):
            if i:
                w._add_probe()
            monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a, p=path: (str(p), ""))
            w._load_probe_file(i)
            assert w.probes[i].xml_path is None
            buttons = {b.text() for b in w.probe_rows_layout.itemAt(i).widget().findChildren(QPushButton)}
            assert "Probe XML or IMRO" in buttons and "Load IMRO" not in buttons
        assert w.n_channels.value() == 1152
        assert [p.n_channels for p in w.probes] == [384, 384, 384]
        assert {ch for group in w.groups for ch in group.channels} == set(range(1152))
        assert len(w._probe_scene()[1]) == 3 * 5120
        assert w.probes[2].channel_map.site_ids[0] == "s2_front_0384"
        for probe, cmap in zip(w.probes, ["red_white_black", "coolwarm", "blue_white_black"]):
            probe.cmap = cmap
        w.color_mode.setCurrentText("per probe")
        # Reload while count stays 384: geometry sorting must also refresh colors.
        w._set_probe_channel_map(2, load_recording_channel_map(paths[2]))
        for i, probe in enumerate(w.probes):
            channels = [ch for g in w.groups for ch in g.channels if i * 384 <= ch < (i + 1) * 384]
            assert [w.channel_colors[ch] for ch in channels] == palette_from_name(probe.cmap, 384)

        frames = 128
        data = np.rint(100 * np.sin(np.arange(frames)[:, None] * .18 + np.arange(1152)[None, :] * .02)).astype(np.int16)
        dat = tmp_path / "neuropixels_ap.dat"
        data.tofile(dat)
        w.dat_path.setText(str(dat))
        w.sampling_rate.setValue(20000)
        w._load_window(silent=True)
        assert w._current_data.shape == (frames, 1152)
        assert np.array_equal(w._current_data[:, 1100], data[:, 1100])
        w.resize(1280, 800)
        w.show()
        app.processEvents()
        assert {i.channel for i in w.viewer._layout_items} == set(range(1152))
        assert w.viewer.minimumHeight() >= 48 + 1152 * 8
        assert w.signal_scroll.verticalScrollBar().maximum() > 0
        w.signal_scroll.verticalScrollBar().setValue(w.signal_scroll.verticalScrollBar().maximum())
        app.processEvents()
        w.viewer.grab()
        assert set(w.viewer._path_cache) == set(range(1152))
        # The last probe's rows are on screen after scrolling to the bottom.
        last_channel = w.viewer._layout_items[-1].channel
        assert last_channel >= 768
        y = w.viewer._trace_image_cache[2][last_channel][2]
        assert 0 <= y - w.signal_scroll.verticalScrollBar().value() < w.signal_scroll.viewport().height()

        w.display_mode.setCurrentText("bitmap")
        app.processEvents()
        w.viewer.grab()
        assert w.viewer.minimumHeight() >= 48 + 1152
        assert w.viewer._bitmap_cache[1][0].height() == 1152
        w.display_mode.setCurrentText("trace")
        app.processEvents()
        assert w.scale.value() == 1.0
        assert w.viewer.minimumHeight() >= 48 + 1152 * 8

        saved = tmp_path / "session.xml"
        monkeypatch.setattr(module.QFileDialog, "getSaveFileName", lambda *a: (str(saved), ""))
        monkeypatch.setattr(module.QMessageBox, "information", lambda *a: None)
        w._save_xml()
        assert len(read_probe_settings(saved)["probes"]) == 3
        assert not saved.with_suffix(".probes.json").exists()
        for path in paths:
            path.unlink()
        other = MainWindow()
        other._apply_xml_metadata(saved)
        assert len(other.probes) == 3 and other.n_channels.value() == 1152
        assert {i.channel for i in other.viewer._layout_items} == set(range(1152))
    finally:
        if other is not None:
            other.close()
        w.close()
