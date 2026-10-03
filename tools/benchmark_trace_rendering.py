"""Compare offscreen synthetic waveform rendering with a Git baseline.

Run from the repository root: python tools/benchmark_trace_rendering.py --baseline main
This excludes disk I/O and signal filtering. Timings are descriptive, not tests.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from pyneuroscope.main_window import MainWindow, ProbeConfig
from pyneuroscope.probe_geometry import load_probe_geometry, selected_site_map
from pyneuroscope.signal_layout import TraceLayoutItem
from pyneuroscope.signal_viewer import SignalViewer


def measure(viewer_class, data, time_seconds, app):
    viewer = viewer_class()
    viewer.resize(1000, 700)
    layout = [TraceLayoutItem(ch, 0, ch, 0, "#60aaff", False) for ch in range(data.shape[1])]
    viewer.set_traces(time_seconds, data, layout)
    # grab() forces QWidget rendering and does not need a visible desktop window.
    start = time.perf_counter()
    viewer.grab()
    first = (time.perf_counter() - start) * 1000
    repeats = []
    for _ in range(5):
        start = time.perf_counter()
        viewer.grab()
        repeats.append((time.perf_counter() - start) * 1000)
    viewer.close()
    viewer.deleteLater()
    app.processEvents()
    return dict(first_render_ms=round(first, 2), cached_render_median_ms=round(float(np.median(repeats)), 2))


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--baseline", default="main")
    options = args.parse_args()
    app = QApplication.instance() or QApplication([])
    # The Windows offscreen plugin may have no default system font database.
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"
    if font_path.exists():
        QFontDatabase.addApplicationFont(str(font_path))
        app.setFont(QFont("Arial", 9))
    source = subprocess.check_output(["git", "show", f"{options.baseline}:src/pyneuroscope/signal_viewer.py"], text=True)
    module = types.ModuleType("pyneuroscope._baseline_signal_viewer")
    module.__package__ = "pyneuroscope"
    exec(compile(source, "<baseline signal_viewer>", "exec"), module.__dict__)
    rng = np.random.default_rng(42)
    results = []
    for channels in [384, 768]:
        data = rng.integers(-1000, 1000, (30000, channels), dtype=np.int16)
        time_seconds = np.arange(len(data)) / 30000.0
        results.append(dict(channels=channels, seconds=1, baseline=measure(module.SignalViewer,data,time_seconds,app),
                            updated=measure(SignalViewer,data,time_seconds,app)))
    window = MainWindow()
    window.probes = [ProbeConfig(384, probe_type="Neuropixels 1.0", cmap="Blues")]
    window._refresh_probe_controls()
    window._apply_probe_configs_to_model()
    window.sampling_rate.setValue(30000)
    geometry = load_probe_geometry("Neuropixels 1.0")
    window._set_probe_channel_map(0, selected_site_map(" ".join(map(str,range(384,768))),geometry,384))
    window.color_mode.setCurrentText("per probe")
    window._current_time = np.arange(30000) / 30000.0
    window._current_data = rng.integers(-1000,1000,(30000,384),dtype=np.int16)
    window.scale.setValue(1.0)
    window._select_probe_channels(set(range(16)))
    window.resize(1440, 950)
    image = Path(tempfile.gettempdir()) / "pyneuroscope-probe-updates.png"
    window.grab().save(str(image))
    window.close()
    print(json.dumps(dict(benchmark=results, synthetic_ui_preview=str(image)), indent=2))


if __name__ == "__main__":
    main()
