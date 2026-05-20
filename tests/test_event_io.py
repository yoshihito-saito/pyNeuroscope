from pathlib import Path

from pyneuroscope.event_io import find_event_files, find_spikes_file, load_event_file, load_spikes_cellinfo


def test_load_spikes_cellinfo_from_demo() -> None:
    spikes = load_spikes_cellinfo(Path("demo/sake_day10/sake_day10.spikes.cellinfo.mat"))

    assert spikes.basename == "sake_day10"
    assert len(spikes.units) == 49
    assert spikes.units[0].channel == 53
    assert spikes.units[0].times.size > 0


def test_load_event_file_from_demo() -> None:
    event = load_event_file(Path("demo/sake_day10/sake_day10.ripples.events.mat"))

    assert event.name == "ripples"
    assert event.timestamps.shape == (16602, 2)
    assert event.peaks is not None
    assert event.peaks.shape == (16602,)


def test_find_analysis_files_prefers_matching_basename() -> None:
    base_dir = Path("demo/sake_day10")

    assert find_spikes_file([base_dir], ["sake_day10"]).name == "sake_day10.spikes.cellinfo.mat"
    assert [path.name for path in find_event_files([base_dir], ["sake_day10"])] == ["sake_day10.ripples.events.mat"]


def test_find_event_files_ignores_mergepoints(tmp_path: Path) -> None:
    (tmp_path / "session.MergePoints.events.mat").write_bytes(b"placeholder")
    (tmp_path / "session.ripples.events.mat").write_bytes(b"placeholder")

    assert [path.name for path in find_event_files([tmp_path], ["session"])] == ["session.ripples.events.mat"]
