from pathlib import Path

import numpy as np
import pytest

from pyneuroscope.dat_reader import (
    DatReaderError,
    byte_offset_for_frame,
    compute_window_bounds,
    inspect_dat,
    read_dat_window,
)


WORK_DIR = Path("tests/_tmp_data")


def _work_path(name: str) -> Path:
    WORK_DIR.mkdir(exist_ok=True)
    return WORK_DIR / name


def test_reads_small_interleaved_window() -> None:
    path = _work_path("amplifier.dat")
    data = np.array(
        [
            [0, 1, 2, 3],
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=np.int16,
    )
    data.tofile(path)

    window = read_dat_window(path, n_channels=4, sampling_rate=2, start_seconds=0.5, duration_seconds=1)

    assert window.start_frame == 1
    assert window.frames_read == 2
    assert window.data.tolist() == [[10, 11, 12, 13], [20, 21, 22, 23]]
    assert window.time_seconds.tolist() == [0.5, 1.0]


def test_rejects_incompatible_file_size() -> None:
    path = _work_path("bad.dat")
    path.write_bytes(b"123")

    with pytest.raises(DatReaderError):
        inspect_dat(path, n_channels=2, sampling_rate=20000)


def test_can_ignore_trailing_bytes_for_live_recording() -> None:
    path = _work_path("live.dat")
    path.write_bytes(np.arange(8, dtype=np.int16).tobytes() + b"x")

    info = inspect_dat(path, n_channels=2, sampling_rate=2, allow_trailing_bytes=True)

    assert info.total_frames == 4
    assert info.duration_seconds == 2.0


def test_clips_window_at_end() -> None:
    path = _work_path("short.dat")
    np.arange(12, dtype=np.int16).tofile(path)

    window = read_dat_window(path, n_channels=2, sampling_rate=2, start_seconds=2, duration_seconds=2)

    assert window.clipped is True
    assert window.frames_read == 2


def test_large_file_offsets_are_python_ints() -> None:
    frames_30_hours = 30 * 60 * 60 * 20000
    start_frame, frames, clipped = compute_window_bounds(
        frames_30_hours,
        20000,
        start_seconds=(30 * 60 * 60) - 1,
        duration_seconds=1,
    )

    assert clipped is False
    assert frames == 20000
    assert start_frame > 2**31
    assert byte_offset_for_frame(start_frame, 128 * 2) > 2**39


def test_inspects_demo_recording_when_present() -> None:
    path = Path("demo/test_recording_251224_122225/amplifier.dat")
    if not path.exists():
        pytest.skip("demo recording is not present")

    info = inspect_dat(path, n_channels=128, sampling_rate=20000)

    assert info.file_size_bytes == 8474066944
    assert info.total_frames == 33101824
    assert info.duration_seconds == pytest.approx(1655.0912)


def test_reads_late_demo_window_when_present() -> None:
    path = Path("demo/test_recording_251224_122225/amplifier.dat")
    if not path.exists():
        pytest.skip("demo recording is not present")

    window = read_dat_window(path, 128, 20000, start_seconds=1654, duration_seconds=1)

    assert window.data.shape == (20000, 128)
    assert window.start_frame == 33080000
