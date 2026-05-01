from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import DatInfo, DatWindow


class DatReaderError(ValueError):
    """Raised when a DAT file cannot be interpreted as requested."""


def dtype_from_name(dtype: str) -> np.dtype:
    try:
        result = np.dtype(dtype)
    except TypeError as exc:
        raise DatReaderError(f"Unsupported dtype: {dtype}") from exc
    if result.itemsize <= 0:
        raise DatReaderError(f"Unsupported dtype: {dtype}")
    return result


def inspect_dat(
    path: str | Path,
    n_channels: int,
    sampling_rate: float,
    dtype: str = "int16",
    *,
    allow_trailing_bytes: bool = False,
) -> DatInfo:
    dat_path = Path(path)
    if not dat_path.exists():
        raise DatReaderError(f"DAT file does not exist: {dat_path}")
    if n_channels <= 0:
        raise DatReaderError("n_channels must be positive")
    if sampling_rate <= 0:
        raise DatReaderError("sampling_rate must be positive")

    np_dtype = dtype_from_name(dtype)
    file_size = int(dat_path.stat().st_size)
    frame_bytes = int(n_channels) * int(np_dtype.itemsize)
    if frame_bytes <= 0:
        raise DatReaderError("Frame size must be positive")
    trailing_bytes = file_size % frame_bytes
    if trailing_bytes != 0 and not allow_trailing_bytes:
        raise DatReaderError(
            f"File size {file_size} is not divisible by frame size {frame_bytes}"
        )

    total_frames = file_size // frame_bytes
    duration_seconds = total_frames / float(sampling_rate)
    return DatInfo(
        path=dat_path,
        n_channels=n_channels,
        sampling_rate=float(sampling_rate),
        dtype=str(np_dtype),
        file_size_bytes=file_size,
        total_frames=total_frames,
        duration_seconds=duration_seconds,
        frame_bytes=frame_bytes,
    )


def compute_window_bounds(
    total_frames: int,
    sampling_rate: float,
    start_seconds: float,
    duration_seconds: float,
    *,
    clip: bool = True,
) -> tuple[int, int, bool]:
    if start_seconds < 0:
        raise DatReaderError("start_seconds must be non-negative")
    if duration_seconds <= 0:
        raise DatReaderError("duration_seconds must be positive")
    if sampling_rate <= 0:
        raise DatReaderError("sampling_rate must be positive")

    start_frame = int(round(float(start_seconds) * float(sampling_rate)))
    requested_frames = int(round(float(duration_seconds) * float(sampling_rate)))
    requested_frames = max(1, requested_frames)

    if start_frame >= int(total_frames):
        raise DatReaderError("start_seconds is outside the recording")

    end_frame = start_frame + requested_frames
    clipped = False
    if end_frame > int(total_frames):
        if not clip:
            raise DatReaderError("Requested window exceeds recording duration")
        end_frame = int(total_frames)
        clipped = True

    return start_frame, max(0, end_frame - start_frame), clipped


def byte_offset_for_frame(start_frame: int, frame_bytes: int) -> int:
    if start_frame < 0 or frame_bytes <= 0:
        raise DatReaderError("Invalid byte offset inputs")
    return int(start_frame) * int(frame_bytes)


def read_dat_window(
    path: str | Path,
    n_channels: int,
    sampling_rate: float,
    start_seconds: float = 0.0,
    duration_seconds: float = 1.0,
    dtype: str = "int16",
    *,
    clip: bool = True,
    allow_trailing_bytes: bool = False,
) -> DatWindow:
    info = inspect_dat(path, n_channels, sampling_rate, dtype, allow_trailing_bytes=allow_trailing_bytes)
    np_dtype = dtype_from_name(dtype)
    start_frame, frames_to_read, clipped = compute_window_bounds(
        info.total_frames,
        info.sampling_rate,
        start_seconds,
        duration_seconds,
        clip=clip,
    )
    offset = byte_offset_for_frame(start_frame, info.frame_bytes)
    count = int(frames_to_read) * int(n_channels)

    with info.path.open("rb") as handle:
        handle.seek(offset)
        raw = np.fromfile(handle, dtype=np_dtype, count=count)

    if raw.size != count:
        clipped = True
        usable = raw.size - (raw.size % int(n_channels))
        raw = raw[:usable]
    data = raw.reshape((-1, int(n_channels)))
    frames_read = int(data.shape[0])
    time_seconds = (np.arange(frames_read, dtype=np.float64) + start_frame) / info.sampling_rate
    return DatWindow(time_seconds, data, start_frame, frames_read, clipped=clipped)
