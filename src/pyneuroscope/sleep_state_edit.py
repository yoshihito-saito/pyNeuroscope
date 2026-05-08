from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat


def _in_intervals(timestamps: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    ints = np.asarray(intervals, dtype=np.float64).reshape(-1, 2) if np.asarray(intervals).size else np.empty((0, 2))
    if ints.size == 0:
        return np.zeros(ts.shape, dtype=bool)
    out = np.zeros(ts.shape, dtype=bool)
    for start, stop in ints:
        out |= (ts >= float(start)) & (ts <= float(stop))
    return out


def _intervals_from_mask(mask: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if m.size == 0 or ts.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    edges = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1)
    if m[0]:
        starts = np.r_[0, starts]
    if m[-1]:
        ends = np.r_[ends, m.size - 1]
    if starts.size == 0 or ends.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    n = min(starts.size, ends.size)
    starts = starts[:n]
    ends = ends[:n]
    return np.column_stack((ts[starts], ts[ends])).astype(np.float64)


def idx_to_intervals(states: np.ndarray, timestamps: np.ndarray, statenames: list[str]) -> dict[str, np.ndarray]:
    idx = np.asarray(states).reshape(-1)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    out: dict[str, np.ndarray] = {}
    if idx.size == 0:
        for name in statenames:
            if name:
                out[f"{name}state"] = np.empty((0, 2), dtype=np.float64)
        return out

    for state_id, name in enumerate(statenames, start=1):
        if not name:
            continue
        mask = idx == state_id
        out[f"{name}state"] = _intervals_from_mask(mask, ts)
    return out


def _merge_intervals(intervals: np.ndarray, max_gap: float, min_duration: float) -> np.ndarray:
    arr = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    merged: list[list[float]] = [[float(arr[0, 0]), float(arr[0, 1])]]
    for start, stop in arr[1:]:
        if float(start) - merged[-1][1] <= float(max_gap):
            merged[-1][1] = max(merged[-1][1], float(stop))
        else:
            merged.append([float(start), float(stop)])
    out = np.asarray(merged, dtype=np.float64)
    keep = (out[:, 1] - out[:, 0]) >= float(min_duration)
    return out[keep, :]


def _subtract_intervals(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    bb = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if aa.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if bb.size == 0:
        return aa

    out: list[list[float]] = []
    for start, stop in aa:
        cur = [(float(start), float(stop))]
        for bs, be in bb:
            next_cur: list[tuple[float, float]] = []
            for cs, ce in cur:
                if be <= cs or bs >= ce:
                    next_cur.append((cs, ce))
                    continue
                if bs > cs:
                    next_cur.append((cs, float(bs)))
                if be < ce:
                    next_cur.append((float(be), ce))
            cur = next_cur
        for cs, ce in cur:
            if ce > cs:
                out.append([cs, ce])
    if not out:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _align_metric_to_timestamps(metric: np.ndarray, metric_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    vals = np.asarray(metric, dtype=np.float64).reshape(-1)
    src_t = np.asarray(metric_t, dtype=np.float64).reshape(-1)
    dst_t = np.asarray(target_t, dtype=np.float64).reshape(-1)
    out = np.full(dst_t.shape, np.nan, dtype=np.float64)
    if vals.size == 0 or src_t.size == 0 or dst_t.size == 0:
        return out
    if vals.size != src_t.size:
        if vals.size == dst_t.size:
            return vals.astype(np.float64, copy=False)
        return out
    if vals.size == dst_t.size and np.array_equal(src_t, dst_t):
        return vals.astype(np.float64, copy=False)

    idx = np.searchsorted(src_t, dst_t, side="left")
    idx = np.clip(idx, 0, src_t.size - 1)
    left = np.clip(idx - 1, 0, src_t.size - 1)
    choose_left = np.abs(src_t[left] - dst_t) < np.abs(src_t[idx] - dst_t)
    nearest_idx = np.where(choose_left, left, idx)
    nearest_dist = np.abs(src_t[nearest_idx] - dst_t)

    diffs = np.diff(src_t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    tol = float(np.median(diffs)) * 0.51 if diffs.size else 1e-9
    valid = nearest_dist <= tol
    out[valid] = vals[nearest_idx[valid]]
    return out


def append_theta_epochs(sleep_state: dict[str, Any], basepath: Path, basename: str) -> tuple[dict[str, Any], Path]:
    metrics = sleep_state["detectorinfo"]["detectionparms"]["SleepScoreMetrics"]
    hists = metrics["histsandthreshs"]

    thratio = np.asarray(metrics["thratio"], dtype=np.float64).reshape(-1)
    emg = np.asarray(metrics["EMG"], dtype=np.float64).reshape(-1)
    ththr = float(hists["THthresh"])
    emgthr = float(hists["EMGthresh"])

    states = np.asarray(sleep_state["idx"]["states"]).reshape(-1)
    timestamps = np.asarray(sleep_state["idx"]["timestamps"]).reshape(-1).astype(np.float64)
    metric_t = np.asarray(metrics.get("t_clus", timestamps), dtype=np.float64).reshape(-1)
    if thratio.size != states.size or emg.size != states.size:
        thratio = _align_metric_to_timestamps(thratio, metric_t, timestamps)
        emg = _align_metric_to_timestamps(emg, metric_t, timestamps)
    theta_ndx = np.isfinite(thratio) & np.isfinite(emg) & (thratio > ththr) & (emg > emgthr)

    theta_states = np.zeros(states.shape, dtype=np.uint8)
    theta_states[theta_ndx] = 7
    non_theta = (states == 1) & (theta_states == 0)
    theta_states[non_theta] = 9

    statenames = np.asarray(["", "", "", "", "", "", "THETA", "", "nonTHETA"], dtype=object)
    theta_idx = {
        "states": theta_states.reshape(-1, 1),
        "timestamps": np.asarray(sleep_state["idx"]["timestamps"]).reshape(-1, 1),
        "statenames": statenames.reshape(1, -1),
    }

    theta_ints = idx_to_intervals(theta_states, timestamps, list(statenames))
    sleep_state.setdefault("idx", {})["theta_epochs"] = theta_idx
    sleep_state.setdefault("ints", {})["THETA"] = np.asarray(
        theta_ints.get("THETAstate", np.empty((0, 2))), dtype=np.float64
    ).reshape(-1, 2)
    sleep_state["ints"]["nonTHETA"] = np.asarray(
        theta_ints.get("nonTHETAstate", np.empty((0, 2))), dtype=np.float64
    ).reshape(-1, 2)

    out_path = basepath / f"{basename}.SleepState.states.mat"
    savemat(out_path, {"SleepState": sleep_state}, do_compression=True)
    return sleep_state, out_path


def states_to_episodes(sleep_state: dict[str, Any], basepath: Path, basename: str) -> tuple[dict[str, Any], Path]:
    ints = sleep_state.get("ints", {})
    nrem = np.asarray(ints.get("NREMstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    wake = np.asarray(ints.get("WAKEstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    rem = np.asarray(ints.get("REMstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)

    min_packet = 30.0
    min_w_episode = 20.0
    min_n_episode = 20.0
    min_r_episode = 20.0
    max_micro = 100.0
    max_w_interrupt = 40.0
    max_n_interrupt = max_micro
    max_r_interrupt = 40.0

    packet = nrem[(nrem[:, 1] - nrem[:, 0]) >= min_packet] if nrem.size else np.empty((0, 2))
    wake_len = (wake[:, 1] - wake[:, 0]) if wake.size else np.asarray([])
    ma = wake[wake_len <= max_micro] if wake.size else np.empty((0, 2))
    wake_intervals = wake[wake_len > max_micro] if wake.size else np.empty((0, 2))

    wake_episode = _merge_intervals(wake_intervals, max_w_interrupt, min_w_episode)
    nrem_episode = _merge_intervals(nrem, max_n_interrupt, min_n_episode)
    rem_episode = _merge_intervals(rem, max_r_interrupt, min_r_episode)

    if nrem_episode.size and rem.size:
        kept: list[list[float]] = []
        for ns, ne in nrem_episode:
            inside = rem[(rem[:, 0] >= ns) & (rem[:, 0] <= ne)]
            if inside.size == 0:
                kept.append([float(ns), float(ne)])
                continue
            cur_start = float(ns)
            for rs, re in inside:
                if rs > cur_start:
                    kept.append([cur_start, float(rs)])
                cur_start = float(re)
            if ne > cur_start:
                kept.append([cur_start, float(ne)])
        nrem_episode = np.asarray(kept, dtype=np.float64) if kept else np.empty((0, 2), dtype=np.float64)
        if nrem_episode.size:
            keep = (nrem_episode[:, 1] - nrem_episode[:, 0]) >= min_n_episode
            nrem_episode = nrem_episode[keep, :]

    ma_rem = np.empty((0, 2), dtype=np.float64)
    if ma.size and rem.size:
        flags = _in_intervals(ma[:, 0], rem) | _in_intervals(ma[:, 1], rem)
        ma_rem = ma[flags, :]
        ma = ma[~flags, :]

    no_overlap = [wake_episode.copy(), nrem_episode.copy(), rem_episode.copy()]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            no_overlap[i] = _subtract_intervals(no_overlap[i], no_overlap[j])
    wake_episode, nrem_episode, rem_episode = no_overlap

    det = {
        "originaldetectorinfo": sleep_state["detectorinfo"],
        "detectionparms": {
            "EpisodeDetectionParms": {
                "minPacketDuration": min_packet,
                "minWAKEEpisodeDuration": min_w_episode,
                "minNREMEpisodeDuration": min_n_episode,
                "minREMEpisodeDuration": min_r_episode,
                "maxMicroarousalDuration": max_micro,
                "maxWAKEEpisodeInterruption": max_w_interrupt,
                "maxNREMEpisodeInterruption": max_n_interrupt,
                "maxREMEpisodeInterruption": max_r_interrupt,
            }
        },
        "detectiondate": datetime.now().strftime("%Y-%m-%d"),
    }
    episodes = {
        "ints": {
            "NREMepisode": np.asarray(nrem_episode, dtype=np.float64).reshape(-1, 2),
            "REMepisode": np.asarray(rem_episode, dtype=np.float64).reshape(-1, 2),
            "WAKEepisode": np.asarray(wake_episode, dtype=np.float64).reshape(-1, 2),
            "NREMpacket": np.asarray(packet, dtype=np.float64).reshape(-1, 2),
            "MA": np.asarray(ma, dtype=np.float64).reshape(-1, 2),
            "MA_REM": np.asarray(ma_rem, dtype=np.float64).reshape(-1, 2),
        },
        "detectorinfo": det,
    }
    out_path = basepath / f"{basename}.SleepStateEpisodes.states.mat"
    savemat(out_path, {"SleepStateEpisodes": episodes}, do_compression=True)
    return episodes, out_path
