# Vendored from h2r_collection (github.com/ravikola02/h2r_collection, MIT) scripts/lerobot_v30.py at 0783627.
# egoumi change: observation.state / action are optional (None -> video-only episode, length from n_frames).
"""LeRobot v3.0 dataset writer (``--dataset-version v30``).

Torch-free: importable outside the hawor env, which keeps the layout
testable without running HaWoR. The format follows lerobot 0.4.x
(``CODEBASE_VERSION = "v3.0"``):

* ``data/chunk-XXX/file-XXX.parquet`` -- many episodes per parquet, files
  rotate when they would exceed ``data_files_size_in_mb`` (default 100).
* ``videos/{video_key}/chunk-XXX/file-XXX.mp4`` -- episode MP4s
  concatenated back-to-back (stream copy), rotating at
  ``video_files_size_in_mb`` (default 500). Episode boundaries are stored
  as ``from_timestamp``/``to_timestamp`` in the episodes metadata.
* ``meta/episodes/chunk-000/file-000.parquet`` -- one row per episode:
  file locations, global frame range, video timestamp ranges, and
  flattened per-episode stats (``stats/{feature}/{min,max,mean,std,count,
  q01,q10,q50,q90,q99}``).
* ``meta/tasks.parquet`` -- task strings as the index, ``task_index`` col.
* ``meta/info.json`` (v3.0 fields) + ``meta/stats.json`` (aggregated).

Frame rows keep the same columns as v2.1 (``timestamp`` stays
episode-relative; video seeks add ``from_timestamp``), which is exactly
what lerobot's own v2.1 -> v3.0 converter produces.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from video_io import (concat_videos, encode_video, sample_video_frames,
                      video_duration_s)

CODEBASE_VERSION_V30 = "v3.0"
V30_CHUNKS_SIZE = 1000
V30_DATA_FILE_SIZE_MB = 100
V30_VIDEO_FILE_SIZE_MB = 500

V30_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V30_VIDEO_PATH = ("videos/{video_key}/chunk-{chunk_index:03d}/"
                  "file-{file_index:03d}.mp4")
V30_EPISODES_PATH = ("meta/episodes/chunk-{chunk_index:03d}/"
                     "file-{file_index:03d}.parquet")
V30_TASKS_PATH = "meta/tasks.parquet"

# Matches lerobot's DEFAULT_QUANTILES (stat key -> quantile).
STATS_QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50,
                   "q90": 0.90, "q99": 0.99}

IMAGE_SAMPLES_PER_EPISODE = 8


# ---------------------------------------------------------------------------
# frame table (shared with the v2.1 writer in convert_to_lerobot.py)
# ---------------------------------------------------------------------------

def episode_frame_table(state: Optional[np.ndarray], action: Optional[np.ndarray],
                        fps: float, episode_index: int, task_index: int,
                        global_index_offset: int,
                        n_frames: Optional[int] = None) -> pa.Table:
    """Per-frame rows for one episode, LeRobot column layout.

    state/action stored as fixed-size list<float32>[D]; ``timestamp`` is
    episode-relative (frame_index / fps) in both v2.1 and v3.0. With
    state/action None the table has only the index columns (video-only).
    """
    n = int(state.shape[0]) if state is not None else int(n_frames)
    frame_index = np.arange(n, dtype=np.int64)
    timestamp = (frame_index / fps).astype(np.float32)
    cols = {}
    for key, arr in (("observation.state", state), ("action", action)):
        if arr is not None:
            arr = arr.astype(np.float32)
            cols[key] = pa.array([row.tolist() for row in arr],
                                 type=pa.list_(pa.float32(), arr.shape[1]))

    return pa.table({
        **cols,
        "timestamp":     timestamp,
        "frame_index":   frame_index,
        "episode_index": np.full(n, episode_index, dtype=np.int64),
        "index":         (global_index_offset + frame_index).astype(np.int64),
        "task_index":    np.full(n, task_index, dtype=np.int64),
    })


# ---------------------------------------------------------------------------
# stats -- lerobot-compatible shapes: vectors (D,), scalar columns (1,),
# images (3,1,1) in [0,1]; every feature carries q01..q99 quantiles
# ---------------------------------------------------------------------------

def _feature_stats(arr2d: np.ndarray, count: int) -> Dict[str, np.ndarray]:
    """min/max/mean/std/count + quantiles over axis 0 of an (N, D) array."""
    stats = {
        "min":   arr2d.min(axis=0),
        "max":   arr2d.max(axis=0),
        "mean":  arr2d.mean(axis=0),
        "std":   arr2d.std(axis=0),
        "count": np.array([count], dtype=np.int64),
    }
    for key, q in STATS_QUANTILES.items():
        stats[key] = np.quantile(arr2d, q, axis=0)
    return stats


def _image_stats(samples: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-channel stats from (S, H, W, 3) uint8 frames, shaped (3,1,1) and
    normalized to [0,1]. ``count`` is the number of sampled frames (matches
    lerobot's compute_episode_stats)."""
    arr = samples.reshape(-1, 3).astype(np.float64) / 255.0
    stats = _feature_stats(arr, count=int(samples.shape[0]))
    return {k: (v if k == "count" else v.reshape(3, 1, 1))
            for k, v in stats.items()}


def compute_episode_stats(state: Optional[np.ndarray], action: Optional[np.ndarray],
                          fps: float, episode_index: int, task_index: int,
                          global_index_offset: int,
                          video_samples: Dict[str, np.ndarray],
                          n_frames: Optional[int] = None) -> Dict:
    """Stats for every feature of one episode (numeric exact, images from
    sampled frames)."""
    n = int(state.shape[0]) if state is not None else int(n_frames)
    frame_index = np.arange(n, dtype=np.float64)
    cols = {}
    if state is not None:
        cols["observation.state"] = state.astype(np.float64)
    if action is not None:
        cols["action"] = action.astype(np.float64)
    cols.update({
        "timestamp":     (frame_index / fps).reshape(-1, 1),
        "frame_index":   frame_index.reshape(-1, 1),
        "episode_index": np.full((n, 1), episode_index, dtype=np.float64),
        "index":         (global_index_offset + frame_index).reshape(-1, 1),
        "task_index":    np.full((n, 1), task_index, dtype=np.float64),
    })
    stats = {key: _feature_stats(arr, n) for key, arr in cols.items()}
    for video_key, samples in video_samples.items():
        stats[video_key] = _image_stats(samples)
    return stats


def aggregate_stats(stats_list: List[Dict]) -> Dict:
    """Aggregate per-episode stats into dataset-wide stats (lerobot's
    parallel-variance scheme; quantiles are count-weighted means)."""
    keys: List[str] = []
    for stats in stats_list:
        keys += [k for k in stats if k not in keys]

    out: Dict[str, Dict] = {}
    for key in keys:
        per = [s[key] for s in stats_list if key in s]
        counts = np.stack([p["count"] for p in per]).astype(np.float64)
        c = counts
        means = np.stack([p["mean"] for p in per])
        while c.ndim < means.ndim:
            c = np.expand_dims(c, axis=-1)
        total = c.sum(axis=0)

        mean = (means * c).sum(axis=0) / total
        variances = np.stack([p["std"] for p in per]) ** 2
        delta = means - mean
        std = np.sqrt(((variances + delta ** 2) * c).sum(axis=0) / total)

        agg = {
            "min":   np.min(np.stack([p["min"] for p in per]), axis=0),
            "max":   np.max(np.stack([p["max"] for p in per]), axis=0),
            "mean":  mean,
            "std":   std,
            "count": counts.sum(axis=0).astype(np.int64),
        }
        for qk in STATS_QUANTILES:
            qv = np.stack([p[qk] for p in per])
            agg[qk] = (qv * c).sum(axis=0) / total
        out[key] = {k: v.tolist() for k, v in agg.items()}
    return out


def _flatten_stats(stats: Dict) -> Dict[str, List]:
    """{'action': {'mean': arr, ...}} -> {'stats/action/mean': list, ...}"""
    flat = {}
    for feature, feature_stats in stats.items():
        for stat_name, value in feature_stats.items():
            flat[f"stats/{feature}/{stat_name}"] = np.asarray(value).tolist()
    return flat


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------

def _bump(chunk_idx: int, file_idx: int) -> Tuple[int, int]:
    if file_idx == V30_CHUNKS_SIZE - 1:
        return chunk_idx + 1, 0
    return chunk_idx, file_idx + 1


class V30DatasetWriter:
    """Accumulate episodes into the LeRobot v3.0 file layout.

    Episodes are added one at a time (``add_episode``); frame rows are
    buffered in RAM per data file (trajectories are tiny -- 14 floats/frame)
    and flushed when the file's size budget is reached, while episode MP4s
    are encoded to a temp dir and concatenated into the packed video file
    on rotation or at ``finalize``. Episode indices are assigned
    contiguously in add order.
    """

    def __init__(self, out_root: Path, fps: float, size: Tuple[int, int],
                 roles: List[str], camera_keys: Dict[str, str],
                 data_file_size_mb: float = V30_DATA_FILE_SIZE_MB,
                 video_file_size_mb: float = V30_VIDEO_FILE_SIZE_MB):
        self.out_root = Path(out_root)
        self.fps = fps
        self.size = size
        self.roles = roles
        self.camera_keys = camera_keys
        self.data_file_size_mb = data_file_size_mb
        self.video_file_size_mb = video_file_size_mb

        self._data_chunk = 0
        self._data_file = 0
        self._tables: List[pa.Table] = []
        self._tables_bytes = 0

        # per-camera accumulation for the current packed video file
        self._vid = {role: {"chunk": 0, "file": 0, "paths": [],
                            "mb": 0.0, "dur": 0.0} for role in roles}
        self._tmp_dir = self.out_root / "tmp_v30_videos"

        self.global_index = 0          # running frame counter
        self.episode_rows: List[Dict] = []
        self._ep_stats: List[Dict] = []

    @property
    def episode_count(self) -> int:
        return len(self.episode_rows)

    # -- data parquets ------------------------------------------------------

    def _flush_data(self) -> None:
        if not self._tables:
            return
        path = self.out_root / V30_DATA_PATH.format(
            chunk_index=self._data_chunk, file_index=self._data_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(self._tables), path)
        self._tables = []
        self._tables_bytes = 0
        self._data_chunk, self._data_file = _bump(
            self._data_chunk, self._data_file)

    # -- packed videos --------------------------------------------------------

    def _flush_videos(self, role: str) -> None:
        v = self._vid[role]
        if not v["paths"]:
            return
        video_key = self.camera_keys[role]
        out_path = self.out_root / V30_VIDEO_PATH.format(
            video_key=video_key, chunk_index=v["chunk"], file_index=v["file"])
        concat_videos(v["paths"], out_path)
        for p in v["paths"]:
            Path(p).unlink(missing_ok=True)
        v["chunk"], v["file"] = _bump(v["chunk"], v["file"])
        v["paths"], v["mb"], v["dur"] = [], 0.0, 0.0

    # -- episodes -------------------------------------------------------------

    def add_episode(self, task_label: str, task_index: int,
                    state: Optional[np.ndarray], action: Optional[np.ndarray],
                    clips: Dict[str, Path],
                    verbose: bool = True,
                    n_frames: Optional[int] = None) -> Dict[str, int]:
        """Append one episode (frame rows + camera videos + stats).

        Returns the per-role encoded frame counts, mirroring the v2.1
        writer's sanity output.
        """
        episode_index = self.episode_count
        n = int(state.shape[0]) if state is not None else int(n_frames)

        table = episode_frame_table(state, action, self.fps, episode_index,
                                    task_index, self.global_index, n_frames=n)
        if (self._tables and (self._tables_bytes + table.nbytes) / (1 << 20)
                >= self.data_file_size_mb):
            self._flush_data()
        self._tables.append(table)
        self._tables_bytes += table.nbytes

        row = {
            "episode_index": episode_index,
            "tasks": [task_label],
            "length": n,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "data/chunk_index": self._data_chunk,
            "data/file_index": self._data_file,
            "dataset_from_index": self.global_index,
            "dataset_to_index": self.global_index + n,
        }

        video_samples: Dict[str, np.ndarray] = {}
        frame_counts: Dict[str, int] = {}
        for role in self.roles:
            video_key = self.camera_keys[role]
            v = self._vid[role]
            clip = clips.get(role)
            if clip is None:
                # Camera missing for this episode (warned upstream): record
                # an empty timestamp range in the current file.
                row[f"videos/{video_key}/chunk_index"] = v["chunk"]
                row[f"videos/{video_key}/file_index"] = v["file"]
                row[f"videos/{video_key}/from_timestamp"] = v["dur"]
                row[f"videos/{video_key}/to_timestamp"] = v["dur"]
                continue

            tmp = self._tmp_dir / video_key / f"episode_{episode_index:06d}.mp4"
            cnt = encode_video(clip, tmp, self.fps, n, self.size)
            frame_counts[role] = cnt
            ep_mb = tmp.stat().st_size / (1 << 20)
            ep_dur = video_duration_s(tmp)
            if ep_dur <= 0:
                ep_dur = n / self.fps

            if v["paths"] and v["mb"] + ep_mb >= self.video_file_size_mb:
                self._flush_videos(role)

            row[f"videos/{video_key}/chunk_index"] = v["chunk"]
            row[f"videos/{video_key}/file_index"] = v["file"]
            row[f"videos/{video_key}/from_timestamp"] = v["dur"]
            row[f"videos/{video_key}/to_timestamp"] = v["dur"] + ep_dur
            v["paths"].append(tmp)
            v["mb"] += ep_mb
            v["dur"] += ep_dur

            samples = sample_video_frames(tmp, IMAGE_SAMPLES_PER_EPISODE,
                                          self.size)
            if samples is not None:
                video_samples[video_key] = samples
            if verbose:
                flag = "" if cnt == n else f"  (!= {n} parquet rows)"
                print(f"  [G] {role:11} -> {video_key} "
                      f"chunk-{v['chunk']:03d}/file-{v['file']:03d} @ "
                      f"{row[f'videos/{video_key}/from_timestamp']:.2f}s  "
                      f"{cnt} frames{flag}")

        stats = compute_episode_stats(state, action, self.fps, episode_index,
                                      task_index, self.global_index,
                                      video_samples, n_frames=n)
        self._ep_stats.append(stats)
        row.update(_flatten_stats(stats))
        self.episode_rows.append(row)
        self.global_index += n
        return frame_counts

    # -- meta -----------------------------------------------------------------

    def finalize(self, features: Dict, tasks: List[str],
                 extra_info: Optional[Dict] = None) -> None:
        """Flush pending files and write meta/ (info, episodes, tasks, stats).

        ``features`` is the same dict the v2.1 writer puts in info.json;
        v3.0 additionally carries ``fps`` on every non-video feature.
        """
        self._flush_data()
        for role in self.roles:
            self._flush_videos(role)
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

        meta_dir = self.out_root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        total_episodes = self.episode_count
        fps_int = int(round(self.fps))

        tasks_path = self.out_root / V30_TASKS_PATH
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"task_index": range(len(tasks))},
                     index=list(tasks)).to_parquet(tasks_path)

        ep_path = self.out_root / V30_EPISODES_PATH.format(
            chunk_index=0, file_index=0)
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.episode_rows).to_parquet(ep_path, index=False)

        features = json.loads(json.dumps(features))  # deep copy
        for ft in features.values():
            if ft.get("dtype") != "video":
                ft["fps"] = fps_int

        info = {
            "codebase_version": CODEBASE_VERSION_V30,
            "robot_type": None,
            "total_episodes": total_episodes,
            "total_frames": self.global_index,
            "total_tasks": len(tasks),
            "chunks_size": V30_CHUNKS_SIZE,
            "data_files_size_in_mb": self.data_file_size_mb,
            "video_files_size_in_mb": self.video_file_size_mb,
            "fps": fps_int,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": V30_DATA_PATH,
            "video_path": V30_VIDEO_PATH,
            "features": features,
        }
        if extra_info:
            info.update(extra_info)
        (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

        (meta_dir / "stats.json").write_text(
            json.dumps(aggregate_stats(self._ep_stats), indent=2))
