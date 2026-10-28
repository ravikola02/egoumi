"""LeRobot v2.1 dataset writer (one parquet + one MP4 per camera per episode).

The generic half of h2r_collection's scripts/convert_to_lerobot.py (stage G + meta writers, at 0783627),
with two egoumi changes: observation.state / action are optional (video-only datasets), and
meta/episodes_stats.jsonl is written (LeRobot 2.1 expects it; h2r_collection's v2.1 exports lacked it).
Stats come from the same per-episode code as the v3.0 writer, so both layouts carry identical numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from lerobot_v30 import (IMAGE_SAMPLES_PER_EPISODE, aggregate_stats,
                         compute_episode_stats, episode_frame_table)
from video_io import encode_video, sample_video_frames

CODEBASE_VERSION_V21 = "v2.1"
V21_CHUNKS_SIZE = 1000
V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
V21_STAT_KEYS = ("min", "max", "mean", "std", "count")     # v2.1 per-episode stats carry no quantiles


class V21DatasetWriter:
    """Same add_episode / finalize interface as lerobot_v30.V30DatasetWriter."""

    def __init__(self, out_root: Path, fps: float, size: Tuple[int, int],
                 roles: List[str], camera_keys: Dict[str, str]):
        self.out_root = Path(out_root)
        self.fps = fps
        self.size = size
        self.roles = roles
        self.camera_keys = camera_keys
        self.global_index = 0
        self.episode_rows: List[Dict] = []
        self._ep_stats: List[Dict] = []

    @property
    def episode_count(self) -> int:
        return len(self.episode_rows)

    def _path(self, template: str, episode_index: int, **kw) -> Path:
        return self.out_root / template.format(episode_chunk=episode_index // V21_CHUNKS_SIZE,
                                               episode_index=episode_index, **kw)

    def add_episode(self, task_label: str, task_index: int,
                    state: Optional[np.ndarray], action: Optional[np.ndarray],
                    clips: Dict[str, Path], verbose: bool = True,
                    n_frames: Optional[int] = None,
                    extra: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, int]:
        episode_index = self.episode_count
        n = int(state.shape[0]) if state is not None else int(n_frames)

        table = episode_frame_table(state, action, self.fps, episode_index,
                                    task_index, self.global_index, n_frames=n, extra=extra)
        path = self._path(V21_DATA_PATH, episode_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)

        frame_counts: Dict[str, int] = {}
        video_samples: Dict[str, np.ndarray] = {}
        for role in self.roles:
            clip = clips.get(role)
            if clip is None:
                continue
            video_key = self.camera_keys[role]
            out = self._path(V21_VIDEO_PATH, episode_index, video_key=video_key)
            cnt = encode_video(clip, out, self.fps, n, self.size)
            frame_counts[role] = cnt
            samples = sample_video_frames(out, IMAGE_SAMPLES_PER_EPISODE, self.size)
            if samples is not None:
                video_samples[video_key] = samples
            if verbose:
                flag = "" if cnt == n else f"  (!= {n} parquet rows)"
                print(f"  [G] {role:11} -> {out.relative_to(self.out_root)}  {cnt} frames{flag}")

        stats = compute_episode_stats(state, action, self.fps, episode_index, task_index,
                                      self.global_index, video_samples, n_frames=n, extra=extra)
        self._ep_stats.append(stats)
        self.episode_rows.append({"episode_index": episode_index, "tasks": [task_label], "length": n})
        self.global_index += n
        return frame_counts

    def finalize(self, features: Dict, tasks: List[str], extra_info: Optional[Dict] = None) -> None:
        meta = self.out_root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        total = self.episode_count
        n_videos = sum(1 for ft in features.values() if ft.get("dtype") == "video")
        info = {
            "codebase_version": CODEBASE_VERSION_V21,
            "robot_type": None,
            "total_episodes": total,
            "total_frames": self.global_index,
            "total_tasks": len(tasks),
            "total_videos": total * n_videos,
            "total_chunks": (total + V21_CHUNKS_SIZE - 1) // V21_CHUNKS_SIZE or 1,
            "chunks_size": V21_CHUNKS_SIZE,
            "fps": float(self.fps),
            "splits": {"train": f"0:{total}"},
            "data_path": V21_DATA_PATH,
            "video_path": V21_VIDEO_PATH,
            "features": features,
        }
        if extra_info:
            info.update(extra_info)
        (meta / "info.json").write_text(json.dumps(info, indent=2))
        with (meta / "episodes.jsonl").open("w") as f:
            for row in self.episode_rows:
                f.write(json.dumps(row) + "\n")
        with (meta / "tasks.jsonl").open("w") as f:
            for i, t in enumerate(tasks):
                f.write(json.dumps({"task_index": i, "task": t}) + "\n")
        with (meta / "episodes_stats.jsonl").open("w") as f:
            for row, st in zip(self.episode_rows, self._ep_stats):
                f.write(json.dumps({"episode_index": row["episode_index"],
                                    "stats": {k: {s: np.asarray(v[s]).tolist() for s in V21_STAT_KEYS}
                                              for k, v in st.items()}}) + "\n")
        agg = aggregate_stats(self._ep_stats)
        agg = {k: {s: v[s] for s in V21_STAT_KEYS} for k, v in agg.items()}
        (meta / "stats.json").write_text(json.dumps(agg, indent=2))
