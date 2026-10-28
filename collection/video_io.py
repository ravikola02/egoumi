# Vendored from h2r_collection (github.com/ravikola02/h2r_collection, MIT) scripts/video_io.py at 0783627.
"""ffmpeg/ffprobe helpers shared by the LeRobot dataset writers.

Torch-free (safe to import outside the hawor env). Everything shells out to
the system ffmpeg/ffprobe; ``convert_to_lerobot.py`` fixes PATH so the
non-snap ffmpeg is used.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def encode_video(clip: Path, out_path: Path, fps: float,
                 n_frames: int, size: Tuple[int, int]) -> int:
    """ffmpeg clip -> single H.264 MP4 at `size` and `fps`. Returns the
    written frame count (probed back so we can sanity-check vs parquet)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    w, h = size
    cmd = [
        "ffmpeg", "-v", "error", "-noautorotate",
        "-i", str(clip),
        "-vf", f"fps={fps},scale={w}:{h}",
        "-frames:v", str(n_frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "23",
        "-an",  # no audio
        "-y", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"ffmpeg encode failed for {clip.name}:\n"
                           + "\n".join(tail[-10:]))
    # Probe back the actual frame count.
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1", str(out_path),
    ], capture_output=True, text=True)
    try:
        return int(probe.stdout.strip())
    except ValueError:
        return -1


def sample_video_frames(video_path: Path, n_samples: int,
                        size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Decode `n_samples` evenly-spaced JPEG frames via ffmpeg and return
    (S, H, W, 3) uint8. None on failure."""
    if not video_path.is_file():
        return None
    w, h = size
    # `select` keeps every k-th frame where k = nb_frames / n_samples;
    # easiest: scale + thumbnail filter is fine for a few samples.
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", f"thumbnail=10,scale={w}:{h}",
        "-frames:v", str(n_samples),
        "-pix_fmt", "rgb24",
        "-f", "rawvideo", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or len(proc.stdout) < w * h * 3:
        return None
    frame_bytes = w * h * 3
    got = len(proc.stdout) // frame_bytes
    if got == 0:
        return None
    arr = np.frombuffer(proc.stdout[:got * frame_bytes], dtype=np.uint8)
    return arr.reshape(got, h, w, 3).copy()


def video_duration_s(video_path: Path) -> float:
    """Video-stream duration in seconds via ffprobe; -1.0 on failure."""
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=nokey=1:noprint_wrappers=1", str(video_path),
    ], capture_output=True, text=True)
    try:
        return float(probe.stdout.strip())
    except ValueError:
        return -1.0


def concat_videos(paths: List[Path], out_path: Path) -> None:
    """Concatenate identically-encoded MP4s into one file (concat demuxer,
    stream copy -- no re-encode). All inputs must share codec/size/fps,
    which holds because every episode goes through ``encode_video`` with the
    same settings. This is the same mechanism LeRobot itself uses to pack
    episodes into its v3.0 video files."""
    if not paths:
        raise ValueError("concat_videos called with no input paths")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = out_path.with_suffix(".ffconcat")
    lines = ["ffconcat version 1.0"]
    lines += [f"file '{Path(p).resolve()}'" for p in paths]
    list_path.write_text("\n".join(lines) + "\n")
    cmd = [
        "ffmpeg", "-v", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        "-y", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    list_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"ffmpeg concat failed for {out_path.name}:\n"
                           + "\n".join(tail[-10:]))
