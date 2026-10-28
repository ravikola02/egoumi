#!/usr/bin/env python3
"""Cut episode clips (head + wrist_left + wrist_right) -> LeRobot dataset (v3.0 default, or v2.1).

    python3 collection/convert_to_lerobot.py --cut-sheet data/output/<c>/clips.json \
        --clips-dir data/output/<c>/episode_clips --out data/output/<c>/lerobot_v3

Same inputs, camera keys and output layouts as h2r_collection's converter, without HaWoR: the episode
length comes from the clips themselves (shortest camera at --fps), and observation.state / action come
from a pluggable source (collection/state_sources.py; default `none` = video-only, `zeros` = flagged
14-d placeholder). End-effector reconstruction and gripper values are not implemented yet.
"""

import os
import sys
from pathlib import Path

# Force /usr/bin/ffmpeg ahead of the snap ffmpeg (sandboxed: cannot read non-home paths such as the SSD).
if Path("/usr/bin/ffmpeg").is_file():
    os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")

import argparse
import json
import math
import subprocess
import traceback
from fractions import Fraction
from typing import Dict, List, Tuple

from episodes import ROLES, load_episodes
from lerobot_v21 import V21DatasetWriter
from lerobot_v30 import V30DatasetWriter
from state_sources import SOURCES, get_source
from video_io import video_duration_s

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_VERSION = "egoumi-collection-v0.1"
CAMERA_KEYS = {
    "head":        "observation.images.head",
    "wrist_left":  "observation.images.wrist_left",
    "wrist_right": "observation.images.wrist_right",
}


def parse_size(s: str) -> Tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def episode_length(clips: Dict[str, Path], fps: float) -> Tuple[int, Dict[str, float]]:
    """Frames every camera can supply at `fps`: floor(shortest duration * fps)."""
    durs = {role: video_duration_s(p) for role, p in clips.items()}
    bad = [r for r, d in durs.items() if d <= 0]
    if bad:
        raise RuntimeError(f"could not probe duration of {bad}")
    return int(math.floor(min(durs.values()) * fps + 1e-6)), durs


def probe_fps(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=avg_frame_rate", "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(Fraction(out)) if out and out != "0/0" else 30.0


def build_features(fps: float, size: Tuple[int, int], roles: List[str], source) -> Dict:
    w, h = size
    features: Dict[str, Dict] = {}
    if source.dim:
        for key in ["observation.state", "action"] + list(getattr(source, "extra_keys", [])):
            features[key] = {"dtype": "float32", "shape": [source.dim], "names": list(source.names)}
    features.update({
        "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
        "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
        "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
        "index":         {"dtype": "int64",   "shape": [1], "names": None},
        "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
    })
    for role in roles:
        features[CAMERA_KEYS[role]] = {
            "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channel"],
            "info": {"video.fps": float(fps), "video.codec": "h264", "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "has_audio": False},
        }
    return features


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cut-sheet", required=True, help="clips.json written by the tagger / sync_videos.py")
    p.add_argument("--clips-dir", required=True, help="dir with {name}_{role}.mp4 clips")
    p.add_argument("--out", required=True, help="dataset root (must not exist yet, or pass --overwrite)")
    p.add_argument("--dataset-version", choices=["v21", "v30"], default="v30")
    p.add_argument("--state-source", choices=sorted(SOURCES), default="none",
                   help="none = video-only; zeros = flagged placeholder; sim_gt = sim ground-truth trajectory; "
                        "aruco = gripper pose reconstructed from the head clips")
    p.add_argument("--sim-gt", help="<session>/sim_gt of a sim/render_capture.py session (sim_gt source, aruco scoring "
                                    "and its head-camera motion)")
    p.add_argument("--camera", help="head intrinsics yaml for --state-source aruco (sim: <session>/sim_gt/cameras.yaml)")
    p.add_argument("--markers", help="marker layout yaml (default configs/gripper_markers.yaml)")
    p.add_argument("--cam-motion", default="sim_gt", help="head-camera motion for aruco: sim_gt (SLAM not wired yet)")
    p.add_argument("--ema-pos", type=float, default=0.18,
                   help="aruco: causal EMA alpha on the predicted TCP position (default 0.18 from sim/ema_sweep.py: "
                        "one-step delta noise at the G1 FK level; h2r kitting used 0.164; 1.0 = off)")
    p.add_argument("--ema-grip", type=float, default=1.0,
                   help="aruco: causal EMA alpha on the predicted gripper value (default 1.0 = off: the wrist-camera "
                        "width is already clean; h2r_collection used 0.3 for MANO)")
    p.add_argument("--fps", default="auto",
                   help="dataset + video fps; auto = the head clips' own rate (default; v30 needs an integer)")
    p.add_argument("--image-size", default="1280x720", help="WxH of the encoded camera streams")
    p.add_argument("--session-id", help="recorded in info.json (default: cut-sheet parent dir name)")
    p.add_argument("--limit", type=int, help="process at most N episodes")
    p.add_argument("--overwrite", action="store_true", help="delete an existing --out first")
    args = p.parse_args(argv)

    cut_sheet, clips_dir, out_root = Path(args.cut_sheet), Path(args.clips_dir), Path(args.out)
    if out_root.exists():
        if not args.overwrite:
            sys.exit(f"{out_root} exists; pass --overwrite to replace it")
        import shutil
        shutil.rmtree(out_root)
    size = parse_size(args.image_size)
    source = get_source(args.state_source, cut_sheet=cut_sheet, sim_gt=args.sim_gt, camera=args.camera,
                        markers=args.markers, cam_motion=args.cam_motion, ema_pos=args.ema_pos,
                        ema_grip=args.ema_grip)
    session_id = args.session_id or cut_sheet.parent.name

    episodes = load_episodes(cut_sheet, clips_dir)
    if args.limit:
        episodes = episodes[: args.limit]
    roles = [r for r in ROLES if any(r in ep.clips for ep in episodes)]
    if args.fps == "auto":
        rates = sorted({round(probe_fps(ep.head_clip), 3) for ep in episodes})
        if len(rates) > 1:
            print(f"WARN head clips have different frame rates {rates}; resampling all to {rates[0]:g}")
        args.fps = rates[0]
    args.fps = float(args.fps)
    if args.dataset_version == "v30" and args.fps != int(args.fps):
        sys.exit(f"LeRobot v3.0 needs an integer fps, got {args.fps:g}; pass --fps")
    Writer = V30DatasetWriter if args.dataset_version == "v30" else V21DatasetWriter
    writer = Writer(out_root, fps=args.fps, size=size, roles=roles, camera_keys=CAMERA_KEYS)

    print(f"Resolved {len(episodes)} episode(s) from {cut_sheet}")
    print(f"  version:  LeRobot {'v3.0' if args.dataset_version == 'v30' else 'v2.1'}")
    print(f"  cameras:  {roles}")
    print(f"  state:    {source.name}" + ("  (PLACEHOLDER)" if source.placeholder else ""))
    print(f"  size:     {size[0]}x{size[1]} @ {args.fps:g} fps")
    print(f"  out:      {out_root}")

    tasks: List[str] = []
    task_idx: Dict[str, int] = {}
    for ep in episodes:
        label = ep.task_label or ""
        if label not in task_idx:
            task_idx[label] = len(tasks); tasks.append(label)

    summaries, failures = [], []
    for ep in episodes:
        episode_index = writer.episode_count
        print(f"\n=== episode {episode_index:06d} (id={ep.episode_id:04d}) '{ep.name}' "
              f"[{ep.task_label or 'no-label'}] ===")
        try:
            n, durs = episode_length(ep.clips, args.fps)
            if n < 2:
                raise RuntimeError(f"episode too short: {n} frames (durations {durs})")
            spread = max(durs.values()) - min(durs.values())
            if spread > 1.5 / args.fps:
                print(f"  WARN camera durations differ by {spread * 1000:.0f} ms: "
                      + ", ".join(f"{r} {d:.3f}s" for r, d in durs.items()))
            state, action = source.compute(ep, n, args.fps)
            counts = writer.add_episode(task_label=ep.task_label or "", task_index=task_idx[ep.task_label or ""],
                                        state=state, action=action, clips=ep.clips, n_frames=n,
                                        extra=getattr(source, "extra", None) or None)
            short = {r: c for r, c in counts.items() if c != n}
            ep_info = dict(getattr(source, "episode_info", {}) or {})
            if ep_info:
                print("  [S] " + "  ".join(f"{k}={v}" for k, v in ep_info.items()))
            summaries.append({"episode_index": episode_index, "name": ep.name, "length": n,
                              "frames": counts, "short": short, "task": ep.task_label or "", "state": ep_info})
        except Exception as exc:  # noqa: BLE001 - isolate per-episode failures
            print(f"ERROR {ep.name}: {exc}", file=sys.stderr)
            traceback.print_exc()
            failures.append((ep.name, str(exc)))

    if summaries:
        extra = {"egoumi": {"session_id": session_id, "cameras": roles, "bridge_version": BRIDGE_VERSION,
                            "camera_roles": {"head": "egocentric head camera",
                                             "wrist_left": "left gripper/wrist camera",
                                             "wrist_right": "right gripper/wrist camera"},
                            **source.info(),
                            "notes": "Built by egoumi collection/ (h2r_collection pipeline without HaWoR)."}}
        writer.finalize(features=build_features(args.fps, size, roles, source), tasks=tasks, extra_info=extra)
        (out_root / "meta" / "egoumi_episodes.json").write_text(json.dumps(summaries, indent=1))

    print("\n===== summary =====")
    for s in summaries:
        flag = f"  SHORT {s['short']}" if s["short"] else ""
        print(f"  ep{s['episode_index']:06d}  '{s['name']}'  rows={s['length']}  frames={s['frames']}{flag}")
    for name, err in failures:
        print(f"  {name}  FAILED: {err}", file=sys.stderr)
    print(f"{len(summaries)} ok, {len(failures)} failed -> {out_root}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
