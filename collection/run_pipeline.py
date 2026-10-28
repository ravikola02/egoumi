#!/usr/bin/env python3
"""egoumi collection pipeline: raw 3-camera session -> sync + tag + cut -> LeRobot dataset.

    python3 collection/run_pipeline.py --session <session>
    python3 collection/run_pipeline.py --session A --session B --collection combo
    python3 collection/run_pipeline.py --session <session> --skip-tag        # re-convert existing clips

Input:  data/raw/<session>/{top_cam,left_w_cam,right_w_cam}/<one>.MP4  (head, wrist_left, wrist_right)
Output: data/output/<collection>/{clips.json, episode_clips/, lerobot_v3/ (or lerobot_v2/)}

Same flow as h2r_collection's scripts/run_pipeline.py (the tagger and sync are vendored from it), but the
converter runs without HaWoR: no end-effector reconstruction or gripper values yet, see --state-source.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import sync_videos as sv          # only its path-taking helpers are used here

REPO_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data"


def child_env(data_root: Path) -> dict:
    """Both stages shell out to ffmpeg; the snap build silently breaks audio sync on non-home paths."""
    env = os.environ.copy()
    env["EGOUMI_DATA_ROOT"] = str(data_root)
    if Path("/usr/bin/ffmpeg").is_file():
        env["PATH"] = "/usr/bin:" + env.get("PATH", "")
    return env


def resolve_session(raw_root: Path, name: str | None) -> Path:
    if name:
        candidate = raw_root / name
        if not candidate.is_dir():
            sys.exit(f"Session folder not found: {candidate}")
        return candidate
    sessions = [s for s in sv.discover_sessions(raw_root) if sv.has_reference_camera(s)]
    if not sessions:
        sys.exit(f"No session with a head camera under {raw_root} (pass --session NAME)")
    return sessions[-1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", action="append", dest="sessions",
                   help="session folder under --raw-root (repeatable; 2+ need --collection)")
    p.add_argument("--collection", help="combined output name (default: the single session's name)")
    p.add_argument("--data-root", default=os.environ.get("EGOUMI_DATA_ROOT", str(DEFAULT_DATA_ROOT)),
                   help="holds raw/<session>/ and output/<collection>/ (default: %(default)s; "
                        "env EGOUMI_DATA_ROOT)")
    p.add_argument("--dataset-version", choices=["v21", "v30"], default="v30")
    p.add_argument("--state-source", default="none", help="none (video-only, default) | zeros | sim_gt | aruco")
    p.add_argument("--sim-gt", help="passed to the converter (sim sessions)")
    p.add_argument("--camera", help="passed to the converter (head intrinsics for aruco)")
    p.add_argument("--markers", help="passed to the converter (marker layout)")
    p.add_argument("--ema-pos", help="passed to the converter (aruco position EMA alpha, default 0.18)")
    p.add_argument("--ema-grip", help="passed to the converter (aruco gripper EMA alpha, default 1.0 = off)")
    p.add_argument("--fps", default="auto", help="auto = the head clips' own rate")
    p.add_argument("--image-size", default="1280x720")
    p.add_argument("--skip-tag", action="store_true", help="reuse the existing cut sheet + clips")
    p.add_argument("--overwrite", action="store_true", help="replace an existing dataset dir")
    p.add_argument("--python", default=sys.executable, help="interpreter for both stages")
    args = p.parse_args(argv)

    data_root = Path(args.data_root).resolve()
    raw_root, output_root = data_root / "raw", data_root / "output"
    sessions = [resolve_session(raw_root, n) for n in (args.sessions or [None])]
    multi = len(sessions) > 1
    collection = args.collection or (None if multi else sessions[0].name)
    if collection is None:
        sys.exit("Multiple --session require --collection NAME")
    # Same cut-sheet location rule as h2r_collection: a bare single session keeps data/raw/<s>/clips.json.
    cut_sheet = (output_root / collection / "clips.json") if (args.collection or multi) else sessions[0] / "clips.json"
    clips_dir = output_root / collection / "episode_clips"
    out = output_root / collection / ("lerobot_v3" if args.dataset_version == "v30" else "lerobot_v2")

    print("=" * 70)
    print(f"  collection: {collection}")
    for s in sessions:
        print(f"  session:    {s.name}   ({sv.detect_session_mode(s)})")
    print(f"  cut sheet:  {cut_sheet}\n  clips dir:  {clips_dir}\n  dataset:    {out}")
    print(f"  state:      {args.state_source}")
    print("=" * 70)

    env = child_env(data_root)
    if args.skip_tag:
        if not cut_sheet.is_file():
            sys.exit(f"--skip-tag set but {cut_sheet} does not exist")
        n = len(json.loads(cut_sheet.read_text()).get("clips", []))
        print(f"[stage 1] tagger SKIPPED  ({n} episodes in cut sheet)")
    else:
        cmd = [args.python, str(HERE / "episode_tagger.py")]
        for s in sessions:
            cmd += ["--session", s.name]
        cmd += ["--raw-root", str(raw_root), "--collection", collection, "--out", str(cut_sheet),
                "--output-dir", str(clips_dir.relative_to(output_root))]
        print(f"[stage 1] tagger -> {' '.join(cmd)}")
        if subprocess.call(cmd, cwd=str(REPO_ROOT), env=env) != 0:
            sys.exit("Tagger failed")

    if not cut_sheet.is_file():
        sys.exit(f"No cut sheet at {cut_sheet} -- did the tagger save? (Save + Cut)")
    if not any(clips_dir.glob("*_head.mp4")):
        sys.exit(f"No head clips in {clips_dir}")

    cmd = [args.python, str(HERE / "convert_to_lerobot.py"), "--cut-sheet", str(cut_sheet),
           "--clips-dir", str(clips_dir), "--out", str(out), "--dataset-version", args.dataset_version,
           "--state-source", args.state_source, "--fps", str(args.fps), "--image-size", args.image_size]
    if args.overwrite:
        cmd.append("--overwrite")
    for flag in ("sim_gt", "camera", "markers", "ema_pos", "ema_grip"):
        if getattr(args, flag):
            cmd += [f"--{flag.replace('_', '-')}", getattr(args, flag)]
    print(f"[stage 2] converter -> {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)
    print("\nDone.  Dataset at:", out) if rc == 0 else print(f"\nConverter exited with status {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
