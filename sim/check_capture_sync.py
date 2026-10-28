#!/usr/bin/env python3
"""Score the collection pipeline on a rendered capture (sim/render_capture.py) against its ground truth.

    python3 sim/check_capture_sync.py --session data/raw/sim_capture --clips data/output/sim_capture/episode_clips

1. Sync: offsets in <session>/sync.json (written by collection/sync_videos.py) vs the true record-start offsets.
2. Cut alignment: the first and last frame of every cut clip are matched by image content against the raw
   recording of the same camera; the matched raw frame is compared with where ground truth says it should be.
   Blank feeds (all-black) are skipped for content matching.
"""
import argparse, json, os, subprocess, sys
import numpy as np

FFMPEG = "/usr/bin/ffmpeg" if os.path.isfile("/usr/bin/ffmpeg") else "ffmpeg"
DIRS = {"head": "top_cam", "wrist_right": "right_w_cam", "wrist_left": "left_w_cam"}
W, H = 160, 90


def decode(path, start_frame=None, n=None, fps=30):
    """All frames (or a window) as (N, H, W) float32 grey thumbnails."""
    cmd = [FFMPEG, "-v", "error"]
    if start_frame is not None:
        cmd += ["-ss", f"{max(start_frame, 0) / fps:.6f}"]
    cmd += ["-i", path, "-vf", f"scale={W}:{H}", "-pix_fmt", "gray", "-f", "rawvideo"]
    if n is not None:
        cmd += ["-frames:v", str(n)]
    raw = subprocess.run(cmd + ["-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)


def best_match(frame, raw, center, radius):
    lo, hi = max(0, center - radius), min(len(raw), center + radius + 1)
    err = [np.mean((raw[k] - frame) ** 2) for k in range(lo, hi)]
    k = int(np.argmin(err)) + lo
    s = sorted(err)
    return k, s[0], (s[1] if len(s) > 1 else np.inf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--clips", required=True, help="episode_clips dir cut from this session")
    ap.add_argument("--cut-sheet", default=None, help="default <session>/sim_gt/clips_gt.json")
    a = ap.parse_args()

    gt = json.load(open(os.path.join(a.session, "sim_gt", "gt.json")))
    fps = gt["fps"]
    sync = json.load(open(os.path.join(a.session, "sync.json")))
    sheet = json.load(open(a.cut_sheet or os.path.join(a.session, "sim_gt", "clips_gt.json")))
    ok = True

    print("== sync: recovered vs true offset (camera record start - head record start)")
    for role, cam in sync["cameras"].items():
        true = gt["true_offset_vs_head_s"][role]
        got = cam["offset_used_s"]
        meta = cam["metadata_offset_s"] - sync["cameras"]["head"]["metadata_offset_s"]
        err_ms = (got - true) * 1000
        flag = "OK" if abs(err_ms) <= 0.5 / fps * 1000 else "FAIL"
        ok &= flag == "OK"
        print(f"  {role:11s} true {true:+.4f}s  recovered {got:+.4f}s ({cam['method']}, conf {cam['confidence']:.1f})"
              f"  err {err_ms:+6.1f} ms = {err_ms / 1000 * fps:+.2f} frames  [{flag}]"
              f"   (metadata alone: {meta:+.3f}s, err {(meta - true) * 1000:+.0f} ms)")

    print("\n== cut alignment: clip first/last frame matched against the raw recording by content")
    raws = {}
    for role in ("head", "wrist_right"):
        vid = [f for f in os.listdir(os.path.join(a.session, DIRS[role])) if f.lower().endswith((".mp4", ".mov"))][0]
        raws[role] = decode(os.path.join(a.session, DIRS[role], vid), fps=fps)
    worst = 0
    for c in sheet["clips"]:
        for role in ("head", "wrist_right"):
            clip = os.path.join(a.clips, f"{c['name']}_{role}.mp4")
            if not os.path.isfile(clip):
                print(f"  {c['name']} {role}: MISSING clip"); ok = False; continue
            frames = decode(clip, fps=fps)
            off = gt["true_offset_vs_head_s"][role]
            exp0 = int(round((c["start"] - off) * fps))           # raw frame index that should open the clip
            k0, e0, e0b = best_match(frames[0], raws[role], exp0, 15)
            kN, eN, _ = best_match(frames[-1], raws[role], exp0 + len(frames) - 1, 15)
            d0, dN = k0 - exp0, kN - (exp0 + len(frames) - 1)
            worst = max(worst, abs(d0), abs(dN))
            flag = "OK" if abs(d0) <= 1 and abs(dN) <= 1 else "FAIL"
            ok &= flag == "OK"
            print(f"  {c['name']:22s} {role:11s} {len(frames):4d} frames  start Δ {d0:+d}  end Δ {dN:+d} frames"
                  f"  (match mse {e0:.1f}, next-best {e0b:.1f})  [{flag}]")
        blank = os.path.join(a.clips, f"{c['name']}_wrist_left.mp4")
        if os.path.isfile(blank):
            b = decode(blank, fps=fps)
            print(f"  {c['name']:22s} wrist_left  {len(b):4d} frames  blank (max pixel {b.max():.0f})")
    print(f"\nworst cut misalignment: {worst} frame(s)   overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
