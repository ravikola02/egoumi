#!/usr/bin/env python3
"""Cut sheet from the gripper (wrist) camera: one episode per working stretch between put-downs.

For long takes where the gripper works in stretches (e.g. bin picking: pick every part from a box into a bin, put the
gripper down, refill the box by hand, repeat). Replaces hand-tagging in collection/episode_tagger.py.

  1. per wrist frame (every --stride-th): motion energy (mean abs difference of 160x90 grey frames) and the pixel centres
     of the gripper's two jaw tags;
  2. rest = smoothed motion below --rest-frac of its p99 for at least --rest-s seconds (the gripper lies still on the
     table); the stretches between rests are candidate episodes;
  3. jaw closure = pixel distance between the two jaw tags, normalised per session (robust to a camera that shifts on
     the gripper); grasps = hysteresis at 30 % / 70 % of each stretch's p20..p80 closure range;
  4. episode = first grasp - --pad .. last release + --pad, clipped to the stretch and to the sync overlap window;
     kept when it has >= --min-grasps grasps, lasts >= --min-s seconds and its closure range is > 0.15.
Times are on the session timeline (head camera = reference) from data/raw/<session>/sync.json, as the tagger writes them.

  python3 scripts/segment_by_wrist_activity.py --session bin_picking_1 --session bin_picking_2 --gripper left \\
      --task "place the parts from the box into the bin" --out data/output/bin_picking/clips.json
Check the boundaries on video before cutting (collection/sync_videos.py --config ... per session).
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from umi_pipe import pose as P                     # noqa: E402
from umi_pipe.model import GripperModel            # noqa: E402


def wrist_signals(path, markers, gripper, stride):
    m = GripperModel(markers, gripper); det = P.make_detector(m)
    T1, T2 = [t.id for t in sorted(m.jaw_tags, key=lambda t: t.center[0])]
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS)
    t, c1, c2, mot = [], [], [], []; prev = None; k = -1
    while True:
        if not cap.grab():
            break
        k += 1
        if k % stride:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
        mot.append(np.nan if prev is None else float(np.mean(np.abs(g - prev)))); prev = g
        d = P.detect(det, img)
        for tid, lst in ((T1, c1), (T2, c2)):
            lst.append(np.asarray(d[tid], float).reshape(4, 2).mean(0) if tid in d else [np.nan, np.nan])
        t.append(k / fps)
    return np.array(t), np.array(c1), np.array(c2), np.array(mot)


def runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def segment(t, c1, c2, mot, lo, hi, a):
    sep = np.linalg.norm(c2 - c1, axis=1)
    s_lo, s_hi = np.nanpercentile(sep, [1, 99])
    g = np.clip((s_hi - sep) / (s_hi - s_lo), 0, 1)                    # 1 = closed
    mm = np.convolve(np.nan_to_num(mot), np.ones(15) / 15, "same")
    dt = np.median(np.diff(t))
    rest = [(i, j) for i, j in runs(mm / np.percentile(mm, 99) < a.rest_frac) if (j - i) * dt >= a.rest_s]
    bounds = [0] + [x for ij in rest for x in ij] + [len(t)]
    eps = []
    for i, j in zip(bounds[::2], bounds[1::2]):
        if j - i < 2:
            continue
        idx = np.arange(i, j); gi = g[idx]; ok = np.isfinite(gi); idx, gi = idx[ok], gi[ok]
        q20, q80 = np.percentile(gi, [20, 80]) if len(gi) else (0.0, 1.0)
        th_hi, th_lo = q20 + 0.7 * (q80 - q20), q20 + 0.3 * (q80 - q20)
        closed, grasps, c0 = False, [], None
        for k, v in zip(idx, gi):
            if not closed and v > th_hi:
                closed, c0 = True, k
            elif closed and v < th_lo:
                closed = False; grasps.append((c0, k))
        if closed:
            grasps.append((c0, idx[-1]))
        s, e = t[i], t[j - 1]
        if grasps:
            s = max(t[i], t[grasps[0][0]] - a.pad); e = min(t[j - 1], t[grasps[-1][1]] + a.pad)
        s, e = max(s, lo + 0.05), min(e, hi - 0.05)
        keep = len(grasps) >= a.min_grasps and e - s >= a.min_s and (q80 - q20) > 0.15
        print(f"  active {t[i]:7.1f}-{t[j - 1]:7.1f}s  grasps {len(grasps):3d}  -> {s:7.1f}-{e:7.1f}s ({e - s:6.1f}s)  "
              f"{'KEEP' if keep else 'drop'}")
        if keep:
            eps.append((round(float(s), 3), round(float(e), 3), len(grasps)))
    return eps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", action="append", required=True, dest="sessions")
    ap.add_argument("--data-root", default=os.environ.get("EGOUMI_DATA_ROOT", os.path.join(ROOT, "data")))
    ap.add_argument("--markers", default=os.path.join(ROOT, "configs", "gripper_markers.yaml"))
    ap.add_argument("--gripper", default="0", help="gripper in the marker layout (name or index); sets the wrist camera")
    ap.add_argument("--task", required=True, help="task_label of every episode")
    ap.add_argument("--out", required=True, help="cut sheet to write")
    ap.add_argument("--output-dir", help="cut sheet output_dir (default: <out parent name>/episode_clips)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--rest-frac", type=float, default=0.15)
    ap.add_argument("--rest-s", type=float, default=4.0)
    ap.add_argument("--pad", type=float, default=1.0)
    ap.add_argument("--min-grasps", type=int, default=5)
    ap.add_argument("--min-s", type=float, default=20.0)
    a = ap.parse_args()

    hand = GripperModel(a.markers, a.gripper).hand
    role, sub = f"wrist_{hand}", f"{hand}_w_cam"
    jobs = {}
    with ProcessPoolExecutor(len(a.sessions)) as ex:
        for s in a.sessions:
            d = os.path.join(a.data_root, "raw", s, sub)
            vids = sorted(f for f in os.listdir(d) if f.lower().endswith((".mp4", ".mov")))
            jobs[s] = ex.submit(wrist_signals, os.path.join(d, vids[0]), a.markers, a.gripper, a.stride)
        sig = {s: f.result() for s, f in jobs.items()}

    clips = []; grasps = []
    for s in a.sessions:
        sy = json.load(open(os.path.join(a.data_root, "raw", s, "sync.json")))
        lo, hi = sy["overlap_window_s"]; off = sy["cameras"][role]["offset_used_s"]
        t, c1, c2, mot = sig[s]
        print(f"{s}: wrist offset {off:+.4f}s, overlap {lo:.1f}-{hi:.1f}s")
        for st, en, ng in segment(t + off, c1, c2, mot, lo, hi, a):
            i = len(clips) + 1
            clips.append({"name": f"ep{i:03d}", "start": st, "end": en, "episode_id": i, "task_label": a.task,
                          "dominant_hand": hand, "session": s})
            grasps.append(ng)
    out_dir = a.output_dir or f"{os.path.basename(os.path.dirname(os.path.abspath(a.out)))}/episode_clips"
    sheet = {"output_dir": out_dir, "clips": clips,
             "segmentation": {"method": "scripts/segment_by_wrist_activity.py: wrist-camera motion rests (gripper put "
                                        f"down, >= {a.rest_s:g} s below {a.rest_frac:g} x p99) split working stretches; "
                                        f"episode = first grasp - {a.pad:g} s .. last release + {a.pad:g} s (jaw-tag "
                                        "separation, per-stretch hysteresis 30/70 % of p20..p80); kept with >= "
                                        f"{a.min_grasps} grasps and >= {a.min_s:g} s", "gripper": a.gripper,
                              "grasps": grasps}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(sheet, open(a.out, "w"), indent=1)
    print(f"{len(clips)} episodes, {sum(c['end'] - c['start'] for c in clips):.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
