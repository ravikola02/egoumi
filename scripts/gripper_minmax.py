#!/usr/bin/env python3
"""Gripper value from the wrist camera by min/max normalisation, written into an existing egoumi LeRobot v3 dataset.

Why: the mount-based jaw fit (collection/state_sources.py, wrist T_body_cam + nominal 600 px pinhole) needs the real
mount and lens model; with the nominal ones it reads every frame as wider than open (gripper 0). This needs neither:
  1. per frame and per jaw tag (6 thumb side, 7 index side): the tag's pixel centre. The camera is rigid on the body,
     so each tag's centre moves along a fixed 1-D image path; its position along that path's principal direction
     (signed so closing is positive) is the jaw signal. (PnP translation of the 18 mm tag was tried first: its
     two-fold planar ambiguity made the signal jump between two levels.) Monotonic but not linear in jaw width
     (perspective, lens distortion), so gripper is a normalised, monotonic, not exactly linear width measure;
  2. the index-side signal is mapped onto the thumb-side one by a linear fit over frames that see both (the jaws are
     coupled, so the two move together; the index tag leaves the frame when the jaws open);
  3. gripper = (s - s_open) / (s_closed - s_open), clipped to [0, 1] (1 = closed, as h2r_collection);
     s_open / s_closed = the --lo / --hi percentiles of the session (or of a calibration clip, --calib-clip,
     recorded opening and closing fully before the session);
  4. rows without a jaw tag are interpolated inside the episode (held at the ends).
Writes observation.state[:, 13] and action[:, 13] (= next state's, h2r convention) and refreshes their stats.

  python3 scripts/gripper_minmax.py --dataset data/output/<c>/lerobot_v3 --clips data/output/<c>/episode_clips
"""
import argparse, glob, json, os, shutil, sys
import numpy as np, pandas as pd, cv2, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "collection"))
from umi_pipe import pose as P                     # noqa: E402
from umi_pipe.model import GripperModel            # noqa: E402
from state_sources import scaled_K, row_frames     # noqa: E402

GRIP = 13          # right.gripper in the h2r 14-d layout


def signals(path, n, fps, det, jaw, K, dist, axes=None):
    """(n, 2) thumb/index signals in metres along each tag's closing axis (NaN where unseen), and raw axes."""
    out = np.full((n, 2), np.nan); raw = {6: [], 7: []}
    it = row_frames(path, n, fps) if n else enumerate(iter(lambda c=cv2.VideoCapture(path): c.read()[1], None))
    for k, img in it:
        if img is None:
            break
        d = P.detect(det, img)
        for j, tid in enumerate((6, 7)):
            if tid not in d:
                continue
            raw[tid].append((k, np.asarray(d[tid], float).reshape(4, 2).mean(0)))
    return raw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--clips", required=True, help="episode_clips dir with <name>_wrist_right.mp4")
    ap.add_argument("--camera", default=os.path.join(ROOT, "configs", "cameras_osmo_nano.yaml"))
    ap.add_argument("--markers", default=os.path.join(ROOT, "configs", "gripper_markers.yaml"))
    ap.add_argument("--calib-clip", help="wrist clip that opens and closes fully (sets open/closed instead of the session)")
    ap.add_argument("--lo", type=float, default=1.0, help="percentile taken as fully open (default 1)")
    ap.add_argument("--hi", type=float, default=99.5, help="percentile taken as closed (default 99.5)")
    ap.add_argument("--no-backup", action="store_true")
    a = ap.parse_args()

    ds = a.dataset
    info = json.load(open(os.path.join(ds, "meta", "info.json"))); fps = float(info["fps"])
    eps = json.load(open(os.path.join(ds, "meta", "egoumi_episodes.json")))
    m = GripperModel(a.markers); jaw = {t.id: t for t in m.jaw_tags}; det = P.make_detector(m)
    wr = yaml.safe_load(open(a.camera))["wrist_right"]
    K = dist = None

    def run(path, n):
        nonlocal K, dist
        if K is None:
            cap = cv2.VideoCapture(path); ok, img = cap.read(); cap.release()
            K, dist = scaled_K(wr, img.shape[1], img.shape[0])
        return signals(path, n, fps, det, jaw, K, dist)

    raws = {}
    for e in eps:
        p = os.path.join(a.clips, f"{e['name']}_wrist_right.mp4")
        raws[e["episode_index"]] = (run(p, e["length"]) if os.path.isfile(p) else {6: [], 7: []}, e["length"])
    calib = run(a.calib_clip, 0) if a.calib_clip else None

    # one image path direction per tag over everything (principal axis of its pixel centres); closing = toward the
    # image centre line between the two tags (thumb tag sits left, index tag right in the wrist view)
    pool = [rr for rr, _ in raws.values()] + ([calib] if calib else [])
    ax, mid = {}, {}
    for tid in (6, 7):
        c = np.array([x[1] for r in pool for x in r[tid]])
        mid[tid] = c.mean(0)
        v = np.linalg.svd(c - mid[tid], full_matrices=False)[2][0]
        ax[tid] = v * (1 if (v[0] > 0) == (tid == 6) else -1) / 1000.0      # "mm" = px / 1000, for the prints

    def series(r, n):
        s = np.full((n, 2), np.nan)
        for j, tid in enumerate((6, 7)):
            for k, c in r[tid]:
                if k < n:
                    s[k, j] = ax[tid] @ (c - mid[tid])
        return s

    S = {i: series(r, n) for i, (r, n) in raws.items()}
    allS = np.concatenate(list(S.values()))
    both = np.isfinite(allS).all(1)
    A = np.c_[allS[both, 1], np.ones(both.sum())]
    coef = np.linalg.lstsq(A, allS[both, 0], rcond=None)[0]           # thumb ~ a * index + b
    fitres = allS[both, 0] - A @ coef
    print(f"index->thumb map: slope {coef[0]:.3f}, offset {coef[1] * 1000:.1f} px, residual median "
          f"{1000 * np.median(np.abs(fitres)):.2f} px over {both.sum()} frames")

    def merged(s):
        g = s[:, 0].copy(); fill = ~np.isfinite(g) & np.isfinite(s[:, 1])
        g[fill] = coef[0] * s[fill, 1] + coef[1]
        return g, fill

    ref = merged(series(calib, max(x[0] for t in (6, 7) for x in calib[t]) + 1))[0] if calib else \
        np.concatenate([merged(s)[0] for s in S.values()])
    ref = ref[np.isfinite(ref)]
    s_open, s_closed = np.percentile(ref, a.lo), np.percentile(ref, a.hi)
    print(f"open {1000 * s_open:.1f} px, closed {1000 * s_closed:.1f} px  (span {1000 * (s_closed - s_open):.1f} px of "
          f"tag motion; source: {'calibration clip' if calib else f'session p{a.lo:g}/p{a.hi:g}'})")

    grip = {}; stats = []
    for i, s in S.items():
        g, fill = merged(s)
        seen = np.isfinite(g); n = len(g)
        x = np.clip((g - s_open) / (s_closed - s_open), 0, 1)
        if seen.any():
            x = np.interp(np.arange(n), np.where(seen)[0], x[seen])
        else:
            x = np.zeros(n)
        grip[i] = x.astype(np.float32)
        stats.append((i, 100 * seen.mean(), 100 * (fill & seen).mean(), x.min(), x.max()))
    for i, sv, fi, lo, hi in stats:
        print(f"  episode {i}: jaw seen {sv:5.1f} %  (index-side only {fi:4.1f} %)  gripper {lo:.2f}-{hi:.2f}")

    # write into the parquet files
    files = sorted(glob.glob(os.path.join(ds, "data", "**", "*.parquet"), recursive=True))
    if not a.no_backup:
        bk = os.path.join(ds, "..", "lerobot_v3_backup_gripper")
        if not os.path.exists(bk):
            shutil.copytree(ds, bk); print(f"backup: {os.path.abspath(bk)}")
    allst, allac = [], []
    for f in files:
        df = pd.read_parquet(f)
        st = np.stack(df["observation.state"].values).astype(np.float32)
        for i in df["episode_index"].unique():
            rows = np.where(df["episode_index"].values == i)[0]
            order = rows[np.argsort(df["frame_index"].values[rows])]
            st[order, GRIP] = grip[int(i)][:len(order)]
        ac = np.stack(df["action"].values).astype(np.float32)
        for i in df["episode_index"].unique():
            rows = np.where(df["episode_index"].values == i)[0]
            order = rows[np.argsort(df["frame_index"].values[rows])]
            ac[order, GRIP] = np.r_[st[order[1:], GRIP], st[order[-1], GRIP]]
        df["observation.state"] = list(st); df["action"] = list(ac)
        df.to_parquet(f, index=False)
        allst.append(st); allac.append(ac)

    # stats: global stats.json + per-episode stats in meta/episodes
    def upd(d, x):
        d["min"] = x.min(0).tolist(); d["max"] = x.max(0).tolist(); d["mean"] = x.mean(0).tolist()
        d["std"] = x.std(0).tolist()
        for q in ("q01", "q10", "q50", "q90", "q99"):
            if q in d:
                d[q] = np.percentile(x, int(q[1:]), axis=0).tolist()
    sp = os.path.join(ds, "meta", "stats.json")
    if os.path.isfile(sp):
        st_all = json.load(open(sp))
        for key, x in (("observation.state", np.concatenate(allst)), ("action", np.concatenate(allac))):
            if key in st_all:
                upd(st_all[key], x)
        json.dump(st_all, open(sp, "w"), indent=4)
    for f in glob.glob(os.path.join(ds, "meta", "episodes", "**", "*.parquet"), recursive=True):
        me = pd.read_parquet(f)
        X = {"observation.state": np.concatenate(allst), "action": np.concatenate(allac)}
        epi = np.concatenate([pd.read_parquet(g)["episode_index"].values for g in files])
        for key in X:
            for stat in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
                col = f"stats/{key}/{stat}"
                if col not in me.columns:
                    continue
                vals = []
                for i in me["episode_index"].values:
                    x = X[key][epi == i]
                    v = {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0)}.get(stat)
                    if v is None:
                        v = np.percentile(x, int(stat[1:]), axis=0)
                    vals.append(v.astype(np.float64))
                me[col] = vals
        me.to_parquet(f, index=False)

    info.setdefault("egoumi", {})["gripper"] = {
        "method": "scripts/gripper_minmax.py: wrist-camera jaw-tag pixel centre along its image path, min/max normalised",
        "open_px": float(s_open * 1000), "closed_px": float(s_closed * 1000),
        "reference": "calibration clip" if calib else f"session percentiles p{a.lo:g} / p{a.hi:g}",
        "index_to_thumb": [float(coef[0]), float(coef[1])],
        "note": "1 = the most-closed jaw position seen in the reference, not necessarily fully closed"}
    json.dump(info, open(os.path.join(ds, "meta", "info.json"), "w"), indent=4)
    print("dataset updated")


if __name__ == "__main__":
    main()
