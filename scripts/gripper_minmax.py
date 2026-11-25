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
  4. rows without a jaw tag are interpolated inside the episode (held at the ends);
  5. causal EMA per episode (--ema, default 0.18 = the pose's alpha, so state and gripper carry one lag; 1 = off).
Writes observation.state[:, <hand>.gripper] and action[:, <hand>.gripper] (= next state's, h2r convention) and
refreshes their stats: column 13 for the right-hand gripper, 12 for the left-hand one (--gripper).

--measure separation (for a camera that is not rigid on the body, e.g. the clip-on DJI cover): steps 1-2 assume the
camera never moves on the gripper, so each tag slides along one fixed image path. When the mount shifts between or
within sessions the paths smear (bin_picking 2026-11-24: index->thumb residual 57-68 px in two of four sessions).
The pixel distance between the two jaw tags moves with the jaws but hardly with the camera, so it replaces the path
signal; frames that see one tag only take a per-group linear map from that tag's path coordinate to the distance.
--group session normalises (paths, maps, open/closed percentiles) per recording session of the cut sheet, since a
remounted camera changes every pixel signal.

  python3 scripts/gripper_minmax.py --dataset data/output/<c>/lerobot_v3 --clips data/output/<c>/episode_clips
  python3 scripts/gripper_minmax.py ... --gripper left --measure separation --group session
"""
import argparse, glob, json, os, shutil, sys
import numpy as np, pandas as pd, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "collection"))
from umi_pipe import pose as P                     # noqa: E402
from umi_pipe.model import GripperModel            # noqa: E402
from state_sources import row_frames, ema, SLOTS     # noqa: E402


def signals(path, n, fps, det, ids):
    """Per jaw tag id: [(row, pixel centre)] for the rows where it is detected."""
    raw = {t: [] for t in ids}
    it = row_frames(path, n, fps) if n else enumerate(iter(lambda c=cv2.VideoCapture(path): c.read()[1], None))
    for k, img in it:
        if img is None:
            break
        d = P.detect(det, img)
        for tid in ids:
            if tid not in d:
                continue
            raw[tid].append((k, np.asarray(d[tid], float).reshape(4, 2).mean(0)))
    return raw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--clips", required=True, help="episode_clips dir with <name>_wrist_<hand>.mp4")
    ap.add_argument("--gripper", default="0", help="gripper in the marker layout: name (right, left) or index (default: "
                    "the first, right); sets the jaw tag ids, the wrist clip and the state column")
    ap.add_argument("--measure", choices=["path", "separation"], default="path",
                    help="path (default): each jaw tag's position along its image path (rigid camera); separation: "
                         "pixel distance between the two jaw tags (camera that shifts on the body)")
    ap.add_argument("--group", choices=["none", "session"], default="none",
                    help="normalise over all episodes (none, default) or per recording session of the cut sheet")
    ap.add_argument("--cut-sheet", help="cut sheet with each clip's session (default: <dataset>/../clips.json)")
    ap.add_argument("--camera", default=os.path.join(ROOT, "configs", "cameras_osmo_nano.yaml"),
                    help="unused (pixel signals need no intrinsics); kept for old command lines")
    ap.add_argument("--markers", default=os.path.join(ROOT, "configs", "gripper_markers.yaml"))
    ap.add_argument("--calib-clip", help="wrist clip that opens and closes fully (sets open/closed instead of the session)")
    ap.add_argument("--lo", type=float, default=1.0, help="percentile taken as fully open (default 1)")
    ap.add_argument("--hi", type=float, default=99.5, help="percentile taken as closed (default 99.5)")
    ap.add_argument("--ema", type=float, default=0.18, help="causal EMA alpha on the gripper value (1 = off)")
    ap.add_argument("--no-backup", action="store_true")
    a = ap.parse_args()

    ds = a.dataset
    info = json.load(open(os.path.join(ds, "meta", "info.json"))); fps = float(info["fps"])
    eps = json.load(open(os.path.join(ds, "meta", "egoumi_episodes.json")))
    m = GripperModel(a.markers, a.gripper); det = P.make_detector(m)
    GRIP = SLOTS[m.hand]["grip"]                                    # 13 right.gripper, 12 left.gripper
    role = f"wrist_{m.hand}"
    # tag on the left holder first (it sits left in the wrist view), then the right holder: closing moves them together
    T1, T2 = [t.id for t in sorted(m.jaw_tags, key=lambda t: t.center[0])]
    print(f"gripper '{m.name}' ({m.hand} hand): jaw tags {T1}/{T2}, clips *_{role}.mp4, state column {GRIP}, "
          f"measure {a.measure}, group {a.group}")
    if a.calib_clip and (a.group != "none" or a.measure != "path"):
        sys.exit("--calib-clip works with --measure path --group none only")

    def run(path, n):
        return signals(path, n, fps, det, (T1, T2))

    raws = {}
    for e in eps:
        p = os.path.join(a.clips, f"{e['name']}_{role}.mp4")
        raws[e["episode_index"]] = (run(p, e["length"]) if os.path.isfile(p) else {T1: [], T2: []}, e["length"])
    calib = run(a.calib_clip, 0) if a.calib_clip else None

    groups = {"all": list(raws)}
    if a.group == "session":
        cs = a.cut_sheet or os.path.join(ds, "..", "clips.json")
        sess = {c["name"]: c.get("session", "") for c in json.load(open(cs))["clips"]}
        groups = {}
        for e in eps:
            groups.setdefault(sess.get(e["name"], ""), []).append(e["episode_index"])

    def fit_group(members):
        """Signals of one group's episodes on a common scale (closing positive); returns ({ep: (s, fill)}, params)."""
        pool = [raws[i][0] for i in members] + ([calib] if calib else [])
        ax, mid = {}, {}
        for tid in (T1, T2):
            c = np.array([x[1] for r in pool for x in r[tid]]).reshape(-1, 2)
            if len(c) < 3:
                ax[tid], mid[tid] = np.array([1e-3, 0.0]), np.zeros(2)
                continue
            mid[tid] = c.mean(0)
            v = np.linalg.svd(c - mid[tid], full_matrices=False)[2][0]
            ax[tid] = v * (1 if (v[0] > 0) == (tid == T1) else -1) / 1000.0  # "mm" = px / 1000, for the prints

        def series(r, n):        # columns: path coordinate of T1, of T2, minus the T1-T2 pixel distance
            s = np.full((n, 3), np.nan); cen = {T1: {}, T2: {}}
            for j, tid in enumerate((T1, T2)):
                for k, c in r[tid]:
                    if k < n:
                        s[k, j] = ax[tid] @ (c - mid[tid]); cen[tid][k] = c
            for k in set(cen[T1]) & set(cen[T2]):
                s[k, 2] = -np.linalg.norm(cen[T2][k] - cen[T1][k]) / 1000.0
            return s

        S = {i: series(*raws[i]) for i in members}
        allS = np.concatenate(list(S.values()))
        prim, fb = (2, (0, 1)) if a.measure == "separation" else (0, (1,))
        maps = {}
        for j in fb:                                 # primary ~ slope * fallback + offset over frames that see both
            both = np.isfinite(allS[:, prim]) & np.isfinite(allS[:, j])
            if both.sum() < 10:
                continue
            A = np.c_[allS[both, j], np.ones(both.sum())]
            maps[j] = np.linalg.lstsq(A, allS[both, prim], rcond=None)[0]
            res = allS[both, prim] - A @ maps[j]
            print(f"    map col {j}->{prim}: slope {maps[j][0]:.3f}, residual median {1000 * np.median(np.abs(res)):.2f} px "
                  f"over {both.sum()} frames")

        def merged(s):
            g = s[:, prim].copy(); fill = np.zeros(len(g), bool)
            for j, (sl, of) in maps.items():
                f = ~np.isfinite(g) & np.isfinite(s[:, j])
                g[f] = sl * s[f, j] + of; fill |= f
            return g, fill

        out = {i: merged(s) for i, s in S.items()}
        if calib:
            ref = merged(series(calib, max(x[0] for t in (T1, T2) for x in calib[t]) + 1))[0]
        else:
            ref = np.concatenate([g for g, _ in out.values()])
        ref = ref[np.isfinite(ref)]
        s_open, s_closed = np.percentile(ref, a.lo), np.percentile(ref, a.hi)
        print(f"    open {1000 * s_open:.1f} px, closed {1000 * s_closed:.1f} px  (span {1000 * (s_closed - s_open):.1f} px; "
              f"source: {'calibration clip' if calib else f'percentiles p{a.lo:g}/p{a.hi:g}'})")
        return out, {"open_px": float(s_open * 1000), "closed_px": float(s_closed * 1000),
                     "fallback_maps": {str(j): [float(v) for v in c] for j, c in maps.items()},
                     "episodes": [int(i) for i in members]}

    grip = {}; stats = []; params = {}
    for gname, members in groups.items():
        print(f"  group '{gname}': {len(members)} episode(s)")
        out, params[gname] = fit_group(members)
        s_open, s_closed = params[gname]["open_px"] / 1000, params[gname]["closed_px"] / 1000
        for i, (g, fill) in out.items():
            seen = np.isfinite(g); n = len(g)
            x = np.clip((g - s_open) / (s_closed - s_open), 0, 1)
            if seen.any():
                x = np.interp(np.arange(n), np.where(seen)[0], x[seen])
            else:
                x = np.zeros(n)
            x = ema(x, a.ema)
            grip[i] = x.astype(np.float32)
            stats.append((i, 100 * seen.mean(), 100 * (fill & seen).mean(), x.min(), x.max()))
    for i, sv, fi, lo, hi in sorted(stats):
        print(f"  episode {i}: jaw seen {sv:5.1f} %  (from the fallback tag {fi:4.1f} %)  gripper {lo:.2f}-{hi:.2f}")

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

    method = {"path": "wrist-camera jaw-tag pixel centre along its image path",
              "separation": "pixel distance between the two wrist-camera jaw tags (single-tag frames: linear map from "
                            "that tag's path coordinate)"}[a.measure]
    g0 = next(iter(params.values()))
    info.setdefault("egoumi", {})["gripper"] = {
        "method": f"scripts/gripper_minmax.py: {method}, min/max normalised"
                  + (", per recording session" if a.group == "session" else ""),
        "gripper": m.name, "hand": m.hand, "state_column": GRIP, "jaw_tags": [T1, T2], "measure": a.measure,
        "open_px": g0["open_px"], "closed_px": g0["closed_px"], "fallback_maps": g0["fallback_maps"],
        "reference": "calibration clip" if calib else f"percentiles p{a.lo:g} / p{a.hi:g}",
        "note": "1 = the most-closed jaw position seen in the reference, not necessarily fully closed"}
    if a.group == "session":
        info["egoumi"]["gripper"]["groups"] = params
    info["egoumi"]["gripper"]["ema_alpha"] = float(a.ema)
    if isinstance(info["egoumi"].get("ema"), dict):
        info["egoumi"]["ema"]["gripper_alpha"] = float(a.ema)
    json.dump(info, open(os.path.join(ds, "meta", "info.json"), "w"), indent=4)
    print("dataset updated")


if __name__ == "__main__":
    main()
