#!/usr/bin/env python3
"""Score HaWoR SLAM head-camera trajectories (collection/head_slam.py output) against a rendered capture's ground truth.

    python3 sim/check_slam.py --session data/raw/sim_capture --slam-cache data/output/sim_capture/slam_cache

Per episode, both trajectories are expressed in their own first frame (the dataset convention), then:
  raw     position / rotation error as used by the pipeline (metric scale from Metric3D)
  scale   best-fit Sim3 scale factor (1.0 = Metric3D's scale was right) and the residual after a Sim3 fit
  jitter  median per-frame step of the SLAM camera vs the true camera
"""
import argparse, glob, json, os
import numpy as np
from scipy.spatial.transform import Rotation as R


def first_frame(T):
    T0i = np.linalg.inv(T[0])
    return np.einsum("ij,njk->nik", T0i, T)


def umeyama(p, q):
    """s, R, t minimising |q - (s R p + t)|."""
    mp, mq = p.mean(0), q.mean(0)
    X, Y = p - mp, q - mq
    U, D, Vt = np.linalg.svd(Y.T @ X / len(p))
    S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rr = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / X.var(0).sum()
    return s, Rr, mq - s * Rr @ mp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--slam-cache", required=True)
    ap.add_argument("--json", help="write the per-episode numbers here")
    a = ap.parse_args()
    gt = np.load(os.path.join(a.session, "sim_gt", "gt_head.npz"))["T_world_cam"]
    fps = json.load(open(os.path.join(a.session, "sim_gt", "gt.json")))["fps"]
    clips = {c["name"]: c for c in json.load(open(os.path.join(a.session, "clips.json")))["clips"]}
    rows = []
    for path in sorted(glob.glob(os.path.join(a.slam_cache, "*", "cameras.npz"))):
        tag = os.path.basename(os.path.dirname(path))
        name = next((n for n in clips if tag.startswith(n + "_")), None)
        if name is None:
            continue
        z = np.load(path)
        S = first_frame(z["T_world_cam"])
        f0 = int(round(clips[name]["start"] * fps))
        G = first_frame(gt[f0:f0 + len(S)])
        p, q = S[:, :3, 3], G[:, :3, 3]
        rot = np.degrees(np.linalg.norm(R.from_matrix(np.einsum("nji,njk->nik", S[:, :3, :3], G[:, :3, :3])).as_rotvec(), axis=1))
        raw = np.linalg.norm(p - q, axis=1) * 1000
        s, Rr, t = umeyama(p, q)
        res = np.linalg.norm((s * (Rr @ p.T)).T + t - q, axis=1) * 1000
        step = lambda x: np.median(np.linalg.norm(np.diff(x, axis=0), axis=1)) * 1000
        r = dict(episode=name, masked=bool(z["masked"]), metric3d_scale=round(float(z["scale"]), 3),
                 raw_pos_mm_median=round(float(np.median(raw)), 1), raw_pos_mm_max=round(float(raw.max()), 1),
                 rot_deg_median=round(float(np.median(rot)), 2), rot_deg_max=round(float(rot.max()), 2),
                 scale_correction=round(float(s), 3), sim3_residual_mm_median=round(float(np.median(res)), 1),
                 sim3_residual_mm_max=round(float(res.max()), 1),
                 step_mm_slam=round(float(step(p)), 2), step_mm_true=round(float(step(q)), 2),
                 seconds=round(float(z["seconds"]), 1))
        rows.append(r)
        print(f"{name:20s} mask={'on ' if r['masked'] else 'off'} raw {r['raw_pos_mm_median']:6.1f}/{r['raw_pos_mm_max']:6.1f} mm"
              f"  rot {r['rot_deg_median']:.2f}/{r['rot_deg_max']:.2f} deg  scale x{r['scale_correction']:.3f}"
              f" (Metric3D {r['metric3d_scale']})  Sim3 residual {r['sim3_residual_mm_median']:.1f}/{r['sim3_residual_mm_max']:.1f} mm"
              f"  step {r['step_mm_slam']:.1f} vs {r['step_mm_true']:.1f} mm  {r['seconds']:.0f}s")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
