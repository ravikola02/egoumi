#!/usr/bin/env python3
"""Choose the EMA alpha for --state-source aruco by the h2r criterion: the least smoothing whose one-step delta noise
reaches the G1 encoder-FK level (per-frame floor ~0.27 mm/axis -> ~0.68 mm per step over 3 axes).

    python3 sim/ema_sweep.py <repo root> <dir holding lerobot_v3_raw (aruco, --ema-pos 1 --ema-grip 1) and lerobot_v3_simgt>

Needs ground truth, so it runs on rendered captures; re-run it when the capture setup changes.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1] + "/collection")
from state_sources import ema
b = sys.argv[2]
rd = lambda n: pd.read_parquet(f"{b}/{n}/data/chunk-000/file-000.parquet")
R_, G_ = rd("lerobot_v3_raw"), rd("lerobot_v3_simgt")
SR = np.stack(R_["observation.state"].to_numpy()); SG = np.stack(G_["observation.state"].to_numpy())
ep = R_["episode_index"].to_numpy()
G1_FLOOR = 0.27                                   # mm, G1 encoder-FK per-frame noise floor (depth; 0.26-0.30 all axes)

def per_ep(fn):
    return np.concatenate([fn(np.where(ep == e)[0]) for e in np.unique(ep)])

def robust_sd(x):                                 # 1.4826 * MAD: spread of the bulk, ignores rare spikes
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)

raw_res = per_ep(lambda m: SR[m][:, 6:9] - SG[m][:, 6:9]) * 1000
print(f"raw ArUco position noise (pred - GT), per axis x / y / z(depth) [mm]:")
print(f"  robust sd {np.round(robust_sd(raw_res), 3)}   rms {np.round(np.sqrt((raw_res ** 2).mean(0)), 3)}   "
      f"max |e| {np.round(np.abs(raw_res).max(0), 1)}")
print(f"  (h2r kitting human: 1.33 / 1.85 / 4.35 mm; G1 robot FK: 0.30 / 0.26 / 0.27 mm)\n")

print(f"{'alpha':>6s} {'lag fr':>6s} | noise after filter, robust sd x/y/z [mm] | rms x/y/z [mm]       | "
      f"1-step delta rms [mm] | err vs raw GT med/p90 [mm]")
rows = []
for a in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.164, 0.12):
    noise = per_ep(lambda m: ema(SR[m][:, 6:9], a) - ema(SG[m][:, 6:9], a)) * 1000
    dnoise = per_ep(lambda m: np.diff(ema(SR[m][:, 6:9], a) - ema(SG[m][:, 6:9], a), axis=0)) * 1000
    tot = per_ep(lambda m: np.linalg.norm(ema(SR[m][:, 6:9], a) - SG[m][:, 6:9], axis=1)) * 1000
    rs, rms, drms = robust_sd(noise), np.sqrt((noise ** 2).mean(0)), np.sqrt((dnoise ** 2).mean(0))
    rows.append((a, rs, rms))
    print(f"{a:6.3f} {(1 - a) / a:6.2f} | {rs[0]:6.3f} {rs[1]:6.3f} {rs[2]:6.3f}                  | "
          f"{rms[0]:5.2f} {rms[1]:5.2f} {rms[2]:5.2f}   | {np.linalg.norm(drms):6.3f}               | "
          f"{np.median(tot):5.2f} / {np.percentile(tot, 90):5.2f}")
best_robust = max((a for a, rs, _ in rows if rs.max() <= G1_FLOOR), default=None)
best_rms = max((a for a, _, rms in rows if rms.max() <= G1_FLOOR), default=None)
print(f"\nleast smoothing reaching the G1 floor ({G1_FLOOR} mm on every axis): robust sd -> alpha {best_robust}; "
      f"rms (incl. spikes) -> alpha {best_rms}")

gr, gg = SR[:, 13], SG[:, 13]
print("\ngripper (wrist camera), noise in jaw-width mm (x 72 mm):")
for a in (1.0, 0.6, 0.3):
    gn = per_ep(lambda m: ema(gr[m], a) - ema(gg[m], a)) * 72
    gl = per_ep(lambda m: np.abs(ema(gr[m], a) - gg[m])) * 72
    print(f"  alpha {a:4.2f} (lag {(1 - a) / a:4.2f} fr): noise robust sd {robust_sd(gn):.3f}  rms {np.sqrt((gn ** 2).mean()):.3f}  "
          f"max {np.abs(gn).max():.2f} | err vs raw GT median {np.median(gl):.2f}  p90 {np.percentile(gl, 90):.2f}")
