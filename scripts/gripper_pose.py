#!/usr/bin/env python3
"""ArUco gripper pose for one sequence (HaWoR layout: <seq>/extracted_images/*.jpg + <seq>/camera.yaml).

    ~/anaconda3/envs/hawor/bin/python scripts/gripper_pose.py --seq sim/episodes/pick_place --cam-source gt

Per frame: detect tags -> multi-tag PnP of the body -> jaw travel from the holder tags -> world pose
T_world_tcp = T_world_cam . T_cam_body . T_body_tcp. Short gaps are interpolated, long gaps stay invalid.

--cam-source picks T_world_cam (what HaWoR's SLAM provides in the real pipeline):
    gt     sim ground truth (<seq>/gt.npz) = perfect SLAM; isolates the marker pipeline
    table  per-frame table-tag pose (frames without the table tag are invalid)
    none   camera frame only (T_world_cam = I)
Writes <seq>/gripper_pose/poses.npz and detections.json.
"""
import os, sys, glob, json, argparse
import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as R, Slerp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from umi_pipe.geometry import inv, make_T
from umi_pipe.model import GripperModel, MARKER_CFG
from umi_pipe import pose as P

SRC_NONE, SRC_MEAS, SRC_INTERP = 0, 1, 2


def load_camera(path):
    c = yaml.safe_load(open(path))
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])
    return K, np.asarray(c.get("dist", [0] * 5), float), c


def fill_gaps(T, src, max_gap):
    """Interpolate runs of SRC_NONE no longer than max_gap between two measured poses (in place)."""
    n = len(T); i = 0
    while i < n:
        if src[i] != SRC_NONE:
            i += 1; continue
        j = i
        while j < n and src[j] == SRC_NONE:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            a, b = i - 1, j
            s = Slerp([0, 1], R.from_matrix([T[a][:3, :3], T[b][:3, :3]]))
            for k in range(i, j):
                u = (k - a) / (b - a)
                T[k] = make_T(s(u).as_matrix(), (1 - u) * T[a][:3, 3] + u * T[b][:3, 3])
                src[k] = SRC_INTERP
        i = j


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--markers", default=MARKER_CFG)
    ap.add_argument("--camera", default=None, help="default <seq>/camera.yaml")
    ap.add_argument("--cam-source", default="gt", choices=["gt", "table", "none"])
    ap.add_argument("--max-gap", type=int, default=5, help="max frames to interpolate across")
    ap.add_argument("--outlier-px", type=float, default=3.0)
    ap.add_argument("--no-subpix", action="store_true")
    ap.add_argument("--no-joint", action="store_true", help="skip the joint body+jaw refinement")
    ap.add_argument("--min-body-tags", type=int, default=1, help="frames with fewer body tags are invalid")
    a = ap.parse_args()

    model = GripperModel(a.markers)
    K, dist, _ = load_camera(a.camera or os.path.join(a.seq, "camera.yaml"))
    frames = sorted(glob.glob(os.path.join(a.seq, "extracted_images", "*.jpg")))
    n = len(frames)
    assert n, f"no frames in {a.seq}/extracted_images"
    det = P.make_detector(model, subpix=not a.no_subpix)

    if a.cam_source == "gt":
        T_world_cam = np.load(os.path.join(a.seq, "gt.npz"))["T_world_cam"][:n].copy()
    else:
        T_world_cam = np.tile(np.eye(4), (n, 1, 1))
    cam_src = np.full(n, SRC_MEAS if a.cam_source != "table" else SRC_NONE)

    T_cam_body = np.tile(np.eye(4), (n, 1, 1)); body_src = np.zeros(n, int)
    travel = np.full((n, 2), np.nan); travel_rms = np.full((n, 2), np.nan)
    rms = np.full(n, np.nan); n_tags = np.zeros(n, int); single = np.zeros(n, bool); confident = np.zeros(n, bool)
    detections = []
    T_prev = None; T_table_prev = None
    for f, path in enumerate(frames):
        img = cv2.imread(path)
        dets = P.detect(det, img)
        rec = {"frame": f, "tags": {str(k): v.round(3).tolist() for k, v in dets.items()}}
        b = P.estimate_body(dets, model, K, dist, T_prev, outlier_px=a.outlier_px)
        if b is not None and len(b["ids"]) >= a.min_body_tags:
            T_cam_body[f] = b["T_cam_body"]; body_src[f] = SRC_MEAS; T_prev = b["T_cam_body"]
            rms[f] = b["rms"]; n_tags[f] = len(b["ids"]); single[f] = b["single"]; confident[f] = b["confident"]
            rec.update(used=b["ids"], rejected=b["rejected"])
            tr1 = P.estimate_travel(dets, model, K, dist, b["T_cam_body"])
            for side, (tr, e) in tr1.items():
                k = 0 if side == "left" else 1
                travel[f, k] = tr; travel_rms[f, k] = e
            if tr1 and not a.no_joint:
                T_j, tr_j, rms_j = P.refine_joint(dets, model, K, dist, b["T_cam_body"],
                                                  {sd: v[0] for sd, v in tr1.items()}, b["ids"])
                if rms_j is not None and rms_j < max(2.0, 2 * b["rms"]):
                    T_cam_body[f] = T_j; T_prev = T_j; rms[f] = rms_j
                    # width: 1-D re-fit on the joint pose (the joint travels trade error with the pose)
                    for side, (tr, e) in P.estimate_travel(dets, model, K, dist, T_j).items():
                        k = 0 if side == "left" else 1
                        travel[f, k] = tr; travel_rms[f, k] = e
                    rec["joint"] = True
        if a.cam_source == "table":
            t = P.estimate_table(dets, model, K, dist, T_table_prev)
            if t is not None:
                T_world_cam[f] = inv(t["T_cam_world"]); cam_src[f] = SRC_MEAS; T_table_prev = t["T_cam_world"]
        detections.append(rec)
        if f % 60 == 0:
            print(f"frame {f + 1}/{n}: tags {sorted(dets)}", flush=True)

    # gaps: camera (table mode) and body in their own frames, then compose
    if a.cam_source == "table":
        fill_gaps(T_world_cam, cam_src, a.max_gap)
    T_world_body = T_world_cam @ T_cam_body
    world_src = np.where((body_src == SRC_MEAS) & (cam_src != SRC_NONE), SRC_MEAS, SRC_NONE)
    world_src[(body_src == SRC_MEAS) & (cam_src == SRC_INTERP)] = SRC_INTERP
    fill_gaps(T_world_body, world_src, a.max_gap)
    T_world_tcp = T_world_body @ model.T_body_tcp
    T_cam_body_all = inv_batch(T_world_cam) @ T_world_body        # interpolated frames re-expressed in camera

    # width: average of the sides seen (mechanism is synchronous); gaps <= max_gap interpolated, longer ones NaN
    tr = np.nanmean(travel, axis=1) if np.isfinite(travel).any() else np.full(n, np.nan)
    width_src = np.where(np.isfinite(tr), SRC_MEAS, SRC_NONE)
    tr_f = tr.copy()
    Tw = np.tile(np.eye(4), (n, 1, 1)); Tw[:, 0, 3] = np.nan_to_num(tr)       # reuse fill_gaps on a 1-D track
    fill_gaps(Tw, width_src, a.max_gap)
    tr_f[width_src == SRC_INTERP] = Tw[width_src == SRC_INTERP, 0, 3]
    width = model.width_from_travel(np.clip(tr_f, 0, model.travel_max))

    out = os.path.join(a.seq, "gripper_pose")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "poses.npz"), T_world_cam=T_world_cam, cam_src=cam_src,
                        T_cam_body=T_cam_body_all, body_src=body_src, world_src=world_src,
                        T_world_body=T_world_body, T_world_tcp=T_world_tcp, travel=travel, travel_rms=travel_rms,
                        width=width, width_src=width_src, width_weak=(width_src == SRC_MEAS) & single, rms=rms, n_tags=n_tags, single=single, confident=confident,
                        cam_source=a.cam_source, K=K, dist=dist)
    with open(os.path.join(out, "detections.json"), "w") as fh:
        json.dump(detections, fh)
    meas = (world_src == SRC_MEAS).mean() * 100; interp = (world_src == SRC_INTERP).mean() * 100
    print(f"{n} frames: pose measured {meas:.1f} %, interpolated {interp:.1f} %, invalid {100 - meas - interp:.1f} %; "
          f"width measured {(width_src == SRC_MEAS).mean() * 100:.1f} % -> {out}")


def inv_batch(T):
    return np.stack([inv(x) for x in T])


if __name__ == "__main__":
    main()
