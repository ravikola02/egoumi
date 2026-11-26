#!/usr/bin/env python3
"""Two-gripper preview of one collection episode in a GRAVITY-ALIGNED table frame.

    PYOPENGL_PLATFORM=egl ~/anaconda3/envs/hawor/bin/python scripts/dual_preview.py \
        --collection data/output/<c> --episode ep001

Inputs (all written by the converter, `--state-source aruco --grippers right,left`, and scripts/episode_preview.py):
  <c>/slam_cache/<ep>_*/cameras.npz       head-camera poses (masked DROID-SLAM), world = first head-camera frame
  <c>/slam_cache/<ep>_*/aruco_poses.npz   per-gripper body pose in the head camera, spike-rejected (valid mask)
  <c>/preview/<ep>/extracted_images/      head frames at the dataset rows; camera.yaml = their intrinsics (staged from
                                          <c>/episode_clips/<ep>_head.mp4 + configs/cameras_osmo_nano.yaml if missing)
World frame of the preview:
  up      = the table-plane normal: Metric3D metric depth on --frames head frames, RANSAC plane per frame, rotated into
            the SLAM world and robust-averaged (as h2r_il scripts/gravity_from_visuals.py). The floor is parallel, so
            either plane gives gravity; the table HEIGHT is checked against the TCP tracks (see --table-from).
  origin  = on the table plane, under the first head-camera position; x = the first head camera's view direction
            made horizontal, y = left, z = up.
Gaps <= --max-gap rows are interpolated; longer gaps hide that gripper (not held). Jaw width is not shown (open).
Writes <c>/preview/<ep>/preview_dual/dual_preview.mp4 and gravity.json.
"""
import argparse, glob, json, os, subprocess, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "collection"), os.path.join(ROOT, "scripts")):
    sys.path.insert(0, p)
from umi_pipe.geometry import inv, apply, look_at_cv               # noqa: E402
from umi_pipe.model import GripperModel, MARKER_CFG, load_meshes    # noqa: E402
from umi_pipe.render import Scene3D                                 # noqa: E402
from state_sources import fill_gaps, ema, ema_rot, row_frames, scaled_K   # noqa: E402
from preview_gripper import (project, draw_poly, frustum_lines, banner, legend, label, camera_meshes,  # noqa: E402
                             CAM_BGR, FFMPEG)

METRIC3D = os.path.expanduser("~/dev/h2r_collection/HaWoR/thirdparty/Metric3D")
COLOR = {"right": ((0.56, 0.36, 0.86, 1.0), (219, 92, 143)),     # purple (rgba, bgr)
         "left": ((0.15, 0.78, 0.80, 1.0), (204, 199, 38))}       # teal


def fit_plane(P, rng, iters=400, thr=0.015):
    """RANSAC + SVD refine on camera-frame points. Returns (unit normal toward the camera, distance, inlier frac)."""
    n = len(P); best, best_c = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(n, 3, replace=False)]
        nv = np.cross(b - a, c - a); L = np.linalg.norm(nv)
        if L < 1e-9:
            continue
        nv /= L; d = -nv @ a
        cnt = int((np.abs(P @ nv + d) < thr).sum())
        if cnt > best_c:
            best_c, best = cnt, (nv, d)
    nv, d = best
    inl = np.abs(P @ nv + d) < thr
    Q = P[inl]; c0 = Q.mean(0)
    nv = np.linalg.svd(Q - c0, full_matrices=False)[2][-1]; d = -nv @ c0
    if d < 0:
        nv, d = -nv, -d
    return nv, d, inl.mean()


def robust_mean_dir(V, iters=5):
    m = V.mean(0); m /= np.linalg.norm(m)
    for _ in range(iters):
        ang = np.degrees(np.arccos(np.clip(V @ m, -1, 1)))
        w = 1.0 / np.maximum(ang, 0.5)
        m = (w[:, None] * V).sum(0); m /= np.linalg.norm(m)
    return m, np.degrees(np.arccos(np.clip(V @ m, -1, 1)))


def estimate_gravity(frames, K, T_wc, n_frames, dmin, dmax, G):
    """up (world), table offset h (plane: up . x = h, METRIC), per-frame spread [deg], inlier frac, depth scale.
    Metric3D's depth scale is checked against the ArUco tags (metric): ratio = tag depth from the PnP pose / Metric3D
    depth at the tag centre, median over the visible camera-facing tags; plane distances are multiplied by it."""
    cwd = os.getcwd(); sys.path.insert(0, METRIC3D); os.chdir(METRIC3D)
    from metric import Metric3D
    model = Metric3D(os.path.join(METRIC3D, "weights/metric_depth_vit_large_800k.pth")); os.chdir(cwd)
    rng = np.random.default_rng(0); ups, ds, cs, inl, ratios = [], [], [], [], []
    for k in np.unique(np.linspace(0, len(T_wc) - 1, n_frames).round().astype(int)):
        img = cv2.cvtColor(cv2.imread(frames[k]), cv2.COLOR_BGR2RGB); h, w = img.shape[:2]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        D = cv2.resize(model(img, np.array([fx, fy, cx, cy])), (w, h), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.mgrid[0:h:4, 0:w:4]; z = D[ys, xs]; ok = (z > dmin) & (z < dmax)
        P = np.stack([(xs[ok] - cx) / fx * z[ok], (ys[ok] - cy) / fy * z[ok], z[ok]], 1)
        if len(P) < 500:
            continue
        nc, d, fr = fit_plane(P, rng)
        ups.append(T_wc[k, :3, :3] @ nc); ds.append(d); cs.append(T_wc[k, :3, 3]); inl.append(fr)
        for g in G.values():
            if not g["vis_meas"][k]:
                continue
            for tg in g["model"].body_tags:
                pc = apply(g["T_cb"][k], tg.center[None])[0]; nrm = g["T_cb"][k][:3, :3] @ tg.normal
                if pc[2] < 0.05 or nrm @ (pc / np.linalg.norm(pc)) > -0.5:      # behind, or not facing the camera
                    continue
                u, v = (K @ pc)[:2] / pc[2]
                if 0 <= u < w and 0 <= v < h and D[int(v), int(u)] > 0:
                    ratios.append(pc[2] / D[int(v), int(u)])
    up, spread = robust_mean_dir(np.array(ups))
    scale = float(np.median(ratios)) if len(ratios) >= 5 else 1.0
    hs = [up @ c - scale * d for c, d in zip(cs, ds)]
    print(f"Metric3D depth x {scale:.2f} to match the ArUco tags ({len(ratios)} tag samples, IQR "
          f"{np.percentile(ratios, 25):.2f}-{np.percentile(ratios, 75):.2f})" if ratios else "no tag samples")
    return up, float(np.median(hs)), spread, float(np.median(inl)), scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--grippers", default="right,left")
    ap.add_argument("--frames", type=int, default=12, help="head frames for the Metric3D table fit")
    ap.add_argument("--min-depth", type=float, default=0.35, help="drop nearer points (operator's body, hands)")
    ap.add_argument("--max-depth", type=float, default=2.5)
    ap.add_argument("--table-from", choices=["auto", "plane", "tcp"], default="auto",
                    help="table height: the fitted plane, the lowest TCP (p1 - 1 cm), or auto = plane unless it lies "
                         "more than 15 cm under the lowest TCP (then it was the floor)")
    ap.add_argument("--max-gap", type=int, default=5)
    ap.add_argument("--ema", type=float, default=0.18,
                    help="world view: the converter's causal EMA on position + SO(3) rotation, restarted after every "
                         "tracking gap (1 = raw). The camera overlay always shows the raw pose")
    ap.add_argument("--panel-width", type=int, default=800)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--scene-only", action="store_true", help="write preview_dual/scene.npz (for the web viewer) and stop")
    a = ap.parse_args()

    col = os.path.abspath(a.collection); seq = os.path.join(col, "preview", a.episode)
    cache = sorted(glob.glob(os.path.join(col, "slam_cache", f"{a.episode}_*", "cameras.npz")), key=os.path.getmtime)[-1]
    T_wc = np.load(cache)["T_world_cam"]
    ar = np.load(os.path.join(os.path.dirname(cache), "aruco_poses.npz"))
    if not glob.glob(os.path.join(seq, "extracted_images", "*.jpg")):   # stage: head frames at the SLAM rows + K
        info = os.path.join(col, "lerobot_v3", "meta", "info.json")
        fps_ds = float(json.load(open(info))["fps"]) if os.path.isfile(info) else 30.0
        os.makedirs(os.path.join(seq, "extracted_images"), exist_ok=True); img = None
        for k, img in row_frames(os.path.join(col, "episode_clips", f"{a.episode}_head.mp4"), len(T_wc), fps_ds):
            cv2.imwrite(os.path.join(seq, "extracted_images", f"{k + 1:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        hc = yaml.safe_load(open(os.path.join(ROOT, "configs", "cameras_osmo_nano.yaml"))); hc = hc.get("head", hc)
        Kh, dh = scaled_K(hc, img.shape[1], img.shape[0])
        yaml.safe_dump({"width": int(img.shape[1]), "height": int(img.shape[0]), "fx": float(Kh[0, 0]), "fy": float(Kh[1, 1]),
                        "cx": float(Kh[0, 2]), "cy": float(Kh[1, 2]), "dist": [float(v) for v in dh], "fps": fps_ds},
                       open(os.path.join(seq, "camera.yaml"), "w"))
        print(f"staged {len(T_wc)} head frames -> {seq}/extracted_images")
    frames = sorted(glob.glob(os.path.join(seq, "extracted_images", "*.jpg")))
    cam = yaml.safe_load(open(os.path.join(seq, "camera.yaml")))
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]]); W, H = cam["width"], cam["height"]
    fps = cam.get("fps", 30.0)
    n = min(len(frames), len(T_wc))                     # poses + gravity use the whole episode
    n_render = min(n, a.max_frames) if a.max_frames else n
    out_dir = os.path.join(seq, "preview_dual"); os.makedirs(out_dir, exist_ok=True)

    # ---- per-gripper poses (SLAM world) ----
    G = {}
    for name in a.grippers.split(","):
        m = GripperModel(MARKER_CFG, index=name)
        T_cb = ar["T_cam_body" + m.suffix][:n].copy(); src = ar["valid" + m.suffix][:n].astype(int)
        T_wb = T_wc[:n] @ T_cb
        fill_gaps(T_wb, src, a.max_gap)
        T_cb_f = np.einsum("nij,njk->nik", np.linalg.inv(T_wc[:n]), T_wb)      # gap-filled, back in the camera
        G[name] = dict(model=m, T_wb=T_wb, T_cb=T_cb_f, vis=src > 0, vis_meas=src == 1)
        print(f"{name}: pose on {100 * (src > 0).mean():.1f} % of rows ({100 * (src == 1).mean():.1f} % measured)")

    # ---- gravity-aligned table frame ----
    gpath = os.path.join(out_dir, "gravity.json")
    if os.path.isfile(gpath):
        gj = json.load(open(gpath)); up = np.array(gj["up_world"]); h = gj["plane_offset"]
        print(f"gravity from cache {gpath}")
    else:
        up, h, spread, inl, scale = estimate_gravity(frames, K, T_wc, a.frames, a.min_depth, a.max_depth, G)
        gj = dict(up_world=up.tolist(), plane_offset=h, metric3d_scale=scale, spread_deg_median=float(np.median(spread)),
                  spread_deg_max=float(spread.max()), inlier_frac=inl)
        json.dump(gj, open(gpath, "w"), indent=1)
    c0 = T_wc[0, :3, 3]; f0 = T_wc[0, :3, 2]
    # heading. The head looks nearly straight down (head->hands horizontal offset ~5 cm on the first sample), so neither
    # the optical axis nor head->hands gives a stable forward. Two grippers: y (operator's left) = the horizontal
    # chord right TCP -> left TCP (medians), x = y x up. One gripper: head -> TCP, horizontal.
    hz = lambda v: v - (v @ up) * up
    med = {nm: np.median((g["T_wb"] @ g["model"].T_body_tcp)[g["vis"], :3, 3], 0) for nm, g in G.items()}
    if "right" in med and "left" in med:
        y = hz(med["left"] - med["right"]); y /= np.linalg.norm(y); x = np.cross(y, up)
    else:
        x = hz(next(iter(med.values())) - c0); x /= np.linalg.norm(x); y = np.cross(up, x)
    R = np.stack([x, y, up])                                       # rows: table-frame axes in the SLAM world
    tcp_up = np.concatenate([(g["T_wb"] @ g["model"].T_body_tcp)[g["vis"], :3, 3] @ up for g in G.values()])
    p1 = float(np.percentile(tcp_up, 1))
    h_tab = h if a.table_from == "plane" else (p1 - 0.01 if a.table_from == "tcp" else (h if p1 - h < 0.15 else p1 - 0.01))
    o = c0 - (c0 @ up - h_tab) * up
    T_GW = np.eye(4); T_GW[:3, :3] = R; T_GW[:3, 3] = -R @ o
    pitch = np.degrees(np.arcsin(np.clip(-(f0 @ up), -1, 1)))
    gj.update(table_offset_used=h_tab, lowest_tcp_offset_p1=p1, head_height_above_table=float(c0 @ up - h_tab),
              head_pitch_down_deg=float(pitch))
    json.dump(gj, open(gpath, "w"), indent=1)
    print(f"gravity: spread median {gj['spread_deg_median']:.1f} max {gj['spread_deg_max']:.1f} deg, inliers "
          f"{gj['inlier_frac']:.2f}; plane {'used' if h_tab == h else 'REPLACED by lowest TCP'} "
          f"(plane {100 * (p1 - h):.0f} cm under the lowest TCP); head {100 * gj['head_height_above_table']:.0f} cm above "
          f"the table, pitched {pitch:.0f} deg down")

    cams = np.einsum("ij,njk->nik", T_GW, T_wc[:n]); cam_c = cams[:, :3, 3]
    for g in G.values():
        g["T_gb"] = np.einsum("ij,njk->nik", T_GW, g["T_wb"])
        if a.ema < 1:                                   # per visible run, as the dataset sees it
            v = np.r_[False, g["vis"], False]; st = np.where(~v[:-1] & v[1:])[0]; en = np.where(v[:-1] & ~v[1:])[0]
            for i0, i1 in zip(st, en):
                seg = g["T_gb"][i0:i1]
                seg[:, :3, 3] = ema(seg[:, :3, 3], a.ema); seg[:, :3, :3] = ema_rot(seg[:, :3, :3], a.ema)
        g["tcp"] = (g["T_gb"] @ g["model"].T_body_tcp)[:, :3, 3]

    allxy = np.concatenate([g["tcp"][g["vis"], :2] for g in G.values()])
    lo, hi = allxy.min(0) - 0.15, allxy.max(0) + 0.15
    tab_size = (max(hi[0] - lo[0], 0.6), max(hi[1] - lo[1], 0.5)); tab_off = tuple((lo + hi) / 2)
    np.savez_compressed(os.path.join(out_dir, "scene.npz"), fps=fps, n=n, cams=cams, table_size=tab_size,
                        table_offset=tab_off, grippers=np.array(list(G)), gravity=json.dumps(gj),
                        **{f"{k}_{nm}": g[k] for nm, g in G.items() for k in ("T_gb", "vis", "tcp")})
    print(f"scene -> {os.path.join(out_dir, 'scene.npz')}")
    if a.scene_only:
        return
    # ---- scenes ----
    pw = a.panel_width; s = pw / W; ph = int(round(H * s))
    Ks = K.copy(); Ks[:2, :2] *= s; Ks[:2, 2] = (K[:2, 2] + 0.5) * s - 0.5
    Sc = Scene3D(pw, ph, ambient=0.45); Sc.add_light((0.2, 0.4, 1.0), 1.8)
    Sw = Scene3D(pw, ph, ambient=0.35); Sw.add_light((0.3, -0.4, -1.0), 1.6); Sw.add_light((-0.5, 0.6, -0.6), 0.6)
    any_m = next(iter(G.values()))["model"]
    Sw.add_table(any_m, size=tab_size, offset=tab_off)
    for name, g in G.items():
        meshes = load_meshes(with_gopro=True, model=g["model"])
        Sc.add_gripper(name, g["model"], style="flat", rgba=COLOR[name][0], meshes=meshes)
        Sw.add_gripper(name, g["model"], style="flat", rgba=COLOR[name][0], meshes=meshes)
    Sc.set_camera(K, np.eye(4), scale=s)
    cb, cl = camera_meshes(); cam_nodes = [Sw.add_mesh(cb, (0.78, 0.80, 0.83, 1.0)), Sw.add_mesh(cl, (1.0, 0.62, 0.16, 1.0))]
    sb, sl = camera_meshes(scale=0.35); ego_nodes = [Sw.add_mesh(sb, (0.78, 0.80, 0.83, 1.0)), Sw.add_mesh(sl, (1.0, 0.62, 0.16, 1.0))]
    PARK = np.eye(4); PARK[:3, 3] = 1e3
    Kw = Ks.copy(); Kw[0, 0] = Kw[1, 1] = 0.75 * pw
    # raised front view, framing both TCP tracks and the head path
    pts = np.concatenate([np.percentile(g["tcp"][g["vis"]], [3, 97], axis=0) for g in G.values()] +
                         [np.percentile(cam_c, [3, 97], axis=0)])   # robust extent: the grippers + the head
    ctr = (pts.min(0) + pts.max(0)) / 2; ctr[2] = max(ctr[2], 0.1)
    r = max(np.linalg.norm(pts - ctr, axis=1).max(), 0.2); half = np.arctan((ph / 2) / Kw[1, 1])
    d = np.array([1.0, 0.35, 0.8]); d /= np.linalg.norm(d)          # raised front view: left/right separate
    T_side = look_at_cv(ctr + d * r / np.sin(half) * 0.95, ctr)
    fr_lines, fr_small = frustum_lines(K, W, H), frustum_lines(K, W, H, depth=0.035)

    npan = 3
    proc = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{npan * pw}x{ph}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", os.path.join(out_dir, "dual_preview.mp4")], stdin=subprocess.PIPE)
    ego_eye = ego_tgt = None
    gmid = np.array([*np.mean(allxy, 0), 0.05])          # gripper midpoint until one is seen
    for f in range(n_render):
        t = f / fps
        img = cv2.resize(cv2.imread(frames[f]), (pw, ph), interpolation=cv2.INTER_AREA)
        for name, g in G.items():
            Sc.set_gripper(name, g["T_cb"][f], 0.0, visible=bool(g["vis"][f]))
        colr, _ = Sc.render(); msk = colr[..., 3:4] > 0
        img[:] = np.where(msk, img * (1 - a.alpha) + cv2.cvtColor(colr[..., :3], cv2.COLOR_RGB2BGR) * a.alpha, img).astype(np.uint8)
        mins, secs = divmod(t, 60)
        banner(img, "RECORDING", "head camera + reconstructed grippers", right=f"{int(mins):02d}:{secs:05.2f}",
               dot=(40, 40, 235) if (f // 15) % 2 == 0 else (70, 70, 110))
        for name, g in G.items():
            Sw.set_gripper(name, g["T_gb"][f], 0.0, visible=bool(g["vis"][f]))

        def world_panel(T_wv, sub, ego=False):
            for nd in cam_nodes:
                Sw.scene.set_pose(nd, PARK if ego else cams[f])
            for nd in ego_nodes:
                Sw.scene.set_pose(nd, cams[f] if ego else PARK)
            Sw.set_camera(Kw, T_wv); T_vw = inv(T_wv)
            c, _ = Sw.render(); wv = cv2.cvtColor(c[..., :3], cv2.COLOR_RGB2BGR); wv[c[..., 3] == 0] = (38, 33, 29)
            uv, ok = project(Kw, T_vw, cam_c[:f + 1]); draw_poly(wv, uv, ok, CAM_BGR, 2)
            for name, g in G.items():
                sel = np.where(g["vis"][:f + 1])[0]
                if len(sel):
                    uv, ok = project(Kw, T_vw, g["tcp"][sel]); draw_poly(wv, uv, ok, COLOR[name][1], 2)
                if g["vis"][f] and not ego:
                    uv, ok = project(Kw, T_vw, g["tcp"][f][None])
                    if ok[0]:
                        label(wv, f"{name} {100 * g['tcp'][f][2]:.0f} cm", uv[0], COLOR[name][1])
            for seg in (fr_small if ego else fr_lines):
                uv, ok = project(Kw, T_vw, apply(cams[f], np.array(seg))); draw_poly(wv, uv, ok, CAM_BGR, 2)
            banner(wv, "RECONSTRUCTION", sub)
            legend(wv, [(COLOR["right"][1], "right gripper"), (COLOR["left"][1], "left gripper"),
                        (CAM_BGR, "head camera"), ((90, 130, 160), "table (gravity-aligned, z = 0)")])
            return wv

        side = world_panel(T_side, "front view, gravity up" + (f", EMA {a.ema:g}" if a.ema < 1 else ""))
        Tc = cams[f]; vis_now = [g["tcp"][f] for g in G.values() if g["vis"][f]]
        if vis_now:
            gmid = np.mean(vis_now, 0)
        fwd = np.array([1.0, 0, 0])                          # the episode heading (table frame +x), not per frame
        eye_t = Tc[:3, 3] - 0.45 * fwd + np.array([0, 0, 0.25]); tgt_t = gmid
        ego_eye = eye_t if ego_eye is None else ego_eye + 0.08 * (eye_t - ego_eye)
        ego_tgt = tgt_t if ego_tgt is None else ego_tgt + 0.08 * (tgt_t - ego_tgt)
        ego = world_panel(look_at_cv(ego_eye, ego_tgt), "ego view, behind the head camera", ego=True)
        proc.stdin.write(np.hstack([img, side, ego]).tobytes())
        if f % 120 == 0:
            print(f"dual preview {f + 1}/{n_render}", flush=True)
    proc.stdin.close(); proc.wait(); Sc.close(); Sw.close()
    print(f"wrote {os.path.join(out_dir, 'dual_preview.mp4')}")


if __name__ == "__main__":
    main()
