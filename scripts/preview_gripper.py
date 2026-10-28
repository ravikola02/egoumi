#!/usr/bin/env python3
"""Gripper-pose preview video (the ArUco counterpart of HaWoR's MANO cam/world visualisation).

    PYOPENGL_PLATFORM=egl ~/anaconda3/envs/hawor/bin/python scripts/preview_gripper.py --seq sim/episodes/pick_place

Reads <seq>/extracted_images, <seq>/camera.yaml, <seq>/gripper_pose/{poses.npz,detections.json} and, when present
(sim), <seq>/gt.npz. Writes <seq>/preview/gripper_preview.mp4, metrics.json and timeline.png.

Layout:  [ camera view: estimated gripper mesh over the frame, tags, TCP axes ] [ world view: est vs GT, trails ]
         [ timeline: TCP position / rotation error, jaw width est vs GT, body tags used + invalid frames      ]
Colours: purple = estimate (HaWoR's hand colour), green = ground truth, grey frames = interpolated, red = invalid.
"""
import os, sys, glob, json, argparse, subprocess
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import cv2
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from umi_pipe.geometry import inv, apply, pose_error, look_at_cv, make_T
from umi_pipe.model import GripperModel, MARKER_CFG, load_meshes
from umi_pipe.render import Scene3D

EST = (0.56, 0.36, 0.86, 1.0)          # purple
GT = (0.20, 0.85, 0.35, 0.45)          # translucent green
EST_BGR = (219, 92, 143); GT_BGR = (90, 204, 51)
FFMPEG = "/usr/bin/ffmpeg" if os.path.isfile("/usr/bin/ffmpeg") else "ffmpeg"   # snap ffmpeg can't read the SSD


def project(K, T_cam_world, pts):
    p = apply(T_cam_world, pts)
    z = p[:, 2:3]
    uv = (p @ K.T)[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    return uv, z[:, 0] > 1e-3


def draw_poly(img, uv, ok, color, thick=2, closed=False):
    pts = [tuple(np.round(p).astype(int)) for p, o in zip(uv, ok) if o]
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, np.int32)], closed, color, thick, cv2.LINE_AA)


def frustum_lines(K, w, h, depth=0.08):
    Ki = np.linalg.inv(K)
    c = [Ki @ [u, v, 1.0] * depth for u, v in [(0, 0), (w, 0), (w, h), (0, h)]]
    o = np.zeros(3)
    return [[o, c[0]], [o, c[1]], [o, c[2]], [o, c[3]], [c[0], c[1], c[2], c[3], c[0]]]


def put(img, text, org, scale=0.55, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def compute_metrics(p, g, model, n):
    m = {}
    valid = p["world_src"] > 0
    meas = p["world_src"] == 1
    e_tcp = np.array([pose_error(a, b) for a, b in zip(p["T_world_tcp"], g["T_world_tcp"][:n])])
    T_cam_body_gt = np.stack([inv(c) @ b for c, b in zip(g["T_world_cam"][:n], g["T_world_body"][:n])])
    e_cam = np.array([pose_error(a, b) for a, b in zip(p["T_cam_body"], T_cam_body_gt)])
    e_w = (p["width"] - g["width"][:n]) * 1000
    def stats(x, sel):
        x = x[sel & np.isfinite(x)]
        return None if not len(x) else {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
                                         "max": float(np.max(x)), "n": int(len(x))}
    m["coverage_measured_pct"] = float(meas.mean() * 100)
    m["coverage_valid_pct"] = float(valid.mean() * 100)
    m["width_measured_pct"] = float((p["width_src"] == 1).mean() * 100)
    m["tcp_world_mm"] = stats(e_tcp[:, 0] * 1000, valid)
    m["tcp_world_deg"] = stats(e_tcp[:, 1], valid)
    m["tcp_world_mm_measured_only"] = stats(e_tcp[:, 0] * 1000, meas)
    m["body_in_camera_mm"] = stats(e_cam[:, 0] * 1000, p["body_src"] == 1)
    m["body_in_camera_deg"] = stats(e_cam[:, 1], p["body_src"] == 1)
    m["width_abs_err_mm"] = stats(np.abs(e_w), np.isfinite(e_w))
    m["width_abs_err_mm_measured_only"] = stats(np.abs(e_w), p["width_src"] == 1)
    if "width_weak" in p:
        strong = (p["width_src"] == 1) & ~p["width_weak"]
        m["width_abs_err_mm_strong"] = stats(np.abs(e_w), strong)
        m["width_weak_pct_of_measured"] = float(p["width_weak"][p["width_src"] == 1].mean() * 100) if (p["width_src"] == 1).any() else 0.0
    m["width_valid_pct"] = float(np.isfinite(p["width"]).mean() * 100)
    m["single_tag_frames_pct"] = float(p["single"][p["body_src"] == 1].mean() * 100) if (p["body_src"] == 1).any() else 0.0
    m["cam_source"] = str(p["cam_source"])
    return m, e_tcp, e_w


def timeline(t, p, e_tcp, e_w, g, model, path, W=1920, H=360, dpi=100):
    fig, ax = plt.subplots(1, 3, figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.patch.set_facecolor("#14161a")
    for a in ax:
        a.set_facecolor("#1d2026"); a.tick_params(colors="#c9ced6", labelsize=8)
        for s in a.spines.values():
            s.set_color("#3a404a")
        a.grid(alpha=0.2)
    src = p["world_src"]
    for a in ax:
        for k in np.where(src == 0)[0]:
            a.axvspan(t[k] - 0.5 / 30, t[k] + 0.5 / 30, color="#ff6b6b", alpha=0.25, lw=0)
        for k in np.where(src == 2)[0]:
            a.axvspan(t[k] - 0.5 / 30, t[k] + 0.5 / 30, color="#9aa3af", alpha=0.25, lw=0)
    if e_tcp is not None:
        ax[0].plot(t, e_tcp[:, 0] * 1000, color="#c48cff", lw=1.3, label="TCP pos err [mm]")
        a2 = ax[0].twinx(); a2.plot(t, e_tcp[:, 1], color="#ffb454", lw=1.0, label="rot err [deg]")
        a2.tick_params(colors="#ffb454", labelsize=8)
        ax[0].set_title("TCP error (purple mm, orange deg)", color="#e8eaed", fontsize=9)
    else:
        ax[0].plot(t, p["rms"], color="#c48cff", lw=1.3); ax[0].set_title("body reprojection RMS [px]", color="#e8eaed", fontsize=9)
    ax[1].plot(t, p["width"] * 1000, color="#c48cff", lw=1.5, label="est")
    if g is not None:
        ax[1].plot(t, g["width"][:len(t)] * 1000, color="#33cc5a", lw=1.2, ls="--", label="GT")
    ws = p["width_src"]
    ax[1].scatter(t[ws == 1], p["width"][ws == 1] * 1000, s=6, color="#c48cff", zorder=3)
    ax[1].set_title("jaw width [mm] (dots = jaw tag seen)", color="#e8eaed", fontsize=9)
    ax[1].legend(fontsize=7, facecolor="#1d2026", labelcolor="#e8eaed", loc="lower left")
    ax[2].step(t, p["n_tags"], where="mid", color="#4c8dff", lw=1.3)
    ax[2].set_title("body tags used (red = invalid, grey = interpolated)", color="#e8eaed", fontsize=9)
    ax[2].set_ylim(-0.2, len(model.body_tags) + 0.5)
    for a in ax:
        a.set_xlim(t[0], t[-1]); a.set_xlabel("t [s]", color="#c9ced6", fontsize=8)
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    # pixel x-range of each axis' data area for the cursor
    xr = []
    for a in ax:
        bb = a.get_window_extent()
        xr.append((bb.x0, bb.x1, H - bb.y1, H - bb.y0))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), xr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--markers", default=MARKER_CFG)
    ap.add_argument("--out", default=None, help="default <seq>/preview/gripper_preview.mp4")
    ap.add_argument("--panel-width", type=int, default=960)
    ap.add_argument("--alpha", type=float, default=0.55, help="overlay opacity of the estimated mesh")
    ap.add_argument("--view", default=None, help="world-view eye 'x,y,z' [m]; default auto from the TCP track")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=float, default=None)
    a = ap.parse_args()

    model = GripperModel(a.markers)
    cam = yaml.safe_load(open(os.path.join(a.seq, "camera.yaml")))
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    W, H = cam["width"], cam["height"]
    pw = a.panel_width; s = pw / W; ph = int(round(H * s))
    Ks = K.copy(); Ks[:2, :2] *= s; Ks[:2, 2] = (K[:2, 2] + 0.5) * s - 0.5     # OpenCV K at panel resolution
    frames = sorted(glob.glob(os.path.join(a.seq, "extracted_images", "*.jpg")))
    pz = np.load(os.path.join(a.seq, "gripper_pose", "poses.npz"))
    p = {k: pz[k] for k in pz.files}
    dets = json.load(open(os.path.join(a.seq, "gripper_pose", "detections.json")))
    n = min(len(frames), len(p["T_world_tcp"]))
    if a.max_frames:
        n = min(n, a.max_frames)
    p = {k: (v[:n] if isinstance(v, np.ndarray) and v.ndim and len(v) >= n and k not in ("K", "dist") else v) for k, v in p.items()}
    gpath = os.path.join(a.seq, "gt.npz")
    g = dict(np.load(gpath)) if os.path.isfile(gpath) else None
    fps = a.fps or cam.get("fps", 30)
    t = np.arange(n) / fps

    out_dir = os.path.join(a.seq, "preview"); os.makedirs(out_dir, exist_ok=True)
    out = a.out or os.path.join(out_dir, "gripper_preview.mp4")
    metrics, e_tcp, e_w = (compute_metrics(p, g, model, n) if g is not None else ({}, None, None))
    if metrics:
        with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=1)
        print(json.dumps(metrics, indent=1))
    tl, xr = timeline(t, p, e_tcp, e_w, g, model, os.path.join(out_dir, "timeline.png"), W=2 * pw)
    tl = cv2.resize(tl, (2 * pw, tl.shape[0]))

    meshes = load_meshes(with_gopro=True)
    # camera view: world = camera frame, gripper at T_cam_body
    Sc = Scene3D(pw, ph, ambient=0.45); Sc.add_light((0.2, 0.4, 1.0), 1.8)
    Sc.add_gripper("est", model, style="flat", rgba=EST, meshes=meshes)
    Sc.set_camera(K, np.eye(4), scale=s)
    # world view
    Sw = Scene3D(pw, ph, ambient=0.35); Sw.add_light((0.3, -0.4, -1.0), 1.6); Sw.add_light((-0.5, 0.6, -0.6), 0.6)
    if p["cam_source"] != "none":
        Sw.add_table(model)
    if g is not None:
        Sw.add_gripper("gt", model, style="flat", rgba=GT, meshes=meshes)
    Sw.add_gripper("est", model, style="flat", rgba=EST, meshes=meshes)
    obj = None
    if g is not None and "T_world_object" in g:
        import trimesh
        obj = Sw.add_mesh(trimesh.creation.box(extents=[0.04] * 3), (0.75, 0.15, 0.12, 1.0))
    tcp_e = p["T_world_tcp"][:, :3, 3]
    valid = p["world_src"] > 0
    ctr = np.median(tcp_e[valid], axis=0) if valid.any() else np.zeros(3)
    eye = np.array([float(v) for v in a.view.split(",")]) if a.view else ctr + [0.45, -0.75, 0.55]
    T_wv = look_at_cv(eye, ctr)
    Kw = Ks.copy(); Kw[0, 0] = Kw[1, 1] = 0.75 * pw
    Sw.set_camera(Kw, T_wv)   # Kw is already at panel resolution
    T_vw = inv(T_wv)
    fr_lines = frustum_lines(K, W, H)

    proc = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                             f"{2 * pw}x{ph + tl.shape[0]}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                            stdin=subprocess.PIPE)
    tag_ids_body = {tt.id for tt in model.body_tags}; tag_ids_jaw = {tt.id for tt in model.jaw_tags}
    for f in range(n):
        img = cv2.resize(cv2.imread(frames[f]), (pw, ph), interpolation=cv2.INTER_AREA)
        tr = model.travel_from_width(p["width"][f]) if np.isfinite(p["width"][f]) else 0.0
        src = int(p["world_src"][f]); body_ok = p["body_src"][f] > 0 or src > 0
        # ---- camera view ----
        if body_ok:
            Sc.set_gripper("est", p["T_cam_body"][f], tr)
            col, dep = Sc.render()
            m = (dep > 0)[..., None]
            img = np.where(m, (img * (1 - a.alpha) + cv2.cvtColor(col[..., :3], cv2.COLOR_RGB2BGR) * a.alpha), img).astype(np.uint8)
        d = dets[f]
        used = set(d.get("used", [])); rej = set(d.get("rejected", []))
        for k, c in d["tags"].items():
            k = int(k); c = (np.array(c) + 0.5) * s - 0.5
            colr = (0, 0, 255) if k in rej else (80, 220, 80) if k in used else (0, 215, 255) if k in tag_ids_jaw \
                else (255, 220, 0) if k == model.table_tag.id else (160, 160, 160)
            cv2.polylines(img, [np.round(c).astype(np.int32)], True, colr, 2, cv2.LINE_AA)
            cv2.circle(img, tuple(np.round(c[0]).astype(int)), 3, colr, -1, cv2.LINE_AA)
            put(img, str(k), tuple(np.round(c.mean(0) + [-6, 5]).astype(int)), 0.45, colr)
        if body_ok:
            T_ct = p["T_cam_body"][f] @ model.T_body_tcp
            ax_pts = apply(T_ct, np.array([[0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03]]))
            uv, ok = project(Ks, np.eye(4), ax_pts)
            if ok.all():
                for j, bgr in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                    cv2.line(img, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[j]).astype(int)), bgr, 2, cv2.LINE_AA)
        if g is not None:
            T_cw_gt = inv(g["T_world_cam"][f])
            uv, ok = project(Ks, T_cw_gt, g["T_world_tcp"][f][None, :3, 3])
            if ok[0]:
                cv2.drawMarker(img, tuple(np.round(uv[0]).astype(int)), GT_BGR, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        srcname = {0: "INVALID", 1: "measured", 2: "interpolated"}[src]
        put(img, f"frame {f + 1}/{n}  t={t[f]:.2f}s  pose: {srcname}", (10, 22),
            color=(255, 255, 255) if src == 1 else (180, 180, 180) if src == 2 else (90, 90, 255))
        line2 = f"body tags {sorted(used)}  rms {p['rms'][f]:.2f}px" if p["body_src"][f] else "no body tags"
        if p["single"][f]:
            line2 += "  SINGLE-TAG" + ("" if p["confident"][f] else " (ambiguous)")
        put(img, line2, (10, 44))
        wtxt = f"width {p['width'][f] * 1000:.1f} mm" if np.isfinite(p["width"][f]) else "width n/a"
        wtxt += (" (jaw tag, WEAK: 1 body tag)" if p.get("width_weak", np.zeros(n, bool))[f] else " (jaw tag)") if p["width_src"][f] == 1 else " (interp)" if p["width_src"][f] == 2 else ""
        if g is not None:
            wtxt += f"  GT {g['width'][f] * 1000:.1f}"
            if src:
                wtxt = f"err {e_tcp[f, 0] * 1000:.1f} mm {e_tcp[f, 1]:.2f} deg  |  " + wtxt
        put(img, wtxt, (10, 66))
        put(img, "camera view: estimate (purple)  tags: green used / yellow jaw / cyan table / red rejected",
            (10, ph - 12), 0.45, (200, 200, 200))

        # ---- world view ----
        if g is not None:
            Sw.set_gripper("gt", g["T_world_body"][f], g["travel"][f])
            if obj is not None:
                Sw.scene.set_pose(obj, g["T_world_object"][f])
        Sw.set_gripper("est", p["T_world_body"][f], tr, visible=src > 0)
        col, _ = Sw.render()
        wv = cv2.cvtColor(col[..., :3], cv2.COLOR_RGB2BGR)
        bg = (col[..., 3] == 0)
        wv[bg] = (38, 33, 29)
        if g is not None:
            uv, ok = project(Kw, T_vw, g["T_world_tcp"][:f + 1, :3, 3]); draw_poly(wv, uv, ok, GT_BGR, 2)
        sel = np.where(valid[:f + 1])[0]
        if len(sel):
            uv, ok = project(Kw, T_vw, tcp_e[sel]); draw_poly(wv, uv, ok, EST_BGR, 2)
        for T_wc, colr in ([(g["T_world_cam"][f], GT_BGR)] if g is not None else []) + \
                          ([(p["T_world_cam"][f], EST_BGR)] if p["cam_source"] == "table" or g is None else []):
            for seg in fr_lines:
                uv, ok = project(Kw, T_vw, apply(T_wc, np.array(seg))); draw_poly(wv, uv, ok, colr, 2)
        put(wv, "world view: estimate (purple) vs GT (green ghost), TCP trails, head camera", (10, 22))
        put(wv, f"cam source: {p['cam_source']}", (10, 44))
        if metrics:
            mm = metrics["tcp_world_mm"]; dg = metrics["tcp_world_deg"]
            if mm:
                put(wv, f"episode: TCP median {mm['median']:.1f} mm / p90 {mm['p90']:.1f} mm, "
                        f"{dg['median']:.2f} deg; valid {metrics['coverage_valid_pct']:.0f} %", (10, 66))

        # ---- timeline cursor ----
        tl_f = tl.copy()
        u = (t[f] - t[0]) / max(t[-1] - t[0], 1e-9)
        for (x0, x1, y0, y1) in xr:
            x = int(round((x0 + u * (x1 - x0)) * tl_f.shape[1] / (2 * pw)))
            cv2.line(tl_f, (x, int(y0)), (x, int(y1)), (255, 255, 255), 1, cv2.LINE_AA)
        frame = np.vstack([np.hstack([img, wv]), tl_f])
        proc.stdin.write(frame.tobytes())
        if f % 60 == 0:
            print(f"preview {f + 1}/{n}", flush=True)
    proc.stdin.close(); proc.wait()
    Sc.close(); Sw.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
