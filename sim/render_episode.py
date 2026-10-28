#!/usr/bin/env python3
"""Render a synthetic head-camera episode of the UMI pinch v3 gripper with ArUco markers.

    PYOPENGL_PLATFORM=egl ~/anaconda3/envs/hawor/bin/python sim/render_episode.py --out sim/episodes/pick_place

Writes, in the HaWoR sequence layout so the real pipeline can run on it unchanged:
    <out>/extracted_images/0001.jpg ...   head-camera frames (JPEG q2, like the ffmpeg extraction)
    <out>/camera.yaml                      intrinsics (pinhole, principal point at the centre)
    <out>/gt.npz                           ground truth: T_world_cam, T_world_body, T_world_tcp, width, travel,
                                           T_world_object, grasped, t
    <out>/gt_masks.npz (--save-masks)      packed bool (T,H,W) gripper+hand silhouette = what model_masks.npy should be

World frame = table tag (configs/gripper_markers.yaml table_tag). Trajectory: a pick-and-place of a 40 mm cube
from keyframes (preset below) + head bob/turn + hand tremor; the gripper closes to the cube width.
"""
import os, sys, argparse, shutil, yaml
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import cv2
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R, Slerp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from umi_pipe.geometry import make_T, inv, frame_from_axes, look_at_cv
from umi_pipe.model import GripperModel, ROOT
from umi_pipe.render import Scene3D

OBJ = 0.040   # cube edge

# Keyframes for the TCP: (t [s], position [m], finger dir (body +Y), cover-up dir (body +Z), width [m]).
# Positions in the table-tag world frame; the cube starts at PICK and ends at PLACE.
PICK, PLACE = np.array([0.10, 0.18, OBJ / 2]), np.array([-0.12, 0.20, OBJ / 2])
PRESETS = {
    "pick_place": [
        (0.0, [0.14, -0.05, 0.20], [0, 1, -0.25], [0, 0.25, 1], 0.072),
        (1.4, PICK + [0, -0.06, 0.08], [0, 1, -0.9], [0, 0.9, 1], 0.072),
        (2.3, PICK + [0, -0.005, 0.0], [0, 1, -1.0], [0, 1.0, 1], 0.072),
        (2.9, PICK + [0, -0.005, 0.0], [0, 1, -1.0], [0, 1.0, 1], OBJ),
        (3.8, (PICK + PLACE) / 2 + [0, -0.02, 0.14], [0.2, 1, -0.6], [0, 0.6, 1], OBJ),
        (4.8, PLACE + [0, -0.005, 0.0], [0, 1, -1.0], [0, 1.0, 1], OBJ),
        (5.3, PLACE + [0, -0.005, 0.0], [0, 1, -1.0], [0, 1.0, 1], 0.072),
        (6.5, [-0.05, -0.02, 0.22], [0, 1, -0.2], [0, 0.2, 1], 0.072),
    ],
    # wrist rolled +-35 deg while carrying: tests the side (cube left/right) tags and single-tag frames
    "roll": [
        (0.0, [0.10, 0.00, 0.18], [0, 1, -0.3], [0, 0.3, 1], 0.072),
        (1.5, [0.05, 0.10, 0.15], [0, 1, -0.3], [0.6, 0.3, 1], 0.050),
        (3.0, [-0.05, 0.12, 0.15], [0, 1, -0.3], [-0.6, 0.3, 1], 0.030),
        (4.5, [0.00, 0.05, 0.20], [0, 1, -0.6], [0, 0.6, 1], 0.072),
    ],
}


def interp_keyframes(keys, t):
    kt = np.array([k[0] for k in keys])
    pos = PchipInterpolator(kt, np.array([k[1] for k in keys], float))(t)
    rots = R.from_matrix(np.stack([frame_from_axes(k[2], k[3]) for k in keys]))
    rot = Slerp(kt, rots)(np.clip(t, kt[0], kt[-1]))
    width = np.interp(t, kt, [k[4] for k in keys])
    return pos, rot, width


def smooth_noise(rng, n, fps, amp, freqs=(0.3, 0.7, 1.3)):
    """Sum of random-phase sinusoids, (n, 3), amplitude ~amp per axis."""
    t = np.arange(n) / fps
    out = np.zeros((n, 3))
    for f in freqs:
        ph = rng.uniform(0, 2 * np.pi, 3)
        out += np.sin(2 * np.pi * f * t[:, None] * rng.uniform(0.8, 1.2, 3) + ph)
    return out * amp / len(freqs) * 1.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="pick_place", choices=sorted(PRESETS))
    ap.add_argument("--camera", default=os.path.join(ROOT, "configs", "camera_sim.yaml"))
    ap.add_argument("--markers", default=os.path.join(ROOT, "configs", "gripper_markers.yaml"))
    ap.add_argument("--head", default="0.0,-0.42,0.58", help="nominal head-camera position in the world frame [m]")
    ap.add_argument("--look", default="0.0,0.12,0.0", help="nominal look-at point [m]")
    ap.add_argument("--head-bob", type=float, default=0.02, help="head translation wobble amplitude [m]")
    ap.add_argument("--head-turn", type=float, default=4.0, help="head rotation wobble amplitude [deg]")
    ap.add_argument("--tremor", type=float, default=0.002, help="hand position tremor amplitude [m]")
    ap.add_argument("--blur", type=int, default=1, help="motion-blur sub-frames per frame (1 = none; ~1/fps exposure)")
    ap.add_argument("--exposure", type=float, default=0.25, help="blur shutter as a fraction of the frame time")
    ap.add_argument("--noise", type=float, default=2.0, help="Gaussian pixel noise sigma [0-255]")
    ap.add_argument("--supersample", type=int, default=2, help="render at Nx and downsample (anti-aliasing)")
    ap.add_argument("--no-hand", action="store_true")
    ap.add_argument("--no-gopro", action="store_true")
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    cam = yaml.safe_load(open(a.camera))
    W, H, fps = cam["width"], cam["height"], cam["fps"]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    ss = max(1, a.supersample)
    model = GripperModel(a.markers)

    keys = PRESETS[a.preset]
    n = int(round(keys[-1][0] * fps)) + 1
    if a.max_frames:
        n = min(n, a.max_frames)

    # sub-frame times for blur: centred on each frame time
    sub = np.linspace(-0.5, 0.5, a.blur) * a.exposure / fps if a.blur > 1 else np.zeros(1)
    t_all = (np.arange(n)[:, None] / fps + sub[None]).ravel()

    head = np.array([float(v) for v in a.head.split(",")]); look = np.array([float(v) for v in a.look.split(",")])
    # wobble evaluated on the fine time grid by interpolating a per-frame noise track
    t_frames = np.arange(n + 1) / fps
    bob = smooth_noise(rng, n + 1, fps, a.head_bob)
    turn = smooth_noise(rng, n + 1, fps, np.radians(a.head_turn))
    trem = smooth_noise(rng, n + 1, fps, a.tremor, freqs=(2.0, 3.1, 4.7))
    lerp = lambda X, tt: np.stack([np.interp(tt, t_frames, X[:, i]) for i in range(3)], 1)

    def cam_pose(tt):
        T = look_at_cv(head + lerp(bob, [tt])[0], look)
        return T @ make_T(R.from_rotvec(lerp(turn, [tt])[0]).as_matrix())

    pos, rot, width = interp_keyframes(keys, t_all)
    pos = pos + lerp(trem, t_all)
    T_world_tcp = np.stack([make_T(rot[i].as_matrix(), pos[i]) for i in range(len(t_all))])
    T_tcp_body = inv(model.T_body_tcp)
    T_world_body = T_world_tcp @ T_tcp_body
    travel = model.travel_from_width(width)

    # object: rests at PICK, rides with the TCP while the jaws are at the cube width, rests at PLACE after release
    kt = np.array([k[0] for k in keys]); kw = np.array([k[4] for k in keys])
    closed = np.where(kw <= OBJ + 1e-6)[0]
    grasped = np.zeros(len(t_all), bool)
    if len(closed):
        t0, t1 = kt[closed[0]], kt[closed[-1]]
        grasped = (t_all >= t0) & (t_all <= t1)
    T_world_obj = np.zeros((len(t_all), 4, 4))
    T_tcp_obj = None; last = make_T(t=PICK)
    for i in range(len(t_all)):
        if grasped[i]:
            if T_tcp_obj is None:
                T_tcp_obj = inv(T_world_tcp[i]) @ last
            last = T_world_tcp[i] @ T_tcp_obj
        elif T_tcp_obj is not None:       # released: drop to the table, keep yaw
            yaw = np.arctan2(last[1, 0], last[0, 0])
            last = make_T(R.from_euler("z", yaw).as_matrix(), [last[0, 3], last[1, 3], OBJ / 2])
            T_tcp_obj = None
        T_world_obj[i] = last

    # ---- scene ----
    import trimesh
    S = Scene3D(W * ss, H * ss)
    S.add_light((0.3, -0.4, -1.0), 3.0); S.add_light((-0.5, 0.6, -0.6), 1.0)
    S.add_table(model)
    floor = trimesh.creation.box(extents=[4, 4, 0.01]); floor.apply_translation([0, 0, -0.75])
    S.add_mesh(floor, (0.45, 0.43, 0.40, 1.0))
    wall = trimesh.creation.box(extents=[4, 0.01, 3]); wall.apply_translation([0, 1.2, 0.5])
    S.add_mesh(wall, (0.62, 0.64, 0.66, 1.0))
    obj_node = S.add_mesh(trimesh.creation.box(extents=[OBJ] * 3), (0.75, 0.15, 0.12, 1.0))
    g = S.add_gripper("g", model, with_gopro=not a.no_gopro, with_hand=not a.no_hand)

    out_img = os.path.join(a.out, "extracted_images")
    if os.path.isdir(out_img):
        shutil.rmtree(out_img)
    os.makedirs(out_img)
    seg_nodes = {nd: (255, 255, 255) for nd in list(g["nodes"].values()) + [v[0] for v in g["tags"].values()]
                 + list(g["hand"].values())}
    masks = []
    T_world_cam = np.zeros((n, 4, 4))
    nb = len(sub)
    for f in range(n):
        acc = np.zeros((H * ss, W * ss, 3), np.float64)
        for j in range(nb):
            i = f * nb + j
            S.set_gripper("g", T_world_body[i], travel[i])
            S.scene.set_pose(obj_node, T_world_obj[i])
            S.set_camera(K, cam_pose(t_all[i]), scale=ss)
            col, _ = S.render()
            acc += col[..., :3]
        mid = f * nb + nb // 2
        T_world_cam[f] = cam_pose(t_all[mid])
        img = acc / nb
        if ss > 1:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        img = np.clip(img + rng.normal(0, a.noise, img.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_img, f"{f + 1:04d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        if a.save_masks:
            S.set_gripper("g", T_world_body[mid], travel[mid]); S.set_camera(K, T_world_cam[f], scale=ss)
            seg = S.render_seg(seg_nodes)[..., 0] > 0
            masks.append(cv2.resize(seg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool))
        if f % 30 == 0:
            print(f"frame {f + 1}/{n}", flush=True)
    S.close()

    mids = np.arange(n) * nb + nb // 2
    with open(os.path.join(a.out, "camera.yaml"), "w") as fh:
        yaml.safe_dump(cam, fh, sort_keys=False)
    np.savez_compressed(os.path.join(a.out, "gt.npz"), t=np.arange(n) / fps, T_world_cam=T_world_cam,
                        T_world_body=T_world_body[mids], T_world_tcp=T_world_tcp[mids], width=width[mids],
                        travel=travel[mids], T_world_object=T_world_obj[mids], grasped=grasped[mids],
                        preset=a.preset, K=K)
    if a.save_masks:
        m = np.stack(masks)
        np.savez_compressed(os.path.join(a.out, "gt_masks.npz"), packed=np.packbits(m, axis=-1), shape=m.shape)
    print(f"wrote {n} frames to {out_img}")


if __name__ == "__main__":
    main()
