#!/usr/bin/env python3
"""Render a dummy raw capture session for the collection pipeline: head camera + right gripper (wrist) camera
+ a blank left wrist camera, as three independently started recordings with skewed clocks and shared room audio.

    PYOPENGL_PLATFORM=egl ~/anaconda3/envs/hawor/bin/python sim/render_capture.py --out data/raw/sim_capture

Writes the collection/ input layout plus ground truth:
    <out>/top_cam/head.mp4              head camera (pyrender), 30 fps, AAC audio
    <out>/right_w_cam/wrist_right.mp4   GoPro on the gripper, looking along the fingers (pyrender, pinhole)
    <out>/left_w_cam/wrist_left.mp4     blank (black) feed: the other hand's camera, audio only
    <out>/sim_gt/gt.json                per-camera start times, clock skew, true sync offsets, episode bounds
    <out>/sim_gt/gt_<camera>.npz        per-frame camera / gripper / TCP poses and jaw width
    <out>/sim_gt/clips_gt.json          true episode boundaries as a cut sheet (head timeline)
    <out>/sim_gt/cameras.yaml           intrinsics of both rendered cameras (+ wrist camera mount)

Episodes run back to back with rests between them (--episodes, default pick_place,roll,place_back). Each camera
starts recording at its own time (--starts) and stamps creation_time with a clock error (--clock-skew), so the
pipeline's audio sync has to recover the true offsets from the claps in the shared soundtrack.
"""
import os, sys, json, argparse, subprocess, shutil, wave
from datetime import datetime, timedelta, timezone
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import cv2
import yaml
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R, Slerp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from umi_pipe.geometry import make_T, inv, frame_from_axes, look_at_cv
from umi_pipe.model import GripperModel, ROOT
from umi_pipe.render import Scene3D
from render_episode import PRESETS, PICK, PLACE, OBJ, smooth_noise   # noqa: E402

FFMPEG = "/usr/bin/ffmpeg" if os.path.isfile("/usr/bin/ffmpeg") else "ffmpeg"
AUDIO_SR = 48000
# GoPro dummy lens: front face of the lens housing at y = 14.5 mm, centred x 0, z ~83.4 mm (camera_dummy.stl).
# Camera sits 1.5 mm in front of it looking along +Y (fingers); image right = body +X, image down = body -Z.
T_BODY_WRISTCAM = make_T(np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]), [0.0, 0.016, 0.0834])


def place_back_keys():
    """pick_place with the pick and place spots swapped (puts the cube back)."""
    d = np.r_[(PLACE - PICK)[:2], 0.0]
    out = []
    for t, p, dr, up, w in PRESETS["pick_place"]:
        p = np.asarray(p, float)
        if np.linalg.norm(p[:2] - PICK[:2]) < 0.09:
            p = p + d
        elif np.linalg.norm(p[:2] - PLACE[:2]) < 0.09:
            p = p - d
        out.append((t, p, dr, up, w))
    return out


EPISODE_KEYS = {**PRESETS, "place_back": place_back_keys()}
EPISODE_TASKS = {"pick_place": "move the red cube to the left", "roll": "roll the wrist",
                 "place_back": "move the red cube back to the right"}


def build_timeline(names, rest, lead):
    """Global TCP keyframes on the head-camera clock (t = 0 at head record start) + episode bounds."""
    keys, bounds, t0 = [], [], lead
    for i, nm in enumerate(names):
        ek = EPISODE_KEYS[nm]
        if i == 0:
            keys.append((0.0,) + tuple(ek[0][1:]))           # hold the first pose during the lead-in
        for k in ek:
            keys.append((t0 + k[0],) + tuple(k[1:]))
        bounds.append((nm, t0, t0 + ek[-1][0]))
        t0 += ek[-1][0] + rest
    keys.append((t0,) + tuple(keys[-1][1:]))                  # hold the last pose through the final rest
    return keys, bounds, t0


class Trajectory:
    def __init__(self, keys, tremor, rng, fps):
        kt = np.array([k[0] for k in keys])
        self.t0, self.t1 = kt[0], kt[-1]
        self.pos = PchipInterpolator(kt, np.array([np.asarray(k[1], float) for k in keys]))
        self.rot = Slerp(kt, R.from_matrix(np.stack([frame_from_axes(k[2], k[3]) for k in keys])))
        self.kt, self.kw = kt, np.array([k[4] for k in keys])
        n = int(np.ceil((self.t1 + 6) * fps)) + 1
        self.tg = np.arange(n) / fps - 3.0
        self.trem = smooth_noise(rng, n, fps, tremor, freqs=(2.0, 3.1, 4.7))

    def tcp(self, t):
        tc = np.clip(t, self.t0, self.t1)
        p = self.pos(tc) + np.array([np.interp(t, self.tg, self.trem[:, i]) for i in range(3)])
        return make_T(self.rot(tc).as_matrix(), p)

    def width(self, t):
        return float(np.interp(t, self.kt, self.kw))


class HeadCam:
    def __init__(self, head, look, bob, turn_deg, rng, fps, span):
        n = int(np.ceil(span * fps)) + 1
        self.tg = np.arange(n) / fps - 3.0
        self.bob = smooth_noise(rng, n, fps, bob)
        self.turn = smooth_noise(rng, n, fps, np.radians(turn_deg))
        self.head, self.look = np.asarray(head, float), np.asarray(look, float)

    def pose(self, t):
        b = np.array([np.interp(t, self.tg, self.bob[:, i]) for i in range(3)])
        r = np.array([np.interp(t, self.tg, self.turn[:, i]) for i in range(3)])
        return look_at_cv(self.head + b, self.look) @ make_T(R.from_rotvec(r).as_matrix())


class ObjectTrack:
    """Cube rests on the table; attaches to the TCP when the jaws close on it near the TCP, drops on release."""
    def __init__(self, start):
        self.T, self.T_tcp_obj = make_T(t=start), None

    def update(self, T_tcp, width):
        closed = width <= OBJ + 1e-6
        if closed and self.T_tcp_obj is None and np.linalg.norm(T_tcp[:3, 3] - self.T[:3, 3]) < 0.02:
            self.T_tcp_obj = inv(T_tcp) @ self.T
        if self.T_tcp_obj is not None:
            if closed:
                self.T = T_tcp @ self.T_tcp_obj
            else:
                yaw = np.arctan2(self.T[1, 0], self.T[0, 0])
                self.T = make_T(R.from_euler("z", yaw).as_matrix(), [self.T[0, 3], self.T[1, 3], OBJ / 2])
                self.T_tcp_obj = None
        return self.T


def room_audio(rng, span, claps):
    """Shared soundtrack on the absolute clock: low ambient noise + sharp claps/knocks."""
    n = int(span * AUDIO_SR)
    x = np.cumsum(rng.normal(0, 1, n)); x -= np.convolve(x, np.ones(4801) / 4801, mode="same")
    x = 0.01 * x / (np.abs(x).max() + 1e-9)
    for tc in claps:
        i = int(tc * AUDIO_SR); L = int(0.04 * AUDIO_SR)
        if 0 <= i < n - L:
            x[i:i + L] += 0.6 * rng.normal(0, 1, L) * np.exp(-np.arange(L) / (0.006 * AUDIO_SR))
    return x


def write_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(AUDIO_SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def open_writer(path, W, H, fps, wav, creation_time):
    return subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                             "-r", str(fps), "-i", "-", "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                             "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                             "-shortest", "-metadata", f"creation_time={creation_time}", path],
                            stdin=subprocess.PIPE)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond:06d}Z"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "raw", "sim_capture"))
    ap.add_argument("--episodes", default="pick_place,roll,place_back")
    ap.add_argument("--rest", type=float, default=1.5, help="seconds between episodes")
    ap.add_argument("--lead", type=float, default=2.0, help="head-clock time before the first episode")
    ap.add_argument("--starts", default="0.0,0.817,-0.433",
                    help="record start of head,wrist_right,wrist_left on the absolute clock [s]")
    ap.add_argument("--tails", default="0.0,0.6,0.9", help="extra seconds each camera keeps recording after the head stops")
    ap.add_argument("--clock-skew", default="0.0,0.35,-0.6", help="creation_time error per camera [s]")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--head-f", type=float, default=1050.0, help="head focal [px]")
    ap.add_argument("--wrist-hfov", type=float, default=120.0, help="wrist camera horizontal FOV [deg] (pinhole)")
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--noise", type=float, default=2.0)
    ap.add_argument("--max-seconds", type=float, default=0, help="truncate the head recording (quick test)")
    ap.add_argument("--head", default="0.18,0.10,0.74",
                    help="nominal head-camera position [m]; default = overhead-right view for jaw-tag visibility "
                         "(original view: 0.0,-0.42,0.58)")
    ap.add_argument("--look", default="0.0,0.10,0.08", help="nominal look-at point [m] (original: 0.0,0.12,0.0)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    W, H = (int(v) for v in a.size.split("x")); ss = a.supersample
    K_head = np.array([[a.head_f, 0, (W - 1) / 2], [0, a.head_f, (H - 1) / 2], [0, 0, 1.0]])
    fw = (W / 2) / np.tan(np.radians(a.wrist_hfov) / 2)
    K_wrist = np.array([[fw, 0, (W - 1) / 2], [0, fw, (H - 1) / 2], [0, 0, 1.0]])
    starts = [float(v) for v in a.starts.split(",")]; tails = [float(v) for v in a.tails.split(",")]
    skew = [float(v) for v in a.clock_skew.split(",")]
    model = GripperModel()

    names = a.episodes.split(",")
    keys, bounds, head_dur = build_timeline(names, a.rest, a.lead)
    if a.max_seconds:
        head_dur = min(head_dur, a.max_seconds)
        bounds = [b for b in bounds if b[2] <= head_dur]
    roles = ["head", "wrist_right", "wrist_left"]
    # every camera stops `tail` after the head stops; it started at its own time
    durs = {r: head_dur + tails[i] - (starts[i] - starts[0]) for i, r in enumerate(roles)}
    clock0 = min(starts)
    span = max(starts[i] + durs[r] for i, r in enumerate(roles)) - clock0 + 1

    traj = Trajectory(keys, 0.002, rng, a.fps)
    headcam = HeadCam([float(v) for v in a.head.split(",")], [float(v) for v in a.look.split(",")],
                      0.02, 4.0, rng, a.fps, span + 6)
    T0 = datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc)

    # ---- audio: one absolute-clock soundtrack, each camera hears its own slice + its own mic noise ----
    claps = [starts[0] + b[1] - 0.6 for b in bounds] + [starts[0] + 0.4, starts[0] + head_dur - 0.5]
    shared = room_audio(rng, span, [c - clock0 for c in claps])
    gt_dir = os.path.join(a.out, "sim_gt"); os.makedirs(gt_dir, exist_ok=True)
    wavs = {}
    for i, r in enumerate(roles):
        i0 = int(round((starts[i] - clock0) * AUDIO_SR)); n = int(durs[r] * AUDIO_SR)
        x = shared[i0:i0 + n] * (0.8 + 0.4 * rng.random()) + rng.normal(0, 0.003, n)
        wavs[r] = os.path.join(gt_dir, f"_{r}.wav"); write_wav(wavs[r], x)

    # ---- scene ----
    import trimesh
    S = Scene3D(W * ss, H * ss, bg=(0.58, 0.59, 0.61, 1.0))   # room grey beyond the wall/floor
    S.add_light((0.3, -0.4, -1.0), 3.0); S.add_light((-0.5, 0.6, -0.6), 1.0)
    S.add_table(model)
    floor = trimesh.creation.box(extents=[4, 4, 0.01]); floor.apply_translation([0, 0, -0.75])
    S.add_mesh(floor, (0.45, 0.43, 0.40, 1.0))
    wall = trimesh.creation.box(extents=[4, 0.01, 3]); wall.apply_translation([0, 1.2, 0.5])
    S.add_mesh(wall, (0.62, 0.64, 0.66, 1.0))
    obj_node = S.add_mesh(trimesh.creation.box(extents=[OBJ] * 3), (0.75, 0.15, 0.12, 1.0))
    S.add_gripper("g", model, with_hand=True)
    T_tcp_body = inv(model.T_body_tcp)

    # the cube's state must evolve on one clock: precompute on a fine grid of absolute times
    t_grid = np.arange(clock0 - 1, clock0 + span, 1.0 / (4 * a.fps))
    obj = ObjectTrack(PICK)
    T_obj_grid = np.stack([obj.update(traj.tcp(t - starts[0]), traj.width(t - starts[0])).copy() for t in t_grid])

    def scene_at(t_abs):
        th = t_abs - starts[0]
        T_tcp = traj.tcp(th); w = traj.width(th)
        T_body = T_tcp @ T_tcp_body
        S.set_gripper("g", T_body, model.travel_from_width(w))
        S.scene.set_pose(obj_node, T_obj_grid[int(np.clip(np.searchsorted(t_grid, t_abs), 0, len(t_grid) - 1))])
        return T_body, T_tcp, w

    def render(K, T_wc, znear):
        S.set_camera(K, T_wc, scale=ss, znear=znear)
        col, _ = S.render()
        img = col[..., :3].astype(np.float32)
        if ss > 1:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        img = np.clip(img + rng.normal(0, a.noise, img.shape), 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    dirs = {"head": "top_cam", "wrist_right": "right_w_cam", "wrist_left": "left_w_cam"}
    files = {"head": "head.mp4", "wrist_right": "wrist_right.mp4", "wrist_left": "wrist_left.mp4"}
    gt = {"fps": a.fps, "starts_abs_s": dict(zip(roles, starts)), "clock_skew_s": dict(zip(roles, skew)),
          "true_offset_vs_head_s": {r: starts[i] - starts[0] for i, r in enumerate(roles)}, "episodes": names}
    for i, r in enumerate(roles):
        d = os.path.join(a.out, dirs[r])
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        n = int(round(durs[r] * a.fps))
        ct = iso(T0 + timedelta(seconds=starts[i] + skew[i]))
        proc = open_writer(os.path.join(d, files[r]), W, H, a.fps, wavs[r], ct)
        poses = {"T_world_cam": [], "T_world_body": [], "T_world_tcp": [], "width": [], "t_head": []}
        black = np.zeros((H, W, 3), np.uint8)
        for k in range(n):
            t_abs = starts[i] + k / a.fps
            if r == "wrist_left":
                proc.stdin.write(black.tobytes()); continue
            T_body, T_tcp, w = scene_at(t_abs)
            if r == "head":
                T_wc = headcam.pose(t_abs - starts[0]); frame = render(K_head, T_wc, 0.01)
            else:
                T_wc = T_body @ T_BODY_WRISTCAM; frame = render(K_wrist, T_wc, 0.005)
            proc.stdin.write(frame.tobytes())
            for key, v in (("T_world_cam", T_wc), ("T_world_body", T_body), ("T_world_tcp", T_tcp),
                           ("width", w), ("t_head", t_abs - starts[0])):
                poses[key].append(v)
            if k % 60 == 0:
                print(f"{r}: frame {k + 1}/{n}", flush=True)
        proc.stdin.close(); proc.wait()
        if r != "wrist_left":
            np.savez_compressed(os.path.join(gt_dir, f"gt_{r}.npz"), **{k: np.asarray(v) for k, v in poses.items()})
        print(f"{r}: wrote {n} frames ({durs[r]:.2f} s), creation_time {ct}")
    S.close()
    for w_ in wavs.values():
        os.remove(w_)

    clips = {"clips": [{"name": f"ep{j + 1:03d}_{nm}", "start": round(t0, 4), "end": round(t1, 4),
                        "task_label": EPISODE_TASKS.get(nm, nm)} for j, (nm, t0, t1) in enumerate(bounds)]}
    json.dump(clips, open(os.path.join(gt_dir, "clips_gt.json"), "w"), indent=1)
    gt["episode_bounds_head_s"] = [[nm, t0, t1] for nm, t0, t1 in bounds]
    gt["durations_s"] = durs
    gt["claps_head_s"] = [c - starts[0] for c in claps]
    json.dump(gt, open(os.path.join(gt_dir, "gt.json"), "w"), indent=1)
    cams = {"head": {"width": W, "height": H, "fx": a.head_f, "fy": a.head_f, "cx": (W - 1) / 2, "cy": (H - 1) / 2},
            "wrist_right": {"width": W, "height": H, "fx": float(fw), "fy": float(fw), "cx": (W - 1) / 2,
                            "cy": (H - 1) / 2, "T_body_cam": T_BODY_WRISTCAM.tolist(), "hfov_deg": a.wrist_hfov}}
    yaml.safe_dump(cams, open(os.path.join(gt_dir, "cameras.yaml"), "w"), sort_keys=False)
    print(f"session written to {a.out}; true offsets vs head {gt['true_offset_vs_head_s']}")


if __name__ == "__main__":
    main()
