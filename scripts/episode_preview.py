#!/usr/bin/env python3
"""Preview of one episode of a collection run (real or rendered capture), via scripts/preview_gripper.py.

    python3 scripts/episode_preview.py --collection data/output/<c> --episode ep001 [--sim-gt data/raw/<s>/sim_gt]
    PYOPENGL_PLATFORM=egl ~/anaconda3/envs/hawor/bin/python scripts/preview_gripper.py \
        --seq data/output/<c>/preview/<episode> --style presentation --ego

Builds <collection>/preview/<episode>/ in the layout preview_gripper.py reads:
  extracted_images/   head clip frames at the dataset rows
  camera.yaml         head intrinsics at the clip size (configs/cameras_osmo_nano.yaml, or the sim's cameras.yaml)
  gripper_pose/       scripts/gripper_pose.py on those frames (body pose in the head camera), then:
                      head-camera motion = the converter's HaWoR SLAM (<collection>/slam_cache), and
                      jaw width = the dataset's gripper value (gripper camera), width_open * (1 - gripper)
  gt.npz              (sim only) ground truth for the same rows, in the sim world
  objects.npz         (optional, --grasped-mesh / --container-mesh) object meshes placed FROM THE GRIPPER, not
                      measured: the grasped object stands on the table under the TCP where the jaws close, moves
                      rigidly with the TCP while they stay closed and stays where it was released; the container
                      stands on the table under the release point. Yaw is not observed (meshes keep their own).
                      Meshes: npz with vertices (m, z up, bottom at z = 0, footprint centred), faces, colors (RGB)
                      e.g. the SAM3D kitting meshes in egovla_setup/kitting_assets/{hub,box}_table.npz.
World frame of the preview:
  sim:  the sim world (z up); the SLAM track is anchored at the true first head-camera pose, so its scale and drift
        errors show against the green ground truth.
  real: the first head-camera frame LEVELLED for display: up = the mean gripper body +Z over the episode (the gripper
        is held top-cover up; APPROXIMATE, ~8 deg frame-to-frame spread on the first capture), forward = the first
        head camera's view direction made horizontal, and a table plane at z = 0, 3 cm under the TCP at the grasp
        (APPROXIMATE). Display only: the dataset keeps the h2r frame (first head-camera optical frame).
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np, cv2, yaml, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "collection"))
from state_sources import row_frames, scaled_K      # noqa: E402

GRIP = 13


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", required=True, help="data/output/<collection> (episode_clips/, lerobot_v3/, slam_cache/)")
    ap.add_argument("--episode", required=True, help="episode name, e.g. ep001")
    ap.add_argument("--dataset", help="default <collection>/lerobot_v3")
    ap.add_argument("--camera", help="head intrinsics yaml (default configs/cameras_osmo_nano.yaml, or <sim-gt>/cameras.yaml)")
    ap.add_argument("--sim-gt", help="<raw session>/sim_gt of a rendered capture: adds ground truth, sim world")
    ap.add_argument("--width-open", type=float, default=0.072)
    ap.add_argument("--grasped-mesh", help="npz mesh of the grasped object (see objects.npz above)")
    ap.add_argument("--container-mesh", help="npz mesh of the container the object is placed in")
    ap.add_argument("--closed", type=float, default=0.5, help="gripper value counted as closed on the object")
    ap.add_argument("--python", default=os.path.expanduser("~/anaconda3/envs/hawor/bin/python"),
                    help="interpreter for scripts/gripper_pose.py")
    a = ap.parse_args()

    col = os.path.abspath(a.collection); ds = a.dataset or os.path.join(col, "lerobot_v3")
    eps = {e["name"]: e for e in json.load(open(os.path.join(ds, "meta", "egoumi_episodes.json")))}
    if a.episode not in eps:
        sys.exit(f"{a.episode} not in {ds}: {sorted(eps)}")
    info = eps[a.episode]; n = info["length"]
    fps = float(json.load(open(os.path.join(ds, "meta", "info.json")))["fps"])
    out = os.path.join(col, "preview", a.episode); os.makedirs(os.path.join(out, "extracted_images"), exist_ok=True)

    # 1. head frames at the dataset rows + intrinsics at the clip size
    img = None
    for k, img in row_frames(os.path.join(col, "episode_clips", f"{a.episode}_head.mp4"), n, fps):
        cv2.imwrite(os.path.join(out, "extracted_images", f"{k + 1:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cam_yaml = a.camera or (os.path.join(a.sim_gt, "cameras.yaml") if a.sim_gt else
                            os.path.join(ROOT, "configs", "cameras_osmo_nano.yaml"))
    cams = yaml.safe_load(open(cam_yaml)); head = cams.get("head", cams)
    K, dist = scaled_K(head, img.shape[1], img.shape[0])
    yaml.safe_dump({"width": int(img.shape[1]), "height": int(img.shape[0]), "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                    "cx": float(K[0, 2]), "cy": float(K[1, 2]), "dist": [float(x) for x in dist], "fps": fps},
                   open(os.path.join(out, "camera.yaml"), "w"))

    # 2. body pose in the head camera
    subprocess.check_call([a.python, os.path.join(ROOT, "scripts", "gripper_pose.py"), "--seq", out,
                           "--cam-source", "none"], cwd=ROOT)
    pz = dict(np.load(os.path.join(out, "gripper_pose", "poses.npz"), allow_pickle=True))
    T_cam_tcp = pz["T_world_tcp"]                      # cam-source none: "world" = head camera

    # 3. head motion (SLAM) + jaw width (dataset gripper value)
    hits = glob.glob(os.path.join(col, "slam_cache", f"{a.episode}_*", "cameras.npz"))
    if not hits:
        sys.exit(f"no SLAM cache for {a.episode} in {col}/slam_cache (convert with --cam-motion hawor_slam)")
    slam = np.load(sorted(hits, key=os.path.getmtime)[-1])["T_world_cam"][:n]
    df = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(ds, "data", "**", "*.parquet"), recursive=True)])
    d = df[df.episode_index == info["episode_index"]].sort_values("frame_index")
    grip = np.stack(d["observation.state"].values)[:n, GRIP]
    pz["width"] = a.width_open * (1 - grip); pz["width_src"] = np.ones(n, int)

    if a.sim_gt:
        g = np.load(os.path.join(a.sim_gt, "gt_head.npz"))
        clip = json.load(open(os.path.join(os.path.dirname(a.sim_gt), "clips.json")))["clips"]
        start = {c["name"]: c["start"] for c in clip}[a.episode]
        idx = int(round(start * fps)) + np.arange(n)
        gt = {k: g[k][idx] for k in ("T_world_cam", "T_world_body", "T_world_tcp", "width")}
        gt["travel"] = (a.width_open - gt["width"]) / 2
        np.savez_compressed(os.path.join(out, "gt.npz"), **gt)
        W = gt["T_world_cam"][0] @ slam                    # SLAM world (first head frame) -> sim world
        src = "hawor_slam"
    else:
        W0 = slam
        up = np.array([(W0[k] @ pz["T_cam_body"][k])[:3, 2] for k in range(n)]).mean(0); up /= np.linalg.norm(up)
        fw = np.array([0, 0, 1.0]) - up * up[2]; fw /= np.linalg.norm(fw)
        L = np.eye(4); L[:3, :3] = np.stack([np.cross(fw, up), fw, up])
        tcp = np.array([(L @ W0[k] @ T_cam_tcp[k])[:3, 3] for k in range(n)])
        kg = int(np.argmax(grip > 0.6))
        c = (tcp.min(0) + tcp.max(0)) / 2
        L[:3, 3] = -np.array([c[0], c[1] - 0.15, tcp[kg, 2] - 0.03])
        W = L @ slam
        src = "hawor_slam"
        print(f"levelled for display: head camera {W[0, 2, 3]:.2f} m above the table plane (approximate)")
    pz["T_world_cam"] = W
    pz["T_world_body"] = W @ pz["T_cam_body"]
    pz["T_world_tcp"] = W @ T_cam_tcp
    pz["cam_source"] = np.array(src)

    # objects, placed from the gripper (display only)
    if a.grasped_mesh or a.container_mesh:
        tcp = pz["T_world_tcp"]; closed = grip > a.closed
        if not closed.any():
            sys.exit("gripper never closes in this episode: cannot place objects")
        k0 = int(np.argmax(closed)); k1 = k0
        while k1 + 1 < n and closed[k1 + 1]:
            k1 += 1
        table_z = 0.0                                 # table plane at z = 0 in both worlds
        names, meshes, poses = [], [], []
        if a.grasped_mesh:
            T0 = np.eye(4); T0[:2, 3] = tcp[k0, :2, 3]; T0[2, 3] = table_z
            T_tcp_obj = np.linalg.inv(tcp[k0]) @ T0
            P = np.tile(T0, (n, 1, 1))
            for k in range(k0, n):
                P[k] = tcp[min(k, k1)] @ T_tcp_obj
            names.append("grasped"); meshes.append(os.path.abspath(a.grasped_mesh)); poses.append(P)
        if a.container_mesh:
            T1 = np.eye(4); T1[:2, 3] = tcp[k1, :2, 3]; T1[2, 3] = table_z
            names.append("container"); meshes.append(os.path.abspath(a.container_mesh)); poses.append(np.tile(T1, (n, 1, 1)))
        np.savez_compressed(os.path.join(out, "objects.npz"), names=np.array(names), meshes=np.array(meshes),
                            poses=np.stack(poses))
        print(f"objects: {names}, grasp at row {k0}, release at row {k1} (placed from the gripper, approximate)")
    np.savez_compressed(os.path.join(out, "gripper_pose", "poses.npz"), **pz)
    print(f"ready: {out}\nnext: PYOPENGL_PLATFORM=egl {a.python} scripts/preview_gripper.py --seq {out} "
          f"--style presentation --ego")


if __name__ == "__main__":
    main()
