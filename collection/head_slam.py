#!/usr/bin/env python3
"""Head-camera motion for one episode with HaWoR's SLAM (masked DROID-SLAM + Metric3D scale), as h2r_collection.

Runs in the HaWoR conda env (torch + DROID-SLAM); called by `--state-source aruco --cam-motion hawor_slam`:

    ~/anaconda3/envs/hawor/bin/python collection/head_slam.py --head-clip ep_head.mp4 --rows 195 --fps 30 \
        --poses poses.npz --camera configs/cameras_osmo_nano.yaml --work slam_cache/ep --out cams.npz

1. Writes the head frames at the dataset-row timestamps into HaWoR's layout (<work>/head/extracted_images/%04d.jpg),
   downscaled to --max-width with the focal scaled to match.
2. Masks what moves: the gripper CAD plus a hand/forearm stand-in, rendered at each row's ArUco pose
   (T_cam_body from --poses; rows without a pose reuse the nearest posed row), dilated by --dilate px, written to
   <work>/head/tracks_0_<n>/model_masks.npy (True = ignored by SLAM). The markers are the strongest features in
   view, so without this SLAM tracks the gripper (--no-mask to measure that).
3. Calls HaWoR's hawor_slam() unchanged, from inside the HaWoR checkout (its weights paths are relative).
4. Saves T_world_cam (n, 4, 4) camera-to-world per row (world = DROID's first frame, metric via Metric3D) + scale.
"""
import os, sys, argparse, contextlib, shutil, time, json
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import cv2
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from state_sources import row_frames, scaled_K           # noqa: E402

DEFAULT_HAWOR = os.environ.get("HAWOR_REPO", os.path.join(os.path.dirname(ROOT), "h2r_collection", "HaWoR"))


@contextlib.contextmanager
def chdir(path):
    old = os.getcwd(); os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def render_masks(T_cam_body, valid, K, w, h, dilate, markers=None):
    """(n, h, w) bool: gripper + hand stand-in at each row's pose, dilated."""
    from umi_pipe.model import GripperModel, MARKER_CFG, load_meshes
    from umi_pipe.render import Scene3D
    model = GripperModel(markers or MARKER_CFG)
    S = Scene3D(w, h)
    S.add_gripper("g", model, style="flat", rgba=(1, 1, 1, 1), meshes=load_meshes(with_gopro=True), with_hand=True)
    S.set_camera(K, np.eye(4), znear=0.01, zfar=5.0)
    idx_valid = np.where(valid)[0]
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)) if dilate > 0 else None
    masks = np.zeros((len(valid), h, w), bool)
    if not len(idx_valid):
        print("  WARN no ArUco pose in this episode: SLAM runs unmasked")
        return masks
    for k in range(len(valid)):
        j = k if valid[k] else idx_valid[np.argmin(np.abs(idx_valid - k))]
        S.set_gripper("g", T_cam_body[j], 0.0)
        _, depth = S.render()
        m = (depth > 0).astype(np.uint8)
        masks[k] = (cv2.dilate(m, kern) if kern is not None else m) > 0
    S.close()
    return masks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head-clip", required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--poses", required=True, help="npz with T_cam_body (n,4,4) and valid (n,) from the ArUco pass")
    ap.add_argument("--camera", required=True, help="cameras yaml (head entry)")
    ap.add_argument("--markers", default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hawor-repo", default=DEFAULT_HAWOR)
    ap.add_argument("--max-width", type=int, default=1920)
    ap.add_argument("--dilate", type=int, default=25, help="mask dilation [px at the SLAM resolution]")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--keep-frames", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    cams = yaml.safe_load(open(a.camera)); cam = cams.get("head", cams)
    pz = np.load(a.poses); T_cam_body, valid = pz["T_cam_body"], pz["valid"].astype(bool)
    seq = os.path.join(a.work, "head")
    img_dir = os.path.join(seq, "extracted_images")
    shutil.rmtree(seq, ignore_errors=True); os.makedirs(img_dir)

    # 1. frames at the row timestamps (HaWoR names them 0001.jpg ...)
    n = 0; w = h = None; s = 1.0
    for k, img in row_frames(a.head_clip, a.rows, a.fps):
        if w is None:
            H0, W0 = img.shape[:2]
            s = min(1.0, a.max_width / W0)
            w, h = int(round(W0 * s)), int(round(H0 * s))
        if s < 1.0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(img_dir, f"{k + 1:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n = k + 1
    if n < a.rows:
        print(f"  WARN head clip gave {n} of {a.rows} rows")
    K, _ = scaled_K(cam, w, h)

    # 2. masks
    tr_dir = os.path.join(seq, f"tracks_0_{n}"); os.makedirs(tr_dir)
    masks = (np.zeros((n, h, w), bool) if a.no_mask else
             render_masks(T_cam_body[:n], valid[:n], K, w, h, a.dilate, a.markers))
    np.save(os.path.join(tr_dir, "model_masks.npy"), masks)

    # 3. HaWoR SLAM, unchanged
    class Args:  # the fields hawor_slam reads
        video_path = os.path.join(a.work, "head.mp4"); img_focal = float(K[0, 0]); input_type = "file"
    repo = os.path.abspath(a.hawor_repo)
    with chdir(repo):
        sys.path.insert(0, repo); sys.path.insert(0, os.path.join(repo, "scripts", "scripts_test_video"))
        from hawor_slam import hawor_slam
        from lib.eval_utils.custom_utils import load_slam_cam
        hawor_slam(Args, 0, n)
        slam_path = os.path.join(seq, "SLAM", f"hawor_slam_w_scale_0_{n}.npz")
        R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(slam_path)
    sl = dict(np.load(slam_path, allow_pickle=True))

    # 4. per-row camera-to-world
    T_world_cam = np.tile(np.eye(4), (n, 1, 1))
    T_world_cam[:, :3, :3] = R_c2w.cpu().numpy()[:n]
    T_world_cam[:, :3, 3] = t_c2w.cpu().numpy()[:n]
    np.savez_compressed(a.out, T_world_cam=T_world_cam, scale=float(sl["scale"]), img_focal=float(K[0, 0]),
                        slam_size=np.array([w, h]), masked=not a.no_mask, dilate=a.dilate,
                        mask_frac=float(masks.mean()), seconds=time.time() - t0)
    if not a.keep_frames:
        shutil.rmtree(img_dir, ignore_errors=True)
        os.remove(os.path.join(tr_dir, "model_masks.npy"))
    print(json.dumps({"rows": n, "scale": float(sl["scale"]), "mask_frac": round(float(masks.mean()), 4),
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
