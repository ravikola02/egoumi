"""Where observation.state / action come from, one class per source.

    none    video-only dataset: no observation.state / action columns (default)
    zeros   14-d all-zero placeholder in the h2r_collection layout, flagged in info.json;
            only for smoke-testing training tooling, never for training
    (planned) gripper   ArUco gripper pose + jaw width (scripts/gripper_pose.py output)

A source returns (state, action) arrays of shape (n, dim), or (None, None) for `none`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Same 14 dims and names as h2r_collection's --eef-point grasp export, so h2r_il tooling reads it unchanged.
H2R_14D_NAMES = [
    "left.eef.x", "left.eef.y", "left.eef.z", "left.eef.roll", "left.eef.pitch", "left.eef.yaw",
    "right.eef.x", "right.eef.y", "right.eef.z", "right.eef.roll", "right.eef.pitch", "right.eef.yaw",
    "left.gripper", "right.gripper",
]
H2R_14D_LAYOUT = "[L pos 3, L rpy 3, R pos 3, R rpy 3, L grip 1, R grip 1]"


def derive_action_from_state(state: np.ndarray) -> np.ndarray:
    """action[t] = state[t+1]; last frame repeats (h2r_collection convention)."""
    return np.concatenate([state[1:], state[-1:]], axis=0)


class NoneSource:
    name = "none"
    dim = 0
    names: List[str] = []
    placeholder = False

    def compute(self, episode, n: int, fps: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return None, None

    def info(self) -> Dict:
        return {"state_source": self.name, "state_placeholder": False,
                "state_signal": "No observation.state / action: video-only dataset."}


class ZerosSource:
    name = "zeros"
    dim = len(H2R_14D_NAMES)
    names = H2R_14D_NAMES
    placeholder = True

    def compute(self, episode, n: int, fps: float):
        state = np.zeros((n, self.dim), np.float32)
        return state, derive_action_from_state(state)

    def info(self) -> Dict:
        return {"state_source": self.name, "state_placeholder": True, "state_layout": H2R_14D_LAYOUT,
                "state_signal": "PLACEHOLDER: all zeros. End-effector pose and gripper values are not "
                                "reconstructed yet. Do not train on this."}


# ---------------------------------------------------------------------------------------------------------
# Gripper trajectories (one gripper, on the right arm). Frame and angle conventions follow h2r_collection:
#   world = the episode's first head-camera frame, OpenCV optical axes (x right, y down, z forward), metres;
#   eef frame = h2r_collection's hand frame mapped onto the gripper (standard parallel-gripper EEF):
#     +z = approach (body +Y, finger direction)       [h2r: wrist -> pinch point]
#     +y = jaw axis toward the thumb-side jaw (body -X) [h2r: z x x, thumb side for the right hand]
#     +x = y x z = grip-plane normal (body -Z)           [h2r: palm normal, x_sign -1 for the right hand]
#   Identical to h2r's _wrist_rotmats when its wrist point lies on the approach line; a real wrist below the grip
#   would tilt h2r's literal formula ~23 deg in pitch, so the gripper's own finger axis is used as z;
#   rpy = intrinsic XYZ Euler of that frame (scipy "XYZ"), wrapped to (-pi, pi] like h2r_collection;
#   gripper = 1 - width / width_open, clipped to [0, 1] (1 = closed).
# The left arm has no gripper in these captures: its 7 dims are zero and flagged in info.json.
# ---------------------------------------------------------------------------------------------------------

R_TCP_EEF = np.array([[0.0, -1, 0], [0, 0, 1], [-1, 0, 0]])    # columns: eef x, y, z in the gripper body frame


def inv_batch(T: np.ndarray) -> np.ndarray:
    Ti = np.zeros_like(T)
    Rt = np.transpose(T[:, :3, :3], (0, 2, 1))
    Ti[:, :3, :3] = Rt
    Ti[:, :3, 3] = -np.einsum("nij,nj->ni", Rt, T[:, :3, 3])
    Ti[:, 3, 3] = 1.0
    return Ti


def fill_gaps(T: np.ndarray, src: np.ndarray, max_gap: int) -> None:
    """Interpolate runs of src == 0 no longer than max_gap between two measured poses (in place; src -> 2)."""
    from scipy.spatial.transform import Rotation as R, Slerp
    n, i = len(T), 0
    while i < n:
        if src[i] != 0:
            i += 1; continue
        j = i
        while j < n and src[j] == 0:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            a, b = i - 1, j
            sl = Slerp([0, 1], R.from_matrix([T[a][:3, :3], T[b][:3, :3]]))
            for k in range(i, j):
                u = (k - a) / (b - a)
                T[k][:3, :3] = sl(u).as_matrix()
                T[k][:3, 3] = (1 - u) * T[a][:3, 3] + u * T[b][:3, 3]
                src[k] = 2
        i = j


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA over axis 0, as h2r_collection's _ema: y[0] = x[0]; y[t] = alpha*x[t] + (1-alpha)*y[t-1].
    alpha >= 1 is a no-op. Lag ~ (1 - alpha) / alpha frames (alpha 0.164 -> 5.1 frames = 170 ms at 30 fps)."""
    x = np.asarray(x, np.float64)
    if alpha >= 1.0 or len(x) < 2:
        return x.copy()
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1.0 - alpha) * y[t - 1]
    return y


def ema_rot(Rs: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA on rotations (N, 3, 3), done on SO(3) so Euler wraps cannot break it:
    R_f[0] = R[0]; R_f[t] = R_f[t-1] * exp(alpha * log(R_f[t-1]^T R[t])), i.e. move alpha of the way (geodesic slerp)
    from the previous filtered rotation toward the new one. Same lag as ema() at the same alpha."""
    from scipy.spatial.transform import Rotation as R
    Rs = np.asarray(Rs, np.float64)
    if alpha >= 1.0 or len(Rs) < 2:
        return Rs.copy()
    out = np.empty_like(Rs)
    out[0] = Rs[0]
    for t in range(1, len(Rs)):
        step = R.from_matrix(out[t - 1].T @ Rs[t]).as_rotvec() * alpha
        out[t] = out[t - 1] @ R.from_rotvec(step).as_matrix()
    return out


def _repo_imports():
    import os, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def _pose_rows(T_c0_tcp: np.ndarray, grip: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    n = len(T_c0_tcp)
    state = np.zeros((n, 14), np.float32)
    state[:, 6:9] = T_c0_tcp[:, :3, 3]
    state[:, 9:12] = R.from_matrix(T_c0_tcp[:, :3, :3] @ R_TCP_EEF).as_euler("XYZ")
    state[:, 13] = grip
    return state


def _load_cut_entries(cut_sheet) -> Dict[str, Dict]:
    import json
    return {c.get("name", f"episode_{i + 1:03d}"): c
            for i, c in enumerate(json.load(open(cut_sheet)).get("clips", []))}


class _GripperBase:
    dim = len(H2R_14D_NAMES)
    names = H2R_14D_NAMES
    placeholder = False

    def __init__(self, cut_sheet=None, sim_gt=None, **_):
        if cut_sheet is None:
            raise ValueError(f"state source {self.name!r} needs the cut sheet")
        self.entries = _load_cut_entries(cut_sheet)
        self.sim_gt = sim_gt
        self.episode_info: Dict = {}
        self.extra: Dict[str, np.ndarray] = {}      # extra per-row float columns for the dataset
        self._gt = None
        if sim_gt:
            import os
            g = np.load(os.path.join(sim_gt, "gt_head.npz"))
            self._gt = {k: g[k] for k in g.files}
            self._gt_fps = __import__("json").load(open(os.path.join(sim_gt, "gt.json")))["fps"]

    def _head_frames(self, episode, n: int, fps: float) -> np.ndarray:
        """Raw head-recording frame index of every dataset row (head = sync reference, offset 0)."""
        entry = self.entries[episode.name]
        f0 = int(round(float(entry["start"]) * fps))
        return f0 + np.arange(n)

    def _gt_world_cam(self, idx):
        if self._gt is None:
            raise ValueError("this needs --sim-gt <session>/sim_gt (head-camera motion from the sim)")
        if abs(self._gt_fps - round(self._gt_fps)) > 1e-6 or idx[-1] >= len(self._gt["T_world_cam"]):
            raise ValueError("dataset rows run past the sim ground truth")
        return self._gt["T_world_cam"][idx]


class SimGtSource(_GripperBase):
    """Ground-truth trajectory from a sim/render_capture.py session: tests the trajectory plumbing."""
    name = "sim_gt"

    def compute(self, episode, n: int, fps: float):
        idx = self._head_frames(episode, n, fps)
        T_wc = self._gt_world_cam(idx)
        T_c0_tcp = inv_batch(T_wc[:1])[0] @ self._gt["T_world_tcp"][idx]
        grip = np.clip(1.0 - self._gt["width"][idx] / 0.072, 0.0, 1.0)
        self.episode_info = {"head_frames": [int(idx[0]), int(idx[-1])]}
        state = _pose_rows(T_c0_tcp, grip)
        return state, derive_action_from_state(state)

    def info(self) -> Dict:
        return {"state_source": self.name, "state_placeholder": False, "state_layout": H2R_14D_LAYOUT,
                "state_signal": "SIM GROUND TRUTH (sim/render_capture.py): right gripper TCP pose in the episode's "
                                "first head-camera optical frame; eef frame as h2r_collection (+z approach = body +Y, +y jaw axis toward the thumb side = body -X, +x = body -Z); "
                                "intrinsic-XYZ rpy; right.gripper = 1 - width/0.072 m (1 = closed). "
                                "Left arm absent: left.* dims are zero."}


class ArucoSource(_GripperBase):
    """Gripper pose reconstructed from the HEAD clip with the ArUco pipeline (umi_pipe.pose)."""
    name = "aruco"

    def __init__(self, cut_sheet=None, sim_gt=None, camera=None, markers=None, cam_motion="sim_gt",
                 max_gap=5, ema_pos=0.18, ema_grip=1.0, ema_rot=None, **kw):
        super().__init__(cut_sheet=cut_sheet, sim_gt=sim_gt)
        _repo_imports()
        import yaml
        from umi_pipe.model import GripperModel, MARKER_CFG
        from umi_pipe import pose as P
        if camera is None:
            raise ValueError("state source 'aruco' needs --camera (head intrinsics yaml)")
        cams = yaml.safe_load(open(camera))
        cam = cams.get("head", cams)
        self.K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
        self.dist = np.asarray(cam.get("dist", [0] * 5), float)
        # gripper (wrist) camera: if its mount on the gripper is known, jaw width comes from it (UMI-style: the
        # camera is rigid to the body, so each jaw tag only slides along its rail in this view)
        wr = cams.get("wrist_right") if isinstance(cams, dict) else None
        self.wrist = None
        if wr and "T_body_cam" in wr:
            from umi_pipe.geometry import inv
            self.wrist = {"K": np.array([[wr["fx"], 0, wr["cx"]], [0, wr["fy"], wr["cy"]], [0, 0, 1.0]]),
                          "dist": np.asarray(wr.get("dist", [0] * 5), float),
                          "T_cam_body": inv(np.asarray(wr["T_body_cam"], float))}
        self.model = GripperModel(markers or MARKER_CFG)
        self.P = P
        self.cam_motion = cam_motion
        self.max_gap = max_gap
        # h2r criterion (sim/ema_sweep.py): least smoothing whose one-step delta noise reaches the G1 FK level ->
        # position 0.18 (h2r kitting used 0.164 for HaWoR); the wrist-camera gripper is clean, so no gripper EMA
        self.ema_pos = 1.0 if ema_pos is None else float(ema_pos)
        self.ema_grip = 1.0 if ema_grip is None else float(ema_grip)
        # orientation: same alpha as position by default, so the whole pose carries one lag
        self.ema_rot = self.ema_pos if ema_rot is None else float(ema_rot)

    def compute(self, episode, n: int, fps: float):
        import cv2
        from umi_pipe.geometry import pose_error
        idx = self._head_frames(episode, n, fps)
        if self.cam_motion != "sim_gt":
            raise ValueError(f"cam_motion {self.cam_motion!r} not implemented (SLAM is not wired yet)")
        T_wc = self._gt_world_cam(idx)

        det = self.P.make_detector(self.model)
        cap = cv2.VideoCapture(str(episode.clips["head"]))
        T_cam_body = np.tile(np.eye(4), (n, 1, 1)); src = np.zeros(n, int)
        travel = np.full(n, np.nan); n_tags = np.zeros(n, int); T_prev = None
        for k in range(n):
            ok, img = cap.read()
            if not ok:
                break
            dets = self.P.detect(det, img)
            b = self.P.estimate_body(dets, self.model, self.K, self.dist, T_prev)
            if b is None:
                continue
            T_cam_body[k] = T_prev = b["T_cam_body"]; src[k] = 1; n_tags[k] = len(b["ids"])
            tr = self.P.estimate_travel(dets, self.model, self.K, self.dist, b["T_cam_body"])
            if tr:
                travel[k] = np.mean([v[0] for v in tr.values()])
        cap.release()
        travel_head = travel.copy()
        travel_wrist = np.full(n, np.nan)
        if self.wrist is not None and "wrist_right" in episode.clips:
            cap = cv2.VideoCapture(str(episode.clips["wrist_right"]))
            for k in range(n):
                ok, img = cap.read()
                if not ok:
                    break
                tr = self.P.estimate_travel(self.P.detect(det, img), self.model, self.wrist["K"], self.wrist["dist"],
                                            self.wrist["T_cam_body"])
                if tr:
                    travel_wrist[k] = np.mean([v[0] for v in tr.values()])
            cap.release()
            travel = np.where(np.isfinite(travel_wrist), travel_wrist, travel_head)

        # pose in the episode frame, gaps <= max_gap interpolated, longer gaps held (and reported)
        T_c0_tcp = inv_batch(T_wc[:1])[0] @ T_wc @ T_cam_body @ self.model.T_body_tcp
        fill_gaps(T_c0_tcp, src, self.max_gap)
        held = 0
        for k in range(n):
            if src[k] == 0:
                prev = np.where(src[:k] > 0)[0]
                nxt = np.where(src[k:] > 0)[0]
                j = prev[-1] if len(prev) else (k + nxt[0] if len(nxt) else None)
                if j is not None:
                    T_c0_tcp[k] = T_c0_tcp[j]; held += 1
        width_ok = np.isfinite(travel)
        # frames without a jaw tag: interpolate the travel between measured frames (hold at the ends); 0 = open
        # only when the jaw tags were never seen in the whole episode
        if width_ok.any():
            k_ok = np.where(width_ok)[0]
            travel = np.interp(np.arange(n), k_ok, travel[k_ok])
            grip = np.clip(1.0 - self.model.width_from_travel(np.clip(travel, 0, self.model.travel_max)) /
                           self.model.width_open, 0, 1)
        else:
            grip = np.zeros(n)
        self.episode_info = {"head_frames": [int(idx[0]), int(idx[-1])],
                             "pose_measured_pct": round(float((src == 1).mean() * 100), 1),
                             "pose_interpolated_pct": round(float((src == 2).mean() * 100), 1),
                             "pose_held_pct": round(held / n * 100, 1),
                             "single_tag_pct": round(float(((n_tags == 1) & (src == 1)).mean() * 100), 1),
                             "gripper_measured_pct": round(float(width_ok.mean() * 100), 1),
                             "gripper_from_wrist_pct": round(float(np.isfinite(travel_wrist).mean() * 100), 1),
                             "gripper_from_head_only_pct": round(float((np.isfinite(travel_head) &
                                                                        ~np.isfinite(travel_wrist)).mean() * 100), 1)}
        if self._gt is not None:   # score against the sim ground truth in the same frame, and ship it as a tag
            G = inv_batch(T_wc[:1])[0] @ self._gt["T_world_tcp"][idx]
            gt_state = _pose_rows(G, np.clip(1.0 - self._gt["width"][idx] / self.model.width_open, 0.0, 1.0))
            self.extra = {"observation.state_gt": gt_state, "action_gt": derive_action_from_state(gt_state)}
            e = np.array([pose_error(a, b) for a, b in zip(T_c0_tcp, G)])
            self.episode_info.update({"vs_gt_pos_mm_median": round(float(np.median(e[:, 0]) * 1000), 2),
                                      "vs_gt_pos_mm_p90": round(float(np.percentile(e[:, 0], 90) * 1000), 2),
                                      "vs_gt_rot_deg_median": round(float(np.median(e[:, 1])), 3),
                                      "vs_gt_pos_mm_max": round(float(e[:, 0].max() * 1000), 2)})
        # causal EMA on the PREDICTED pose (the GT tags stay raw): orientation on SO(3) before it becomes Euler angles,
        # position and gripper per component
        T_f = T_c0_tcp.copy()
        T_f[:, :3, :3] = ema_rot(T_c0_tcp[:, :3, :3], self.ema_rot)
        state = _pose_rows(T_f, grip)
        state[:, 6:9] = ema(state[:, 6:9], self.ema_pos)
        state[:, 13] = ema(state[:, 13], self.ema_grip)
        if self._gt is not None:
            e2 = np.linalg.norm(state[:, 6:9] - gt_state[:, 6:9], axis=1) * 1000
            r2 = np.array([pose_error(a, b)[1] for a, b in zip(T_f, G)])
            self.episode_info.update({"vs_gt_pos_mm_median_after_ema": round(float(np.median(e2)), 2),
                                      "vs_gt_pos_mm_p90_after_ema": round(float(np.percentile(e2, 90)), 2),
                                      "vs_gt_pos_mm_max_after_ema": round(float(e2.max()), 2),
                                      "vs_gt_rot_deg_median_after_ema": round(float(np.median(r2)), 3),
                                      "vs_gt_rot_deg_max_after_ema": round(float(r2.max()), 3)})
        return state, derive_action_from_state(state)

    @property
    def extra_keys(self) -> List[str]:
        return ["observation.state_gt", "action_gt"] if self._gt is not None else []

    def info(self) -> Dict:
        return {"state_source": self.name, "state_placeholder": False, "state_layout": H2R_14D_LAYOUT,
                "camera_motion": self.cam_motion,
                "ema": {"position_alpha": self.ema_pos, "gripper_alpha": self.ema_grip,
                        "orientation_alpha": self.ema_rot,
                        "orientation": "filtered on SO(3) (geodesic slerp toward each new rotation), before conversion "
                                       "to Euler angles, so the +-pi wrap cannot break it",
                        "applies_to": "observation.state / action only (ground-truth columns are raw)",
                        "lag_frames_position": round((1 - self.ema_pos) / self.ema_pos, 2) if self.ema_pos < 1 else 0.0,
                        "note": "causal, reset per episode; the filtered state lags the unshifted videos. Inference must "
                                "apply the same filter to live observations, or train on a raw (alpha 1.0) export."},
                "ground_truth_columns": ({"observation.state_gt": "sim ground truth of observation.state (same frame, "
                                                                  "layout and conventions)",
                                          "action_gt": "action_gt[t] = observation.state_gt[t+1]"}
                                         if self._gt is not None else {}),
                "state_signal": "ArUco gripper pose from the head clip (multi-tag PnP), right gripper TCP in the "
                                "episode's first head-camera optical frame; eef frame as h2r_collection (+z approach = "
                                "body +Y, +y jaw axis toward the thumb side = body -X, +x = body -Z); intrinsic-XYZ rpy. "
                                "Head-camera motion: " +
                                ("SIM GROUND TRUTH standing in for SLAM (not wired yet)." if self.cam_motion == "sim_gt"
                                 else self.cam_motion) +
                                " right.gripper = 1 - width/width_open from the jaw tags, seen by the gripper (wrist) camera "
                                "when its mount is known (head camera as fallback), interpolated across frames without a "
                                "jaw tag (meta/egoumi_episodes.json gripper_measured_pct), 0 if never seen in an episode. Left arm absent: left.* = 0."}


SOURCES = {s.name: s for s in (NoneSource, ZerosSource, SimGtSource, ArucoSource)}


def get_source(name: str, **opts):
    if name not in SOURCES:
        raise ValueError(f"unknown state source {name!r}; choose from {sorted(SOURCES)}")
    cls = SOURCES[name]
    return cls(**opts) if issubclass(cls, _GripperBase) else cls()
