"""ArUco gripper pose: detection, multi-tag rigid PnP for the body, 1-D jaw-travel fit, table-tag camera pose.

This is the core of the planned h2r_collection scripts/gripper_pose.py (docs/h2r_gripper_plan.md, step 2).
"""
import numpy as np
import cv2
from scipy.optimize import minimize_scalar

from .geometry import rvec_tvec_to_T, T_to_rvec_tvec, apply, pose_error


def make_detector(model, subpix=True):
    p = cv2.aruco.DetectorParameters()
    if subpix:
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 3
    return cv2.aruco.ArucoDetector(model.dictionary, p)


def detect(detector, img_bgr):
    """{id: (4,2) float corners TL,TR,BR,BL}."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def _reproj(K, dist, T_cam_obj, obj_pts):
    rv, tv = T_to_rvec_tvec(T_cam_obj)
    p, _ = cv2.projectPoints(obj_pts, rv, tv, K, dist)
    return p.reshape(-1, 2)


def _rms(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _planar_pose(obj, img, K, dist, T_prev=None):
    """Single planar tag: both IPPE solutions, pick the one closest to T_prev (rotation) or lowest error."""
    n, rvs, tvs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    cands = [rvec_tvec_to_T(rvs[i], tvs[i]) for i in range(n)]
    errs = np.ravel(errs)
    if T_prev is not None and len(cands) > 1:
        rot = [pose_error(T, T_prev)[1] for T in cands]
        k = int(np.argmin(rot))
        # trust the previous frame only if it separates the two candidates clearly
        if rot[1 - k] - rot[k] > 10.0 or errs[k] < 2 * errs[1 - k]:
            return cands[k], True
    k = int(np.argmin(errs))
    ambiguous = len(cands) > 1 and errs[1 - k] < 1.5 * errs[k] + 0.3
    return cands[k], not ambiguous


def estimate_body(dets, model, K, dist, T_prev=None, outlier_px=3.0, min_tags=1):
    """Rigid multi-tag PnP over the body tags. Returns dict or None.
    keys: T_cam_body, rms, ids (used), rejected, single (only one tag), confident (single-tag ambiguity resolved)."""
    tags = {t.id: t for t in model.body_tags}
    ids = [i for i in dets if i in tags]
    rejected = []
    while len(ids) >= min_tags and ids:
        obj = np.concatenate([tags[i].corners() for i in ids])
        img = np.concatenate([dets[i] for i in ids])
        if len(ids) == 1:
            T, confident = _planar_pose(obj, img, K, dist, T_prev)
        else:
            ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                return None
            rv, tv = cv2.solvePnPRefineLM(obj, img, K, dist, rv, tv)
            T, confident = rvec_tvec_to_T(rv, tv), True
        proj = _reproj(K, dist, T, obj)
        per_tag = [_rms(proj[4 * k:4 * k + 4], img[4 * k:4 * k + 4]) for k in range(len(ids))]
        worst = int(np.argmax(per_tag))
        if per_tag[worst] > outlier_px and len(ids) >= 3:
            rejected.append(ids.pop(worst))
            continue
        if T[2, 3] <= 0:
            return None
        return dict(T_cam_body=T, rms=_rms(proj, img), per_tag=dict(zip(ids, per_tag)), ids=list(ids),
                    rejected=rejected, single=len(ids) == 1, confident=confident)
    return None


def estimate_travel(dets, model, K, dist, T_cam_body, bounds=(-0.005, None)):
    """Per-side jaw travel from the holder tags with the body pose held fixed (1-D reprojection fit).
    Returns {side: (travel, rms_px)} for the sides whose tag is visible."""
    hi = model.travel_max + 0.005 if bounds[1] is None else bounds[1]
    out = {}
    for t in model.jaw_tags:
        if t.id not in dets:
            continue
        img = dets[t.id]
        f = lambda d: _rms(_reproj(K, dist, T_cam_body, t.corners(travel=d)), img)
        r = minimize_scalar(f, bounds=(bounds[0], hi), method="bounded", options={"xatol": 1e-5})
        side = "left" if t.center[0] < 0 else "right"
        out[side] = (float(r.x), float(r.fun))
    return out


def refine_joint(dets, model, K, dist, T_cam_body, travel, body_ids, huber_px=1.0):
    """Joint least-squares over body pose (6) + per-side travel (<=2) using every visible body and jaw tag.
    travel: {side: t} initial values (sides without a visible jaw tag stay fixed/absent).
    Returns (T_cam_body, {side: t}, rms_px)."""
    from scipy.optimize import least_squares
    from .geometry import T_to_rvec_tvec
    tags = {t.id: t for t in model.body_tags}
    jaws = [(t, "left" if t.center[0] < 0 else "right") for t in model.jaw_tags if t.id in dets]
    jaws = [(t, s) for t, s in jaws if s in travel]
    if not jaws:
        return T_cam_body, travel, None
    sides = sorted({s for _, s in jaws})
    rv, tv = T_to_rvec_tvec(T_cam_body)
    x0 = np.concatenate([rv.ravel(), tv.ravel(), [travel[s] for s in sides]])
    body_obj = np.concatenate([tags[i].corners() for i in body_ids])
    body_img = np.concatenate([dets[i] for i in body_ids])

    def resid(x):
        r = []
        p, _ = cv2.projectPoints(body_obj, x[:3], x[3:6], K, dist)
        r.append((p.reshape(-1, 2) - body_img).ravel())
        for t, s in jaws:
            p, _ = cv2.projectPoints(t.corners(travel=x[6 + sides.index(s)]), x[:3], x[3:6], K, dist)
            r.append((p.reshape(-1, 2) - dets[t.id]).ravel())
        return np.concatenate(r)

    lo = np.r_[np.full(6, -np.inf), np.full(len(sides), -0.005)]
    hi = np.r_[np.full(6, np.inf), np.full(len(sides), model.travel_max + 0.005)]
    x0[6:] = np.clip(x0[6:], lo[6:] + 1e-9, hi[6:] - 1e-9)
    res = least_squares(resid, x0, bounds=(lo, hi), loss="huber", f_scale=huber_px, x_scale="jac")
    T = rvec_tvec_to_T(res.x[:3], res.x[3:6])
    rms = float(np.sqrt(np.mean(resid(res.x).reshape(-1, 2) ** 2 * 2)))
    return T, {s: float(res.x[6 + k]) for k, s in enumerate(sides)}, rms


def estimate_table(dets, model, K, dist, T_prev=None):
    """T_cam_world from the static table tag (world = tag frame), or None."""
    t = model.table_tag
    if t.id not in dets:
        return None
    T, confident = _planar_pose(t.corners(), dets[t.id], K, dist, T_prev)
    return dict(T_cam_world=T, confident=confident,
                rms=_rms(_reproj(K, dist, T, t.corners()), dets[t.id]))
