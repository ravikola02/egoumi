"""SE(3) helpers. Poses are 4x4 numpy arrays T_a_b mapping points in frame b to frame a."""
import numpy as np
from scipy.spatial.transform import Rotation as R

CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])     # OpenCV camera (x right, y down, z fwd) <-> OpenGL (y up, -z fwd)


def make_T(Rm=None, t=None):
    T = np.eye(4)
    if Rm is not None:
        T[:3, :3] = Rm
    if t is not None:
        T[:3, 3] = t
    return T


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def apply(T, pts):
    pts = np.asarray(pts, float)
    return pts @ T[:3, :3].T + T[:3, 3]


def rvec_tvec_to_T(rvec, tvec):
    return make_T(R.from_rotvec(np.ravel(rvec)).as_matrix(), np.ravel(tvec))


def T_to_rvec_tvec(T):
    return R.from_matrix(T[:3, :3]).as_rotvec().reshape(3, 1), T[:3, 3].reshape(3, 1).copy()


def frame_from_axes(y_dir, z_dir):
    """Rotation whose +Y is y_dir and +Z is z_dir (z orthogonalised against y)."""
    y = np.asarray(y_dir, float); y /= np.linalg.norm(y)
    z = np.asarray(z_dir, float); z = z - y * (z @ y); z /= np.linalg.norm(z)
    return np.stack([np.cross(y, z), y, z], axis=1)


def look_at_cv(eye, target, up=(0, 0, 1)):
    """T_world_cam (OpenCV convention) for a camera at eye looking at target."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_T(np.stack([x, y, z], axis=1), eye)


def pose_error(T_est, T_gt):
    """(translation error [m], rotation error [deg])."""
    dt = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
    dR = T_est[:3, :3].T @ T_gt[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return dt, ang
