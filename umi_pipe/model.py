"""Gripper + marker model shared by the sim, the pose estimator and the preview.

Body frame = CAD assembly frame in metres (configs/gripper_markers.yaml).
"""
import os
import numpy as np
import yaml
import cv2

from .geometry import make_T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(ROOT, "sim", "assets", "meshes")
MARKER_CFG = os.path.join(ROOT, "configs", "gripper_markers.yaml")

# Parts the head camera can see. The GoPro dummy is optional (mount kept on v3, camera may not be fitted).
STATIC_PARTS = ["bottom_plate", "top_cover", "cube", "grip", "slider", "gear_left", "gear_right",
                "linkage_left", "linkage_right"]


class Tag:
    def __init__(self, d, sticker_offset, moving_axis=None):
        self.id = int(d["id"])
        self.size = float(d["size"])
        self.margin = float(d.get("margin", 0.0))
        n = np.asarray(d["normal"], float); self.normal = n / np.linalg.norm(n)
        u = np.asarray(d["up"], float); self.up = u / np.linalg.norm(u)
        self.right = np.cross(self.up, self.normal)          # image +x seen from the front
        self.center = np.asarray(d["center"], float) + self.normal * sticker_offset
        self.axis = None if moving_axis is None else np.asarray(moving_axis, float)
        self.plate = float(d.get("plate", 0.0))                # >0: sim renders a backing flag this thick [m]

    def corners(self, half=None, travel=0.0):
        """4x3 corners TL, TR, BR, BL (OpenCV order) of the black square (or a square of half-size `half`)."""
        h = self.size / 2 if half is None else half
        c = self.center + (0 if self.axis is None else travel * self.axis)
        r, u = self.right * h, self.up * h
        return np.stack([c - r + u, c + r + u, c + r - u, c - r - u])


class GripperModel:
    def __init__(self, cfg_path=MARKER_CFG, index=0):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        g = self.cfg["grippers"][index]
        off = float(self.cfg.get("sticker_offset", 0.0))
        self.name = g["name"]
        self.body_tags = [Tag(d, off) for d in g["body_tags"]]
        self.jaw_tags = [Tag(d, off, d["axis"]) for d in g["jaw_tags"]]
        self.moving_parts = g["moving_parts"]
        self.width_open = float(g["width_open"])
        self.travel_max = float(g["travel_max"])
        self.T_body_tcp = make_T(t=g["tcp_in_body"])
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.cfg["dictionary"]))
        tt = self.cfg["table_tag"]
        self.table_tag = Tag({"id": tt["id"], "size": tt["size"], "margin": tt["margin"],
                              "center": [0, 0, 0], "normal": [0, 0, 1], "up": [0, 1, 0]}, 0.0)
        # jaw side -> axis, from the jaw tag that rides on that side's holder
        self.side_axis = {}
        for t in self.jaw_tags:
            self.side_axis["left" if t.center[0] < 0 else "right"] = t.axis

    # ---- articulation -------------------------------------------------------------------
    def travel_from_width(self, width):
        return np.clip((self.width_open - width) / 2.0, 0.0, self.travel_max)

    def width_from_travel(self, travel):
        return self.width_open - 2.0 * travel

    def part_transforms(self, travel):
        """{part: T_body_part} for a jaw travel. Gears/linkages/slider are not articulated (hidden under the cover)."""
        out = {p: np.eye(4) for p in STATIC_PARTS}
        for side, parts in self.moving_parts.items():
            for p in parts:
                out[p] = make_T(t=travel * self.side_axis[side])
        return out

    # ---- markers ------------------------------------------------------------------------
    def tag_by_id(self):
        return {t.id: t for t in self.body_tags + self.jaw_tags}

    def tag_image(self, tag, px_per_module=40):
        """RGB texture: marker plus white margin, oriented so row 0 is the `up` edge."""
        n_mod = self.dictionary.markerSize + 2
        side = n_mod * px_per_module
        img = cv2.aruco.generateImageMarker(self.dictionary, tag.id, side)
        m = int(round(tag.margin / tag.size * side))
        img = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def load_meshes(with_gopro=True):
    import trimesh
    names = list(STATIC_PARTS) + ["finger_holder_left_thumb", "finger_holder_right_index",
                                  "soft_finger_left", "soft_finger_right"]
    if with_gopro:
        names.append("camera_dummy")
    return {n: trimesh.load(os.path.join(MESH_DIR, f"{n}.stl")) for n in names}
