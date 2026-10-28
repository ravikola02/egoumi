"""pyrender scenes for the sim (textured gripper, hand stand-in, table, table tag, object)
and for the preview (flat-coloured gripper ghosts, camera frustums, trails).

Run with PYOPENGL_PLATFORM=egl for headless rendering.
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import trimesh
import pyrender

from .geometry import make_T, CV_TO_GL
from .model import load_meshes

PLA = (0.78, 0.78, 0.76, 1.0)          # light-grey PLA+
TPU = (0.15, 0.15, 0.17, 1.0)          # black TPU fingers
SKIN = (0.80, 0.58, 0.46, 1.0)


def _mat(rgba, metallic=0.0, rough=0.8, alpha_blend=False):
    return pyrender.MetallicRoughnessMaterial(baseColorFactor=rgba, metallicFactor=metallic, roughnessFactor=rough,
                                              alphaMode="BLEND" if alpha_blend else "OPAQUE", doubleSided=True)


def textured_quad(corners, img):
    """corners: 4x3 TL, TR, BR, BL; img: HxWx3 uint8 with row 0 = top edge."""
    uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)     # pyrender: v=0 is the image's last row
    tex = pyrender.Texture(source=img, source_channels="RGB",
                           sampler=pyrender.Sampler(magFilter=9729, minFilter=9987))   # LINEAR, LINEAR_MIPMAP_LINEAR
    mat = pyrender.MetallicRoughnessMaterial(baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=0.9,
                                             doubleSided=True)
    prim = pyrender.Primitive(positions=np.asarray(corners, np.float32), texcoord_0=uv,
                              normals=np.tile(np.cross(corners[1] - corners[0], corners[3] - corners[0]), (4, 1)).astype(np.float32),
                              indices=np.array([[0, 2, 1], [0, 3, 2]], np.uint32), material=mat)
    return pyrender.Mesh([prim])


def capsule(p0, p1, r):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0; L = np.linalg.norm(v)
    m = trimesh.creation.capsule(height=L, radius=r, count=[12, 12])
    z = v / L; x = np.cross(z, [0, 0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0, 0]); x /= np.linalg.norm(x)
    m.apply_transform(make_T(np.stack([x, np.cross(z, x), z], 1), p0))   # capsule axis = +Z from 0..L
    return m


def hand_meshes(model):
    """Crude right hand in the body frame: palm behind the grip, three fingers round the grip, thumb and
    index reaching the rings (they translate with their holder). mm numbers from cad/build_v2.py ring params."""
    mm = 1e-3
    palm = trimesh.creation.box(extents=[60 * mm, 26 * mm, 80 * mm]); palm.apply_translation([0, -30 * mm, -60 * mm])
    forearm = capsule([0, -40 * mm, -95 * mm], [0, -150 * mm, -310 * mm], 30 * mm)
    wrap = [capsule([-22 * mm, -18 * mm, z * mm], [22 * mm, 28 * mm, z * mm], 9 * mm) for z in (-50, -70, -90)]
    static = trimesh.util.concatenate([palm, forearm] + wrap)
    thumb = capsule([-28 * mm, -20 * mm, -30 * mm], [-50 * mm, 41 * mm, -27 * mm], 10 * mm)     # thumb ring centre
    index = capsule([28 * mm, -18 * mm, -40 * mm], [40 * mm, 71 * mm, -36.5 * mm], 9 * mm)     # index ring centre
    return static, {"left": thumb, "right": index}


class Scene3D:
    def __init__(self, width, height, ambient=0.35):
        self.scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(ambient,) * 3)
        self.r = pyrender.OffscreenRenderer(width, height)
        self.w, self.h = width, height
        self.cam_node = None
        self.grippers = {}

    # ---- camera / lights --------------------------------------------------------------
    def set_camera(self, K, T_world_cam_cv, scale=1, znear=0.01, zfar=10.0):
        """K in the OpenCV convention (pixel centres at integers) for the final image; `scale` = supersampling
        factor of this renderer. pyrender measures from pixel edges, hence the +0.5."""
        cam = pyrender.IntrinsicsCamera(K[0, 0] * scale, K[1, 1] * scale, (K[0, 2] + 0.5) * scale,
                                        (K[1, 2] + 0.5) * scale, znear=znear, zfar=zfar)
        pose = T_world_cam_cv @ CV_TO_GL
        if self.cam_node is None:
            self.cam_node = self.scene.add(cam, pose=pose)
        else:
            self.cam_node.camera = cam
            self.scene.set_pose(self.cam_node, pose)

    def add_light(self, direction=(0.3, -0.4, -1.0), intensity=3.0):
        d = np.asarray(direction, float); d /= np.linalg.norm(d)
        z = -d; x = np.cross([0, 1.0, 0], z); x /= np.linalg.norm(x)
        self.scene.add(pyrender.DirectionalLight(intensity=intensity), pose=make_T(np.stack([x, np.cross(z, x), z], 1)))

    # ---- static world -------------------------------------------------------------------
    def add_table(self, model, size=(1.2, 0.8), offset=(0.0, 0.15), color=(0.22, 0.24, 0.27, 1.0), tag=True):
        box = trimesh.creation.box(extents=[size[0], size[1], 0.02]); box.apply_translation([offset[0], offset[1], -0.0101])
        self.scene.add(pyrender.Mesh.from_trimesh(box, material=_mat(color), smooth=False))
        if tag:
            t = model.table_tag
            img = model.tag_image(t)
            corners = t.corners(half=t.size / 2 + t.margin) + [0, 0, 0.0003]
            self.scene.add(textured_quad(corners, img))

    def add_mesh(self, mesh, rgba, pose=np.eye(4), alpha_blend=False, smooth=False):
        return self.scene.add(pyrender.Mesh.from_trimesh(mesh, material=_mat(rgba, alpha_blend=alpha_blend), smooth=smooth),
                              pose=pose)

    def add_lines(self, pts, rgba, pose=np.eye(4)):
        """Polyline (N,3) as GL_LINE_STRIP."""
        prim = pyrender.Primitive(positions=np.asarray(pts, np.float32), mode=3,
                                  material=_mat(rgba), color_0=np.tile(np.asarray(rgba, np.float32), (len(pts), 1)))
        return self.scene.add(pyrender.Mesh([prim]), pose=pose)

    def remove(self, node):
        if node is not None and self.scene.has_node(node):
            self.scene.remove_node(node)

    # ---- gripper ------------------------------------------------------------------------
    def add_gripper(self, key, model, style="textured", rgba=None, with_gopro=True, with_hand=False, meshes=None):
        """style 'textured': PLA/TPU parts + ArUco stickers (sim). style 'flat': one colour (preview ghost)."""
        meshes = meshes or load_meshes(with_gopro)
        nodes = {}
        for name, m in meshes.items():
            if style == "textured":
                c = TPU if name.startswith("soft_finger") else ((0.1, 0.1, 0.1, 1.0) if name == "camera_dummy" else PLA)
            else:
                c = rgba
            nodes[name] = self.scene.add(pyrender.Mesh.from_trimesh(m, material=_mat(c, alpha_blend=c[3] < 1), smooth=False))
        tags = {}
        if style == "textured":
            for t in model.body_tags + model.jaw_tags:
                q = textured_quad(t.corners(half=t.size / 2 + t.margin), model.tag_image(t))
                tags[t.id] = (self.scene.add(q), t)
                if t.plate > 0:      # backing flag (candidate mount, not in the CAD)
                    e = t.size + 2 * t.margin
                    box = trimesh.creation.box(extents=[e + 0.002, e + 0.002, t.plate])
                    box.apply_transform(make_T(np.stack([t.right, t.up, t.normal], 1),
                                               t.center - t.normal * (t.plate / 2 + 0.0002)))
                    tags[f"plate{t.id}"] = (self.scene.add(pyrender.Mesh.from_trimesh(box, material=_mat(PLA))), t)
        hand = {}
        if with_hand:
            static, fingers = hand_meshes(model)
            hand["static"] = self.add_mesh(static, SKIN, smooth=True)
            for side, m in fingers.items():
                hand[side] = self.add_mesh(m, SKIN, smooth=True)
        self.grippers[key] = dict(model=model, nodes=nodes, tags=tags, hand=hand)
        return self.grippers[key]

    def set_gripper(self, key, T_world_body, travel, visible=True):
        g = self.grippers[key]; model = g["model"]
        parts = model.part_transforms(travel)
        for name, node in g["nodes"].items():
            self.scene.set_pose(node, T_world_body @ parts.get(name, np.eye(4)))
            node.mesh.is_visible = visible
        for node, t in g["tags"].values():
            shift = np.eye(4) if t.axis is None else make_T(t=travel * t.axis)
            self.scene.set_pose(node, T_world_body @ shift)
            node.mesh.is_visible = visible
        for side, node in g["hand"].items():
            shift = np.eye(4) if side == "static" else make_T(t=travel * model.side_axis[side])
            self.scene.set_pose(node, T_world_body @ shift)
            node.mesh.is_visible = visible

    # ---- render -------------------------------------------------------------------------
    def render(self, flags=0):
        color, depth = self.r.render(self.scene, flags=flags | pyrender.RenderFlags.RGBA)
        return color, depth

    def render_seg(self, node_colors):
        """node_colors: {node: (r,g,b)} -> HxWx3 uint8 label image (others black)."""
        color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.SEG, seg_node_map=node_colors)
        return color

    def close(self):
        try:
            self.r.delete()
        except Exception:      # pyrender/EGL: deleting a second renderer in one process can fail; harmless at exit
            pass
