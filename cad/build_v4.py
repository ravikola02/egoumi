"""UMI pinch v4 = v3 with the GoPro mount on the top cover replaced by an inverted-L clip bracket for the
DJI Osmo Nano Magnetic Hat Clip (the spring clip in the Osmo Nano kit; the camera snaps magnetically onto it).

Bracket: a vertical post across X standing on the cover, then a flat "brim" running forward (+Y, toward the fingers)
from its top, drooping BRIM_TILT so the clip sits on it the way it sits on a cap brim and the camera looks down at the
jaws (the v3 sim used a 15 deg down pitch). Gussets at both ends (post-to-cover, post-to-brim) leave the middle free.
Preview: `cd cad && ../.venv-cad/bin/python build_v4.py` -> renders/v4_*.png (no print files yet).
"""
import os
import math
import cadquery as cq
import build_v2 as V2

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------- parameters (mm, deg; assembly frame) ----------------
COVER_TOP_Z = 19.8      # top face of the cover plate (bottom z 17.4)
WALL_W = 80.0           # along X (post and brim); plate is 107 mm wide at the post's back edge (y -8.5)
WALL_H = 30.0           # post height above the cover (to the brim's top surface at the back)
WALL_T = 3.5            # post and brim thickness; cap brims are ~2.5-4 mm. UNVERIFIED: measure the clip's jaw range
WALL_Y0 = -5.0          # y of the post's front face
BRIM_D = 45.0           # brim length forward of the post's back face
BRIM_TILT = 15.0        # brim droop = camera pitch down
EDGE_ROUND = 1.2        # brim front edge fillet so the clip slides on
GUSSET = dict(t=4.0, d=22.0, h=28.0)   # base gussets at both post ends (thickness, depth on plate, height)
BRIM_GUSSET = 18.0      # post-to-brim gusset leg length

# ---------------- top cover: v3 flap cut + GoPro mount removed ----------------
cover = V2.parts["top_cover"]
keep_center = cq.Solid.makeBox(70, 400, 200, cq.Vector(-35, -200, COVER_TOP_Z))
cover = cover.cut(cq.Solid.makeBox(400, 400, 200, cq.Vector(-200, -200, COVER_TOP_Z)).cut(keep_center))
# GoPro mount: x -28.6..28.7, y -24..14, up to z 54.1, all above the plate top
cover = cover.cut(cq.Solid.makeBox(70, 60, 100, cq.Vector(-35, -30, COVER_TOP_Z))).clean()

# ---------------- inverted-L bracket ----------------
a = math.radians(BRIM_TILT)
Z_TOP = COVER_TOP_Z + WALL_H
PIVOT = cq.Vector(0, WALL_Y0 - WALL_T, Z_TOP)   # back top edge of the post = back edge of the brim's top surface


def on_brim(shape):
    """brim frame (x across, y forward along the brim, z up from its top surface, origin PIVOT) -> assembly."""
    return shape.rotate(cq.Vector(0, 0, 0), cq.Vector(1, 0, 0), -BRIM_TILT).translate(PIVOT)


post = cq.Solid.makeBox(WALL_W, WALL_T, Z_TOP - COVER_TOP_Z + 1.0, cq.Vector(-WALL_W / 2, WALL_Y0 - WALL_T, COVER_TOP_Z - 1.0))
brim = cq.Solid.makeBox(WALL_W, BRIM_D, WALL_T, cq.Vector(-WALL_W / 2, 0, -WALL_T))
brim = brim.fillet(EDGE_ROUND, [e for e in brim.Edges() if abs(e.Center().y - BRIM_D) < 1e-6 and e.geomType() == "LINE"
                                and abs(e.Center().x) < 1e-6])
brim = on_brim(brim)
below_brim_top = on_brim(cq.Solid.makeBox(400, 400, 400, cq.Vector(-200, -100, -400)))


def gusset(xc):
    t, d, h = GUSSET["t"], GUSSET["d"], GUSSET["h"]
    pts = [(WALL_Y0 - 1.0, COVER_TOP_Z - 1.0), (WALL_Y0 + d, COVER_TOP_Z - 1.0), (WALL_Y0 + d, COVER_TOP_Z),
           (WALL_Y0 - 1.0, COVER_TOP_Z + h)]
    return cq.Workplane("YZ").polyline(pts).close().extrude(t).translate((xc - t / 2, 0, 0)).val()


def brim_gusset(xc):
    g, t = BRIM_GUSSET, GUSSET["t"]
    zb = Z_TOP - WALL_T            # brim underside at the post (approx.; the far corner is buried in the brim)
    pts = [(WALL_Y0 - 1.0, zb - g), (WALL_Y0 - 1.0, zb + 1.0), (WALL_Y0 + g, zb + 1.0 - (g + WALL_T) * math.tan(a))]
    return cq.Workplane("YZ").polyline(pts).close().extrude(t).translate((xc - t / 2, 0, 0)).val()


half = WALL_W / 2 - GUSSET["t"] / 2
bracket = post.fuse(brim, gusset(-half), gusset(half), brim_gusset(-half), brim_gusset(half)).intersect(below_brim_top)
bracket = bracket.intersect(cq.Solid.makeBox(400, 400, 200, cq.Vector(-200, -200, COVER_TOP_Z - 1.0)))
top_cover = cover.fuse(bracket).clean()

parts = dict(V2.parts)
parts["top_cover"] = top_cover

# ---------------- preview-only dummies: Osmo Nano (57.3 x 29.5 x 28) on a hat clip ----------------
# Clip and camera sizes are placeholders; the clip's real geometry is not published. Shown with the clip pushed onto
# the brim's front edge and the camera standing on it, lens forward.
CLIP = dict(w=30.0, d=25.0, jaw=4.0)
u0 = BRIM_D - CLIP["d"]
clip = cq.Solid.makeBox(CLIP["w"], CLIP["d"], CLIP["jaw"], cq.Vector(-CLIP["w"] / 2, u0, 0.1))
clip = clip.fuse(cq.Solid.makeBox(CLIP["w"], CLIP["d"] * 0.7, CLIP["jaw"], cq.Vector(-CLIP["w"] / 2, u0 + CLIP["d"] * 0.3, -WALL_T - 0.1 - CLIP["jaw"])))
clip = clip.fuse(cq.Solid.makeBox(CLIP["w"], 3.0, WALL_T + 0.2 + 2 * CLIP["jaw"], cq.Vector(-CLIP["w"] / 2, BRIM_D + 0.1, -WALL_T - 0.1 - CLIP["jaw"])))
clip = on_brim(clip)
cam_v0 = 0.1 + CLIP["jaw"]
cam = cq.Workplane("XY").box(29.5, 28.0, 57.3, centered=(True, False, False)).edges("|Y").fillet(8).val() \
    .translate(cq.Vector(0, BRIM_D - 28.0, cam_v0))
cam = cam.fuse(cq.Solid.makeCylinder(8.0, 2.0, cq.Vector(0, BRIM_D, cam_v0 + 57.3 - 14.0), cq.Vector(0, 1, 0)))
cam = on_brim(cam)
lens = on_brim(cq.Solid.makeSphere(0.1, cq.Vector(0, BRIM_D + 2.0, cam_v0 + 57.3 - 14.0)))

if __name__ == "__main__":
    from render import render
    print("top_cover valid", top_cover.isValid(), "vol", round(top_cover.Volume()),
          "bbox", [round(v, 1) for v in (top_cover.BoundingBox().ymin, top_cover.BoundingBox().ymax,
                                         top_cover.BoundingBox().zmax)])
    lc = lens.Center()
    print(f"dummy lens centre y {lc.y:.1f} z {lc.z:.1f} (v3 GoPro: y 16, z 83.4), pitch {BRIM_TILT} deg down")
    R = os.path.join(_ROOT, "renders")
    render([("top_cover v4 (inverted-L clip bracket)", top_cover)], os.path.join(R, "v4_cover.png"),
           views=((25, -60), (25, -130), (0, 0)), size=5, title="v4 top cover: GoPro mount -> Osmo Nano inverted-L clip bracket")
    cols = {"top_cover": "#E45756", "finger_holder_left_thumb": "#F58518", "finger_holder_right_index": "#F58518",
            "grip": "#4C78A8", "hat clip (dummy)": "#555555", "Osmo Nano (dummy)": "#BAB0AC"}
    show = [(n, s) for n, s in parts.items() if n != "camera_dummy"] + [("hat clip (dummy)", clip),
                                                                       ("Osmo Nano (dummy)", cam)]
    render(show, os.path.join(R, "v4_assembly.png"), views=((20, -60), (0, 0), (15, -150)), colors=cols,
           title="UMI pinch v4: Osmo Nano on the hat clip, clip on the cover's brim (clip/camera = placeholders)")
