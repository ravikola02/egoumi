"""UMI pinch v2: original palm grip restored + thumb/index rings placed for a pistol-grip hand.

Hand pose (right hand): palm + middle/ring/little fingers wrap the original `grip` post.
Thumb lies along the LEFT side of the post top, index along the RIGHT side, both pointing +Y.
  - thumb pushes +X (adduction across the front of the post)  -> drives LEFT holder inward
  - index pulls  -X (MCP flexion, like a trigger squeeze)       -> drives RIGHT holder inward
Gears + linkages + cut slider keep both jaws synchronised; spring still opens.
For a LEFT-hand user mirror the ring placement (swap holders).

Units mm, original assembly frame: +Y fingers, X jaw travel, +Z camera up.
Hand positions below are ESTIMATES from typical adult proportions - verify with a fit print.
"""
import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
import os
import cadquery as cq
from inspect_asm import load_parts

# ---------------- parameters ----------------
T_CLOSE = 36.0            # holder travel per side until soft fingertips meet (from v1 check)
# index ring (on RIGHT holder), sits on index middle phalanx
IDX = dict(ri=10.0, wall=3.0, x_open=40.0, z=-36.5, y0=62.0, y1=80.0)
# thumb ring (on LEFT holder), sits on thumb distal phalanx, behind the index ring
THB = dict(ri=11.0, wall=3.0, x_open=-50.0, z=-27.0, y0=33.0, y1=49.0)
# thumb arm (left holder underside -> thumb ring); x chosen to clear finger-screw pocket at x<-71.8
ARM = dict(x0=-71.5, x1=-61.5, y0=35.0, y1=112.0, z0=-36.0)
HOLDER_BOTTOM = -21.4
STEM_CUT_Z = -6.4
OUT = os.path.join(_ROOT, "out_v2")
os.makedirs(OUT, exist_ok=True)

P, fingers = {}, []
for nm, s in load_parts(os.path.join(_ROOT, "input", "vis_assembly.step")):
    (fingers.append(s) if nm == "soft_gripper_finger" else P.__setitem__(nm, s))

slider = P["handle"].intersect(
    cq.Solid.makeBox(400, 400, 200, cq.Vector(-200, -200, STEM_CUT_Z))).clean()


def cyl_y(xc, zc, r, y0, y1):
    return cq.Solid.makeCylinder(r, y1 - y0, cq.Vector(xc, y0, zc), cq.Vector(0, 1, 0))


def box(x0, x1, y0, y1, z0, z1):
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0))


def finish_ring(solid, xc, zc, ri, y0, y1):
    """cut the bore and round its entry edges"""
    s = solid.cut(cyl_y(xc, zc, ri, y0 - 5, y1 + 5)).clean()
    edges = [e for e in s.Edges() if e.geomType() == "CIRCLE" and abs(e.radius() - ri) < 1e-3]
    try:
        s = s.fillet(1.2, edges)
    except Exception as ex:
        print("bore fillet skipped:", ex)
    return s


# ---- index ring + web + gusset on RIGHT holder ----
r_o = IDX["ri"] + IDX["wall"]
xi, zi = IDX["x_open"], IDX["z"]
idx = cyl_y(xi, zi, r_o, IDX["y0"], IDX["y1"])
web = box(xi - 2, xi + 10, IDX["y0"], 78.5, zi + IDX["ri"] + 0.5, HOLDER_BOTTOM + 0.6)   # under holder (y>=71.7)
gus = box(xi - 2, xi + 10, IDX["y0"], 72.0, zi + IDX["ri"] + 0.5, -11.5)                 # up the holder back face (y=71.2)
idx = idx.fuse(web).fuse(gus)
idx = finish_ring(idx, xi, zi, IDX["ri"], IDX["y0"], IDX["y1"])
holder_R = P["finger_holder_right"].fuse(idx).clean()

# ---- thumb ring + arm on LEFT holder ----
r_ot = THB["ri"] + THB["wall"]
xt, zt = THB["x_open"], THB["z"]
thb = cyl_y(xt, zt, r_ot, THB["y0"], THB["y1"])
arm = box(ARM["x0"], ARM["x1"], ARM["y0"], ARM["y1"], ARM["z0"], HOLDER_BOTTOM + 0.6)
thb = thb.fuse(arm)
thb = finish_ring(thb, xt, zt, THB["ri"], THB["y0"], THB["y1"])
holder_L = P["finger_holder_left"].fuse(thb).clean()

parts = {
    "bottom_plate": P["bottom_plate"], "top_cover": P["top_cover"], "cube": P["cube"],
    "camera_dummy": P["Part 3"], "grip": P["grip"],
    "gear_left": P["gear_left"], "gear_right": P["gear_right"],
    "linkage_left": P["linkage_left"], "linkage_right": P["linkage_right"],
    "slider": slider,
    "finger_holder_left_thumb": holder_L, "finger_holder_right_index": holder_R,
    "soft_finger_left": fingers[0], "soft_finger_right": fingers[1],
}
MOVING_L = ("finger_holder_left_thumb", "soft_finger_left")
MOVING_R = ("finger_holder_right_index", "soft_finger_right")


def posed(T):
    out = {}
    for n, s in parts.items():
        if n in MOVING_L:
            s = s.translate(cq.Vector(T, 0, 0))
        elif n in MOVING_R:
            s = s.translate(cq.Vector(-T, 0, 0))
        out[n] = s
    return out


if __name__ == "__main__":
    for nm, s in parts.items():
        print(f"{nm:28s} valid={s.isValid()} vol={s.Volume():9.0f}")
    asm = cq.Assembly(name="umi_pinch_v2")
    for nm, s in parts.items():
        asm.add(s, name=nm)
    asm.export(os.path.join(OUT, "umi_pinch_v2_assembly.step"))
    for nm in ("slider", "finger_holder_left_thumb", "finger_holder_right_index"):
        cq.exporters.export(parts[nm], os.path.join(OUT, f"{nm}.step"))
        cq.exporters.export(parts[nm], os.path.join(OUT, f"{nm}.stl"), tolerance=0.05, angularTolerance=0.1)
