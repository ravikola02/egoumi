"""UMI two-finger (pinch) variant, v1.

Changes vs. Stanford UMI gripper:
  - handle: trigger stem cut off at the bottom-plate underside; the slider stays,
    so linkages + gears still synchronise both jaws and the spring still opens them.
  - grip: removed, replaced by `hand_mount` (back-of-hand plate + strap slots) on the
    same 4 M3 holes.
  - finger_holder_left/right: a finger ring is merged onto the underside
    (thumb -> left ring, index -> right ring for a right hand, palm down).
All other parts unchanged. Units mm, frame of the original assembly
(+Y = finger direction, X = jaw travel, +Z = camera up).
"""
import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
import os
import cadquery as cq
from inspect_asm import load_parts

# ---------------- parameters ----------------
RING_ID = 22.0          # finger ring inner diameter (adult index/thumb; adjust or add TPU liner)
RING_WALL = 3.0
RING_Y = (90.0, 110.0)  # ring extent along Y (clear of finger screw pockets at y~83 and y~118)
RING_X_OPEN = 56.0      # |x| of ring centre with jaws fully open
HOLDER_BOTTOM = -21.4
WEB_H = 4.0             # web between holder underside and ring top
STEM_CUT_Z = -6.4       # handle cut plane (bottom plate underside is -6.5)
MOUNT = dict(x=40.0, y0=-8.0, y1=66.0, t=5.0, z_top=-6.5)
STRAP_W, STRAP_SLOT_T = 27.0, 4.0  # for 25 mm velcro
OUT = os.path.join(_ROOT, "out_v1")
os.makedirs(OUT, exist_ok=True)

P = {}
fingers = []
for nm, s in load_parts(os.path.join(_ROOT, "input", "vis_assembly.step")):
    if nm == "soft_gripper_finger":
        fingers.append(s)
    else:
        P[nm] = s

# ---------------- handle: cut stem ----------------
keep = cq.Solid.makeBox(400, 400, 200, cq.Vector(-200, -200, STEM_CUT_Z))
slider = P["handle"].intersect(keep).clean()

# ---------------- finger ring ----------------
ro = RING_ID / 2 + RING_WALL
zc = HOLDER_BOTTOM - WEB_H - ro + 1.0  # 1 mm overlap of ring top into web
L = RING_Y[1] - RING_Y[0]


def ring_at(xc):
    ring = (cq.Workplane("XZ", origin=(0, RING_Y[1], 0))
            .center(xc, zc).circle(ro).circle(RING_ID / 2).extrude(L))
    try:
        ring = ring.faces("<Y or >Y").edges().fillet(1.2)
    except Exception:
        pass
    web = cq.Workplane("XY").box(2 * ro * 0.75, L, (HOLDER_BOTTOM + 0.5) - (zc + ro - 1.0),
                                 centered=(True, False, False)) \
        .translate((xc, RING_Y[0], zc + ro - 1.0))
    return ring.union(web).val()


holder_L = P["finger_holder_left"].fuse(ring_at(-RING_X_OPEN)).clean()
holder_R = P["finger_holder_right"].fuse(ring_at(+RING_X_OPEN)).clean()

# ---------------- back-of-hand mount (replaces grip) ----------------
m = MOUNT
mount = (cq.Workplane("XY", origin=(0, 0, m["z_top"] - m["t"]))
         .center(0, (m["y0"] + m["y1"]) / 2)
         .rect(2 * m["x"], m["y1"] - m["y0"]).extrude(m["t"]))
mount = mount.edges("|Z").fillet(6).faces("<Z").edges().fillet(2)
holes = [(-26, 3), (26, 3), (-30, 61), (30, 61)]
mount = (mount.faces("<Z").workplane(centerOption="ProjectedOrigin", origin=(0, 0, 0))
         .pushPoints([(x, -y) for x, y in holes])  # <Z workplane mirrors Y
         .cboreHole(3.4, 6.5, 3.2))
# strap slots
for sx in (-1, 1):
    slot = cq.Workplane("XY").box(STRAP_SLOT_T, STRAP_W, 20).translate((sx * (m["x"] - 6), 30, m["z_top"]))
    mount = mount.cut(slot)
# clearance pocket under slider slot
pocket = cq.Workplane("XY").box(16, 32, 3.0).translate((0, 42, m["z_top"]))
mount = mount.cut(pocket)
hand_mount = mount.val()

parts = {
    "bottom_plate": P["bottom_plate"], "top_cover": P["top_cover"], "cube": P["cube"],
    "camera_dummy": P["Part 3"],
    "gear_left": P["gear_left"], "gear_right": P["gear_right"],
    "linkage_left": P["linkage_left"], "linkage_right": P["linkage_right"],
    "slider": slider, "hand_mount": hand_mount,
    "finger_holder_left_ring": holder_L, "finger_holder_right_ring": holder_R,
    "soft_finger_left": fingers[0], "soft_finger_right": fingers[1],
}

if __name__ == "__main__":
    import pickle
    for nm, s in parts.items():
        print(f"{nm:26s} valid={s.isValid()} vol={s.Volume():9.0f}")
    asm = cq.Assembly(name="umi_pinch_v1")
    for nm, s in parts.items():
        asm.add(s, name=nm)
    asm.save(os.path.join(OUT, "umi_pinch_v1_assembly.step"))
    for nm in ("slider", "hand_mount", "finger_holder_left_ring", "finger_holder_right_ring"):
        cq.exporters.export(parts[nm], os.path.join(OUT, f"{nm}.step"))
        cq.exporters.export(parts[nm], os.path.join(OUT, f"{nm}.stl"), tolerance=0.05, angularTolerance=0.1)
    print("ring centre z", zc, "ring OD", 2 * ro)
