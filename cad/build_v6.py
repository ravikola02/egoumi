"""UMI pinch v6: screwless finger mounting on Akshat's drop-in fit, with one TPU finger that fits both our holder
and Akshat's mount, and a removable tag cap that carries the jaw ArUco tag. No screws anywhere on the holder.

The joint is copied from the files Akshat sent (UMI_Claw_Print.stl, PLA_Rigid_Mount_Reinforced.stl): the finger's base
is a tenon whose outer edge has two dovetail sockets; the mount has a back slab and a wall with two matching dovetail
tabs. The finger drops in from above, and the tabs lock it in x and y. The outlines below are his, measured from
the STLs (claw sits in the mount at (-3.76, +11.85) mm; zero clearance across, about 0.3 mm along).

  * common TPU finger (one per side, mirror images): the UMI fin-ray blade from the holder's front forward, a 2 mm
    solid flange at its base, and Akshat's tenon behind it: same outline, 24 mm tall, inset 0.9 mm top and bottom of
    the 25.8 mm blade, with his 2.5 x 10 mm slot for the cross pin of his mount. Tip, inner face and height are those
    of the UMI finger, so it fits his mount where his claw does and our holder where the UMI finger did.
  * holder (one per side, mirror images): the Stanford finger holder with its top plate cut off over the finger
    (from y = Y_SPLIT forward), the bolt holes and nut pockets filled, Akshat's back slab and tabbed wall built
    between the plates, a floor under the tenon at his height (0.9 mm above the blade's underside), and the v5
    dovetail rail underneath (ring mounts unchanged);
  * tag cap (one per side): the cut-off piece of top plate. It does the job of his cross pin: its tongue goes into a
    slot in the holder's top plate, then the cap is swung down and a snap arm at its outer edge clicks under the
    bottom plate. A pad on its underside squeezes the tenon PAD_SQUEEZE mm, so nothing rattles. The jaw tag sticker
    sits in a recess on top whose floor is at the height the marker config already uses (no config change).
    Release: push the snap arm's tip outward from below, swing the cap up, pull it forward, lift the finger out.

Finger tip, jaw travel, tag pose and the outer envelope are those of v3 (the finger sits 0.2 mm lower, centred in
the 26.2 mm gap; the right side is an exact mirror, so its finger is 0.57 mm further out than the Stanford one).
Units mm, assembly frame (+Y fingers, X jaw travel, +Z up). The fits are Akshat's (zero clearance across the
dovetails) and the cap's (CLEAR, PAD_SQUEEZE, snap lip); untested on our printer: print one holder, cap and finger first.
Run: cd cad && ../.venv-cad/bin/python build_v6.py   -> out_v6/ (STEP, STL), ../tmp_print/ (3MF, git-ignored)
"""
import os
import sys
import numpy as np
import cadquery as cq
import build_v2 as V2
import build_v5 as B5          # dovetail rail and ring mounts (unchanged)

_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_ROOT, "out_v6")
PRINT_DIR = os.path.join(_ROOT, "..", "tmp_print")

# ---------------- Akshat's joint, in his own coordinates (mm) ----------------
# claw: blade z 0..25.8, tenon z 0.9..24.9; blade starts at y 41.15; inner (gripping) face x 36.96; tip y 119.91
AK_TENON = [(14.76, 0.00), (36.96, 0.00), (36.96, 41.15), (18.33, 41.15), (18.33, 37.00), (11.43, 37.00),
            (11.43, 32.83), (14.43, 33.77), (14.43, 26.23), (11.43, 27.17), (11.43, 23.00), (14.76, 23.00),
            (14.76, 18.84), (19.76, 20.50), (19.76, 2.50), (14.76, 4.16)]
AK_TENON_Z = (0.9, 24.9)
AK_BLADE_Y0 = 41.15
AK_SLOT = dict(x=(19.76, 29.76), y=(10.25, 12.75), z=(11.65, 14.15))   # cross-pin slot, open into the rear socket
# mount: back slab + tabbed wall above its floor (z 9 .. 33.3); claw = mount + (3.76, -11.85), tenon rests at z 9.0
AK_WALL = [(0.00, 0.00), (32.70, 0.00), (32.70, 11.85), (11.00, 11.85), (11.00, 16.50), (16.00, 14.68),
           (16.00, 32.32), (11.00, 30.50), (11.00, 35.00), (7.67, 35.00), (7.67, 39.50), (10.67, 38.41),
           (10.67, 45.59), (7.67, 44.50), (7.67, 49.15), (14.31, 49.15), (14.31, 52.95), (0.00, 52.95)]
AK_MOUNT_TO_CLAW = (3.76, -11.85)

# ---------------- our frame (LEFT side; the right side is its mirror image) ----------------
Z_LOW_TOP, Z_LOW_BOT = -15.5, V2.HOLDER_BOTTOM     # bottom plate, top and underside (-21.4)
Z_TOP_BOT, Z_TOP_TOP = 10.7, 16.6                  # top plate, underside and top
Z_BLADE0 = -15.3                                   # finger underside: 25.8 blade centred in the 26.2 mm gap
CLAW_TO_US = (-40.0 - 36.96, 201.35 - 119.91, Z_BLADE0)   # inner face -> x -40, tip -> UMI tip y 201.35
Y_J = AK_BLADE_Y0 + CLAW_TO_US[1]                  # blade start, 122.59 (holder front face 123.3)
FLANGE = 2.0                                       # solid base of the blade (Akshat's claw has one too)
Y_SPLIT = 76.0                                     # cap = top plate from here forward (the jaw tag starts at y 77.5)
CLEAR = 0.15                                       # per-face clearance of the cap fits
FILL_R_LOW, FILL_R_TOP = 3.7, 3.6                  # fill the Stanford bolt holes and nut pockets (plate edge is 4.0 away)
PAD_SQUEEZE = 0.1
PAD = dict(x_outer=-57.0, x_inner=-40.5, y0=82.0, y1=122.0)   # over the tenon only (outer edge clear of the tabs)
TONGUE = dict(w=34.0, depth=4.0, z0=12.6, z1=15.0)  # cap tongue into the holder's top plate
SLOT_EXTRA_UP = 0.3                                # extra slot height above the tongue so the cap can go in tilted
SNAP = dict(y0=88.0, y1=100.0, t=2.5, lip=1.3, ramp=2.0)   # snap arm outside the bottom plate's outer edge
TAG = dict(size=20.4, depth=0.2, raise_=0.2)       # 20 mm sticker recess; cap top raised 0.2 so the sticker face
                                                   # (0.2 sticker) lands at 16.8 = config centre z 16.6 + 0.2 offset
TAG_XY = (-60.0, 87.5)                             # configs/gripper_markers.yaml jaw tag 6 (left; tag 7 = mirror)
STANFORD_BOLTS = ((-75.0, 83.0), (-74.4, 118.0), (-45.0, 82.5))
X_OUT, X_IN = -80.0, -40.0                         # holder outer / inner edge over the finger


def box(x0, x1, y0, y1, z0, z1):
    return cq.Solid.makeBox(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0), cq.Vector(min(x0, x1), min(y0, y1), min(z0, z1)))


def cyl_z(x, y, r, z0, z1):
    return cq.Solid.makeCylinder(r, z1 - z0, cq.Vector(x, y, z0), cq.Vector(0, 0, 1))


def claw_pt(x, y):
    return (x + CLAW_TO_US[0], y + CLAW_TO_US[1])


def mount_pt(x, y):
    return claw_pt(x + AK_MOUNT_TO_CLAW[0], y + AK_MOUNT_TO_CLAW[1])


def prism(pts, z0, z1):
    return cq.Workplane("XY").polyline(pts).close().extrude(z1 - z0).val().translate(cq.Vector(0, 0, z0))


def make_finger():
    """common TPU finger, left side: UMI blade from Y_J forward + flange + Akshat's tenon (with his slot)."""
    umi = V2.fingers[0].translate(cq.Vector(0, 0, Z_BLADE0 - V2.fingers[0].BoundingBox().zmin))
    blade = umi.intersect(box(-200, 200, Y_J, 400, -100, 100)).clean()
    # flange: the blade's cross-section at its base with the fin-ray cells filled
    sl = umi.intersect(box(-200, 200, Y_J, Y_J + FLANGE, -100, 100))
    faces = [f for f in sl.Faces() if abs(f.Center().y - Y_J) < 1e-6 and abs(abs(f.normalAt().y) - 1) < 1e-6]
    flange = None
    for f in faces:
        sol = cq.Solid.extrudeLinear(cq.Face.makeFromWires(f.outerWire()), cq.Vector(0, FLANGE, 0))
        flange = sol if flange is None else flange.fuse(sol)
    tpts = [claw_pt(x, y + (0.4 if abs(y - AK_BLADE_Y0) < 1e-6 else 0.0)) for x, y in AK_TENON]   # 0.4 into the flange
    tenon = prism(tpts, Z_BLADE0 + AK_TENON_Z[0], Z_BLADE0 + AK_TENON_Z[1])
    (sx0, sy0), (sx1, sy1) = claw_pt(AK_SLOT["x"][0], AK_SLOT["y"][0]), claw_pt(AK_SLOT["x"][1], AK_SLOT["y"][1])
    slot = box(sx0 - 0.01, sx1, sy0, sy1, Z_BLADE0 + AK_SLOT["z"][0], Z_BLADE0 + AK_SLOT["z"][1])
    return blade.fuse(flange, tenon).cut(slot).clean()


def make_holder_and_cap():
    """left side."""
    H = V2.P["finger_holder_left"]
    H = H.fuse(*[cyl_z(x, y, FILL_R_LOW, Z_LOW_BOT, Z_LOW_TOP) for x, y in STANFORD_BOLTS],
               *[cyl_z(x, y, FILL_R_TOP, Z_TOP_BOT, Z_TOP_TOP) for x, y in STANFORD_BOLTS]).clean()
    cap_zone = box(-200, 200, Y_SPLIT, 200, Z_TOP_BOT, 50)
    cap = H.intersect(cap_zone).clean()
    holder = H.cut(cap_zone).fuse(B5.rail_L).clean()

    # Akshat's back slab and tabbed wall between the plates, the outer strip out to our holder edge, and the floor
    wall_pts = [mount_pt(x, y) for x, y in AK_WALL]
    y_wall_front = wall_pts[-2][1]
    wall = prism(wall_pts, Z_LOW_TOP - 0.1, Z_TOP_BOT).intersect(box(-200, 200, Y_SPLIT, 400, -100, 100))
    outer = box(X_OUT, wall_pts[0][0] + 0.01, Y_SPLIT, y_wall_front, Z_LOW_TOP - 0.1, Z_TOP_BOT)
    floor = box(wall_pts[0][0], X_IN, Y_SPLIT, y_wall_front, Z_LOW_TOP - 0.1, Z_BLADE0 + AK_TENON_Z[0])
    holder = holder.fuse(wall, outer, floor).clean()
    xc = (X_IN + X_OUT) / 2
    tw, td = TONGUE["w"], TONGUE["depth"]
    holder = holder.cut(box(xc - tw / 2 - CLEAR, xc + tw / 2 + CLEAR, Y_SPLIT - td - CLEAR, Y_SPLIT + 1.0,
                            TONGUE["z0"] - CLEAR, TONGUE["z1"] + CLEAR + SLOT_EXTRA_UP)).clean()

    # cap: raised top, tag recess, tongue, pad, snap arm (+ bridge to the plate)
    raised = cap.intersect(box(-200, 200, -200, 200, Z_TOP_TOP - 1.0, Z_TOP_TOP)).translate(cq.Vector(0, 0, TAG["raise_"]))
    cap = cap.fuse(raised).clean()
    tx, ty = TAG_XY
    h = TAG["size"] / 2
    cap = cap.cut(box(tx - h, tx + h, ty - h, ty + h, Z_TOP_TOP + TAG["raise_"] - TAG["depth"], 50))
    tongue = box(xc - tw / 2, xc + tw / 2, Y_SPLIT - td, Y_SPLIT + 0.5, TONGUE["z0"], TONGUE["z1"])
    try:   # chamfer the tongue's leading edges so it finds the slot when tilted
        tongue = tongue.chamfer(0.6, None, [e for e in tongue.Edges() if abs(e.Center().y - (Y_SPLIT - td)) < 1e-6
                                            and e.geomType() == "LINE" and abs(e.Center().x - xc) < 1e-6])
    except Exception as ex:
        print("tongue chamfer skipped:", ex)
    z_tenon_top = Z_BLADE0 + AK_TENON_Z[1]
    pad = box(PAD["x_outer"], PAD["x_inner"], PAD["y0"], PAD["y1"], z_tenon_top - PAD_SQUEEZE, Z_TOP_BOT + 0.5)
    a0, a1 = X_OUT - CLEAR * 0.7, X_OUT - CLEAR * 0.7 - SNAP["t"]          # arm faces (inner, outer)
    z_lip_top = Z_LOW_BOT - 0.05
    arm = box(a0, a1, SNAP["y0"], SNAP["y1"], z_lip_top - SNAP["ramp"], Z_TOP_BOT + 0.5)
    lip = B5._wp(SNAP["y1"]).polyline([(a0, z_lip_top), (X_OUT + SNAP["lip"], z_lip_top), (a0, z_lip_top - SNAP["ramp"])]) \
        .close().extrude(SNAP["y1"] - SNAP["y0"]).val()
    bridge = box(X_OUT + 0.5, a1, SNAP["y0"], SNAP["y1"], Z_TOP_BOT, Z_TOP_TOP)
    cap = cap.fuse(tongue, pad, arm, lip, bridge).clean()
    return holder, cap


finger_L = make_finger()
holder_L, cap_L = make_holder_and_cap()
built = {+1: (holder_L, cap_L, finger_L), -1: (holder_L.mirror("YZ"), cap_L.mirror("YZ"), finger_L.mirror("YZ"))}
SIDES = {+1: "left", -1: "right"}
parts6 = {n: s for n, s in V2.parts.items() if n not in ("finger_holder_left_thumb", "finger_holder_right_index",
                                                         "soft_finger_left", "soft_finger_right", "top_cover")}
parts6["top_cover"] = B5.top_cover
for s, nm in SIDES.items():
    holder, cap, finger = built[s]
    parts6[f"finger_holder_{nm}"] = holder
    parts6[f"tag_cap_{nm}"] = cap
    parts6[f"tpu_finger_{nm}"] = finger
NEW6 = ("finger_holder_left", "finger_holder_right", "tag_cap_left", "tag_cap_right")
FINGERS6 = ("tpu_finger_left", "tpu_finger_right")
MOVING = {"finger_holder_left": +1, "tag_cap_left": +1, "tpu_finger_left": +1,
          "finger_holder_right": -1, "tag_cap_right": -1, "tpu_finger_right": -1}


def posed(T):
    return {n: s.translate(cq.Vector(MOVING.get(n, 0) * T, 0, 0)) for n, s in parts6.items()}


inter = B5.inter


def section_xy(shape, z):
    """area-weighted outline of a solid's section at height z, as a cq face list"""
    return shape.intersect(box(-300, 300, -300, 400, z - 0.005, z + 0.005))


def akshat_match():
    """our tenon and pocket against his, in his coordinates: the area where they differ (should be ~0)."""
    def area_diff(a, b):
        return a.cut(b).Volume() / 0.01 + b.cut(a).Volume() / 0.01
    zt = Z_BLADE0 + 12.0
    ours_tenon = section_xy(finger_L, zt).intersect(box(-300, 300, -300, Y_J, -100, 100))
    his_tenon = prism([claw_pt(x, y) for x, y in AK_TENON], zt - 0.005, zt + 0.005).intersect(box(-300, 300, -300, Y_J, -100, 100))
    his_tenon = his_tenon.cut(box(*[v for p in zip(claw_pt(AK_SLOT["x"][0], AK_SLOT["y"][0]), claw_pt(AK_SLOT["x"][1], AK_SLOT["y"][1])) for v in p], -100, 100)) \
        if AK_SLOT["z"][0] <= 12.0 + 0 <= AK_SLOT["z"][1] else his_tenon
    zw = 0.0
    ours_wall = section_xy(holder_L, zw).intersect(box(mount_pt(0, 0)[0], X_IN, Y_SPLIT, mount_pt(0, 52.95)[1], -100, 100))
    his_wall = prism([mount_pt(x, y) for x, y in AK_WALL], zw - 0.005, zw + 0.005).intersect(box(-300, 300, Y_SPLIT, 400, -100, 100))
    return area_diff(ours_tenon, his_tenon), area_diff(ours_wall, his_wall)


def check():
    ok = True
    dt, dw = akshat_match()
    print(f"match to Akshat's files: tenon outline differs by {dt:.2f} mm2, wall outline by {dw:.2f} mm2 (0 = identical)")
    ok &= dt < 1.0 and dw < 1.0
    for s, nm in SIDES.items():
        holder, cap, finger = built[s]
        seated = {"cap x holder": inter(cap, holder), "holder x finger": inter(holder, finger)}
        sq = inter(cap, finger)
        pad_area = (PAD["x_inner"] - PAD["x_outer"]) * (PAD["y1"] - PAD["y0"])
        print(f"{nm}: seated overlaps " + ", ".join(f"{k} {v:.2f}" for k, v in seated.items())
              + f" | pad squeezes the tenon {sq:.1f} mm3 (expected {PAD_SQUEEZE * pad_area:.0f})")
        # Akshat's own fit: one face of each dovetail tab sits 0.03-0.19 mm inside the claw's socket (his claw overlaps his
        # mount the same way), the opposite face has ~0.3 mm play. Kept as he designed it: about 12 mm3 per finger.
        ok &= seated["cap x holder"] < 0.05 and seated["holder x finger"] < 15.0
        ok &= abs(sq - PAD_SQUEEZE * pad_area) < 0.2 * PAD_SQUEEZE * pad_area
        lock = {lab: inter(cap.translate(cq.Vector(*v)), holder) for lab, v in
                (("down 0.5", (0, 0, -0.5)), ("back 0.5", (0, -0.5, 0)), ("up 0.5", (0, 0, 0.5)),
                 ("inward 0.5", (s * 0.5, 0, 0)), ("outward 0.5", (-s * 0.5, 0, 0)))}
        print(f"{nm}: cap moved " + ", ".join(f"{k} -> {v:.1f}" for k, v in lock.items())
              + "  (forward: held by the snap arm and tongue together; it can only leave by swinging up)")
        ok &= all(v > 0.05 for v in lock.values())
        # the dovetails hold the finger in every horizontal direction; the 0.3 mm lengthwise play of Akshat's fit is
        # allowed, so the finger is moved 0.5 mm
        flock = {lab: inter(finger.translate(cq.Vector(*v)), holder) for lab, v in
                 (("forward 0.5", (0, 0.5, 0)), ("back 0.5", (0, -0.5, 0)), ("inward 0.5", (s * 0.5, 0, 0)),
                  ("outward 0.5", (-s * 0.5, 0, 0)), ("down 0.5", (0, 0, -0.5)), ("up 0.5 (cap)", (0, 0, 0.5)))}
        held = {k: v + (inter(finger.translate(cq.Vector(0, 0, 0.5)), cap) if k.startswith("up") else 0) for k, v in flock.items()}
        print(f"{nm}: finger moved " + ", ".join(f"{k} -> {v:.1f}" for k, v in held.items()))
        ok &= all(v > 0.05 for v in held.values())
        print(f"{nm}: valid holder {holder.isValid()} ({len(holder.Solids())} solid), cap {cap.isValid()} ({len(cap.Solids())}), "
              f"finger {finger.isValid()} ({len(finger.Solids())})")
        ok &= all(x.isValid() and len(x.Solids()) == 1 for x in (holder, cap, finger))
    # cap and finger against every other part over the jaw travel (the Stanford holder bodies already overlap the
    # unanimated gears / linkages / slider in this pose model, as in v2, so the holder body itself is not re-tested)
    for T in (0.0, 12.0, 24.0, V2.T_CLOSE, 38.0):
        S = posed(T)
        hits = []
        for s, nm in SIDES.items():
            for lab in (f"tag_cap_{nm}", f"tpu_finger_{nm}"):
                for m, o in S.items():
                    if m in (f"tag_cap_{nm}", f"finger_holder_{nm}", f"tpu_finger_{nm}"):
                        continue
                    v = inter(S[lab], o)
                    if v > 0.05:
                        hits.append(f"{lab} x {m}: {v:.1f}")
        ok &= not hits
        print(f"T={T:4.1f}:", hits if hits else "no collisions")
    tips = posed(V2.T_CLOSE)
    v3 = V2.fingers[0].translate(cq.Vector(V2.T_CLOSE, 0, 0)).distance(V2.fingers[1].translate(cq.Vector(-V2.T_CLOSE, 0, 0)))
    print(f"finger gap at {V2.T_CLOSE} mm travel per side: {tips['tpu_finger_left'].distance(tips['tpu_finger_right']):.2f} mm "
          f"(stock v3 fingers: {v3:.2f} mm; the difference is the right side being an exact mirror)")
    # ring mounts on the new holders
    for nm, (shape, sg) in B5.ring_mounts.items():
        h = parts6["finger_holder_left" if sg > 0 else "finger_holder_right"]
        ov = {lab: inter(shape.translate(cq.Vector(*v)), h) for lab, v in
              (("seated", (0, 0, 0)), ("fwd", (0, 1, 0)), ("back", (0, -1, 0)), ("down", (0, 0, -1)), ("side", (1, 0, 0)))}
        good = ov["seated"] < 0.05 and all(v > 0.05 for k, v in ov.items() if k != "seated")
        hit = inter(shape, parts6["tag_cap_left" if sg > 0 else "tag_cap_right"]) + inter(shape, parts6["tpu_finger_left" if sg > 0 else "tpu_finger_right"])
        print(f"{nm:36s} on v6 holder: " + ", ".join(f"{k} {v:.1f}" for k, v in ov.items()) + f", x cap/finger {hit:.1f}"
              + ("" if good and hit < 0.05 else "  <-- PROBLEM"))
        ok &= good and hit < 0.05
    for pair in (B5.RING_RH, B5.RING_SMALL_RH):
        for T in (0.0, 18.0, V2.T_CLOSE):
            S = posed(T)
            for nm in pair:
                shape, sg = B5.ring_mounts[nm]
                a = shape.translate(cq.Vector(sg * T, 0, 0))
                side = "left" if sg > 0 else "right"
                hits = [f"{m} {inter(a, o):.1f}" for m, o in S.items()
                        if m not in (f"finger_holder_{side}", f"tag_cap_{side}") and inter(a, o) > 0.05]
                if hits:
                    ok = False
                    print(f"{nm} T={T}: {hits}")
    print("ring mounts over the travel: done (any collision is listed above)")
    return ok


def orient_side(V3, V, F):
    """holders: lie on a side (less overhang than top-down, and the walls print without bridging over the pocket)."""
    best = None
    for name, R in V3.ROTS.items():
        if abs((R @ np.array([0, 0, 1.0]))[2]) > 1e-6:
            continue
        W = V @ R.T
        a, b, c = W[F[:, 0]], W[F[:, 1]], W[F[:, 2]]
        n = np.cross(b - a, c - a)
        area = np.linalg.norm(n, axis=1) / 2
        nz = n[:, 2] / (2 * area + 1e-12)
        tz = np.maximum(np.maximum(a[:, 2], b[:, 2]), c[:, 2])
        on_bed = (nz < -0.999) & (tz < W[:, 2].min() + 0.05)
        ov, bed = area[(nz < -0.7071) & ~on_bed].sum(), area[on_bed].sum()
        if bed > 100 and (best is None or ov < best[2]):
            best = (name, R, ov, bed)
    return best


def export():
    import build_v3 as V3
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PRINT_DIR, exist_ok=True)
    for nm in NEW6 + FINGERS6:
        shape = parts6[nm]
        cq.exporters.export(shape, os.path.join(OUT, f"{nm}.step"))
        V, F = V3.mesh(shape)
        if nm.startswith("tpu_finger"):              # flat, as the UMI finger: blade underside on the bed
            oname, Rm = "-Z down (as modelled)", np.eye(3)
            ov = bed = 0.0
        else:
            oname, Rm, ov, bed = V3.orient(V, F)
        W = V @ Rm.T
        if np.linalg.det(Rm) < 0:
            F = F[:, ::-1]
        lo, hi = W.min(0), W.max(0)
        W = W - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        bad, vol = V3.mesh_stats(W, F)
        V3.prep[nm] = dict(V=W, F=F, orient=oname, overhang=ov, bed=bed, size=hi - lo, bad_edges=bad, vol=vol)
        V3.OVR[nm] = ({"sparse_infill_density": "100%", "sparse_infill_pattern": "line"} if nm.startswith("tpu_finger")
                      else {"enable_support": "1", "support_type": "tree(auto)"})
        print(f"print {nm:20s} {oname:22s} size={np.round(hi - lo, 1)} overhang={ov:6.0f} mm2 bed={bed:6.0f} mm2 "
              f"open_edges={bad} vol={vol:8.0f}")
        t = W[F].astype("<f4")
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = n, t
        with open(os.path.join(OUT, f"{nm}.stl"), "wb") as fh:
            fh.write(nm.encode().ljust(80, b" ") + np.uint32(len(t)).tobytes() + rec.tobytes())
    for f in ("UMI_v6_holders_caps_PLA.3mf",):
        if os.path.exists(os.path.join(PRINT_DIR, f)):
            os.remove(os.path.join(PRINT_DIR, f))
    V3.write_3mf(os.path.join(PRINT_DIR, "UMI_v6_holders_caps_PLA.3mf"), "UMI_v6_holders_caps_PLA", list(NEW6), V3.pack(list(NEW6)))
    V3.write_3mf(os.path.join(PRINT_DIR, "UMI_v6_fingers_TPU.3mf"), "UMI_v6_fingers_TPU", list(FINGERS6), V3.pack(list(FINGERS6)))
    asm = cq.Assembly(name="umi_pinch_v6")
    for nm, s in parts6.items():
        asm.add(s, name=nm)
    asm.export(os.path.join(OUT, "umi_pinch_v6_assembly.step"))


if __name__ == "__main__":
    ok = check()
    export()
    sys.exit(0 if ok else 1)
