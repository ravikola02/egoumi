"""UMI pinch v6: screwless finger mounting. The stock TPU soft finger drops in from above onto pins, and a
removable tag cap closes the holder over it and carries the jaw ArUco tag. No screws anywhere on the holder.

Adapted from the drop-in fit of the PLA claw and mount Akshat sent (finger base dropped into a matching pocket from
above). Our holder clamps the finger between a top and a bottom plate, so the same idea becomes:

  * holder (one per side): the Stanford finger holder with its top plate cut off over the finger (from y = Y_SPLIT
    forward), the bolt holes and nut pockets filled, three PIN_D pins standing up from the bottom plate at the
    finger's own three 4 mm holes, and the v5 dovetail rail underneath (ring mounts unchanged);
  * TPU soft finger: the stock UMI finger, unchanged (cad/out_v3/soft_finger_*.stl). It drops onto the pins from
    above; the pins hold it in x and y, so pull-out along the finger and the grip force go into the pins;
  * tag cap (one per side): the cut-off piece of top plate. Its tongue goes into a slot in the holder's top plate,
    then the cap is swung down and a snap arm at its outer edge clicks under the bottom plate. A pad on its underside
    squeezes the finger PAD_SQUEEZE mm, so nothing rattles. The jaw tag sticker sits in a recess on top whose floor
    is at the height the marker config already uses, so configs/gripper_markers.yaml does not change.
    Release: push the snap arm's tip outward from below, swing the cap up, pull it forward.

Finger tip, jaw travel, tag pose and the outer envelope are those of v3.
Units mm, assembly frame (+Y fingers, X jaw travel, +Z up). The fits (CLEAR, PIN_D, PAD_SQUEEZE, snap lip) are
untested and printer-dependent: print one holder and one cap first.
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

# ---------------- parameters (mm) ----------------
Z_LOW_TOP, Z_LOW_BOT = -15.5, V2.HOLDER_BOTTOM     # bottom plate, top and underside (-21.4)
Z_TOP_BOT, Z_TOP_TOP = 10.7, 16.6                  # top plate, underside and top
Y_SPLIT = 76.0                                     # cap = top plate from here forward (the jaw tag starts at y 77.5)
Y_FRONT = 123.3                                    # holder front face
CLEAR = 0.15                                       # per-face clearance of the printed fits
PIN_D, PIN_TOP, PIN_CHAMFER = 3.9, 9.9, 0.5        # pins in the finger's 4.0 mm holes; tops stay 0.3 below the pad
FILL_R_LOW, FILL_R_TOP = 3.7, 3.6                  # fill the Stanford bolt holes and nut pockets (plate edge is 4.0 away)
PAD_SQUEEZE = 0.1                                  # finger is 25.8 tall in a 26.2 gap: pad = 0.4 + squeeze
PAD = dict(y0=78.5, y1=87.5, inset=0.5)            # pad over the finger's solid back strip only (the fin ray stays free)
TONGUE = dict(w=34.0, depth=4.0, z0=12.6, z1=15.0)  # cap tongue into the holder's top plate
SLOT_EXTRA_UP = 0.3                                # extra slot height above the tongue so the cap can go in tilted
SNAP = dict(y0=88.0, y1=100.0, t=2.5, lip=1.3, ramp=2.0)   # snap arm outside the bottom plate's outer edge
TAG = dict(size=20.4, depth=0.2, raise_=0.2)       # 20 mm sticker recess; cap top raised 0.2 so the sticker face
                                                   # (0.2 sticker) lands at 16.8 = config centre z 16.6 + 0.2 offset
TAG_XY = {+1: (-60.0, 87.5), -1: (60.0, 87.5)}     # configs/gripper_markers.yaml jaw tags 6 (left) and 7 (right)
STANFORD_BOLTS = ((-75.0, 83.0), (-74.4, 118.0), (-45.0, 82.5))   # left holder; x mirrored on the right


def box(x0, x1, y0, y1, z0, z1):
    return cq.Solid.makeBox(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0), cq.Vector(min(x0, x1), min(y0, y1), min(z0, z1)))


def cyl_z(x, y, r, z0, z1):
    return cq.Solid.makeCylinder(r, z1 - z0, cq.Vector(x, y, z0), cq.Vector(0, 0, 1))


def finger_holes(finger):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    hs = set()
    for f in finger.Faces():
        a = BRepAdaptor_Surface(f.wrapped)
        if a.GetType() == GeomAbs_Cylinder and abs(a.Cylinder().Radius() - 2.0) < 1e-3:
            loc = a.Cylinder().Location()
            hs.add((round(loc.X(), 3), round(loc.Y(), 3)))
    return sorted(hs)


def pin(x, y):
    p = cyl_z(x, y, PIN_D / 2, Z_LOW_TOP - 0.1, PIN_TOP)
    try:
        p = p.chamfer(PIN_CHAMFER, None, [e for e in p.Edges() if e.geomType() == "CIRCLE" and abs(e.Center().z - PIN_TOP) < 1e-6])
    except Exception as ex:
        print("pin chamfer skipped:", ex)
    return p


def build_side(side):
    """side = +1 left (thumb in v3), -1 right. Returns (holder, cap, finger, pins)."""
    s = side
    H = V2.P["finger_holder_left" if s > 0 else "finger_holder_right"]
    finger = V2.fingers[0 if s > 0 else 1]
    rail = B5.rail_L if s > 0 else B5.rail_L.mirror("YZ")
    bolts = [(x if s > 0 else -x, y) for x, y in STANFORD_BOLTS]
    xin, xout = (-40.0, -80.0) if s > 0 else (40.0, 80.0)                      # holder inner / outer edge here
    fb = finger.BoundingBox()

    # fill the old bolt holes and nut pockets (both plates), then split off the cap
    H = H.fuse(*[cyl_z(x, y, FILL_R_LOW, Z_LOW_BOT, Z_LOW_TOP) for x, y in bolts],
               *[cyl_z(x, y, FILL_R_TOP, Z_TOP_BOT, Z_TOP_TOP) for x, y in bolts]).clean()
    cap_zone = box(-200, 200, Y_SPLIT, 200, Z_TOP_BOT, 50)
    cap = H.intersect(cap_zone).clean()
    holder = H.cut(cap_zone).fuse(rail).clean()

    # holder: pins at the finger's holes; slot for the cap tongue
    holes = finger_holes(finger)
    pins = None
    for x, y in holes:
        p = pin(x, y)
        pins = p if pins is None else pins.fuse(p)
    holder = holder.fuse(pins).clean()
    xc = (xin + xout) / 2
    tw, td = TONGUE["w"], TONGUE["depth"]
    slot = box(xc - tw / 2 - CLEAR, xc + tw / 2 + CLEAR, Y_SPLIT - td - CLEAR, Y_SPLIT + 1.0,
               TONGUE["z0"] - CLEAR, TONGUE["z1"] + CLEAR + SLOT_EXTRA_UP)
    holder = holder.cut(slot).clean()

    # cap: raised top, tag recess, tongue, pad, snap arm
    raised = cap.intersect(box(-200, 200, -200, 200, Z_TOP_TOP - 1.0, Z_TOP_TOP)).translate(cq.Vector(0, 0, TAG["raise_"]))
    cap = cap.fuse(raised).clean()
    tx, ty = TAG_XY[s]
    h = TAG["size"] / 2
    cap = cap.cut(box(tx - h, tx + h, ty - h, ty + h, Z_TOP_TOP + TAG["raise_"] - TAG["depth"], 50))
    tongue = box(xc - tw / 2, xc + tw / 2, Y_SPLIT - td, Y_SPLIT + 0.5, TONGUE["z0"], TONGUE["z1"])
    try:   # chamfer the tongue's leading edges so it finds the slot when tilted
        tongue = tongue.chamfer(0.6, None, [e for e in tongue.Edges() if abs(e.Center().y - (Y_SPLIT - td)) < 1e-6 and e.geomType() == "LINE"
                                      and abs(e.Center().x - xc) < 1e-6])
    except Exception as ex:
        print("tongue chamfer skipped:", ex)
    pad_t = (Z_TOP_BOT - Z_LOW_TOP) - (fb.zmax - fb.zmin) + PAD_SQUEEZE
    pad = box(fb.xmin + PAD["inset"], fb.xmax - PAD["inset"], PAD["y0"], PAD["y1"], Z_TOP_BOT - pad_t, Z_TOP_BOT + 0.5)
    sg = -1 if s > 0 else 1                                   # outward direction in x
    xo = xout                                                 # bottom plate outer edge
    a0, a1 = xo + sg * CLEAR * 0.7, xo + sg * (CLEAR * 0.7 + SNAP["t"])        # arm faces (inner, outer)
    z_lip_top = Z_LOW_BOT - 0.05
    arm = box(a0, a1, SNAP["y0"], SNAP["y1"], z_lip_top - SNAP["ramp"], Z_TOP_BOT + 0.5)
    lip = B5._wp(SNAP["y1"]).polyline([(a0, z_lip_top), (xo - sg * SNAP["lip"], z_lip_top), (a0, z_lip_top - SNAP["ramp"])]) \
        .close().extrude(SNAP["y1"] - SNAP["y0"]).val()
    bridge = box(xo - sg * 0.5, a1, SNAP["y0"], SNAP["y1"], Z_TOP_BOT, Z_TOP_TOP)   # joins the arm to the cap plate
    cap = cap.fuse(tongue, pad, arm, lip, bridge).clean()
    return holder, cap, finger, pins


SIDES = {+1: "left", -1: "right"}
built = {s: build_side(s) for s in SIDES}
parts6 = dict(V2.parts)
for k in ("finger_holder_left_thumb", "finger_holder_right_index", "top_cover"):
    parts6.pop(k, None)
parts6["top_cover"] = B5.top_cover
for s, nm in SIDES.items():
    holder, cap, finger, _ = built[s]
    parts6[f"finger_holder_{nm}"] = holder
    parts6[f"tag_cap_{nm}"] = cap
NEW6 = ("finger_holder_left", "finger_holder_right", "tag_cap_left", "tag_cap_right")
MOVING = {"finger_holder_left": +1, "tag_cap_left": +1, "soft_finger_left": +1,
          "finger_holder_right": -1, "tag_cap_right": -1, "soft_finger_right": -1}


def posed(T):
    return {n: s.translate(cq.Vector(MOVING.get(n, 0) * T, 0, 0)) for n, s in parts6.items()}


inter = B5.inter


def seated_finger(finger):
    """the CAD finger floats 0.4 mm under the top plate; in the v6 holder it rests on the bottom plate."""
    return finger.translate(cq.Vector(0, 0, Z_LOW_TOP - finger.BoundingBox().zmin))


def check():
    ok = True
    for s, nm in SIDES.items():
        holder, cap, finger, pins = built[s]
        finger = seated_finger(finger)
        sq = inter(cap, finger)
        pad_area = (finger.BoundingBox().xlen - 2 * PAD["inset"]) * (PAD["y1"] - PAD["y0"])
        seated = {"cap x holder": inter(cap, holder), "pins x finger": inter(pins, finger),
                  "holder x finger": inter(holder, finger)}
        bad = {k: v for k, v in seated.items() if v > 0.05}
        print(f"{nm}: seated overlaps " + ", ".join(f"{k} {v:.2f}" for k, v in seated.items())
              + f" | pad squeezes the finger {sq:.1f} mm3 (expected about {PAD_SQUEEZE * pad_area * 0.8:.0f}, solid strip minus holes)")
        ok &= not bad
        # cap is held: it collides with the holder if moved down, back, sideways or up (lip under the plate)
        lock = {lab: inter(cap.translate(cq.Vector(*v)), holder) for lab, v in
                (("down 0.5", (0, 0, -0.5)), ("back 0.5", (0, -0.5, 0)), ("up 0.5", (0, 0, 0.5)),
                 ("inward 0.5", (s * 0.5, 0, 0)), ("outward 0.5", (-s * 0.5, 0, 0)))}
        print(f"{nm}: cap moved " + ", ".join(f"{k} -> {v:.1f}" for k, v in lock.items())
              + "  (forward is held by the snap arm and tongue together: it can only leave by swinging up)")
        ok &= all(v > 0.05 for v in lock.values())
        # finger held: moved forward, sideways or back it hits the pins / back wall
        flock = {lab: inter(finger.translate(cq.Vector(*v)), holder) for lab, v in
                 (("forward 0.5", (0, 0.5, 0)), ("back 0.5", (0, -0.5, 0)), ("inward 0.5", (s * 0.5, 0, 0)), ("outward 0.5", (-s * 0.5, 0, 0)))}
        print(f"{nm}: finger moved " + ", ".join(f"{k} -> {v:.1f}" for k, v in flock.items()))
        ok &= all(v > 0.05 for v in flock.values())
        print(f"{nm}: valid holder {holder.isValid()} ({len(holder.Solids())} solid), cap {cap.isValid()} ({len(cap.Solids())} solid)")
        ok &= holder.isValid() and cap.isValid() and len(holder.Solids()) == 1 and len(cap.Solids()) == 1
    # new geometry against every other part over the jaw travel (the Stanford holder bodies already overlap the
    # unanimated gears / linkages / slider in this pose model, as in v2, so only the cap and pins are tested)
    for T in (0.0, 12.0, 24.0, V2.T_CLOSE, 38.0):
        S = posed(T)
        hits = []
        for s, nm in SIDES.items():
            cap = S[f"tag_cap_{nm}"]
            pins = built[s][3].translate(cq.Vector(s * T, 0, 0))
            for m, o in S.items():
                if m in (f"tag_cap_{nm}", f"finger_holder_{nm}", f"soft_finger_{nm}"):
                    continue
                for lab, g in (("cap", cap), ("pins", pins)):
                    v = inter(g, o)
                    if v > 0.05:
                        hits.append(f"{lab} {nm} x {m}: {v:.1f}")
        ok &= not hits
        print(f"T={T:4.1f}:", hits if hits else "no collisions")
    # ring mounts on the new holders: seated, locked, slide-in path, travel
    for nm, (shape, sg) in B5.ring_mounts.items():
        h = parts6["finger_holder_left" if sg > 0 else "finger_holder_right"]
        ov = {lab: inter(shape.translate(cq.Vector(*v)), h) for lab, v in
              (("seated", (0, 0, 0)), ("fwd", (0, 1, 0)), ("back", (0, -1, 0)), ("down", (0, 0, -1)), ("side", (1, 0, 0)))}
        good = ov["seated"] < 0.05 and all(v > 0.05 for k, v in ov.items() if k != "seated")
        hit_cap = inter(shape, parts6["tag_cap_left" if sg > 0 else "tag_cap_right"])
        print(f"{nm:36s} on v6 holder: " + ", ".join(f"{k} {v:.1f}" for k, v in ov.items()) + f", x cap {hit_cap:.1f}"
              + ("" if good and hit_cap < 0.05 else "  <-- PROBLEM"))
        ok &= good and hit_cap < 0.05
    for pair in (B5.RING_RH, B5.RING_SMALL_RH):
        for T in (0.0, 18.0, V2.T_CLOSE):
            S = posed(T)
            for nm in pair:
                shape, sg = B5.ring_mounts[nm]
                a = shape.translate(cq.Vector(sg * T, 0, 0))
                own = ("finger_holder_left", "tag_cap_left") if sg > 0 else ("finger_holder_right", "tag_cap_right")
                hits = [f"{m} {inter(a, o):.1f}" for m, o in S.items() if m not in own and inter(a, o) > 0.05]
                if hits:
                    ok = False
                    print(f"{nm} T={T}: {hits}")
    print("ring mounts over the travel: done (any collision is listed above)")
    return ok


def orient_pins_flat(V3, V, F):
    """holders: lie on a side so the pins print horizontally (layers along each pin, strongest in bending, no pin
    starting in mid-air); among those, least overhang. The stock picker prefers the bigger top-down bed contact."""
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
    for nm in NEW6:
        shape = parts6[nm]
        cq.exporters.export(shape, os.path.join(OUT, f"{nm}.step"))
        V, F = V3.mesh(shape)
        oname, Rm, ov, bed = V3.orient(V, F) if not nm.startswith("finger_holder") else orient_pins_flat(V3, V, F)
        W = V @ Rm.T
        if np.linalg.det(Rm) < 0:
            F = F[:, ::-1]
        lo, hi = W.min(0), W.max(0)
        W = W - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        bad, vol = V3.mesh_stats(W, F)
        V3.prep[nm] = dict(V=W, F=F, orient=oname, overhang=ov, bed=bed, size=hi - lo, bad_edges=bad, vol=vol)
        V3.OVR[nm] = {"enable_support": "1", "support_type": "tree(auto)"}
        print(f"print {nm:20s} {oname:22s} size={np.round(hi - lo, 1)} overhang={ov:6.0f} mm2 bed={bed:6.0f} mm2 "
              f"open_edges={bad} vol={vol:8.0f}")
        t = W[F].astype("<f4")
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = n, t
        with open(os.path.join(OUT, f"{nm}.stl"), "wb") as fh:
            fh.write(nm.encode().ljust(80, b" ") + np.uint32(len(t)).tobytes() + rec.tobytes())
    names = list(NEW6)
    V3.write_3mf(os.path.join(PRINT_DIR, "UMI_v6_holders_caps_PLA.3mf"), "UMI_v6_holders_caps_PLA", names, V3.pack(names))
    asm = cq.Assembly(name="umi_pinch_v6")
    for nm, s in parts6.items():
        asm.add(s, name=nm)
    asm.export(os.path.join(OUT, "umi_pinch_v6_assembly.step"))


if __name__ == "__main__":
    for s, nm in SIDES.items():
        print(f"{nm}: pins at the finger's holes {finger_holes(built[s][2])}")
    ok = check()
    export()
    sys.exit(0 if ok else 1)
