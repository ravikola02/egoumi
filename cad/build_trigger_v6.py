"""v6 trigger: a shorter, solid trigger on the unchanged Stanford slider plate.

Problem with the Stanford trigger (cad/input: part "handle"): its stem hangs ~100 mm below the slider and the
fingers pull it backwards along most of that length. Pulling low on a long lever twists the slider: its rear end
lifts into the lid (top cover) and the front presses on the bottom plate, and the bending peaks at the stem's root.
The push on the lid is roughly finger force x lever / slider length, about as large as the finger pull itself.

This trigger:
  * keeps the slider plate as it is (everything above z = STEM_CUT_Z), so it fits the same mechanism;
  * replaces the stem with a solid one that is only as long as the fingers that pull it (FINGERS x FINGER_PITCH),
    with finger scallops on its front face that put the pull as high as possible;
  * adds fillets where the stem meets the slider plate (front and back, inside the stem's 13 mm width);
  * prints lying on its side, so the layers run along the stem.
Two sizes: 2 fingers (index + middle) and 1 finger (index only). Lever = distance from the slider's mid-plane to the
centre of the finger pull.
"""
import numpy as np
import cadquery as cq
import build_v2 as V2

HALF_W = 6.5                  # stem half-width in x (the original stem's width; the slots it moves in fit this)
Z_TOP = V2.STEM_CUT_Z         # slider plate underside (-6.4)
Z_SLIDER_MID = 5.2            # slider plate mid-plane, the lever's reference
FINGER_PITCH = 21.0           # height per finger
FIRST_FINGER_Z = -17.0        # centre of the top finger scallop
SCALLOP_R = 11.0              # finger scallop radius, depth SCALLOP_DEPTH into the front face
SCALLOP_DEPTH = 2.0
ROOT_R = 6.0                  # fillets at the stem's root
# original stem profile, from the Stanford part: back edge y(z) and front edge y(z) (measured, linear fits)
BACK = lambda z: 31.5 + (-7.0 - z) * 0.0425 if z > -20 else 35.3 - (-20.0 - z) * 0.0875
FRONT_LOW = lambda z: 66.3 - (-20.0 - z) * 0.0875     # below z -20
FRONT_ROOT_Y = 53.5           # the front edge meets the slider plate here (plate front at y 53)


def profile(n_fingers):
    """YZ outline of the stem (y forward, z up)."""
    z_bot = FIRST_FINGER_Z - (n_fingers - 1) * FINGER_PITCH - FINGER_PITCH / 2 - 2.0
    pts = [(BACK(Z_TOP) - ROOT_R, Z_TOP + 0.6)]
    # back root fillet (concave, quarter circle), then down the back edge
    for a in np.linspace(90, 0, 10):
        pts.append((BACK(Z_TOP) - ROOT_R + ROOT_R * np.cos(np.radians(a)) * 1.0, Z_TOP - ROOT_R + ROOT_R * np.sin(np.radians(a))))
    pts = [(BACK(Z_TOP) - ROOT_R, Z_TOP + 0.6), (BACK(Z_TOP) - ROOT_R, Z_TOP)] + [
        (BACK(Z_TOP) - ROOT_R * (1 - np.sin(np.radians(t))), Z_TOP - ROOT_R * (1 - np.cos(np.radians(t)))) for t in np.linspace(0, 90, 10)][1:]
    for z in np.linspace(Z_TOP - ROOT_R, z_bot + 4.0, 12):
        pts.append((BACK(z), z))
    # rounded bottom
    yb, yf = BACK(z_bot + 4.0), FRONT_LOW(z_bot + 4.0)
    for t in np.linspace(180, 360, 13):
        pts.append(((yb + yf) / 2 + (yf - yb) / 2 * np.cos(np.radians(t)), z_bot + 4.0 + 4.0 * np.sin(np.radians(t))))
    # front edge up to z -20, then the original 45 deg chamfer to the plate, with a front root fillet
    for z in np.linspace(z_bot + 4.0, -20.0, 10):
        pts.append((FRONT_LOW(z), z))
    pts += [(FRONT_LOW(-20.0), -20.0), (FRONT_ROOT_Y + 1.0, Z_TOP - 1.0), (FRONT_ROOT_Y - 0.5, Z_TOP + 0.6)]
    clean = [pts[0]]
    for p in pts[1:]:
        if np.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 0.05:
            clean.append(p)
    return clean, z_bot


def make_trigger(n_fingers):
    pts, z_bot = profile(n_fingers)
    wp = cq.Workplane(cq.Plane(origin=(-HALF_W, 0, 0), xDir=(0, 1, 0), normal=(1, 0, 0)))
    stem = wp.polyline(pts).close().extrude(2 * HALF_W).val()
    for k in range(n_fingers):        # finger scallops on the front face
        zc = FIRST_FINGER_Z - k * FINGER_PITCH
        yc = FRONT_LOW(min(zc, -20.0)) + SCALLOP_R - SCALLOP_DEPTH
        stem = stem.cut(cq.Solid.makeCylinder(SCALLOP_R, 40, cq.Vector(-20, yc, zc), cq.Vector(1, 0, 0)))
    try:  # soften the stem's vertical edges for the fingers
        stem = stem.fillet(1.5, [e for e in stem.Edges() if e.geomType() == "LINE" and abs(e.tangentAt(0).z) > 0.9
                                 and e.BoundingBox().zmax < Z_TOP - ROOT_R])
    except Exception as ex:
        print("stem edge rounding skipped:", ex)
    plate = V2.P["handle"].intersect(cq.Solid.makeBox(400, 400, 200, cq.Vector(-200, -200, Z_TOP))).clean()
    return plate.fuse(stem).clean(), z_bot


def lever(n_fingers):
    zc = np.mean([FIRST_FINGER_Z - k * FINGER_PITCH for k in range(n_fingers)])
    return Z_SLIDER_MID - zc


ORIG_LEVER = Z_SLIDER_MID - (-55.0)    # Stanford: fingers spread over the 100 mm stem, centre ~z -55
triggers = {f"trigger_{n}finger": make_trigger(n) for n in (2, 1)}


def check(inter):
    import build_v4 as B4
    import build_cube_v6 as CB
    ok = True
    fixed = {"grip": V2.parts["grip"], "bottom_plate": V2.parts["bottom_plate"], "top_cover v4": B4.top_cover, "cube v6": CB.cube6}
    orig = V2.P["handle"]
    for nm, (t, z_bot) in triggers.items():
        n = int(nm.split("_")[1][0])
        hits = []
        for dy in (-2, 0, 4, 8, 12, 16):
            tm = t.translate(cq.Vector(0, -dy, 0))
            for fn, f in fixed.items():
                v = inter(tm, f) - inter(orig.translate(cq.Vector(0, -dy, 0)), f)
                if v > 0.5:
                    hits.append(f"{fn} at {dy} mm back: {v:.1f}")
        root = t.intersect(cq.Solid.makeBox(20, 200, 0.2, cq.Vector(-10, -100, Z_TOP - 1.0))).Volume() / 0.2
        print(f"{nm}: valid {t.isValid()} ({len(t.Solids())} solid), stem to z {z_bot:.0f} (Stanford -105), "
              f"lever {lever(n):.0f} mm vs ~{ORIG_LEVER:.0f} -> lid push about {lever(n) / ORIG_LEVER:.0%} of the Stanford trigger's; "
              f"root section {root:.0f} mm2; travel 0-16 mm: " + (", ".join(hits) if hits else "no collisions"))
        ok &= t.isValid() and len(t.Solids()) == 1 and not hits
    return ok
