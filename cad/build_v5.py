"""UMI pinch v5 = v3 with the two rigid finger rings replaced by removable, slide-on strap mounts, so one print fits
any hand, left or right. Only the two finger holders are reprinted (once); every other v3 part is reused as printed.

Why: in v3 the thumb and index rings are 90 mm apart when the jaws are open, their positions and bores are fixed,
they are fused to the holders, and the left/right holders differ, so the gripper is right-handed only.

Holder: the Stanford finger holder plus a dovetail rail under its lower plate, running along Y (no ring, no arm).
The two holders are mirror images.

Strap mount (one per holder, mirror images): slides onto the dovetail from the front until its closed end stops on
the rail; a small bump on the rail clicks into a groove. No screws. Use loads go into the dovetail flanks (jaw
direction) and into the stop (pulling back). It carries a fin OUTSIDE the gripper body (|x| 86..94 with the jaws
open) with two strap windows: a rear, high one where a thumb reaches and a front, low one where an index reaches.
A back-to-back hook-and-loop strap ("one-wrap", STRAP["w"] wide) goes through a window, around the bar under it,
around the finger, and closes on itself. Closing the jaws only ever PULLS on the strap (the thumb pushes inward, the
index pulls inward, each away from its own holder) and the spring opens them.
  * loop length sets where the finger sits, anywhere from the fin inward, so any hand picks its own open spread;
  * the two windows sit STAGGER apart in height, so the thumb passes OVER the index (a key pinch) instead of meeting
    it side by side. That, not the anchor position, is what lets a narrow spread still close the jaws: each finger
    always travels the full jaw travel, so two fingers at the same height meet early however the strap is anchored;
  * right hand: thumb in the left mount's rear window, index in the right mount's front window. Left hand: swap;
  * any other wearable (rings, cradles) can be a new part on the same dovetail without touching the holders.

Units mm, assembly frame (+Y fingers, X jaw travel, +Z up). Finger sizes and positions are ESTIMATES, as in v2.
The fit (CLEAR, DETENT) is untested and printer-dependent: print one mount first and tune.
Run: cd cad && ../.venv-cad/bin/python build_v5.py   -> out_v5/ (STEP, STL), ../tmp_print/ (3MF, git-ignored)
"""
import os
import numpy as np
import cadquery as cq
import build_v2 as V2

_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_ROOT, "out_v5")
PRINT_DIR = os.path.join(_ROOT, "..", "tmp_print")

# ---------------- parameters (LEFT side; the right side is the mirror image) ----------------
HOLDER_BOTTOM = V2.HOLDER_BOTTOM                      # underside of the holder's lower plate, z -21.4
STRAP = dict(w=20.0, t=1.6)                           # strap width and thickness
# dovetail rail under the lower plate; x range sits where the v2 thumb arm was, clear of the nut pocket at x < -72.5
DOVE = dict(xc=-63.0, w_root=9.0, w_tip=13.0, h=4.0, y0=74.0, y1=110.0)
CLEAR = 0.15                                          # gap per face between rail and channel
DETENT = dict(y=80.0, r=0.6, h=0.3)                   # bump across the rail's lower face / groove in the channel floor
MOUNT = dict(wall=2.5, floor=3.4, stop=3.0, gap=0.2)  # channel walls, floor under the rail, closed front end, gap to plate
FIN = dict(x0=-94.0, x1=-86.0, y0=27.0)               # outboard of the body (|x| <= 86) when the jaws are open
BRIDGE_Y0 = 88.0                                      # bridge from the channel block to the fin, ahead of the nut pocket
BAR_H, SLOT_H = 4.0, 3.4                              # bar under each window, window height
# strap windows: y range and bar-centre height. rear = thumb (v3: y 41, z -27), front = index (v3: y 71, z -36.5)
WINDOWS = dict(rear=dict(y=(30.0, 52.0), zc=-32.0), front=dict(y=(60.0, 82.0), zc=-55.0))
STAGGER = WINDOWS["rear"]["zc"] - WINDOWS["front"]["zc"]
EDGE = 1.2                                            # rounding of the bar edges the strap wraps
FINGER = dict(thumb=10.0, index=9.0)                  # stand-in finger radii
SPREADS = (100.0, 85.0, 72.0)                         # open thumb-to-index distances shown in the preview
for _w in WINDOWS.values():
    _w["bar"] = (_w["zc"] - BAR_H / 2, _w["zc"] + BAR_H / 2)
    _w["slot"] = (_w["bar"][1], _w["bar"][1] + SLOT_H)
Z_TOP = HOLDER_BOTTOM - MOUNT["gap"]                  # top face of the mount
Z_FLOOR = HOLDER_BOTTOM - DOVE["h"] - CLEAR           # channel floor
Z_BLOCK = Z_FLOOR - MOUNT["floor"]                    # underside of the channel block
Y_FRONT = DOVE["y1"] + CLEAR + MOUNT["stop"]          # front end of the mount


def box(x0, x1, y0, y1, z0, z1):
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0))


def _wp(y1):
    return cq.Workplane(cq.Plane(origin=(0, y1, 0), xDir=(1, 0, 0), normal=(0, -1, 0)))   # local (x, y) = (X, Z)


def dovetail(y0, y1, grow=0.0, top=0.6):
    """trapezoid prism hanging under the plate, wider at the bottom; grow offsets every face (channel = rail + CLEAR)."""
    d, zr, zt = DOVE, HOLDER_BOTTOM + top, HOLDER_BOTTOM - DOVE["h"] - grow
    slope = (d["w_tip"] - d["w_root"]) / 2 / d["h"]
    hw = lambda z: d["w_root"] / 2 + slope * (HOLDER_BOTTOM - z) + grow * float(np.hypot(1, slope))
    pts = [(d["xc"] - hw(zr), zr), (d["xc"] + hw(zr), zr), (d["xc"] + hw(zt), zt), (d["xc"] - hw(zt), zt)]
    return _wp(y1).polyline(pts).close().extrude(y1 - y0).val()


def detent(r):
    """half-round rib across X on the rail's lower face."""
    zc = HOLDER_BOTTOM - DOVE["h"] + (r - DETENT["h"])
    return cq.Solid.makeCylinder(r, DOVE["w_tip"] - 2.0, cq.Vector(DOVE["xc"] - DOVE["w_tip"] / 2 + 1.0, DETENT["y"], zc),
                                 cq.Vector(1, 0, 0)).intersect(box(-200, 200, 0, 200, -200, HOLDER_BOTTOM - DOVE["h"] + 1e-3))


def make_rail():
    return dovetail(DOVE["y0"], DOVE["y1"]).fuse(detent(DETENT["r"])).clean()


def make_mount():
    d, m, f = DOVE, MOUNT, FIN
    half = d["w_tip"] / 2 + CLEAR * 1.2 + m["wall"]
    block = box(d["xc"] - half, d["xc"] + half, d["y0"] - 2.0, Y_FRONT, Z_BLOCK, Z_TOP)
    bridge = box(f["x0"], d["xc"] - half + 1.0, BRIDGE_Y0, Y_FRONT, Z_BLOCK, Z_TOP)
    rear_z0 = WINDOWS["rear"]["bar"][0]
    fin = box(f["x0"], f["x1"], f["y0"], Y_FRONT, rear_z0, Z_TOP)
    fy0, fy1 = WINDOWS["front"]["y"]
    fin = fin.fuse(box(f["x0"], f["x1"], fy0 - 3.0, fy1 + 3.0, WINDOWS["front"]["bar"][0], rear_z0 + 1.0))
    for w in WINDOWS.values():
        fin = fin.cut(box(f["x0"] - 1, f["x1"] + 1, w["y"][0], w["y"][1], *w["slot"]))

    def on_bar(e):   # the four long edges of each bar: the strap bends around them
        c = e.Center()
        return e.geomType() == "LINE" and any(
            w["y"][0] < c.y < w["y"][1] and abs(e.Length() - (w["y"][1] - w["y"][0])) < 1e-6 and c.z < w["bar"][1] + 1e-6
            for w in WINDOWS.values())
    try:
        fin = fin.fillet(EDGE, [e for e in fin.Edges() if on_bar(e)])
    except Exception as ex:
        print("bar edge rounding skipped:", ex)
    # channel: open at the rear, closed at the front (the stop); groove for the detent bump
    channel = dovetail(d["y0"] - 10.0, d["y1"] + CLEAR, grow=CLEAR, top=5.0).fuse(detent(DETENT["r"] + 0.1))
    return block.fuse(bridge).fuse(fin).cut(channel).clean()


def fill_top_recess(holder, sx):
    """v3: fill the 0.3 mm frame groove on the top face so the jaw tag sits flat."""
    x0, x1 = sorted((sx * 50.0, sx * 70.0))
    return holder.fuse(cq.Solid.makeBox(x1 - x0, 20.0, 0.31, cq.Vector(x0, 77.5, 16.3))).clean()


rail_L = make_rail()
mount_L = make_mount()
mount_R = mount_L.mirror("YZ")
holder_L = fill_top_recess(V2.P["finger_holder_left"].fuse(rail_L).clean(), -1)
holder_R = fill_top_recess(V2.P["finger_holder_right"].fuse(rail_L.mirror("YZ")).clean(), +1)

# top cover as v3 (mirror flaps removed); unchanged, shown for context only
_keep = cq.Solid.makeBox(70, 400, 200, cq.Vector(-35, -200, 19.8))
top_cover = V2.parts["top_cover"].cut(cq.Solid.makeBox(400, 400, 200, cq.Vector(-200, -200, 19.8)).cut(_keep)).clean()

parts = dict(V2.parts)
parts.update(top_cover=top_cover, finger_holder_left=holder_L, finger_holder_right=holder_R,
             strap_mount_left=mount_L, strap_mount_right=mount_R)
del parts["finger_holder_left_thumb"], parts["finger_holder_right_index"]
NEW = ("finger_holder_left", "finger_holder_right", "strap_mount_left", "strap_mount_right")
MOVING_L = ("finger_holder_left", "strap_mount_left", "soft_finger_left")
MOVING_R = ("finger_holder_right", "strap_mount_right", "soft_finger_right")


# ---------------- preview-only soft parts (right-hand layout; mirror for a left hand) ----------------
def _hull_prism(circles, y0, y1):
    """prism over the convex hull of circles [(x, z, r)], r = 0 for a point."""
    from scipy.spatial import ConvexHull
    a = np.radians(np.linspace(0, 360, 72, endpoint=False))
    p = np.concatenate([np.c_[x + r * np.cos(a), z + r * np.sin(a)] if r > 0 else np.array([[x, z]]) for x, z, r in circles])
    p = p[ConvexHull(p).vertices]
    return _wp(y1).polyline([tuple(q) for q in p]).close().extrude(y1 - y0).val()


def strap(kind, window, xc, g=0.3):
    """strap loop on the LEFT mount: around the bar, through the window, around a finger centred at x = xc."""
    w = WINDOWS[window]
    y0 = sum(w["y"]) / 2 - STRAP["w"] / 2
    y1, t, r = y0 + STRAP["w"], STRAP["t"], FINGER[kind]
    z0, z1 = w["bar"][0] - g, w["bar"][1] + g
    bar = [(x, z) for x in (FIN["x0"] - g, FIN["x1"] + g) for z in (z0, z1)]
    face = [(FIN["x1"] + g, z0), (FIN["x1"] + g, z1)]
    inner1 = _hull_prism([(x, z, 0) for x, z in bar], y0, y1)
    inner2 = _hull_prism([(x, z, 0) for x, z in face] + [(xc, w["zc"], r + g)], y0, y1)
    outer = _hull_prism([(x, z, t) for x, z in bar], y0, y1).fuse(
        _hull_prism([(x, z, t) for x, z in face] + [(xc, w["zc"], r + g + t)], y0, y1))
    return outer.cut(inner1).cut(inner2).cut(mount_L)     # the real strap is squeezed where the dummy meets the fin


def finger(kind, window, xc, back=40.0, front=22.0):
    w = WINDOWS[window]
    yc, r = sum(w["y"]) / 2, FINGER[kind]
    c = cq.Solid.makeCylinder(r, back + front, cq.Vector(xc, yc - back, w["zc"]), cq.Vector(0, 1, 0))
    return c.fuse(cq.Solid.makeSphere(r, cq.Vector(xc, yc + front, w["zc"])))


def soft(spread):
    """right-hand layout: thumb on the left mount's rear window, index on the right mount's front window, their
    centres `spread` apart with the jaws open. Returns name -> (shape, +1 moves with the left holder / -1 right)."""
    x = -spread / 2
    return {
        "strap_thumb": (strap("thumb", "rear", x), +1), "thumb (est.)": (finger("thumb", "rear", x), +1),
        "strap_index": (strap("index", "front", x).mirror("YZ"), -1),
        "index (est.)": (finger("index", "front", x).mirror("YZ"), -1),
    }


def meet_gap(spread, stagger=STAGGER):
    """soft-fingertip gap left when the stand-in thumb and index (with their straps) touch; 0 = the jaws shut first.
    With stagger >= the two radii plus straps the thumb passes over the index and they never touch; below a spread of
    2 * T_CLOSE the thumb would have to cross past the index, which is counted as the limit."""
    reach = FINGER["thumb"] + FINGER["index"] + 2 * STRAP["t"]
    dx_min = float(np.sqrt(max(0.0, reach ** 2 - stagger ** 2)))
    return max(0.0, 2 * V2.T_CLOSE - (spread - dx_min))


def posed(T, spread=None):
    out = {}
    for n, s in parts.items():
        dx = T if n in MOVING_L else -T if n in MOVING_R else 0.0
        out[n] = s.translate(cq.Vector(dx, 0, 0))
    if spread is not None:
        for n, (s, sg) in soft(spread).items():
            out[n] = s.translate(cq.Vector(sg * T, 0, 0))
    return out


def inter(a, b):
    try:
        return a.intersect(b).Volume()
    except Exception:
        return -1.0


def check():
    """Over the jaw travel: the rails and mounts against every other part (a mount must not touch even its own
    holder: the dovetail has clearance), and the straps (widest and narrowest preview spread, both hands' windows)
    against every rigid part except their own mount. The Stanford holder bodies are not re-tested because they
    already overlap the gears, linkages and slider in this pose model (those parts are not animated), as in v2."""
    L, R = ("finger_holder_left", "strap_mount_left"), ("finger_holder_right", "strap_mount_right")
    added = {"rail left": (rail_L, +1, L[:1]), "rail right": (rail_L.mirror("YZ"), -1, R[:1]),
             "mount left": (mount_L, +1, L[1:]), "mount right": (mount_R, -1, R[1:])}
    for sp in (SPREADS[0], SPREADS[-1]):
        for kind, win in (("thumb", "rear"), ("index", "front")):
            s = strap(kind, win, -sp / 2)
            added[f"strap {kind} left spread={sp:.0f}"] = (s, +1, L[1:])
            added[f"strap {kind} right spread={sp:.0f}"] = (s.mirror("YZ"), -1, R[1:])
    ok = True
    for T in (0.0, 12.0, 24.0, V2.T_CLOSE, 38.0):
        S = posed(T)
        hits = []
        for n, (shape, sg, own) in added.items():
            a = shape.translate(cq.Vector(sg * T, 0, 0))
            for m, s in S.items():
                if m in own:
                    continue
                v = inter(a, s)
                if v > 0.05:
                    hits.append(f"{n} x {m}: {v:.1f} mm3")
        ok &= not hits
        print(f"T={T:4.1f}:", hits if hits else "no collisions")
    return ok


def export_print():
    """print-oriented STLs + one Bambu Studio 3MF for the four new parts, with v3's orientation, packing and settings."""
    import build_v3 as V3
    for nm in NEW:
        V, F = V3.mesh(parts[nm])
        name, Rm, ov, bed = V3.orient(V, F)
        W = V @ Rm.T
        if np.linalg.det(Rm) < 0:
            F = F[:, ::-1]
        lo, hi = W.min(0), W.max(0)
        W = W - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        bad, vol = V3.mesh_stats(W, F)
        V3.prep[nm] = dict(V=W, F=F, orient=name, overhang=ov, bed=bed, size=hi - lo, bad_edges=bad, vol=vol)
        V3.OVR[nm] = {"enable_support": "1", "support_type": "tree(auto)"}
        print(f"print {nm:20s} {name:22s} size={np.round(hi - lo, 1)} overhang={ov:6.0f} mm2 bed={bed:6.0f} mm2 "
              f"open_edges={bad} vol={vol:8.0f}")
        t = W[F].astype("<f4")
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = n, t
        with open(os.path.join(OUT, f"{nm}.stl"), "wb") as fh:
            fh.write(nm.encode().ljust(80, b" ") + np.uint32(len(t)).tobytes() + rec.tobytes())
    os.makedirs(PRINT_DIR, exist_ok=True)       # git-ignored: the 3MF is a local print file, not part of the design
    V3.write_3mf(os.path.join(PRINT_DIR, "UMI_v5_holders_mounts_PLA.3mf"), "UMI_v5_holders_mounts_PLA", list(NEW), V3.pack(list(NEW)))


if __name__ == "__main__":
    import sys
    os.makedirs(OUT, exist_ok=True)
    for nm in NEW:
        s = parts[nm]
        b = s.BoundingBox()
        print(f"{nm:20s} valid={s.isValid()} solids={len(s.Solids())} vol={s.Volume():8.0f} "
              f"x[{b.xmin:.1f},{b.xmax:.1f}] y[{b.ymin:.1f},{b.ymax:.1f}] z[{b.zmin:.1f},{b.zmax:.1f}]")
    print(f"mirror check: |vol L - vol R| = {abs(holder_L.Volume() - holder_R.Volume()):.2f} mm3")
    # the joint: seated = no overlap; it must not lift off downward or move sideways (dovetail), only slide along +Y
    for label, v in (("seated", (0, 0, 0)), ("pulled down 1 mm", (0, 0, -1)), ("pushed inward 1 mm", (1, 0, 0)),
                     ("pushed outward 1 mm", (-1, 0, 0)), ("pulled back 1 mm (stop)", (0, -1, 0)), ("slid forward 20 mm", (0, 20, 0))):
        print(f"joint, mount {label:24s}: overlap with holder {inter(mount_L.translate(cq.Vector(*v)), holder_L):6.1f} mm3")
    print(f"window stagger {STAGGER:.0f} mm (v3 rings: 9.5)")
    for sp in SPREADS:
        print(f"open spread {sp:5.1f} mm (v3: 90): fingertip gap when the stand-in fingers touch: "
              f"{meet_gap(sp):4.1f} mm staggered, {meet_gap(sp, 0.0):4.1f} mm if they met at the same height")
    ok = check()
    asm = cq.Assembly(name="umi_pinch_v5")
    for nm, s in parts.items():
        asm.add(s, name=nm)
    asm.export(os.path.join(OUT, "umi_pinch_v5_assembly.step"))
    for nm in NEW:
        cq.exporters.export(parts[nm], os.path.join(OUT, f"{nm}.step"))
    export_print()
    sys.exit(0 if ok else 1)
