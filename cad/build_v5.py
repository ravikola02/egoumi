"""UMI pinch v5 = v3 with the two rigid finger rings replaced by strap anchors, so one print fits any hand, left or
right. Only the two finger holders change; every other v3 part is reused as printed.

Why: in v3 the thumb and index rings are 90 mm apart when the jaws are open, and their positions and bores are fixed.
That is too wide for smaller hands, and the left/right holders differ, so the gripper is right-handed only.

Each holder now carries the same rail on its underside (the v2 thumb arm, mirrored onto the other holder) with two
strap windows: a rear one where a thumb reaches and a front one where an index reaches. A back-to-back hook-and-loop
strap ("one-wrap", STRAP["w"] wide) goes through a window, around the bar under it, around the finger, and closes on
itself. Closing the jaws only ever PULLS on the strap (the thumb pushes inward, the index pulls inward, each away
from its own holder) and the spring opens them, so a strap is all the load path needs.
  * loop length sets where the finger sits: a longer loop moves it inward, narrowing the hand's open spread;
  * the loop takes any finger size, and the finger finds its own height and angle;
  * right hand: thumb in the left holder's rear window, index in the right holder's front window. Left hand: swap.

Limit (kinematic, not a design choice): each finger still travels the full jaw travel, T_CLOSE per side. Moving both
fingers inward by d narrows the open spread by 2d, and the fingers then meet 2d earlier, before the jaws are shut.

Units mm, assembly frame (+Y fingers, X jaw travel, +Z up). Finger sizes and positions are ESTIMATES, as in v2.
Run: cd cad && ../.venv-cad/bin/python build_v5.py   -> out_v5/
"""
import os
import numpy as np
import cadquery as cq
import build_v2 as V2

_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_ROOT, "out_v5")

# ---------------- parameters (LEFT holder; the right one is its mirror image) ----------------
STRAP = dict(w=20.0, t=1.6)                          # strap width and thickness
RAIL = dict(x0=-71.5, x1=-61.5, y0=27.0, y1=112.0, z0=-36.6)   # v2 thumb arm, 0.6 mm deeper; x clears the screw pocket
TUNNEL = dict(z0=-32.6, z1=-29.2)                    # strap window through the rail; the bar is what is left below it
WINDOWS = dict(rear=(30.0, 52.0), front=(60.0, 82.0))   # y ranges; rear = thumb (v3 centre y 41), front = index (y 71)
EDGE = 1.2                                           # rounding of the bar edges the strap wraps
HOLDER_BOTTOM = V2.HOLDER_BOTTOM
FACE_X = RAIL["x1"]                                  # rail face the finger rests against at the tightest setting
FINGER = dict(thumb=10.0, index=9.0)                 # stand-in finger radii
Z_FINGER = (RAIL["z0"] + TUNNEL["z0"]) / 2           # the loop centres on the bar


def box(x0, x1, y0, y1, z0, z1):
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0))


def make_rail():
    rail = box(RAIL["x0"], RAIL["x1"], RAIL["y0"], RAIL["y1"], RAIL["z0"], HOLDER_BOTTOM + 0.6)
    for y0, y1 in WINDOWS.values():
        rail = rail.cut(box(RAIL["x0"] - 1, RAIL["x1"] + 1, y0, y1, TUNNEL["z0"], TUNNEL["z1"]))

    def on_bar(e):   # the four long edges of each bar: the strap bends around them
        c = e.Center()
        in_win = any(y0 < c.y < y1 and abs(e.Length() - (y1 - y0)) < 1e-6 for y0, y1 in WINDOWS.values())
        return in_win and e.geomType() == "LINE" and c.z < TUNNEL["z0"] + 1e-6
    try:
        rail = rail.fillet(EDGE, [e for e in rail.Edges() if on_bar(e)])
    except Exception as ex:
        print("bar edge rounding skipped:", ex)
    return rail


def fill_top_recess(holder, sx):
    """v3: fill the 0.3 mm frame groove on the top face so the jaw tag sits flat."""
    x0, x1 = sorted((sx * 50.0, sx * 70.0))
    return holder.fuse(cq.Solid.makeBox(x1 - x0, 20.0, 0.31, cq.Vector(x0, 77.5, 16.3))).clean()


rail_L = make_rail()
rail_R = rail_L.mirror("YZ")
holder_L = fill_top_recess(V2.P["finger_holder_left"].fuse(rail_L).clean(), -1)
holder_R = fill_top_recess(V2.P["finger_holder_right"].fuse(rail_R).clean(), +1)

# top cover as v3 (mirror flaps removed); unchanged, shown for context only
_keep = cq.Solid.makeBox(70, 400, 200, cq.Vector(-35, -200, 19.8))
top_cover = V2.parts["top_cover"].cut(cq.Solid.makeBox(400, 400, 200, cq.Vector(-200, -200, 19.8)).cut(_keep)).clean()

parts = dict(V2.parts)
parts.update(top_cover=top_cover, finger_holder_left=holder_L, finger_holder_right=holder_R)
del parts["finger_holder_left_thumb"], parts["finger_holder_right_index"]
NEW = ("finger_holder_left", "finger_holder_right")
MOVING_L = ("finger_holder_left", "soft_finger_left")
MOVING_R = ("finger_holder_right", "soft_finger_right")


# ---------------- preview-only soft parts (right-hand layout; mirror for a left hand) ----------------
def _wp(y1):
    return cq.Workplane(cq.Plane(origin=(0, y1, 0), xDir=(1, 0, 0), normal=(0, -1, 0)))   # local (x, y) = (X, Z)


def _hull_prism(circles, y0, y1):
    """prism over the convex hull of circles [(x, z, r)], r = 0 for a point."""
    from scipy.spatial import ConvexHull
    a = np.radians(np.linspace(0, 360, 72, endpoint=False))
    p = np.concatenate([np.c_[x + r * np.cos(a), z + r * np.sin(a)] if r > 0 else np.array([[x, z]]) for x, z, r in circles])
    p = p[ConvexHull(p).vertices]
    return _wp(y1).polyline([tuple(q) for q in p]).close().extrude(y1 - y0).val()


def finger_x(kind, d):
    """finger centre on the LEFT holder, d mm inward of the tightest setting (finger against the rail face)."""
    return FACE_X + FINGER[kind] + d


def strap(kind, window, d, g=0.3):
    """strap loop on the LEFT holder: around the bar, through the window, around the finger."""
    y0 = sum(WINDOWS[window]) / 2 - STRAP["w"] / 2
    y1, t, r = y0 + STRAP["w"], STRAP["t"], FINGER[kind]
    bar = [(x, z) for x in (RAIL["x0"] - g, RAIL["x1"] + g) for z in (RAIL["z0"] - g, TUNNEL["z0"] + g)]
    face = [(RAIL["x1"] + g, RAIL["z0"] - g), (RAIL["x1"] + g, TUNNEL["z0"] + g)]
    fc = (finger_x(kind, d), Z_FINGER)
    inner1 = _hull_prism([(x, z, 0) for x, z in bar], y0, y1)
    inner2 = _hull_prism([(x, z, 0) for x, z in face] + [(*fc, r + g)], y0, y1)
    outer = _hull_prism([(x, z, t) for x, z in bar], y0, y1).fuse(
        _hull_prism([(x, z, t) for x, z in face] + [(*fc, r + g + t)], y0, y1))
    return outer.cut(inner1).cut(inner2).cut(rail_L)      # the real strap is squeezed where the dummy meets the rail


def finger(kind, window, d, back=40.0, front=22.0):
    yc, r = sum(WINDOWS[window]) / 2, FINGER[kind]
    c = cq.Solid.makeCylinder(r, back + front, cq.Vector(finger_x(kind, d), yc - back, Z_FINGER), cq.Vector(0, 1, 0))
    return c.fuse(cq.Solid.makeSphere(r, cq.Vector(finger_x(kind, d), yc + front, Z_FINGER)))


def soft(d):
    """right-hand layout: thumb on the left holder's rear window, index on the right holder's front window.
    Returns name -> (shape, +1 if it moves with the left holder / -1 with the right)."""
    return {
        "strap_thumb": (strap("thumb", "rear", d), +1), "thumb (est.)": (finger("thumb", "rear", d), +1),
        "strap_index": (strap("index", "front", d).mirror("YZ"), -1),
        "index (est.)": (finger("index", "front", d).mirror("YZ"), -1),
    }


def open_spread(d):
    """thumb-to-index centre distance with the jaws open."""
    return -finger_x("index", d) - finger_x("thumb", d)


def meet_gap(d):
    """soft-fingertip gap at which thumb and index meet side by side (with their straps between them); 0 = jaws shut."""
    t_max = (open_spread(d) - (FINGER["thumb"] + FINGER["index"] + 2 * STRAP["t"])) / 2
    return max(0.0, 2 * (V2.T_CLOSE - t_max))


def posed(T, d=None):
    out = {}
    for n, s in parts.items():
        dx = T if n in MOVING_L else -T if n in MOVING_R else 0.0
        out[n] = s.translate(cq.Vector(dx, 0, 0))
    if d is not None:
        for n, (s, sg) in soft(d).items():
            out[n] = s.translate(cq.Vector(sg * T, 0, 0))
    return out


def inter(a, b):
    try:
        return a.intersect(b).Volume()
    except Exception:
        return -1.0


def check():
    """Over the jaw travel: the two rails against every other part, and the straps (tightest and a long setting, both
    hands' windows) against every rigid part except their own holder. Only the additions are tested because the
    Stanford holders already overlap the gears, linkages and slider in this pose model (those parts are not animated)."""
    added = {"rail left": (rail_L, +1, "finger_holder_left"), "rail right": (rail_R, -1, "finger_holder_right")}
    for d in (0.0, 16.0):
        for kind, win in (("thumb", "rear"), ("index", "front")):
            s = strap(kind, win, d)
            added[f"strap {kind} left d={d:.0f}"] = (s, +1, "finger_holder_left")
            added[f"strap {kind} right d={d:.0f}"] = (s.mirror("YZ"), -1, "finger_holder_right")
    ok = True
    for T in (0.0, 12.0, 24.0, V2.T_CLOSE, 38.0):
        S = posed(T)
        hits = []
        for n, (shape, sg, own) in added.items():
            a = shape.translate(cq.Vector(sg * T, 0, 0))
            for m, s in S.items():
                if m == own:
                    continue
                v = inter(a, s)
                if v > 0.05:
                    hits.append(f"{n} x {m}: {v:.1f} mm3")
        ok &= not hits
        print(f"T={T:4.1f}:", hits if hits else "no collisions")
    return ok


def export_print():
    """print-oriented STLs + one Bambu Studio 3MF for the two holders, with v3's orientation, packing and settings."""
    import build_v3 as V3
    for nm in NEW:
        V, F = V3.mesh(parts[nm])
        name, Rm, ov, bed = V3.orient(V, F)
        W = V @ Rm.T
        lo, hi = W.min(0), W.max(0)
        W = W - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        bad, vol = V3.mesh_stats(W, F)
        V3.prep[nm] = dict(V=W, F=F, orient=name, overhang=ov, bed=bed, size=hi - lo, bad_edges=bad, vol=vol)
        V3.OVR[nm] = {"enable_support": "1", "support_type": "tree(auto)"}
        print(f"print {nm:20s} {name:20s} size={np.round(hi - lo, 1)} overhang={ov:6.0f} mm2 bed={bed:6.0f} mm2 "
              f"open_edges={bad} vol={vol:8.0f}")
        t = W[F].astype("<f4")
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = n, t
        with open(os.path.join(OUT, f"{nm}.stl"), "wb") as fh:
            fh.write(nm.encode().ljust(80, b" ") + np.uint32(len(t)).tobytes() + rec.tobytes())
    V3.write_3mf(os.path.join(OUT, "UMI_v5_holders_PLA.3mf"), "UMI_v5_holders_PLA", list(NEW), V3.pack(list(NEW)))


if __name__ == "__main__":
    import sys
    os.makedirs(OUT, exist_ok=True)
    for nm, old in zip(NEW, ("finger_holder_left_thumb", "finger_holder_right_index")):
        s = parts[nm]
        b = s.BoundingBox()
        print(f"{nm:20s} valid={s.isValid()} solids={len(s.Solids())} vol={s.Volume():8.0f} (v2 {V2.parts[old].Volume():8.0f}) "
              f"x[{b.xmin:.1f},{b.xmax:.1f}] y[{b.ymin:.1f},{b.ymax:.1f}] z[{b.zmin:.1f},{b.zmax:.1f}]")
    print(f"mirror check: |vol L - vol R| = {abs(holder_L.Volume() - holder_R.Volume()):.2f} mm3")
    for d in (0.0, 7.0, 14.0, 21.0):
        print(f"loop +{d:4.1f} mm/side: open spread {open_spread(d):5.1f} mm (v3: {V2.IDX['x_open'] - V2.THB['x_open']:.0f}), "
              f"fingers meet at a fingertip gap of ~{meet_gap(d):4.1f} mm (side-by-side estimate)")
    ok = check()
    asm = cq.Assembly(name="umi_pinch_v5")
    for nm, s in parts.items():
        asm.add(s, name=nm)
    asm.export(os.path.join(OUT, "umi_pinch_v5_assembly.step"))
    for nm in NEW:
        cq.exporters.export(parts[nm], os.path.join(OUT, f"{nm}.step"))
    export_print()
    sys.exit(0 if ok else 1)
