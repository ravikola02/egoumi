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


# ---------------- ring mounts: the v3 rings as slide-on parts on the same dovetail ----------------
# v3 sizes and positions, in LEFT-holder coordinates (the index ring is modelled mirrored, then flipped to the right).
RINGS = dict(thumb=dict(xc=V2.THB["x_open"], zc=V2.THB["z"], ri=V2.THB["ri"], wall=V2.THB["wall"], y=(V2.THB["y0"], V2.THB["y1"])),
             index=dict(xc=-V2.IDX["x_open"], zc=V2.IDX["z"], ri=V2.IDX["ri"], wall=V2.IDX["wall"], y=(V2.IDX["y0"], V2.IDX["y1"])))
# small-hand rings: closer together, smaller bores, and far enough apart in HEIGHT that the thumb passes over the
# index. With rigid rings each finger travels the full jaw travel, so a 70 mm open spread ends with the thumb 2 mm
# past the index: they can only close if one is above the other by a finger radius plus the other ring's outer radius.
SMALL = dict(spread=70.0, thumb_ri=10.0, index_ri=9.0, wall=3.0, thumb_z=-26.5, stagger=22.0)
_c = (V2.THB["x_open"] + V2.IDX["x_open"]) / 2      # v3 centre line between the rings (x -5)
RINGS["thumb_small"] = dict(xc=_c - SMALL["spread"] / 2, zc=SMALL["thumb_z"], ri=SMALL["thumb_ri"], wall=SMALL["wall"], y=RINGS["thumb"]["y"])
RINGS["index_small"] = dict(xc=-(_c + SMALL["spread"] / 2), zc=SMALL["thumb_z"] - SMALL["stagger"], ri=SMALL["index_ri"],
                            wall=SMALL["wall"], y=RINGS["index"]["y"])
LATCH = dict(w=6.0, len=23.0, t=2.0, slit=0.6, hook=1.0, play=0.3)   # snap tongue in the channel floor, hook behind the rail


def ring_mount(kind):
    """Slides onto the rail from BEHIND (the thumb ring rises above the holder's underside, so it cannot pass under
    the lower plate from the front). Closed rear end = stop against pushing forward; a snap tongue in the floor hooks
    behind the rail's front end = lock against pulling back. Press the tongue tip down to release."""
    d, m, r, L = DOVE, MOUNT, RINGS[kind], LATCH
    half = d["w_tip"] / 2 + CLEAR * 1.2 + m["wall"]
    x0, x1 = d["xc"] - half, d["xc"] + half
    y_stop = d["y0"] - CLEAR                       # rear stop face
    block = box(x0, x1, y_stop - m["stop"], Y_FRONT, Z_BLOCK, Z_TOP)
    ro = r["ri"] + r["wall"]
    ring = cq.Solid.makeCylinder(ro, r["y"][1] - r["y"][0], cq.Vector(r["xc"], r["y"][0], r["zc"]), cq.Vector(0, 1, 0))
    if kind == "thumb":      # v2 arm, now running back from the channel block
        link = box(V2.ARM["x0"], V2.ARM["x1"], V2.ARM["y0"], y_stop, V2.ARM["z0"], Z_TOP)
    elif kind == "thumb_small":   # same arm, moved inward with the ring
        link = box(r["xc"] - ro - 9.0, r["xc"] - ro + 3.0, V2.ARM["y0"], y_stop, V2.ARM["z0"], Z_TOP)
    elif kind == "index":    # block extended rearward and inward to the ring
        link = box(x0, r["xc"] - r["ri"] - 0.5, r["y"][0], y_stop, Z_BLOCK, Z_TOP).fuse(
            box(x1 - 1.0, r["xc"] - r["ri"] - 0.5, y_stop - 1.0, r["y"][1], Z_BLOCK, Z_TOP))
    else:                    # low ring: the block reaches only to the ring's OUTER edge and a post drops to it there,
        xe = r["xc"] - ro    # so nothing stands where the thumb passes over the index
        link = box(x0, xe, r["y"][0], y_stop, Z_BLOCK, Z_TOP).fuse(box(x1 - 1.0, xe, y_stop - 1.0, r["y"][1], Z_BLOCK, Z_TOP))
        link = link.fuse(box(xe - 6.0, xe, r["y"][0], r["y"][1], r["zc"] - 5.0, Z_TOP))
        link = link.fuse(box(xe - 1.0, r["xc"] - r["ri"] - 0.3, r["y"][0], r["y"][1], r["zc"] - 5.0, r["zc"] + 5.0))
    # the ring is bored and rounded on its own (outer edges 1.0, bore edges 1.2, as v3), then joined
    bore = cq.Solid.makeCylinder(r["ri"], r["y"][1] - r["y"][0] + 2.0, cq.Vector(r["xc"], r["y"][0] - 1.0, r["zc"]), cq.Vector(0, 1, 0))
    ring = ring.cut(bore)
    for rad, fil in ((ro, 1.0), (r["ri"], 1.2)):
        try:
            ring = ring.fillet(fil, [e for e in ring.Edges() if e.geomType() == "CIRCLE" and abs(e.radius() - rad) < 1e-3])
        except Exception as ex:
            print(f"{kind} ring rounding (r={rad}) skipped:", ex)
    body = block.fuse(link.cut(bore)).fuse(ring).clean()     # the bore never reaches the channel block
    channel = dovetail(y_stop, Y_FRONT + 10.0, grow=CLEAR, top=5.0).fuse(detent(DETENT["r"] + 0.1))
    body = body.cut(channel)
    # snap tongue: two slits through the floor, thinned from below, hook with a lead-in ramp at its tip
    ty0 = Y_FRONT - L["len"]
    for sx in (-1, 1):
        xs = d["xc"] + sx * L["w"] / 2
        body = body.cut(box(min(xs, xs + sx * L["slit"]), max(xs, xs + sx * L["slit"]), ty0, Y_FRONT + 1, Z_BLOCK - 1, Z_FLOOR + 0.01))
    body = body.cut(box(d["xc"] - L["w"] / 2, d["xc"] + L["w"] / 2, ty0, Y_FRONT + 1, Z_BLOCK - 1, Z_FLOOR - L["t"]))
    yh = d["y1"] + L["play"]
    hook = cq.Workplane("YZ").polyline([(yh, Z_FLOOR - 0.1), (yh, Z_FLOOR + L["hook"]), (yh + 0.7, Z_FLOOR + L["hook"]),
                                        (Y_FRONT, Z_FLOOR + 0.1), (Y_FRONT, Z_FLOOR - 0.1)]).close().extrude(L["w"] - 0.4).val()
    return body.fuse(hook.translate(cq.Vector(d["xc"] - L["w"] / 2 + 0.2, 0, 0))).clean()


def fill_top_recess(holder, sx):
    """v3: fill the 0.3 mm frame groove on the top face so the jaw tag sits flat."""
    x0, x1 = sorted((sx * 50.0, sx * 70.0))
    return holder.fuse(cq.Solid.makeBox(x1 - x0, 20.0, 0.31, cq.Vector(x0, 77.5, 16.3))).clean()


rail_L = make_rail()
mount_L = make_mount()
mount_R = mount_L.mirror("YZ")
holder_L = fill_top_recess(V2.P["finger_holder_left"].fuse(rail_L).clean(), -1)
holder_R = fill_top_recess(V2.P["finger_holder_right"].fuse(rail_L.mirror("YZ")).clean(), +1)

# ring mounts. Right hand (as v3): thumb on the left holder, index on the right. Left hand: the mirrored pair.
_rt, _ri = ring_mount("thumb"), ring_mount("index")
ring_mounts = {"ring_mount_thumb_left_holder": (_rt, +1), "ring_mount_index_right_holder": (_ri.mirror("YZ"), -1),
               "ring_mount_thumb_right_holder": (_rt.mirror("YZ"), -1), "ring_mount_index_left_holder": (_ri, +1)}
RING_RH = ("ring_mount_thumb_left_holder", "ring_mount_index_right_holder")
_st, _si = ring_mount("thumb_small"), ring_mount("index_small")
ring_mounts.update({"ring_mount_small_thumb_left_holder": (_st, +1), "ring_mount_small_index_right_holder": (_si.mirror("YZ"), -1),
                    "ring_mount_small_thumb_right_holder": (_st.mirror("YZ"), -1), "ring_mount_small_index_left_holder": (_si, +1)})
RING_SMALL_RH = ("ring_mount_small_thumb_left_holder", "ring_mount_small_index_right_holder")

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


def check_rings():
    """ring mounts: the joint (seated / locked / path in from behind) and collisions over the jaw travel."""
    ok = True
    hold = {+1: holder_L, -1: holder_R}
    for nm, (shape, sg) in ring_mounts.items():
        ov = {lab: inter(shape.translate(cq.Vector(*v)), hold[sg]) for lab, v in
              (("seated", (0, 0, 0)), ("pushed forward 1", (0, 1, 0)), ("pulled back 1", (0, -1, 0)), ("down 1", (0, 0, -1)),
               ("inward 1", (sg, 0, 0)), ("outward 1", (-sg, 0, 0)))}
        locked = ov["seated"] < 0.05 and all(v > 0.05 for k, v in ov.items() if k != "seated")
        ok &= locked
        print(f"{nm:30s} valid={shape.isValid()} solids={len(shape.Solids())} joint: " + ", ".join(f"{k} {v:.1f}" for k, v in ov.items())
              + ("" if locked else "  <-- NOT LOCKED"))
    for nm in RING_RH + RING_SMALL_RH:     # slide-in path from behind with the jaws open (the latch hook is meant to touch the rail on the way)
        shape, sg = ring_mounts[nm]
        S = posed(0.0)
        hits = [f"{m} at -{dy}: {inter(shape.translate(cq.Vector(0, -dy, 0)), s):.1f}" for dy in (10, 25, 40) for m, s in S.items()
                if not m.startswith(("finger_holder", "strap_mount")) and inter(shape.translate(cq.Vector(0, -dy, 0)), s) > 0.05]
        print(f"{nm:30s} slide-in path:", hits if hits else "clear of every other part")
        ok &= not hits
    for label, pair in (("rings", RING_RH), ("small rings", RING_SMALL_RH)):
      for T in (0.0, 12.0, 24.0, V2.T_CLOSE, 38.0):
        S = posed(T)
        hits = []
        for nm in pair:
            shape, sg = ring_mounts[nm]
            a = shape.translate(cq.Vector(sg * T, 0, 0))
            hits += [f"{nm} x {m}: {inter(a, s):.1f} mm3" for m, s in S.items()
                     if not m.startswith("strap_mount") and m != ("finger_holder_left" if sg > 0 else "finger_holder_right") and inter(a, s) > 0.05]
        b = ring_mounts[pair[0]][0].translate(cq.Vector(T, 0, 0)), ring_mounts[pair[1]][0].translate(cq.Vector(-T, 0, 0))
        if inter(*b) > 0.05:
            hits.append(f"thumb ring x index ring: {inter(*b):.1f} mm3")
        ok &= not hits
        print(f"{label} T={T:4.1f}:", hits if hits else "no collisions")
    return ok


def export_print():
    """print-oriented STLs + one Bambu Studio 3MF for the four new parts, with v3's orientation, packing and settings."""
    import build_v3 as V3
    shapes = {**{n: parts[n] for n in NEW}, **{n: s for n, (s, _) in ring_mounts.items()}}
    for nm, shape in shapes.items():
        V, F = V3.mesh(shape)
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
    V3.write_3mf(os.path.join(PRINT_DIR, "UMI_v5_ring_mounts_PLA.3mf"), "UMI_v5_ring_mounts_PLA", list(RING_RH), V3.pack(list(RING_RH)))
    V3.write_3mf(os.path.join(PRINT_DIR, "UMI_v5_ring_mounts_small_PLA.3mf"), "UMI_v5_ring_mounts_small_PLA", list(RING_SMALL_RH),
                 V3.pack(list(RING_SMALL_RH)))


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
    ok = check() & check_rings()
    asm = cq.Assembly(name="umi_pinch_v5")
    for nm, s in parts.items():
        asm.add(s, name=nm)
    asm.export(os.path.join(OUT, "umi_pinch_v5_assembly.step"))
    for nm in NEW:
        cq.exporters.export(parts[nm], os.path.join(OUT, f"{nm}.step"))
    for nm, (shape, _) in ring_mounts.items():
        cq.exporters.export(shape, os.path.join(OUT, f"{nm}.step"))
    export_print()
    sys.exit(0 if ok else 1)
