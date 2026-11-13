"""v6 marker cube: the four 60 mm cube markers (IDs 0-3) go on printed tiles that slide into the cube and click in,
instead of being stuck straight onto the cube's faces.

Each face gets a pocket with undercut grooves along two edges, open toward one edge of the cube: the rear and side
faces toward the cube's open bottom, the top face toward the front edge of the top wall. A tile (a sticker on a
1.6 mm printed plate) slides in until it stops at the closed end, and a bump on its back clicks into a hole in the
pocket floor. A 1.6 mm backing plate inside the wall keeps the floor under the pocket 2.4 mm thick.

The tile's face is flush with the old face, so the sticker sits where configs/gripper_markers.yaml already puts it
(centre, normal and up unchanged). Each face is exactly 70 mm across, so the tiles are TILE mm to leave a rim and to
keep neighbouring tiles clear of each other at the corners: the sticker is cut to TILE mm, i.e. the white margin
around the 60 mm marker drops from 5 to 2 mm. Print the cube and tiles in a light colour, so the 3 mm cube rim
around each tile extends the white border.
Swap a tile: push the bump out from inside (the cube is open underneath) and slide the tile out.
"""
import numpy as np
import cadquery as cq
import build_v2 as V2

TILE = 64.0                 # tile face = sticker size (marker 60 + 2 x 2 margin)
T_FRONT, T_BACK = 0.8, 0.8  # tile: front layer (face width) and back layer (flanges)
FLANGE = 1.0                # back layer reaches this far beyond the face width on the two groove edges
CLEAR = 0.15                # per side, in the plane
LIP_GAP = 0.05              # flange thickness clearance in the groove
BUMP = dict(r=1.5, h=0.4, hole_r=1.15, from_end=3.0)  # detent near the trailing end; cap footprint r 1.02
BACKING = 1.6               # inside the wall, behind the pocket
WALL = 2.4                  # cube wall
HALF = 35.0                 # cube half-width (every face is 70 mm across in at least one direction)

# face frames: centre (= marker centre in the config), outward normal, marker "up", entry side along up (-1 = from
# the low-v edge, i.e. the cube bottom; +1 = from the high-v edge, the top wall's front edge)
FACES = {
    "rear":  dict(id=0, c=(0.0, -107.5, 28.5), n=(0, -1, 0), up=(0, 0, 1), entry=-1),
    "top":   dict(id=1, c=(0.0, -72.5, 63.5), n=(0, 0, 1), up=(0, 1, 0), entry=+1),
    "left":  dict(id=2, c=(-35.0, -72.5, 28.5), n=(-1, 0, 0), up=(0, 0, 1), entry=-1),
    "right": dict(id=3, c=(35.0, -72.5, 28.5), n=(1, 0, 0), up=(0, 0, 1), entry=-1),
}


def frame(f):
    """location: local (u, v, w) -> assembly; w = 0 on the face, w < 0 into the cube."""
    n, v = np.array(f["n"], float), np.array(f["up"], float)
    u = np.cross(v, n)
    return cq.Location(cq.Plane(origin=cq.Vector(*f["c"]), xDir=cq.Vector(*u), normal=cq.Vector(*n)))


def lbox(u0, u1, v0, v1, w0, w1):
    return cq.Solid.makeBox(u1 - u0, v1 - v0, w1 - w0, cq.Vector(u0, v0, w0))


def place(shape, f):
    return shape.moved(frame(f))


def face_span(f, cube):
    """v range of the face's flat wall in local coords (the top wall stops short of the cube's front)."""
    loc = cube.moved(frame(f).inverse)
    sl = loc.intersect(lbox(-1, 1, -200, 200, -WALL + 0.2, -0.2))
    bb = sl.BoundingBox()
    return bb.ymin, bb.ymax


def build(cube=None):
    cube = cube or V2.parts["cube"]
    h, a = TILE / 2, TILE / 2 + FLANGE
    tiles, cuts, adds = {}, [], []
    for nm, f in FACES.items():
        vlo, vhi = face_span(f, cube)
        e = f["entry"]
        # tile: from the entry edge of the face to the stop end of the sticker area
        tv0, tv1 = (vlo, h) if e < 0 else (-h, vhi)
        tile = lbox(-h, h, tv0, tv1, -T_FRONT, 0).fuse(lbox(-a, a, tv0, tv1, -T_FRONT - T_BACK, -T_FRONT))
        vb = (tv0 + BUMP["from_end"]) if e < 0 else (tv1 - BUMP["from_end"])
        bump = cq.Solid.makeSphere(BUMP["r"], cq.Vector(0, vb, -T_FRONT - T_BACK + BUMP["r"] - BUMP["h"]),
                                   angleDegrees1=-90, angleDegrees2=90) \
            .intersect(lbox(-5, 5, vb - 5, vb + 5, -10, -T_FRONT - T_BACK + 0.01))
        tiles[nm] = place(tile.fuse(bump).clean(), f)
        # pocket in the cube: open past the entry edge, closed at the stop end
        pv0, pv1 = (vlo - 1.0, h + CLEAR) if e < 0 else (-h - CLEAR, vhi + 1.0)
        pocket = lbox(-h - CLEAR, h + CLEAR, pv0, pv1, -T_FRONT - T_BACK, 1.0)
        groove = lbox(-a - CLEAR, a + CLEAR, pv0, pv1, -T_FRONT - T_BACK - LIP_GAP, -T_FRONT + LIP_GAP)
        hole = cq.Solid.makeCylinder(BUMP["hole_r"], 20, cq.Vector(0, vb, -20), cq.Vector(0, 0, 1))
        cuts.append(place(pocket.fuse(groove, hole), f))
        # backing plate inside the wall, over the face area (inside the hollow only)
        inner = HALF - WALL
        bv0 = max(vlo, -HALF) if e < 0 else max(vlo, -HALF) + WALL
        bv1 = min(vhi, HALF) - WALL if e < 0 else min(vhi, HALF)
        adds.append(place(lbox(-inner, inner, bv0, bv1, -WALL - BACKING, -WALL + 0.01), f))
        # the hole must also pass through the backing (cut after the fuse below)
    c6 = cube.fuse(*adds).clean()
    for k in cuts:
        c6 = c6.cut(k)
    return c6.clean(), tiles


cube6, tiles6 = build()


def check(inter):
    ok = True
    print(f"cube v6: valid {cube6.isValid()} ({len(cube6.Solids())} solid), volume {cube6.Volume():.0f} "
          f"(v3 {V2.parts['cube'].Volume():.0f})")
    ok &= cube6.isValid() and len(cube6.Solids()) == 1
    names = list(tiles6)
    for i, nm in enumerate(names):
        t, f = tiles6[nm], FACES[nm]
        n, v = np.array(f["n"], float), np.array(f["up"], float)
        u = np.cross(v, n)
        e = f["entry"]
        seated = inter(t, cube6)
        mv = {"outward": n, "into the cube": -n, "sideways +": u, "sideways -": -u,
              "toward the stop": -e * v, "toward the entry (detent)": e * v}
        res = {k: inter(t.translate(cq.Vector(*(0.5 * d))), cube6) for k, d in mv.items()}
        held = all(x > 0.01 for x in res.values())
        # the sticker area: TILE square centred on the config's marker centre, flush with the old face
        bb = t.BoundingBox()
        print(f"tile {nm} (ID {f['id']}): valid {t.isValid()}, seated overlap {seated:.2f}; moved 0.5 mm " +
              ", ".join(f"{k} -> {x:.2f}" for k, x in res.items()) + ("" if held else "  <-- NOT HELD"))
        ok &= t.isValid() and seated < 0.01 and held
        for other in names[i + 1:]:
            x = inter(t, tiles6[other])
            if x > 0.01:
                print(f"  tile {nm} x tile {other}: {x:.2f}  <-- COLLISION")
                ok = False
    return ok
