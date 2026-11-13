"""v6 marker cube: snaps onto the bottom plate instead of being screwed on. Only the cube changes; the bottom plate
is the printed Stanford one.

The cube hooks onto the back of the bottom plate with two ears. Each ear sits on a 5 mm shelf of the plate (z -6.5 to
-1.5) and was held by an M3 screw through the shelf's 3.6 mm hole into a nut pocket underneath. Now, per ear:
  * a solid peg (PEG_R) drops into the shelf's existing hole: it locates the cube front-back and sideways, together
    with the ears' inner faces against the plate's tabs (as before);
  * a hook arm hangs down just behind the shelf's rounded back edge and its lip snaps under the shelf: the cube can't
    lift. The arm flexes backwards while the cube is pushed down, and its catch face is flat.
  * Release: from under the cube (it is open underneath), push the two arm tips backwards and lift the cube.
  * Print the cube rear face down (as the v3 cube): the arms then lie flat, with layers along their length.
(The shelf hole is too small for a strong snap pin: it has a 45 deg chamfer from r 1.8 to r 2.4 above the nut pocket.)

The cube carries the four main tracking markers and is held only at its front, so play here tilts them: 0.1 mm of
lift at the ears moves the rear marker about 1 mm. The lip is set to touch the shelf's underside with no gap; untested
in print.
"""
import cadquery as cq
import build_v2 as V2

EARS = ((26.4, -18.1), (-26.4, -18.1))     # old screw axes (both ears)
Z_SHELF_TOP, Z_SHELF_BOT = -1.5, -6.5
HOLE_R, HOLE_STRAIGHT_TO = 1.8, -3.4       # straight part of the shelf hole (chamfer below it)
SHELF_R = 6.0                              # shelf's rounded end around the old screw axis
PEG_R = 1.7                                # 3.4 mm peg in the 3.6 mm hole
ARM = dict(w=5.0, t=1.6, gap=0.2, root_z=6.0, lip=1.3, ramp=1.6)   # hook arm behind the shelf


def cyl(x, y, r, z0, z1):
    return cq.Solid.makeCylinder(r, z1 - z0, cq.Vector(x, y, z0), cq.Vector(0, 0, 1))


def box(x0, x1, y0, y1, z0, z1):
    return cq.Solid.makeBox(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0), cq.Vector(min(x0, x1), min(y0, y1), min(z0, z1)))


def hook_arm(x, y):
    """arm behind the shelf's back edge (the edge is the arc of radius SHELF_R around the old screw axis)."""
    a = ARM
    y_front = y - SHELF_R - a["gap"]                     # arm's front face, just behind the arc's rearmost point
    y_back = y_front - a["t"]
    x0, x1 = x - a["w"] / 2, x + a["w"] / 2
    z_bot = Z_SHELF_BOT - a["ramp"]
    arm = box(x0, x1, y_back, y_front, z_bot, a["root_z"] + 1.0)
    # lip under the shelf: flat top on the shelf's underside, ramped below so pushing the cube down bends the arm back
    lip = cq.Workplane("YZ").polyline([(y_front - 0.01, Z_SHELF_BOT), (y_front + a["lip"], Z_SHELF_BOT),
                                       (y_front - 0.01, z_bot)]).close().extrude(a["w"]).val().translate(cq.Vector(x0, 0, 0))
    # bridge from the arm's top to the ear (the ear's footprint ends at r 5.4 around the old axis)
    bridge = box(x0, x1, y_back, y - 5.0, a["root_z"], 7.7)
    return arm.fuse(lip, bridge)


def build():
    cube = V2.parts["cube"]
    adds = []
    for x, y in EARS:
        # fill the old screw hole and head counterbore, add the peg below the ear and the hook arm behind the shelf
        cube = cube.fuse(cyl(x, y, HOLE_R + 0.01, Z_SHELF_TOP, 1.6), cyl(x, y, 3.26, 1.4, 10.95)).clean()
        adds += [cyl(x, y, PEG_R, HOLE_STRAIGHT_TO + 0.1, Z_SHELF_TOP + 0.01), hook_arm(x, y)]
    return cube.fuse(*adds).clean()


cube6 = build()


def check(inter):
    """cube against the bottom plate: no overlap seated, held in every direction, arm strain, nothing hit below."""
    bp = V2.parts["bottom_plate"]
    seated = inter(cube6, bp)
    held = {lab: inter(cube6.translate(cq.Vector(*v)), bp) for lab, v in
            (("up 0.3", (0, 0, 0.3)), ("down 0.3", (0, 0, -0.3)), ("back 0.3", (0, -0.3, 0)), ("forward 0.3", (0, 0.3, 0)),
             ("sideways +x 0.3", (0.3, 0, 0)), ("sideways -x 0.3", (-0.3, 0, 0)))}
    print(f"cube v6: valid {cube6.isValid()} ({len(cube6.Solids())} solid); seated overlap with the bottom plate "
          f"{seated:.2f} mm3")
    print("cube moved: " + ", ".join(f"{k} -> {v:.1f}" for k, v in held.items()))
    L = ARM["root_z"] - (Z_SHELF_BOT - ARM["ramp"])
    strain = 3 * ARM["t"] * (ARM["lip"] + ARM["gap"]) / (2 * L ** 2)
    print(f"hook arm: {L:.1f} mm flex length, {ARM['lip'] + ARM['gap']:.1f} mm deflection to pass the shelf, "
          f"bending strain {100 * strain:.1f} % (PLA: keep under ~2.5 %)")
    under = cq.Solid.makeBox(200, 200, 60, cq.Vector(-100, -100, Z_SHELF_BOT - 60))
    others = {n: inter(cube6.intersect(under), s) for n, s in V2.parts.items() if n not in ("cube", "bottom_plate")}
    hits = {n: v for n, v in others.items() if v > 0.05}
    print("parts below the shelf touched by the arms:", hits or "none")
    return (cube6.isValid() and len(cube6.Solids()) == 1 and seated < 0.05
            and all(v > 0.05 for v in held.values()) and strain < 0.025 and not hits)
