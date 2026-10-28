import cadquery as cq
from build_v1 import parts, RING_X_OPEN
from render import render
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder

# hand-mount hole positions
cyl = set()
for f in parts["hand_mount"].Faces():
    a = BRepAdaptor_Surface(f.wrapped)
    if a.GetType() == GeomAbs_Cylinder:
        c = a.Cylinder(); l = c.Axis().Location()
        cyl.add((round(c.Radius(), 2), round(l.X(), 1), round(l.Y(), 1)))
print("mount holes:", sorted(c for c in cyl if c[0] < 4))

def inter(a, b):
    try:
        return a.intersect(b).Volume()
    except Exception as e:
        return -1

new = ["hand_mount", "finger_holder_left_ring", "finger_holder_right_ring", "slider"]
for n in new:
    for m, s in parts.items():
        if m == n:
            continue
        v = inter(parts[n], s)
        if v > 1:
            print(f"OPEN overlap {n} x {m}: {v:.0f} mm3")

# closed state: holders + fingers move inward by T
for T in (30.0, 36.0, 40.0):
    L = parts["finger_holder_left_ring"].translate(cq.Vector(T, 0, 0))
    R = parts["finger_holder_right_ring"].translate(cq.Vector(-T, 0, 0))
    print(f"T={T}: ring-ring overlap {inter(L, R):.0f}, holderL x mount {inter(L, parts['hand_mount']):.0f}, "
          f"holderR x mount {inter(R, parts['hand_mount']):.0f}, ring centre spacing {2*(RING_X_OPEN-T):.0f} mm")

view = [(n, s) for n, s in parts.items()]
cols = {"finger_holder_left_ring": "#E45756", "finger_holder_right_ring": "#E45756",
        "hand_mount": "#54A24B", "slider": "#F58518"}
render(view, "v1_open.png", views=((20, -60), (-25, -120), (0, -90)), colors=cols, title="UMI pinch v1 - open")
T = 36.0
closed = []
for n, s in parts.items():
    if n in ("finger_holder_left_ring", "soft_finger_left"):
        s = s.translate(cq.Vector(T, 0, 0))
    if n in ("finger_holder_right_ring", "soft_finger_right"):
        s = s.translate(cq.Vector(-T, 0, 0))
    closed.append((n, s))
render(closed, "v1_closed.png", views=((-25, -120), (0, -90), (-90, -90)), colors=cols,
       title=f"UMI pinch v1 - jaws closed {T:.0f} mm/side (gears/slider not re-posed)")
render([(n, parts[n]) for n in ("finger_holder_left_ring", "hand_mount", "slider")], "v1_newparts.png",
       views=((25, -60), (-30, -120)), colors=cols, title="New / modified parts")
