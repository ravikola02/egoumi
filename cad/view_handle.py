import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
import cadquery as cq
from inspect_asm import load_parts
from render import render

P = {}
for nm, s in load_parts(os.path.join(_ROOT, "input", "vis_assembly.step")):
    P.setdefault(nm, s)

render([("handle", P["handle"]), ("grip", P["grip"])], "orig_handle_grip.png",
       views=((15, -60), (15, 30), (0, 0)), title="handle + grip", alpha=0.85)

# section at x=0 (YZ plane) of grip, handle, bottom plate
sec = []
for nm in ("grip", "handle", "bottom_plate", "top_cover", "cube"):
    box = cq.Solid.makeBox(2, 400, 400, cq.Vector(-1, -200, -200))
    sec.append((nm, P[nm].intersect(box)))
render(sec, "orig_section_x0.png", views=((0, 0),), title="section x=0 (YZ)", size=8)

for nm in ("handle", "grip", "bottom_plate"):
    s = P[nm]
    print(nm, "faces", len(s.Faces()), "bb", s.BoundingBox().__dict__ if False else "")
# cylindrical faces (holes / spring bores) on grip & handle
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
for nm in ("grip", "handle", "bottom_plate", "gear_left", "finger_holder_left", "linkage_left"):
    cyl = []
    for f in P[nm].Faces():
        a = BRepAdaptor_Surface(f.wrapped)
        if a.GetType() == GeomAbs_Cylinder:
            c = a.Cylinder()
            ax = c.Axis()
            loc, d = ax.Location(), ax.Direction()
            cyl.append((round(c.Radius(), 2), round(loc.X(), 1), round(loc.Y(), 1), round(loc.Z(), 1),
                        tuple(round(v, 2) for v in (d.X(), d.Y(), d.Z()))))
    print(nm, sorted(set(cyl))[:40])
