"""Export the v3 assembly parts as STL meshes in the ASSEMBLY frame (mm -> m) for the sim.

out_v3/*.stl are print-oriented, so the sim tessellates umi_pinch_v3_assembly.step instead.
Needs CadQuery (any 2.x): `.venv-cad/bin/python sim/export_meshes.py` from umi_project/.
Writes sim/assets/meshes/<part>.stl and sim/assets/meshes/bounds.json.
"""
import os, sys, json
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "..", "cad"))
import cadquery as cq
from inspect_asm import load_parts

STEP = os.path.join(_ROOT, "..", "cad", "out_v3", "umi_pinch_v3_assembly.step")
OUT = os.path.join(_ROOT, "assets", "meshes")
os.makedirs(OUT, exist_ok=True)

bounds = {}
for name, shp in load_parts(STEP):
    m = shp.scale(1e-3)                               # mm -> m, frame unchanged
    cq.exporters.export(m, os.path.join(OUT, f"{name}.stl"), tolerance=1e-4, angularTolerance=0.2)
    bb = shp.BoundingBox()
    bounds[name] = {"min": [bb.xmin, bb.ymin, bb.zmin], "max": [bb.xmax, bb.ymax, bb.zmax]}
    print(f"{name:26s} x[{bb.xmin:7.1f},{bb.xmax:7.1f}] y[{bb.ymin:7.1f},{bb.ymax:7.1f}] z[{bb.zmin:7.1f},{bb.zmax:7.1f}]")
with open(os.path.join(OUT, "bounds.json"), "w") as f:
    json.dump(bounds, f, indent=1)
