"""Verify the 3MF projects: spec-level read with lib3mf, plate bounds, overlaps between footprints,
per-object settings parse and key validity against Bambu Studio's PrintConfig.cpp; render plates."""
import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
import zipfile, re, os, glob
import xml.etree.ElementTree as ET
import numpy as np
import lib3mf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(_ROOT, "out_v3")
_PC_PATH = os.environ.get("BBS_PRINTCONFIG", "")  # optional: Bambu Studio src/libslic3r/PrintConfig.cpp
PC = open(_PC_PATH).read() if _PC_PATH and os.path.isfile(_PC_PATH) else None
H2D = dict(x=(0, 350), y=(0, 320), left=(0, 325), right=(25, 350), zmax=325)

wrapper = lib3mf.get_wrapper()
for path in sorted(glob.glob(os.path.join(OUT, "*.3mf"))):
    print("==", os.path.basename(path))
    model = wrapper.CreateModel()
    model.QueryReader("3mf").ReadFromFile(path)
    objs = {}
    it = model.GetMeshObjects()
    while it.MoveNext():
        mo = it.GetCurrentMeshObject()
        objs[mo.GetResourceID()] = mo
    print("   lib3mf read OK: objects", len(objs))
    # build items + world bbox
    fps = []
    bi = model.GetBuildItems()
    while bi.MoveNext():
        item = bi.GetCurrent()
        mo = item.GetObjectResource()
        t = item.GetObjectTransform()
        tx, ty, tz = t.Fields[3][0], t.Fields[3][1], t.Fields[3][2]
        verts = np.array([[v.Coordinates[0], v.Coordinates[1], v.Coordinates[2]] for v in mo.GetVertices()])
        W = verts + [tx, ty, tz]
        lo, hi = W.min(0), W.max(0)
        name = mo.GetName()
        ok = (lo[0] >= H2D["right"][0] and hi[0] <= H2D["left"][1] and lo[1] >= 0 and hi[1] <= 320 and hi[2] <= H2D["zmax"] and abs(lo[2]) < 1e-6)
        print(f"   {name:27s} x[{lo[0]:6.1f},{hi[0]:6.1f}] y[{lo[1]:6.1f},{hi[1]:6.1f}] z[{lo[2]:.2f},{hi[2]:6.1f}] tris={mo.GetTriangleCount():6d} in_both_nozzle_area={ok}")
        fps.append((name, lo, hi, W, mo))
    # footprint overlap check (bbox)
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            a, b = fps[i], fps[j]
            if a[1][0] < b[2][0] and b[1][0] < a[2][0] and a[1][1] < b[2][1] and b[1][1] < a[2][1]:
                print("   BBOX OVERLAP:", a[0], b[0])
    # per-object settings
    z = zipfile.ZipFile(path)
    cfg = ET.fromstring(z.read("Metadata/model_settings.config"))
    for o in cfg.findall("object"):
        meta = {m.get("key"): m.get("value") for m in o.findall("metadata")}
        bad = [k for k in meta if PC and k != "name" and f'this->add("{k}"' not in PC]
        part = o.find("part")
        assert part is not None and part.get("id") == o.get("id")
        print(f"   settings {meta['name']:27s} {' '.join(f'{k}={v}' for k, v in meta.items() if k != 'name')}" + (f"  UNKNOWN KEYS {bad}" if bad else ""))
    # render plate (top view)
    fig, ax = plt.subplots(figsize=(7, 6.4))
    ax.add_patch(plt.Rectangle((0, 0), 350, 320, fill=False, ec="#888", lw=1))
    ax.add_patch(plt.Rectangle((25, 0), 300, 320, fill=True, fc="#eef4ea", ec="none"))
    for name, lo, hi, W, mo in fps:
        tri = np.array([[t.Indices[0], t.Indices[1], t.Indices[2]] for t in mo.GetTriangleIndices()])
        P = W[tri][:, :, :2]
        from matplotlib.collections import PolyCollection
        ax.add_collection(PolyCollection(P, facecolors="#4C78A8", edgecolors="none", alpha=0.35))
        ax.text((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, name.replace("finger_holder_", "holder_").replace("_", "\n", 1),
                ha="center", va="center", fontsize=7)
    ax.set_xlim(-5, 355); ax.set_ylim(-5, 325); ax.set_aspect("equal")
    ax.set_title(os.path.basename(path) + "  (H2D plate, green = reachable by both nozzles)", fontsize=9)
    fig.tight_layout(); fig.savefig(path.replace(".3mf", "_plate.png"), dpi=110); plt.close(fig)
