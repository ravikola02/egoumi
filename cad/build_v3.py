"""UMI pinch v3 = v2 + mirror flaps removed + holder top recess filled + rounded ring edges,
plus print-ready orientation, plate packing and Bambu Studio 3MF projects (H2D, 0.4 mm nozzle).
"""
import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
import os, json, zipfile, uuid
import numpy as np
import cadquery as cq
import build_v2 as V2

OUT = os.path.join(_ROOT, "out_v3")
os.makedirs(OUT, exist_ok=True)
parts = dict(V2.parts)

# ---------------- 1. top cover: remove mirror flaps ----------------
COVER_TOP_Z = 19.8          # top face of the cover plate
keep_center = cq.Solid.makeBox(70, 400, 200, cq.Vector(-35, -200, COVER_TOP_Z))      # GoPro mount, |x|<=35
cut_flaps = cq.Solid.makeBox(400, 400, 200, cq.Vector(-200, -200, COVER_TOP_Z)).cut(keep_center)
top_cover = parts["top_cover"].cut(cut_flaps).clean()
parts["top_cover"] = top_cover

# ---------------- 2. holders: fill top frame recess, round ring outer edges ----------------
def fill_and_round(holder, sx, ring_xc, ring_zc, ring_ro):
    # 0.3 mm deep 20x20 frame groove on the top face (x 50..70 mirrored, y 77.5..97.5, z 16.3..16.6)
    x0, x1 = sorted((sx * 50.0, sx * 70.0))
    patch = cq.Solid.makeBox(x1 - x0, 20.0, 0.31, cq.Vector(x0, 77.5, 16.3))
    h = holder.fuse(patch).clean()
    # round the outer circular edges of the finger ring
    edges = [e for e in h.Edges() if e.geomType() == "CIRCLE" and abs(e.radius() - ring_ro) < 1e-3
             and abs(e.Center().x - ring_xc) < 0.01 and abs(e.Center().z - ring_zc) < 0.01]
    try:
        h = h.fillet(1.0, edges)
    except Exception as ex:
        print("outer ring fillet skipped:", ex)
    return h

parts["finger_holder_left_thumb"] = fill_and_round(
    parts["finger_holder_left_thumb"], -1, V2.THB["x_open"], V2.THB["z"], V2.THB["ri"] + V2.THB["wall"])
parts["finger_holder_right_index"] = fill_and_round(
    parts["finger_holder_right_index"], +1, V2.IDX["x_open"], V2.IDX["z"], V2.IDX["ri"] + V2.IDX["wall"])

# ---------------- 3. print groups (Stanford UMI hardware guide) ----------------
# Stanford: PLA+ 0.6 nozzle, 0.3 layer, 1.2 wall, 1.2 top/bottom, 20% gyroid; grip 100% infill + 1.8 wall;
# holders: supports; linkages: 2.4 wall + 1.8 top/bottom; TPU finger 100% infill lines.
# H2D 0.4 nozzle translation (line width 0.42 outer / 0.45 inner):
BASE = {"layer_height": "0.2", "wall_loops": "3", "top_shell_layers": "6", "top_shell_thickness": "1.2",
        "bottom_shell_layers": "6", "bottom_shell_thickness": "1.2", "sparse_infill_density": "20%",
        "sparse_infill_pattern": "gyroid", "enable_support": "0", "extruder": "1"}
OVR = {
    # grip: Stanford lists no supports, but ~30 cm2 under the top flange overhangs >45 deg -> tree supports
    "grip": {"sparse_infill_density": "100%", "wall_loops": "4", "enable_support": "1", "support_type": "tree(auto)"},
    "linkage_left": {"wall_loops": "6", "top_shell_layers": "9", "top_shell_thickness": "1.8",
                     "bottom_shell_layers": "9", "bottom_shell_thickness": "1.8"},
    "linkage_right": {"wall_loops": "6", "top_shell_layers": "9", "top_shell_thickness": "1.8",
                      "bottom_shell_layers": "9", "bottom_shell_thickness": "1.8"},
    "finger_holder_left_thumb": {"enable_support": "1", "support_type": "tree(auto)"},
    "finger_holder_right_index": {"enable_support": "1", "support_type": "tree(auto)"},
    "soft_finger_left": {"sparse_infill_density": "100%", "sparse_infill_pattern": "line"},
    "soft_finger_right": {"sparse_infill_density": "100%", "sparse_infill_pattern": "line"},
}
GROUPS = {
    "UMI_print1_covers_PLA": ["bottom_plate", "top_cover"],
    "UMI_print2_internals_PLA": ["grip", "slider", "cube", "finger_holder_left_thumb", "finger_holder_right_index",
                                 "gear_left", "gear_right", "linkage_left", "linkage_right"],
    "UMI_print3_fingers_TPU": ["soft_finger_left", "soft_finger_right"],
}

# ---------------- mesh helpers ----------------
def mesh(shape, tol=0.03, ang=0.15):
    v, f = shape.tessellate(tol, ang)
    V = np.array([[p.x, p.y, p.z] for p in v], float)
    F = np.array(f, int)
    # weld duplicated face-boundary vertices (within 1e-3 mm), union-find over close pairs
    from scipy.spatial import cKDTree
    parent = np.arange(len(V))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in cKDTree(V).query_pairs(1e-3):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(len(V))])
    uniq, inv = np.unique(roots, return_inverse=True)
    V = V[uniq]; F = inv[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return V, F


def mesh_stats(V, F):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6
    return int((cnt != 2).sum()), vol


ROTS = {  # candidate print orientations: which original direction ends up DOWN
    "-Z down (as modelled)": np.eye(3),
    "+Z down (flipped)": np.diag([1, -1, -1.0]),
    "-Y down": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]),
    "+Y down": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]]),
    "-X down": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0.0]]),
    "+X down": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0.0]]),
}


def orient(V, F):
    cands = []
    for name, R in ROTS.items():
        W = V @ R.T
        a, b, c = W[F[:, 0]], W[F[:, 1]], W[F[:, 2]]
        n = np.cross(b - a, c - a)
        area = np.linalg.norm(n, axis=1) / 2
        nz = n[:, 2] / (2 * area + 1e-12)
        zmin = W[:, 2].min()
        tri_zmax = np.maximum(np.maximum(a[:, 2], b[:, 2]), c[:, 2])
        on_bed = (nz < -0.999) & (tri_zmax < zmin + 0.05)
        overhang = (nz < -0.7071) & ~on_bed
        cands.append((name, R, area[overhang].sum(), area[on_bed].sum()))
    # keep orientations with at least half of the best flat bed contact, then least overhang
    max_bed = max(c[3] for c in cands)
    ok = [c for c in cands if c[3] >= 0.5 * max_bed]
    name, R, ov, bed = min(ok, key=lambda c: (round(c[2], 3), -c[3]))
    return name, R, ov, bed


# ---------------- 4. prepare printable meshes ----------------
prep = {}
for grp, names in GROUPS.items():
    for nm in names:
        # bottom_plate: finer mesh avoids overlapping triangles at a zero-thickness edge in the original CAD
        V, F = mesh(parts[nm], *({"bottom_plate": (0.01, 0.1)}.get(nm, (0.03, 0.15))))
        name, R, ov, bed = orient(V, F)
        W = V @ R.T
        if np.linalg.det(R) < 0:
            F = F[:, ::-1]
        lo, hi = W.min(0), W.max(0)
        W = W - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        bad_edges, vol = mesh_stats(W, F)
        prep[nm] = dict(V=W, F=F, orient=name, overhang=ov, bed=bed, size=(hi - lo), bad_edges=bad_edges, vol=vol)

# ---------------- 5. pack each group on one H2D plate ----------------
PLATE = dict(x0=27.0, x1=323.0, y0=8.0, y1=312.0)   # inside both H2D nozzle areas (x 25..325)
GAP = 8.0


def pack(names):
    items = []
    for nm in names:
        sx, sy = prep[nm]["size"][0], prep[nm]["size"][1]
        rot = sx < sy and sy <= PLATE["x1"] - PLATE["x0"]   # lay long side along X
        w, d = (sy, sx) if rot else (sx, sy)
        items.append([nm, w, d, rot])
    items.sort(key=lambda t: -t[2])
    x, y, row_d, place = PLATE["x0"], PLATE["y0"], 0.0, {}
    for nm, w, d, rot in items:
        if x + w > PLATE["x1"]:
            x, y, row_d = PLATE["x0"], y + row_d + GAP, 0.0
        if y + d > PLATE["y1"]:
            raise RuntimeError(f"{nm} does not fit on the plate")
        place[nm] = (x + w / 2, y + d / 2, rot)
        x += w + GAP
        row_d = max(row_d, d)
    return place


def write_3mf(path, title, names, place):
    objs, items, cfg = [], [], []
    for i, nm in enumerate(names, start=1):
        p = prep[nm]
        V, F = p["V"], p["F"]
        cx, cy, rot = place[nm]
        if rot:   # rotate 90 deg about Z (baked into mesh)
            V = V @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]).T
        vtx = "\n".join(f'     <vertex x="{a:.5f}" y="{b:.5f}" z="{c:.5f}"/>' for a, b, c in V)
        tri = "\n".join(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in F)
        objs.append(f'  <object id="{i}" name="{nm}" type="model">\n   <mesh>\n    <vertices>\n{vtx}\n    </vertices>\n'
                    f'    <triangles>\n{tri}\n    </triangles>\n   </mesh>\n  </object>')
        items.append(f'  <item objectid="{i}" transform="1 0 0 0 1 0 0 0 1 {cx:.4f} {cy:.4f} 0" printable="1"/>')
        settings = dict(BASE); settings.update(OVR.get(nm, {}))
        meta = "\n".join(f'    <metadata key="{k}" value="{v}"/>' for k, v in settings.items())
        cfg.append(f'  <object id="{i}">\n    <metadata key="name" value="{nm}"/>\n{meta}\n'
                   f'    <part id="{i}" subtype="normal_part">\n      <metadata key="name" value="{nm}"/>\n'
                   f'      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n    </part>\n  </object>')
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
             f' <metadata name="Title">{title}</metadata>\n'
             ' <metadata name="Designer">UMI pinch v3 (derived from Stanford UMI gripper)</metadata>\n'
             ' <resources>\n' + "\n".join(objs) + '\n </resources>\n <build>\n' + "\n".join(items) + '\n </build>\n</model>\n')
    content_types = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
                     ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
                     ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
                     ' <Default Extension="config" ContentType="text/xml"/>\n'
                     ' <Default Extension="png" ContentType="image/png"/>\n</Types>\n')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            ' <Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n</Relationships>\n')
    model_cfg = '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n' + "\n".join(cfg) + '\n</config>\n'
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model)
        z.writestr("Metadata/model_settings.config", model_cfg)


if __name__ == "__main__":
    report = {}
    for grp, names in GROUPS.items():
        place = pack(names)
        write_3mf(os.path.join(OUT, grp + ".3mf"), grp, names, place)
        report[grp] = place
        for nm in names:
            p = prep[nm]
            print(f"{grp:26s} {nm:27s} {p['orient']:22s} size={np.round(p['size'],1)} overhang={p['overhang']:7.0f}mm2 "
                  f"bed={p['bed']:7.0f}mm2 open_edges={p['bad_edges']} vol={p['vol']:9.0f} at=({place[nm][0]:.0f},{place[nm][1]:.0f}) rot90={place[nm][2]}")
    # oriented STLs
    for nm, p in prep.items():
        V, F = p["V"], p["F"]
        tris = V[F]
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        with open(os.path.join(OUT, f"{nm}.stl"), "w") as fh:
            fh.write(f"solid {nm}\n")
            for nn, t in zip(n, tris):
                fh.write(f" facet normal {nn[0]:.6f} {nn[1]:.6f} {nn[2]:.6f}\n  outer loop\n")
                for q in t:
                    fh.write(f"   vertex {q[0]:.5f} {q[1]:.5f} {q[2]:.5f}\n")
                fh.write("  endloop\n endfacet\n")
            fh.write(f"endsolid {nm}\n")
    # updated assembly + modified parts as STEP (design frame)
    asm = cq.Assembly(name="umi_pinch_v3")
    for nm, s in parts.items():
        asm.add(s, name=nm)
    asm.export(os.path.join(OUT, "umi_pinch_v3_assembly.step"))
    for nm in ("top_cover", "finger_holder_left_thumb", "finger_holder_right_index", "slider"):
        cq.exporters.export(parts[nm], os.path.join(OUT, f"{nm}.step"))
    json.dump({g: {k: [round(float(v[0]), 2), round(float(v[1]), 2), bool(v[2])] for k, v in pl.items()} for g, pl in report.items()},
              open(os.path.join(OUT, "_placement.json"), "w"), indent=1)
    print("top_cover valid", parts["top_cover"].isValid(), "vol", round(parts["top_cover"].Volume()),
          "zmax", round(parts["top_cover"].BoundingBox().zmax, 1))
    for nm in ("finger_holder_left_thumb", "finger_holder_right_index"):
        print(nm, "valid", parts[nm].isValid(), "vol", round(parts[nm].Volume()))
