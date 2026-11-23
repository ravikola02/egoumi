"""Print files for the DJI action-cam top cover (cad/build_v4.py: the hat-clip bracket for the Osmo Nano).

This is the default top cover from now on; the v3 GoPro cover in cad/out_v3 is legacy.
Writes cad/out_v4/top_cover_dji.{stl,step} (committed) and, in the git-ignored tmp_print/:
  UMI_top_cover_DJI_PLA.3mf      the cover alone
  UMI_print1_covers_DJI_PLA.3mf  bottom plate + DJI cover (replaces UMI_print1_covers_PLA.3mf from cad/out_v3)
Run: cd cad && ../.venv-cad/bin/python export_dji_cover.py
"""
import os
import numpy as np
import cadquery as cq
import build_v3 as V3
import build_v4 as B4

_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_ROOT, "out_v4")
PRINT_DIR = os.path.join(_ROOT, "..", "tmp_print")


def prep(nm, shape):
    V, F = V3.mesh(shape)
    oname, Rm, ov, bed = V3.orient(V, F)
    W = V @ Rm.T
    if np.linalg.det(Rm) < 0:
        F = F[:, ::-1]
    lo, hi = W.min(0), W.max(0)
    W = W - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    bad, vol = V3.mesh_stats(W, F)
    V3.prep[nm] = dict(V=W, F=F, orient=oname, overhang=ov, bed=bed, size=hi - lo, bad_edges=bad, vol=vol)
    print(f"{nm:16s} {oname:22s} size={np.round(hi - lo, 1)} overhang={ov:6.0f} mm2 bed={bed:6.0f} mm2 open_edges={bad}")
    return W, F


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PRINT_DIR, exist_ok=True)
    W, F = prep("top_cover_dji", B4.top_cover)
    V3.OVR["top_cover_dji"] = {"enable_support": "1", "support_type": "tree(auto)"}
    cq.exporters.export(B4.top_cover, os.path.join(OUT, "top_cover_dji.step"))
    t = W[F].astype("<f4")
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]); rec["n"], rec["v"] = n, t
    with open(os.path.join(OUT, "top_cover_dji.stl"), "wb") as fh:
        fh.write(b"top_cover_dji".ljust(80, b" ") + np.uint32(len(t)).tobytes() + rec.tobytes())
    V3.write_3mf(os.path.join(PRINT_DIR, "UMI_top_cover_DJI_PLA.3mf"), "UMI_top_cover_DJI_PLA", ["top_cover_dji"], V3.pack(["top_cover_dji"]))
    names = ["bottom_plate", "top_cover_dji"]           # bottom_plate keeps its v3 prep and settings
    V3.write_3mf(os.path.join(PRINT_DIR, "UMI_print1_covers_DJI_PLA.3mf"), "UMI_print1_covers_DJI_PLA", names, V3.pack(names))
