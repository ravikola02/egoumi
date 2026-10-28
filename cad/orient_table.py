import numpy as np
import build_v3 as B

for nm in ("grip", "finger_holder_left_thumb", "finger_holder_right_index", "slider", "cube", "bottom_plate", "top_cover", "gear_left"):
    V, F = B.mesh(B.parts[nm])
    print(f"== {nm}  open_edges={B.mesh_stats(V, F)[0]}")
    for name, R in B.ROTS.items():
        W = V @ R.T
        a, b, c = W[F[:, 0]], W[F[:, 1]], W[F[:, 2]]
        n = np.cross(b - a, c - a); area = np.linalg.norm(n, axis=1) / 2
        nz = n[:, 2] / (2 * area + 1e-12)
        zmin = W[:, 2].min(); tz = np.maximum(np.maximum(a[:, 2], b[:, 2]), c[:, 2])
        bed = (nz < -0.999) & (tz < zmin + 0.05)
        ov = (nz < -0.7071) & ~bed
        h = W[:, 2].max() - zmin
        print(f"   {name:22s} bed={area[bed].sum():7.0f}  overhang={area[ov].sum():7.0f}  height={h:6.1f}")
