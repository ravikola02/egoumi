import numpy as np
import build_v3 as B

V, F = B.mesh(B.parts["bottom_plate"])
e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
u, cnt = np.unique(e, axis=0, return_counts=True)
bad = u[cnt != 2]
print("open/non-manifold edges:", len(bad), "counts:", np.unique(cnt[cnt != 2], return_counts=True))
P = V[bad]
mid = P.mean(1)
L = np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
order = np.argsort(mid[:, 0])
for k in order:
    print(np.round(mid[k], 2), "len", round(L[k], 3))
# check B-rep: free edges / small faces
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE
s = B.parts["bottom_plate"]
print("faces", len(s.Faces()), "shells", len(s.Shells()), "solids", len(s.Solids()), "closed shells", [sh.Closed() for sh in s.Shells()])
small = [f for f in s.Faces() if f.Area() < 0.05]
print("tiny faces:", len(small), [round(f.Area(), 4) for f in small][:10])
