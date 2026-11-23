"""Print files for the second (left-hand) gripper. The marker sheet for it is assets/print/egoumi_aruco_markers_left.pdf
(IDs 10-17, scripts/make_marker_sheet.py --gripper left); every other part not listed here is the same for both hands.

  tmp_print/main_v3/UMI_print2_internals_PLA_LEFT_HAND.3mf   v3 (main): the internals plate with the two finger
        holders swapped and mirrored (thumb ring on the right holder, index ring on the left). Stanford's holders,
        gears and linkages are exact mirror images, so the racks still mesh. Covers (print 1 / the DJI covers plate)
        and the TPU fingers (print 3) are the same for both hands.
  tmp_print/UMI_ring_mounts_LEFT_HAND_PLA.3mf                v5/v6 (only where cad/build_v5.py exists, i.e. the
        wearables-v5 branch): the left-hand ring-mount pairs; the v6 holders, caps, cube and fingers are hand-neutral.
Run: cd cad && ../.venv-cad/bin/python export_left_hand.py
"""
import os
import numpy as np
import build_v3 as V3

_ROOT = os.path.dirname(os.path.abspath(__file__))
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
    V3.OVR[nm] = {"enable_support": "1", "support_type": "tree(auto)"}
    print(f"{nm:34s} {oname:22s} size={np.round(hi - lo, 1)} open_edges={bad}")


if __name__ == "__main__":
    os.makedirs(os.path.join(PRINT_DIR, "main_v3"), exist_ok=True)
    # v3 internals, left hand
    lh = {"finger_holder_right_thumb": V3.parts["finger_holder_left_thumb"].mirror("YZ"),
          "finger_holder_left_index": V3.parts["finger_holder_right_index"].mirror("YZ")}
    for nm, s in lh.items():
        prep(nm, s)
    names = [n for n in V3.GROUPS["UMI_print2_internals_PLA"] if not n.startswith("finger_holder")] + list(lh)
    V3.write_3mf(os.path.join(PRINT_DIR, "main_v3", "UMI_print2_internals_PLA_LEFT_HAND.3mf"),
                 "UMI_print2_internals_PLA_LEFT_HAND", names, V3.pack(names))
    # ring mounts, left hand (v5/v6 designs; not on every branch)
    if not os.path.isfile(os.path.join(_ROOT, "build_v5.py")):
        print("wrote", os.path.join(PRINT_DIR, "main_v3", "UMI_print2_internals_PLA_LEFT_HAND.3mf"),
              "(no cad/build_v5.py here: no ring-mount file)")
        raise SystemExit(0)
    import build_v5 as B5
    rings = ["ring_mount_thumb_right_holder", "ring_mount_index_left_holder",
             "ring_mount_small_thumb_right_holder", "ring_mount_small_index_left_holder"]
    for nm in rings:
        prep(nm, B5.ring_mounts[nm][0])
    V3.write_3mf(os.path.join(PRINT_DIR, "UMI_ring_mounts_LEFT_HAND_PLA.3mf"), "UMI_ring_mounts_LEFT_HAND_PLA", rings, V3.pack(rings))
    print("wrote", os.path.join(PRINT_DIR, "main_v3", "UMI_print2_internals_PLA_LEFT_HAND.3mf"), "and",
          os.path.join(PRINT_DIR, "UMI_ring_mounts_LEFT_HAND_PLA.3mf"))
