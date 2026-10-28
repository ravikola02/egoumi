import cadquery as cq
from build_v2 import parts, posed, T_CLOSE
from render import render

NEW = ("finger_holder_left_thumb", "finger_holder_right_index")


def inter(a, b):
    try:
        return a.intersect(b).Volume()
    except Exception:
        return -1.0


# baseline overlaps in the ORIGINAL parts (press fits / modelling contacts) to ignore
base = {}
for T in (0.0, 30.0, T_CLOSE, 38.0):
    S = posed(T)
    hits = []
    for n in NEW:
        for m, s in S.items():
            if m == n:
                continue
            v = inter(S[n], s)
            if v > 0.5:
                hits.append(f"{n} x {m}: {v:.1f}")
    print(f"T={T:4.1f}:", hits if hits else "no collisions")

# hand proxy (estimates) for the render only
def capsule(p0, p1, r):
    p0, p1 = cq.Vector(*p0), cq.Vector(*p1)
    d = p1 - p0
    c = cq.Solid.makeCylinder(r, d.Length, p0, d.normalized())
    return c.fuse(cq.Solid.makeSphere(r, p0)).fuse(cq.Solid.makeSphere(r, p1))


def hand(T):
    idx_ring = (40.0 - T, 71.0, -36.5)
    thb_ring = (-50.0 + T, 41.0, -27.0)
    index = capsule((21, 18, -22), (idx_ring[0] + (idx_ring[0] - 21) * 0.35, 89, -40), 8.5)
    thumb = capsule((-22, -8, -20), (thb_ring[0] + (thb_ring[0] + 22) * 0.25, 52, -28), 9.5)
    return [("index (est.)", index), ("thumb (est.)", thumb)]


cols = {"finger_holder_left_thumb": "#E45756", "finger_holder_right_index": "#E45756",
        "grip": "#4C78A8", "slider": "#F58518", "index (est.)": "#F2C9A0", "thumb (est.)": "#E8B88A"}
keep = ("bottom_plate", "grip", "finger_holder_left_thumb", "finger_holder_right_index",
        "soft_finger_left", "soft_finger_right", "cube", "camera_dummy", "top_cover")
for T, tag in ((0.0, "open"), (T_CLOSE, "closed")):
    S = posed(T)
    view = [(n, S[n]) for n in keep] + hand(T)
    render(view, f"v2_{tag}.png", views=((-20, -125), (-89, -90), (0, -90)), colors=cols,
           title=f"UMI pinch v2 - {tag} (T={T:.0f} mm/side); skin-coloured = estimated thumb/index")
render([(n, parts[n]) for n in NEW], "v2_rings.png", views=((-25, -125), (-89, -90)), colors=cols,
       title="Modified holders (thumb ring+arm on left, index ring on right)")
