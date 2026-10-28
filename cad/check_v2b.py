import cadquery as cq
from build_v2 import parts, P, fingers

static = ("gear_left", "gear_right", "linkage_left", "linkage_right", "slider")
for T in (30.0, 36.0, 38.0):
    for new, orig, sgn in (("finger_holder_left_thumb", "finger_holder_left", 1),
                           ("finger_holder_right_index", "finger_holder_right", -1)):
        a = parts[new].translate(cq.Vector(sgn * T, 0, 0))
        b = P[orig].translate(cq.Vector(sgn * T, 0, 0))
        others = [parts[m] for m in parts if m not in static and m != new and m not in
                  ("finger_holder_left_thumb", "finger_holder_right_index", "soft_finger_left", "soft_finger_right")]
        # ring-only material = new minus original holder
        ring = a.cut(b)
        hits = {m: ring.intersect(parts[m]).Volume() for m in static}
        print(T, new, "ring-feature vs static mech:", {k: round(v, 1) for k, v in hits.items() if v > 0.1})
# rings vs each other and vs opposite holder when closed
for T in (36.0, 38.0):
    L = parts["finger_holder_left_thumb"].translate(cq.Vector(T, 0, 0))
    R = parts["finger_holder_right_index"].translate(cq.Vector(-T, 0, 0))
    print(T, "L x R", round(L.intersect(R).Volume(), 1),
          "L x fingerR", round(L.intersect(fingers[1].translate(cq.Vector(-T, 0, 0))).Volume(), 1),
          "R x fingerL", round(R.intersect(fingers[0].translate(cq.Vector(T, 0, 0))).Volume(), 1),
          "L x grip", round(L.intersect(parts["grip"]).Volume(), 1),
          "R x grip", round(R.intersect(parts["grip"]).Volume(), 1))
