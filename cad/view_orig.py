import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
from inspect_asm import load_parts
from render import render

parts = load_parts(os.path.join(_ROOT, "input", "vis_assembly.step"))
# dedupe names for second finger
seen = {}
named = []
for nm, s in parts:
    seen[nm] = seen.get(nm, 0) + 1
    named.append((nm if seen[nm] == 1 else f"{nm}_{seen[nm]}", s))

render(named, "orig_full.png", views=((20, -60), (20, 30), (90, -90)), title="Original UMI assembly")
mech = [p for p in named if p[0] in ("gear_left", "gear_right", "linkage_left", "linkage_right",
                                     "handle", "finger_holder_left", "finger_holder_right")]
render(mech, "orig_mech.png", views=((90, -90), (20, -60), (0, -90)), title="Mechanism only")
mech2 = [p for p in named if p[0] in ("gear_left", "gear_right", "linkage_left", "linkage_right")]
render(mech2, "orig_gears.png", views=((90, -90), (30, -60)), title="Gears + linkages")
