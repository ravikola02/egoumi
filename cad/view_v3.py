import build_v2 as V2
import build_v3 as B
from render import render

cols = {"top_cover": "#E45756", "finger_holder_left_thumb": "#F58518", "finger_holder_right_index": "#F58518",
        "grip": "#4C78A8"}
render([("top_cover (v2, with mirror flaps)", V2.parts["top_cover"]), ], "v3_cover_before.png",
       views=((25, -60), (0, -90)), size=5, labels=True)
render([("top_cover (v3, flaps removed)", B.parts["top_cover"])], "v3_cover_after.png",
       views=((25, -60), (0, -90)), size=5, labels=True)
render([("holder left v3 (top recess filled, rings rounded)", B.parts["finger_holder_left_thumb"])],
       "v3_holder.png", views=((60, -60), (-30, -120)), size=6)
keep = [n for n in B.parts if n != "camera_dummy"] + ["camera_dummy"]
render([(n, B.parts[n]) for n in keep], "v3_assembly.png", views=((20, -60), (-20, -125), (0, -90)),
       colors=cols, title="UMI pinch v3 (mirror flaps removed, palm grip, thumb/index rings)")
