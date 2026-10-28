# EgoUMI: UMI pinch gripper + ArUco head-camera tracking

> 🚧 **Work in progress.** The hardware is designed but not yet printed or fit-tested, and the tracking pipeline
> has so far been tested **only in simulation**. Expect breaking changes.

A non-powered, UMI-style handheld gripper that you close with a thumb + index pinch, tracked with ArUco markers
from an egocentric **head** camera, so that human demonstrations can become robot-learning datasets.

It builds on my [**h2r_collection**](https://github.com/ravikola02/h2r_collection) pipeline (egocentric head video →
HaWoR hand pose + head SLAM → LeRobot dataset). The goal is a `--source gripper` mode there that replaces the
HaWoR hand pose with the gripper pose from the markers and keeps HaWoR's head-camera SLAM
([plan](docs/h2r_gripper_plan.md)).

[![Sim preview: estimated gripper over the head-camera frame, world view vs ground truth, error timeline](assets/preview_pick_place.gif)](assets/preview_pick_place.mp4)

*Simulated pick-and-place ([mp4](assets/preview_pick_place.mp4), [wrist-roll episode](assets/preview_roll.mp4)).
Left: head-camera frame with the estimated gripper mesh (purple), detected tags and the TCP axes. Right: world view,
estimate vs ground-truth ghost (green), TCP trails and the head camera. Bottom: TCP error, jaw width estimate vs
ground truth, and the number of body tags used.*

## What's here
| Part | Status |
|---|---|
| `cad/`: gripper derived from the [Stanford UMI](https://umi-gripper.github.io/) CAD; pistol grip kept, trigger replaced by a thumb ring and an index ring on the finger holders; mirrors removed to make room for tags | CAD done and collision-checked, print files for a Bambu H2D in `cad/out_v3` ([print notes](docs/PRINT_NOTES.md)); **not printed yet** |
| `sim/`: synthetic head-camera episodes (textured CAD, ArUco stickers, hand stand-in, table tag, object, head bob, tremor, blur, noise) in HaWoR's frame layout, with ground truth | working |
| `umi_pipe/` + `scripts/gripper_pose.py`: subpixel ArUco → multi-tag PnP (SQPNP + LM, outlier rejection, single-tag disambiguation) → joint body + jaw refinement → jaw width → world TCP pose | working in sim |
| `scripts/preview_gripper.py`: the preview above, the gripper counterpart of HaWoR's MANO cam/world visualisation; also runs on real sequences without ground truth | working |
| h2r_collection integration (`--source gripper`, camera calibration, SLAM masks, 14-d state) | planned |

## Quick start (simulation)
Runs in a Python env with `pyrender`, `trimesh`, `opencv-python` (≥ 4.7, ArUco), `scipy`, `pyyaml` and
`matplotlib` (the HaWoR env from h2r_collection has all of them), with headless EGL rendering and `ffmpeg`.
```bash
sim/run_sim_test.sh pick_place                                            # render -> pose -> preview
MARKERS=configs/gripper_markers_jawflags.yaml sim/run_sim_test.sh pick_place_jawflags
sim/run_sim_test.sh roll --preset roll --seed 1
sim/run_sim_test.sh blur_table --blur 5 -- --cam-source table              # args after -- go to gripper_pose.py
# output: sim/episodes/<name>/preview/{gripper_preview.mp4,metrics.json,timeline.png}
```
Set `PY=/path/to/python` if your interpreter is not the HaWoR conda env. `sim/assets/meshes` is checked in;
`sim/export_meshes.py` regenerates it from the STEP file (needs CadQuery).

## First sim results
| Episode | TCP error median / p90 / max | Jaw width measured |
|---|---|---|
| pick-and-place, jaw tags on the holder tops (original plan) | 1.7 / 5.8 / 15.8 mm | **0 % of frames** |
| pick-and-place, jaw tags on 45° flags facing the user | 1.4 / 5.8 / 15.8 mm | 55 %, median error 0.08 mm |
| wrist roll ±35°, flags | 0.56 / 0.97 / 2.3 mm | 56 %, max error 1.8 mm |
| pick-and-place + motion blur, camera pose from the table tag | 7.2 / 52 / 90 mm | 60 % |

What the sim showed so far (details in [sim/README.md](sim/README.md)):
- Tags on the holder tops are never visible from a head camera, so jaw width needs user-facing tags. The 45° flags
  are a sim-only candidate and are not in the CAD yet.
- Frames where only one tag is visible cause most of the error: the TCP is about 29 cm from the cube tags.
- A single table tag is not a usable camera pose (pose flips). Camera motion comes from SLAM; the table tag only
  gives scale and the world frame.

Not modelled yet: lens distortion, rolling shutter, glare, real SLAM drift. The TCP point is an estimate from the CAD.

## Layout
```
cad/            CadQuery build scripts (v1 → v3), Stanford UMI input STEP, v3 print files, renders
configs/        marker layouts (body frame = CAD assembly frame, metres), sim camera
docs/           pipeline plan for h2r_collection, print notes
sim/            episode renderer, mesh export, one-command test runner
umi_pipe/       shared model / rendering / pose-estimation code
scripts/        gripper_pose.py (pipeline), preview_gripper.py (preview)
assets/         README media
```

## Credits and licence
- Gripper mechanism and base CAD: [Universal Manipulation Interface](https://umi-gripper.github.io/) (Chi et al.,
  Stanford). The files in `cad/input/` are their design exported from Onshape; their terms apply to that design.
- Data pipeline: [h2r_collection](https://github.com/ravikola02/h2r_collection), which uses
  [HaWoR](https://github.com/ThunderVVV/HaWoR) (CC BY-NC-ND).
- The code in this repository is MIT licensed (see `LICENSE`).
