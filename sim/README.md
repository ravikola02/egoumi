# Sim + preview for the ArUco gripper pipeline

Synthetic head-camera episodes of the v3 gripper, the pose pipeline run on them, and a preview video
(the gripper counterpart of HaWoR's MANO cam/world visualisation, `HaWoR/lib/vis/run_vis2.py`).
Everything runs in the HaWoR conda env from h2r_collection (or any env with pyrender, OpenCV >= 4.7, scipy) with headless EGL.

```bash
cd egoumi                  # repo root
sim/run_sim_test.sh pick_place                                    # render -> pose -> preview
MARKERS=configs/gripper_markers_jawflags.yaml sim/run_sim_test.sh pick_place_jawflags
sim/run_sim_test.sh roll --preset roll --seed 1
sim/run_sim_test.sh blur_table --blur 5 -- --cam-source table     # args after -- go to gripper_pose.py
```
Output: `sim/episodes/<name>/preview/gripper_preview.mp4`, `metrics.json` and `timeline.png`.

## Pieces
| File | Role |
|---|---|
| `sim/export_meshes.py` | v3 STEP → assembly-frame STLs in metres (`sim/assets/meshes/`); needs `.venv-cad` (CadQuery 2.7.0; 2.8.0 is not on PyPI for py3.10) |
| `configs/gripper_markers.yaml` | tag IDs, sizes, poses in the body frame, jaw axes, TCP, width model, table tag |
| `configs/camera_sim.yaml` | pinhole camera, 1920×1080, f = 1050 px, 30 fps |
| `umi_pipe/` | shared code: `model.py` (tags, jaw motion), `render.py` (pyrender scenes), `pose.py` (detection, PnP, jaw fit, table tag) |
| `sim/render_episode.py` | keyframed pick-place / roll, head bob, hand tremor, motion blur, noise, 2× supersampling, hand stand-in, object; writes HaWoR layout `extracted_images/%04d.jpg` + `gt.npz` (+ `gt_masks.npz`) |
| `scripts/gripper_pose.py` | per frame: subpixel ArUco → multi-tag SQPNP+LM (outlier tags dropped; single tag = IPPE disambiguated by the previous frame) → joint body+jaw refine → 1-D jaw travel per side → world pose; short gaps interpolated |
| `scripts/preview_gripper.py` | camera view (estimated mesh over the frame, tags, TCP axes, GT cross) + world view (estimate vs GT ghost, trails, the moving head camera) + timeline; the world view auto-frames the gripper track and the head-camera path; `--style presentation` shows the raw recording next to the reconstruction with title bars and a legend (README media); works without `gt.npz` on real sequences |

`--cam-source`: `gt` = perfect SLAM (isolates the marker pipeline), `none` = camera frame, `table` = per-frame
table-tag pose (needs a table tag; the sim no longer renders one: `Scene3D.add_table(tag=True)` puts it back).

## Results (2026-10-28, sim, f = 1050 px, gripper and markers as in the CAD)
Head camera: the egocentric view behind and above the user's hands (`--head 0.0,-0.42,0.58 --look 0.0,0.12,0.0`).

| Episode | TCP median / p90 / max [mm] | rot median [deg] | jaw width from the head camera | width err median / max [mm] | single-tag frames |
|---|---|---|---|---|---|
| pick_place | 1.7 / 6.1 / 16.2 | 0.31 | 21 % of frames | 0.00 / 0.44 | 25 % |
| roll | 0.60 / 0.98 / 1.8 | 0.07 | 73 % | 0.21 / 0.94 | 1 % |

Jaw width is meant to come from the **gripper (wrist) camera**, as in UMI: it is rigid to the gripper, so each jaw
tag only slides along its rail in that view. `render_capture.py` renders that camera at the GoPro lens, pitched
down 15° (`--wrist-tilt`) so both jaw tags stay in its 120° pinhole frame over the full travel; at 0° they fall just
outside the bottom edge (a real GoPro's ~155° fisheye covers them without the tilt). From 5° down, both tags are
detected at every travel, 95–116 px across. `--state-source aruco` takes the gripper value from that feed and uses
the head camera's jaw sightings only as a fallback.

Findings:
- From the egocentric head view the GoPro and the cube block the holder-top jaw tags at level poses, and they face
  away in top-down grasps; the head sees them in 21–73 % of frames. The wrist camera sees them in every frame.
- Single-tag frames (cube rear only, 25 % of pick_place) cause the large TCP errors: the TCP is about 290 mm from
  the cube, so about 1° at the tags becomes about 5 mm at the fingertips.
- Coplanar body tags (only the two top-cover tags in view) have a two-fold planar ambiguity; `estimate_body` also
  scores both IPPE solutions.
- Earlier numbers in this file (45° jaw flags, blur + table-tag camera) were measured before two renderer fixes
  (sheared jaw-tag textures; sticker normals pointing into the face, so stickers got only ambient light). They are
  removed until re-run; the claim that holder-top jaw tags are never visible came from those bugs.

## Not modelled / caveats
- Lens distortion (renders are ideal pinhole), rolling shutter, auto-exposure, specular glare, sticker wrinkles.
- Gears, linkages and slider are not articulated (hidden under the cover); fingers are rigid boxes in the hand stand-in.
- TCP point is an estimate from the soft-finger bounding box (`tcp_in_body`).
- Real-sequence use: the preview needs `<seq>/camera.yaml`; `--cam-source slam` (HaWoR SLAM) is not wired yet.
