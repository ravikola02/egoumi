# Sim + preview for the ArUco gripper pipeline

Synthetic head-camera episodes of the v3 gripper, the pose pipeline run on them, and a preview video
(the gripper counterpart of HaWoR's MANO cam/world visualisation, `HaWoR/lib/vis/run_vis2.py`).
Everything runs in the HaWoR conda env from h2r_collection (or any env with pyrender, OpenCV >= 4.7, scipy) with headless EGL.

```bash
cd umi_pinch_gripper        # repo root
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
| `scripts/preview_gripper.py` | camera view (estimated mesh over the frame, tags, TCP axes, GT cross) + world view (estimate vs GT ghost, trails, frustums) + timeline; works without `gt.npz` on real sequences |

`--cam-source`: `gt` = perfect SLAM (isolates the marker pipeline), `table` = per-frame table-tag pose, `none` = camera frame.

## Results so far (2026-10-28, sim, f = 1050 px, head about 0.6 m from the gripper)
| Episode | TCP median / p90 / max [mm] | rot median [deg] | width measured | width err, measured [mm] |
|---|---|---|---|---|
| pick_place, holder-top jaw tags (CAD plan) | 1.7 / 5.8 / 15.8 | 0.31 | **0 %** | – |
| pick_place, 45° jaw flags (candidate) | 1.4 / 5.8 / 15.8 | 0.22 | 55 % | median 0.08, max 13.4 (0.05 / 2.6 with ≥ 2 body tags) |
| roll, jaw flags | 0.56 / 0.97 / 2.3 | 0.09 | 56 % | median 0.17, max 1.8 |
| pick_place + blur 5, cam from table tag | 7.2 / 52 / 90 | 1.56 | 60 % | – |

Findings:
- Holder-top jaw tags (plan) are never detected from the head camera: they sit behind the GoPro and cube at level
  poses and are edge-on (about 90°) in top-down grasps. Jaw width needs tags that face the user. The 45° flags in
  `gripper_markers_jawflags.yaml` are a sim-only candidate, not in the CAD.
- Single-tag frames (cube rear only, 21 % of pick_place) give the large TCP errors: the TCP is about 290 mm from
  the cube, so about 1° at the tags becomes about 5 mm at the fingertips. When the body pose rests on one tag the
  jaw fit absorbs its error (7–13 mm width error at under 1 px reprojection), so `poses.npz` marks those frames
  `width_weak`.
- A single 150 mm table tag is not a usable camera pose (IPPE flips: 12 mm median, 153 mm max). Keep SLAM for
  T_world_cam, as the plan does; use the tag for scale and the world frame only, or use a multi-tag board.
- Motion blur (5 sub-frames, 1/120 s) costs little in the camera frame: body pose 1.4 mm median, 5.7 mm p90.

## Not modelled / caveats
- Lens distortion (renders are ideal pinhole), rolling shutter, auto-exposure, specular glare, sticker wrinkles.
- Gears, linkages and slider are not articulated (hidden under the cover); fingers are rigid boxes in the hand stand-in.
- TCP point is an estimate from the soft-finger bounding box (`tcp_in_body`).
- Real-sequence use: the preview needs `<seq>/camera.yaml`; `--cam-source slam` (HaWoR SLAM) is not wired yet.
