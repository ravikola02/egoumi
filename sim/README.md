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
| `scripts/preview_gripper.py` | camera view (estimated mesh over the frame, tags, TCP axes, GT cross) + world view (estimate vs GT ghost, trails, frustums) + timeline; works without `gt.npz` on real sequences |

`--cam-source`: `gt` = perfect SLAM (isolates the marker pipeline), `table` = per-frame table-tag pose, `none` = camera frame.

## Results (2026-10-28, sim, f = 1050 px, gripper and markers as in the CAD)
The default head camera is the **overhead-right view** (`--head 0.18,0.10,0.74 --look 0.0,0.10,0.08`): about 74 cm
above the table, 18 cm right of the work area, looking down. It was picked from a sweep of 25 placements around the
workspace for jaw-tag visibility. The original view (`--head 0.0,-0.42,0.58 --look 0.0,0.12,0.0`), from behind and
above the user's hand, is kept for comparison.

| Episode | Head view | TCP median / p90 / max [mm] | rot median [deg] | width measured | width err median / max [mm] | single-tag frames |
|---|---|---|---|---|---|---|
| pick_place | original | 1.8 / 6.4 / 16.2 | 0.31 | 21 % | 0.00 / 0.44 | 25 % |
| pick_place | **overhead-right** | **0.41 / 0.74 / 1.8** | 0.12 | **95 %** | 0.03 / 1.1 | 0 % |
| roll | overhead-right | 0.18 / 0.39 / 1.4 | 0.12 | 100 % | 0.06 / 0.43 | 0 % |

Jaw-tag visibility from the placement sweep (pick_place, every 6th frame, real detector):

| Placement | either jaw tag | both jaw tags | ≥ 2 body tags |
|---|---|---|---|
| original (behind the hand) | 18 % | 3 % | 70 % |
| overhead-right (default) | 97 % | 61 % | 100 % |
| facing the user (opposite side of the table) | 100 % | 100 % | 100 % (but no longer a head camera) |

Findings:
- The holder-top jaw tags are visible from a head camera; *where* the head is matters. From behind the hand, the
  GoPro and the cube block them at level poses and they face away in top-down grasps. From above and to the right of
  the work they are seen in almost every frame.
- Single-tag frames (cube rear only) caused the large TCP errors in the original view: the TCP is about 290 mm from
  the cube, so about 1° at the tags becomes about 5 mm at the fingertips. The overhead-right view has none.
- Coplanar body tags (only the two top-cover tags in view) have a two-fold planar ambiguity; `estimate_body` now
  scores both IPPE solutions as well as SQPNP.
- Earlier numbers in this file (45° jaw flags, blur + table-tag camera) were measured before two renderer fixes
  (sheared jaw-tag textures; sticker normals pointing into the face, so stickers got only ambient light). They are
  removed until re-run. The single-tag table-tag camera pose was unusable then; that needs re-checking too.

## Not modelled / caveats
- Lens distortion (renders are ideal pinhole), rolling shutter, auto-exposure, specular glare, sticker wrinkles.
- Gears, linkages and slider are not articulated (hidden under the cover); fingers are rigid boxes in the hand stand-in.
- TCP point is an estimate from the soft-finger bounding box (`tcp_in_body`).
- Real-sequence use: the preview needs `<seq>/camera.yaml`; `--cam-source slam` (HaWoR SLAM) is not wired yet.
