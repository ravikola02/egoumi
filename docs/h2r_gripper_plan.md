# h2r_collection: `--source gripper` (ArUco gripper pose + HaWoR head SLAM)

Based on reading h2r_collection `scripts/convert_to_lerobot.py` and HaWoR `scripts/scripts_test_video/{detect_track_video,hawor_video,hawor_slam}.py`,
`lib/pipeline/{masked_droid_slam,est_scale}.py`, `lib/eval_utils/custom_utils.py`.

> **Status 2026-10-29:** implemented in this repo's `collection/` instead of inside h2r_collection: ArUco gripper pose
> (`--state-source aruco`), jaw width from the gripper camera, and head motion from `hawor_slam` called unchanged with
> our gripper + hand mask (`collection/head_slam.py`). Tested on rendered captures (see `collection/README.md`,
> *Head-camera motion*); real captures pending.

> 2026-10-28: the table marker is dropped from the sim and is not needed: like h2r_collection, the dataset world frame
> is each episode's first head-camera frame. The table-tag items below are optional (SLAM scale/drift check only).

## Key finding
`hawor_slam(args, s, e)` only needs: extracted frames (`<seq>/extracted_images/*.jpg`), a focal (`args.img_focal`,
principal point assumed at image centre by `est_calib`), and a dynamic-object mask
`<seq>/tracks_{s}_{e}/model_masks.npy` — bool (T, H, W) at extracted-frame resolution (SLAM and scale estimation
resize it internally). Stage 2 (`hawor_motion_estimation`) only produces that file by rendering MANO hands.
So the hand network, MANO and infiller can be dropped; write our own mask to the same path; HaWoR stays unmodified.

## Stage changes
| Stage in run_hawor_or_cache | hand path (today) | gripper path |
|---|---|---|
| 1 detect_track_video | hand boxes | keep (only to mask the holding hand) |
| 2 hawor_motion_estimation | hand net + MANO → model_masks.npy | replace: ArUco gripper pose + masks to same file |
| 3 hawor_slam | masked DROID-SLAM + Metric3D scale | unchanged |
| 4 hawor_infiller | hand infill | drop |
| state | MANO wrist + thumb spread | gripper TCP from markers + measured jaw width |

## Per-episode flow
1. Frames: extract at 30 fps; undistort to pinhole with principal point at image centre; pass real focal via
   `--img-focal`. Write JPEG with `-q:v 2`.
2. Gripper pose: OpenCV ArucoDetector (subpixel corners); per gripper, rigid multi-tag PnP with all visible body tags
   (tag corner coords in gripper frame from CAD; solvePnP SQPNP/IPPE → LM refine) → T_cam_body; CAD offset → TCP.
   Jaw width = distance between the two holder-top tags along the rail axis in the gripper frame.
3. Masks: projected gripper CAD silhouette at T_cam_body (dilated) ∪ dilated stage-1 hand boxes → model_masks.npy.
4. SLAM: `hawor_slam` unchanged → `load_slam_cam`.
5. World + scale: static table ArUco tag (like UMI tx_slam_tag); least-squares SLAM scale s and T_world_table
   (tag must be static) — replaces/cross-checks Metric3D; gives cross-episode world frame; residual drift = QA score.
6. Compose: T_world_tcp = T_world_cam(SLAM, scaled) · T_cam_body · T_body_tcp; interpolate short gaps, mark long gaps
   invalid; smooth.
7. State: same 14-d layout (h2r_il unchanged): TCP position, gripper-frame RPY, width / calibrated max.

## Repo changes
- configs/gripper_markers.yaml: dictionary, tag IDs per gripper, sizes, tag poses in gripper frame, TCP offset,
  jaw-tag IDs, width calibration.
- scripts/calibrate_camera.py: ChArUco intrinsics + distortion per camera mode.
- scripts/gripper_pose.py: detection, multi-tag PnP, width, axis-overlay QA video.
- scripts/gripper_masks.py: CAD silhouette + hand boxes → model_masks.npy.
- convert_to_lerobot.py: `--source {hand,gripper}`, `run_gripper_or_cache()`, `build_state_14d_gripper()`,
  BRIDGE_VERSION "aruco-v0.1", source recorded in info.json["h2r"].

## Markers on the v3 gripper (reuse UMI DICT_4X4_50 scheme)
- Top cover top face: 2 × ~35 mm (space freed by removing mirror flaps).
- Cube: 60 mm on rear + top faces (face the head camera).
- Holder tops: ~18 mm each (filled 20 × 20 mm area) → jaw width; visible above the finger rings.
- Second gripper: separate ID set.

## Capture rules
Stabilisation off; fixed resolution/FOV (calibrate that mode); tags ≥ ~25 px (20 mm at 0.5 m ≈ 40 px in 4K wide,
≈ 20 px in 1080p — estimate); shutter ≈ 1/500 s or faster; table tag in view.

## Validation (proposed targets)
1. Static: gripper at known positions on a printed grid.
2. Dynamic: gripper bolted to UR/Dobot flange, robot FK as ground truth. Reference: head-camera glove paper
   (arXiv 2610.27622) 6.5 mm / 1.6° median, 93–100 % coverage. Target ≤ 10 mm, ≤ 3°, ≥ 90 % frames.
3. SLAM: Metric3D vs table-tag scale; table-tag drift per episode.
4. End-to-end: 5 episodes, overlay videos, replay one on the Dobot.

## Timeline (estimate)
Days 1–2 calibration/undistortion/markers · 3–5 pose+width module · 6–7 masks + SLAM end-to-end ·
8–9 table-tag scale/world, state builder · 10–12 robot validation, then data collection.

## Risks / scope
- Occlusion and blur → multi-tag rigid PnP + capture rules.
- Gripper data fits Dobot X-Trainer (parallel gripper); keep the hand path for G1/H2 dexterous hands.
- Licensing: HaWoR CC-BY-NC-ND, MANO non-commercial. Gripper path drops MANO/HaWoR net; calling DROID-SLAM (BSD)
  and Metric3D directly could remove the NC dependency (Metric3D licence unverified; not legal advice).
