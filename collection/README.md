# collection/: 3-camera capture → LeRobot dataset

> 🚧 Work in progress. Cameras, sync, cutting and the dataset writers work (tested on the
> [h2r_collection](https://github.com/ravikola02/h2r_collection) Sample session). **End-effector pose
> and gripper values are not reconstructed yet**, so datasets are video-only (or carry a flagged zero placeholder).

The same flow as h2r_collection's pipeline, with HaWoR removed:

1. **Sync:** cameras aligned by creation time and refined by audio cross-correlation (`sync_videos.py`).
2. **Tag + cut:** a local web tagger to mark episodes; **Save + Cut** writes the cut sheet and frame-exact clips (`episode_tagger.py`).
3. **Convert:** clips → LeRobot v3.0 (default) or v2.1 (`convert_to_lerobot.py`).

```bash
# one command: tag in the browser, then convert
python3 collection/run_pipeline.py --session <session>

# several captures into one dataset (additive cut sheet)
python3 collection/run_pipeline.py --session A --session B --collection combo

# re-convert existing clips only; LeRobot 2.1 instead of 3.0; 14-d zero placeholder state
python3 collection/run_pipeline.py --session <session> --skip-tag --overwrite --dataset-version v21 --state-source zeros

# review the cuts before converting
python3 collection/preview_clips.py --session <session>
```
Needs Python ≥ 3.9 with `numpy`, `pandas` and `pyarrow`, plus a non-snap `ffmpeg`/`ffprobe` (the scripts put
`/usr/bin` first on `PATH`: the snap build cannot read external drives and silently breaks the audio sync).

## Data layout
Everything lives under the data root, `<repo>/data` by default or `--data-root` / `EGOUMI_DATA_ROOT`
(for example an external SSD).
```
data/raw/<session>/
├── top_cam/      <one>.MP4    # head (egocentric, required)
├── left_w_cam/   <one>.MP4    # wrist_left  (left gripper camera, optional)
└── right_w_cam/  <one>.MP4    # wrist_right (right gripper camera, optional)

data/output/<collection>/
├── clips.json                 # cut sheet (source of truth; a bare single session keeps data/raw/<s>/clips.json)
├── episode_clips/{ep}_{head,wrist_left,wrist_right}.mp4
└── lerobot_v3/ (or lerobot_v2/)
```
Camera keys: `observation.images.{head,wrist_left,wrist_right}` (identical to h2r_collection).

## Dataset contents
| `--state-source` | `observation.state` / `action` | use |
|---|---|---|
| `none` (default) | absent: video-only dataset | viewing, visual pre-training, checking cuts and sync |
| `zeros` | 14-d zeros in the h2r_collection layout (`left/right.eef.{x,y,z,roll,pitch,yaw}`, `left/right.gripper`), `info.json → egoumi.state_placeholder: true` | smoke-testing training code only; **never train on it** |
| `gripper` (planned) | ArUco gripper TCP pose + jaw width (`scripts/gripper_pose.py`) | the real thing |

- Episode length = the shortest camera clip at `--fps` (floor), so every camera has exactly one frame per row.
- `--fps auto` (default) keeps the cameras' own rate instead of resampling (h2r_collection forced 30 for HaWoR).
- v2.1 output also writes `meta/episodes_stats.jsonl`, so LeRobot's own v2.1 → v3.0 converter accepts it unchanged.
- `meta/egoumi_episodes.json` maps dataset episode indices to cut-sheet names and per-camera frame counts.

## Dummy capture for testing (pyrender)
`sim/render_capture.py` renders a raw session in this layout: head camera + right gripper camera (a GoPro on the
gripper looking along the fingers, 120° pinhole) + a **blank** left wrist feed (black frames, audio only). The three
recordings start at different times, carry deliberately skewed `creation_time` clocks, and share a room
soundtrack with claps, so the audio sync has to do real work. Ground truth goes to `<session>/sim_gt/`.
```bash
PYOPENGL_PLATFORM=egl ~/anaconda3/envs/hawor/bin/python sim/render_capture.py --out data/raw/sim_capture
python3 collection/sync_videos.py --session sim_capture --config data/raw/sim_capture/sim_gt/clips_gt.json \
    --output sim_capture/episode_clips                           # or tag it by hand with run_pipeline.py
python3 collection/convert_to_lerobot.py --cut-sheet data/raw/sim_capture/sim_gt/clips_gt.json \
    --clips-dir data/output/sim_capture/episode_clips --out data/output/sim_capture/lerobot_v3
python3 sim/check_capture_sync.py --session data/raw/sim_capture --clips data/output/sim_capture/episode_clips
```
The left feed's blank recording is deliberate: it stands in for the other hand's wrist camera when only one
gripper is used.

## Provenance
`sync_videos.py`, `episode_tagger.py`, `episodes.py`, `video_io.py`, `lerobot_v30.py` and `preview_clips.py` are
vendored from h2r_collection at commit `0783627` (MIT, same author). Changes are marked `egoumi change`: a
data-root override, and optional state/action in the writers. `lerobot_v21.py` is the generic part of
h2r_collection's converter. `convert_to_lerobot.py`, `state_sources.py` and `run_pipeline.py` are new.

## Verified (2026-10-28, h2r_collection Sample session: 3 cameras, 25 fps, 3 test episodes)
- Cut → convert for all four combinations (v3.0 / v2.1 × `none` / `zeros`): 277 frames, all cameras equal to the
  parquet row count.
- v3.0 loads in LeRobot 0.4.1 (`LeRobotDataset`): episode boundaries, tasks and all three 720p streams decode.
- v2.1 converts with LeRobot's upstream `convert_dataset_v21_to_v30` and then loads.
- The tagger starts, finds the session (head_wrist, overlap 4.93–17.48 s) and serves its UI. Tagging itself is a
  manual browser step and was not exercised here.

## Verified on the pyrender dummy capture (2026-10-28: 3 episodes, head + gripper GoPro + blank left feed)
- Audio sync recovered both offsets exactly (+0.817 s, −0.433 s; confidence 800–920). The MP4 container keeps
  `creation_time` to whole seconds only, so metadata alone was off by 183 ms and 1,567 ms: the claps matter.
- Every clip matches the raw recording frame-exactly at both ends (`sim/check_capture_sync.py`); the blank feed
  stays blank; 525 frames load in LeRobot 0.4.1 with both the PyAV and torchcodec backends.
- **Cutter fix** (`sync_videos.clip_camera`, marked `egoumi change`): for a camera whose start offset is not a whole
  number of frames, clip frame 0 used to duplicate frame 1 (one frame dropped at every cut). It now seeks on the
  camera's own frame grid and caps at `-frames:v` with passthrough timestamps. h2r_collection still has the old code.
