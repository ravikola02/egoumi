# collection/: 3-camera capture → LeRobot dataset

> 🚧 Work in progress. Cameras, sync, cutting and the dataset writers work (tested on the
> [h2r_collection](https://github.com/ravikola02/h2r_collection) Sample session and a rendered capture). The gripper
> trajectory (`--state-source aruco`) works **in sim only**: head-camera motion still comes from the sim, because
> SLAM is not wired in yet. Without it, datasets are video-only (or carry a flagged zero placeholder).

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

## Cameras
Recording settings and per-session checklist: [docs/recording_setup_osmo_nano.md](../docs/recording_setup_osmo_nano.md).

The same setup as h2r_collection: **DJI Osmo Nano** (OW001) on the head and as the wrist cameras. The gripper camera
is the right wrist feed, on the gripper's camera mount; the left wrist feed may be blank. Intrinsics are the ones
h2r_collection used, in `configs/cameras_osmo_nano.yaml` (the default `--camera` for `--state-source aruco`):

| | h2r recording modes | intrinsics |
|---|---|---|
| head | 1920×1080 @ 59.94 fps, or 3840×2160 10-bit @ 59.94 / 47.95 | HaWoR focal 600 px at 1920 wide (1200 px at 3840), principal point at the image centre, no distortion model |
| wrist / gripper camera | 1920×1080 @ 29.97 fps | the same lens; gripper camera mount: nominal, at the GoPro-mount lens position, pitched 15° down |

- Intrinsics are rescaled to each clip's resolution, and every dataset row takes the frame nearest its timestamp
  from each camera's own rate, so a 59.94 fps head and 29.97 fps wrists line up on the 30 fps rows.
- `--fps auto` uses the slowest camera, rounded to an integer (30 here), the rate h2r_collection resampled to.
- **Unverified:** lens distortion. The Osmo Nano is a 12 mm-equivalent wide lens; HaWoR never modelled distortion,
  so markers near the image edge will be biased until a calibration replaces these numbers. The gripper-camera
  mount is nominal too: the Osmo Nano's lens is not where the GoPro dummy's is.

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
| `aruco` | right gripper TCP pose reconstructed from the **head clips** with the ArUco pipeline (`umi_pipe.pose`); `right.gripper` from the jaw tags seen by the **gripper (wrist) camera** (head-camera jaw sightings as a fallback), interpolated across frames without one; left arm zero. Causal EMA α 0.18 on position (`--ema-pos`) and on orientation (`--ema-rot`, filtered on SO(3) so the ±π wrap cannot break it; defaults to the position α so the whole pose has one lag), none on the gripper (`--ema-grip 1.0`); see below for how α was chosen. Head-camera motion comes from `--cam-motion` (only `sim_gt` for now: SLAM is not wired). With `--sim-gt` the same dataset also gets **`observation.state_gt` / `action_gt`**, the ground truth in the same frame and layout | testing reconstruction against ground truth; real captures once SLAM is in |
| `sim_gt` | the sim ground truth only (`sim/render_capture.py` sessions) | testing the trajectory plumbing |

- Episode length = the shortest camera clip at `--fps` (floor), so every camera has exactly one frame per row.
- `--fps auto` (default) keeps the cameras' own rate instead of resampling (h2r_collection forced 30 for HaWoR).
- v2.1 output also writes `meta/episodes_stats.jsonl`, so LeRobot's own v2.1 → v3.0 converter accepts it unchanged.
- `meta/egoumi_episodes.json` maps dataset episode indices to cut-sheet names and per-camera frame counts, plus the
  state source's per-episode report (pose coverage, single-tag share, gripper coverage, error vs ground truth).
- Trajectory convention (as in h2r_collection): world = the episode's first head-camera frame, OpenCV optical axes,
  metres; eef frame = h2r_collection's hand frame on the gripper: +z approach (body +Y, finger direction), +y jaw
  axis toward the thumb-side jaw (body -X), +x grip-plane normal (body -Z); rpy = intrinsic XYZ, wrapped; gripper 1 = closed;
  `action[t] = state[t+1]`.

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
- Trajectory test (`run_pipeline.py --skip-tag --state-source aruco --sim-gt ... --camera ...`; egocentric head
  camera, gripper camera pitched 15° down): one dataset with both `observation.state` (predicted) and
  `observation.state_gt` (raw ground truth). Pose on 100 % of frames; gripper from the wrist camera on 100 %, within
  1.1 mm of jaw width. Before the EMA, TCP error median 1.5 / 0.56 / 2.4 mm (pick-and-place / wrist roll / put back),
  with single-tag stretches in the head view reaching 17–30 mm. `state_gt` matches the independent `sim_gt` build
  exactly; `action` / `action_gt` are the next state within every episode.
- **Choosing the EMA** (`sim/ema_sweep.py`, same criterion as the h2r datasets): the least smoothing whose one-step
  delta noise, which is what the actions see, reaches the G1 encoder-FK level (0.30 / 0.26 / 0.27 mm per-frame floor ->
  0.68 mm per step over 3 axes). The raw ArUco deltas are 4.2 mm.

  | position α | lag | delta noise | error vs raw GT, median |
  |---|---|---|---|
  | 1.0 (off) | 0 ms | 4.20 mm | 1.2 mm |
  | 0.3 | 78 ms | 1.08 mm | 9.0 mm |
  | 0.2 | 133 ms | 0.72 mm | 14.9 mm |
  | **0.18 (default)** | **152 ms** | **0.65 mm** | **16.5 mm** |
  | 0.164 (h2r kitting) | 170 ms | 0.59 mm | 18.4 mm |

  Orientation uses the same α (one lag for the whole pose), filtered on SO(3):

  | orientation α | lag | 1-step delta noise | noise vs equally filtered GT, median / max | error vs raw GT, median |
  |---|---|---|---|---|
  | 1.0 (off) | 0 ms | 0.76° | 0.21 / 4.0° | 0.21° |
  | 0.3 | 78 ms | 0.19° | 0.15 / 2.1° | 1.5° |
  | **0.18 (default)** | **152 ms** | **0.11°** | **0.15 / 1.6°** | **2.9°** |

  The EMA does little for the per-frame error itself: that is slow bias from single-tag stretches, not jitter. The
  gripper needs no EMA: the wrist-camera width is already clean (median error 0.00 mm), and α 0.3 would only add lag
  (p90 error 0.87 -> 4.2 mm). These are sim numbers; re-run the sweep once real captures have ground truth.
- **Cutter fix** (`sync_videos.clip_camera`, marked `egoumi change`): for a camera whose start offset is not a whole
  number of frames, clip frame 0 used to duplicate frame 1 (one frame dropped at every cut). It now seeks on the
  camera's own frame grid and caps at `-frames:v` with passthrough timestamps. h2r_collection still has the old code.
