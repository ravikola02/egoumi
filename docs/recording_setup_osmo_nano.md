# Recording setup: DJI Osmo Nano (head + gripper camera)

The same cameras as h2r_collection: a **DJI Osmo Nano** (OW001) on the head and one on the gripper's camera mount
(the right wrist feed). A second wrist camera is optional; in one-gripper captures its feed can be blank.
Intrinsics used by the pipeline: `configs/cameras_osmo_nano.yaml` (h2r_collection / HaWoR: focal 600 px at 1920 wide,
principal point at the image centre, no distortion model).

## Settings

| | Head camera | Gripper (wrist) camera |
|---|---|---|
| Resolution / frame rate | **3840×2160 @ 59.94 fps** (as h2r box_folding_1 / flipping_1) | **1920×1080 @ 29.97 fps** (as h2r) |
| Colour | normal (not D-Log M); 8-bit if offered | normal |
| Field of view | the **same FOV mode h2r recorded in** (see below) | same |
| Stabilization | **off** (RockSteady and HorizonBalancing) | **off** |
| Orientation | **locked to landscape** | **locked to landscape** |
| Shutter | about **1/500 s or faster**, good light; lock exposure and white balance if the app allows | same |
| Audio | **on** | **on** |
| Mount | head mount, fixed for the session | gripper camera mount, hinge tilted **~15° down**, fixed for the session |

Why:
- **4K head.** At 1080p the markers are small: in the sim with the Osmo Nano intrinsics, 41 % of pick-and-place
  frames see only one body tag and the TCP error is 2.7 mm median. 4K doubles every marker's size in pixels. The
  pipeline rescales the intrinsics (1200 px at 3840) and resamples to 30 fps rows.
- **Normal colour, 8-bit.** Better marker contrast than a log profile; h2r's 10-bit 4K head files caused decoding
  trouble in h2r_collection.
- **Field of view.** The camera offers Standard (DeWarp), Wide and Ultra Wide (143°). The 600 px intrinsics hold only
  for the mode h2r used, which is **not confirmed** (check the camera). DeWarp would remove the lens distortion the
  pipeline does not model, but changes the focal length, so it needs a new calibration.
- **Stabilization off.** Both modes crop and warp every frame, which breaks fixed intrinsics and the assumption that
  the gripper camera is rigid to the gripper.
- **Landscape locked.** The camera switches landscape / portrait by orientation on its own; h2r's wrist files had to
  be rotated 180° afterwards.
- **Gripper camera tilt.** At 0° both jaw tags fall just outside the frame; from 5° down they are detected at every
  jaw opening (sim, Osmo Nano intrinsics), about 114 px across at 15°. The mount pose in
  `configs/cameras_osmo_nano.yaml` is nominal: the Osmo Nano's lens is not where the GoPro dummy's is.
- **Shutter.** Motion blur is the main real-world risk for marker corners.

## Per session

1. **Clap once, sharply,** at the start of every take: MP4 timestamps keep whole seconds only, so the sync comes from
   the audio (in the sim it recovers the offsets exactly).
2. Keep **settings and mounts unchanged** for the whole session.
3. Storage: 4K60 HEVC at 120 Mbps is about 54 GB per hour.
4. **Optional insurance, about 1 minute per camera:** before the first take, record 30 s of a calibration board moved
   slowly through the view, in the exact recording mode. The pipeline can keep h2r's 600 px, but this lets the focal
   length be checked and the distortion measured later without re-recording.
5. Copy each take into `data/raw/<session>/{top_cam,right_w_cam,left_w_cam}/` and run
   `python3 collection/run_pipeline.py --session <session>` (see `collection/README.md`).

Sources for the camera capabilities: [DJI Osmo Nano support](https://www.dji.com/support/product/nano),
[FAQ](https://www.dji.com/nano/faq), [Newsshooter](https://www.newsshooter.com/2025/09/23/dji-osmo-nano/),
[Engadget review](https://www.engadget.com/cameras/dji-osmo-nano-review-high-quality-video-in-a-truly-tiny-action-cam-120040319.html).
