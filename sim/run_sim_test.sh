#!/usr/bin/env bash
# Sim test of the ArUco gripper pipeline: render -> pose -> preview.
#   sim/run_sim_test.sh <name> [render_episode.py args...] [-- gripper_pose.py args...]
# e.g. sim/run_sim_test.sh pick_place
#      sim/run_sim_test.sh roll_blur --preset roll --blur 5 -- --cam-source table
# MARKERS=configs/<variant>.yaml selects the marker layout for all three steps.
# Two grippers: sim/run_sim_test.sh both --hands both  (right + left-hand gripper; pose + preview per gripper).
# Software rendering (no GPU): PYOPENGL_PLATFORM=osmesa, and add --chunk 40 to the render args (OSMesa leaks memory per
# frame); PREVIEW_ARGS=--metrics-only skips the preview video.
# Output: sim/episodes/<name>/{extracted_images,gt.npz,gripper_pose/,preview/gripper_preview.mp4,preview/metrics.json}
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/anaconda3/envs/hawor/bin/python}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
name=$1; shift
render_args=(); pose_args=(); dst=render_args
for a in "$@"; do
  if [[ $a == "--" ]]; then dst=pose_args; continue; fi
  eval "$dst+=(\"\$a\")"
done
seq=sim/episodes/$name
mk=(--markers "${MARKERS:-configs/gripper_markers.yaml}")
"$PY" sim/render_episode.py --out "$seq" "${mk[@]}" "${render_args[@]}"
# --hands right (default) | left | both: one pose + preview pass per gripper in the episode
hands=right
for ((i = 0; i < ${#render_args[@]}; i++)); do [[ ${render_args[$i]} == --hands ]] && hands=${render_args[$((i + 1))]}; done
case $hands in both) grippers=(right left) ;; left) grippers=(left) ;; *) grippers=(right) ;; esac
for g in "${grippers[@]}"; do
  "$PY" scripts/gripper_pose.py --seq "$seq" --gripper "$g" "${mk[@]}" "${pose_args[@]}"
  "$PY" scripts/preview_gripper.py --seq "$seq" --gripper "$g" "${mk[@]}" ${PREVIEW_ARGS:-}
  [[ $g == right ]] && sfx="" || sfx="_$g"
  echo "preview ($g gripper): $seq/preview$sfx/gripper_preview.mp4"
done
