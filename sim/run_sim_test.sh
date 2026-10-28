#!/usr/bin/env bash
# Sim test of the ArUco gripper pipeline: render -> pose -> preview.
#   sim/run_sim_test.sh <name> [render_episode.py args...] [-- gripper_pose.py args...]
# e.g. sim/run_sim_test.sh pick_place
#      sim/run_sim_test.sh roll_blur --preset roll --blur 5 -- --cam-source table
# MARKERS=configs/<variant>.yaml selects the marker layout for all three steps.
# Output: sim/episodes/<name>/{extracted_images,gt.npz,gripper_pose/,preview/gripper_preview.mp4,preview/metrics.json}
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/anaconda3/envs/hawor/bin/python}
export PYOPENGL_PLATFORM=egl
name=$1; shift
render_args=(); pose_args=(); dst=render_args
for a in "$@"; do
  if [[ $a == "--" ]]; then dst=pose_args; continue; fi
  eval "$dst+=(\"\$a\")"
done
seq=sim/episodes/$name
mk=(--markers "${MARKERS:-configs/gripper_markers.yaml}")
"$PY" sim/render_episode.py --out "$seq" "${mk[@]}" "${render_args[@]}"
"$PY" scripts/gripper_pose.py --seq "$seq" "${mk[@]}" "${pose_args[@]}"
"$PY" scripts/preview_gripper.py --seq "$seq" "${mk[@]}"
echo "preview: $seq/preview/gripper_preview.mp4"
