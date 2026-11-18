#!/usr/bin/env bash
set -euo pipefail

script_name="$(basename "$0")"
usage() {
  echo "usage: ${script_name} [--with-images] [OUTPUT_DIRECTORY]" >&2
}

with_images=false
if [[ ${1:-} == "--with-images" ]]; then
  with_images=true
  shift
fi
if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi

stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${1:-/tmp/yellow_track_passage_${stamp}}"
mkdir -p "${output_dir}"

topics=(
  /tf
  /tf_static
  /scan
  /odom
  /amcl_pose
  /move_base/status
  /move_base/current_goal
  /move_base/SBPLLatticePlanner/plan
  /move_base/TebLocalPlannerROS/local_plan
  /move_base/global_costmap/costmap_updates
  /move_base/local_costmap/costmap_updates
  /yellow_track_avoidance/debug/boundaries
  /yellow_track_avoidance/debug/disks
  /yellow_track_disc_costmap_bridge/status
  /yellow_track_avoidance/passage/status
  /yellow_track_avoidance/passage/markers
  /yellow_track_navigation_gate/status
  /yellow_track_avoidance/move_base_cmd_vel
  /yellow_track_avoidance/passage_cmd_vel
  /cmd_vel
)
if ${with_images}; then
  topics+=(
    /usb_cam/image_raw
    /usb_cam/camera_info
    /yellow_track_avoidance/debug/raw_overlay
    /yellow_track_avoidance/debug/bev
  )
fi

echo "recording passage validation to ${output_dir}/passage_validation.bag"
echo "stop with Ctrl-C after the trial"
exec rosbag record --lz4 \
  -O "${output_dir}/passage_validation.bag" \
  "${topics[@]}"
