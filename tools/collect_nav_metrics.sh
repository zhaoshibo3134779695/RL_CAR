#!/usr/bin/env bash

set -u

WORKSPACE="/home/mo-bit/newznzc_ws"
ROS_SETUP="/opt/ros/noetic/setup.bash"
WS_SETUP="$WORKSPACE/devel/setup.bash"
MAP_FILE_DEFAULT="$WORKSPACE/src/navigation/nav_demo/map/mymap.yaml"

if [[ -f "$ROS_SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$ROS_SETUP"
else
  echo "ROS setup not found: $ROS_SETUP"
  exit 1
fi

if [[ -f "$WS_SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$WS_SETUP"
else
  echo "Workspace setup not found: $WS_SETUP"
  exit 1
fi

MAP_FILE="${1:-$MAP_FILE_DEFAULT}"

section() {
  echo
  echo "========== $1 =========="
}

run_cmd() {
  echo "\$ $*"
  "$@"
}

section "ROS Master"
if ! rosnode list >/dev/null 2>&1; then
  echo "ROS master is not reachable."
  echo "Please launch your robot/navigation stack first, for example:"
  echo "  roslaunch nav_demo slam_manual_mapping.launch"
  echo "or"
  echo "  roslaunch nav_demo rviz_manual_nav.launch"
  exit 2
fi
echo "ROS master is reachable."

section "Map YAML"
if [[ -f "$MAP_FILE" ]]; then
  echo "map_file: $MAP_FILE"
  grep -E '^(image|resolution|origin):' "$MAP_FILE" || true
else
  echo "Map file not found: $MAP_FILE"
fi

section "Configured Lidar Driver Values"
grep -nE 'max_range|min_range|static_transform_publisher|args=' \
  "$WORKSPACE/src/drivers/leishen/lslidar_driver/launch/lslidar_serial.launch" || true

section "Live /scan Sample"
timeout 8s rostopic echo -n 1 /scan | grep -E '^(angle_min|angle_max|angle_increment|range_min|range_max):' || true

section "Live /odom Rate"
timeout 8s rostopic hz /odom 2>&1 | tail -n 5 || true

section "Live TF base_link -> laser_link"
timeout 6s rosrun tf tf_echo base_link laser_link 2>&1 | grep -E 'Translation:|Rotation in RPY:' || true

section "Current DWA Key Params"
grep -nE 'max_vel_trans|min_vel_trans|max_vel_x|min_vel_x|max_vel_y|min_vel_y|max_vel_theta|min_vel_theta|acc_lim_x|acc_lim_y|acc_lim_th|xy_goal_tolerance|yaw_goal_tolerance|sim_time|vx_samples|vy_samples|vth_samples|path_distance_bias|goal_distance_bias|occdist_scale|holonomic_robot' \
  "$WORKSPACE/src/navigation/nav_demo/param/dwa_local_planner_params.yaml" || true

section "Current Costmap Key Params"
grep -nE 'footprint:|obstacle_range:|raytrace_range:|inflation_radius:|cost_scaling_factor:|width:|height:|resolution:' \
  "$WORKSPACE/src/navigation/nav_demo/param/costmap_common_params.yaml" \
  "$WORKSPACE/src/navigation/nav_demo/param/global_costmap_params.yaml" \
  "$WORKSPACE/src/navigation/nav_demo/param/local_costmap_params.yaml" || true

section "Notes For Manual Fill"
echo "Please also tell me these manual measurements if you know them:"
echo "  stable max forward speed (m/s)"
echo "  stable max reverse speed (m/s)"
echo "  stable max angular speed (rad/s)"
echo "  braking distance from 0.20 m/s (m)"
