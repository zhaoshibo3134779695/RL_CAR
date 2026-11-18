#!/usr/bin/env bash
set -o pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$(cd "${script_dir}/.." && pwd)"
package_source_dir="$(cd "${package_dir}/.." && pwd)"

profile="donor"
if [[ ${1:-} == "--profile" && -n ${2:-} ]]; then
  profile="$2"
  shift 2
fi
if [[ $# -ne 0 || ( "${profile}" != "donor" && "${profile}" != "current_car" ) ]]; then
  echo "usage: $0 [--profile donor|current_car]" >&2
  exit 2
fi

if [[ ${ROS_DISTRO:-} != "noetic" ]] && [[ -r /opt/ros/noetic/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
elif ! command -v rospack >/dev/null 2>&1; then
  echo "[FAIL] /opt/ros/noetic/setup.bash not found"
  exit 1
fi
case ":${ROS_PACKAGE_PATH:-}:" in
  *":${package_source_dir}:"*) ;;
  *) export ROS_PACKAGE_PATH="${package_source_dir}:${ROS_PACKAGE_PATH:-/opt/ros/noetic/share}" ;;
esac
set -u

failures=0
warnings=0

ok() { echo "[ OK ] $*"; }
warn() { echo "[WARN] $*"; warnings=$((warnings + 1)); }
fail() { echo "[FAIL] $*"; failures=$((failures + 1)); }

echo "yellow_track_avoidance migration preflight"
echo "package: ${package_dir}"
echo "profile: ${profile}"
echo "host: $(uname -m), $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")"

for command_name in python3 rospack roslaunch rosservice rostopic xmllint file ldd; do
  if command -v "${command_name}" >/dev/null 2>&1; then
    ok "command ${command_name}"
  else
    fail "missing command ${command_name}"
  fi
done

required_packages=(
  yellow_track_avoidance amcl car_bringup cv_bridge lslidar_driver map_server
  move_base mowen robot_localization robot_state_publisher rqt_image_view rviz
  sbpl_lattice_planner teb_local_planner tf
)
if [[ "${profile}" == "current_car" ]]; then
  required_packages+=(nav_demo usb_cam)
else
  required_packages+=(orbbec_camera)
fi
for package_name in "${required_packages[@]}"; do
  if rospack find "${package_name}" >/dev/null 2>&1; then
    ok "ROS package ${package_name}"
  else
    fail "missing ROS package ${package_name}"
  fi
done
if [[ "${profile}" == "donor" ]]; then
  if rospack find wit_ros_imu >/dev/null 2>&1; then
    ok "optional ROS package wit_ros_imu"
  else
    warn "optional wit_ros_imu not found (start_imu must remain false)"
  fi
else
  ok "current_car entrypoint keeps IMU startup disabled"
fi

if python3 -c 'import cv2, numpy, yaml' >/dev/null 2>&1; then
  ok "Python imports cv2/numpy/yaml"
else
  fail "Python cannot import cv2, numpy, or yaml"
fi
if PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${package_dir}/src:${PYTHONPATH:-}" \
  python3 -c 'import yellow_track_avoidance' >/dev/null 2>&1; then
  ok "Python imports yellow_track_avoidance from this package"
else
  fail "Python cannot import yellow_track_avoidance from ${package_dir}/src"
fi

essential_files=(
  package.xml CMakeLists.txt config/vehicle_profiles.yaml
  maps/track_7_15_5x4.yaml vendor/sbpl_signed_char/lib/libsbpl.so
)
if [[ "${profile}" == "current_car" ]]; then
  essential_files+=(
    config/current_car.yaml launch/current_car_camera.launch
    launch/current_car_navigation.launch maps/competition_blank_5x4.yaml
    maps/competition_blank_5x4.pgm scripts/competition_corridor_map.py
  )
else
  essential_files+=(
    config/calibrated.yaml launch/real_navigation.launch
  )
fi
for relative_path in "${essential_files[@]}"; do
  if [[ -r "${package_dir}/${relative_path}" ]]; then
    ok "file ${relative_path}"
  else
    fail "missing file ${relative_path}"
  fi
done

xml_failed=0
while IFS= read -r launch_file; do
  if ! xmllint --noout "${launch_file}" >/dev/null 2>&1; then
    fail "invalid launch XML ${launch_file#${package_dir}/}"
    xml_failed=1
  fi
done < <(find "${package_dir}/launch" -maxdepth 1 -name '*.launch' -type f | sort)
if [[ ${xml_failed} -eq 0 ]]; then
  ok "all launch XML files are well formed"
fi

sbpl_library="${package_dir}/vendor/sbpl_signed_char/lib/libsbpl.so"
architecture="$(uname -m)"
library_description="$(file "${sbpl_library}" 2>/dev/null || true)"
case "${architecture}" in
  x86_64) expected_elf="x86-64" ;;
  aarch64) expected_elf="ARM aarch64" ;;
  *) expected_elf="${architecture}" ;;
esac
if [[ "${library_description}" == *"${expected_elf}"* ]]; then
  ok "vendored SBPL architecture matches ${architecture} host"
else
  fail "vendored SBPL does not match ${architecture}; rebuild with tools/build_sbpl_runtime.sh"
fi
if ldd_output="$(ldd "${sbpl_library}" 2>&1)"; then
  if [[ "${ldd_output}" == *"not found"* ]]; then
    fail "vendored SBPL has unresolved shared-library dependencies"
  else
    ok "vendored SBPL dynamic dependencies resolve"
  fi
else
  fail "could not inspect vendored SBPL: ${ldd_output}"
fi

for device_path in /dev/lslidar /dev/carserial; do
  if [[ -e "${device_path}" ]]; then
    if [[ -r "${device_path}" && -w "${device_path}" ]]; then
      ok "device ${device_path} is readable and writable"
    else
      warn "device ${device_path} exists but current user lacks read/write access"
    fi
  else
    warn "device ${device_path} is absent (expected when hardware is disconnected)"
  fi
done

if [[ "${profile}" == "current_car" ]]; then
  camera_device="/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Orbbec_DaBai_DCW2_RGB_Camera_CH8Z16501C1-video-index0"
  if [[ -e "${camera_device}" ]]; then
    ok "camera ${camera_device}"
  else
    warn "camera ${camera_device} is absent (override video_device if its ID changed)"
  fi
fi

echo
echo "result: ${failures} failure(s), ${warnings} warning(s)"
if [[ ${failures} -ne 0 ]]; then
  exit 1
fi
