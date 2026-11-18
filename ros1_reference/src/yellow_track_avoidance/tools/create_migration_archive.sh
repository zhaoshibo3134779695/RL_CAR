#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$(cd "${script_dir}/.." && pwd)"
package_parent="$(dirname "${package_dir}")"
package_name="$(basename "${package_dir}")"
date_tag="$(date +%Y%m%d)"
default_archive="${package_parent}/${package_name}_migration_${date_tag}.zip"
archive_path="$(realpath -m "${1:-${default_archive}}")"

case "${archive_path}" in
  "${package_dir}"/*)
    echo "Refusing to create the archive inside the package being archived." >&2
    exit 2
    ;;
esac
if [[ "${archive_path}" != *.zip ]]; then
  echo "Archive path must end in .zip: ${archive_path}" >&2
  exit 2
fi
if [[ -e "${archive_path}" || -e "${archive_path}.sha256" ]]; then
  echo "Output already exists; choose a new path: ${archive_path}" >&2
  exit 2
fi

mkdir -p "$(dirname "${archive_path}")"
(
  cd "${package_parent}"
  zip -rq "${archive_path}" "${package_name}" \
    -x "${package_name}/.git/*" \
       "${package_name}/.pytest_cache/*" \
       "${package_name}/__pycache__/*" \
       "${package_name}/*/__pycache__/*" \
       "${package_name}/*/*/__pycache__/*" \
       "${package_name}/*.pyc" \
       "${package_name}/*/*.pyc" \
       "${package_name}/Log/*" \
       "${package_name}/debug_captures/*" \
       "${package_name}/disc_acceptance_runs/*" \
       "${package_name}/navigation_runs/*" \
       "${package_name}/offline_navigation_runs/*" \
       "${package_name}/calibration_runs/archive/*" \
       "${package_name}/calibration_runs/current_car_640x480/*" \
       "${package_name}/calibration_runs/current.zip" \
       "${package_name}/calibration_runs/current/live0000.*" \
       "${package_name}/calibration_runs/current/live_bev_snapshot.png" \
       "${package_name}/calibration_runs/current/offline_*/*" \
       "${package_name}/calibration_runs/max_1920x1080_30/*" \
       "${package_name}/calibration_runs/profile_*.tar.gz" \
       "${package_name}/*.zip" \
       "${package_name}/*.tar.gz" \
       "${package_name}/*.tgz"
)

unzip -tqq "${archive_path}"
sha256sum "${archive_path}" > "${archive_path}.sha256"
file_count="$(unzip -Z1 "${archive_path}" | wc -l)"
archive_size="$(du -h "${archive_path}" | awk '{print $1}')"

echo "Created ${archive_path}"
echo "SHA256: ${archive_path}.sha256"
echo "Entries: ${file_count}; size: ${archive_size}"
