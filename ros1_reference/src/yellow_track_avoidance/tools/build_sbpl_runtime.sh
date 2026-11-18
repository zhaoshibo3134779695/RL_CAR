#!/usr/bin/env bash
set -euo pipefail

# Build the exact SBPL release used by this package for the current host.
# -fsigned-char preserves the package's XYTheta lattice action semantics on
# every architecture, including AArch64 where plain char is otherwise unsigned.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$(cd "${script_dir}/.." && pwd)"
output_dir="${package_dir}/vendor/sbpl_signed_char/lib"
source_dir="$(mktemp -d /tmp/yta_sbpl_source.XXXXXX)"
build_dir="$(mktemp -d /tmp/yta_sbpl_build.XXXXXX)"
readonly expected_commit="4d654845aa3b92aa1b4282a342e9011bac95aeb9"

cleanup() {
  rm -rf "${source_dir}" "${build_dir}"
}
trap cleanup EXIT

git -c advice.detachedHead=false clone --quiet --branch 1.3.1 --depth 1 \
  https://github.com/sbpl/sbpl.git "${source_dir}"

actual_commit="$(git -C "${source_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
  echo "Unexpected SBPL 1.3.1 commit: ${actual_commit}" >&2
  exit 1
fi

cmake -S "${source_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-fsigned-char
cmake --build "${build_dir}" --target sbpl --parallel 2

mkdir -p "${output_dir}"
install -m 0755 "${build_dir}/libsbpl.so" "${output_dir}/libsbpl.so"

echo "Installed $(uname -m) signed-char SBPL to ${output_dir}/libsbpl.so"
