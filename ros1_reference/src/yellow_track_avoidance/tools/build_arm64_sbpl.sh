#!/usr/bin/env bash
set -euo pipefail

# Compatibility entrypoint retained for existing migration notes.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/build_sbpl_runtime.sh" "$@"
