#!/usr/bin/env bash
set -u

run() {
  printf '\n$'
  printf ' %q' "$@"
  printf '\n'
  "$@" 2>&1
  printf '[exit=%s]\n' "$?"
}

printf 'SCOPE phase-1 environment probe\n'
printf 'probe_started_utc=%s\n' "$(date --iso-8601=seconds)"
run date --iso-8601=seconds
run hostname
run whoami
run pwd
run printenv HOME
run uname -a
run sed -n 1,120p /etc/os-release
if test -e /.dockerenv; then
  run ls -l /.dockerenv
else
  printf '/.dockerenv: absent\n'
fi
run lscpu
run nproc
run free -h
if command -v nvidia-smi >/dev/null 2>&1; then
  run nvidia-smi
  run nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
else
  printf 'nvidia-smi: not found\n'
fi
if command -v nvcc >/dev/null 2>&1; then
  run nvcc --version
else
  printf 'nvcc: not found\n'
fi
run df -h /home/jovyan/work
run python --version
run python3 --version
for tool in conda mamba micromamba git curl wget; do
  printf '\n## %s\n' "$tool"
  command -v "$tool" 2>&1 || true
  "$tool" --version 2>&1 | head -n 6 || true
done
run conda info
run conda env list
run git --version
printf '\n## Existing scope-related paths\n'
find /home/jovyan/work -maxdepth 5 -iname '*scope*' -print 2>&1
printf '\n## Existing Git worktrees under /home/jovyan/work\n'
find /home/jovyan/work -maxdepth 5 -type d -name .git -print 2>&1
for url in \
  https://github.com/TempleRAIL/scope.git \
  https://zenodo.org \
  https://download.pytorch.org/whl/torch_stable.html; do
  printf '\n## Network probe: %s\n' "$url"
  curl -L -I --max-time 20 --connect-timeout 10 "$url" 2>&1 | sed -n '1,40p'
  printf '[curl_pipeline_exit=%s]\n' "${PIPESTATUS[0]}"
done
printf '\nprobe_finished_utc=%s\n' "$(date --iso-8601=seconds)"
