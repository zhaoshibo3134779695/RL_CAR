#!/usr/bin/env bash
set -u

repro_root=/home/jovyan/work/research/scope_repro
compat_repo=$repro_root/repos/scope_modern_gpu
test_data=$repro_root/datasets/OGM-datasets/OGM-Turtlebot2/test
modern_bin=$repro_root/environments/scope_modern_gpu/bin
gnu_time=$repro_root/environments/scope_official/bin/time
run_root=$repro_root/outputs/modern_gpu_parallel_20260726T010500Z
log_root=$repro_root/logs/modern_gpu_parallel
shard_count=8
sample_count=17000

if test -e "$run_root"; then
  echo "Refusing to overwrite existing run root: $run_root" >&2
  exit 2
fi

mkdir -p "$run_root" "$log_root"
printf 'started_utc=%s\n' "$(date --iso-8601=seconds)"
printf 'run_root=%s\n' "$run_root"
printf 'shard_count=%s\n' "$shard_count"
printf 'sample_count=%s\n' "$sample_count"
printf 'base_commit=%s\n' "$(git -C "$compat_repo" rev-parse HEAD)"
git -C "$compat_repo" status --short --branch

nvidia-smi \
  --query-gpu=timestamp,memory.used,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop-ms=500 > "$log_root/gpu_memory.csv" 2>&1 &
monitor_pid=$!

pids=()
shard=0
while test "$shard" -lt "$shard_count"; do
  start=$((shard * sample_count / shard_count))
  end=$(((shard + 1) * sample_count / shard_count))
  shard_name=$(printf 'shard_%02d_%05d_%05d' "$shard" "$start" "$end")
  shard_root=$run_root/$shard_name
  mkdir -p "$shard_root"
  cp -a "$compat_repo" "$shard_root/runtime"
  mv "$shard_root/runtime/output" "$shard_root/runtime/upstream_output_baseline"
  mkdir -p "$shard_root/runtime/output"
  (
    shard_log=$log_root/${shard_name}_console.log
    {
      printf 'shard=%s\n' "$shard"
      printf 'start_index=%s\n' "$start"
      printf 'end_index=%s\n' "$end"
      printf 'started_utc=%s\n' "$(date --iso-8601=seconds)"
      printf 'command=SCOPE_START_INDEX=%s SCOPE_END_INDEX=%s sh run_eval_demo.sh %s\n' \
        "$start" "$end" "$test_data"
      cd "$shard_root/runtime"
      env \
        SCOPE_START_INDEX="$start" \
        SCOPE_END_INDEX="$end" \
        PYTHONUNBUFFERED=1 \
        MPLBACKEND=Agg \
        PATH="$modern_bin:$PATH" \
        "$gnu_time" -v sh run_eval_demo.sh "$test_data"
      code=$?
      printf 'exit_code=%s\n' "$code"
      printf 'finished_utc=%s\n' "$(date --iso-8601=seconds)"
      exit "$code"
    } > "$shard_log" 2>&1
  ) &
  pids+=("$!")
  printf 'launched %s pid=%s\n' "$shard_name" "$!"
  shard=$((shard + 1))
done

overall=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then
    printf 'completed pid=%s exit=0\n' "$pid"
  else
    code=$?
    printf 'completed pid=%s exit=%s\n' "$pid" "$code"
    overall=1
  fi
done

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

combined=$run_root/combined_output
mkdir -p "$combined"
for shard_dir in "$run_root"/shard_*; do
  find "$shard_dir/runtime/output" -maxdepth 1 -type f \
    \( -name 'mask*.jpg' -o -name 'pred*.jpg' \) \
    -exec cp -t "$combined" {} +
  cp "$shard_dir/runtime/output/01_decode_dev.log" \
    "$combined/01_decode_dev_$(basename "$shard_dir").log"
done

printf 'combined_mask_files='
find "$combined" -maxdepth 1 -type f -name 'mask*.jpg' | wc -l
printf 'combined_pred_files='
find "$combined" -maxdepth 1 -type f -name 'pred*.jpg' | wc -l
printf 'finished_utc=%s\n' "$(date --iso-8601=seconds)"
printf 'overall_exit=%s\n' "$overall"
exit "$overall"
