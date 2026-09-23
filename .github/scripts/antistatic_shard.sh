#!/bin/bash
# Runs one shard of the antistatic endgame experiment (magpie_test
# antistaticeg), splitting its seeds across parallel processes.
#
# usage: antistatic_shard.sh <first_seed> <games> <out_dir> [processes]
# Other settings pass through as ANTISTATIC_SECONDS, ANTISTATIC_FIRSTWIN and
# ANTISTATIC_LEX. Each process writes its summary to <out_dir>/part_<n>.txt
# and its game logs to <out_dir>/part_<n>/.
set -euo pipefail

first_seed=$1
games=$2
out_dir=$3
processes=${4:-$(nproc)}
mkdir -p "$out_dir"

per_process=$(((games + processes - 1) / processes))
pids=()
for ((p = 0; p < processes; p++)); do
  seed=$((first_seed + p * per_process))
  count=$((games - p * per_process))
  ((count > per_process)) && count=$per_process
  ((count <= 0)) && break
  ANTISTATIC_SEED=$seed ANTISTATIC_GAMES=$count \
    ANTISTATIC_OUT="$out_dir/part_$p" \
    bin/magpie_test antistaticeg >"$out_dir/part_$p.txt" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit $status
