#!/usr/bin/env bash
# Run pegeq shards from a queue with N parallel workers, stopping new shards
# once a time budget is used, so a run that takes many hours can be done in
# rounds. Re-run the same command to continue; finished shards are skipped.
#
# usage: tools/pegeq_shards.sh OUTDIR NSHARDS WORKERS BUDGET_SECONDS -- <pegeq args>
# Each shard writes OUTDIR/shard_<i>.txt and marks completion with OUTDIR/shard_<i>.done.
set -u
outdir=$1; nshards=$2; workers=$3; budget=$4; shift 4
[ "$1" = "--" ] && shift
mkdir -p "$outdir"
start=$(date +%s)
here=$(cd "$(dirname "$0")/.." && pwd)

next_shard() {
  # atomically claim the next shard that is neither done nor claimed this round
  for ((i = 0; i < nshards; i++)); do
    [ -e "$outdir/shard_$i.done" ] && continue
    if mkdir "$outdir/shard_$i.claim" 2>/dev/null; then
      echo "$i"
      return
    fi
  done
  echo ""
}

worker() {
  while :; do
    now=$(date +%s)
    [ $((now - start)) -ge "$budget" ] && return
    i=$(next_shard)
    [ -z "$i" ] && return
    if "$here/bin/pegeq" "$@" -shard "$i" -nshards "$nshards" \
        -out "$outdir/shard_$i.txt" > "$outdir/shard_$i.log" 2>&1 \
        && grep -q '^done' "$outdir/shard_$i.log"; then
      touch "$outdir/shard_$i.done"
    fi
    rmdir "$outdir/shard_$i.claim" 2>/dev/null
  done
}

rm -rf "$outdir"/shard_*.claim
for ((w = 0; w < workers; w++)); do
  worker "$@" &
done
wait
done_n=$(ls "$outdir"/shard_*.done 2>/dev/null | wc -l)
echo "shards done: $done_n / $nshards   elapsed $(( $(date +%s) - start ))s"
