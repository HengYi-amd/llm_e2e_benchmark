#!/usr/bin/env bash
# Fan the sweep out across the node's GPUs: one shard = one
# (model, dtype, profile, repeat), run as an independent single-GPU job.
#
# Both arms of a shard must run back to back on the SAME GPU; splitting a pair
# across cards lets clock and thermal differences look like an arm difference.
# Repeats may land on different cards, since each carries its own arm pair.
# Concurrent shards inflate absolute latency, so only within-shard arm ratios
# are comparable; each point records e2e_shard_concurrency for auditing.
set -Eeuo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"
cd "$E2E_ROOT"

RUN_DIR="${RUN_DIR:-$E2E_ARTIFACTS/runs/$(cat "$E2E_ROOT/state/current_run_id")}"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/raw/e2e"
export RUN_DIR

read -r -a GPUS <<< "${E2E_GPUS:-0 1 2 3 4 5 6 7}"
NG=${#GPUS[@]}

SHARDS=()
for model in $E2E_MODELS; do
  for dtype in $E2E_DTYPES; do
    for prof in $E2E_PROFILES; do
      for rep in $(seq 1 "$E2E_REPEATS"); do
        SHARDS+=("$model|$dtype|$prof|$rep")
      done
    done
  done
done
echo "[shard] ${#SHARDS[@]} shards over $NG GPUs"

pids=(); i=0
for s in "${SHARDS[@]}"; do
    IFS='|' read -r model dtype prof rep <<< "$s"
    gpu="${GPUS[$(( i % NG ))]}"
    tag="$(echo "$model" | tr '/' '_')_${dtype}_${prof}_r${rep}"
    echo "[shard] GPU $gpu <- $tag"
    env BENCH_GPU="$gpu" BENCH_PORT=$((8300 + gpu)) \
        E2E_MODELS="$model" E2E_DTYPES="$dtype" E2E_PROFILES="$prof" \
        E2E_REP_LIST="$rep" E2E_SHARDS="$NG" \
        bash "$E2E_ROOT/scripts/bench/run_e2e.sh" \
        > "$RUN_DIR/logs/shard_gpu${gpu}_${tag}.log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
    # Wait for a free slot rather than oversubscribing a card.
    if [ "${#pids[@]}" -ge "$NG" ]; then
        wait -n 2>/dev/null || true
        alive=(); for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
        pids=("${alive[@]}")
    fi
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[shard] all shards finished rc=$rc, $(ls -1 "$RUN_DIR/raw/e2e"/*.json 2>/dev/null | wc -l) points"
exit $rc
