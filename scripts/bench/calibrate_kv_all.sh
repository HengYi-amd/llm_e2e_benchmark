#!/usr/bin/env bash
# Calibrate the KV pin for every model, one GPU each, in parallel.
#
# The pin is per model because the byte budget left over after weights differs
# by tens of gigabytes between model sizes; one global value either wastes
# capacity on the small model or fails to fit the large one.
set -Eeuo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"
cd "$E2E_ROOT"

read -r -a GPUS <<< "${E2E_GPUS:-0 1 2 3 4 5 6 7}"
rm -rf "$E2E_ROOT/state/kv_pin.d"; mkdir -p "$E2E_ROOT/state/kv_pin.d"

pids=(); i=0
for model in $E2E_MODELS; do
    gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
    echo "[kvcal-all] GPU $gpu <- $model"
    env KV_CALIB_MODEL="$model" BENCH_GPU="$gpu" BENCH_PORT=$((8400 + gpu)) \
        bash "$E2E_ROOT/scripts/bench/calibrate_kv.sh" \
        > "$E2E_ROOT/logs/kvcal_$(echo "$model" | tr -c 'A-Za-z0-9' '_').log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[kvcal-all] done rc=$rc"
cat "$E2E_ROOT"/state/kv_pin.d/*.env 2>/dev/null || true
exit "$rc"
