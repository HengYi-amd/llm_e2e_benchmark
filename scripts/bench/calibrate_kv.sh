#!/usr/bin/env bash
# Pin KV cache capacity so every arm gets the same one.
#
# vLLM sizes the KV cache from peak memory during profiling, and arms that
# compile different numbers of autotune candidates end up with different block
# counts; more blocks means more concurrency and a throughput win unrelated to
# the kernels under test. Probe each arm once, take the minimum, keep headroom,
# and write state/kv_pin.env, which env.sh picks up.
set -Eeuo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"
. "$E2E_VENV/bin/activate"
cd "$E2E_ROOT"

# This stage measures how much memory is left for the KV cache, which does not
# depend on which GEMM kernel won. Autotuning it EXHAUSTIVE costs hours and
# changes nothing it reports, so it runs DEFAULT unless told otherwise.
GPU="${BENCH_GPU:-0}"
PORT="${BENCH_PORT:-$((8400 + GPU))}"
export HIP_VISIBLE_DEVICES="$GPU"
unset ROCR_VISIBLE_DEVICES
OUT="$E2E_ROOT/logs/kv_calib/$(echo "${KV_CALIB_MODEL:-default}" | tr -c 'A-Za-z0-9' '_')"
mkdir -p "$OUT" "$E2E_KV_PIN_DIR"

MODEL="${KV_CALIB_MODEL:-$(echo "$E2E_MODELS" | awk '{print $1}')}"
# Shell-safe suffix for the per-model variable name.
MODEL_KEY="$(echo "$MODEL" | tr -c 'A-Za-z0-9' '_')"
DTYPE="$(echo "$E2E_DTYPES" | awk '{print $1}')"
SWEEP="$E2E_CONCURRENCY_SWEEP"
ISL="$(echo "${E2E_PROFILE_showcase:-32:512}" | cut -d: -f1)"
COMPILE_SIZES="[$(
    for c in $SWEEP; do
        echo "$c"
        p=$(( ISL * c ))
        [ "$p" -gt "$E2E_MAX_NUM_BATCHED_TOKENS" ] && p="$E2E_MAX_NUM_BATCHED_TOKENS"
        echo "$p"
    done | sort -n -u | paste -sd, -)]"

# Calibrating with a pin in effect would only measure the pin.
unset E2E_KV_CACHE_BYTES

probe_arm() {
    local arm="$1"
    local bev="E2E_BACKENDS_$arm"; local be="${!bev}"
    export ARM_TAG="$arm"
    export TORCHINDUCTOR_CACHE_DIR="$E2E_ROOT/caches/inductor/kvcal_${arm}"
    export VLLM_CACHE_ROOT="$E2E_ROOT/caches/vllm/kvcal_${arm}"
    export TRITON_CACHE_DIR="$E2E_ROOT/caches/triton/kvcal_${arm}"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"
    export VLLM_FORCE_ATEN_LINEAR=1 VLLM_ROCM_USE_SKINNY_GEMM=0
    export TORCHINDUCTOR_ORIGAMI="$E2E_ORIGAMI" TORCHINDUCTOR_ORIGAMI_TOPK="$E2E_ORIGAMI_TOPK"
    export FLYDSL_ENABLE_AUTOTUNING="$E2E_FLYDSL_AUTOTUNING"

    # Reading compiled artifacts back from the AOT cache can livelock inside the
    # kernel loader while holding the GIL: the engine stops logging right after
    # weight loading and one core spins indefinitely. Compiling fresh costs time
    # but is the only reliable way past it, and it also rules out a stale cache
    # serving artifacts that were never autotuned in this run.
    export VLLM_DISABLE_COMPILE_CACHE="${E2E_DISABLE_COMPILE_CACHE:-1}"

    local cc="{\"mode\":3,\"backend\":\"inductor\",\"compile_sizes\":$COMPILE_SIZES,\"cudagraph_mode\":\"PIECEWISE\",\"inductor_compile_config\":{\"max_autotune\":true,\"max_autotune_gemm\":true,\"max_autotune_gemm_backends\":\"$be\",\"max_autotune_gemm_search_space\":\"${E2E_KV_SEARCH_SPACE:-DEFAULT}\",\"flydsl_enable_autotuning\":$([ "$E2E_FLYDSL_AUTOTUNING" = 1 ] && echo true || echo false)}}"

    vllm serve "$MODEL" --dtype "$DTYPE" --tensor-parallel-size "$E2E_TP" \
        --port "$PORT" --host 127.0.0.1 \
        --max-model-len "$E2E_MAX_MODEL_LEN" \
        --max-num-batched-tokens "$E2E_MAX_NUM_BATCHED_TOKENS" \
        --gpu-memory-utilization "$E2E_GPU_MEM_UTIL" \
        --no-enable-prefix-caching --seed 0 -cc "$cc" \
        > "$OUT/$arm.log" 2>&1 &
    local pid=$!
    local t0; t0=$(date +%s)
    while : ; do
        curl -fsS --connect-timeout 5 --max-time 10 \
             "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        kill -0 "$pid" 2>/dev/null || { echo "[kvcal] $arm server died" >&2; return 1; }
        [ $(( $(date +%s) - t0 )) -gt "$E2E_SERVER_TIMEOUT_S" ] && {
            echo "[kvcal] $arm server timed out" >&2; kill -KILL "$pid" 2>/dev/null; return 1; }
        sleep 5
    done
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true

    # Report bytes: --kv-cache-memory-bytes takes a byte budget, and the token
    # count it yields varies with the block size the engine picks.
    local gib
    gib=$(grep -ao 'Available KV cache memory: [0-9.]* GiB' "$OUT/$arm.log" \
          | tail -1 | grep -oE '[0-9.]+')
    [ -n "$gib" ] || return 1
    python - "$gib" <<'PY'
import sys
print(int(float(sys.argv[1]) * 1024 ** 3))
PY
}

echo "[kvcal] model=$MODEL dtype=$DTYPE arms=$E2E_ARMS"
declare -A TOK
for arm in $E2E_ARMS; do
    t="$(probe_arm "$arm" || true)"
    [ -n "$t" ] || { echo "[kvcal] could not read capacity for $arm" >&2; exit 1; }
    TOK[$arm]="$t"
    echo "[kvcal]   $arm: $t bytes"
done

python - "$OUT/summary.json" "$E2E_KV_PIN_DIR/$MODEL_KEY.env" "$MODEL_KEY" "${TOK[@]}" <<'PY'
import json, os, sys
out_json, out_env, model_key = sys.argv[1], sys.argv[2], sys.argv[3]
vals = [int(x) for x in sys.argv[4:]]
lo = min(vals)
spread = (max(vals) - lo) / lo if lo else 0.0
# Headroom: the same byte budget can yield a slightly different block count run
# to run, so aiming exactly at the minimum can leave an arm a block short.
pinned = int(lo * 0.98)
json.dump({"bytes_by_arm": vals, "min_bytes": lo, "spread": spread,
           "pinned_bytes": pinned}, open(out_json, "w"), indent=2)
# Writing this file is the point of the stage. Computing the value and not
# writing it lets the stage pass green while every arm still sizes its own
# cache, which puts a memory difference inside the throughput comparison.
os.makedirs(os.path.dirname(out_env), exist_ok=True)
with open(out_env, "w") as f:
    f.write("# generated by scripts/bench/calibrate_kv.sh\n")
    f.write(f"export E2E_KV_CACHE_BYTES__{model_key}={pinned}\n")
print(f"[kvcal] capacities differ by {spread:.2%} across arms; pinning "
      f"{pinned} bytes ({pinned / 1024 ** 3:.2f} GiB) -> {out_env}")
PY
echo "[kvcal] wrote $OUT/summary.json"

