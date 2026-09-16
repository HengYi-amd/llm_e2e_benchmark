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

GPU="${BENCH_GPU:-0}"
PORT="${BENCH_PORT:-$((8400 + GPU))}"
export HIP_VISIBLE_DEVICES="$GPU"
unset ROCR_VISIBLE_DEVICES
OUT="$E2E_ROOT/logs/kv_calib"
mkdir -p "$OUT"

MODEL="${KV_CALIB_MODEL:-$(echo "$E2E_MODELS" | awk '{print $1}')}"
DTYPE="$(echo "$E2E_DTYPES" | awk '{print $1}')"
SWEEP="$E2E_CONCURRENCY_SWEEP"
COMPILE_SIZES="[$(echo "$SWEEP $E2E_MAX_NUM_BATCHED_TOKENS" | tr ' ' '\n' | sort -n -u | paste -sd, -)]"

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

    local cc="{\"mode\":3,\"backend\":\"inductor\",\"compile_sizes\":$COMPILE_SIZES,\"cudagraph_mode\":\"PIECEWISE\",\"inductor_compile_config\":{\"max_autotune\":true,\"max_autotune_gemm\":true,\"max_autotune_gemm_backends\":\"$be\",\"max_autotune_gemm_search_space\":\"$E2E_AUTOTUNE_SEARCH_SPACE\",\"flydsl_enable_autotuning\":$([ "$E2E_FLYDSL_AUTOTUNING" = 1 ] && echo true || echo false)}}"

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
        curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        kill -0 "$pid" 2>/dev/null || { echo "[kvcal] $arm server died" >&2; return 1; }
        [ $(( $(date +%s) - t0 )) -gt "$E2E_SERVER_TIMEOUT_S" ] && {
            echo "[kvcal] $arm server timed out" >&2; kill -KILL "$pid" 2>/dev/null; return 1; }
        sleep 5
    done
    kill -TERM "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true

    grep -ao 'KV cache size: [0-9,]* tokens' "$OUT/$arm.log" | tail -1 \
        | grep -o '[0-9,]*' | tr -d ',' | tail -1
}

echo "[kvcal] model=$MODEL dtype=$DTYPE arms=$E2E_ARMS"
declare -A TOK
for arm in $E2E_ARMS; do
    t="$(probe_arm "$arm" || true)"
    [ -n "$t" ] || { echo "[kvcal] could not read capacity for $arm" >&2; exit 1; }
    TOK[$arm]="$t"
    echo "[kvcal]   $arm: $t tokens"
done

python - "$OUT/summary.json" "$E2E_ROOT/state/kv_pin.env" "${TOK[@]}" <<'PY'
import json, sys
out_json, out_env = sys.argv[1], sys.argv[2]
toks = [int(x) for x in sys.argv[3:]]
lo = min(toks)
spread = (max(toks) - lo) / lo if lo else 0.0
# Headroom: the pin is a byte budget and the block count it yields varies
# slightly, so aiming exactly at the minimum can leave an arm a block short.
pinned_tokens = int(lo * 0.98)
json.dump({"tokens_by_arm": toks, "min_tokens": lo, "spread": spread,
           "pinned_tokens": pinned_tokens}, open(out_json, "w"), indent=2)
print(f"[kvcal] capacities differ by {spread:.2%} across arms; pinning "
      f"{pinned_tokens} tokens")
PY
echo "[kvcal] wrote $OUT/summary.json"
echo "[kvcal] NOTE: vLLM pins bytes, not tokens. Set E2E_KV_CACHE_BYTES in"
echo "        config/default.env (or state/kv_pin.env) from the byte figure"
echo "        reported in the server log if you need a hard pin."
