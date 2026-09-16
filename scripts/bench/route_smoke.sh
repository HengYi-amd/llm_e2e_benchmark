#!/usr/bin/env bash
# Gate: prove the GEMM backend under test is actually routed to before the
# sweep starts. Without it a broken route still produces a full A/A run
# labelled as A/B.
set -Eeuo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"
. "$E2E_VENV/bin/activate"
# Run from the repo root: a sibling checkout can shadow an installed package of
# the same name, which surfaces as a model-loading error.
cd "$E2E_ROOT"

SMOKE_CACHE_TAG="${SMOKE_CACHE_TAG:-smoke}"
RUN_DIR="${RUN_DIR:-$E2E_ARTIFACTS/runs/$(cat "$E2E_ROOT/state/current_run_id")}"
mkdir -p "$RUN_DIR/raw"
OUT="$RUN_DIR/raw/route_smoke${SMOKE_CACHE_TAG:+_$SMOKE_CACHE_TAG}.log"
JSON="$RUN_DIR/raw/route_smoke${SMOKE_CACHE_TAG:+_$SMOKE_CACHE_TAG}.json"

# Set only one device mask: ROCR and HIP masks compose, leaving no visible GPU.
export HIP_VISIBLE_DEVICES=${BENCH_GPU:-0}
unset ROCR_VISIBLE_DEVICES
export ARM_TAG=smoke

# Caches must be isolated per attempt: a shared FX graph cache serves stale
# artifacts, autotune never runs, and the same configuration passes or fails at
# random.
export TORCHINDUCTOR_CACHE_DIR="$E2E_ROOT/caches/inductor/$SMOKE_CACHE_TAG"
export VLLM_CACHE_ROOT="$E2E_ROOT/caches/vllm/$SMOKE_CACHE_TAG"
export VLLM_DISABLE_COMPILE_CACHE=1

# The probe must compile under the same backends and autotune breadth as the
# treatment arm, or the gate proves routing for a configuration never swept.
# SMOKE_* overrides remain available for bisecting.
: "${SMOKE_GEMM_BACKENDS:=${E2E_BACKENDS_treatment:-ATEN,TRITON,FLYDSL}}"
: "${SMOKE_AUTOTUNE_SEARCH_SPACE:=${E2E_AUTOTUNE_SEARCH_SPACE:-DEFAULT}}"
: "${SMOKE_MAX_NUM_BATCHED_TOKENS:=${E2E_MAX_NUM_BATCHED_TOKENS:-8192}}"
export SMOKE_GEMM_BACKENDS SMOKE_AUTOTUNE_SEARCH_SPACE \
       SMOKE_MAX_NUM_BATCHED_TOKENS
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="$SMOKE_GEMM_BACKENDS"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

MODEL="${SMOKE_MODEL:-Qwen/Qwen3-0.6B}"
echo "[smoke] model=$MODEL" | tee "$OUT"

# The probe must stay a separate file: in a pipeline a heredoc attaches to the
# last command, so python would read empty stdin and the log would look like a
# routing failure.
#
# Startup cudagraph capture can go invalid; retry only on that one signature so
# every other failure is reported immediately.
ATTEMPT=0
while : ; do
    ATTEMPT=$((ATTEMPT+1))
    : > "$OUT.attempt"
    PROBE_RC=0
    TORCH_LOGS="+torch._inductor.select_algorithm" \
    python "$E2E_ROOT/scripts/bench/route_smoke_probe.py" "$MODEL" 2>&1 \
        | tee "$OUT.attempt" || PROBE_RC=${PIPESTATUS[0]}
    cat "$OUT.attempt" >> "$OUT"
    grep -q '^GENERATED:' "$OUT.attempt" && break
    if [ "$ATTEMPT" -ge "${SMOKE_CAPTURE_RETRIES:-4}" ] || \
       ! grep -aq 'hipStreamCaptureStatusInvalidated' "$OUT.attempt"; then
        echo "[smoke] FAIL: probe did not finish (rc=$PROBE_RC, attempt" \
             "$ATTEMPT), no GENERATED line in the log" | tee -a "$OUT"
        rm -f "$OUT.attempt"; exit 1
    fi
    echo "[smoke] capture went invalid, retry $ATTEMPT" | tee -a "$OUT"
    # Fresh caches, or the retry reloads the half-written artifacts.
    export TORCHINDUCTOR_CACHE_DIR="$E2E_ROOT/caches/inductor/${SMOKE_CACHE_TAG}_r$ATTEMPT"
    export VLLM_CACHE_ROOT="$E2E_ROOT/caches/vllm/${SMOKE_CACHE_TAG}_r$ATTEMPT"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
done
rm -f "$OUT.attempt"
echo "[smoke] capture_retries=$((ATTEMPT-1))" | tee -a "$OUT"

echo "[smoke] === routing evidence ===" | tee -a "$OUT"
python "$E2E_ROOT/scripts/bench/tally_route.py" "$OUT" "$JSON" | tee -a "$OUT"
tot=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["autotune_total"])' "$JSON")
fly=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["flydsl_participated"])' "$JSON")
win=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["flydsl_wins"])' "$JSON")

if [ "${tot:-0}" -eq 0 ]; then
    echo "[smoke] FAIL: no AUTOTUNE mm. Zero can also mean a stale cache hit" \
         "instead of a routing failure -- rerun with a fresh SMOKE_CACHE_TAG" \
         "to tell the two apart" | tee -a "$OUT"
    exit 1
fi
if [ "${fly:-0}" -eq 0 ]; then
    echo "[smoke] FAIL: FlyDSL never entered the candidate set -- check the" \
         "backend gate and FLYDSL_ENABLE_AUTOTUNING" | tee -a "$OUT"
    exit 1
fi
# Wins are recorded but never gated on: on small GEMMs the winner is decided by
# benchmarking noise. This gate proves routing only.
echo "[smoke] FlyDSL participated $fly/$tot times, won $win" | tee -a "$OUT"
echo "[smoke] PASS"
