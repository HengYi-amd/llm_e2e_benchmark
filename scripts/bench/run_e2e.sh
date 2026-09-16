#!/usr/bin/env bash
# One shard of the end-to-end sweep on one GPU: start one server per
# (model, dtype, arm, profile, repeat) and sweep concurrency against it with
# `vllm bench serve`, which reports prefill (TTFT) and decode (TPOT) separately.
# Compiling once per server keeps every concurrency point on the same artifacts.
set -Eeuo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"
. "$E2E_VENV/bin/activate"
cd "$E2E_ROOT"

RUN_DIR="${RUN_DIR:-$E2E_ARTIFACTS/runs/$(cat "$E2E_ROOT/state/current_run_id")}"
mkdir -p "$RUN_DIR/raw/e2e" "$RUN_DIR/logs"

GPU="${BENCH_GPU:-0}"
PORT="${BENCH_PORT:-$((8300 + GPU))}"
# The two visibility masks compose, so setting both would leave no visible GPU.
export HIP_VISIBLE_DEVICES="$GPU"
unset ROCR_VISIBLE_DEVICES

DEADLINE=$(( $(date +%s) + E2E_BUDGET_S ))
SWEEP="${E2E_CONCURRENCY_SWEEP}"

# Must cover both phases (decode: num_tokens = concurrency; prefill: up to
# max_num_batched_tokens). A missing size compiles as a dynamic graph, where
# max_autotune is off and the candidate backend is never considered.
COMPILE_SIZES="[$(echo "$SWEEP $E2E_MAX_NUM_BATCHED_TOKENS" | tr ' ' '\n' \
    | sort -n -u | paste -sd, -)]"

server_pid=""
cleanup() {
    [ -n "$server_pid" ] && kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 30); do
        [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null || break
        sleep 1
    done
    [ -n "$server_pid" ] && kill -KILL "$server_pid" 2>/dev/null || true
    server_pid=""
}
trap cleanup EXIT TERM INT

arm_backends() {
    local v="E2E_BACKENDS_$1"
    echo "${!v:-ATEN,TRITON}"
}

start_server() {
    local model="$1" dtype="$2" arm="$3" tag="$4"
    local be; be="$(arm_backends "$arm")"

    export ARM_TAG="$arm"
    # Caches must be per arm and per shard, or one arm reuses the other's
    # compiled artifacts and skips autotune, silently comparing identical code.
    export TORCHINDUCTOR_CACHE_DIR="$E2E_ROOT/caches/inductor/${arm}_g${GPU}"
    export TRITON_CACHE_DIR="$E2E_ROOT/caches/triton/${arm}_g${GPU}"
    export VLLM_CACHE_ROOT="$E2E_ROOT/caches/vllm/${arm}_g${GPU}"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

    # Without the ATen linear path, unquantized GEMM stays inside a custom op
    # that Inductor cannot autotune, and both arms would run identical kernels.
    export VLLM_FORCE_ATEN_LINEAR=1
    export VLLM_ROCM_USE_SKINNY_GEMM=0

    # Pinned explicitly: defaults are invisible in the command line yet change
    # what each arm means.
    export TORCHINDUCTOR_ORIGAMI="$E2E_ORIGAMI"
    export TORCHINDUCTOR_ORIGAMI_TOPK="$E2E_ORIGAMI_TOPK"
    export FLYDSL_ENABLE_AUTOTUNING="$E2E_FLYDSL_AUTOTUNING"

    local cc
    cc=$(cat <<JSON
{"mode":3,"backend":"inductor","compile_sizes":$COMPILE_SIZES,
 "cudagraph_mode":"PIECEWISE",
 "inductor_compile_config":{"max_autotune":true,"max_autotune_gemm":true,
   "max_autotune_gemm_backends":"$be",
   "max_autotune_gemm_search_space":"$E2E_AUTOTUNE_SEARCH_SPACE",
   "flydsl_enable_autotuning":$([ "$E2E_FLYDSL_AUTOTUNING" = 1 ] && echo true || echo false)}}
JSON
)
    vllm serve "$model" \
        --dtype "$dtype" --tensor-parallel-size "$E2E_TP" \
        --port "$PORT" --host 127.0.0.1 \
        --max-model-len "$E2E_MAX_MODEL_LEN" \
        --max-num-batched-tokens "$E2E_MAX_NUM_BATCHED_TOKENS" \
        --gpu-memory-utilization "$E2E_GPU_MEM_UTIL" \
        --no-enable-prefix-caching --seed 0 \
        ${E2E_KV_CACHE_BYTES:+--kv-cache-memory-bytes "$E2E_KV_CACHE_BYTES"} \
        -cc "$(echo "$cc" | tr -d '\n')" \
        > "$RUN_DIR/logs/server_${tag}.log" 2>&1 &
    server_pid=$!

    local t0; t0=$(date +%s)
    while : ; do
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "  server up after $(( $(date +%s) - t0 ))s (pid $server_pid)"
            return 0
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "  server died during startup, see server_${tag}.log" >&2
            return 1
        fi
        if [ $(( $(date +%s) - t0 )) -gt "$E2E_SERVER_TIMEOUT_S" ]; then
            echo "  server did not become ready within ${E2E_SERVER_TIMEOUT_S}s" >&2
            return 1
        fi
        sleep 5
    done
}

run_point() {
    local model="$1" dtype="$2" arm="$3" prof="$4" isl="$5" osl="$6" conc="$7" rep="$8"
    local short; short=$(echo "$model" | tr '/' '_')
    local pid_="e__${short}__${dtype}__tp${E2E_TP}__${prof}__c${conc}__${arm}__r${rep}"
    local json="$RUN_DIR/raw/e2e/${pid_}.json"
    [ -f "$json" ] && { echo "    skip c=$conc (already have it)"; return 0; }

    local nprompts=$(( conc * E2E_NUM_PROMPTS_MULT ))
    [ "$nprompts" -lt "$E2E_NUM_PROMPTS_MIN" ] && nprompts=$E2E_NUM_PROMPTS_MIN

    local rc=0
    timeout "$E2E_CLIENT_TIMEOUT_S" vllm bench serve \
        --backend vllm --model "$model" --host 127.0.0.1 --port "$PORT" \
        --dataset-name random \
        --random-input-len "$isl" --random-output-len "$osl" \
        --random-range-ratio 0 \
        --num-prompts "$nprompts" --max-concurrency "$conc" \
        --ignore-eos --seed 0 --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-filename "$json" \
        > "$RUN_DIR/logs/${pid_}.log" 2>&1 || rc=$?

    if [ "$rc" -ne 0 ] || [ ! -f "$json" ]; then
        echo "    FAILED c=$conc rc=$rc (see ${pid_}.log)"
        python - "$json" "$arm" "$model" "$dtype" "$prof" "$isl" "$osl" "$conc" "$rep" "$rc" <<'PY'
import json, sys
rc = int(sys.argv[10])
json.dump({"status": "failed", "arm": sys.argv[2], "model_id": sys.argv[3],
           "dtype": sys.argv[4], "profile": sys.argv[5],
           "isl": int(sys.argv[6]), "osl": int(sys.argv[7]),
           "concurrency": int(sys.argv[8]), "repeat_idx": int(sys.argv[9]),
           "fail_rc": rc,
           "fail_kind": {124: "client_timeout"}.get(rc, f"rc{rc}")},
          open(sys.argv[1], "w"), indent=2)
PY
        return 0
    fi

    # Workload provenance travels with the number; a metric without ISL/OSL/
    # concurrency is not comparable.
    python - "$json" "$arm" "$model" "$dtype" "$prof" "$isl" "$osl" "$conc" "$rep" \
             "$RUN_DIR/logs/server_${arm}_${short}_${dtype}_${prof}_r${rep}.log" <<'PY'
import json, os, re, sys
p = sys.argv[1]
d = json.load(open(p))
d.update(status="ok", arm=sys.argv[2], model_id=sys.argv[3], dtype=sys.argv[4],
         profile=sys.argv[5], isl=int(sys.argv[6]), osl=int(sys.argv[7]),
         concurrency=int(sys.argv[8]), repeat_idx=int(sys.argv[9]),
         tp_size=int(os.environ.get("E2E_TP", "1")),
         max_num_batched_tokens=int(os.environ.get("E2E_MAX_NUM_BATCHED_TOKENS", "0")),
         gemm_backends=os.environ.get("E2E_BACKENDS_" + sys.argv[2], ""),
         autotune_search_space=os.environ.get("E2E_AUTOTUNE_SEARCH_SPACE", ""),
         origami=os.environ.get("TORCHINDUCTOR_ORIGAMI", ""),
         origami_topk=os.environ.get("TORCHINDUCTOR_ORIGAMI_TOPK", ""),
         flydsl_autotuning=os.environ.get("FLYDSL_ENABLE_AUTOTUNING", ""),
         e2e_shard_concurrency=int(os.environ.get("E2E_SHARDS", "1")))
# KV capacity granted; if it differs across arms the throughput comparison
# includes a memory difference, so it is recorded for the per-cell check.
log = sys.argv[10] if len(sys.argv) > 10 else None
if log and os.path.exists(log):
    t = open(log, errors="ignore").read()
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", t)
    if m:
        d["kv_cache_tokens"] = int(m[-1].replace(",", ""))
json.dump(d, open(p, "w"), indent=2)
PY
    echo "    c=$conc ok"
}

for model in $E2E_MODELS; do
  for dtype in $E2E_DTYPES; do
    for prof in $E2E_PROFILES; do
      spec_var="E2E_PROFILE_${prof}"; spec="${!spec_var:-}"
      [ -n "$spec" ] || { echo "[e2e] unknown profile '$prof'" >&2; continue; }
      isl="${spec%%:*}"; osl="${spec##*:}"
      # E2E_REP_LIST hands one specific repeat to this shard; without it every
      # shard runs repeat 1 and they overwrite each other.
      for rep in ${E2E_REP_LIST:-$(seq 1 "$E2E_REPEATS")}; do
        # ABBA across repeats so drift over time cannot look like an arm effect.
        arms="$E2E_ARMS"
        [ $((rep % 2)) -eq 0 ] && arms="$(echo "$E2E_ARMS" | tr ' ' '\n' | tac | tr '\n' ' ')"
        for arm in $arms; do
          [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "[e2e] time budget reached"; break 5; }
          short=$(echo "$model" | tr '/' '_')
          tag="${arm}_${short}_${dtype}_${prof}_r${rep}"

          # Skip the server entirely when every point already exists.
          need=0
          for c in $SWEEP; do
            [ -f "$RUN_DIR/raw/e2e/e__${short}__${dtype}__tp${E2E_TP}__${prof}__c${c}__${arm}__r${rep}.json" ] || need=1
          done
          [ "$need" -eq 0 ] && { echo "[e2e] $tag: all points present, skipping"; continue; }

          echo "[e2e] $tag  (compile_sizes=$COMPILE_SIZES)"
          if start_server "$model" "$dtype" "$arm" "$tag"; then
            for c in $SWEEP; do
              run_point "$model" "$dtype" "$arm" "$prof" "$isl" "$osl" "$c" "$rep"
            done
          fi
          cleanup
        done
      done
    done
  done
done

echo "[e2e] done, $(ls -1 "$RUN_DIR/raw/e2e"/*.json 2>/dev/null | wc -l) points on disk"
