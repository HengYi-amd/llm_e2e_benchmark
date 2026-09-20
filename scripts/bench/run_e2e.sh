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

RUN_DIR="${RUN_DIR:-$E2E_ARTIFACTS/$(cat "$E2E_ROOT/state/current_run_id")}"
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
# Signal the whole group, not the pid: the engine runs in child processes that
# would otherwise be reparented and keep holding GPU memory.
# Collect a process and all of its descendants, leaves last.
descendants() {
    local queue="$1" next pid
    while [ -n "$queue" ]; do
        next=""
        for pid in $queue; do
            echo "$pid"
            next="$next $(pgrep -P "$pid" 2>/dev/null | tr '\n' ' ')"
        done
        queue="$next"
    done
}

# Signal the engine's children too, not just the server process: they hold the
# GPU memory and are reparented if the parent dies first. Walking parent links
# rather than the process group, because the group id of a backgrounded child
# is not reliably its own.
cleanup() {
    [ -n "$server_pid" ] || return 0
    local tree; tree=$(descendants "$server_pid" | sort -u)
    for p in $(echo "$tree" | sort -rn); do kill -TERM "$p" 2>/dev/null || true; done
    for _ in $(seq 30); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
    for p in $(echo "$tree" | sort -rn); do kill -KILL "$p" 2>/dev/null || true; done
    server_pid=""
    return 0
}

trap cleanup EXIT
trap 'cleanup; exit 143' TERM INT HUP

arm_backends() {
    local v="E2E_BACKENDS_$1"
    echo "${!v:-ATEN,TRITON}"
}

start_server() {
    local model="$1" dtype="$2" arm="$3" tag="$4"
    local be; be="$(arm_backends "$arm")"

    # KV capacity is pinned per model: the byte budget that fits one model's
    # weights overruns a larger one's. An unpinned model simply keeps vLLM's
    # own sizing.
    local kv_var="E2E_KV_CACHE_BYTES__$(echo "$model" | tr -c 'A-Za-z0-9' '_')"
    local kv="${!kv_var:-}"

    export ARM_TAG="$arm"
    # Caches must be per arm and per shard, or one arm reuses the other's
    # compiled artifacts and skips autotune, silently comparing identical code.
    # Per arm, per shard AND per repeat, wiped before use. Inductor caches the
    # autotune winner as well as the compiled code, so a warm cache would make a
    # repeat replay the first repeat's selection instead of measuring it again.
    local ck="${arm}_g${GPU}_r${REP_TAG:-1}"
    export TORCHINDUCTOR_CACHE_DIR="$E2E_ROOT/caches/inductor/$ck"
    export TRITON_CACHE_DIR="$E2E_ROOT/caches/triton/$ck"
    export VLLM_CACHE_ROOT="$E2E_ROOT/caches/vllm/$ck"
    # Rename rather than delete: on NFS a directory whose files are still open
    # by a just-killed process keeps .nfsXXXX stubs that make rm fail, and under
    # `set -e` that failure takes the whole shard down. Moving the old tree aside
    # guarantees a fresh cache even when the old one cannot be removed yet.
    local _c
    for _c in "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"; do
        if [ -d "$_c" ]; then
            mv "$_c" "${_c}.stale.$$" 2>/dev/null || true
            ( rm -rf "${_c}.stale.$$" >/dev/null 2>&1 & ) || true
        fi
        mkdir -p "$_c"
    done

    # Without the ATen linear path, unquantized GEMM stays inside a custom op
    # that Inductor cannot autotune, and both arms would run identical kernels.
    export VLLM_FORCE_ATEN_LINEAR=1
    export VLLM_ROCM_USE_SKINNY_GEMM=0

    # Pinned explicitly: defaults are invisible in the command line yet change
    # what each arm means.
    export TORCHINDUCTOR_ORIGAMI="$E2E_ORIGAMI"
    export TORCHINDUCTOR_ORIGAMI_TOPK="$E2E_ORIGAMI_TOPK"
    export FLYDSL_ENABLE_AUTOTUNING="$E2E_FLYDSL_AUTOTUNING"

    # Reading compiled artifacts back from the AOT cache can livelock inside the
    # kernel loader while holding the GIL: the engine stops logging right after
    # weight loading and one core spins indefinitely. Compiling fresh costs time
    # but is the only reliable way past it, and it also rules out a stale cache
    # serving artifacts that were never autotuned in this run.
    export VLLM_DISABLE_COMPILE_CACHE="${E2E_DISABLE_COMPILE_CACHE:-1}"

    # Choices that miss the precompilation deadline are dropped, so the surviving
    # candidate set would depend on machine load rather than on the search space.
    # Shape-aware pruning keeps the pool small enough that nothing is dropped.
    export TORCHINDUCTOR_PRECOMPILATION_TIMEOUT_SECONDS="$E2E_PRECOMPILE_TIMEOUT_S"
    # Inductor otherwise caps its compile-worker pool at 32 regardless of cores.
    export TORCHINDUCTOR_COMPILE_THREADS="$E2E_COMPILE_THREADS"

    # EXHAUSTIVE benchmarks thousands of candidates, each allocating its own
    # output and workspace tensors; the caching allocator fragments badly enough
    # that a 2 MiB request can fail with tens of gigabytes still free.
    export PYTORCH_HIP_ALLOC_CONF="${E2E_ALLOC_CONF:-expandable_segments:True}"

    # vLLM gives the engine core this long to report ready. Under EXHAUSTIVE the
    # core spends hours autotuning before it gets there, so the stock ten
    # minutes is far short of what this workload needs.
    export VLLM_ENGINE_READY_TIMEOUT_S="${E2E_ENGINE_READY_TIMEOUT_S:-86400}"

    # Triton's exhaustive list is thousands of configs and dominates compile
    # time; the candidate backend's is a fraction of it. Keeping Triton on its
    # default list makes the sweep tractable.
    export TORCHINDUCTOR_TRITON_DEFAULT_SPACE="${E2E_TRITON_DEFAULT_SPACE:-0}"

    # A card handed over from a finished shard can still hold its engine's
    # allocation: the process is gone before the driver has reclaimed the VRAM,
    # and vLLM refuses to start when free memory is under its requested
    # utilization. Wait for the card to drain rather than fail the point.
    local need_mib="${E2E_MIN_FREE_MIB:-268000}"
    local waited=0
    while [ "$waited" -lt "${E2E_VRAM_WAIT_S:-1200}" ]; do
        local used
        used=$(rocm-smi --showmeminfo vram 2>/dev/null \
               | grep -oE "GPU\[$GPU\].*Used Memory \(B\): [0-9]+" \
               | grep -oE '[0-9]+$' | tail -1)
        [ -n "$used" ] || break
        local free_mib=$(( (309220868096 - used) / 1048576 ))
        [ "$free_mib" -ge "$need_mib" ] && break
        [ "$waited" -eq 0 ] && echo "    waiting for GPU $GPU to drain (${free_mib}MiB free)"
        sleep 10
        waited=$((waited + 10))
    done

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
        --max-num-seqs "$E2E_MAX_NUM_SEQS" \
        --gpu-memory-utilization "$E2E_GPU_MEM_UTIL" \
        --no-enable-prefix-caching --seed 0 \
        ${kv:+--kv-cache-memory-bytes "$kv"} \
        -cc "$(echo "$cc" | tr -d '\n')" \
        > "$RUN_DIR/logs/server_${tag}.log" 2>&1 &
    server_pid=$!

    # Two separate limits. The wall-clock one bounds a legitimately slow
    # compile; the log-stall one catches a startup that has stopped making any
    # progress at all, which otherwise burns the full wall-clock budget.
    local t0 last_sz last_move; t0=$(date +%s); last_sz=0; last_move=$t0
    while : ; do
        local sz; sz=$(stat -c %s "$RUN_DIR/logs/server_${tag}.log" 2>/dev/null || echo 0)
        if [ "$sz" -ne "$last_sz" ]; then last_sz=$sz; last_move=$(date +%s); fi
        if [ $(( $(date +%s) - last_move )) -gt "${E2E_SERVER_STALL_S:-900}" ]; then
            echo "  server log has not advanced in ${E2E_SERVER_STALL_S:-900}s, treating as hung" >&2
            cleanup
            return 1
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "  server exited during startup, see server_${tag}.log" >&2
            return 1
        fi
        # Without a timeout the probe can block on an established but
        # unanswered connection, freezing both limits below.
        if curl -fsS --connect-timeout 5 --max-time 10 \
                "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
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

    # A zero exit code does not mean the requests succeeded; the harness
    # reports per-request failures inside the JSON.
    if [ "$rc" -eq 0 ] && [ -f "$json" ]; then
        if ! python -c "
import json,sys
d=json.load(open(sys.argv[1]))
c=d.get('completed'); n=int(sys.argv[2])
sys.exit(0 if c is None or int(c) >= n else 1)
" "$json" "$nprompts" 2>/dev/null; then
            echo "    c=$conc: incomplete responses, marking failed"
            rc=125
            rm -f "$json"
        fi
    fi
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
             "$RUN_DIR/logs/server_${tag}.log" <<'PY'
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
         disable_compile_cache=os.environ.get("VLLM_DISABLE_COMPILE_CACHE", ""),
         compile_threads=os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", ""),
         triton_default_space=os.environ.get("TORCHINDUCTOR_TRITON_DEFAULT_SPACE", ""),
         max_num_seqs=int(os.environ.get("E2E_MAX_NUM_SEQS", "0")),
         precompile_timeout_s=os.environ.get(
             "TORCHINDUCTOR_PRECOMPILATION_TIMEOUT_SECONDS", ""),
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
      # Prefill runs at ISL x concurrency tokens, capped by the chunk size, and
      # decode at the concurrency itself. Compiling the chunk size instead of
      # the real prefill width leaves prefill on a dynamic graph, where
      # max_autotune is off and the candidate backend is never considered.
      COMPILE_SIZES="[$(
          for c in $SWEEP; do
              echo "$c"
              p=$(( isl * c ))
              [ "$p" -gt "$E2E_MAX_NUM_BATCHED_TOKENS" ] && p="$E2E_MAX_NUM_BATCHED_TOKENS"
              echo "$p"
          done | sort -n -u | paste -sd, -)]"
      # E2E_REP_LIST hands one specific repeat to this shard; without it every
      # shard runs repeat 1 and they overwrite each other.
      for rep in ${E2E_REP_LIST:-$(seq 1 "$E2E_REPEATS")}; do
        export REP_TAG="$rep"
        # ABBA across repeats so drift over time cannot look like an arm effect.
        arms="$E2E_ARMS"
        [ $((rep % 2)) -eq 0 ] && arms="$(echo "$E2E_ARMS" | tr ' ' '\n' | tac | tr '\n' ' ')"
        for arm in $arms; do
          [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "[e2e] time budget reached"; break 5; }
          short=$(echo "$model" | tr '/' '_')
          # Concurrency belongs in the tag: shards of one model that differ
          # only by concurrency run at the same time, and a shared tag makes
          # them truncate each other's server log - the same log each point
          # reads its KV capacity back out of.
          tag="${arm}_${short}_${dtype}_${prof}_r${rep}_c$(echo "$SWEEP" | tr ' ' '-')"

          # Skip the server entirely when every point already exists.
          need=0
          for c in $SWEEP; do
            [ -f "$RUN_DIR/raw/e2e/e__${short}__${dtype}__tp${E2E_TP}__${prof}__c${c}__${arm}__r${rep}.json" ] || need=1
          done
          [ "$need" -eq 0 ] && { echo "[e2e] $tag: all points present, skipping"; continue; }

          echo "[e2e] $tag  (compile_sizes=$COMPILE_SIZES)"
          # Graph capture fails at startup on this platform at a low but steady
          # rate, independent of the backend under test. Retrying on that one
          # signature costs a recompile; not retrying costs the whole arm, and
          # an arm missing from a cell removes the comparison the run exists for.
          ok=0
          for try in $(seq 1 "${E2E_SERVER_RETRIES:-3}"); do
              if start_server "$model" "$dtype" "$arm" "$tag"; then ok=1; break; fi
              if grep -aq 'StreamCaptureInvalidated\|operation failed due to a previous error during capture' \
                   "$RUN_DIR/logs/server_${tag}.log" 2>/dev/null; then
                  echo "  graph capture flake on attempt $try, retrying"
                  mv "$RUN_DIR/logs/server_${tag}.log" \
                     "$RUN_DIR/logs/server_${tag}.capturefail${try}.log" 2>/dev/null || true
                  cleanup
                  continue
              fi
              echo "  server failed for a reason other than the capture flake; not retrying"
              break
          done
          if [ "$ok" = 1 ]; then
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
