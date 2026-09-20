#!/usr/bin/env bash
# Run the whole pipeline. Stage status is recorded under state/stages/ and a
# stage marked PASSED is skipped, so a re-run resumes.
#
#   bash scripts/run_all.sh              # everything
#   E2E_FROM=05 bash scripts/run_all.sh  # from a stage onwards
#   E2E_ONLY="07 08 09" bash scripts/run_all.sh
set -Eeuo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)/env.sh"
cd "$E2E_ROOT"

STATE="$E2E_ROOT/state/stages"
LOG="$E2E_ROOT/logs/run_all.log"
mkdir -p "$STATE" "$E2E_ROOT/logs"

if [ ! -f "$E2E_ROOT/state/current_run_id" ]; then
    date -u +"%Y%m%dT%H%M%SZ__$(echo "$E2E_DTYPES" | tr ' ' '-')" \
        > "$E2E_ROOT/state/current_run_id"
fi
RUN_ID="$(cat "$E2E_ROOT/state/current_run_id")"
RUN_DIR="$E2E_ARTIFACTS/$RUN_ID"
mkdir -p "$RUN_DIR"/{raw/e2e,logs,normalized,figures,manifest}
export RUN_DIR

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }

# Only this instance's own descendants. Matching on the project path instead
# would also select shards and supervisors started by hand or by another
# pipeline, and the exit trap would kill work this script never owned.
descendants() {
    local frontier="$$" out="" next p child ppid
    while [ -n "$frontier" ]; do
        next=""
        for p in $frontier; do
            for child in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
                ppid=$(awk '{print $4}' "/proc/$child/stat" 2>/dev/null) || continue
                [ "$ppid" = "$p" ] && { out="$out $child"; next="$next $child"; }
            done
        done
        frontier="$next"
    done
    echo $out
}

cleanup() {
    rc=$?
    log "cleanup (rc=$rc)"
    # Shells first, and with KILL: a shard waiting on a foreground child defers
    # its own TERM handler, so a polite signal leaves it free to start another
    # server while the rest of the teardown runs.
    for p in $(descendants); do
        tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null \
            | grep -qE 'run_e2e\.sh|run_sharded\.sh|calibrate_kv' \
            && kill -KILL "$p" 2>/dev/null || true
    done
    sleep 2
    for p in $(descendants); do kill -TERM "$p" 2>/dev/null || true; done
    for _ in $(seq 30); do
        [ -z "$(descendants)" ] && break
        sleep 1
    done
    for p in $(descendants); do kill -KILL "$p" 2>/dev/null || true; done
    rm -rf "$TMPDIR"
    log "cleanup done ($(descendants | wc -w) still alive)"
    exit $rc
}
trap cleanup EXIT TERM INT HUP QUIT

run_stage() {
    local id="$1" cmd="$2"
    if [ -n "${E2E_ONLY:-}" ] && ! echo " $E2E_ONLY " | grep -q " ${id%%_*} "; then return 0; fi
    if [ -n "${E2E_FROM:-}" ] && [ "${id%%_*}" \< "$E2E_FROM" ]; then return 0; fi
    if [ "$(cat "$STATE/$id.status" 2>/dev/null)" = PASSED ]; then
        log "SKIP  $id (already PASSED)"; return 0
    fi
    log "START $id"
    local t0; t0=$(date +%s)
    # A stage without a timeout can consume the whole run budget and starve
    # every stage after it.
    local tmo="${E2E_STAGE_TIMEOUT_S:-10800}"
    case "$id" in 05_*) tmo="${E2E_SWEEP_TIMEOUT_S:-39600}" ;; esac
    # Run the stage in the background and wait on it. Bash defers trap handling
    # while a foreground child runs, so a TERM sent during a long stage would sit
    # pending until that stage finished - long enough for the servers it started
    # to keep holding the GPUs. `wait` is interruptible, so cleanup runs at once.
    local spid rc2
    timeout -k 60 "$tmo" bash -c "$cmd" >> "$E2E_ROOT/logs/$id.log" 2>&1 &
    spid=$!
    wait "$spid" && rc2=0 || rc2=$?
    if [ "$rc2" -eq 0 ]; then
        echo PASSED > "$STATE/$id.status"
        log "PASS  $id ($(( $(date +%s) - t0 ))s)"
    else
        echo FAILED > "$STATE/$id.status"
        log "FAIL  $id ($(( $(date +%s) - t0 ))s) - see logs/$id.log"
        return 1
    fi
}

PY="source $E2E_VENV/bin/activate &&"

log "=== run $RUN_ID ==="
run_stage 00_preflight   "bash $E2E_ROOT/scripts/preflight/check_gpus.sh"
run_stage 01_verify      "$PY python $E2E_ROOT/scripts/setup/verify_stack.py"
run_stage 02_patch_vllm  "bash $E2E_ROOT/scripts/setup/patch_vllm.sh"
# Gate: without proof that the backend is actually routed to, a green sweep can
# mean it was never exercised at all.
run_stage 03_route_smoke "bash $E2E_ROOT/scripts/bench/route_smoke.sh" \
    || log "route smoke failed; 06_route_proof decides whether results are publishable"
run_stage 04_kv_pin      "bash $E2E_ROOT/scripts/bench/calibrate_kv_all.sh" \
    || log "KV pin unavailable; normalize will flag cells whose arms differ"
run_stage 05_e2e         "bash $E2E_ROOT/scripts/bench/run_sharded.sh"
# Must precede the report: reported speedups are only attributable to the
# kernel if routing was verified for the same run.
run_stage 06_route_proof "$PY python $E2E_ROOT/scripts/proof/route_evidence.py $RUN_DIR"
run_stage 07_normalize   "$PY python $E2E_ROOT/scripts/normalize/normalize.py --run-dir $RUN_DIR"
run_stage 08_plot        "$PY python $E2E_ROOT/scripts/plot/plot.py --run-dir $RUN_DIR --metrics $(echo "$E2E_METRICS" | tr ' ' ',')"
run_stage 09_publish    "$PY python $E2E_ROOT/scripts/publish.py --run-dir $RUN_DIR --result-dir $E2E_RESULT --suffix '${E2E_RESULT_SUFFIX:-}'"
# Cross-model comparison figures. Skipped rather than failed when the sweep
# covers a single model: the chart only exists to put models side by side.
run_stage 10_combined   "$PY python $E2E_ROOT/scripts/plot/combined.py --run-dir $RUN_DIR --out-dir $E2E_RESULT/${E2E_COMBINED_DIR:-bf16_result_png}" \
    || log "combined figures skipped (single model, or not enough cells)"

log "artifacts: $RUN_DIR"
log "figures:   $(ls -1 "$RUN_DIR/figures"/*.png 2>/dev/null | wc -l)"
log "results:   $E2E_RESULT"
