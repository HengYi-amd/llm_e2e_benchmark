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
RUN_DIR="$E2E_ARTIFACTS/runs/$RUN_ID"
mkdir -p "$RUN_DIR"/{raw/e2e,logs,normalized,figures,report,manifest}
export RUN_DIR

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }

# Long-lived GPU processes must not outlive the pipeline holding VRAM. Match
# only this project's own processes; a broad pattern would hit other users.
cleanup() {
    rc=$?
    log "cleanup (rc=$rc)"
    pkill -u "$(id -u)" -f "$E2E_ROOT/scripts/bench" 2>/dev/null || true
    for p in $(pgrep -u "$(id -u)" -f "vllm serve" 2>/dev/null); do
        tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "$E2E_ROOT" && kill -TERM "$p" 2>/dev/null
    done
    sleep 5
    for p in $(pgrep -u "$(id -u)" -f "vllm serve" 2>/dev/null); do
        tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "$E2E_ROOT" && kill -KILL "$p" 2>/dev/null
    done
    rm -rf "$TMPDIR"
    log "cleanup done"
    exit $rc
}
trap cleanup EXIT TERM INT

run_stage() {
    local id="$1" cmd="$2"
    if [ -n "${E2E_ONLY:-}" ] && ! echo " $E2E_ONLY " | grep -q " ${id%%_*} "; then return 0; fi
    if [ -n "${E2E_FROM:-}" ] && [ "${id%%_*}" \< "$E2E_FROM" ]; then return 0; fi
    if [ "$(cat "$STATE/$id.status" 2>/dev/null)" = PASSED ]; then
        log "SKIP  $id (already PASSED)"; return 0
    fi
    log "START $id"
    local t0; t0=$(date +%s)
    if bash -c "$cmd" >> "$E2E_ROOT/logs/$id.log" 2>&1; then
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
run_stage 03_route_smoke "bash $E2E_ROOT/scripts/bench/route_smoke.sh"
run_stage 04_kv_pin      "bash $E2E_ROOT/scripts/bench/calibrate_kv.sh"
run_stage 05_e2e         "bash $E2E_ROOT/scripts/bench/run_sharded.sh"
# Must precede the report: reported speedups are only attributable to the
# kernel if routing was verified for the same run.
run_stage 06_route_proof "$PY python $E2E_ROOT/scripts/proof/route_evidence.py $RUN_DIR"
run_stage 07_normalize   "$PY python $E2E_ROOT/scripts/normalize/normalize.py --run-dir $RUN_DIR"
run_stage 08_plot        "$PY python $E2E_ROOT/scripts/plot/plot.py --run-dir $RUN_DIR --metrics $(echo "$E2E_METRICS" | tr ' ' ',')"
run_stage 09_report      "$PY python $E2E_ROOT/scripts/report/report.py --run-dir $RUN_DIR"

log "artifacts: $RUN_DIR"
log "figures:   $(ls -1 "$RUN_DIR/figures"/*.png 2>/dev/null | wc -l)"
log "report:    $RUN_DIR/report/REPORT.md"
