#!/usr/bin/env bash
# One-line status of the current run, with a completion estimate derived from
# the points already collected rather than from a model of compile cost.
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)/env.sh"
cd "$E2E_ROOT"

RID="$(cat "$E2E_ROOT/state/current_run_id" 2>/dev/null)"
[ -n "$RID" ] || { echo "no active run"; exit 0; }
RUN_DIR="$E2E_ARTIFACTS/$RID"

want=0
for m in $E2E_MODELS; do for c in $E2E_CONCURRENCY_SWEEP; do
    want=$((want + 2 * E2E_REPEATS)); done; done

ok=$(grep -l '"status": "ok"' "$RUN_DIR"/raw/e2e/*.json 2>/dev/null | wc -l)
bad=$(grep -l '"status": "failed"' "$RUN_DIR"/raw/e2e/*.json 2>/dev/null | wc -l)
stage=$(tail -1 "$E2E_ROOT/logs/run_all.log" 2>/dev/null | sed 's/^\[[^]]*\] //')

# Elapsed since the sweep stage began, not since the pipeline began: the stages
# before it are fixed cost and would flatten the rate.
t0=$(grep -a "START 05_e2e" "$E2E_ROOT/logs/run_all.log" 2>/dev/null | tail -1 \
     | sed 's/^\[\([^]]*\)\].*/\1/')
now=$(date +%s)
if [ -n "$t0" ]; then
    s0=$(date -d "$t0" +%s 2>/dev/null || echo "$now")
    el=$(( now - s0 ))
else
    el=0
fi

printf "run %s\n" "$RID"
printf "  stage      %s\n" "${stage:-?}"
printf "  points     %d ok / %d failed  of %d\n" "$ok" "$bad" "$want"
if [ "$ok" -gt 0 ] && [ "$el" -gt 60 ]; then
    rate=$(awk -v o="$ok" -v e="$el" 'BEGIN{print o/e}')
    left=$(awk -v w="$want" -v o="$ok" -v r="$rate" 'BEGIN{printf "%d", (w-o)/r}')
    printf "  elapsed    %dh%02dm   rate %.2f points/h\n" $((el/3600)) $(((el%3600)/60)) \
        "$(awk -v r="$rate" 'BEGIN{print r*3600}')"
    printf "  remaining  ~%dh%02dm  (eta %s)\n" $((left/3600)) $(((left%3600)/60)) \
        "$(date -d "@$((now+left))" '+%H:%M')"
else
    printf "  elapsed    %dh%02dm   (too early for an estimate)\n" $((el/3600)) $(((el%3600)/60))
fi
if [ -f "$E2E_ROOT/state/contention.json" ]; then
    n=$(grep -o '"foreign_samples": [0-9]*' "$E2E_ROOT/state/contention.json" | grep -oE '[0-9]+')
    [ "${n:-0}" -gt 0 ] && printf "  CONTENTION %s samples with a foreign GPU process\n" "$n"
fi
