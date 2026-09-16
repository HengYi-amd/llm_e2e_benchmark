#!/usr/bin/env bash
# Progress and stall monitor for a running sweep.
#
# Keep the loop in this file: run inline via `bash -c`, the shell carries
# the match pattern in its own cmdline and reports a phantom stall every cycle.
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)/env.sh"
cd "$E2E_ROOT"

RID="$(cat "$E2E_ROOT/state/current_run_id")"
RAW="$E2E_ARTIFACTS/runs/$RID/raw/e2e"
DAEMON_LOG="$E2E_ROOT/logs/daemon.log"
PGID_FILE="$E2E_ROOT/state/daemon.pgid"

# Clients run under `timeout <client timeout> vllm ...`; the wrapper's argv is
# what identifies a benchmark client still in flight.
PAT="timeout $E2E_CLIENT_TIMEOUT_S vllm"
STALL_S="${E2E_WATCH_STALL_S:-$E2E_CLIENT_TIMEOUT_S}"
INTERVAL_S="${E2E_WATCH_INTERVAL_S:-120}"

nwords() { set -- $1; echo $#; }
TOTAL="${E2E_EXPECTED_POINTS:-$((
    $(nwords "$E2E_MODELS") * $(nwords "$E2E_DTYPES") * $(nwords "$E2E_PROFILES") *
    $(nwords "$E2E_ARMS") * $(nwords "$E2E_CONCURRENCY_SWEEP") * E2E_REPEATS ))}"

DL=$(wc -l < "$DAEMON_LOG")
prev=$(ls "$RAW"/*.json 2>/dev/null | wc -l)
seen=""
while true; do
    sleep "$INTERVAL_S"
    n=$(ls "$RAW"/*.json 2>/dev/null | wc -l)
    [ "$n" -ge $((prev+2)) ] && { echo "e2e progress $n/$TOTAL"; prev=$n; }
    for f in $(grep -l '"status": *"failed"' "$RAW"/*.json 2>/dev/null); do
        case "$seen" in *"$f"*) ;; *) echo "new failed point: $(basename "$f")"; seen="$seen $f";; esac
    done
    # Only argv that starts with the pattern counts; this also excludes the
    # monitor's own tree.
    ps -u "$(id -u)" -o pid=,etimes=,args= | while read -r p e a; do
        case "$a" in "$PAT"*) [ "$e" -gt "$STALL_S" ] && echo "possible stall in cache load: pid $p for $((e/60)) min";; esac
    done
    tail -n +$((DL+1)) "$DAEMON_LOG" 2>/dev/null | grep -E '(START|PASS|FAIL) +(10|10b|11|12|13)_' || true
    DL=$(wc -l < "$DAEMON_LOG")
    pgrep -g "$(tr -dc '0-9' < "$PGID_FILE")" >/dev/null || {
        echo "daemon exited, e2e=$n/$TOTAL, last line: $(tail -1 "$DAEMON_LOG")"; break; }
done
