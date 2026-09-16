#!/usr/bin/env bash
# In-container stop for the benchmark daemon. Kill only this project's own
# process tree, TERM before KILL so vLLM can release its VRAM.
set -uo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)/env.sh"

PGID_FILE="$E2E_ROOT/state/daemon.pgid"
[ -f "$PGID_FILE" ] || { echo "no $PGID_FILE, daemon is not running"; exit 0; }
PGID="$(tr -dc '0-9' < "$PGID_FILE")"
[ -n "$PGID" ] || { echo "pgid file is empty, refusing to act"; exit 1; }

# Recorded pids can be recycled: refuse to signal unless the group leader's
# cmdline is still one of this project's entry points. The repo may be mounted
# at another path, so a match on the repo dir name also counts.
REPO_NAME="$(basename "$E2E_ROOT")"
leader_cmd="$(cat "/proc/$PGID/cmdline" 2>/dev/null | tr "\0" " ")"
case "$leader_cmd" in
    *"$E2E_ROOT/scripts/run_all.sh"*|*"$REPO_NAME/scripts/run_all.sh"*|\
        *"$E2E_ROOT/scripts/finish_run.sh"*|*"$REPO_NAME/scripts/finish_run.sh"*|\
        *"$E2E_ROOT/scripts/run_ablation.sh"*|*"$REPO_NAME/scripts/run_ablation.sh"*) ;;
    "") echo "pgid $PGID is gone, nothing to stop"
        rm -f "$PGID_FILE" "$E2E_ROOT/state/daemon.pid"; exit 0 ;;
    *) echo "pgid $PGID does not lead this project's run_all.sh (it is: $leader_cmd)"
       echo "that pid was most likely recycled - refusing to kill."; exit 1 ;;
esac

# Walk the process tree, not the group: `timeout` puts vllm in a group of its
# own, so a group kill leaves the vLLM subtree alive and holding VRAM.
descendants() {
    local root=$1 queue=$1 next pid
    while [ -n "$queue" ]; do
        next=""
        for pid in $queue; do
            echo "$pid"
            next="$next $(pgrep -P "$pid" 2>/dev/null | tr '\n' ' ')"
        done
        queue="$next"
    done
}
alive() { descendants "$PGID" | sort -u | while read -r p; do kill -0 "$p" 2>/dev/null && echo "$p"; done; }

TREE="$(descendants "$PGID" | sort -u | tr '\n' ' ')"
echo "stopping process tree rooted at $PGID, $(echo $TREE | wc -w) processes"
# Leaves first, or children get reparented to init before being signalled.
for p in $(descendants "$PGID" | sort -rn); do kill -TERM "$p" 2>/dev/null; done
for _ in $(seq 60); do
    [ -z "$(alive)" ] && break
    sleep 1
done
left="$(alive | wc -l)"
if [ "$left" -gt 0 ]; then
    # A process wedged in GPU teardown never acts on the TERM.
    echo "$left still alive after TERM, sending KILL"
    for p in $(alive); do kill -KILL "$p" 2>/dev/null; done
    sleep 3
fi
# Final sweep by pgid, for anything that called setpgid and broke the chain.
kill -KILL -- "-$PGID" 2>/dev/null || true
rm -f "$PGID_FILE" "$E2E_ROOT/state/daemon.pid"
rm -rf "$TMPDIR"

echo "--- leftovers (this user only) ---"
pgrep -u "$(id -u)" -af 'vllm|run_e2e|run_all.sh' | grep -v stop.sh || echo "none"
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showpidgpus 2>/dev/null | head -20 || true
fi
