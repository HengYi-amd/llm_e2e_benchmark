#!/usr/bin/env bash
# Stop the pipeline without leaving anything behind that still holds VRAM.
#
#   bash scripts/daemon/stop.sh          # TERM, so the traps get to clean up
#   bash scripts/daemon/stop.sh --force  # KILL once the gentle path has failed
#
# Order is load-bearing: TERM the group so run_all.sh's trap closes the vLLM
# server, wait, KILL what remains, sweep leftovers, then re-check the GPUs.
set -uo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"

E2E_CONTAINER="${E2E_CONTAINER:-}"
# Where this repo appears from inside the container; a plain bind mount keeps
# the same path.
CROOT="${E2E_CONTAINER_ROOT:-$E2E_ROOT}"

PIDFILE="$E2E_ROOT/state/daemon.pid"
PGIDFILE="$E2E_ROOT/state/daemon.pgid"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# Single point that knows whether a container is being addressed.
cexec() {
    if [ -n "$E2E_CONTAINER" ]; then
        podman exec "$E2E_CONTAINER" bash -lc "$1"
    else
        bash -lc "$1"
    fi
}

pid=$(cat "$PIDFILE" 2>/dev/null || true)
pgid=$(cat "$PGIDFILE" 2>/dev/null || true)
suppid=$(cat "$E2E_ROOT/state/supervisor.pid" 2>/dev/null || true)
suppgid=$(cat "$E2E_ROOT/state/supervisor.pgid" 2>/dev/null || true)

# The supervisor restarts the pipeline, so it must go first.
if [ -n "$suppgid" ]; then
    echo "-> SIGTERM supervisor process group $suppgid"
    cexec "kill -TERM -$suppgid 2>/dev/null" || true
fi

if [ -z "$pid" ] && [ -z "$pgid" ]; then
    echo "no daemon on record; still running the leftover sweep and the GPU check."
fi

if [ -n "$pgid" ]; then
    echo "-> SIGTERM process group $pgid"
    cexec "kill -TERM -$pgid 2>/dev/null" || true
elif [ -n "$pid" ]; then
    echo "-> SIGTERM pid $pid"
    cexec "kill -TERM $pid 2>/dev/null" || true
fi

for i in $(seq 1 60); do
    if [ -n "$pid" ] && cexec "kill -0 $pid 2>/dev/null"; then sleep 1; else break; fi
done

if [ -n "$pid" ] && cexec "kill -0 $pid 2>/dev/null"; then
    if [ "$FORCE" = "1" ]; then
        echo "-> still alive, SIGKILL process group $pgid"
        cexec "kill -KILL -${pgid:-$pid} 2>/dev/null" || true
    else
        echo "!! still alive after 60s. Force it: bash $0 --force" >&2
    fi
fi

# Leftover sweep, off by default. The targeted process-group kill above already
# covers everything this daemon started; a pattern sweep additionally matches
# shards launched by hand or by a second sweep running in parallel, and killing
# those destroys work this script never owned. Set E2E_STOP_SWEEP=1 to opt in.
if [ "${E2E_STOP_SWEEP:-0}" != "1" ]; then
    echo "-> skipping leftover sweep (set E2E_STOP_SWEEP=1 to force)"
    exit 0
fi
PROJ_DIR="$(dirname "$CROOT")"
SWEEP_PAT="$CROOT|$PROJ_DIR/$(basename "$E2E_TORCH_SRC")|$PROJ_DIR/$(basename "$E2E_VLLM_SRC")"
SWEEP_PAT="$SWEEP_PAT|run_all.sh|run_e2e.sh|run_sharded.sh"
echo "-> sweeping for leftovers matching $SWEEP_PAT"
cexec "
  for p in \$(pgrep -u "$(id -u)" -f '$SWEEP_PAT' 2>/dev/null); do
     [ \"\$p\" = \"\$\$\" ] && continue
     echo \"   kill \$p : \$(ps -o args= -p \$p 2>/dev/null | cut -c1-100)\"
     kill -TERM \$p 2>/dev/null
  done
  sleep 5
  for p in \$(pgrep -u "$(id -u)" -f '$SWEEP_PAT' 2>/dev/null); do
     [ \"\$p\" = \"\$\$\" ] && continue
     kill -KILL \$p 2>/dev/null
  done
" || true

rm -f "$PIDFILE" "$PGIDFILE" \
      "$E2E_ROOT/state/supervisor.pid" "$E2E_ROOT/state/supervisor.pgid"

# TMPDIR is the only thing a run writes outside the repo; nothing else
# reclaims it.
echo "-> removing TMPDIR $TMPDIR"
cexec "rm -rf '$TMPDIR' 2>/dev/null; echo '   done'" || true

# A process that died mid-teardown keeps its allocations and the next run then
# fails on memory for no visible reason, so confirm VRAM is back near idle.
echo "-> GPU re-check (expect near-idle VRAM and gfx=0%)"
cexec "
for g in ${E2E_GPUS:-0 1 2 3 4 5 6 7}; do
  v=\$(amd-smi metric -g \$g -m 2>/dev/null | awk '/USED_VRAM/{print \$2; exit}')
  a=\$(amd-smi metric -g \$g -u 2>/dev/null | awk '/GFX_ACTIVITY/{print \$2; exit}')
  printf '   GPU%s used_vram=%sMB gfx=%s%%\n' \"\$g\" \"\$v\" \"\$a\"
done"

# Not through cexec: a container pid namespace hides the orphans this looks for.
echo "-> processes holding a GPU, host side:"
n=0
for p in $(ls /sys/class/kfd/kfd/proc/ 2>/dev/null); do
  echo "   PID=$p $(ps -o args= -p "$p" 2>/dev/null | cut -c1-90)"; n=$((n+1))
done
[ "$n" = "0" ] && echo "   (none)"
echo "stopped."
