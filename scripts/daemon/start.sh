#!/usr/bin/env bash
# Start the pipeline as a background daemon.
#
#   bash scripts/daemon/start.sh
#
# With E2E_CONTAINER set every command runs in that container, otherwise here.
# setsid makes run_all.sh a process group leader so scripts/daemon/stop.sh can
# take down the whole group.
set -Eeuo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"

E2E_CONTAINER="${E2E_CONTAINER:-}"
# Where this repo appears from inside the container; a plain bind mount keeps
# the same path.
CROOT="${E2E_CONTAINER_ROOT:-$E2E_ROOT}"

PIDFILE="$E2E_ROOT/state/daemon.pid"
PGIDFILE="$E2E_ROOT/state/daemon.pgid"
MAINLOG="$E2E_ROOT/logs/daemon.log"

# Single point that knows whether a container is being addressed. -d detaches.
cexec() {
    local detach=0
    [ "${1:-}" = "-d" ] && { detach=1; shift; }
    if [ -n "$E2E_CONTAINER" ]; then
        if [ "$detach" = 1 ]; then
            podman exec -d "$E2E_CONTAINER" bash -lc "$1"
        else
            podman exec "$E2E_CONTAINER" bash -lc "$1"
        fi
    elif [ "$detach" = 1 ]; then
        setsid bash -lc "$1" </dev/null >/dev/null 2>&1 &
        disown
    else
        bash -lc "$1"
    fi
}

mkdir -p "$E2E_ROOT/state" "$E2E_ROOT/logs"

if [ -f "$PIDFILE" ]; then
    oldpid=$(cat "$PIDFILE" 2>/dev/null || true)
    if [ -n "$oldpid" ] && cexec "kill -0 $oldpid 2>/dev/null"; then
        echo "A daemon is already running (pid=$oldpid). Run stop.sh first." >&2
        exit 1
    fi
    rm -f "$PIDFILE" "$PGIDFILE"
fi

echo "== starting daemon $(date --iso-8601=seconds) ==" | tee -a "$MAINLOG"

# pid and pgid are written from inside the new session into the repo, so the
# stop scripts find them wherever they run from.
cexec -d "
    cd $CROOT
    setsid bash -c '
        echo \$\$ > $CROOT/state/daemon.pid
        ps -o pgid= -p \$\$ | tr -d \" \" > $CROOT/state/daemon.pgid
        exec bash $CROOT/scripts/run_all.sh
    ' >> $CROOT/logs/daemon.log 2>&1
"

sleep 3
if [ -f "$PIDFILE" ]; then
    echo "daemon started: pid=$(cat "$PIDFILE") pgid=$(cat "$PGIDFILE" 2>/dev/null || echo '?')"
    echo "log:      $MAINLOG"
    echo "progress: bash $E2E_ROOT/scripts/daemon/status.sh"
    echo "stop:     bash $E2E_ROOT/scripts/daemon/stop.sh"
else
    echo "startup may have failed, check $MAINLOG" >&2
    tail -20 "$MAINLOG" 2>/dev/null || true
    exit 1
fi
