#!/usr/bin/env bash
# Gate before a run: are the target GPUs free, and is a CI runner using them?
# An idle GPU still reports some used VRAM and non-zero activity, so the
# thresholds are "a bit above idle", not "exactly zero".
set -uo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
VRAM_LIMIT=${VRAM_LIMIT:-2000}
ACT_LIMIT=${ACT_LIMIT:-5}
MIN_FREE_GB=${E2E_MIN_FREE_GB:-100}

# Exact match, not pgrep -f: a -f pattern also hits container supervisors and
# this script's own grep.
if pgrep -x "Runner.Listener" >/dev/null 2>&1 || pgrep -x "Runner.Worker" >/dev/null 2>&1; then
    echo "[preflight] !! a GitHub Actions runner is live, refusing to start"
    exit 1
fi

# Either SMI tool will do; with neither installed, warn and proceed rather than
# block the run on a missing tool.
if command -v amd-smi >/dev/null 2>&1; then
    SMI=amd-smi
elif command -v rocm-smi >/dev/null 2>&1; then
    SMI=rocm-smi
else
    SMI=""
    echo "[preflight] neither amd-smi nor rocm-smi found, skipping the GPU check"
fi

gpu_vram_mb() {
    case "$SMI" in
        amd-smi)  amd-smi metric -g "$1" -m 2>/dev/null | awk '/USED_VRAM/{print $2; exit}' ;;
        rocm-smi) rocm-smi -d "$1" --showmeminfo vram --csv 2>/dev/null |
                      awk -F, 'NR>1 && $3+0 > 0 {printf "%d\n", $3/1048576; exit}' ;;
    esac
}
gpu_activity() {
    case "$SMI" in
        amd-smi)  amd-smi metric -g "$1" -u 2>/dev/null | awk '/GFX_ACTIVITY/{print $2; exit}' ;;
        rocm-smi) rocm-smi -d "$1" --showuse --csv 2>/dev/null |
                      awk -F, 'NR>1 {gsub(/[^0-9]/,"",$2); if ($2 != "") print $2; exit}' ;;
    esac
}

busy=""
if [ -n "$SMI" ]; then
    echo "[preflight] GPU usage:"
    for g in $GPUS; do
        v=$(gpu_vram_mb "$g")
        a=$(gpu_activity "$g")
        v=${v:-99999}; a=${a:-100}
        printf "  GPU%s used_vram=%sMB gfx=%s%%\n" "$g" "$v" "$a"
        if [ "$v" -gt "$VRAM_LIMIT" ] 2>/dev/null || [ "$a" -ge "$ACT_LIMIT" ] 2>/dev/null; then
            busy="$busy $g"
        fi
    done
fi

if [ -n "$busy" ]; then
    echo "[preflight] !! these GPUs are in use:$busy"
    [ "${ALLOW_BUSY:-0}" = "1" ] || exit 1
fi

echo "[preflight] disk:"
df -h "$E2E_ROOT" | tail -1
avail=$(df -BG "$E2E_ROOT" | tail -1 | awk '{gsub("G","",$4); print $4}')
[ "${avail:-0}" -lt "$MIN_FREE_GB" ] && { echo "[preflight] !! less than ${MIN_FREE_GB}G free"; exit 1; }

echo "[preflight] OK"
