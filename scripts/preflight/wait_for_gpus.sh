#!/usr/bin/env bash
# Wait for the target GPUs to go idle. Other jobs are never preempted or
# killed, and the wait has a hard deadline so it cannot become permanent.
set -uo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
VRAM_LIMIT=${VRAM_LIMIT:-2000}
DEADLINE=$(( $(date +%s) + ${GPU_WAIT_TIMEOUT_S:-43200} ))
INTERVAL=${GPU_WAIT_INTERVAL_S:-60}

# Without an SMI tool every GPU reads as busy and the full deadline would be
# waited out for nothing, so an absent tool means "go ahead", with a warning.
if command -v amd-smi >/dev/null 2>&1; then
    SMI=amd-smi
elif command -v rocm-smi >/dev/null 2>&1; then
    SMI=rocm-smi
else
    echo "[wait_gpu] neither amd-smi nor rocm-smi found, not waiting"
    exit 0
fi

gpu_vram_mb() {
    case "$SMI" in
        amd-smi)  amd-smi metric -g "$1" -m 2>/dev/null | awk '/USED_VRAM/{print $2; exit}' ;;
        rocm-smi) rocm-smi -d "$1" --showmeminfo vram --csv 2>/dev/null |
                      awk -F, 'NR>1 && $3+0 > 0 {printf "%d\n", $3/1048576; exit}' ;;
    esac
}

while true; do
    busy=""
    for g in $GPUS; do
        v=$(gpu_vram_mb "$g")
        v=${v:-99999}
        [ "$v" -gt "$VRAM_LIMIT" ] 2>/dev/null && busy="$busy $g"
    done
    [ -z "$busy" ] && { echo "[wait_gpu] all idle, continuing"; exit 0; }
    [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "[wait_gpu] !! timed out, still in use:$busy"; exit 1; }
    echo "[wait_gpu] $(date --iso-8601=seconds) in use:$busy - waiting (not touching other jobs)"
    sleep "$INTERVAL"
done
