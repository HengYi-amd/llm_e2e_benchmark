#!/usr/bin/env bash
# Keep every missing data point covered by a live shard, one shard at a time.
#
# The whole-pipeline watchdog is the wrong tool once a run is most of the way
# through: tearing everything down to recover one dead shard throws away the
# hours the other seven have invested. This supervisor instead asks, for each
# point that does not exist yet, whether some process is currently working on
# it, and starts one only for the points where nobody is. Running shards are
# never signalled.
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)/env.sh"
cd "$E2E_ROOT"

LOG="$E2E_ROOT/logs/supervisor.log"
DIAG="$E2E_ROOT/logs/failures.log"
mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date --iso-8601=seconds)] $*" >> "$LOG"; }
diag() { echo "[$(date --iso-8601=seconds)] $*" >> "$DIAG"; }

: "${SV_POLL_S:=300}"
: "${SV_MIN_FREE_MIB:=250000}"
: "${SV_MAX_LAUNCH:=24}"
: "${SV_UTIL:=0.85}"

RID="${RUN_ID:-$(cat "$E2E_ROOT/state/current_run_id" 2>/dev/null)}"
RUN_DIR="${RUN_DIR:-$E2E_ARTIFACTS/$RID}"
launches=0

declare -A HIP2DEV
while read -r h d _; do HIP2DEV["$h"]="$d"; done \
    < <(bash "$E2E_ROOT/scripts/bench/gpumap.sh" 2>/dev/null)

point_file() {   # model conc arm rep -> path
    local short; short="$(echo "$1" | tr '/' '_')"
    echo "$RUN_DIR/raw/e2e/e__${short}__${E2E_DTYPES}__tp${E2E_TP}__${E2E_PROFILES}__c${2}__${3}__r${4}.json"
}

# (model,conc) pairs some live run_e2e.sh is currently working on.
covered_pairs() {
    local p c g m cc
    for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
        c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
        case "$c" in *"bench/run_e2e.sh"*) ;; *) continue ;; esac
        m=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -m1 '^E2E_MODELS=' | cut -d= -f2)
        cc=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -m1 '^E2E_CONCURRENCY_SWEEP=' | cut -d= -f2)
        rr=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -m1 '^E2E_REP_LIST=' | cut -d= -f2)
        [ -n "$m" ] && [ -n "$cc" ] && echo "$m|$cc|${rr:-1}"
    done
}

# HIP indices with no shard of ours and enough free memory for a fresh engine.
free_gpus() {
    local busy p g h used free
    busy=""
    for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
        c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
        case "$c" in *"bench/run_e2e.sh"*) ;; *) continue ;; esac
        g=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep '^BENCH_GPU=' | cut -d= -f2)
        [ -n "$g" ] && busy="$busy $g"
    done
    for h in "${!HIP2DEV[@]}"; do
        case " $busy " in *" $h "*) continue ;; esac
        used=$(rocm-smi --showmeminfo vram 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
               | grep -oE "GPU\[${HIP2DEV[$h]}\].*Used Memory \(B\): [0-9]+" \
               | grep -oE '[0-9]+$' | tail -1)
        [ -n "$used" ] || continue
        free=$(( (309220868096 - used) / 1048576 ))
        [ "$free" -ge "$SV_MIN_FREE_MIB" ] && echo "$h"
    done
}

# Why the previous attempt at this point died, so the night leaves a record
# rather than just a gap.
record_failure() {
    local model="$1" conc="$2" arm="$3" rep="$4" short log sig
    short="$(echo "$model" | tr '/' '_')"
    log="$RUN_DIR/logs/server_${arm}_${short}_${E2E_DTYPES}_${E2E_PROFILES}_r${rep}_c${conc}.log"
    [ -f "$log" ] || { diag "$short c=$conc $arm r$rep: no server log to inspect"; return; }
    sig=$(grep -aoE "OutOfMemoryError[^\"]{0,60}|LLVM ERROR: out of memory|MemoryError|hipError[A-Za-z]+|CUDA error: [a-z ]+|Engine core initialization failed" \
          "$log" 2>/dev/null | sort -u | head -3 | tr '\n' ';')
    diag "$short c=$conc $arm r$rep: rounds=$(grep -ac 'AUTOTUNE mm(' "$log") signature=${sig:-<none>}"
}

say "supervisor started (poll ${SV_POLL_S}s, util $SV_UTIL, run $RID)"
while : ; do
    missing=0 relaunched=0
    cov=" $(covered_pairs | tr '\n' ' ') "
    gpus=($(free_gpus))
    gi=0
    for model in $E2E_MODELS; do
        for conc in $E2E_CONCURRENCY_SWEEP; do
          for rep in $(seq 1 "${E2E_REPEATS:-1}"); do
            for arm in $E2E_ARMS; do
                [ -f "$(point_file "$model" "$conc" "$arm" "$rep")" ] && continue
                missing=$((missing + 1))
                case "$cov" in *" $model|$conc|$rep "*) continue ;; esac
                [ "$gi" -ge "${#gpus[@]}" ] && continue
                [ "$launches" -ge "$SV_MAX_LAUNCH" ] && continue
                g="${gpus[$gi]}"; gi=$((gi + 1))
                launches=$((launches + 1)); relaunched=$((relaunched + 1))
                record_failure "$model" "$conc" "$arm" "$rep"
                say "relaunch $(basename "$model") c=$conc $arm r$rep on Device${HIP2DEV[$g]} (hip $g)"
                ( cd "$E2E_ROOT" && \
                  RUN_DIR="$RUN_DIR" BENCH_GPU="$g" BENCH_PORT=$((8400 + g)) \
                  E2E_MODELS="$model" E2E_CONCURRENCY_SWEEP="$conc" E2E_ARMS="$arm" \
                  E2E_REP_LIST="$rep" E2E_GPU_MEM_UTIL="$SV_UTIL" E2E_SHARDS=8 \
                  nohup bash scripts/bench/run_e2e.sh \
                    > "$RUN_DIR/logs/sv_$(echo "$model" | tr '/' '_')_c${conc}_${arm}_r${rep}.log" 2>&1 & )
                # Let it claim the card before the next candidate looks at it.
                sleep 20
                cov="$cov $model|$conc|$rep "
            done
          done
        done
    done
    [ "$relaunched" -gt 0 ] && say "cycle: $missing missing, $relaunched relaunched"
    [ "$missing" -eq 0 ] && { say "all points present; supervisor exiting"; exit 0; }
    sleep "$SV_POLL_S"
done
