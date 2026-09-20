#!/usr/bin/env bash
# Fan the sweep out across the node's GPUs, claiming cards as they free up.
#
# One shard = one (model, dtype, profile, repeat, concurrency), run as an
# independent single-GPU job. Both arms of a shard run back to back on the SAME
# card: splitting a pair across cards lets clock and thermal differences look
# like an arm difference.
#
# Cards are acquired opportunistically rather than assigned up front. The node is
# shared, so a card a colleague is using is simply skipped; when their job ends
# and the card drains, the next shard takes it. That reaches full occupancy
# without signalling anyone else's work, and it degrades gracefully when it
# cannot - fewer cards means a longer run, not a failed one.
set -Eeuo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"
cd "$E2E_ROOT"

RUN_DIR="${RUN_DIR:-$E2E_ARTIFACTS/$(cat "$E2E_ROOT/state/current_run_id")}"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/raw/e2e"
export RUN_DIR

: "${E2E_CLAIM_POLL_S:=30}"
: "${E2E_CLAIM_MIN_FREE_MIB:=268000}"
# Cards this run may ever use, as HIP indices; empty means every GPU present.
CANDIDATES="${E2E_GPUS:-}"
if [ -z "$CANDIDATES" ]; then
    CANDIDATES=$(bash "$E2E_ROOT/scripts/bench/gpumap.sh" | awk '{print $1}' | tr '\n' ' ')
fi

# hip index -> rocm-smi device number, so free memory can be read per card.
declare -A HIP2DEV
while read -r hip dev _; do HIP2DEV["$hip"]="$dev"; done \
    < <(bash "$E2E_ROOT/scripts/bench/gpumap.sh")

free_mib() {
    local dev="${HIP2DEV[$1]:-}"
    [ -n "$dev" ] || { echo 0; return; }
    rocm-smi --showmeminfo vram 2>/dev/null \
        | sed 's/\x1b\[[0-9;]*m//g' \
        | grep -oE "GPU\[$dev\].*Used Memory \(B\): [0-9]+" \
        | grep -oE '[0-9]+$' | tail -1 \
        | awk '{printf "%d", (309220868096 - $1) / 1048576}'
}

declare -A BUSY          # hip index -> pid of the shard holding it
release_finished() {
    local hip pid
    for hip in "${!BUSY[@]}"; do
        pid="${BUSY[$hip]}"
        kill -0 "$pid" 2>/dev/null || unset 'BUSY[$hip]'
    done
}

# GUIDs of cards another user currently has queues on. Free memory alone is not
# enough of a test: a neighbour running small kernels leaves most of the card
# free, and taking it would make their next allocation fail. Claim only cards
# nobody else is on.
foreign_guids() {
    local p c out=""
    for p in $(ls /sys/class/kfd/kfd/proc/ 2>/dev/null); do
        c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
        case "$c" in *"$E2E_ROOT"*) continue ;; esac
        case "$(readlink "/proc/$p/cwd" 2>/dev/null)" in
            "$E2E_ROOT"|"$E2E_ROOT"/*) continue ;;
        esac
        tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null \
            | grep -q "^TORCHINDUCTOR_CACHE_DIR=$E2E_ROOT" && continue
        for q in /sys/class/kfd/kfd/proc/"$p"/queues/*/; do
            [ -f "$q/gpuid" ] && out="$out $(cat "$q/gpuid")"
        done
    done
    echo "$out"
}

declare -A HIP2GUID
while read -r hip _ guid; do HIP2GUID["$hip"]="$guid"; done \
    < <(bash "$E2E_ROOT/scripts/bench/gpumap.sh")

# A card is claimable when no shard of ours holds it, nobody else has queues on
# it, and it has drained enough for the engine to start.
acquire() {
    local hip f busy_guids waited=0 polite=1
    while : ; do
        release_finished
        busy_guids=" $(foreign_guids) "
        for hip in $CANDIDATES; do
            [ -n "${BUSY[$hip]:-}" ] && continue
            if [ "$polite" = 1 ]; then
                case "$busy_guids" in *" ${HIP2GUID[$hip]:-x} "*) continue ;; esac
            fi
            f=$(free_mib "$hip")
            if [ "${f:-0}" -ge "$E2E_CLAIM_MIN_FREE_MIB" ]; then
                echo "$hip"
                return 0
            fi
        done
        sleep "$E2E_CLAIM_POLL_S"
        waited=$((waited + E2E_CLAIM_POLL_S))
        # Yielding to a neighbour is the right default, but yielding forever is
        # not: a night spent waiting produces nothing. After the grace period the
        # memory test alone decides, which still cannot evict anyone - it only
        # stops treating a card as reserved because someone ran a kernel on it.
        if [ "$polite" = 1 ] && [ "$waited" -ge "${E2E_CLAIM_POLITE_S:-1800}" ]; then
            echo "[shard] no free card for ${waited}s; claiming on free memory alone" >&2
            polite=0
        fi
    done
}

SHARD_BY_CONC="${E2E_SHARD_BY_CONCURRENCY:-1}"
SHARDS=()
for model in $E2E_MODELS; do
  for dtype in $E2E_DTYPES; do
    for prof in $E2E_PROFILES; do
      for rep in $(seq 1 "$E2E_REPEATS"); do
        if [ "$SHARD_BY_CONC" = 1 ]; then
          for c in $E2E_CONCURRENCY_SWEEP; do
            SHARDS+=("$model|$dtype|$prof|$rep|$c")
          done
        else
          SHARDS+=("$model|$dtype|$prof|$rep|$E2E_CONCURRENCY_SWEEP")
        fi
      done
    done
  done
done

devs=""
for h in $CANDIDATES; do devs="$devs Device${HIP2DEV[$h]:-?}"; done
echo "[shard] ${#SHARDS[@]} shards; candidate cards:$devs (claimed as they free up)"

pids=()
for s in "${SHARDS[@]}"; do
    IFS='|' read -r model dtype prof rep conc <<< "$s"
    gpu="$(acquire)"
    ctag="$(echo "$conc" | tr ' ' '-')"
    tag="$(echo "$model" | tr '/' '_')_${dtype}_${prof}_r${rep}_c${ctag}"
    echo "[shard] claimed Device${HIP2DEV[$gpu]:-?} (hip $gpu) <- $tag"
    env BENCH_GPU="$gpu" BENCH_PORT=$((8300 + gpu)) \
        E2E_MODELS="$model" E2E_DTYPES="$dtype" E2E_PROFILES="$prof" \
        E2E_CONCURRENCY_SWEEP="$conc" \
        E2E_REP_LIST="$rep" E2E_SHARDS="${#CANDIDATES}" \
        bash "$E2E_ROOT/scripts/bench/run_e2e.sh" \
        > "$RUN_DIR/logs/shard_gpu${gpu}_${tag}.log" 2>&1 &
    BUSY[$gpu]=$!
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[shard] all shards finished rc=$rc, $(ls -1 "$RUN_DIR/raw/e2e"/*.json 2>/dev/null | wc -l) points"
exit $rc
