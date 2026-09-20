#!/usr/bin/env bash
# Progress of the running pipeline: processes, stage states, log tail, GPUs.
set -uo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"

E2E_CONTAINER="${E2E_CONTAINER:-}"

# Single point that knows whether a container is being addressed.
cexec() {
    if [ -n "$E2E_CONTAINER" ]; then
        podman exec "$E2E_CONTAINER" bash -lc "$1"
    else
        bash -lc "$1"
    fi
}

echo "== processes =="
# The second grep drops this listing's own pipeline, which carries the pattern.
cexec '
  found=0
  while read -r pid args; do
    [ -z "$pid" ] && continue
    printf "  %-8s %s\n" "$pid" "$(echo "$args" | cut -c1-70)"
    found=1
  done < <(ps -eo pid=,args= | grep -E "run_all\.sh|run_sharded\.sh|run_e2e\.sh|vllm serve|vllm bench" | grep -vE "ps -eo|cut -c")
  [ "$found" = "0" ] && echo "  (nothing running - possibly finished; see the stage states below)"
' 2>/dev/null

echo
echo "== stages =="
if [ -d "$E2E_ROOT/state/stages" ]; then
    for f in $(ls -1 "$E2E_ROOT/state/stages"/*.status 2>/dev/null | sort); do
        printf "  %-28s %s\n" "$(basename "$f" .status)" "$(cat "$f")"
    done
else
    echo "  (not started yet)"
fi

echo
echo "== tail of the current stage log =="
last=$(ls -1t "$E2E_ROOT/logs"/*.log 2>/dev/null | head -1)
[ -n "$last" ] && { echo "  ($last)"; tail -15 "$last" | sed 's/^/  /'; }

echo
echo "== GPU =="
cexec "
for g in ${E2E_GPUS:-0 1 2 3 4 5 6 7}; do
  v=\$(amd-smi metric -g \$g -m 2>/dev/null | awk '/USED_VRAM/{print \$2; exit}')
  a=\$(amd-smi metric -g \$g -u 2>/dev/null | awk '/GFX_ACTIVITY/{print \$2; exit}')
  printf '  GPU%s vram=%sMB gfx=%s%%\n' \"\$g\" \"\$v\" \"\$a\"
done" 2>/dev/null

echo
echo "== artifacts =="
[ -d "$E2E_ARTIFACTS" ] &&
    find "$E2E_ARTIFACTS" -maxdepth 3 -type d 2>/dev/null | head -10 | sed 's|^|  |'
echo "  plots: $(ls -1 "$E2E_ARTIFACTS"/runs/*/plots/*.png 2>/dev/null | wc -l)"
