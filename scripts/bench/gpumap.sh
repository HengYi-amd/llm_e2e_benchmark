#!/usr/bin/env bash
# Emit "hip_index rocm_device guid" for each GPU, derived at run time.
#
# HIP_VISIBLE_DEVICES indexes devices in kfd node order while rocm-smi numbers
# them in its own order, and on this class of machine the two disagree. Excluding
# "the card a colleague is on" needs the real mapping, not the assumption that
# both orderings start at zero and agree.
set -uo pipefail

declare -A GUID2DEV
while read -r dev guid; do
    [ -n "${guid:-}" ] && GUID2DEV["$guid"]="$dev"
done < <(rocm-smi 2>/dev/null \
         | sed 's/\x1b\[[0-9;]*m//g' \
         | awk '/^[0-9]+[[:space:]]/{gsub(",","",$3); print $1, $4}')

hip=0
for n in $(ls /sys/class/kfd/kfd/topology/nodes/ 2>/dev/null | sort -n); do
    p="/sys/class/kfd/kfd/topology/nodes/$n/properties"
    [ -f "$p" ] || continue
    sc=$(awk '/^simd_count /{print $2; exit}' "$p")
    [ "${sc:-0}" = "0" ] && continue          # CPU node, not a GPU
    guid=$(cat "/sys/class/kfd/kfd/topology/nodes/$n/gpu_id" 2>/dev/null)
    echo "$hip ${GUID2DEV[$guid]:-?} ${guid:-?}"
    hip=$((hip + 1))
done
