#!/usr/bin/env bash
# Shared environment. Sourced by every script; override anything from the
# environment before sourcing.

# Resolve the repo root from this file's location so scripts work from any
# working directory and from inside or outside a container.
if [ -z "${E2E_ROOT:-}" ]; then
    _env_src="${BASH_SOURCE[0]:-$0}"
    E2E_ROOT="$(cd "$(dirname "$_env_src")" && pwd -P)"
fi
export E2E_ROOT

# Only needed when building from source (scripts/build/); a stock install works.
export E2E_VLLM_SRC="${E2E_VLLM_SRC:-$(dirname "$E2E_ROOT")/vllm_e2e}"

# One environment per numeric format. bf16 was measured against a PyTorch at a
# specific commit; the mxfp work needs an unmerged PR branch whose Inductor
# internals that commit does not have. Sharing one venv would make the published
# bf16 run unreproducible the moment the mxfp branch is installed.
if [ "${E2E_PRECISION:-bf16}" = "bf16" ]; then
    export E2E_VENV="${E2E_VENV:-$E2E_ROOT/.venv-e2e}"
else
    export E2E_VENV="${E2E_VENV:-$E2E_ROOT/.venv-mxfp}"
fi
# Source checkout matching the venv above.
if [ "${E2E_PRECISION:-bf16}" = "bf16" ]; then
    export E2E_TORCH_SRC="${E2E_TORCH_SRC:-$(dirname "$E2E_ROOT")/pytorch_e2e}"
else
    export E2E_TORCH_SRC="${E2E_TORCH_SRC:-$(dirname "$E2E_ROOT")/pytorch_mxfp}"
fi

# Everything a run writes stays inside the repo except TMPDIR, which must be on
# a local filesystem: compile workers do heavy small-file I/O and a network
# home directory makes compilation crawl. stop.sh removes it.
export E2E_ARTIFACTS="${E2E_ARTIFACTS:-$E2E_ROOT/runs}"
# Deliverables: one flat directory per (model, dtype).
# One deliverable tree per numeric format: bf16_result, mxfp8_result,
# mxfp4_result. Keeping them apart means a new format cannot overwrite a
# published one, and the format is visible in the path.
export E2E_RESULT="${E2E_RESULT:-$E2E_ROOT/${E2E_RESULT_ROOT:-${E2E_PRECISION:-bf16}_result}}"
export TMPDIR="${E2E_TMPDIR:-/var/tmp/llm_e2e_bench}"

# Weights kept inside the repo so a run is self-contained; a shared cache can
# have a model evicted from under a sweep.
export HF_HOME="${HF_HOME:-$E2E_ROOT/models}"
export HF_HUB_DISABLE_TELEMETRY=1

# Caches must be isolated per arm: a shared Inductor cache lets the second arm
# hit the first arm's artifacts, so autotune never runs and the measured
# difference is a cache hit. ARM_TAG is set by the runner.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$E2E_ROOT/caches/inductor/${ARM_TAG:-default}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$E2E_ROOT/caches/triton/${ARM_TAG:-default}}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$E2E_ROOT/caches/vllm/${ARM_TAG:-default}}"
export FLYDSL_RUNTIME_CACHE_DIR="${FLYDSL_RUNTIME_CACHE_DIR:-$E2E_ROOT/caches/flydsl/${ARM_TAG:-default}}"

mkdir -p "$TMPDIR" "$E2E_ARTIFACTS" "$E2E_RESULT" "$E2E_ROOT/logs" "$E2E_ROOT/state" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$FLYDSL_RUNTIME_CACHE_DIR" "$VLLM_CACHE_ROOT" 2>/dev/null || true

# Keep CWD off sys.path: a neighbouring checkout can shadow the installed
# package of the same name, and the failure surfaces as a model-loading error.
export PYTHONSAFEPATH=1

# Config: file first, environment wins. Exported, because the provenance each
# result carries is read from the environment by child processes; unexported,
# every result would record an empty configuration.
# shellcheck disable=SC1091
if [ -f "$E2E_ROOT/config/default.env" ]; then
    set -a
    . "$E2E_ROOT/config/default.env"
    set +a
fi

# KV capacity pin produced by scripts/bench/calibrate_kv.sh. Written as an if
# rather than a `&&` list: a trailing test that fails makes sourcing this file
# return non-zero, which silently aborts any caller running under `set -e`.
# shellcheck disable=SC1091
# Pins live under a per-format namespace: a KV capacity calibrated for mxfp4
# weights is meaningless for bf16, and a flat directory let one overwrite the
# other, which would silently make an already-published run unreproducible.
export E2E_KV_PIN_DIR="$E2E_ROOT/state/kv_pin.d/${E2E_PRECISION:-bf16}"
mkdir -p "$E2E_KV_PIN_DIR" 2>/dev/null || true
for _kv in "$E2E_KV_PIN_DIR"/*.env; do
    [ -f "$_kv" ] && . "$_kv"
done
unset _kv
:
