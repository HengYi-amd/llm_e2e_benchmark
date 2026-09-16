#!/usr/bin/env bash
# Add a VLLM_FORCE_ATEN_LINEAR switch to vLLM's dispatch_unquantized_gemm.
#
# The platform routes this op through a registered custom op the compiler cannot
# see through, so without the switch the linear layers never lower to aten.mm and
# no GEMM backend is ever a candidate. Both arms run with the switch on.
set -Eeuo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"
. "$E2E_VENV/bin/activate"
# Run from the repo so a checkout in the working directory cannot shadow the
# installed package of the same name.
cd "$E2E_ROOT"

# Patch the module Python actually imports; fall back to the source checkout.
TGT=$(python -c "import vllm.model_executor.layers.utils as u;print(u.__file__)" \
      2>/dev/null || true)
if [ -z "$TGT" ]; then
    TGT="$E2E_VLLM_SRC/vllm/model_executor/layers/utils.py"
    [ -f "$TGT" ] || { echo "[patch] cannot find vLLM: set E2E_VLLM_SRC" >&2; exit 1; }
fi
echo "[patch] target: $TGT"
mkdir -p "$E2E_ROOT/patches"
sha_before=$(sha256sum "$TGT" | cut -d' ' -f1)

if grep -q "VLLM_FORCE_ATEN_LINEAR" "$TGT"; then
    echo "[patch] already applied, skipping"
else
    cp "$TGT" "$E2E_ROOT/patches/layers_utils.orig.py"
    python - "$TGT" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:\n    if current_platform.is_rocm():"
new = ("def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:\n"
       "    # Bypass the platform custom-op wrapper so Inductor sees an aten.mm\n"
       "    # it can autotune.\n"
       "    import os as _os\n"
       "    if _os.environ.get('VLLM_FORCE_ATEN_LINEAR', '0') == '1':\n"
       "        return default_unquantized_gemm\n"
       "    if current_platform.is_rocm():")
assert old in s, "anchor not found - this vLLM version may have moved it"
open(p, "w").write(s.replace(old, new))
print("patched")
PY
fi

sha_after=$(sha256sum "$TGT" | cut -d' ' -f1)
cat > "$E2E_ROOT/patches/vllm_force_aten_linear.json" <<JSON
{"target": "$TGT", "sha256_before": "$sha_before", "sha256_after": "$sha_after",
 "env_switch": "VLLM_FORCE_ATEN_LINEAR"}
JSON

echo "[patch] checking the switch takes effect"
python - <<'PY'
import os
os.environ["VLLM_PLUGINS"] = ""
import vllm.model_executor.layers.utils as u
os.environ["VLLM_FORCE_ATEN_LINEAR"] = "1"
assert u.dispatch_unquantized_gemm() is u.default_unquantized_gemm, \
    "switch had no effect"
os.environ["VLLM_FORCE_ATEN_LINEAR"] = "0"
print("OK: switch works as expected")
PY
