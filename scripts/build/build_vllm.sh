#!/usr/bin/env bash
# Optional: rebuild vLLM against a locally built PyTorch.
#
# Only needed after scripts/build/build_torch.sh. Uses vLLM's own
# use_existing_torch.py so the build never pulls a torch wheel of its own.
set -Eeuo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"
# shellcheck disable=SC1091
. "$E2E_VENV/bin/activate"

VLLM_PIN="${E2E_VLLM_PIN:-6e448d0ea9}"
TORCH_PIN="${E2E_TORCH_PIN:-313e0fb9ff2ca1a073cb1cba3faf0f34ea27d8d8}"

# Idempotent: vllm lives in the venv and torch stays the locally built one.
if python -c "
import pathlib,vllm,torch,sys
ok = '$E2E_VENV' in pathlib.Path(vllm.__file__).resolve().as_posix() \
     and torch.version.git_version=='$TORCH_PIN'
sys.exit(0 if ok else 1)" 2>/dev/null; then
    echo '[build_vllm] already in place, skipping'; exit 0
fi

TORCH_BEFORE=$(python -c "import torch;print(torch.version.git_version)")

# A vLLM checkout shipped with the environment, if any; copying it saves a clone.
VLLM_BASE_SRC="${E2E_VLLM_BASE_SRC:-/app/vllm}"

if [ ! -d "$E2E_VLLM_SRC/.git" ]; then
    if [ -d "$VLLM_BASE_SRC/.git" ]; then
        echo "[build_vllm] copying source from $VLLM_BASE_SRC"
        cp -a "$VLLM_BASE_SRC" "$E2E_VLLM_SRC"
    else
        echo "[build_vllm] clone -> $E2E_VLLM_SRC"
        git clone https://github.com/vllm-project/vllm.git "$E2E_VLLM_SRC"
    fi
fi
cd "$E2E_VLLM_SRC"
git checkout -f "$VLLM_PIN" 2>/dev/null || echo "[warn] checkout $VLLM_PIN failed, using current HEAD"
git reset --hard HEAD

# A copied checkout carries CMakeCache.txt and *-subbuild trees with the original
# source path baked in, and git reset leaves them (untracked). Keep *-src, which is
# downloaded source and path-independent, and drop the rest.
rm -rf build .deps/*-subbuild .deps/*-build CMakeCache.txt CMakeFiles
echo "[build_vllm] HEAD=$(git rev-parse HEAD)"

# Strip the torch/torchvision/torchaudio pins from requirements and pyproject.
python use_existing_torch.py

# That script cannot reach third-party metadata: timm pulls torchvision, which
# pins an exact torch, and the resolver then replaces the local torch with a
# released build plus CUDA wheels. Text-model benchmarks do not need timm.
sed -i -E '/^\s*(timm|torchvision|torchaudio)([<>=~!]|$)/d' requirements/*.txt
echo "[build_vllm] remaining timm/torchvision lines: $(grep -rcE '^\s*(timm|torchvision|torchaudio)' requirements/*.txt 2>/dev/null | grep -v ':0' | wc -l)"

export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-rocm}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx950}"

# FetchContent clones a large repository over the network and fails often enough
# to be worth avoiding. Reuse a local clone only when the commit matches the pin,
# so the version is still the one vLLM expects.
TK_PIN="${E2E_TRITON_KERNELS_PIN:-0f380657dbf3ee86eb57558ff71df24f03b5d4e7}"
TK_LOCAL="$E2E_VLLM_SRC/.deps/triton_kernels-src"
TK_BASE="$VLLM_BASE_SRC/.deps/triton_kernels-src"
TK_INNER=python/triton_kernels/triton_kernels
if [ ! -d "$TK_LOCAL/$TK_INNER" ] \
   && [ "$(git -C "$TK_BASE" rev-parse HEAD 2>/dev/null)" = "$TK_PIN" ]; then
    echo "[build_vllm] reusing triton_kernels @ $TK_PIN from $TK_BASE"
    rm -rf "$TK_LOCAL"; mkdir -p "$(dirname "$TK_LOCAL")"
    cp -a "$TK_BASE" "$TK_LOCAL"
fi
if [ -d "$TK_LOCAL/$TK_INNER" ]; then
    export TRITON_KERNELS_SRC_DIR="$TK_LOCAL/$TK_INNER"
    echo "[build_vllm] TRITON_KERNELS_SRC_DIR=$TRITON_KERNELS_SRC_DIR"
else
    echo "[warn] no local triton_kernels, falling back to FetchContent over the network"
fi

# An editable install redirects torch.__file__ to the checkout, so
# torch.utils.cmake_prefix_path (what vLLM's CMakeLists uses) points at a
# share/cmake that does not exist. Hand cmake the real location instead.
TORCH_CMAKE=$(python -c "
import sysconfig, pathlib
p = pathlib.Path(sysconfig.get_paths()['purelib']) / 'torch' / 'share' / 'cmake'
print(p if (p / 'Torch' / 'TorchConfig.cmake').exists() else '')")
[ -n "$TORCH_CMAKE" ] || { echo "[FATAL] no TorchConfig.cmake, cannot build vLLM"; exit 1; }
echo "[build_vllm] Torch cmake prefix: $TORCH_CMAKE"
export CMAKE_PREFIX_PATH="$TORCH_CMAKE${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export Torch_DIR="$TORCH_CMAKE/Torch"

# Abort before anything is downloaded if the resolution plan would replace torch
# or add CUDA wheels. The pattern is deliberately general, not timm-specific.
echo "[build_vllm] dry-run dependency resolution"
DRY=$(mktemp -p "${TMPDIR:-/tmp}")
if pip install -e . --no-build-isolation --dry-run > "$DRY" 2>&1; then
    if grep -qE "Would install .*(torch-[0-9]|torchvision-[0-9]|nvidia-|cuda-toolkit)" "$DRY"; then
        echo "[FATAL] dry-run would replace torch or add CUDA wheels, aborting:"
        grep -oE "(torch-[0-9][^ ]*|torchvision-[0-9][^ ]*|nvidia-[^ ]*|cuda-toolkit-[^ ]*)" "$DRY" | sort -u | sed 's/^/    /'
        rm -f "$DRY"; exit 1
    fi
    echo "[build_vllm] dry-run clean, torch will not be replaced"
else
    echo "[warn] dry-run itself failed (not necessarily fatal), tail:"; tail -20 "$DRY"
fi
rm -f "$DRY"

echo "[build_vllm] building"
time pip install -e . --no-build-isolation -v

TORCH_AFTER=$(python -c "import torch;print(torch.version.git_version)")
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
    echo "[FATAL] the vLLM install replaced torch: $TORCH_BEFORE -> $TORCH_AFTER"; exit 1
fi
echo '[build_vllm] DONE'
