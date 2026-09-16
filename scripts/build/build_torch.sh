#!/usr/bin/env bash
# Optional: build PyTorch from source at the pinned commit.
#
# Only needed if the installed PyTorch predates the Inductor backend under test.
# Idempotent: exits early once the pinned build is importable.
set -Eeuo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"

PIN="${E2E_TORCH_PIN:-313e0fb9ff2ca1a073cb1cba3faf0f34ea27d8d8}"

if python -c "
import torch,sys
sys.exit(0 if torch.version.git_version=='$PIN' and torch.distributed.is_available() else 1)
" 2>/dev/null; then
    echo '[build_torch] already in place, skipping'; exit 0
fi

# Build the venv on the interpreter that owns the runtimes linked in below.
_base_python=/opt/venv/bin/python3.12
[ -x "$_base_python" ] || _base_python="$(command -v python3.12 || command -v python3)"
E2E_BASE_PYTHON="${E2E_BASE_PYTHON:-$_base_python}"

if [ ! -f "$E2E_VENV/bin/activate" ]; then
    echo "[build_torch] creating venv with $E2E_BASE_PYTHON"
    "$E2E_BASE_PYTHON" -m venv --system-site-packages "$E2E_VENV"
fi
# shellcheck disable=SC1091
. "$E2E_VENV/bin/activate"
python -V

# When the base interpreter is itself a venv, --system-site-packages inherits the
# site-packages it was created from, not the base venv's own, and the runtimes go
# missing silently. Symlink them rather than adding the whole directory, which
# would also shadow the torch built here.
BASE_SP="${E2E_BASE_SITE_PACKAGES:-$("$E2E_BASE_PYTHON" -c \
    'import sysconfig;print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)}"
OUR_SP="$(python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
if [ -n "$BASE_SP" ] && [ -d "$BASE_SP" ] && [ -d "$OUR_SP" ] \
   && [ "$BASE_SP" != "$OUR_SP" ]; then
    for n in triton triton_kernels flydsl flydsl.libs; do
        [ -e "$BASE_SP/$n" ] && ln -sfn "$BASE_SP/$n" "$OUR_SP/$n"
    done
    for d in "$BASE_SP"/triton-*.dist-info "$BASE_SP"/triton_kernels-*.dist-info \
             "$BASE_SP"/flydsl-*.dist-info; do
        [ -e "$d" ] && ln -sfn "$d" "$OUR_SP/$(basename "$d")"
    done
fi

if ! command -v ccache >/dev/null 2>&1; then
    echo '[build_torch] installing ccache'
    apt-get update -qq && apt-get install -y -qq ccache \
        || echo '[warn] no ccache, continuing (just slower)'
fi
command -v ccache >/dev/null 2>&1 && ccache -M "${E2E_CCACHE_SIZE:-80G}" && ccache -z

# cmake 4.x rejects the pre-3.5 cmake_minimum_required() still in this build.
python -m pip install -q "cmake==3.31.6" ninja

# --no-build-isolation installs no PEP 517 build requirements; these mirror the
# pinned tree's pyproject.toml and must stay in step with it.
python -m pip install -q "scikit-build-core>=1.0" "packaging>=24.2" \
    "typing-extensions>=4.10.0" numpy pyyaml six setuptools wheel
python -c "import scikit_build_core; print('scikit_build_core', scikit_build_core.__version__)"

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx950}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export BUILD_TEST="${BUILD_TEST:-0}"
export USE_KINETO="${USE_KINETO:-1}"
export USE_DISTRIBUTED="${USE_DISTRIBUTED:-1}"
export USE_GLOO="${USE_GLOO:-1}"
export USE_NCCL="${USE_NCCL:-1}"
export USE_SYSTEM_NCCL="${USE_SYSTEM_NCCL:-1}"
export NCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR:-$ROCM_PATH/include/rccl}"
export NCCL_LIB_DIR="${NCCL_LIB_DIR:-/usr/local/lib}"
export CMAKE_ARGS="${CMAKE_ARGS:--DCMAKE_EXE_LINKER_FLAGS=$ROCM_PATH/lib/libhsa-runtime64.so}"

# Some packagings ship an rccl cmake config naming a versioned librccl.so.* that
# is not where the library landed; recreate the name find_package(rccl) asks for.
RCCL_REAL=$(readlink -f "$NCCL_LIB_DIR/librccl.so.1" 2>/dev/null)
RCCL_WANT=$(grep -oE 'librccl\.so\.[0-9.]+' \
    "$ROCM_PATH/lib/cmake/rccl/rccl-targets-relwithdebinfo.cmake" 2>/dev/null | head -1)
if [ -n "$RCCL_REAL" ] && [ -n "$RCCL_WANT" ] && [ ! -e "$ROCM_PATH/lib/$RCCL_WANT" ]; then
    echo "[build_torch] linking $ROCM_PATH/lib/$RCCL_WANT -> $RCCL_REAL"
    ln -sf "$RCCL_REAL"  "$ROCM_PATH/lib/$RCCL_WANT"
    ln -sf "$RCCL_WANT"  "$ROCM_PATH/lib/librccl.so.1"
    ln -sf librccl.so.1  "$ROCM_PATH/lib/librccl.so"
fi

if [ ! -d "$E2E_TORCH_SRC/.git" ]; then
    echo "[build_torch] clone -> $E2E_TORCH_SRC"
    git clone https://github.com/pytorch/pytorch.git "$E2E_TORCH_SRC"
fi
cd "$E2E_TORCH_SRC"
git fetch --all --tags --prune 2>/dev/null || true
git checkout -f "$PIN"
git submodule update --init --recursive --jobs 16

echo "[build_torch] HEAD=$(git rev-parse HEAD)  version.txt=$(cat version.txt)"
[ "$(git rev-parse HEAD)" = "$PIN" ] || { echo "[FATAL] HEAD is not the pinned SHA"; exit 1; }

python tools/amd_build/build_amd.py

echo "[build_torch] building MAX_JOBS=$MAX_JOBS PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"
echo "[build_torch]   USE_DISTRIBUTED=$USE_DISTRIBUTED USE_NCCL=$USE_NCCL NCCL_INCLUDE_DIR=$NCCL_INCLUDE_DIR NCCL_LIB_DIR=$NCCL_LIB_DIR"
time pip install -e . -v --no-build-isolation

command -v ccache >/dev/null 2>&1 && ccache -s | head -8

# Self-check from outside the source tree so the checkout cannot satisfy the
# import instead of the installed build.
cd /
python - <<'EOF'
import pathlib, torch
print("torch      :", torch.__version__)
print("file       :", pathlib.Path(torch.__file__).resolve())
print("git_version:", torch.version.git_version)
print("hip        :", torch.version.hip)
print("distributed:", torch.distributed.is_available(), "nccl:", torch.distributed.is_nccl_available())
import torch._inductor.config as c
print("flydsl cfg :", hasattr(c, "flydsl_enable_autotuning"))
from torch._inductor.codegen.flydsl import flydsl_utils
print("flydsl rt  :", flydsl_utils.runtime_available())
EOF
echo '[build_torch] DONE'
