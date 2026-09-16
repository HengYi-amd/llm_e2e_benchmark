#!/usr/bin/env bash
# Optional: build torchvision from source against a locally built PyTorch.
#
# Only needed alongside scripts/build/build_torch.sh. It is a hard runtime
# dependency of vLLM's warmup path and must be built with --no-deps: its pip
# metadata pins an exact torch and would replace the local build, while a
# prebuilt binary compiled against another torch fails at op registration.
set -Eeuo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"
# shellcheck disable=SC1091
. "$E2E_VENV/bin/activate"

# Sibling checkout by default, like E2E_TORCH_SRC / E2E_VLLM_SRC.
E2E_VISION_SRC="${E2E_VISION_SRC:-$(dirname "$E2E_ROOT")/vision_e2e}"

if python -c "from torchvision.transforms import InterpolationMode" 2>/dev/null; then
    echo "[tv] already usable: $(python -c 'import torchvision;print(torchvision.__version__)')"
    exit 0
fi

[ -d "$E2E_VISION_SRC/.git" ] \
    || git clone --depth 1 -b main https://github.com/pytorch/vision "$E2E_VISION_SRC"
cd "$E2E_VISION_SRC"
echo "[tv] commit=$(git rev-parse --short HEAD)"

TORCH_BEFORE=$(python -c "import torch;print(torch.__version__)")
# CPU ops are enough: the operators only need to be registered, not executed.
FORCE_CUDA=0 pip install --no-deps --no-build-isolation .
TORCH_AFTER=$(python -c "import torch;print(torch.__version__)")
[ "$TORCH_BEFORE" = "$TORCH_AFTER" ] \
    || { echo "[FATAL] torch was replaced: $TORCH_BEFORE -> $TORCH_AFTER"; exit 1; }

cd "$E2E_ROOT"
python -c "
from torchvision.transforms import InterpolationMode
import torchvision, torch
print('[tv] OK', torchvision.__version__, '| torch', torch.__version__)
"
echo "[tv] DONE"
