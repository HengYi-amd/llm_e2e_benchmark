#!/usr/bin/env python
"""Gate on the software stack; run after installing anything.

These mismatches fail late and look like benchmark noise, so they are checked
once up front and recorded in state/verify_stack.json.
"""
import json
import os
import pathlib
import sys

ROOT = os.environ.get("E2E_ROOT") or str(pathlib.Path(__file__).resolve().parents[2])
VENV = os.environ.get("E2E_VENV") or os.path.join(ROOT, ".venv-e2e")
TORCH_SRC = os.environ.get("E2E_TORCH_SRC") or os.path.join(
    os.path.dirname(ROOT), "pytorch_e2e")
VLLM_SRC = os.environ.get("E2E_VLLM_SRC") or os.path.join(
    os.path.dirname(ROOT), "vllm_e2e")

# Pinned torch commit and expected GPU arch; override both from the environment.
PIN = os.environ.get("E2E_TORCH_GIT_PIN", "313e0fb9ff2ca1a073cb1cba3faf0f34ea27d8d8")
ARCH = os.environ.get("E2E_GPU_ARCH", "gfx950")

fails, info = [], {}


def check(name, cond, detail=""):
    info[name] = {"ok": bool(cond), "detail": str(detail)}
    print(f"{'OK  ' if cond else 'FAIL'}  {name}: {detail}")
    if not cond:
        fails.append(name)


import torch  # noqa: E402

tf = pathlib.Path(torch.__file__).resolve().as_posix()
check("torch.version", torch.version.git_version == PIN, torch.version.git_version)
# A stray torch wheel on the path carries no backend under test, so the arm
# silently falls back and the comparison measures nothing.
check("torch.from_our_build", TORCH_SRC in tf or VENV in tf, tf)
check("torch.hip", torch.version.hip is not None, torch.version.hip)
check("torch.distributed", torch.distributed.is_available(), "is_available")
check("torch.nccl", torch.distributed.is_nccl_available(), "is_nccl_available")

import torch._inductor.config as c  # noqa: E402

check("inductor.flydsl_config", hasattr(c, "flydsl_enable_autotuning"),
      getattr(c, "flydsl_enable_autotuning", "MISSING"))
try:
    from torch._inductor.codegen.flydsl import flydsl_utils
    check("flydsl.runtime_available", flydsl_utils.runtime_available(),
          flydsl_utils._flydsl_runtime_unavailable_reason() or "ready")
except Exception as e:  # noqa: BLE001
    check("flydsl.runtime_available", False, repr(e))

try:
    import flydsl
    check("flydsl.version_0_3_x", flydsl.__version__.startswith("0.3."), flydsl.__version__)
except Exception as e:  # noqa: BLE001
    check("flydsl.version_0_3_x", False, repr(e))

try:
    import triton
    check("triton.3_8_x", triton.__version__.startswith("3.8."), triton.__version__)
except Exception as e:  # noqa: BLE001
    check("triton.3_8_x", False, repr(e))

if "--with-vllm" in sys.argv:
    try:
        import vllm
        vf = pathlib.Path(vllm.__file__).resolve().as_posix()
        check("vllm.from_our_venv", VENV in vf or VLLM_SRC in vf,
              f"{vllm.__version__} {vf}")
    except Exception as e:  # noqa: BLE001
        check("vllm.import", False, repr(e))

check("cuda.available", torch.cuda.is_available(), f"{torch.cuda.device_count()} devices")
if torch.cuda.is_available():
    arch = torch.cuda.get_device_properties(0).gcnArchName
    check(f"arch.{ARCH}", arch.split(":")[0] == ARCH, arch)

out = pathlib.Path(ROOT) / "state" / "verify_stack.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(info, indent=2))

print("\n" + ("STACK OK" if not fails else f"STACK FAILED: {fails}"))
sys.exit(1 if fails else 0)
