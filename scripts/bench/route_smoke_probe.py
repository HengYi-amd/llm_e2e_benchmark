"""Routing probe: start vLLM, make Inductor autotune aten.mm, and emit the
selection log on stdout for scripts/bench/tally_route.py to count.

Must stay a standalone file, not a heredoc; see scripts/bench/route_smoke.sh.
"""
import os
import sys

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig


def _inductor_cfg():
    backends = os.environ.get("SMOKE_GEMM_BACKENDS", "ATEN,TRITON,FLYDSL")
    cfg = {
        "max_autotune": True,
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": backends,
        "max_autotune_gemm_search_space": os.environ.get(
            "SMOKE_AUTOTUNE_SEARCH_SPACE", "DEFAULT"),
        "flydsl_enable_autotuning": "FLYDSL" in backends,
    }
    # Combo kernels reach the compile-time autotune codegen path; switchable
    # so failures there can be bisected.
    if os.environ.get("SMOKE_NO_COMBO") == "1":
        cfg["combo_kernels"] = False
        cfg["benchmark_combo_kernel"] = False
    # Overrides the unconditional compile-time autotune of the standalone
    # compile path without disabling that path.
    if os.environ.get("SMOKE_NO_CT_AUTOTUNE") == "1":
        cfg["triton.autotune_at_compile_time"] = False
    return cfg


llm = LLM(
    model=sys.argv[1], dtype="bfloat16", tensor_parallel_size=1,
    max_model_len=2048, gpu_memory_utilization=0.30,
    max_num_batched_tokens=int(
        os.environ.get("SMOKE_MAX_NUM_BATCHED_TOKENS") or 8192),
    enable_prefix_caching=False, seed=0,
    compilation_config=CompilationConfig(
        mode=3, backend="inductor",
        # Short on purpose: the gate only needs static-shape graphs with a
        # candidate; each extra size costs another full autotune.
        compile_sizes=[1, 2, 4, 8, 16],
        cudagraph_mode="PIECEWISE",
        inductor_compile_config=_inductor_cfg(),
    ),
)
out = llm.generate(["Hello, my name is"] * 8, SamplingParams(max_tokens=16, temperature=0))
print("GENERATED:", out[0].outputs[0].text[:60])
