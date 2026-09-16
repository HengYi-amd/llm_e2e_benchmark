# llm_e2e_benchmark

An A/B harness for TorchInductor GEMM backends, measured end to end through
vLLM.

It answers one question: **when a GEMM backend is added to Inductor's autotune
candidate list, what happens to a real model's serving latency?** Two arms
differ in exactly one setting — the backend list — and everything else is held
identical, so any difference is attributable to that change.

The harness is backend-, model- and dtype-agnostic. The shipped defaults target
the FlyDSL backend at bf16; see [Extending it](#extending-it).

---

## Contents

- [What it measures](#what-it-measures)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Pipeline stages](#pipeline-stages)
- [Design decisions](#design-decisions)
- [Controls the harness enforces](#controls-the-harness-enforces)
- [Extending it](#extending-it)
- [Interpreting the output](#interpreting-the-output)

## What it measures

| arm | `max_autotune_gemm_backends` |
|---|---|
| `baseline` | `ATEN,TRITON` |
| `treatment` | `ATEN,TRITON,FLYDSL` |

Three metrics, collected with `vllm bench serve`:

| metric | isolates | role |
|---|---|---|
| **TPOT** — time per output token | decode steady state; GEMM M = concurrency | headline |
| **TTFT** — time to first token | prefill; large-M GEMMs | supporting |
| **output throughput** — tokens/s | serving capacity | supporting |

TTFT and TPOT are reported separately rather than as one fused latency. A GEMM
backend moves the two phases by different factors: decode GEMMs are skinny and
memory-bound, prefill GEMMs are large-M and compute-bound. A fused wall-clock
number blends them, hides which one moved, and can be shifted by changing the
output length alone.

## Requirements

- A GPU with a working PyTorch and vLLM install. The harness drives vLLM
  through its CLI; it does not import it.
- A Python environment containing both, reachable via `E2E_VENV`.
- `curl` for server readiness probing, and a GPU query tool (`amd-smi` or
  `rocm-smi`) for the preflight checks; a missing query tool degrades to a
  warning rather than a failure.

`scripts/build/` holds optional source-build helpers if a specific revision of
PyTorch or vLLM is needed.

## Quick start

```bash
git clone https://github.com/HengYi-amd/llm_e2e_benchmark.git
cd llm_e2e_benchmark

export E2E_VENV=/path/to/python/env      # required
export E2E_VLLM_SRC=/path/to/vllm        # for the source patch step
export E2E_TORCH_SRC=/path/to/pytorch    # for the build helpers

bash scripts/run_all.sh
```

Narrow the sweep without editing anything:

```bash
E2E_MODELS="<hf-model-id>" \
E2E_PROFILES=blog \
E2E_CONCURRENCY_SWEEP="1 8 64" \
  bash scripts/run_all.sh
```

Re-run only post-processing against an existing run:

```bash
E2E_ONLY="07 08 09" bash scripts/run_all.sh
```

Run detached, and stop safely:

```bash
bash scripts/daemon/start.sh
bash scripts/daemon/status.sh
bash scripts/daemon/stop.sh
```

`stop.sh` verifies the recorded process still belongs to this run before
signalling, kills the process tree rather than the process group, removes the
temporary directory, and re-checks every GPU afterwards. On a shared machine a
stop that leaves a server holding memory, or that kills an unrelated job, is
worse than no stop at all.

Set `E2E_CONTAINER` when the GPU stack lives in a container and the daemon is
driven from outside it; leave it empty to run everything directly.

## Repository layout

```text
config/default.env          all tunables and their defaults
env.sh                      paths and cache isolation; sourced by every script

scripts/
  run_all.sh                the pipeline; stages are resumable
  preflight/                GPU availability checks
  setup/
    verify_stack.py         assert the installed versions match expectations
    patch_vllm.sh           add the switch that exposes GEMM to the compiler
  bench/
    route_smoke.sh          gate: prove the backend is reachable
    route_smoke_probe.py    the probe the gate runs
    calibrate_kv.sh         pin KV capacity equal across arms
    run_e2e.sh              one shard: a server per arm, swept over concurrency
    run_sharded.sh          fan shards across GPUs
    tally_route.py          parse autotune selections from a log
  proof/route_evidence.py   did the backend actually execute?
  normalize/normalize.py    raw results into tidy CSVs, with integrity checks
  plot/plot.py              speedup bars and absolute-value lines
  report/report.py          assemble REPORT.md
  daemon/                   start / status / stop
  stop.sh                   process-tree kill used by daemon/stop.sh

patches/flydsl_raw_args.patch   optional compiler-side fix, see below
```

Output goes to `artifacts/runs/<run_id>/`, which is not tracked:

```text
raw/e2e/*.json      one file per point, carrying its full workload description
normalized/*.csv    per-point and per-cell tables
figures/*.png       each with a JSON sidecar
report/REPORT.md
manifest/run.json   environment fingerprint
logs/
```

## Configuration

Every value in [`config/default.env`](config/default.env) can be overridden
from the environment.

| knob | default | note |
|---|---|---|
| `E2E_MODELS` | two dense models | dense only; MoE changes which GEMMs exist |
| `E2E_DTYPES` | `bfloat16` | |
| `E2E_ARMS` | `baseline treatment` | backend lists in `E2E_BACKENDS_<arm>` |
| `E2E_CONCURRENCY_SWEEP` | `1 … 128` | becomes the decode GEMM's M |
| `E2E_PROFILES` | `chunked blog` | workload shapes, below |
| `E2E_TP` | `1` | |
| `E2E_MAX_NUM_BATCHED_TOKENS` | `8192` | pinned, not defaulted |
| `E2E_MAX_MODEL_LEN` | `4096` | |
| `E2E_GPU_MEM_UTIL` | `0.85` | |
| `E2E_AUTOTUNE_SEARCH_SPACE` | `DEFAULT` | |
| `E2E_FLYDSL_AUTOTUNING` | `1` | required; see below |
| `E2E_REPEATS` | `2` | ABBA-interleaved |
| `E2E_GPUS` | `0 … 7` | shards, not tensor parallelism |
| `E2E_CONTAINER` | empty | run directly, or wrap in a container |

## Pipeline stages

| stage | does |
|---|---|
| `00_preflight` | GPUs present and free |
| `01_verify` | installed versions match expectations |
| `02_patch_vllm` | add the `VLLM_FORCE_ATEN_LINEAR` switch |
| `03_route_smoke` | gate: the backend produces candidates at all |
| `04_kv_pin` | calibrate KV capacity, pin it equal across arms |
| `05_e2e` | the sweep, sharded across GPUs |
| `06_route_proof` | confirm the backend executed at runtime |
| `07_normalize` | tidy tables, integrity checks |
| `08_plot` | figures |
| `09_report` | `REPORT.md` |

A stage already marked passed is skipped, so a re-run resumes rather than
recompiling from scratch.

## Design decisions

### Workload profiles

**ISL** (input sequence length) sets the prefill work; **OSL** (output sequence
length) sets the number of decode steps.

vLLM schedules prefill in chunks of at most `max_num_batched_tokens`. That
default differs between the offline and server code paths, so the harness pins
it: a benchmark whose chunk size depends on which entry point was used is not
reproducible.

| profile | ISL | OSL | purpose |
|---|---|---|---|
| `chunked` | 2048 | 128 | prefill reaches a full chunk, so TTFT reflects real prefill GEMMs |
| `blog` | 32 | 128 | short-prompt reference shape; one prefill step against 127 decode steps, so read it as decode-only |

Do not draw a prefill conclusion from the `blog` profile: at that prompt length
TTFT is dominated by scheduling overhead rather than GEMM time. `plot.py` omits
that combination for this reason.

### Tensor parallelism

`E2E_TP=1`. Sharding divides each Linear's N or K by the TP size, so the backend
would be tested on GEMMs a fraction of their real width, and each decoder layer
would add collective calls to the decode critical path — time both arms pay
equally, which only dilutes the signal.

TP=1 does not mean one GPU is used. `run_sharded.sh` occupies every configured
GPU by running independent single-GPU shards. Both arms of a cell run on the
same GPU, because the arm ratio is the result and inter-card clock or thermal
spread would otherwise look like a backend difference.

### Autotune breadth

`E2E_AUTOTUNE_SEARCH_SPACE=DEFAULT`. The exhaustive space turns each GEMM from
a few dozen benchmarked candidates into thousands, costing hours per model, and
on some platforms it bypasses the shipped selection heuristic — measuring a
path that is not the one users get.

`E2E_FLYDSL_AUTOTUNING=1` is required. That setting defaults to off upstream,
and with it off the backend contributes a single hardcoded configuration
against a tuned competitor, which does not compare backends at all.

`TORCHINDUCTOR_ORIGAMI` is pinned and recorded because its default is on and
invisible: where its supporting package is installed it changes which Triton
candidates exist, producing two very different arms from identical command
lines.

### The compiler-visibility patch

On ROCm, vLLM does not route unquantized GEMM through `aten.mm`. It dispatches
to a registered custom op wrapping a hand-written kernel selection tree.
**Inductor cannot see through a custom op**, so on the stock path there is no
`aten.mm` to autotune and no GEMM backend is ever a candidate.

[`scripts/setup/patch_vllm.sh`](scripts/setup/patch_vllm.sh) adds one
environment switch, `VLLM_FORCE_ATEN_LINEAR`, selecting the plain linear path
instead. **Both arms set it.** It is not part of the A/B — it is the
precondition that makes the A/B possible.

This bounds what the result claims: the comparison is **conditional** on the
Inductor autotune path being in use. It is not a statement about the engine's
default path on that platform, which uses the vendor kernels instead. Where the
default dispatch already is `aten.mm`, no patch is needed.

A second, compiler-side patch is in
[`patches/flydsl_raw_args.patch`](patches/flydsl_raw_args.patch). It lets the
template kernel pass raw arguments through the kernel-call generator, which
compile-time autotuning needs in order to build example tensors for a
candidate. Apply it only if the installed PyTorch lacks it.

## Controls the harness enforces

| control | without it |
|---|---|
| Per-arm compile caches | the second arm loads the first arm's artifacts, skips autotune, and reports a cache hit as a result |
| KV capacity pinned equal | the arm granted more KV blocks wins on throughput for reasons unrelated to GEMMs |
| Compile size list covers both phases | a missing size falls back to a dynamic graph where the backend is never a candidate |
| Both arms of a cell on one GPU | inter-card variation is read as a backend effect |
| ABBA repeat ordering | thermal drift over a long sweep is read as an arm effect |
| Explicit failure accounting | a cell silently dropping a repeat reads as a complete measurement |
| Route evidence with controls | a speedup is reported without evidence the backend ran |
| Full provenance per point | a metric without its workload is not comparable to anything |

## Extending it

**Another model.** Add it to `E2E_MODELS`. Prefer dense models: MoE routing
changes which GEMMs exist, so dense and MoE models are not measuring the same
thing. Check the model config for expert-related keys before adding.

**Another backend.**

```bash
E2E_BACKENDS_treatment="ATEN,TRITON,YOURBACKEND" bash scripts/run_all.sh
```

[`scripts/proof/route_evidence.py`](scripts/proof/route_evidence.py) is
backend-specific — it looks for the symbols a backend emits — so extend it too,
and keep its positive and negative controls. A detector that cannot detect a
known-positive case reports "no wins" indefinitely.

**Another dtype or a quantized path.** Two things change:

1. The visibility patch targets the *unquantized* dispatch. Quantized paths go
   through separate dispatch code, so the equivalent switch must be located
   there.
2. The backend's own dtype gate decides whether it produces candidates at all.
   Run `scripts/bench/route_smoke.sh` first: if it reports zero candidates, the
   sweep would silently compare the baseline against itself.

## Interpreting the output

`REPORT.md` states, for every claim, either its evidence or that the evidence
is missing. Read three things before the numbers:

1. **Route evidence.** If the backend did not execute, the speedups are
   correlation, and the report says so.
2. **Data integrity.** Failed points, and whether KV capacity was in fact equal
   across arms.
3. **Scope.** The conditional nature of the claim, and that autotune ran in its
   default, pruned search space — the result is what the shipped heuristics
   pick, not the ceiling either backend could reach.

Figures carry the same context in their footnote, so a chart lifted into a
presentation still says what it measured.
