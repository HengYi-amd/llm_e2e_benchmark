# llm_e2e_benchmark

An A/B harness for TorchInductor GEMM backends, measured end to end through
vLLM.

It answers one question: **when a GEMM backend is added to Inductor's autotune
candidate list, what happens to a real model's serving latency?** Two arms
differ in exactly one setting — the backend list — and everything else is held
identical, so any difference is attributable to that change.

The harness is backend-, model- and dtype-agnostic. The shipped defaults target
the FlyDSL backend at bf16; see [Extending it](#extending-it).

For the exact parameters of the published run, and why each was chosen, see
[RUN_Recipe.md](RUN_Recipe.md). For a self-contained reproduction procedure,
see [.claude/REPRODUCE.md](.claude/REPRODUCE.md).

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
- [Reading a result](#reading-a-result)
- [Extending it](#extending-it)

## What it measures

| arm | `max_autotune_gemm_backends` |
|---|---|
| `baseline` | `ATEN,TRITON` |
| `treatment` | `ATEN,TRITON,FLYDSL` |

The treatment arm is a superset. Autotune benchmarks every candidate and keeps
the fastest, so adding a backend can only change the outcome where that backend
measured faster. Any end-to-end regression therefore points at measurement
noise or at an autotune decision made inside its own noise — not at a slower
kernel. See [Reading a result](#reading-a-result).

Metrics, collected with `vllm bench serve`:

| metric | isolates | role |
|---|---|---|
| **TPOT** — time per output token | decode steady state; GEMM M = concurrency | headline |
| **TTFT** — time to first token | prefill; large-M GEMMs | supporting |
| **end-to-end latency** | whole request | supporting |
| **output throughput** — tokens/s | serving capacity | supporting |

TTFT and TPOT are reported separately rather than as one fused latency. A GEMM
backend moves the two phases by different factors: decode GEMMs are skinny and
memory-bound, prefill GEMMs are large-M and compute-bound. A fused wall-clock
number blends them, hides which one moved, and can be shifted by changing the
output length alone. (`mean_e2el_ms` does equal `TTFT + TPOT x (OSL-1)` to
within rounding, so nothing is lost by reporting the parts.)

## Requirements

- A GPU with a working PyTorch and vLLM install. The harness drives vLLM
  through its CLI; it does not import it.
- A Python environment containing both, reachable via `E2E_VENV`.
- `curl` for server readiness probing, and a GPU query tool (`amd-smi` or
  `rocm-smi`) for the preflight checks and for shard scheduling.
- Enough VRAM for the largest model at TP=1. The shipped defaults assume a
  single card can hold the 70B weights plus the pinned KV cache.

`scripts/build/` holds optional source-build helpers if a specific revision of
PyTorch or vLLM is needed.

## Quick start

```bash
git clone <this-repository>
cd llm_e2e_benchmark

export E2E_VENV=/path/to/python/env      # required
export E2E_VLLM_SRC=/path/to/vllm        # for the source patch step
export E2E_TORCH_SRC=/path/to/pytorch    # for the build helpers

bash scripts/run_all.sh
```

Narrow the sweep without editing anything:

```bash
E2E_MODELS="<hf-model-id>" \
E2E_CONCURRENCY_SWEEP="8 64" \
  bash scripts/run_all.sh
```

Re-run only post-processing against an existing run:

```bash
E2E_ONLY="07 08 09 10" bash scripts/run_all.sh
```

Guard a long sweep. The supervisor relaunches only points that no live shard is
working on, and never signals a healthy one:

```bash
bash scripts/supervisor.sh &
```

Run detached, and stop safely:

```bash
bash scripts/daemon/start.sh
bash scripts/daemon/status.sh
bash scripts/daemon/stop.sh
```

`stop.sh` verifies the recorded process still belongs to this run before
signalling, and kills the process tree rather than the process group. On a
shared machine a stop that leaves a server holding memory, or that kills an
unrelated job, is worse than no stop at all. `daemon/stop.sh` will not sweep for
leftovers by pattern unless `E2E_STOP_SWEEP=1` is set, because such a sweep also
matches shards started by hand or by a second run.

Set `E2E_CONTAINER` when the GPU stack lives in a container and the pipeline is
driven from outside it; leave it empty to run everything directly.

## Repository layout

```text
config/default.env          all tunables and their defaults
env.sh                      paths and cache isolation; sourced by every script

scripts/
  run_all.sh                the pipeline; stages are resumable
  supervisor.sh             relaunch missing points without disturbing live ones
  progress.sh               point count, per-shard state, elapsed time
  stop.sh                   process-tree kill used by daemon/stop.sh
  preflight/                GPU availability checks
  setup/
    verify_stack.py         assert the installed versions match expectations
    patch_vllm.sh           add the switch that exposes GEMM to the compiler
  bench/
    route_smoke.sh          gate: prove the backend is reachable
    route_smoke_probe.py    the probe the gate runs
    calibrate_kv.sh         measure KV capacity for one model
    calibrate_kv_all.sh     pin it, per model, equal across arms
    gpumap.sh               map HIP device index to the GPU tool's numbering
    run_e2e.sh              one shard: a server per arm, swept over concurrency
    run_sharded.sh          fan shards across GPUs, claiming cards as they free
    tally_route.py          parse autotune selections from a log
  proof/route_evidence.py   did the backend actually execute?
  normalize/normalize.py    raw results into tidy CSVs, with integrity checks
  plot/
    plot.py                 per-model speedup bars and absolute-value lines
    combined.py             one chart per metric with every model on it
  publish.py                the deliverable layout under result/
  daemon/                   start / status / stop

patches/flydsl_raw_args.patch   optional compiler-side fix, see below
```

Everything a run writes stays out of version control. `runs/<run_id>/`:

```text
raw/e2e/*.json      one file per point, carrying its full workload description
normalized/*.csv    per-point and per-cell tables
figures/*.png       each with a JSON sidecar
logs/               one server log per (arm, model, repeat, concurrency)
```

and `result/` holds the published copies: one flat directory per
(model, dtype), plus a directory of cross-model comparison figures.

## Configuration

Every value in [`config/default.env`](config/default.env) can be overridden
from the environment. The defaults are the published recipe.

| knob | default | note |
|---|---|---|
| `E2E_MODELS` | two dense models | dense only; MoE changes which GEMMs exist |
| `E2E_DTYPES` | `bfloat16` | |
| `E2E_ARMS` | `baseline treatment` | backend lists in `E2E_BACKENDS_<arm>` |
| `E2E_CONCURRENCY_SWEEP` | `8 16 32 64 128 256` | becomes the decode GEMM's M |
| `E2E_PROFILES` | `showcase` | workload shape, below |
| `E2E_PROFILE_showcase` | `256:512` | ISL:OSL |
| `E2E_TP` | `1` | |
| `E2E_MAX_NUM_BATCHED_TOKENS` | `2048` | pinned, not defaulted |
| `E2E_MAX_MODEL_LEN` | `4096` | |
| `E2E_GPU_MEM_UTIL` | `0.85` | with KV pinned; higher starved the compile workspace |
| `E2E_AUTOTUNE_SEARCH_SPACE` | `EXHAUSTIVE` | applies to the candidate backend |
| `E2E_TRITON_DEFAULT_SPACE` | `1` | keeps Triton on its default space regardless |
| `E2E_FLYDSL_AUTOTUNING` | `1` | required; see below |
| `E2E_REPEATS` | `2` | ABBA-interleaved; see below |
| `E2E_GPUS` | every GPU present | shards, not tensor parallelism |
| `E2E_CONTAINER` | empty | run directly, or wrap in a container |

`config/default.env` is sourced with `set -a`, so every value is exported and
inherited by child processes. **A process already running does not see an edit
to that file.** Change a value and relaunch; to confirm a running process picked
one up, read it back from `/proc/<pid>/environ` rather than from the file.

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
| `08_plot` | per-model figures |
| `09_publish` | the deliverable layout under `result/` |
| `10_combined` | cross-model comparison figures |

A stage already marked passed is skipped, so a re-run resumes rather than
recompiling from scratch. `E2E_FROM=05` starts from a stage; `E2E_ONLY="07 08"`
runs only those.

## Design decisions

### Workload profile

**ISL** (input sequence length) sets the prefill work; **OSL** (output sequence
length) sets the number of decode steps.

The shipped `showcase` profile is ISL=256 / OSL=512: long enough that prefill is
real work rather than scheduling overhead, and decode-dominated enough that TPOT
is the headline. Two decode steps per prefill token keeps the steady state in
the concurrency-shaped GEMM regime the backend is being tested on.

vLLM schedules prefill in chunks of at most `max_num_batched_tokens`. That
default differs between the offline and server code paths, so the harness pins
it: a benchmark whose chunk size depends on which entry point was used is not
reproducible. **The pinned value, not ISL, is what sets the prefill GEMM's M.**
A prefill step is `min(ISL x concurrency, max_num_batched_tokens)` tokens wide,
so raising concurrency reaches the cap long before raising ISL does.

`plot.py` refuses to draw TTFT when the widest prefill in the sweep is under 512
tokens: at that width TTFT measures scheduling, not GEMM time.

### Repeats

`E2E_REPEATS=2` is the floor, not a nicety. The measured repeat spread on this
workload is **~1.2% median and ~2.6% worst case for TPOT**, and worse for TTFT.
A single pass therefore cannot distinguish a 2% effect from drift, and the sign
of such an effect flips between runs. Aggregation takes the median per cell and
records a `*_spread` column beside every metric; `normalize.py` warns when a
cell's repeats disagree by more than 5%.

Repeats are ABBA-interleaved: repeat 1 runs `baseline` then `treatment`, repeat
2 runs them in the opposite order, so drift over the sweep cannot masquerade as
an arm effect.

### Tensor parallelism

`E2E_TP=1`. Sharding divides each Linear's N or K by the TP size, so the backend
would be tested on GEMMs a fraction of their real width, and each decoder layer
would add collective calls to the decode critical path — time both arms pay
equally, which only dilutes the signal.

TP=1 does not mean one GPU is used. `run_sharded.sh` occupies every available
GPU by running independent single-GPU shards. Both arms of a cell run on the
same GPU, because the arm ratio is the result and inter-card clock or thermal
spread would otherwise look like a backend difference.

Cards are claimed opportunistically: a card another user has queues on is
skipped rather than taken, and picked up later when it drains. After a grace
period the free-memory test alone decides, which still cannot evict anyone.

### Autotune breadth

The candidate backend runs its **exhaustive** space; Triton stays on its
**default** space (`E2E_TRITON_DEFAULT_SPACE=1`). Triton's exhaustive space
costs hours of precompilation per model for a selection that, on the shapes
under test here, did not differ measurably from its default space. Spending
that time does not change the comparison, and the precompilation timeout
silently drops candidates under load, which makes the surviving set depend on
machine load rather than on the configuration.

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
| Per-arm compile caches, wiped per point | the second arm loads the first arm's artifacts, skips autotune, and reports a cache hit as a result |
| KV capacity pinned equal | the arm granted more KV blocks wins on throughput for reasons unrelated to GEMMs |
| Compile size list covers both phases | a missing size falls back to a dynamic graph where the backend is never a candidate |
| Both arms of a cell on one GPU | inter-card variation is read as a backend effect |
| ABBA repeat ordering | thermal drift over a long sweep is read as an arm effect |
| At least two repeats, spread recorded | an effect smaller than the noise floor is reported as a result |
| Explicit failure accounting | a cell silently dropping a repeat reads as a complete measurement |
| Route evidence with controls | a speedup is reported without evidence the backend ran |
| Full provenance per point | a metric without its workload is not comparable to anything |

Cache directories are replaced by renaming and deleting in the background, not
by deleting in place: on a network filesystem an in-place delete can fail on a
still-open file handle, and under `set -e` that failure takes the whole sweep
down silently.

## Reading a result

Read these before the numbers:

1. **Route evidence.** If the backend never won an autotune decision, the arms
   are functionally identical and any difference is noise. `route_evidence.py`
   answers this; so does counting winners in the server logs. The winner of an
   autotune round is the **first candidate line after the `dtypes:` line**, not
   the line immediately after the `AUTOTUNE` header.
2. **Repeat spread.** Compare each cell's reported change against that cell's
   own `*_spread`. A change smaller than the spread is not a result, whatever
   its sign.
3. **Provenance parity.** Both arms of a cell must share search space, KV
   capacity, batched-token cap and sequence cap. `normalize.py` checks KV; the
   rest is in each point's JSON.
4. **Scope.** The claim is conditional on the Inductor autotune path being in
   use, and on the shapes this workload generates.

A backend that wins autotune by a margin smaller than the benchmark's own noise
is a coin flip, not a speedup: the selection can pick a kernel that is not
actually faster in the serving loop, where cache state and contention differ
from the isolated microbenchmark. Expect kernel-level wins to convert to
end-to-end gains only where the model is not already bandwidth-bound — compare
measured TPOT against the weight-streaming roofline before attributing a flat
result to the backend.

Figures carry their workload in a footnote, so a chart lifted into a
presentation still says what it measured.

## Extending it

**Another model.** Add it to `E2E_MODELS`. Prefer dense models: MoE routing
changes which GEMMs exist, so dense and MoE models are not measuring the same
thing. Check the model config for expert-related keys before adding. Models
without bias produce `mm`; models with bias also produce `addmm`, which a
backend may or may not hook — check before assuming coverage.

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
