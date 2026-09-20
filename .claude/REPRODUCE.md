# Reproducing the FlyDSL end-to-end GEMM benchmark

You are an automation agent. This file is self-contained: following it end to end
reproduces the published result without reading any other document.

**What this measures.** Two vLLM serving runs that differ in exactly one setting —
the TorchInductor GEMM autotune backend list — compared on TPOT, TTFT, end-to-end
latency and output throughput.

| arm | `max_autotune_gemm_backends` |
|---|---|
| `baseline` | `ATEN,TRITON` |
| `treatment` | `ATEN,TRITON,FLYDSL` |

**Published result** (ISL=256 / OSL=512, concurrency 8–256, n>=2 in every cell):

| model | TPOT | TTFT | e2e latency | output throughput |
|---|---|---|---|---|
| Qwen3-32B | 1.0173x | 1.0106x | 1.0169x | 1.0168x |
| Llama-3.3-70B-Instruct | 1.0065x | 1.0026x | 1.0066x | 1.0065x |

**Do not target these numbers.** All twelve cells now meet the collection
criteria — n>=2 repeats, both arms on the same Triton search space, both arms
collected back to back on the same card, symmetric autotune coverage, identical
KV capacity — but the geometric means are still larger than what the data
supports per cell.

**Noise floor**: pooling all 24 replicated measurements gives a within-run
standard deviation of **1.71%** in log space. At n=2 that is roughly the standard
error of a single cell's ratio, so **a per-cell change under about 1.7% is one
sigma**. By that standard only **Qwen c=64 (+5.7%)** clears the noise — its
baseline was measured 6 times and its treatment twice, putting the effect near
3.6 sigma. The other eleven cells, including all six Llama cells, sit inside the
noise, so the honest reading for Llama is "no significant difference", not
"0.65% faster".

Repeats change conclusions in both directions, which is why n=1 is useless here:

| cell | few repeats | after re-measurement |
|---|---|---|
| Llama c=8 | -3.4% (n=1) | +1.1% (n=2) |
| Llama c=16 | +3.8% (n=2) | +0.3% (n=3) |
| Qwen c=8 | +3.5% (n=1) | +0.2% (n=2) |
| Qwen c=64 | +0.3% (n=1) | +5.7% (baseline n=6) |

A single pass invents gains that are not there *and* hides gains that are.

So: run the full matrix in **one batch** with `E2E_REPEATS=3`, and judge the
result by whether the checks in section 4 pass — not by whether it matches these
numbers.

## 1. Prerequisites

Hardware: one node with **8 GPUs**, each with enough VRAM to hold the 70B model's
bf16 weights plus a ~94 GB KV cache at TP=1 (measured: ~288 GB cards). Fewer or
smaller GPUs still work — the sweep just takes proportionally longer, and the 70B
model may not fit at all.

Software:

- A Python environment with a working PyTorch and vLLM. The harness drives vLLM
  through its CLI; it never imports it.
- `curl`, and a GPU query tool (`rocm-smi` or `amd-smi`).
- Roughly 400 GB of disk for model weights, plus ~80 GB for compile caches.

Required environment variables:

```bash
export E2E_VENV=/path/to/python/env      # must contain torch and vllm
export E2E_VLLM_SRC=/path/to/vllm        # source checkout, for the patch step
```

If the GPU stack lives in a container and you are driving from outside it, set
`E2E_CONTAINER=<container-name>` and `E2E_CONTAINER_ROOT=<repo path inside it>`;
every command below then runs through `podman exec`/`docker exec` automatically.
Otherwise leave both empty and run the commands directly.

---

## 2. One-command path

From the repository root:

```bash
bash scripts/run_all.sh
```

This runs stages 00–10 and is resumable: a stage already marked passed in
`state/stages/` is skipped. All defaults in `config/default.env` are already the
published recipe — **do not override anything** unless this document says to.

Guard a long sweep in parallel. The supervisor relaunches only points that no
live shard is working on and never signals a healthy shard:

```bash
bash scripts/supervisor.sh &
```

Expected wall time on 8 idle GPUs: **1.5–2 hours** for the full 2-model matrix.

If you prefer to drive stages yourself, they are:

| stage | command |
|---|---|
| `00_preflight` | `bash scripts/preflight/check_gpus.sh` |
| `01_verify` | `python scripts/setup/verify_stack.py` |
| `02_patch_vllm` | `bash scripts/setup/patch_vllm.sh` |
| `03_route_smoke` | `bash scripts/bench/route_smoke.sh` |
| `04_kv_pin` | `bash scripts/bench/calibrate_kv_all.sh` |
| `05_e2e` | `bash scripts/bench/run_sharded.sh` |
| `06_route_proof` | `python scripts/proof/route_evidence.py runs/<run_id>` |
| `07_normalize` | `python scripts/normalize/normalize.py --run-dir runs/<run_id>` |
| `08_plot` | `python scripts/plot/plot.py --run-dir runs/<run_id>` |
| `09_publish` | `python scripts/publish.py --run-dir runs/<run_id> --result-dir result --suffix _exhaustive` |
| `10_combined` | `python scripts/plot/combined.py --run-dir runs/<run_id> --out-dir result/bf16_result_png` |

`state/current_run_id` holds the active run id.

---

## 3. The recipe, and the three settings that actually matter

Everything below is already the default. It is listed so you can verify, not so
you can set it.

| knob | value |
|---|---|
| `E2E_MODELS` | `Qwen/Qwen3-32B meta-llama/Llama-3.3-70B-Instruct` |
| `E2E_DTYPES` | `bfloat16` |
| `E2E_PROFILE_showcase` | `256:512` (ISL:OSL) |
| `E2E_CONCURRENCY_SWEEP` | `8 16 32 64 128 256` |
| `E2E_REPEATS` | `2` |
| `E2E_TP` | `1` |
| `E2E_MAX_NUM_BATCHED_TOKENS` | `2048` |
| `E2E_MAX_MODEL_LEN` | `4096` |
| `E2E_MAX_NUM_SEQS` | `256` |
| `E2E_GPU_MEM_UTIL` | `0.85` |
| `E2E_AUTOTUNE_SEARCH_SPACE` | `EXHAUSTIVE` |
| `E2E_TRITON_DEFAULT_SPACE` | `1` |
| `E2E_FLYDSL_AUTOTUNING` | `1` |
| `E2E_PRECOMPILE_TIMEOUT_S` | `3600` |

Three of these are load-bearing and silently ruin the result if wrong:

1. **`E2E_REPEATS=2`.** The pooled within-run standard deviation on this
   workload is **1.71%** in log space, and a single arm's repeat range reached
   7.7% in one cell. With `n=1` you cannot distinguish a 2% effect from drift and
   the sign flips between runs — three cells in the published batch did exactly
   that once repeats were added. This default was once `1`, which produced a
   published-looking result that reversed on re-measurement. Prefer `3`: at n=2 a
   single outlying run still moves the median.

2. **`E2E_TRITON_DEFAULT_SPACE=1`.** Keeps Triton on its default candidate space
   while the candidate backend runs exhaustive. Triton's exhaustive space adds
   ~816 s of precompilation per model for no measurable change in what it picks,
   and it pushes candidate counts past the precompilation timeout — after which
   candidates are *silently dropped*, making the surviving set depend on machine
   load rather than on configuration.

3. **`E2E_FLYDSL_AUTOTUNING=1`.** Off upstream. With it off the backend
   contributes one hardcoded configuration against a tuned competitor, which does
   not compare backends at all.

`config/default.env` is sourced with `set -a`, so values are exported and
inherited by children. **Editing that file does not affect a process that is
already running.** To confirm a running process picked a value up, read it back
from `/proc/<pid>/environ` — never assume the file is the truth.

---

## 4. Verifying the run is valid

Run every check. Each one has caught a real failure that otherwise looked like a
clean result.

### 4.1 All points present, and n=2

```bash
RID=$(cat state/current_run_id)
ls runs/$RID/raw/e2e/*.json | wc -l          # expect 48 for the full matrix
ls runs/$RID/raw/e2e/ | grep -c _r2          # expect half of the above
grep -L '"status": "ok"' runs/$RID/raw/e2e/*.json   # expect no output
```

A point count that looks right while `_r2` is zero means `E2E_REPEATS` did not
take effect. The data is then single-pass and cannot support any conclusion.

### 4.2 The backend actually won autotune decisions

This is the check that decides whether the comparison means anything. If the
candidate backend never won, the two arms are functionally identical and any
difference is noise.

In a server log an autotune block looks like:

```text
AUTOTUNE mm(8x8192, 8192x10240)
strides: [8192, 1], [1, 8192]
dtypes: torch.bfloat16, torch.bfloat16
  flydsl_mm_flydsl_7 0.0268 ms 100.0% FlyDSL template ...
  triton_mm_2379 0.0281 ms 95.4% ...
```

**The winner is the first candidate line after the `dtypes:` line — not the line
immediately after the `AUTOTUNE` header.** Taking the header's next line reads
`strides:` and yields the false conclusion that the backend never won.

```bash
RID=$(cat state/current_run_id)
python3 - "$RID" <<'PY'
import glob, re, sys, collections, statistics as st
hdr = re.compile(r'AUTOTUNE mm\((\d+)x')
cnd = re.compile(r'^\(EngineCore[^)]*\)\s+(\S+)\s+([\d.]+) ms')
for arm in ("baseline", "treatment"):
    wins, margins = collections.Counter(), []
    for f in glob.glob(f"runs/{sys.argv[1]}/logs/server_{arm}_*.log"):
        L = open(f, errors="ignore").read().splitlines()
        i = 0
        while i < len(L):
            if not hdr.search(L[i]):
                i += 1; continue
            cands, j = [], i + 1
            while j < len(L) and len(cands) < 5:
                c = cnd.match(L[j])
                if c: cands.append((c.group(1), float(c.group(2))))
                elif cands: break
                j += 1
            if cands:
                k = ("flydsl" if cands[0][0].startswith("flydsl")
                     else "aten" if cands[0][0] == "mm" else "triton")
                wins[k] += 1
                if k == "flydsl":
                    r = next((x for x in cands[1:]
                              if not x[0].startswith("flydsl")), None)
                    if r: margins.append((r[1] - cands[0][1]) / cands[0][1] * 100)
            i = j if j > i else i + 1
    tot = sum(wins.values())
    m = f", median win margin {st.median(margins):+.2f}%" if margins else ""
    print(f"{arm:10s} {tot:4d} decisions  flydsl={wins['flydsl']} "
          f"aten={wins['aten']} triton={wins['triton']}{m}")
PY
```

Expected: `baseline` shows **flydsl=0** (the backend is not in its list), and
`treatment` shows a non-zero flydsl count. Reference values from the published
run: Qwen 51 of 168 decisions (30%, median margin +5.58%), Llama 76 of 182 (42%,
+3.29%).

A treatment arm with `flydsl=0` means the backend never routed. Stop and debug
before reading any speedup — re-run `bash scripts/bench/route_smoke.sh`, which
gates exactly this.

### 4.3 Both arms of every cell measured the same thing

```bash
RID=$(cat state/current_run_id)
python3 - "$RID" <<'PY'
import json, glob, sys, collections
cells = collections.defaultdict(dict)
for f in glob.glob(f"runs/{sys.argv[1]}/raw/e2e/*.json"):
    d = json.load(open(f))
    if d.get("status") != "ok": continue
    cells[(d["model_id"], d["concurrency"])].setdefault(d["arm"], []).append(d)
keys = ("autotune_search_space", "triton_default_space", "kv_cache_tokens",
        "max_num_batched_tokens", "max_num_seqs")
bad = 0
for (m, c), arms in sorted(cells.items()):
    if len(arms) < 2:
        print(f"  INCOMPLETE {m} c={c}: {list(arms)}"); bad += 1; continue
    sig = {a: tuple(str(v[0].get(k)) for k in keys) for a, v in arms.items()}
    if len(set(sig.values())) > 1:
        print(f"  MISMATCH {m} c={c}: {sig}"); bad += 1
print("all cells comparable" if not bad else f"{bad} cell(s) need attention")
PY
```

`kv_cache_tokens` differing between arms is the most damaging case: the arm with
more KV blocks wins on throughput for reasons unrelated to GEMMs. `normalize.py`
also reports this as `KV capacity identical across arms in N cells`.

### 4.4 Every reported change is larger than its own noise

```bash
RID=$(cat state/current_run_id)
python3 - "$RID" <<'PY'
import csv, sys, math
rows = list(csv.DictReader(open(f"runs/{sys.argv[1]}/normalized/e2e_agg.csv")))
for m in sorted({r["model_id"] for r in rows}):
    vals = []
    print(m)
    for c in sorted({int(r["concurrency"]) for r in rows if r["model_id"] == m}):
        g = lambda a: next((r for r in rows if r["model_id"] == m
                            and int(r["concurrency"]) == c and r["arm"] == a), None)
        b, t = g("baseline"), g("treatment")
        if not b or not t: continue
        sp = float(b["mean_tpot_ms_mean"]) / float(t["mean_tpot_ms_mean"])
        vals.append(sp)
        spread = max(abs(float(x.get("mean_tpot_ms_spread") or 0)) for x in (b, t)) * 100
        delta = (sp - 1) * 100
        flag = "  <- inside noise" if abs(delta) <= spread else ""
        print(f"  c={c:<4d} {delta:+6.1f}%   spread {spread:4.1f}%{flag}")
    print(f"  geomean {math.exp(sum(map(math.log, vals))/len(vals)):.4f}x\n")
PY
```

Any cell marked `inside noise` must not be reported as a speedup, whatever its
sign. In the published run every Llama cell is inside noise, which is why the
correct statement for that model is "no significant difference", not "faster".

---

## 5. Interpreting what you get

The treatment arm is a **superset** of the baseline's candidate list. Autotune
keeps the fastest measured candidate, so adding a backend can only change the
outcome where that backend measured faster. A genuine end-to-end regression is
therefore not evidence of a slower kernel — it is evidence of one of:

- **measurement noise** (check §4.4 first), or
- **an autotune decision made inside its own noise**: when the winning margin is
  a couple of percent, the isolated microbenchmark cannot rank candidates
  reliably, and the kernel it picks may not be faster in the serving loop where
  cache state and contention differ.

Kernel-level wins convert to end-to-end gains only where the model is not already
bandwidth-bound. Before attributing a flat result to the backend, compare measured
TPOT against the weight-streaming roofline:

```text
floor_ms = (weight_bytes + kv_bytes_per_token * concurrency * avg_context) / bandwidth
```

In the published run Llama-70B sits at 83% of that floor at concurrency 8 and
Qwen-32B at 67%. Llama has roughly 17% of headroom, shared with attention,
sampling and scheduling — which is why a 3.57% kernel win does not show up
end to end, while Qwen's 9.50% win does.

---

## 6. Failure modes and what to do

| symptom | cause | action |
|---|---|---|
| Point count right, `_r2` count zero | `E2E_REPEATS` not in effect | check `/proc/<pid>/environ`, relaunch |
| `treatment` shows `flydsl=0` | backend never routed | re-run `scripts/bench/route_smoke.sh`; check `E2E_FLYDSL_AUTOTUNING=1` and the vLLM patch |
| Whole sweep dies with no error output | cache cleanup hit an open file handle on a network filesystem | already handled (rename + background delete); if you changed it, restore that |
| Servers killed mid-compile | stall threshold below the silent precompilation window | `E2E_SERVER_STALL_S=5400` or higher |
| Every server killed at round 2 | `E2E_SERVER_TIMEOUT_S` too low | `43200` |
| KV allocation fails on the largest model | `E2E_GPU_MEM_UTIL` too high, starving the compile workspace | `0.85` |
| `autotune_total: 0` in the smoke probe | stale smoke cache produced a cache hit | set `SMOKE_CACHE_TAG` to the run id |
| A shard dies and nothing restarts it | supervisor not running | `bash scripts/supervisor.sh &` |
| `hipErrorInvalidValue` during KV allocation | **root cause unknown**; seen 3 times in long sweeps | supervisor relaunches the affected point |

**Do not** use a pattern sweep (`pkill -f run_e2e.sh` or similar) to clean up. It
also matches shards started by hand or by a parallel run, and killing those
destroys work you did not own. `scripts/stop.sh` kills a verified process tree;
`scripts/daemon/stop.sh` will not sweep by pattern unless `E2E_STOP_SWEEP=1`.

On a shared machine, `run_sharded.sh` claims cards opportunistically: it skips a
card another user has queues on and picks it up when it drains. Never kill another
user's process to free a GPU.

---

## 7. Deliverables

```text
result/Qwen3-32B_bf16_exhaustive/                 8 PNGs + summary.csv + raw_points.csv
result/Llama-3.3-70B-Instruct_bf16_exhaustive/    same
result/bf16_result_png/                           4 cross-model comparison PNGs
```

Each figure carries its workload in a footnote, so a chart lifted into a
presentation still states what it measured. Speedup bars are labelled in percent
with a zero baseline; changes within 1% are deliberately left unlabelled, because
labelling them would present noise as a measurement.
