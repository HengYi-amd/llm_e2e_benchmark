"""Assemble one run into a markdown report.

Speedups are oriented so higher is always better: latency is baseline/treatment,
throughput is treatment/baseline. A speedup is labelled correlation-only unless
the route-evidence stage confirmed the backend actually ran.
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

METRIC_LABEL = {
    "mean_tpot_ms": "TPOT (time per output token, decode steady state)",
    "mean_ttft_ms": "TTFT (time to first token, prefill)",
    "output_throughput": "output throughput (tokens/s)",
}


def geomean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--treatment-arm", default="treatment")
    a = ap.parse_args()
    run = pathlib.Path(a.run_dir)
    norm = run / "normalized"
    L = []
    A = L.append

    A(f"# End-to-end GEMM backend A/B — {run.name}\n")

    mf = run / "manifest" / "run.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        A("## 1. Environment\n")
        A("| key | value |\n|---|---|")
        for k, v in m.items():
            A(f"| `{k}` | `{v}` |")
        A("")

    agg = norm / "e2e_agg.csv"
    if not agg.exists():
        A("## 2. Results\n\n**Not produced** — normalize did not run.\n")
        (run / "report").mkdir(parents=True, exist_ok=True)
        (run / "report" / "REPORT.md").write_text("\n".join(L))
        return
    g = pd.read_csv(agg)
    t = g[g.arm == a.treatment_arm]

    A("## 2. Results\n")
    A("Speedup is the treatment arm over the baseline arm, oriented so that "
      "**higher is always better**. Latency metrics are baseline/treatment, "
      "throughput is treatment/baseline.\n")
    for metric, label in METRIC_LABEL.items():
        col = f"{metric}_speedup"
        if col not in t.columns:
            continue
        A(f"### {label}\n")
        A("| model | dtype | profile | geomean | points faster | n per point |")
        A("|---|---|---|---|---|---|")
        for (model, dtype, prof), d in t.groupby(["model_id", "dtype", "profile"]):
            s = d[col].dropna()
            if not len(s):
                continue
            nmin, nmax = int(d.n.min()), int(d.n.max())
            reps = f"{nmin}" if nmin == nmax else f"{nmin}-{nmax}"
            A(f"| {model} | {dtype} | {prof} | **{geomean(s):.4f}x** | "
              f"{int((s > 1).sum())}/{len(s)} | {reps} |")
        A("")

    A("## 3. Data integrity `[measured]`\n")
    raw = pd.read_csv(norm / "e2e_bench.csv")
    nfail = int((raw.get("status", "ok") != "ok").sum())
    A(f"- Points collected: {len(raw)}; failed: {nfail}"
      + ("" if not nfail else " (excluded from the means; affected cells have a reduced n)"))
    if "kv_consistent" in g.columns:
        bad = int((~g.kv_consistent.fillna(False)).sum())
        A(f"- KV cache capacity identical across arms: "
          + ("yes, in every cell" if not bad else
             f"**no — {bad} cell(s) differ**, so throughput there also reflects a "
             "memory difference and must not be read as a GEMM result"))
    if nfail:
        A("\n| model | profile | concurrency | arm | rep | failure |\n|---|---|---|---|---|---|")
        for _, r in raw[raw.get("status", "ok") != "ok"].iterrows():
            A(f"| {r.get('model_id')} | {r.get('profile')} | {int(r.get('concurrency', 0))} | "
              f"{r.get('arm')} | {int(r.get('repeat_idx', 0))} | {r.get('fail_kind', '?')} |")
    A("")

    rev = run / "raw" / "route_evidence.json"
    A("## 4. Route evidence: did the backend actually run? `[measured]`\n")
    if rev.exists():
        ev = json.loads(rev.read_text())
        s = ev.get("summary", ev)
        A("| metric | treatment arm | baseline arm |\n|---|---|---|")
        A(f"| candidate modules | {s.get('flydsl_candidate_modules', '?')} | "
          f"{s.get('baseline_candidate_modules', '?')} |")
        A(f"| graph modules containing a GEMM | {s.get('flydsl_mm_modules', '?')} | — |")
        A(f"| **modules where it won and ran** | **{s.get('flydsl_winner_modules', '?')}** | "
          f"**{s.get('baseline_winner_modules', '?')}** |")
        A(f"\n{s.get('verdict', '')}\n")
    else:
        A("**Not produced.** Without it the speedups in §2 cannot be attributed to "
          "the kernel and should be read as correlation only.\n")

    A("## 5. What these numbers are, and are not\n")
    A("- The metrics come from `vllm bench serve`, which separates **TTFT** "
      "(prefill) from **TPOT** (decode steady state). A single fused wall-clock "
      "number would blend the two, and a GEMM backend moves them by different "
      "factors — the decode GEMM is skinny (M = concurrency) and memory-bound, "
      "the prefill GEMM is large-M and compute-bound.")
    A("- The comparison is **conditional**: it says what adding the backend to "
      "the Inductor candidate list does, *given that the Inductor "
      "`max_autotune` path is already in use*. It is not a claim about the "
      "engine's default path.")
    A("- Autotune runs in its `DEFAULT` search space, not exhaustive. The "
      "reported result is what the shipped selection heuristics pick, not the "
      "best config either backend could reach.")
    A("")

    (run / "report").mkdir(parents=True, exist_ok=True)
    out = run / "report" / "REPORT.md"
    out.write_text("\n".join(L))
    pngs = list((run / "figures").glob("*.png"))
    print(f"report -> {out}  ({len(L)} lines, {len(pngs)} figures)")


if __name__ == "__main__":
    main()
