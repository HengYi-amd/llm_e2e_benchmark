"""Grouped speedup figures: one chart per metric, one bar group per concurrency.

Each model is a colour; the sign of a bar is carried by its direction, not by a
second colour, so a bar that drops below the axis reads as slower at a glance and
the legend stays about models. Changes inside one percent are left unlabelled:
the run-to-run spread is wider than that, so a number there would dress up noise.
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated for CVD separation against the chart surface (dE 23.2 protan,
# 30.1 normal, both >= 3:1 contrast).
MODEL_COLORS = ["#2a78d6", "#d1603d"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

METRICS = [
    ("mean_tpot_ms_speedup", "TPOT", "decode steady state"),
    ("mean_ttft_ms_speedup", "TTFT", "prefill"),
    ("mean_e2el_ms_speedup", "end-to-end latency", "whole request"),
    ("output_throughput_speedup", "output throughput", "capacity"),
]
SLUG = {"TPOT": "tpot", "TTFT": "ttft",
        "end-to-end latency": "e2e_latency", "output throughput": "throughput"}

# Symmetric auto-scaling wastes most of the panel when a metric's values sit
# almost entirely on one side of zero, which is the case for TTFT. Pin the axis
# for those so the bars fill the frame; anything unlisted keeps auto-scaling.
YLIM = {"TTFT": (-2.0, 10.0)}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "legend.frameon": False,
})


def one_figure(agg, raw, metric, name, phase, out_dir):
    t = agg[agg.arm == "treatment"]
    models = sorted(t.model_id.unique())
    concs = sorted(t.concurrency.unique())
    if not models or not concs or metric not in t.columns:
        return None

    width = 0.8 / len(models)
    x = np.arange(len(concs))
    fig, ax = plt.subplots(figsize=(11.5, 5.4))

    for i, m in enumerate(models):
        g = t[t.model_id == m].set_index("concurrency")
        vals = [((g[metric].get(c, np.nan) - 1.0) * 100.0) for c in concs]
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, vals, width=width * 0.86, color=MODEL_COLORS[i % len(MODEL_COLORS)],
               label=m.split("/")[-1], zorder=3)
        for xi, v in zip(pos, vals):
            if not np.isfinite(v) or abs(v) < 1.0:
                continue
            ax.text(xi, v + (0.12 if v > 0 else -0.12), f"{v:.1f}%",
                    ha="center", va="bottom" if v > 0 else "top",
                    fontsize=8.5, color=INK2, zorder=5)

    ax.axhline(0.0, color=AXIS, lw=1.0, zorder=4)
    allv = [((t[t.model_id == m].set_index("concurrency")[metric].get(c, np.nan) - 1) * 100)
            for m in models for c in concs]
    allv = [v for v in allv if np.isfinite(v)]
    if name in YLIM:
        lo, hi = YLIM[name]
        if allv and (min(allv) < lo or max(allv) > hi):
            print(f"  warning: {name} data spans {min(allv):+.1f}..{max(allv):+.1f}%, "
                  f"outside the pinned axis {lo}..{hi}")
        ax.set_ylim(lo, hi)
    else:
        span = max(max(abs(v) for v in allv), 1.5) if allv else 1.5
        ax.set_ylim(-span * 1.5, span * 1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in concs])
    ax.set_xlabel("Batch Size (concurrent requests)")
    ax.set_ylabel("Speedup (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(loc="upper left", fontsize=9)

    isl = int(raw.isl.iloc[0]) if "isl" in raw.columns else 0
    osl = int(raw.osl.iloc[0]) if "osl" in raw.columns else 0
    chunk = int(raw.max_num_batched_tokens.iloc[0]) if "max_num_batched_tokens" in raw.columns else 0
    ax.set_title(f"vLLM Inference: BF16, MI355X\n{name} ({phase})",
                 fontsize=12, loc="center")
    foot = (f"Baseline: ATEN+TRITON  |  Treatment: ATEN+TRITON+FLYDSL  |  "
            f"torch.compile, max-autotune  |  TP=1\n"
            f"vllm bench serve, ISL={isl} / OSL={osl}, "
            f"max_num_batched_tokens={chunk}, prefix caching off, ignore_eos, seed 0  |  "
            f"bars above zero are faster; changes within 1% are left unlabelled")
    fig.text(0.5, -0.02, foot, ha="center", va="top", fontsize=8,
             color=MUTED, linespacing=1.6)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bf16_isl{isl}osl{osl}_{SLUG.get(name, name)}_speedup"
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    run = pathlib.Path(a.run_dir)
    agg = pd.read_csv(run / "normalized" / "e2e_agg.csv")
    raw = pd.read_csv(run / "normalized" / "e2e_bench.csv")

    # Only draw a metric when every model has every concurrency in both arms;
    # a grouped chart with a hole in it invites the reader to fill the gap.
    need = len(agg.model_id.unique()) * len(agg.concurrency.unique()) * 2
    if len(agg) < need:
        print(f"incomplete: {len(agg)} of {need} cells; not drawing")
        return

    out = pathlib.Path(a.out_dir)
    for metric, name, phase in METRICS:
        stem = one_figure(agg, raw, metric, name, phase, out)
        if stem:
            print(f"  {stem}.png")
    print(f"combined figures -> {out}")


if __name__ == "__main__":
    main()
