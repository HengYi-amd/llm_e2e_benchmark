"""Render one figure per (model, dtype, profile, metric) from the aggregated CSV.

The footnote is part of the figure because a speedup without its workload is not
comparable. Keep all figure text ASCII: the default font lacks other glyphs.
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Contrast-checked categorical palette.
C_MAIN, C_ALT = "#1baf7a", "#2a78d6"
# Diverging pair for the change-vs-baseline bars: faster and slower are
# opposite polarities, not two categories. Validated for CVD separation.
C_UP, C_DOWN = "#2a78d6", "#d1603d"
# One colour per model, never per sign: a bar's direction already carries
# the sign, and recolouring by it makes two models look like four series.
MODEL_COLORS = ["#2a78d6", "#d1603d"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CRIT = "#d03b3b"

DTYPE_SHORT = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}

METRIC_LABEL = {
    "mean_tpot_ms": ("TPOT", "time per output token", "decode steady state"),
    "mean_ttft_ms": ("TTFT", "time to first token", "prefill"),
    "mean_e2el_ms": ("end-to-end latency", "TTFT + TPOT x (OSL - 1)",
                     "whole request"),
    "output_throughput": ("output throughput", "tokens/s", "capacity"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 160,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "legend.frameon": False,
})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6, ls="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)


def fname(short, dtype, frame, metric_name):
    """Figure name: model, dtype, workload, metric. Nothing else."""
    isl, osl = int(frame.isl.iloc[0]), int(frame.osl.iloc[0])
    tag = metric_name.lower().replace(" ", "")
    return f"{short}_{DTYPE_SHORT.get(dtype, dtype)}_isl{isl}osl{osl}_{tag}"


def save(fig, out, name, meta):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    (out / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    plt.close(fig)
    print(f"  {name}.png")


def speedup_bars(d, out, model, dtype, prof, metric, arms, ctx, color=MODEL_COLORS[0]):
    col = f"{metric}_speedup"
    t = d[(d.arm == arms[1]) & d[col].notna()].sort_values("concurrency")
    if t.empty:
        return
    sp = t[col].values
    # Plot the change against the baseline, not the ratio: zero is the baseline,
    # a faster point rises and a slower one falls. Direction carries the polarity
    # on its own, so the two colours are reinforcement rather than the only cue.
    pct = (sp - 1.0) * 100.0
    x = np.arange(len(t))
    short = model.split("/")[-1]
    name, unit, phase = METRIC_LABEL.get(metric, (metric, "", ""))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x, pct, width=0.62, color=color, zorder=3)
    ax.axhline(0.0, color=AXIS, lw=1.0, zorder=4)
    span = max(float(np.abs(pct).max()), 1.0)
    pad = span * 0.14
    for xi, v in zip(x, pct):
        # Anything inside a point of the baseline is noise at this sample size;
        # labelling it would dress up a number the data cannot support.
        if abs(v) < 1.0:
            continue
        ax.text(xi, v + (pad * 0.12 if v > 0 else -pad * 0.12), f"{v:+.1f}%",
                ha="center", va="bottom" if v > 0 else "top",
                fontsize=9, color=INK, fontweight="bold", zorder=5)
    ax.set_ylim(min(pct.min() - pad, -pad), max(pct.max() + pad, pad))
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in t.concurrency])
    ax.set_xlabel("concurrency (requests in flight)")
    ax.set_ylabel(f"{name} change vs baseline (%)")
    gm_pct = (float(np.exp(np.log(sp).mean())) - 1.0) * 100.0
    ax.set_title(f"Adding FlyDSL to the Inductor GEMM backends - {short} {dtype}\n"
                 f"{name} ({phase}): geomean {gm_pct:+.1f}%, "
                 f"{int((sp > 1).sum())}/{len(sp)} points faster",
                 fontsize=11, loc="left")
    style(ax)

    nfail = int(t.n_failed.sum()) if "n_failed" in t else 0
    foot = workload_footnote(
        t, ctx, arms,
        extra=(f"; {nfail} failed point(s) excluded" if nfail else "")
              + f"; {name} = {unit}; bars above zero are faster than the baseline; "
                "changes within 1% are left unlabelled")
    fig.text(0.008, -0.055, foot, ha="left", va="top",
             fontsize=7.5, color=MUTED, linespacing=1.6)
    save(fig, out, fname(short, dtype, t, name) + "_speedup",
         {"model": model, "dtype": dtype, "profile": prof, "metric": metric,
          "geomean": float(np.exp(np.log(sp).mean())), "geomean_pct": gm_pct, "wins": int((sp > 1).sum()), "points": len(sp)})


def workload_footnote(d, ctx, arms, extra=""):
    """The workload line that every figure carries.

    A speedup without its workload is not a result: the batch width, the prefill
    chunk width and the search space each move the number more than the kernel
    under test does.

    Only settings a reader would misread the chart without are listed. Outputs
    are fixed-length because the client passes --ignore-eos, which is what makes
    the arms comparable: with natural stopping the decode step count would track
    how much the model chose to say. The seed is a reproducibility detail that
    changes no bar, so it stays in each point's provenance JSON and in the
    published raw_points.csv rather than on the chart.
    """
    t = d.iloc[0]
    return (
        f"Baseline: {arms[0]} = {ctx['backends'][0]}   |   "
        f"Treatment: {arms[1]} = {ctx['backends'][1]}   |   "
        f"torch.compile max-autotune, "
        f"origami={ctx['origami']}/top{ctx['origami_topk']}\n"
        f"vllm bench serve, TP={ctx['tp']}, "
        f"ISL={int(t.isl)} / OSL={int(t.osl)} (fixed-length outputs), "
        f"max_num_batched_tokens={ctx['chunk']}, prefix caching off\n"
        + extra.lstrip("; ")
    )


def absolute_lines(d, out, model, dtype, prof, metric, arms, ctx):
    """Absolute values per arm; a ratio alone hides whether both arms are slow."""
    name, unit, _ = METRIC_LABEL.get(metric, (metric, "", ""))
    short = model.split("/")[-1]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for arm, col in zip(arms, (C_ALT, C_MAIN)):
        s = d[d.arm == arm].sort_values("concurrency")
        if s.empty:
            continue
        ax.plot(s.concurrency, s[f"{metric}_mean"], "-o", lw=1.8, ms=5,
                color=col, label=arm, zorder=3)
    c = sorted(int(v) for v in d.concurrency.unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(c)
    ax.set_xticklabels([str(v) for v in c])
    ax.minorticks_off()
    ax.set_xlabel("concurrency (requests in flight)")
    ax.set_ylabel(f"{name} ({unit})")
    ax.set_title(f"{name} - {short} {dtype}, {prof} profile", fontsize=10.5, loc="left")
    ax.legend(fontsize=8)
    style(ax)
    fig.text(0.008, -0.055, workload_footnote(d, ctx, arms),
             ha="left", va="top", fontsize=7.5, color=MUTED, linespacing=1.6)
    save(fig, out, fname(short, dtype, d, name),
         {"model": model, "dtype": dtype, "profile": prof, "metric": metric})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--metrics",
                default="mean_tpot_ms,mean_ttft_ms,mean_e2el_ms,output_throughput")
    ap.add_argument("--arms", default="baseline,treatment")
    a = ap.parse_args()
    run = pathlib.Path(a.run_dir)
    out = pathlib.Path(a.out_dir) if a.out_dir else run / "figures"
    agg = run / "normalized" / "e2e_agg.csv"
    if not agg.exists():
        print("plot: no e2e_agg.csv; run normalize first")
        return
    g = pd.read_csv(agg)
    raw = pd.read_csv(run / "normalized" / "e2e_bench.csv")
    arms = a.arms.split(",")

    # Footnote carries the run's actual settings, not the defaults.
    first = raw[raw.get("status", "ok") == "ok"].head(1)
    def pick(col, dflt=""):
        return str(first[col].iloc[0]) if col in first and len(first) else dflt
    ctx = {
        "backends": [
            raw[raw.arm == arms[0]].gemm_backends.dropna().iloc[0]
            if "gemm_backends" in raw and (raw.arm == arms[0]).any() else "ATEN,TRITON",
            raw[raw.arm == arms[1]].gemm_backends.dropna().iloc[0]
            if "gemm_backends" in raw and (raw.arm == arms[1]).any() else "ATEN,TRITON,FLYDSL",
        ],
        "search_space": pick("autotune_search_space", "DEFAULT"),
        "origami": pick("origami", "?"), "origami_topk": pick("origami_topk", "?"),
        "tp": pick("tp_size", "1"), "chunk": pick("max_num_batched_tokens", "?"),
    }

    # isl/osl live on the raw rows; bring them onto the aggregate for the footnote.
    if "isl" in raw.columns:
        key = ["model_id", "dtype", "profile", "concurrency"]
        g = g.merge(raw[key + ["isl", "osl"]].drop_duplicates(key), on=key, how="left")

    print(f"figures -> {out}")
    order = sorted(g.model_id.unique())
    for (model, dtype, prof), d in g.groupby(["model_id", "dtype", "profile"]):
        mcolor = MODEL_COLORS[order.index(model) % len(MODEL_COLORS)]
        for m in a.metrics.split(","):
            # A short prompt makes TTFT a measure of scheduling overhead rather
            # than of prefill GEMM time; plotting it invites a prefill reading
            # the data cannot support.
            # Prefill is ISL x concurrency tokens wide, not ISL: a short ISL
            # still exercises prefill once enough requests are in flight.
            prefill_max = int(d.isl.iloc[0]) * int(d.concurrency.max())
            if m == "mean_ttft_ms" and prefill_max < 512:
                print(f"  skip TTFT: widest prefill is {prefill_max} tokens")
                continue
            if f"{m}_speedup" in d.columns:
                speedup_bars(d, out, model, dtype, prof, m, arms, ctx, mcolor)
                absolute_lines(d, out, model, dtype, prof, m, arms, ctx)
    print("done")


if __name__ == "__main__":
    main()
