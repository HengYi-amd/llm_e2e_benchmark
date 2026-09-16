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
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CRIT = "#d03b3b"

METRIC_LABEL = {
    "mean_tpot_ms": ("TPOT", "time per output token", "decode steady state"),
    "mean_ttft_ms": ("TTFT", "time to first token", "prefill"),
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


def save(fig, out, name, meta):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    (out / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    plt.close(fig)
    print(f"  {name}.png")


def speedup_bars(d, out, model, dtype, prof, metric, arms, ctx):
    col = f"{metric}_speedup"
    t = d[(d.arm == arms[1]) & d[col].notna()].sort_values("concurrency")
    if t.empty:
        return
    sp = t[col].values
    x = np.arange(len(t))
    short = model.split("/")[-1]
    name, unit, phase = METRIC_LABEL.get(metric, (metric, "", ""))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x, sp, width=0.62, color=C_MAIN, zorder=3)
    ax.axhline(1.0, color=AXIS, lw=1.0, zorder=4)
    for xi, v in zip(x, sp):
        if v >= 1.0:
            ax.text(xi, v + 0.006, f"{v:.3f}x", ha="center", va="bottom",
                    fontsize=8.5, color=INK, fontweight="bold", zorder=5)
        else:
            # Above a sub-1.0 bar the label would sit on the 1.0 line and read
            # as parity, so place it inside the bar.
            ax.text(xi, v - 0.008, f"{v:.3f}x", ha="center", va="top",
                    fontsize=8.5, color="#ffffff", fontweight="bold", zorder=5)
    ax.set_ylim(min(0.94, float(sp.min()) - 0.04), float(sp.max()) + 0.06)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in t.concurrency])
    ax.set_xlabel("concurrency (requests in flight)")
    ax.set_ylabel(f"{name} speedup vs baseline")
    gm = float(np.exp(np.log(sp).mean()))
    ax.set_title(f"Adding FlyDSL to the Inductor GEMM backends - {short} {dtype}\n"
                 f"{name} ({phase}): geomean {gm:.3f}x, "
                 f"{int((sp > 1).sum())}/{len(sp)} points faster",
                 fontsize=11, loc="left")
    style(ax)

    nmin, nmax = int(t.n.min()), int(t.n.max())
    reps = f"{nmin}" if nmin == nmax else f"{nmin}-{nmax}"
    nfail = int(t.n_failed.sum()) if "n_failed" in t else 0
    foot = (
        f"Baseline: {ctx['backends'][0]}   |   Treatment: {ctx['backends'][1]}   |   "
        f"torch.compile, max-autotune (search space {ctx['search_space']}, "
        f"origami={ctx['origami']}/top{ctx['origami_topk']})\n"
        f"vllm bench serve, {model}, {dtype}, TP={ctx['tp']}, "
        f"ISL={int(t.isl.iloc[0])} / OSL={int(t.osl.iloc[0])}, "
        f"max_num_batched_tokens={ctx['chunk']}, prefix caching off, ignore_eos\n"
        f"{name} = {unit}; mean of {reps} repeat(s) per point"
        + (f"; {nfail} failed point(s) excluded" if nfail else "")
        + "; higher is better, 1.0 = baseline"
    )
    fig.text(0.008, -0.055, foot, ha="left", va="top",
             fontsize=7.5, color=MUTED, linespacing=1.6)
    save(fig, out, f"speedup_{metric}_{short}_{dtype}_{prof}",
         {"model": model, "dtype": dtype, "profile": prof, "metric": metric,
          "geomean": gm, "wins": int((sp > 1).sum()), "points": len(sp)})


def absolute_lines(d, out, model, dtype, prof, metric, arms):
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
    save(fig, out, f"absolute_{metric}_{short}_{dtype}_{prof}",
         {"model": model, "dtype": dtype, "profile": prof, "metric": metric})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--metrics", default="mean_tpot_ms,mean_ttft_ms,output_throughput")
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
    for (model, dtype, prof), d in g.groupby(["model_id", "dtype", "profile"]):
        for m in a.metrics.split(","):
            if f"{m}_speedup" in d.columns:
                speedup_bars(d, out, model, dtype, prof, m, arms, ctx)
                absolute_lines(d, out, model, dtype, prof, m, arms)
    print("done")


if __name__ == "__main__":
    main()
