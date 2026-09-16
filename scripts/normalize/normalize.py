"""Aggregate the raw per-point JSON into two tidy CSVs.

e2e_bench.csv  one row per measured point
e2e_agg.csv    one row per (model, dtype, profile, concurrency, arm) cell, with
               the arm-relative speedup for every reported metric
"""
import argparse
import json
import pathlib

import pandas as pd

# Speedup orientation: latency inverts (baseline/treatment), throughput does not.
# Reversing this silently inverts every reported result.
LOWER_IS_BETTER = {"mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
                   "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
                   "mean_itl_ms", "mean_e2el_ms"}
HIGHER_IS_BETTER = {"output_throughput", "total_token_throughput",
                    "request_throughput"}

CELL = ["model_id", "dtype", "profile", "concurrency"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--baseline-arm", default="baseline")
    a = ap.parse_args()
    run = pathlib.Path(a.run_dir)
    out = run / "normalized"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted((run / "raw" / "e2e").glob("*.json")):
        try:
            rows.append(json.load(open(f)))
        except (OSError, ValueError):
            print(f"normalize: unreadable, skipped: {f.name}")
    if not rows:
        print("normalize: no raw points")
        return
    df = pd.DataFrame(rows)
    df["schema_version"] = "e2e/v2"
    df.to_csv(out / "e2e_bench.csv", index=False)

    ok = df[df.get("status", "ok") == "ok"]
    metrics = [m for m in (LOWER_IS_BETTER | HIGHER_IS_BETTER) if m in ok.columns]
    if not metrics:
        print("normalize: no known metric columns present")
        return

    g = ok.groupby(CELL + ["arm"], dropna=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
        n=(metrics[0], "count"),
    ).reset_index()

    # Failures are counted, never averaged away: a cell dropping to a smaller n
    # otherwise reads as a complete measurement.
    fail = df[df.get("status", "ok") != "ok"]
    if len(fail):
        nf = fail.groupby(CELL + ["arm"], dropna=False).size().reset_index(name="n_failed")
        g = g.merge(nf, on=CELL + ["arm"], how="left")
    g["n_failed"] = g.get("n_failed", 0)
    g["n_failed"] = g["n_failed"].fillna(0).astype(int)
    if int(g["n_failed"].sum()):
        print(f"e2e_agg: {int(g['n_failed'].sum())} failed point(s), excluded from "
              "the means (affected cells have a reduced n)")

    # Speedup, oriented per metric.
    base = g[g["arm"] == a.baseline_arm]
    for m in metrics:
        b = base[CELL + [f"{m}_mean"]].rename(columns={f"{m}_mean": "_b"})
        g = g.merge(b, on=CELL, how="left")
        g[f"{m}_speedup"] = (g["_b"] / g[f"{m}_mean"] if m in LOWER_IS_BETTER
                             else g[f"{m}_mean"] / g["_b"])
        g = g.drop(columns=["_b"])

    # KV capacity must be identical across arms in a cell; otherwise a throughput
    # difference also reflects a memory difference and is not a backend result.
    if "kv_cache_tokens" in ok.columns:
        kv = ok.groupby(CELL, dropna=False)["kv_cache_tokens"].nunique()
        bad = kv[kv > 1]
        if len(bad):
            print(f"e2e_agg: WARNING — KV capacity differs across arms in {len(bad)} "
                  "cell(s); throughput there includes a memory difference, not just "
                  "a GEMM difference. Pin it with scripts/bench/calibrate_kv.sh.")
            g = g.merge(kv.rename("kv_distinct").reset_index(), on=CELL, how="left")
            g["kv_consistent"] = g["kv_distinct"] == 1
        else:
            print(f"e2e_agg: KV capacity identical across arms in all {len(kv)} cells")
            g["kv_consistent"] = True
    else:
        print("e2e_agg: WARNING — no kv_cache_tokens parsed; consistency unverified")
        g["kv_consistent"] = False

    g.to_csv(out / "e2e_agg.csv", index=False)
    print(f"e2e_bench: {len(df)} rows -> {out / 'e2e_bench.csv'}")
    print(f"e2e_agg:   {len(g)} rows -> {out / 'e2e_agg.csv'}")


if __name__ == "__main__":
    main()
