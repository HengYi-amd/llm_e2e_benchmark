"""Write the deliverable layout: result/<Model>_<dtype>_data/ holding CSVs and PNGs.

One flat directory per (model, dtype), no nesting. Raw per-point JSON and logs
stay in the working run directory; what lands here is the data someone else
would read.
"""
import argparse
import pathlib
import shutil

import pandas as pd

DTYPE_SHORT = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--suffix", default="_data",
                    help="tail of each result directory name, after "
                         "<model>_<dtype>; e.g. _exhaustive")
    ap.add_argument("--no-figures", action="store_true",
                    help="publish the data but not the figures, for runs "
                         "whose routing proof did not pass")
    a = ap.parse_args()
    run = pathlib.Path(a.run_dir)
    result = pathlib.Path(a.result_dir)
    agg = run / "normalized" / "e2e_agg.csv"
    if not agg.exists():
        print("publish: no e2e_agg.csv; nothing to publish")
        return

    g = pd.read_csv(agg)
    raw = pd.read_csv(run / "normalized" / "e2e_bench.csv")

    for (model, dtype), d in g.groupby(["model_id", "dtype"]):
        short = str(model).split("/")[-1]
        out = result / f"{short}_{DTYPE_SHORT.get(dtype, dtype)}{a.suffix}"
        out.mkdir(parents=True, exist_ok=True)

        d.to_csv(out / "summary.csv", index=False)
        raw[(raw.model_id == model) & (raw.dtype == dtype)].to_csv(
            out / "raw_points.csv", index=False)

        # Figures are named by model and dtype already; copy them flat.
        n = 0
        tag = DTYPE_SHORT.get(dtype, dtype)
        if a.no_figures:
            print(f"publish: {out}  (data only, figures withheld)")
            continue
        for png in sorted((run / "figures").glob(f"{short}_{tag}_*.png")):
            shutil.copy2(png, out / png.name)
            n += 1
        print(f"publish: {out}  (summary.csv, raw_points.csv, {n} figure(s))")


if __name__ == "__main__":
    main()
