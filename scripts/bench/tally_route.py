"""Count what select_algorithm picked, from the autotune log.

The verdict is the JSON summary line ending each autotune, not the line after
`AUTOTUNE mm(` (that one is `strides:`).

Participation, not winning, is the signal: ATen contributes exactly one
candidate, so num_choices - num_triton_choices > 1 means a FlyDSL candidate
entered the set. Winning is the property under measurement, so gating on it
would fail the harness for the result it exists to report.
"""
import json
import sys

total = participated = wins = 0
kernels = []
for line in open(sys.argv[1], errors="replace"):
    i = line.find('{"num_choices"')
    if i < 0:
        continue
    try:
        d = json.loads(line[i : line.rindex("}") + 1])
    except ValueError:
        continue
    total += 1
    if d.get("num_choices", 0) - d.get("num_triton_choices", 0) > 1:
        participated += 1
    best = str(d.get("best_kernel", ""))
    if "flydsl" in best:
        wins += 1
        kernels.append(best)

out = {
    "autotune_total": total,
    "flydsl_participated": participated,
    "flydsl_wins": wins,
    "flydsl_win_rate_among_participated": round(wins / participated, 4) if participated else None,
    "flydsl_kernels": kernels,
}
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=2)
print(json.dumps(out, indent=2))
