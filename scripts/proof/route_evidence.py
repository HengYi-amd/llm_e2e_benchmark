"""Route evidence, per arm and per shard: did the GEMM backend actually run?

Match symbols with comment lines stripped, never the raw text: generated
modules carry the cache path in a header comment, so every treatment-arm file
"contains flydsl" regardless of routing.

Candidate and winner are different symbols; confusing them yields a confident
zero. A candidate is `def flydsl_mm_flydsl_main(...)`, compiled during autotune
and called by its own module to benchmark it, so a call proves nothing. A
winner is `<sym> = async_compile.flydsl(...)` in a graph module that `call()`
then runs; mm_modules is the denominator for winners.

self_test() therefore runs positive and negative controls first and blocks the
verdict if either fails: a detector that can only print zero is not evidence.
"""
import json
import os
import pathlib
import re
import sys

CAND_RE = re.compile(r"^\s*def\s+flydsl_mm_flydsl_main\s*\(", re.M)
WIN_DEF_RE = re.compile(r"^\s*(\w+)\s*=\s*async_compile\.flydsl\(", re.M)
MM_RE = re.compile(r"triton_tem_fused|extern_kernels\.(mm|addmm|bmm)|async_compile\.flydsl\(")

# Controls: caches from scripts/bench/route_smoke.sh built with FLYDSL as the
# only backend (must report winners) and without it (must report none).
POS_PROBES = os.environ.get(
    "E2E_PROBE_POS_CACHES", "probe_FLYDSL_M4 probe_FLYDSL_M2").split()
NEG_PROBES = os.environ.get(
    "E2E_PROBE_NEG_CACHES", "probe_ATEN_TRITON_M4 probe_ATEN_TRITON_M2").split()


def repo_root():
    return pathlib.Path(
        os.environ.get("E2E_ROOT") or pathlib.Path(__file__).resolve().parents[2])


def arm_backends(arm):
    return os.environ.get(f"E2E_BACKENDS_{arm}", "")


def arms():
    """Split configured arms into those under test and the rest.

    An arm is under test when its backend list names FLYDSL, the only thing the
    arms differ in, so no second source of truth is needed.
    """
    names = os.environ.get("E2E_ARMS", "baseline treatment").split()
    treatment = [a for a in names if "FLYDSL" in arm_backends(a).upper()]
    if not treatment:
        treatment = names[-1:]
    return treatment, [a for a in names if a not in treatment]


def strip_comments(text):
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


def scan(arm_dir):
    cand = winners = mm_mods = total = 0
    run_calls = 0
    winner_syms = set()
    for p in pathlib.Path(arm_dir).rglob("*.py"):
        try:
            raw = p.read_text(errors="replace")
        except OSError:
            continue
        total += 1
        code = strip_comments(raw)
        if CAND_RE.search(code):
            cand += 1
        syms = WIN_DEF_RE.findall(code)
        if MM_RE.search(code):
            mm_mods += 1
        if syms:
            # Defined is not executed: it counts only once call() runs it.
            hit = [s for s in syms if re.search(rf"\b{re.escape(s)}\.run\(", code)]
            if hit:
                winners += 1
                run_calls += sum(len(re.findall(rf"\b{re.escape(s)}\.run\(", code)) for s in hit)
                winner_syms |= set(hit)
    return {
        "generated_modules": total,
        "candidate_modules": cand,
        "mm_modules": mm_mods,
        "winner_modules": winners,
        "winner_run_calls": run_calls,
        "winner_symbols": sorted(winner_syms),
    }


def self_test(root):
    """Controls. No verdict is emitted unless both directions pass."""
    cases = [(n, "pos") for n in POS_PROBES] + [(n, "neg") for n in NEG_PROBES]
    ok = True
    rows = []
    for name, kind in cases:
        # Same tree the arm scan uses: the engine writes its compiled modules
        # under the vLLM cache root, not the Inductor one.
        d = root / "caches" / "vllm" / name
        if not d.is_dir():
            rows.append((name, kind, "missing", False)); ok = False; continue
        r = scan(d)
        good = r["winner_modules"] > 0 if kind == "pos" else r["winner_modules"] == 0
        ok &= good
        rows.append((name, kind, f"winner={r['winner_modules']} cand={r['candidate_modules']}", good))
    for name, kind, detail, good in rows:
        print(f"[self-test] {'PASS' if good else 'FAIL'} {name} ({kind}) {detail}")
    return ok


def main():
    root = repo_root()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test(root) else 1)
    run_dir = pathlib.Path(args[0]) if args else None

    if not self_test(root):
        print("detector controls failed, refusing to report a route verdict",
              file=sys.stderr)
        sys.exit(1)

    treatment, baseline = arms()
    # One cache directory per arm and shard, named <arm>_g<gpu> by run_e2e.sh.
    out = {}
    for arm in treatment + baseline:
        for d in sorted((root / "caches" / "vllm").glob(f"{arm}_g[0-7]")):
            out[d.name] = scan(d)

    def total(names, key):
        return sum(v[key] for k, v in out.items()
                   if any(k.startswith(f"{a}_g") for a in names))

    summary = {
        "flydsl_candidate_modules": total(treatment, "candidate_modules"),
        "flydsl_winner_modules": total(treatment, "winner_modules"),
        "flydsl_mm_modules": total(treatment, "mm_modules"),
        "baseline_candidate_modules": total(baseline, "candidate_modules"),
        "baseline_winner_modules": total(baseline, "winner_modules"),
    }
    participated = summary["flydsl_candidate_modules"] > 0
    won = summary["flydsl_winner_modules"] > 0
    summary["flydsl_participated"] = participated
    summary["flydsl_won_any"] = won
    if participated and not won:
        summary["verdict"] = (
            "FlyDSL was wired in but won no GEMM: the treatment arm executes "
            "exactly the same kernels as the baseline, so the end-to-end "
            "difference between the arms is measurement noise and concurrency "
            "jitter, not a performance result."
        )
    elif not participated:
        summary["verdict"] = (
            "No FlyDSL candidate was ever compiled: the backend is not in the "
            "picture and this A/B is invalid."
        )
    else:
        summary["verdict"] = (
            f"FlyDSL won {summary['flydsl_winner_modules']}/"
            f"{summary['flydsl_mm_modules']} of the graph modules containing a "
            "GEMM, so the end-to-end difference can be attributed to the kernel."
        )

    res = {"per_arm_shard": out, "summary": summary}
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if run_dir:
        (run_dir / "raw").mkdir(parents=True, exist_ok=True)
        json.dump(res, open(run_dir / "raw" / "route_evidence.json", "w"),
                  indent=2, ensure_ascii=False)

    # The gate has to be able to fail. Writing a verdict that says the
    # comparison is invalid and then exiting 0 is not a gate: downstream stages
    # would publish a backend speedup that nothing shows the backend produced.
    if not summary.get("flydsl_won_any"):
        print("route proof FAILED: the backend never won a GEMM, so the two arms "
              "ran the same kernels", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
