"""Phase 7 — ablation runner.

Two modes:

1. **Default (no ``--milestone``)** — runs the legacy 4-config sweep on a
   single benchmark::

        A. initial.plc
        B. LKHPlacer + HPWL surrogate only
        C. LKHPlacer + CostApproximator
        D. LKHPlacer + CostApproximator + ChainPolicy

2. **Milestone mode (``--milestone {A,B,C,D,E}``)** — added by the LKH
   critique fix plan §"Validation harness". Selects one configuration
   matching a fix milestone, runs it across all benchmarks listed in
   ``--benchmark``, and (optionally) appends a row to a JSON history file
   so the per-milestone delta against any previous run can be computed
   later::

        --milestone A  -> surrogate only, gate_mode=hpwl
        --milestone B  -> surrogate only, gate_mode=hpwl   (perf-only fix)
        --milestone C  -> + approximator, gate_mode=hpwl
        --milestone D  -> + approximator + policy, gate_mode=hpwl
        --milestone E  -> + approximator + policy, gate_mode=predicted_proxy

   Baseline (pre-fix code) is intentionally absent: the current codebase
   no longer carries the pre-A semantics on the inference path. To get a
   true baseline number, ``git checkout`` the pre-Milestone-A commit and
   run with ``--milestone A`` against that checkout.

Usage::

    uv run python submissions/lkh/ablate.py
    uv run python submissions/lkh/ablate.py --benchmark ibm03 --time-budget 30
    uv run python submissions/lkh/ablate.py --milestone E \\
        --benchmark ibm01,ibm05,ibm07,ibm10 \\
        --time-budget 60 --history-file ablate_history.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

_spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE.parent / "placer.py"))
placer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(placer_mod)


# Default checkpoint paths used by Milestone mode and the legacy 4-config
# sweep. ``BOGUS`` forces the corresponding loader inside LKHPlacer to skip.
_BOGUS_CKPT = Path("/dev/null/does_not_exist.pt")
_APPROX_CKPT = _HERE.parent / "checkpoints" / "cost_approximator.pt"
_POLICY_CKPT = _HERE.parent / "checkpoints" / "chain_policy.pt"


def _eval_placement(name: str, placement, benchmark, plc, runtime: float):
    costs = compute_proxy_cost(placement, benchmark, plc)
    return {
        "method": name,
        "proxy": float(costs["proxy_cost"]),
        "wl": float(costs["wirelength_cost"]),
        "den": float(costs["density_cost"]),
        "cong": float(costs["congestion_cost"]),
        "overlaps": int(costs["overlap_count"]),
        "runtime": runtime,
    }


# Per-milestone config: (label, approx_path, policy_path, use_policy, gate_mode).
# ``None`` for an approximator/policy path means "skip" (we pass _BOGUS_CKPT).
_MILESTONE_CONFIG = {
    "A": ("A: Milestone-A surrogate", None, None, False, "hpwl"),
    "B": ("B: Milestone-B surrogate", None, None, False, "hpwl"),
    "C": ("C: Milestone-C +approx", _APPROX_CKPT, None, False, "hpwl"),
    "D": ("D: Milestone-D +policy", _APPROX_CKPT, _POLICY_CKPT, True, "hpwl"),
    "E": ("E: Milestone-E predproxy", _APPROX_CKPT, _POLICY_CKPT, True, "predicted_proxy"),
}


def _build_placer(*, milestone: str, seed: int, time_budget_s: float,
                  max_chains: int, max_chain_length: int):
    """Instantiate an LKHPlacer for the requested milestone config."""
    if milestone not in _MILESTONE_CONFIG:
        raise ValueError(f"unknown milestone {milestone!r}; "
                         f"expected one of {sorted(_MILESTONE_CONFIG)}")
    _, approx, policy, use_policy, gate_mode = _MILESTONE_CONFIG[milestone]
    approx_path = str(approx) if approx is not None else str(_BOGUS_CKPT)
    policy_path = str(policy) if policy is not None else str(_BOGUS_CKPT)
    return placer_mod.LKHPlacer(
        seed=seed, time_budget_s=time_budget_s,
        max_chains=max_chains, max_chain_length=max_chain_length,
        checkpoint_path=approx_path, policy_path=policy_path,
        use_policy=use_policy, gate_mode=gate_mode,
    )


def run_milestone(milestone: str, benchmark_names: list[str], *,
                  time_budget_s: float, max_chains: int,
                  max_chain_length: int, seed: int) -> dict:
    """Run one milestone config across multiple benchmarks.

    Returns a dict shaped for ``--history-file`` consumption::

        {"milestone": "E", "params": {...},
         "per_benchmark": {"ibm01": {proxy/wl/den/cong/overlaps/runtime},
                           "ibm05": {...}, ...},
         "aggregate": {"proxy_mean": ..., "overlaps_total": ...,
                       "runtime_total": ...}}
    """
    label = _MILESTONE_CONFIG[milestone][0]
    print(f"\n=== Milestone-mode ablation: {label} ===")
    per_benchmark: dict[str, dict] = {}
    for bench_name in benchmark_names:
        bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / bench_name
        if not bench_dir.exists():
            print(f"  [skip] {bench_name}: {bench_dir} not found")
            continue
        print(f"\n[{bench_name}] loading benchmark…")
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))
        placer = _build_placer(milestone=milestone, seed=seed,
                               time_budget_s=time_budget_s,
                               max_chains=max_chains,
                               max_chain_length=max_chain_length)
        t0 = time.time()
        placement = placer.place(benchmark)
        runtime = time.time() - t0
        per_benchmark[bench_name] = _eval_placement(
            label, placement, benchmark, plc, runtime,
        )
        r = per_benchmark[bench_name]
        print(f"  proxy={r['proxy']:.4f}  wl={r['wl']:.3f}  den={r['den']:.3f}  "
              f"cong={r['cong']:.3f}  overlaps={r['overlaps']}  time={r['runtime']:.1f}s")

    if per_benchmark:
        proxies = [r["proxy"] for r in per_benchmark.values()]
        overlaps = sum(r["overlaps"] for r in per_benchmark.values())
        runtimes = sum(r["runtime"] for r in per_benchmark.values())
        aggregate = {
            "proxy_mean": sum(proxies) / len(proxies),
            "proxy_min": min(proxies),
            "proxy_max": max(proxies),
            "overlaps_total": overlaps,
            "runtime_total": runtimes,
            "n_benchmarks": len(per_benchmark),
        }
    else:
        aggregate = {"proxy_mean": float("nan"), "n_benchmarks": 0,
                     "overlaps_total": 0, "runtime_total": 0.0}

    return {
        "milestone": milestone,
        "params": {
            "time_budget_s": time_budget_s,
            "max_chains": max_chains,
            "max_chain_length": max_chain_length,
            "seed": seed,
        },
        "per_benchmark": per_benchmark,
        "aggregate": aggregate,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _print_milestone_summary(record: dict, history_path: Path | None) -> None:
    label = _MILESTONE_CONFIG[record["milestone"]][0]
    print(f"\n=== Milestone {record['milestone']} summary: {label} ===")
    if not record["per_benchmark"]:
        print("  (no benchmarks ran)")
        return
    print(f"  {'benchmark':<12} {'proxy':>8} {'wl':>7} {'den':>7} {'cong':>7} "
          f"{'overlaps':>8} {'time(s)':>8}")
    print("  " + "-" * 64)
    for name, r in record["per_benchmark"].items():
        print(f"  {name:<12} {r['proxy']:>8.4f} {r['wl']:>7.3f} {r['den']:>7.3f} "
              f"{r['cong']:>7.3f} {r['overlaps']:>8d} {r['runtime']:>8.1f}")
    agg = record["aggregate"]
    print(f"\n  aggregate: proxy_mean={agg['proxy_mean']:.4f}  "
          f"overlaps_total={agg['overlaps_total']}  "
          f"runtime_total={agg['runtime_total']:.1f}s  "
          f"n={agg['n_benchmarks']}")

    if history_path is None:
        return
    history: list[dict] = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
            if not isinstance(history, list):
                print(f"  [warn] {history_path} is not a JSON list, overwriting")
                history = []
        except json.JSONDecodeError:
            print(f"  [warn] {history_path} is malformed, overwriting")
            history = []

    # Compare against the most recent matching-config record (same set of
    # benchmarks). The plan's success criteria are stated as "delta vs.
    # previous milestone", so compute that delta if we can find one.
    prev = None
    for past in reversed(history):
        if past.get("milestone") == record["milestone"]:
            continue  # skip same-milestone re-runs
        if set(past.get("per_benchmark", {}).keys()) == set(record["per_benchmark"].keys()):
            prev = past
            break
    if prev is not None:
        prev_agg = prev["aggregate"]
        d_proxy = record["aggregate"]["proxy_mean"] - prev_agg["proxy_mean"]
        d_overlaps = record["aggregate"]["overlaps_total"] - prev_agg["overlaps_total"]
        d_runtime = record["aggregate"]["runtime_total"] - prev_agg["runtime_total"]
        sign = lambda x: "+" if x >= 0 else ""
        print(f"\n  delta vs. previous milestone={prev['milestone']!r}: "
              f"Δproxy_mean={sign(d_proxy)}{d_proxy:.4f}  "
              f"Δoverlaps={sign(d_overlaps)}{d_overlaps}  "
              f"Δruntime={sign(d_runtime)}{d_runtime:.1f}s")

    history.append(record)
    history_path.write_text(json.dumps(history, indent=2))
    print(f"  history appended -> {history_path}")


# ---------------------------------------------------------------------------
# Legacy 4-config sweep, retained for back-compat callers.

def run_ablation(benchmark_name: str, *, time_budget_s: float,
                 max_chains: int, max_chain_length: int, seed: int) -> list[dict]:
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark_name
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    bogus = str(_BOGUS_CKPT)
    approx_ckpt = str(_APPROX_CKPT)
    policy_ckpt = str(_POLICY_CKPT)

    results = []

    print("\n[A] Initial placement (no placer)")
    t0 = time.time()
    placement = benchmark.macro_positions.clone()
    results.append(_eval_placement("A: initial.plc", placement, benchmark, plc,
                                    time.time() - t0))

    print("\n[B] LKHPlacer — HPWL surrogate only (no checkpoints)")
    placer = placer_mod.LKHPlacer(
        seed=seed, time_budget_s=time_budget_s, max_chains=max_chains,
        max_chain_length=max_chain_length,
        checkpoint_path=bogus, policy_path=bogus,
        use_policy=False,
    )
    t0 = time.time()
    placement = placer.place(benchmark)
    results.append(_eval_placement("B: surrogate", placement, benchmark, plc,
                                    time.time() - t0))

    print("\n[C] LKHPlacer — CostApproximator only")
    placer = placer_mod.LKHPlacer(
        seed=seed, time_budget_s=time_budget_s, max_chains=max_chains,
        max_chain_length=max_chain_length,
        checkpoint_path=approx_ckpt, policy_path=bogus,
        use_policy=False,
    )
    t0 = time.time()
    placement = placer.place(benchmark)
    results.append(_eval_placement("C: + approximator", placement, benchmark, plc,
                                    time.time() - t0))

    print("\n[D] LKHPlacer — CostApproximator + ChainPolicy")
    placer = placer_mod.LKHPlacer(
        seed=seed, time_budget_s=time_budget_s, max_chains=max_chains,
        max_chain_length=max_chain_length,
        checkpoint_path=approx_ckpt, policy_path=policy_ckpt,
        use_policy=True,
    )
    t0 = time.time()
    placement = placer.place(benchmark)
    results.append(_eval_placement("D: + policy", placement, benchmark, plc,
                                    time.time() - t0))

    return results


def _print_table(results: list[dict], benchmark_name: str):
    print(f"\n=== Phase 7 ablation: {benchmark_name} ===")
    print(f"  {'method':<22} {'proxy':>8} {'wl':>7} {'den':>7} {'cong':>7} "
          f"{'overlaps':>8} {'time(s)':>8}")
    print("  " + "-" * 72)
    base_proxy = results[0]["proxy"] if results else 0.0
    for r in results:
        delta = (r["proxy"] - base_proxy)
        sign = "+" if delta >= 0 else ""
        print(f"  {r['method']:<22} {r['proxy']:>8.4f} {r['wl']:>7.3f} "
              f"{r['den']:>7.3f} {r['cong']:>7.3f} "
              f"{r['overlaps']:>8d} {r['runtime']:>8.1f}  "
              f"(Δproxy = {sign}{delta:.4f})")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="ibm01",
                   help="Single benchmark name OR comma-separated list "
                        "(e.g. ibm01,ibm05,ibm07,ibm10). Comma form is "
                        "only used in --milestone mode.")
    p.add_argument("--time-budget", type=float, default=15.0,
                   help="Seconds per placer run (per benchmark).")
    p.add_argument("--max-chains", type=int, default=2000)
    p.add_argument("--max-chain-length", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--milestone", choices=sorted(_MILESTONE_CONFIG),
                   default=None,
                   help="If set, run a single milestone config across the "
                        "benchmarks listed in --benchmark instead of the "
                        "legacy 4-config sweep.")
    p.add_argument("--history-file", default=None,
                   help="JSON file to append per-milestone records to. "
                        "Used to compute Δ-vs-previous-milestone in subsequent "
                        "runs. Only consulted in --milestone mode.")
    args = p.parse_args()

    if args.milestone is None:
        # Legacy single-benchmark 4-config sweep.
        print(f"=== Phase 7 ablation runner ===")
        print(f"  benchmark={args.benchmark}  time_budget={args.time_budget}s")
        results = run_ablation(
            args.benchmark,
            time_budget_s=args.time_budget,
            max_chains=args.max_chains,
            max_chain_length=args.max_chain_length,
            seed=args.seed,
        )
        _print_table(results, args.benchmark)
        return

    bench_names = [s.strip() for s in args.benchmark.split(",") if s.strip()]
    print(f"=== Phase 7 milestone ablation runner ===")
    print(f"  milestone={args.milestone}  benchmarks={bench_names}  "
          f"time_budget={args.time_budget}s  max_chains={args.max_chains}")
    record = run_milestone(
        args.milestone, bench_names,
        time_budget_s=args.time_budget,
        max_chains=args.max_chains,
        max_chain_length=args.max_chain_length,
        seed=args.seed,
    )
    history_path = Path(args.history_file) if args.history_file else None
    _print_milestone_summary(record, history_path)


if __name__ == "__main__":
    main()
