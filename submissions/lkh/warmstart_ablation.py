"""Warm-start ablation.

Is our placer actually *optimizing*, or just polishing a placement
(initial.plc) that already sits near proxy 1.46? To find out, run the exact
same LKHPlacer (same checkpoints, same time budget) from three different
starting points and compare the real proxy cost the official evaluator reports:

    initial    — the challenge-provided initial.plc (our competition entry)
    random     — movable hard macros scattered uniformly at random
    analytical — a self-contained quadratic / force-directed global placement

If `random` and `analytical` land far worse than `initial`, the method mostly
polishes a good seed. If they converge near `initial`, the chain loop has real
optimization power and the warm start is not the bottleneck.

Usage:
    uv run python submissions/lkh/warmstart_ablation.py --benchmark ibm01
    uv run python submissions/lkh/warmstart_ablation.py \
        --benchmark ibm01,ibm07,ibm13 --time-budget 30
    uv run python submissions/lkh/warmstart_ablation.py --benchmark ibm01 \
        --init-modes initial,analytical --seeds 3
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

_spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
_placer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_placer)
LKHPlacer = _placer.LKHPlacer

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from train import parse_benchmarks  # shared CLI helper


def _eval_one(benchmark, plc, *, init_mode: str, time_budget_s: float,
              seed: int) -> dict:
    placer = LKHPlacer(seed=seed, time_budget_s=time_budget_s,
                       max_chains=5000, max_chain_length=8,
                       init_mode=init_mode)
    t0 = time.time()
    placement = placer.place(benchmark)
    runtime = time.time() - t0
    costs = compute_proxy_cost(placement, benchmark, plc)
    return {
        "proxy_cost": float(costs["proxy_cost"]),
        "wirelength": float(costs["wirelength_cost"]),
        "density": float(costs["density_cost"]),
        "congestion": float(costs["congestion_cost"]),
        "overlaps": int(costs["overlap_count"]),
        "runtime_s": runtime,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="ibm01",
                   help="Single name, comma-separated list, or 'all'.")
    p.add_argument("--init-modes", default="initial,random,analytical",
                   help="Comma-separated subset of initial,random,analytical.")
    p.add_argument("--time-budget", type=float, default=30.0,
                   help="Seconds per placer run.")
    p.add_argument("--seeds", type=int, default=1,
                   help="Number of seeds per (benchmark, init_mode); results "
                        "are averaged. random/analytical benefit from >1.")
    p.add_argument("--plot", default="",
                   help="If set, write a grouped-bar PNG of proxy by init_mode.")
    p.add_argument("--json", default="",
                   help="If set, dump the results dict to this JSON path.")
    args = p.parse_args()

    benchmarks = parse_benchmarks(args.benchmark)
    init_modes = [m.strip() for m in args.init_modes.split(",") if m.strip()]

    print(f"=== Warm-start ablation ===")
    print(f"  benchmarks={benchmarks}  init_modes={init_modes}  "
          f"time_budget={args.time_budget}s  seeds={args.seeds}")

    # results[bench][mode] = averaged dict
    results: dict[str, dict[str, dict]] = {}
    for name in benchmarks:
        bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))
        results[name] = {}
        for mode in init_modes:
            runs = [
                _eval_one(benchmark, plc, init_mode=mode,
                          time_budget_s=args.time_budget, seed=42 + s)
                for s in range(args.seeds)
            ]
            agg = {
                k: float(np.mean([r[k] for r in runs]))
                for k in ("proxy_cost", "wirelength", "density",
                          "congestion", "runtime_s")
            }
            agg["overlaps"] = int(max(r["overlaps"] for r in runs))
            agg["proxy_std"] = float(np.std([r["proxy_cost"] for r in runs]))
            results[name][mode] = agg

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'benchmark':>10} {'init_mode':>11} {'proxy':>8} {'±std':>7} "
          f"{'overlaps':>9} {'runtime':>8}  Δ vs initial")
    print("-" * 72)
    for name in benchmarks:
        base = results[name].get("initial", {}).get("proxy_cost")
        for mode in init_modes:
            r = results[name][mode]
            if base is not None and mode != "initial" and base > 0:
                delta = f"{(r['proxy_cost'] - base) / base * 100:+.1f}%"
            else:
                delta = "—"
            print(f"{name:>10} {mode:>11} {r['proxy_cost']:>8.4f} "
                  f"{r['proxy_std']:>7.4f} {r['overlaps']:>9} "
                  f"{r['runtime_s']:>7.1f}s  {delta:>10}")

    # Aggregate across benchmarks (only if every benchmark ran every mode).
    if len(benchmarks) > 1:
        print(f"\n{'AVG':>10}")
        for mode in init_modes:
            proxies = [results[n][mode]["proxy_cost"] for n in benchmarks]
            tot_ov = sum(results[n][mode]["overlaps"] for n in benchmarks)
            print(f"{'':>10} {mode:>11} {np.mean(proxies):>8.4f} "
                  f"{'':>7} {tot_ov:>9}")

    if args.json:
        import json
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"time_budget_s": args.time_budget, "seeds": args.seeds,
             "results": results}, indent=2))
        print(f"\n[warmstart] results -> {args.json}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"initial": "#2ca02c", "random": "#d62728",
                  "analytical": "#1f77b4"}
        x = np.arange(len(benchmarks))
        width = 0.8 / max(len(init_modes), 1)
        fig, ax = plt.subplots(figsize=(1.6 * len(benchmarks) + 3, 4.5))
        for k, mode in enumerate(init_modes):
            vals = [results[n][mode]["proxy_cost"] for n in benchmarks]
            errs = [results[n][mode].get("proxy_std", 0.0) for n in benchmarks]
            ax.bar(x + k * width, vals, width, yerr=errs, capsize=3,
                   label=mode, color=colors.get(mode, None))
        ax.set_xticks(x + width * (len(init_modes) - 1) / 2)
        ax.set_xticklabels(benchmarks)
        ax.set_ylabel("proxy cost (lower = better)")
        ax.set_title(f"warm-start ablation "
                     f"({args.time_budget:.0f}s budget, {args.seeds} seed(s))")
        ax.axhline(1.4578, color="k", ls="--", lw=1,
                   label="RePlAce baseline (1.458)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=130)
        print(f"[warmstart] plot -> {args.plot}")


if __name__ == "__main__":
    main()
