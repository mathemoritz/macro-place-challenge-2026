"""Phase 7 — ablation runner.

Compares four placements on a chosen benchmark:
    A. Initial placement (initial.plc, no placer ran)
    B. LKHPlacer with HPWL surrogate only (no checkpoints used)
    C. LKHPlacer with CostApproximator only (no policy)
    D. LKHPlacer with CostApproximator + ChainPolicy

For each, reports proxy cost components, overlap pair count, and wall time.

The toggles are implemented by passing temporary checkpoint paths to
``LKHPlacer`` — we don't mutate the real checkpoint files.

Usage:
    uv run python submissions/lkh/ablate.py
    uv run python submissions/lkh/ablate.py --benchmark ibm03 --time-budget 30
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

_spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
placer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(placer_mod)


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


def run_ablation(benchmark_name: str, *, time_budget_s: float,
                 max_chains: int, max_chain_length: int, seed: int) -> list[dict]:
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark_name
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    bogus = Path("/dev/null/does_not_exist.pt")  # forces _load_*_ to return None
    approx_ckpt = _HERE / "checkpoints" / "cost_approximator.pt"
    policy_ckpt = _HERE / "checkpoints" / "chain_policy.pt"

    results = []

    # A. Initial placement.
    print("\n[A] Initial placement (no placer)")
    t0 = time.time()
    placement = benchmark.macro_positions.clone()
    results.append(_eval_placement("A: initial.plc", placement, benchmark, plc,
                                    time.time() - t0))

    # B. Surrogate only.
    print("\n[B] LKHPlacer — HPWL surrogate only (no checkpoints)")
    placer = placer_mod.LKHPlacer(
        seed=seed, time_budget_s=time_budget_s, max_chains=max_chains,
        max_chain_length=max_chain_length,
        checkpoint_path=str(bogus), policy_path=str(bogus),
        use_policy=False,
    )
    t0 = time.time()
    placement = placer.place(benchmark)
    results.append(_eval_placement("B: surrogate", placement, benchmark, plc,
                                    time.time() - t0))

    # C. Approximator only.
    print("\n[C] LKHPlacer — CostApproximator only")
    placer = placer_mod.LKHPlacer(
        seed=seed, time_budget_s=time_budget_s, max_chains=max_chains,
        max_chain_length=max_chain_length,
        checkpoint_path=str(approx_ckpt), policy_path=str(bogus),
        use_policy=False,
    )
    t0 = time.time()
    placement = placer.place(benchmark)
    results.append(_eval_placement("C: + approximator", placement, benchmark, plc,
                                    time.time() - t0))

    # D. Approximator + policy.
    print("\n[D] LKHPlacer — CostApproximator + ChainPolicy")
    placer = placer_mod.LKHPlacer(
        seed=seed, time_budget_s=time_budget_s, max_chains=max_chains,
        max_chain_length=max_chain_length,
        checkpoint_path=str(approx_ckpt), policy_path=str(policy_ckpt),
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
    p.add_argument("--benchmark", default="ibm01")
    p.add_argument("--time-budget", type=float, default=15.0,
                   help="Seconds per placer run (B/C/D).")
    p.add_argument("--max-chains", type=int, default=2000)
    p.add_argument("--max-chain-length", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

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


if __name__ == "__main__":
    main()
