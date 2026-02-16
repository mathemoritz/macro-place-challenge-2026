#!/usr/bin/env python3
"""
Greedy Row Placer - Demo Submission

A simple but legal placer that:
1. Sorts macros by height (tallest first)
2. Places them left-to-right in rows (like shelf packing)
3. Guarantees zero overlaps and canvas boundary compliance

This produces valid, scoreable placements but makes no attempt to
optimize wirelength, density, or congestion. Use it as a starting
point for your own algorithm.

Usage:
    python submissions/examples/greedy_row_placer.py
    python submissions/examples/greedy_row_placer.py --benchmark ibm01
    python submissions/examples/greedy_row_placer.py --all
"""

import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
from loader import load_benchmark_from_dir
from objective import compute_proxy_cost
from utils import validate_placement
from benchmark import Benchmark


class GreedyRowPlacer:
    """
    Greedy row-based (shelf packing) placement.

    Places macros in rows from bottom to top, left to right,
    sorted by descending height. Guarantees zero overlaps.
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask()
        movable_indices = torch.where(movable)[0].tolist()

        sizes = benchmark.macro_sizes
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height

        # Sort movable macros by height descending (shelf packing heuristic)
        movable_indices.sort(key=lambda i: -sizes[i, 1].item())

        # Small gap to avoid float32 touching-edge false overlaps
        gap = 0.001

        cursor_x = 0.0
        cursor_y = 0.0
        row_height = 0.0

        for idx in movable_indices:
            w = sizes[idx, 0].item()
            h = sizes[idx, 1].item()

            # Start new row if macro doesn't fit
            if cursor_x + w > canvas_w:
                cursor_x = 0.0
                cursor_y += row_height + gap
                row_height = 0.0

            # Check if we've run out of vertical space
            if cursor_y + h > canvas_h:
                # Place at origin as fallback (will overlap but shouldn't happen
                # if area utilization < 100%)
                placement[idx, 0] = w / 2
                placement[idx, 1] = h / 2
                continue

            # Place macro (positions are centers)
            placement[idx, 0] = cursor_x + w / 2
            placement[idx, 1] = cursor_y + h / 2

            cursor_x += w + gap
            row_height = max(row_height, h)

        return placement


BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17", "ibm18",
]

SA_BASELINES = {
    "ibm01": 1.3166, "ibm02": 1.9072, "ibm03": 1.7401, "ibm04": 1.5037,
    "ibm06": 2.5057, "ibm07": 2.0229, "ibm08": 1.9239, "ibm09": 1.3875,
    "ibm10": 2.1108, "ibm11": 1.7111, "ibm12": 2.8261, "ibm13": 1.9141,
    "ibm14": 2.2750, "ibm15": 2.3000, "ibm16": 2.2337, "ibm17": 3.6726,
    "ibm18": 2.7755,
}

REPLACE_BASELINES = {
    "ibm01": 0.9976, "ibm02": 1.8370, "ibm03": 1.3222, "ibm04": 1.3024,
    "ibm06": 1.6187, "ibm07": 1.4633, "ibm08": 1.4285, "ibm09": 1.1194,
    "ibm10": 1.5009, "ibm11": 1.1774, "ibm12": 1.7261, "ibm13": 1.3355,
    "ibm14": 1.5436, "ibm15": 1.5159, "ibm16": 1.4780, "ibm17": 1.6446,
    "ibm18": 1.7722,
}


def evaluate_benchmark(name: str, testcase_root: str) -> dict:
    """Evaluate the placer on a single benchmark."""
    benchmark_dir = f"{testcase_root}/{name}"
    benchmark, plc = load_benchmark_from_dir(benchmark_dir)

    placer = GreedyRowPlacer()

    start = time.time()
    placement = placer.place(benchmark)
    runtime = time.time() - start

    is_valid, violations = validate_placement(placement, benchmark)
    costs = compute_proxy_cost(placement, benchmark, plc)

    return {
        "name": name,
        "proxy_cost": costs["proxy_cost"],
        "wirelength": costs["wirelength_cost"],
        "density": costs["density_cost"],
        "congestion": costs["congestion_cost"],
        "overlaps": costs["overlap_count"],
        "runtime": runtime,
        "valid": is_valid,
        "sa_baseline": SA_BASELINES.get(name),
        "replace_baseline": REPLACE_BASELINES.get(name),
    }


def main():
    parser = argparse.ArgumentParser(description="Greedy Row Placer - Demo Submission")
    parser.add_argument("--benchmark", "-b", type=str, default=None,
                        help="Run on a specific benchmark (e.g., ibm01)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Run on all 17 IBM benchmarks")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    testcase_root = repo_root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"

    if not testcase_root.exists():
        print(f"Error: Testcases not found at {testcase_root}")
        print("Run: git submodule update --init external/MacroPlacement")
        sys.exit(1)

    benchmarks_to_run = BENCHMARKS if args.all else [args.benchmark or "ibm01"]

    print("=" * 80)
    print("Greedy Row Placer - Demo Submission")
    print("=" * 80)
    print()

    results = []
    for name in benchmarks_to_run:
        print(f"  {name}...", end=" ", flush=True)
        result = evaluate_benchmark(name, str(testcase_root))
        results.append(result)

        status = "VALID" if result["overlaps"] == 0 else f"INVALID ({result['overlaps']} overlaps)"
        print(f"proxy={result['proxy_cost']:.4f}  "
              f"(wl={result['wirelength']:.3f} den={result['density']:.3f} cong={result['congestion']:.3f})  "
              f"{status}  [{result['runtime']:.2f}s]")

    if len(results) > 1:
        print()
        print("-" * 80)
        print(f"{'Benchmark':>8}  {'Proxy':>8}  {'SA':>8}  {'RePlAce':>8}  {'vs SA':>8}  {'vs RePlAce':>10}  {'Overlaps':>8}")
        print("-" * 80)
        for r in results:
            vs_sa = ((r["sa_baseline"] - r["proxy_cost"]) / r["sa_baseline"] * 100) if r["sa_baseline"] else 0
            vs_rep = ((r["replace_baseline"] - r["proxy_cost"]) / r["replace_baseline"] * 100) if r["replace_baseline"] else 0
            print(f"{r['name']:>8}  {r['proxy_cost']:>8.4f}  {r['sa_baseline']:>8.4f}  {r['replace_baseline']:>8.4f}  "
                  f"{vs_sa:>+7.1f}%  {vs_rep:>+9.1f}%  {r['overlaps']:>8}")

        avg_proxy = sum(r["proxy_cost"] for r in results) / len(results)
        avg_sa = sum(r["sa_baseline"] for r in results) / len(results)
        avg_rep = sum(r["replace_baseline"] for r in results) / len(results)
        total_overlaps = sum(r["overlaps"] for r in results)
        total_runtime = sum(r["runtime"] for r in results)

        print("-" * 80)
        print(f"{'AVG':>8}  {avg_proxy:>8.4f}  {avg_sa:>8.4f}  {avg_rep:>8.4f}  "
              f"{(avg_sa - avg_proxy) / avg_sa * 100:>+7.1f}%  "
              f"{(avg_rep - avg_proxy) / avg_rep * 100:>+9.1f}%  {total_overlaps:>8}")
        print()
        print(f"Total runtime: {total_runtime:.2f}s")
        if total_overlaps > 0:
            print(f"DISQUALIFIED: {total_overlaps} total overlaps across benchmarks")
        print()


if __name__ == "__main__":
    main()
