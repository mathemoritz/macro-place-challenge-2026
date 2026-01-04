#!/usr/bin/env python3
"""
Minimal evaluation script for macro placement.

Usage:
    python scripts/evaluate.py --benchmark ariane133 --placer random
"""

import argparse
import sys
import time
from pathlib import Path

# Add src and baselines to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "baselines"))

import torch
from marco_place.data.tensor_schema import CircuitTensorData
from marco_place.metrics.wirelength import compute_hpwl_from_circuit
from random_placer import RandomPlacer


def load_benchmark(benchmark_name: str) -> CircuitTensorData:
    """
    Load a benchmark from the processed directory.

    Args:
        benchmark_name: Name of the benchmark (e.g., 'ariane133')

    Returns:
        CircuitTensorData instance
    """
    benchmark_path = project_root / "benchmarks" / "processed" / "public" / f"{benchmark_name}.pt"

    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark not found: {benchmark_path}\n"
            f"Available benchmarks should be in: benchmarks/processed/public/"
        )

    print(f"Loading benchmark from: {benchmark_path}")
    circuit_data = CircuitTensorData.load(str(benchmark_path))
    print(f"Loaded: {circuit_data}")

    return circuit_data


def get_placer(placer_name: str):
    """
    Get placer instance by name.

    Args:
        placer_name: Name of the placer ('random', etc.)

    Returns:
        Placer instance with .place(circuit_data) method
    """
    if placer_name == "random":
        return RandomPlacer()
    else:
        raise ValueError(f"Unknown placer: {placer_name}")


def evaluate(circuit_data: CircuitTensorData, placer, verbose: bool = True):
    """
    Evaluate a placer on a benchmark.

    Args:
        circuit_data: Circuit data
        placer: Placer instance with .place(circuit_data) method
        verbose: Print detailed output

    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("Running placement...")
        print("=" * 70)

    # Run placer
    start_time = time.time()
    placement = placer.place(circuit_data)
    runtime = time.time() - start_time

    if verbose:
        print(f"Placement completed in {runtime:.2f} seconds")
        print(f"Placement shape: {placement.shape}")

    # Compute HPWL
    hpwl = compute_hpwl_from_circuit(placement, circuit_data)

    if verbose:
        print("\n" + "=" * 70)
        print("Evaluation Results")
        print("=" * 70)
        print(f"Design:    {circuit_data.design_name}")
        print(f"Macros:    {circuit_data.num_macros}")
        print(f"Nets:      {circuit_data.num_nets}")
        print(f"Canvas:    {circuit_data.canvas_width:.1f} x {circuit_data.canvas_height:.1f} um")
        print(f"Runtime:   {runtime:.3f} seconds")
        print(f"HPWL:      {hpwl:.2f} um")
        print("=" * 70)

    results = {
        'design_name': circuit_data.design_name,
        'num_macros': circuit_data.num_macros,
        'num_nets': circuit_data.num_nets,
        'runtime': runtime,
        'hpwl': hpwl,
        'placement': placement,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate macro placement algorithm on benchmark"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Benchmark name (e.g., ariane133)"
    )
    parser.add_argument(
        "--placer",
        type=str,
        default="random",
        help="Placer to use (default: random)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save placement (optional)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    try:
        # Load benchmark
        circuit_data = load_benchmark(args.benchmark)

        # Get placer
        placer = get_placer(args.placer)

        # Evaluate
        results = evaluate(circuit_data, placer, verbose=not args.quiet)

        # Save placement if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(results, str(output_path))
            print(f"\nResults saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
