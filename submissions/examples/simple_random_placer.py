#!/usr/bin/env python3
"""
Simple Random Placer - Example Submission

This demonstrates the complete workflow:
1. Load a benchmark using our loader
2. Implement a simple placement algorithm
3. Compute proxy cost

This serves as a minimal working example using the PyTorch infrastructure.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
from loader import load_benchmark_from_dir
from objective import compute_proxy_cost
from utils import validate_placement
from benchmark import Benchmark


class SimpleRandomPlacer:
    """
    Simple random placement algorithm (baseline).

    Places all movable macros at random positions within canvas bounds.
    Respects fixed macros.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize placer.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Generate random placement.

        Args:
            benchmark: Benchmark object with circuit data

        Returns:
            placement: [num_macros, 2] tensor of (x, y) center positions
        """
        torch.manual_seed(self.seed)

        # Start with zeros
        placement = torch.zeros(benchmark.num_macros, 2)

        # Place each macro randomly
        for i in range(benchmark.num_macros):
            # Skip fixed macros - we'll set them at the end
            if benchmark.macro_fixed[i]:
                continue

            w, h = benchmark.macro_sizes[i]

            # Random position ensuring macro fits in canvas
            # Macro position is the center, so we need to ensure edges stay within bounds
            x_min = w / 2
            x_max = benchmark.canvas_width - w / 2
            y_min = h / 2
            y_max = benchmark.canvas_height - h / 2

            # Generate random position
            x = torch.rand(1).item() * (x_max - x_min) + x_min
            y = torch.rand(1).item() * (y_max - y_min) + y_min

            placement[i, 0] = x
            placement[i, 1] = y

        # Restore fixed macro positions
        fixed_mask = benchmark.macro_fixed
        placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

        return placement


def main():
    """Run the complete placement workflow."""

    # Path to benchmark
    benchmark_dir = "external/MacroPlacement/Testcases/ICCAD04/ibm01"

    print("=" * 60)
    print("Simple Random Placer - Example Workflow")
    print("=" * 60)
    print()

    # Step 1: Load benchmark
    print(f"[1/4] Loading benchmark from {benchmark_dir}...")
    benchmark, plc = load_benchmark_from_dir(benchmark_dir)

    print(f"  ✓ Loaded {benchmark.name}")
    print(f"    - Macros: {benchmark.num_macros} ({(~benchmark.macro_fixed).sum().item()} movable, {benchmark.macro_fixed.sum().item()} fixed)")
    print(f"    - Nets: {benchmark.num_nets}")
    print(f"    - Canvas: {benchmark.canvas_width:.1f} × {benchmark.canvas_height:.1f} μm")
    print(f"    - Grid: {benchmark.grid_rows} × {benchmark.grid_cols}")
    print()

    # Step 2: Run placer
    print("[2/4] Running placer...")
    placer = SimpleRandomPlacer(seed=42)
    placement = placer.place(benchmark)

    print(f"  ✓ Generated placement for {benchmark.num_macros} macros")
    print()

    # Step 3: Validate placement
    print("[3/4] Validating placement...")
    is_valid, violations = validate_placement(placement, benchmark)

    if is_valid:
        print("  ✓ Placement is valid!")
        print("    - All macros within canvas bounds")
        print("    - No NaN/Inf values")
        print("    - Fixed macros at original positions")
    else:
        print("  ✗ Placement has violations:")
        for violation in violations:
            print(f"    - {violation}")
    print()

    # Step 4: Compute proxy cost and overlap metrics
    print("[4/4] Computing proxy cost and overlap metrics...")
    costs = compute_proxy_cost(placement, benchmark, plc)

    print("  ✓ Costs computed:")
    print(f"    - Wirelength:  {costs['wirelength_cost']:.6f}")
    print(f"    - Density:     {costs['density_cost']:.6f}")
    print(f"    - Congestion:  {costs['congestion_cost']:.6f}")
    print(f"    - Proxy Cost:  {costs['proxy_cost']:.6f} ⭐")
    print()

    print("  ✓ Overlap analysis:")
    print(f"    - Overlapping pairs:       {costs['overlap_count']}")
    print(f"    - Macros with overlaps:    {costs['num_macros_with_overlaps']} ({costs['overlap_ratio']*100:.1f}%)")
    print(f"    - Total overlap area:      {costs['total_overlap_area']:.3f} μm²")
    if costs['overlap_count'] > 0:
        print(f"    - Max single overlap:      {costs['max_overlap_area']:.6f} μm²")
        print(f"    - Avg overlap per pair:    {costs['total_overlap_area']/costs['overlap_count']:.6f} μm²")
    print()

    # Compare with initial placement
    print("Comparison with initial placement:")
    initial_costs = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)

    improvement = ((initial_costs['proxy_cost'] - costs['proxy_cost']) / initial_costs['proxy_cost']) * 100

    print(f"  Initial proxy cost:   {initial_costs['proxy_cost']:.6f} (overlaps: {initial_costs['overlap_count']})")
    print(f"  Random proxy cost:    {costs['proxy_cost']:.6f} (overlaps: {costs['overlap_count']})")

    if improvement > 0:
        print(f"  Improvement: +{improvement:.2f}% better! 🎉")
    else:
        print(f"  Improvement: {improvement:.2f}% (worse)")

    print()
    print("=" * 60)
    print("Example workflow complete!")
    print()
    print("Next steps:")
    print("  1. Implement a smarter placement algorithm (SA, RL, GNN, etc.)")
    print("  2. Run on all benchmarks in ICCAD04")
    print("  3. Submit your solution to beat the baselines!")
    print("=" * 60)


if __name__ == "__main__":
    main()
