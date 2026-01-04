#!/usr/bin/env python3
"""
Simple benchmark converter - generates toy benchmarks for testing.

This script creates simple synthetic benchmarks to test the evaluation pipeline.
For Phase 2, we'll implement full LEF/DEF/protobuf parsing.

Usage:
    python benchmarks/scripts/convert_simple.py --output benchmarks/processed/public/toy_small.pt
"""

import argparse
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from marco_place.data.tensor_schema import CircuitTensorData


def generate_toy_benchmark(
    num_macros: int = 10,
    canvas_width: float = 1000.0,
    canvas_height: float = 1000.0,
    connectivity: float = 0.3,
    seed: int = 42
) -> CircuitTensorData:
    """
    Generate a synthetic toy benchmark for testing.

    Args:
        num_macros: Number of macros to generate
        canvas_width: Canvas width in microns
        canvas_height: Canvas height in microns
        connectivity: Net connectivity density (0-1)
        seed: Random seed for reproducibility

    Returns:
        CircuitTensorData instance
    """
    torch.manual_seed(seed)

    # Generate macro sizes (random between 50-200 microns)
    macro_sizes = torch.rand(num_macros, 2) * 150 + 50

    # Generate initial positions (will be overridden by placer)
    macro_positions = torch.rand(num_macros, 2) * torch.tensor([canvas_width, canvas_height])

    # All macros are movable
    macro_is_fixed = torch.zeros(num_macros, dtype=torch.bool)

    # Generate nets (hypergraph connectivity)
    # Each net connects 2-4 random macros
    num_nets = int(num_macros * num_macros * connectivity / 2)
    net_to_nodes = []
    net_weights = []

    for _ in range(num_nets):
        # Random number of pins per net (2-4)
        num_pins = torch.randint(2, 5, (1,)).item()

        # Random macro indices
        node_indices = torch.randperm(num_macros)[:num_pins]
        net_to_nodes.append(node_indices)

        # Random weight (0.5 - 2.0)
        weight = torch.rand(1).item() * 1.5 + 0.5
        net_weights.append(weight)

    net_weights = torch.tensor(net_weights)

    # Create metadata
    metadata = {
        'design_name': f'toy_{num_macros}macros',
        'num_macros': num_macros,
        'canvas_width': canvas_width,
        'canvas_height': canvas_height,
        'target_density': 0.6,
    }

    # Create CircuitTensorData
    circuit_data = CircuitTensorData(
        metadata=metadata,
        macro_positions=macro_positions,
        macro_sizes=macro_sizes,
        macro_is_fixed=macro_is_fixed,
        net_to_nodes=net_to_nodes,
        net_weights=net_weights,
    )

    return circuit_data


def generate_ariane_like_benchmark() -> CircuitTensorData:
    """
    Generate a benchmark similar to Ariane133 for testing.

    This is a placeholder until we can parse the actual Ariane133 benchmark.

    Returns:
        CircuitTensorData instance
    """
    # Ariane has ~133 macros (memory SRAMs)
    num_macros = 133
    canvas_width = 5000.0  # Approximate
    canvas_height = 5000.0

    # Generate SRAM-like macro sizes (256x16 bit SRAMs)
    # Typical SRAM dimensions: ~200-300 microns
    macro_sizes = torch.ones(num_macros, 2) * 250 + torch.rand(num_macros, 2) * 50

    # Initial positions (grid-like)
    cols = int(torch.ceil(torch.sqrt(torch.tensor(num_macros))))
    rows = int(torch.ceil(torch.tensor(num_macros) / cols))

    macro_positions = torch.zeros(num_macros, 2)
    for i in range(num_macros):
        row = i // cols
        col = i % cols
        macro_positions[i, 0] = (col + 0.5) * (canvas_width / cols)
        macro_positions[i, 1] = (row + 0.5) * (canvas_height / rows)

    macro_is_fixed = torch.zeros(num_macros, dtype=torch.bool)

    # Generate nets (more realistic connectivity)
    # Processor designs typically have ~10-20 nets per macro on average
    num_nets = num_macros * 15
    net_to_nodes = []
    net_weights = []

    torch.manual_seed(133)

    for _ in range(num_nets):
        # Most nets connect 2-3 macros, some connect more
        num_pins = torch.randint(2, 6, (1,)).item()
        node_indices = torch.randperm(num_macros)[:num_pins]
        net_to_nodes.append(node_indices)

        # Weight based on criticality (most nets have weight ~1)
        weight = torch.randn(1).abs().item() * 0.5 + 0.75
        net_weights.append(weight)

    net_weights = torch.tensor(net_weights)

    metadata = {
        'design_name': 'ariane133',
        'num_macros': num_macros,
        'canvas_width': canvas_width,
        'canvas_height': canvas_height,
        'target_density': 0.6,
        'description': 'Synthetic benchmark similar to Ariane133 (for testing)',
    }

    circuit_data = CircuitTensorData(
        metadata=metadata,
        macro_positions=macro_positions,
        macro_sizes=macro_sizes,
        macro_is_fixed=macro_is_fixed,
        net_to_nodes=net_to_nodes,
        net_weights=net_weights,
    )

    return circuit_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic benchmarks for testing"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ariane133",
        choices=["ariane133", "toy_small", "toy_medium", "toy_large"],
        help="Benchmark to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: benchmarks/processed/public/<benchmark>.pt)"
    )

    args = parser.parse_args()

    # Generate benchmark
    if args.benchmark == "ariane133":
        print("Generating Ariane133-like benchmark...")
        circuit_data = generate_ariane_like_benchmark()
    elif args.benchmark == "toy_small":
        print("Generating small toy benchmark...")
        circuit_data = generate_toy_benchmark(num_macros=10)
    elif args.benchmark == "toy_medium":
        print("Generating medium toy benchmark...")
        circuit_data = generate_toy_benchmark(num_macros=50)
    elif args.benchmark == "toy_large":
        print("Generating large toy benchmark...")
        circuit_data = generate_toy_benchmark(num_macros=200)

    print(f"Generated: {circuit_data}")
    print(f"  Macros: {circuit_data.num_macros}")
    print(f"  Nets:   {circuit_data.num_nets}")
    print(f"  Canvas: {circuit_data.canvas_width:.1f} x {circuit_data.canvas_height:.1f} um")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "benchmarks" / "processed" / "public" / f"{args.benchmark}.pt"

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save benchmark
    circuit_data.save(str(output_path))
    print(f"\nSaved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
