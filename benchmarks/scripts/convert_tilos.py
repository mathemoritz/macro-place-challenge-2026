#!/usr/bin/env python3
"""
Convert TILOS MacroPlacement benchmarks to tensor format.

This script finds all protobuf netlist files from the TILOS repository
and converts them to our PyTorch tensor format.

Usage:
    python benchmarks/scripts/convert_tilos.py --tilos_dir /path/to/MacroPlacement
"""

import argparse
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from marco_place.data.converters.protobuf_parser import parse_protobuf_netlist


def find_netlist_files(tilos_dir: Path) -> dict:
    """
    Find all netlist.pb.txt files in TILOS repository.

    Args:
        tilos_dir: Path to TILOS MacroPlacement directory

    Returns:
        Dictionary mapping benchmark names to netlist file paths
    """
    netlists = {}

    # Search in CodeElements/Plc_client/test/
    test_dir = tilos_dir / "CodeElements" / "Plc_client" / "test"

    if not test_dir.exists():
        print(f"Warning: Test directory not found: {test_dir}")
        return netlists

    # Find all netlist.pb.txt files
    for netlist_file in test_dir.glob("*/netlist.pb.txt"):
        benchmark_name = netlist_file.parent.name
        netlists[benchmark_name] = netlist_file

    return netlists


def convert_benchmark(
    benchmark_name: str,
    netlist_file: Path,
    output_dir: Path
) -> bool:
    """
    Convert a single benchmark to tensor format.

    Args:
        benchmark_name: Name of the benchmark
        netlist_file: Path to netlist.pb.txt
        output_dir: Output directory for converted benchmark

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n{'='*70}")
        print(f"Converting {benchmark_name}...")
        print(f"{'='*70}")

        # Parse the netlist
        circuit_data = parse_protobuf_netlist(str(netlist_file))

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save to file
        output_file = output_dir / f"{benchmark_name}.pt"
        circuit_data.save(str(output_file))

        print(f"✓ Saved to: {output_file}")
        return True

    except Exception as e:
        print(f"✗ Error converting {benchmark_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert TILOS benchmarks to tensor format"
    )
    parser.add_argument(
        "--tilos_dir",
        type=str,
        default="/home/will/partcl/MacroPlacement",
        help="Path to TILOS MacroPlacement repository"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (default: benchmarks/processed/public)"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        help="Specific benchmarks to convert (default: all)"
    )

    args = parser.parse_args()

    # Set paths
    tilos_dir = Path(args.tilos_dir)
    if not tilos_dir.exists():
        print(f"Error: TILOS directory not found: {tilos_dir}")
        return 1

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "benchmarks" / "processed" / "public"

    # Find all netlist files
    print("Searching for netlist files...")
    netlists = find_netlist_files(tilos_dir)

    if not netlists:
        print("No netlist files found!")
        return 1

    print(f"\nFound {len(netlists)} benchmarks:")
    for name in sorted(netlists.keys()):
        print(f"  - {name}")

    # Filter benchmarks if specified
    if args.benchmarks:
        netlists = {k: v for k, v in netlists.items() if k in args.benchmarks}
        if not netlists:
            print(f"\nError: None of the specified benchmarks found")
            return 1

    print(f"\nConverting {len(netlists)} benchmarks...\n")

    # Convert each benchmark
    success_count = 0
    for benchmark_name, netlist_file in sorted(netlists.items()):
        if convert_benchmark(benchmark_name, netlist_file, output_dir):
            success_count += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"Conversion Summary")
    print(f"{'='*70}")
    print(f"Total:      {len(netlists)}")
    print(f"Successful: {success_count}")
    print(f"Failed:     {len(netlists) - success_count}")
    print(f"\nOutput directory: {output_dir}")

    return 0 if success_count == len(netlists) else 1


if __name__ == "__main__":
    sys.exit(main())
