#!/usr/bin/env python3
"""Test PlacementCost directly."""

import sys
from pathlib import Path

# Add external/MacroPlacement to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "MacroPlacement" / "CodeElements" / "Plc_client"))

from plc_client_os import PlacementCost

def main():
    benchmark_dir = "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    netlist_file = f"{benchmark_dir}/netlist.pb.txt"
    plc_file = f"{benchmark_dir}/initial.plc"

    print("Creating PlacementCost...")
    plc = PlacementCost(netlist_file)
    print(f"Grid: {plc.grid_row} x {plc.grid_col}")
    print(f"Canvas: {plc.width} x {plc.height}")

    print("\nRestoring placement...")
    plc.restore_placement(plc_file, ifInital=True, ifReadComment=True)
    print(f"Grid after restore: {plc.grid_row} x {plc.grid_col}")
    print(f"Canvas after restore: {plc.width} x {plc.height}")

    print("\nComputing costs...")
    try:
        wl = plc.get_cost()
        print(f"Wirelength: {wl}")

        density = plc.get_density_cost()
        print(f"Density: {density}")

        congestion = plc.get_congestion_cost()
        print(f"Congestion: {congestion}")

        print("\n✓ Success!")
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
