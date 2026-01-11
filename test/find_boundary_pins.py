#!/usr/bin/env python3
"""Find pins at canvas boundaries."""

import sys
import math
from pathlib import Path

# Add external/MacroPlacement to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "MacroPlacement" / "CodeElements" / "Plc_client"))

from plc_client_os import PlacementCost

def main():
    benchmark_dir = "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    netlist_file = f"{benchmark_dir}/netlist.pb.txt"
    plc_file = f"{benchmark_dir}/initial.plc"

    plc = PlacementCost(netlist_file)
    plc.restore_placement(plc_file, ifInital=True, ifReadComment=True)

    grid_width = float(plc.width / plc.grid_col)
    grid_height = float(plc.height / plc.grid_row)

    print(f"Canvas: {plc.width} x {plc.height}")
    print(f"Grid: {plc.grid_row} rows x {plc.grid_col} cols")
    print(f"Cell size: {grid_width} x {grid_height}")
    print()

    boundary_issues = []

    # Check all pins
    for pin_idx in plc.hard_macro_pin_indices[:100]:  # Check first 100
        pin = plc.modules_w_pins[pin_idx]
        x, y = pin.get_pos()

        row = math.floor(y / grid_height)
        col = math.floor(x / grid_width)

        if row >= plc.grid_row or col >= plc.grid_col or row < 0 or col < 0:
            boundary_issues.append({
                'name': pin.get_name(),
                'pos': (x, y),
                'row': row,
                'col': col,
            })

    print(f"Found {len(boundary_issues)} pins with boundary issues:")
    for issue in boundary_issues[:10]:
        print(f"  {issue['name']}: pos=({issue['pos'][0]:.4f}, {issue['pos'][1]:.4f})")
        print(f"    → row={issue['row']} (max={plc.grid_row-1}), col={issue['col']} (max={plc.grid_col-1})")

    if boundary_issues:
        print()
        print("✗ Boundary condition bug confirmed!")
        print("   Pins at canvas edge compute grid cells >= grid dimensions")
    else:
        print()
        print("✓ No boundary issues in first 100 pins")

if __name__ == "__main__":
    sys.exit(main())
