#!/usr/bin/env python3
"""Debug congestion computation."""

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
    print(f"After __init__:")
    print(f"  grid_row={plc.grid_row}, grid_col={plc.grid_col}")
    print(f"  width={plc.width}, height={plc.height}")
    print(f"  H_routing_cong size={len(plc.H_routing_cong)}")
    print()

    print("Restoring placement...")
    plc.restore_placement(plc_file, ifInital=True, ifReadComment=True)
    print(f"After restore_placement:")
    print(f"  grid_row={plc.grid_row}, grid_col={plc.grid_col}")
    print(f"  width={plc.width}, height={plc.height}")
    print(f"  H_routing_cong size={len(plc.H_routing_cong)}")
    print(f"  Expected size={plc.grid_row * plc.grid_col}")
    print()

    # Check some macro positions
    print("Checking macro positions and grid cells:")
    for i, idx in enumerate(plc.hard_macro_indices[:5]):
        node = plc.modules_w_pins[idx]
        x, y = node.get_pos()

        # Compute grid cell manually
        grid_width = float(plc.width / plc.grid_col)
        grid_height = float(plc.height / plc.grid_row)
        row = int(y / grid_height)
        col = int(x / grid_width)
        grid_idx = row * plc.grid_col + col

        print(f"  Macro {i} ({node.get_name()[:30]}): pos=({x:.2f}, {y:.2f})")
        print(f"    → grid cell row={row}, col={col}, idx={grid_idx}")
        print(f"    → within bounds? {0 <= grid_idx < len(plc.H_routing_cong)}")

        if row >= plc.grid_row or col >= plc.grid_col:
            print(f"    ⚠️  OUT OF BOUNDS! row={row} >= {plc.grid_row} or col={col} >= {plc.grid_col}")
    print()

    # Try to compute congestion and catch where it fails
    print("Attempting congestion computation...")
    try:
        plc.FLAG_UPDATE_CONGESTION = True
        plc.get_routing()
        print("✓ Routing computation succeeded!")

        congestion = plc.get_congestion_cost()
        print(f"✓ Congestion cost: {congestion}")
    except IndexError as e:
        print(f"✗ IndexError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    sys.exit(main())
