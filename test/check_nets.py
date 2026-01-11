#!/usr/bin/env python3
"""Check net information in PlacementCost."""

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
    plc.restore_placement(plc_file, ifInital=True, ifReadComment=True)

    print(f"\nPlacementCost internal state:")
    print(f"  net_cnt: {plc.net_cnt}")
    print(f"  Number of modules_w_pins: {len(plc.modules_w_pins)}")
    print(f"  Number of hard macros: {len(plc.hard_macro_indices)}")
    print(f"  Number of hard macro pins: {len(plc.hard_macro_pin_indices)}")

    print(f"\nComputing wirelength:")
    wl = plc.get_wirelength()
    print(f"  Total wirelength (HPWL): {wl}")

    wl_cost = plc.get_cost()
    print(f"  Normalized wirelength cost: {wl_cost}")

    canvas_w, canvas_h = plc.get_canvas_width_height()
    print(f"\nNormalization:")
    print(f"  Canvas: {canvas_w} x {canvas_h}")
    print(f"  Expected: {wl} / (({canvas_w} + {canvas_h}) * {plc.net_cnt}) = {wl / ((canvas_w + canvas_h) * plc.net_cnt)}")

if __name__ == "__main__":
    sys.exit(main())
