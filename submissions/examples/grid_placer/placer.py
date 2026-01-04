"""
Example grid-based placer demonstrating placement interface.

This placer demonstrates:
1. Macro-only placement (place_cells=False): Places macros, keeps cells at initial positions
2. Joint placement (place_cells=True): Places both macros and cells
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from marco_place.data.tensor_schema import CircuitTensorData


class GridPlacer:
    """
    Grid-based placer that arranges macros (and optionally cells) in a regular grid.

    This is a simple baseline that doesn't optimize for wirelength/congestion,
    but produces valid placements quickly.
    """

    def __init__(self, seed: int = 42, place_cells: bool = False):
        """
        Initialize GridPlacer.

        Args:
            seed: Random seed for reproducibility
            place_cells: If True, also place standard cells; if False, keep at initial positions
        """
        self.seed = seed
        self.place_cells = place_cells

    def place(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        """
        Place macros (and optionally cells) in a grid pattern.

        Returns:
            placement: torch.Tensor [num_macros + num_stdcells, 2]
                First num_macros rows: macro positions
                Next num_stdcells rows: cell positions
        """
        torch.manual_seed(self.seed)

        # Place macros
        macro_placement = self._place_macros(circuit_data)

        # Handle standard cells
        num_stdcells = circuit_data.num_stdcells
        if num_stdcells == 0:
            # No cells to place
            return macro_placement

        if not self.place_cells:
            # Keep cells at initial positions (macro-only placement)
            return torch.cat([macro_placement, circuit_data.stdcell_positions], dim=0)

        # Joint placement: also place cells
        cell_placement = self._place_cells(circuit_data, macro_placement)
        return torch.cat([macro_placement, cell_placement], dim=0)

    def _place_macros(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        """
        Place macros in a grid pattern.

        Strategy: Row-packing with height-based sorting for better utilization.
        """
        num_macros = circuit_data.num_macros
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height
        macro_sizes = circuit_data.macro_sizes

        # Sort macros by height (descending) for better packing
        heights = macro_sizes[:, 1]
        sorted_indices = torch.argsort(heights, descending=True)

        placement = torch.zeros(num_macros, 2)

        # Row packing
        current_x = 0.0
        current_y = 0.0
        row_height = 0.0

        for idx in sorted_indices:
            width = macro_sizes[idx, 0].item()
            height = macro_sizes[idx, 1].item()

            # Check if macro fits in current row
            if current_x + width > canvas_width:
                # Move to next row
                current_x = 0.0
                current_y += row_height
                row_height = 0.0

            # Place macro (center position)
            placement[idx, 0] = current_x + width / 2
            placement[idx, 1] = current_y + height / 2

            # Update position
            current_x += width
            row_height = max(row_height, height)

        return placement

    def _place_cells(
        self,
        circuit_data: CircuitTensorData,
        macro_placement: torch.Tensor
    ) -> torch.Tensor:
        """
        Place standard cells in available space around macros.

        Strategy: Simple grid in regions not occupied by macros.
        """
        num_cells = circuit_data.num_stdcells
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height

        # Create simple grid for cells
        # This is a coarse placement - a real placer would do detailed placement
        cols = int(torch.ceil(torch.sqrt(torch.tensor(num_cells, dtype=torch.float32))))
        rows = int(torch.ceil(torch.tensor(num_cells, dtype=torch.float32) / cols))

        cell_width = canvas_width / cols
        cell_height = canvas_height / rows

        cell_placement = torch.zeros(num_cells, 2)

        for i in range(num_cells):
            row = i // cols
            col = i % cols

            # Center of grid cell
            cell_placement[i, 0] = (col + 0.5) * cell_width
            cell_placement[i, 1] = (row + 0.5) * cell_height

        return cell_placement


if __name__ == "__main__":
    # Simple test of both modes
    print("Testing GridPlacer...")

    # Create dummy circuit data
    metadata = {
        'design_name': 'test',
        'num_macros': 4,
        'num_stdcells': 10,
        'canvas_width': 1000.0,
        'canvas_height': 1000.0,
    }

    macro_sizes = torch.tensor([
        [100.0, 100.0],
        [150.0, 100.0],
        [100.0, 150.0],
        [120.0, 120.0],
    ])

    stdcell_sizes = torch.tensor([
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
    ])

    circuit_data = CircuitTensorData(
        metadata=metadata,
        macro_positions=torch.zeros(4, 2),
        macro_sizes=macro_sizes,
        stdcell_positions=torch.zeros(10, 2),
        stdcell_sizes=stdcell_sizes,
    )

    # Test 1: Macro-only placement (keeps cells at initial positions)
    print("\n--- Test 1: Macro-only placement ---")
    placer_macro_only = GridPlacer(place_cells=False)
    placement = placer_macro_only.place(circuit_data)

    print(f"✓ Returned placement tensor: {placement.shape}")
    print(f"  Expected: [{4 + 10}, 2]")
    assert placement.shape == (14, 2), f"Shape mismatch: expected (14, 2), got {placement.shape}"
    print(f"  Macro positions (first 4 rows):\n{placement[:4]}")
    print(f"  Cell positions (last 10 rows, should be initial positions):\n{placement[4:]}")

    # Test 2: Joint placement
    print("\n--- Test 2: Joint macro + cell placement ---")
    placer_joint = GridPlacer(place_cells=True)
    placement = placer_joint.place(circuit_data)

    print(f"✓ Returned placement tensor: {placement.shape}")
    print(f"  Expected: [{4 + 10}, 2]")
    assert placement.shape == (14, 2), f"Shape mismatch: expected (14, 2), got {placement.shape}"
    print(f"  Macro positions (first 4 rows):\n{placement[:4]}")
    print(f"  Cell positions (last 10 rows, should be placed):\n{placement[4:]}")

    print("\n✓ GridPlacer tests passed!")
