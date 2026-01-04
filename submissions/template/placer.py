"""
Submission template for macro placement competition.

All submissions must inherit from BasePlacerInterface and implement the place() method.
"""

from abc import ABC, abstractmethod
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from marco_place.data.tensor_schema import CircuitTensorData


class BasePlacerInterface(ABC):
    """
    Base interface for macro placement algorithms.

    All submissions must inherit from this class and implement the place() method.
    Optionally, implement place_cells() for two-stage placement (macros + standard cells).

    Your placer will be evaluated on:
    1. Placement legality (no overlaps, within boundaries)
    2. Proxy cost (wirelength + density + congestion)
    3. Runtime (must complete within 1 hour per benchmark)

    Prize Eligibility:
    - The $20K prize is awarded ONLY if your submission beats the Circuit Training
      baseline on aggregate across all benchmarks
    - Aggregate score must be > 0 (positive improvement over baseline)
    """

    @abstractmethod
    def place(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        """
        Compute macro placement for the given circuit.

        Args:
            circuit_data: CircuitTensorData object containing:
                - metadata: dict with design_name, num_macros, canvas_width, canvas_height
                - macro_sizes: [num_macros, 2] tensor of (width, height)
                - net_to_nodes: list of tensors containing node indices for each net
                - net_weights: [num_nets] tensor of net weights
                - Other optional fields (see tensor_schema.py for full specification)

        Returns:
            placement: torch.Tensor of shape [num_macros, 2]
                      containing (x, y) coordinates of macro centers in microns

        Constraints:
        - Placement must be within canvas boundaries:
          0 <= x - width/2 and x + width/2 <= canvas_width
          0 <= y - height/2 and y + height/2 <= canvas_height

        - Macros must not overlap:
          No two macros can have overlapping areas

        - Must respect placement blockages (if present):
          Macros cannot be placed in blockage regions

        - Runtime limit:
          Must complete within 1 hour (3600 seconds) per benchmark

        Notes:
        - You may use any PyTorch-based approach (GNN, RL, optimization, etc.)
        - Pre-trained models are allowed if included in submission
        - You have access to all circuit_data fields for your algorithm
        - The evaluation harness will validate your placement and compute metrics
        - Invalid placements will receive a score of 0 for that benchmark

        Example:
            >>> def place(self, circuit_data):
            >>>     num_macros = circuit_data.num_macros
            >>>     canvas_width = circuit_data.canvas_width
            >>>     canvas_height = circuit_data.canvas_height
            >>>
            >>>     # Your algorithm here
            >>>     placement = your_algorithm(circuit_data)
            >>>
            >>>     return placement  # [num_macros, 2] tensor
        """
        pass

    def place_cells(
        self,
        circuit_data: CircuitTensorData,
        macro_placement: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute standard cell placement given fixed macro positions (OPTIONAL).

        This method is optional. If not implemented, only macros will be placed.
        For two-stage placement, implement this method to place standard cells
        after macros have been fixed.

        Args:
            circuit_data: CircuitTensorData object with design information
            macro_placement: [num_macros, 2] tensor of fixed macro positions

        Returns:
            cell_placement: torch.Tensor of shape [num_stdcells, 2]
                          containing (x, y) coordinates of cell centers in microns

        Constraints:
        - Same constraints as macro placement (no overlaps, within boundaries)
        - Cells must not overlap with fixed macros
        - Runtime is included in the total 1-hour timeout

        Notes:
        - Standard cells are typically clustered (800-1000 clusters per design)
        - You can use coarse placement (cells don't need detailed legalization)
        - If not implemented, cells will keep their initial positions

        Example:
            >>> def place_cells(self, circuit_data, macro_placement):
            >>>     num_cells = circuit_data.num_stdcells
            >>>     # Place cells in remaining space
            >>>     cell_placement = your_cell_placement_algorithm(
            >>>         circuit_data, macro_placement
            >>>     )
            >>>     return cell_placement  # [num_stdcells, 2] tensor
        """
        # Default: return None to indicate cells are not placed
        # Evaluation will use initial cell positions if None is returned
        return None


class TemplatePlacer(BasePlacerInterface):
    """
    Template placer implementation.

    Replace this with your own algorithm!

    This is a simple grid-based placement that serves as a starting point.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize your placer.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed

    def place(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        """
        Simple grid-based placement.

        Replace this with your own algorithm!
        """
        torch.manual_seed(self.seed)

        num_macros = circuit_data.num_macros
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height

        # Create grid
        cols = int(torch.ceil(torch.sqrt(torch.tensor(num_macros, dtype=torch.float32))))
        rows = int(torch.ceil(torch.tensor(num_macros, dtype=torch.float32) / cols))

        cell_width = canvas_width / cols
        cell_height = canvas_height / rows

        # Place macros in grid
        placement = torch.zeros(num_macros, 2)

        for i in range(num_macros):
            row = i // cols
            col = i % cols

            # Center of grid cell
            placement[i, 0] = (col + 0.5) * cell_width
            placement[i, 1] = (row + 0.5) * cell_height

        return placement


if __name__ == "__main__":
    # Simple test
    print("Testing TemplatePlacer...")

    # Create dummy circuit data
    metadata = {
        'design_name': 'test',
        'num_macros': 4,
        'canvas_width': 1000.0,
        'canvas_height': 1000.0,
    }

    macro_sizes = torch.tensor([
        [100.0, 100.0],
        [150.0, 100.0],
        [100.0, 150.0],
        [120.0, 120.0],
    ])

    circuit_data = CircuitTensorData(
        metadata=metadata,
        macro_positions=torch.zeros(4, 2),
        macro_sizes=macro_sizes,
    )

    # Test placer
    placer = TemplatePlacer()
    placement = placer.place(circuit_data)

    print(f"Generated placement for {circuit_data.num_macros} macros:")
    print(placement)
    print("\n✓ Template placer works!")
