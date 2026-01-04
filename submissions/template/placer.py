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
        Compute placement for the given circuit.

        Args:
            circuit_data: CircuitTensorData object containing:
                - metadata: dict with design_name, num_macros, num_stdcells, canvas dimensions
                - macro_sizes: [num_macros, 2] tensor of (width, height)
                - stdcell_sizes: [num_stdcells, 2] tensor of (width, height)
                - net_to_nodes: list of tensors containing node indices for each net
                - net_weights: [num_nets] tensor of net weights
                - Other optional fields (see tensor_schema.py for full specification)

        Returns:
            placement: torch.Tensor [num_macros + num_stdcells, 2]
                - First num_macros rows: macro positions (x, y)
                - Next num_stdcells rows: standard cell positions (x, y)
                - If you don't want to place cells, return their initial positions

        Constraints:
        - All positions must be within canvas boundaries
        - No overlaps between macros
        - No overlaps between macros and cells
        - Runtime limit: 1 hour (3600 seconds) per benchmark

        Notes:
        - How you compute the placement is up to you (single-stage, two-stage, etc.)
        - You can place only macros and keep cells at initial positions
        - Standard cells are pre-clustered (typically 800-1000 clusters per design)

        Examples:
            >>> # Example 1: Macro-only placement (keep cells at initial positions)
            >>> def place(self, circuit_data):
            >>>     macro_placement = your_macro_algorithm(circuit_data)
            >>>     # Keep cells at initial positions
            >>>     return torch.cat([macro_placement, circuit_data.stdcell_positions], dim=0)

            >>> # Example 2: Joint placement (single-stage algorithm)
            >>> def place(self, circuit_data):
            >>>     all_positions = your_joint_algorithm(circuit_data)
            >>>     return all_positions  # [num_macros + num_stdcells, 2]

            >>> # Example 3: Two-stage placement (internal implementation detail)
            >>> def place(self, circuit_data):
            >>>     # Stage 1: Place macros
            >>>     macro_placement = place_macros_first(circuit_data)
            >>>     # Stage 2: Place cells around macros
            >>>     cell_placement = place_cells_given_macros(circuit_data, macro_placement)
            >>>     # Return concatenated result
            >>>     return torch.cat([macro_placement, cell_placement], dim=0)
        """
        pass


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
        num_stdcells = circuit_data.num_stdcells
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height

        # Create grid for macros
        cols = int(torch.ceil(torch.sqrt(torch.tensor(num_macros, dtype=torch.float32))))
        rows = int(torch.ceil(torch.tensor(num_macros, dtype=torch.float32) / cols))

        cell_width = canvas_width / cols
        cell_height = canvas_height / rows

        # Place macros in grid
        macro_placement = torch.zeros(num_macros, 2)

        for i in range(num_macros):
            row = i // cols
            col = i % cols

            # Center of grid cell
            macro_placement[i, 0] = (col + 0.5) * cell_width
            macro_placement[i, 1] = (row + 0.5) * cell_height

        # Keep standard cells at initial positions (macro-only placement)
        if num_stdcells > 0:
            return torch.cat([macro_placement, circuit_data.stdcell_positions], dim=0)
        else:
            return macro_placement


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
