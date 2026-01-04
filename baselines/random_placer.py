"""
Random baseline placer for macro placement.

This is the simplest baseline that randomly places macros within the canvas
while ensuring no overlaps and respecting canvas boundaries.
"""

import torch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marco_place.data.tensor_schema import CircuitTensorData


class RandomPlacer:
    """
    Random placement baseline.

    Randomly places macros within the canvas, ensuring:
    - No overlaps between macros
    - All macros are within canvas boundaries
    """

    def __init__(self, max_attempts: int = 10000, seed: int = 42):
        """
        Initialize RandomPlacer.

        Args:
            max_attempts: Maximum attempts to place all macros without overlaps
            seed: Random seed for reproducibility
        """
        self.max_attempts = max_attempts
        self.seed = seed

    def place(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        """
        Generate random macro placement.

        Args:
            circuit_data: Circuit data containing macro sizes and canvas dimensions

        Returns:
            placement: [num_macros, 2] - (x, y) coordinates of macro centers
        """
        torch.manual_seed(self.seed)

        num_macros = circuit_data.num_macros
        macro_sizes = circuit_data.macro_sizes
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height

        # Try to generate a valid placement
        for attempt in range(self.max_attempts):
            placement = self._generate_random_placement(
                num_macros, macro_sizes, canvas_width, canvas_height
            )

            if self._is_valid_placement(placement, macro_sizes, canvas_width, canvas_height):
                return placement

        # If we couldn't find a valid placement, use grid-based placement
        print(f"Warning: Could not find overlap-free random placement after {self.max_attempts} attempts.")
        print("Falling back to grid-based placement.")
        return self._grid_placement(num_macros, macro_sizes, canvas_width, canvas_height)

    def _generate_random_placement(
        self,
        num_macros: int,
        macro_sizes: torch.Tensor,
        canvas_width: float,
        canvas_height: float
    ) -> torch.Tensor:
        """
        Generate random positions for all macros.

        Args:
            num_macros: Number of macros
            macro_sizes: [num_macros, 2] - (width, height) of each macro
            canvas_width: Canvas width
            canvas_height: Canvas height

        Returns:
            placement: [num_macros, 2] - (x, y) coordinates
        """
        placement = torch.zeros(num_macros, 2)

        for i in range(num_macros):
            width, height = macro_sizes[i]

            # Generate random position ensuring macro stays within canvas
            # x, y are center coordinates
            x_min = width / 2
            x_max = canvas_width - width / 2
            y_min = height / 2
            y_max = canvas_height - height / 2

            if x_max <= x_min or y_max <= y_min:
                # Macro is too large for canvas
                placement[i, 0] = canvas_width / 2
                placement[i, 1] = canvas_height / 2
            else:
                placement[i, 0] = torch.rand(1) * (x_max - x_min) + x_min
                placement[i, 1] = torch.rand(1) * (y_max - y_min) + y_min

        return placement

    def _is_valid_placement(
        self,
        placement: torch.Tensor,
        macro_sizes: torch.Tensor,
        canvas_width: float,
        canvas_height: float
    ) -> bool:
        """
        Check if placement is valid (no overlaps, within boundaries).

        Args:
            placement: [num_macros, 2] - (x, y) coordinates
            macro_sizes: [num_macros, 2] - (width, height) of each macro
            canvas_width: Canvas width
            canvas_height: Canvas height

        Returns:
            True if valid, False otherwise
        """
        num_macros = len(placement)

        # Check boundaries
        for i in range(num_macros):
            x, y = placement[i]
            width, height = macro_sizes[i]

            left = x - width / 2
            right = x + width / 2
            bottom = y - height / 2
            top = y + height / 2

            if left < 0 or right > canvas_width or bottom < 0 or top > canvas_height:
                return False

        # Check overlaps
        for i in range(num_macros):
            for j in range(i + 1, num_macros):
                if self._rectangles_overlap(placement[i], macro_sizes[i],
                                           placement[j], macro_sizes[j]):
                    return False

        return True

    def _rectangles_overlap(
        self,
        pos1: torch.Tensor,
        size1: torch.Tensor,
        pos2: torch.Tensor,
        size2: torch.Tensor
    ) -> bool:
        """
        Check if two axis-aligned rectangles overlap.

        Args:
            pos1: (x, y) center of first rectangle
            size1: (width, height) of first rectangle
            pos2: (x, y) center of second rectangle
            size2: (width, height) of second rectangle

        Returns:
            True if rectangles overlap, False otherwise
        """
        x1, y1 = pos1
        w1, h1 = size1
        x2, y2 = pos2
        w2, h2 = size2

        left1, right1 = x1 - w1/2, x1 + w1/2
        bottom1, top1 = y1 - h1/2, y1 + h1/2

        left2, right2 = x2 - w2/2, x2 + w2/2
        bottom2, top2 = y2 - h2/2, y2 + h2/2

        # No overlap if one is completely to the side of the other
        if right1 <= left2 or right2 <= left1:
            return False
        if top1 <= bottom2 or top2 <= bottom1:
            return False

        return True

    def _grid_placement(
        self,
        num_macros: int,
        macro_sizes: torch.Tensor,
        canvas_width: float,
        canvas_height: float
    ) -> torch.Tensor:
        """
        Place macros on a grid (fallback when random placement fails).

        Args:
            num_macros: Number of macros
            macro_sizes: [num_macros, 2] - (width, height) of each macro
            canvas_width: Canvas width
            canvas_height: Canvas height

        Returns:
            placement: [num_macros, 2] - (x, y) coordinates
        """
        # Determine grid size
        cols = int(torch.ceil(torch.sqrt(torch.tensor(num_macros))))
        rows = int(torch.ceil(torch.tensor(num_macros) / cols))

        cell_width = canvas_width / cols
        cell_height = canvas_height / rows

        placement = torch.zeros(num_macros, 2)

        for i in range(num_macros):
            row = i // cols
            col = i % cols

            # Place macro at center of grid cell
            x = (col + 0.5) * cell_width
            y = (row + 0.5) * cell_height

            placement[i, 0] = x
            placement[i, 1] = y

        return placement


if __name__ == "__main__":
    # Simple test
    print("Random Placer Test")
    print("=" * 50)

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
        macro_positions=torch.zeros(4, 2),  # Dummy initial positions
        macro_sizes=macro_sizes,
    )

    # Run random placer
    placer = RandomPlacer()
    placement = placer.place(circuit_data)

    print(f"Generated placement for {circuit_data.num_macros} macros:")
    for i in range(circuit_data.num_macros):
        x, y = placement[i]
        w, h = macro_sizes[i]
        print(f"  Macro {i}: pos=({x:.1f}, {y:.1f}), size=({w:.1f}, {h:.1f})")
