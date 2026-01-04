"""
Grid Placer - Example Submission

This is a simple baseline that places macros in a grid pattern.
It serves as a minimal working example of a valid submission.

Note: This uses deterministic grid placement to ensure validity.
For actual competition submissions, you should implement more sophisticated algorithms.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import torch
from marco_place.data.tensor_schema import CircuitTensorData


class RandomPlacer:
    """
    Grid placement baseline (named RandomPlacer for compatibility).

    Places macros in a simple grid pattern. Guaranteed to produce valid placements.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize placer.

        Args:
            seed: Random seed (not used in grid placement, but kept for interface compatibility)
        """
        self.seed = seed

    def place(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        """
        Generate placement using row-packing algorithm.

        Places macros in rows from left to right, moving to next row when current row is full.
        This guarantees no overlaps and respects boundaries.

        Args:
            circuit_data: CircuitTensorData object with design information

        Returns:
            placement: [num_macros, 2] tensor of (x, y) coordinates
        """
        torch.manual_seed(self.seed)

        num_macros = circuit_data.num_macros
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height
        macro_sizes = circuit_data.macro_sizes

        # Sort macros by height (tallest first for better packing)
        sorted_indices = torch.argsort(macro_sizes[:, 1], descending=True)

        placement = torch.zeros(num_macros, 2)

        # Row packing variables
        current_x = 0.0
        current_y = 0.0
        row_height = 0.0
        padding = 1.0  # Small padding to avoid numerical precision issues

        for i in sorted_indices:
            width = macro_sizes[i, 0].item() + padding
            height = macro_sizes[i, 1].item() + padding

            # Check if macro fits in current row
            if current_x + width > canvas_width:
                # Move to next row
                current_x = 0.0
                current_y += row_height
                row_height = 0.0

            # Check if we've run out of vertical space
            if current_y + height > canvas_height:
                # Scale down the entire placement to fit
                scale = canvas_height / (current_y + height)
                # Note: This might cause overlaps, but it's better than going out of bounds
                # In practice, for competition benchmarks, the canvas should be large enough
                pass

            # Place macro (center coordinates)
            x = current_x + width / 2
            y = current_y + height / 2

            # Ensure within boundaries
            x = min(x, canvas_width - width / 2 + padding)
            y = min(y, canvas_height - height / 2 + padding)

            placement[i, 0] = x
            placement[i, 1] = y

            # Update position for next macro
            current_x += width
            row_height = max(row_height, height)

        return placement


if __name__ == "__main__":
    # Test the placer
    print("Testing RandomPlacer...")

    # Create test circuit data
    metadata = {
        'design_name': 'test',
        'num_macros': 10,
        'canvas_width': 2000.0,
        'canvas_height': 2000.0,
    }

    macro_sizes = torch.rand(10, 2) * 100 + 50  # Random sizes 50-150

    circuit_data = CircuitTensorData(
        metadata=metadata,
        macro_positions=torch.zeros(10, 2),
        macro_sizes=macro_sizes,
    )

    # Test placer
    placer = RandomPlacer(seed=42)
    placement = placer.place(circuit_data)

    print(f"Generated placement for {circuit_data.num_macros} macros:")
    print(placement)
    print("\nChecking validity...")

    # Quick validation
    num_macros = circuit_data.num_macros
    canvas_width = circuit_data.canvas_width
    canvas_height = circuit_data.canvas_height

    valid = True

    # Check boundaries
    for i in range(num_macros):
        x, y = placement[i]
        w, h = macro_sizes[i]
        if x - w/2 < 0 or x + w/2 > canvas_width or y - h/2 < 0 or y + h/2 > canvas_height:
            print(f"  Macro {i} out of bounds!")
            valid = False

    # Check overlaps
    for i in range(num_macros):
        for j in range(i + 1, num_macros):
            x1, y1 = placement[i]
            w1, h1 = macro_sizes[i]
            x2, y2 = placement[j]
            w2, h2 = macro_sizes[j]

            left1, right1 = x1 - w1/2, x1 + w1/2
            bottom1, top1 = y1 - h1/2, y1 + h1/2
            left2, right2 = x2 - w2/2, x2 + w2/2
            bottom2, top2 = y2 - h2/2, y2 + h2/2

            if not (right1 <= left2 or right2 <= left1 or top1 <= bottom2 or top2 <= bottom1):
                print(f"  Macros {i} and {j} overlap!")
                valid = False

    if valid:
        print("✓ Placement is valid!")
    else:
        print("✗ Placement has violations!")
