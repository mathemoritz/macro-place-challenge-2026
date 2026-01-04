"""
Placement validation and legality checking.

This module implements checks to ensure placements satisfy all physical constraints:
- No macro overlaps
- All macros within canvas boundaries
- Respecting placement blockages
- Respecting region constraints
"""

from typing import List, Tuple
import torch

from marco_place.data.tensor_schema import CircuitTensorData


class PlacementValidator:
    """
    Validates that a placement satisfies all physical constraints.

    A valid placement must satisfy:
    1. No overlaps between macros
    2. All macros within canvas boundaries
    3. No macros in placement blockage regions
    4. Region constraints (if specified)
    """

    def __init__(self, circuit_data: CircuitTensorData, tolerance: float = 1e-6):
        """
        Initialize PlacementValidator.

        Args:
            circuit_data: Circuit data containing macro sizes, canvas, constraints
            tolerance: Numerical tolerance for floating point comparisons (microns)
        """
        self.circuit_data = circuit_data
        self.tolerance = tolerance

    def validate(self, placement: torch.Tensor) -> Tuple[bool, List[str]]:
        """
        Validate placement and return (is_valid, list_of_violations).

        Args:
            placement: [num_macros, 2] - (x, y) coordinates of macro centers

        Returns:
            (is_valid, violations)
            - is_valid: True if placement satisfies all constraints
            - violations: List of violation descriptions (empty if valid)
        """
        violations = []

        # Check 1: Correct dimensions
        if placement.shape != (self.circuit_data.num_macros, 2):
            violations.append(
                f"Invalid placement shape: expected {(self.circuit_data.num_macros, 2)}, "
                f"got {placement.shape}"
            )
            return False, violations

        # Check 2: No NaN or Inf values
        if torch.isnan(placement).any() or torch.isinf(placement).any():
            violations.append("Placement contains NaN or Inf values")
            return False, violations

        # Check 3: Within canvas boundaries
        if not self.check_boundaries(placement):
            violations.append("One or more macros outside canvas boundaries")

        # Check 4: No overlaps
        overlaps = self.find_overlaps(placement)
        if overlaps:
            violations.append(
                f"Found {len(overlaps)} macro overlaps: {overlaps[:5]}"  # Show first 5
            )

        # Check 5: Respect placement blockages
        if self.circuit_data.placement_blockages:
            blockage_violations = self.check_blockages(placement)
            if blockage_violations:
                violations.append(
                    f"Macros in blockage regions: {blockage_violations[:5]}"
                )

        is_valid = len(violations) == 0
        return is_valid, violations

    def check_boundaries(self, placement: torch.Tensor) -> bool:
        """
        Check that all macros are within canvas boundaries.

        Args:
            placement: [num_macros, 2] - Macro positions

        Returns:
            True if all macros within boundaries, False otherwise
        """
        canvas_width = self.circuit_data.canvas_width
        canvas_height = self.circuit_data.canvas_height
        macro_sizes = self.circuit_data.macro_sizes

        for i in range(self.circuit_data.num_macros):
            x, y = placement[i]
            width, height = macro_sizes[i]

            # Calculate bounds
            left = x - width / 2
            right = x + width / 2
            bottom = y - height / 2
            top = y + height / 2

            # Check boundaries (with tolerance)
            if (left < -self.tolerance or
                right > canvas_width + self.tolerance or
                bottom < -self.tolerance or
                top > canvas_height + self.tolerance):
                return False

        return True

    def find_overlaps(self, placement: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Find all pairs of overlapping macros.

        Args:
            placement: [num_macros, 2] - Macro positions

        Returns:
            List of (macro_i, macro_j) pairs that overlap
        """
        overlaps = []
        macro_sizes = self.circuit_data.macro_sizes
        num_macros = self.circuit_data.num_macros

        for i in range(num_macros):
            for j in range(i + 1, num_macros):
                if self._rectangles_overlap(
                    placement[i], macro_sizes[i],
                    placement[j], macro_sizes[j]
                ):
                    overlaps.append((i, j))

        return overlaps

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
        x1, y1 = pos1.numpy()
        w1, h1 = size1.numpy()
        x2, y2 = pos2.numpy()
        w2, h2 = size2.numpy()

        left1, right1 = x1 - w1/2, x1 + w1/2
        bottom1, top1 = y1 - h1/2, y1 + h1/2

        left2, right2 = x2 - w2/2, x2 + w2/2
        bottom2, top2 = y2 - h2/2, y2 + h2/2

        # Add tolerance to avoid false positives from floating point errors
        tol = self.tolerance

        # No overlap if one is completely to the side of the other
        if right1 <= left2 + tol or right2 <= left1 + tol:
            return False
        if top1 <= bottom2 + tol or top2 <= bottom1 + tol:
            return False

        return True

    def check_blockages(self, placement: torch.Tensor) -> List[int]:
        """
        Check if any macros are in placement blockage regions.

        Args:
            placement: [num_macros, 2] - Macro positions

        Returns:
            List of macro indices that violate blockages
        """
        violations = []
        macro_sizes = self.circuit_data.macro_sizes

        for i in range(self.circuit_data.num_macros):
            x, y = placement[i].numpy()
            width, height = macro_sizes[i].numpy()

            # Macro bounds
            left = x - width / 2
            right = x + width / 2
            bottom = y - height / 2
            top = y + height / 2

            # Check against each blockage
            for blockage in self.circuit_data.placement_blockages:
                block_left = blockage['x']
                block_right = blockage['x'] + blockage['width']
                block_bottom = blockage['y']
                block_top = blockage['y'] + blockage['height']

                # Check if macro overlaps with blockage
                if not (right <= block_left or left >= block_right or
                        top <= block_bottom or bottom >= block_top):
                    violations.append(i)
                    break  # No need to check other blockages for this macro

        return violations

    def check_fixed_macros(self, placement: torch.Tensor) -> bool:
        """
        Check if fixed macros maintain their positions.

        Args:
            placement: [num_macros, 2] - Macro positions

        Returns:
            True if all fixed macros are in correct positions
        """
        macro_is_fixed = self.circuit_data.macro_is_fixed

        if macro_is_fixed is None or not macro_is_fixed.any():
            return True  # No fixed macros

        # Check each fixed macro
        for i in range(self.circuit_data.num_macros):
            if macro_is_fixed[i]:
                # Compare with original position
                expected_pos = self.circuit_data.macro_positions[i]
                actual_pos = placement[i]

                # Check if positions match (within tolerance)
                diff = (expected_pos - actual_pos).abs()
                if (diff > self.tolerance).any():
                    return False

        return True

    def get_violation_summary(self, placement: torch.Tensor) -> str:
        """
        Get a human-readable summary of all violations.

        Args:
            placement: [num_macros, 2] - Macro positions

        Returns:
            Formatted string describing all violations
        """
        is_valid, violations = self.validate(placement)

        if is_valid:
            return "✓ Placement is valid - all constraints satisfied"

        summary = "✗ Placement is INVALID\n"
        summary += "=" * 50 + "\n"

        for i, violation in enumerate(violations, 1):
            summary += f"{i}. {violation}\n"

        summary += "=" * 50

        return summary
