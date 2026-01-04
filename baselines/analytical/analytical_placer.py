"""
Analytical Placer - Force-Directed Placement Baseline

This implements a simplified analytical placement algorithm using force-directed methods.
The approach models the placement problem as a physical system:
- Nets act as springs (attractive forces to minimize wirelength)
- Macros repel each other (repulsive forces to avoid overlaps and spread evenly)

Algorithm:
1. Start with grid-based initial placement
2. Iteratively compute and apply forces until convergence
3. Legalize placement to remove any remaining overlaps
4. Return final placement
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import numpy as np
from marco_place.data.tensor_schema import CircuitTensorData
from marco_place.validation.legality import PlacementValidator


class AnalyticalPlacer:
    """
    Force-directed analytical placer.

    Uses attractive forces (from nets) and repulsive forces (between macros)
    to find a good placement.
    """

    def __init__(
        self,
        seed: int = 42,
        num_iterations: int = 100,
        step_size: float = 0.1,
        attractive_weight: float = 1.0,
        repulsive_weight: float = 0.5,
        legalization_passes: int = 10
    ):
        """
        Initialize Analytical placer.

        Args:
            seed: Random seed for reproducibility
            num_iterations: Number of force-directed iterations
            step_size: Step size for position updates
            attractive_weight: Weight for attractive forces (net connections)
            repulsive_weight: Weight for repulsive forces (macro spreading)
            legalization_passes: Number of legalization passes to remove overlaps
        """
        self.seed = seed
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.attractive_weight = attractive_weight
        self.repulsive_weight = repulsive_weight
        self.legalization_passes = legalization_passes

        torch.manual_seed(seed)
        np.random.seed(seed)

    def place(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        """
        Generate macro placement using force-directed method.

        Args:
            circuit_data: CircuitTensorData with design information

        Returns:
            placement: [num_macros, 2] tensor of (x, y) coordinates
        """
        print(f"Analytical Placer: Placing {circuit_data.num_macros} macros...")

        # Start with grid placement
        placement = self._initial_placement(circuit_data)

        # Force-directed optimization
        placement = self._force_directed_placement(placement, circuit_data)

        # Legalize to remove overlaps
        placement = self._legalize(placement, circuit_data)

        print("Analytical Placer: Placement complete")

        return placement

    def _initial_placement(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        """Generate initial grid-based placement."""
        num_macros = circuit_data.num_macros
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height
        macro_sizes = circuit_data.macro_sizes

        # Sort by height for better packing
        sorted_indices = torch.argsort(macro_sizes[:, 1], descending=True)

        placement = torch.zeros(num_macros, 2)

        # Row packing with generous padding
        current_x = 0.0
        current_y = 0.0
        row_height = 0.0
        padding = 10.0  # Generous padding

        for i in sorted_indices:
            width = macro_sizes[i, 0].item() + padding
            height = macro_sizes[i, 1].item() + padding

            if current_x + width > canvas_width:
                current_x = 0.0
                current_y += row_height
                row_height = 0.0

            x = current_x + width / 2
            y = current_y + height / 2

            # Clamp to boundaries
            x = min(x, canvas_width - width / 2 + padding)
            y = min(y, canvas_height - height / 2 + padding)

            placement[i, 0] = x
            placement[i, 1] = y

            current_x += width
            row_height = max(row_height, height)

        return placement

    def _force_directed_placement(
        self,
        placement: torch.Tensor,
        circuit_data: CircuitTensorData
    ) -> torch.Tensor:
        """
        Apply force-directed optimization.

        Computes attractive forces (from nets) and repulsive forces (between macros).
        """
        placement = placement.clone()
        num_macros = circuit_data.num_macros
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height
        macro_sizes = circuit_data.macro_sizes

        # If no netlist, skip force-directed step
        if circuit_data.net_to_nodes is None or len(circuit_data.net_to_nodes) == 0:
            return placement

        for iteration in range(self.num_iterations):
            forces = torch.zeros_like(placement)

            # Attractive forces from nets
            for net_idx, node_indices in enumerate(circuit_data.net_to_nodes):
                if len(node_indices) < 2:
                    continue

                # Compute center of net
                net_positions = placement[node_indices]
                center = net_positions.mean(dim=0)

                # Pull each macro toward center
                for node_idx in node_indices:
                    diff = center - placement[node_idx]
                    force = self.attractive_weight * diff
                    forces[node_idx] += force

            # Repulsive forces between macros (prevent overlaps)
            for i in range(num_macros):
                for j in range(i + 1, num_macros):
                    diff = placement[i] - placement[j]
                    distance = torch.norm(diff) + 1e-6  # Avoid division by zero

                    # Stronger repulsion for closer macros
                    repulsion_strength = self.repulsive_weight / (distance ** 2 + 1)
                    force = repulsion_strength * diff / distance

                    forces[i] += force
                    forces[j] -= force

            # Update positions
            placement += self.step_size * forces

            # Clamp to boundaries (accounting for macro sizes)
            for i in range(num_macros):
                w, h = macro_sizes[i]
                placement[i, 0] = max(w/2, min(placement[i, 0], canvas_width - w/2))
                placement[i, 1] = max(h/2, min(placement[i, 1], canvas_height - h/2))

            # Reduce step size over time (simulated annealing-like)
            if (iteration + 1) % 20 == 0:
                self.step_size *= 0.9

        return placement

    def _legalize(
        self,
        placement: torch.Tensor,
        circuit_data: CircuitTensorData
    ) -> torch.Tensor:
        """
        Legalize placement by removing overlaps.

        Uses a simple greedy approach: for each overlapping pair,
        push macros apart along the shortest separation vector.
        """
        placement = placement.clone()
        num_macros = circuit_data.num_macros
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height
        macro_sizes = circuit_data.macro_sizes

        for pass_idx in range(self.legalization_passes):
            overlaps_found = False

            for i in range(num_macros):
                for j in range(i + 1, num_macros):
                    # Check for overlap
                    if self._check_overlap(
                        placement[i], macro_sizes[i],
                        placement[j], macro_sizes[j]
                    ):
                        overlaps_found = True

                        # Compute separation vector
                        diff = placement[i] - placement[j]
                        distance = torch.norm(diff)

                        if distance < 1e-6:
                            # Macros at same position - add random offset
                            diff = torch.randn(2) * 10
                            distance = torch.norm(diff)

                        # Required separation
                        required_sep = (macro_sizes[i] + macro_sizes[j]).norm() / 2 * 1.1

                        # Push apart
                        push = (required_sep - distance) / 2
                        direction = diff / distance

                        placement[i] += push * direction
                        placement[j] -= push * direction

            # Clamp to boundaries
            for i in range(num_macros):
                w, h = macro_sizes[i]
                placement[i, 0] = max(w/2, min(placement[i, 0], canvas_width - w/2))
                placement[i, 1] = max(h/2, min(placement[i, 1], canvas_height - h/2))

            if not overlaps_found:
                print(f"Legalization converged after {pass_idx + 1} passes")
                break

        return placement

    def _check_overlap(
        self,
        pos1: torch.Tensor,
        size1: torch.Tensor,
        pos2: torch.Tensor,
        size2: torch.Tensor
    ) -> bool:
        """Check if two macros overlap."""
        x1, y1 = pos1
        w1, h1 = size1
        x2, y2 = pos2
        w2, h2 = size2

        left1, right1 = x1 - w1/2, x1 + w1/2
        bottom1, top1 = y1 - h1/2, y1 + h1/2
        left2, right2 = x2 - w2/2, x2 + w2/2
        bottom2, top2 = y2 - h2/2, y2 + h2/2

        return not (right1 <= left2 or right2 <= left1 or
                   top1 <= bottom2 or top2 <= bottom1)


if __name__ == "__main__":
    # Test analytical placer
    print("Testing Analytical Placer...")

    # Create test circuit
    metadata = {
        'design_name': 'test',
        'num_macros': 20,
        'canvas_width': 2000.0,
        'canvas_height': 2000.0,
    }

    macro_sizes = torch.rand(20, 2) * 80 + 40  # Sizes 40-120

    # Create simple nets
    net_to_nodes = [
        torch.tensor([i, (i+1) % 20]) for i in range(20)
    ]
    net_weights = torch.ones(20)

    circuit_data = CircuitTensorData(
        metadata=metadata,
        macro_positions=torch.zeros(20, 2),
        macro_sizes=macro_sizes,
        net_to_nodes=net_to_nodes,
        net_weights=net_weights,
    )

    # Run analytical placer
    placer = AnalyticalPlacer(
        seed=42,
        num_iterations=100,
        step_size=0.1,
        attractive_weight=1.0,
        repulsive_weight=0.5,
        legalization_passes=10
    )

    placement = placer.place(circuit_data)

    print(f"\nGenerated placement for {circuit_data.num_macros} macros")
    print(f"Placement shape: {placement.shape}")

    # Validate
    validator = PlacementValidator(circuit_data)
    is_valid, violations = validator.validate(placement)

    if is_valid:
        print("✓ Placement is valid!")
    else:
        print(f"✗ Placement invalid: {violations}")

    # Compute cost
    from marco_place.metrics.proxy_cost import compute_proxy_cost
    costs = compute_proxy_cost(placement, circuit_data)
    print(f"\nProxy cost: {costs['proxy_cost']:.6f}")
    print(f"  Wirelength: {costs['wirelength_cost']:.6f}")
    print(f"  Density:    {costs['density_cost']:.6f}")
    print(f"  Congestion: {costs['congestion_cost']:.6f}")
