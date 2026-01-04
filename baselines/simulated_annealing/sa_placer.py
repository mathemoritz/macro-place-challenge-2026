"""
Simulated Annealing Placer - Classical Optimization Baseline

This implements a simulated annealing algorithm for macro placement.
SA is a probabilistic technique that explores the solution space by
accepting worse solutions with decreasing probability as temperature decreases.

Algorithm:
1. Start with grid-based initial placement
2. Iteratively perturb placement (swap or shift macros)
3. Accept improvements, probabilistically accept worse solutions
4. Cool down temperature using exponential schedule
5. Return best placement found
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import numpy as np
import time
from marco_place.data.tensor_schema import CircuitTensorData
from marco_place.metrics.proxy_cost import compute_proxy_cost
from marco_place.validation.legality import PlacementValidator


class SimulatedAnnealingPlacer:
    """
    Simulated Annealing placer for macro placement.

    Uses probabilistic search to optimize proxy cost.
    """

    def __init__(
        self,
        seed: int = 42,
        max_iterations: int = 5000,
        initial_temp: float = 0.5,
        final_temp: float = 0.001,
        cooling_rate: float = 0.98,
        max_time: float = 3000.0  # 50 minutes (leave buffer)
    ):
        """
        Initialize Simulated Annealing placer.

        Args:
            seed: Random seed for reproducibility
            max_iterations: Maximum number of SA iterations (reduced for speed)
            initial_temp: Starting temperature
            final_temp: Temperature at which to stop
            cooling_rate: Temperature decay factor (T_new = cooling_rate * T_old)
            max_time: Maximum runtime in seconds (with buffer for timeout)
        """
        self.seed = seed
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_time = max_time

        torch.manual_seed(seed)
        np.random.seed(seed)

    def place(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        """
        Generate macro placement using Simulated Annealing.

        Args:
            circuit_data: CircuitTensorData with design information

        Returns:
            placement: [num_macros, 2] tensor of (x, y) coordinates
        """
        start_time = time.time()

        # Generate initial placement (grid-based)
        current_placement = self._initial_placement(circuit_data)

        # Validate initial placement
        validator = PlacementValidator(circuit_data)
        is_valid, _ = validator.validate(current_placement)

        if not is_valid:
            print("Warning: Initial placement invalid, attempting to fix...")
            current_placement = self._fix_placement(current_placement, circuit_data)

        # Compute initial cost
        try:
            current_cost = compute_proxy_cost(current_placement, circuit_data)['proxy_cost']
        except Exception as e:
            print(f"Error computing initial cost: {e}")
            return current_placement

        best_placement = current_placement.clone()
        best_cost = current_cost

        # SA parameters
        temperature = self.initial_temp
        iteration = 0
        accepts = 0
        rejects = 0

        # Main SA loop
        while temperature > self.final_temp and iteration < self.max_iterations:
            # Check timeout
            if time.time() - start_time > self.max_time:
                print(f"SA timeout after {iteration} iterations")
                break

            # Generate neighbor
            neighbor_placement = self._generate_neighbor(
                current_placement, circuit_data, temperature
            )

            # Check validity
            is_valid, _ = validator.validate(neighbor_placement)

            if not is_valid:
                rejects += 1
                iteration += 1
                continue

            # Compute cost
            try:
                neighbor_cost = compute_proxy_cost(neighbor_placement, circuit_data)['proxy_cost']
            except Exception as e:
                rejects += 1
                iteration += 1
                continue

            # Acceptance criterion
            delta_cost = neighbor_cost - current_cost

            if delta_cost < 0:
                # Better solution - always accept
                current_placement = neighbor_placement
                current_cost = neighbor_cost
                accepts += 1

                # Update best
                if current_cost < best_cost:
                    best_placement = current_placement.clone()
                    best_cost = current_cost

            else:
                # Worse solution - accept with probability
                acceptance_prob = np.exp(-delta_cost / temperature)

                if np.random.rand() < acceptance_prob:
                    current_placement = neighbor_placement
                    current_cost = neighbor_cost
                    accepts += 1
                else:
                    rejects += 1

            # Cool down (more frequent cooling)
            if (iteration + 1) % 50 == 0:
                temperature *= self.cooling_rate

                # Progress update every 500 iterations
                if (iteration + 1) % 500 == 0:
                    elapsed = time.time() - start_time
                    accept_rate = accepts / (accepts + rejects) if (accepts + rejects) > 0 else 0
                    print(f"Iter {iteration + 1}: T={temperature:.4f}, "
                          f"cost={current_cost:.6f}, best={best_cost:.6f}, "
                          f"accept={accept_rate:.2%}, time={elapsed:.1f}s")

            iteration += 1

        elapsed = time.time() - start_time
        print(f"SA completed: {iteration} iterations, {elapsed:.2f}s, "
              f"best_cost={best_cost:.6f}")

        return best_placement

    def _initial_placement(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        """Generate initial grid-based placement."""
        num_macros = circuit_data.num_macros
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height
        macro_sizes = circuit_data.macro_sizes

        # Sort by height for better packing
        sorted_indices = torch.argsort(macro_sizes[:, 1], descending=True)

        placement = torch.zeros(num_macros, 2)

        # Row packing
        current_x = 0.0
        current_y = 0.0
        row_height = 0.0
        padding = 5.0  # Padding to avoid tight fits

        for i in sorted_indices:
            width = macro_sizes[i, 0].item() + padding
            height = macro_sizes[i, 1].item() + padding

            # Check if fits in current row
            if current_x + width > canvas_width:
                current_x = 0.0
                current_y += row_height
                row_height = 0.0

            # Place macro
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

    def _generate_neighbor(
        self,
        placement: torch.Tensor,
        circuit_data: CircuitTensorData,
        temperature: float
    ) -> torch.Tensor:
        """
        Generate neighbor placement by perturbing current placement.

        Uses three types of moves:
        1. Swap two macros (40%)
        2. Small random shift (40%)
        3. Large random shift (20%)
        """
        neighbor = placement.clone()
        num_macros = circuit_data.num_macros
        macro_sizes = circuit_data.macro_sizes
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height

        move_type = np.random.rand()

        if move_type < 0.4:
            # Swap two macros
            i, j = np.random.choice(num_macros, size=2, replace=False)
            neighbor[i], neighbor[j] = neighbor[j].clone(), neighbor[i].clone()

        elif move_type < 0.8:
            # Small random shift (temperature-dependent)
            i = np.random.randint(num_macros)
            max_shift = min(canvas_width, canvas_height) * 0.1 * temperature

            dx = (np.random.rand() - 0.5) * 2 * max_shift
            dy = (np.random.rand() - 0.5) * 2 * max_shift

            neighbor[i, 0] += dx
            neighbor[i, 1] += dy

        else:
            # Large random shift
            i = np.random.randint(num_macros)
            neighbor[i, 0] = np.random.rand() * canvas_width
            neighbor[i, 1] = np.random.rand() * canvas_height

        # Ensure within boundaries
        for i in range(num_macros):
            w, h = macro_sizes[i]
            neighbor[i, 0] = max(w/2, min(neighbor[i, 0], canvas_width - w/2))
            neighbor[i, 1] = max(h/2, min(neighbor[i, 1], canvas_height - h/2))

        return neighbor

    def _fix_placement(
        self,
        placement: torch.Tensor,
        circuit_data: CircuitTensorData
    ) -> torch.Tensor:
        """Attempt to fix invalid placement by spreading out macros."""
        num_macros = circuit_data.num_macros
        macro_sizes = circuit_data.macro_sizes
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height

        # Simple fix: add random jitter to overlapping macros
        fixed = placement.clone()

        for i in range(num_macros):
            w, h = macro_sizes[i]

            # Ensure within boundaries
            fixed[i, 0] = max(w/2, min(fixed[i, 0], canvas_width - w/2))
            fixed[i, 1] = max(h/2, min(fixed[i, 1], canvas_height - h/2))

            # Add small jitter
            jitter = torch.randn(2) * 10
            fixed[i] += jitter

            # Re-clamp
            fixed[i, 0] = max(w/2, min(fixed[i, 0], canvas_width - w/2))
            fixed[i, 1] = max(h/2, min(fixed[i, 1], canvas_height - h/2))

        return fixed


if __name__ == "__main__":
    # Test SA placer
    print("Testing Simulated Annealing Placer...")

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

    # Run SA with small iteration count for testing
    placer = SimulatedAnnealingPlacer(
        seed=42,
        max_iterations=500,
        initial_temp=1.0,
        final_temp=0.01,
        cooling_rate=0.95
    )

    placement = placer.place(circuit_data)

    print(f"\nGenerated placement for {circuit_data.num_macros} macros")
    print(f"Placement shape: {placement.shape}")

    # Validate
    from marco_place.validation.legality import PlacementValidator
    validator = PlacementValidator(circuit_data)
    is_valid, violations = validator.validate(placement)

    if is_valid:
        print("✓ Placement is valid!")
    else:
        print(f"✗ Placement invalid: {violations}")

    # Compute cost
    costs = compute_proxy_cost(placement, circuit_data)
    print(f"\nProxy cost: {costs['proxy_cost']:.6f}")
    print(f"  Wirelength: {costs['wirelength_cost']:.6f}")
    print(f"  Density:    {costs['density_cost']:.6f}")
    print(f"  Congestion: {costs['congestion_cost']:.6f}")
