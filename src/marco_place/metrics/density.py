"""
Density metrics for macro placement evaluation.

This module implements density calculation and cost functions.
"""

from typing import Tuple
import torch

from marco_place.data.tensor_schema import CircuitTensorData


def compute_density_map(
    macro_positions: torch.Tensor,
    macro_sizes: torch.Tensor,
    grid_shape: Tuple[int, int],
    canvas_size: Tuple[float, float]
) -> torch.Tensor:
    """
    Compute density map on grid.

    For each grid cell, calculates the total macro area overlapping with that cell.

    Args:
        macro_positions: [num_macros, 2] - (x, y) center coordinates
        macro_sizes: [num_macros, 2] - (width, height)
        grid_shape: (rows, cols) - number of grid cells
        canvas_size: (width, height) - canvas dimensions in microns

    Returns:
        density_map: [grid_rows, grid_cols] - density in each grid cell (0-1 scale)
    """
    grid_rows, grid_cols = grid_shape
    canvas_width, canvas_height = canvas_size

    density_map = torch.zeros(grid_rows, grid_cols)

    cell_width = canvas_width / grid_cols
    cell_height = canvas_height / grid_rows
    cell_area = cell_width * cell_height

    num_macros = len(macro_positions)

    for i in range(num_macros):
        x, y = macro_positions[i]
        w, h = macro_sizes[i]

        # Find grid cells overlapping with this macro
        # Macro bounds
        left = x - w / 2
        right = x + w / 2
        bottom = y - h / 2
        top = y + h / 2

        # Convert to grid indices
        col_start = int(torch.floor(left / cell_width).item())
        col_end = int(torch.floor(right / cell_width).item())
        row_start = int(torch.floor(bottom / cell_height).item())
        row_end = int(torch.floor(top / cell_height).item())

        # Clip to grid bounds
        col_start = max(0, min(col_start, grid_cols - 1))
        col_end = max(0, min(col_end, grid_cols - 1))
        row_start = max(0, min(row_start, grid_rows - 1))
        row_end = max(0, min(row_end, grid_rows - 1))

        # Add macro area to overlapping cells
        # Simplified: distribute area evenly across overlapping cells
        num_cells = (col_end - col_start + 1) * (row_end - row_start + 1)
        if num_cells > 0:
            macro_area = w * h
            area_per_cell = macro_area / num_cells

            density_map[row_start:row_end+1, col_start:col_end+1] += area_per_cell / cell_area

    return density_map


def compute_density_cost(
    placement: torch.Tensor,
    circuit_data: CircuitTensorData,
    target_density: float = 0.6,
    grid_shape: Tuple[int, int] = (32, 32)
) -> float:
    """
    Density cost based on deviation from target density.

    Penalizes both over-dense and under-dense regions using mean squared error.

    Args:
        placement: [num_macros, 2] - Macro positions
        circuit_data: Circuit data containing macro sizes and canvas info
        target_density: Target density (default: 0.6)
        grid_shape: Grid resolution for density calculation

    Returns:
        Density cost (normalized, typically 0-1 range)
    """
    density_map = compute_density_map(
        placement,
        circuit_data.macro_sizes,
        grid_shape,
        (circuit_data.canvas_width, circuit_data.canvas_height)
    )

    # Cost is mean squared deviation from target
    deviation = (density_map - target_density).pow(2)
    cost = deviation.mean().item()

    return cost
