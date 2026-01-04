"""
Congestion metrics for macro placement evaluation.

This module implements congestion estimation using routing demand estimation.
"""

from typing import Tuple
import torch

from marco_place.data.tensor_schema import CircuitTensorData


def compute_congestion_map(
    placement: torch.Tensor,
    circuit_data: CircuitTensorData,
    grid_shape: Tuple[int, int] = (32, 32)
) -> torch.Tensor:
    """
    Estimate routing congestion using bounding box routing model.

    This is a simplified congestion model that estimates routing demand
    by distributing each net's routing along its bounding box perimeter.

    Args:
        placement: [num_nodes, 2] - Node positions
        circuit_data: Circuit data containing nets
        grid_shape: (rows, cols) - Grid resolution

    Returns:
        congestion_map: [grid_rows, grid_cols] - Routing demand in each cell
    """
    grid_rows, grid_cols = grid_shape
    canvas_width = circuit_data.canvas_width
    canvas_height = circuit_data.canvas_height

    # Create congestion map (routing demand per cell)
    congestion_map = torch.zeros(grid_rows, grid_cols)

    cell_width = canvas_width / grid_cols
    cell_height = canvas_height / grid_rows

    # For each net, estimate routing demand
    for net_idx, node_indices in enumerate(circuit_data.net_to_nodes):
        if len(node_indices) < 2:
            continue

        # Get positions of nodes in this net
        net_positions = placement[node_indices]

        # Compute bounding box
        x_coords = net_positions[:, 0]
        y_coords = net_positions[:, 1]

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Get net weight
        net_weight = circuit_data.net_weights[net_idx] if net_idx < len(circuit_data.net_weights) else 1.0

        # Convert to grid coordinates
        col_min = int(torch.floor(x_min / cell_width).item())
        col_max = int(torch.floor(x_max / cell_width).item())
        row_min = int(torch.floor(y_min / cell_height).item())
        row_max = int(torch.floor(y_max / cell_height).item())

        # Clip to grid bounds
        col_min = max(0, min(col_min, grid_cols - 1))
        col_max = max(0, min(col_max, grid_cols - 1))
        row_min = max(0, min(row_min, grid_rows - 1))
        row_max = max(0, min(row_max, grid_rows - 1))

        # Distribute routing demand over bounding box
        # Simplified model: add routing demand to all cells in bounding box
        num_cells = (col_max - col_min + 1) * (row_max - row_min + 1)
        if num_cells > 0:
            demand_per_cell = net_weight / num_cells

            congestion_map[row_min:row_max+1, col_min:col_max+1] += demand_per_cell

    return congestion_map


def compute_congestion_cost(
    placement: torch.Tensor,
    circuit_data: CircuitTensorData,
    grid_shape: Tuple[int, int] = (32, 32),
    alpha: float = 0.5
) -> float:
    """
    Compute congestion cost based on routing demand.

    The cost combines maximum congestion (worst bottleneck) and
    average congestion (overall routing load).

    Args:
        placement: [num_nodes, 2] - Node positions
        circuit_data: Circuit data
        grid_shape: Grid resolution for congestion calculation
        alpha: Weight for max vs mean congestion (0-1)
               alpha=1.0: only penalize max (bottlenecks)
               alpha=0.0: only penalize average

    Returns:
        Congestion cost (normalized)
    """
    congestion_map = compute_congestion_map(placement, circuit_data, grid_shape)

    # Get maximum and mean congestion
    max_congestion = congestion_map.max().item()
    mean_congestion = congestion_map.mean().item()

    # Weighted combination
    # Emphasize worst bottleneck but also consider average
    cost = alpha * max_congestion + (1 - alpha) * mean_congestion

    # Normalize by number of nets to make comparable across designs
    if circuit_data.num_nets > 0:
        cost = cost / circuit_data.num_nets

    return cost


def get_congestion_hotspots(
    placement: torch.Tensor,
    circuit_data: CircuitTensorData,
    grid_shape: Tuple[int, int] = (32, 32),
    threshold: float = 0.8
) -> torch.Tensor:
    """
    Identify congestion hotspots (highly congested regions).

    Args:
        placement: [num_nodes, 2] - Node positions
        circuit_data: Circuit data
        grid_shape: Grid resolution
        threshold: Threshold for hotspot detection (as fraction of max congestion)

    Returns:
        hotspot_mask: [grid_rows, grid_cols] - Boolean mask of hotspots
    """
    congestion_map = compute_congestion_map(placement, circuit_data, grid_shape)

    max_congestion = congestion_map.max()

    # Identify cells with congestion above threshold
    hotspot_threshold = threshold * max_congestion
    hotspot_mask = congestion_map > hotspot_threshold

    return hotspot_mask
