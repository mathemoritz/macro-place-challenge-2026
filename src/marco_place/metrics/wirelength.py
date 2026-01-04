"""
Wirelength metrics for macro placement evaluation.

This module implements Half-Perimeter Wirelength (HPWL), a standard metric
for estimating routing wirelength in VLSI placement.
"""

from typing import List, Optional
import torch

from marco_place.data.tensor_schema import CircuitTensorData


def compute_hpwl(
    placement: torch.Tensor,
    net_to_nodes: List[torch.Tensor],
    net_weights: Optional[torch.Tensor] = None
) -> float:
    """
    Compute Half-Perimeter Wirelength (HPWL).

    For each net, HPWL is calculated as:
        HPWL = (max_x - min_x) + (max_y - min_y)

    This represents the half-perimeter of the bounding box containing all
    pins in the net, which is a standard approximation for routing length.

    Args:
        placement: [num_nodes, 2] - (x, y) coordinates of all nodes
        net_to_nodes: List of tensors, each containing node indices for one net
        net_weights: [num_nets] - Optional weights for each net (default: all 1.0)

    Returns:
        Total weighted HPWL across all nets
    """
    if len(net_to_nodes) == 0:
        return 0.0

    if net_weights is None:
        net_weights = torch.ones(len(net_to_nodes))

    total_hpwl = 0.0

    for net_idx, node_indices in enumerate(net_to_nodes):
        if len(node_indices) == 0:
            continue

        # Get positions of all nodes in this net
        positions = placement[node_indices]  # [num_pins_in_net, 2]

        # Compute bounding box
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Half-perimeter wirelength
        hpwl = (x_max - x_min) + (y_max - y_min)

        # Apply weight
        weighted_hpwl = hpwl * net_weights[net_idx]
        total_hpwl += weighted_hpwl.item()

    return total_hpwl


def compute_hpwl_from_circuit(
    macro_placement: torch.Tensor,
    circuit_data: CircuitTensorData
) -> float:
    """
    Compute HPWL for a macro placement given circuit data.

    This is a convenience function that extracts the necessary information
    from CircuitTensorData and calls compute_hpwl().

    Args:
        macro_placement: [num_macros, 2] - (x, y) coordinates of macro centers
        circuit_data: CircuitTensorData containing net connectivity

    Returns:
        Total weighted HPWL
    """
    return compute_hpwl(
        placement=macro_placement,
        net_to_nodes=circuit_data.net_to_nodes,
        net_weights=circuit_data.net_weights
    )


def compute_normalized_wirelength(
    macro_placement: torch.Tensor,
    circuit_data: CircuitTensorData
) -> float:
    """
    Compute normalized wirelength cost for proxy cost calculation.

    The wirelength is normalized by the canvas perimeter and number of nets
    to produce a dimensionless cost metric between 0 and ~1.

    Args:
        macro_placement: [num_macros, 2] - (x, y) coordinates of macro centers
        circuit_data: CircuitTensorData containing net connectivity and canvas info

    Returns:
        Normalized wirelength cost
    """
    hpwl = compute_hpwl_from_circuit(macro_placement, circuit_data)

    # Normalize by canvas size and number of nets
    canvas_perimeter = 2 * (circuit_data.canvas_width + circuit_data.canvas_height)
    num_nets = circuit_data.num_nets

    if num_nets == 0:
        return 0.0

    normalized = hpwl / (canvas_perimeter * num_nets)

    return normalized
