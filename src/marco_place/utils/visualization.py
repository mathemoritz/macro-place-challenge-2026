"""
Visualization utilities for macro placement.

This module provides functions to visualize placements, including:
- Macro positions and sizes
- Canvas boundaries
- Placement blockages
- Net connections
"""

from typing import Optional, Tuple
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure

from marco_place.data.tensor_schema import CircuitTensorData


def plot_placement(
    placement: torch.Tensor,
    circuit_data: CircuitTensorData,
    show_nets: bool = False,
    max_nets_to_show: int = 100,
    figsize: Tuple[int, int] = (12, 12),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot macro placement on canvas.

    Args:
        placement: [num_macros, 2] - (x, y) coordinates of macro centers
        circuit_data: Circuit data containing macro sizes and canvas info
        show_nets: Whether to draw net connections (can be slow for large designs)
        max_nets_to_show: Maximum number of nets to draw (if show_nets=True)
        figsize: Figure size (width, height) in inches
        title: Optional title for the plot
        ax: Optional matplotlib axes to plot on (if None, creates new figure)

    Returns:
        matplotlib Figure object
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    canvas_width = circuit_data.canvas_width
    canvas_height = circuit_data.canvas_height
    macro_sizes = circuit_data.macro_sizes

    # Set axis limits
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.set_aspect('equal')

    # Draw canvas boundary
    canvas_rect = patches.Rectangle(
        (0, 0), canvas_width, canvas_height,
        linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
    )
    ax.add_patch(canvas_rect)

    # Draw placement blockages
    if circuit_data.placement_blockages:
        for blockage in circuit_data.placement_blockages:
            block_rect = patches.Rectangle(
                (blockage['x'], blockage['y']),
                blockage['width'], blockage['height'],
                linewidth=1, edgecolor='red', facecolor='red',
                alpha=0.3, linestyle='-'
            )
            ax.add_patch(block_rect)

    # Draw nets (if requested and not too many)
    if show_nets and len(circuit_data.net_to_nodes) > 0:
        num_nets_to_draw = min(len(circuit_data.net_to_nodes), max_nets_to_show)

        for net_idx in range(num_nets_to_draw):
            node_indices = circuit_data.net_to_nodes[net_idx]
            if len(node_indices) < 2:
                continue

            # Get positions of nodes in this net
            net_positions = placement[node_indices].numpy()

            # Draw bounding box
            x_coords = net_positions[:, 0]
            y_coords = net_positions[:, 1]

            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            # Draw as rectangle (half-perimeter bounding box)
            net_rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=0.5, edgecolor='blue', facecolor='none',
                alpha=0.3, linestyle='-'
            )
            ax.add_patch(net_rect)

    # Draw macros
    num_macros = circuit_data.num_macros
    colors = plt.cm.tab20(torch.linspace(0, 1, min(num_macros, 20)))

    for i in range(num_macros):
        x, y = placement[i].numpy()
        width, height = macro_sizes[i].numpy()

        # Calculate rectangle corner (bottom-left)
        rect_x = x - width / 2
        rect_y = y - height / 2

        # Choose color
        color = colors[i % len(colors)]

        # Draw macro rectangle
        macro_rect = patches.Rectangle(
            (rect_x, rect_y), width, height,
            linewidth=1.5, edgecolor='darkblue', facecolor=color,
            alpha=0.7
        )
        ax.add_patch(macro_rect)

        # Add label (for smaller designs)
        if num_macros <= 50:
            ax.text(
                x, y, f'M{i}',
                ha='center', va='center',
                fontsize=8, fontweight='bold', color='white'
            )

    # Set title
    if title is None:
        title = f"{circuit_data.design_name} - Macro Placement\n"
        title += f"{num_macros} macros, Canvas: {canvas_width:.0f}x{canvas_height:.0f} um"
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set labels
    ax.set_xlabel('X (um)', fontsize=12)
    ax.set_ylabel('Y (um)', fontsize=12)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    plt.tight_layout()

    return fig


def plot_comparison(
    placements: dict,
    circuit_data: CircuitTensorData,
    figsize: Tuple[int, int] = (18, 6)
) -> Figure:
    """
    Plot multiple placements side-by-side for comparison.

    Args:
        placements: Dictionary mapping labels to placement tensors
                   e.g., {'Random': placement1, 'SA': placement2}
        circuit_data: Circuit data
        figsize: Figure size (width, height) in inches

    Returns:
        matplotlib Figure object
    """
    num_placements = len(placements)

    fig, axes = plt.subplots(1, num_placements, figsize=figsize)

    if num_placements == 1:
        axes = [axes]

    for ax, (label, placement) in zip(axes, placements.items()):
        plot_placement(
            placement,
            circuit_data,
            show_nets=False,
            title=f"{label}\n{circuit_data.design_name}",
            ax=ax
        )

    plt.tight_layout()

    return fig


def plot_density_map(
    placement: torch.Tensor,
    circuit_data: CircuitTensorData,
    grid_shape: Tuple[int, int] = (20, 20),
    figsize: Tuple[int, int] = (10, 10)
) -> Figure:
    """
    Plot placement density heatmap.

    Args:
        placement: [num_macros, 2] - Macro positions
        circuit_data: Circuit data
        grid_shape: (rows, cols) for density grid
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    from marco_place.metrics.density import compute_density_map

    # Compute density map
    density_map = compute_density_map(
        placement,
        circuit_data.macro_sizes,
        grid_shape,
        (circuit_data.canvas_width, circuit_data.canvas_height)
    )

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot heatmap
    im = ax.imshow(
        density_map.numpy(),
        origin='lower',
        extent=[0, circuit_data.canvas_width, 0, circuit_data.canvas_height],
        cmap='hot',
        alpha=0.7
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Density', fontsize=12)

    # Draw macro outlines
    macro_sizes = circuit_data.macro_sizes
    for i in range(len(placement)):
        x, y = placement[i].numpy()
        width, height = macro_sizes[i].numpy()

        rect_x = x - width / 2
        rect_y = y - height / 2

        macro_rect = patches.Rectangle(
            (rect_x, rect_y), width, height,
            linewidth=1, edgecolor='cyan', facecolor='none'
        )
        ax.add_patch(macro_rect)

    # Set labels
    ax.set_xlim(0, circuit_data.canvas_width)
    ax.set_ylim(0, circuit_data.canvas_height)
    ax.set_aspect('equal')
    ax.set_xlabel('X (um)', fontsize=12)
    ax.set_ylabel('Y (um)', fontsize=12)
    ax.set_title(
        f'Density Map - {circuit_data.design_name}\n'
        f'Grid: {grid_shape[0]}x{grid_shape[1]}',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()

    return fig


def save_placement_figure(
    fig: Figure,
    output_path: str,
    dpi: int = 150
):
    """
    Save placement figure to file.

    Args:
        fig: Matplotlib figure to save
        output_path: Output file path (e.g., 'placement.png')
        dpi: Resolution in dots per inch
    """
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
