"""
Tensor representation of circuit netlists for macro placement.

This module defines the core data structure (CircuitTensorData) that represents
a circuit netlist as PyTorch tensors, suitable for optimization and ML approaches.
"""

from typing import Dict, List, Any, Optional
import torch


class CircuitTensorData:
    """
    PyTorch tensor representation of a circuit netlist for macro placement.

    This class stores the essential information needed to perform macro placement:
    - Macro positions and sizes
    - Net connectivity (hypergraph)
    - Canvas dimensions and constraints

    All spatial coordinates are in microns.
    """

    def __init__(
        self,
        # Metadata
        metadata: Dict[str, Any],
        # Macro data
        macro_positions: torch.Tensor,
        macro_sizes: torch.Tensor,
        macro_is_fixed: Optional[torch.Tensor] = None,
        # Netlist (hypergraph)
        net_to_nodes: Optional[List[torch.Tensor]] = None,
        net_weights: Optional[torch.Tensor] = None,
        # Extended node data (optional)
        node_features: Optional[torch.Tensor] = None,
        node_types: Optional[torch.Tensor] = None,
        # Standard cell data (optional)
        stdcell_positions: Optional[torch.Tensor] = None,
        stdcell_sizes: Optional[torch.Tensor] = None,
        # Port data (optional)
        port_positions: Optional[torch.Tensor] = None,
        port_sides: Optional[torch.Tensor] = None,
        # Constraints (optional)
        placement_blockages: Optional[List[Dict]] = None,
        # Grid information (optional)
        grid_rows: Optional[int] = None,
        grid_cols: Optional[int] = None,
        # Additional fields
        **kwargs
    ):
        """
        Initialize CircuitTensorData.

        Args:
            metadata: Dictionary containing:
                - design_name (str): Name of the design
                - num_macros (int): Number of macros
                - canvas_width (float): Canvas width in microns
                - canvas_height (float): Canvas height in microns
                - target_density (float, optional): Target placement density (default: 0.6)
                - num_stdcells (int, optional): Number of standard cells
                - num_ports (int, optional): Number of I/O ports

            macro_positions: [num_macros, 2] - (x, y) coordinates of macro centers
            macro_sizes: [num_macros, 2] - (width, height) of each macro
            macro_is_fixed: [num_macros] - Boolean mask for fixed macros (optional)

            net_to_nodes: List of tensors, each containing node indices for one net
            net_weights: [num_nets] - Weight for each net in wirelength calculation

            node_features: [num_nodes, feature_dim] - Node features (optional)
            node_types: [num_nodes] - Node types: 0=macro, 1=stdcell, 2=port (optional)

            stdcell_positions: [num_stdcells, 2] - Standard cell positions (optional)
            stdcell_sizes: [num_stdcells, 2] - Standard cell sizes (optional)

            port_positions: [num_ports, 2] - I/O port positions (optional)
            port_sides: [num_ports] - Port sides: 0=left, 1=right, 2=top, 3=bottom (optional)

            placement_blockages: List of dicts with 'x', 'y', 'width', 'height' (optional)

            grid_rows: Number of placement grid rows (optional)
            grid_cols: Number of placement grid columns (optional)

            **kwargs: Additional fields for extended functionality
        """
        # Validate metadata
        required_keys = ['design_name', 'num_macros', 'canvas_width', 'canvas_height']
        for key in required_keys:
            if key not in metadata:
                raise ValueError(f"Missing required metadata key: {key}")

        # Store metadata
        self.metadata = metadata
        if 'target_density' not in self.metadata:
            self.metadata['target_density'] = 0.6

        # Store macro data
        self.macro_positions = macro_positions
        self.macro_sizes = macro_sizes

        if macro_is_fixed is None:
            # By default, no macros are fixed
            self.macro_is_fixed = torch.zeros(metadata['num_macros'], dtype=torch.bool)
        else:
            self.macro_is_fixed = macro_is_fixed

        # Store netlist data
        if net_to_nodes is None:
            self.net_to_nodes = []
        else:
            self.net_to_nodes = net_to_nodes

        if net_weights is None:
            # Default: equal weight for all nets
            self.net_weights = torch.ones(len(self.net_to_nodes))
        else:
            self.net_weights = net_weights

        # Store extended node data
        self.node_features = node_features
        self.node_types = node_types

        # Store standard cell data
        self.stdcell_positions = stdcell_positions
        self.stdcell_sizes = stdcell_sizes

        # Store port data
        self.port_positions = port_positions
        self.port_sides = port_sides

        # Store constraints
        self.placement_blockages = placement_blockages if placement_blockages else []

        # Store grid information
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        # Store additional fields
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def num_macros(self) -> int:
        """Number of macros in the design."""
        return self.metadata['num_macros']

    @property
    def num_stdcells(self) -> int:
        """Number of standard cells (clustered) in the design."""
        return self.metadata.get('num_stdcells', 0)

    @property
    def num_nets(self) -> int:
        """Number of nets in the design."""
        return len(self.net_to_nodes)

    @property
    def canvas_width(self) -> float:
        """Canvas width in microns."""
        return self.metadata['canvas_width']

    @property
    def canvas_height(self) -> float:
        """Canvas height in microns."""
        return self.metadata['canvas_height']

    @property
    def design_name(self) -> str:
        """Design name."""
        return self.metadata['design_name']

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary containing all data fields
        """
        data = {
            'version': '1.0',
            'metadata': self.metadata,
            'macro_positions': self.macro_positions,
            'macro_sizes': self.macro_sizes,
            'macro_is_fixed': self.macro_is_fixed,
            'net_to_nodes': self.net_to_nodes,
            'net_weights': self.net_weights,
        }

        # Add optional fields if present
        if self.node_features is not None:
            data['node_features'] = self.node_features
        if self.node_types is not None:
            data['node_types'] = self.node_types
        if self.stdcell_positions is not None:
            data['stdcell_positions'] = self.stdcell_positions
        if self.stdcell_sizes is not None:
            data['stdcell_sizes'] = self.stdcell_sizes
        if self.port_positions is not None:
            data['port_positions'] = self.port_positions
        if self.port_sides is not None:
            data['port_sides'] = self.port_sides
        if self.placement_blockages:
            data['placement_blockages'] = self.placement_blockages
        if self.grid_rows is not None:
            data['grid_rows'] = self.grid_rows
        if self.grid_cols is not None:
            data['grid_cols'] = self.grid_cols

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CircuitTensorData':
        """
        Create CircuitTensorData from dictionary.

        Args:
            data: Dictionary containing circuit data

        Returns:
            CircuitTensorData instance
        """
        # Extract required fields
        metadata = data['metadata']
        macro_positions = data['macro_positions']
        macro_sizes = data['macro_sizes']

        # Extract optional fields
        macro_is_fixed = data.get('macro_is_fixed')
        net_to_nodes = data.get('net_to_nodes')
        net_weights = data.get('net_weights')
        node_features = data.get('node_features')
        node_types = data.get('node_types')
        stdcell_positions = data.get('stdcell_positions')
        stdcell_sizes = data.get('stdcell_sizes')
        port_positions = data.get('port_positions')
        port_sides = data.get('port_sides')
        placement_blockages = data.get('placement_blockages')
        grid_rows = data.get('grid_rows')
        grid_cols = data.get('grid_cols')

        return cls(
            metadata=metadata,
            macro_positions=macro_positions,
            macro_sizes=macro_sizes,
            macro_is_fixed=macro_is_fixed,
            net_to_nodes=net_to_nodes,
            net_weights=net_weights,
            node_features=node_features,
            node_types=node_types,
            stdcell_positions=stdcell_positions,
            stdcell_sizes=stdcell_sizes,
            port_positions=port_positions,
            port_sides=port_sides,
            placement_blockages=placement_blockages,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )

    def save(self, path: str):
        """
        Save to file using PyTorch's save function.

        Args:
            path: Output file path (should end in .pt)
        """
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str) -> 'CircuitTensorData':
        """
        Load from file.

        Args:
            path: Input file path

        Returns:
            CircuitTensorData instance
        """
        data = torch.load(path)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        parts = [
            f"design='{self.design_name}'",
            f"num_macros={self.num_macros}",
        ]
        if self.num_stdcells > 0:
            parts.append(f"num_stdcells={self.num_stdcells}")
        parts.extend([
            f"num_nets={self.num_nets}",
            f"canvas={self.canvas_width:.1f}x{self.canvas_height:.1f}um"
        ])
        return f"CircuitTensorData({', '.join(parts)})"
