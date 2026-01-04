# Tensor Format Specification

This document describes the `CircuitTensorData` format used to represent circuit benchmarks in the Macro Placement Challenge.

## Table of Contents

1. [Overview](#overview)
2. [CircuitTensorData Class](#circuittensordata-class)
3. [Field Descriptions](#field-descriptions)
4. [Usage Examples](#usage-examples)
5. [Loading and Saving](#loading-and-saving)
6. [Advanced Topics](#advanced-topics)

---

## Overview

Circuits are represented as **PyTorch tensors** for efficient computation and ML integration. The `CircuitTensorData` class encapsulates all information needed for macro placement:

- Macro dimensions and positions
- Netlist connectivity (hypergraph)
- Canvas constraints
- Optional metadata

This format is inspired by PyTorch Geometric's graph data structures and is optimized for:
- Fast tensor operations
- GPU acceleration (if needed)
- Compatibility with GNNs and other ML models

---

## CircuitTensorData Class

### Location

```python
from marco_place.data.tensor_schema import CircuitTensorData
```

### Constructor

```python
CircuitTensorData(
    metadata: Dict[str, Any],
    macro_positions: torch.Tensor,
    macro_sizes: torch.Tensor,
    macro_is_fixed: Optional[torch.Tensor] = None,
    net_to_nodes: Optional[List[torch.Tensor]] = None,
    net_weights: Optional[torch.Tensor] = None,
    node_features: Optional[torch.Tensor] = None,
    node_types: Optional[torch.Tensor] = None,
    stdcell_positions: Optional[torch.Tensor] = None,
    stdcell_sizes: Optional[torch.Tensor] = None,
    port_positions: Optional[torch.Tensor] = None,
    port_sides: Optional[torch.Tensor] = None,
    placement_blockages: Optional[List[Dict]] = None
)
```

---

## Field Descriptions

### Required Fields

#### `metadata: Dict[str, Any]`

Design-level information stored as a dictionary.

**Required keys**:
- `design_name` (str): Name of the circuit design
- `num_macros` (int): Number of hard macros
- `canvas_width` (float): Canvas width in microns
- `canvas_height` (float): Canvas height in microns

**Optional keys**:
- `target_density` (float): Target utilization (default: 0.6)
- `num_ports` (int): Number of I/O ports
- `num_soft_macros` (int): Number of soft macros (standard cells)
- `num_nets` (int): Total number of nets

**Example**:
```python
metadata = {
    'design_name': 'ariane133',
    'num_macros': 133,
    'canvas_width': 1302.8,
    'canvas_height': 1302.8,
    'target_density': 0.6
}
```

#### `macro_positions: torch.Tensor`

Initial positions of hard macros (typically from previous placement or initialization).

- **Shape**: `[num_macros, 2]`
- **Type**: `torch.float32`
- **Format**: `[[x0, y0], [x1, y1], ..., [xN, yN]]`
- **Units**: Microns
- **Coordinates**: (x, y) represent the **center** of each macro

**Example**:
```python
macro_positions = torch.tensor([
    [651.4, 651.4],  # Macro 0 center at (651.4, 651.4)
    [195.4, 195.4],  # Macro 1 center at (195.4, 195.4)
    # ...
])
```

#### `macro_sizes: torch.Tensor`

Dimensions of each hard macro.

- **Shape**: `[num_macros, 2]`
- **Type**: `torch.float32`
- **Format**: `[[width0, height0], [width1, height1], ...]`
- **Units**: Microns

**Example**:
```python
macro_sizes = torch.tensor([
    [128.5, 64.2],   # Macro 0: 128.5 μm wide, 64.2 μm tall
    [256.0, 128.0],  # Macro 1: 256.0 μm wide, 128.0 μm tall
    # ...
])
```

**Note**: For a macro with center (x, y) and size (w, h):
- Left edge: `x - w/2`
- Right edge: `x + w/2`
- Bottom edge: `y - h/2`
- Top edge: `y + h/2`

---

### Optional Fields (Netlist)

#### `net_to_nodes: List[torch.Tensor]`

Hypergraph connectivity representing the netlist. Each net connects multiple nodes (macros).

- **Type**: List of `torch.LongTensor`
- **Length**: Number of nets
- **Element**: Each element is a tensor of node indices connected by that net

**Example**:
```python
net_to_nodes = [
    torch.tensor([0, 5, 12]),      # Net 0 connects macros 0, 5, and 12
    torch.tensor([1, 2]),           # Net 1 connects macros 1 and 2
    torch.tensor([3, 7, 9, 15]),   # Net 2 connects macros 3, 7, 9, and 15
    # ...
]
```

**Notes**:
- Node indices are 0-based
- A net with 2 nodes is an edge
- A net with 3+ nodes is a hyperedge
- For this competition, nets only include macro-to-macro connections (ports and standard cells are pre-filtered)

#### `net_weights: torch.Tensor`

Importance/weight of each net (used in wirelength calculation).

- **Shape**: `[num_nets]`
- **Type**: `torch.float32`
- **Default**: All weights = 1.0

**Example**:
```python
net_weights = torch.tensor([
    1.0,  # Net 0 has default weight
    2.5,  # Net 1 is 2.5x more important
    0.5,  # Net 2 is less critical
    # ...
])
```

---

### Optional Fields (Advanced)

#### `macro_is_fixed: torch.Tensor`

Boolean mask indicating which macros have fixed positions (cannot be moved).

- **Shape**: `[num_macros]`
- **Type**: `torch.bool`
- **Default**: `None` (all macros are movable)

**Example**:
```python
macro_is_fixed = torch.tensor([False, False, True, False, ...])
# Macro 2 is fixed in place
```

#### `node_features: torch.Tensor`

Feature vectors for each node (e.g., for GNN models).

- **Shape**: `[num_nodes, feature_dim]`
- **Type**: `torch.float32`
- **Default**: `None`

**Example**:
```python
# Features could include: area, aspect ratio, connectivity degree, etc.
node_features = torch.tensor([
    [128.5 * 64.2, 2.0, 5.0],  # Macro 0: area, aspect ratio, degree
    [256.0 * 128.0, 2.0, 3.0], # Macro 1
    # ...
])
```

#### `node_types: torch.Tensor`

Type indicator for each node.

- **Shape**: `[num_nodes]`
- **Type**: `torch.long`
- **Values**:
  - `0`: Hard macro
  - `1`: Standard cell / soft macro
  - `2`: I/O port

**Example**:
```python
node_types = torch.tensor([0, 0, 0, 1, 1, 2, ...])
# First 3 nodes are macros, next 2 are standard cells, then a port
```

#### `placement_blockages: List[Dict]`

Regions where macros cannot be placed (e.g., power grid, pre-placed blocks).

- **Type**: List of dictionaries
- **Format**: Each dict has keys: `x`, `y`, `width`, `height`
- **Units**: Microns
- **Coordinates**: (x, y) is the bottom-left corner

**Example**:
```python
placement_blockages = [
    {'x': 100.0, 'y': 100.0, 'width': 50.0, 'height': 50.0},
    {'x': 500.0, 'y': 500.0, 'width': 100.0, 'height': 75.0},
]
```

---

## Usage Examples

### Basic Usage

```python
import torch
from marco_place.data.tensor_schema import CircuitTensorData

# Create minimal circuit data
metadata = {
    'design_name': 'my_design',
    'num_macros': 10,
    'canvas_width': 1000.0,
    'canvas_height': 1000.0,
}

macro_positions = torch.zeros(10, 2)  # Initial positions
macro_sizes = torch.rand(10, 2) * 100 + 50  # Random sizes 50-150

circuit_data = CircuitTensorData(
    metadata=metadata,
    macro_positions=macro_positions,
    macro_sizes=macro_sizes,
)

print(circuit_data)
# CircuitTensorData(design='my_design', num_macros=10, num_nets=0, canvas=1000.0x1000.0um)
```

### With Netlist

```python
# Define nets
net_to_nodes = [
    torch.tensor([0, 1, 2]),     # Net connects macros 0, 1, 2
    torch.tensor([1, 3]),         # Net connects macros 1, 3
    torch.tensor([2, 3, 4, 5]),  # Net connects macros 2, 3, 4, 5
]

net_weights = torch.ones(len(net_to_nodes))

circuit_data = CircuitTensorData(
    metadata=metadata,
    macro_positions=macro_positions,
    macro_sizes=macro_sizes,
    net_to_nodes=net_to_nodes,
    net_weights=net_weights,
)

print(f"Number of nets: {circuit_data.num_nets}")
# Number of nets: 3
```

### Accessing Properties

```python
# Access metadata
print(f"Design: {circuit_data.design_name}")
print(f"Macros: {circuit_data.num_macros}")
print(f"Canvas: {circuit_data.canvas_width} x {circuit_data.canvas_height} μm")

# Access tensors
print(f"Macro sizes shape: {circuit_data.macro_sizes.shape}")
print(f"First macro size: {circuit_data.macro_sizes[0]}")

# Iterate over nets
for i, nodes in enumerate(circuit_data.net_to_nodes):
    weight = circuit_data.net_weights[i]
    print(f"Net {i}: connects macros {nodes.tolist()}, weight={weight}")
```

---

## Loading and Saving

### Load from File

```python
from marco_place.data.dataset import MacroPlacementDataset

# Load dataset
dataset = MacroPlacementDataset('benchmarks/processed', split='public')

# Get a specific benchmark
circuit_data = dataset.get_by_name('ariane133')

# Or by index
circuit_data = dataset[0]
```

### Direct Load

```python
import torch

# Load directly
circuit_data = torch.load('benchmarks/processed/public/ariane133.pt')
```

### Save to File

```python
import torch

# Save circuit data
torch.save(circuit_data, 'my_benchmark.pt')
```

---

## Advanced Topics

### Computing Macro Bounding Boxes

```python
def get_macro_bbox(placement, macro_sizes, macro_idx):
    """Get bounding box for a macro."""
    x, y = placement[macro_idx]
    w, h = macro_sizes[macro_idx]

    return {
        'left': x - w/2,
        'right': x + w/2,
        'bottom': y - h/2,
        'top': y + h/2,
        'width': w,
        'height': h,
    }
```

### Building Edge Lists from Hypergraph

```python
def hypergraph_to_edges(net_to_nodes):
    """Convert hypergraph to edge list (for GNN models)."""
    edges = []

    for net in net_to_nodes:
        nodes = net.tolist()
        # Create clique: connect all pairs
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                edges.append([nodes[i], nodes[j]])
                edges.append([nodes[j], nodes[i]])  # Bidirectional

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index
```

### Checking Placement Validity

```python
def is_valid_placement(placement, macro_sizes, canvas_width, canvas_height):
    """Check if placement satisfies boundary constraints."""
    for i in range(len(placement)):
        x, y = placement[i]
        w, h = macro_sizes[i]

        # Check boundaries
        if x - w/2 < 0 or x + w/2 > canvas_width:
            return False
        if y - h/2 < 0 or y + h/2 > canvas_height:
            return False

    return True
```

### Computing Total Macro Area

```python
def compute_total_area(macro_sizes):
    """Compute total area of all macros."""
    areas = macro_sizes[:, 0] * macro_sizes[:, 1]
    return areas.sum().item()

# Usage
total_area = compute_total_area(circuit_data.macro_sizes)
canvas_area = circuit_data.canvas_width * circuit_data.canvas_height
utilization = total_area / canvas_area

print(f"Utilization: {utilization:.2%}")
```

---

## Summary

The `CircuitTensorData` format provides:

✅ **Efficient**: PyTorch tensors for fast computation
✅ **Flexible**: Optional fields for advanced use cases
✅ **ML-friendly**: Easy integration with GNNs, RL, etc.
✅ **Complete**: All information needed for macro placement

For questions or clarifications, see the [Getting Started Guide](getting_started.md) or open an issue on GitHub.
