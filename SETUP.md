# Setup Instructions

## Quick Start with UV

```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt

# Test the infrastructure
source .venv/bin/activate
pytest
```

## What's Working

The PyTorch infrastructure is fully operational:

- ✅ **Loader** (`src/loader.py`): Extracts macro data from PlacementCost into PyTorch tensors
- ✅ **Benchmark** (`src/benchmark.py`): Pure tensor representation (macros as tensors, nets in PlacementCost)
- ✅ **Objective** (`src/objective.py`): Computes all three cost components via PlacementCost
  - **Wirelength**: 0.064080 (uses 7269 nets from PlacementCost)
  - **Density**: 0.811984 (top 10% grid cells)
  - **Congestion**: 1.136852 (top 5% routing segments with smoothing)
  - **Deterministic**: All costs reproducible to 10 decimal places
- ✅ **Utils** (`src/utils.py`): Validation and visualization helpers

### Congestion Fix

Fixed a boundary condition bug in PlacementCost's `__get_grid_cell_location` via monkey-patch. When pins are at canvas boundaries, the original code computed out-of-bounds grid cells. The fix clamps row/col values to valid ranges.

**Note on Nets**: The net connectivity is stored in PlacementCost but not extracted into PyTorch tensors. This is fine for cost computation but may need extraction if participants want to write custom net-based algorithms in pure PyTorch.

**Note on Overlaps**:
- The **density cost** implicitly penalizes overlaps (overlapping macros cause grid cell density > 1.0)
- The **objective function can be computed** for overlapping placements (they just score poorly)
- The **SA baseline enforces zero overlap** as a hard constraint via `IsFeasible()`
- Our validation now checks for overlaps by default (can disable with `check_overlaps=False`)

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test
pytest test/test_infrastructure.py::test_compute_costs -v

# Debug scripts (if issues arise)
python test/debug_congestion.py
python test/check_nets.py
```

## Exporting to DEF Format

You can export placement results to industry-standard DEF (Design Exchange Format):

```python
from loader import load_benchmark_from_dir
from def_writer import write_def

# Load benchmark
flows_dir = 'external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping'
benchmark, plc = load_benchmark_from_dir(flows_dir)

# ... run your placer and update positions in plc ...

# Export to DEF
write_def(plc, 'output.def', design_name='ariane133')
```

The DEF file includes:
- Die area
- Row definitions (for standard cells)
- Component positions (macros and standard cells)
- I/O pins
- Net connectivity

This allows you to:
- Import results into commercial EDA tools (Innovus, ICC2)
- Visualize with OpenROAD
- Run detailed routing and analysis

## Modern Benchmarks

The competition uses modern chip designs from the TILOS MacroPlacement repository:

- **ariane133/136**: RISC-V processor cores
- **nvdla**: NVIDIA Deep Learning Accelerator
- **mempool_tile**: Memory pooling architecture

These benchmarks are already converted to PyTorch tensor format in `benchmarks/processed/public/`.

To regenerate:

```bash
python scripts/convert_modern_benchmarks.py
python scripts/compute_initial_baselines.py
```
