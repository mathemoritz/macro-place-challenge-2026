# Macro Placement Benchmarks

This directory contains benchmarks for the macro placement competition.

## Directory Structure

```
benchmarks/
├── processed/          # Converted tensor format benchmarks
│   ├── public/        # Public benchmarks (available to participants)
│   └── hidden/        # Hidden test cases (for final evaluation)
├── raw/               # Original LEF/DEF/protobuf files (gitignored)
├── metadata/          # Benchmark statistics and baseline scores
└── scripts/           # Conversion and download scripts
```

## Available Benchmarks

### Public Benchmarks (Phase 1)

These synthetic benchmarks are currently available for testing:

| Benchmark | Macros | Nets | Canvas (um) | Description |
|-----------|--------|------|-------------|-------------|
| `toy_small` | 10 | 15 | 1000 x 1000 | Small test case for quick prototyping |
| `toy_medium` | 50 | 375 | 1000 x 1000 | Medium-sized benchmark |
| `toy_large` | 200 | 6000 | 1000 x 1000 | Larger benchmark for scalability testing |
| `ariane133` | 133 | 1995 | 5000 x 5000 | Synthetic RISC-V processor-like design |

**Note**: These are synthetic benchmarks for Phase 1 testing. Real benchmarks from the TILOS repository will be added in Phase 2.

### Real Benchmarks (Coming in Phase 2)

The following benchmarks from [TILOS-AI-Institute/MacroPlacement](https://github.com/TILOS-AI-Institute/MacroPlacement) will be converted:

| Benchmark | Design Type | Macros | Description |
|-----------|-------------|--------|-------------|
| **Ariane133** | RISC-V Processor | 133 | 64-bit RISC-V processor with 133 SRAM macros (256x16-bit) |
| **Ariane136** | RISC-V Processor | 136 | Variant of Ariane133 with 136 SRAMs |
| **MemPool Tile** | Memory System | 18K+ cells | Memory-centric design with 18,278 flops, mixed SRAM sizes |
| **MemPool Group** | Memory System | 360K+ cells | Large memory system with 360,724 flops |
| **NVDLA** | ML Accelerator | 128 | NVIDIA Deep Learning Accelerator with 128 SRAMs (256x64-bit) |
| **BlackParrot** | Multi-core CPU | Variable | Multi-core processor design with 214,441 flops |

### Hidden Test Cases (Phase 7)

Hidden benchmarks will be created to prevent overfitting to public benchmarks. These will only be revealed during final evaluation.

Characteristics of hidden benchmarks:
- Mix of sizes (small, medium, large)
- Different macro/stdcell ratios
- Various connectivity patterns
- Sourced from ISPD/DAC contests or custom RTL designs

## Benchmark Format

All benchmarks are stored as PyTorch `.pt` files containing `CircuitTensorData` objects.

### Loading a Benchmark

```python
from marco_place.data.tensor_schema import CircuitTensorData

# Method 1: Direct load
circuit_data = CircuitTensorData.load('benchmarks/processed/public/ariane133.pt')

# Method 2: Using Dataset loader
from marco_place.data.dataset import load_benchmark
circuit_data = load_benchmark('ariane133')

# Method 3: Using Dataset class
from marco_place.data.dataset import MacroPlacementDataset
dataset = MacroPlacementDataset('benchmarks/processed', split='public')
circuit_data = dataset[0]  # or dataset.get_by_name('ariane133')
```

### Benchmark Schema

Each benchmark contains:

**Metadata**:
- `design_name`: Name of the design
- `num_macros`: Number of macros
- `canvas_width`, `canvas_height`: Canvas dimensions (microns)
- `target_density`: Target placement density (typically 0.6)

**Macro Data**:
- `macro_positions`: [num_macros, 2] - Initial (x, y) positions
- `macro_sizes`: [num_macros, 2] - (width, height) of each macro
- `macro_is_fixed`: [num_macros] - Boolean mask for fixed macros

**Netlist (Hypergraph)**:
- `net_to_nodes`: List of tensors containing node indices for each net
- `net_weights`: [num_nets] - Weight for wirelength calculation

**Optional Fields** (Phase 2+):
- `node_features`, `node_types`: Extended node information
- `stdcell_positions`, `stdcell_sizes`: Standard cell data
- `port_positions`, `port_sides`: I/O port information
- `placement_blockages`: Forbidden placement regions
- `grid_rows`, `grid_cols`: Placement grid information

## Generating Benchmarks

### Toy Benchmarks

Generate synthetic benchmarks for testing:

```bash
# Small benchmark (10 macros)
python benchmarks/scripts/convert_simple.py --benchmark toy_small

# Medium benchmark (50 macros)
python benchmarks/scripts/convert_simple.py --benchmark toy_medium

# Large benchmark (200 macros)
python benchmarks/scripts/convert_simple.py --benchmark toy_large

# Ariane133-like benchmark (133 macros)
python benchmarks/scripts/convert_simple.py --benchmark ariane133
```

### Real Benchmarks (Phase 2)

Scripts to download and convert TILOS benchmarks will be added in Phase 2:

```bash
# Download original benchmarks from TILOS repo
bash benchmarks/scripts/download_from_tilos.sh

# Convert all benchmarks to tensor format
python benchmarks/scripts/convert_all.py
```

## Evaluation Metrics

Benchmarks will be evaluated on:

1. **Proxy Cost** (fast, for rapid iteration):
   - Wirelength (HPWL)
   - Density uniformity
   - Congestion estimation
   - Formula: `0.5 * wirelength + 0.25 * density + 0.25 * congestion`

2. **Post-Routing QoR** (slow, for final evaluation):
   - Routed wirelength
   - Total power
   - Timing (TNS, WNS)
   - Requires OpenROAD or Innovus

## Baseline Scores

Baseline scores for each benchmark will be recorded in `benchmarks/metadata/baseline_scores.json` as baselines are implemented.

Expected baselines:
- **Random**: Grid-based or random placement
- **Simulated Annealing**: Classical optimization
- **Analytical**: Force-directed or quadratic optimization
- **Circuit Training**: GNN + RL (baseline to beat for prize)

## References

- TILOS MacroPlacement: https://github.com/TILOS-AI-Institute/MacroPlacement
- Kahng et al. (2023): https://arxiv.org/abs/2302.11014
- Circuit Training: https://github.com/google-research/circuit_training

## Contact

For questions about benchmarks:
- Open an issue: https://github.com/partcl/partcl-marco-place-challenge/issues
- Check documentation: `docs/`
