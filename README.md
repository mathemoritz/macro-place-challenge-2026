# Partcl Macro Placement Challenge

**Win $20,000 by beating state-of-the-art macro placement algorithms!**

This competition challenges you to develop better algorithms for macro placement in VLSI chip design. Your goal is to beat the benchmarks from the influential paper ["Assessment of Reinforcement Learning for Macro Placement"](https://arxiv.org/abs/2302.11014) (ISPD 2023).

## 🎯 Prize Details

- **Prize Amount**: $20,000 (winner-takes-all)
- **Winner**: The SINGLE team with the highest aggregate score
- **Eligibility**: Prize is awarded ONLY if your submission beats the **Circuit Training baseline** (best performing method in Kahng's paper) on aggregate across all benchmarks
- **If no submission beats the baseline**: No prize will be awarded
- **Second place and beyond**: No monetary prize (but recognized on leaderboard)

## 📊 The Challenge

### What is Macro Placement?

Macro placement is a critical step in chip design where large memory blocks (macros) need to be positioned on the chip canvas. Given:
- **246 hard macros** of varying sizes (ranging from 0.8 to 27 μm²)
- **7,269 nets** connecting macros to each other and to 894 pre-placed standard cell clusters
- **A 22.9 × 23.0 μm canvas** with 42.8% area utilization

You must find positions that optimize:

### Objective Function

```
proxy_cost = 1.0 × wirelength + 0.5 × density + 0.5 × congestion
```

**Lower is better!** Each component is normalized:

1. **Wirelength** (weight = 1.0): Half-perimeter wirelength (HPWL) of all nets, normalized by total wire capacity
2. **Density** (weight = 0.5): Average of the top 10% densest grid cells
3. **Congestion** (weight = 0.5): Average of the top 5% most congested routing segments

### Hard Constraints

Your placement MUST satisfy:
- ✅ **All macros within canvas bounds** (0 ≤ x ≤ 22.9, 0 ≤ y ≤ 23.0)
- ✅ **Zero macro overlaps** (matching SA baseline behavior)
- ✅ **Fixed macros stay fixed** (if any)

**Note on Overlaps**: While the density cost implicitly penalizes overlaps (grid cells can exceed 100% density), valid competition submissions should have **zero overlaps**. The SA baseline enforces this as a hard constraint.

### Baselines to Beat

Your algorithm must outperform:
1. **Simulated Annealing** (classical optimization)
2. **RePlAce** (analytical placement)
3. **Circuit Training** (Google's RL method) ← **You must beat this!**

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/partcl/partcl-marco-place-challenge.git
cd partcl-marco-place-challenge

# Initialize TILOS MacroPlacement submodule (required for evaluation)
git submodule update --init external/MacroPlacement

# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt

# Test the infrastructure
pytest
```

### Run Your First Example

```bash
# Run the simple random placer example
python submissions/examples/simple_random_placer.py
```

You should see output like:
```
[4/4] Computing proxy cost and overlap metrics...
  ✓ Costs computed:
    - Wirelength:  0.128768
    - Density:     1.276113
    - Congestion:  2.248285
    - Proxy Cost:  1.890967 ⭐

  ✓ Overlap analysis:
    - Overlapping pairs:       211
    - Macros with overlaps:    198 (80.5%)
    - Total overlap area:      99.632 μm²

Comparison with initial placement:
  Initial proxy cost:   1.038498 (overlaps: 9)
  Random proxy cost:    1.890967 (overlaps: 211)
  Improvement: -82.09% (worse)
```

The random placer performs poorly - your job is to do better!

## 🎓 How It Works

### 1. Benchmark Representation

Benchmarks are represented as **PyTorch tensors** for easy integration with ML approaches:

```python
from loader import load_benchmark_from_dir

# Load a benchmark
benchmark, plc = load_benchmark_from_dir('external/MacroPlacement/Testcases/ICCAD04/ibm01')

print(f"Benchmark: {benchmark.name}")
print(f"Macros: {benchmark.num_macros}")
print(f"Nets: {benchmark.num_nets}")
print(f"Canvas: {benchmark.canvas_width} × {benchmark.canvas_height} μm")

# Access data
print(f"Macro positions: {benchmark.macro_positions.shape}")  # [246, 2]
print(f"Macro sizes: {benchmark.macro_sizes.shape}")          # [246, 2]
print(f"Fixed macros: {benchmark.macro_fixed.shape}")         # [246] (bool)
```

### 2. Implementing Your Placer

Create a class with a `.place()` method:

```python
import torch
from benchmark import Benchmark

class MyPlacer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Generate macro placement.

        Args:
            benchmark: Benchmark object with:
                - num_macros: Number of macros (246 for ibm01)
                - macro_sizes: [num_macros, 2] (width, height) in μm
                - macro_fixed: [num_macros] bool (True if fixed)
                - canvas_width, canvas_height: Canvas dimensions
                - num_nets: Number of nets (7269 for ibm01)

        Returns:
            placement: [num_macros, 2] tensor of (x, y) center positions
        """
        placement = torch.zeros(benchmark.num_macros, 2)

        # Your algorithm here!
        # - Use GNNs, RL, SA, optimization, or any approach
        # - Ensure no overlaps and within canvas boundaries
        # - Minimize proxy cost

        # Remember to respect fixed macros!
        fixed_mask = benchmark.macro_fixed
        placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

        return placement
```

### 3. Evaluation

```python
from loader import load_benchmark_from_dir
from objective import compute_proxy_cost
from utils import validate_placement

# Load benchmark
benchmark, plc = load_benchmark_from_dir('external/MacroPlacement/Testcases/ICCAD04/ibm01')

# Run your placer
placer = MyPlacer()
placement = placer.place(benchmark)

# Validate placement legality
is_valid, violations = validate_placement(placement, benchmark)
if not is_valid:
    print(f"Invalid placement: {violations}")

# Compute cost and overlap metrics
costs = compute_proxy_cost(placement, benchmark, plc)
print(f"Proxy cost: {costs['proxy_cost']:.6f}")
print(f"Overlaps: {costs['overlap_count']} pairs")
print(f"Macros with overlaps: {costs['num_macros_with_overlaps']}")
```

### 4. Scoring

```python
# Per-benchmark improvement over Circuit Training baseline
improvement = (CT_baseline_cost - your_cost) / CT_baseline_cost * 100

# Final score = geometric mean across all benchmarks
final_score = geometric_mean(improvements)

# To win the prize: final_score must be > 0
```

## 📋 Competition Rules

### Allowed

1. **Any approach**: RL, GNN, SA, analytical methods, hybrid approaches, etc.
2. **Any framework**: PyTorch, TensorFlow, JAX, or pure Python
3. **Any optimization technique**: Gradient descent, evolutionary algorithms, etc.
4. **Learn from data**: You can train on the public benchmarks

### Not Allowed

1. ❌ Modifying the evaluation functions (use PlacementCost as-is)
2. ❌ Hardcoding solutions for specific benchmarks
3. ❌ Using external/proprietary placement tools (must be open-source submission)

### Constraints

Your placement MUST satisfy:
- ✅ All macros within canvas bounds
- ✅ Zero macro overlaps (matching SA baseline)
- ✅ Respect fixed macros (keep at original positions)
- ✅ No NaN/Inf values

### Runtime

- Maximum 1 hour per benchmark
- Evaluated on standard hardware (details TBD)

### Evaluation

- **Public benchmarks**: Available now for development (ibm01-18)
- **Hidden test cases**: 3-5 additional benchmarks for final evaluation
- **Prevents overfitting**: Hidden tests ensure general solutions

## 🎯 Public Benchmarks

We provide real benchmarks from the ICCAD04 suite (via TILOS MacroPlacement repository):

| Benchmark | Macros | Nets | Canvas (μm) | Area Util. |
|-----------|--------|------|-------------|------------|
| **ibm01** | 246 | 7,269 | 22.9×23.0 | 42.8% |
| **ibm02** | 254 | 7,538 | 23.2×23.5 | 43.1% |
| **ibm03** | 269 | 8,045 | 24.1×24.3 | 44.2% |
| ... | ... | ... | ... | ... |

Each benchmark includes:
- Hard macros (you place these)
- Soft macros (pre-placed standard cell clusters)
- Nets connecting all components
- Initial placement (human-designed, quite good!)

## 💡 Why This Is Hard

Despite "only" 246 macros, this problem is extremely challenging:

1. **Massive search space**: ~10^800 possible placements (even with constraints)
2. **Conflicting objectives**: Wirelength wants clustering, density wants spreading, congestion wants routing space
3. **Non-convex landscape**: Millions of local minima, discontinuities, plateaus
4. **Long-range dependencies**: Moving one macro affects costs globally through 7,269 nets
5. **Hard constraints**: No overlaps between heterogeneous sizes (33× size variation)
6. **Tight packing**: 42.8% area utilization leaves little slack

State-of-the-art methods (SA, RePlAce, Circuit Training) still have significant room for improvement!

## 📖 Documentation

- **Getting Started**: [`docs/getting_started.md`](docs/getting_started.md)
- **API Reference**: [`SETUP.md`](SETUP.md)
- **Example Submissions**: [`submissions/examples/`](submissions/examples/)

## 📚 References

- **Kahng et al. (2023)**: ["Assessment of Reinforcement Learning for Macro Placement"](https://arxiv.org/abs/2302.11014)
- **TILOS MacroPlacement**: [GitHub Repository](https://github.com/TILOS-AI-Institute/MacroPlacement)
- **Circuit Training**: [Google Research](https://github.com/google-research/circuit_training)

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/partcl/partcl-marco-place-challenge/issues)
- **Email**: contact@partcl.com

## 📄 License

This project is licensed under the PolyForm Noncommercial License 1.0.0 - see [LICENSE.md](LICENSE.md) for details.

---

**Ready to win $20,000?**

Good luck! 🚀
