# Partcl Macro Placement Challenge

**Win $20,000 by developing better macro placement algorithms!**

This competition challenges you to beat classical macro placement baselines on modern chip designs including RISC-V processors (Ariane), AI accelerators (NVDLA), and memory architectures (MemPool). Your goal is to minimize placement cost while maintaining fast runtime and zero overlaps.

## 🎯 Prize Details

- **Prize Amount**: $20,000 (winner-takes-all)
- **Winner**: The SINGLE team with the highest aggregate score across all benchmarks
- **Eligibility**: Prize is awarded ONLY if your submission beats the baseline initial placements on aggregate score
- **If no submission beats the baselines**: No prize will be awarded
- **Second place and beyond**: No monetary prize (but recognized on leaderboard)

## 📊 The Challenge

### What is Macro Placement?

Macro placement is a critical step in chip design where large memory blocks (macros) need to be positioned on the chip canvas. For example, the **ariane133** benchmark (a RISC-V processor core) has:
- **133 hard macros** including SRAMs, register files, and custom blocks
- **22,584 nets** connecting macros to each other and to standard cell clusters
- **A 1.43 × 1.43 mm canvas** (real-world chip scale)
- **Zero overlaps required** (enforced by fabrication constraints)

You must find positions that optimize the objective function while maintaining legality.

### Objective Function

```
proxy_cost = 1.0 × wirelength + 0.5 × density + 0.5 × congestion
```

**Lower is better!** Each component is normalized:

1. **Wirelength** (weight = 1.0): Half-perimeter wirelength (HPWL) of all nets, normalized by total wire capacity
2. **Density** (weight = 0.5): Average of the top 10% densest grid cells
3. **Congestion** (weight = 0.5): Average of the top 5% most congested routing segments

These metrics are computed using the TILOS MacroPlacement evaluator (the same evaluator used in academic research).

### Scoring System

Your final score combines three factors:

```python
# Per-benchmark score
if overlap_count > 0:
    score = -1000  # Disqualified for overlaps
else:
    quality_score = (baseline_cost - your_cost) / baseline_cost  # Higher is better
    runtime_penalty = max(0, (your_runtime - 300) / 300)  # Penalty for runtime > 5min
    score = quality_score - 0.1 × runtime_penalty

# Final score = geometric mean across all benchmarks
final_score = geometric_mean([scores for all IBM benchmarks])
```

**To win**: `final_score` > 0 (meaning you beat the baseline on average)

### Hard Constraints (Automatic Disqualification if Violated)

Your placement MUST satisfy:
- ✅ **Zero macro overlaps** (any overlap = automatic -1000 score for that benchmark)
- ✅ **All macros within canvas bounds**
- ✅ **Fixed macros stay fixed** (if any)
- ✅ **No NaN/Inf values**
- ✅ **Runtime < 1 hour per benchmark** (hard timeout)

**Note on Overlaps**: While the density cost implicitly penalizes overlaps (grid cells can exceed 100% density), **any overlap is an automatic disqualification** for that benchmark. Zero tolerance.

### Baselines to Beat

Your algorithm must outperform the **hand-crafted initial placements** provided with each benchmark:

- **Initial Placements**: Expert-designed placements that serve as starting points
  - Created by chip designers with domain knowledge
  - Already optimized for basic constraints
  - Zero overlaps guaranteed
  - Proxy costs: 0.71-0.96 (see table below)

**Note**: We will also run classical baselines (Simulated Annealing, RePlAce) for comparison, but you only need to beat the initial placements to be eligible for the prize.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/partcleda/partcl-macro-place-challenge.git
cd partcl-macro-place-challenge

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
  Score: -1000 (DISQUALIFIED: 211 overlaps)
```

The random placer has overlaps and is automatically disqualified - your job is to do better!

## 🎓 How It Works

### 1. Benchmark Representation

Benchmarks are represented as **PyTorch tensors** for easy integration with ML approaches:

```python
from benchmark import Benchmark

# Load a pre-processed benchmark
benchmark = Benchmark.load('benchmarks/processed/public/ariane133_ng45.pt')

print(f"Benchmark: {benchmark.name}")
print(f"Macros: {benchmark.num_macros}")
print(f"Nets: {benchmark.num_nets}")
print(f"Canvas: {benchmark.canvas_width} × {benchmark.canvas_height} mm")

# Access data
print(f"Macro positions: {benchmark.macro_positions.shape}")  # [133, 2]
print(f"Macro sizes: {benchmark.macro_sizes.shape}")          # [133, 2]
print(f"Fixed macros: {benchmark.macro_fixed.shape}")         # [133] (bool)
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
                - num_macros: Number of macros (133 for ariane133)
                - macro_sizes: [num_macros, 2] (width, height) in mm
                - macro_fixed: [num_macros] bool (True if fixed)
                - canvas_width, canvas_height: Canvas dimensions in mm
                - num_nets: Number of nets (22584 for ariane133)

        Returns:
            placement: [num_macros, 2] tensor of (x, y) center positions
        """
        placement = torch.zeros(benchmark.num_macros, 2)

        # Your algorithm here!
        # - Use GNNs, RL, SA, optimization, or any approach
        # - MUST have zero overlaps (automatic disqualification otherwise)
        # - MUST be within canvas boundaries
        # - Minimize proxy cost while keeping runtime reasonable

        # Remember to respect fixed macros!
        fixed_mask = benchmark.macro_fixed
        placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

        return placement
```

### 3. Evaluation

```python
import time
from benchmark import Benchmark
from loader import load_benchmark_from_dir
from objective import compute_proxy_cost
from utils import validate_placement

# Load benchmark (need both tensor and PlacementCost for evaluation)
flows_dir = "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping"
benchmark, plc = load_benchmark_from_dir(flows_dir)

# Run your placer with timing
start_time = time.time()
placer = MyPlacer()
placement = placer.place(benchmark)
runtime = time.time() - start_time

# Validate placement legality
is_valid, violations = validate_placement(placement, benchmark)
if not is_valid:
    print(f"Invalid placement: {violations}")

# Compute cost and overlap metrics
costs = compute_proxy_cost(placement, benchmark, plc)
print(f"Proxy cost: {costs['proxy_cost']:.6f}")
print(f"Overlaps: {costs['overlap_count']} pairs")
print(f"Runtime: {runtime:.2f}s")

# Compute score
if costs['overlap_count'] > 0:
    score = -1000  # Disqualified
else:
    baseline_cost = 1.0  # Replace with actual baseline
    quality = (baseline_cost - costs['proxy_cost']) / baseline_cost
    runtime_penalty = max(0, (runtime - 300) / 300)
    score = quality - 0.1 * runtime_penalty

print(f"Score: {score}")
```

## 📋 Competition Rules

### Allowed

1. **Any algorithmic approach**: SA, RL, GNN, analytical methods, hybrid approaches, learning-based, etc.
2. **Any framework**: PyTorch, TensorFlow, JAX, or pure Python/C++
3. **Any optimization technique**: Gradient descent, evolutionary algorithms, local search, etc.
4. **Training on public benchmarks**: You can learn from the IBM benchmark data

### Not Allowed

1. ❌ Modifying the evaluation functions (must use TILOS MacroPlacement evaluator as-is)
2. ❌ Hardcoding solutions for specific benchmarks (must be general algorithm)
3. ❌ Using external/proprietary placement tools (must be open-source submission)
4. ❌ Exceeding runtime limits (1 hour per benchmark hard timeout)

### Runtime Constraints

- **Soft limit**: 5 minutes per benchmark (no penalty)
- **Penalty zone**: 5-60 minutes (linear penalty up to -0.1 quality score)
- **Hard timeout**: 1 hour (automatic disqualification)

Runtime measured on standard hardware:
- CPU: AMD EPYC 7763 (64 cores) or equivalent
- RAM: 256GB
- No GPU acceleration in evaluation (but you can use GPU during development)

### Overlap Tolerance: ZERO

Unlike density cost which is continuous, overlaps result in automatic disqualification:
- 0 overlaps: ✅ Eligible for scoring
- 1+ overlaps: ❌ Score = -1000 (disqualified for that benchmark)

This matches the constraints enforced by the SA baseline.

## 🎯 Modern Benchmark Suite

We evaluate on real chip designs from the TILOS MacroPlacement repository:

| Benchmark | Design Type | Macros | Nets | Canvas (mm) | Initial Baseline ⭐ |
|-----------|-------------|--------|------|-------------|---------------------|
| **ariane133** | RISC-V Processor | 133 | 22,584 | 1.43×1.43 | **0.7109** |
| **ariane136** | RISC-V Processor | 136 | 23,067 | 1.45×1.45 | **0.7097** |
| **nvdla** | AI Accelerator | 128 | 40,606 | 2.13×2.13 | **0.7569** |
| **mempool_tile** | Memory Architecture | 20 | 32,944 | 0.89×0.89 | **0.9610** |

### Why These Benchmarks?

**Real Chip Designs**:
- **Ariane**: RISC-V processor cores used in actual tape-outs
- **NVDLA**: NVIDIA's open-source deep learning accelerator
- **MemPool**: Research memory architecture from ETH Zurich

**Modern Scale**:
- 20-136 macros (compact, high-quality placements)
- 22K-41K nets (dense connectivity)
- Real-world canvas sizes (0.9-2.1 mm)
- NanGate45 technology (45nm process)

**High-Quality Baselines**:
- Initial placements already optimized by experts
- Zero overlaps guaranteed
- Lower proxy costs (0.71-0.96) than IBM benchmarks (1.0-3.7)
- Much harder to improve upon

Each benchmark includes:
- Hard macros (you place these)
- Standard cell clusters (pre-placed, fixed during evaluation)
- Nets connecting all components
- Initial placement (hand-crafted by experts, serves as baseline)

## 💡 Why This Is Hard

Despite "only" 20-136 macros, this problem is extremely challenging:

1. **Already-optimized baselines**: Initial placements are hand-crafted by experts with domain knowledge
2. **Conflicting objectives**: Wirelength wants clustering, density wants spreading, congestion wants routing space
3. **Non-convex landscape**: Millions of local minima, discontinuities, plateaus
4. **Long-range dependencies**: Moving one macro affects costs globally through tens of thousands of nets
5. **Hard constraints**: No overlaps between heterogeneous macro sizes
6. **Dense connectivity**: 22K-41K nets create complex optimization landscape
7. **Runtime matters**: Must be fast enough to be practical (< 5 minutes ideal)

The initial placements are already quite good (proxy costs 0.71-0.96), making further improvement challenging!

## 📖 Documentation

- **Setup Guide**: [`SETUP.md`](SETUP.md) - Infrastructure details, testing, cost computation
- **API Reference**: [`SETUP.md`](SETUP.md) - Benchmark format, loader, objective functions
- **Example Submissions**: [`submissions/examples/`](submissions/examples/) - Random placer example

## 📚 References

- **TILOS MacroPlacement**: [GitHub Repository](https://github.com/TILOS-AI-Institute/MacroPlacement)
  - Source of evaluation infrastructure
  - ICCAD04 benchmarks
  - SA and RePlAce baseline implementations

- **ICCAD04 Benchmarks**: Classic macro placement benchmark suite used in academic research

## 🤔 FAQ

**Q: Why modern benchmarks instead of IBM (ICCAD04)?**
A: Modern benchmarks are real chip designs (RISC-V processors, AI accelerators) with better-quality initial placements, making them more representative of current industry challenges.

**Q: Why is runtime part of the score?**
A: Real chip design requires practical algorithms. A solution that takes hours is less useful than one that takes minutes, even if slightly lower quality.

**Q: Can I use GPU?**
A: Yes during development, but evaluation runs on CPU-only hardware for fairness.

**Q: Are these benchmarks too small (only 20-136 macros)?**
A: The challenge comes from the high-quality initial placements (already optimized by experts) and dense connectivity (22K-41K nets), not just macro count.

**Q: Are there hidden test cases?**
A: No. All 4 benchmarks are public. The aggregate score across all 4 determines the winner.

**Q: What counts as "beating" the baseline?**
A: Your geometric mean score across all benchmarks must be positive (meaning on average you beat the initial placements).

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/partcleda/partcl-macro-place-challenge/issues)
- **Email**: contact@partcl.com

## 📄 License

This project is licensed under the PolyForm Noncommercial License 1.0.0 - see [LICENSE.md](LICENSE.md) for details.

---

**Ready to win $20,000?**

Beat expert-designed initial placements on real chip designs with zero overlaps and reasonable runtime!

Good luck! 🚀
