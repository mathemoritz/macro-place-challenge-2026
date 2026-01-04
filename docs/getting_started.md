# Getting Started Guide

Welcome to the Macro Placement Challenge! This guide will walk you through everything you need to know to create your first submission.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Understanding the Problem](#understanding-the-problem)
4. [Your First Evaluation](#your-first-evaluation)
5. [Creating Your Own Placer](#creating-your-own-placer)
6. [Testing and Debugging](#testing-and-debugging)
7. [Submission Guidelines](#submission-guidelines)
8. [Tips and Best Practices](#tips-and-best-practices)

---

## Prerequisites

### Required Knowledge

- **Python**: Intermediate level (classes, functions, file I/O)
- **PyTorch**: Basic familiarity with tensors and operations
- **Algorithms**: Understanding of optimization or machine learning concepts (helpful but not required)

### System Requirements

- **OS**: Linux, macOS, or Windows (Linux recommended)
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB+ recommended for larger benchmarks
- **Storage**: 2GB for repository and benchmarks

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/partcl/partcl-marco-place-challenge.git
cd partcl-marco-place-challenge
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n macro-place python=3.10
conda activate macro-place
```

### Step 3: Install Dependencies

```bash
# Install the package in development mode
pip install -e .

# Verify installation
python -c "import marco_place; print('Installation successful!')"
```

### Step 4: Verify Benchmarks

Check that benchmarks are available:

```bash
ls benchmarks/processed/public/
# Should show: ariane133.pt, nvdla_asap7.pt, blackparrot_asap7.pt, mempool_asap7.pt, etc.
```

---

## Understanding the Problem

### What is Macro Placement?

Macro placement is a critical step in VLSI chip design where large memory blocks (macros) must be positioned on a 2D canvas to minimize:

1. **Wirelength**: Total wire length connecting macros
2. **Density**: Even distribution to avoid hotspots
3. **Congestion**: Prevent routing bottlenecks

### Input: Circuit Data

Your placer receives a `CircuitTensorData` object containing:

```python
class CircuitTensorData:
    metadata: Dict  # design_name, num_macros, canvas_width, canvas_height
    macro_sizes: Tensor  # [num_macros, 2] - (width, height) in microns
    net_to_nodes: List[Tensor]  # Connectivity: list of macro indices per net
    net_weights: Tensor  # [num_nets] - importance of each net
```

### Output: Placement

Your placer must return:

```python
placement: torch.Tensor  # [num_macros, 2] - (x, y) coordinates of macro centers
```

### Constraints

1. **No Overlaps**: Macros cannot overlap each other
2. **Boundary Constraints**: All macros must be fully within the canvas
3. **Legal Positions**: For a macro with center (x, y) and size (w, h):
   - Left edge: `x - w/2 >= 0`
   - Right edge: `x + w/2 <= canvas_width`
   - Bottom edge: `y - h/2 >= 0`
   - Top edge: `y + h/2 <= canvas_height`

### Scoring

Your submission is scored by computing a **proxy cost** that combines:

```python
proxy_cost = 0.5 * wirelength_cost + 0.25 * density_cost + 0.25 * congestion_cost
```

The **aggregate score** is the geometric mean of improvements over the Circuit Training baseline across all benchmarks.

**To win the $20K prize**: Your aggregate score must be **positive** (beat Circuit Training on average).

---

## Your First Evaluation

### Step 1: Examine the Example Submission

Look at the provided example:

```bash
cat submissions/examples/random_placer/placer.py
```

This shows a simple row-packing placer that:
1. Sorts macros by height
2. Places them left-to-right in rows
3. Moves to the next row when the current row is full

### Step 2: Run Your First Evaluation

Evaluate the example submission on a single benchmark:

```bash
python scripts/evaluate_submission.py \
    --submission submissions/examples/random_placer/placer.py \
    --name "GridPlacer" \
    --benchmarks ariane133
```

You should see output like:

```
Loading submission from: submissions/examples/random_placer/placer.py
✓ Loaded placer class: RandomPlacer

======================================================================
Evaluating Submission: GridPlacer
======================================================================

--- Evaluating ariane133 ---
Design: CircuitTensorData(design='ariane133', num_macros=133, num_nets=757, canvas=1302.8x1302.8um)
Running placer (timeout: 3600s)...
✓ Completed in 0.00s
Validating placement...
✓ Placement is valid
Computing metrics...
  Proxy cost: 0.200452
    - Wirelength: 0.219019
    - Density:    0.361067
    - Congestion: 0.002702

======================================================================
EVALUATION COMPLETE
======================================================================
Submission: GridPlacer
Aggregate Score: +0.00%
```

### Step 3: Evaluate on All Public Benchmarks

```bash
python scripts/evaluate_submission.py \
    --submission submissions/examples/random_placer/placer.py \
    --name "GridPlacer" \
    --split public
```

This will evaluate on all 7 public benchmarks (4 real + 3 toy).

### Step 4: View the Leaderboard

```bash
python scripts/generate_leaderboard.py
```

You'll see a ranked leaderboard with all submissions.

---

## Creating Your Own Placer

### Step 1: Copy the Template

```bash
cp -r submissions/template submissions/my_placer
cd submissions/my_placer
```

### Step 2: Implement the `place()` Method

Edit `placer.py`:

```python
import torch
from marco_place.data.tensor_schema import CircuitTensorData
from abc import ABC, abstractmethod

class BasePlacerInterface(ABC):
    @abstractmethod
    def place(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        pass

class MyPlacer(BasePlacerInterface):
    def __init__(self):
        # Initialize your algorithm here
        pass

    def place(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        """
        Your algorithm goes here!

        Args:
            circuit_data: CircuitTensorData with macro sizes, nets, canvas info

        Returns:
            placement: [num_macros, 2] tensor of (x, y) coordinates
        """
        num_macros = circuit_data.num_macros
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height
        macro_sizes = circuit_data.macro_sizes

        # Example: Random placement (you should implement something better!)
        placement = torch.rand(num_macros, 2)
        placement[:, 0] *= canvas_width
        placement[:, 1] *= canvas_height

        # Ensure within boundaries
        for i in range(num_macros):
            w, h = macro_sizes[i]
            placement[i, 0] = torch.clamp(placement[i, 0], w/2, canvas_width - w/2)
            placement[i, 1] = torch.clamp(placement[i, 1], h/2, canvas_height - h/2)

        return placement
```

### Step 3: Test Locally

```bash
# Test your placer standalone
python submissions/my_placer/placer.py

# Run evaluation on a small benchmark
python scripts/evaluate_submission.py \
    --submission submissions/my_placer/placer.py \
    --name "MyPlacer" \
    --benchmarks toy_small
```

### Step 4: Iterate and Improve

Common approaches:

1. **Optimization-based**:
   - Simulated Annealing
   - Genetic Algorithms
   - Gradient-based optimization

2. **ML-based**:
   - Graph Neural Networks (GNNs)
   - Reinforcement Learning (RL)
   - Transformers

3. **Hybrid**:
   - GNN for initial placement + optimization for refinement
   - RL for macro selection + analytical for positioning

---

## Testing and Debugging

### Quick Validation

Test your placer on the smallest benchmark first:

```bash
python scripts/evaluate_submission.py \
    --submission submissions/my_placer/placer.py \
    --name "MyPlacer" \
    --benchmarks toy_small \
    --timeout 60
```

### Common Issues

#### 1. Placement Outside Boundaries

**Error**: `One or more macros outside canvas boundaries`

**Fix**: Ensure macro centers account for macro sizes:

```python
x = max(width/2, min(x, canvas_width - width/2))
y = max(height/2, min(y, canvas_height - height/2))
```

#### 2. Macro Overlaps

**Error**: `Found N macro overlaps`

**Fix**: Check overlap between all macro pairs:

```python
def check_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    left1, right1 = x1 - w1/2, x1 + w1/2
    bottom1, top1 = y1 - h1/2, y1 + h1/2
    left2, right2 = x2 - w2/2, x2 + w2/2
    bottom2, top2 = y2 - h2/2, y2 + h2/2

    return not (right1 <= left2 or right2 <= left1 or
                top1 <= bottom2 or top2 <= bottom1)
```

#### 3. Timeout

**Error**: `Timeout (3600s)`

**Fix**: Optimize your algorithm or use early stopping:

```python
import time
start = time.time()
max_time = 3000  # Leave buffer

while time.time() - start < max_time:
    # Your optimization loop
    if converged:
        break
```

### Visualization

To visualize placements (coming soon):

```python
from marco_place.utils.visualization import plot_placement

plot_placement(placement, circuit_data, save_path="my_placement.png")
```

---

## Submission Guidelines

### What to Submit

1. **Required**:
   - `placer.py` - Your placer implementation
   - `requirements.txt` - Additional dependencies (if any)
   - `README.md` - Brief description of your approach

2. **Optional**:
   - Pre-trained model weights
   - Supporting code files
   - Documentation

### File Structure

```
submissions/my_placer/
├── placer.py           # Required: contains your placer class
├── requirements.txt    # Required: list dependencies
├── README.md          # Required: describe your approach
├── model.py           # Optional: supporting code
└── weights/           # Optional: pre-trained weights
    └── model.pth
```

### Requirements

1. **Interface**: Must have a class with `.place(circuit_data)` method
2. **Dependencies**: Only PyTorch + standard libraries (numpy, scipy, etc.)
3. **No External APIs**: Cannot use external services
4. **Reproducible**: Must produce consistent results with same random seed
5. **Runtime**: Must complete within 1 hour per benchmark

---

## Tips and Best Practices

### 1. Start Simple

Begin with a simple algorithm (e.g., grid placement) and gradually add sophistication.

### 2. Leverage the Graph Structure

The netlist is a hypergraph. Consider:
- Node embeddings with GNNs
- Community detection for clustering
- Spectral methods for initial placement

### 3. Multi-Stage Approaches

Many successful placers use multiple stages:
1. **Global placement**: Rough positions (allows overlaps)
2. **Legalization**: Remove overlaps
3. **Detailed placement**: Local refinement

### 4. Balance the Cost Components

The proxy cost has 3 components with different weights:
- Wirelength: 50% (most important)
- Density: 25%
- Congestion: 25%

Don't optimize one at the expense of others.

### 5. Study the Benchmarks

Different benchmarks have different characteristics:
- **ariane133**: Moderate size, balanced connectivity
- **nvdla_asap7**: Dense connectivity, small canvas
- **mempool_asap7**: Sparse connectivity, many macros
- **blackparrot_asap7**: Large, complex design

### 6. Use the Toy Benchmarks for Debugging

The toy benchmarks are fast to evaluate. Use them for:
- Debugging your implementation
- Hyperparameter tuning
- Quick iteration

### 7. Monitor Runtime

Keep your algorithm under the 1-hour timeout:
- Profile your code
- Use early stopping
- Consider time-quality tradeoffs

### 8. Read the Kahng Paper

The [assessment paper](https://arxiv.org/abs/2302.11014) provides valuable insights:
- What works and what doesn't
- Baseline performance numbers
- Common pitfalls

---

## Next Steps

1. **Experiment**: Try different algorithms and hyperparameters
2. **Read Documentation**:
   - [Tensor Format Specification](tensor_format.md)
   - [Competition Rules](competition_rules.md)
3. **Join the Community**: Check GitHub Issues for discussions
4. **Submit**: When ready, follow the submission guidelines

---

## Need Help?

- **Documentation**: See [`docs/`](../docs/)
- **Issues**: [GitHub Issues](https://github.com/partcl/partcl-marco-place-challenge/issues)
- **Examples**: Study `submissions/examples/` for working code

Good luck! Remember, the prize is only awarded if you beat the Circuit Training baseline. 🚀
