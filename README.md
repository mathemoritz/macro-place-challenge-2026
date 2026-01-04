# Macro Placement Challenge 💰

**Win $20,000 by beating state-of-the-art macro placement algorithms!**

This competition challenges you to develop better algorithms for macro placement in VLSI chip design. Your goal is to beat the benchmarks from Andrew Kahng's influential paper ["Assessment of Reinforcement Learning for Macro Placement"](https://arxiv.org/abs/2302.11014) (ISPD 2023).

## 🎯 Prize Details

- **Prize Amount**: $20,000
- **Eligibility**: Prize is awarded ONLY if your submission beats the **Circuit Training baseline** (best performing method in Kahng's paper) on aggregate across all benchmarks
- **If no submission beats the baseline**: No prize will be awarded

## 📊 The Challenge

Macro placement is a critical step in chip design where large memory blocks (macros) need to be positioned on the chip canvas to optimize:
- **Wirelength**: Total wire length connecting components
- **Density**: Even distribution of components
- **Congestion**: Avoiding routing bottlenecks

Your algorithm should produce better placements than:
1. **Simulated Annealing** (classical optimization)
2. **RePlAce** (analytical placement)
3. **Circuit Training** (Google's RL-based method) ← **This is the baseline you must beat!**

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/partcl/partcl-marco-place-challenge.git
cd partcl-marco-place-challenge

# Install dependencies
pip install -e .
```

### Run Your First Evaluation

```bash
# Evaluate the example grid placer on a single benchmark
python scripts/evaluate_submission.py \
    --submission submissions/examples/random_placer/placer.py \
    --name "MyFirstSubmission" \
    --benchmarks ariane133

# Evaluate on all public benchmarks
python scripts/evaluate_submission.py \
    --submission submissions/examples/random_placer/placer.py \
    --name "MyFirstSubmission" \
    --split public

# Generate leaderboard
python scripts/generate_leaderboard.py
```

You should see output like:
```
======================================================================
Evaluating Submission: MyFirstSubmission
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
Submission: MyFirstSubmission
Aggregate Score: +0.00%

✗ This submission does not yet beat the Circuit Training baseline.
   Keep optimizing to qualify for the prize!
```

## 📁 Repository Structure

```
partcl-marco-place-challenge/
├── benchmarks/          # Benchmark circuits
│   ├── processed/       # Converted tensor format benchmarks
│   └── scripts/         # Conversion scripts
├── src/marco_place/     # Core library
│   ├── data/           # Data structures and loaders
│   ├── metrics/        # HPWL, density, congestion metrics
│   ├── validation/     # Placement legality checking
│   └── evaluation/     # Evaluation harness
├── baselines/          # Baseline implementations
│   └── random_placer.py
├── submissions/        # Submission templates and examples
│   ├── template/       # Required interface
│   └── examples/       # Example submissions
├── scripts/            # Evaluation scripts
│   └── evaluate.py
└── docs/               # Documentation
```

## 🎓 How It Works

### 1. Circuit Representation

Circuits are represented as **PyTorch tensors** for easy integration with ML approaches:

```python
from marco_place.data.tensor_schema import CircuitTensorData

# Load a benchmark
circuit_data = CircuitTensorData.load('benchmarks/processed/public/ariane133.pt')

print(circuit_data)
# CircuitTensorData(design='ariane133', num_macros=133, num_nets=1995, canvas=5000.0x5000.0um)

# Access data
print(f"Macro sizes: {circuit_data.macro_sizes.shape}")  # [133, 2]
print(f"Nets: {len(circuit_data.net_to_nodes)}")         # 1995
```

### 2. Implementing Your Placer

Create a class with a `.place()` method that takes circuit data and returns macro positions:

```python
import torch
from marco_place.data.tensor_schema import CircuitTensorData

class MyPlacer:
    def place(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        """
        Generate macro placement.

        Args:
            circuit_data: Circuit data with macro sizes, nets, canvas info

        Returns:
            placement: [num_macros, 2] tensor of (x, y) coordinates
        """
        # Your algorithm here!
        # - Use GNNs, RL, optimization, or any PyTorch-based approach
        # - Ensure no overlaps and within canvas boundaries

        placement = your_algorithm(circuit_data)
        return placement
```

### 3. Evaluation

The evaluation harness will:
1. Load benchmarks (public + hidden test cases)
2. Run your placer with 1-hour timeout per benchmark
3. Validate placement legality (no overlaps, within boundaries)
4. Compute **proxy cost** = 0.5×wirelength + 0.25×density + 0.25×congestion
5. Rank by aggregate improvement over Circuit Training baseline

### 4. Scoring

```python
# Per-benchmark improvement
improvement = (CT_baseline_cost - your_cost) / CT_baseline_cost * 100

# Final score = geometric mean across all benchmarks
final_score = geometric_mean(improvements)

# To win the prize: final_score must be > 0
```

## 📋 Competition Rules

1. **Submissions**: Any approach is allowed short of changing the evaluation functions.
2. **Constraints**:
   - No macro overlaps
   - All macros within canvas boundaries
   - Respect placement blockages
3. **Runtime**: Maximum 1 hour per benchmark
4. **Evaluation**:
   - Public benchmarks (available now)
   - Hidden test cases (used for final evaluation)
5. **Prize**: Awarded only if you beat Circuit Training baseline on aggregate

See [`docs/competition_rules.md`](docs/competition_rules.md) (coming in Phase 5) for full details.

## 🎯 Benchmarks

### Public Benchmarks (Available Now)

We provide real benchmarks from the TILOS MacroPlacement repository:

| Benchmark | Macros | Nets | Canvas Size | Description |
|-----------|--------|------|-------------|-------------|
| **ariane133** | 133 | 757 | 1302.8×1302.8 μm | RISC-V processor (Ariane) |
| **nvdla_asap7** | 128 | 2308 | 346.2×346.2 μm | Deep learning accelerator |
| **blackparrot_asap7** | 220 | 1892 | 576.5×576.5 μm | Multi-core processor |
| **mempool_asap7** | 324 | 422 | 375.7×375.7 μm | Memory-centric design |

Plus synthetic toy benchmarks for quick testing:
- `toy_small`: 10 macros, 15 nets
- `toy_medium`: 50 macros, 375 nets
- `toy_large`: 200 macros, 6000 nets (note: unrealistic macro sizes)

### Hidden Test Cases (Final Evaluation)

3-5 hidden benchmarks will be used for final prize evaluation to prevent overfitting.

## 📖 Documentation

- **Getting Started**: [`docs/getting_started.md`](docs/) (coming in Phase 5)
- **Tensor Format**: [`docs/tensor_format.md`](docs/) (coming in Phase 5)
- **Competition Rules**: [`docs/competition_rules.md`](docs/) (coming in Phase 5)
- **API Reference**: [`docs/api_reference.md`](docs/) (coming in Phase 5)

## 🏆 Current Baselines

| Baseline | Status | Description |
|----------|--------|-------------|
| Grid Placer | ✅ Implemented | Simple row-packing placement (example submission) |
| Simulated Annealing | 🚧 Phase 6 | Classical optimization approach |
| Analytical | 🚧 Phase 6 | Force-directed placement |
| Circuit Training | 🚧 Phase 6 | GNN + RL (baseline to beat for prize!) |

**Note**: Baseline scores from the Kahng paper will be used to compute improvement percentages.

## 🤝 Contributing

This is a competition repository. Participants should:
1. Fork the repository
2. Implement your placer
3. Test on public benchmarks
4. Submit your code when ready

## 📚 References

- **Kahng et al. (2023)**: ["Assessment of Reinforcement Learning for Macro Placement"](https://arxiv.org/abs/2302.11014)
- **TILOS MacroPlacement**: [GitHub Repository](https://github.com/TILOS-AI-Institute/MacroPlacement)
- **Circuit Training**: [Google Research](https://github.com/google-research/circuit_training)

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/partcl/partcl-marco-place-challenge/issues)
- **Email**: [contact@partcl.com](contact@partcl.com)

## 📄 License

This project is licensed under the PolyForm Noncommercial License 1.0.0 - see [LICENSE.md](LICENSE.md) for details.

---

**Ready to win $20,000? Start by evaluating the example submission and then creating your own!**

```bash
# Test the example submission
python scripts/evaluate_submission.py \
    --submission submissions/examples/random_placer/placer.py \
    --name "GridPlacer" \
    --split public

# Copy the template and create your own
cp -r submissions/template submissions/my_placer
# Edit submissions/my_placer/placer.py with your algorithm

# Evaluate your submission
python scripts/evaluate_submission.py \
    --submission submissions/my_placer/placer.py \
    --name "MyPlacer" \
    --split public

# Generate leaderboard
python scripts/generate_leaderboard.py
```

Good luck! 🚀
