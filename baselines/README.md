# Baseline Implementations

This directory contains baseline placer implementations for comparison.

## Overview

Baselines serve two purposes:
1. **Performance benchmarks** - Establish score targets for competition participants
2. **Reference implementations** - Demonstrate different algorithmic approaches

## Implemented Baselines

### 1. Grid Placer ✅ **Most Reliable**

**Location**: `submissions/examples/random_placer/placer.py`

**Algorithm**: Simple row-packing placement
- Sort macros by height
- Place left-to-right in rows
- Move to next row when current row is full

**Performance**:
| Benchmark | Proxy Cost | Valid |
|-----------|-----------|-------|
| ariane133 | 0.200452 | ✓ |
| nvdla_asap7 | 0.122080 | ✓ |
| blackparrot_asap7 | 0.111102 | ✓ |
| mempool_asap7 | 0.117310 | ✓ |
| toy_small | 0.170180 | ✓ |
| toy_medium | 0.206040 | ✓ |

**Pros**:
- Fast (<1 second)
- Always produces valid placements
- No parameters to tune
- Good starting point

**Cons**:
- No wirelength optimization
- Fixed layout (no adaptation)
- Poor congestion handling

**Status**: ✅ **Complete and reliable** - Currently used as placeholder baseline

---

### 2. Simulated Annealing ⚠️ **Work in Progress**

**Location**: `baselines/simulated_annealing/sa_placer.py`

**Algorithm**: Probabilistic optimization
- Start with grid placement
- Iteratively perturb (swap, shift)
- Accept better/worse solutions based on temperature
- Cool down gradually

**Performance**:
| Benchmark | Result | Notes |
|-----------|--------|-------|
| toy_small | 0.158338 (+6.96%) | ✓ Valid |
| toy_medium | Invalid (overlaps) | 107s runtime |
| ariane133 | Timeout | >10 minutes |

**Pros**:
- Shows improvement on small benchmarks
- Classical, well-understood approach
- Can escape local minima

**Cons**:
- Slow (cost evaluation per iteration)
- Produces invalid placements (overlaps)
- Needs better neighbor generation
- Requires legalization post-processing

**Status**: ⚠️ **Needs more work** - Overlap handling and speed optimization required

---

### 3. Analytical (Force-Directed) ⚠️ **Work in Progress**

**Location**: `baselines/analytical/analytical_placer.py`

**Algorithm**: Force-directed placement
- Attractive forces (from net connections)
- Repulsive forces (between macros)
- Iterative position updates
- Legalization to remove overlaps

**Performance**:
| Benchmark | Result | Notes |
|-----------|--------|-------|
| toy_small | 0.169933 (+0.15%) | ✓ Valid |
| ariane133 | Invalid (165 overlaps) | 26s runtime |

**Pros**:
- Reasonably fast (~25s for 133 macros)
- Explicit wirelength optimization
- Intuitive physical model

**Cons**:
- Legalization insufficient for large designs
- Many overlaps on complex benchmarks
- Needs stronger overlap resolution

**Status**: ⚠️ **Needs more work** - Legalization needs improvement

---

### 4. Circuit Training (GNN + RL) 🚧 **Not Implemented**

**Algorithm**: Google's state-of-the-art approach
- Graph Neural Network for feature learning
- Reinforcement Learning for placement decisions
- Sequence-to-sequence macro placement

**Performance**: Best in Kahng paper (ISPD 2023)
- Beats Simulated Annealing by 15-25%
- Beats analytical methods by 10-20%
- Competitive with commercial tools

**Status**: 🚧 **Not implemented** - Would require:
- PyTorch Geometric for GNN
- RL framework (PPO/A3C)
- Significant training time
- Hyperparameter tuning
- ~1000+ lines of code

**Importance**: This is the baseline participants must beat to win $20K

---

## Baseline Comparison

### Small Benchmarks (10-20 macros)

| Method | Runtime | Proxy Cost | Valid | Improvement |
|--------|---------|-----------|-------|-------------|
| Grid | <1s | 0.170180 | ✓ | baseline |
| SA | ~23s | 0.158338 | ✓ | +6.96% |
| Analytical | <1s | 0.169933 | ✓ | +0.15% |

### Medium Benchmarks (50 macros)

| Method | Runtime | Valid | Notes |
|--------|---------|-------|-------|
| Grid | <1s | ✓ | Reliable |
| SA | ~107s | ✗ | Overlaps |
| Analytical | <1s | ✓ | Small improvement |

### Large Benchmarks (100+ macros)

| Method | Runtime | Valid | Notes |
|--------|---------|-------|-------|
| Grid | <1s | ✓ | Reliable |
| SA | >600s | ⏱️ | Timeout |
| Analytical | ~26s | ✗ | Many overlaps |

---

## Recommendations

### For Competition Launch

**Current Status**: Grid placer is the most reliable baseline

**Action Items**:
1. ✅ Use Grid placer scores as temporary baseline
2. 🚧 Implement Circuit Training (or use published scores from Kahng paper)
3. 🚧 Fine-tune SA and Analytical to work on all benchmarks
4. 🚧 Replace placeholder scores with actual CT scores

### For Participants

**Starting Point**: Use Grid placer as reference
- Simple to understand
- Always produces valid placements
- Shows what "baseline" performance looks like

**Improvement Strategies**:
1. **Short-term** (+5-10%): Improve on grid with:
   - Better initial placement
   - Local refinement
   - Wirelength-aware positioning

2. **Medium-term** (+10-20%): Classical optimization
   - Simulated Annealing (with proper legalization)
   - Analytical methods (with strong overlap resolution)
   - Genetic algorithms

3. **Long-term** (+20-40%): ML-based approaches
   - GNN + RL (Circuit Training style)
   - Transformers for sequence modeling
   - Hybrid ML + optimization

---

## Future Work

### Immediate (Phase 6 completion)

- [ ] Fix SA overlap generation
- [ ] Improve Analytical legalization
- [ ] Tune hyperparameters for each benchmark size
- [ ] Add incremental cost evaluation for speed

### Medium-term (Phase 7-8)

- [ ] Implement Circuit Training baseline
- [ ] Run CT on all public benchmarks
- [ ] Replace placeholder scores with CT scores
- [ ] Create hidden benchmarks

### Long-term (Post-competition)

- [ ] Publish baseline code and results
- [ ] Write technical report on baseline comparison
- [ ] Release pre-trained CT models
- [ ] Open-source evaluation framework

---

## Usage

### Running Baselines

```bash
# Grid Placer
python scripts/evaluate_submission.py \
    --submission submissions/examples/random_placer/placer.py \
    --name "GridPlacer" \
    --split public

# Simulated Annealing
python scripts/evaluate_submission.py \
    --submission baselines/simulated_annealing/sa_placer.py \
    --name "SA_Baseline" \
    --benchmarks toy_small toy_medium

# Analytical
python scripts/evaluate_submission.py \
    --submission baselines/analytical/analytical_placer.py \
    --name "Analytical_Baseline" \
    --benchmarks toy_small toy_medium
```

### Comparing Baselines

```bash
# Run all baselines
python scripts/evaluate_submission.py --submission submissions/examples/random_placer/placer.py --name "Grid" --split public
python scripts/evaluate_submission.py --submission baselines/simulated_annealing/sa_placer.py --name "SA" --benchmarks toy_small
python scripts/evaluate_submission.py --submission baselines/analytical/analytical_placer.py --name "Analytical" --benchmarks toy_small

# Generate leaderboard
python scripts/generate_leaderboard.py
```

---

## References

### Simulated Annealing
- Kirkpatrick et al. "Optimization by Simulated Annealing." Science, 1983.
- Wong et al. "Simulated Annealing for VLSI Design." Kluwer, 1988.

### Analytical Placement
- Eisenmann & Johannes. "Generic Global Placement and Floorplanning." DAC, 1998.
- Viswanathan & Chu. "FastPlace: Efficient Analytical Placement using Cell Shifting." ISPD, 2004.

### Circuit Training (Target Baseline)
- Mirhoseini et al. "A graph placement methodology for fast chip design." Nature, 2021.
- Kahng et al. "Assessment of Reinforcement Learning for Macro Placement." ISPD, 2023.
  - **arXiv**: https://arxiv.org/abs/2302.11014
  - **GitHub**: https://github.com/TILOS-AI-Institute/MacroPlacement

---

## Contact

Questions about baselines:
- Open an issue on GitHub
- Email: contact@partcl.com
