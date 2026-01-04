# Benchmark Metadata

This directory contains metadata files for the macro placement benchmarks.

## Files

### baseline_scores.json

Contains baseline proxy cost scores for each benchmark. These scores are used to compute improvement percentages for submissions.

**Current Status**: **Placeholder baselines**

The current baseline scores are from a simple grid placer. For the actual competition launch, these should be replaced with **Circuit Training** scores from the Kahng paper (arXiv:2302.11014).

**Format**:

```json
{
  "benchmark_name": {
    "proxy_cost": float,        // Combined cost (50% WL + 25% density + 25% congestion)
    "wirelength_cost": float,   // Normalized wirelength cost
    "density_cost": float,      // Density penalty
    "congestion_cost": float,   // Congestion penalty
    "method": string            // Baseline method name
  }
}
```

**Usage**:

The evaluation harness automatically loads these scores to compute improvements:

```python
from marco_place.evaluation.harness import load_baseline_scores

baselines = load_baseline_scores("benchmarks/metadata/baseline_scores.json")

# For each benchmark, compute improvement
improvement = (baseline_cost - submission_cost) / baseline_cost * 100
```

## TODO for Competition Launch

Before launching the competition officially:

1. **Run Circuit Training** on all public benchmarks
2. **Extract proxy costs** from Circuit Training results
3. **Replace placeholder scores** in baseline_scores.json with CT scores
4. **Document Circuit Training setup** (hyperparameters, runtime, hardware)
5. **Verify reproducibility** of baseline scores

## Baseline Method: Circuit Training

Circuit Training is the state-of-the-art method from Google Research:

- **Paper**: [Mirhoseini et al. "A graph placement methodology for fast chip design." Nature, 2021]
- **GitHub**: https://github.com/google-research/circuit_training
- **Approach**: Graph Neural Network + Reinforcement Learning
- **Performance**: Best results in Kahng's assessment paper

## Expected Baseline Performance

Based on the Kahng paper, Circuit Training typically achieves:
- **Proxy cost**: 0.05-0.15 (varies by benchmark)
- **Improvement over SA**: +15-25%
- **Improvement over RePlAce**: +10-20%

For participants to win the $20K prize, they must **beat these Circuit Training scores on aggregate**.

## Contact

For questions about baselines or to request additional metadata:
- Open an issue on GitHub
- Email: contact@partcl.com
