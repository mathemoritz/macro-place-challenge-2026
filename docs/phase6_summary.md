# Phase 6 Summary: Baseline Implementations

**Date**: 2026-01-03
**Status**: Completed (with known limitations)

---

## Overview

Phase 6 focused on implementing competitive baseline algorithms to establish performance targets for the competition. The goal was to create multiple placement algorithms that participants could compare against.

## Implemented Baselines

### 1. Grid Placer ✅ **Production Ready**

**Status**: Complete and reliable

**Implementation**: `submissions/examples/random_placer/placer.py`

**Algorithm**:
- Row-packing with height-based sorting
- Guaranteed valid placements
- No optimization (pure geometric placement)

**Performance**:
- Runtime: <1 second on all benchmarks
- Valid on 6/7 benchmarks (toy_large has unrealistic macro sizes)
- Proxy costs: 0.11-0.21 across benchmarks

**Verdict**: **Recommended as temporary baseline** until Circuit Training is implemented

---

### 2. Simulated Annealing ⚠️ **Prototype**

**Status**: Works on small benchmarks, needs improvements for production

**Implementation**: `baselines/simulated_annealing/sa_placer.py`

**Algorithm**:
- Probabilistic optimization with temperature cooling
- Three move types: swap, small shift, large shift
- Metropolis acceptance criterion
- Exponential cooling schedule

**Performance**:
- **toy_small** (10 macros): 0.158338 (+6.96% improvement) ✓
- **toy_medium** (50 macros): Invalid (overlaps) ✗
- **ariane133** (133 macros): Timeout (>600s) ✗

**Issues**:
1. **Overlap Generation**: Swap/shift moves frequently create overlaps
2. **Slow Convergence**: Cost evaluation on every iteration is expensive
3. **Invalid Final States**: No guarantee of legality

**Improvements Needed**:
- Overlap-aware move generation
- Incremental cost evaluation
- Post-processing legalization step
- Adaptive cooling schedule

---

### 3. Analytical (Force-Directed) ⚠️ **Prototype**

**Status**: Fast but unreliable on large benchmarks

**Implementation**: `baselines/analytical/analytical_placer.py`

**Algorithm**:
- Attractive forces from net connections
- Repulsive forces between macros
- Iterative force-directed updates
- Greedy overlap resolution

**Performance**:
- **toy_small** (10 macros): 0.169933 (+0.15% improvement) ✓
- **ariane133** (133 macros): Invalid (165 overlaps) ✗

**Issues**:
1. **Weak Legalization**: Greedy push-apart doesn't scale
2. **Force Imbalance**: Attractive vs repulsive weights need tuning
3. **Convergence**: May get stuck in local minima

**Improvements Needed**:
- Quadratic programming for overlap resolution
- Better force model (cell spreading)
- Multi-stage approach (global → detailed → legalization)

---

### 4. Circuit Training 🚧 **Not Implemented**

**Status**: Not implemented (recommended for full competition launch)

**Why It's Important**:
- Best method in Kahng's assessment paper
- Target baseline for $20K prize eligibility
- Represents state-of-the-art in macro placement

**Implementation Complexity**:
- ~1000-2000 lines of code
- Requires PyTorch Geometric (GNN)
- Requires RL framework (PPO/A3C)
- Needs significant training time (days/weeks)
- Hyperparameter tuning required

**Alternative Approach**:
- Use published scores from Kahng paper directly
- Document that "Circuit Training" is the target
- Participants beat these published scores to win

---

## Performance Summary

### Reliability Ranking (Valid Placements)

1. **Grid Placer**: 6/7 benchmarks ✓✓✓
2. **Analytical**: 1/2 tested ✓
3. **SA**: 1/3 tested ✓

### Speed Ranking (Runtime)

1. **Grid Placer**: <1 second ⚡⚡⚡
2. **Analytical**: ~25 seconds ⚡⚡
3. **SA**: 23-600+ seconds ⚡

### Quality Ranking (Proxy Cost on toy_small)

1. **SA**: 0.158338 (-7% vs grid) 🏆
2. **Analytical**: 0.169933 (-0.1% vs grid) 🥈
3. **Grid**: 0.170180 (baseline) 🥉

---

## Baseline Scores Generated

Created `benchmarks/metadata/baseline_scores.json` with Grid Placer scores:

```json
{
  "ariane133": {"proxy_cost": 0.200452},
  "nvdla_asap7": {"proxy_cost": 0.122080},
  "blackparrot_asap7": {"proxy_cost": 0.111102},
  "mempool_asap7": {"proxy_cost": 0.117310},
  "toy_small": {"proxy_cost": 0.170180},
  "toy_medium": {"proxy_cost": 0.206040}
}
```

**Note**: These are **placeholder scores**. For official competition, replace with Circuit Training scores from Kahng paper.

---

## Lessons Learned

### 1. Legality is Hard

Both SA and Analytical struggle with overlap resolution. Valid placement generation is more difficult than expected, especially for:
- Dense designs (many macros, small canvas)
- Irregular macro sizes
- Tight packing requirements

**Takeaway**: Competition participants will face the same challenges. This makes the problem interesting!

### 2. Speed-Quality Tradeoff

| Method | Speed | Quality | Reliability |
|--------|-------|---------|-------------|
| Grid | ⚡⚡⚡ | ⭐ | ✓✓✓ |
| Analytical | ⚡⚡ | ⭐⭐ | ✓ |
| SA | ⚡ | ⭐⭐⭐ | ✓ |

- Fast methods (Grid) → Simple but low quality
- Slow methods (SA) → Better quality but not production-ready
- Need sweet spot: Fast + Good + Reliable

### 3. Implementation Complexity

| Method | Lines of Code | Time to Implement |
|--------|--------------|-------------------|
| Grid | ~100 | 1 hour |
| Analytical | ~300 | 3 hours |
| SA | ~350 | 4 hours |
| Circuit Training | ~1500 | 1-2 weeks |

**Takeaway**: Circuit Training is significantly more complex. Using published scores is practical for beta launch.

---

## Recommendations

### For Beta Launch (Current)

✅ **Use Grid Placer as baseline**
- Reliable and fast
- Produces valid placements
- Easy to understand
- Good reference point

✅ **Document SA and Analytical as prototypes**
- Show participants different approaches
- Demonstrate common challenges
- Provide starting points for improvement

### For Official Launch (Future)

🚧 **Replace with Circuit Training scores**
- Use published results from Kahng paper (ISPD 2023)
- Document proxy cost methodology
- Clearly state: "To win $20K, beat these Circuit Training scores"

🚧 **Improve SA and Analytical**
- Fix overlap issues
- Optimize for speed
- Make production-ready

🚧 **Implement Circuit Training (Optional)**
- Provides verifiable baseline
- Allows participants to study approach
- Demonstrates feasibility

---

## Action Items

### Immediate (Before Beta Launch)

- [x] Document baseline implementations
- [x] Generate Grid Placer baseline scores
- [x] Create baselines README
- [x] Test on all public benchmarks

### Before Official Launch

- [ ] Replace placeholder scores with Circuit Training scores
- [ ] Fix SA overlap generation
- [ ] Improve Analytical legalization
- [ ] Add baseline comparison visualization
- [ ] Publish baseline results paper

### Optional (Enhancement)

- [ ] Implement Circuit Training from scratch
- [ ] Create baseline training scripts
- [ ] Add more classical baselines (GA, tabu search)
- [ ] Benchmark against commercial tools

---

## Conclusion

**Phase 6 Status**: ✅ **Complete** (with known limitations)

**Key Achievements**:
1. ✅ Implemented 3 baselines (Grid, SA, Analytical)
2. ✅ Generated baseline scores for 6 benchmarks
3. ✅ Documented performance and limitations
4. ✅ Identified challenges for participants

**Known Limitations**:
1. ⚠️ SA and Analytical need improvements for large benchmarks
2. ⚠️ Placeholder baseline scores (not Circuit Training)
3. ⚠️ Circuit Training not implemented

**Ready for Beta?**: ✅ **YES**
- Grid Placer is reliable enough for beta testing
- Participants can compare against Grid scores
- SA and Analytical show what's possible with optimization
- Documentation clearly explains limitations

**Ready for Official Launch?**: 🚧 **NOT YET**
- Need Circuit Training scores from Kahng paper
- Should improve SA and Analytical (optional but recommended)
- Consider implementing or licensing Circuit Training

---

## Timeline Estimate

To make Phase 6 production-ready:

**Option A: Use Published Scores** (Recommended)
- Time: 1 week
- Extract CT scores from Kahng paper
- Update baseline_scores.json
- Document methodology

**Option B: Implement Circuit Training**
- Time: 2-4 weeks
- Implement GNN + RL from scratch
- Train on public benchmarks
- Verify results match paper

**Current Status**: Grid Placer provides adequate baseline for beta launch. Circuit Training scores needed for official launch.

---

## Contact

Questions about Phase 6:
- Will (Implementation Lead)
- GitHub Issues: https://github.com/partcl/partcl-marco-place-challenge/issues
