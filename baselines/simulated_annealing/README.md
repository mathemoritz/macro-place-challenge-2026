# Simulated Annealing Baseline

Classical optimization baseline using simulated annealing for macro placement.

## Status

⚠️ **Work in Progress** - Currently produces invalid placements on larger benchmarks due to overlap issues.

## Algorithm

Simulated Annealing (SA) is a probabilistic optimization technique that explores the solution space by:

1. **Initial Placement**: Start with grid-based row packing
2. **Iterative Improvement**: Generate neighbor solutions via:
   - Swapping two macros (40% probability)
   - Small random shift (40% probability)
   - Large random shift (20% probability)
3. **Acceptance Criterion**:
   - Accept better solutions always
   - Accept worse solutions with probability `exp(-ΔE / T)`
4. **Cooling Schedule**: Exponential temperature decay
5. **Return**: Best solution found

## Parameters

- **Max iterations**: 5000 (reduced for speed)
- **Initial temperature**: 0.5
- **Final temperature**: 0.001
- **Cooling rate**: 0.98 (applied every 50 iterations)
- **Max time**: 50 minutes

## Performance

### toy_small (10 macros)
- **Proxy cost**: 0.158338
- **Improvement over grid**: +6.96%
- **Runtime**: ~23 seconds
- **Status**: ✓ Valid placement

### toy_medium (50 macros)
- **Status**: ✗ Invalid (overlaps)
- **Runtime**: ~107 seconds
- **Issue**: Overlap handling needs improvement

### ariane133 (133 macros)
- **Status**: ⏱️ Timeout after 10 minutes
- **Issue**: Too slow for large benchmarks

## Known Issues

1. **Overlap Generation**: Swap and shift moves frequently create overlaps
2. **Slow Cost Evaluation**: Computing proxy cost on every iteration is expensive
3. **Invalid Final State**: Validation only checks final placement, not intermediate states

## Improvements Needed

1. **Better Neighbor Generation**:
   - Check for overlaps before accepting moves
   - Use guided perturbations that preserve legality
   - Implement overlap-aware swap selection

2. **Performance Optimization**:
   - Batch cost evaluations
   - Use incremental cost updates (delta evaluation)
   - Reduce validation overhead

3. **Legalization**:
   - Add explicit legalization step after SA
   - Use force-directed spreading to resolve overlaps
   - Implement constraint-aware moves

## Usage

```bash
# Test on small benchmark
python scripts/evaluate_submission.py \
    --submission baselines/simulated_annealing/sa_placer.py \
    --name "SA_Baseline" \
    --benchmarks toy_small

# Test standalone
python baselines/simulated_annealing/sa_placer.py
```

## Future Work

- Implement overlap-aware moves
- Add legalization post-processing
- Optimize cost evaluation (incremental updates)
- Tune hyperparameters for each benchmark size
- Consider parallel tempering or other SA variants

## References

- Kirkpatrick et al. "Optimization by Simulated Annealing." Science, 1983.
- Wong et al. "Simulated Annealing for VLSI Design." Kluwer, 1988.
