# Random Placer - Example Submission

This is a simple baseline implementation that serves as a working example of a valid submission for the Macro Placement Challenge.

## Algorithm

The RandomPlacer uses rejection sampling to place macros:

1. For each macro, generate random (x, y) coordinates within the canvas
2. Check if the position is valid (no overlaps, within boundaries)
3. If valid, place the macro; otherwise, try again
4. After a maximum number of attempts, fall back to grid placement

This is **not** expected to produce good placements, but demonstrates:
- Correct submission structure
- Valid placement generation (no overlaps, within boundaries)
- Proper interface implementation

## Files

- `placer.py` - Main placer implementation
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Usage

### Testing Locally

```bash
# Test the placer standalone
python submissions/examples/random_placer/placer.py
```

### Running Evaluation

```bash
# Evaluate on all public benchmarks
python scripts/evaluate_submission.py \
    --submission submissions/examples/random_placer/placer.py \
    --name "RandomPlacer"

# Evaluate on specific benchmarks
python scripts/evaluate_submission.py \
    --submission submissions/examples/random_placer/placer.py \
    --name "RandomPlacer" \
    --benchmarks ariane133 nvdla_asap7
```

### Generate Leaderboard

```bash
# After running evaluations
python scripts/generate_leaderboard.py
```

## Expected Performance

This baseline is **not prize eligible** as it will not beat the Circuit Training baseline. It serves purely as:
- A template for creating your own submission
- A sanity check that the evaluation pipeline works
- A lower bound on performance

## Modifying for Your Submission

1. Copy this directory to create your own submission
2. Replace the random placement logic with your algorithm
3. Ensure your placer class has a `place(circuit_data)` method that returns a `[num_macros, 2]` tensor
4. Test locally first, then run the full evaluation

## Implementation Notes

- **Seed**: Uses random seed for reproducibility
- **Max attempts**: Falls back to grid after 10,000 failed attempts
- **No optimization**: Does not consider wirelength, density, or congestion
- **Runtime**: Should complete quickly on all benchmarks (< 1 minute)

## Prize Eligibility

To be eligible for the $20,000 prize, your submission must:
1. Beat the Circuit Training baseline on aggregate across all benchmarks
2. Produce valid placements (no overlaps, within boundaries)
3. Complete within the timeout (1 hour per benchmark)

This random baseline will **not** qualify for the prize.
