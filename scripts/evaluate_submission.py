#!/usr/bin/env python3
"""
Evaluate a submission on all benchmarks.

This script evaluates a placer submission on all benchmarks and generates
aggregate scores for the leaderboard.

Usage:
    python scripts/evaluate_submission.py --submission submissions/examples/random_placer/placer.py
    python scripts/evaluate_submission.py --submission path/to/your/placer.py --name "MyPlacer"
"""

import argparse
import sys
import importlib.util
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from marco_place.evaluation.harness import EvaluationHarness, load_baseline_scores


def load_placer_class(submission_file: Path):
    """
    Dynamically load placer class from Python file.

    Args:
        submission_file: Path to Python file containing placer

    Returns:
        Placer class
    """
    # Load module
    spec = importlib.util.spec_from_file_location("submission", submission_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find placer class (should inherit from BasePlacerInterface or have place() method)
    placer_class = None

    for attr_name in dir(module):
        attr = getattr(module, attr_name)

        # Check if it's a class with a place method
        if isinstance(attr, type) and hasattr(attr, 'place'):
            # Skip BasePlacerInterface itself
            if attr_name == 'BasePlacerInterface':
                continue

            placer_class = attr
            break

    if placer_class is None:
        raise ValueError(f"No placer class found in {submission_file}")

    return placer_class


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate submission on all benchmarks"
    )
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Path to submission Python file (must contain placer class)"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Submission name for leaderboard (default: filename)"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        help="Specific benchmarks to evaluate (default: all)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="public",
        choices=["public", "hidden", "all"],
        help="Benchmark split to evaluate (default: public)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds per benchmark (default: 3600 = 1 hour)"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="benchmarks/metadata/baseline_scores.json",
        help="Path to baseline scores JSON (for computing improvements)"
    )

    args = parser.parse_args()

    # Load submission
    submission_file = Path(args.submission)
    if not submission_file.exists():
        print(f"Error: Submission file not found: {submission_file}")
        return 1

    print(f"Loading submission from: {submission_file}")

    try:
        placer_class = load_placer_class(submission_file)
        print(f"✓ Loaded placer class: {placer_class.__name__}")
    except Exception as e:
        print(f"Error loading placer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Determine submission name
    if args.name:
        submission_name = args.name
    else:
        submission_name = submission_file.stem

    # Load baseline scores
    baseline_scores = load_baseline_scores(args.baseline)
    if baseline_scores:
        print(f"✓ Loaded baseline scores for {len(baseline_scores)} benchmarks")

    # Create evaluation harness
    harness = EvaluationHarness(
        benchmark_dir="benchmarks/processed",
        split=args.split,
        output_dir="results",
        timeout_seconds=args.timeout,
        baseline_scores=baseline_scores
    )

    # Evaluate submission
    results = harness.evaluate_submission(
        placer_class=placer_class,
        submission_name=submission_name,
        benchmark_names=args.benchmarks,
        verbose=True
    )

    # Print final summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Submission: {submission_name}")
    print(f"Aggregate Score: {results['summary'].get('aggregate_score', 0):+.2f}%")

    if results['summary'].get('prize_eligible', False):
        print("\n🏆 CONGRATULATIONS! This submission qualifies for the $20,000 prize!")
        print("   (Pending verification and final evaluation on hidden test cases)")
    else:
        print("\n✗ This submission does not yet beat the Circuit Training baseline.")
        print("   Keep optimizing to qualify for the prize!")

    print("\nNext steps:")
    print("  1. Review results in: results/")
    print("  2. Generate leaderboard: python scripts/generate_leaderboard.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
