#!/usr/bin/env python3
"""
Generate leaderboard from evaluation results.

This script reads all result JSON files and generates a ranked leaderboard.

Usage:
    python scripts/generate_leaderboard.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all result JSON files.

    Args:
        results_dir: Directory containing result files

    Returns:
        List of result dictionaries
    """
    results = []

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return results

    for result_file in results_dir.glob("*_results.json"):
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")

    return results


def generate_leaderboard(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate leaderboard from results.

    Args:
        results: List of result dictionaries

    Returns:
        Leaderboard dictionary
    """
    leaderboard = {
        'last_updated': None,
        'entries': []
    }

    # Find most recent timestamp
    timestamps = [r.get('timestamp') for r in results if r.get('timestamp')]
    if timestamps:
        leaderboard['last_updated'] = max(timestamps)

    # Create leaderboard entries
    for result in results:
        summary = result.get('summary', {})

        entry = {
            'submission_name': result.get('submission_name', 'Unknown'),
            'timestamp': result.get('timestamp'),
            'aggregate_score': summary.get('aggregate_score', 0.0),
            'valid_benchmarks': summary.get('valid_benchmarks', 0),
            'total_benchmarks': summary.get('total_benchmarks', 0),
            'total_runtime': summary.get('total_runtime', 0.0),
            'prize_eligible': summary.get('prize_eligible', False),
            'mean_improvement': summary.get('mean_improvement', 0.0),
        }

        leaderboard['entries'].append(entry)

    # Sort by aggregate score (descending)
    leaderboard['entries'].sort(key=lambda x: x['aggregate_score'], reverse=True)

    # Add rank
    for i, entry in enumerate(leaderboard['entries'], 1):
        entry['rank'] = i

    return leaderboard


def print_leaderboard(leaderboard: Dict[str, Any]):
    """Print formatted leaderboard."""
    print("\n" + "=" * 100)
    print("MACRO PLACEMENT CHALLENGE - LEADERBOARD")
    print("=" * 100)

    if leaderboard['last_updated']:
        print(f"Last Updated: {leaderboard['last_updated']}")

    print("\n Prize Eligibility: Aggregate Score > 0 (must beat Circuit Training baseline)")
    print("=" * 100)

    if not leaderboard['entries']:
        print("\nNo submissions yet!")
        return

    # Header
    print(f"\n{'Rank':<6} {'Submission':<30} {'Aggregate':<12} {'Valid/Total':<12} {'Runtime':<12} {'Prize':<10}")
    print(f"{'':6} {'':30} {'Score (%)':<12} {'Benchmarks':<12} {'(seconds)':<12} {'Eligible':<10}")
    print("-" * 100)

    # Entries
    for entry in leaderboard['entries']:
        rank = entry['rank']
        name = entry['submission_name'][:28]
        score = entry['aggregate_score']
        valid = entry['valid_benchmarks']
        total = entry['total_benchmarks']
        runtime = entry['total_runtime']
        prize = "✓ YES" if entry['prize_eligible'] else "✗ No"

        # Add medal emojis for top 3
        medal = ""
        if rank == 1:
            medal = "🥇 "
        elif rank == 2:
            medal = "🥈 "
        elif rank == 3:
            medal = "🥉 "

        print(f"{medal}{rank:<4} {name:<30} {score:>+10.2f}  {valid:>2}/{total:<8} {runtime:>10.1f}  {prize:<10}")

    print("=" * 100)

    # Prize eligibility summary
    prize_eligible = sum(1 for e in leaderboard['entries'] if e['prize_eligible'])
    print(f"\nPrize Eligible Submissions: {prize_eligible}/{len(leaderboard['entries'])}")

    if prize_eligible > 0:
        winner = leaderboard['entries'][0]
        if winner['prize_eligible']:
            print(f"\n🏆 Current Leader: {winner['submission_name']} (Score: {winner['aggregate_score']:+.2f}%)")
            print(f"   This submission currently qualifies for the $20,000 prize!")


def save_leaderboard(leaderboard: Dict[str, Any], output_file: Path):
    """Save leaderboard to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(leaderboard, f, indent=2)

    print(f"\n✓ Leaderboard saved to: {output_file}")


def main():
    results_dir = project_root / "results"
    output_file = results_dir / "leaderboard.json"

    # Load results
    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)
    print(f"Found {len(results)} submissions")

    if not results:
        print("No results found. Run evaluations first!")
        return 1

    # Generate leaderboard
    leaderboard = generate_leaderboard(results)

    # Print leaderboard
    print_leaderboard(leaderboard)

    # Save leaderboard
    save_leaderboard(leaderboard, output_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
