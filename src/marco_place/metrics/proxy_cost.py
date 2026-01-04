"""
Proxy cost calculation for macro placement.

The proxy cost is a weighted combination of wirelength, density, and congestion
that serves as a fast approximation of placement quality.
"""

from typing import Dict, Optional, Tuple
import torch

from marco_place.data.tensor_schema import CircuitTensorData
from marco_place.metrics.wirelength import compute_normalized_wirelength
from marco_place.metrics.density import compute_density_cost
from marco_place.metrics.congestion import compute_congestion_cost


def compute_proxy_cost(
    placement: torch.Tensor,
    circuit_data: CircuitTensorData,
    weights: Optional[Dict[str, float]] = None,
    grid_shape: Tuple[int, int] = (32, 32),
    target_density: float = 0.6
) -> Dict[str, float]:
    """
    Compute proxy cost as weighted sum of wirelength, density, and congestion.

    The proxy cost provides fast feedback for placement optimization without
    requiring time-consuming place-and-route runs.

    Default weights (from literature):
    - Wirelength: 0.5 (most important)
    - Density: 0.25 (uniform distribution)
    - Congestion: 0.25 (avoid routing bottlenecks)

    Args:
        placement: [num_macros, 2] - (x, y) coordinates of macro centers
        circuit_data: Circuit data containing macro sizes, nets, canvas info
        weights: Optional custom weights dict with keys 'wirelength', 'density', 'congestion'
        grid_shape: Grid resolution for density and congestion calculation
        target_density: Target density for density cost calculation

    Returns:
        Dictionary with individual costs and total proxy cost:
        {
            'wirelength_cost': float,
            'density_cost': float,
            'congestion_cost': float,
            'proxy_cost': float  # weighted sum
        }
    """
    # Default weights if not specified
    if weights is None:
        weights = {
            'wirelength': 0.5,
            'density': 0.25,
            'congestion': 0.25
        }

    # Validate weights
    if not isinstance(weights, dict):
        raise TypeError("weights must be a dictionary")

    required_keys = ['wirelength', 'density', 'congestion']
    for key in required_keys:
        if key not in weights:
            raise ValueError(f"Missing weight for '{key}'")

    # Compute individual costs
    wirelength_cost = compute_normalized_wirelength(placement, circuit_data)

    density_cost = compute_density_cost(
        placement,
        circuit_data,
        target_density=target_density,
        grid_shape=grid_shape
    )

    congestion_cost = compute_congestion_cost(
        placement,
        circuit_data,
        grid_shape=grid_shape
    )

    # Compute weighted proxy cost
    proxy_cost = (
        weights['wirelength'] * wirelength_cost +
        weights['density'] * density_cost +
        weights['congestion'] * congestion_cost
    )

    return {
        'wirelength_cost': wirelength_cost,
        'density_cost': density_cost,
        'congestion_cost': congestion_cost,
        'proxy_cost': proxy_cost,
    }


def compute_improvement(
    submission_cost: float,
    baseline_cost: float
) -> float:
    """
    Compute percentage improvement of submission over baseline.

    Args:
        submission_cost: Proxy cost of submission
        baseline_cost: Proxy cost of baseline (Circuit Training)

    Returns:
        Improvement as percentage (positive = better than baseline)
        E.g., 10.0 means 10% better than baseline
    """
    if baseline_cost == 0:
        raise ValueError("Baseline cost cannot be zero")

    improvement = (baseline_cost - submission_cost) / baseline_cost * 100

    return improvement


def compute_aggregate_score(
    improvements: list[float],
    method: str = 'geometric_mean'
) -> float:
    """
    Compute aggregate score across multiple benchmarks.

    Args:
        improvements: List of per-benchmark improvement percentages
        method: Aggregation method
                'geometric_mean': Geometric mean (default, recommended)
                'arithmetic_mean': Arithmetic mean
                'median': Median improvement

    Returns:
        Aggregate score

    Note:
        Geometric mean is recommended because it penalizes solutions
        that sacrifice one benchmark to improve another.
    """
    if len(improvements) == 0:
        return 0.0

    # Filter out invalid benchmarks (those with 0 improvement due to failures)
    valid_improvements = [imp for imp in improvements if imp != 0]

    if len(valid_improvements) == 0:
        return 0.0

    if method == 'geometric_mean':
        # Shift values to make all positive for geometric mean
        # Add 100 so that 0% improvement becomes 100
        shifted = [imp + 100 for imp in valid_improvements]

        # Compute geometric mean
        product = 1.0
        for val in shifted:
            product *= val

        geom_mean = product ** (1.0 / len(shifted))

        # Shift back
        return geom_mean - 100

    elif method == 'arithmetic_mean':
        return sum(valid_improvements) / len(valid_improvements)

    elif method == 'median':
        sorted_improvements = sorted(valid_improvements)
        n = len(sorted_improvements)
        if n % 2 == 0:
            return (sorted_improvements[n//2 - 1] + sorted_improvements[n//2]) / 2
        else:
            return sorted_improvements[n//2]

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def is_prize_eligible(aggregate_score: float) -> bool:
    """
    Check if aggregate score qualifies for prize.

    Prize is awarded ONLY if aggregate score > 0, meaning the submission
    beats the Circuit Training baseline on average across all benchmarks.

    Args:
        aggregate_score: Aggregate improvement percentage

    Returns:
        True if eligible for $20K prize, False otherwise
    """
    return aggregate_score > 0
