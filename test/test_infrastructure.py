"""Test PyTorch infrastructure for macro placement."""

import sys
from pathlib import Path
import torch
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loader import load_benchmark_from_dir
from objective import compute_proxy_cost
from utils import validate_placement
from benchmark import Benchmark


BENCHMARK_DIR = "external/MacroPlacement/Testcases/ICCAD04/ibm01"


@pytest.fixture
def benchmark_and_plc():
    """Load benchmark once for tests that need it."""
    return load_benchmark_from_dir(BENCHMARK_DIR)


def test_load_benchmark(benchmark_and_plc):
    """Test loading benchmark from PlacementCost."""
    benchmark, plc = benchmark_and_plc

    assert benchmark.name == "ibm01"
    assert benchmark.num_macros == 246
    assert benchmark.num_nets == 7269
    assert benchmark.macro_positions.shape == (246, 2)
    assert benchmark.macro_sizes.shape == (246, 2)
    assert benchmark.macro_fixed.shape == (246,)
    assert benchmark.canvas_width > 0
    assert benchmark.canvas_height > 0
    assert benchmark.grid_rows == 41
    assert benchmark.grid_cols == 45


def test_validate_initial_placement(benchmark_and_plc):
    """Test validation of initial placement."""
    benchmark, plc = benchmark_and_plc

    is_valid, violations = validate_placement(benchmark.macro_positions, benchmark)

    assert is_valid, f"Initial placement invalid: {violations}"
    assert len(violations) == 0


def test_compute_costs(benchmark_and_plc):
    """Test computing proxy cost with PlacementCost."""
    benchmark, plc = benchmark_and_plc

    costs = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)

    # Check all cost components are present
    assert 'proxy_cost' in costs
    assert 'wirelength_cost' in costs
    assert 'density_cost' in costs
    assert 'congestion_cost' in costs

    # Check overlap metrics are present
    assert 'overlap_count' in costs
    assert 'total_overlap_area' in costs
    assert 'max_overlap_area' in costs
    assert 'num_macros_with_overlaps' in costs
    assert 'overlap_ratio' in costs

    # Check costs are positive
    assert costs['wirelength_cost'] > 0
    assert costs['density_cost'] > 0
    assert costs['congestion_cost'] > 0
    assert costs['proxy_cost'] > 0

    # Check expected values (regression test)
    assert abs(costs['wirelength_cost'] - 0.064080) < 1e-5
    assert abs(costs['density_cost'] - 0.811984) < 1e-5
    assert abs(costs['congestion_cost'] - 1.136852) < 1e-5
    assert abs(costs['proxy_cost'] - 1.038498) < 1e-5

    # Initial placement has some overlaps (this is expected)
    assert costs['overlap_count'] >= 0
    assert costs['overlap_ratio'] >= 0.0
    assert costs['overlap_ratio'] <= 1.0


def test_random_placement(benchmark_and_plc):
    """Test random placement generation and validation."""
    benchmark, plc = benchmark_and_plc

    torch.manual_seed(42)
    random_placement = torch.zeros_like(benchmark.macro_positions)

    for i in range(benchmark.num_macros):
        w, h = benchmark.macro_sizes[i]
        x_min, x_max = w / 2, benchmark.canvas_width - w / 2
        y_min, y_max = h / 2, benchmark.canvas_height - h / 2

        x = torch.rand(1) * (x_max - x_min) + x_min
        y = torch.rand(1) * (y_max - y_min) + y_min

        random_placement[i, 0] = x
        random_placement[i, 1] = y

    # Respect fixed macros
    fixed_mask = benchmark.macro_fixed
    random_placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

    # Validate
    is_valid, violations = validate_placement(random_placement, benchmark)
    assert is_valid, f"Random placement invalid: {violations}"

    # Compute cost
    costs_random = compute_proxy_cost(random_placement, benchmark, plc)
    assert costs_random['proxy_cost'] > 0


def test_save_and_load_benchmark(benchmark_and_plc, tmp_path):
    """Test saving and loading benchmark to .pt file."""
    benchmark, plc = benchmark_and_plc

    output_file = tmp_path / "test_benchmark.pt"

    # Save
    benchmark.save(str(output_file))

    # Load
    loaded = Benchmark.load(str(output_file))

    # Compare
    assert loaded.name == benchmark.name
    assert loaded.num_macros == benchmark.num_macros
    assert loaded.num_nets == benchmark.num_nets
    assert torch.allclose(loaded.macro_positions, benchmark.macro_positions)
    assert torch.allclose(loaded.macro_sizes, benchmark.macro_sizes)
    assert torch.equal(loaded.macro_fixed, benchmark.macro_fixed)


def test_determinism():
    """Test that cost computation is deterministic."""
    results = []

    for _ in range(3):
        benchmark, plc = load_benchmark_from_dir(BENCHMARK_DIR)
        costs = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)
        results.append(costs)

    # Check all runs match exactly
    for key in ['wirelength_cost', 'density_cost', 'congestion_cost', 'proxy_cost']:
        values = [r[key] for r in results]
        assert len(set(values)) == 1, f"{key} is not deterministic: {values}"
