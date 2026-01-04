"""
PyTorch Dataset interface for macro placement benchmarks.

This module provides Dataset classes for loading and iterating over benchmarks.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
import torch
from torch.utils.data import Dataset

from marco_place.data.tensor_schema import CircuitTensorData


class MacroPlacementDataset(Dataset):
    """
    PyTorch Dataset for macro placement benchmarks.

    Loads benchmarks from a directory and provides standard Dataset interface
    for training and evaluation.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'public',
        benchmark_names: Optional[List[str]] = None
    ):
        """
        Initialize MacroPlacementDataset.

        Args:
            data_dir: Path to benchmarks directory (should contain 'public' and/or 'hidden' subdirs)
            split: Which split to load - 'public', 'hidden', or 'all'
            benchmark_names: Optional list of specific benchmark names to load.
                           If None, loads all benchmarks in the split.
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # Determine which directories to search
        if split == 'all':
            search_dirs = [self.data_dir / 'public', self.data_dir / 'hidden']
        else:
            search_dirs = [self.data_dir / split]

        # Find all benchmark files
        self.benchmark_files = []
        self.benchmark_names_list = []

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # Get all .pt files
            pt_files = sorted(search_dir.glob('*.pt'))

            for pt_file in pt_files:
                benchmark_name = pt_file.stem

                # Filter by benchmark_names if specified
                if benchmark_names is not None and benchmark_name not in benchmark_names:
                    continue

                self.benchmark_files.append(pt_file)
                self.benchmark_names_list.append(benchmark_name)

        if len(self.benchmark_files) == 0:
            raise ValueError(
                f"No benchmarks found in {data_dir}/{split}. "
                f"Make sure benchmark files exist."
            )

    def __len__(self) -> int:
        """Return number of benchmarks in dataset."""
        return len(self.benchmark_files)

    def __getitem__(self, idx: int) -> CircuitTensorData:
        """
        Load and return a single benchmark.

        Args:
            idx: Index of benchmark to load

        Returns:
            CircuitTensorData instance
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        benchmark_file = self.benchmark_files[idx]
        circuit_data = CircuitTensorData.load(str(benchmark_file))

        return circuit_data

    def get_by_name(self, name: str) -> CircuitTensorData:
        """
        Load benchmark by design name.

        Args:
            name: Design name (e.g., 'ariane133')

        Returns:
            CircuitTensorData instance

        Raises:
            ValueError: If benchmark name not found
        """
        if name not in self.benchmark_names_list:
            raise ValueError(
                f"Benchmark '{name}' not found. "
                f"Available: {', '.join(self.benchmark_names_list)}"
            )

        idx = self.benchmark_names_list.index(name)
        return self[idx]

    def get_benchmark_names(self) -> List[str]:
        """
        Get list of all benchmark names in dataset.

        Returns:
            List of benchmark names
        """
        return self.benchmark_names_list.copy()

    def __repr__(self) -> str:
        return (
            f"MacroPlacementDataset("
            f"split='{self.split}', "
            f"num_benchmarks={len(self)}, "
            f"data_dir='{self.data_dir}')"
        )


def load_benchmark(
    benchmark_name: str,
    data_dir: Optional[Union[str, Path]] = None,
    split: str = 'public'
) -> CircuitTensorData:
    """
    Convenience function to load a single benchmark.

    Args:
        benchmark_name: Name of benchmark (e.g., 'ariane133')
        data_dir: Path to benchmarks directory (default: auto-detect)
        split: Which split to search - 'public', 'hidden', or 'all'

    Returns:
        CircuitTensorData instance

    Example:
        >>> circuit_data = load_benchmark('ariane133')
        >>> print(circuit_data.num_macros)
        133
    """
    if data_dir is None:
        # Try to auto-detect data directory
        # Assume we're running from project root or scripts/
        possible_dirs = [
            Path.cwd() / 'benchmarks' / 'processed',
            Path.cwd().parent / 'benchmarks' / 'processed',
        ]

        for pdir in possible_dirs:
            if pdir.exists():
                data_dir = pdir
                break

        if data_dir is None:
            raise ValueError(
                "Could not auto-detect data directory. "
                "Please specify data_dir parameter."
            )

    data_dir = Path(data_dir)
    dataset = MacroPlacementDataset(data_dir, split=split)

    return dataset.get_by_name(benchmark_name)
