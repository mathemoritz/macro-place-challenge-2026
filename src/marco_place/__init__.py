"""
Macro Placement Challenge - Beat Kahng's benchmarks for $20K

This package provides tools for macro placement optimization:
- Tensor-based circuit representation
- Evaluation metrics (proxy cost, HPWL, density, congestion)
- Validation engine for placement legality
- Evaluation harness for submissions
"""

__version__ = "0.1.0"

from marco_place.data.tensor_schema import CircuitTensorData

__all__ = ["CircuitTensorData", "__version__"]
