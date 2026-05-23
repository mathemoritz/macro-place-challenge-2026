"""Phase 3 — fast cost approximator (Model 1) for the LK chain.

The plan's Phase 3 builds on Phase 2's GNN+CNN encoder. We skip Phase 2 here
(it requires torch_geometric / torch_scatter which aren't in pyproject.toml)
and instead feed a small MLP hand-crafted features over the SAME physical
quantities the encoder would capture — HPWL Δ, local density, overlap count
change, neighbor attraction, geometry. Exit criterion is unchanged: Pearson
correlation with exact compute_proxy_cost Δ on a held-out set of moves.

Features are produced in placer.py (`_features_for_move`) so the inference
path and the data-collection path use the same code.
"""

import torch
import torch.nn as nn

# Must match the length of the vector returned by placer._features_for_move.
FEATURE_DIM = 16


class CostApproximator(nn.Module):
    """Predicts (normalized) exact Δproxy_cost from a 16-dim feature vector."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
