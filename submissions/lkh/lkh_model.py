"""Neural models for the LK-chain pipeline.

Phase 3 — CostApproximator: predicts Δproxy_cost from a (state, macro, move)
feature vector. Replaces compute_proxy_cost inside the chain.

Phase 4 — ChainPolicy: actor-critic over chain actions. At each step the
state is (global summary, current macro, candidate moves, chain progress);
the policy emits a categorical over (K candidates + STOP) and a state value.

Phase 2 (GNN+CNN encoder) is skipped — we use hand-crafted features built in
placer.py and chain_env.py over the same physical signals (HPWL Δ, local
density, overlap counts, neighbor attraction) so models 1 and 2 share a
consistent feature schema.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ── Phase 4: ChainPolicy ────────────────────────────────────────────────────

# Per-step inputs to the policy (kept small; matches chain_env feature schemas).
GLOBAL_DIM = 5    # hpwl_norm, overlap_norm, pos_x_var, pos_y_var, movable_frac
MACRO_DIM = 6     # w, h, deg, n_self_overlap, x, y (all normalized)
CAND_DIM = FEATURE_DIM
CHAIN_DIM = 3     # length_norm, hpwl_gain_norm, n_displaced_norm


class ChainPolicy(nn.Module):
    """Actor-critic for LK chain action selection.

    Per chain step: ``K`` candidate moves are scored together with a single
    STOP action. ``forward`` returns a length-(K+1) logit vector and a scalar
    state value, both differentiable.

    Task 1c (MaskRegulate): ``cand_dim`` accepts the augmented 17-dim
    candidate schema. Existing 16-dim policies still load — ``_load_chain_policy``
    reads ``cand_dim`` from the checkpoint and reconstructs the right shape.
    """

    def __init__(self, hidden: int = 64, cand_dim: int = CAND_DIM):
        super().__init__()
        self.cand_dim = int(cand_dim)
        gd, md, cd, chd = GLOBAL_DIM, MACRO_DIM, self.cand_dim, CHAIN_DIM
        self.move_head = nn.Sequential(
            nn.Linear(gd + md + cd + chd, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(gd + chd, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(gd + chd, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_feat: torch.Tensor, macro_feat: torch.Tensor,
                cand_feats: torch.Tensor, chain_feat: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        global_feat  : (GLOBAL_DIM,)
        macro_feat   : (MACRO_DIM,)
        cand_feats   : (K, CAND_DIM)
        chain_feat   : (CHAIN_DIM,)

        Returns (logits[K+1], value scalar).
        """
        K = cand_feats.shape[0]
        g = global_feat.unsqueeze(0).expand(K, -1)
        m = macro_feat.unsqueeze(0).expand(K, -1)
        c = chain_feat.unsqueeze(0).expand(K, -1)
        move_inp = torch.cat([g, m, cand_feats, c], dim=-1)
        move_logits = self.move_head(move_inp).squeeze(-1)            # (K,)

        sv_inp = torch.cat([global_feat, chain_feat], dim=-1)
        stop_logit = self.stop_head(sv_inp)                            # (1,)
        value = self.value_head(sv_inp).squeeze(-1)                    # scalar

        logits = torch.cat([move_logits, stop_logit], dim=-1)          # (K+1,)
        return logits, value
