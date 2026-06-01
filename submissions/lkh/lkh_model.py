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

    Task 4b (MaskRegulate): optional encoder dims. When
    ``encoder_global_dim > 0`` and/or ``encoder_macro_dim > 0`` the policy
    accepts those embeddings via ``forward`` and concatenates them with the
    hand-crafted features. Defaults of 0 preserve the pre-Task-4 shape so
    existing checkpoints load unchanged.
    """

    def __init__(self, hidden: int = 64, cand_dim: int = CAND_DIM,
                 encoder_global_dim: int = 0,
                 encoder_macro_dim: int = 0):
        super().__init__()
        self.cand_dim = int(cand_dim)
        self.encoder_global_dim = int(encoder_global_dim)
        self.encoder_macro_dim = int(encoder_macro_dim)
        gd, md, cd, chd = GLOBAL_DIM, MACRO_DIM, self.cand_dim, CHAIN_DIM
        egd = self.encoder_global_dim
        emd = self.encoder_macro_dim
        # Move head sees (per-candidate): global + macro + cand + chain
        # + optional encoder dims (global + per-macro embedding).
        self.move_head = nn.Sequential(
            nn.Linear(gd + md + cd + chd + egd + emd, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Stop/value heads see only the state-level features (no per-cand).
        self.stop_head = nn.Sequential(
            nn.Linear(gd + chd + egd, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(gd + chd + egd, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_feat: torch.Tensor, macro_feat: torch.Tensor,
                cand_feats: torch.Tensor, chain_feat: torch.Tensor,
                encoder_global: torch.Tensor | None = None,
                encoder_macro: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        global_feat     : (GLOBAL_DIM,)
        macro_feat      : (MACRO_DIM,)
        cand_feats      : (K, CAND_DIM)
        chain_feat      : (CHAIN_DIM,)
        encoder_global  : (encoder_global_dim,) or None — only required if
                          encoder_global_dim > 0.
        encoder_macro   : (encoder_macro_dim,)  or None — only required if
                          encoder_macro_dim  > 0.

        Returns (logits[K+1], value scalar).
        """
        K = cand_feats.shape[0]
        g = global_feat.unsqueeze(0).expand(K, -1)
        m = macro_feat.unsqueeze(0).expand(K, -1)
        c = chain_feat.unsqueeze(0).expand(K, -1)
        move_parts = [g, m, cand_feats, c]
        sv_parts = [global_feat, chain_feat]
        # Task 4b: optional encoder dims. Tensors are required when the
        # corresponding *_dim is > 0; supplying them when *_dim == 0 is an
        # error (it would silently introduce ghost dims).
        if self.encoder_global_dim > 0:
            if encoder_global is None:
                raise ValueError(
                    "ChainPolicy was built with encoder_global_dim > 0 but "
                    "encoder_global was not passed to forward()"
                )
            move_parts.append(encoder_global.unsqueeze(0).expand(K, -1))
            sv_parts.append(encoder_global)
        if self.encoder_macro_dim > 0:
            if encoder_macro is None:
                raise ValueError(
                    "ChainPolicy was built with encoder_macro_dim > 0 but "
                    "encoder_macro was not passed to forward()"
                )
            move_parts.append(encoder_macro.unsqueeze(0).expand(K, -1))
        move_inp = torch.cat(move_parts, dim=-1)
        move_logits = self.move_head(move_inp).squeeze(-1)            # (K,)

        sv_inp = torch.cat(sv_parts, dim=-1)
        stop_logit = self.stop_head(sv_inp)                            # (1,)
        value = self.value_head(sv_inp).squeeze(-1)                    # scalar

        logits = torch.cat([move_logits, stop_logit], dim=-1)          # (K+1,)
        return logits, value
