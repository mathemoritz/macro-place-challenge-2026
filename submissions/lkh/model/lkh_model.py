"""Neural models for the LK-chain pipeline.

Phase 3 — CostApproximator: predicts Δproxy_cost from a (state, macro, move)
feature vector. Replaces compute_proxy_cost inside the chain.

Phase 4 — ChainPolicy: actor-critic over chain actions. At each step the
state is (global summary, current macro, candidate moves, chain progress);
the policy emits a categorical over (K candidates + STOP) and a state value.

Phase 2.5 — ProxyCostPredictor: GNN+CNN encoder trained via supervised
regression to predict (wl_cost, density_cost, congestion_cost) for a full
placement. Delta-proxy for a candidate move is encode(after) - encode(before)
with only the moved macro's x,y updated in node_features. Replaces
CostApproximator once the encoder achieves sufficient Pearson r on held-out
benchmarks.

Phase 3 — CostApproximator: predicts delta-proxy_cost from a (state, macro,
move) feature vector. Replaced by ProxyCostPredictor after Phase 2.5.
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── lazy-load encoder (lives in model/ subdirectory) ───────────────────────
_HERE = Path(__file__).resolve().parent
_encoder_spec = importlib.util.spec_from_file_location(
    "lkh_encoder", str(_HERE / "encoder.py")
)
_encoder_mod = importlib.util.module_from_spec(_encoder_spec)
_encoder_spec.loader.exec_module(_encoder_mod)
StateEncoder = _encoder_mod.StateEncoder

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


# ── Phase 2.5: ProxyCostPredictor ──────────────────────────────────────────


class ProxyCostPredictor(nn.Module):
    """GNN+CNN encoder + 3-output head predicting (wl, density, congestion).

    Training (train_encoder.py): supervised MSE on per-term proxy costs
    collected from random placements.  The head is discarded after training;
    ``self.encoder`` can be extracted and plugged into ChainPolicy inputs.

    Inference (placer.py run_greedy): for each candidate move, clone
    node_features, update node_features[macro_idx, 0:2] with the new (x/hp,
    y/hp), re-run forward, subtract before-costs from after-costs, then
    combine with benchmark-specific alpha weights (stored in the checkpoint)
    to obtain delta-proxy.

    Args:
        hidden_dim: encoder hidden size H.
        num_gnn_layers: number of edge-centric GCN layers.
        edge_fc_layers: MLP depth inside each GCN layer.
        grid_size: CNN canvas raster resolution (square).
        use_cnn: if False, the CNN output is excluded from the head input
            (head_in = 5H instead of 6H). Lets you ablate the canvas branch
            without changing the encoder; the CNN still runs but its output is
            dropped before the head. Set at construction time — changing this
            after instantiation would require re-initialising the head weights.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        edge_fc_layers: int = 1,
        grid_size: int = 128,
        use_cnn: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_cnn = use_cnn

        self.encoder = StateEncoder(hidden_dim, num_gnn_layers, edge_fc_layers, grid_size)

        # Without current_node: gnn_global is 7H; cnn_global is H.
        head_in = (8 if use_cnn else 7) * hidden_dim
        self.proxy_head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [wl_pred, density_pred, congestion_pred]
        )

    def forward(
        self,
        node_features: torch.Tensor,  # [N_total, 8]
        edge_index: torch.Tensor,     # [2, E]
        edge_attr: torch.Tensor,      # [E, 1]
        canvas: torch.Tensor,         # [3, grid, grid]
        metadata: torch.Tensor,       # [12]
        n_hard: int,
    ) -> torch.Tensor:
        """Returns [3] tensor: (wl_pred, density_pred, congestion_pred).

        Predictions are in normalized space (per-benchmark/per-term z-score);
        denormalize using the stats stored in the checkpoint before combining
        with alpha weights.
        """
        _, gnn_global, cnn_global = self.encoder(
            node_features, edge_index, edge_attr, canvas, metadata,
            n_hard=n_hard, current_node=None,
        )
        head_input = (
            torch.cat([gnn_global, cnn_global], dim=-1) if self.use_cnn else gnn_global
        )
        return self.proxy_head(head_input)


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
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        gd, md, cd, chd = GLOBAL_DIM, MACRO_DIM, CAND_DIM, CHAIN_DIM
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


# GNN_GLOBAL_FACTOR * H = gnn_global dim (7H without current_node attention)
GNN_GLOBAL_FACTOR = 7


class ChainPolicyWithGNN(nn.Module):
    """ChainPolicy augmented with frozen GNN embeddings.

    Adds two projection heads:
      gnn_global_proj:  Linear(7H → proj_dim) + LayerNorm + ReLU
      macro_gnn_proj:   Linear(H  → proj_dim) + LayerNorm + ReLU

    Projected embeddings are concatenated to global_feat and macro_feat
    before the MLP heads. When both GNN tensors are None, behavior is
    identical to ChainPolicy (projections are zero-padded).
    """

    def __init__(self, hidden: int = 64, gnn_H: int = 32, proj_dim: int = 16):
        super().__init__()
        self.gnn_H = gnn_H
        self.proj_dim = proj_dim

        self.gnn_global_proj = nn.Sequential(
            nn.Linear(GNN_GLOBAL_FACTOR * gnn_H, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.macro_gnn_proj = nn.Sequential(
            nn.Linear(gnn_H, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )

        gd  = GLOBAL_DIM + proj_dim   # 5  + 16 = 21
        md  = MACRO_DIM  + proj_dim   # 6  + 16 = 22
        cd  = CAND_DIM                # 16
        chd = CHAIN_DIM               # 3

        self.move_head = nn.Sequential(
            nn.Linear(gd + md + cd + chd, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(gd + chd, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(gd + chd, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        global_feat:   torch.Tensor,
        macro_feat:    torch.Tensor,
        cand_feats:    torch.Tensor,
        chain_feat:    torch.Tensor,
        gnn_global:    torch.Tensor | None = None,
        macro_gnn_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev = global_feat.device
        g_gnn = self.gnn_global_proj(gnn_global) if gnn_global is not None \
                else torch.zeros(self.proj_dim, dtype=global_feat.dtype, device=dev)
        m_gnn = self.macro_gnn_proj(macro_gnn_emb) if macro_gnn_emb is not None \
                else torch.zeros(self.proj_dim, dtype=macro_feat.dtype, device=dev)

        g_aug = torch.cat([global_feat, g_gnn])
        m_aug = torch.cat([macro_feat,  m_gnn])

        K = cand_feats.shape[0]
        move_inp = torch.cat([
            g_aug.unsqueeze(0).expand(K, -1),
            m_aug.unsqueeze(0).expand(K, -1),
            cand_feats,
            chain_feat.unsqueeze(0).expand(K, -1),
        ], dim=-1)
        move_logits = self.move_head(move_inp).squeeze(-1)

        sv_inp = torch.cat([g_aug, chain_feat])
        stop_logit = self.stop_head(sv_inp)
        value      = self.value_head(sv_inp).squeeze(-1)
        return torch.cat([move_logits, stop_logit]), value
