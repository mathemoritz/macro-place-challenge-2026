"""GNN-only encoder runtime.

Wraps ``encoder.PlacementGNN`` so the cost approximator and the policy
can consume a single GNN-only encoder. The combined GNN+CNN variant
lives in ``encoder_runtime_gnncnn.py`` and exposes the same API.

Output convention:
    per_node : [N, hidden_dim]      — one embedding per hard macro
    graph_vec: [hidden_dim]         — mean-pooled global summary

The embedding the cost approximator sees for a single move is::

    [ per_node[macro_idx]  (hidden_dim)
    ; graph_vec            (hidden_dim)
    ; hand_feats[16]       (16) ]

So ``embed_dim = 2 * hidden_dim``. With hidden=128 and the 16-dim
hand-feature vector, the approximator's input dim is 272.

Performance note: the encoder forward is ~5-30 ms per state on CPU. The
placer caches the encoder output once per chain and reuses ``per_node``
and ``graph_vec`` across all candidate scores within the chain.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Lazy import of encoder.py — avoid macro_place package init.
_enc_spec = importlib.util.spec_from_file_location("lkh_encoder", str(_HERE / "encoder.py"))
_enc_mod = importlib.util.module_from_spec(_enc_spec)
_enc_spec.loader.exec_module(_enc_mod)

PlacementGNN = _enc_mod.PlacementGNN
build_node_features = _enc_mod.build_node_features
build_edge_index_and_attr = _enc_mod.build_edge_index_and_attr


class GNNEncoderRuntime(nn.Module):
    """Thin wrapper around ``PlacementGNN`` that exposes the encoder API the
    cost approximator and the policy share.

    Keeps a stable interface even if we later swap the underlying GNN —
    callers go through this module, not through ``encoder.PlacementGNN``
    directly.
    """

    KIND = "gnn"

    def __init__(self, hidden_dim: int = 128, num_gnn_layers: int = 3):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_gnn_layers = int(num_gnn_layers)
        self.gnn = PlacementGNN(
            node_dim=8, edge_dim=4,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_gnn_layers,
        )

    @property
    def embed_dim(self) -> int:
        """Dim of the (per_node concat graph_vec) vector before hand_feats."""
        return 2 * self.hidden_dim

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """node_features [N, 8], edge_index [2, 2E], edge_attr [2E, 4] →
        (per_node [N, hidden_dim], graph_vec [hidden_dim])."""
        return self.gnn(node_features, edge_index, edge_attr)


def build_encoder_inputs(state) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (node_feats, edge_index, edge_attr) for the GNN.

    Edges only depend on the netlist (constant for a benchmark); node
    features depend on positions and update per state.
    """
    node_feats = build_node_features(state)
    edge_index, edge_attr = build_edge_index_and_attr(state)
    return node_feats, edge_index, edge_attr


def encode_state_gnn(state, encoder: GNNEncoderRuntime, *,
                      with_grad: bool = False
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run encoder on ``state``. Caches nothing — caller manages caching.

    ``with_grad=False`` (default) is the inference path: ``no_grad`` saves
    memory and time. ``with_grad=True`` is used during joint encoder +
    approximator regression training (see ``train_encoder_joint.py``).
    """
    node_feats, edge_index, edge_attr = build_encoder_inputs(state)
    if with_grad:
        return encoder(node_feats, edge_index, edge_attr)
    with torch.no_grad():
        return encoder(node_feats, edge_index, edge_attr)


def make_cand_embedding(per_node: torch.Tensor, graph_vec: torch.Tensor,
                         macro_idx: int, hand_feats: np.ndarray
                         ) -> np.ndarray:
    """Build the per-candidate input vector for the cost approximator.

    Layout: ``[per_node[macro_idx]; graph_vec; hand_feats]``. The
    approximator sees this as one flat vector of length
    ``2*hidden_dim + 16``.

    Returned as np.ndarray for consistency with ``_features_for_move``
    (which the rest of the pipeline currently expects as numpy).
    """
    macro_emb = per_node[macro_idx].detach().numpy()
    global_emb = graph_vec.detach().numpy()
    return np.concatenate([macro_emb, global_emb, hand_feats.astype(np.float32)],
                          axis=0).astype(np.float32)


def macro_embedding(per_node: torch.Tensor, macro_idx: int) -> np.ndarray:
    """Just ``per_node[macro_idx]`` as numpy. Used by the policy and seed head."""
    return per_node[macro_idx].detach().numpy().astype(np.float32)


def global_embedding(graph_vec: torch.Tensor) -> np.ndarray:
    """Just ``graph_vec`` as numpy. Used by the policy and seed head."""
    return graph_vec.detach().numpy().astype(np.float32)


def save_encoder(encoder: GNNEncoderRuntime, path: Path, *,
                  trained_on: list[str] | None = None,
                  extra: dict | None = None) -> None:
    """Save encoder weights + architecture metadata so the placer can rebuild."""
    payload: dict = {
        "state_dict": encoder.state_dict(),
        "kind": GNNEncoderRuntime.KIND,
        "hidden_dim": encoder.hidden_dim,
        "num_gnn_layers": encoder.num_gnn_layers,
        "embed_dim": encoder.embed_dim,
        "trained_on": list(trained_on) if trained_on else [],
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_encoder(path: Path) -> dict | None:
    """Load a GNN-only encoder checkpoint. Returns a bundle dict or None.

    Bundle keys: ``encoder`` (GNNEncoderRuntime, eval mode), ``embed_dim``,
    ``hidden_dim``, ``num_gnn_layers``, ``kind``, ``trained_on``.

    Returns ``None`` if the file doesn't exist OR if the checkpoint kind
    isn't ``"gnn"`` — the GNN+CNN variant has its own loader.
    """
    path = Path(path)
    if not path.exists():
        return None
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if ckpt.get("kind") not in (None, GNNEncoderRuntime.KIND):
        return None  # wrong variant — let the GNN+CNN loader try

    encoder = GNNEncoderRuntime(
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        num_gnn_layers=int(ckpt.get("num_gnn_layers", 3)),
    )
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.eval()
    return {
        "encoder": encoder,
        "embed_dim": encoder.embed_dim,
        "hidden_dim": encoder.hidden_dim,
        "num_gnn_layers": encoder.num_gnn_layers,
        "kind": GNNEncoderRuntime.KIND,
        "trained_on": ckpt.get("trained_on", []),
    }
