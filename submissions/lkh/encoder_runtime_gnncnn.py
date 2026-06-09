"""Building block — GNN + CNN encoder runtime (Fix 2, optional secondary path).

The poster diagram shows a single GNN encoder; the primary runtime path
(``encoder_runtime.py``) matches that literally. This file is an optional
extra: it wraps the existing ``encoder.StateEncoder`` which combines a
GNN with a canvas CNN for spatial context. Enable with
``encoder_kind="gnncnn"`` at placer construction.

Output convention:
    per_node : [N, hidden_dim]                   — one per macro
    graph_vec: [2 * hidden_dim]                  — graph-pool concat with CNN

The cand embedding layout::

    [ per_node[macro_idx]  (hidden_dim)
    ; graph_vec            (2*hidden_dim)
    ; hand_feats[16]       (16) ]

So ``embed_dim = 3 * hidden_dim``. With hidden=128 that's 384, plus 16
hand-features = 400-dim input to the approximator.

API mirrors ``encoder_runtime.py`` so the placer can swap the import
based on a single flag.
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

_enc_spec = importlib.util.spec_from_file_location("lkh_encoder", str(_HERE / "encoder.py"))
_enc_mod = importlib.util.module_from_spec(_enc_spec)
_enc_spec.loader.exec_module(_enc_mod)

StateEncoder = _enc_mod.StateEncoder
build_node_features = _enc_mod.build_node_features
build_edge_index_and_attr = _enc_mod.build_edge_index_and_attr
rasterize_canvas = _enc_mod.rasterize_canvas


class GNNCNNEncoderRuntime(nn.Module):
    """Thin wrapper around the existing ``encoder.StateEncoder`` (GNN + CNN).

    Same API surface as ``GNNEncoderRuntime`` in encoder_runtime.py so the
    placer can pick one via flag.
    """

    KIND = "gnncnn"

    def __init__(self, hidden_dim: int = 128, num_gnn_layers: int = 3,
                 grid_size: int = 128):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_gnn_layers = int(num_gnn_layers)
        self.grid_size = int(grid_size)
        self.encoder = StateEncoder(
            hidden_dim=self.hidden_dim,
            num_gnn_layers=self.num_gnn_layers,
            grid_size=self.grid_size,
        )

    @property
    def embed_dim(self) -> int:
        """Dim of (per_node[m] concat graph_vec) before hand_feats.
        per_node is hidden_dim; graph_vec is 2*hidden_dim (graph_pool + CNN).
        So embed = hidden + 2*hidden = 3*hidden."""
        return 3 * self.hidden_dim

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, canvas: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(node_features, edge_index, edge_attr, canvas)


def build_encoder_inputs(state, grid_size: int = 128
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (node_feats, edge_index, edge_attr, canvas) for GNN+CNN."""
    node_feats = build_node_features(state)
    edge_index, edge_attr = build_edge_index_and_attr(state)
    canvas = rasterize_canvas(state, grid_size=grid_size)
    return node_feats, edge_index, edge_attr, canvas


def encode_state_gnncnn(state, encoder: GNNCNNEncoderRuntime, *,
                         with_grad: bool = False
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run encoder on ``state``. See encoder_runtime.encode_state_gnn for the
    sibling that handles GNN-only inputs."""
    node_feats, edge_index, edge_attr, canvas = build_encoder_inputs(
        state, grid_size=encoder.grid_size
    )
    if with_grad:
        return encoder(node_feats, edge_index, edge_attr, canvas)
    with torch.no_grad():
        return encoder(node_feats, edge_index, edge_attr, canvas)


def make_cand_embedding(per_node: torch.Tensor, graph_vec: torch.Tensor,
                         macro_idx: int, hand_feats: np.ndarray
                         ) -> np.ndarray:
    """``[per_node[m]; graph_vec; hand_feats]`` as numpy. Same contract as
    encoder_runtime.make_cand_embedding."""
    macro_emb = per_node[macro_idx].detach().numpy()
    global_emb = graph_vec.detach().numpy()
    return np.concatenate([macro_emb, global_emb, hand_feats.astype(np.float32)],
                          axis=0).astype(np.float32)


def macro_embedding(per_node: torch.Tensor, macro_idx: int) -> np.ndarray:
    return per_node[macro_idx].detach().numpy().astype(np.float32)


def global_embedding(graph_vec: torch.Tensor) -> np.ndarray:
    return graph_vec.detach().numpy().astype(np.float32)


def save_encoder(encoder: GNNCNNEncoderRuntime, path: Path, *,
                  trained_on: list[str] | None = None,
                  extra: dict | None = None) -> None:
    payload: dict = {
        "state_dict": encoder.state_dict(),
        "kind": GNNCNNEncoderRuntime.KIND,
        "hidden_dim": encoder.hidden_dim,
        "num_gnn_layers": encoder.num_gnn_layers,
        "grid_size": encoder.grid_size,
        "embed_dim": encoder.embed_dim,
        "trained_on": list(trained_on) if trained_on else [],
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_encoder(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if ckpt.get("kind") != GNNCNNEncoderRuntime.KIND:
        return None

    encoder = GNNCNNEncoderRuntime(
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        num_gnn_layers=int(ckpt.get("num_gnn_layers", 3)),
        grid_size=int(ckpt.get("grid_size", 128)),
    )
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.eval()
    return {
        "encoder": encoder,
        "embed_dim": encoder.embed_dim,
        "hidden_dim": encoder.hidden_dim,
        "num_gnn_layers": encoder.num_gnn_layers,
        "grid_size": encoder.grid_size,
        "kind": GNNCNNEncoderRuntime.KIND,
        "trained_on": ckpt.get("trained_on", []),
    }
