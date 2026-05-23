"""Phase 2 — placement state encoder (GNN + CNN).

Pure-PyTorch implementation; no torch-geometric / torch-scatter dependency
(both have CUDA-version-pinned binary wheels and would force a custom
Dockerfile because the judges run with --network none). Message passing
uses ``torch.Tensor.index_add_`` instead of scatter ops.

The encoder produces:
    per_node  [N, hidden_dim]            — one embedding per hard macro
    global    [2 * hidden_dim]           — graph-pool concat with canvas CNN

Components match the plan §2:
    PlacementGNN: 3-layer message-passing over the hard-macro netlist graph.
    CanvasCNN:    3-channel raster (occupancy, density, edge-endpoint
                  weights) → fixed-size vector.
    StateEncoder: glue.

Integration path (kept SEPARATE from this file so this is a drop-in):

    1. Train CostApproximator with encoder features. In train.py replace
       _features_for_move(state, m, d) with
           per_node, global_vec = encode_state(state, encoder)
           feats = torch.cat([per_node[m], global_vec, hand_features])
       Bump in_dim in CostApproximator(in_dim=hidden+2*hidden+FEATURE_DIM).

    2. Train ChainPolicy similarly: in chain_env.state_for_policy(), replace
       compute_{global,macro}_features with encoder outputs. Update the
       *_DIM constants in lkh_model.ChainPolicy at construction time.

    3. In placer.py, load the encoder alongside the cost approximator /
       policy. Cache encoder outputs PER CHAIN (compute at chain init, reuse
       across the chain's steps) to keep inference fast; the staleness is
       small since one chain only moves a handful of macros.

Smoke test:
    uv run python submissions/lkh/encoder.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ── Architecture ───────────────────────────────────────────────────────────

class GNNLayer(nn.Module):
    """One message-passing step: edge MLP -> sum-aggregate -> node MLP."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """h: [N, node_dim]; edge_index: [2, E] (long); edge_attr: [E, edge_dim]."""
        src = edge_index[0]
        dst = edge_index[1]
        # Edge messages computed from source node features + edge features.
        msgs = self.edge_mlp(torch.cat([h[src], edge_attr], dim=-1))
        # Sum-aggregate at destination.
        N = h.shape[0]
        agg = torch.zeros(N, msgs.shape[-1], dtype=h.dtype, device=h.device)
        agg.index_add_(0, dst, msgs)
        return self.node_mlp(torch.cat([h, agg], dim=-1))


class PlacementGNN(nn.Module):
    """3-layer message-passing GNN with residual connections.

    Edges are passed in both directions (the netlist graph is treated as
    undirected) — build_edge_index_and_attr() handles the duplication.
    """

    def __init__(self, node_dim: int = 8, edge_dim: int = 4,
                 hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GNNLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(node_features)
        for layer in self.layers:
            h = h + layer(h, edge_index, edge_attr)
        graph_vec = h.mean(dim=0)
        return h, graph_vec


class CanvasCNN(nn.Module):
    """3-channel canvas raster -> fixed-size vector (hidden_dim)."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # -> [B, hidden_dim]
        )

    def forward(self, canvas: torch.Tensor) -> torch.Tensor:
        """canvas: [C, H, W] or [B, C, H, W]. Returns [hidden_dim] or [B, hidden_dim]."""
        squeeze = False
        if canvas.dim() == 3:
            canvas = canvas.unsqueeze(0)
            squeeze = True
        out = self.net(canvas)
        return out.squeeze(0) if squeeze else out


class StateEncoder(nn.Module):
    """GNN + CNN joint encoder; returns (per_node, global)."""

    NODE_DIM = 8
    EDGE_DIM = 4
    CNN_CHANNELS = 3

    def __init__(self, hidden_dim: int = 128, num_gnn_layers: int = 3,
                 grid_size: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.gnn = PlacementGNN(self.NODE_DIM, self.EDGE_DIM, hidden_dim, num_gnn_layers)
        self.cnn = CanvasCNN(self.CNN_CHANNELS, hidden_dim)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, canvas: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        h, g = self.gnn(node_features, edge_index, edge_attr)
        c = self.cnn(canvas)
        return h, torch.cat([g, c], dim=-1)


# ── Feature builders (from PlacementState) ─────────────────────────────────

def build_node_features(state) -> torch.Tensor:
    """Per-hard-macro node features [N, 8]:
        w_norm, h_norm, area_norm, degree_norm, is_fixed,
        x_norm, y_norm, n_overlap_norm.
    """
    n = state.n
    feats = np.zeros((n, StateEncoder.NODE_DIM), dtype=np.float32)
    feats[:, 0] = state.sizes[:, 0] / max(state.cw, 1e-6)
    feats[:, 1] = state.sizes[:, 1] / max(state.ch, 1e-6)
    feats[:, 2] = (state.sizes[:, 0] * state.sizes[:, 1]) / max(state.cw * state.ch, 1e-6)
    degs = np.array([len(state.neighbors[i]) for i in range(n)], dtype=np.float32)
    feats[:, 3] = degs / max(n, 1)
    feats[:, 4] = (~state.movable).astype(np.float32)
    feats[:, 5] = state.pos[:, 0] / max(state.cw, 1e-6)
    feats[:, 6] = state.pos[:, 1] / max(state.ch, 1e-6)
    # Per-node overlap count via the pairwise AABB test (vectorized N×N).
    dx = np.abs(state.pos[:, 0:1] - state.pos[:, 0:1].T)
    dy = np.abs(state.pos[:, 1:2] - state.pos[:, 1:2].T)
    ov = (dx + 1e-6 < state.sep_x) & (dy + 1e-6 < state.sep_y)
    np.fill_diagonal(ov, False)
    feats[:, 7] = ov.sum(axis=1) / max(n, 1)
    return torch.from_numpy(feats)


def build_edge_index_and_attr(state) -> tuple[torch.Tensor, torch.Tensor]:
    """Edges as both directions. Returns (edge_index [2, 2E], edge_attr [2E, 4]).
    Attributes: weight, dx/cw, dy/ch, dist/canvas_diag.
    """
    E = len(state.edges)
    if E == 0:
        return (torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, StateEncoder.EDGE_DIM, dtype=torch.float32))
    edges = state.edges
    weights = state.edge_weights.astype(np.float32)
    pos = state.pos.astype(np.float32)
    canvas_diag = float((state.cw ** 2 + state.ch ** 2) ** 0.5)
    src_orig = edges[:, 0]
    dst_orig = edges[:, 1]
    src = np.concatenate([src_orig, dst_orig])
    dst = np.concatenate([dst_orig, src_orig])
    dx = pos[dst, 0] - pos[src, 0]
    dy = pos[dst, 1] - pos[src, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    w_dup = np.concatenate([weights, weights])
    attr = np.stack([
        w_dup,
        dx / state.cw,
        dy / state.ch,
        dist / canvas_diag,
    ], axis=1).astype(np.float32)
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0).astype(np.int64))
    edge_attr = torch.from_numpy(attr)
    return edge_index, edge_attr


def rasterize_canvas(state, grid_size: int = 128) -> torch.Tensor:
    """3-channel raster of the canvas [3, grid_size, grid_size]:
        channel 0: occupancy (1 in any cell covered by a hard macro)
        channel 1: density (sum of macro areas accumulated per cell)
        channel 2: edge-endpoint weights (sum over edges with endpoint in cell)
                   — congestion proxy. Plan's exact congestion model isn't
                   specified; this captures "many wires terminate here".
    """
    canvas = np.zeros((3, grid_size, grid_size), dtype=np.float32)
    cell_w = state.cw / grid_size
    cell_h = state.ch / grid_size

    # Per-macro occupancy + density.
    for i in range(state.n):
        cx, cy = state.pos[i]
        w, h = state.sizes[i]
        x0 = max(0, int((cx - w / 2) / cell_w))
        x1 = min(grid_size, int((cx + w / 2) / cell_w) + 1)
        y0 = max(0, int((cy - h / 2) / cell_h))
        y1 = min(grid_size, int((cy + h / 2) / cell_h) + 1)
        canvas[0, y0:y1, x0:x1] = 1.0
        canvas[1, y0:y1, x0:x1] += float(w * h)

    # Edge-endpoint scatter (vectorized).
    if len(state.edges) > 0:
        edges = state.edges
        weights = state.edge_weights.astype(np.float32)
        idx_all = np.concatenate([edges[:, 0], edges[:, 1]])
        w_all = np.concatenate([weights, weights])
        gx = np.clip((state.pos[idx_all, 0] / cell_w).astype(int), 0, grid_size - 1)
        gy = np.clip((state.pos[idx_all, 1] / cell_h).astype(int), 0, grid_size - 1)
        np.add.at(canvas[2], (gy, gx), w_all)

    return torch.from_numpy(canvas)


def encode_state(state, encoder: StateEncoder) -> tuple[torch.Tensor, torch.Tensor]:
    """End-to-end: build inputs and run the encoder. Returns (per_node, global)."""
    node_feats = build_node_features(state)
    edge_index, edge_attr = build_edge_index_and_attr(state)
    canvas = rasterize_canvas(state, grid_size=encoder.grid_size)
    with torch.no_grad():
        return encoder(node_feats, edge_index, edge_attr, canvas)


# ── Smoke test ─────────────────────────────────────────────────────────────

def _smoke_test():
    sys.path.insert(0, str(_HERE.parent.parent))

    spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
    placer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(placer)

    from macro_place.loader import load_benchmark_from_dir
    benchmark, _ = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    edges, weights = placer._hard_macro_edges(benchmark)
    state = placer.PlacementState(benchmark, edges, weights)

    print(f"=== Phase 2 encoder smoke test ===")
    print(f"  benchmark = {benchmark.name}")
    print(f"  num_hard_macros = {state.n}, num_edges = {len(state.edges)}")

    encoder = StateEncoder(hidden_dim=128, num_gnn_layers=3, grid_size=128)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  encoder params = {n_params:,}")

    # Feature build (and time it)
    t = time.time()
    nf = build_node_features(state)
    ei, ea = build_edge_index_and_attr(state)
    cv = rasterize_canvas(state, grid_size=128)
    feat_ms = (time.time() - t) * 1000
    print(f"  feature build: {feat_ms:.1f} ms")
    print(f"    node_features = {tuple(nf.shape)} dtype={nf.dtype}")
    print(f"    edge_index    = {tuple(ei.shape)} dtype={ei.dtype}")
    print(f"    edge_attr     = {tuple(ea.shape)} dtype={ea.dtype}")
    print(f"    canvas        = {tuple(cv.shape)} dtype={cv.dtype}")

    encoder.eval()

    # Warm-up + timed forward pass average.
    with torch.no_grad():
        _ = encoder(nf, ei, ea, cv)

    n_runs = 5
    t = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            per_node, global_vec = encoder(nf, ei, ea, cv)
    fwd_ms = (time.time() - t) * 1000 / n_runs
    print(f"  forward pass:  {fwd_ms:.1f} ms (avg of {n_runs})")
    print(f"    per_node = {tuple(per_node.shape)}")
    print(f"    global   = {tuple(global_vec.shape)}")

    # Sanity: gradients flow.
    encoder.train()
    nf2 = nf.detach().clone()
    nf2.requires_grad = False  # node feats are inputs, not parameters
    per_node, global_vec = encoder(nf, ei, ea, cv)
    loss = per_node.pow(2).mean() + global_vec.pow(2).mean()
    loss.backward()
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in encoder.parameters())
    print(f"  backward pass: gradients finite = {grad_ok}")


if __name__ == "__main__":
    _smoke_test()
