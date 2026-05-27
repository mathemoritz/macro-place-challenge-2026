"""Phase 2 — placement state encoder (edge-centric GNN + CNN).

Architecture matches Google's CircuitTrainingModel (circuit_training/model/model_lib.py)
with three deviations noted inline:
  1. CNN canvas raster added (occupancy, density, congestion) as a complementary
     global view — not in Google's model.
  2. `is_node_placed` replaced by `is_movable` — in refinement all macros are
     already placed, so `is_node_placed` is always 1 and carries no signal.
  3. Pure-PyTorch; no TF dependency.

StateEncoder.forward produces three tensors:
    per_node   [n_hard, H]    — one embedding per hard macro
    gnn_global [5H or 7H]     — GNN-only global summary (without / with current_node)
    cnn_global [H]            — CNN canvas features (exposed separately for ablation)

encode_state() concatenates gnn_global + cnn_global → [6H or 8H] for callers
that want the full global vector without caring about the CNN split.

Components:
    GNNLayer:      One edge-centric message-passing step.
    PlacementGNN:  feature_encoder projection + 3 GNNLayers.
    CanvasCNN:     3-channel raster → fixed-size vector.
    StateEncoder:  metadata_encoder + GNN + CNN + self-attention readout.

Integration path (see ARCHITECTURE.md §2):
    encode_state(state, encoder, current_node=None)
        → (per_node [n_hard, H], global [6H or 8H])
    StateEncoder.forward returns (per_node, gnn_global, cnn_global) separately;
    encode_state concatenates them so callers see the same [6H/8H] shape as before.

Smoke test:
    uv run python submissions/lkh/encoder.py
"""

from __future__ import annotations

import importlib.util
import math
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
    """One edge-centric message-passing step.

    For every undirected edge (i, j):
      1. Form both directed edge representations:
           h_ij = concat(h_i, h_j, weight)
           h_ji = concat(h_j, h_i, weight)
      2. Average their edge-FC outputs:
           h_edge = (edge_fc(h_ij) + edge_fc(h_ji)) / 2
      3. Each node's new embedding = mean of all its incident edge embeddings
         (both as src and as dst). Skip connection adds the old embedding.

    Matches CircuitTrainingModel.gather_to_edges / scatter_to_nodes
    (model_lib.py:300-352) and the per-layer edge_fc (model_lib.py:139-148).
    """

    def __init__(self, hidden_dim: int, edge_fc_layers: int = 1):
        super().__init__()
        in_dim = 2 * hidden_dim + 1  # two node embeds + scalar weight
        layers: list[nn.Module] = []
        for _ in range(edge_fc_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.edge_fc = nn.Sequential(*layers)

    def forward(
        self,
        h: torch.Tensor,  # [N, H]
        edge_index: torch.Tensor,  # [2, E] long
        edge_attr: torch.Tensor,  # [E, 1] scalar weight
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_updated [N, H], h_edges [E, H])."""
        E = edge_index.shape[1]
        N, H = h.shape
        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]

        if E == 0:
            return h, torch.zeros(0, H, dtype=h.dtype, device=h.device)

        # Build directed edge representations and average (bidirectional)
        h_ij = torch.cat([h[src], h[dst], edge_attr], dim=-1)  # [E, 2H+1]
        h_ji = torch.cat([h[dst], h[src], edge_attr], dim=-1)  # [E, 2H+1]
        h_edges = (self.edge_fc(h_ij) + self.edge_fc(h_ji)) * 0.5  # [E, H]

        # Scatter edge embeddings to both endpoints — mean aggregation
        agg = torch.zeros(N, H, dtype=h.dtype, device=h.device)
        cnt = torch.zeros(N, H, dtype=h.dtype, device=h.device)
        ones = torch.ones_like(h_edges)
        agg.index_add_(0, src, h_edges)
        agg.index_add_(0, dst, h_edges)
        cnt.index_add_(0, src, ones)
        cnt.index_add_(0, dst, ones)
        h_new = agg / (cnt + 1e-6)

        return h + h_new, h_edges  # skip connection


class PlacementGNN(nn.Module):
    """Edge-centric GNN with initial feature projection.

    Matches CircuitTrainingModel._feature_encoder + edge-centric GCN layers
    (model_lib.py:128-136, 535-549).
    """

    def __init__(
        self,
        node_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        edge_fc_layers: int = 1,
    ):
        super().__init__()
        # Project raw node features to hidden_dim before first GCN layer.
        # Matches model_lib.py:128-136 (_feature_encoder).
        self.feature_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            [GNNLayer(hidden_dim, edge_fc_layers) for _ in range(num_layers)]
        )

    def forward(
        self,
        node_features: torch.Tensor,  # [N, node_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor,  # [E, 1]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_nodes [N, H], h_edges [E, H]) from the final layer."""
        h = self.feature_encoder(node_features)
        H = h.shape[1]
        E = edge_index.shape[1]
        h_edges = torch.zeros(E, H, dtype=h.dtype, device=h.device)
        for layer in self.layers:
            h, h_edges = layer(h, edge_index, edge_attr)
        return h, h_edges


class CanvasCNN(nn.Module):
    """3-channel canvas raster -> fixed-size vector (hidden_dim).

    Unchanged from original encoder.py.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
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
    """Edge-centric GNN + metadata encoder + CNN + self-attention readout.

    Matches Google's CircuitTrainingModel (model_lib.py) with the CNN added
    as a complementary canvas-level global view.

    Output (per_node, gnn_global, cnn_global):
        gnn_global components (matching model_lib.py:551-584):
            h_metadata      [H]   — netlist-level circuit properties
            h_edges_mean    [H]   — mean edge embedding (global wiring summary)
            h_edges_var     [H]   — variance (spread of wire costs)
            h_edges_max     [H]   — max (worst-case wires)
            h_edges_min     [H]   — min
            h_attended      [H]   — current node's attention over all nodes (optional)
            h_current_node  [H]   — current node's own embedding (optional)
        cnn_global: [H]  — canvas density/congestion (returned separately for ablation)
    """

    NODE_DIM = 8
    METADATA_DIM = 12
    CNN_CHANNELS = 3

    def __init__(
        self,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        edge_fc_layers: int = 1,
        grid_size: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size

        self.gnn = PlacementGNN(self.NODE_DIM, hidden_dim, num_gnn_layers, edge_fc_layers)
        self.cnn = CanvasCNN(self.CNN_CHANNELS, hidden_dim)

        # Matches model_lib.py:119-127 (_metadata_encoder): Dense(12→H) + ReLU
        self.metadata_encoder = nn.Sequential(
            nn.Linear(self.METADATA_DIM, hidden_dim),
            nn.ReLU(),
        )

        # Luong dot-product attention — matches model_lib.py:156-172
        self.attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim)
        self.attn_value = nn.Linear(hidden_dim, hidden_dim)

    def _self_attention(
        self,
        h_current: torch.Tensor,  # [H]
        h_nodes: torch.Tensor,  # [N, H]
    ) -> torch.Tensor:  # [H]
        """Current node attends over all nodes (Luong-style)."""
        q = self.attn_query(h_current)  # [H]
        k = self.attn_key(h_nodes)  # [N, H]
        v = self.attn_value(h_nodes)  # [N, H]
        scores = torch.softmax(q @ k.T / math.sqrt(self.hidden_dim), dim=-1)  # [N]
        return scores @ v  # [H]

    def forward(
        self,
        node_features: torch.Tensor,  # [N_total, 8]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor,  # [E, 1]
        canvas: torch.Tensor,  # [3, grid, grid]
        metadata: torch.Tensor,  # [12]
        n_hard: int,
        current_node: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            n_hard: number of hard macro nodes (first n_hard rows of node_features).
                    Anchor nodes follow after; per_node output is sliced to n_hard.
            current_node: index into node_features for the macro currently being
                    considered. When provided, adds h_attended and h_current_node
                    to gnn_global.

        Returns:
            per_node:   [n_hard, H]
            gnn_global: [5H] without current_node, [7H] with current_node
                        = concat(h_meta, h_edges_mean, h_edges_var, h_edges_max,
                                 h_edges_min [, h_attended, h_current_node])
            cnn_global: [H]  — CNN canvas features (exposed separately for ablation)
        """
        h_nodes, h_edges = self.gnn(node_features, edge_index, edge_attr)
        h_meta = self.metadata_encoder(metadata)
        h_cnn = self.cnn(canvas)

        # Edge statistics — matches model_lib.py:554-565
        if h_edges.shape[0] > 0:
            h_edges_mean = h_edges.mean(dim=0)
            h_edges_var = h_edges.var(dim=0)
            h_edges_max = h_edges.max(dim=0).values
            h_edges_min = h_edges.min(dim=0).values
        else:
            H = self.hidden_dim
            h_edges_mean = torch.zeros(H, dtype=h_nodes.dtype, device=h_nodes.device)
            h_edges_var = torch.zeros(H, dtype=h_nodes.dtype, device=h_nodes.device)
            h_edges_max = torch.zeros(H, dtype=h_nodes.dtype, device=h_nodes.device)
            h_edges_min = torch.zeros(H, dtype=h_nodes.dtype, device=h_nodes.device)

        gnn_parts = [h_meta, h_edges_mean, h_edges_var, h_edges_max, h_edges_min]

        if current_node is not None:
            h_curr = h_nodes[current_node]
            h_attended = self._self_attention(h_curr, h_nodes)
            gnn_parts.extend([h_attended, h_curr])

        return h_nodes[:n_hard], torch.cat(gnn_parts, dim=-1), h_cnn


# ── Feature builders (from PlacementState) ─────────────────────────────────


def _build_anchors(state) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate hf_anchors to unique anchor node positions.

    Returns:
        unique_pos [M, 2]: unique (x, y) positions for anchor nodes.
        inverse    [E_hf]: maps each hf edge → its anchor node index (0-based).
    """
    if len(state.hf_anchors) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.int64)
    # Round to 4 decimal places before deduplication (float precision)
    unique_pos, inverse = np.unique(np.round(state.hf_anchors, 4), axis=0, return_inverse=True)
    return unique_pos.astype(np.float32), inverse.astype(np.int64)


def build_node_features(state, anchors: np.ndarray) -> torch.Tensor:
    """Per-node features [N_hard + N_anchors, 8]:
        x_norm, y_norm, w_norm, h_norm,
        is_hard_macro, is_soft_macro, is_port_cluster, is_movable.

    Deviates from Google in one place: `is_movable` replaces `is_node_placed`.
    In refinement placement all macros already have positions, so is_node_placed
    is always 1 and carries no signal. is_movable distinguishes which hard macros
    the chain can move (1) from fixed anchor nodes (0).

    Normalization: positions and sizes divided by half_perimeter = cw + ch,
    matching Google's _normalize_locations_by_canvas and
    _normalize_macro_size_by_canvas (observation_extractor.py).
    """
    hp = float(state.cw + state.ch)
    n = state.n
    M = len(anchors)
    feats = np.zeros((n + M, 8), dtype=np.float32)

    # Hard macro nodes
    feats[:n, 0] = state.pos[:, 0] / hp  # x_norm
    feats[:n, 1] = state.pos[:, 1] / hp  # y_norm
    feats[:n, 2] = state.sizes[:, 0] / hp  # w_norm
    feats[:n, 3] = state.sizes[:, 1] / hp  # h_norm
    feats[:n, 4] = 1.0  # is_hard_macro
    feats[:n, 5] = 0.0  # is_soft_macro
    feats[:n, 6] = 0.0  # is_port_cluster
    feats[:n, 7] = state.movable.astype(np.float32)  # is_movable

    # Anchor nodes (port clusters / soft macro contributions)
    if M > 0:
        feats[n:, 0] = anchors[:, 0] / hp  # x_norm
        feats[n:, 1] = anchors[:, 1] / hp  # y_norm
        # w=0, h=0, is_hard=0, is_soft=0, is_port=1, is_movable=0
        feats[n:, 6] = 1.0  # is_port_cluster

    return torch.from_numpy(feats)


def build_edge_index_and_attr(
    state, anchor_inverse: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Undirected edges stored once each. Returns (edge_index [2, E], edge_attr [E, 1]).

    Includes:
      - Hard-hard edges (state.edges / state.edge_weights)
      - Hard-to-anchor edges (state.hf_macros → anchor node N_hard + inverse[k])

    Weights are normalized by their mean (matching observation_extractor.py
    _normalize_adj_matrix), so the scale is consistent across benchmarks.

    Edge attributes are 1-dim scalar weights only (no dx/dy/dist). The
    edge-centric GCN captures positional relationships by concatenating node
    embeddings (which contain x, y) at each edge — so positional edge features
    would duplicate information already in the nodes.
    """
    n = state.n
    src_parts, dst_parts, weight_parts = [], [], []

    if len(state.edges) > 0:
        src_parts.append(state.edges[:, 0].astype(np.int64))
        dst_parts.append(state.edges[:, 1].astype(np.int64))
        weight_parts.append(state.edge_weights.astype(np.float64))

    if len(state.hf_macros) > 0:
        anchor_node_idx = (n + anchor_inverse).astype(np.int64)
        src_parts.append(state.hf_macros.astype(np.int64))
        dst_parts.append(anchor_node_idx)
        weight_parts.append(state.hf_weights.astype(np.float64))

    if not src_parts:
        return (
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, 1, dtype=torch.float32),
        )

    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    weights = np.concatenate(weight_parts).astype(np.float32)

    # Normalize by mean weight (matching _normalize_adj_matrix)
    mean_w = float(weights.mean()) + 1e-6
    weights = weights / mean_w

    edge_index = torch.from_numpy(np.stack([src, dst], axis=0))
    edge_attr = torch.from_numpy(weights[:, None])  # [E, 1]
    return edge_index, edge_attr


def build_metadata_features(state, anchor_inverse: np.ndarray) -> torch.Tensor:
    """12-dim netlist metadata vector matching observation_config.NETLIST_METADATA.

    Features unavailable from PlacementState (routing resources, grid dims)
    default to 0.0. These can be populated from the benchmark object later
    if the data is exposed.
    """
    MAX_NODES = 5000
    MAX_EDGES = 70000
    hp = float(state.cw + state.ch)
    n_anchors = int(anchor_inverse.max()) + 1 if len(anchor_inverse) > 0 else 0
    n_total_edges = len(state.edges) + len(state.hf_macros)

    meta = np.array(
        [
            n_total_edges / MAX_EDGES,  # normalized_num_edges
            state.n / MAX_NODES,  # normalized_num_hard_macros
            0.0,  # normalized_num_soft_macros (unavailable)
            n_anchors / MAX_NODES,  # normalized_num_port_clusters
            0.0,  # total_horizontal_routes_k (unavailable)
            0.0,  # total_vertical_routes_k
            0.0,  # total_macro_horizontal_routes_k
            0.0,  # total_macro_vertical_routes_k
            0.0,  # grid_cols (unavailable)
            0.0,  # grid_rows
            state.cw / hp,  # canvas_width (normalized)
            state.ch / hp,  # canvas_height (normalized)
        ],
        dtype=np.float32,
    )

    return torch.from_numpy(meta)


def rasterize_canvas(state, grid_size: int = 128) -> torch.Tensor:
    """3-channel raster of the canvas [3, grid_size, grid_size]:
        channel 0: occupancy (1 in any cell covered by a hard macro)
        channel 1: density (sum of macro areas accumulated per cell)
        channel 2: edge-endpoint weights (sum over edges with endpoint in cell)

    Unchanged from original encoder.py.
    """
    canvas = np.zeros((3, grid_size, grid_size), dtype=np.float32)
    cell_w = state.cw / grid_size
    cell_h = state.ch / grid_size

    for i in range(state.n):
        cx, cy = state.pos[i]
        w, h = state.sizes[i]
        x0 = max(0, int((cx - w / 2) / cell_w))
        x1 = min(grid_size, int((cx + w / 2) / cell_w) + 1)
        y0 = max(0, int((cy - h / 2) / cell_h))
        y1 = min(grid_size, int((cy + h / 2) / cell_h) + 1)
        canvas[0, y0:y1, x0:x1] = 1.0
        canvas[1, y0:y1, x0:x1] += float(w * h)

    if len(state.edges) > 0:
        edges = state.edges
        weights = state.edge_weights.astype(np.float32)
        idx_all = np.concatenate([edges[:, 0], edges[:, 1]])
        w_all = np.concatenate([weights, weights])
        gx = np.clip((state.pos[idx_all, 0] / cell_w).astype(int), 0, grid_size - 1)
        gy = np.clip((state.pos[idx_all, 1] / cell_h).astype(int), 0, grid_size - 1)
        np.add.at(canvas[2], (gy, gx), w_all)

    return torch.from_numpy(canvas)


def encode_state(
    state,
    encoder: StateEncoder,
    current_node: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """End-to-end: build all inputs and run the encoder.

    Args:
        state: PlacementState — must have been constructed with a full HpwlEdges
               bundle (i.e. state.hf_anchors / state.hf_macros are populated).
        encoder: StateEncoder instance.
        current_node: index into the hard macro list (0 … state.n-1) for the
               macro currently being considered. Enables self-attention readout.

    Returns:
        per_node [state.n, H], global [6H or 8H]
    """
    anchors, anchor_inverse = _build_anchors(state)
    node_feats = build_node_features(state, anchors)
    edge_index, edge_attr = build_edge_index_and_attr(state, anchor_inverse)
    metadata = build_metadata_features(state, anchor_inverse)
    canvas = rasterize_canvas(state, grid_size=encoder.grid_size)
    with torch.no_grad():
        per_node, gnn_global, cnn_global = encoder(
            node_feats,
            edge_index,
            edge_attr,
            canvas,
            metadata,
            n_hard=state.n,
            current_node=current_node,
        )
    return per_node, torch.cat([gnn_global, cnn_global], dim=-1)


# ── Smoke test ─────────────────────────────────────────────────────────────


def _smoke_test():
    # _HERE = submissions/lkh/model; project root is _HERE.parent.parent
    sys.path.insert(0, str(_HERE.parent.parent))

    spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE.parent / "placer.py"))
    placer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(placer)

    from macro_place.loader import load_benchmark_from_dir

    benchmark, _ = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    hpwl_edges = placer._hard_macro_edges(benchmark)
    state = placer.PlacementState(benchmark, hpwl_edges)

    print("=== Phase 2 encoder smoke test (edge-centric GNN) ===")
    print(f"  benchmark       = {benchmark.name}")
    print(f"  n_hard_macros   = {state.n}")
    print(f"  hh_edges        = {len(state.edges)}")
    print(f"  hf_edges        = {len(state.hf_macros)}")

    anchors, anchor_inverse = _build_anchors(state)
    print(f"  anchor_nodes    = {len(anchors)}  (unique hf anchors)")

    encoder = StateEncoder(hidden_dim=128, num_gnn_layers=3, edge_fc_layers=1, grid_size=128)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  encoder params  = {n_params:,}")

    t = time.time()
    nf = build_node_features(state, anchors)
    ei, ea = build_edge_index_and_attr(state, anchor_inverse)
    md = build_metadata_features(state, anchor_inverse)
    cv = rasterize_canvas(state, 128)
    feat_ms = (time.time() - t) * 1000
    print(f"\n  feature build   : {feat_ms:.1f} ms")
    print(f"    node_features : {tuple(nf.shape)}  dtype={nf.dtype}")
    print(f"    edge_index    : {tuple(ei.shape)}  dtype={ei.dtype}")
    print(f"    edge_attr     : {tuple(ea.shape)}  dtype={ea.dtype}")
    print(f"    metadata      : {tuple(md.shape)}  dtype={md.dtype}")
    print(f"    canvas        : {tuple(cv.shape)}  dtype={cv.dtype}")

    encoder.eval()
    with torch.no_grad():
        _ = encoder(nf, ei, ea, cv, md, n_hard=state.n)  # warm-up

    n_runs = 5
    t = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            per_node, gnn_global, cnn_global = encoder(nf, ei, ea, cv, md, n_hard=state.n)
    fwd_ms = (time.time() - t) * 1000 / n_runs
    print(f"\n  forward (no current_node): {fwd_ms:.1f} ms avg over {n_runs}")
    print(f"    per_node   : {tuple(per_node.shape)}   expected ({state.n}, 128)")
    print(f"    gnn_global : {tuple(gnn_global.shape)}  expected (640,)  [5×128]")
    print(f"    cnn_global : {tuple(cnn_global.shape)}  expected (128,)  [1×128]")

    with torch.no_grad():
        per_node_c, gnn_global_c, cnn_global_c = encoder(
            nf, ei, ea, cv, md, n_hard=state.n, current_node=0
        )
    print(f"\n  forward (current_node=0):")
    print(f"    per_node   : {tuple(per_node_c.shape)}   expected ({state.n}, 128)")
    print(f"    gnn_global : {tuple(gnn_global_c.shape)}  expected (896,)  [7×128]")
    print(f"    cnn_global : {tuple(cnn_global_c.shape)}  expected (128,)  [1×128]")

    # encode_state still returns the concatenated form for backward compat
    p2, g2 = encode_state(state, encoder, current_node=None)
    assert g2.shape == (768,), f"encode_state global shape (no current): {g2.shape}"
    p3, g3 = encode_state(state, encoder, current_node=0)
    assert g3.shape == (1024,), f"encode_state global shape (with current): {g3.shape}"

    # Gradient flow
    encoder.train()
    per_node2, gnn2, cnn2 = encoder(nf, ei, ea, cv, md, n_hard=state.n, current_node=0)
    loss = per_node2.pow(2).mean() + gnn2.pow(2).mean() + cnn2.pow(2).mean()
    loss.backward()
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    print(f"\n  backward pass   : gradients finite = {grad_ok}")

    assert per_node.shape == (state.n, 128), f"per_node shape mismatch: {per_node.shape}"
    assert gnn_global.shape == (640,), f"gnn_global shape mismatch: {gnn_global.shape}"
    assert cnn_global.shape == (128,), f"cnn_global shape mismatch: {cnn_global.shape}"
    assert gnn_global_c.shape == (896,), f"gnn_global shape (current) mismatch: {gnn_global_c.shape}"
    assert grad_ok, "gradient flow broken"
    print("\n  All assertions passed.")


if __name__ == "__main__":
    _smoke_test()
