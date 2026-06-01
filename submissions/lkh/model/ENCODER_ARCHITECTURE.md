# Encoder Architecture — Deep Dive

Source: [encoder.py](../encoder.py)  
Reference: [external/circuit_training/circuit_training/model/model_lib.py](../../../external/circuit_training/circuit_training/model/model_lib.py)

---

## Overview

The encoder converts a placement state (macro positions, netlist connectivity, canvas geometry) into two tensors:

| Output | Shape | Meaning |
|---|---|---|
| `per_node` | `[n_hard, H]` | One embedding per hard macro |
| `global` | `[6·H]` or `[8·H]` | Circuit-wide summary vector |

where `H = hidden_dim = 128`.

There are four sub-networks, assembled in [StateEncoder](../encoder.py#L178):

```
PlacementState
      │
      ├── build_node_features  ──► PlacementGNN ──────────► per_node [n_hard, H]
      │         (node feats)          │                           │
      │                               └── h_edges [E, H]          │
      │                                        │                   │
      ├── build_metadata_features ──► metadata_encoder [H]        │
      │                                                            │
      ├── rasterize_canvas ─────────► CanvasCNN [H]               │
      │                                                            │
      └── (current_node optional) ──► self_attention [H] ─────────┘
                                                        └── concat → global [6H or 8H]
```

---

## Dimension Comparison: Ours vs Google

This is the most important difference between the two implementations.

| Hyperparameter | Our implementation | Google (`CircuitTrainingModel`) |
|---|---|---|
| Core hidden dim | **128** | **8** (`gcn_node_dim`) |
| GNN layers | 3 | 3 |
| Edge FC layers per GNN step | 1 | 1 |
| Node input features | 8 | 8 |
| Metadata input features | 12 | 12 (`NETLIST_METADATA`) |
| Global vector components | 8 (or 10) × H | 7 × H (always with current_node) |

Our `hidden_dim = 128` is **16× larger** than Google's `gcn_node_dim = 8`. Every projection — feature encoder, edge FC, attention Q/K/V, metadata encoder — maps into this 128-dim space. Google's model is deliberately tiny because it was designed to run inside a TF-Agents training loop on thousands of TPU rollouts; the small hidden dim trades expressivity for training throughput. Our model runs offline (no live RL), so the 128-dim space gives it far more capacity to represent placement quality.

---

## Component 1 — Node Feature Construction

**Code:** [encoder.py:306–342](../encoder.py#L306-L342)

Each node in the graph gets an 8-dimensional feature vector:

```
[x_norm, y_norm, w_norm, h_norm, is_hard_macro, is_soft_macro, is_port_cluster, is_movable]
```

Positions and sizes are normalized by the half-perimeter (`cw + ch`) of the canvas, matching Google's `_normalize_locations_by_canvas` and `_normalize_macro_size_by_canvas` in `observation_extractor.py`. This makes the features scale-invariant across benchmarks of different physical sizes.

The graph contains two node types:
- **Hard macro nodes** (indices `0…n-1`): real placeable macros. `is_hard_macro=1`, `is_movable` reflects whether the chain can currently move this macro.
- **Anchor nodes** (indices `n…n+M-1`): deduplicated positions of high-fanout (HF) net endpoints — port clusters and soft macro contributions. `is_port_cluster=1`, `is_movable=0`, `w=h=0`.

**Deviation from Google:** The 8th feature is `is_movable` instead of `is_node_placed`. In Google's formulation, `is_node_placed` starts at 0 and flips to 1 as macros are sequentially placed during an RL episode. In our refinement setting, all macros already have positions at the start of every call, so `is_node_placed` would be 1 everywhere and carry zero information. `is_movable` instead encodes which hard macros the LK-chain is permitted to move on this pass — a meaningful binary signal that varies per-macro.

---

## Component 2 — Edge Construction

**Code:** [encoder.py:345–392](../encoder.py#L345-L392)

The graph has undirected edges of two kinds:

1. **Hard–hard edges** (`state.edges`, `state.edge_weights`): wires between pairs of hard macros. Each edge has a scalar weight proportional to connectivity strength (e.g. net fanout or half-perimeter contribution).
2. **Hard–anchor edges** (`state.hf_macros`, `state.hf_weights`): high-fanout nets that connect a hard macro to a fixed point in the canvas. The anchor positions are deduplicated via [`_build_anchors`](../encoder.py#L290-L303) — if 50 nets share the same anchor point, they collapse to one anchor node with multiple edges, rather than 50 identical nodes.

All weights are **normalized by their mean** before being passed to the GNN, matching Google's `_normalize_adj_matrix`. This prevents the GNN from seeing absolute wire counts that differ by orders of magnitude across benchmarks.

Edge attributes are a 1-dim scalar (weight only). Positional information (dx, dy) is not added to edge features because the node embeddings already encode `x` and `y` — the edge-centric GCN sees position through the node embeddings it aggregates, so adding positional edge features would duplicate signal already present.

Google's implementation pads all edge tensors to a fixed `max_num_edges` with zeros and uses a weight-mask to suppress the padding. Our implementation uses dynamic-sized sparse tensors — no padding required since we aren't running inside a TF static-graph loop.

---

## Component 3 — PlacementGNN

**Code:** [encoder.py:109–147](../encoder.py#L109-L147)  
**Google equivalent:** [`model_lib.py:128–136`](../../../external/circuit_training/circuit_training/model/model_lib.py#L128-L136) + [`model_lib.py:535–549`](../../../external/circuit_training/circuit_training/model/model_lib.py#L535-L549)

### 3a — Feature Encoder (initial projection)

```python
self.feature_encoder = nn.Sequential(
    nn.Linear(node_dim=8, hidden_dim=128),
    nn.ReLU(),
)
```

The raw 8-dim node features are projected into the 128-dim hidden space before any message passing. This is a learned embedding that gives the GNN a richer starting representation than raw normalized scalars. Google's `_feature_encoder` does the same thing but projects into 8 dims.

### 3b — GNNLayer (edge-centric message passing)

**Code:** [encoder.py:51–106](../encoder.py#L51-L106)  
**Google equivalent:** [`model_lib.py:266–352`](../../../external/circuit_training/circuit_training/model/model_lib.py#L266-L352)

Each GNN layer performs one round of edge-centric message passing. "Edge-centric" means the model first builds edge representations, then aggregates them into nodes — rather than the typical approach of directly aggregating neighbor node embeddings.

**Step 1 — Gather to edges:**
For each undirected edge (i, j), form two directed representations:
```
h_ij = concat(h_i, h_j, weight)   # [2H+1]
h_ji = concat(h_j, h_i, weight)   # [2H+1]
```
Both orderings are computed to preserve symmetry. The scalar edge weight is appended so the GNN sees how strongly the two macros are connected.

**Step 2 — Edge FC:**
Each directed edge embedding is passed through a small MLP (`edge_fc`: Linear(2H+1 → H) + ReLU), then the two directions are averaged:
```
h_edge = (edge_fc(h_ij) + edge_fc(h_ji)) / 2     # [H]
```
This is the new edge representation for this layer. It summarizes the relationship between macros i and j in context of their current embeddings.

**Step 3 — Scatter to nodes (mean aggregation):**
Each node's new embedding is the mean of all its incident edge embeddings (both as source and destination):
```
h_i_new = mean over all edges (i,*) or (*,i) of h_edge
```
This aggregation is implemented with `index_add_` for both `src` and `dst` endpoints, followed by division by the total count — matching Google's `scatter_to_nodes` / `_scatter_count`.

**Step 4 — Skip connection:**
```
h = h + h_new
```
The residual connection lets gradients flow through many layers without vanishing. Google's code does the same at [model_lib.py:549](../../../external/circuit_training/circuit_training/model/model_lib.py#L549).

**Why edge-centric?** The reward in circuit placement (wirelength, congestion) depends on wire properties, not just node positions. By building explicit edge representations, the GNN can reason about wire quality directly. An alternative is to apply GCN on the "line graph" (where wire–wire adjacency is the connectivity), but the line graph adjacency matrix grows quadratically and becomes intractable.

**Dimension note:** Our `edge_fc` maps `2×128+1 = 257 → 128`. Google's maps `2×8+1 = 17 → 8`. The larger hidden dim lets our edge MLP capture richer relationships between the pair of macros on each wire.

The GNN runs for **3 layers** in both implementations. Three layers means each node's embedding can incorporate information from nodes up to 3 hops away through the wire graph.

---

## Component 4 — Metadata Encoder

**Code:** [encoder.py:213–217](../encoder.py#L213-L217), [encoder.py:395–423](../encoder.py#L395-L423)  
**Google equivalent:** [`model_lib.py:119–127`](../../../external/circuit_training/circuit_training/model/model_lib.py#L119-L127)

```python
self.metadata_encoder = nn.Sequential(
    nn.Linear(12, hidden_dim=128),
    nn.ReLU(),
)
```

A 12-dim vector of circuit-level statistics is projected to `[H]`. These features describe the netlist as a whole rather than individual macros:

| Index | Feature | Value in our impl |
|---|---|---|
| 0 | normalized_num_edges | `n_edges / 70000` |
| 1 | normalized_num_hard_macros | `n / 5000` |
| 2 | normalized_num_soft_macros | `0.0` (unavailable) |
| 3 | normalized_num_port_clusters | `n_anchors / 5000` |
| 4–7 | routing resource counts | `0.0` (unavailable) |
| 8–9 | grid dimensions | `0.0` (unavailable) |
| 10 | canvas_width_norm | `cw / (cw+ch)` |
| 11 | canvas_height_norm | `ch / (cw+ch)` |

Several features are set to 0 because the `PlacementState` object doesn't expose routing grid information. This is a known gap — populating these from the benchmark object would give the model better awareness of routing congestion constraints. Google's implementation has all 12 values populated from the full chip netlist.

The metadata embedding is always included in the global vector. It gives the model a "prior" about the scale and complexity of the circuit before looking at individual macro positions.

---

## Component 5 — CanvasCNN (our addition)

**Code:** [encoder.py:150–175](../encoder.py#L150-L175)  
**Google equivalent:** _Not present._

```python
self.net = nn.Sequential(
    Conv2d(3, 32, 3, padding=1),  ReLU(),   MaxPool2d(2),   # → H/2 × W/2
    Conv2d(32, 64, 3, padding=1), ReLU(),   MaxPool2d(2),   # → H/4 × W/4
    Conv2d(64, 128, 3, padding=1), ReLU(),
    AdaptiveAvgPool2d(1),                                    # → 1×1
    Flatten(),                                               # → [128]
)
```

The canvas is rasterized into a `[3, grid_size, grid_size]` image with three channels:
- **Channel 0 (occupancy):** 1 in every cell covered by a hard macro — shows where macros are packed.
- **Channel 1 (density):** accumulated macro area per cell — highlights heavily congested regions.
- **Channel 2 (edge weight):** sum of edge weights whose endpoints land in each cell — shows where wiring hotspots are.

The CNN processes this raster and produces a single `[H=128]` vector via global average pooling. This gives the encoder a **spatial global view** that the GNN cannot easily capture — the GNN operates on topology (which macros are connected), while the CNN sees the actual spatial arrangement and density.

**Rasterization code:** [encoder.py:426–457](../encoder.py#L426-L457). Macro footprints are painted cell-by-cell; edge weight accumulation uses `np.add.at` for vectorized scatter.

Google's model has no CNN branch. It was designed for sequential RL placement where spatial feedback comes through the policy's own grid-logit output head (a deconv CNN that produces 128×128 logit maps). In our refinement setting we don't produce grid logits — the CNN canvas branch fills the gap by giving the encoder spatial awareness.

---

## Component 6 — Self-Attention Readout

**Code:** [encoder.py:224–236](../encoder.py#L224-L236)  
**Google equivalent:** [`model_lib.py:354–376`](../../../external/circuit_training/circuit_training/model/model_lib.py#L354-L376)

```python
q = attn_query(h_current)      # [H]
k = attn_key(h_nodes)          # [N, H]
v = attn_value(h_nodes)        # [N, H]
scores = softmax(q @ k.T / sqrt(H))   # [N]
h_attended = scores @ v        # [H]
```

This is **Luong-style (dot-product) attention** [Luong et al., 2015], applied when a `current_node` is specified. The current macro queries all node embeddings, producing attention weights that reflect how relevant each other macro is to the current one. The output is a weighted sum of all node value-embeddings — a context vector summarizing the "neighborhood of interest" around the current macro.

**Why this matters:** The GNN gives each macro a local embedding from its direct wiring neighbors. Self-attention with the current macro as the query lets the model additionally ask "which globally distant macros are most relevant right now?" — useful for macros whose quality depends on far-away constraints.

Both implementations use the same three-linear-layer (Q, K, V) structure. Ours scales the dot products by `sqrt(H)` to stabilize softmax gradients for large `H`; Google's `tf.keras.layers.Attention` does the same internally.

When `current_node=None`, this block is skipped entirely and the global vector is 6×H instead of 8×H. Google's model always requires a current node.

---

## Component 7 — Global Vector Assembly

**Code:** [encoder.py:264–284](../encoder.py#L264-L284)  
**Google equivalent:** [`model_lib.py:551–584`](../../../external/circuit_training/circuit_training/model/model_lib.py#L551-L584)

After the GNN runs, node embeddings and edge embeddings from the last layer are pooled into the global vector:

| Component | Intuition |
|---|---|
| `h_nodes_mean` | Mean hard-macro embedding — aggregate spatial distribution |
| `h_nodes_max` | Max hard-macro embedding — captures extreme positions |
| `h_edges_mean` | Average wire cost across all nets — overall wiring quality |
| `h_edges_var` | Spread of wire costs — uniformity vs hotspots |
| `h_edges_max` | Worst-case wire — most critical net |
| `h_edges_min` | Best-case wire — baseline |

Google's model always includes a `current_node`, which contributes `h_attended` (attention-weighted mean of **all** node embeddings) and `h_current_node` (the focal macro's embedding) — giving it two node-level components in every forward pass. Our `ProxyCostPredictor` runs without a focal macro, so we substitute unconditional mean/max pooling over the hard macro embeddings, which plays the same role: providing a summary of the node embedding space to the head.

Google's model includes the same four edge statistics (controlled by `include_min_max_var=True` at [model_lib.py:84](../../../external/circuit_training/circuit_training/model/model_lib.py#L84)) for a total of 7×8=56 dims. Our global vector is 7×128=896 (without current_node) or 9×128=1152 (with current_node); `encode_state` appends `h_cnn` to give 8×128=1024 or 10×128=1280.

**Key difference from prior version:** The global vector previously omitted any node-level summary, leaving the head unable to predict density cost (which requires knowing where all macros are, not just connected pairs). The node pooling fixes this.

---

## What Google's Model Does That Ours Doesn't

Google's `CircuitTrainingModel` includes two heads that are absent from our encoder:

**Policy head** ([model_lib.py:194–264](../../../external/circuit_training/circuit_training/model/model_lib.py#L194-L264)): A deconvolutional CNN that upsamples the global vector `h` into a 128×128 grid of placement logits. The architecture is:
```
Dense(8×8×32) → ReLU → Reshape(8,8,32) →
Conv2DTranspose(16, 3, stride=2) →  # 16×16
Conv2DTranspose(8, 3, stride=2)  →  # 32×32
Conv2DTranspose(4, 3, stride=2)  →  # 64×64
Conv2DTranspose(2, 3, stride=2)  →  # 128×128
Conv2DTranspose(1, 3, stride=1)  →  # 128×128×1
Flatten                              # 16384 logits
```
This is the RL policy output — it selects a grid cell to place the current macro. We don't need this because our pipeline uses LK-chain moves, not grid-cell selection.

**Value head** ([model_lib.py:174–183](../../../external/circuit_training/circuit_training/model/model_lib.py#L174-L183)):
```
Dense(32) → ReLU → Dense(8) → ReLU → Dense(1)
```
Predicts the expected cumulative reward from the current state. Our `CostApproximator` in `lkh_model.py` plays a similar role but is structurally separate and takes hand-crafted move features rather than the full GNN embedding.

---

## Summary of Deviations from Google

| # | Deviation | Location | Reason |
|---|---|---|---|
| 1 | `hidden_dim = 128` instead of `gcn_node_dim = 8` | All projection layers | Capacity: we run offline, not live RL |
| 2 | `is_movable` instead of `is_node_placed` | [encoder.py:333](../encoder.py#L333) | All macros are pre-placed in refinement; `is_node_placed` is always 1 |
| 3 | CanvasCNN branch added | [encoder.py:150–175](../encoder.py#L150-L175) | Spatial density/congestion signal that the GNN topology can't see |
| 4 | `current_node` optional | [encoder.py:279–283](../encoder.py#L279-L283) | Encoder used for global state summaries without a focal macro |
| 5 | No policy head, no value head | — | Not doing grid-cell RL; heads live in `lkh_model.py` separately |
| 6 | Pure PyTorch, no TF/gin/TPU logic | Entire file | Framework choice |
| 7 | Dynamic-size sparse tensors (no padding) | [encoder.py:80–106](../encoder.py#L80-L106) | Not inside a TF static graph; padding would waste compute |
