# LKH Placer — Architecture Walkthrough

A bottom-up tour of how the code thinks. Each section answers three
questions: **why does this exist?**, **how does it work?**, and **where does
it live in the codebase?** If you want the operational guide instead, see
`README.md`.

## Table of contents

1. [The problem we're solving](#1-the-problem-were-solving)
2. [The big picture](#2-the-big-picture)
3. [`PlacementState` — the geometry layer](#3-placementstate--the-geometry-layer)
4. [HPWL surrogate — the cheap cost function](#4-hpwl-surrogate--the-cheap-cost-function)
5. [`LKChain` — the cascade move pattern](#5-lkchain--the-cascade-move-pattern)
6. [The commit gate — lex `(overlap_pairs, hpwl)`](#6-the-commit-gate--lex-overlap_pairs-hpwl)
7. [Legalization — spiral search to zero overlaps](#7-legalization--spiral-search-to-zero-overlaps)
8. [`CostApproximator` — MLP cost model](#8-costapproximator--mlp-cost-model)
9. [`ChainPolicy` — PPO actor-critic](#9-chainpolicy--ppo-actor-critic)
10. [Iterative training loop](#10-iterative-training-loop)
11. [Inference (`LKHPlacer.place`)](#11-inference-lkhplacerplace)
12. [Ablation](#12-ablation)
13. [State encoder (GNN + CNN)](#13-state-encoder-gnn--cnn)
14. [Feature schema reference](#14-feature-schema-reference)
15. [The Modal architecture](#15-the-modal-architecture)
16. [Design trade-offs FAQ](#16-design-trade-offs-faq)

---

## 1. The problem we're solving

**Macro placement** = decide where ~hundreds of fixed-size rectangular
blocks (SRAMs, etc.) go on a chip, minimizing wirelength, density, and
routing congestion, subject to a hard "no overlap" constraint.

The competition gives us an `initial.plc` (a placement from Cadence Innovus)
and asks us to improve its **proxy cost** (a weighted sum of HPWL,
top-10% density, top-5% congestion) without introducing overlaps among
hard macros.

The hard part is that **moving one macro to fix overlap X creates overlap Y
somewhere else**. Greedy "fix-the-worst-overlap" doesn't terminate.

### The LK insight

Lin-Kernighan chains generalize a 2-opt move into a **cascade**:

```
move macro A      (creates overlap with B)
  └─ move macro B  (resolves A's overlap, creates overlap with C)
     └─ move macro C  (resolves B's overlap, clean landing)
```

Each individual step might *worsen* the cost locally, but the full cascade
can be a net improvement. This is the mechanism that lets us escape local
minima without throwing away progress.

The plan layers two learned models on top of this base mechanism so that
chains pick better moves and know when to stop.

---

## 2. The big picture

```
┌────────────────────────────────────────────────────────────────────┐
│                       LKHPlacer.place(benchmark)                   │
│                                                                    │
│   load_benchmark_from_dir  ─►  PlacementState (numpy)              │
│                                                                    │
│   while time_left:                                                 │
│       seed_macro ← weighted pick (high HPWL contribution)          │
│       LKChain.run(state, seed_macro)                               │
│           ├── candidate_positions                                  │
│           ├── score with policy / approximator / surrogate         │
│           ├── apply best, cascade to displaced macro               │
│           └── commit if (overlap_pairs, hpwl) improves             │
│       if stagnation > threshold: perturb 5 macros                  │
│                                                                    │
│   state.pos ← best_pos                                             │
│   if overlap_pairs > 0: legalize(state)   ← guarantee 0 overlaps   │
│                                                                    │
│   return placement (hard from LK, soft from initial.plc)           │
└────────────────────────────────────────────────────────────────────┘
```

Training fills in the two `score with ...` boxes:

- `CostApproximator` replaces the surrogate with a learned
  Δproxy_cost predictor.
- `ChainPolicy` replaces argmin-over-candidates with a learned
  policy that also has a STOP action.

The placer is **always** the same shape; learned models just plug into the
scorer. If no checkpoints exist, the HPWL surrogate runs and the placer is
still functional.

---

## 3. `PlacementState` — the geometry layer

**Why?** Everything below needs fast move/undo, overlap counting, and HPWL
computation. We can't afford to call into the TILOS `PlacementCost`
(`compute_proxy_cost`, ~1 s/call) inside an inner loop. `PlacementState` is
a pure-numpy mirror of the macro geometry, optimized for these queries.

**Where?** `placer.py`, class `PlacementState`.

### What it holds

```python
self.n         # num hard macros (= benchmark.num_hard_macros)
self.pos       # [N, 2] float64 — centers
self.sizes     # [N, 2] float64 — (w, h)
self.half_w    # [N]   float64 — sizes[:, 0] / 2  (precomputed)
self.half_h    # [N]   float64 — sizes[:, 1] / 2
self.movable   # [N]   bool    — True if not fixed
self.cw, self.ch    # canvas dimensions (float)

# Pairwise pre-sums so overlap tests are O(N) per macro, not O(N^2) on the fly:
self.sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0   # [N, N]
self.sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0   # [N, N]

# HPWL surrogate graph
self.edges          # [E, 2] long — undirected hard-macro pairs
self.edge_weights   # [E]    float — see "HPWL surrogate" below
self.neighbors      # list[list[int]] — adjacency list for fast lookups
```

### Key operations

| Method | What it does | Cost |
|---|---|---|
| `clamp(idx, x, y)` | Project a position into the canvas bounds for macro `idx` | O(1) |
| `overlapping_with(idx)` | Vectorized AABB test: which macros overlap macro `idx` | O(N) |
| `has_overlap(idx)` | Faster "any overlap with macro `idx`?" | O(N) |
| `overlap_pairs()` | Full pairwise overlap count (matches the evaluator) | O(N²), but as a single numpy op (fast) |
| `hpwl()` | Sum over edges of (w · (|Δx| + |Δy|)) | O(E) |
| `candidate_positions(idx)` | Generate move candidates for macro `idx` | O(K), K ≈ 16 |

### Overlap-gap invariant

The cached `overlap_pairs` count uses `gap=1e-6` so its zero-point matches
the evaluator's strict `<` semantics up to float precision. Any larger
gap would treat close-but-not-touching pairs as overlapping, and the
commit gate would lex-compare on an inflated count; chains could then
"improve" by decreasing the inflated count while increasing the real
count. `placer.py` calls this invariant out in a long comment.

---

## 4. HPWL surrogate — the cheap cost function

**Why?** `compute_proxy_cost` from TILOS takes ~1–100 s per call (scales
with benchmark size). A single LK chain might score 8 candidates × 8 steps
= 64 cost evaluations. Using exact cost in the chain would make each
chain take 1 minute to several hours. Unworkable.

We need a function that:
1. Computes in *milliseconds*
2. Tracks the things that drive proxy cost
3. Lets us undo moves trivially

**Half-perimeter wirelength (HPWL)** is the standard answer. For each net,
the HPWL is the perimeter of the bounding box of its terminals (divided
by 2). A net with two macros has HPWL = |Δx| + |Δy|. We approximate
multi-pin nets by their **clique expansion** with weight `1 / (k - 1)`
where `k` is the net size — this preserves the "every pair pulls" intuition.

```
benchmark.net_pin_nodes
   ↓ (extract hard-macro participation)
edges: list of (i, j) hard-macro pairs
weights: 1 / (net_size - 1) for each net containing both i and j
   ↓
state.edges, state.edge_weights
   ↓
state.hpwl() = Σ_e weight_e · ( |x_i - x_j| + |y_i - y_j| )
```

**Where?** `placer.py::_hard_macro_edges` (graph extraction) and
`PlacementState.hpwl` (the actual sum).

### What HPWL misses

HPWL ≈ wirelength. The proxy cost is `1.0 × wirelength + 0.5 × density +
0.5 × congestion`. So HPWL is only ~50–66% of the signal. The chain can
reduce HPWL while increasing density (by pushing macros into dense
regions). That's why we need:

- The **commit gate** (next section) to enforce that overlaps don't grow,
  so density doesn't explode.
- The **CostApproximator** to learn a closer approximation to
  the *real* Δproxy_cost.

---

## 5. `LKChain` — the cascade move pattern

**Why?** Bridges between the placement state and whatever scorer is
deciding moves. Owns the chain semantics (cascade, commit/revert).

**Where?** `placer.py`, class `LKChain`.

### Episode

```
1. Snapshot state.pos
2. current = seed_macro
3. for step in 0..max_chain_length:
       a. cands = state.candidate_positions(current)
       b. score each candidate (policy / approximator / surrogate)
       c. apply best candidate → state mutates
       d. displaced = movable macros now overlapping current
       e. if not displaced or step == max-1: break
       f. current = closest displaced macro   ← cascade
4. Commit if (overlap_pairs, hpwl) lex-improved vs snapshot; else revert
```

### Candidate generation

`PlacementState.candidate_positions(idx)` returns up to ~16 positions:

1. **Connected centroid** — mean of `state.pos[neighbors[idx]]`. Where
   HPWL would pull this macro.
2. **Grid jitter** — 12 positions at `±0.5·max(w,h)` step in a small ring
   around the current position. Captures local refinement.
3. **Random canvas jumps** — 4 uniformly random in-bounds positions.
   Captures long-distance moves (escape local minima).

Each is `clamp`-ed to the canvas bounds so any candidate is at least
geometrically valid.

### Cascade selection

After applying the chosen move, the macro likely overlaps one or more
others. The chain follows the **nearest** displaced macro (smallest
Manhattan distance from `current`). This biases toward a tight cascade
that quickly resolves itself, rather than wandering far across the chip.

### Two code paths

```python
LKChain.run(max_length)
├── self.policy_bundle is not None → run_with_policy()
│       (ChainEnv-driven, policy picks action including STOP)
└── else → run_greedy()
        (argmin scorer + cascade)
```

Both end with the same commit gate; the only difference is **how the next
move is chosen at each step**.

---

## 6. The commit gate — lex `(overlap_pairs, hpwl)`

The most important design decision. Every chain ends with this check:

```python
better = (
    end_overlap_pairs < start_overlap_pairs
    or (end_overlap_pairs == start_overlap_pairs
        and end_hpwl < start_hpwl)
)
```

### Why this exact form

A chain produces a candidate state. We want:

1. **Never increase overlaps.** That would defeat the whole point —
   overlaps must end up at 0.
2. **Subject to that, decrease wirelength.** Cheap improvements.

The lex order achieves both: overlap_pairs is the primary key, HPWL is the
tiebreaker. The chain *can* worsen HPWL temporarily mid-cascade (in the
scoring), but the **net** chain has to land on lex-better state, or it
gets reverted.

### Why not just check "overlaps ≤ start AND hpwl ≤ start"?

That's actually equivalent at the commit gate level, but lex ordering
extends cleanly to the **best-pos tracker** in `LKHPlacer.place`:

```python
best_key = (state.overlap_pairs(), state.hpwl())
# ...
if cur_key < best_key:    # Python tuple comparison is lex
    best_key = cur_key
    best_pos = state.pos.copy()
```

Lex on tuples is one line of code and means the best-ever placement
strictly dominates every alternative on this combined criterion.

### Where the count comes from

`PlacementState.overlap_pairs(gap=1e-6)` does a vectorized N×N AABB test.
The `gap=1e-6` is the strict-overlap variant that matches the evaluator's
`<` (not `<=`) semantics. See section 3 for the bug we hit when we used
`gap=0.05`.

---

## 7. Legalization — spiral search to zero overlaps

**Why?** The chain commit gate can drive overlaps **down**, but in
practice it rarely reaches 0 on `ibm01-18`. On `ibm07` for example, chains
typically leave 10–15 residual overlap pairs. A submission with any
overlaps is disqualified. So after all chains, we need a **legalization**
step that snaps the remaining overlapping macros into legal spots.

**Where?** `placer.py::_legalize`.

### Algorithm

```
1. Sort movable macros by area, largest first
2. For each macro in order:
   - If fixed: mark as placed, continue
   - If current position is non-colliding with already-placed: keep, mark placed
   - Else: spiral search outward
       for ring r in 1..max_search_radius:
           for each cell on the perimeter of the r×r ring at step = 0.25·max(w,h):
               candidate = clamp(current + (dx·step, dy·step))
               if no collision with already-placed: track if best so far
           if found a valid candidate in this ring: break
       keep the candidate with smallest displacement from original
```

### Intuition

- Largest first → biggest macros get their preferred spots; small macros
  can squeeze into gaps.
- Step = 0.25 · max(w, h) → fine enough to find a slot without skipping,
  coarse enough to bound the search.
- Ring search bounded at 150 rings × step = 37.5 macro widths away
  before giving up.
- Output: every movable macro either keeps its position (if already
  legal) or jumps to the **nearest** legal spot. Fixed macros are anchors.

### Trade-off

Legalization tends to *increase* HPWL slightly: a macro that was happily
overlapping in a tight cluster gets pushed to the cluster's edge. The
proxy cost can go up modestly after legalization, but the competition
rewards validity, so this is the right trade.

### Why not "always legalize"?

We do — unconditionally call `_legalize` after the chain loop. If
`overlap_pairs() == 0` already, it's a no-op (every macro's current
position passes the collision check). The branch is `if overlap_pairs > 0`
just for the log message.

---

## 8. `CostApproximator` — MLP cost model

**Why?** The HPWL surrogate captures wirelength but misses density and
congestion. A learned model that takes (state, macro, move) features and
predicts the **actual Δproxy_cost** would let the chain make better
decisions.

**Where?** `lkh_model.py::CostApproximator` (the model), `train.py` (the
trainer).

### The model

```python
class CostApproximator(nn.Module):
    def __init__(self, in_dim=16, hidden=64):
        self.net = nn.Sequential(
            Linear(16, 64), ReLU,
            Linear(64, 64), ReLU,
            Linear(64, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)
```

Tiny — about 10k parameters. Justified by:
- We aren't using a GNN/CNN (those would imply ~300k+ params for a
  similar-quality encoder).
- The 16 input features (see [§14](#14-feature-schema-reference))
  already include rich, hand-crafted signals: HPWL Δ, overlap count
  change, local density, neighbor attraction Δ, macro geometry.
- The mapping from these to Δproxy_cost is not very wiggly.

### Training data

`train._collect_one_benchmark` runs **state-drift sampling**:

```python
state = PlacementState(initial)
for sample in range(N):
    macro = random movable macro
    for each candidate position:
        feats = _features_for_move(state, macro, Δ)
        apply move; new_cost = compute_proxy_cost(state)
        target = new_cost - base_cost
        record (feats, target)
        revert
    with prob 0.3: commit the best candidate as new base
                    (drift to keep sampling distribution diverse)
```

The drift is critical. Without it, every sample comes from the initial
placement, and the model learns *only* the initial-state cost surface.
At deployment, the placer drifts far from initial, and the model is
out-of-distribution.

### Loss and normalization

- Features normalized to zero mean / unit variance on the training set
- Targets (`Δproxy_cost`) normalized to zero mean / unit variance too
- Loss: `SmoothL1Loss` (Huber) — less sensitive to occasional outliers
- Optimizer: Adam, lr=1e-3, batch=64, 60 epochs typical

### Exit criterion

Target is Spearman ρ > 0.8 on a held-out 20% split. On multi-benchmark
runs with 1000+ examples this is usually achieved within 30 epochs.

---

## 9. `ChainPolicy` — PPO actor-critic

**Why?** A good cost approximator tells you *if* a move is good. A
policy decides *which* move to make and *when to stop*. The chain has
discrete actions (which candidate, or STOP), so we use PPO with a
categorical action distribution.

**Where?** `lkh_model.py::ChainPolicy` (the model), `chain_env.py::ChainEnv`
(the RL environment), `train_policy.py` (the trainer).

### The model

Actor-critic with three heads:

```
                       per-step inputs
        ┌──────────────────────────────────────┐
state ──┤  global_feat (5d)                    │
        │  macro_feat  (6d)                    │
        │  cand_feats  (K, 16d)                │
        │  chain_feat  (3d)                    │
        └──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
   move_head       stop_head       value_head
   (per cand)      (one)          (state value)
       │               │               │
       └─────softmax(K+1 logits)──┐    │
                                  ▼    ▼
                          action distrib   V(state)
```

- `move_head`: scores each of K candidate moves
- `stop_head`: scores the implicit "STOP" action
- `value_head`: V(state) for advantage estimation
- Sample action ∈ [0..K]; index K means STOP

### The environment

`ChainEnv` wraps `PlacementState` Gym-style:

```python
env = ChainEnv(state, seed_macro, rng)
while not env.done:
    sp = env.state_for_policy()         # build all features
    K = sp["candidates"].shape[0]
    logits, value = policy(sp...)
    action = sample(softmax(logits))
    reward, done, info = env.step(action, sp["raw_candidates"])
```

- **Reward**: `-Δhpwl_surrogate` per step (cumulative = -hpwl_gain via
  telescoping). Optional terminal bonus on commit. We use the surrogate
  here, not exact cost, for the same speed reason as before.
- **Episode end**: STOP action, clean landing (no displaced macros),
  or max chain length (8).
- **Commit gate**: same lex `(overlap_pairs, hpwl)` as `LKChain.run_greedy`.

### PPO update

Standard clipped objective with GAE:

```python
log_prob_ratio = exp(log_prob_new - log_prob_old)
surr1 = ratio * advantage
surr2 = clip(ratio, 1 - clip, 1 + clip) * advantage
policy_loss = -min(surr1, surr2).mean()

value_loss   = SmoothL1Loss(value, returns)
entropy_bonus = -mean(probs * log_probs)

loss = policy_loss + 0.5 · value_loss - 0.01 · entropy_bonus
```

### Why PPO

- Discrete action space → categorical → easy.
- Variable K (number of candidates differs per step) → process
  transitions individually in the update (no padding needed).
- Sample-efficient enough that small batches (4 trajectories/iter) work.
- Stable enough that the policy doesn't catastrophically collapse on
  small data (our scale).

### Reading PPO logs

Healthy training signals:

- `ent` (entropy) dropping → policy is becoming less random
- `commit` rate climbing → policy finds improving chains
- `R/traj` positive → chains are net-improving

If `ent` collapses too early, bump `--ent-coef`.

---

## 10. Iterative training loop

**Why?** The cost approximator and the policy each train one model, but a good policy
relies on a good approximator, and a good approximator's *useful* data
distribution depends on what the policy actually does at deployment. We
break this chicken-and-egg with multiple rounds:

```
round 0:   collect data (heuristic candidates)  → train approx + policy
round 1+:  retrain approx and policy with cached data, warm-started policy
```

**Where?** `train_iter.py`.

### What each round does

```
collect_data(benchmarks, num_examples_per_benchmark)
    ↓
train CostApproximator
    ↓
train ChainPolicy (warm-started from previous round if any)
    ↓
evaluate placer end-to-end on every benchmark
    ↓
save per-round checkpoint to output_root/round_<i>/
    ↓
append to history.json
```

### Cache behaviour

- Aggregated `chain_data.pt` is saved after round 0 and **reused** for
  rounds 1+. So subsequent rounds skip the slow data collection.
- Per-benchmark caches (`per_bench/<name>.pt`) survive worker preemption.
- `--force-recollect-each-round` invalidates both.

### Why warm-start the policy across rounds

The cost approximator's quality changes between rounds (more data, better
calibration). If we re-train the policy from scratch each round, we waste
all the previously learned behaviour. Warm-start lets the policy adapt
incrementally to the changing approximator.

---

## 11. Inference (`LKHPlacer.place`)

**Why?** This is the actual function the competition evaluator calls.
It needs to be wall-clock-bounded, valid, and use whatever checkpoints
exist.

**Where?** `placer.py::LKHPlacer.place`.

### Top-level structure

```python
def place(self, benchmark):
    state = PlacementState(benchmark, edges, edge_weights)
    seed_weights = self._seed_weights(state, movable_idx)
    best_pos, best_key = state.pos.copy(), (state.overlap_pairs(), state.hpwl())

    while chains < max and elapsed < budget:
        seed = rng.choices(movable_idx, weights=seed_weights, k=1)[0]
        result = LKChain(state, seed, rng,
                          approximator=self.approximator,
                          policy_bundle=self.policy_bundle,
                          ).run(max_chain_length)
        if result["committed"]:
            cur_key = (state.overlap_pairs(), state.hpwl())
            if cur_key < best_key:
                best_key, best_pos = cur_key, state.pos.copy()
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1
        if stagnation >= stagnation_threshold:
            self._perturb(state, rng)
            stagnation = 0

    state.pos[:] = best_pos
    if state.overlap_pairs() > 0:
        state.pos[:] = _legalize(state)
    return build_full_placement(state)
```

### Three things that make this robust

1. **Seed weighting.** We prefer to start chains from macros that
   contribute a lot to current HPWL — those are the ones that benefit
   most from a move. Falls back to uniform if no edges.
2. **Stagnation perturbation.** If 50 chains in a row fail to improve
   `best_key`, randomly relocate 5 macros to break out of the local
   minimum. The chain loop then has to recover from this disruption,
   but if best_pos was already good, we keep it via the `state.pos[:] =
   best_pos` restore at the end.
3. **Legalization.** Final spiral search to drive overlap_pairs → 0.

### Soft macros

Soft macros (standard-cell clusters) are **not** moved by this placer.
They stay at their `initial.plc` positions. We optimize hard-macro
positions only; the proxy cost computation treats soft macros as fixed
during scoring. This matches the convention in `submissions/will_seed`
and keeps the search space tractable.

---

## 12. Ablation

**Why?** Compare four placer variants on the same benchmark to see what
each learned component contributes.

**Where?** `ablate.py`.

### The four conditions

| Method | Approximator? | Policy? | Notes |
|---|:---:|:---:|---|
| A: initial.plc | – | – | no placer; baseline |
| B: surrogate only | no | no | LK chains with HPWL+overlap-penalty scoring |
| C: + approximator | yes | no | LK chains scored by CostApproximator predictions |
| D: + policy | yes | yes | Full system; policy picks actions |

Each runs for `--time-budget` seconds and reports `(proxy, wl, den, cong,
overlaps, runtime)`. Useful for visualizing what each piece adds.

### How conditions are forced

`LKHPlacer` takes `checkpoint_path` and `policy_path`. We pass:
- B: `checkpoint_path="/dev/null/...", policy_path="/dev/null/..."` →
  loaders return None → falls back to surrogate
- C: real `checkpoint_path`, bogus `policy_path`, `use_policy=False`
- D: both real paths

So the ablation script doesn't need to mutate the filesystem.

---

## 13. State encoder (GNN + CNN)

**Why?** The learned state representation: a 3-layer message-passing GNN
over the netlist hypergraph plus a CNN over a canvas raster. Outputs
per-macro embeddings and a global vector that feed both the cost
approximator and the policy with topology- and spatial-context-aware
features the hand-crafted 16-dim vector can't express.

**Where?** `encoder.py` (architecture: `PlacementGNN`, `CanvasCNN`,
`StateEncoder`) plus the runtime wrappers `encoder_runtime.py` (GNN-only,
primary path) and `encoder_runtime_gnncnn.py` (GNN+CNN variant) used by
the placer and trainers.

### What's in `encoder.py`

```python
class GNNLayer(nn.Module):
    # edge MLP → index_add aggregation → node MLP

class PlacementGNN(nn.Module):
    # 3 stacked GNNLayers with residual connections
    # output: (per_node [N, hidden], graph_pool [hidden])

class CanvasCNN(nn.Module):
    # 3 conv layers → adaptive pool → flatten
    # input: 3-channel canvas raster
    # output: [hidden]

class StateEncoder(nn.Module):
    # combines GNN + CNN; returns (per_node [N, hidden], global [2*hidden])
```

### How it's wired

- `LKHPlacer(feature_mode="encoder")` loads `checkpoints/encoder.pt`,
  builds an encoder cache once per chain, and routes the cost
  approximator + policy inputs through `[per_node[m]; graph_vec; hand_feats]`.
- `train_encoder_joint.py` runs the regression pretrain: joint
  optimizer over encoder + cost approximator parameters, Smooth-L1 on
  true Δproxy labels.
- `train_policy.py --use-encoder` (or `--encoder-fine-tune-in-ppo`)
  continues training the encoder during PPO so the embeddings adapt to
  the policy's needs.
- `CostApproximator.in_dim = encoder.embed_dim + 16` (272 for hidden=128).
- `ChainPolicy(encoder_global_dim=..., encoder_macro_dim=...)` enables
  the encoder concatenation paths in `move_head`, `stop_head`, `value_head`.
- The encoder forward runs once per chain (cached in the `ChainEnv`)
  so candidate scoring stays at thousands of moves per second.

---

## 14. Feature schema reference

Every model in the pipeline reads from a consistent feature schema. If
you ever swap features (e.g., wire in the encoder), keep this aligned
across files.

### Per-move features — `placer._features_for_move` → 16 dims

Computed for every candidate move:

| idx | feature | what it captures |
|---:|---|---|
| 0 | macro width | size |
| 1 | macro height | size |
| 2 | macro area | size |
| 3 | move Δx | which direction |
| 4 | move Δy | which direction |
| 5 | \|Δx\| | magnitude only |
| 6 | \|Δy\| | magnitude only |
| 7 | target_x / cw | normalized target position |
| 8 | target_y / ch | normalized target position |
| 9 | HPWL after - HPWL before | direct surrogate signal |
| 10 | n_overlap at source | how crowded the start was |
| 11 | n_overlap at target | how crowded the destination is |
| 12 | local-density count at source | broader crowding (3·size box) |
| 13 | local-density count at target | broader crowding |
| 14 | neighbor-attraction Δ | +ve = moving toward connected neighbors |
| 15 | macro degree | how connected this macro is |

### Global features — `chain_env.compute_global_features` → 5 dims

State summary; computed once per chain step:

```
[ hpwl_norm,       # current hpwl / (N · canvas_diagonal)
  overlap_norm,    # current overlap_pairs / max_possible
  pos_x_var,       # numpy.var(state.pos[:,0]) / cw^2
  pos_y_var,       # spatial spread
  movable_frac ]   # fraction of macros that are movable
```

### Per-macro features — `chain_env.compute_macro_features` → 6 dims

Per-step, for the *current* macro being moved:

```
[ w/cw, h/ch,             # geometry
  degree/N,               # connectivity
  n_overlap_self/N,       # current overlap count
  x/cw, y/ch ]            # current position
```

### Chain features — `chain_env.compute_chain_features` → 3 dims

Per-step, the chain's progress:

```
[ chain_length / max_length,                  # how deep in the cascade
  hpwl_gain / (|hpwl_gain| + 1),              # bounded
  n_displaced / 10 ]                          # arbitrary normalization
```

### Total dim flowing to each model

- `CostApproximator.in_dim` = 16 (per-move features only)
- `ChainPolicy.move_head` input = global (5) + macro (6) + cand (16) + chain (3) = 30
- `ChainPolicy.stop_head` and `value_head` input = global (5) + chain (3) = 8

If you wire in the encoder (§13), these go to ~400 and ~280.

---

## 15. The Modal architecture

**Why?** Cost-approximator training across 17 IBM benchmarks needs ~15 hours
sequentially. Modal lets us parallelize per-benchmark (one container per
benchmark), bringing total wall time down to ~3 hours.

**Where?** `modal_run.py`.

### App and image

One `modal.App("lkh-macro-place")`, one `modal.Image.debian_slim` with
torch + numpy + matplotlib + tqdm + absl-py. Local repo is added via
`image.add_local_dir(..., ignore=_ignore)`, with `_ignore` filtering out
the heavy TILOS subdirs (Flows, Docs, ExperimentalData, the
non-`Plc_client` parts of CodeElements). Result: ~800 MB mount instead
of ~3.5 GB.

### Function topology

```
                                          @app.function ┐
@app.local_entrypoint iter_ ─spawn──> run_iter           │ nonpreemptible
                                       │                  │ 8 CPU, 16 GB
                                       │                  │ timeout 24 h
                                       │                  │
                                       │                  ├──┐
                                       │                  │  │
                                       ├──starmap──> collect_one_benchmark_modal × 17
                                       │                     2 CPU, 4 GB, max_containers=20
                                       │                     each writes one cache file
                                       │                     each commits the volume
                                       │
                                       ├── train_iter.run_round (×N rounds)
                                       │      ├── train.collect_data (cache-aware)
                                       │      ├── train.train_model
                                       │      ├── train_policy.train_chain_policy
                                       │      └── per-benchmark eval
                                       │
                                       └── final _persist + volume.commit
```

### Why the patterns we chose

- `.spawn()` (not `.remote()`) on the local entrypoint → the call survives
  local terminal disconnect.
- `--detach` on `modal run` → the App survives local entrypoint return.
  Both flags are needed; using only one means the App tears down before
  the spawned function gets to run.
- `.starmap()` from inside `run_iter` → parallel per-benchmark collection.
  Modal handles the parallelism; we just consume the result iterator.
- `nonpreemptible=True` → no mid-collection restarts. Costs 3× the
  list CPU rate but eliminates an entire class of failures we hit.
- **Per-benchmark cache**: each container writes its result and commits
  immediately. If any container fails or times out, the others keep going.
- **Per-round archival** to `/output/iter/round_<i>/` + an aggregated
  `history.json`. Survives the most catastrophic failures.

### Volume layout after a run

```
/iter
├── history.json              ← per-round metrics, aggregated
├── checkpoints/              ← LATEST round's canonical checkpoints
│   ├── cost_approximator.pt
│   └── chain_policy.pt
├── data/chain_data.pt        ← aggregated training data
├── per_bench/                ← per-benchmark caches (preemption-safe)
│   ├── ibm01.pt
│   └── ...
├── round_0/                  ← archival per-round snapshots
├── round_1/
└── round_N/
```

---

## 16. Design trade-offs FAQ

**Q: Why hand-crafted features instead of the GNN+CNN encoder?**
The GNN+CNN needs `torch-geometric` + `torch-scatter`, which have
CUDA-pinned wheels and need a custom Dockerfile (judges run with
`--network none`). Hand-crafted features capture the same physical
signals (HPWL Δ, density, overlap counts, attraction) and run ~5× faster
per-step. The encoder is implemented in `encoder.py` for when you want
to swap it in.

**Q: Why a chain commit gate instead of just optimizing proxy cost?**
The commit gate is a fast, predictable filter. Optimizing proxy cost
directly inside the chain would require calling `compute_proxy_cost` per
candidate per step, which is unworkably slow. The gate's `(overlap_pairs,
hpwl)` lex order strictly improves both quantities we care about and is
cheap to evaluate.

**Q: Why PPO instead of DQN / SAC / random search?**
PPO is stable on small batches, supports the variable-sized action
space (K candidates + STOP) cleanly, and the categorical actor-critic
form is one of the simplest RL patterns. DQN would need q-network
inputs to handle variable K; SAC is continuous-action.

**Q: Why does the commit gate use overlap count not overlap area?**
Because the competition evaluator uses pair count (`overlap_count` in
`compute_proxy_cost`). Optimizing what's scored is the simpler answer.

**Q: Why is `compute_proxy_cost` so slow?**
It's pure Python in TILOS's `plc_client_os.py`. Each call walks all
nets, all modules, builds congestion maps. We can't rewrite it (the
judges use this exact implementation) so we use it only for **training
targets** and **final evaluation**, never inside the inner loop.

**Q: Why is the legalizer guaranteed to terminate?**
The spiral search is bounded at 150 rings per macro. If a macro can't
find a legal spot within 150 · step ≈ 37 macro widths, it stays at its
best-found position (which might still overlap). In practice 150 rings
is way more than needed on ICCAD04 designs; we see termination at ring
1–5 typically.

**Q: Why does the placer score candidates with `policy_OR_approximator
_OR_surrogate` instead of always using the best one?**
Robustness. If a checkpoint file is missing or corrupted, the placer
should still produce a valid output. The fallback chain
(`policy → approximator → surrogate`) makes the system degrade
gracefully instead of crashing.

**Q: What's the role of the stagnation perturbation?**
Random escape from local minima. When 50+ chains fail to improve the
best-key, the search is stuck. Moving 5 random macros to random positions
disrupts the configuration; subsequent chains then have to recover from
the disruption. If the disruption is worse than `best_pos`, we don't
update best_pos, so we lose nothing. If the recovery finds something
better, we win.

**Q: Why does training run multi-benchmark instead of fine-tuning per benchmark?**
Generalization. A single-benchmark policy memorizes that benchmark's cost
landscape. The competition tests on hidden NG45 designs in Tier 2 — our
policy needs to generalize. Training on the IBM set forces it to learn
*transferable* heuristics rather than memorizing positions.

**Q: How would I add a new feature to `_features_for_move`?**
1. Add it in `placer.py::_features_for_move` (the function and its
   assertion on `FEATURE_DIM`).
2. Bump `FEATURE_DIM` in `lkh_model.py` and `placer.py`.
3. Retrain CostApproximator (existing checkpoint becomes incompatible).
4. The ChainPolicy reads candidate features via `_features_for_move`
   too, so it also needs retraining.

**Q: Where would I tweak the chain candidate generation?**
`PlacementState.candidate_positions` in `placer.py`. Currently it returns
~16 positions per macro (1 connected centroid + 12 grid jitter + 4
random). You could add more types (gradient step, swap with same-size
macro, ...), or change how they're mixed.
