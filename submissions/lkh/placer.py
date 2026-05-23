"""
LKHPlacer — Phase 1 of the LK-Chain RL plan, non-learned foundation (MVP)

Implements the LK cascade move pattern from
background/mvp_implementation_plan_lkh.md, adapted to this repo's real API:
- Positions are CENTERS, not corners.
- compute_proxy_cost is expensive and requires a PlacementCost; we use a fast
  HPWL surrogate during the chain and only fall back to exact proxy cost for
  the final commit decision (when plc is available on disk).
- Only hard macros [0, num_hard_macros) are placed; soft macros are left at
  their initial-placement positions (matches will_seed pattern).

Deferred to later phases of the plan (need GPU/training infrastructure):
- Phase 2: GNN+CNN state encoder (torch_geometric)
- Phase 3: learned cost approximator (replaces _surrogate_delta below)
- Phase 4: PPO move-picking policy (replaces _greedy_step below)
- Phase 5: iterative training, drift detection
- Phase 6: full inference with learned models

Usage:
    uv run evaluate submissions/lkh/placer.py
    uv run evaluate submissions/lkh/placer.py -b ibm01
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark

# Make the sibling lkh_model.py importable when this file is loaded via
# importlib.util.spec_from_file_location (the evaluate.py path).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ── HPWL edge graph extraction ─────────────────────────────────────────────

def _hard_macro_edges(benchmark: Benchmark) -> tuple[np.ndarray, np.ndarray]:
    """Build weighted hard-macro-to-hard-macro edge list.

    Prefers ``net_pin_nodes`` (pin-level, available on newer benchmarks); falls
    back to ``net_nodes`` (macro-level) for legacy ``.pt`` files where pin info
    isn't persisted. Edge weight = sum over nets containing both macros of
    1 / (net_size - 1), matching the surrogate weighting used in will_seed.
    """
    n_hard = benchmark.num_hard_macros
    weights: dict[tuple[int, int], float] = {}

    if benchmark.net_pin_nodes:
        nets_owners = (
            sorted({int(o) for o in net_pins[:, 0].tolist() if o < n_hard})
            for net_pins in benchmark.net_pin_nodes
        )
    else:
        nets_owners = (
            sorted({int(o) for o in nodes.tolist() if o < n_hard})
            for nodes in benchmark.net_nodes
        )

    for owners in nets_owners:
        if len(owners) < 2:
            continue
        w = 1.0 / (len(owners) - 1)
        for i in range(len(owners)):
            for j in range(i + 1, len(owners)):
                key = (owners[i], owners[j])
                weights[key] = weights.get(key, 0.0) + w

    if not weights:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64)
    edges = np.array(list(weights.keys()), dtype=np.int64)
    w = np.array([weights[k] for k in weights], dtype=np.float64)
    return edges, w


def _load_plc_if_available(name: str):
    """Best-effort PLC reload for exact final-commit cost. Mirrors will_seed."""
    from macro_place.loader import load_benchmark, load_benchmark_from_dir

    iccad = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if iccad.exists():
        _, plc = load_benchmark_from_dir(str(iccad))
        return plc
    ng45 = {
        "ariane133": "ariane133",
        "ariane136": "ariane136",
        "nvdla": "nvdla",
        "mempool_tile": "mempool_tile",
    }
    design = ng45.get(name) or ng45.get(name.replace("_ng45", ""))
    if design:
        base = Path("external/MacroPlacement/Flows/NanGate45") / design / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


# ── Placement state ────────────────────────────────────────────────────────

class PlacementState:
    """Mutable hard-macro positions with O(N) overlap checks and HPWL surrogate."""

    def __init__(self, benchmark: Benchmark, edges: np.ndarray, edge_weights: np.ndarray):
        n_hard = benchmark.num_hard_macros
        self.n = n_hard
        self.pos = benchmark.macro_positions[:n_hard].numpy().astype(np.float64).copy()
        self.sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        self.half_w = self.sizes[:, 0] / 2.0
        self.half_h = self.sizes[:, 1] / 2.0
        self.movable = benchmark.get_movable_mask()[:n_hard].numpy()
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)

        # Pairwise separation thresholds for AABB overlap (centers).
        self.sep_x = (self.sizes[:, 0:1] + self.sizes[:, 0:1].T) / 2.0
        self.sep_y = (self.sizes[:, 1:2] + self.sizes[:, 1:2].T) / 2.0

        # HPWL surrogate edges (macro-pair, weight).
        self.edges = edges
        self.edge_weights = edge_weights
        self.neighbors: list[list[int]] = [[] for _ in range(n_hard)]
        for (i, j) in edges:
            self.neighbors[i].append(int(j))
            self.neighbors[j].append(int(i))

    # ── Geometry ──────────────────────────────────────────────────────────

    def clamp(self, idx: int, x: float, y: float) -> tuple[float, float]:
        return (
            float(np.clip(x, self.half_w[idx], self.cw - self.half_w[idx])),
            float(np.clip(y, self.half_h[idx], self.ch - self.half_h[idx])),
        )

    def overlapping_with(self, idx: int, gap: float = 0.05) -> list[int]:
        """Return indices of macros currently overlapping macro idx."""
        dx = np.abs(self.pos[idx, 0] - self.pos[:, 0])
        dy = np.abs(self.pos[idx, 1] - self.pos[:, 1])
        hits = (dx < self.sep_x[idx] + gap) & (dy < self.sep_y[idx] + gap)
        hits[idx] = False
        return np.where(hits)[0].tolist()

    def has_overlap(self, idx: int, gap: float = 0.05) -> bool:
        dx = np.abs(self.pos[idx, 0] - self.pos[:, 0])
        dy = np.abs(self.pos[idx, 1] - self.pos[:, 1])
        hits = (dx < self.sep_x[idx] + gap) & (dy < self.sep_y[idx] + gap)
        hits[idx] = False
        return bool(hits.any())

    def overlap_pairs(self, gap: float = 0.05) -> int:
        """Count overlapping hard-macro pairs (matches the evaluator semantics)."""
        n = self.n
        dx = np.abs(self.pos[:, 0:1] - self.pos[:, 0:1].T)
        dy = np.abs(self.pos[:, 1:2] - self.pos[:, 1:2].T)
        ov = (dx < self.sep_x + gap) & (dy < self.sep_y + gap)
        np.fill_diagonal(ov, False)
        return int(ov.sum() // 2)

    # ── HPWL surrogate ────────────────────────────────────────────────────

    def hpwl(self) -> float:
        if len(self.edges) == 0:
            return 0.0
        dx = np.abs(self.pos[self.edges[:, 0], 0] - self.pos[self.edges[:, 1], 0])
        dy = np.abs(self.pos[self.edges[:, 0], 1] - self.pos[self.edges[:, 1], 1])
        return float((self.edge_weights * (dx + dy)).sum())

    # ── Candidate generation ──────────────────────────────────────────────

    def count_within(self, idx: int, x: float, y: float, radius: float) -> int:
        """Count macro centers (excluding ``idx``) within an axis-aligned box
        of half-side ``radius`` around (x, y). Cheap proxy for local density."""
        dx = np.abs(self.pos[:, 0] - x)
        dy = np.abs(self.pos[:, 1] - y)
        hits = (dx < radius) & (dy < radius)
        hits[idx] = False
        return int(hits.sum())

    def candidate_positions(self, idx: int, num_candidates: int = 16,
                            rng: random.Random | None = None) -> list[tuple[float, float]]:
        """Candidate centers for macro idx: HPWL-attractor + small grid jitter."""
        r = rng or random
        cands: list[tuple[float, float]] = []

        # Connected centroid attractor (where HPWL would pull this macro).
        nbrs = self.neighbors[idx]
        if nbrs:
            cx = float(self.pos[nbrs, 0].mean())
            cy = float(self.pos[nbrs, 1].mean())
            cands.append(self.clamp(idx, cx, cy))

        # Local grid jitter steps proportional to macro size.
        step = max(self.sizes[idx, 0], self.sizes[idx, 1]) * 0.5
        for dxm, dym in [(-2, 0), (2, 0), (0, -2), (0, 2),
                         (-1, 0), (1, 0), (0, -1), (0, 1),
                         (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            x, y = self.clamp(idx,
                              self.pos[idx, 0] + dxm * step,
                              self.pos[idx, 1] + dym * step)
            cands.append((x, y))

        # Random canvas jumps (open-region proxy without expensive search).
        for _ in range(4):
            x, y = self.clamp(idx,
                              r.uniform(self.half_w[idx], self.cw - self.half_w[idx]),
                              r.uniform(self.half_h[idx], self.ch - self.half_h[idx]))
            cands.append((x, y))

        return cands[:num_candidates]


# ── Feature extraction (for Phase 3 cost approximator) ────────────────────

FEATURE_DIM = 16


def _features_for_move(state: "PlacementState", macro_idx: int,
                        move_delta: np.ndarray) -> np.ndarray:
    """16-dim feature vector for predicting Δproxy_cost of moving ``macro_idx``
    by ``move_delta``. Mirrors the physical signals a GNN+CNN encoder would
    surface: HPWL change, local density at source/target, overlap change,
    neighbor-attraction change, and geometry."""
    old_x = float(state.pos[macro_idx, 0])
    old_y = float(state.pos[macro_idx, 1])
    new_x = old_x + float(move_delta[0])
    new_y = old_y + float(move_delta[1])
    w = float(state.sizes[macro_idx, 0])
    h = float(state.sizes[macro_idx, 1])

    # HPWL surrogate Δ (analytic).
    hpwl_before = state.hpwl()
    saved = state.pos[macro_idx].copy()
    state.pos[macro_idx, 0] = new_x
    state.pos[macro_idx, 1] = new_y
    hpwl_after = state.hpwl()
    n_overlap_new = len(state.overlapping_with(macro_idx))
    state.pos[macro_idx] = saved
    n_overlap_old = len(state.overlapping_with(macro_idx))

    # Local-density proxy: macro centers within 3·max(w,h) of each spot.
    radius = max(w, h) * 3.0
    n_local_old = state.count_within(macro_idx, old_x, old_y, radius)
    n_local_new = state.count_within(macro_idx, new_x, new_y, radius)

    # Neighbor-attraction signal (Manhattan).
    nbrs = state.neighbors[macro_idx]
    if nbrs:
        cx = float(state.pos[nbrs, 0].mean())
        cy = float(state.pos[nbrs, 1].mean())
        attract_old = abs(old_x - cx) + abs(old_y - cy)
        attract_new = abs(new_x - cx) + abs(new_y - cy)
        deg = len(nbrs)
    else:
        attract_old = attract_new = 0.0
        deg = 0

    feats = np.array([
        w, h, w * h,                                # macro geometry
        float(move_delta[0]), float(move_delta[1]),
        abs(float(move_delta[0])), abs(float(move_delta[1])),
        new_x / state.cw, new_y / state.ch,         # normalized target
        hpwl_after - hpwl_before,                    # HPWL Δ (surrogate)
        float(n_overlap_old), float(n_overlap_new),  # local overlap counts
        float(n_local_old), float(n_local_new),      # local density
        attract_old - attract_new,                   # +ve = pulled toward neighbors
        float(deg),                                  # macro degree
    ], dtype=np.float32)
    assert feats.shape == (FEATURE_DIM,), f"feature dim drift: {feats.shape}"
    return feats


# ── Cost approximator loading (Phase 3) ────────────────────────────────────

def _load_cost_approximator(ckpt_path: Path):
    """Returns (model, feat_mean, feat_std, target_mean, target_std) or None."""
    if not ckpt_path.exists():
        return None
    try:
        from lkh_model import CostApproximator
    except ImportError:
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CostApproximator(
        in_dim=ckpt.get("feature_dim", FEATURE_DIM),
        hidden=ckpt.get("hidden", 64),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {
        "model": model,
        "feat_mean": ckpt["feat_mean"].numpy(),
        "feat_std": ckpt["feat_std"].numpy(),
        "target_mean": float(ckpt["target_mean"]),
        "target_std": float(ckpt["target_std"]),
        "pearson_r_val": ckpt.get("pearson_r_val"),
        "trained_on": ckpt.get("trained_on"),
    }


# ── LK chain (greedy, optionally approximator-guided) ──────────────────────

class LKChain:
    """Cascading move chain: pick seed; move to best candidate (by trained
    approximator if available, else HPWL+overlap surrogate); follow first
    displaced macro; cap at max_length; commit only on lex (overlap_pairs,
    hpwl) improvement vs the chain's start state."""

    def __init__(self, state: PlacementState, seed_macro: int,
                 rng: random.Random, approximator: dict | None = None):
        self.state = state
        self.rng = rng
        self.seed = seed_macro
        self.approx = approximator

    def run_greedy(self, max_length: int = 8) -> dict:
        st = self.state
        if not st.movable[self.seed]:
            return {"chain_gain": 0.0, "committed": False, "length": 0}

        snapshot = st.pos.copy()
        start_hpwl = st.hpwl()
        start_overlap_pairs = st.overlap_pairs()
        current = self.seed
        length = 0

        for step in range(max_length):
            cands = st.candidate_positions(current, num_candidates=12, rng=self.rng)
            if not cands:
                break

            # Score each candidate. With a trained approximator: predicted
            # Δproxy_cost (lower is better). Without: HPWL surrogate + per-
            # candidate overlap penalty (the pre-Phase-3 fallback).
            best_pos = None
            best_score = float("inf")
            hpwl_scale = max(st.cw, st.ch)
            old_x = float(st.pos[current, 0])
            old_y = float(st.pos[current, 1])

            if self.approx is not None:
                feats = np.stack([
                    _features_for_move(
                        st, current,
                        np.array([cx - old_x, cy - old_y], dtype=np.float64),
                    )
                    for (cx, cy) in cands
                ])
                norm = (feats - self.approx["feat_mean"]) / self.approx["feat_std"]
                with torch.no_grad():
                    pred = self.approx["model"](torch.tensor(norm, dtype=torch.float32))
                pred_delta = pred.numpy() * self.approx["target_std"] + self.approx["target_mean"]

                # Safety floor: candidates that would introduce extra hard-
                # macro overlaps get penalized in the same units as proxy cost.
                for k, (cx, cy) in enumerate(cands):
                    st.pos[current, 0] = cx
                    st.pos[current, 1] = cy
                    new_ov = len(st.overlapping_with(current))
                    st.pos[current, 0] = old_x
                    st.pos[current, 1] = old_y
                    score = float(pred_delta[k]) + 0.5 * new_ov
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy)
            else:
                saved = st.pos[current].copy()
                for (cx, cy) in cands:
                    st.pos[current, 0] = cx
                    st.pos[current, 1] = cy
                    new_ov = len(st.overlapping_with(current))
                    score = st.hpwl() + new_ov * hpwl_scale
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy)
                st.pos[current] = saved

            if best_pos is None:
                break

            # Apply best move.
            st.pos[current, 0], st.pos[current, 1] = best_pos
            length += 1

            # Cascade to a displaced macro, if any.
            displaced = [j for j in st.overlapping_with(current) if st.movable[j]]
            if not displaced:
                break  # clean landing, chain complete
            # Prefer the most-overlapping (closest) displaced macro.
            dx = np.abs(st.pos[displaced, 0] - st.pos[current, 0])
            dy = np.abs(st.pos[displaced, 1] - st.pos[current, 1])
            order = np.argsort(dx + dy)
            current = int(displaced[order[0]])

        # Commit gate: improvement in (overlap_pairs, hpwl), lex-ordered.
        # Initial placement may already be illegal; we want strict progress on
        # overlap pairs (matches evaluator), tie-break on HPWL improvement.
        end_hpwl = st.hpwl()
        end_overlap_pairs = st.overlap_pairs()
        gain = start_hpwl - end_hpwl

        better = (
            end_overlap_pairs < start_overlap_pairs
            or (end_overlap_pairs == start_overlap_pairs and gain > 0)
        )
        if better:
            return {"chain_gain": gain, "committed": True, "length": length,
                    "overlap_delta": end_overlap_pairs - start_overlap_pairs}

        st.pos[:] = snapshot
        return {"chain_gain": 0.0, "committed": False, "length": length,
                "overlap_delta": 0}


# ── Submission placer ──────────────────────────────────────────────────────

class LKHPlacer:
    """Run many random-seed greedy LK chains under a wall-clock budget."""

    def __init__(self, seed: int = 42, time_budget_s: float = 60.0,
                 max_chains: int = 5000, max_chain_length: int = 8,
                 checkpoint_path: str | None = None):
        self.seed = seed
        self.time_budget_s = time_budget_s
        self.max_chains = max_chains
        self.max_chain_length = max_chain_length
        ckpt = Path(checkpoint_path) if checkpoint_path else (
            _HERE / "checkpoints" / "cost_approximator.pt"
        )
        self.approximator = _load_cost_approximator(ckpt)
        if self.approximator is not None:
            r = self.approximator.get("pearson_r_val")
            trained_on = self.approximator.get("trained_on")
            print(f"[LKHPlacer] cost approximator loaded "
                  f"(r={r:.3f}, trained on {trained_on})")
        else:
            print(f"[LKHPlacer] no approximator at {ckpt} — using HPWL surrogate")

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        rng = random.Random(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        edges, edge_weights = _hard_macro_edges(benchmark)
        state = PlacementState(benchmark, edges, edge_weights)

        movable_idx = np.where(state.movable)[0].tolist()
        if not movable_idx:
            return benchmark.macro_positions.clone()

        # Cost-weighted seed sampling: macros with highest neighbor-displacement
        # are sampled more often. Falls back to uniform if no edges.
        seed_weights = self._seed_weights(state, movable_idx)

        start = time.time()
        chains = 0
        best_pos = state.pos.copy()
        best_key = (state.overlap_pairs(), state.hpwl())

        while chains < self.max_chains and (time.time() - start) < self.time_budget_s:
            seed_macro = rng.choices(movable_idx, weights=seed_weights, k=1)[0]
            result = LKChain(state, seed_macro, rng,
                              approximator=self.approximator).run_greedy(self.max_chain_length)
            chains += 1
            if result["committed"]:
                # Track best on the same lex criterion the chain commit gate uses.
                cur_key = (state.overlap_pairs(), state.hpwl())
                if cur_key < best_key:
                    best_key = cur_key
                    best_pos = state.pos.copy()

        state.pos[:] = best_pos

        # Build full placement: hard macros from LK, soft macros untouched.
        full = benchmark.macro_positions.clone()
        full[: state.n] = torch.tensor(state.pos, dtype=torch.float32)
        return full

    @staticmethod
    def _seed_weights(state: PlacementState, movable_idx: list[int]) -> list[float]:
        """Higher weight for macros with more / longer connections (likely
        to benefit most from a move)."""
        if len(state.edges) == 0:
            return [1.0] * len(movable_idx)

        score = np.zeros(state.n, dtype=np.float64)
        ex = np.abs(state.pos[state.edges[:, 0], 0] - state.pos[state.edges[:, 1], 0])
        ey = np.abs(state.pos[state.edges[:, 0], 1] - state.pos[state.edges[:, 1], 1])
        d = state.edge_weights * (ex + ey)
        np.add.at(score, state.edges[:, 0], d)
        np.add.at(score, state.edges[:, 1], d)
        w = [max(float(score[i]), 1e-6) for i in movable_idx]
        return w
