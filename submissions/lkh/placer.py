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
from typing import NamedTuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark

# Make the sibling lkh_model.py importable when this file is loaded via
# importlib.util.spec_from_file_location (the evaluate.py path).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ── HPWL edge graph extraction ─────────────────────────────────────────────

class HpwlEdges(NamedTuple):
    """Bundle of edge arrays driving the HPWL surrogate.

    C.1 (LKH critique fix): pre-fix the surrogate had only hard-hard pair
    edges and weighted them as 1 / (num_hard_owners_on_net - 1), pretending
    soft macros and IO ports did not exist. Proxy cost includes nets that
    connect hard macros to those fixed terminals, so the surrogate had a
    systematic blind spot that grew with soft-macro fan-out. The new
    surrogate uses the full clique-pair weight 1 / (k - 1) where k = total
    terminals on the net (hard + soft + ports), and adds a parallel
    hard-to-fixed-anchor edge set.
    """
    hh_edges: np.ndarray   # [E_hh, 2] hard-macro-to-hard-macro pairs
    hh_weights: np.ndarray  # [E_hh] per-pair weights
    hf_macros: np.ndarray   # [E_hf] hard-macro indices for each hard-to-fixed contribution
    hf_anchors: np.ndarray  # [E_hf, 2] (x, y) of the fixed terminal
    hf_weights: np.ndarray  # [E_hf] per-edge weights


def _hard_macro_edges(benchmark: Benchmark) -> HpwlEdges:
    """Build the HPWL surrogate edges including hard-to-fixed contributions.

    Prefers ``net_pin_nodes`` (pin-level, available on newer benchmarks);
    falls back to ``net_nodes`` (macro-level) for legacy ``.pt`` files where
    pin info isn't persisted.

    Indexing: owner ``o`` is a hard macro iff ``o < num_hard_macros``; soft
    macro iff ``num_hard_macros <= o < num_macros``; IO port iff
    ``o >= num_macros`` (port index = ``o - num_macros``).
    """
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.num_macros
    positions = benchmark.macro_positions.numpy()
    if len(benchmark.port_positions) > 0:
        ports = benchmark.port_positions.numpy()
    else:
        ports = np.zeros((0, 2), dtype=np.float64)

    def fixed_owner_pos(o: int) -> tuple[float, float]:
        if o < n_total:
            return float(positions[o, 0]), float(positions[o, 1])
        return float(ports[o - n_total, 0]), float(ports[o - n_total, 1])

    if benchmark.net_pin_nodes:
        nets_owners = (
            sorted({int(o) for o in net_pins[:, 0].tolist()})
            for net_pins in benchmark.net_pin_nodes
        )
    else:
        nets_owners = (
            sorted({int(o) for o in nodes.tolist()})
            for nodes in benchmark.net_nodes
        )

    hh_w: dict[tuple[int, int], float] = {}
    hf_w: dict[tuple[int, int], float] = {}
    for owners in nets_owners:
        if len(owners) < 2:
            continue
        hard = [o for o in owners if o < n_hard]
        fixed = [o for o in owners if o >= n_hard]
        k = len(hard) + len(fixed)
        if k < 2 or len(hard) == 0:
            continue  # all-fixed nets contribute nothing to the surrogate
        w = 1.0 / (k - 1)
        for i in range(len(hard)):
            for j in range(i + 1, len(hard)):
                key = (hard[i], hard[j])
                hh_w[key] = hh_w.get(key, 0.0) + w
            for f in fixed:
                key = (hard[i], f)
                hf_w[key] = hf_w.get(key, 0.0) + w

    if hh_w:
        hh_edges = np.array(list(hh_w.keys()), dtype=np.int64)
        hh_weights = np.array(list(hh_w.values()), dtype=np.float64)
    else:
        hh_edges = np.zeros((0, 2), dtype=np.int64)
        hh_weights = np.zeros(0, dtype=np.float64)

    if hf_w:
        keys = list(hf_w.keys())
        hf_macros = np.array([k[0] for k in keys], dtype=np.int64)
        hf_anchors = np.array([fixed_owner_pos(k[1]) for k in keys], dtype=np.float64)
        hf_weights = np.array([hf_w[k] for k in keys], dtype=np.float64)
    else:
        hf_macros = np.zeros(0, dtype=np.int64)
        hf_anchors = np.zeros((0, 2), dtype=np.float64)
        hf_weights = np.zeros(0, dtype=np.float64)

    return HpwlEdges(hh_edges, hh_weights, hf_macros, hf_anchors, hf_weights)


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

    # B.1 + B.2 (LKH critique fix): incremental HPWL and overlap caches.
    # Pre-fix the inner loop recomputed full HPWL (O(E)) and full pairwise
    # overlap (O(N^2)) per candidate evaluation. With caches, each move is
    # O(degree) for HPWL and O(N) for overlap (the per-macro row update).
    # The contract: any code path that writes ``state.pos`` directly must
    # call ``rebuild_caches()`` afterward, or the cached totals drift.
    # Set ``state._debug_incremental = True`` to get post-move assertions
    # against from-scratch recomputation.

    _DEFAULT_GAP = 1e-6

    def __init__(self, benchmark: Benchmark, edges, edge_weights=None,
                 hf_macros=None, hf_anchors=None, hf_weights=None):
        """Build a placement state.

        Two calling conventions are accepted for back-compat with code paths
        that pre-date C.1:

        * ``PlacementState(bench, hpwl_edges)`` — pass the ``HpwlEdges``
          bundle returned by ``_hard_macro_edges``.
        * ``PlacementState(bench, hh_edges, hh_weights)`` — legacy 2-array
          form; soft-macro / IO-port contributions default to empty.
        """
        if isinstance(edges, HpwlEdges):
            bundle = edges
            hh_edges = bundle.hh_edges
            hh_weights = bundle.hh_weights
            hf_macros = bundle.hf_macros
            hf_anchors = bundle.hf_anchors
            hf_weights = bundle.hf_weights
        else:
            hh_edges = edges
            hh_weights = edge_weights if edge_weights is not None \
                else np.zeros(0, dtype=np.float64)
            hf_macros = hf_macros if hf_macros is not None \
                else np.zeros(0, dtype=np.int64)
            hf_anchors = hf_anchors if hf_anchors is not None \
                else np.zeros((0, 2), dtype=np.float64)
            hf_weights = hf_weights if hf_weights is not None \
                else np.zeros(0, dtype=np.float64)

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
        self.edges = hh_edges
        self.edge_weights = hh_weights
        # C.1: parallel hard-to-fixed-anchor edge set.
        self.hf_macros = hf_macros
        self.hf_anchors = hf_anchors
        self.hf_weights = hf_weights

        self.neighbors: list[list[int]] = [[] for _ in range(n_hard)]
        # Per-pair weight lookup for connectivity-aware candidate generation
        # (A.3) and cascade-follow scoring. `neighbor_weight[i].get(j, 0.0)`
        # is O(1) — important because cascade-follow scores every displaced
        # macro per step.
        self.neighbor_weight: list[dict[int, float]] = [{} for _ in range(n_hard)]
        # Incident hh-edge index for fast HPWL delta on a single-macro move.
        macro_inc_hh: list[list[int]] = [[] for _ in range(n_hard)]
        for k, (i, j) in enumerate(hh_edges):
            ii, jj = int(i), int(j)
            w = float(hh_weights[k])
            self.neighbors[ii].append(jj)
            self.neighbors[jj].append(ii)
            self.neighbor_weight[ii][jj] = w
            self.neighbor_weight[jj][ii] = w
            macro_inc_hh[ii].append(k)
            macro_inc_hh[jj].append(k)
        self.macro_edge_indices: list[np.ndarray] = [
            np.array(lst, dtype=np.int64) for lst in macro_inc_hh
        ]
        # C.1: incident hf-edge index keyed by hard macro.
        macro_inc_hf: list[list[int]] = [[] for _ in range(n_hard)]
        for k, h in enumerate(hf_macros):
            macro_inc_hf[int(h)].append(k)
        self.macro_hf_indices: list[np.ndarray] = [
            np.array(lst, dtype=np.int64) for lst in macro_inc_hf
        ]

        # Incremental caches (filled by rebuild_caches below).
        self._debug_incremental = False
        self.per_edge_hpwl: np.ndarray = np.zeros(len(hh_edges), dtype=np.float64)
        self.per_edge_hf_hpwl: np.ndarray = np.zeros(len(hf_macros), dtype=np.float64)
        self._total_hpwl: float = 0.0
        self._overlap_partners: list[dict[int, float]] = [
            {} for _ in range(n_hard)
        ]
        self._total_overlap_pairs: int = 0
        self._total_overlap_area: float = 0.0
        self.rebuild_caches()

    # ── Geometry ──────────────────────────────────────────────────────────

    def clamp(self, idx: int, x: float, y: float) -> tuple[float, float]:
        return (
            float(np.clip(x, self.half_w[idx], self.cw - self.half_w[idx])),
            float(np.clip(y, self.half_h[idx], self.ch - self.half_h[idx])),
        )

    # Gap default of 1e-6 = "match the evaluator (strict overlap) up to float
    # precision". ibm01's initial.plc has ~190 pairs sitting within 0.05 μm
    # of touching; using gap=0.05 there miscounts those as overlaps and lets
    # chains "commit" reductions in the inflated count while real overlaps
    # increase. Callers that want a breathing-room margin must pass gap
    # explicitly.

    def overlapping_with(self, idx: int, gap: float = _DEFAULT_GAP) -> list[int]:
        """Return indices of macros currently overlapping macro idx."""
        if gap == self._DEFAULT_GAP:
            return list(self._overlap_partners[idx].keys())
        # Non-default gap: full recompute (rare path).
        dx = np.abs(self.pos[idx, 0] - self.pos[:, 0])
        dy = np.abs(self.pos[idx, 1] - self.pos[:, 1])
        hits = (dx + gap < self.sep_x[idx]) & (dy + gap < self.sep_y[idx])
        hits[idx] = False
        return np.where(hits)[0].tolist()

    def has_overlap(self, idx: int, gap: float = _DEFAULT_GAP) -> bool:
        if gap == self._DEFAULT_GAP:
            return bool(self._overlap_partners[idx])
        dx = np.abs(self.pos[idx, 0] - self.pos[:, 0])
        dy = np.abs(self.pos[idx, 1] - self.pos[:, 1])
        hits = (dx + gap < self.sep_x[idx]) & (dy + gap < self.sep_y[idx])
        hits[idx] = False
        return bool(hits.any())

    def overlap_pairs(self, gap: float = _DEFAULT_GAP) -> int:
        """Count overlapping hard-macro pairs (matches the evaluator semantics).

        O(1) cached read after B.2; pre-fix this was a full O(N^2) AABB.
        """
        if gap == self._DEFAULT_GAP:
            return self._total_overlap_pairs
        dx = np.abs(self.pos[:, 0:1] - self.pos[:, 0:1].T)
        dy = np.abs(self.pos[:, 1:2] - self.pos[:, 1:2].T)
        ov = (dx + gap < self.sep_x) & (dy + gap < self.sep_y)
        np.fill_diagonal(ov, False)
        return int(ov.sum() // 2)

    def overlap_area(self) -> float:
        """Sum of pairwise hard-macro overlap rectangle areas (cached).

        Used as the lex-tiebreaker between ``overlap_pairs`` and ``hpwl`` in
        the chain commit gate (A.2): chains that reduce overlap area without
        reducing the discrete pair count are rewarded. Drives legalization
        toward a near-no-op as the chain loop progresses.
        """
        return self._total_overlap_area

    # ── HPWL surrogate ────────────────────────────────────────────────────

    def hpwl(self) -> float:
        """Total HPWL surrogate (B.1 cached). Pre-fix this was O(E) per call."""
        return self._total_hpwl

    # ── Regularity (MaskRegulate Task 1) ──────────────────────────────────
    # Distance-to-nearest-edge proxy: low = edge-ward = good. Domain practice
    # is to push macros toward the canvas perimeter so the core stays open
    # for routing. Cheap (O(1) per macro) and analytic, so it gives a
    # zero-variance signal alongside the noisy learned approximator —
    # particularly useful for the congestion term, which the MLP ranks badly.

    def regularity_of(self, idx: int, x: float, y: float) -> float:
        """Distance from (x, y) to the nearest canvas edge along x + along y.

        Lower is better (edge-ward). Independent of macro size / orientation;
        evaluator semantics are positional, not corner-based.
        """
        return min(x, self.cw - x) + min(y, self.ch - y)

    def total_regularity(self) -> float:
        """Sum of ``regularity_of`` over **movable** macros.

        Used as the regularity term in the chain's blended commit gate /
        PPO reward. Fixed macros are excluded because they don't contribute
        to the optimization signal — including them would inflate the term
        and disguise improvements coming from movable repositioning.
        """
        if self.n == 0 or not self.movable.any():
            return 0.0
        xs = self.pos[self.movable, 0]
        ys = self.pos[self.movable, 1]
        dx_edge = np.minimum(xs, self.cw - xs)
        dy_edge = np.minimum(ys, self.ch - ys)
        return float((dx_edge + dy_edge).sum())

    # ── Incremental cache machinery (B.1 + B.2) ───────────────────────────

    def rebuild_caches(self) -> None:
        """Recompute HPWL and overlap caches from ``self.pos`` from scratch.
        Call this after any bulk ``state.pos[:] = ...`` assignment. Keeps
        the contract that cached totals always agree with ``self.pos``.
        """
        if len(self.edges) == 0:
            self.per_edge_hpwl = np.zeros(0, dtype=np.float64)
        else:
            dx = np.abs(self.pos[self.edges[:, 0], 0] - self.pos[self.edges[:, 1], 0])
            dy = np.abs(self.pos[self.edges[:, 0], 1] - self.pos[self.edges[:, 1], 1])
            self.per_edge_hpwl = (self.edge_weights * (dx + dy)).astype(np.float64)
        if len(self.hf_macros) == 0:
            self.per_edge_hf_hpwl = np.zeros(0, dtype=np.float64)
        else:
            dx = np.abs(self.pos[self.hf_macros, 0] - self.hf_anchors[:, 0])
            dy = np.abs(self.pos[self.hf_macros, 1] - self.hf_anchors[:, 1])
            self.per_edge_hf_hpwl = (self.hf_weights * (dx + dy)).astype(np.float64)
        self._total_hpwl = float(self.per_edge_hpwl.sum() + self.per_edge_hf_hpwl.sum())

        self._overlap_partners = [{} for _ in range(self.n)]
        self._total_overlap_pairs = 0
        self._total_overlap_area = 0.0
        if self.n > 1:
            dx = np.abs(self.pos[:, 0:1] - self.pos[:, 0:1].T)
            dy = np.abs(self.pos[:, 1:2] - self.pos[:, 1:2].T)
            is_overlap = (dx + self._DEFAULT_GAP < self.sep_x) & \
                         (dy + self._DEFAULT_GAP < self.sep_y)
            np.fill_diagonal(is_overlap, False)
            ox = np.maximum(0.0, self.sep_x - dx)
            oy = np.maximum(0.0, self.sep_y - dy)
            pair_area = ox * oy
            rows, cols = np.where(np.triu(is_overlap, k=1))
            for r, c in zip(rows.tolist(), cols.tolist()):
                a = float(pair_area[r, c])
                self._overlap_partners[r][c] = a
                self._overlap_partners[c][r] = a
                self._total_overlap_pairs += 1
                self._total_overlap_area += a

    def apply_move(self, idx: int, new_x: float, new_y: float
                   ) -> tuple[float, float]:
        """Move macro ``idx`` to (``new_x``, ``new_y``); incrementally
        update HPWL and overlap caches. Returns the previous (x, y) so a
        caller can revert with ``state.apply_move(idx, *prev)``.

        Cost: O(degree(idx)) for HPWL + O(N) for the overlap row update.
        Replaces the pre-B.1 pattern where every candidate evaluation
        re-summed all E edges and all N^2 pairs.
        """
        old_x = float(self.pos[idx, 0])
        old_y = float(self.pos[idx, 1])

        # 1. Subtract idx's current overlap contributions (symmetric removals).
        for j, area in list(self._overlap_partners[idx].items()):
            self._overlap_partners[j].pop(idx, None)
            self._total_overlap_pairs -= 1
            self._total_overlap_area -= area
        self._overlap_partners[idx].clear()

        # 2. Subtract idx's current HPWL contributions (hh + hf incident).
        incident = self.macro_edge_indices[idx]
        incident_hf = self.macro_hf_indices[idx]
        if len(incident) > 0:
            self._total_hpwl -= float(self.per_edge_hpwl[incident].sum())
        if len(incident_hf) > 0:
            self._total_hpwl -= float(self.per_edge_hf_hpwl[incident_hf].sum())

        # 3. Apply the move.
        self.pos[idx, 0] = new_x
        self.pos[idx, 1] = new_y

        # 4. Recompute incident hh-edge HPWL contributions and add them in.
        if len(incident) > 0:
            ie = self.edges[incident]
            ddx = np.abs(self.pos[ie[:, 0], 0] - self.pos[ie[:, 1], 0])
            ddy = np.abs(self.pos[ie[:, 0], 1] - self.pos[ie[:, 1], 1])
            new_per_edge = self.edge_weights[incident] * (ddx + ddy)
            self.per_edge_hpwl[incident] = new_per_edge
            self._total_hpwl += float(new_per_edge.sum())
        # 4b. Recompute incident hf-edge HPWL contributions (C.1).
        if len(incident_hf) > 0:
            anchors = self.hf_anchors[incident_hf]
            ddx = np.abs(self.pos[idx, 0] - anchors[:, 0])
            ddy = np.abs(self.pos[idx, 1] - anchors[:, 1])
            new_per_hf = self.hf_weights[incident_hf] * (ddx + ddy)
            self.per_edge_hf_hpwl[incident_hf] = new_per_hf
            self._total_hpwl += float(new_per_hf.sum())

        # 5. Recompute idx's overlap row (O(N), one vectorized AABB).
        dx_row = np.abs(self.pos[idx, 0] - self.pos[:, 0])
        dy_row = np.abs(self.pos[idx, 1] - self.pos[:, 1])
        sep_x_row = self.sep_x[idx]
        sep_y_row = self.sep_y[idx]
        is_overlap_row = (dx_row + self._DEFAULT_GAP < sep_x_row) & \
                         (dy_row + self._DEFAULT_GAP < sep_y_row)
        is_overlap_row[idx] = False
        new_partners_idx = np.where(is_overlap_row)[0]
        if len(new_partners_idx) > 0:
            ox = np.maximum(0.0, sep_x_row - dx_row)
            oy = np.maximum(0.0, sep_y_row - dy_row)
            for j_arr in new_partners_idx:
                j = int(j_arr)
                a = float(ox[j] * oy[j])
                self._overlap_partners[idx][j] = a
                self._overlap_partners[j][idx] = a
                self._total_overlap_pairs += 1
                self._total_overlap_area += a

        if self._debug_incremental:
            self._assert_caches_consistent()

        return old_x, old_y

    def _assert_caches_consistent(self) -> None:
        """Debug-mode invariant check: cached totals must agree with full
        recomputation. Float tolerance via ``np.allclose``.
        """
        cached_hpwl = self._total_hpwl
        cached_pairs = self._total_overlap_pairs
        cached_area = self._total_overlap_area

        # Tolerances: float64 cumulative round-off across thousands of
        # subtract/add operations realistically accrues ~1e-5 relative error
        # on aggregate sums; we want to catch logic bugs (which would drift
        # by orders of magnitude more) without firing on legitimate float
        # noise.
        rtol, atol = 1e-6, 1e-6
        true_hpwl = 0.0
        if len(self.edges) > 0:
            dx = np.abs(self.pos[self.edges[:, 0], 0] - self.pos[self.edges[:, 1], 0])
            dy = np.abs(self.pos[self.edges[:, 0], 1] - self.pos[self.edges[:, 1], 1])
            true_hpwl += float((self.edge_weights * (dx + dy)).sum())
        if len(self.hf_macros) > 0:
            dx = np.abs(self.pos[self.hf_macros, 0] - self.hf_anchors[:, 0])
            dy = np.abs(self.pos[self.hf_macros, 1] - self.hf_anchors[:, 1])
            true_hpwl += float((self.hf_weights * (dx + dy)).sum())
        assert np.isclose(cached_hpwl, true_hpwl, rtol=rtol, atol=atol), (
            f"HPWL cache drift: cached={cached_hpwl} true={true_hpwl}"
        )

        if self.n > 1:
            dx = np.abs(self.pos[:, 0:1] - self.pos[:, 0:1].T)
            dy = np.abs(self.pos[:, 1:2] - self.pos[:, 1:2].T)
            is_overlap = (dx + self._DEFAULT_GAP < self.sep_x) & \
                         (dy + self._DEFAULT_GAP < self.sep_y)
            np.fill_diagonal(is_overlap, False)
            true_pairs = int(is_overlap.sum() // 2)
            ox = np.maximum(0.0, self.sep_x - dx)
            oy = np.maximum(0.0, self.sep_y - dy)
            # Cache semantics: area accumulates only for pairs that the
            # strict gap test counts as overlapping (apply_move and
            # rebuild_caches both gate on ``is_overlap``). Mask the
            # from-scratch area the same way for an apples-to-apples check.
            pair_area = np.where(is_overlap, ox * oy, 0.0)
            true_area = float(pair_area.sum() / 2.0)
        else:
            true_pairs, true_area = 0, 0.0
        assert cached_pairs == true_pairs, (
            f"overlap_pairs cache drift: cached={cached_pairs} true={true_pairs}"
        )
        assert np.isclose(cached_area, true_area, rtol=rtol, atol=atol), (
            f"overlap_area cache drift: cached={cached_area} true={true_area}"
        )

    # ── PositionMask (MaskRegulate Task 3) ────────────────────────────────
    # A boolean grid_size × grid_size mask: True where placing macro ``idx``
    # creates **no** AABB overlap against any other macro. Cell semantics
    # match ``self.sep_x`` / ``self.sep_y``: a center separation strictly
    # less than the sum-of-half-sides means an overlap. Strict-overlap test
    # mirrors ``_DEFAULT_GAP`` semantics (1e-6) used elsewhere.

    def regularity_grid(self, idx: int, grid_size: int = 64) -> np.ndarray:
        """Per-cell regularity ``min(x, cw-x) + min(y, ch-y)`` for macro
        ``idx``'s legal box, on a ``[grid_size, grid_size]`` raster. Higher
        = farther from edge (worse per MaskRegulate). Used as a CNN channel
        in Task 4's encoder rasterizer.
        """
        x_lo = float(self.half_w[idx])
        x_hi = float(self.cw - self.half_w[idx])
        y_lo = float(self.half_h[idx])
        y_hi = float(self.ch - self.half_h[idx])
        if x_hi <= x_lo or y_hi <= y_lo:
            return np.zeros((grid_size, grid_size), dtype=np.float32)
        x_centers = np.linspace(x_lo, x_hi, grid_size)
        y_centers = np.linspace(y_lo, y_hi, grid_size)
        dx_edge = np.minimum(x_centers, self.cw - x_centers)
        dy_edge = np.minimum(y_centers, self.ch - y_centers)
        return (dy_edge[:, None] + dx_edge[None, :]).astype(np.float32)

    def position_mask(self, idx: int, grid_size: int = 64) -> np.ndarray:
        """Return a ``[grid_size, grid_size]`` bool grid: True iff placing
        macro ``idx`` at the cell center yields **no** overlap with any
        other macro. Axis order matches WireMask: ``mask[gy, gx]``.

        Cost: ``O(N × grid_size)`` for N hard macros — the per-cell test is
        vectorized over the grid, but iterated over macros (early-skips on
        macros that lie entirely outside the legal box for ``idx``). For
        the IBM benchmarks (N=246–537, grid=64) this is microseconds.
        """
        # Legal placement box for this macro.
        x_lo = float(self.half_w[idx])
        x_hi = float(self.cw - self.half_w[idx])
        y_lo = float(self.half_h[idx])
        y_hi = float(self.ch - self.half_h[idx])
        if x_hi <= x_lo or y_hi <= y_lo:
            return np.zeros((grid_size, grid_size), dtype=bool)
        gx = np.linspace(x_lo, x_hi, grid_size)
        gy = np.linspace(y_lo, y_hi, grid_size)
        sep_x_row = self.sep_x[idx]
        sep_y_row = self.sep_y[idx]
        forbidden = np.zeros((grid_size, grid_size), dtype=bool)
        for j in range(self.n):
            if j == idx:
                continue
            ox = float(self.pos[j, 0])
            oy = float(self.pos[j, 1])
            sx = float(sep_x_row[j]) - self._DEFAULT_GAP
            sy = float(sep_y_row[j]) - self._DEFAULT_GAP
            if sx <= 0 or sy <= 0:
                continue
            # 1-D blocked tests; if either axis has no blocked cell, the
            # outer product is empty and we can skip the OR.
            x_block = np.abs(gx - ox) < sx
            if not x_block.any():
                continue
            y_block = np.abs(gy - oy) < sy
            if not y_block.any():
                continue
            forbidden |= y_block[:, None] & x_block[None, :]
        return ~forbidden

    def _filter_cells_through_mask(self, idx: int,
                                   cells: list[tuple[float, float]],
                                   grid_size: int = 64,
                                   ) -> list[tuple[float, float]]:
        """Drop cells from ``cells`` that the ``position_mask`` reports
        illegal (overlapping). Order preserved. Cells outside the
        ``position_mask``'s legal box are conservatively kept (they don't
        index a mask entry anyway). Used by ``candidate_positions`` when
        ``use_position_mask=True`` to gate WireMask candidates pre-scoring.
        """
        if not cells:
            return cells
        mask = self.position_mask(idx, grid_size=grid_size)
        x_lo = float(self.half_w[idx])
        x_hi = float(self.cw - self.half_w[idx])
        y_lo = float(self.half_h[idx])
        y_hi = float(self.ch - self.half_h[idx])
        if x_hi <= x_lo or y_hi <= y_lo:
            return cells
        # Map each cell back to its nearest grid index and consult the mask.
        out: list[tuple[float, float]] = []
        for cx, cy in cells:
            gxi = int(round((cx - x_lo) / (x_hi - x_lo) * (grid_size - 1)))
            gyi = int(round((cy - y_lo) / (y_hi - y_lo) * (grid_size - 1)))
            gxi = max(0, min(grid_size - 1, gxi))
            gyi = max(0, min(grid_size - 1, gyi))
            if mask[gyi, gxi]:
                out.append((cx, cy))
        return out

    # ── WireMask (MaskRegulate Task 2) ────────────────────────────────────
    # Analytic HPWL-optimal candidate cells for one macro. The macro's added
    # HPWL contribution at cell (x, y) is sum over incident edges of
    # ``w * (|x - anchor_x| + |y - anchor_y|)``, which is **separable** in
    # x and y. That lets us compute the full grid map in O(grid × degree)
    # by summing two 1-D cost arrays via an outer sum, instead of the naive
    # O(grid² × degree). The minima of that grid are the HPWL-best cells.

    def wire_mask_grid(self, idx: int, grid_size: int = 64
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-cell HPWL cost grid for macro ``idx``. Returns
        ``(cost_grid, x_centers, y_centers)``. ``cost_grid[gy, gx]`` is the
        macro's added HPWL contribution if placed at ``(x_centers[gx],
        y_centers[gy])``. Returns ``(empty, empty, empty)`` if the macro
        has no incident edges or the legal box collapses.

        Shared by ``wire_mask_best`` (top-k extraction) and the Task 4
        encoder rasterizer (full grid as a CNN channel).
        """
        incident_hh = self.macro_edge_indices[idx]
        incident_hf = self.macro_hf_indices[idx]
        if len(incident_hh) == 0 and len(incident_hf) == 0:
            empty = np.zeros((0,), dtype=np.float64)
            return np.zeros((0, 0)), empty, empty

        anchors_x_parts = []
        anchors_y_parts = []
        weights_parts = []
        if len(incident_hh) > 0:
            ie = self.edges[incident_hh]
            # The "other endpoint" is whichever of (i, j) is not ``idx``.
            other = np.where(ie[:, 0] == idx, ie[:, 1], ie[:, 0])
            anchors_x_parts.append(self.pos[other, 0])
            anchors_y_parts.append(self.pos[other, 1])
            weights_parts.append(self.edge_weights[incident_hh])
        if len(incident_hf) > 0:
            anchors_x_parts.append(self.hf_anchors[incident_hf, 0])
            anchors_y_parts.append(self.hf_anchors[incident_hf, 1])
            weights_parts.append(self.hf_weights[incident_hf])

        anchors_x = np.concatenate(anchors_x_parts)
        anchors_y = np.concatenate(anchors_y_parts)
        weights = np.concatenate(weights_parts).astype(np.float64)

        # Legal placement box for this macro.
        x_lo = float(self.half_w[idx])
        x_hi = float(self.cw - self.half_w[idx])
        y_lo = float(self.half_h[idx])
        y_hi = float(self.ch - self.half_h[idx])
        if x_hi <= x_lo or y_hi <= y_lo:
            empty = np.zeros((0,), dtype=np.float64)
            return np.zeros((0, 0)), empty, empty

        x_centers = np.linspace(x_lo, x_hi, grid_size)
        y_centers = np.linspace(y_lo, y_hi, grid_size)

        # Separable: cost(x, y) = cost_x(x) + cost_y(y) with
        # cost_x[gx] = sum_k w_k * |x_centers[gx] - anchors_x[k]|.
        cost_x = (weights[:, None] *
                  np.abs(x_centers[None, :] - anchors_x[:, None])).sum(axis=0)
        cost_y = (weights[:, None] *
                  np.abs(y_centers[None, :] - anchors_y[:, None])).sum(axis=0)
        cost_grid = cost_y[:, None] + cost_x[None, :]
        return cost_grid, x_centers, y_centers

    def wire_mask_best(self, idx: int, grid_size: int = 64,
                       top_k: int = 3) -> list[tuple[float, float]]:
        """Top-``top_k`` cell centers minimizing macro ``idx``'s HPWL
        contribution, evaluated on a ``grid_size × grid_size`` raster of the
        legal placement box for this macro. Returns ``[]`` if the macro has
        no incident edges (then there is no analytic optimum to recommend).

        Cells are returned in best-first order, clamped to the legal canvas
        and de-duplicated (so they don't clash with adjacent grid neighbors
        when ``top_k`` is small).
        """
        cost_grid, x_centers, y_centers = self.wire_mask_grid(idx, grid_size)
        if cost_grid.size == 0:
            return []

        # k smallest cells. argpartition is O(N) vs sort's O(N log N).
        flat = cost_grid.ravel()
        k_eff = min(top_k, flat.size)
        # Slightly oversample (top_k * 4) so de-dup after clamp still yields
        # ``top_k`` distinct cells if the grid is coarse.
        oversample = min(flat.size, max(top_k * 4, top_k))
        part = np.argpartition(flat, oversample - 1)[:oversample]
        # Sort just the oversample slice.
        part = part[np.argsort(flat[part])]
        seen: set[tuple[float, float]] = set()
        out: list[tuple[float, float]] = []
        for fi in part:
            gy, gx = divmod(int(fi), grid_size)
            cx, cy = self.clamp(idx,
                                 float(x_centers[gx]),
                                 float(y_centers[gy]))
            key = (round(cx, 6), round(cy, 6))
            if key in seen:
                continue
            seen.add(key)
            out.append((cx, cy))
            if len(out) >= top_k:
                break
        return out

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
                            rng: random.Random | None = None,
                            use_wiremask: bool = False,
                            wiremask_k: int = 3,
                            wiremask_grid_size: int = 64,
                            use_position_mask: bool = False,
                            position_mask_grid_size: int = 64,
                            ) -> list[tuple[float, float]]:
        """Candidate centers for macro ``idx``: connected centroid + top-K
        connected partners + grid jitter + random jumps.

        A.3 (LKH critique fix): the pre-fix mix was 1 connected centroid +
        12 grid jitter + 4 random — only 1/16 carried connectivity signal.
        We now spend 4 of the 16 slots on positions stepping toward each of
        the macro's top-4 most strongly-connected partners (alpha-nearness
        analog), keeping 7 grid-jitter and 4 random.

        Task 2 (MaskRegulate WireMask): when ``use_wiremask=True``, replace
        the last ``wiremask_k`` random-jump slots with the analytic HPWL-best
        cells from ``wire_mask_best``. Total slot count is unchanged so
        downstream policy ``cand_dim`` and per-step compute stay constant.
        """
        r = rng or random
        cands: list[tuple[float, float]] = []

        # Connected centroid attractor (where HPWL would pull this macro).
        nbrs = self.neighbors[idx]
        if nbrs:
            cx = float(self.pos[nbrs, 0].mean())
            cy = float(self.pos[nbrs, 1].mean())
            cands.append(self.clamp(idx, cx, cy))

        # Connected-partner offsets: step toward each of the top-4 highest-
        # weight neighbors by 0.5 * max(w, h). Skip near-zero distances so
        # we don't add a useless duplicate of the current position.
        step = max(self.sizes[idx, 0], self.sizes[idx, 1]) * 0.5
        cur_x = float(self.pos[idx, 0])
        cur_y = float(self.pos[idx, 1])
        if self.neighbor_weight[idx]:
            top_partners = sorted(self.neighbor_weight[idx].items(),
                                  key=lambda kv: -kv[1])[:4]
            for partner_j, _w in top_partners:
                px = float(self.pos[partner_j, 0])
                py = float(self.pos[partner_j, 1])
                vdx = px - cur_x
                vdy = py - cur_y
                norm = (vdx * vdx + vdy * vdy) ** 0.5
                if norm < 1e-9:
                    continue
                cands.append(self.clamp(
                    idx,
                    cur_x + vdx * step / norm,
                    cur_y + vdy * step / norm,
                ))

        # Local grid jitter steps proportional to macro size. Trimmed from
        # 12 to 7 (kept 4 wide cardinals + 3 narrow cardinals; dropped 4
        # diagonals and 1 narrow cardinal to make room for the connected-
        # partner offsets above).
        for dxm, dym in [(-2, 0), (2, 0), (0, -2), (0, 2),
                         (-1, 0), (1, 0), (0, -1)]:
            x, y = self.clamp(idx,
                              cur_x + dxm * step,
                              cur_y + dym * step)
            cands.append((x, y))

        # Random canvas jumps. Task 2: when use_wiremask is on, reserve the
        # last ``wiremask_k`` slots for the analytic HPWL-optimal cells
        # instead. The wire_mask_best() call is O(grid * degree); on ibm01
        # at grid=64, degree~30, that's ~2k ops per macro per chain step.
        n_random = max(0, 4 - wiremask_k) if use_wiremask else 4
        for _ in range(n_random):
            x, y = self.clamp(idx,
                              r.uniform(self.half_w[idx], self.cw - self.half_w[idx]),
                              r.uniform(self.half_h[idx], self.ch - self.half_h[idx]))
            cands.append((x, y))

        if use_wiremask:
            wm = self.wire_mask_best(idx, grid_size=wiremask_grid_size,
                                     top_k=wiremask_k)
            # Task 3: pre-score legality filter. WireMask cells are HPWL-
            # optimal but ignore overlap — a fully-clustered design can
            # have its WireMask minima inside other macros. Filtering them
            # away here is cheaper than letting the chain score and reject
            # them (the surrogate's per-overlap penalty isn't always large
            # enough to disqualify them, as Task 2's ibm01 run showed).
            if use_position_mask:
                wm = self._filter_cells_through_mask(
                    idx, wm, grid_size=position_mask_grid_size,
                )
            cands.extend(wm)

        return cands[:num_candidates]


# ── Feature extraction (for Phase 3 cost approximator) ────────────────────

FEATURE_DIM = 16
# Task 1c (MaskRegulate): augmented schema with a regularity-Δ feature
# appended. Kept as a SEPARATE constant so existing 16-dim checkpoints
# remain loadable without modification — the approximator's ``in_dim`` is
# read from the checkpoint and ``_features_for_move`` is invoked with
# ``use_reg_feature`` matching that ``in_dim``.
FEATURE_DIM_WITH_REG = 17


def _features_for_move(state: "PlacementState", macro_idx: int,
                        move_delta: np.ndarray,
                        use_reg_feature: bool = False) -> np.ndarray:
    """Feature vector for predicting Δproxy_cost of moving ``macro_idx``
    by ``move_delta``. 16-dim by default; 17-dim with ``use_reg_feature=True``
    (Task 1c — appends signed regularity Δ).

    Mirrors the physical signals a GNN+CNN encoder would surface: HPWL
    change, local density at source/target, overlap change, neighbor-
    attraction change, and geometry. The optional 17th feature is
    ``reg_before - reg_after`` (positive = pushed edge-ward = good per
    MaskRegulate's regularity term).
    """
    old_x = float(state.pos[macro_idx, 0])
    old_y = float(state.pos[macro_idx, 1])
    new_x = old_x + float(move_delta[0])
    new_y = old_y + float(move_delta[1])
    w = float(state.sizes[macro_idx, 0])
    h = float(state.sizes[macro_idx, 1])

    # HPWL surrogate Δ via incremental apply_move + revert (B.3). Pre-fix
    # this called full state.hpwl() twice (O(E) each).
    hpwl_before = state.hpwl()
    n_overlap_old = len(state.overlapping_with(macro_idx))
    state.apply_move(macro_idx, new_x, new_y)
    hpwl_after = state.hpwl()
    n_overlap_new = len(state.overlapping_with(macro_idx))
    state.apply_move(macro_idx, old_x, old_y)

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

    base = [
        w, h, w * h,                                # macro geometry
        float(move_delta[0]), float(move_delta[1]),
        abs(float(move_delta[0])), abs(float(move_delta[1])),
        new_x / state.cw, new_y / state.ch,         # normalized target
        hpwl_after - hpwl_before,                    # HPWL Δ (surrogate)
        float(n_overlap_old), float(n_overlap_new),  # local overlap counts
        float(n_local_old), float(n_local_new),      # local density
        attract_old - attract_new,                   # +ve = pulled toward neighbors
        float(deg),                                  # macro degree
    ]
    if use_reg_feature:
        # Task 1c: regularity Δ. Positive = pushed toward edge.
        reg_before = state.regularity_of(macro_idx, old_x, old_y)
        reg_after = state.regularity_of(macro_idx, new_x, new_y)
        base.append(reg_before - reg_after)

    feats = np.array(base, dtype=np.float32)
    expected_dim = FEATURE_DIM_WITH_REG if use_reg_feature else FEATURE_DIM
    assert feats.shape == (expected_dim,), \
        f"feature dim drift: {feats.shape} expected ({expected_dim},)"
    return feats


# ── Legalization (Phase 6 finishing step) ──────────────────────────────────

def _legalize(state: PlacementState, *, gap: float = 0.001,
              max_search_radius: int = 150) -> np.ndarray:
    """Greedy outward-spiral legalization to zero hard-macro overlap pairs.

    Largest movable macros are placed first; each one keeps its position if it
    doesn't collide with already-placed macros, otherwise it snaps to the
    nearest legal slot via an outward (Chebyshev-ring) spiral around the
    target position. Fixed macros are anchors. The safety ``gap`` (default
    1 nm) sits inside the evaluator's strict overlap definition, so the
    result passes ``validate_placement``.

    Adapted from submissions/will_seed/placer.py with a tighter gap (was
    0.05 μm) and bounded search radius (configurable).

    Returns the legalized [N, 2] array; the input ``state.pos`` is unchanged.
    """
    n = state.n
    sizes = state.sizes
    half_w = state.half_w
    half_h = state.half_h
    cw, ch = state.cw, state.ch
    movable = state.movable
    pos = state.pos.copy()

    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0 + gap
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0 + gap

    order = sorted(range(n), key=lambda i: -float(sizes[i, 0] * sizes[i, 1]))
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        # If current position is already legal w.r.t. placed neighbours, keep it.
        if placed.any():
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            collide = (dx < sep_x[idx]) & (dy < sep_y[idx]) & placed
            collide[idx] = False
            if not collide.any():
                placed[idx] = True
                continue

        # Outward spiral search. ``step`` is 1/4 of the macro's longest side
        # so we don't skip plausible slots while keeping the search bounded.
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
        best_p = legal[idx].copy()
        best_d = float("inf")
        for r in range(1, max_search_radius + 1):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = float(np.clip(pos[idx, 0] + dxm * step,
                                        half_w[idx], cw - half_w[idx]))
                    cy = float(np.clip(pos[idx, 1] + dym * step,
                                        half_h[idx], ch - half_h[idx]))
                    if placed.any():
                        dx = np.abs(cx - legal[:, 0])
                        dy = np.abs(cy - legal[:, 1])
                        c = (dx < sep_x[idx]) & (dy < sep_y[idx]) & placed
                        c[idx] = False
                        if c.any():
                            continue
                    d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                    if d < best_d:
                        best_d = d
                        best_p = np.array([cx, cy])
                        found = True
            if found:
                break

        legal[idx] = best_p
        placed[idx] = True

    return legal


# ── Legalization-time probe (A.4 of LKH critique fix) ─────────────────────

def _probe_legalization_cost(state: PlacementState, n_probe: int = 10) -> float:
    """Estimate full ``_legalize`` wall time by replicating the spiral
    inner-loop (read-only) on ``n_probe`` largest movable macros and
    extrapolating to all movable macros.

    The probe runs a fixed 3-ring sweep per macro (vs. up to 150 in real
    legalization), so it consistently under-estimates dense placements and
    over-estimates sparse ones. The caller multiplies by a safety factor to
    bias toward over-estimation; the legalization budget is a soft reserve
    (chains stop early if the budget runs low) so a moderate over-shoot is
    cheap and an under-shoot only delays the cutoff slightly.
    """
    n = state.n
    if n == 0:
        return 0.0
    movable_idx = np.where(state.movable)[0]
    if len(movable_idx) == 0:
        return 0.0
    order = sorted(movable_idx.tolist(),
                   key=lambda i: -float(state.sizes[i, 0] * state.sizes[i, 1]))
    sample = order[:min(n_probe, len(order))]

    sep_x = (state.sizes[:, 0:1] + state.sizes[:, 0:1].T) / 2.0 + 0.001
    sep_y = (state.sizes[:, 1:2] + state.sizes[:, 1:2].T) / 2.0 + 0.001

    t0 = time.time()
    for idx in sample:
        step = max(state.sizes[idx, 0], state.sizes[idx, 1]) * 0.25
        for r in range(1, 4):  # 3-ring probe; full _legalize allows up to 150
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = float(state.pos[idx, 0] + dxm * step)
                    cy = float(state.pos[idx, 1] + dym * step)
                    dx = np.abs(cx - state.pos[:, 0])
                    dy = np.abs(cy - state.pos[:, 1])
                    _ = (dx < sep_x[idx]) & (dy < sep_y[idx])
    elapsed = time.time() - t0
    return elapsed * len(movable_idx) / max(len(sample), 1)


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
    feat_dim = int(ckpt.get("feature_dim", FEATURE_DIM))
    # Task 1c: infer ``use_reg_feature`` from the ckpt. Prefer an explicit
    # field; otherwise fall back to a dim check so older 16-d checkpoints
    # still resolve correctly even if they predate the field.
    use_reg_feature = bool(
        ckpt.get("use_reg_feature", feat_dim == FEATURE_DIM_WITH_REG)
    )
    model = CostApproximator(
        in_dim=feat_dim,
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
        "use_reg_feature": use_reg_feature,
    }


# ── Chain policy loading (Phase 4) ─────────────────────────────────────────

def _load_chain_policy(ckpt_path: Path):
    """Returns a dict with policy + chain_env helpers, or None if missing.

    Also reconstructs the optional ``SeedSelectionHead`` (Fix 3) if the
    checkpoint co-stored it under ``seed_head_state_dict``. The head is
    returned under bundle key ``"seed_head"`` or None.
    """
    if not ckpt_path.exists():
        return None
    try:
        from lkh_model import ChainPolicy
    except ImportError:
        return None
    # Import chain_env via importlib to avoid circular imports.
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("chain_env", str(_HERE / "chain_env.py"))
    env_mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(env_mod)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # ChainPolicy may have been trained with MaskRegulate's cand_dim /
    # encoder dims or the session building-block's base-dim overrides.
    # Take whichever is present in the checkpoint; default to legacy.
    policy_kwargs = {"hidden": ckpt.get("hidden", 64)}
    for k in ("global_dim", "macro_dim", "cand_dim", "chain_dim",
              "encoder_global_dim", "encoder_macro_dim"):
        if k in ckpt:
            policy_kwargs[k] = int(ckpt[k])
    policy = ChainPolicy(**policy_kwargs)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    # MaskRegulate Task 4b: if the checkpoint co-stored a StateEncoder,
    # rebuild it for inference. Legacy checkpoints have no encoder.
    encoder = None
    enc_global_dim = int(ckpt.get("encoder_global_dim", 0))
    enc_state = ckpt.get("encoder_state_dict")
    if enc_global_dim > 0 and enc_state is not None:
        try:
            spec_enc = _ilu.spec_from_file_location(
                "lkh_encoder", str(_HERE / "encoder.py"))
            enc_mod = _ilu.module_from_spec(spec_enc)
            spec_enc.loader.exec_module(enc_mod)
            encoder = enc_mod.StateEncoder(
                hidden_dim=int(ckpt.get("encoder_hidden", 64)),
                num_gnn_layers=3,
                grid_size=int(ckpt.get("encoder_grid_size", 64)),
            )
            encoder.load_state_dict(enc_state)
            encoder.eval()
        except Exception as exc:                                   # noqa: BLE001
            print(f"[LKHPlacer] encoder reconstruct failed ({exc}); "
                  f"running policy without encoder")
            encoder = None

    # Session building-block: seed-head co-stored in checkpoint dict.
    seed_head = None
    if "seed_head_state_dict" in ckpt:
        try:
            from seed_head import rebuild_seed_head_from_policy_ckpt
            seed_head = rebuild_seed_head_from_policy_ckpt(ckpt)
        except Exception:                                             # noqa: BLE001
            seed_head = None

    return {
        "policy": policy,
        "ChainEnv": env_mod.ChainEnv,
        "trained_on": ckpt.get("trained_on"),
        "n_iterations": ckpt.get("n_iterations"),
        "encoder": encoder,
        "encoder_global_dim": enc_global_dim,
        "encoder_macro_dim": int(ckpt.get("encoder_macro_dim", 0)),
        "encoder_grid_size": int(ckpt.get("encoder_grid_size", 64)),
        "encoder_hidden": int(ckpt.get("encoder_hidden", 64)),
        "encoder_path": ckpt.get("encoder_path"),
        "seed_head": seed_head,
    }


def _load_encoder(ckpt_path: Path, kind: str = "gnn"):
    """Load a GNN encoder runtime (Fix 2). ``kind`` selects which runtime
    module to use: ``"gnn"`` (primary) or ``"gnncnn"`` (optional extra).
    Returns the bundle dict from the respective ``load_encoder`` or None.
    """
    if not ckpt_path.exists():
        return None
    try:
        if kind == "gnn":
            from encoder_runtime import load_encoder
        elif kind == "gnncnn":
            from encoder_runtime_gnncnn import load_encoder
        else:
            raise ValueError(f"unknown encoder kind={kind!r}; expected 'gnn' or 'gnncnn'")
    except ImportError:
        return None
    return load_encoder(ckpt_path)


# ── LK chain (greedy, optionally approximator-guided) ──────────────────────

class LKChain:
    """Cascading move chain: pick seed; move to best candidate (by trained
    approximator if available, else HPWL+overlap surrogate); follow first
    displaced macro; cap at max_length; commit only on lex (overlap_pairs,
    hpwl) improvement vs the chain's start state.

    If a learned ``policy`` is provided, the policy chooses the action at
    each step instead of the greedy approximator/surrogate. The commit gate
    is unchanged so the placer still guarantees monotone (overlap_pairs,
    hpwl) progress regardless of which scorer the chain uses.
    """

    def __init__(self, state: PlacementState, seed_macro: int,
                 rng: random.Random, approximator: dict | None = None,
                 policy_bundle: dict | None = None,
                 gate_mode: str = "hpwl",
                 # MaskRegulate opt-ins
                 reg_weight: float = 0.0,
                 use_wiremask: bool = False,
                 use_position_mask: bool = False,
                 encoder=None,
                 # Session building-block opt-ins
                 feature_mode: str = "handcrafted",
                 encoder_cache: dict | None = None,
                 lam: float = 0.01):
        """``gate_mode`` selects the third lex-coordinate of the commit gate:

        * ``"hpwl"`` (default): ``(overlap_pairs, overlap_area, hpwl)``.
          Milestone-A behavior.
        * ``"predicted_proxy"``: ``(overlap_pairs, overlap_area,
          cumulative_predicted_Δproxy)``. Requires an approximator. Milestone E
          (LKH critique fix §3.3 deferred).
        * ``"scalar_penalty"``: single-scalar commit (session). Score is
          ``predicted_Δproxy + lam × n_new_overlap_pairs``. Requires an
          approximator. See ``commit_gate_scalar.py``.

        MaskRegulate Task 1b: ``reg_weight`` (default 0.0 → unchanged) blends
        an analytic regularity term into the third lex coordinate. When
        ``reg_weight > 0``, the third coord becomes
        ``base_norm + reg_weight * reg_norm`` where ``base_norm`` is HPWL
        (or predicted Δproxy) divided by ``max(cw, ch)`` and ``reg_norm`` is
        ``total_regularity()`` divided by ``cw + ch``.

        ``feature_mode`` selects the candidate-feature pipeline:

        * ``"handcrafted"`` (default): ``_features_for_move`` 16-dim vector.
        * ``"encoder"``: GNN-encoder embedding + hand_feats; needs
          ``encoder_cache`` (``{"per_node": Tensor, "graph_vec": Tensor}``).
        """
        self.state = state
        self.rng = rng
        self.seed = seed_macro
        self.approx = approximator
        self.policy_bundle = policy_bundle
        self.gate_mode = gate_mode
        # MaskRegulate Task 1b/2/3/4b state
        self.reg_weight = float(reg_weight)
        self.use_wiremask = bool(use_wiremask)
        self.use_position_mask = bool(use_position_mask)
        self.encoder = encoder
        # Session building-block state
        self.feature_mode = feature_mode
        self.encoder_cache = encoder_cache
        self.lam = float(lam)
        if gate_mode not in ("hpwl", "predicted_proxy", "scalar_penalty"):
            raise ValueError(
                f"unknown gate_mode={gate_mode!r}; expected 'hpwl', "
                f"'predicted_proxy', or 'scalar_penalty'"
            )
        if feature_mode not in ("handcrafted", "encoder"):
            raise ValueError(
                f"unknown feature_mode={feature_mode!r}; expected "
                f"'handcrafted' or 'encoder'"
            )
        if gate_mode == "predicted_proxy" and approximator is None:
            # Soft fallback: predicted_proxy without an approximator is
            # meaningless, so silently fall back to hpwl. This keeps the
            # ablation harness from blowing up when the user pairs the new
            # gate with a pre-Milestone-C checkpoint that's been removed.
            self.gate_mode = "hpwl"
        if gate_mode == "scalar_penalty" and approximator is None:
            # Same fallback rationale as predicted_proxy.
            self.gate_mode = "hpwl"
        if feature_mode == "encoder" and encoder_cache is None:
            # Encoder mode without a cache silently falls back; PPO rollouts
            # never enter this state because the placer fills the cache, but
            # this protects ablation harnesses.
            self.feature_mode = "handcrafted"

    def _compute_third(self, st: PlacementState,
                       predicted_cumulative: float) -> float:
        """Compute the third lex coordinate for the commit gate.

        When ``reg_weight == 0`` (default), returns the original behavior:
        ``st.hpwl()`` under hpwl gate or cumulative predicted Δproxy under
        predicted_proxy gate. When ``reg_weight > 0``, blends the base
        coord with the canvas-normalized regularity term (Task 1b).
        """
        if self.gate_mode == "predicted_proxy":
            base = predicted_cumulative
        else:
            base = st.hpwl()
        if self.reg_weight <= 0.0:
            return base
        hpwl_scale = max(st.cw, st.ch)
        reg_scale = st.cw + st.ch
        base_norm = base / max(hpwl_scale, 1e-9)
        reg_norm = st.total_regularity() / max(reg_scale, 1e-9)
        return base_norm + self.reg_weight * reg_norm

    def run_with_policy(self, max_length: int = 8) -> dict:
        """Drive the chain with the trained policy via ChainEnv."""
        st = self.state
        if not st.movable[self.seed]:
            return {"chain_gain": 0.0, "committed": False, "length": 0,
                    "overlap_delta": 0, "best_prefix_index": 0}

        ChainEnv = self.policy_bundle["ChainEnv"]
        policy = self.policy_bundle["policy"]
        env = ChainEnv(st, self.seed, rng=self.rng,
                        max_chain_length=max_length, max_candidates=8,
                        gate_mode=self.gate_mode,
                        approximator=self.approx,
                        reg_weight=self.reg_weight,
                        use_wiremask=self.use_wiremask,
                        use_position_mask=self.use_position_mask,
                        encoder=self.encoder,
                        feature_mode=self.feature_mode,
                        encoder_cache=self.encoder_cache,
                        lam=self.lam)
        import torch.nn.functional as _F
        final_info: dict = {}
        while not env.done:
            sp = env.state_for_policy()
            K = sp["candidates"].shape[0]
            if K == 0:
                _r, _d, info = env.step(0, [])
                if _d:
                    final_info = info
                break
            # Task 4b: forward encoder embeddings to policies that were
            # trained with them (encoder_global_dim / encoder_macro_dim
            # > 0). Legacy 16/17-d policies ignore these — they were built
            # with the dims at 0, so the kwargs default to None and pass
            # through harmlessly.
            enc_global = (torch.from_numpy(sp["encoder_global"])
                          if "encoder_global" in sp else None)
            enc_macro = (torch.from_numpy(sp["encoder_macro"])
                         if "encoder_macro" in sp else None)
            with torch.no_grad():
                logits, _v = policy(
                    torch.from_numpy(sp["global"]),
                    torch.from_numpy(sp["macro"]),
                    torch.from_numpy(sp["candidates"]),
                    torch.from_numpy(sp["chain"]),
                    encoder_global=enc_global,
                    encoder_macro=enc_macro,
                )
                action = int(_F.softmax(logits, dim=-1).argmax().item())
            _r, _d, info = env.step(action, sp["raw_candidates"])
            if _d:
                final_info = info

        # env's _finalize already restored state.pos to the best-prefix snapshot
        # (A.1) under the lex (overlap_pairs, overlap_area, hpwl) gate (A.2),
        # so st.hpwl() / st.overlap_pairs() now reflect the committed state.
        return {
            "chain_gain": float(final_info.get("hpwl_gain", 0.0)),
            "committed": bool(final_info.get("committed", False)),
            "length": int(final_info.get("chain_length", env.chain_length)),
            "overlap_delta": int(final_info.get("overlap_delta", 0)),
            "best_prefix_index": int(final_info.get("best_prefix_index", 0)),
        }

    def run(self, max_length: int = 8) -> dict:
        """Run with policy if available, else fall back to greedy."""
        if self.policy_bundle is not None:
            return self.run_with_policy(max_length)
        return self.run_greedy(max_length)

    def run_greedy(self, max_length: int = 8) -> dict:
        st = self.state
        if not st.movable[self.seed]:
            return {"chain_gain": 0.0, "committed": False, "length": 0,
                    "overlap_delta": 0, "best_prefix_index": 0}

        snapshot = st.pos.copy()
        start_hpwl = st.hpwl()
        start_overlap_pairs = st.overlap_pairs()
        start_overlap_area = st.overlap_area()

        # Best-prefix tracking (A.1): the canonical Lin-Kernighan gain
        # criterion commits the lex-best PREFIX of the cascade, not the
        # endpoint. Lex order on (overlap_pairs, overlap_area, third) is
        # the gate. Third coordinate depends on gate_mode + reg_weight:
        #   - "hpwl"             -> st.hpwl() [+ reg_weight * reg_norm]
        #   - "predicted_proxy"  -> cumulative Δproxy [+ reg_weight * reg_norm]
        #   - "scalar_penalty"   -> single scalar via ScalarBestTracker
        use_scalar_gate = (self.gate_mode == "scalar_penalty")
        if use_scalar_gate:
            from commit_gate_scalar import ScalarBestTracker
            scalar_tracker = ScalarBestTracker(
                start_overlap_pairs=start_overlap_pairs,
                start_pos_snapshot=snapshot,
                lam=self.lam,
            )
            best_key = (start_overlap_pairs, start_overlap_area, 0.0)
            best_pos_snapshot = snapshot
            best_prefix_index = 0
        else:
            scalar_tracker = None
            start_third = self._compute_third(st, predicted_cumulative=0.0)
            start_key = (start_overlap_pairs, start_overlap_area, start_third)
            best_key = start_key
            best_pos_snapshot = snapshot
            best_prefix_index = 0
        predicted_cumulative = 0.0

        current = self.seed
        length = 0
        # A.3: track visited macros so the cascade can't oscillate between
        # the same two macros under combined-score follow.
        visited: set[int] = {self.seed}

        for step in range(max_length):
            # A.3: candidate_positions returns up to 16 (1 centroid + 4
            # connected-partner offsets + 7 grid jitter + 4 random jumps).
            # Pre-fix the inner loop asked for 12, which deterministically
            # truncated *all* random jumps because they're appended last —
            # the chain lost its canvas-jump diversifier. Ask for 16 so the
            # random jumps are preserved; per-candidate cost is O(degree+N)
            # so the 33% extra calls is bounded.
            cands = st.candidate_positions(current, num_candidates=16,
                                            rng=self.rng,
                                            use_wiremask=self.use_wiremask,
                                            use_position_mask=self.use_position_mask)
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

            best_predicted_delta = 0.0  # used iff gate uses predicted Δproxy
            if self.approx is not None:
                # Feature schema follows the loaded checkpoint (MaskRegulate
                # 17-dim vs legacy 16-dim). In session encoder mode we
                # further wrap with [macro_emb; global_vec; hand_feats]
                # using the chain-level encoder cache.
                use_reg_feat = bool(self.approx.get("use_reg_feature", False))
                hand_feats_list = [
                    _features_for_move(
                        st, current,
                        np.array([cx - old_x, cy - old_y], dtype=np.float64),
                        use_reg_feature=use_reg_feat,
                    )
                    for (cx, cy) in cands
                ]
                if self.feature_mode == "encoder" and self.encoder_cache is not None:
                    per_node = self.encoder_cache["per_node"]
                    graph_vec = self.encoder_cache["graph_vec"]
                    macro_emb = per_node[current].detach().numpy().astype(np.float32)
                    global_emb = graph_vec.detach().numpy().astype(np.float32)
                    feats = np.stack([
                        np.concatenate([macro_emb, global_emb, hf.astype(np.float32)])
                        for hf in hand_feats_list
                    ])
                else:
                    feats = np.stack(hand_feats_list)
                norm = (feats - self.approx["feat_mean"]) / self.approx["feat_std"]
                with torch.no_grad():
                    pred = self.approx["model"](torch.tensor(norm, dtype=torch.float32))
                pred_delta = pred.numpy() * self.approx["target_std"] + self.approx["target_mean"]

                # Safety floor: candidates that would introduce extra hard-
                # macro overlaps get penalized in the same units as proxy
                # cost. Use incremental apply_move/revert (B.3).
                for k, (cx, cy) in enumerate(cands):
                    st.apply_move(current, cx, cy)
                    new_ov = len(st.overlapping_with(current))
                    st.apply_move(current, old_x, old_y)
                    score = float(pred_delta[k]) + 0.5 * new_ov
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy)
                        best_predicted_delta = float(pred_delta[k])
            else:
                # B.3: per-candidate apply + revert. The HPWL read is O(1)
                # cached and apply_move is O(degree)+O(N) — pre-fix this
                # was 12 * (O(E) + O(N^2)) per chain step.
                for (cx, cy) in cands:
                    st.apply_move(current, cx, cy)
                    new_ov = len(st.overlapping_with(current))
                    score = st.hpwl() + new_ov * hpwl_scale
                    st.apply_move(current, old_x, old_y)
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy)

            if best_pos is None:
                break

            # Apply best move (B.3: cache-aware).
            st.apply_move(current, best_pos[0], best_pos[1])
            length += 1
            if self.gate_mode in ("predicted_proxy", "scalar_penalty"):
                predicted_cumulative += best_predicted_delta

            # Best-prefix snapshot (A.1): the state *after* this move is a
            # candidate commit point.
            if use_scalar_gate:
                scalar_tracker.update(
                    st, predicted_delta=best_predicted_delta,
                    chain_step_after_move=length,
                )
            else:
                third = self._compute_third(st, predicted_cumulative)
                cur_key = (st.overlap_pairs(), st.overlap_area(), third)
                if cur_key < best_key:
                    best_key = cur_key
                    best_pos_snapshot = st.pos.copy()
                    best_prefix_index = length

            # Cascade to a displaced macro, if any. A.3: combined score
            # = manhattan_dist / (connectivity_weight + eps). Lower is
            # better; we prefer displaced macros that are both close *and*
            # strongly connected, rather than pure-Manhattan (which pulls
            # the chain into geometrically nearby but topologically
            # unrelated regions). The visited set blocks oscillation.
            displaced = [j for j in st.overlapping_with(current)
                         if st.movable[j] and j not in visited]
            if not displaced:
                break  # clean landing, chain complete
            eps = 1e-3
            scores = []
            for j in displaced:
                dxj = abs(float(st.pos[j, 0] - st.pos[current, 0]))
                dyj = abs(float(st.pos[j, 1] - st.pos[current, 1]))
                w = st.neighbor_weight[current].get(j, 0.0)
                scores.append((dxj + dyj) / (w + eps))
            current = int(displaced[int(np.argmin(scores))])
            visited.add(current)

        # Best-prefix commit (A.1 + A.2): restore the lex-best prefix on
        # the lex tuple. If no prefix beat start, full revert (worst case =
        # current behavior). Bulk pos[:]= writes bypass apply_move so we
        # rebuild_caches() afterward (B.3 contract). chain_gain is reported
        # in actual HPWL units regardless of gate_mode for monitoring.
        if use_scalar_gate:
            committed = scalar_tracker.commit_best(st)
            if committed:
                end_overlap_pairs = st.overlap_pairs()
                return {"chain_gain": start_hpwl - st.hpwl(), "committed": True,
                        "length": length,
                        "overlap_delta": end_overlap_pairs - start_overlap_pairs,
                        "best_prefix_index": scalar_tracker.best_prefix_index}
            # commit_best already restored snapshot when no prefix won; just
            # report the no-op outcome.
            return {"chain_gain": 0.0, "committed": False, "length": length,
                    "overlap_delta": 0, "best_prefix_index": 0}

        if best_prefix_index > 0:
            st.pos[:] = best_pos_snapshot
            st.rebuild_caches()
            end_overlap_pairs = best_key[0]
            return {"chain_gain": start_hpwl - st.hpwl(), "committed": True,
                    "length": length,
                    "overlap_delta": end_overlap_pairs - start_overlap_pairs,
                    "best_prefix_index": best_prefix_index}

        st.pos[:] = snapshot
        st.rebuild_caches()
        return {"chain_gain": 0.0, "committed": False, "length": length,
                "overlap_delta": 0, "best_prefix_index": 0}


# ── Analytical warm start (Step 2: warm-start ablation) ────────────────────

def _analytical_warmstart(state: PlacementState, *, n_iters: int = 40,
                          anchor_pull: float = 1.0) -> np.ndarray:
    """Pure-numpy quadratic / force-directed global placement.

    Step 2 of the "what's holding us back" plan: the placer currently only
    *refines* the challenge-provided ``initial.plc``, which already sits near
    proxy 1.46. Refining a near-optimal placement is a different (and
    lower-headroom) problem than placing from scratch — so we need a way to
    seed the chain loop from something other than ``initial.plc`` to tell
    whether the method has real optimization power or is only polishing.

    This is the analytical-warm-start arm (RePlAce/DREAMPlace are external
    tools that can't run under the judges' ``--network none`` image; this is
    a self-contained stand-in in their spirit). Each movable hard macro is
    relaxed (Jacobi sweeps) toward the weighted centroid of (a) its connected
    hard-macro neighbours and (b) its fixed terminals (soft macros + IO
    ports), using the same HPWL-surrogate edge weights the chain loop uses.
    The fixed anchors are the boundary conditions that stop everything from
    collapsing to a single point — exactly the role of fixed pins in classic
    quadratic placement.

    Returns a new ``[N, 2]`` position array. The result generally has
    overlaps (analytical placers ignore them); the chain loop's overlap-aware
    commit gate and the final ``_legalize`` pass drive overlaps back to zero,
    so callers should ``rebuild_caches()`` and run the loop as usual.
    """
    n = state.n
    pos = state.pos.copy()
    movable = state.movable
    if n == 0:
        return pos

    # Pre-extract per-macro fixed-anchor contributions (constant across sweeps).
    hf_anchor_x = [np.zeros(0) for _ in range(n)]
    hf_anchor_y = [np.zeros(0) for _ in range(n)]
    hf_w = [np.zeros(0) for _ in range(n)]
    for i in range(n):
        hf_idx = state.macro_hf_indices[i]
        if len(hf_idx) > 0:
            hf_anchor_x[i] = state.hf_anchors[hf_idx, 0]
            hf_anchor_y[i] = state.hf_anchors[hf_idx, 1]
            hf_w[i] = state.hf_weights[hf_idx]

    for _ in range(n_iters):
        new_pos = pos.copy()
        for i in range(n):
            if not movable[i]:
                continue
            ax = ay = wsum = 0.0
            # Connected hard-macro neighbours (current positions).
            nw = state.neighbor_weight[i]
            if nw:
                for j, w in nw.items():
                    ax += w * pos[j, 0]
                    ay += w * pos[j, 1]
                    wsum += w
            # Fixed terminals (soft macros + IO ports).
            if len(hf_w[i]) > 0:
                wf = hf_w[i]
                ax += anchor_pull * float((wf * hf_anchor_x[i]).sum())
                ay += anchor_pull * float((wf * hf_anchor_y[i]).sum())
                wsum += anchor_pull * float(wf.sum())
            if wsum > 1e-12:
                nx, ny = state.clamp(i, ax / wsum, ay / wsum)
                new_pos[i, 0] = nx
                new_pos[i, 1] = ny
        pos = new_pos
    return pos


# ── Submission placer ──────────────────────────────────────────────────────

class LKHPlacer:
    """Run many random-seed LK chains under a wall-clock budget.

    Uses the learned ChainPolicy when ``checkpoints/chain_policy.pt`` exists,
    else the learned CostApproximator when ``checkpoints/cost_approximator.pt``
    exists, else a pure HPWL+overlap surrogate. Phase 6 inference adds a
    stagnation counter that triggers a random perturbation of K random
    macros when no chain has improved the best-seen placement in a while.
    """

    def __init__(self, seed: int = 42, time_budget_s: float = 60.0,
                 max_chains: int = 5000, max_chain_length: int = 8,
                 checkpoint_path: str | None = None,
                 policy_path: str | None = None,
                 stagnation_threshold: int = 50,
                 perturb_macros: int = 5,
                 use_policy: bool = True,
                 legalization_reserve_s: float | None = None,
                 legalization_reserve_frac: float = 0.05,
                 seed_refresh_interval: int = 100,
                 gate_mode: str = "hpwl",
                 # MaskRegulate opt-ins
                 reg_weight: float = 0.0,
                 use_wiremask: bool = False,
                 use_position_mask: bool = False,
                 encoder=None,
                 # Where the chain loop starts from:
                 #   "initial"    — initial.plc (default).
                 #   "random"     — random valid positions.
                 #   "analytical" — built-in quadratic warm-start (RePlAce-like).
                 #   "replace"    — load a pre-computed .plc from ``seeds_dir``
                 #                  (e.g. external RePlAce/DREAMPlace output).
                 init_mode: str = "initial",
                 seeds_dir: str | None = None,
                 # Session building-block opt-ins
                 feature_mode: str = "handcrafted",
                 encoder_kind: str = "gnn",
                 encoder_path: str | None = None,
                 seed_mode: str = "heuristic",
                 lam: float = 0.01):
        if init_mode not in ("initial", "random", "analytical", "replace"):
            raise ValueError(
                f"unknown init_mode={init_mode!r}; expected 'initial', "
                f"'random', 'analytical', or 'replace'"
            )
        self.init_mode = init_mode
        self.seeds_dir = seeds_dir
        self.seed = seed
        self.time_budget_s = time_budget_s
        self.max_chains = max_chains
        self.max_chain_length = max_chain_length
        self.stagnation_threshold = stagnation_threshold
        self.perturb_macros = perturb_macros
        # A.4 (LKH critique fix): explicit legalization budget.
        # ``legalization_reserve_s=None`` means "probe and choose adaptively"
        # — the chain loop cuts off early enough for ``_legalize`` to finish
        # within ``time_budget_s``. An explicit value disables the probe.
        self.legalization_reserve_s = legalization_reserve_s
        self.legalization_reserve_frac = legalization_reserve_frac
        # A.5 (LKH critique fix): seed-weight staleness. Pre-fix the weights
        # were a static snapshot of HPWL contribution at place() entry; the
        # placement drifts across thousands of chains so the ranking goes
        # stale. Refresh every ``seed_refresh_interval`` chains and after
        # every kick.
        self.seed_refresh_interval = seed_refresh_interval
        # E.1: commit-gate mode propagated to every LKChain spawned by this
        # placer. ``"hpwl"`` is the Milestone-A default (no retrain needed);
        # ``"predicted_proxy"`` and ``"scalar_penalty"`` (Fix 5) require an
        # approximator.
        self.gate_mode = gate_mode
        # MaskRegulate Task 1b: regularity blend weight (0.0 = unchanged).
        self.reg_weight = float(reg_weight)
        # MaskRegulate Task 2: WireMask candidate injection.
        self.use_wiremask = bool(use_wiremask)
        # MaskRegulate Task 3: legality filter on WireMask candidates.
        self.use_position_mask = bool(use_position_mask)
        # Task 4b: optional StateEncoder propagated to every LKChain spawn.
        # None = legacy path (no encoder branch in the policy forward).
        self.encoder = encoder

        # Fix 2: feature pipeline selection.
        if feature_mode not in ("handcrafted", "encoder"):
            raise ValueError(
                f"unknown feature_mode={feature_mode!r}; expected "
                f"'handcrafted' or 'encoder'"
            )
        if encoder_kind not in ("gnn", "gnncnn"):
            raise ValueError(
                f"unknown encoder_kind={encoder_kind!r}; expected "
                f"'gnn' or 'gnncnn'"
            )
        self.feature_mode = feature_mode
        self.encoder_kind = encoder_kind

        # Fix 3: who picks the seed macro.
        if seed_mode not in ("heuristic", "policy"):
            raise ValueError(
                f"unknown seed_mode={seed_mode!r}; expected 'heuristic' or 'policy'"
            )
        self.seed_mode = seed_mode

        # Fix 5: scalar-gate overlap weight.
        self.lam = float(lam)

        approx_ckpt = Path(checkpoint_path) if checkpoint_path else (
            _HERE / "checkpoints" / "cost_approximator.pt"
        )
        self.approximator = _load_cost_approximator(approx_ckpt)
        if self.approximator is not None:
            r = self.approximator.get("pearson_r_val")
            trained_on = self.approximator.get("trained_on")
            print(f"[LKHPlacer] cost approximator loaded "
                  f"(r={r:.3f}, trained on {trained_on})")
        else:
            print(f"[LKHPlacer] using HPWL surrogate")

        # Fix 2: optionally load the encoder. Only attempted when the user
        # asks for encoder feature mode — saves load time + memory in default.
        self.encoder_bundle = None
        if self.feature_mode == "encoder":
            enc_ckpt = Path(encoder_path) if encoder_path else (
                _HERE / "checkpoints" / "encoder.pt"
            )
            self.encoder_bundle = _load_encoder(enc_ckpt, kind=self.encoder_kind)
            if self.encoder_bundle is not None:
                print(f"[LKHPlacer] encoder loaded")
            else:
                print(f"[LKHPlacer] using default features")
                self.feature_mode = "handcrafted"

        self.policy_bundle = None
        if use_policy:
            policy_ckpt = Path(policy_path) if policy_path else (
                _HERE / "checkpoints" / "chain_policy.pt"
            )
            self.policy_bundle = _load_chain_policy(policy_ckpt)
            if self.policy_bundle is not None:
                print(f"[LKHPlacer] chain policy loaded "
                      f"(iters={self.policy_bundle.get('n_iterations')}, "
                      f"trained on {self.policy_bundle.get('trained_on')})")
                # MaskRegulate: attach co-trained StateEncoder if present.
                if self.encoder is None and self.policy_bundle.get("encoder") is not None:
                    self.encoder = self.policy_bundle["encoder"]
                    print(f"[LKHPlacer] using co-trained StateEncoder")
                if self.policy_bundle.get("seed_head") is not None:
                    print(f"[LKHPlacer] seed head loaded")
            else:
                print(f"[LKHPlacer] using greedy chains")

    def _perturb(self, state: PlacementState, rng: random.Random,
                 seed_weights: list[float] | None = None,
                 movable_idx: list[int] | None = None) -> None:
        """Move ``perturb_macros`` movable macros to random positions. A.5
        (LKH critique fix): bias the picks toward the top-quartile of
        ``seed_weights`` (the highest current HPWL-contribution macros) so
        the kick targets stuck/high-cost regions instead of arbitrary ones.
        Falls back to uniform-random when seed weights aren't supplied."""
        if movable_idx is None:
            movable_idx = np.where(state.movable)[0].tolist()
        if not movable_idx:
            return
        if seed_weights is not None and len(seed_weights) == len(movable_idx):
            # Restrict to the top-quartile of the seed-weight distribution.
            order = np.argsort(seed_weights)[::-1]
            cutoff = max(1, len(movable_idx) // 4)
            pool = [movable_idx[k] for k in order[:cutoff].tolist()]
        else:
            pool = movable_idx
        for _ in range(self.perturb_macros):
            i = int(rng.choice(pool))
            x = rng.uniform(state.half_w[i], state.cw - state.half_w[i])
            y = rng.uniform(state.half_h[i], state.ch - state.half_h[i])
            # B.3: route the random kick through apply_move so the
            # incremental caches stay consistent across the perturbation.
            state.apply_move(i, x, y)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        rng = random.Random(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        hpwl_edges = _hard_macro_edges(benchmark)
        state = PlacementState(benchmark, hpwl_edges)

        movable_idx = np.where(state.movable)[0].tolist()
        if not movable_idx:
            return benchmark.macro_positions.clone()

        # init_mode controls where the chain loop starts from. Bulk pos
        # writes bypass apply_move so we rebuild_caches() afterwards (B.3).
        if self.init_mode == "replace":
            # Load a pre-computed .plc from seeds_dir (e.g. external
            # RePlAce/DREAMPlace output). Falls back loudly if missing.
            from seed_loader import load_seed_positions
            seed_positions = load_seed_positions(
                benchmark, seeds_dir=self.seeds_dir
            )
            if seed_positions is None:
                print(f"[LKHPlacer] using default seed")
            else:
                state.pos[:] = seed_positions[: state.n]
                state.rebuild_caches()
                print(f"[LKHPlacer] seed loaded")
        elif self.init_mode == "random":
            for i in movable_idx:
                state.pos[i, 0] = rng.uniform(state.half_w[i],
                                              state.cw - state.half_w[i])
                state.pos[i, 1] = rng.uniform(state.half_h[i],
                                              state.ch - state.half_h[i])
            state.rebuild_caches()
            print(f"[LKHPlacer] init_mode=random: scattered {len(movable_idx)} "
                  f"movable macros (overlap_pairs={state.overlap_pairs()})")
        elif self.init_mode == "analytical":
            state.pos[:] = _analytical_warmstart(state)
            state.rebuild_caches()
            print(f"[LKHPlacer] init_mode=analytical: quadratic warm start "
                  f"(hpwl={state.hpwl():.2f}, overlap_pairs={state.overlap_pairs()})")

        # Cost-weighted seed sampling: macros with highest neighbor-displacement
        # are sampled more often. Falls back to uniform if no edges.
        seed_weights = self._seed_weights(state, movable_idx)

        # Fix 3: optionally use the learned seed-selection head. Requires
        # both feature_mode="encoder" (so we have embeddings) and a policy
        # bundle that includes a seed head.
        seed_head = None
        if (self.seed_mode == "policy"
                and self.policy_bundle is not None
                and self.policy_bundle.get("seed_head") is not None
                and self.feature_mode == "encoder"
                and self.encoder_bundle is not None):
            seed_head = self.policy_bundle["seed_head"]
            print(f"[LKHPlacer] learned seed selection enabled")
        elif self.seed_mode == "policy":
            print(f"[LKHPlacer] using default seed selection")

        # A.4: reserve wall time for the post-loop _legalize step. If the
        # caller didn't pin a value, probe the spiral on the 10 largest
        # movable macros and pad by 1.5x. Floor at frac * budget so a
        # mis-estimating probe can't starve legalization.
        if self.legalization_reserve_s is not None:
            legalization_reserve = float(self.legalization_reserve_s)
        else:
            probe_estimate = _probe_legalization_cost(state, n_probe=10)
            # Tier-1 Fix D: density-aware scaling. The probe runs a fixed
            # 3-ring sweep on 10 macros — calibrated for sparse states. On
            # dense designs (ibm17/18 routinely hit chain-loop end with 50+
            # overlap pairs) real legalization needs up to 150 rings per
            # macro, so the 3-ring probe undershoots by ~10x. Without this
            # scaling, ibm17 wall time hit 127s on a 60s budget — chain
            # loop ran its 57s, then legalization ran free for ~70s.
            #
            # Heuristic: count initial overlap pairs and ramp the reserve
            # linearly above 20 pairs, capped at 8x. ibm01 typically has
            # ~70 initial pairs → 3.5x; ibm17 ~150-300 → 7-8x.
            initial_overlaps = state.overlap_pairs()
            density_multiplier = max(1.0, min(8.0, initial_overlaps / 20.0))
            floor = self.legalization_reserve_frac * self.time_budget_s
            legalization_reserve = max(
                probe_estimate * 1.5 * density_multiplier, floor, 1.0
            )
            print(f"[LKHPlacer] legalization reserve: probe={probe_estimate:.2f}s "
                  f"× density {density_multiplier:.1f}x "
                  f"(initial_overlaps={initial_overlaps}) "
                  f"-> reserved {legalization_reserve:.2f}s of "
                  f"{self.time_budget_s:.1f}s budget")
        # Cap the reserve at half the budget so the chain loop still gets
        # meaningful time even on a wildly mis-estimating probe.
        legalization_reserve = min(legalization_reserve, self.time_budget_s * 0.5)
        chain_loop_deadline = self.time_budget_s - legalization_reserve

        start = time.time()
        chains = 0
        best_pos = state.pos.copy()
        # A.2: outer-loop best-tracking must match the chain-internal lex
        # gate `(overlap_pairs, overlap_area, third)`. With the pre-A.2
        # 2-tuple, a chain that reduced `overlap_area` at slight HPWL cost
        # would commit (lex-better in 3-tuple) but the outer would prefer
        # an earlier higher-area state with lower HPWL — and legalization
        # would then have to undo a larger overlap, regressing the very
        # invariant A.2 was designed to protect.
        # Task 1b: when reg_weight > 0, the outer third blends HPWL with
        # regularity (canvas-normalized). Independent of gate_mode — the
        # outer compares across chains and needs a state-level scalar,
        # not the per-chain cumulative-Δproxy used inside predicted_proxy.
        _hpwl_scale = max(state.cw, state.ch)
        _reg_scale = state.cw + state.ch
        def _outer_third() -> float:
            if self.reg_weight <= 0.0:
                return state.hpwl()
            return (state.hpwl() / _hpwl_scale
                    + self.reg_weight * state.total_regularity() / _reg_scale)
        best_key = (state.overlap_pairs(), state.overlap_area(), _outer_third())
        stagnation = 0

        while chains < self.max_chains and (time.time() - start) < chain_loop_deadline:
            # Fix 2: refresh encoder cache once per chain. The same per-node
            # + graph embeddings are reused for the chain's seed selection
            # and every candidate-scoring decision inside the chain.
            encoder_cache = None
            if self.feature_mode == "encoder" and self.encoder_bundle is not None:
                encoder = self.encoder_bundle["encoder"]
                if self.encoder_kind == "gnn":
                    from encoder_runtime import encode_state_gnn
                    per_node, graph_vec = encode_state_gnn(state, encoder, with_grad=False)
                else:
                    from encoder_runtime_gnncnn import encode_state_gnncnn
                    per_node, graph_vec = encode_state_gnncnn(state, encoder, with_grad=False)
                encoder_cache = {"per_node": per_node, "graph_vec": graph_vec}

            # Fix 3: seed pick. Either heuristic (default) or policy-driven.
            if seed_head is not None and encoder_cache is not None:
                from seed_head import choose_seed_by_policy
                movable_mask_arr = state.movable
                macro_idx, _logp = choose_seed_by_policy(
                    encoder_cache["per_node"], encoder_cache["graph_vec"],
                    movable_mask_arr, seed_head,
                    stochastic=False, rng=rng,
                )
                seed_macro = int(macro_idx)
            else:
                seed_macro = rng.choices(movable_idx, weights=seed_weights, k=1)[0]

            result = LKChain(state, seed_macro, rng,
                              approximator=self.approximator,
                              policy_bundle=self.policy_bundle,
                              gate_mode=self.gate_mode,
                              reg_weight=self.reg_weight,
                              use_wiremask=self.use_wiremask,
                              use_position_mask=self.use_position_mask,
                              encoder=self.encoder,
                              feature_mode=self.feature_mode,
                              encoder_cache=encoder_cache,
                              lam=self.lam,
                              ).run(self.max_chain_length)
            chains += 1
            if result["committed"]:
                cur_key = (state.overlap_pairs(), state.overlap_area(),
                           _outer_third())
                if cur_key < best_key:
                    best_key = cur_key
                    best_pos = state.pos.copy()
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1

            # A.5: refresh seed weights every ``seed_refresh_interval``
            # chains so the seed-sampling distribution tracks current
            # placement, not the stale initial-placement HPWL contributions.
            if (self.seed_refresh_interval > 0
                    and chains % self.seed_refresh_interval == 0):
                seed_weights = self._seed_weights(state, movable_idx)

            # Phase 6.1: kick on stagnation. A.5: bias the kick toward
            # top-quartile-cost macros, then refresh seed weights so the
            # post-kick chain loop resumes with a non-stale ranking.
            if stagnation >= self.stagnation_threshold:
                self._perturb(state, rng, seed_weights=seed_weights,
                              movable_idx=movable_idx)
                seed_weights = self._seed_weights(state, movable_idx)
                stagnation = 0

        state.pos[:] = best_pos
        state.rebuild_caches()

        # Phase 6 finishing — legalize residual hard-macro overlaps to zero.
        # This is the difference between an "invalid" submission (overlap_count
        # > 0, disqualified) and a valid one. The LK commit gate prevents
        # *adding* overlaps but cannot drive them to zero unaided.
        n_overlaps_before = state.overlap_pairs()
        if n_overlaps_before > 0:
            state.pos[:] = _legalize(state)
            state.rebuild_caches()
            n_overlaps_after = state.overlap_pairs()
            print(f"[LKHPlacer] legalization: "
                  f"{n_overlaps_before} -> {n_overlaps_after} overlap pairs")

        # Build full placement: hard macros from LK, soft macros untouched.
        full = benchmark.macro_positions.clone()
        full[: state.n] = torch.tensor(state.pos, dtype=torch.float32)
        return full

    @staticmethod
    def _seed_weights(state: PlacementState, movable_idx: list[int]) -> list[float]:
        """Higher weight for macros with more / longer connections (likely
        to benefit most from a move). C.1: also accounts for hard-to-fixed
        edges so macros connected mainly to soft macros / IO ports are
        sampled in proportion to their actual HPWL contribution; pre-C.1
        such a macro had effectively zero seed weight."""
        if len(state.edges) == 0 and len(state.hf_macros) == 0:
            return [1.0] * len(movable_idx)

        score = np.zeros(state.n, dtype=np.float64)
        if len(state.edges) > 0:
            ex = np.abs(state.pos[state.edges[:, 0], 0] - state.pos[state.edges[:, 1], 0])
            ey = np.abs(state.pos[state.edges[:, 0], 1] - state.pos[state.edges[:, 1], 1])
            d = state.edge_weights * (ex + ey)
            np.add.at(score, state.edges[:, 0], d)
            np.add.at(score, state.edges[:, 1], d)
        if len(state.hf_macros) > 0:
            fx = np.abs(state.pos[state.hf_macros, 0] - state.hf_anchors[:, 0])
            fy = np.abs(state.pos[state.hf_macros, 1] - state.hf_anchors[:, 1])
            df = state.hf_weights * (fx + fy)
            np.add.at(score, state.hf_macros, df)
        w = [max(float(score[i]), 1e-6) for i in movable_idx]
        return w
