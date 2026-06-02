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

    hh_edges: np.ndarray  # [E_hh, 2] hard-macro-to-hard-macro pairs
    hh_weights: np.ndarray  # [E_hh] per-pair weights
    hf_macros: np.ndarray  # [E_hf] hard-macro indices for each hard-to-fixed contribution
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
        nets_owners = (sorted({int(o) for o in nodes.tolist()}) for nodes in benchmark.net_nodes)

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
        base = (
            Path("external/MacroPlacement/Flows/NanGate45")
            / design
            / "netlist"
            / "output_CT_Grouping"
        )
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

    def __init__(
        self,
        benchmark: Benchmark,
        edges,
        edge_weights=None,
        hf_macros=None,
        hf_anchors=None,
        hf_weights=None,
    ):
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
            hh_weights = edge_weights if edge_weights is not None else np.zeros(0, dtype=np.float64)
            hf_macros = hf_macros if hf_macros is not None else np.zeros(0, dtype=np.int64)
            hf_anchors = (
                hf_anchors if hf_anchors is not None else np.zeros((0, 2), dtype=np.float64)
            )
            hf_weights = hf_weights if hf_weights is not None else np.zeros(0, dtype=np.float64)

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
        self._overlap_partners: list[dict[int, float]] = [{} for _ in range(n_hard)]
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
            is_overlap = (dx + self._DEFAULT_GAP < self.sep_x) & (
                dy + self._DEFAULT_GAP < self.sep_y
            )
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

    def apply_move(self, idx: int, new_x: float, new_y: float) -> tuple[float, float]:
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
        is_overlap_row = (dx_row + self._DEFAULT_GAP < sep_x_row) & (
            dy_row + self._DEFAULT_GAP < sep_y_row
        )
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
        assert np.isclose(
            cached_hpwl, true_hpwl, rtol=rtol, atol=atol
        ), f"HPWL cache drift: cached={cached_hpwl} true={true_hpwl}"

        if self.n > 1:
            dx = np.abs(self.pos[:, 0:1] - self.pos[:, 0:1].T)
            dy = np.abs(self.pos[:, 1:2] - self.pos[:, 1:2].T)
            is_overlap = (dx + self._DEFAULT_GAP < self.sep_x) & (
                dy + self._DEFAULT_GAP < self.sep_y
            )
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
        assert (
            cached_pairs == true_pairs
        ), f"overlap_pairs cache drift: cached={cached_pairs} true={true_pairs}"
        assert np.isclose(
            cached_area, true_area, rtol=rtol, atol=atol
        ), f"overlap_area cache drift: cached={cached_area} true={true_area}"

    # ── Candidate generation ──────────────────────────────────────────────

    def count_within(self, idx: int, x: float, y: float, radius: float) -> int:
        """Count macro centers (excluding ``idx``) within an axis-aligned box
        of half-side ``radius`` around (x, y). Cheap proxy for local density."""
        dx = np.abs(self.pos[:, 0] - x)
        dy = np.abs(self.pos[:, 1] - y)
        hits = (dx < radius) & (dy < radius)
        hits[idx] = False
        return int(hits.sum())

    def candidate_positions(
        self, idx: int, num_candidates: int = 16, rng: random.Random | None = None
    ) -> list[tuple[float, float]]:
        """Candidate centers for macro ``idx``: connected centroid + top-K
        connected partners + grid jitter + random jumps.

        A.3 (LKH critique fix): the pre-fix mix was 1 connected centroid +
        12 grid jitter + 4 random — only 1/16 carried connectivity signal.
        We now spend 4 of the 16 slots on positions stepping toward each of
        the macro's top-4 most strongly-connected partners (alpha-nearness
        analog), keeping 7 grid-jitter and 4 random."""
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
            top_partners = sorted(self.neighbor_weight[idx].items(), key=lambda kv: -kv[1])[:4]
            for partner_j, _w in top_partners:
                px = float(self.pos[partner_j, 0])
                py = float(self.pos[partner_j, 1])
                vdx = px - cur_x
                vdy = py - cur_y
                norm = (vdx * vdx + vdy * vdy) ** 0.5
                if norm < 1e-9:
                    continue
                cands.append(
                    self.clamp(
                        idx,
                        cur_x + vdx * step / norm,
                        cur_y + vdy * step / norm,
                    )
                )

        # Local grid jitter steps proportional to macro size. Trimmed from
        # 12 to 7 (kept 4 wide cardinals + 3 narrow cardinals; dropped 4
        # diagonals and 1 narrow cardinal to make room for the connected-
        # partner offsets above).
        for dxm, dym in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, 0), (1, 0), (0, -1)]:
            x, y = self.clamp(idx, cur_x + dxm * step, cur_y + dym * step)
            cands.append((x, y))

        # Random canvas jumps (open-region proxy without expensive search).
        for _ in range(4):
            x, y = self.clamp(
                idx,
                r.uniform(self.half_w[idx], self.cw - self.half_w[idx]),
                r.uniform(self.half_h[idx], self.ch - self.half_h[idx]),
            )
            cands.append((x, y))

        return cands[:num_candidates]


# ── Feature extraction (for Phase 3 cost approximator) ────────────────────

FEATURE_DIM = 16


def _features_for_move(
    state: "PlacementState", macro_idx: int, move_delta: np.ndarray
) -> np.ndarray:
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

    feats = np.array(
        [
            w,
            h,
            w * h,  # macro geometry
            float(move_delta[0]),
            float(move_delta[1]),
            abs(float(move_delta[0])),
            abs(float(move_delta[1])),
            new_x / state.cw,
            new_y / state.ch,  # normalized target
            hpwl_after - hpwl_before,  # HPWL Δ (surrogate)
            float(n_overlap_old),
            float(n_overlap_new),  # local overlap counts
            float(n_local_old),
            float(n_local_new),  # local density
            attract_old - attract_new,  # +ve = pulled toward neighbors
            float(deg),  # macro degree
        ],
        dtype=np.float32,
    )
    assert feats.shape == (FEATURE_DIM,), f"feature dim drift: {feats.shape}"
    return feats


# ── Legalization (Phase 6 finishing step) ──────────────────────────────────


def _legalize(
    state: PlacementState, *, gap: float = 0.001, max_search_radius: int = 150
) -> np.ndarray:
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
                    cx = float(np.clip(pos[idx, 0] + dxm * step, half_w[idx], cw - half_w[idx]))
                    cy = float(np.clip(pos[idx, 1] + dym * step, half_h[idx], ch - half_h[idx]))
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
    order = sorted(
        movable_idx.tolist(), key=lambda i: -float(state.sizes[i, 0] * state.sizes[i, 1])
    )
    sample = order[: min(n_probe, len(order))]

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
        from submissions.lkh.model.lkh_model import CostApproximator
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


# ── Chain policy loading (Phase 4) ─────────────────────────────────────────


def _load_chain_policy(ckpt_path: Path):
    """Returns a dict with policy + chain_env helpers, or None if missing."""
    if not ckpt_path.exists():
        return None
    try:
        from submissions.lkh.model.lkh_model import ChainPolicy, ChainPolicyWithGNN
    except ImportError:
        return None
    # Import chain_env via importlib to avoid circular imports.
    # chain_env.py lives in the environment/ subdirectory.
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "chain_env", str(_HERE / "environment" / "chain_env.py")
    )
    env_mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(env_mod)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy_class = ckpt.get("policy_class", "ChainPolicy")
    hidden = ckpt.get("hidden", 64)
    if policy_class == "ChainPolicyWithGNN":
        gnn_H = ckpt.get("gnn_H", 32)
        proj_dim = ckpt.get("proj_dim", 16)
        policy = ChainPolicyWithGNN(hidden=hidden, gnn_H=gnn_H, proj_dim=proj_dim)
    else:
        policy = ChainPolicy(hidden=hidden)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return {
        "policy": policy,
        "ChainEnv": env_mod.ChainEnv,
        "trained_on": ckpt.get("trained_on"),
        "n_iterations": ckpt.get("n_iterations"),
        "policy_class": policy_class,
        "gnn_H": ckpt.get("gnn_H"),
    }


# ── Proxy cost encoder loading (Phase 2.5) ─────────────────────────────────

# Lazy-load encoder helpers on first call to _load_proxy_cost_encoder so
# loading placer.py for a HPWL-only run doesn't pay the torch.nn import cost.
_encoder_helpers: dict | None = None


def _get_encoder_helpers() -> dict:
    global _encoder_helpers
    if _encoder_helpers is not None:
        return _encoder_helpers
    import importlib.util as _ilu

    enc_spec = _ilu.spec_from_file_location("lkh_encoder", str(_HERE / "model" / "encoder.py"))
    enc_mod = _ilu.module_from_spec(enc_spec)
    enc_spec.loader.exec_module(enc_mod)
    _encoder_helpers = {
        "_build_anchors": enc_mod._build_anchors,
        "build_node_features": enc_mod.build_node_features,
        "build_edge_index_and_attr": enc_mod.build_edge_index_and_attr,
        "build_metadata_features": enc_mod.build_metadata_features,
        "rasterize_canvas": enc_mod.rasterize_canvas,
    }
    return _encoder_helpers


def _load_proxy_cost_encoder(ckpt_path: Path):
    """Load a ProxyCostPredictor checkpoint. Returns a bundle dict or None.

    The returned bundle includes:
        model         ProxyCostPredictor (eval mode)
        norm_stats    {bench_name: {wl_mean, wl_std, den_mean, den_std, ...}}
        proxy_weights {wl: α, den: α, cong: α}
        trained_on    list[str]

    If the checkpoint does not carry ``"type": "proxy_cost_encoder"`` it is an
    old MLP checkpoint and is ignored here (handled by _load_cost_approximator).
    """
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("type") != "proxy_cost_encoder":
        return None
    try:
        from submissions.lkh.model.lkh_model import ProxyCostPredictor
    except ImportError:
        return None
    model = ProxyCostPredictor(
        hidden_dim=ckpt["hidden_dim"],
        num_gnn_layers=ckpt["num_gnn_layers"],
        edge_fc_layers=ckpt.get("edge_fc_layers", 1),
        grid_size=ckpt.get("grid_size", 128),
        use_cnn=ckpt.get("use_cnn", True),
    )
    model.encoder.load_state_dict(ckpt["encoder_state"])
    model.proxy_head.load_state_dict(ckpt["head_state"])
    model.eval()
    return {
        "model": model,
        "norm_stats": ckpt.get("norm_stats", {}),
        "proxy_weights": ckpt.get("proxy_weights", {"wl": 1.0, "den": 0.5, "cong": 0.5}),
        "pearson_r_wl": ckpt.get("pearson_r_wl"),
        "pearson_r_den": ckpt.get("pearson_r_den"),
        "pearson_r_cong": ckpt.get("pearson_r_cong"),
        "trained_on": ckpt.get("trained_on"),
    }


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

    def __init__(
        self,
        state: PlacementState,
        seed_macro: int,
        rng: random.Random,
        approximator: dict | None = None,
        policy_bundle: dict | None = None,
        gate_mode: str = "hpwl",
        proxy_encoder: dict | None = None,
    ):
        """``gate_mode`` selects the third lex-coordinate of the commit gate:

        * ``"hpwl"`` (default): ``(overlap_pairs, overlap_area, hpwl)``.
          Milestone-A behavior.
        * ``"predicted_proxy"``: ``(overlap_pairs, overlap_area,
          cumulative_predicted_Δproxy)``. Requires an approximator. Milestone E
          (LKH critique fix §3.3 deferred).
        """
        self.state = state
        self.rng = rng
        self.seed = seed_macro
        self.approx = approximator
        self.policy_bundle = policy_bundle
        self.gate_mode = gate_mode
        # proxy_encoder bundle includes {model, effective_weights, ...} for encoder scoring.
        # effective_weights = {wl: α_wl * wl_std, den: α_den * den_std, cong: α_cong * cong_std}
        # so delta scoring avoids denormalization arithmetic per candidate.
        self.proxy_encoder = proxy_encoder
        if gate_mode not in ("hpwl", "predicted_proxy"):
            raise ValueError(
                f"unknown gate_mode={gate_mode!r}; expected 'hpwl' or 'predicted_proxy'"
            )
        if gate_mode == "predicted_proxy" and approximator is None:
            # Soft fallback: predicted_proxy without an approximator is
            # meaningless, so silently fall back to hpwl. This keeps the
            # ablation harness from blowing up when the user pairs the new
            # gate with a pre-Milestone-C checkpoint that's been removed.
            self.gate_mode = "hpwl"

    def run_with_policy(self, max_length: int = 8) -> dict:
        """Drive the chain with the trained policy via ChainEnv."""
        st = self.state
        if not st.movable[self.seed]:
            return {
                "chain_gain": 0.0,
                "committed": False,
                "length": 0,
                "overlap_delta": 0,
                "best_prefix_index": 0,
            }

        ChainEnv = self.policy_bundle["ChainEnv"]
        policy = self.policy_bundle["policy"]
        env = ChainEnv(
            st,
            self.seed,
            rng=self.rng,
            max_chain_length=max_length,
            max_candidates=8,
            gate_mode=self.gate_mode,
            approximator=self.approx,
        )
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
            with torch.no_grad():
                logits, _v = policy(
                    torch.from_numpy(sp["global"]),
                    torch.from_numpy(sp["macro"]),
                    torch.from_numpy(sp["candidates"]),
                    torch.from_numpy(sp["chain"]),
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
            return {
                "chain_gain": 0.0,
                "committed": False,
                "length": 0,
                "overlap_delta": 0,
                "best_prefix_index": 0,
            }

        snapshot = st.pos.copy()
        start_hpwl = st.hpwl()
        start_overlap_pairs = st.overlap_pairs()
        start_overlap_area = st.overlap_area()

        # Best-prefix tracking (A.1): the canonical Lin-Kernighan gain
        # criterion commits the lex-best PREFIX of the cascade, not the
        # endpoint. Lex order on (overlap_pairs, overlap_area, third) is
        # the gate. Third coordinate depends on gate_mode (E.1):
        #   - "hpwl"             -> st.hpwl()
        #   - "predicted_proxy"  -> cumulative approximator-predicted Δproxy
        #     (starts at 0; each move adds the chosen candidate's pred_delta)
        if self.gate_mode == "predicted_proxy":
            start_third = 0.0
        else:
            start_third = start_hpwl
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

        # Phase 2.5: build encoder inputs once at chain start. Topology
        # (edge_index, edge_attr, metadata) is static; node_feats and canvas
        # are updated after each committed move.
        _enc_nf = _enc_ei = _enc_ea = _enc_canvas = _enc_md = _enc_hp = None
        if self.proxy_encoder is not None:
            _eh = _get_encoder_helpers()
            _enc_anchors, _enc_anchor_inv = _eh["_build_anchors"](st)
            _enc_nf = _eh["build_node_features"](st, _enc_anchors)
            _enc_ei, _enc_ea = _eh["build_edge_index_and_attr"](st, _enc_anchor_inv)
            _enc_md = _eh["build_metadata_features"](st, _enc_anchor_inv)
            _enc_canvas = _eh["rasterize_canvas"](
                st, grid_size=self.proxy_encoder["model"].encoder.grid_size
            )
            _enc_hp = float(st.cw + st.ch)

        for step in range(max_length):
            # A.3: candidate_positions returns up to 16 (1 centroid + 4
            # connected-partner offsets + 7 grid jitter + 4 random jumps).
            # Pre-fix the inner loop asked for 12, which deterministically
            # truncated *all* random jumps because they're appended last —
            # the chain lost its canvas-jump diversifier. Ask for 16 so the
            # random jumps are preserved; per-candidate cost is O(degree+N)
            # so the 33% extra calls is bounded.
            cands = st.candidate_positions(current, num_candidates=16, rng=self.rng)
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

            best_predicted_delta = 0.0  # used iff gate_mode == "predicted_proxy"
            if self.proxy_encoder is not None:
                # Phase 2.5: encoder scoring. Score each candidate by the
                # encoder-predicted delta-proxy (before minus after). Canvas is
                # stale for candidate evaluation (too expensive to recompute per
                # candidate); updated after the committed move.
                enc_model = self.proxy_encoder["model"]
                ew = self.proxy_encoder["effective_weights"]
                n_hard_enc = st.n
                with torch.no_grad():
                    pred_before = enc_model(
                        _enc_nf, _enc_ei, _enc_ea, _enc_canvas, _enc_md, n_hard_enc
                    )
                cost_before = (
                    ew["wl"] * pred_before[0]
                    + ew["den"] * pred_before[1]
                    + ew["cong"] * pred_before[2]
                ).item()
                for cx, cy in cands:
                    nf_after = _enc_nf.clone()
                    nf_after[current, 0] = cx / _enc_hp
                    nf_after[current, 1] = cy / _enc_hp
                    with torch.no_grad():
                        pred_after = enc_model(
                            nf_after, _enc_ei, _enc_ea, _enc_canvas, _enc_md, n_hard_enc
                        )
                    delta = (
                        ew["wl"] * pred_after[0]
                        + ew["den"] * pred_after[1]
                        + ew["cong"] * pred_after[2]
                    ).item() - cost_before
                    st.apply_move(current, cx, cy)
                    new_ov = len(st.overlapping_with(current))
                    st.apply_move(current, old_x, old_y)
                    score = delta + 0.5 * new_ov
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy)
                        best_predicted_delta = delta
            elif self.approx is not None:
                feats = np.stack(
                    [
                        _features_for_move(
                            st,
                            current,
                            np.array([cx - old_x, cy - old_y], dtype=np.float64),
                        )
                        for (cx, cy) in cands
                    ]
                )
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
                for cx, cy in cands:
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
            if self.gate_mode == "predicted_proxy":
                predicted_cumulative += best_predicted_delta

            # Update encoder inputs to reflect committed position (canvas is
            # expensive; recompute it only once per committed move, not per
            # candidate). node_feats update is O(1).
            if self.proxy_encoder is not None:
                _enc_nf[current, 0] = best_pos[0] / _enc_hp
                _enc_nf[current, 1] = best_pos[1] / _enc_hp
                _enc_canvas = _get_encoder_helpers()["rasterize_canvas"](
                    st, grid_size=self.proxy_encoder["model"].encoder.grid_size
                )

            # Best-prefix snapshot (A.1): the state *after* this move is a
            # candidate commit point. If it lex-beats the current best, take
            # a pos snapshot. Only one snapshot is retained (the running
            # winner), so the per-chain memory cost is one extra N-by-2
            # array, not max_length of them.
            if self.gate_mode == "predicted_proxy":
                third = predicted_cumulative
            else:
                third = st.hpwl()
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
            displaced = [
                j for j in st.overlapping_with(current) if st.movable[j] and j not in visited
            ]
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
        if best_prefix_index > 0:
            st.pos[:] = best_pos_snapshot
            st.rebuild_caches()
            end_overlap_pairs = best_key[0]
            return {
                "chain_gain": start_hpwl - st.hpwl(),
                "committed": True,
                "length": length,
                "overlap_delta": end_overlap_pairs - start_overlap_pairs,
                "best_prefix_index": best_prefix_index,
            }

        st.pos[:] = snapshot
        st.rebuild_caches()
        return {
            "chain_gain": 0.0,
            "committed": False,
            "length": length,
            "overlap_delta": 0,
            "best_prefix_index": 0,
        }


# ── Submission placer ──────────────────────────────────────────────────────


class LKHPlacer:
    """Run many random-seed LK chains under a wall-clock budget.

    Uses the learned ChainPolicy when ``checkpoints/chain_policy.pt`` exists,
    else the learned CostApproximator when ``checkpoints/cost_approximator.pt``
    exists, else a pure HPWL+overlap surrogate. Phase 6 inference adds a
    stagnation counter that triggers a random perturbation of K random
    macros when no chain has improved the best-seen placement in a while.
    """

    def __init__(
        self,
        seed: int = 42,
        time_budget_s: float = 60.0,
        max_chains: int = 5000,
        max_chain_length: int = 8,
        checkpoint_path: str | None = None,
        policy_path: str | None = None,
        stagnation_threshold: int = 50,
        perturb_macros: int = 5,
        use_policy: bool = True,
        legalization_reserve_s: float | None = None,
        legalization_reserve_frac: float = 0.05,
        seed_refresh_interval: int = 100,
        gate_mode: str = "hpwl",
    ):
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
        # ``"predicted_proxy"`` requires a cascade-aware approximator from C.
        self.gate_mode = gate_mode

        # Phase 2.5: try loading the proxy cost encoder first. If its checkpoint
        # carries type=="proxy_cost_encoder", it supersedes the MLP approximator
        # for candidate scoring. The MLP is still loaded as a fallback in case
        # the encoder hasn't been trained yet (no checkpoint, or untrained).
        enc_ckpt = (
            Path(checkpoint_path)
            if checkpoint_path
            else (_HERE / "checkpoints" / "proxy_cost_encoder.pt")
        )
        self.proxy_encoder = _load_proxy_cost_encoder(enc_ckpt)
        if self.proxy_encoder is not None:
            r_wl = self.proxy_encoder.get("pearson_r_wl")
            r_den = self.proxy_encoder.get("pearson_r_den")
            r_cong = self.proxy_encoder.get("pearson_r_cong")
            trained_on = self.proxy_encoder.get("trained_on")
            print(
                f"[LKHPlacer] proxy cost encoder loaded "
                f"(r_wl={r_wl}, r_den={r_den}, r_cong={r_cong}, "
                f"trained on {trained_on})"
            )

        approx_ckpt = (
            Path(checkpoint_path)
            if checkpoint_path
            else (_HERE / "checkpoints" / "cost_approximator.pt")
        )
        # Only load MLP approximator if encoder isn't available (it's superseded).
        self.approximator = (
            None if self.proxy_encoder is not None else _load_cost_approximator(approx_ckpt)
        )
        if self.approximator is not None:
            r = self.approximator.get("pearson_r_val")
            trained_on = self.approximator.get("trained_on")
            print(f"[LKHPlacer] cost approximator loaded " f"(r={r:.3f}, trained on {trained_on})")
        elif self.proxy_encoder is None:
            print(f"[LKHPlacer] no approximator at {approx_ckpt} — using HPWL surrogate")

        self.policy_bundle = None
        if use_policy:
            policy_ckpt = (
                Path(policy_path) if policy_path else (_HERE / "checkpoints" / "chain_policy.pt")
            )
            self.policy_bundle = _load_chain_policy(policy_ckpt)
            if self.policy_bundle is not None:
                print(
                    f"[LKHPlacer] chain policy loaded "
                    f"({self.policy_bundle.get('policy_class', 'ChainPolicy')}, "
                    f"iters={self.policy_bundle.get('n_iterations')}, "
                    f"trained on {self.policy_bundle.get('trained_on')})"
                )
            else:
                print(f"[LKHPlacer] no policy at {policy_ckpt} — using greedy chains")

    def _perturb(
        self,
        state: PlacementState,
        rng: random.Random,
        seed_weights: list[float] | None = None,
        movable_idx: list[int] | None = None,
    ) -> None:
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

        # Phase 2.5: resolve benchmark-specific encoder effective weights.
        # effective_weights = {wl: α_wl * wl_std, den: α_den * den_std, cong: α_cong * cong_std}
        # so per-candidate delta scoring avoids per-term denormalization.
        bench_proxy_encoder: dict | None = None
        if self.proxy_encoder is not None:
            bench_name = benchmark.name
            ns = self.proxy_encoder["norm_stats"].get(bench_name)
            if ns is not None:
                pw = self.proxy_encoder["proxy_weights"]
                bench_proxy_encoder = dict(self.proxy_encoder)  # shallow copy
                bench_proxy_encoder["effective_weights"] = {
                    "wl": pw["wl"] * ns["wl_std"],
                    "den": pw["den"] * ns["den_std"],
                    "cong": pw["cong"] * ns["cong_std"],
                }
            else:
                print(
                    f"[LKHPlacer] encoder has no norm_stats for {bench_name!r} "
                    f"— falling back to HPWL surrogate"
                )

        # Cost-weighted seed sampling: macros with highest neighbor-displacement
        # are sampled more often. Falls back to uniform if no edges.
        seed_weights = self._seed_weights(state, movable_idx)

        # A.4: reserve wall time for the post-loop _legalize step. If the
        # caller didn't pin a value, probe the spiral on the 10 largest
        # movable macros and pad by 1.5x. Floor at frac * budget so a
        # mis-estimating probe can't starve legalization.
        if self.legalization_reserve_s is not None:
            legalization_reserve = float(self.legalization_reserve_s)
        else:
            probe_estimate = _probe_legalization_cost(state, n_probe=10)
            floor = self.legalization_reserve_frac * self.time_budget_s
            legalization_reserve = max(probe_estimate * 1.5, floor, 1.0)
            print(
                f"[LKHPlacer] legalization reserve: probe={probe_estimate:.2f}s "
                f"-> reserved {legalization_reserve:.2f}s of {self.time_budget_s:.1f}s budget"
            )
        # Cap the reserve at half the budget so the chain loop still gets
        # meaningful time even on a wildly mis-estimating probe.
        legalization_reserve = min(legalization_reserve, self.time_budget_s * 0.5)
        chain_loop_deadline = self.time_budget_s - legalization_reserve

        start = time.time()
        chains = 0
        best_pos = state.pos.copy()
        # A.2: outer-loop best-tracking must match the chain-internal lex
        # gate `(overlap_pairs, overlap_area, hpwl)`. With the pre-A.2
        # 2-tuple, a chain that reduced `overlap_area` at slight HPWL cost
        # would commit (lex-better in 3-tuple) but the outer would prefer
        # an earlier higher-area state with lower HPWL — and legalization
        # would then have to undo a larger overlap, regressing the very
        # invariant A.2 was designed to protect.
        best_key = (state.overlap_pairs(), state.overlap_area(), state.hpwl())
        stagnation = 0

        while chains < self.max_chains and (time.time() - start) < chain_loop_deadline:
            seed_macro = rng.choices(movable_idx, weights=seed_weights, k=1)[0]
            result = LKChain(
                state,
                seed_macro,
                rng,
                approximator=self.approximator,
                policy_bundle=self.policy_bundle,
                gate_mode=self.gate_mode,
                proxy_encoder=bench_proxy_encoder,
            ).run(self.max_chain_length)
            chains += 1
            if result["committed"]:
                cur_key = (state.overlap_pairs(), state.overlap_area(), state.hpwl())
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
            if self.seed_refresh_interval > 0 and chains % self.seed_refresh_interval == 0:
                seed_weights = self._seed_weights(state, movable_idx)

            # Phase 6.1: kick on stagnation. A.5: bias the kick toward
            # top-quartile-cost macros, then refresh seed weights so the
            # post-kick chain loop resumes with a non-stale ranking.
            if stagnation >= self.stagnation_threshold:
                self._perturb(state, rng, seed_weights=seed_weights, movable_idx=movable_idx)
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
            print(
                f"[LKHPlacer] legalization: "
                f"{n_overlaps_before} -> {n_overlaps_after} overlap pairs"
            )

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
