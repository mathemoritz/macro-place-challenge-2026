"""State snapshot for deferred-feature training.

``train.py`` freezes ``(features, target)`` tensors at collect time.
That works for the hand-feature path, but with the encoder wired in
the cached features would be a snapshot of the encoder at random init
— training the encoder afterward would do nothing because the cache
is frozen.

This module stores the *state* at collect time and re-runs the
encoder forward inside each batch step at training time, so the
gradient flows into the encoder.

A snapshot bundles:
- ``benchmark_name``      to look up cached netlist edges + sizes
- ``pos[N, 2]``           macro positions at the moment of sampling
- ``macro_idx``           which macro is being moved
- ``move_delta[2]``       the (dx, dy) of the move
- ``target_delta_proxy``  true Δproxy from compute_proxy_cost (the label)
- ``hand_feats[16]``      cached _features_for_move — these don't depend
                          on the encoder, so caching them avoids
                          recomputing per batch step
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
_placer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_placer)

PlacementState = _placer.PlacementState


@dataclass
class StateSnapshot:
    benchmark_name: str
    pos: np.ndarray              # [N_hard, 2]
    macro_idx: int
    move_delta: np.ndarray       # [2]
    target_delta_proxy: float
    hand_feats: np.ndarray       # [16]


def collect_snapshot(benchmark_name: str, state, macro_idx: int,
                      move_delta: np.ndarray, target_delta_proxy: float,
                      hand_feats: np.ndarray) -> StateSnapshot:
    """Build a snapshot from the current state. Copies ``pos`` so future
    state mutations don't bleed into the snapshot."""
    return StateSnapshot(
        benchmark_name=benchmark_name,
        pos=state.pos.copy(),
        macro_idx=int(macro_idx),
        move_delta=np.asarray(move_delta, dtype=np.float64).copy(),
        target_delta_proxy=float(target_delta_proxy),
        hand_feats=np.asarray(hand_feats, dtype=np.float32).copy(),
    )


@dataclass
class BenchmarkCache:
    """Per-benchmark static info used to rebuild a PlacementState from
    a snapshot.

    Edges and sizes don't depend on positions, so we build them once per
    benchmark and reuse across snapshots.
    """
    benchmark: object                          # macro_place.benchmark.Benchmark
    hpwl_edges: object                         # placer.HpwlEdges bundle


def build_benchmark_cache(benchmark) -> BenchmarkCache:
    """Build a benchmark cache for snapshot replay. Lightweight: no plc."""
    hpwl_edges = _placer._hard_macro_edges(benchmark)
    return BenchmarkCache(benchmark=benchmark, hpwl_edges=hpwl_edges)


def replay_state(snapshot: StateSnapshot, cache: BenchmarkCache):
    """Build a fresh ``PlacementState`` at the snapshot's positions.

    Constant per-benchmark info (sizes, canvas, edges) comes from
    ``cache``. The state's positions are bulk-overwritten and
    ``rebuild_caches`` is called so HPWL/overlap totals are correct.
    """
    state = PlacementState(cache.benchmark, cache.hpwl_edges)
    n = snapshot.pos.shape[0]
    if n != state.n:
        raise ValueError(
            f"replay_state: snapshot has {n} macros but benchmark "
            f"{snapshot.benchmark_name!r} has {state.n}"
        )
    state.pos[:] = snapshot.pos
    state.rebuild_caches()
    return state


def stack_snapshots_by_benchmark(snapshots: list[StateSnapshot]
                                   ) -> dict[str, list[StateSnapshot]]:
    """Group snapshots by benchmark name for batched encoder forward.

    Used by the joint trainer to amortize edge-building (constant per
    benchmark) across all snapshots in a mini-batch that share a
    benchmark.
    """
    by_bench: dict[str, list[StateSnapshot]] = {}
    for s in snapshots:
        by_bench.setdefault(s.benchmark_name, []).append(s)
    return by_bench


def save_snapshots(snapshots: list[StateSnapshot], path: Path) -> None:
    """Persist a snapshot list. torch.save handles dataclasses + np arrays."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"snapshots": snapshots, "format_version": 1}, str(path))


def load_snapshots(path: Path) -> list[StateSnapshot]:
    payload = torch.load(str(path), weights_only=False)
    return payload["snapshots"]
