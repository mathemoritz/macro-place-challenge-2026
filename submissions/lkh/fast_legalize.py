"""Fast per-chain legalizer.

Used to compute the post-legalization reward signal during PPO. The pass
needs to be (a) fast enough to run thousands of times during training and
(b) report the moves it applied so the cost approximator can score each
one and the predicted post-leg Δproxy can be summed.

``fast_legalize`` uses the same outward-spiral algorithm as
``placer._legalize`` but routes moves through ``state.apply_move`` (so
the incremental caches stay coherent), skips already-legal macros, and
returns the moves with the feature vector captured *before* each move
is applied.

Meant to run inside ``ChainEnv._finalize`` and other training-time
contexts. The careful spiral legalizer in ``placer._legalize`` is still
the final safety net at the end of ``LKHPlacer.place``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Use importlib to avoid macro_place package init when this is loaded by
# the evaluator path (mirrors chain_env.py's pattern).
_spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
_placer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_placer)

_features_for_move = _placer._features_for_move
PlacementState = _placer.PlacementState


# Each fast_legalize move is captured as:
#   (macro_idx, old_x, old_y, new_x, new_y, feats_at_move)
# The feats_at_move is the 16-dim _features_for_move vector measured BEFORE
# the move is applied — feeding this to the approximator yields the
# predicted Δproxy for the legalization move. The caller stacks all feats
# and runs one batched approximator forward pass.
LegMove = tuple[int, float, float, float, float, np.ndarray]


def fast_legalize(state: PlacementState, *, gap: float = 0.001,
                  max_search_radius: int = 50,
                  feature_builder=None) -> list[LegMove]:
    """Legalize ``state`` in-place by nudging overlapping macros to nearest
    legal slots. Returns the list of moves applied with per-move features.

    Algorithm: largest-area movable macro first; skip macros already legal;
    spiral outward (Chebyshev rings) on a 0.25 · max(w, h) grid; pick
    nearest-distance non-colliding slot; apply via ``state.apply_move`` so
    the cached HPWL + overlap totals stay consistent.

    ``feature_builder`` (optional): a callable ``(macro_idx, delta) -> np.ndarray``
    that builds the feature vector the cost approximator consumes. When
    ``None`` we fall back to the 16-dim ``_features_for_move`` schema (the
    handcrafted-mode default). In encoder mode the env passes its
    ``_build_feat_vec`` so legalization features match the approximator's
    actual input dim (``encoder_embed + 16``).

    The ``feats_at_move`` is captured *before* the move is applied so it's
    a valid input to the approximator (which was trained on pre-move
    features).
    """
    n = state.n
    moves: list[LegMove] = []
    if n == 0:
        return moves

    # Order by descending area, largest-first (same as placer._legalize).
    order = sorted(range(n), key=lambda i: -float(state.sizes[i, 0] * state.sizes[i, 1]))

    for idx in order:
        if not state.movable[idx]:
            continue
        if not state.has_overlap(idx, gap):
            continue  # already legal — no-op, no move recorded

        old_x = float(state.pos[idx, 0])
        old_y = float(state.pos[idx, 1])
        step = max(float(state.sizes[idx, 0]), float(state.sizes[idx, 1])) * 0.25

        best_pos: tuple[float, float] | None = None
        best_dist = float("inf")
        for r in range(1, max_search_radius + 1):
            found_in_ring = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx, cy = state.clamp(idx, old_x + dxm * step, old_y + dym * step)
                    # Probe: apply, check overlap, revert. apply_move keeps
                    # caches consistent so the revert is a clean round-trip.
                    state.apply_move(idx, cx, cy)
                    bad = state.has_overlap(idx, gap)
                    state.apply_move(idx, old_x, old_y)
                    if bad:
                        continue
                    d = (cx - old_x) ** 2 + (cy - old_y) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_pos = (cx, cy)
                        found_in_ring = True
            if found_in_ring:
                break  # nearest-distance candidate found within this ring

        if best_pos is None:
            # Couldn't legalize this macro within the budget. Leave it; the
            # outer placer._legalize spiral (max_search_radius=150) is the
            # safety net at the end of LKHPlacer.place().
            continue

        new_x, new_y = best_pos
        # Capture features BEFORE applying the move (approximator expects
        # pre-move features). Use the env-supplied builder if present so the
        # vector dim matches the approximator's training schema (encoder
        # mode adds the GNN embeddings on top of the 16-dim hand_feats).
        delta = np.array([new_x - old_x, new_y - old_y], dtype=np.float64)
        if feature_builder is not None:
            feats = feature_builder(idx, delta)
        else:
            feats = _features_for_move(state, idx, delta)
        state.apply_move(idx, new_x, new_y)
        moves.append((idx, old_x, old_y, new_x, new_y, feats))

    return moves


def score_legalization_moves(moves: list[LegMove], approximator: dict
                              ) -> float:
    """Sum predicted Δproxy across the legalization moves.

    Runs one batched approximator forward pass on the stacked features.
    ``approximator`` is the dict returned by ``placer._load_cost_approximator``
    (keys: ``model``, ``feat_mean``, ``feat_std``, ``target_mean``,
    ``target_std``). Returns the cumulative predicted Δproxy.
    """
    if not moves:
        return 0.0
    import torch

    feats = np.stack([m[5] for m in moves])
    norm = (feats - approximator["feat_mean"]) / approximator["feat_std"]
    with torch.no_grad():
        pred_norm = approximator["model"](torch.tensor(norm, dtype=torch.float32))
    pred = pred_norm.numpy() * approximator["target_std"] + approximator["target_mean"]
    return float(pred.sum())
