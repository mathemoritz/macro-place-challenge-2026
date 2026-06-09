"""Building block — scalar commit gate (Fix 5).

The poster's chain-commit rule scores intermediate states by a single
scalar: ``wirelength + density + congestion + overlap penalty``. The
current code uses a strict lex 3-tuple ``(overlap_pairs, overlap_area,
hpwl)`` that absolutely vetoes any overlap-pair increase, regardless of
the proxy gain.

This module provides the scalar replacement. Used by both
``LKChain.run_greedy`` (in placer.py) and ``ChainEnv._finalize`` (in
chain_env.py) when ``gate_mode="scalar_penalty"``. Co-locating the logic
here keeps the two consumers from drifting out of sync.

Score definition:
    score(step) = cumulative_predicted_Δproxy(step)
                  + λ × max(0, current_overlap_pairs - start_overlap_pairs)

The ``max(0, ...)`` term — "new overlap pairs" — penalizes only overlaps
*introduced* by the chain, not pre-existing ones the chain hasn't yet
removed. This matches the poster's "overlap penalty" framing: bad chains
add overlaps, good chains don't have to fix the world's overlaps.

The committed prefix is the one with the lowest score; if no step beats
the start score (0.0 + λ·0 = 0), we revert.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Avoid runtime import to dodge package-init issues; only needed for typing.
    from .placer import PlacementState  # type: ignore


def scalar_score(predicted_delta_proxy: float, n_new_overlap_pairs: int,
                  lam: float) -> float:
    """Single-scalar commit score. Lower is better.

    Args:
        predicted_delta_proxy: cumulative predicted Δproxy across moves
            applied so far (from the cost approximator).
        n_new_overlap_pairs: count of overlap pairs *introduced* by the
            chain so far (clamped at zero — we don't reward chains for
            reducing pre-existing overlap below the start state).
        lam: overlap penalty weight (proxy units per overlap pair).
    """
    return float(predicted_delta_proxy) + lam * float(n_new_overlap_pairs)


def new_overlap_pairs(start_overlap_pairs: int, current_overlap_pairs: int
                      ) -> int:
    """Overlap pairs the chain has *introduced* relative to its start.

    Clamped at zero: chains that reduce overlap below the start get a 0
    penalty (not negative). Reducing overlap is captured via the proxy
    component of the score (legalization-cost proxy via the approximator),
    not via this penalty term.
    """
    return max(0, int(current_overlap_pairs) - int(start_overlap_pairs))


class ScalarBestTracker:
    """Best-prefix tracker for the scalar commit gate.

    Mirrors the lex-tuple bookkeeping in ``LKChain.run_greedy`` (placer.py
    lines 922-1066) and ``ChainEnv`` (chain_env.py lines 148-337) but
    against the single scalar ``scalar_score``. Caller seeds it with the
    chain's start state, calls ``update`` after each move, and ends with
    ``commit_best`` to restore the lex-best prefix into the live state.
    """

    def __init__(self, start_overlap_pairs: int, start_pos_snapshot: np.ndarray,
                 lam: float):
        self.start_overlap_pairs = int(start_overlap_pairs)
        self.lam = float(lam)
        # Start "score" is 0 (no moves applied, no new overlaps). Any prefix
        # whose scalar is strictly < 0 wins.
        self.best_score: float = 0.0
        self.best_pos_snapshot: np.ndarray = start_pos_snapshot.copy()
        self.best_prefix_index: int = 0
        self.predicted_cumulative: float = 0.0

    def update(self, state, predicted_delta: float, chain_step_after_move: int
               ) -> None:
        """Call after each in-chain ``state.apply_move``.

        ``predicted_delta`` is the cost approximator's prediction for the
        just-applied move. ``chain_step_after_move`` is the 1-indexed
        count of moves applied so far in the chain (used as
        ``best_prefix_index`` if this step wins).
        """
        self.predicted_cumulative += float(predicted_delta)
        n_new = new_overlap_pairs(self.start_overlap_pairs, state.overlap_pairs())
        score = scalar_score(self.predicted_cumulative, n_new, self.lam)
        if score < self.best_score:
            self.best_score = score
            self.best_pos_snapshot = state.pos.copy()
            self.best_prefix_index = int(chain_step_after_move)

    def commit_best(self, state) -> bool:
        """Restore the lex-best prefix into ``state``. Returns True iff a
        non-trivial prefix was committed (else state was reverted to start).

        Bulk pos[:]= writes bypass the incremental cache machinery; we call
        ``state.rebuild_caches()`` afterward (B.3 contract in placer.py).
        """
        committed = self.best_prefix_index > 0
        state.pos[:] = self.best_pos_snapshot
        state.rebuild_caches()
        return committed
