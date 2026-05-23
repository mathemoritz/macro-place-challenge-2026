"""ChainEnv — LK chain as an RL environment for Phase 4 PPO training.

Episode:
    reset implicitly at __init__: snapshot the placement, pick a seed macro
    step(action_idx, candidates):
        if action_idx == len(candidates): STOP -> finalize
        else: apply candidate move; cascade to nearest displaced macro
    finalize:
        commit (keep current state) iff lex (overlap_pairs, hpwl) improved
        else revert to snapshot
        emits info {committed, hpwl_gain, overlap_delta, chain_length}

Reward shaping: -Δhpwl_surrogate per step (so cumulative ≈ chain HPWL gain
via telescoping) plus an optional terminal bonus on commit. We deliberately
use the HPWL surrogate, not compute_proxy_cost — chain training would
otherwise be O(1 s) per step and the full curriculum would take days.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Load placer.py for PlacementState + _features_for_move without triggering
# the macro_place package __init__ chain.
_spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
_placer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_placer)

PlacementState = _placer.PlacementState
_features_for_move = _placer._features_for_move
FEATURE_DIM = _placer.FEATURE_DIM


# Feature-dim constants must match lkh_model.py exactly.
GLOBAL_DIM = 5
MACRO_DIM = 6
CHAIN_DIM = 3


def compute_global_features(state: PlacementState) -> np.ndarray:
    """5-d global state summary."""
    n = max(state.n, 1)
    canvas_diag = (state.cw ** 2 + state.ch ** 2) ** 0.5
    hpwl_norm = state.hpwl() / max(n * canvas_diag, 1e-6)
    overlap_norm = state.overlap_pairs() / max(n * (n - 1) / 2.0, 1.0)
    pos_x_var = float(np.var(state.pos[:, 0])) / max(state.cw ** 2, 1e-6)
    pos_y_var = float(np.var(state.pos[:, 1])) / max(state.ch ** 2, 1e-6)
    n_mov_frac = float(state.movable.mean())
    return np.array([hpwl_norm, overlap_norm, pos_x_var, pos_y_var, n_mov_frac],
                    dtype=np.float32)


def compute_macro_features(state: PlacementState, idx: int) -> np.ndarray:
    """6-d per-macro features (normalized)."""
    return np.array([
        state.sizes[idx, 0] / max(state.cw, 1e-6),
        state.sizes[idx, 1] / max(state.ch, 1e-6),
        len(state.neighbors[idx]) / max(state.n, 1),
        len(state.overlapping_with(idx)) / max(state.n, 1),
        state.pos[idx, 0] / max(state.cw, 1e-6),
        state.pos[idx, 1] / max(state.ch, 1e-6),
    ], dtype=np.float32)


def compute_chain_features(chain_length: int, hpwl_gain: float,
                            n_displaced: int, max_chain_length: int
                            ) -> np.ndarray:
    """3-d chain progress features."""
    return np.array([
        chain_length / max(max_chain_length, 1),
        hpwl_gain / (abs(hpwl_gain) + 1.0),     # bounded in (-1, 1)
        n_displaced / 10.0,                      # arbitrary normalization
    ], dtype=np.float32)


class ChainEnv:
    """LK chain as an RL env. See module docstring for episode semantics."""

    def __init__(self, state: PlacementState, seed_macro: int, rng,
                 max_chain_length: int = 8, max_candidates: int = 8,
                 terminal_commit_bonus: float = 0.0):
        self.state = state
        self.seed_macro = seed_macro
        self.rng = rng
        self.max_chain_length = max_chain_length
        self.max_candidates = max_candidates
        self.terminal_commit_bonus = terminal_commit_bonus

        self.snapshot = state.pos.copy()
        self.start_hpwl = state.hpwl()
        self.start_overlaps = state.overlap_pairs()

        self.current_macro = seed_macro
        self.chain_length = 0
        self.hpwl_gain = 0.0
        self.done = False

    def state_for_policy(self) -> dict:
        """Build the policy inputs for the current step.

        Returns a dict with NumPy arrays plus the raw candidate positions so
        the caller can map a chosen action_idx back to a (cx, cy).
        """
        cands = self.state.candidate_positions(
            self.current_macro,
            num_candidates=self.max_candidates,
            rng=self.rng,
        )
        if cands:
            old_x = self.state.pos[self.current_macro, 0]
            old_y = self.state.pos[self.current_macro, 1]
            cand_feats = np.stack([
                _features_for_move(
                    self.state, self.current_macro,
                    np.array([cx - old_x, cy - old_y], dtype=np.float64),
                )
                for (cx, cy) in cands
            ]).astype(np.float32)
        else:
            cand_feats = np.zeros((0, FEATURE_DIM), dtype=np.float32)

        n_displaced = len(self.state.overlapping_with(self.current_macro))
        return {
            "global": compute_global_features(self.state),
            "macro": compute_macro_features(self.state, self.current_macro),
            "candidates": cand_feats,
            "chain": compute_chain_features(
                self.chain_length, self.hpwl_gain, n_displaced,
                self.max_chain_length,
            ),
            "raw_candidates": cands,
        }

    def step(self, action_idx: int, candidates: list[tuple[float, float]]
             ) -> tuple[float, bool, dict]:
        """Apply action. STOP when action_idx == len(candidates)."""
        if self.done:
            raise RuntimeError("step() called on terminated env")

        K = len(candidates)
        if K == 0 or action_idx >= K:
            return self._finalize(reward=0.0)

        cx, cy = candidates[action_idx]
        old_hpwl = self.state.hpwl()
        self.state.pos[self.current_macro, 0] = cx
        self.state.pos[self.current_macro, 1] = cy
        new_hpwl = self.state.hpwl()
        delta_hpwl = new_hpwl - old_hpwl
        reward = -float(delta_hpwl)
        self.hpwl_gain += -float(delta_hpwl)
        self.chain_length += 1

        # Cascade to nearest displaced movable macro.
        displaced = [j for j in self.state.overlapping_with(self.current_macro)
                     if self.state.movable[j]]
        if not displaced or self.chain_length >= self.max_chain_length:
            return self._finalize(reward=reward)

        dx = np.abs(self.state.pos[displaced, 0] - self.state.pos[self.current_macro, 0])
        dy = np.abs(self.state.pos[displaced, 1] - self.state.pos[self.current_macro, 1])
        self.current_macro = int(displaced[int(np.argmin(dx + dy))])
        return reward, False, {}

    def _finalize(self, reward: float = 0.0) -> tuple[float, bool, dict]:
        end_hpwl = self.state.hpwl()
        end_overlaps = self.state.overlap_pairs()
        committed = (
            end_overlaps < self.start_overlaps
            or (end_overlaps == self.start_overlaps and end_hpwl < self.start_hpwl)
        )
        if not committed:
            self.state.pos[:] = self.snapshot
        self.done = True
        if committed:
            reward += self.terminal_commit_bonus
        info = {
            "committed": committed,
            "chain_length": self.chain_length,
            "hpwl_gain": self.start_hpwl - end_hpwl,
            "overlap_delta": end_overlaps - self.start_overlaps,
        }
        return reward, True, info
