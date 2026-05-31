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
                 terminal_commit_bonus: float = 0.0,
                 terminal_reward_mode: str = "committed_gain",
                 gate_mode: str = "hpwl",
                 approximator: dict | None = None,
                 reg_weight: float = 0.0,
                 use_wiremask: bool = False,
                 use_position_mask: bool = False):
        """
        D.1 (LKH critique fix): ``terminal_reward_mode`` selects the PPO
        reward shape.

        * ``"committed_gain"`` (default): per-step reward is 0, terminal
          reward is the best-prefix HPWL gain (= ``start_hpwl - best_hpwl``,
          0 if no prefix beat start). The policy directly optimizes the
          quantity the gate measures, so STOP becomes a meaningful action
          (stopping at the best prefix is what's rewarded).
        * ``"hpwl_telescope_legacy"``: per-step reward = ``-Δhpwl``, terminal
          adds ``terminal_commit_bonus``. This was the pre-fix shape; the
          telescope sums to ``start_hpwl - end_hpwl``, but only at the
          *endpoint*, leaving STOP effectively unrewarded and rewarding
          chains that end on a worse prefix than they passed through.

        E.1 (LKH critique fix §3.3 deferred): ``gate_mode`` selects the
        third lex-coordinate of the commit gate.

        * ``"hpwl"`` (default): ``(overlap_pairs, overlap_area, hpwl)``.
          Milestone-A semantics; matches LKChain.run_greedy.
        * ``"predicted_proxy"``: ``(overlap_pairs, overlap_area,
          cumulative_predicted_Δproxy)``. Requires ``approximator`` (the
          same dict bundle LKChain consumes); falls back to ``"hpwl"`` if
          the approximator is missing so the env is robust to the
          (rare) policy-only inference path.
        """
        self.state = state
        self.seed_macro = seed_macro
        self.rng = rng
        self.max_chain_length = max_chain_length
        self.max_candidates = max_candidates
        self.terminal_commit_bonus = terminal_commit_bonus
        self.terminal_reward_mode = terminal_reward_mode
        if terminal_reward_mode not in ("committed_gain", "hpwl_telescope_legacy"):
            raise ValueError(
                f"unknown terminal_reward_mode={terminal_reward_mode!r}; "
                f"expected 'committed_gain' or 'hpwl_telescope_legacy'"
            )
        if gate_mode not in ("hpwl", "predicted_proxy"):
            raise ValueError(
                f"unknown gate_mode={gate_mode!r}; expected 'hpwl' or 'predicted_proxy'"
            )
        self.gate_mode = gate_mode
        self.approximator = approximator
        if gate_mode == "predicted_proxy" and approximator is None:
            # Soft fallback so policy rollouts that lack an approximator
            # don't blow up; matches the LKChain.__init__ contract.
            self.gate_mode = "hpwl"
        # Task 1b: regularity blend weight. 0.0 = unchanged. When > 0, the
        # gate's third coord and the ``committed_gain`` reward both include
        # a canvas-normalized regularity term — see ``_compute_third`` and
        # ``_finalize``. predicted_proxy path is intentionally untouched
        # (per the implementation brief) until Task 1c retrains the model.
        self.reg_weight = float(reg_weight)
        # MaskRegulate Task 2: WireMask candidate injection in
        # ``state_for_policy``. Mirrors LKChain.use_wiremask.
        self.use_wiremask = bool(use_wiremask)
        # MaskRegulate Task 3: legality filter on WireMask candidates.
        self.use_position_mask = bool(use_position_mask)
        # Pre-compute scales so per-step ``_compute_third`` is allocation-
        # free; both are constants for this env's lifetime.
        self._hpwl_scale = max(state.cw, state.ch)
        self._reg_scale = state.cw + state.ch

        self.snapshot = state.pos.copy()
        self.start_hpwl = state.hpwl()
        self.start_overlaps = state.overlap_pairs()
        self.start_overlap_area = state.overlap_area()
        # Cache start regularity for the reward shape in _finalize.
        self.start_regularity = (state.total_regularity()
                                  if self.reg_weight > 0.0 else 0.0)

        # Best-prefix tracking (A.1) under the lex (overlap_pairs,
        # overlap_area, third) gate (A.2 / E.1). See LKChain.run_greedy
        # for the parallel implementation; the env mirrors it so RL
        # rollouts and greedy chains share commit semantics.
        start_third = self._compute_third(predicted_cumulative=0.0)
        self.start_key = (self.start_overlaps, self.start_overlap_area,
                          start_third)
        self.best_key = self.start_key
        self.best_pos = self.snapshot
        self.best_prefix_index = 0
        # E.1: running sum of predicted Δproxy across applied moves.
        self.predicted_cumulative = 0.0

        self.current_macro = seed_macro
        self.chain_length = 0
        self.hpwl_gain = 0.0
        self.done = False
        # A.3: visited set blocks the cascade from oscillating between the
        # same two macros under the combined manhattan/connectivity follow
        # rule.
        self.visited: set[int] = {seed_macro}

    def _compute_third(self, predicted_cumulative: float) -> float:
        """Third lex coordinate. Mirrors ``LKChain._compute_third``.

        ``reg_weight == 0`` (default) preserves the original behavior: HPWL
        under the hpwl gate, cumulative predicted Δproxy under predicted_proxy.
        ``reg_weight > 0`` blends in the canvas-normalized regularity term.
        """
        if self.gate_mode == "predicted_proxy":
            base = predicted_cumulative
        else:
            base = self.state.hpwl()
        if self.reg_weight <= 0.0:
            return base
        base_norm = base / max(self._hpwl_scale, 1e-9)
        reg_norm = self.state.total_regularity() / max(self._reg_scale, 1e-9)
        return base_norm + self.reg_weight * reg_norm

    def _predict_delta(self, old_x: float, old_y: float,
                       cx: float, cy: float) -> float:
        """Approximator-predicted Δproxy for moving current_macro -> (cx, cy).

        Used only when ``gate_mode == "predicted_proxy"``. Mirrors the
        denormalization performed by LKChain.run_greedy so the predicted
        cumulative is in the same units as the offline calibration set.
        """
        # Task 1c: feature schema follows the approximator checkpoint.
        use_reg_feat = bool(self.approximator.get("use_reg_feature", False))
        feats = _features_for_move(
            self.state, self.current_macro,
            np.array([cx - old_x, cy - old_y], dtype=np.float64),
            use_reg_feature=use_reg_feat,
        )
        norm = (feats - self.approximator["feat_mean"]) / self.approximator["feat_std"]
        with torch.no_grad():
            pred = self.approximator["model"](torch.tensor(norm, dtype=torch.float32))
        return float(pred.item() * self.approximator["target_std"]
                     + self.approximator["target_mean"])

    def state_for_policy(self) -> dict:
        """Build the policy inputs for the current step.

        Returns a dict with NumPy arrays plus the raw candidate positions so
        the caller can map a chosen action_idx back to a (cx, cy).
        """
        cands = self.state.candidate_positions(
            self.current_macro,
            num_candidates=self.max_candidates,
            rng=self.rng,
            use_wiremask=self.use_wiremask,
            use_position_mask=self.use_position_mask,
        )
        # Task 1c: policy candidate features follow the approximator's schema
        # (so the PPO policy sees the same dim downstream code expects).
        use_reg_feat = (
            bool(self.approximator.get("use_reg_feature", False))
            if self.approximator is not None else False
        )
        cand_dim = (_placer.FEATURE_DIM_WITH_REG if use_reg_feat
                    else FEATURE_DIM)
        if cands:
            old_x = self.state.pos[self.current_macro, 0]
            old_y = self.state.pos[self.current_macro, 1]
            cand_feats = np.stack([
                _features_for_move(
                    self.state, self.current_macro,
                    np.array([cx - old_x, cy - old_y], dtype=np.float64),
                    use_reg_feature=use_reg_feat,
                )
                for (cx, cy) in cands
            ]).astype(np.float32)
        else:
            cand_feats = np.zeros((0, cand_dim), dtype=np.float32)

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
        old_x = float(self.state.pos[self.current_macro, 0])
        old_y = float(self.state.pos[self.current_macro, 1])
        # E.1: predicted Δproxy for the chosen candidate must be measured
        # *before* apply_move (the approximator is trained on
        # pre-move features). Skipped when gate_mode == "hpwl".
        if self.gate_mode == "predicted_proxy":
            step_pred_delta = self._predict_delta(old_x, old_y, cx, cy)
        else:
            step_pred_delta = 0.0
        # B.3: apply_move keeps the incremental caches consistent so the
        # gate evaluation in _finalize and the per-step state_for_policy
        # features both read O(1) cached quantities.
        self.state.apply_move(self.current_macro, cx, cy)
        new_hpwl = self.state.hpwl()
        delta_hpwl = new_hpwl - old_hpwl
        # D.1: per-step reward depends on terminal_reward_mode.
        if self.terminal_reward_mode == "committed_gain":
            reward = 0.0
        else:  # legacy telescope
            reward = -float(delta_hpwl)
        self.hpwl_gain += -float(delta_hpwl)
        self.chain_length += 1
        if self.gate_mode == "predicted_proxy":
            self.predicted_cumulative += step_pred_delta

        # Best-prefix snapshot under the lex (overlap_pairs, overlap_area,
        # third) gate. Third coord depends on gate_mode (E.1) and Task 1b's
        # regularity blend.
        third = self._compute_third(self.predicted_cumulative)
        cur_key = (self.state.overlap_pairs(), self.state.overlap_area(),
                   third)
        if cur_key < self.best_key:
            self.best_key = cur_key
            self.best_pos = self.state.pos.copy()
            self.best_prefix_index = self.chain_length

        # Cascade to a displaced macro under the combined manhattan/
        # connectivity score (A.3); visited set blocks oscillation.
        displaced = [j for j in self.state.overlapping_with(self.current_macro)
                     if self.state.movable[j] and j not in self.visited]
        if not displaced or self.chain_length >= self.max_chain_length:
            return self._finalize(reward=reward)

        eps = 1e-3
        scores = []
        for j in displaced:
            dxj = abs(float(self.state.pos[j, 0] - self.state.pos[self.current_macro, 0]))
            dyj = abs(float(self.state.pos[j, 1] - self.state.pos[self.current_macro, 1]))
            w = self.state.neighbor_weight[self.current_macro].get(j, 0.0)
            scores.append((dxj + dyj) / (w + eps))
        self.current_macro = int(displaced[int(np.argmin(scores))])
        self.visited.add(self.current_macro)
        return reward, False, {}

    def _finalize(self, reward: float = 0.0) -> tuple[float, bool, dict]:
        # Best-prefix commit (A.1 + A.2 + E.1): restore the lex-best prefix.
        # If no prefix beat start, full revert (worst case = pre-A.1
        # behavior). Bulk pos[:]= writes bypass apply_move so we
        # rebuild_caches() afterward (B.3 contract).
        committed = self.best_prefix_index > 0
        if committed:
            self.state.pos[:] = self.best_pos
        else:
            self.state.pos[:] = self.snapshot
        self.state.rebuild_caches()
        self.done = True
        end_overlaps = self.best_key[0]
        # HPWL gain is reported in info regardless of gate_mode for
        # monitoring; under predicted_proxy gate, best_key[2] is the
        # cumulative predicted Δproxy at the committed prefix, not HPWL.
        committed_hpwl_gain = (self.start_hpwl - self.state.hpwl()) if committed else 0.0
        # Task 1b: committed regularity gain (start - end). Positive = pushed
        # toward edges = good. Only computed when reg_weight > 0; otherwise
        # ``start_regularity`` is 0.0 and this is zero too.
        committed_reg_gain = (
            (self.start_regularity - self.state.total_regularity())
            if (committed and self.reg_weight > 0.0) else 0.0
        )
        # D.1 + Task 1b: terminal reward shape.
        if self.terminal_reward_mode == "committed_gain":
            # Per-step reward was 0; terminal reward = committed gain in
            # the units the gate measures. STOP is now a first-class
            # action — the policy is rewarded for stopping at the best
            # prefix even if continuing the cascade would have produced a
            # numerically larger telescope.
            if self.gate_mode == "predicted_proxy" and committed:
                # Negative predicted cumulative = improvement = reward.
                # Task 1b note: predicted_proxy reward path is intentionally
                # untouched until a regularity-aware approximator exists.
                reward = -float(self.best_key[2])
            elif self.reg_weight > 0.0:
                # Task 1b: convex blend of canvas-normalized hpwl gain and
                # regularity gain. α = 1/(1+reg_weight) puts the two on
                # comparable footing for the policy update — α → 1 as
                # reg_weight → 0 recovers the original hpwl-only reward
                # (modulo the normalization, which only kicks in for the
                # regularity branch).
                hpwl_gain_norm = (committed_hpwl_gain
                                   / max(self._hpwl_scale, 1e-9))
                reg_gain_norm = committed_reg_gain / max(self._reg_scale, 1e-9)
                alpha = 1.0 / (1.0 + self.reg_weight)
                reward = (alpha * hpwl_gain_norm
                           + (1.0 - alpha) * reg_gain_norm)
            else:
                reward = committed_hpwl_gain
        else:  # legacy telescope
            if committed:
                reward += self.terminal_commit_bonus
        info = {
            "committed": committed,
            "chain_length": self.chain_length,
            "best_prefix_index": self.best_prefix_index,
            "hpwl_gain": committed_hpwl_gain,
            "reg_gain": committed_reg_gain,
            "predicted_proxy_gain": (-float(self.best_key[2])
                                     if (self.gate_mode == "predicted_proxy"
                                         and committed) else 0.0),
            "overlap_delta": (end_overlaps - self.start_overlaps) if committed else 0,
        }
        return reward, True, info
