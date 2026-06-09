"""ChainEnv — LK chain as an RL environment for PPO training.

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

# Encoder for per-macro mask channels. Loaded lazily so the legacy
# inference path doesn't import torch_geometric-shaped code on
# Modal images that don't carry it.
_encoder_mod = None
def _get_encoder_module():
    global _encoder_mod
    if _encoder_mod is None:
        _spec_enc = importlib.util.spec_from_file_location(
            "lkh_encoder", str(_HERE / "encoder.py")
        )
        _encoder_mod = importlib.util.module_from_spec(_spec_enc)
        _spec_enc.loader.exec_module(_encoder_mod)
    return _encoder_mod


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
                 # Regularity / mask opt-ins
                 reg_weight: float = 0.0,
                 use_wiremask: bool = False,
                 use_position_mask: bool = False,
                 encoder=None,
                 # Feature-mode opt-ins
                 feature_mode: str = "handcrafted",
                 encoder_cache: dict | None = None,
                 lam: float = 0.01):
        """
        ``terminal_reward_mode`` options:
        * ``"committed_gain"`` (default): per-step 0, terminal = best-prefix HPWL gain.
        * ``"hpwl_telescope_legacy"``: per-step ``-Δhpwl`` + terminal bonus.
        * ``"predicted_proxy_with_postleg"``: reward is the negative
          cumulative predicted Δproxy across (committed chain moves + the
          ``fast_legalize`` moves applied to the committed best-prefix
          state). Post-legalization reward signal.
          Requires ``approximator`` (uses predictions, not real proxy).

        ``gate_mode`` options:
        * ``"hpwl"`` (default): lex (overlap_pairs, overlap_area, hpwl).
        * ``"predicted_proxy"``: lex with cumulative predicted Δproxy third.
        * ``"scalar_penalty"``: single scalar
          ``predicted_Δproxy + lam × n_new_overlap_pairs``.

        ``feature_mode`` options:
        * ``"handcrafted"`` (default): hand-built 16/5/6/3-dim features.
        * ``"encoder"``: encoder embeddings + chain features (lengths).
          Requires ``encoder_cache`` with ``per_node`` and ``graph_vec``
          tensors. The cache is built ONCE per episode (we do not refresh
          mid-chain — accepted staleness for speed).
        """
        self.state = state
        self.seed_macro = seed_macro
        self.rng = rng
        self.max_chain_length = max_chain_length
        self.max_candidates = max_candidates
        self.terminal_commit_bonus = terminal_commit_bonus
        self.terminal_reward_mode = terminal_reward_mode
        if terminal_reward_mode not in ("committed_gain", "hpwl_telescope_legacy",
                                          "predicted_proxy_with_postleg"):
            raise ValueError(
                f"unknown terminal_reward_mode={terminal_reward_mode!r}"
            )
        if gate_mode not in ("hpwl", "predicted_proxy", "scalar_penalty"):
            raise ValueError(
                f"unknown gate_mode={gate_mode!r}; expected 'hpwl', "
                f"'predicted_proxy', or 'scalar_penalty'"
            )
        if feature_mode not in ("handcrafted", "encoder"):
            raise ValueError(
                f"unknown feature_mode={feature_mode!r}"
            )
        self.gate_mode = gate_mode
        self.approximator = approximator
        self.feature_mode = feature_mode
        self.encoder_cache = encoder_cache
        self.lam = float(lam)
        if gate_mode in ("predicted_proxy", "scalar_penalty") and approximator is None:
            # Soft fallback so policy rollouts without an approximator
            # don't blow up; matches LKChain.__init__ contract.
            self.gate_mode = "hpwl"
        # Regularity blend weight. 0.0 = unchanged.
        self.reg_weight = float(reg_weight)
        # WireMask + position-mask candidate handling.
        self.use_wiremask = bool(use_wiremask)
        self.use_position_mask = bool(use_position_mask)
        # Optional StateEncoder. The encoder's GNN forward depends
        # on the WHOLE state (positions + edges), so we cache it ONCE at
        # chain init — subsequent steps recompute only the CNN branch (per-
        # macro mask raster). The cache is invalidated naturally because
        # each chain spawns a fresh ChainEnv.
        self.encoder = encoder
        self._encoder_gnn_cache: tuple | None = None
        if self.encoder is not None:
            _enc = _get_encoder_module()
            nf = _enc.build_node_features(state)
            ei, ea = _enc.build_edge_index_and_attr(state)
            with torch.no_grad():
                h_gnn, g_gnn = self.encoder.gnn(nf, ei, ea)
            self._encoder_gnn_cache = (h_gnn.detach(), g_gnn.detach())
        # Pre-compute scales so per-step ``_compute_third`` is allocation-free.
        self._hpwl_scale = max(state.cw, state.ch)
        self._reg_scale = state.cw + state.ch
        # Fallbacks for missing dependencies.
        if feature_mode == "encoder" and encoder_cache is None:
            self.feature_mode = "handcrafted"
        if (terminal_reward_mode == "predicted_proxy_with_postleg"
                and approximator is None):
            self.terminal_reward_mode = "committed_gain"

        self.snapshot = state.pos.copy()
        self.start_hpwl = state.hpwl()
        self.start_overlaps = state.overlap_pairs()
        self.start_overlap_area = state.overlap_area()
        # Cache start regularity for the reward shape in _finalize.
        self.start_regularity = (state.total_regularity()
                                  if self.reg_weight > 0.0 else 0.0)

        # Best-prefix tracking. Scalar gate uses ScalarBestTracker.
        # Else: lex (overlap_pairs, overlap_area, _compute_third) where
        # _compute_third blends hpwl/predicted_proxy with optional
        # reg_weight regularity.
        self.use_scalar_gate = (self.gate_mode == "scalar_penalty")
        if self.use_scalar_gate:
            from commit_gate_scalar import ScalarBestTracker
            self.scalar_tracker = ScalarBestTracker(
                start_overlap_pairs=self.start_overlaps,
                start_pos_snapshot=self.snapshot,
                lam=self.lam,
            )
            self.best_key = (self.start_overlaps, self.start_overlap_area, 0.0)
            self.best_pos = self.snapshot
            self.best_prefix_index = 0
        else:
            self.scalar_tracker = None
            start_third = self._compute_third(predicted_cumulative=0.0)
            self.start_key = (self.start_overlaps, self.start_overlap_area,
                              start_third)
            self.best_key = self.start_key
            self.best_pos = self.snapshot
            self.best_prefix_index = 0
        # Running sum of predicted Δproxy across applied moves.
        self.predicted_cumulative = 0.0

        self.current_macro = seed_macro
        self.chain_length = 0
        self.hpwl_gain = 0.0
        self.done = False
        # Visited set blocks the cascade from oscillating between the
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

    def _build_feat_vec(self, macro_idx: int, delta: np.ndarray,
                          use_reg_feature: bool = False) -> np.ndarray:
        """Build the approximator's input feature vector for a candidate.

        In handcrafted mode this is the 16-dim (or 17-dim under
        ``use_reg_feature``) ``_features_for_move`` vector.
        In encoder mode, the vector is
        ``[per_node[macro]; graph_vec; hand_feats]`` using the chain-level
        encoder cache. The approximator's stored ``feat_mean``/``feat_std``
        normalization matches whichever schema it was trained on.
        """
        hand = _features_for_move(self.state, macro_idx, delta,
                                    use_reg_feature=use_reg_feature)
        if (self.feature_mode == "encoder" and self.encoder_cache is not None):
            per_node = self.encoder_cache["per_node"]
            graph_vec = self.encoder_cache["graph_vec"]
            macro_emb = per_node[macro_idx].detach().numpy().astype(np.float32)
            global_emb = graph_vec.detach().numpy().astype(np.float32)
            return np.concatenate([macro_emb, global_emb, hand.astype(np.float32)])
        return hand

    def _predict_delta(self, old_x: float, old_y: float,
                       cx: float, cy: float) -> float:
        """Approximator-predicted Δproxy for moving current_macro -> (cx, cy).

        Used when ``gate_mode`` is ``predicted_proxy`` or ``scalar_penalty``.
        Mirrors the denormalization performed by LKChain.run_greedy.
        """
        # Feature schema follows the approximator checkpoint.
        use_reg_feat = bool(self.approximator.get("use_reg_feature", False))
        feats = self._build_feat_vec(
            self.current_macro,
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
        # Policy candidate features follow the approximator's schema.
        use_reg_feat = (
            bool(self.approximator.get("use_reg_feature", False))
            if self.approximator is not None else False
        )
        # Determine effective feature dim for empty-candidate fallback.
        base_feat_dim = (_placer.FEATURE_DIM_WITH_REG if use_reg_feat
                          else FEATURE_DIM)
        if self.feature_mode == "encoder" and self.encoder_cache is not None:
            per_node = self.encoder_cache["per_node"]
            graph_vec = self.encoder_cache["graph_vec"]
            embed_dim = int(per_node.shape[1] + graph_vec.shape[0])
            cand_dim = embed_dim + base_feat_dim
        else:
            cand_dim = base_feat_dim

        if cands:
            old_x = self.state.pos[self.current_macro, 0]
            old_y = self.state.pos[self.current_macro, 1]
            cand_feats = np.stack([
                self._build_feat_vec(
                    self.current_macro,
                    np.array([cx - old_x, cy - old_y], dtype=np.float64),
                    use_reg_feature=use_reg_feat,
                )
                for (cx, cy) in cands
            ]).astype(np.float32)
        else:
            cand_feats = np.zeros((0, cand_dim), dtype=np.float32)

        n_displaced = len(self.state.overlapping_with(self.current_macro))

        # Encoder-mode global/macro features come from the
        # encoder cache; chain features stay hand-crafted (the chain
        # progress signals — length, gain, n_displaced — aren't
        # network-derived). When encoder_cache is absent we fall back
        # to the handcrafted features.
        if self.feature_mode == "encoder" and self.encoder_cache is not None:
            per_node = self.encoder_cache["per_node"]
            graph_vec = self.encoder_cache["graph_vec"]
            global_vec = graph_vec.detach().numpy().astype(np.float32)
            macro_vec = per_node[self.current_macro].detach().numpy().astype(np.float32)
        else:
            global_vec = compute_global_features(self.state)
            macro_vec = compute_macro_features(self.state, self.current_macro)

        out = {
            "global": global_vec,
            "macro": macro_vec,
            "candidates": cand_feats,
            "chain": compute_chain_features(
                self.chain_length, self.hpwl_gain, n_displaced,
                self.max_chain_length,
            ),
            "raw_candidates": cands,
        }
        # When an encoder is attached, append per-macro mask-CNN
        # output to the cached GNN globals, plus the cached per-node
        # embedding for the current macro. ``state_for_policy`` is the only
        # call site at which the encoder fires — once per chain step.
        if self.encoder is not None and self._encoder_gnn_cache is not None:
            _enc = _get_encoder_module()
            canvas = _enc.rasterize_canvas(
                self.state, idx=self.current_macro,
                grid_size=self.encoder.grid_size,
            )
            with torch.no_grad():
                cnn_out = self.encoder.cnn(canvas)
            h_gnn, g_gnn = self._encoder_gnn_cache
            encoder_global = torch.cat([g_gnn, cnn_out], dim=-1)
            encoder_macro = h_gnn[self.current_macro]
            out["encoder_global"] = encoder_global.numpy().astype(np.float32)
            out["encoder_macro"] = encoder_macro.numpy().astype(np.float32)
        return out

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
        # Predicted Δproxy needed for both predicted_proxy and
        # scalar_penalty gates. Measured *before* apply_move (approximator
        # was trained on pre-move features).
        if self.gate_mode in ("predicted_proxy", "scalar_penalty"):
            step_pred_delta = self._predict_delta(old_x, old_y, cx, cy)
        else:
            step_pred_delta = 0.0
        # apply_move keeps the incremental caches consistent so the
        # gate evaluation in _finalize and the per-step state_for_policy
        # features both read O(1) cached quantities.
        self.state.apply_move(self.current_macro, cx, cy)
        new_hpwl = self.state.hpwl()
        delta_hpwl = new_hpwl - old_hpwl
        # Per-step reward depends on terminal_reward_mode.
        if self.terminal_reward_mode in ("committed_gain",
                                           "predicted_proxy_with_postleg"):
            reward = 0.0
        else:  # legacy telescope
            reward = -float(delta_hpwl)
        self.hpwl_gain += -float(delta_hpwl)
        self.chain_length += 1
        if self.gate_mode in ("predicted_proxy", "scalar_penalty"):
            self.predicted_cumulative += step_pred_delta

        # Best-prefix snapshot. Scalar gate uses ScalarBestTracker.
        # Otherwise: lex tuple with _compute_third blending reg term.
        if self.use_scalar_gate:
            self.scalar_tracker.update(
                self.state,
                predicted_delta=step_pred_delta,
                chain_step_after_move=self.chain_length,
            )
            # Mirror into legacy fields so consumers reading best_pos /
            # best_prefix_index after step still get a coherent view.
            self.best_pos = self.scalar_tracker.best_pos_snapshot
            self.best_prefix_index = self.scalar_tracker.best_prefix_index
        else:
            third = self._compute_third(self.predicted_cumulative)
            cur_key = (self.state.overlap_pairs(), self.state.overlap_area(),
                       third)
            if cur_key < self.best_key:
                self.best_key = cur_key
                self.best_pos = self.state.pos.copy()
                self.best_prefix_index = self.chain_length

        # Cascade to a displaced macro under the combined manhattan/
        # connectivity score; visited set blocks oscillation.
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
        # Best-prefix commit: restore the lex-best (or scalar-best) prefix.
        # Bulk pos[:]= writes bypass apply_move so we rebuild_caches()
        # afterward (cache contract).
        if self.use_scalar_gate:
            committed = self.scalar_tracker.commit_best(self.state)
            self.best_prefix_index = self.scalar_tracker.best_prefix_index
        else:
            committed = self.best_prefix_index > 0
            if committed:
                self.state.pos[:] = self.best_pos
            else:
                self.state.pos[:] = self.snapshot
            self.state.rebuild_caches()
        self.done = True

        # Optional post-legalization reward via predicted Δproxy.
        # Runs fast_legalize on the committed best-prefix state and scores
        # each legalization move via the approximator. We pass the env's
        # ``_build_feat_vec`` so legalization features include encoder
        # embeddings when feature_mode="encoder" (matches the approximator's
        # training schema).
        leg_predicted = 0.0
        if self.terminal_reward_mode == "predicted_proxy_with_postleg" and committed:
            from fast_legalize import fast_legalize, score_legalization_moves
            leg_moves = fast_legalize(self.state, feature_builder=self._build_feat_vec)
            leg_predicted = score_legalization_moves(leg_moves, self.approximator)

        end_overlaps = self.state.overlap_pairs() if self.use_scalar_gate else self.best_key[0]
        committed_hpwl_gain = (self.start_hpwl - self.state.hpwl()) if committed else 0.0
        # Committed regularity gain (start - end).
        committed_reg_gain = (
            (self.start_regularity - self.state.total_regularity())
            if (committed and self.reg_weight > 0.0) else 0.0
        )
        committed_predicted_cum = (self.scalar_tracker.predicted_cumulative
                                    if self.use_scalar_gate
                                    else self.predicted_cumulative)

        # Terminal reward shape.
        if self.terminal_reward_mode == "predicted_proxy_with_postleg":
            # Negative cumulative predicted Δproxy = improvement.
            if committed:
                reward = -float(committed_predicted_cum + leg_predicted)
            else:
                reward = 0.0
        elif self.terminal_reward_mode == "committed_gain":
            if self.gate_mode == "predicted_proxy" and committed:
                reward = -float(self.best_key[2])
            elif self.use_scalar_gate and committed:
                reward = -float(committed_predicted_cum)
            elif self.reg_weight > 0.0:
                # Convex blend of canvas-normalized hpwl gain
                # and regularity gain. α = 1/(1+reg_weight) puts the two
                # on comparable footing.
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
            "predicted_proxy_gain": (
                -float(self.best_key[2])
                if (self.gate_mode == "predicted_proxy" and committed
                    and not self.use_scalar_gate)
                else (-float(committed_predicted_cum)
                       if (self.use_scalar_gate and committed) else 0.0)
            ),
            "post_leg_predicted_delta": float(leg_predicted),
            "overlap_delta": (end_overlaps - self.start_overlaps) if committed else 0,
        }
        return reward, True, info
