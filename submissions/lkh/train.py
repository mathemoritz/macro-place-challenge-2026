"""Phase 3 trainer for the LK cost approximator.

Workflow
--------
1. For each benchmark in the list: load it, sample random (macro, candidate)
   moves under state-drift, record (features, exact Δproxy_cost) pairs.
2. Concatenate the per-benchmark arrays into a single dataset.
3. Cache (features, targets) + the benchmarks list to disk so re-training
   doesn't re-pay the expensive exact-cost calls.
4. Train a small MLP (lkh_model.CostApproximator) with Smooth-L1 loss.
   Normalization stats are computed globally over the combined dataset.
5. Validate on a 20% held-out split; report Pearson r and Spearman ρ.
6. Save state-dict + normalization stats to ``checkpoints/cost_approximator.pt``.

Usage:
    uv run python submissions/lkh/train.py                    # default: ibm01
    uv run python submissions/lkh/train.py --benchmark ibm01,ibm02,ibm03
    uv run python submissions/lkh/train.py --benchmark all --num-examples 200
    uv run python submissions/lkh/train.py --num-examples 3000 --epochs 80
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                       # for lkh_model
sys.path.insert(0, str(_HERE.parent.parent))         # repo root for macro_place

# Load placer.py without triggering the package __init__ (which fails when
# the TILOS submodule isn't initialized in some environments).
_spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
_placer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_placer)

from lkh_model import CostApproximator, FEATURE_DIM

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


# ── Log formatting helpers ─────────────────────────────────────────────────
#
# ``compute_proxy_cost`` ranges from sub-second on ibm01 to ~100 s per call on
# ibm17, so a single ``{rate:.1f}/s`` format string is unreadable in the slow
# regime (everything < 0.05 ex/s prints as ``0.0/s``). These helpers show the
# more useful inverse (s/ex) when the rate is small, and convert wall-time
# fields to ``Xh YYm`` / ``MMmSSs`` instead of raw seconds.

def _fmt_time(seconds: float) -> str:
    """Format wall time in human-friendly units."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m{s:02d}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def _fmt_rate(examples: float, seconds: float) -> str:
    """Show ex/s when fast, s/ex when slow. Avoids the ``0.0/s`` regime."""
    if seconds <= 0 or examples <= 0:
        return "—"
    rate = examples / seconds
    if rate >= 0.5:
        return f"{rate:.2f}ex/s"
    return f"{seconds / examples:.1f}s/ex"


# ── Benchmark list parsing ─────────────────────────────────────────────────

def parse_benchmarks(arg: str) -> list[str]:
    """Parse a ``--benchmark`` CLI value into a list of benchmark names.

    Accepts:
        - single name: ``ibm01``
        - comma-separated: ``ibm01,ibm02,ibm03``
        - the literal ``all`` -> the canonical 17-benchmark IBM suite from
          ``macro_place/evaluate.py::IBM_BENCHMARKS``.

    Existence of each benchmark directory is verified by ``load_benchmark_from_dir``
    when collection begins; parse_benchmarks only does the syntactic split so
    callers can build lists programmatically without needing the cwd set.
    """
    arg = (arg or "").strip()
    if arg == "all":
        from macro_place.evaluate import IBM_BENCHMARKS
        return list(IBM_BENCHMARKS)
    names = [n.strip() for n in arg.split(",") if n.strip()]
    if not names:
        raise ValueError("--benchmark must contain at least one benchmark name")
    return names


# ── Data collection ────────────────────────────────────────────────────────

def _collect_one_benchmark(benchmark_name: str, num_examples: int, seed: int,
                            drift_prob: float = 0.3,
                            p_cascade: float = 0.5,
                            cascade_min: int = 2,
                            cascade_max: int = 4,
                            use_reg_feature: bool = False,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Single-benchmark state-drift sampler. Returns (features [N, F], targets [N]).

    The plc object is released when this function returns so the next
    benchmark in a multi-benchmark sweep doesn't accumulate memory.

    C.2 (LKH critique fix): with probability ``p_cascade`` (default 0.5),
    run a ``[cascade_min, cascade_max]``-step greedy mini-cascade before
    sampling so the (features, Δproxy) pair sees a cascade-interior state
    rather than always a near-initial one. This shifts the training
    distribution toward what the approximator faces at chain inference time
    and is a prerequisite for unifying the cost signal in Milestone E.
    """
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark_name
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    hpwl_edges = _placer._hard_macro_edges(benchmark)
    state = _placer.PlacementState(benchmark, hpwl_edges)
    rng = np.random.RandomState(seed)
    py_rng_seed = int(rng.randint(0, 2**31 - 1))

    movable = np.where(state.movable)[0]
    n_hard = state.n
    full_placement = benchmark.macro_positions.clone()

    def exact_cost() -> float:
        full_placement[:n_hard] = torch.tensor(state.pos, dtype=torch.float32)
        return float(compute_proxy_cost(full_placement, benchmark, plc)["proxy_cost"])

    t0 = time.time()
    base_cost = exact_cost()
    print(f"  base proxy_cost = {base_cost:.4f}  (first call: {time.time()-t0:.2f}s)")

    feats_list: list[np.ndarray] = []
    targets_list: list[float] = []

    import random as _random
    py_rng = _random.Random(py_rng_seed)

    movable_list = movable.tolist()

    def _do_preliminary_cascade(num_steps: int) -> list[tuple[int, float, float]]:
        """C.2: greedy mini-cascade. Each step picks a random movable macro,
        applies the lowest-HPWL of 4 candidates. Returns the undo list."""
        saved: list[tuple[int, float, float]] = []
        for _ in range(num_steps):
            i = int(py_rng.choice(movable_list))
            cands_cas = state.candidate_positions(i, num_candidates=4, rng=py_rng)
            if not cands_cas:
                continue
            best_cx, best_cy = None, None
            best_h = float("inf")
            for cx, cy in cands_cas:
                ox = float(state.pos[i, 0]); oy = float(state.pos[i, 1])
                state.apply_move(i, cx, cy)
                h = state.hpwl()
                state.apply_move(i, ox, oy)
                if h < best_h:
                    best_h = h
                    best_cx, best_cy = cx, cy
            if best_cx is None:
                continue
            ox = float(state.pos[i, 0]); oy = float(state.pos[i, 1])
            state.apply_move(i, best_cx, best_cy)
            saved.append((i, ox, oy))
        return saved

    t_collect = time.time()
    last_log = t_collect
    n_cascade_samples = 0
    while len(feats_list) < num_examples:
        # C.2: optionally pre-roll a 2-4 step cascade so this sampling round
        # operates on a cascade-interior state. Reverted at end so the drift
        # mechanic stays on the un-drifted base state.
        cascade_saved: list[tuple[int, float, float]] = []
        local_base = base_cost
        if py_rng.random() < p_cascade:
            n_steps = py_rng.randint(cascade_min, cascade_max)
            cascade_saved = _do_preliminary_cascade(n_steps)
            if cascade_saved:
                local_base = exact_cost()
                n_cascade_samples += 1

        macro_idx = int(rng.choice(movable))
        cands = state.candidate_positions(macro_idx, num_candidates=8, rng=py_rng)
        cand_costs: list[tuple[tuple[float, float], float]] = []
        for cx, cy in cands:
            old_x = float(state.pos[macro_idx, 0])
            old_y = float(state.pos[macro_idx, 1])
            delta = np.array([cx - old_x, cy - old_y], dtype=np.float64)

            feats = _placer._features_for_move(
                state, macro_idx, delta,
                use_reg_feature=use_reg_feature,
            )

            # B.3: incremental apply + revert; the caches stay coherent
            # so subsequent _features_for_move calls read O(1) HPWL/area.
            state.apply_move(macro_idx, cx, cy)
            new_cost = exact_cost()
            state.apply_move(macro_idx, old_x, old_y)

            feats_list.append(feats)
            targets_list.append(new_cost - local_base)
            cand_costs.append(((cx, cy), new_cost))

            if len(feats_list) >= num_examples:
                break

        # Revert the preliminary cascade so the drift mechanic and the next
        # iteration's base_cost stay on the un-drifted state.
        for (i, ox, oy) in reversed(cascade_saved):
            state.apply_move(i, ox, oy)

        # Maybe drift: commit the best candidate as the new base. After C.2
        # the candidate's ``new_cost`` may have been measured in the cascade
        # interior, so when a cascade ran we recompute the post-commit base
        # explicitly rather than reuse the cascade-interior reading.
        if cand_costs and py_rng.random() < drift_prob:
            (cx, cy), new_cost = min(cand_costs, key=lambda t: t[1])
            state.apply_move(macro_idx, cx, cy)
            base_cost = exact_cost() if cascade_saved else new_cost

        if time.time() - last_log > 5.0:
            elapsed = time.time() - t_collect
            n_done = len(feats_list)
            rate = n_done / max(elapsed, 1e-6)
            remaining = num_examples - n_done
            eta = remaining / max(rate, 1e-9)
            print(f"  collected {n_done:4d}/{num_examples}  "
                  f"(elapsed={_fmt_time(elapsed)}, "
                  f"rate={_fmt_rate(n_done, elapsed)}, "
                  f"eta={_fmt_time(eta)}, base={base_cost:.3f})")
            last_log = time.time()

    elapsed = time.time() - t_collect
    print(f"  done in {_fmt_time(elapsed)} "
          f"({_fmt_rate(len(feats_list), elapsed)})  "
          f"cascade-interior rounds: {n_cascade_samples}")
    return np.stack(feats_list), np.array(targets_list, dtype=np.float32)


def collect_calibration_samples(benchmark_name: str, n_samples: int,
                                 time_budget_s: float,
                                 approximator_ckpt_path: str | None,
                                 policy_ckpt_path: str | None,
                                 seed: int,
                                 use_reg_feature: bool = False,
                                 use_wiremask: bool = False,
                                 use_position_mask: bool = False,
                                 reg_weight: float = 0.0,
                                 gate_mode: str = "hpwl",
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """C.3 (LKH critique fix): collect (features, true_Δproxy) pairs from
    states the placer actually visits at inference time. Pre-fix the
    approximator was trained only on near-initial-placement single moves;
    by the time the placer commits a few chains the state distribution
    drifts away from the training distribution, and the approximator's
    Δproxy predictions silently diverge from the true Δproxy.

    Run the current placer for ``time_budget_s`` to drive the state into
    the inference-time distribution, then sample ``n_samples`` single-move
    examples from the post-placement state with exact ``compute_proxy_cost``
    targets. Caller appends these to the next round's training set so the
    next-round approximator learns to predict Δproxy at the residual states.
    """
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark_name
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    placer = _placer.LKHPlacer(
        seed=seed, time_budget_s=time_budget_s,
        checkpoint_path=approximator_ckpt_path,
        policy_path=policy_ckpt_path,
        use_policy=policy_ckpt_path is not None,
        gate_mode=gate_mode,
        reg_weight=reg_weight,
        use_wiremask=use_wiremask,
        use_position_mask=use_position_mask,
    )
    placement = placer.place(benchmark)

    # Re-instantiate the state at the post-placement positions and use that
    # as the sampling base. The placer's internal state is gone after
    # place() returns; we rebuild fresh.
    hpwl_edges = _placer._hard_macro_edges(benchmark)
    state = _placer.PlacementState(benchmark, hpwl_edges)
    n_hard = state.n
    state.pos[:] = placement[:n_hard].numpy().astype(state.pos.dtype)
    state.rebuild_caches()

    movable = np.where(state.movable)[0]
    feat_dim = (_placer.FEATURE_DIM_WITH_REG if use_reg_feature
                else FEATURE_DIM)
    if len(movable) == 0:
        return np.zeros((0, feat_dim), dtype=np.float32), \
               np.zeros(0, dtype=np.float32)

    rng = np.random.RandomState(seed + 7919)
    import random as _random
    py_rng = _random.Random(int(rng.randint(0, 2**31 - 1)))

    full_placement = benchmark.macro_positions.clone()
    def exact_cost() -> float:
        full_placement[:n_hard] = torch.tensor(state.pos, dtype=torch.float32)
        return float(compute_proxy_cost(full_placement, benchmark, plc)["proxy_cost"])

    base_cost = exact_cost()
    feats_list: list[np.ndarray] = []
    targets_list: list[float] = []
    while len(feats_list) < n_samples:
        macro_idx = int(rng.choice(movable))
        cands = state.candidate_positions(macro_idx, num_candidates=4, rng=py_rng)
        if not cands:
            continue
        for cx, cy in cands:
            old_x = float(state.pos[macro_idx, 0])
            old_y = float(state.pos[macro_idx, 1])
            delta = np.array([cx - old_x, cy - old_y], dtype=np.float64)
            feats = _placer._features_for_move(
                state, macro_idx, delta,
                use_reg_feature=use_reg_feature,
            )
            state.apply_move(macro_idx, cx, cy)
            new_cost = exact_cost()
            state.apply_move(macro_idx, old_x, old_y)
            feats_list.append(feats)
            targets_list.append(new_cost - base_cost)
            if len(feats_list) >= n_samples:
                break
    return np.stack(feats_list), np.array(targets_list, dtype=np.float32)


def _save_benchmark_cache(cache_file: Path, feats: np.ndarray, targets: np.ndarray,
                          *, num_examples: int, name: str) -> None:
    """Atomic write of one per-benchmark cache file."""
    tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
    torch.save(
        {"features": feats, "targets": targets,
         "num_examples": num_examples, "name": name},
        str(tmp),
    )
    tmp.replace(cache_file)


def per_bench_caches_complete(benchmark_names: list[str], cache_dir: Path) -> bool:
    """True if every benchmark has a readable ``<name>.pt`` in ``cache_dir``."""
    cache_dir = Path(cache_dir)
    for name in benchmark_names:
        path = cache_dir / f"{name}.pt"
        if not path.exists():
            return False
        try:
            d = torch.load(str(path), weights_only=False)
            if len(d.get("features", [])) == 0:
                return False
        except Exception:                                       # noqa: BLE001
            return False
    return True


def load_per_benchmark_caches(
    benchmark_names: list[str],
    cache_dir: Path,
    *,
    num_examples_per_benchmark: int | None = None,
    collect_missing: bool = False,
    seed: int = 42,
    drift_prob: float = 0.3,
    post_benchmark_callback: Optional[Callable[[str], None]] = None,
    force_recollect: bool = False,
    use_reg_feature: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate per-benchmark ``<name>.pt`` caches.

  Used after parallel Modal collection (or any time all caches exist).
  Set ``collect_missing=True`` only for local runs that may lack a few
  cache files; never re-collects benchmarks whose cache already exists.

  D.2 (LKH critique fix): set ``force_recollect=True`` to bypass existing
  per-benchmark caches and re-collect each one. Pre-fix the Phase 5
  iterative loop had ``--force-recollect-each-round`` only invalidating
  the *aggregated* cache; per-benchmark caches were still loaded as-is,
  silently bypassing the user's explicit "re-collect" request.
    """
    if not benchmark_names:
        raise ValueError("load_per_benchmark_caches: benchmark_names is empty")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_feats: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for b_idx, name in enumerate(benchmark_names):
        cache_file = cache_dir / f"{name}.pt"
        feats: np.ndarray
        targets: np.ndarray

        if cache_file.exists() and not force_recollect:
            try:
                d = torch.load(str(cache_file), weights_only=False)
                feats = d["features"]
                targets = d["targets"]
                print(f"  [{b_idx + 1}/{len(benchmark_names)}] loaded: {name}  "
                      f"({len(feats)} examples)")
            except Exception as exc:                              # noqa: BLE001
                if not collect_missing:
                    raise RuntimeError(
                        f"cache load failed for {name} ({cache_file}): {exc}"
                    ) from exc
                print(f"  [{b_idx + 1}/{len(benchmark_names)}] cache unreadable "
                      f"({exc}); collecting {name}")
                feats, targets = _collect_one_benchmark(
                    name, num_examples_per_benchmark or 0,
                    seed=seed + b_idx * 7919, drift_prob=drift_prob,
                    use_reg_feature=use_reg_feature,
                )
                _save_benchmark_cache(
                    cache_file, feats, targets,
                    num_examples=num_examples_per_benchmark or len(feats),
                    name=name,
                )
        elif collect_missing or force_recollect:
            if num_examples_per_benchmark is None:
                raise ValueError(
                    "num_examples_per_benchmark required when collect_missing or "
                    "force_recollect is True"
                )
            reason = "force_recollect=True" if force_recollect else "missing cache"
            print(f"  [{b_idx + 1}/{len(benchmark_names)}] {reason}; "
                  f"collecting {name}")
            feats, targets = _collect_one_benchmark(
                name, num_examples_per_benchmark,
                seed=seed + b_idx * 7919, drift_prob=drift_prob,
                use_reg_feature=use_reg_feature,
            )
            _save_benchmark_cache(
                cache_file, feats, targets,
                num_examples=num_examples_per_benchmark, name=name,
            )
        else:
            raise FileNotFoundError(
                f"per-benchmark cache missing: {cache_file}\n"
                f"Run parallel collection first (Modal ``.starmap``) or "
                f"``collect_data(..., collect_missing=True)``."
            )

        if (num_examples_per_benchmark is not None
                and len(feats) < num_examples_per_benchmark):
            print(f"  WARNING: {name} has {len(feats)} examples, "
                  f"expected {num_examples_per_benchmark}")

        all_feats.append(feats)
        all_targets.append(targets)
        if post_benchmark_callback is not None:
            try:
                post_benchmark_callback(name)
            except Exception as exc:                              # noqa: BLE001
                print(f"    post_benchmark_callback error for {name}: {exc}")

    return np.concatenate(all_feats, axis=0), np.concatenate(all_targets, axis=0)


def collect_data(benchmark_names: list[str], num_examples_per_benchmark: int,
                  seed: int, drift_prob: float = 0.3,
                  per_benchmark_cache_dir: Optional[Path] = None,
                  post_benchmark_callback: Optional[Callable[[str], None]] = None,
                  use_reg_feature: bool = False,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Collect (or load) per-benchmark data and concatenate.

    Prefer ``load_per_benchmark_caches`` when caches already exist (e.g.
    after Modal parallel collection). This function is for ``train.py`` CLI
    runs that may need to populate missing caches via ``collect_missing``.
    """
    if not benchmark_names:
        raise ValueError("collect_data: benchmark_names is empty")

    if per_benchmark_cache_dir is None:
        all_feats: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        for b_idx, name in enumerate(benchmark_names):
            print(f"\n  [{b_idx + 1}/{len(benchmark_names)}] benchmark = {name}")
            feats, targets = _collect_one_benchmark(
                name, num_examples_per_benchmark,
                seed=seed + b_idx * 7919, drift_prob=drift_prob,
                use_reg_feature=use_reg_feature,
            )
            all_feats.append(feats)
            all_targets.append(targets)
        return np.concatenate(all_feats, axis=0), np.concatenate(all_targets, axis=0)

    return load_per_benchmark_caches(
        benchmark_names, Path(per_benchmark_cache_dir),
        num_examples_per_benchmark=num_examples_per_benchmark,
        collect_missing=True, seed=seed, drift_prob=drift_prob,
        post_benchmark_callback=post_benchmark_callback,
        use_reg_feature=use_reg_feature,
    )


# ── Training ───────────────────────────────────────────────────────────────

def _synthesize_group_ids(n: int, n_calibration_samples: int,
                          main_group_size: int = 8,
                          calibration_group_size: int = 4) -> np.ndarray:
    """Synthesize group_ids for the standard data layout ``[cal | main]``.

    Tier-1 Fix B: existing per-benchmark caches don't store group_ids, but
    ``_collect_one_benchmark`` and ``collect_calibration_samples`` both lay
    out their samples in contiguous (macro_idx, base_state) groups — 8
    candidates per main-group, 4 per calibration-group. The pairwise rank
    loss only needs to know which examples are siblings (collected from the
    same parent state), so we reconstruct group_ids from this layout.

    Layout assumption: when calibration data is prepended (``train_iter``
    Step 1a does this), ``features[:n_calibration_samples]`` is the
    calibration block and the rest is the main block.
    """
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    ids = np.zeros(n, dtype=np.int64)
    cur = 0
    if n_calibration_samples > 0:
        cal_n = min(n_calibration_samples, n)
        ids[:cal_n] = np.arange(cal_n) // calibration_group_size
        cur = int(ids[:cal_n].max()) + 1
    if n > n_calibration_samples:
        rest = n - n_calibration_samples
        ids[n_calibration_samples:] = (np.arange(rest) // main_group_size) + cur
    return ids


def _pairwise_rank_loss(pred: torch.Tensor, true: torch.Tensor,
                        pair_left: torch.Tensor, pair_right: torch.Tensor,
                        pair_sign: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """Margin ranking loss over within-group pairs (Fix B).

    For each pair ``(i, j)`` where ``true[i] - true[j]`` has known sign
    ``s``, penalize predictions that disagree: ``ReLU(margin - s * (pred[i]
    - pred[j]))``. Pre-computed pair indices and signs allow vectorized
    evaluation. ``pred`` and ``true`` are the network output and targets
    for a flat set of examples (positions referenced by ``pair_*``).
    """
    if len(pair_left) == 0:
        return torch.tensor(0.0, dtype=pred.dtype)
    diff_pred = pred[pair_left] - pred[pair_right]
    return F.relu(margin - pair_sign * diff_pred).mean()


def _spearman_corr(pred: np.ndarray, true: np.ndarray) -> float:
    """Rank correlation, the metric the placer actually cares about
    (argmin-over-candidates is a ranking decision)."""
    if len(pred) < 2:
        return 0.0
    return float(np.corrcoef(np.argsort(np.argsort(pred)),
                              np.argsort(np.argsort(true)))[0, 1])


def train_model(features: np.ndarray, targets: np.ndarray, *,
                epochs: int, batch_size: int, lr: float, hidden: int,
                val_frac: float, seed: int,
                n_calibration_samples: int = 0,
                rank_loss_enabled: bool = True,
                rank_weight: float = 1.0,
                rank_margin: float = 0.1,
                main_group_size: int = 8,
                calibration_group_size: int = 4,
                patience: int | None = None) -> tuple[nn.Module, dict]:
    """Supervised regression with optional pairwise rank loss.

    Tier-1 Fix A: best checkpoint is selected by **val Spearman ρ**, not
    val Smooth-L1. The placer uses the model as an argmin-over-candidates
    scorer, so rank correlation is the relevant metric. Pre-fix Smooth-L1
    selection preferred amplitude-correct but rank-poor checkpoints; the
    100-epoch ``long12h`` run showed train_loss → 0.003 with Spearman
    plateauing at ~0.83, then degrading.

    Tier-1 Fix B: when ``rank_loss_enabled`` is True (default) and groups
    can be synthesized from the standard data layout, a pairwise margin
    ranking loss is added. Within-group pairs are pre-computed once and
    evaluated as a single tensor op per epoch.

    Early stopping kicks in after ``patience`` epochs without Spearman
    improvement (default ``max(10, epochs // 5)``).
    """
    rng = np.random.RandomState(seed)
    n = len(features)

    if patience is None:
        patience = max(10, epochs // 5)

    # Group ids drive both the train/val split and rank-loss pair construction.
    # Splitting by group prevents within-group pairs from straddling the split,
    # which would leak rank information into the val Spearman metric.
    if rank_loss_enabled:
        group_ids = _synthesize_group_ids(
            n, n_calibration_samples,
            main_group_size=main_group_size,
            calibration_group_size=calibration_group_size,
        )
        unique_groups = np.unique(group_ids)
        group_perm = rng.permutation(unique_groups)
        n_val_groups = max(1, int(len(unique_groups) * val_frac))
        val_group_set = set(int(g) for g in group_perm[:n_val_groups].tolist())
        is_val = np.fromiter((int(g) in val_group_set for g in group_ids),
                              dtype=bool, count=n)
        val_idx = np.where(is_val)[0]
        train_idx = np.where(~is_val)[0]
    else:
        perm = rng.permutation(n)
        n_val = int(n * val_frac)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        group_ids = None

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    feat_mean = X[train_idx].mean(0)
    feat_std = X[train_idx].std(0).clamp(min=1e-6)
    X = (X - feat_mean) / feat_std

    target_mean = float(y[train_idx].mean())
    target_std = float(y[train_idx].std().clamp(min=1e-6))
    y_norm = (y - target_mean) / target_std

    model = CostApproximator(in_dim=features.shape[1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    X_train, y_train = X[train_idx], y_norm[train_idx]
    X_val, y_val_norm = X[val_idx], y_norm[val_idx]
    y_val_raw = y[val_idx]

    # Pre-compute within-group pair tensors (train side only). Pairs are
    # referenced by position within X_train, not by original cache index.
    train_pair_left = train_pair_right = train_pair_sign = None
    if rank_loss_enabled and group_ids is not None:
        orig_to_train_pos: dict[int, int] = {int(o): pos
                                              for pos, o in enumerate(train_idx)}
        # Bucket positions in X_train by their group id.
        train_groups: dict[int, list[int]] = {}
        for orig in train_idx:
            g = int(group_ids[orig])
            train_groups.setdefault(g, []).append(orig_to_train_pos[int(orig)])

        left, right, sign = [], [], []
        for positions in train_groups.values():
            k = len(positions)
            if k < 2:
                continue
            for i in range(k):
                for j in range(i + 1, k):
                    a, b = positions[i], positions[j]
                    diff = float(y_train[a].item() - y_train[b].item())
                    if abs(diff) < 1e-9:
                        continue
                    left.append(a)
                    right.append(b)
                    sign.append(1.0 if diff > 0 else -1.0)
        if left:
            train_pair_left = torch.tensor(left, dtype=torch.long)
            train_pair_right = torch.tensor(right, dtype=torch.long)
            train_pair_sign = torch.tensor(sign, dtype=torch.float32)
            print(f"  rank loss: {len(left):,} pairs from "
                  f"{len(train_groups)} groups "
                  f"(weight={rank_weight}, margin={rank_margin})")
        else:
            print(f"  rank loss: no usable pairs (group_size too small)")

    best_state = None
    best_spearman = -float("inf")
    best_epoch = 0
    epochs_since_improvement = 0

    for epoch in range(epochs):
        # ── Smooth-L1 pass over mini-batches ──────────────────────────────
        model.train()
        idx = torch.randperm(len(X_train))
        total = 0.0
        for i in range(0, len(X_train), batch_size):
            b = idx[i:i + batch_size]
            pred = model(X_train[b])
            loss = loss_fn(pred, y_train[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        train_loss = total / len(X_train)

        # ── Pairwise rank pass (full-data, chunked for memory) ────────────
        rank_loss_avg = 0.0
        if train_pair_left is not None:
            n_pairs = len(train_pair_left)
            # Cap pair evaluations per epoch to bound time on huge datasets.
            max_pairs_per_epoch = 30_000
            if n_pairs > max_pairs_per_epoch:
                sub = torch.randperm(n_pairs)[:max_pairs_per_epoch]
                pl = train_pair_left[sub]
                pr = train_pair_right[sub]
                ps = train_pair_sign[sub]
            else:
                pl, pr, ps = train_pair_left, train_pair_right, train_pair_sign
            chunk = 4096
            n_used = 0
            for k in range(0, len(pl), chunk):
                la = pl[k:k + chunk]
                ra = pr[k:k + chunk]
                sa = ps[k:k + chunk]
                # Forward each leg separately so backward through both
                # contributes to model params.
                pred_l = model(X_train[la])
                pred_r = model(X_train[ra])
                rank_loss = F.relu(rank_margin - sa * (pred_l - pred_r)).mean()
                opt.zero_grad()
                (rank_weight * rank_loss).backward()
                opt.step()
                rank_loss_avg += float(rank_loss.item()) * len(la)
                n_used += len(la)
            rank_loss_avg /= max(n_used, 1)

        # ── Evaluation ────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
        val_loss = float(loss_fn(val_pred, y_val_norm))
        val_pred_raw = val_pred.numpy() * target_std + target_mean
        y_val_np = y_val_raw.numpy()
        pearson = float(np.corrcoef(val_pred_raw, y_val_np)[0, 1])
        spearman_val = _spearman_corr(val_pred_raw, y_val_np)

        # Best by Spearman — the placer ranks candidates, so this is the
        # metric that actually predicts downstream proxy improvement.
        if spearman_val > best_spearman:
            best_spearman = spearman_val
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch % 5 == 0 or epoch == epochs - 1:
            extra = f"  rank={rank_loss_avg:.4f}" if train_pair_left is not None else ""
            print(f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"pearson_r={pearson:.3f}  spearman_ρ={spearman_val:.3f}{extra}")

        if epochs_since_improvement >= patience:
            print(f"  early-stop at epoch {epoch} "
                  f"(best Spearman={best_spearman:.3f} @ epoch {best_epoch}, "
                  f"patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  loaded best-Spearman checkpoint from epoch {best_epoch}")

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
    val_pred_raw = val_pred.numpy() * target_std + target_mean
    y_val_np = y_val_raw.numpy()
    pearson_final = float(np.corrcoef(val_pred_raw, y_val_np)[0, 1])
    # Spearman = Pearson on rank-transformed arrays (matches plan's "rank corr").
    pred_ranks = np.argsort(np.argsort(val_pred_raw))
    true_ranks = np.argsort(np.argsort(y_val_np))
    spearman_final = float(np.corrcoef(pred_ranks, true_ranks)[0, 1])
    return model, dict(
        feat_mean=feat_mean,
        feat_std=feat_std,
        target_mean=target_mean,
        target_std=target_std,
        pearson=pearson_final,
        spearman=spearman_final,
        n_train=len(train_idx),
        n_val=len(val_idx),
    )


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="ibm01",
                   help="Single name, comma-separated list, or 'all' for the 17 IBM benchmarks. "
                        "Example: --benchmark ibm01,ibm02,ibm07")
    p.add_argument("--num-examples", type=int, default=1500,
                   help="Per-benchmark example count. Total dataset = N * len(benchmarks).")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-out",
                   default=str(_HERE / "data" / "chain_data.pt"))
    p.add_argument("--checkpoint-out",
                   default=str(_HERE / "checkpoints" / "cost_approximator.pt"))
    p.add_argument("--force-recollect", action="store_true",
                   help="Re-run data collection even if cached data exists.")
    p.add_argument("--use-reg-feature", action="store_true",
                   help="MaskRegulate Task 1c: emit a 17th feature "
                        "(reg_before - reg_after) in _features_for_move. "
                        "Invalidates existing 16-d checkpoints; saves "
                        "use_reg_feature=True in the new ckpt for "
                        "downstream loader compatibility.")
    # Step 1 (ranking lever): the placer picks moves by argmin over the
    # approximator's per-candidate scores, so within-state *ranking* — not
    # amplitude — is what matters. The pairwise margin rank loss (Fix B) is
    # on by default but its strength was previously hardcoded; expose it so
    # the rank-weight=0 vs >0 ablation is a one-flag change.
    p.add_argument("--rank-weight", type=float, default=1.0,
                   help="Weight on the within-state pairwise margin rank loss. "
                        "0.0 = pure Smooth-L1 regression (the ablation baseline); "
                        ">1.0 leans harder on ranking. Default 1.0.")
    p.add_argument("--rank-margin", type=float, default=0.1,
                   help="Margin (in normalized-target units) for the rank loss. "
                        "Default 0.1.")
    p.add_argument("--no-rank-loss", action="store_true",
                   help="Disable the pairwise rank loss entirely (equivalent to "
                        "--rank-weight 0 but also skips pair construction).")
    args = p.parse_args()

    benchmarks = parse_benchmarks(args.benchmark)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== Phase 3: Cost Approximator ===")
    print(f"  benchmarks = {benchmarks}")
    print(f"  examples/benchmark = {args.num_examples}, "
          f"total = {args.num_examples * len(benchmarks)}")

    data_path = Path(args.data_out)
    use_cache = False
    if data_path.exists() and not args.force_recollect:
        cached = torch.load(data_path, weights_only=False)
        # Backwards compat: old caches stored "benchmark" (singular string).
        cached_benchmarks = cached.get("benchmarks")
        if cached_benchmarks is None:
            singleton = cached.get("benchmark")
            cached_benchmarks = [singleton] if singleton else []
        if set(cached_benchmarks) == set(benchmarks):
            print(f"\n[1/3] Loading cached data from {data_path}  "
                  f"(benchmarks match: {sorted(cached_benchmarks)})")
            features, targets = cached["features"], cached["targets"]
            use_cache = True
        else:
            print(f"\n[1/3] Cached data covers {sorted(cached_benchmarks)} "
                  f"but request is {sorted(benchmarks)} — re-collecting")

    if not use_cache:
        print(f"\n[1/3] Collecting on {benchmarks}  "
              f"(use_reg_feature={args.use_reg_feature})...")
        features, targets = collect_data(
            benchmarks, args.num_examples, args.seed,
            use_reg_feature=args.use_reg_feature,
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "targets": targets,
                    "benchmarks": benchmarks,
                    "feature_dim": features.shape[1],
                    "use_reg_feature": bool(args.use_reg_feature)},
                   data_path)
        print(f"  saved {len(features)} examples ({features.shape[1]}-d) "
              f"to {data_path}")

    print(f"\n[2/3] Training ({features.shape[0]} examples, {features.shape[1]} feats)...")
    model, info = train_model(
        features, targets,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        hidden=args.hidden, val_frac=args.val_frac, seed=args.seed,
        rank_loss_enabled=not args.no_rank_loss,
        rank_weight=args.rank_weight,
        rank_margin=args.rank_margin,
    )

    print(f"\n[3/3] Final correlation (val):")
    print(f"    Pearson r  = {info['pearson']:.4f}")
    print(f"    Spearman ρ = {info['spearman']:.4f}  (plan's rank-correlation gate)")
    plan_gate = info["spearman"] > 0.8
    mvp_gate = info["spearman"] > 0.5
    print(f"  plan exit criterion ( ρ > 0.8 ): {'PASS' if plan_gate else 'FAIL'}")
    print(f"  MVP gate          ( ρ > 0.5 ): {'PASS' if mvp_gate else 'FAIL'}")

    ckpt = {
        "state_dict": model.state_dict(),
        "feat_mean": info["feat_mean"],
        "feat_std": info["feat_std"],
        "target_mean": info["target_mean"],
        "target_std": info["target_std"],
        "feature_dim": features.shape[1],
        "hidden": args.hidden,
        "pearson_r_val": info["pearson"],
        "spearman_r_val": info["spearman"],
        "trained_on": benchmarks,
        "n_train": info["n_train"],
        "n_val": info["n_val"],
        # Task 1c: schema flag so the placer's loader picks up
        # use_reg_feature from the ckpt and routes _features_for_move
        # accordingly.
        "use_reg_feature": bool(args.use_reg_feature),
    }
    ckpt_path = Path(args.checkpoint_out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"  checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
