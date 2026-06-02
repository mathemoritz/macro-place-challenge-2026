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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # for lkh_model
sys.path.insert(0, str(_HERE.parent.parent))  # repo root for macro_place

# Load placer.py without triggering the package __init__ (which fails when
# the TILOS submodule isn't initialized in some environments).
_spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE.parent / "placer.py"))
_placer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_placer)

from submissions.lkh.model.lkh_model import CostApproximator, FEATURE_DIM

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


def _collect_one_benchmark(
    benchmark_name: str,
    num_examples: int,
    seed: int,
    drift_prob: float = 0.3,
    p_cascade: float = 0.5,
    cascade_min: int = 2,
    cascade_max: int = 4,
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
                ox = float(state.pos[i, 0])
                oy = float(state.pos[i, 1])
                state.apply_move(i, cx, cy)
                h = state.hpwl()
                state.apply_move(i, ox, oy)
                if h < best_h:
                    best_h = h
                    best_cx, best_cy = cx, cy
            if best_cx is None:
                continue
            ox = float(state.pos[i, 0])
            oy = float(state.pos[i, 1])
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

            feats = _placer._features_for_move(state, macro_idx, delta)

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
        for i, ox, oy in reversed(cascade_saved):
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
            print(
                f"  collected {n_done:4d}/{num_examples}  "
                f"(elapsed={_fmt_time(elapsed)}, "
                f"rate={_fmt_rate(n_done, elapsed)}, "
                f"eta={_fmt_time(eta)}, base={base_cost:.3f})"
            )
            last_log = time.time()

    elapsed = time.time() - t_collect
    print(
        f"  done in {_fmt_time(elapsed)} "
        f"({_fmt_rate(len(feats_list), elapsed)})  "
        f"cascade-interior rounds: {n_cascade_samples}"
    )
    return np.stack(feats_list), np.array(targets_list, dtype=np.float32)


def collect_calibration_samples(
    benchmark_name: str,
    n_samples: int,
    time_budget_s: float,
    approximator_ckpt_path: str | None,
    policy_ckpt_path: str | None,
    seed: int,
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
        seed=seed,
        time_budget_s=time_budget_s,
        checkpoint_path=approximator_ckpt_path,
        policy_path=policy_ckpt_path,
        use_policy=policy_ckpt_path is not None,
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
    if len(movable) == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), np.zeros(0, dtype=np.float32)

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
            feats = _placer._features_for_move(state, macro_idx, delta)
            state.apply_move(macro_idx, cx, cy)
            new_cost = exact_cost()
            state.apply_move(macro_idx, old_x, old_y)
            feats_list.append(feats)
            targets_list.append(new_cost - base_cost)
            if len(feats_list) >= n_samples:
                break
    return np.stack(feats_list), np.array(targets_list, dtype=np.float32)


def _save_benchmark_cache(
    cache_file: Path, feats: np.ndarray, targets: np.ndarray, *, num_examples: int, name: str
) -> None:
    """Atomic write of one per-benchmark cache file."""
    tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
    torch.save(
        {"features": feats, "targets": targets, "num_examples": num_examples, "name": name},
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
        except Exception:  # noqa: BLE001
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
                print(
                    f"  [{b_idx + 1}/{len(benchmark_names)}] loaded: {name}  "
                    f"({len(feats)} examples)"
                )
            except Exception as exc:  # noqa: BLE001
                if not collect_missing:
                    raise RuntimeError(
                        f"cache load failed for {name} ({cache_file}): {exc}"
                    ) from exc
                print(
                    f"  [{b_idx + 1}/{len(benchmark_names)}] cache unreadable "
                    f"({exc}); collecting {name}"
                )
                feats, targets = _collect_one_benchmark(
                    name,
                    num_examples_per_benchmark or 0,
                    seed=seed + b_idx * 7919,
                    drift_prob=drift_prob,
                )
                _save_benchmark_cache(
                    cache_file,
                    feats,
                    targets,
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
            print(f"  [{b_idx + 1}/{len(benchmark_names)}] {reason}; " f"collecting {name}")
            feats, targets = _collect_one_benchmark(
                name,
                num_examples_per_benchmark,
                seed=seed + b_idx * 7919,
                drift_prob=drift_prob,
            )
            _save_benchmark_cache(
                cache_file,
                feats,
                targets,
                num_examples=num_examples_per_benchmark,
                name=name,
            )
        else:
            raise FileNotFoundError(
                f"per-benchmark cache missing: {cache_file}\n"
                f"Run parallel collection first (Modal ``.starmap``) or "
                f"``collect_data(..., collect_missing=True)``."
            )

        if num_examples_per_benchmark is not None and len(feats) < num_examples_per_benchmark:
            print(
                f"  WARNING: {name} has {len(feats)} examples, "
                f"expected {num_examples_per_benchmark}"
            )

        all_feats.append(feats)
        all_targets.append(targets)
        if post_benchmark_callback is not None:
            try:
                post_benchmark_callback(name)
            except Exception as exc:  # noqa: BLE001
                print(f"    post_benchmark_callback error for {name}: {exc}")

    return np.concatenate(all_feats, axis=0), np.concatenate(all_targets, axis=0)


def collect_data(
    benchmark_names: list[str],
    num_examples_per_benchmark: int,
    seed: int,
    drift_prob: float = 0.3,
    per_benchmark_cache_dir: Optional[Path] = None,
    post_benchmark_callback: Optional[Callable[[str], None]] = None,
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
                name,
                num_examples_per_benchmark,
                seed=seed + b_idx * 7919,
                drift_prob=drift_prob,
            )
            all_feats.append(feats)
            all_targets.append(targets)
        return np.concatenate(all_feats, axis=0), np.concatenate(all_targets, axis=0)

    return load_per_benchmark_caches(
        benchmark_names,
        Path(per_benchmark_cache_dir),
        num_examples_per_benchmark=num_examples_per_benchmark,
        collect_missing=True,
        seed=seed,
        drift_prob=drift_prob,
        post_benchmark_callback=post_benchmark_callback,
    )


# ── Training ───────────────────────────────────────────────────────────────


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    val_frac: float,
    seed: int,
) -> tuple[nn.Module, dict]:
    """Standard supervised regression with normalized features and targets."""
    rng = np.random.RandomState(seed)
    n = len(features)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

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

    best_state = None
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(X_train))
        total = 0.0
        for i in range(0, len(X_train), batch_size):
            b = idx[i : i + batch_size]
            pred = model(X_train[b])
            loss = loss_fn(pred, y_train[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        train_loss = total / len(X_train)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
        val_loss = float(loss_fn(val_pred, y_val_norm))
        val_pred_raw = val_pred.numpy() * target_std + target_mean
        pearson = float(np.corrcoef(val_pred_raw, y_val_raw.numpy())[0, 1])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                f"pearson_r={pearson:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

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
    p.add_argument(
        "--benchmark",
        default="ibm01",
        help="Single name, comma-separated list, or 'all' for the 17 IBM benchmarks. "
        "Example: --benchmark ibm01,ibm02,ibm07",
    )
    p.add_argument(
        "--num-examples",
        type=int,
        default=1500,
        help="Per-benchmark example count. Total dataset = N * len(benchmarks).",
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-out", default=str(_HERE / "data" / "chain_data.pt"))
    p.add_argument("--checkpoint-out", default=str(_HERE / "checkpoints" / "cost_approximator.pt"))
    p.add_argument(
        "--force-recollect",
        action="store_true",
        help="Re-run data collection even if cached data exists.",
    )
    args = p.parse_args()

    benchmarks = parse_benchmarks(args.benchmark)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== Phase 3: Cost Approximator ===")
    print(f"  benchmarks = {benchmarks}")
    print(
        f"  examples/benchmark = {args.num_examples}, "
        f"total = {args.num_examples * len(benchmarks)}"
    )

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
            print(
                f"\n[1/3] Loading cached data from {data_path}  "
                f"(benchmarks match: {sorted(cached_benchmarks)})"
            )
            features, targets = cached["features"], cached["targets"]
            use_cache = True
        else:
            print(
                f"\n[1/3] Cached data covers {sorted(cached_benchmarks)} "
                f"but request is {sorted(benchmarks)} — re-collecting"
            )

    if not use_cache:
        print(f"\n[1/3] Collecting on {benchmarks}...")
        features, targets = collect_data(benchmarks, args.num_examples, args.seed)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": features,
                "targets": targets,
                "benchmarks": benchmarks,
                "feature_dim": FEATURE_DIM,
            },
            data_path,
        )
        print(f"  saved {len(features)} examples to {data_path}")

    print(f"\n[2/3] Training ({features.shape[0]} examples, {features.shape[1]} feats)...")
    model, info = train_model(
        features,
        targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        val_frac=args.val_frac,
        seed=args.seed,
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
        "trained_on": benchmarks,
        "n_train": info["n_train"],
        "n_val": info["n_val"],
    }
    ckpt_path = Path(args.checkpoint_out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"  checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
