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

import numpy as np
import torch
import torch.nn as nn

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
                            drift_prob: float = 0.3
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Single-benchmark state-drift sampler. Returns (features [N, F], targets [N]).

    The plc object is released when this function returns so the next
    benchmark in a multi-benchmark sweep doesn't accumulate memory.
    """
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark_name
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    edges, edge_weights = _placer._hard_macro_edges(benchmark)
    state = _placer.PlacementState(benchmark, edges, edge_weights)
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

    t_collect = time.time()
    last_log = t_collect
    while len(feats_list) < num_examples:
        macro_idx = int(rng.choice(movable))
        cands = state.candidate_positions(macro_idx, num_candidates=8, rng=py_rng)
        cand_costs: list[tuple[tuple[float, float], float]] = []
        for cx, cy in cands:
            old = state.pos[macro_idx].copy()
            delta = np.array([cx - old[0], cy - old[1]], dtype=np.float64)

            feats = _placer._features_for_move(state, macro_idx, delta)

            state.pos[macro_idx, 0] = cx
            state.pos[macro_idx, 1] = cy
            new_cost = exact_cost()
            state.pos[macro_idx] = old

            feats_list.append(feats)
            targets_list.append(new_cost - base_cost)
            cand_costs.append(((cx, cy), new_cost))

            if len(feats_list) >= num_examples:
                break

        # Maybe drift: commit the best candidate as the new base if it doesn't
        # increase overlap pairs catastrophically.
        if cand_costs and py_rng.random() < drift_prob:
            (cx, cy), new_cost = min(cand_costs, key=lambda t: t[1])
            state.pos[macro_idx, 0] = cx
            state.pos[macro_idx, 1] = cy
            base_cost = new_cost

        if time.time() - last_log > 5.0:
            elapsed = time.time() - t_collect
            rate = len(feats_list) / max(elapsed, 1e-6)
            eta = (num_examples - len(feats_list)) / max(rate, 1e-6)
            print(f"  collected {len(feats_list):4d}/{num_examples}  "
                  f"({elapsed:.1f}s, rate={rate:.1f}/s, eta={eta:.0f}s, base={base_cost:.3f})")
            last_log = time.time()

    elapsed = time.time() - t_collect
    print(f"  done in {elapsed:.1f}s "
          f"({len(feats_list) / max(elapsed, 1e-6):.1f} examples/s)")
    return np.stack(feats_list), np.array(targets_list, dtype=np.float32)


def collect_data(benchmark_names: list[str], num_examples_per_benchmark: int,
                  seed: int, drift_prob: float = 0.3
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Multi-benchmark wrapper. Concatenates per-benchmark arrays.

    Each benchmark contributes ``num_examples_per_benchmark`` samples; the
    seed is offset per benchmark so different benchmarks don't share the
    same RNG trajectory. Total dataset size = N × len(benchmark_names).
    """
    if not benchmark_names:
        raise ValueError("collect_data: benchmark_names is empty")

    all_feats: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for b_idx, name in enumerate(benchmark_names):
        print(f"\n  [{b_idx + 1}/{len(benchmark_names)}] benchmark = {name}")
        feats, targets = _collect_one_benchmark(
            name, num_examples_per_benchmark,
            seed=seed + b_idx * 7919,   # arbitrary stride so offsets don't collide
            drift_prob=drift_prob,
        )
        all_feats.append(feats)
        all_targets.append(targets)
    return np.concatenate(all_feats, axis=0), np.concatenate(all_targets, axis=0)


# ── Training ───────────────────────────────────────────────────────────────

def train_model(features: np.ndarray, targets: np.ndarray, *,
                epochs: int, batch_size: int, lr: float, hidden: int,
                val_frac: float, seed: int) -> tuple[nn.Module, dict]:
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
            b = idx[i:i + batch_size]
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
            print(f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"pearson_r={pearson:.3f}")

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
        print(f"\n[1/3] Collecting on {benchmarks}...")
        features, targets = collect_data(benchmarks, args.num_examples, args.seed)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "targets": targets,
                    "benchmarks": benchmarks, "feature_dim": FEATURE_DIM},
                   data_path)
        print(f"  saved {len(features)} examples to {data_path}")

    print(f"\n[2/3] Training ({features.shape[0]} examples, {features.shape[1]} feats)...")
    model, info = train_model(
        features, targets,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        hidden=args.hidden, val_frac=args.val_frac, seed=args.seed,
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
