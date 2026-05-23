"""Phase 5 — iterative training loop (single or multi-benchmark).

Each round:
    1. Collect (state, move, exact Δcost) triples across all configured
       benchmarks; train the CostApproximator on the combined dataset
       (Phase 3).
    2. Train the ChainPolicy with PPO sampling rollouts uniformly across
       the same benchmark list (Phase 4).
    3. Run end-to-end placer evaluation on EACH benchmark and report
       proxy / overlaps / wall-time.

Multi-benchmark training addresses the generalization gap noted in the
review — both models otherwise overfit to whichever single benchmark they
were trained on.

Usage:
    uv run python submissions/lkh/train_iter.py --rounds 3 --examples 500
    uv run python submissions/lkh/train_iter.py --benchmark ibm01,ibm02,ibm07
    uv run python submissions/lkh/train_iter.py --benchmark all --rounds 1
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

# Load helper modules without dragging the macro_place package init through
# importlib for a second time.
import train as _train          # Phase 3 trainer
import train_policy as _trainp  # Phase 4 trainer
from lkh_model import FEATURE_DIM


def run_round(round_idx: int, *, benchmarks: list[str], num_examples_per_benchmark: int,
              cost_epochs: int, policy_iterations: int,
              trajectories_per_iter: int, seed: int,
              data_path: Path, cost_ckpt: Path, policy_ckpt: Path,
              force_recollect: bool, eval_time_budget_s: float) -> dict:
    print(f"\n========== Round {round_idx} ==========")
    print(f"  benchmarks = {benchmarks}")

    # ── Step 1: data collection + Phase 3 training ────────────────────────
    use_cache = False
    if not force_recollect and data_path.exists():
        cached = torch.load(data_path, weights_only=False)
        cached_benchmarks = cached.get("benchmarks")
        if cached_benchmarks is None:
            singleton = cached.get("benchmark")
            cached_benchmarks = [singleton] if singleton else []
        if set(cached_benchmarks) == set(benchmarks):
            print(f"\n[Round {round_idx}] Step 1a: reusing cached data at {data_path}")
            features, targets = cached["features"], cached["targets"]
            use_cache = True
        else:
            print(f"\n[Round {round_idx}] Step 1a: cached data covers "
                  f"{sorted(cached_benchmarks)} but request is {sorted(benchmarks)} "
                  f"— re-collecting")

    if not use_cache:
        total = num_examples_per_benchmark * len(benchmarks)
        print(f"\n[Round {round_idx}] Step 1a: collect "
              f"{num_examples_per_benchmark}/benchmark = {total} total")
        features, targets = _train.collect_data(
            benchmarks, num_examples_per_benchmark,
            seed=seed + round_idx,
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "targets": targets,
                    "benchmarks": benchmarks, "feature_dim": FEATURE_DIM},
                   data_path)

    print(f"\n[Round {round_idx}] Step 1b: train CostApproximator "
          f"({len(features)} examples, {cost_epochs} epochs)")
    model, info = _train.train_model(
        features, targets, epochs=cost_epochs, batch_size=64,
        lr=1e-3, hidden=64, val_frac=0.2, seed=seed + round_idx,
    )
    cost_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "feat_mean": info["feat_mean"], "feat_std": info["feat_std"],
        "target_mean": info["target_mean"], "target_std": info["target_std"],
        "feature_dim": features.shape[1], "hidden": 64,
        "pearson_r_val": info["pearson"], "spearman_r_val": info["spearman"],
        "trained_on": benchmarks, "round": round_idx,
        "n_train": info["n_train"], "n_val": info["n_val"],
    }, cost_ckpt)
    print(f"  CostApproximator -> Pearson={info['pearson']:.3f}  "
          f"Spearman={info['spearman']:.3f}  saved to {cost_ckpt}")

    # ── Step 2: Phase 4 PPO training ──────────────────────────────────────
    warm_start = policy_ckpt if (round_idx > 0 and policy_ckpt.exists()) else None
    print(f"\n[Round {round_idx}] Step 2: PPO policy ({policy_iterations} iters, "
          f"warm_start={'yes' if warm_start else 'no'})")
    _trainp.train_chain_policy(
        benchmarks,
        n_iterations=policy_iterations,
        trajectories_per_iter=trajectories_per_iter,
        max_chain_length=8, max_candidates=8,
        terminal_commit_bonus=0.0,
        seed=seed + round_idx,
        lr=3e-4, gamma=0.99, lam=0.95,
        clip=0.2, ent_coef=0.01, vf_coef=0.5, ppo_epochs=4,
        output_ckpt=policy_ckpt,
        initial_policy_ckpt=warm_start,
        log_every=max(policy_iterations // 5, 1),
    )

    # ── Step 3: end-to-end evaluation on every benchmark ──────────────────
    print(f"\n[Round {round_idx}] Step 3: evaluating placer end-to-end "
          f"on {len(benchmarks)} benchmark(s)")
    per_bench_eval = {}
    for name in benchmarks:
        print(f"  -- {name} --")
        per_bench_eval[name] = evaluate_round(name, time_budget_s=eval_time_budget_s)

    return {
        "round": round_idx,
        "pearson": info["pearson"],
        "spearman": info["spearman"],
        "per_benchmark": per_bench_eval,
    }


def evaluate_round(benchmark: str, *, time_budget_s: float = 20.0) -> dict:
    """Run the placer + exact proxy cost on a single benchmark."""
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    _spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
    placer_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(placer_mod)

    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark
    bench, plc = load_benchmark_from_dir(str(bench_dir))

    placer = placer_mod.LKHPlacer(seed=123, time_budget_s=time_budget_s,
                                    max_chains=2000, max_chain_length=8)
    t0 = time.time()
    placement = placer.place(bench)
    runtime = time.time() - t0
    costs = compute_proxy_cost(placement, bench, plc)
    print(f"    proxy={costs['proxy_cost']:.4f}  "
          f"wl={costs['wirelength_cost']:.3f}  "
          f"den={costs['density_cost']:.3f}  "
          f"cong={costs['congestion_cost']:.3f}  "
          f"overlaps={costs['overlap_count']}  "
          f"[{runtime:.1f}s]")
    return {
        "proxy_cost": float(costs["proxy_cost"]),
        "wirelength": float(costs["wirelength_cost"]),
        "density": float(costs["density_cost"]),
        "congestion": float(costs["congestion_cost"]),
        "overlaps": int(costs["overlap_count"]),
        "runtime_s": float(runtime),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="ibm01",
                   help="Single name, comma-separated list, or 'all' for the 17 IBM benchmarks.")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--examples", type=int, default=400,
                   help="Per-benchmark example count (Phase 3 data collection).")
    p.add_argument("--cost-epochs", type=int, default=60)
    p.add_argument("--policy-iterations", type=int, default=100)
    p.add_argument("--trajectories-per-iter", type=int, default=4)
    p.add_argument("--eval-time-budget", type=float, default=20.0,
                   help="Seconds per benchmark during the per-round evaluation step.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-recollect-each-round", action="store_true",
                   help="By default, data is re-collected only when missing or when the "
                        "cached benchmark set doesn't match. Set this to refresh data each round.")
    p.add_argument("--data-path",
                   default=str(_HERE / "data" / "chain_data.pt"))
    p.add_argument("--cost-ckpt",
                   default=str(_HERE / "checkpoints" / "cost_approximator.pt"))
    p.add_argument("--policy-ckpt",
                   default=str(_HERE / "checkpoints" / "chain_policy.pt"))
    args = p.parse_args()

    benchmarks = _train.parse_benchmarks(args.benchmark)

    print(f"=== Phase 5: iterative training ===")
    print(f"  benchmarks={benchmarks}  rounds={args.rounds}")
    print(f"  examples/benchmark={args.examples}  cost_epochs={args.cost_epochs}  "
          f"policy_iters={args.policy_iterations}")

    history: list[dict] = []
    for r in range(args.rounds):
        record = run_round(
            r,
            benchmarks=benchmarks,
            num_examples_per_benchmark=args.examples,
            cost_epochs=args.cost_epochs,
            policy_iterations=args.policy_iterations,
            trajectories_per_iter=args.trajectories_per_iter,
            seed=args.seed,
            data_path=Path(args.data_path),
            cost_ckpt=Path(args.cost_ckpt),
            policy_ckpt=Path(args.policy_ckpt),
            force_recollect=args.force_recollect_each_round,
            eval_time_budget_s=args.eval_time_budget,
        )
        history.append(record)

    print(f"\n=== Phase 5 summary ===")
    print(f"  Approximator quality per round:")
    print(f"    round | Pearson | Spearman")
    for h in history:
        print(f"    {h['round']:>5} | {h['pearson']:>7.3f} | {h['spearman']:>8.3f}")

    print(f"\n  Placer evaluation per round × benchmark:")
    print(f"    {'round':>5} | {'benchmark':>10} | {'proxy':>8} | {'overlaps':>8} | {'time(s)':>8}")
    for h in history:
        for name, m in h["per_benchmark"].items():
            print(f"    {h['round']:>5} | {name:>10} | "
                  f"{m['proxy_cost']:>8.4f} | {m['overlaps']:>8d} | {m['runtime_s']:>8.1f}")


if __name__ == "__main__":
    main()
