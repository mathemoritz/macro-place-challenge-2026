"""Iterative training loop (single or multi-benchmark).

Each round:
    1. Load (or re-collect) per-benchmark caches (``per_bench/<name>.pt``)
       and train the CostApproximator on the combined dataset.
       Caches are populated by Modal parallel collection or
       ``train.collect_data``. Data is RE-COLLECTED EACH ROUND by
       default — the iterative loop's stated purpose is to drift
       training data toward the inference distribution, and the earlier
       loop silently bypassed ``--force-recollect-each-round`` for
       per-benchmark caches.
    2. Train the ChainPolicy with PPO sampling rollouts uniformly across
       the same benchmark list.
    3. Run end-to-end placer evaluation on EACH benchmark and report
       proxy / overlaps / wall-time.
    4. Calibration sampling: capture (features, true_Δproxy) at
       inference-time states and stash for next round's training data.

Multi-benchmark training addresses the generalization gap — both models
otherwise overfit to whichever single benchmark they were trained on.

Usage:
    uv run python submissions/lkh/train_iter.py --rounds 3 --examples 500
    uv run python submissions/lkh/train_iter.py --benchmark ibm01,ibm02,ibm07
    uv run python submissions/lkh/train_iter.py --benchmark all --rounds 1

    # Reuse round-0 caches (e.g. when Modal pre-populated them) but still
    # re-collect rounds 1+:
    uv run python submissions/lkh/train_iter.py --rounds 3 --reuse-round-0-data
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

# Load helper modules without dragging the macro_place package init through
# importlib for a second time.
import train as _train          # CostApproximator trainer
import train_policy as _trainp  # PPO policy trainer
from lkh_model import FEATURE_DIM

# Mirror of ``modal_run.ITER_PRESETS`` (training knobs only; no output_tag).
PRESETS: dict[str, dict[str, int | float]] = {
    "long12h": {
        "rounds": 5,
        "examples": 240,
        "cost_epochs": 100,
        "policy_iterations": 3500,
        "trajectories_per_iter": 16,
        "eval_time_budget": 60.0,
        "calibration_samples_per_bench": 75,
        "calibration_time_budget": 15.0,
    },
    "medium4h": {
        "rounds": 3,
        "examples": 240,
        "cost_epochs": 50,
        "policy_iterations": 1000,
        "trajectories_per_iter": 4,
        "eval_time_budget": 25.0,
        "calibration_samples_per_bench": 25,
        "calibration_time_budget": 10.0,
    },
}


def run_round(round_idx: int, *, benchmarks: list[str], num_examples_per_benchmark: int,
              cost_epochs: int, policy_iterations: int,
              trajectories_per_iter: int, seed: int,
              data_path: Path, cost_ckpt: Path, policy_ckpt: Path,
              force_recollect: bool, eval_time_budget_s: float,
              output_root: Optional[Path] = None,
              per_benchmark_cache_dir: Optional[Path] = None,
              post_benchmark_callback: Optional[Callable[[str], None]] = None,
              collect_missing: bool = False,
              enable_calibration: bool = True,
              calibration_samples_per_bench: int = 50,
              calibration_time_budget_s: float = 10.0,
              # Regularity / wiremask flags
              gate_mode: str = "hpwl",
              reg_weight: float = 0.0,
              use_wiremask: bool = False,
              use_position_mask: bool = False,
              use_reg_feature: bool = False,
              rank_loss_enabled: bool = True,
              rank_weight: float = 1.0,
              rank_margin: float = 0.1,
              use_encoder: bool = False,
              encoder_hidden: int = 64,
              encoder_grid: int = 64,
              # Optional encoder/seed opt-ins
              feature_mode: str = "handcrafted",
              encoder_kind: str = "gnn",
              encoder_ckpt: str | None = None,
              scalar_lam: float = 0.01,
              seed_mode: str = "heuristic",
              terminal_reward_mode: str = "committed_gain",
              encoder_fine_tune_in_ppo: bool = False) -> dict:
    """Run one round of cost-model training + PPO + per-benchmark evaluation.

    Canonical writes (so the placer can find the latest checkpoints) go to
    ``cost_ckpt`` and ``policy_ckpt``. When ``output_root`` is provided, each
    round also gets an archival copy in ``output_root / f"round_{round_idx}/"``
    plus a per-round JSON record — this is what makes detached Modal runs
    crash-safe and lets you compare rounds after the fact.

    When ``per_benchmark_cache_dir`` is set, the data-loading step only
    **loads** existing ``<name>.pt`` caches (no re-collection). Populate
    caches via Modal parallel collection or ``train.collect_data`` before
    calling this.
    """
    print(f"\n========== Round {round_idx} ==========")
    print(f"  benchmarks = {benchmarks}")

    # ── Step 1: load training data + cost-model training ──────────────────
    use_cache = False
    bench_cache_dir = (
        Path(per_benchmark_cache_dir) if per_benchmark_cache_dir else None
    )
    per_bench_ready = (
        bench_cache_dir is not None
        and not force_recollect
        and _train.per_bench_caches_complete(benchmarks, bench_cache_dir)
    )

    if per_bench_ready:
        print(f"\n[Round {round_idx}] Step 1a: per-benchmark caches ready in "
              f"{bench_cache_dir} (skipping stale chain_data.pt if any)")
    elif not force_recollect and data_path.exists():
        cached = torch.load(data_path, weights_only=False)
        cached_benchmarks = cached.get("benchmarks")
        if cached_benchmarks is None:
            singleton = cached.get("benchmark")
            cached_benchmarks = [singleton] if singleton else []
        if set(cached_benchmarks) == set(benchmarks):
            print(f"\n[Round {round_idx}] Step 1a: reusing aggregated data at "
                  f"{data_path}")
            features, targets = cached["features"], cached["targets"]
            use_cache = True
        else:
            print(f"\n[Round {round_idx}] Step 1a: {data_path} covers "
                  f"{sorted(cached_benchmarks)} but need {sorted(benchmarks)} "
                  f"— will load or collect per-benchmark data")

    if not use_cache:
        total = num_examples_per_benchmark * len(benchmarks)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        if per_benchmark_cache_dir is not None:
            action = "force-recollect" if force_recollect else "load"
            print(f"\n[Round {round_idx}] Step 1a: {action} per-benchmark caches "
                  f"({num_examples_per_benchmark}/bench × {len(benchmarks)} "
                  f"= {total} total)")
            features, targets = _train.load_per_benchmark_caches(
                benchmarks, Path(per_benchmark_cache_dir),
                num_examples_per_benchmark=num_examples_per_benchmark,
                collect_missing=collect_missing,
                seed=seed + round_idx,
                post_benchmark_callback=post_benchmark_callback,
                force_recollect=force_recollect,
            )
        else:
            print(f"\n[Round {round_idx}] Step 1a: collect "
                  f"{num_examples_per_benchmark}/benchmark = {total} total")
            features, targets = _train.collect_data(
                benchmarks, num_examples_per_benchmark,
                seed=seed + round_idx,
            )
        # Prepend the previous round's calibration samples (if any).
        # Calibration captures (features, true_Δproxy) at inference-time
        # states; mixing them into the next round's training distribution
        # closes the within-round calibration gap.
        n_calibration_samples = 0
        if enable_calibration and round_idx > 0 and output_root is not None:
            prev_cal = output_root / f"round_{round_idx - 1}_calibration.pt"
            if prev_cal.exists():
                cal = torch.load(prev_cal, weights_only=False)
                features = np.concatenate([cal["features"], features], axis=0)
                targets = np.concatenate([cal["targets"], targets], axis=0)
                n_calibration_samples = int(len(cal["features"]))
                print(f"  prepended {n_calibration_samples} calibration "
                      f"samples from {prev_cal.name}")
        torch.save({"features": features, "targets": targets,
                    "benchmarks": benchmarks, "feature_dim": FEATURE_DIM,
                    "n_calibration_samples": n_calibration_samples},
                   data_path)
    else:
        # Cached aggregated chain_data.pt; recover the prepend count for the
        # rank-loss group synthesizer. Falls back to 0 for older caches that
        # predate the rank-loss field (the rank loss then treats the whole
        # array as main).
        n_calibration_samples = int(cached.get("n_calibration_samples", 0))

    print(f"\n[Round {round_idx}] Step 1b: train CostApproximator "
          f"({len(features)} examples, {cost_epochs} epochs)")
    model, info = _train.train_model(
        features, targets, epochs=cost_epochs, batch_size=64,
        lr=1e-3, hidden=64, val_frac=0.2, seed=seed + round_idx,
        n_calibration_samples=n_calibration_samples,
        rank_loss_enabled=rank_loss_enabled,
        rank_weight=rank_weight,
        rank_margin=rank_margin,
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
        "use_reg_feature": bool(use_reg_feature),
    }, cost_ckpt)
    print(f"  CostApproximator -> Pearson={info['pearson']:.3f}  "
          f"Spearman={info['spearman']:.3f}  saved to {cost_ckpt}")

    # ── Step 2: PPO policy training ───────────────────────────────────────
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
        # Regularity / wiremask flags
        gate_mode=gate_mode,
        reg_weight=reg_weight,
        use_wiremask=use_wiremask,
        use_position_mask=use_position_mask,
        approximator_ckpt=str(cost_ckpt),
        use_reg_feature=use_reg_feature,
        use_encoder=use_encoder,
        encoder_hidden=encoder_hidden,
        encoder_grid=encoder_grid,
        # Optional encoder/seed flags
        feature_mode=feature_mode,
        encoder_kind=encoder_kind,
        encoder_ckpt=encoder_ckpt,
        scalar_lam=scalar_lam,
        seed_mode=seed_mode,
        terminal_reward_mode=terminal_reward_mode,
        encoder_fine_tune_in_ppo=encoder_fine_tune_in_ppo,
    )

    # ── Step 3: end-to-end evaluation on every benchmark ──────────────────
    print(f"\n[Round {round_idx}] Step 3: evaluating placer end-to-end "
          f"on {len(benchmarks)} benchmark(s)")
    per_bench_eval = {}
    for name in benchmarks:
        print(f"  -- {name} --")
        per_bench_eval[name] = evaluate_round(
            name, time_budget_s=eval_time_budget_s,
            gate_mode=gate_mode, reg_weight=reg_weight,
            use_wiremask=use_wiremask, use_position_mask=use_position_mask,
        )

    # ── Step 4: calibration sampling ──────────────────────────────────────
    # Capture (features, true_Δproxy) pairs at the placer's inference-time
    # states so the next round's approximator sees the residual distribution
    # that single-move + cascade-interior sampling alone misses.
    if enable_calibration and output_root is not None:
        cal_path = output_root / f"round_{round_idx}_calibration.pt"
        all_cal_features: list[np.ndarray] = []
        all_cal_targets: list[np.ndarray] = []
        print(f"\n[Round {round_idx}] Step 4: calibration sampling "
              f"({calibration_samples_per_bench} samples × {len(benchmarks)} "
              f"benchmarks)")
        for b_idx, name in enumerate(benchmarks):
            try:
                cal_f, cal_t = _train.collect_calibration_samples(
                    name,
                    n_samples=calibration_samples_per_bench,
                    time_budget_s=calibration_time_budget_s,
                    approximator_ckpt_path=str(cost_ckpt) if cost_ckpt.exists() else None,
                    policy_ckpt_path=str(policy_ckpt) if policy_ckpt.exists() else None,
                    seed=seed + round_idx * 1009 + b_idx * 7919,
                    use_reg_feature=use_reg_feature,
                    use_wiremask=use_wiremask,
                    use_position_mask=use_position_mask,
                    reg_weight=reg_weight,
                    gate_mode=gate_mode,
                )
                all_cal_features.append(cal_f)
                all_cal_targets.append(cal_t)
                print(f"  {name}: collected {len(cal_f)} calibration samples")
            except Exception as exc:                                # noqa: BLE001
                # Calibration is best-effort; a failure on one benchmark
                # shouldn't crash the whole round.
                print(f"  {name}: calibration failed ({exc}); skipping")
        if all_cal_features:
            cal_features = np.concatenate(all_cal_features, axis=0)
            cal_targets = np.concatenate(all_cal_targets, axis=0)
            torch.save({"features": cal_features, "targets": cal_targets,
                        "benchmarks": benchmarks, "round": round_idx},
                       cal_path)
            print(f"  saved {len(cal_features)} calibration samples -> {cal_path}")

    record = {
        "round": round_idx,
        "pearson": info["pearson"],
        "spearman": info["spearman"],
        "per_benchmark": per_bench_eval,
    }

    # Per-round archival: keeps every round's checkpoint so we can compare
    # across the iterative loop and so a crashed Modal run still leaves the
    # finished rounds' data on the volume.
    if output_root is not None:
        round_dir = output_root / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        if cost_ckpt.exists():
            shutil.copy(str(cost_ckpt), str(round_dir / cost_ckpt.name))
        if policy_ckpt.exists():
            shutil.copy(str(policy_ckpt), str(round_dir / policy_ckpt.name))
        (round_dir / "round_record.json").write_text(
            json.dumps(record, indent=2, default=str)
        )
        print(f"  archived to {round_dir}")

    return record


def evaluate_round(benchmark: str, *, time_budget_s: float = 20.0,
                   gate_mode: str = "hpwl", reg_weight: float = 0.0,
                   use_wiremask: bool = False,
                   use_position_mask: bool = False) -> dict:
    """Run the placer + exact proxy cost on a single benchmark."""
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    _spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
    placer_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(placer_mod)

    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark
    bench, plc = load_benchmark_from_dir(str(bench_dir))

    placer = placer_mod.LKHPlacer(seed=123, time_budget_s=time_budget_s,
                                    max_chains=2000, max_chain_length=8,
                                    gate_mode=gate_mode,
                                    reg_weight=reg_weight,
                                    use_wiremask=use_wiremask,
                                    use_position_mask=use_position_mask)
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
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="Apply a hyperparameter bundle (overrides other training "
                        "flags except --benchmark and paths). Use 'long12h' for "
                        "~12 h Modal run; prefer modal_run.py::iter_ --preset long12h on cloud.")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--examples", type=int, default=400,
                   help="Per-benchmark example count for data collection.")
    p.add_argument("--cost-epochs", type=int, default=60)
    p.add_argument("--policy-iterations", type=int, default=100)
    p.add_argument("--trajectories-per-iter", type=int, default=4)
    p.add_argument("--eval-time-budget", type=float, default=20.0,
                   help="Seconds per benchmark during the per-round evaluation step.")
    p.add_argument("--seed", type=int, default=42)
    # The iterative loop's stated purpose is to drift training data toward
    # the inference distribution, so re-collect is the right default. The
    # flag invalidates per-benchmark caches too, not just the aggregated
    # cache.
    p.add_argument("--force-recollect-each-round",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Re-collect training data every round. Default: True. "
                        "Pass --no-force-recollect-each-round to keep all rounds "
                        "on the original cache (legacy behavior).")
    p.add_argument("--reuse-round-0-data", action="store_true",
                   help="Override --force-recollect-each-round for round 0 only "
                        "(reuse pre-existing per-benchmark caches in round 0). "
                        "Useful when round 0 was populated by a Modal pre-fetch.")
    p.add_argument("--data-path",
                   default=str(_HERE / "data" / "chain_data.pt"))
    p.add_argument("--cost-ckpt",
                   default=str(_HERE / "checkpoints" / "cost_approximator.pt"))
    p.add_argument("--policy-ckpt",
                   default=str(_HERE / "checkpoints" / "chain_policy.pt"))
    p.add_argument("--output-root",
                   default=str(_HERE / "iter_output"),
                   help="Directory for per-round archival (round_<i>/ + "
                        "history.json). Set to empty string to disable.")
    p.add_argument("--per-benchmark-cache-dir",
                   default=str(_HERE / "data" / "per_bench"),
                   help="Directory of per-benchmark cache files. Empty string disables.")
    p.add_argument("--collect-missing", action="store_true",
                   help="If a per-benchmark cache is missing, collect it instead "
                        "of failing. Modal runs should leave this off.")
    p.add_argument("--no-calibration", action="store_true",
                   help="Disable within-round calibration sampling. "
                        "Default: calibration is on.")
    p.add_argument("--calibration-samples-per-bench", type=int, default=50,
                   help="Per-benchmark calibration samples after each "
                        "round's eval (default 50).")
    p.add_argument("--calibration-time-budget", type=float, default=10.0,
                   help="Seconds per-benchmark for the calibration "
                        "placer pass (default 10).")
    # Regularity / wiremask flags. Default OFF preserves legacy behavior;
    # newer runs flip them on as a bundle.
    p.add_argument("--use-reg-feature", action="store_true",
                   help="17-d features with regularity Δ.")
    p.add_argument("--reg-weight", type=float, default=0.0,
                   help="Regularity blend in chain gate + PPO reward.")
    p.add_argument("--use-wiremask", action="store_true",
                   help="Inject analytic HPWL-best cells as candidates.")
    p.add_argument("--use-position-mask", action="store_true",
                   help="Legality-filter WireMask candidates.")
    # gate-mode supports hpwl/predicted_proxy + 'scalar_penalty'.
    p.add_argument("--gate-mode",
                   choices=("hpwl", "predicted_proxy", "scalar_penalty"),
                   default="hpwl",
                   help="Third lex coord of the commit gate "
                        "(or scalar score under 'scalar_penalty').")
    # Expose the rank-loss knobs to the iterative loop so a full-suite
    # rank-weight sweep is a single flag.
    p.add_argument("--rank-weight", type=float, default=1.0,
                   help="Weight on the CostApproximator's within-state pairwise "
                        "rank loss. 0.0 = pure regression baseline.")
    p.add_argument("--rank-margin", type=float, default=0.1,
                   help="Margin for the rank loss (normalized-target units).")
    p.add_argument("--no-rank-loss", action="store_true",
                   help="Disable the pairwise rank loss entirely.")
    # Co-train the GNN+CNN encoder with the policy.
    p.add_argument("--use-encoder", action="store_true",
                   help="Co-train the GNN+CNN StateEncoder and feed its "
                        "embeddings to the policy (and at inference).")
    p.add_argument("--encoder-hidden", type=int, default=64,
                   help="StateEncoder hidden dim. Default 64.")
    p.add_argument("--encoder-grid", type=int, default=64,
                   help="Per-macro mask raster resolution. Default 64.")
    # Optional encoder/seed flags. Defaults preserve legacy behavior.
    p.add_argument("--feature-mode", choices=["handcrafted", "encoder"],
                   default="handcrafted",
                   help="'encoder' uses GNN encoder features in PPO.")
    p.add_argument("--encoder-kind", choices=["gnn", "gnncnn"], default="gnn")
    p.add_argument("--encoder-ckpt", default=None,
                   help="Path to encoder checkpoint (default: checkpoints/encoder.pt).")
    p.add_argument("--scalar-lam", type=float, default=0.01)
    p.add_argument("--seed-mode", choices=["heuristic", "policy"],
                   default="heuristic",
                   help="'policy' uses a learned SeedSelectionHead.")
    p.add_argument("--terminal-reward-mode",
                   choices=["committed_gain", "hpwl_telescope_legacy",
                            "predicted_proxy_with_postleg"],
                   default="committed_gain",
                   help="'predicted_proxy_with_postleg' rewards post-legalization "
                        "predicted Δproxy via the approximator.")
    p.add_argument("--encoder-fine-tune-in-ppo", action="store_true",
                   help="Experimental: also update encoder during PPO. Default off.")
    args = p.parse_args()

    if args.preset is not None:
        for key, val in PRESETS[args.preset].items():
            setattr(args, key, val)
        print(f"  preset={args.preset!r} applied")

    benchmarks = _train.parse_benchmarks(args.benchmark)
    output_root = Path(args.output_root) if args.output_root else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
    per_benchmark_cache_dir = (
        Path(args.per_benchmark_cache_dir) if args.per_benchmark_cache_dir else None
    )

    print(f"=== Iterative training ===")
    print(f"  benchmarks={benchmarks}  rounds={args.rounds}")
    print(f"  examples/benchmark={args.examples}  cost_epochs={args.cost_epochs}  "
          f"policy_iters={args.policy_iterations}")
    if output_root is not None:
        print(f"  output_root={output_root}")

    history: list[dict] = []
    for r in range(args.rounds):
        # Round 0 may opt back into cache reuse via --reuse-round-0-data.
        # Rounds >= 1 always honor --force-recollect-each-round (default True).
        force_recollect_r = args.force_recollect_each_round and not (
            r == 0 and args.reuse_round_0_data
        )
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
            force_recollect=force_recollect_r,
            eval_time_budget_s=args.eval_time_budget,
            output_root=output_root,
            per_benchmark_cache_dir=per_benchmark_cache_dir,
            collect_missing=args.collect_missing,
            enable_calibration=not args.no_calibration,
            calibration_samples_per_bench=args.calibration_samples_per_bench,
            calibration_time_budget_s=args.calibration_time_budget,
            # Regularity / wiremask flags
            gate_mode=args.gate_mode,
            reg_weight=args.reg_weight,
            use_wiremask=args.use_wiremask,
            use_position_mask=args.use_position_mask,
            use_reg_feature=args.use_reg_feature,
            rank_loss_enabled=not args.no_rank_loss,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            use_encoder=args.use_encoder,
            encoder_hidden=args.encoder_hidden,
            encoder_grid=args.encoder_grid,
            # Optional encoder/seed flags
            feature_mode=args.feature_mode,
            encoder_kind=args.encoder_kind,
            encoder_ckpt=args.encoder_ckpt,
            scalar_lam=args.scalar_lam,
            seed_mode=args.seed_mode,
            terminal_reward_mode=args.terminal_reward_mode,
            encoder_fine_tune_in_ppo=args.encoder_fine_tune_in_ppo,
        )
        history.append(record)
        # Stream the aggregated history to disk after every round so a crash
        # leaves the previous rounds analysable.
        if output_root is not None:
            (output_root / "history.json").write_text(
                json.dumps(history, indent=2, default=str)
            )

    print(f"\n=== Iterative training summary ===")
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
