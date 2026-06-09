"""Joint regression trainer for encoder + cost approximator.

Handles the regression-pretraining stage: collect snapshots, then for
each batch re-run encoder + approximator together with one joint
optimizer.

Writes two checkpoints:
- ``checkpoints/encoder.pt`` — encoder weights + architecture.
- ``checkpoints/cost_approximator.pt`` — approximator weights + the
  feature/target normalization stats + an ``encoder_kind`` field so the
  placer knows which encoder runtime to load.

Usage::

    uv run python submissions/lkh/train_encoder_joint.py \\
        --benchmark all --encoder-kind gnn --num-examples 200 --epochs 60
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent))

# Hard imports for things that don't need package-init dance.
from lkh_model import CostApproximator
import state_snapshot as _snap

# Lazy imports to dodge package-init issues.
_placer_spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
_placer = importlib.util.module_from_spec(_placer_spec)
_placer_spec.loader.exec_module(_placer)


# ── Snapshot collection ────────────────────────────────────────────────────

def collect_snapshots_one_benchmark(benchmark_name: str, num_examples: int,
                                      seed: int, drift_prob: float = 0.3,
                                      p_cascade: float = 0.5,
                                      cascade_min: int = 2,
                                      cascade_max: int = 4
                                      ) -> list[_snap.StateSnapshot]:
    """State-drift sampler that emits ``StateSnapshot`` objects.

    Same drift + cascade-interior sampling as ``train._collect_one_benchmark``
    but instead of returning (features, targets) arrays, returns
    snapshots that include the raw state at each sampling point.

    The hand_feats are computed and cached alongside since they don't
    depend on the encoder.
    """
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark_name
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    hpwl_edges = _placer._hard_macro_edges(benchmark)
    state = _placer.PlacementState(benchmark, hpwl_edges)
    rng = np.random.RandomState(seed)

    movable = np.where(state.movable)[0]
    n_hard = state.n
    full_placement = benchmark.macro_positions.clone()

    def exact_cost() -> float:
        full_placement[:n_hard] = torch.tensor(state.pos, dtype=torch.float32)
        return float(compute_proxy_cost(full_placement, benchmark, plc)["proxy_cost"])

    t0 = time.time()
    base_cost = exact_cost()
    print(f"  base proxy_cost = {base_cost:.4f}  (first call: {time.time()-t0:.2f}s)")

    import random as _random
    py_rng = _random.Random(int(rng.randint(0, 2**31 - 1)))

    movable_list = movable.tolist()
    snapshots: list[_snap.StateSnapshot] = []

    def _do_preliminary_cascade(num_steps: int) -> list[tuple[int, float, float]]:
        saved: list[tuple[int, float, float]] = []
        for _ in range(num_steps):
            i = int(py_rng.choice(movable_list))
            cands_cas = state.candidate_positions(i, num_candidates=4, rng=py_rng)
            if not cands_cas:
                continue
            best_cx = best_cy = None
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

    while len(snapshots) < num_examples:
        cascade_saved: list[tuple[int, float, float]] = []
        local_base = base_cost
        if py_rng.random() < p_cascade:
            n_steps = py_rng.randint(cascade_min, cascade_max)
            cascade_saved = _do_preliminary_cascade(n_steps)
            if cascade_saved:
                local_base = exact_cost()

        macro_idx = int(rng.choice(movable))
        cands = state.candidate_positions(macro_idx, num_candidates=8, rng=py_rng)
        cand_costs: list[tuple[tuple[float, float], float]] = []
        for cx, cy in cands:
            old_x = float(state.pos[macro_idx, 0])
            old_y = float(state.pos[macro_idx, 1])
            delta = np.array([cx - old_x, cy - old_y], dtype=np.float64)

            hand_feats = _placer._features_for_move(state, macro_idx, delta)

            state.apply_move(macro_idx, cx, cy)
            new_cost = exact_cost()
            state.apply_move(macro_idx, old_x, old_y)

            snapshots.append(_snap.collect_snapshot(
                benchmark_name=benchmark_name,
                state=state,
                macro_idx=macro_idx,
                move_delta=delta,
                target_delta_proxy=new_cost - local_base,
                hand_feats=hand_feats,
            ))
            cand_costs.append(((cx, cy), new_cost))

            if len(snapshots) >= num_examples:
                break

        for (i, ox, oy) in reversed(cascade_saved):
            state.apply_move(i, ox, oy)

        if cand_costs and py_rng.random() < drift_prob:
            (cx, cy), new_cost = min(cand_costs, key=lambda t: t[1])
            state.apply_move(macro_idx, cx, cy)
            base_cost = exact_cost() if cascade_saved else new_cost

        if time.time() - last_log > 5.0:
            elapsed = time.time() - t_collect
            n_done = len(snapshots)
            rate = n_done / max(elapsed, 1e-6)
            print(f"  collected {n_done:4d}/{num_examples}  "
                  f"({rate:.2f} ex/s, base={base_cost:.3f})")
            last_log = time.time()

    elapsed = time.time() - t_collect
    print(f"  done in {elapsed:.1f}s ({len(snapshots) / max(elapsed, 1e-6):.2f} ex/s)")
    return snapshots


# ── Joint training ─────────────────────────────────────────────────────────

def _spearman(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) < 2:
        return 0.0
    return float(np.corrcoef(
        np.argsort(np.argsort(pred)),
        np.argsort(np.argsort(true)),
    )[0, 1])


def train_encoder_and_approximator_joint(
    snapshots: list[_snap.StateSnapshot],
    *,
    encoder_kind: str = "gnn",
    encoder_hidden: int = 128,
    num_gnn_layers: int = 3,
    grid_size: int = 128,
    hand_feature_dim: int = 16,
    approx_hidden: int = 64,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 42,
    patience: int | None = None,
):
    """Joint regression training of encoder + cost approximator.

    Returns ``(encoder_module, approximator_module, info_dict)``.
    The returned encoder is in ``eval()`` mode and the approximator weights
    are the best-by-Spearman checkpoint (matches train.py's selection rule).
    """
    if encoder_kind == "gnn":
        from encoder_runtime import GNNEncoderRuntime as EncoderCls
        encoder = EncoderCls(hidden_dim=encoder_hidden,
                              num_gnn_layers=num_gnn_layers)
        encode_inputs_fn = _encode_inputs_gnn
    elif encoder_kind == "gnncnn":
        from encoder_runtime_gnncnn import GNNCNNEncoderRuntime as EncoderCls
        encoder = EncoderCls(hidden_dim=encoder_hidden,
                              num_gnn_layers=num_gnn_layers,
                              grid_size=grid_size)
        encode_inputs_fn = _encode_inputs_gnncnn
    else:
        raise ValueError(f"encoder_kind must be 'gnn' or 'gnncnn'; got {encoder_kind!r}")

    embed_dim = encoder.embed_dim
    approx_in_dim = embed_dim + hand_feature_dim
    approximator = CostApproximator(in_dim=approx_in_dim, hidden=approx_hidden)

    print(f"  encoder kind={encoder_kind}  embed_dim={embed_dim}")
    print(f"  approximator in_dim={approx_in_dim}  hidden={approx_hidden}")
    print(f"  total params: encoder={sum(p.numel() for p in encoder.parameters()):,}  "
          f"approx={sum(p.numel() for p in approximator.parameters()):,}")

    # Build per-benchmark caches once.
    bench_caches: dict[str, _snap.BenchmarkCache] = {}
    for snap in snapshots:
        if snap.benchmark_name not in bench_caches:
            from macro_place.loader import load_benchmark_from_dir
            bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / snap.benchmark_name
            benchmark, _plc = load_benchmark_from_dir(str(bench_dir))
            del _plc  # discard expensive PLC; we just need the lightweight benchmark
            bench_caches[snap.benchmark_name] = _snap.build_benchmark_cache(benchmark)
    print(f"  built caches for {len(bench_caches)} benchmark(s)")

    # Train/val split (snapshot-level).
    rng = np.random.RandomState(seed)
    n = len(snapshots)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx = set(int(i) for i in perm[:n_val].tolist())
    train_snaps = [s for i, s in enumerate(snapshots) if i not in val_idx]
    val_snaps = [s for i, s in enumerate(snapshots) if i in val_idx]

    # Normalize targets.
    targets = np.array([s.target_delta_proxy for s in train_snaps], dtype=np.float32)
    target_mean = float(targets.mean())
    target_std = float(max(targets.std(), 1e-6))

    # Hand-feature normalization (per-dim mean/std from train split).
    hand = np.stack([s.hand_feats for s in train_snaps])
    feat_mean = hand.mean(axis=0).astype(np.float32)
    feat_std = np.maximum(hand.std(axis=0), 1e-6).astype(np.float32)

    if patience is None:
        patience = max(10, epochs // 5)

    params = list(encoder.parameters()) + list(approximator.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.SmoothL1Loss()

    best_state: dict | None = None
    best_spearman = -float("inf")
    best_epoch = 0
    epochs_since_improvement = 0

    for epoch in range(epochs):
        encoder.train()
        approximator.train()
        perm_train = rng.permutation(len(train_snaps))
        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(train_snaps), batch_size):
            batch_snaps = [train_snaps[int(i)]
                            for i in perm_train[batch_start:batch_start + batch_size]]
            # Forward per snapshot, accumulate losses, one optimizer step.
            optimizer.zero_grad()
            losses: list[torch.Tensor] = []
            for snap in batch_snaps:
                cache = bench_caches[snap.benchmark_name]
                state = _snap.replay_state(snap, cache)
                per_node, graph_vec = encode_inputs_fn(state, encoder)
                macro_emb = per_node[snap.macro_idx]
                # Build cand feature vector: [macro_emb; graph_vec; norm_hand_feats]
                norm_hand = ((snap.hand_feats - feat_mean) / feat_std).astype(np.float32)
                hand_t = torch.tensor(norm_hand, dtype=torch.float32)
                feat = torch.cat([macro_emb, graph_vec, hand_t], dim=-1)
                pred = approximator(feat.unsqueeze(0)).squeeze(-1)
                target_norm = torch.tensor(
                    (snap.target_delta_proxy - target_mean) / target_std,
                    dtype=torch.float32,
                )
                losses.append(loss_fn(pred, target_norm))

            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # Validation — Spearman + Pearson on the val split.
        encoder.eval()
        approximator.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for snap in val_snaps:
                cache = bench_caches[snap.benchmark_name]
                state = _snap.replay_state(snap, cache)
                per_node, graph_vec = encode_inputs_fn(state, encoder)
                macro_emb = per_node[snap.macro_idx]
                norm_hand = ((snap.hand_feats - feat_mean) / feat_std).astype(np.float32)
                hand_t = torch.tensor(norm_hand, dtype=torch.float32)
                feat = torch.cat([macro_emb, graph_vec, hand_t], dim=-1)
                pred = approximator(feat.unsqueeze(0)).squeeze(-1)
                val_preds.append(float(pred.item()) * target_std + target_mean)
                val_targets.append(snap.target_delta_proxy)

        val_preds_np = np.array(val_preds)
        val_targets_np = np.array(val_targets)
        if len(val_preds_np) >= 2:
            pearson = float(np.corrcoef(val_preds_np, val_targets_np)[0, 1])
            spearman = _spearman(val_preds_np, val_targets_np)
        else:
            pearson = spearman = 0.0

        if spearman > best_spearman:
            best_spearman = spearman
            best_epoch = epoch
            best_state = {
                "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                "approximator": {k: v.detach().clone()
                                  for k, v in approximator.state_dict().items()},
            }
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  train={train_loss:.4f}  "
                  f"pearson={pearson:.3f}  spearman={spearman:.3f}")

        if epochs_since_improvement >= patience:
            print(f"  early-stop at epoch {epoch} (best Spearman={best_spearman:.3f} @ {best_epoch})")
            break

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        approximator.load_state_dict(best_state["approximator"])
        print(f"  loaded best-by-Spearman checkpoint from epoch {best_epoch}")

    encoder.eval()
    approximator.eval()

    info = {
        "feat_mean": torch.tensor(feat_mean),
        "feat_std": torch.tensor(feat_std),
        "target_mean": target_mean,
        "target_std": target_std,
        "pearson": pearson,
        "spearman": best_spearman,
        "embed_dim": embed_dim,
        "approx_in_dim": approx_in_dim,
        "encoder_kind": encoder_kind,
        "hand_feature_dim": hand_feature_dim,
        "n_train": len(train_snaps),
        "n_val": len(val_snaps),
    }
    return encoder, approximator, info


def _encode_inputs_gnn(state, encoder):
    from encoder_runtime import build_encoder_inputs
    node_feats, edge_index, edge_attr = build_encoder_inputs(state)
    return encoder(node_feats, edge_index, edge_attr)


def _encode_inputs_gnncnn(state, encoder):
    from encoder_runtime_gnncnn import build_encoder_inputs
    node_feats, edge_index, edge_attr, canvas = build_encoder_inputs(
        state, grid_size=encoder.grid_size
    )
    return encoder(node_feats, edge_index, edge_attr, canvas)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_benchmarks(arg: str) -> list[str]:
    arg = (arg or "").strip()
    if arg == "all":
        from macro_place.evaluate import IBM_BENCHMARKS
        return list(IBM_BENCHMARKS)
    names = [n.strip() for n in arg.split(",") if n.strip()]
    if not names:
        raise ValueError("--benchmark must contain at least one name")
    return names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="ibm01")
    p.add_argument("--num-examples", type=int, default=200,
                   help="Per-benchmark example count for the regression dataset.")
    p.add_argument("--encoder-kind", choices=["gnn", "gnncnn"], default="gnn")
    p.add_argument("--encoder-hidden", type=int, default=128)
    p.add_argument("--num-gnn-layers", type=int, default=3)
    p.add_argument("--grid-size", type=int, default=128,
                   help="(gnncnn only) canvas raster resolution.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder-out",
                   default=str(_HERE / "checkpoints" / "encoder.pt"))
    p.add_argument("--approx-out",
                   default=str(_HERE / "checkpoints" / "cost_approximator.pt"))
    p.add_argument("--snapshots-cache",
                   default=str(_HERE / "data" / "encoder_snapshots.pt"),
                   help="Where to cache the collected snapshots so subsequent "
                        "runs skip re-collection.")
    args = p.parse_args()

    benchmarks = parse_benchmarks(args.benchmark)

    snapshots_path = Path(args.snapshots_cache)
    snapshots: list[_snap.StateSnapshot] = []
    if snapshots_path.exists():
        print(f"[1/2] Loading cached snapshots from {snapshots_path}")
        snapshots = _snap.load_snapshots(snapshots_path)
        cached_benches = sorted({s.benchmark_name for s in snapshots})
        if set(cached_benches) != set(benchmarks):
            print(f"  cache covers {cached_benches} but request is "
                  f"{sorted(benchmarks)} — re-collecting")
            snapshots = []

    if not snapshots:
        print(f"[1/2] Collecting snapshots for {benchmarks}")
        for b_idx, name in enumerate(benchmarks):
            print(f"  [{b_idx+1}/{len(benchmarks)}] {name}")
            snapshots.extend(collect_snapshots_one_benchmark(
                name, args.num_examples,
                seed=args.seed + b_idx * 7919,
            ))
        _snap.save_snapshots(snapshots, snapshots_path)
        print(f"  saved {len(snapshots)} snapshots -> {snapshots_path}")

    print(f"\n[2/2] Joint training ({len(snapshots)} snapshots)")
    encoder, approximator, info = train_encoder_and_approximator_joint(
        snapshots,
        encoder_kind=args.encoder_kind,
        encoder_hidden=args.encoder_hidden,
        num_gnn_layers=args.num_gnn_layers,
        grid_size=args.grid_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    print(f"\n  Final: Pearson={info['pearson']:.3f}  Spearman={info['spearman']:.3f}")

    # Save encoder.
    if args.encoder_kind == "gnn":
        from encoder_runtime import save_encoder
    else:
        from encoder_runtime_gnncnn import save_encoder
    save_encoder(encoder, Path(args.encoder_out), trained_on=benchmarks)
    print(f"  encoder -> {args.encoder_out}")

    # Save approximator. We mirror train.py's checkpoint schema so the
    # placer's existing _load_cost_approximator works, and we add the
    # encoder kind/path so the placer knows what to pair it with.
    ckpt = {
        "state_dict": approximator.state_dict(),
        "feat_mean": info["feat_mean"],
        "feat_std": info["feat_std"],
        "target_mean": info["target_mean"],
        "target_std": info["target_std"],
        "feature_dim": info["approx_in_dim"],
        "hidden": 64,
        "pearson_r_val": info["pearson"],
        "spearman_r_val": info["spearman"],
        "trained_on": benchmarks,
        "n_train": info["n_train"],
        "n_val": info["n_val"],
        # Encoder linkage:
        "encoder_kind": info["encoder_kind"],
        "encoder_path": str(args.encoder_out),
        "encoder_embed_dim": info["embed_dim"],
        "hand_feature_dim": info["hand_feature_dim"],
    }
    Path(args.approx_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.approx_out)
    print(f"  approximator -> {args.approx_out}")


if __name__ == "__main__":
    main()
