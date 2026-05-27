"""Phase 2.5 — supervised encoder pre-training for proxy cost prediction.

Trains ProxyCostPredictor (GNN+CNN encoder + 3-output head) to predict
(wl_cost, density_cost, congestion_cost) for randomized macro placements.
The trained encoder can then score candidate moves in LKChain.run_greedy via
delta(encode(state_after) - encode(state_before)).

Workflow:
    1. Per benchmark: randomize movable macro positions, call compute_proxy_cost,
       build encoder inputs (node_feats, canvas, edge_index, edge_attr, metadata).
    2. Cache per-benchmark .pt files in data/encoder/per_bench/.
    3. Train with per-benchmark per-term z-score normalization, MSE loss, Adam +
       cosine LR, early stopping on val combined Pearson r.
    4. Checkpoint -> checkpoints/proxy_cost_encoder.pt.

Usage:
    uv run python submissions/lkh/train_encoder.py --benchmarks ibm01,ibm02 --samples 50 --epochs 5
    uv run python submissions/lkh/train_encoder.py --benchmarks all --samples 200 --epochs 100
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))  # repo root for macro_place

# Load placer.py via importlib to avoid package __init__ issues
_spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
_placer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_placer)

# Load encoder module
_enc_spec = importlib.util.spec_from_file_location(
    "lkh_encoder", str(_HERE / "model" / "encoder.py")
)
_encoder_mod = importlib.util.module_from_spec(_enc_spec)
_enc_spec.loader.exec_module(_encoder_mod)

from lkh_model import ProxyCostPredictor
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

build_node_features = _encoder_mod.build_node_features
build_edge_index_and_attr = _encoder_mod.build_edge_index_and_attr
build_metadata_features = _encoder_mod.build_metadata_features
rasterize_canvas = _encoder_mod.rasterize_canvas
_build_anchors = _encoder_mod._build_anchors

# Proxy weights: hardcoded in macro_place/objective.py
PROXY_WEIGHTS = {"wl": 1.0, "den": 0.5, "cong": 0.5}


# ── Benchmark list parsing ──────────────────────────────────────────────────

def parse_benchmarks(arg: str) -> list[str]:
    arg = (arg or "").strip()
    if arg == "all":
        from macro_place.evaluate import IBM_BENCHMARKS
        return list(IBM_BENCHMARKS)
    names = [n.strip() for n in arg.split(",") if n.strip()]
    if not names:
        raise ValueError("--benchmarks must contain at least one benchmark name")
    return names


# ── Timing helpers ──────────────────────────────────────────────────────────

def _fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{int(s//60)}m{int(s%60):02d}s"
    return f"{int(s//3600)}h{int((s%3600)//60):02d}m"


# ── Data collection ─────────────────────────────────────────────────────────

def collect_proxy_samples(
    benchmark_name: str,
    n_samples: int,
    seed: int = 42,
    grid_size: int = 128,
) -> dict:
    """Collect n_samples random-placement proxy cost observations for one benchmark.

    For each sample: randomize all movable macro positions uniformly on canvas,
    build encoder inputs, call compute_proxy_cost. Canvas is stored as float16.

    Returns a dict with keys:
        node_feats   [n_samples, N_total, 8]  float32
        canvas       [n_samples, 3, G, G]     float16
        edge_index   [2, E]                    int64   (fixed per benchmark)
        edge_attr    [E, 1]                    float32 (fixed per benchmark)
        metadata     [12]                      float32 (fixed per benchmark)
        n_hard       int
        wl_costs     [n_samples]               float32
        den_costs    [n_samples]               float32
        cong_costs   [n_samples]               float32
        proxy_weights dict  {wl: α, den: α, cong: α}
        name         str
    """
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark_name
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    hpwl_edges = _placer._hard_macro_edges(benchmark)
    state = _placer.PlacementState(benchmark, hpwl_edges)
    rng = np.random.RandomState(seed)
    n_hard = state.n
    movable = np.where(state.movable)[0]

    # Build fixed topology features (reused across all samples)
    anchors, anchor_inverse = _build_anchors(state)
    edge_index, edge_attr = build_edge_index_and_attr(state, anchor_inverse)
    metadata = build_metadata_features(state, anchor_inverse)

    full_placement = benchmark.macro_positions.clone()

    nf_list: list[torch.Tensor] = []
    cv_list: list[torch.Tensor] = []
    wl_list: list[float] = []
    den_list: list[float] = []
    cong_list: list[float] = []

    t0 = time.time()
    last_log = t0
    for i in range(n_samples):
        # Randomize all movable macro positions uniformly on canvas
        state.pos[movable, 0] = rng.uniform(0, state.cw, size=len(movable))
        state.pos[movable, 1] = rng.uniform(0, state.ch, size=len(movable))
        state.rebuild_caches()

        nf = build_node_features(state, anchors)  # [N_total, 8]
        cv = rasterize_canvas(state, grid_size=grid_size)  # [3, G, G]

        full_placement[:n_hard] = torch.tensor(state.pos, dtype=torch.float32)
        result = compute_proxy_cost(full_placement, benchmark, plc)

        nf_list.append(nf)
        cv_list.append(cv.to(torch.float16))
        wl_list.append(float(result["wirelength_cost"]))
        den_list.append(float(result["density_cost"]))
        cong_list.append(float(result["congestion_cost"]))

        if time.time() - last_log > 5.0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / max(rate, 1e-9)
            print(f"  [{benchmark_name}] {i+1}/{n_samples} "
                  f"({_fmt_time(elapsed)} elapsed, eta {_fmt_time(eta)})")
            last_log = time.time()

    elapsed = time.time() - t0
    print(f"  [{benchmark_name}] done: {n_samples} samples in {_fmt_time(elapsed)}")

    return {
        "node_feats": torch.stack(nf_list),                        # [N, N_total, 8]
        "canvas": torch.stack(cv_list),                            # [N, 3, G, G]
        "edge_index": edge_index,                                  # [2, E]
        "edge_attr": edge_attr,                                    # [E, 1]
        "metadata": metadata,                                      # [12]
        "n_hard": n_hard,
        "wl_costs": torch.tensor(wl_list, dtype=torch.float32),   # [N]
        "den_costs": torch.tensor(den_list, dtype=torch.float32),  # [N]
        "cong_costs": torch.tensor(cong_list, dtype=torch.float32), # [N]
        "proxy_weights": PROXY_WEIGHTS,
        "name": benchmark_name,
    }


def _save_bench_cache(cache_dir: Path, data: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f"{data['name']}.pt.tmp"
    torch.save(data, str(tmp))
    tmp.rename(cache_dir / f"{data['name']}.pt")


def _load_bench_cache(cache_dir: Path, name: str) -> dict | None:
    p = cache_dir / f"{name}.pt"
    if not p.exists():
        return None
    try:
        return torch.load(str(p), weights_only=False)
    except Exception:
        return None


def load_all_caches(
    benchmark_names: list[str],
    cache_dir: Path,
    n_samples: int,
    seed: int,
    grid_size: int,
    force_recollect: bool = False,
) -> dict[str, dict]:
    """Load or collect per-benchmark caches. Returns {name: data_dict}."""
    result: dict[str, dict] = {}
    for b_idx, name in enumerate(benchmark_names):
        print(f"\n[{b_idx+1}/{len(benchmark_names)}] {name}")
        cached = None if force_recollect else _load_bench_cache(cache_dir, name)
        if cached is not None:
            n_in_cache = cached["node_feats"].shape[0]
            if n_in_cache >= n_samples:
                print(f"  loaded {n_in_cache} samples from cache")
                result[name] = cached
                continue
            print(f"  cache has {n_in_cache} samples, need {n_samples} — re-collecting")
        data = collect_proxy_samples(name, n_samples, seed=seed + b_idx * 7919,
                                     grid_size=grid_size)
        _save_bench_cache(cache_dir, data)
        result[name] = data
    return result


# ── Batched forward ─────────────────────────────────────────────────────────

def _bench_batch_forward(
    model: ProxyCostPredictor,
    bench_data: dict,
    indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Batched ProxyCostPredictor forward for samples from the same benchmark.

    Handles the shared-topology case: edge_index and metadata are the same for
    all samples; only node_feats and canvas vary. Implements PyG-style batching
    by offsetting node indices for each sample in the batch.

    Returns: [B, 3] predictions (normalized).
    """
    B = len(indices)
    nf = bench_data["node_feats"][indices].to(device)       # [B, N, 8]
    cv = bench_data["canvas"][indices].to(torch.float32).to(device)  # [B, 3, G, G]
    ei = bench_data["edge_index"].to(device)                # [2, E]
    ea = bench_data["edge_attr"].to(device)                 # [E, 1]
    md = bench_data["metadata"].to(device)                  # [12]
    n_hard = bench_data["n_hard"]
    N = nf.shape[1]
    E = ei.shape[1]
    H = model.hidden_dim
    enc = model.encoder

    # Flatten node features: [B*N, 8]
    nf_flat = nf.reshape(B * N, 8)

    if E > 0:
        # Offset edge_index for each sample: sample b uses nodes [b*N, (b+1)*N)
        offsets = torch.arange(B, device=device) * N  # [B]
        ei_batch = torch.cat([ei + offsets[b] for b in range(B)], dim=1)  # [2, B*E]
        ea_batch = ea.repeat(B, 1)                                          # [B*E, 1]
    else:
        ei_batch = torch.zeros(2, 0, dtype=torch.long, device=device)
        ea_batch = torch.zeros(0, 1, dtype=torch.float32, device=device)

    # GNN forward on flattened batch
    h_nodes_flat, h_edges_flat = enc.gnn(nf_flat, ei_batch, ea_batch)
    # h_nodes_flat: [B*N, H], h_edges_flat: [B*E, H]

    h_meta = enc.metadata_encoder(md)  # [H]
    h_meta_b = h_meta.unsqueeze(0).expand(B, -1)  # [B, H]

    # Per-sample CNN
    h_cnn = enc.cnn(cv)  # [B, H]

    # Per-sample edge stats
    if E > 0:
        h_edges = h_edges_flat.reshape(B, E, H)   # [B, E, H]
        h_edges_mean = h_edges.mean(dim=1)         # [B, H]
        h_edges_var  = h_edges.var(dim=1)          # [B, H]
        h_edges_max  = h_edges.max(dim=1).values   # [B, H]
        h_edges_min  = h_edges.min(dim=1).values   # [B, H]
    else:
        zeros = torch.zeros(B, H, device=device)
        h_edges_mean = h_edges_var = h_edges_max = h_edges_min = zeros

    gnn_global = torch.cat(
        [h_meta_b, h_edges_mean, h_edges_var, h_edges_max, h_edges_min], dim=-1
    )  # [B, 5H]

    head_input = (
        torch.cat([gnn_global, h_cnn], dim=-1) if model.use_cnn else gnn_global
    )  # [B, 6H or 5H]

    return model.proxy_head(head_input)  # [B, 3]


# ── Training ────────────────────────────────────────────────────────────────

def _pearson(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    r = np.corrcoef(pred, target)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def train_proxy_encoder(
    bench_data: dict[str, dict],
    hidden_dim: int = 128,
    num_gnn_layers: int = 3,
    edge_fc_layers: int = 1,
    grid_size: int = 128,
    use_cnn: bool = True,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 42,
    device: torch.device | None = None,
    patience: int = 15,
) -> tuple[ProxyCostPredictor, dict]:
    """Train ProxyCostPredictor on the given per-benchmark sample dicts.

    Normalization is per-benchmark per-term z-score on training split.
    Early stopping based on combined val Pearson r (mean of wl, den, cong).

    Returns (model, info_dict) where info_dict has val pearson_r per term,
    norm_stats dict, and the proxy_weights stored in the checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device = {device}")

    rng = np.random.RandomState(seed)
    benchmark_names = list(bench_data.keys())

    # ── Stratified 80/20 split per benchmark ───────────────────────────────
    train_idx_by_bench: dict[str, list[int]] = {}
    val_idx_by_bench: dict[str, list[int]] = {}
    norm_stats: dict[str, dict] = {}

    for name in benchmark_names:
        d = bench_data[name]
        n = d["node_feats"].shape[0]
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_frac))
        val_idx = sorted(perm[:n_val].tolist())
        train_idx = sorted(perm[n_val:].tolist())
        train_idx_by_bench[name] = train_idx
        val_idx_by_bench[name] = val_idx

        # Per-term normalization stats on training split
        wl_tr = d["wl_costs"][train_idx]
        den_tr = d["den_costs"][train_idx]
        cong_tr = d["cong_costs"][train_idx]
        norm_stats[name] = {
            "wl_mean": float(wl_tr.mean()),
            "wl_std": float(wl_tr.std().clamp(min=1e-6)),
            "den_mean": float(den_tr.mean()),
            "den_std": float(den_tr.std().clamp(min=1e-6)),
            "cong_mean": float(cong_tr.mean()),
            "cong_std": float(cong_tr.std().clamp(min=1e-6)),
        }

    def normalize_targets(name: str, wl, den, cong):
        ns = norm_stats[name]
        wl_n  = (wl  - ns["wl_mean"])   / ns["wl_std"]
        den_n = (den - ns["den_mean"])  / ns["den_std"]
        cng_n = (cong - ns["cong_mean"]) / ns["cong_std"]
        return torch.stack([wl_n, den_n, cng_n], dim=-1)  # [N, 3]

    # ── Build flattened train/val sample lists ─────────────────────────────
    # Each entry: (benchmark_name, local_sample_idx)
    train_samples: list[tuple[str, int]] = []
    val_samples: list[tuple[str, int]] = []
    for name in benchmark_names:
        for i in train_idx_by_bench[name]:
            train_samples.append((name, i))
        for i in val_idx_by_bench[name]:
            val_samples.append((name, i))

    n_train = len(train_samples)
    n_val = len(val_samples)
    print(f"  train={n_train}  val={n_val}  ({len(benchmark_names)} benchmarks)")

    # ── Model and optimizer ─────────────────────────────────────────────────
    model = ProxyCostPredictor(
        hidden_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
        edge_fc_layers=edge_fc_layers,
        grid_size=grid_size,
        use_cnn=use_cnn,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val_r = -float("inf")
    best_state: dict | None = None
    no_improve = 0

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        perm = rng.permutation(n_train).tolist()

        # Group by benchmark for batching
        bench_batches: dict[str, list[int]] = {n: [] for n in benchmark_names}
        for pos in perm:
            name, idx = train_samples[pos]
            bench_batches[name].append(idx)

        train_loss = 0.0
        n_processed = 0
        optimizer.zero_grad()

        for name in benchmark_names:
            indices = bench_batches[name]
            if not indices:
                continue
            d = bench_data[name]
            targets_all = normalize_targets(
                name,
                d["wl_costs"][indices],
                d["den_costs"][indices],
                d["cong_costs"][indices],
            ).to(device)  # [N_b, 3]

            # Mini-batches within this benchmark
            b_perm = list(range(len(indices)))
            for b_start in range(0, len(indices), batch_size):
                b_end = min(b_start + batch_size, len(indices))
                b_local = b_perm[b_start:b_end]
                b_global = [indices[j] for j in b_local]

                pred = _bench_batch_forward(model, d, b_global, device)  # [B, 3]
                targets_b = targets_all[b_local]

                loss = loss_fn(pred, targets_b) * len(b_local)
                loss.backward()
                train_loss += loss.item()
                n_processed += len(b_local)

                # Step every batch_size samples
                optimizer.step()
                optimizer.zero_grad()

        train_loss /= max(n_processed, 1)
        scheduler.step()

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_pred_all: list[np.ndarray] = []
        val_true_all: list[np.ndarray] = []
        val_loss = 0.0
        n_val_proc = 0

        bench_val: dict[str, list[int]] = {n: [] for n in benchmark_names}
        for name, idx in val_samples:
            bench_val[name].append(idx)

        with torch.no_grad():
            for name in benchmark_names:
                indices = bench_val[name]
                if not indices:
                    continue
                d = bench_data[name]
                targets_all = normalize_targets(
                    name,
                    d["wl_costs"][indices],
                    d["den_costs"][indices],
                    d["cong_costs"][indices],
                ).to(device)

                for b_start in range(0, len(indices), batch_size):
                    b_end = min(b_start + batch_size, len(indices))
                    b_global = indices[b_start:b_end]
                    pred = _bench_batch_forward(model, d, b_global, device)
                    targets_b = targets_all[b_start:b_end]
                    val_loss += loss_fn(pred, targets_b).item() * len(b_global)
                    n_val_proc += len(b_global)
                    val_pred_all.append(pred.cpu().numpy())
                    val_true_all.append(targets_b.cpu().numpy())

        val_loss /= max(n_val_proc, 1)
        val_pred_np = np.concatenate(val_pred_all)   # [N_val, 3]
        val_true_np = np.concatenate(val_true_all)   # [N_val, 3]
        r_wl   = _pearson(val_pred_np[:, 0], val_true_np[:, 0])
        r_den  = _pearson(val_pred_np[:, 1], val_true_np[:, 1])
        r_cong = _pearson(val_pred_np[:, 2], val_true_np[:, 2])
        r_mean = np.mean([r_wl, r_den, r_cong])

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"r_wl={r_wl:.3f}  r_den={r_den:.3f}  r_cong={r_cong:.3f}  "
                  f"r_mean={r_mean:.3f}")

        if r_mean > best_val_r:
            best_val_r = r_mean
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stop at epoch {epoch} (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final val metrics with best weights
    model.eval()
    val_pred_all, val_true_all = [], []
    bench_val = {n: [] for n in benchmark_names}
    for name, idx in val_samples:
        bench_val[name].append(idx)

    with torch.no_grad():
        for name in benchmark_names:
            indices = bench_val[name]
            if not indices:
                continue
            d = bench_data[name]
            targets_all = normalize_targets(
                name,
                d["wl_costs"][indices],
                d["den_costs"][indices],
                d["cong_costs"][indices],
            ).to(device)
            for b_start in range(0, len(indices), batch_size):
                b_end = min(b_start + batch_size, len(indices))
                b_global = indices[b_start:b_end]
                pred = _bench_batch_forward(model, d, b_global, device)
                val_pred_all.append(pred.cpu().numpy())
                val_true_all.append(targets_all[b_start:b_end].cpu().numpy())

    val_pred_np = np.concatenate(val_pred_all)
    val_true_np = np.concatenate(val_true_all)
    final_r_wl   = _pearson(val_pred_np[:, 0], val_true_np[:, 0])
    final_r_den  = _pearson(val_pred_np[:, 1], val_true_np[:, 1])
    final_r_cong = _pearson(val_pred_np[:, 2], val_true_np[:, 2])

    return model, {
        "norm_stats": norm_stats,
        "pearson_r_wl": final_r_wl,
        "pearson_r_den": final_r_den,
        "pearson_r_cong": final_r_cong,
        "n_train": n_train,
        "n_val": n_val,
        "best_val_r_mean": best_val_r,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmarks", default="ibm01",
                   help="Single name, comma-separated, or 'all'")
    p.add_argument("--samples", type=int, default=200,
                   help="Random placement samples per benchmark")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-gnn-layers", type=int, default=3)
    p.add_argument("--edge-fc-layers", type=int, default=1)
    p.add_argument("--grid-size", type=int, default=128)
    p.add_argument("--no-cnn", action="store_true",
                   help="Ablate CNN: exclude canvas features from head input")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=15,
                   help="Early stopping patience (epochs without val r improvement)")
    p.add_argument("--cache-dir",
                   default=str(_HERE / "data" / "encoder" / "per_bench"))
    p.add_argument("--checkpoint-out",
                   default=str(_HERE / "checkpoints" / "proxy_cost_encoder.pt"))
    p.add_argument("--force-recollect", action="store_true")
    args = p.parse_args()

    benchmarks = parse_benchmarks(args.benchmarks)
    use_cnn = not args.no_cnn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=== Phase 2.5: ProxyCostPredictor encoder pre-training ===")
    print(f"  benchmarks = {benchmarks}")
    print(f"  samples/benchmark = {args.samples}  "
          f"total ≈ {args.samples * len(benchmarks)}")
    print(f"  hidden_dim={args.hidden_dim}  gnn_layers={args.num_gnn_layers}  "
          f"use_cnn={use_cnn}")

    print("\n[1/3] Data collection / cache loading...")
    bench_data = load_all_caches(
        benchmarks,
        cache_dir=Path(args.cache_dir),
        n_samples=args.samples,
        seed=args.seed,
        grid_size=args.grid_size,
        force_recollect=args.force_recollect,
    )

    print(f"\n[2/3] Training ({args.epochs} epochs, batch={args.batch_size}, lr={args.lr})...")
    model, info = train_proxy_encoder(
        bench_data,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        edge_fc_layers=args.edge_fc_layers,
        grid_size=args.grid_size,
        use_cnn=use_cnn,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        patience=args.patience,
    )

    print(f"\n[3/3] Final validation Pearson r:")
    print(f"    wl:   {info['pearson_r_wl']:.4f}")
    print(f"    den:  {info['pearson_r_den']:.4f}")
    print(f"    cong: {info['pearson_r_cong']:.4f}")
    print(f"    mean: {info['best_val_r_mean']:.4f}")

    # Save checkpoint
    ckpt = {
        "type": "proxy_cost_encoder",
        "encoder_state": model.encoder.state_dict(),
        "head_state": model.proxy_head.state_dict(),
        "hidden_dim": args.hidden_dim,
        "num_gnn_layers": args.num_gnn_layers,
        "edge_fc_layers": args.edge_fc_layers,
        "grid_size": args.grid_size,
        "use_cnn": use_cnn,
        "norm_stats": info["norm_stats"],
        "proxy_weights": PROXY_WEIGHTS,
        "pearson_r_wl": info["pearson_r_wl"],
        "pearson_r_den": info["pearson_r_den"],
        "pearson_r_cong": info["pearson_r_cong"],
        "best_val_r_mean": info["best_val_r_mean"],
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
