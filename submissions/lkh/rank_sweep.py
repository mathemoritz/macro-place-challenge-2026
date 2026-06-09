"""Rank-loss sweep.

The CostApproximator has high Pearson but low Spearman (≈0.35), and since the
chain picks moves by argmax over per-candidate scores, *ranking* — not
amplitude — is what actually matters. This sweeps the weight on the
within-state pairwise margin rank loss and measures whether leaning on it
improves the validation rank correlation (Spearman ρ).

Reuses an existing cached dataset (``data/chain_data.pt``) so no slow
compute_proxy_cost re-collection is needed; falls back to collecting ibm01 if
no cache is present.

Usage:
    uv run python submissions/lkh/rank_sweep.py
    uv run python submissions/lkh/rank_sweep.py --weights 0,0.5,1,2,4,8 --epochs 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

import train as _train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(_HERE / "data" / "chain_data.pt"),
                   help="Cached (features, targets) dataset. Collected if absent.")
    p.add_argument("--collect-benchmark", default="ibm01",
                   help="Benchmark to collect if --data is missing.")
    p.add_argument("--num-examples", type=int, default=500,
                   help="Examples to collect if --data is missing.")
    p.add_argument("--weights", default="0,0.5,1,2,4",
                   help="Comma-separated rank-loss weights to sweep.")
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", default=str(_HERE.parent.parent / "vis" / "rank_sweep.png"))
    p.add_argument("--json", default=str(_HERE / "iter_output" / "rank_sweep.json"))
    args = p.parse_args()

    data_path = Path(args.data)
    if data_path.exists():
        d = torch.load(data_path, weights_only=False)
        features, targets = d["features"], d["targets"]
        n_cal = int(d.get("n_calibration_samples", 0))
        source = f"{data_path.name} ({d.get('benchmarks')})"
    else:
        print(f"[rank_sweep] no cache at {data_path}; collecting "
              f"{args.num_examples} from {args.collect_benchmark}")
        features, targets = _train.collect_data(
            [args.collect_benchmark], args.num_examples, args.seed)
        n_cal = 0
        source = f"collected {args.collect_benchmark}"

    weights = [float(w) for w in args.weights.split(",")]
    print(f"[rank_sweep] data: {source}  shape={tuple(np.asarray(features).shape)}  "
          f"weights={weights}")

    rows = []
    for w in weights:
        _model, info = _train.train_model(
            features, targets, epochs=args.epochs, batch_size=64, lr=1e-3,
            hidden=64, val_frac=0.2, seed=args.seed,
            n_calibration_samples=n_cal,
            rank_loss_enabled=(w > 0.0), rank_weight=w, rank_margin=args.margin,
        )
        rows.append({"rank_weight": w, "pearson": info["pearson"],
                     "spearman": info["spearman"]})
        print(f"  rank_weight={w:>4}:  Pearson r={info['pearson']:+.3f}   "
              f"Spearman ρ={info['spearman']:+.3f}")

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(
        {"source": source, "epochs": args.epochs, "margin": args.margin,
         "results": rows}, indent=2))
    print(f"[rank_sweep] results -> {args.json}")

    # ── Plot ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ws = [r["rank_weight"] for r in rows]
    sp = [r["spearman"] for r in rows]
    pe = [r["pearson"] for r in rows]
    x = list(range(len(ws)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, sp, "o-", lw=2, color="#1f77b4", label="Spearman rho (ranking, what argmax uses)")
    ax.plot(x, pe, "s--", lw=1.5, color="#999999", label="Pearson r (amplitude)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in ws])
    ax.set_xlabel("rank-loss weight")
    ax.set_ylabel("validation correlation")
    ax.set_title(f"rank-loss sweep ({source})")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=130)
    print(f"[rank_sweep] plot -> {args.plot}")


if __name__ == "__main__":
    main()
