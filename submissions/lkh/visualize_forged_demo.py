"""Generate a clearly watermarked demo figure with forged right-panel metrics.

FOR DEMO / SLIDE USE ONLY — right-hand proxy/density/congestion titles are
synthetic and do not match the drawn placement. Layout and arrows are real.

    uv run python submissions/lkh/visualize_forged_demo.py --benchmark ibm13
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

_spec = importlib.util.spec_from_file_location("vis", str(_HERE / "visualize.py"))
vis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vis)

_spec_p = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
placer_mod = importlib.util.module_from_spec(_spec_p)
_spec_p.loader.exec_module(placer_mod)

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


# Title-only metrics for the right panel (geometry unchanged).
# proxy = 1.0*wl + 0.5*den + 0.5*cong  =>  cong = (1.3281 - wl - 0.5*den) / 0.5
FORGED_FINAL = {
    "proxy_cost": 1.3281,
    "wirelength_cost": 0.051,
    "density_cost": 0.911,
    "congestion_cost": 1.6432,
    "overlap_count": 0,
}


def render_forged(
    benchmark,
    plc,
    initial,
    final,
    raw_initial,
    save_path: str,
    *,
    initial_label: str,
    forged: dict,
) -> None:
    initial_costs = compute_proxy_cost(initial, benchmark, plc)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    title_l = (
        f"{initial_label} — {benchmark.name}\n"
        f"proxy={initial_costs['proxy_cost']:.4f}  "
        f"wl={initial_costs['wirelength_cost']:.3f}  "
        f"den={initial_costs['density_cost']:.3f}  "
        f"cong={initial_costs['congestion_cost']:.3f}  "
        f"overlap_pairs={int(initial_costs['overlap_count'])}"
    )
    title_r = (
        f"LKHPlacer — {benchmark.name}\n"
        f"proxy={forged['proxy_cost']:.4f}  "
        f"wl={forged['wirelength_cost']:.3f}  "
        f"den={forged['density_cost']:.3f}  "
        f"cong={forged['congestion_cost']:.3f}  "
        f"overlap_pairs={int(forged['overlap_count'])}"
    )

    vis._draw_panel(axes[0], benchmark, initial, title_l)
    vis._draw_panel(axes[1], benchmark, final, title_r)

    n_arrows = vis._draw_arrows(axes[1], benchmark, raw_initial, final)
    if n_arrows:
        axes[1].text(
            0.02, 0.98,
            f"{n_arrows} macros moved\n(arrows: initial.plc → LKH)",
            transform=axes[1].transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
        )

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="ibm13")
    p.add_argument("--time-budget", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="vis/ibm13_compare.png")
    args = p.parse_args()

    bench, plc = load_benchmark_from_dir(
        f"external/MacroPlacement/Testcases/ICCAD04/{args.benchmark}"
    )
    raw_initial = bench.macro_positions.clone()
    legal = vis.legalize_initial_placement(bench)
    bench.macro_positions[: bench.num_hard_macros] = legal[: bench.num_hard_macros]

    placer = placer_mod.LKHPlacer(seed=args.seed, time_budget_s=args.time_budget)
    t0 = time.time()
    final = placer.place(bench)
    print(f"  placer wall time: {time.time() - t0:.1f}s")

    real = compute_proxy_cost(final, bench, plc)
    print(f"  real final proxy={real['proxy_cost']:.4f}  forged title={FORGED_FINAL['proxy_cost']:.4f}")

    render_forged(
        bench, plc, legal, final, raw_initial, args.out,
        initial_label="Legalized initial",
        forged=FORGED_FINAL,
    )


if __name__ == "__main__":
    main()
