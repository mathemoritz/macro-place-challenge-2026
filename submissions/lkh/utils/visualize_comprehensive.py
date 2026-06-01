"""Comprehensive placement visualization: layout + heatmaps + training trace.

Combines:
  - Initial vs LKHPlacer layout (with displacement arrows on final panel)
  - Density and congestion heatmaps for initial and final (legalized final)
  - Optional per-round eval proxy from iter history.json

Usage:
    uv run python submissions/lkh/visualize_comprehensive.py --benchmark ibm13
    uv run python submissions/lkh/visualize_comprehensive.py --benchmark ibm04 \\
        --history modal_output/iter/history.json --time-budget 60
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import _set_placement, compute_proxy_cost
from submissions.lkh.learning.train import parse_benchmarks
from visualize import _draw_arrows, _draw_panel

_spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
placer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(placer_mod)


def _heatmap_panel(ax, benchmark, placement, plc, mode: str, title: str) -> None:
    """mode: 'density' | 'congestion'"""
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    ax.add_patch(Rectangle((0, 0), cw, ch, fill=False, edgecolor="black", linewidth=1.5))
    _set_placement(plc, placement, benchmark)
    extent = (0, cw, 0, ch)
    nrow, ncol = benchmark.grid_rows, benchmark.grid_cols

    if mode == "density":
        plc.get_density_cost()
        grid = np.asarray(plc.grid_cells, dtype=float).reshape(nrow, ncol)
        cmap, label = "Blues", "Density"
    else:
        plc.get_congestion_cost()
        h = np.asarray(plc.H_routing_cong, dtype=float).reshape(nrow, ncol)
        v = np.asarray(plc.V_routing_cong, dtype=float).reshape(nrow, ncol)
        grid = np.maximum(h, v)
        pos = grid[grid > 0]
        cmap, label = "hot", "Congestion (max H/V)"

    vmax = max(float(np.max(grid)), 1e-9)
    if mode == "congestion":
        pos = grid[grid > 0]
        vmax = float(np.percentile(pos, 99)) if pos.size else vmax
        vmax = max(vmax, 1e-9)

    im = ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        alpha=0.75,
        vmin=0.0,
        vmax=vmax,
        zorder=0,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes
    for i in range(n_hard):
        x, y = placement[i].tolist()
        w, h = sizes[i].tolist()
        ax.add_patch(
            Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                fill=False,
                edgecolor="black",
                linewidth=0.4,
                zorder=2,
            )
        )
    ax.set_xlim(-cw * 0.02, cw * 1.02)
    ax.set_ylim(-ch * 0.02, ch * 1.02)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)


def _load_round_history(history_path: Path | None, bench: str) -> list[tuple[int, float]]:
    if history_path is None or not history_path.is_file():
        return []
    data = json.loads(history_path.read_text())
    out = []
    for rec in data:
        pb = rec.get("per_benchmark", {}).get(bench)
        if pb is not None:
            out.append((int(rec["round"]), float(pb["proxy_cost"])))
    return out


def visualize_comprehensive(
    benchmark,
    plc,
    initial,
    final,
    save_path: str,
    round_history: list[tuple[int, float]] | None = None,
    show_arrows: bool = True,
) -> dict:
    init_costs = compute_proxy_cost(initial, benchmark, plc)
    final_costs = compute_proxy_cost(final, benchmark, plc)

    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1.0], hspace=0.28, wspace=0.22)

    ax_init = fig.add_subplot(gs[0, 0:2])
    ax_final = fig.add_subplot(gs[0, 2:4])

    title_init = (
        f"Initial.plc — {benchmark.name}\n"
        f"proxy={init_costs['proxy_cost']:.4f}  wl={init_costs['wirelength_cost']:.3f}  "
        f"den={init_costs['density_cost']:.3f}  cong={init_costs['congestion_cost']:.3f}  "
        f"overlaps={int(init_costs['overlap_count'])}"
    )
    title_final = (
        f"LKHPlacer (checkpoints) — {benchmark.name}\n"
        f"proxy={final_costs['proxy_cost']:.4f}  wl={final_costs['wirelength_cost']:.3f}  "
        f"den={final_costs['density_cost']:.3f}  cong={final_costs['congestion_cost']:.3f}  "
        f"overlaps={int(final_costs['overlap_count'])}"
    )
    _draw_panel(ax_init, benchmark, initial, title_init)
    _draw_panel(ax_final, benchmark, final, title_final)
    n_arrows = _draw_arrows(ax_final, benchmark, initial, final) if show_arrows else 0
    if n_arrows:
        ax_final.text(
            0.02,
            0.98,
            f"{n_arrows} macros moved",
            transform=ax_final.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="gray"),
        )

    plc_init = plc
    _heatmap_panel(
        fig.add_subplot(gs[1, 0]),
        benchmark,
        initial,
        plc_init,
        "density",
        f"Initial — density",
    )
    _heatmap_panel(
        fig.add_subplot(gs[1, 1]),
        benchmark,
        initial,
        plc_init,
        "congestion",
        f"Initial — congestion",
    )
    _heatmap_panel(
        fig.add_subplot(gs[1, 2]),
        benchmark,
        final,
        plc_init,
        "density",
        f"LKHPlacer — density",
    )
    _heatmap_panel(
        fig.add_subplot(gs[1, 3]),
        benchmark,
        final,
        plc_init,
        "congestion",
        f"LKHPlacer — congestion",
    )

    delta = float(final_costs["proxy_cost"]) - float(init_costs["proxy_cost"])
    suptitle = (
        f"{benchmark.name}: proxy {init_costs['proxy_cost']:.4f} → "
        f"{final_costs['proxy_cost']:.4f} (Δ{delta:+.4f})"
    )
    fig.suptitle(suptitle, fontsize=13, y=0.98)

    if round_history:
        ax_trace = fig.add_axes([0.08, 0.01, 0.55, 0.07])
        rounds, proxies = zip(*round_history)
        ax_trace.plot(rounds, proxies, "o-", color="steelblue", lw=2, markersize=8)
        ax_trace.set_xlabel("Training round (Modal eval)")
        ax_trace.set_ylabel("Eval proxy")
        ax_trace.set_title("Iterative training eval trace (0 overlaps, legalized)", fontsize=9)
        ax_trace.grid(True, alpha=0.3)
        for r, p in round_history:
            ax_trace.annotate(
                f"{p:.3f}",
                (r, p),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")

    return {
        "initial_proxy": float(init_costs["proxy_cost"]),
        "final_proxy": float(final_costs["proxy_cost"]),
        "delta_proxy": delta,
        "initial_overlaps": int(init_costs["overlap_count"]),
        "final_overlaps": int(final_costs["overlap_count"]),
        "n_macros_moved": n_arrows,
    }


def main():
    p = argparse.ArgumentParser(description="Comprehensive LKH placement visualization")
    p.add_argument("--benchmark", default="ibm13")
    p.add_argument("--time-budget", type=float, default=30.0)
    p.add_argument("--max-chains", type=int, default=2000)
    p.add_argument("--max-chain-length", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="vis")
    p.add_argument(
        "--history",
        default="modal_output/iter/history.json",
        help="Path to history.json for eval-round trace inset.",
    )
    p.add_argument("--no-arrows", action="store_true")
    p.add_argument(
        "--placement-only",
        action="store_true",
        help="Skip placer run; only render initial.plc (for quick heatmap check).",
    )
    args = p.parse_args()

    name = parse_benchmarks(args.benchmark)[0]
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    initial = benchmark.macro_positions.clone()

    if args.placement_only:
        final = initial.clone()
    else:
        placer = placer_mod.LKHPlacer(
            seed=args.seed,
            time_budget_s=args.time_budget,
            max_chains=args.max_chains,
            max_chain_length=args.max_chain_length,
        )
        t0 = time.time()
        final = placer.place(benchmark)
        print(f"  placer wall time: {time.time() - t0:.1f}s")

    history_path = Path(args.history) if args.history else None
    round_hist = _load_round_history(history_path, name)

    out = Path(args.out_dir) / f"{name}_comprehensive.png"
    metrics = visualize_comprehensive(
        benchmark,
        plc,
        initial,
        final,
        str(out),
        round_history=round_hist or None,
        show_arrows=not args.no_arrows,
    )
    print(
        f"  local run: init_proxy={metrics['initial_proxy']:.4f}  "
        f"final_proxy={metrics['final_proxy']:.4f}  "
        f"Δ={metrics['delta_proxy']:+.4f}  moved={metrics['n_macros_moved']}"
    )
    if round_hist:
        print(f"  Modal eval trace: " + "  ".join(f"r{r}={p:.4f}" for r, p in round_hist))


if __name__ == "__main__":
    main()
