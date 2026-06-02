"""visualize.py — Initial vs Final placement comparison.

Side-by-side rendering for the LKHPlacer: the initial.plc layout on the
left and the placer's output on the right, with optional arrows showing
per-macro displacement and proxy-cost metrics in each panel's title.

Usage:
    uv run python submissions/lkh/visualize.py
    uv run python submissions/lkh/visualize.py --benchmark ibm03 --time-budget 30
    uv run python submissions/lkh/visualize.py --benchmark ibm01,ibm07
    uv run python submissions/lkh/visualize.py --benchmark all --no-arrows
    uv run python submissions/lkh/visualize.py --benchmark ibm13 --initial legalized --time-budget 60
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

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_spec = importlib.util.spec_from_file_location("lkh_placer", str(_HERE.parent / "placer.py"))
placer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(placer_mod)

from submissions.lkh.learning.train import parse_benchmarks  # comma-separated / "all" CLI helper

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


def legalize_initial_placement(benchmark) -> torch.Tensor:
    """Return overlap-free hard-macro positions (spiral legalization)."""
    hpwl_edges = placer_mod._hard_macro_edges(benchmark)
    state = placer_mod.PlacementState(benchmark, hpwl_edges)
    n_before = state.overlap_pairs()
    if n_before > 0:
        state.pos[:] = placer_mod._legalize(state)
        state.rebuild_caches()
    n = benchmark.num_hard_macros
    out = benchmark.macro_positions.clone()
    out[:n] = torch.tensor(state.pos, dtype=torch.float32)
    print(f"  legalized initial: {n_before} -> {state.overlap_pairs()} overlap pairs")
    return out


# ── Rendering ──────────────────────────────────────────────────────────────


def _draw_panel(ax, benchmark, placement, title: str) -> None:
    """Canvas border + soft macros (faint) + hard macros (solid)."""
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes

    ax.add_patch(Rectangle((0, 0), cw, ch, fill=False, edgecolor="black", linewidth=2))

    # Soft macros — drawn first so hard macros sit on top.
    for i in range(n_hard, benchmark.num_macros):
        x, y = placement[i].tolist()
        w, h = sizes[i].tolist()
        ax.add_patch(
            Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                fill=True,
                facecolor="lightsteelblue",
                alpha=0.15,
                edgecolor="gray",
                linewidth=0.3,
                linestyle="dashed",
            )
        )

    # Hard macros — fixed in red, movable in blue.
    for i in range(n_hard):
        x, y = placement[i].tolist()
        w, h = sizes[i].tolist()
        is_fixed = bool(benchmark.macro_fixed[i].item())
        face = "red" if is_fixed else "steelblue"
        alpha = 0.3 if is_fixed else 0.5
        ax.add_patch(
            Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                fill=True,
                facecolor=face,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.5,
            )
        )

    ax.set_xlim(-cw * 0.02, cw * 1.02)
    ax.set_ylim(-ch * 0.02, ch * 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.set_title(title, fontsize=11)


def _draw_arrows(ax, benchmark, initial, final, threshold_frac: float = 0.005) -> int:
    """Arrows on ``ax`` from initial→final for each hard macro that moved
    more than ``threshold_frac * canvas_diagonal``. Returns the number
    of arrows actually drawn.
    """
    n_hard = benchmark.num_hard_macros
    init_pos = initial[:n_hard].numpy()
    final_pos = final[:n_hard].numpy()
    deltas = final_pos - init_pos
    dist = np.linalg.norm(deltas, axis=1)
    diag = float((benchmark.canvas_width**2 + benchmark.canvas_height**2) ** 0.5)
    threshold = threshold_frac * diag

    if dist.max() <= threshold:
        return 0
    cmap = plt.get_cmap("viridis")
    norm = max(float(dist.max()), 1e-6)
    n_drawn = 0
    for i in range(n_hard):
        if dist[i] < threshold:
            continue
        color = cmap(min(float(dist[i]) / norm, 1.0))
        ax.annotate(
            "",
            xy=(float(final_pos[i, 0]), float(final_pos[i, 1])),
            xytext=(float(init_pos[i, 0]), float(init_pos[i, 1])),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8, alpha=0.75),
        )
        n_drawn += 1
    return n_drawn


def visualize_initial_vs_final(
    benchmark,
    plc,
    initial,
    final,
    save_path: str,
    show_arrows: bool = True,
    initial_label: str = "Initial",
    arrow_baseline=None,
) -> dict:
    """Two-panel figure: initial.plc on the left, placer output on the right.

    Metrics in each title come from ``compute_proxy_cost`` so the comparison
    is grounded in what the competition evaluator actually scores.
    """
    initial_costs = compute_proxy_cost(initial, benchmark, plc)
    final_costs = compute_proxy_cost(final, benchmark, plc)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    title_l = (
        f"{initial_label} — {benchmark.name}\n"
        f"proxy={initial_costs['proxy_cost']:.4f}  "
        f"wl={initial_costs['wirelength_cost']:.3f}  "
        f"den={initial_costs['density_cost']:.3f}  "
        f"cong={initial_costs['congestion_cost']:.3f}  "
        f"overlap_pairs={int(initial_costs['overlap_count'])}"
    )
    _draw_panel(axes[0], benchmark, initial, title_l)

    title_r = (
        f"LKHPlacer — {benchmark.name}\n"
        f"proxy={final_costs['proxy_cost']:.4f}  "
        f"wl={final_costs['wirelength_cost']:.3f}  "
        f"den={final_costs['density_cost']:.3f}  "
        f"cong={final_costs['congestion_cost']:.3f}  "
        f"overlap_pairs={int(final_costs['overlap_count'])}"
    )
    _draw_panel(axes[1], benchmark, final, title_r)

    arrow_from = initial if arrow_baseline is None else arrow_baseline
    n_arrows = _draw_arrows(axes[1], benchmark, arrow_from, final) if show_arrows else 0
    if show_arrows and n_arrows > 0:
        arrow_note = (
            "arrows: initial.plc → LKH"
            if arrow_baseline is not None
            else "arrows: left panel → LKH"
        )
        axes[1].text(
            0.02,
            0.98,
            f"{n_arrows} macros moved\n({arrow_note})",
            transform=axes[1].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
        )

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")

    return {
        "initial_proxy": float(initial_costs["proxy_cost"]),
        "final_proxy": float(final_costs["proxy_cost"]),
        "initial_overlaps": int(initial_costs["overlap_count"]),
        "final_overlaps": int(final_costs["overlap_count"]),
        "n_macros_moved": int(n_arrows),
    }


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--benchmark", default="ibm01", help="Single name, comma-separated list, or 'all'."
    )
    p.add_argument("--time-budget", type=float, default=15.0)
    p.add_argument("--max-chains", type=int, default=2000)
    p.add_argument("--max-chain-length", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="vis")
    p.add_argument(
        "--no-arrows", action="store_true", help="Hide the per-macro displacement arrows."
    )
    p.add_argument(
        "--initial",
        choices=("raw", "legalized"),
        default="raw",
        help="'raw' = initial.plc as shipped; 'legalized' = spiral-"
        "legalized start (0 overlap pairs) for fair proxy comparison.",
    )
    p.add_argument(
        "--arrows-from",
        choices=("same", "raw"),
        default="same",
        help="Displacement arrows on the right panel: 'same' = from "
        "the left-panel placement; 'raw' = from shipped initial.plc "
        "(use with --initial legalized to show total motion).",
    )
    args = p.parse_args()

    benchmarks = parse_benchmarks(args.benchmark)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for name in benchmarks:
        print(f"\n=== {name} ===")
        bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        bench, plc = load_benchmark_from_dir(str(bench_dir))
        raw_initial = bench.macro_positions.clone()
        if args.initial == "legalized":
            initial = legalize_initial_placement(bench)
            bench.macro_positions[: bench.num_hard_macros] = initial[: bench.num_hard_macros]
            initial_label = "Legalized initial"
        else:
            initial = raw_initial.clone()
            initial_label = "Initial.plc"

        arrow_baseline = None
        if args.arrows_from == "raw":
            arrow_baseline = raw_initial
        elif args.initial == "legalized" and not args.no_arrows:
            print(
                "  hint: with --initial legalized the placer often moves few "
                "macros; add --arrows-from raw to show motion from initial.plc"
            )

        placer = placer_mod.LKHPlacer(
            seed=args.seed,
            time_budget_s=args.time_budget,
            max_chains=args.max_chains,
            max_chain_length=args.max_chain_length,
        )
        t0 = time.time()
        final = placer.place(bench)
        wall = time.time() - t0
        print(f"  placer wall time: {wall:.1f}s")

        save_path = out_dir / f"{name}_compare.png"
        metrics = visualize_initial_vs_final(
            bench,
            plc,
            initial,
            final,
            str(save_path),
            show_arrows=not args.no_arrows,
            initial_label=initial_label,
            arrow_baseline=arrow_baseline,
        )
        metrics["benchmark"] = name
        metrics["wall_s"] = wall
        summary.append(metrics)

    if len(summary) > 1:
        print(f"\n=== Summary ===")
        print(
            f"  {'benchmark':>10}  {'init_proxy':>10}  {'final_proxy':>11}  "
            f"{'Δproxy':>8}  {'init_ov':>7}  {'final_ov':>8}  {'moved':>6}  {'time':>6}"
        )
        for s in summary:
            d = s["final_proxy"] - s["initial_proxy"]
            sign = "+" if d >= 0 else ""
            print(
                f"  {s['benchmark']:>10}  {s['initial_proxy']:>10.4f}  "
                f"{s['final_proxy']:>11.4f}  {sign}{d:>7.4f}  "
                f"{s['initial_overlaps']:>7d}  {s['final_overlaps']:>8d}  "
                f"{s['n_macros_moved']:>6d}  {s['wall_s']:>5.1f}s"
            )


if __name__ == "__main__":
    main()
