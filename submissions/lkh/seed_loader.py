"""Building block — analytical-seed loader (Fix 1).

The poster claims inference is seeded by a state-of-the-art analytical
placement (RePlAce or DREAMPlace), not by the ``initial.plc`` shipped with
the benchmark. This module loads such a seed from ``seeds/<bench>.plc``.

Drop a pre-computed RePlAce/DREAMPlace ``.plc`` output for each benchmark
into ``submissions/lkh/seeds/`` and the placer (with
``seed_source="replace"``) will pick it up instead of the default
initial.plc positions.

If the seed file is missing the loader returns ``None`` so the caller can
decide whether to raise (recommended — silent fallback to initial.plc is
exactly the bug the poster was meant to avoid) or warn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SEEDS_DIR = _HERE / "seeds"


def find_seed_plc(benchmark_name: str,
                   seeds_dir: Path | None = None) -> Path | None:
    """Resolve ``<seeds_dir>/<benchmark_name>.plc``. Returns ``None`` if missing."""
    base = Path(seeds_dir) if seeds_dir is not None else _SEEDS_DIR
    path = base / f"{benchmark_name}.plc"
    return path if path.exists() else None


def _resolve_netlist_dir(benchmark_name: str) -> Path | None:
    """Find the directory containing ``netlist.pb.txt`` for ``benchmark_name``.

    Mirrors ``placer._load_plc_if_available`` (placer.py lines 138-158) — we
    need the netlist to build a fresh ``PlacementCost`` against which we
    apply the seed ``.plc``.
    """
    iccad = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark_name
    if (iccad / "netlist.pb.txt").exists():
        return iccad

    ng45_map = {
        "ariane133": "ariane133",
        "ariane136": "ariane136",
        "nvdla": "nvdla",
        "mempool_tile": "mempool_tile",
    }
    design = ng45_map.get(benchmark_name) or ng45_map.get(
        benchmark_name.replace("_ng45", "")
    )
    if design is not None:
        base = (Path("external/MacroPlacement/Flows/NanGate45")
                / design / "netlist" / "output_CT_Grouping")
        if (base / "netlist.pb.txt").exists():
            return base

    return None


def load_seed_positions(benchmark, seeds_dir: Path | None = None
                        ) -> np.ndarray | None:
    """Load RePlAce/DREAMPlace seed positions for ``benchmark``.

    Returns ``[num_hard_macros, 2]`` numpy array of (x, y) centers, or
    ``None`` if no seed file is found. Raises on file-format/shape mismatch
    so a corrupted seed file fails loud rather than silently regressing.
    """
    seed_path = find_seed_plc(benchmark.name, seeds_dir)
    if seed_path is None:
        return None

    netlist_dir = _resolve_netlist_dir(benchmark.name)
    if netlist_dir is None:
        raise FileNotFoundError(
            f"seed_loader: found seed at {seed_path} for benchmark "
            f"{benchmark.name!r} but no matching netlist.pb.txt under "
            f"external/MacroPlacement/. Submodule may not be initialized."
        )

    from macro_place._plc import PlacementCost

    plc = PlacementCost(str(netlist_dir / "netlist.pb.txt"))
    # ifInital=False — the seed .plc overrides the initial state. ifReadComment
    # keeps comment-encoded metadata (canvas, grid) in sync with the seed.
    plc.restore_placement(str(seed_path), ifInital=False, ifReadComment=True)

    n_hard = benchmark.num_hard_macros
    positions = np.zeros((n_hard, 2), dtype=np.float64)
    for tensor_idx, plc_idx in enumerate(benchmark.hard_macro_indices):
        x, y = plc.modules_w_pins[plc_idx].get_pos()
        positions[tensor_idx, 0] = float(x)
        positions[tensor_idx, 1] = float(y)

    if positions.shape != (n_hard, 2):
        raise ValueError(
            f"seed_loader: seed at {seed_path} yielded shape "
            f"{positions.shape}, expected {(n_hard, 2)}"
        )
    return positions


def require_seed_positions(benchmark, seeds_dir: Path | None = None
                            ) -> np.ndarray:
    """Strict variant of ``load_seed_positions`` — raises if the seed is missing.

    The placer uses this when ``seed_source="replace"`` so an absent seed
    file is a loud error rather than a silent regression to ``initial.plc``.
    """
    positions = load_seed_positions(benchmark, seeds_dir)
    if positions is None:
        base = Path(seeds_dir) if seeds_dir is not None else _SEEDS_DIR
        raise FileNotFoundError(
            f"seed_loader: no seed file at {base / f'{benchmark.name}.plc'}. "
            f"Drop a RePlAce/DREAMPlace output there, or use seed_source='initial'."
        )
    return positions
