"""Run the LKH training pipeline on Modal cloud.

App definition (one ``modal.App``, one ``modal.Image``, one persistent
``modal.Volume``) and four entrypoints — one per training phase plus a
smoke test:

    smoke    : confirm image + mounts + benchmark loading work
    train    : Phase 3 — train CostApproximator
    policy   : Phase 4 — train ChainPolicy via PPO
    iter_    : Phase 5 — full iterative loop (Phase 3 + 4 per round)

Quickstart:

    # First time: install Modal and authenticate
    uv add modal
    uv run modal setup

    # Smoke-test the image and mounts (~2-3 minutes)
    uv run modal run submissions/lkh/modal_run.py::smoke

    # Phase 3 across all 17 benchmarks (per-benchmark example count = N)
    uv run modal run submissions/lkh/modal_run.py::train \\
        --benchmark all --num-examples 300 --epochs 80

    # Phase 4 across all 17 benchmarks, real-scale training
    uv run modal run submissions/lkh/modal_run.py::policy \\
        --benchmark all --iterations 3000

    # Full iterative pipeline (long-running; use --detach)
    uv run modal run --detach submissions/lkh/modal_run.py::iter_ \\
        --benchmark all --rounds 3 --examples 200 --policy-iterations 1500

After completion, all checkpoints, training data, and visualizations are
persisted to the Modal Volume ``lkh-results``. Download with:

    modal volume get lkh-results /iter/checkpoints ./checkpoints

Reference (CS224R Modal Compute Guide, sections 4-5).
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import modal

# ── App, image, volume, mount ──────────────────────────────────────────────

app = modal.App("lkh-macro-place")

# Local repo root (this file is at submissions/lkh/modal_run.py).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Python 3.11 matches the judges' container and our local uv environment.
# Deps mirror submissions/lkh + macro_place's pyproject.toml (CPU torch is fine —
# our MLPs are tiny and the bottleneck is `compute_proxy_cost` which is CPU-only).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "numpy>=1.20",
        "matplotlib>=3.5",
        "tqdm>=4.65",
        "absl-py>=1.0",
    )
)

volume = modal.Volume.from_name("lkh-results", create_if_missing=True)


def _include(path: str) -> bool:
    """Filter for ``Mount.from_local_dir``: skip transient, IDE, and bloat dirs.

    The TILOS MacroPlacement submodule alone is ~3 GB; we drop everything
    except CodeElements/Plc_client (needed for the PlacementCost parser) and
    Testcases/ICCAD04 (the benchmarks we train on).
    """
    # Substrings to skip anywhere in the path.
    drop = [
        "/.git/", "/.venv/", "/__pycache__/", "/.pytest_cache/",
        "/.idea/", "/.vscode/", "/.cursor/", "/.DS_Store",
        "/vis/", "/vis_surrogate/", "/agent-transcripts/",
        # MacroPlacement top-level directories we don't need.
        "/external/MacroPlacement/Flows/",
        "/external/MacroPlacement/Docs/",
        "/external/MacroPlacement/Enablements/",
        "/external/MacroPlacement/ExperimentalData/",
        # Heavy CodeElements subdirs we don't import.
        "/external/MacroPlacement/CodeElements/Clustering/",
        "/external/MacroPlacement/CodeElements/CodeFlowIntegration/",
        "/external/MacroPlacement/CodeElements/EvalCT/",
        "/external/MacroPlacement/CodeElements/FDPlacement/",
        "/external/MacroPlacement/CodeElements/FormatTranslators/",
        "/external/MacroPlacement/CodeElements/Gridding/",
        "/external/MacroPlacement/CodeElements/Grouping/",
        "/external/MacroPlacement/CodeElements/SimulatedAnnealing/",
        "/external/MacroPlacement/CodeElements/SimulatedAnnealingGWTW/",
        "/external/MacroPlacement/CodeElements/StatTest/",
        "/external/MacroPlacement/CodeElements/VisualPlacement/",
    ]
    return not any(s in path for s in drop)


repo_mount = modal.Mount.from_local_dir(
    str(REPO_ROOT),
    remote_path="/root/repo",
    condition=_include,
)


# ── Container-side helpers (run inside Modal) ──────────────────────────────

def _setup_env() -> None:
    """Set the cwd and PYTHONPATH so macro_place + plc_client_os are importable."""
    import os
    import sys
    os.chdir("/root/repo")
    for p in ("/root/repo",
              "/root/repo/submissions/lkh",
              "/root/repo/external/MacroPlacement/CodeElements/Plc_client"):
        if p not in sys.path:
            sys.path.insert(0, p)


def _run(cmd: list[str]) -> None:
    """Run a python subprocess from the repo root with PYTHONPATH set."""
    import os
    _setup_env()
    env = os.environ.copy()
    extra = ":".join([
        "/root/repo",
        "/root/repo/submissions/lkh",
        "/root/repo/external/MacroPlacement/CodeElements/Plc_client",
    ])
    env["PYTHONPATH"] = f"{extra}:{env.get('PYTHONPATH', '')}"
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True, cwd="/root/repo")


def _persist(label: str) -> None:
    """Copy checkpoints / data / vis from the container to the volume under
    ``/output/<label>/``. Existing files are overwritten."""
    targets = [
        ("submissions/lkh/checkpoints", f"/output/{label}/checkpoints"),
        ("submissions/lkh/data",        f"/output/{label}/data"),
        ("vis",                          f"/output/{label}/vis"),
    ]
    for src, dst in targets:
        src_path = Path("/root/repo") / src
        if src_path.exists() and any(src_path.iterdir()):
            shutil.copytree(str(src_path), dst, dirs_exist_ok=True)
            print(f"  persisted {src} -> {dst}", flush=True)
    volume.commit()


# ── Remote functions ───────────────────────────────────────────────────────

@app.function(image=image, mounts=[repo_mount], volumes={"/output": volume},
              cpu=8.0, memory=8192, timeout=30 * 60)
def run_smoke() -> None:
    """Verify the image, mount, and benchmark loader on a fresh container."""
    _setup_env()
    print("=== Modal smoke test ===")
    import sys
    import torch
    import numpy as np
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  Torch  : {torch.__version__}")
    print(f"  NumPy  : {np.__version__}")
    from macro_place.loader import load_benchmark_from_dir
    bench_dir = "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    if not Path(bench_dir).exists():
        print(f"FAIL: testcases not in mount at {bench_dir}")
        return
    bench, _plc = load_benchmark_from_dir(bench_dir)
    print(f"  loaded {bench.name}: "
          f"{bench.num_hard_macros} hard / {bench.num_macros} total macros")
    print("OK")


@app.function(image=image, mounts=[repo_mount], volumes={"/output": volume},
              cpu=8.0, memory=8192, timeout=6 * 3600)
def run_train(benchmark: str, num_examples: int, epochs: int, seed: int,
              force_recollect: bool) -> None:
    cmd = ["python", "submissions/lkh/train.py",
           "--benchmark", benchmark,
           "--num-examples", str(num_examples),
           "--epochs", str(epochs),
           "--seed", str(seed)]
    if force_recollect:
        cmd.append("--force-recollect")
    _run(cmd)
    _persist("train")


@app.function(image=image, mounts=[repo_mount], volumes={"/output": volume},
              cpu=8.0, memory=8192, timeout=12 * 3600)
def run_policy(benchmark: str, iterations: int, trajectories_per_iter: int,
               seed: int, initial_policy: str) -> None:
    cmd = ["python", "submissions/lkh/train_policy.py",
           "--benchmark", benchmark,
           "--iterations", str(iterations),
           "--trajectories-per-iter", str(trajectories_per_iter),
           "--seed", str(seed)]
    if initial_policy:
        cmd += ["--initial-policy", initial_policy]
    _run(cmd)
    _persist("policy")


@app.function(image=image, mounts=[repo_mount], volumes={"/output": volume},
              cpu=8.0, memory=16384, timeout=24 * 3600)
def run_iter(benchmark: str, rounds: int, examples: int, cost_epochs: int,
             policy_iterations: int, trajectories_per_iter: int,
             eval_time_budget: float, seed: int, force_recollect: bool) -> None:
    cmd = ["python", "submissions/lkh/train_iter.py",
           "--benchmark", benchmark,
           "--rounds", str(rounds),
           "--examples", str(examples),
           "--cost-epochs", str(cost_epochs),
           "--policy-iterations", str(policy_iterations),
           "--trajectories-per-iter", str(trajectories_per_iter),
           "--eval-time-budget", str(eval_time_budget),
           "--seed", str(seed)]
    if force_recollect:
        cmd.append("--force-recollect-each-round")
    _run(cmd)
    _persist("iter")


# ── Local entrypoints ──────────────────────────────────────────────────────

@app.local_entrypoint()
def smoke() -> None:
    """Smoke-test the Modal setup (~2-3 minutes the first time)."""
    run_smoke.remote()


@app.local_entrypoint()
def train(benchmark: str = "ibm01", num_examples: int = 1500, epochs: int = 60,
          seed: int = 42, force_recollect: bool = False) -> None:
    """Phase 3 — train CostApproximator on Modal.

    Examples:
        modal run modal_run.py::train --benchmark ibm01,ibm02 --num-examples 1500
        modal run modal_run.py::train --benchmark all --num-examples 300
    """
    run_train.remote(
        benchmark=benchmark, num_examples=num_examples,
        epochs=epochs, seed=seed, force_recollect=force_recollect,
    )


@app.local_entrypoint()
def policy(benchmark: str = "ibm01", iterations: int = 1000,
           trajectories_per_iter: int = 4, seed: int = 42,
           initial_policy: str = "") -> None:
    """Phase 4 — train ChainPolicy via PPO on Modal.

    Examples:
        modal run modal_run.py::policy --benchmark all --iterations 3000
        modal run modal_run.py::policy --benchmark ibm01,ibm07 --iterations 1000
    """
    run_policy.remote(
        benchmark=benchmark, iterations=iterations,
        trajectories_per_iter=trajectories_per_iter,
        seed=seed, initial_policy=initial_policy,
    )


@app.local_entrypoint()
def iter_(benchmark: str = "ibm01", rounds: int = 2, examples: int = 400,
          cost_epochs: int = 60, policy_iterations: int = 1000,
          trajectories_per_iter: int = 4, eval_time_budget: float = 20.0,
          seed: int = 42, force_recollect: bool = False) -> None:
    """Phase 5 — iterative training loop (Phases 3 + 4 per round).

    Examples:
        # Quick: 1 round, 3 benchmarks, modest data
        modal run modal_run.py::iter_ --benchmark ibm01,ibm02,ibm07 \\
            --rounds 1 --examples 200 --policy-iterations 500

        # Full: detached run, all benchmarks, multiple rounds
        modal run --detach modal_run.py::iter_ --benchmark all \\
            --rounds 3 --examples 200 --policy-iterations 1500
    """
    run_iter.remote(
        benchmark=benchmark, rounds=rounds, examples=examples,
        cost_epochs=cost_epochs, policy_iterations=policy_iterations,
        trajectories_per_iter=trajectories_per_iter,
        eval_time_budget=eval_time_budget,
        seed=seed, force_recollect=force_recollect,
    )
