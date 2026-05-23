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

def _ignore(path) -> bool:
    """``ignore`` callable for ``Image.add_local_dir``: True ⇒ exclude file.

    Modal 1.x removed ``modal.Mount`` and replaced ``condition=`` (True = keep)
    with ``ignore=`` (True = drop). The TILOS MacroPlacement submodule alone
    is ~3 GB; we drop everything except CodeElements/Plc_client (needed for the
    PlacementCost parser) and Testcases/ICCAD04 (our 17 benchmarks).

    The path arg is a ``pathlib.Path`` relative to ``local_path``; we normalize
    with a leading slash so ``/segment/`` substring checks work uniformly.
    """
    p = "/" + str(path).replace("\\", "/").strip("/")
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
    return any(s in p for s in drop)


# Python 3.11 matches the judges' container and our local uv environment.
# Deps mirror submissions/lkh + macro_place's pyproject.toml (CPU torch is fine —
# our MLPs are tiny and the bottleneck is `compute_proxy_cost` which is CPU-only).
# ``add_local_dir`` is the modern Modal replacement for ``Mount.from_local_dir``;
# with ``copy=False`` (the default), files are attached at container start, not
# baked into the image — so iterating on local code doesn't rebuild the image.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "numpy>=1.20",
        "matplotlib>=3.5",
        "tqdm>=4.65",
        "absl-py>=1.0",
    )
    .add_local_dir(
        local_path=str(REPO_ROOT),
        remote_path="/root/repo",
        ignore=_ignore,
    )
)

volume = modal.Volume.from_name("lkh-results", create_if_missing=True)


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

@app.function(image=image, volumes={"/output": volume},
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


@app.function(image=image, volumes={"/output": volume},
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


@app.function(image=image, volumes={"/output": volume},
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


@app.function(image=image, volumes={"/output": volume},
              cpu=8.0, memory=16384, timeout=24 * 3600)
def run_iter(benchmark: str, rounds: int, examples: int, cost_epochs: int,
             policy_iterations: int, trajectories_per_iter: int,
             eval_time_budget: float, seed: int, force_recollect: bool) -> None:
    """Crash-safe Phase 5 driver: imports ``train_iter`` in-process and calls
    ``run_round`` in a loop, committing the volume after every round so a
    detached run that dies at hour 9 still leaves rounds 0..N-1 persisted.
    """
    import json as _json
    _setup_env()
    import train as _train
    import train_iter as _ti

    benchmarks = _train.parse_benchmarks(benchmark)
    output_root = Path("/output/iter")
    output_root.mkdir(parents=True, exist_ok=True)
    data_path = Path("/root/repo/submissions/lkh/data/chain_data.pt")
    cost_ckpt = Path("/root/repo/submissions/lkh/checkpoints/cost_approximator.pt")
    policy_ckpt = Path("/root/repo/submissions/lkh/checkpoints/chain_policy.pt")

    print(f"=== Phase 5 on Modal ===")
    print(f"  benchmarks={benchmarks}  rounds={rounds}  output={output_root}")

    history: list[dict] = []
    for r in range(rounds):
        record = _ti.run_round(
            r,
            benchmarks=benchmarks,
            num_examples_per_benchmark=examples,
            cost_epochs=cost_epochs,
            policy_iterations=policy_iterations,
            trajectories_per_iter=trajectories_per_iter,
            seed=seed,
            data_path=data_path,
            cost_ckpt=cost_ckpt,
            policy_ckpt=policy_ckpt,
            force_recollect=force_recollect,
            eval_time_budget_s=eval_time_budget,
            output_root=output_root,
        )
        history.append(record)
        # Stream aggregated history + canonical-path snapshot, then commit so
        # if the container dies on the NEXT round, rounds 0..r are durable.
        (output_root / "history.json").write_text(
            _json.dumps(history, indent=2, default=str)
        )
        _persist("iter")
        volume.commit()
        print(f"[Modal] round {r} committed ({len(history)} rounds done).")


# ── Local entrypoints ──────────────────────────────────────────────────────

@app.local_entrypoint()
def smoke() -> None:
    """Smoke-test the Modal setup (~2-3 minutes the first time).

    Uses ``.remote()`` because we *want* to wait for the result locally
    (it's a fast sanity check).
    """
    run_smoke.remote()


# ---------------------------------------------------------------------------
# Long-running entrypoints use ``.spawn()``, not ``.remote()``.
#
# Per Modal: ``.remote()`` blocks until the function returns, and a closed
# local terminal cancels the in-flight call EVEN with ``--detach``. Using
# ``.spawn()`` dispatches the work and returns a FunctionCall handle
# immediately, so the job survives any local disconnection. The handle's
# ``object_id`` is what you pass to ``modal call logs <id>`` to stream
# output afterwards.
# ---------------------------------------------------------------------------

def _print_spawn_handle(call, label: str) -> None:
    """Print the FunctionCall ID + the commands to monitor / cancel."""
    fid = getattr(call, "object_id", None) or str(call)
    print(f"\nSpawned {label} on Modal.")
    print(f"  function_call_id: {fid}")
    print(f"  monitor:  modal call logs {fid}")
    print(f"  result:   modal call get {fid}")
    print(f"  stop:     modal app stop lkh-macro-place")
    print(f"  dashboard: https://modal.com/apps")


@app.local_entrypoint()
def train(benchmark: str = "ibm01", num_examples: int = 1500, epochs: int = 60,
          seed: int = 42, force_recollect: bool = False) -> None:
    """Phase 3 — train CostApproximator on Modal (background spawn).

    Examples:
        modal run modal_run.py::train --benchmark ibm01,ibm02 --num-examples 1500
        modal run modal_run.py::train --benchmark all --num-examples 300
    """
    call = run_train.spawn(
        benchmark=benchmark, num_examples=num_examples,
        epochs=epochs, seed=seed, force_recollect=force_recollect,
    )
    _print_spawn_handle(call, "run_train")


@app.local_entrypoint()
def policy(benchmark: str = "ibm01", iterations: int = 1000,
           trajectories_per_iter: int = 4, seed: int = 42,
           initial_policy: str = "") -> None:
    """Phase 4 — train ChainPolicy via PPO on Modal (background spawn).

    Examples:
        modal run modal_run.py::policy --benchmark all --iterations 3000
        modal run modal_run.py::policy --benchmark ibm01,ibm07 --iterations 1000
    """
    call = run_policy.spawn(
        benchmark=benchmark, iterations=iterations,
        trajectories_per_iter=trajectories_per_iter,
        seed=seed, initial_policy=initial_policy,
    )
    _print_spawn_handle(call, "run_policy")


@app.local_entrypoint()
def iter_(benchmark: str = "ibm01", rounds: int = 2, examples: int = 400,
          cost_epochs: int = 60, policy_iterations: int = 1000,
          trajectories_per_iter: int = 4, eval_time_budget: float = 20.0,
          seed: int = 42, force_recollect: bool = False) -> None:
    """Phase 5 — iterative training loop, dispatched as a background job.

    Examples:
        # Quick: 1 round, 3 benchmarks, modest data
        modal run modal_run.py::iter_ --benchmark ibm01,ibm02,ibm07 \\
            --rounds 1 --examples 200 --policy-iterations 500

        # Full: dispatch detached, all benchmarks, multiple rounds
        modal run modal_run.py::iter_ --benchmark all \\
            --rounds 3 --examples 200 --policy-iterations 1500

    The job runs to completion regardless of local terminal state — this
    entrypoint returns as soon as the spawn is dispatched. ``--detach`` is
    no longer required (it doesn't hurt either).
    """
    call = run_iter.spawn(
        benchmark=benchmark, rounds=rounds, examples=examples,
        cost_epochs=cost_epochs, policy_iterations=policy_iterations,
        trajectories_per_iter=trajectories_per_iter,
        eval_time_budget=eval_time_budget,
        seed=seed, force_recollect=force_recollect,
    )
    _print_spawn_handle(call, "run_iter")
