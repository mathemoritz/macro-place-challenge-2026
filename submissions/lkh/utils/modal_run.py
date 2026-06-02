"""Run the LKH training pipeline on Modal cloud.

App definition (one ``modal.App``, one ``modal.Image``, one persistent
``modal.Volume``) and four entrypoints — one per training phase plus a
smoke test:

    smoke    : confirm image + mounts + benchmark loading work
    train    : Phase 3 — train CostApproximator
    policy   : Phase 4 — train ChainPolicy via PPO
    iter_    : Phase 5 — full iterative loop (Phase 3 + 4 per round)

Speed: ``iter_`` runs Phase 3 data collection in parallel via Modal's
``.starmap()`` (one container per benchmark, up to 20 concurrent), so the
slow benchmarks (ibm15-18) no longer dominate. Total wall time becomes
max(per-benchmark) instead of sum(per-benchmark), giving ~5× speedup on the
full ICCAD04 sweep.

Quickstart:

    # First time: install Modal and authenticate
    uv add modal
    uv run modal setup

    # Smoke-test the image and mounts (~2-3 minutes)
    uv run modal run submissions/lkh/modal_run.py::smoke

    # Long-running entrypoints use ``.spawn()`` + ``--detach``.
    # ``.spawn()`` makes the call async (immune to local terminal disconnect);
    # ``--detach`` keeps the *app* alive after ``modal run`` exits so the
    # spawned function actually has time to execute. You need both flags.

    # Phase 3 across all 17 benchmarks (per-benchmark example count = N)
    uv run modal run --detach submissions/lkh/modal_run.py::train \\
        --benchmark all --num-examples 300 --epochs 80

    # Phase 4 across all 17 benchmarks, real-scale training
    uv run modal run --detach submissions/lkh/modal_run.py::policy \\
        --benchmark all --iterations 3000

    # Full iterative pipeline (long-running)
    uv run modal run --detach submissions/lkh/modal_run.py::iter_ \\
        --benchmark all --rounds 3 --examples 200 --policy-iterations 1500

After completion, all checkpoints, training data, and visualizations are
persisted to the Modal Volume ``lkh-results``. Download with:

    modal volume get lkh-results /iter/checkpoints ./checkpoints

Reference (CS224R Modal Compute Guide, sections 4-5).

12-hour overnight preset (``iter_12h``) — see ``LONG_RUN_12H`` and ``iter_12h`` below.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import shlex
import shutil
import subprocess
from pathlib import Path

import modal

# ── App, image, volume, mount ──────────────────────────────────────────────

app = modal.App("lkh-macro-place")

# Local repo root (this file is at submissions/lkh/utils/modal_run.py).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


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
        "/.git/",
        "/.venv/",
        "/__pycache__/",
        "/.pytest_cache/",
        "/.idea/",
        "/.vscode/",
        "/.cursor/",
        "/.DS_Store",
        "/vis/",
        "/vis_surrogate/",
        "/agent-transcripts/",
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
        "numpy>=1.20",
        "matplotlib>=3.5",
        "tqdm>=4.65",
        "absl-py>=1.0",
        "wandb>=0.16",
    )
    .pip_install(
        "torch>=2.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
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
    for p in (
        "/root/repo",
        "/root/repo/submissions/lkh",
        "/root/repo/external/MacroPlacement/CodeElements/Plc_client",
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


def _run(cmd: list[str]) -> None:
    """Run a python subprocess from the repo root with PYTHONPATH set."""
    import os

    _setup_env()
    env = os.environ.copy()
    extra = ":".join(
        [
            "/root/repo",
            "/root/repo/submissions/lkh",
            "/root/repo/external/MacroPlacement/CodeElements/Plc_client",
        ]
    )
    env["PYTHONPATH"] = f"{extra}:{env.get('PYTHONPATH', '')}"
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True, cwd="/root/repo")


def _archive_iter_output(output_root: Path) -> Path | None:
    """Copy prior ``/output/iter`` artifacts into ``archives/<UTC-timestamp>/``.

      Called before ``force_recollect`` wipes working caches so old 75-example
    (or partial) collections remain on the volume under ``iter/archives/``.
    """
    per_bench = output_root / "per_bench"
    data_pt = output_root / "data" / "chain_data.pt"
    history = output_root / "history.json"
    ckpt_dir = output_root / "checkpoints"
    round_dirs = sorted(output_root.glob("round_*"))

    has_per_bench = per_bench.exists() and any(per_bench.glob("*.pt"))
    has_other = (
        data_pt.exists()
        or history.exists()
        or (ckpt_dir.exists() and any(ckpt_dir.iterdir()))
        or bool(round_dirs)
    )
    if not has_per_bench and not has_other:
        return None

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = output_root / "archives" / ts
    archive_root.mkdir(parents=True, exist_ok=True)

    if has_per_bench:
        shutil.copytree(per_bench, archive_root / "per_bench", dirs_exist_ok=True)
        n = len(list((archive_root / "per_bench").glob("*.pt")))
        print(f"[Modal] archived {n} per-benchmark cache file(s)")

    if data_pt.exists():
        (archive_root / "data").mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_pt, archive_root / "data" / data_pt.name)

    if history.exists():
        shutil.copy2(history, archive_root / history.name)

    if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
        shutil.copytree(ckpt_dir, archive_root / "checkpoints", dirs_exist_ok=True)

    for rd in round_dirs:
        if rd.is_dir():
            shutil.copytree(rd, archive_root / rd.name, dirs_exist_ok=True)

    print(f"[Modal] archived previous iter output -> {archive_root}")
    return archive_root


def _clear_iter_working_tree(output_root: Path) -> None:
    """Remove active iter artifacts so a forced re-collect starts clean."""
    per_bench = output_root / "per_bench"
    if per_bench.exists():
        for f in per_bench.glob("*.pt"):
            f.unlink()

    data_pt = output_root / "data" / "chain_data.pt"
    if data_pt.exists():
        data_pt.unlink()

    history = output_root / "history.json"
    if history.exists():
        history.unlink()

    ckpt_dir = output_root / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    for rd in output_root.glob("round_*"):
        if rd.is_dir():
            shutil.rmtree(rd)

    for cal in output_root.glob("round_*_calibration.pt"):
        cal.unlink(missing_ok=True)

    print(f"[Modal] cleared working iter tree under {output_root}")


def _persist(label: str) -> None:
    """Copy checkpoints / data / vis from the container to the volume under
    ``/output/<label>/``. Existing files are overwritten."""
    targets = [
        ("submissions/lkh/checkpoints", f"/output/{label}/checkpoints"),
        ("submissions/lkh/data", f"/output/{label}/data"),
        ("vis", f"/output/{label}/vis"),
    ]
    for src, dst in targets:
        src_path = Path("/root/repo") / src
        if src_path.exists() and any(src_path.iterdir()):
            shutil.copytree(str(src_path), dst, dirs_exist_ok=True)
            print(f"  persisted {src} -> {dst}", flush=True)
    volume.commit()


# ── Remote functions ───────────────────────────────────────────────────────


@app.function(image=image, volumes={"/output": volume}, cpu=8.0, memory=8192, timeout=30 * 60)
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
    print(
        f"  loaded {bench.name}: " f"{bench.num_hard_macros} hard / {bench.num_macros} total macros"
    )
    print("OK")


@app.function(
    image=image,
    volumes={"/output": volume},
    cpu=8.0,
    memory=8192,
    timeout=6 * 3600,
    nonpreemptible=True,
)
def run_train(
    benchmark: str, num_examples: int, epochs: int, seed: int, force_recollect: bool
) -> None:
    cmd = [
        "python",
        "submissions/lkh/train.py",
        "--benchmark",
        benchmark,
        "--num-examples",
        str(num_examples),
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
    ]
    if force_recollect:
        cmd.append("--force-recollect")
    _run(cmd)
    _persist("train")


@app.function(
    image=image,
    volumes={"/output": volume},
    gpu="A10G",
    memory=16384,
    timeout=4 * 3600,
    secrets=[modal.Secret.from_name("wandb")],
)
def run_train_encoder(
    benchmarks: str,
    n_samples: int,
    epochs: int,
    seed: int = 42,
    no_cnn: bool = False,
    wandb_project: str = "lkh-encoder",
    patience: int = 50,
) -> None:
    """GPU training only. Assumes per-benchmark caches exist at /output/encoder/per_bench/.

    Called by run_encoder_pipeline after parallel collection completes.
    """
    volume.reload()
    cmd = [
        "python",
        "submissions/lkh/learning/train_encoder.py",
        "--benchmarks",
        benchmarks,
        "--samples",
        str(n_samples),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--seed",
        str(seed),
        "--cache-dir",
        "/output/encoder/per_bench",
        "--skip-collection",
    ]
    if no_cnn:
        cmd.append("--no-cnn")
    if wandb_project:
        cmd += ["--wandb-project", wandb_project]
    _run(cmd)
    _persist("encoder")


@app.function(
    image=image,
    volumes={"/output": volume},
    cpu=8.0,
    memory=8192,
    timeout=12 * 3600,
    nonpreemptible=True,
    secrets=[modal.Secret.from_name("wandb")],
)
def run_policy(
    benchmark: str, iterations: int, trajectories_per_iter: int, seed: int,
    initial_policy: str,
    encoder_ckpt: str = "",
    proj_dim: int = 16,
    gate_mode: str = "hpwl",
    wandb_project: str = "",
    wandb_run_name: str = "",
) -> None:
    cmd = [
        "python",
        "submissions/lkh/learning/train_policy.py",
        "--benchmark",
        benchmark,
        "--iterations",
        str(iterations),
        "--trajectories-per-iter",
        str(trajectories_per_iter),
        "--seed",
        str(seed),
    ]
    if initial_policy:
        cmd += ["--initial-policy", initial_policy]
    if encoder_ckpt:
        cmd += ["--encoder-ckpt", encoder_ckpt, "--proj-dim", str(proj_dim),
                "--gate-mode", gate_mode]
    if wandb_project:
        cmd += ["--wandb-project", wandb_project]
    if wandb_run_name:
        cmd += ["--wandb-run-name", wandb_run_name]
    _run(cmd)
    _persist("policy")


# ---------------------------------------------------------------------------
# Parallel per-benchmark data collection.
#
# ``compute_proxy_cost`` is single-threaded under the Python GIL, so a single
# container can only collect one benchmark at a time even with multiple CPUs.
# This function does exactly one benchmark per container; ``run_iter`` invokes
# it as ``.starmap()`` to collect all 17 benchmarks concurrently.
#
# Wall time becomes max(per-benchmark) instead of sum, which is ~5× faster on
# the full ICCAD04 sweep (3 h instead of 15 h). Each container writes its
# benchmark's cache to /output/iter/per_bench/<name>.pt and commits the volume
# so the next stage (Phase 3 training + PPO) sees a fully populated cache.
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={"/output": volume},
    cpu=2.0,
    memory=4096,
    timeout=8 * 3600,
    nonpreemptible=True,
    max_containers=20,
)
def collect_one_benchmark_modal(name: str, num_examples: int, seed: int) -> dict:
    import time as _time

    _setup_env()
    import torch as _torch
    import submissions.lkh.learning.train as _train

    cache_dir = Path("/output/iter/per_bench")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{name}.pt"

    # Idempotency: if someone managed to commit this benchmark already
    # (e.g. a prior partial run), skip without recomputing.
    if cache_file.exists():
        try:
            d = _torch.load(str(cache_file), weights_only=False)
            return {"name": name, "n": int(len(d["features"])), "wall_s": 0.0, "from_cache": True}
        except Exception:
            print(f"[{name}] cache exists but unreadable; re-collecting")

    t0 = _time.time()
    print(f"[{name}] collecting {num_examples} examples...")
    # _collect_one_benchmark signature: (benchmark_name, num_examples, seed, drift_prob).
    # The outer `collect_data` renames this to `num_examples_per_benchmark`;
    # don't conflate them here or Modal raises TypeError on the worker.
    feats, targets = _train._collect_one_benchmark(
        name,
        num_examples=num_examples,
        seed=seed,
    )
    wall = _time.time() - t0

    tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
    _torch.save(
        {"features": feats, "targets": targets, "num_examples": num_examples, "name": name},
        str(tmp),
    )
    tmp.replace(cache_file)
    volume.commit()
    print(
        f"[{name}] done: {len(feats)} examples in "
        f"{_train._fmt_time(wall)} ({_train._fmt_rate(len(feats), wall)})"
    )
    return {"name": name, "n": int(len(feats)), "wall_s": float(wall), "from_cache": False}


# ── Encoder collection + training pipeline ────────────────────────────────
#
# Data collection is 100 % CPU-bound (PlacementCost C++ backend). Running it
# on the GPU container wastes expensive A10G time. Instead:
#   1. collect_encoder_benchmark: one cheap CPU container per benchmark,
#      all running in parallel via .starmap() in run_encoder_pipeline.
#   2. run_train_encoder: GPU-only training, starts after all caches are committed.
#   3. run_encoder_pipeline: orchestrator that sequences 1 → 2.
#
# Wall time ≈ max(per-benchmark collection) + training
#            ≈ ~60 min collection + ~20–30 min training  (full 17-bench run)
# vs ~6 h if collection ran sequentially on the GPU box.


@app.function(
    image=image,
    volumes={"/output": volume},
    cpu=2.0,
    memory=4096,
    timeout=8 * 3600,
    max_containers=20,
)
def collect_encoder_benchmark(
    name: str, n_samples: int, seed: int, grid_size: int, force_recollect: bool
) -> dict:
    """Collect proxy-cost samples for one benchmark. CPU-only, runs in parallel."""
    import time as _time
    import torch as _torch

    cache_dir = Path("/output/encoder/per_bench")
    cache_file = cache_dir / f"{name}.pt"

    if not force_recollect and cache_file.exists():
        try:
            d = _torch.load(str(cache_file), weights_only=False)
            if d["node_feats"].shape[0] >= n_samples:
                return {
                    "name": name,
                    "n": int(d["node_feats"].shape[0]),
                    "wall_s": 0.0,
                    "from_cache": True,
                }
        except Exception:
            print(f"[{name}] cache unreadable; re-collecting")

    t0 = _time.time()
    _run(
        [
            "python",
            "submissions/lkh/learning/train_encoder.py",
            "--benchmarks",
            name,
            "--samples",
            str(n_samples),
            "--seed",
            str(seed),
            "--grid-size",
            str(grid_size),
            "--cache-dir",
            str(cache_dir),
            "--collection-only",
        ]
    )
    wall = _time.time() - t0
    volume.commit()
    return {"name": name, "n": n_samples, "wall_s": float(wall), "from_cache": False}


@app.function(image=image, volumes={"/output": volume}, cpu=2.0, memory=4096, timeout=12 * 3600)
def run_encoder_pipeline(
    benchmarks: str,
    n_samples: int,
    epochs: int,
    seed: int = 42,
    force_recollect: bool = False,
    no_cnn: bool = False,
    wandb_project: str = "lkh-encoder",
    grid_size: int = 128,
    patience: int = 50,
) -> None:
    """Orchestrator: parallel CPU collection → single GPU training pass."""
    _setup_env()

    if benchmarks == "all":
        from macro_place.evaluate import IBM_BENCHMARKS

        benchmark_list = list(IBM_BENCHMARKS)
    else:
        benchmark_list = [n.strip() for n in benchmarks.split(",") if n.strip()]

    volume.reload()
    cache_dir = Path("/output/encoder/per_bench")

    def _cache_sufficient(name: str) -> bool:
        import torch as _torch

        p = cache_dir / f"{name}.pt"
        if not p.exists():
            return False
        try:
            d = _torch.load(str(p), weights_only=False)
            return int(d["node_feats"].shape[0]) >= n_samples
        except Exception:
            return False

    todo = (
        benchmark_list
        if force_recollect
        else [n for n in benchmark_list if not _cache_sufficient(n)]
    )

    if todo:
        configs = [
            (n, n_samples, seed + i * 7919, grid_size, force_recollect) for i, n in enumerate(todo)
        ]
        n_cached = len(benchmark_list) - len(todo)
        print(
            f"[encoder] collecting {len(todo)} benchmarks in parallel "
            f"({n_cached} already cached)"
        )
        for res in collect_encoder_benchmark.starmap(configs):
            tag = "cached" if res.get("from_cache") else "collected"
            wall = res.get("wall_s", 0.0)
            wall_str = f"{wall:.0f}s" if wall < 60 else f"{wall / 60:.1f}m"
            print(f"  {tag}: {res['name']:>6}  n={res['n']:>4}  wall={wall_str}")
        volume.reload()
    else:
        print(f"[encoder] all {len(benchmark_list)} benchmarks cached — skipping collection")

    print("[encoder] handing off to GPU training...")
    run_train_encoder.remote(
        benchmarks=benchmarks,
        n_samples=n_samples,
        epochs=epochs,
        seed=seed,
        no_cnn=no_cnn,
        wandb_project=wandb_project,
        patience=patience,
    )


# Budgeted for ~12 h wall time on 17 IBM benchmarks (parallel collection +
# 5 iterative rounds). Scales roughly linearly in ``examples`` and in
# ``policy_iterations * trajectories_per_iter``.
LONG_RUN_12H: dict[str, Any] = {
    "benchmark": "all",
    "rounds": 5,
    "examples": 240,
    "cost_epochs": 100,
    "policy_iterations": 3500,
    "trajectories_per_iter": 16,
    "eval_time_budget": 60.0,
    "calibration_samples_per_bench": 75,
    "calibration_time_budget_s": 15.0,
    "seed": 42,
    "force_recollect": False,
    "output_tag": "iter",
    "cache_read_dir": "",
    "skip_collection": False,
}

MEDIUM_RUN_4H: dict[str, Any] = {
    "benchmark": "all",
    "rounds": 3,
    "examples": 240,
    "cost_epochs": 50,
    "policy_iterations": 1000,
    "trajectories_per_iter": 4,
    "eval_time_budget": 25.0,
    "calibration_samples_per_bench": 25,
    "calibration_time_budget_s": 10.0,
    "seed": 43,
    "force_recollect": False,
    "output_tag": "iter_medium",
    "cache_read_dir": "/output/iter/per_bench",
    "skip_collection": True,
}


def _run_iter_impl(
    benchmark: str,
    rounds: int,
    examples: int,
    cost_epochs: int,
    policy_iterations: int,
    trajectories_per_iter: int,
    eval_time_budget: float,
    seed: int,
    force_recollect: bool,
    calibration_samples_per_bench: int,
    calibration_time_budget_s: float,
    output_tag: str,
    cache_read_dir: str,
    skip_collection: bool,
) -> None:
    import json as _json

    _setup_env()
    import submissions.lkh.learning.train as _train
    import train_iter as _ti

    benchmarks = _train.parse_benchmarks(benchmark)
    output_root = Path(f"/output/{output_tag}")
    output_root.mkdir(parents=True, exist_ok=True)

    if cache_read_dir:
        per_benchmark_cache_dir = Path(cache_read_dir)
        print(f"[Modal] cache read:  {per_benchmark_cache_dir}")
        print(f"[Modal] cache write: {output_root}/ (not main iter/)")
    else:
        per_benchmark_cache_dir = output_root / "per_bench"
        per_benchmark_cache_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_root / "data" / "chain_data.pt"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    stale_chain = Path("/root/repo/submissions/lkh/data/chain_data.pt")
    if stale_chain.exists() and stale_chain != data_path:
        stale_chain.unlink()
        print(f"[Modal] removed stale {stale_chain}")
    cost_ckpt = Path("/root/repo/submissions/lkh/checkpoints/cost_approximator.pt")
    policy_ckpt = Path("/root/repo/submissions/lkh/checkpoints/chain_policy.pt")

    volume.reload()

    if skip_collection:
        missing = [n for n in benchmarks if not (per_benchmark_cache_dir / f"{n}.pt").exists()]
        if missing:
            raise RuntimeError(
                f"skip_collection: missing caches {missing} under {per_benchmark_cache_dir}"
            )
        print(f"[Modal] skip_collection: {len(benchmarks)} caches OK.")
    else:
        if force_recollect:
            archived = _archive_iter_output(output_root)
            if archived is not None:
                volume.commit()
            _clear_iter_working_tree(output_root)
            volume.commit()
            print(f"[Modal] force_recollect: cleared {output_root}")
        todo = [n for n in benchmarks if not (per_benchmark_cache_dir / f"{n}.pt").exists()]
        if todo:
            configs = [(n, examples, seed + i * 7919) for i, n in enumerate(todo)]
            print(f"[Modal] parallel collection: {len(todo)} benchmarks")
            for res in collect_one_benchmark_modal.starmap(configs):
                tag = "cached" if res.get("from_cache") else "collected"
                print(
                    f"  {tag}: {res['name']:>6}  n={res['n']:>4}  "
                    f"wall={_train._fmt_time(res.get('wall_s', 0.0))}"
                )
            volume.reload()
        else:
            print(f"[Modal] all {len(benchmarks)} benchmarks already cached.")

    print(f"=== Phase 5 on Modal ({output_tag}) ===")
    print(f"  benchmarks={benchmarks}  rounds={rounds}  output={output_root}")
    print(
        f"  examples/bench={examples}  cost_epochs={cost_epochs}  "
        f"policy_iters={policy_iterations}  traj/iter={trajectories_per_iter}"
    )
    print(
        f"  eval_time_budget={eval_time_budget}s  "
        f"calibration={calibration_samples_per_bench}/bench @ "
        f"{calibration_time_budget_s}s"
    )
    print(f"  per-benchmark cache: {per_benchmark_cache_dir}")

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
            force_recollect=False,
            eval_time_budget_s=eval_time_budget,
            output_root=output_root,
            per_benchmark_cache_dir=per_benchmark_cache_dir,
            calibration_samples_per_bench=calibration_samples_per_bench,
            calibration_time_budget_s=calibration_time_budget_s,
        )
        history.append(record)
        (output_root / "history.json").write_text(_json.dumps(history, indent=2, default=str))
        _persist(output_tag)
        volume.commit()
        print(f"[Modal] round {r} committed ({len(history)} rounds done).")


@app.function(
    image=image,
    volumes={"/output": volume},
    cpu=8.0,
    memory=16384,
    timeout=24 * 3600,
    nonpreemptible=True,
)
def run_iter(
    benchmark: str,
    rounds: int,
    examples: int,
    cost_epochs: int,
    policy_iterations: int,
    trajectories_per_iter: int,
    eval_time_budget: float,
    seed: int,
    force_recollect: bool,
    calibration_samples_per_bench: int = 50,
    calibration_time_budget_s: float = 10.0,
    output_tag: str = "iter",
    cache_read_dir: str = "",
    skip_collection: bool = False,
) -> None:
    _run_iter_impl(
        benchmark,
        rounds,
        examples,
        cost_epochs,
        policy_iterations,
        trajectories_per_iter,
        eval_time_budget,
        seed,
        force_recollect,
        calibration_samples_per_bench,
        calibration_time_budget_s,
        output_tag,
        cache_read_dir,
        skip_collection,
    )


@app.function(
    image=image,
    volumes={"/output": volume},
    cpu=8.0,
    memory=16384,
    timeout=4 * 3600,
    nonpreemptible=True,
)
def run_iter_medium(
    benchmark: str,
    rounds: int,
    examples: int,
    cost_epochs: int,
    policy_iterations: int,
    trajectories_per_iter: int,
    eval_time_budget: float,
    seed: int,
    force_recollect: bool,
    calibration_samples_per_bench: int,
    calibration_time_budget_s: float,
    output_tag: str,
    cache_read_dir: str,
    skip_collection: bool,
) -> None:
    _run_iter_impl(
        benchmark,
        rounds,
        examples,
        cost_epochs,
        policy_iterations,
        trajectories_per_iter,
        eval_time_budget,
        seed,
        force_recollect,
        calibration_samples_per_bench,
        calibration_time_budget_s,
        output_tag,
        cache_read_dir,
        skip_collection,
    )


# ── Local entrypoints ──────────────────────────────────────────────────────


@app.local_entrypoint()
def smoke() -> None:
    """Smoke-test the Modal setup (~2-3 minutes the first time).

    Uses ``.remote()`` because we *want* to wait for the result locally
    (it's a fast sanity check).
    """
    run_smoke.remote()


# ---------------------------------------------------------------------------
# Long-running entrypoints use ``.spawn()`` + ``--detach`` together.
#
# ``.remote()`` blocks until the function returns, and the call is cancelled
# when the local terminal disconnects — even with ``--detach``. ``.spawn()``
# dispatches the work and returns a FunctionCall handle immediately, so the
# call survives the disconnect; ``--detach`` keeps the App alive after the
# CLI exits so the spawned function actually gets to run. Without both,
# the App tears itself down before the spawned function starts (the
# dashboard then shows the App as "stopped" with 0 calls).
# ---------------------------------------------------------------------------


def _spawn_iter(fn: Any, label: str, **kwargs: Any) -> None:
    """Dispatch a Phase-5 Modal function and print hyperparameters."""
    call = fn.spawn(**kwargs)
    _print_spawn_handle(call, label)
    print("\n--- Hyperparameters ---")
    for key in (
        "benchmark",
        "rounds",
        "examples",
        "cost_epochs",
        "policy_iterations",
        "trajectories_per_iter",
        "eval_time_budget",
        "calibration_samples_per_bench",
        "calibration_time_budget_s",
        "seed",
        "force_recollect",
        "output_tag",
        "cache_read_dir",
        "skip_collection",
    ):
        if key in kwargs:
            print(f"  {key}: {kwargs[key]}")


def _print_spawn_handle(call, label: str) -> None:
    """Print the FunctionCall ID + the commands to monitor / cancel.

    NOTE: Modal 1.x has no ``modal call`` CLI subcommand — function calls are
    only accessible programmatically via the Python SDK. App-level commands
    are the practical way to watch / stop a single-call app like ours.
    """
    fid = getattr(call, "object_id", None) or str(call)
    print(f"\nSpawned {label} on Modal.")
    print(f"  function_call_id: {fid}      (handle, mostly informational)")
    print(f"  see status:  modal app list")
    print(f"  stream logs: modal app logs <APP_ID> -f   # ID from app list (e.g. ap-...)")
    print(f"  stop:        modal app stop <APP_ID>")
    print(f"  dashboard:   https://modal.com/apps")


@app.local_entrypoint()
def train(
    benchmark: str = "ibm01",
    num_examples: int = 1500,
    epochs: int = 60,
    seed: int = 42,
    force_recollect: bool = False,
) -> None:
    """Phase 3 — train CostApproximator on Modal (background spawn).

    Examples:
        modal run modal_run.py::train --benchmark ibm01,ibm02 --num-examples 1500
        modal run modal_run.py::train --benchmark all --num-examples 300
    """
    call = run_train.spawn(
        benchmark=benchmark,
        num_examples=num_examples,
        epochs=epochs,
        seed=seed,
        force_recollect=force_recollect,
    )
    _print_spawn_handle(call, "run_train")


@app.local_entrypoint()
def encoder(
    benchmarks: str = "all",
    n_samples: int = 200,
    epochs: int = 300,
    seed: int = 42,
    force_recollect: bool = False,
    no_cnn: bool = False,
    wandb_project: str = "lkh-encoder",
    patience: int = 50,
) -> None:
    """Phase 2.5 — parallel CPU collection + GPU encoder training.

    Collection runs one cheap CPU container per benchmark (up to 20 concurrent);
    training starts on a single A10G once all caches are committed.
    Wall time ≈ max(per-benchmark) + training, not sum(per-benchmark).

    Early stopping is based on val_loss (more stable than Pearson r with small
    val sets). Default patience=50 gives the model room to escape plateaus.

    Examples:
        # Smoke test (reuses cached data from previous run)
        uv run modal run --detach submissions/lkh/modal_run.py::encoder \\
            --benchmarks ibm01,ibm02,ibm07 --n-samples 20 --epochs 20

        # Full run, all 17 benchmarks
        uv run modal run --detach submissions/lkh/modal_run.py::encoder \\
            --benchmarks all --n-samples 200 --epochs 300

    Download checkpoint (works mid-run too — best weights are saved immediately):
        modal volume get lkh-results /encoder/checkpoints/proxy_cost_encoder.pt \\
            submissions/lkh/checkpoints/
    """
    call = run_encoder_pipeline.spawn(
        benchmarks=benchmarks,
        n_samples=n_samples,
        epochs=epochs,
        seed=seed,
        force_recollect=force_recollect,
        no_cnn=no_cnn,
        wandb_project=wandb_project,
        patience=patience,
    )
    _print_spawn_handle(call, "run_encoder_pipeline")


@app.local_entrypoint()
def policy(
    benchmark: str = "ibm01",
    iterations: int = 1000,
    trajectories_per_iter: int = 16,
    seed: int = 42,
    initial_policy: str = "",
    encoder_ckpt: str = "",
    proj_dim: int = 16,
    gate_mode: str = "hpwl",
    wandb_project: str = "",
    wandb_run_name: str = "",
) -> None:
    """Phase 4 — train ChainPolicy via PPO on Modal (background spawn).

    Examples:
        modal run modal_run.py::policy --benchmark all --iterations 3000
        modal run modal_run.py::policy --benchmark ibm01,ibm07 --iterations 1000
        modal run --detach modal_run.py::policy --benchmark all --iterations 3000 \\
            --encoder-ckpt /output/encoder/checkpoints/proxy_cost_encoder.pt
    """
    call = run_policy.spawn(
        benchmark=benchmark,
        iterations=iterations,
        trajectories_per_iter=trajectories_per_iter,
        seed=seed,
        initial_policy=initial_policy,
        encoder_ckpt=encoder_ckpt,
        proj_dim=proj_dim,
        gate_mode=gate_mode,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    _print_spawn_handle(call, "run_policy")


@app.local_entrypoint()
def iter_(
    benchmark: str = "ibm01",
    rounds: int = 2,
    examples: int = 400,
    cost_epochs: int = 60,
    policy_iterations: int = 1000,
    trajectories_per_iter: int = 4,
    eval_time_budget: float = 20.0,
    calibration_samples_per_bench: int = 50,
    calibration_time_budget_s: float = 10.0,
    seed: int = 42,
    force_recollect: bool = False,
) -> None:
    """Phase 5 — iterative training loop, dispatched as a background job.

    Examples:
        # Quick: 1 round, 3 benchmarks, modest data
        modal run --detach modal_run.py::iter_ --benchmark ibm01,ibm02,ibm07 \\
            --rounds 1 --examples 200 --policy-iterations 500

        # Full: all benchmarks, multiple rounds
        modal run --detach modal_run.py::iter_ --benchmark all \\
            --rounds 3 --examples 200 --policy-iterations 1500

        # ~12 h overnight preset (see ``iter_12h``)
        modal run --detach modal_run.py::iter_12h --force-recollect

    Always launch with ``--detach``: ``.spawn()`` makes the call asynchronous
    but the App itself shuts down when ``modal run`` exits, taking the
    spawned function with it. ``--detach`` keeps the App alive so the spawn
    can actually execute.
    """
    _spawn_iter(
        run_iter,
        "run_iter",
        benchmark=benchmark,
        rounds=rounds,
        examples=examples,
        cost_epochs=cost_epochs,
        policy_iterations=policy_iterations,
        trajectories_per_iter=trajectories_per_iter,
        eval_time_budget=eval_time_budget,
        calibration_samples_per_bench=calibration_samples_per_bench,
        calibration_time_budget_s=calibration_time_budget_s,
        seed=seed,
        force_recollect=force_recollect,
        output_tag="iter",
        cache_read_dir="",
        skip_collection=False,
    )


@app.local_entrypoint()
def iter_12h(force_recollect: bool = False, seed: int = 42) -> None:
    """~12-hour overnight training preset (17 benchmarks, parallel collection).

    Wall-time budget (typical):
        - Parallel data collection @ 240 examples/bench  ~7 h
        - 5 rounds × (PPO + eval + calibration + cost train)  ~4.5 h
        - Total  ~11–12 h

    First run on a volume that has an older sweep (e.g. 75-example caches),
    pass ``--force-recollect``. Prior artifacts are copied to
    ``lkh-results/iter/archives/<UTC-timestamp>/`` on the volume, then the
    working tree is cleared and rebuilt at 240 examples/bench::

        uv run modal run --detach submissions/lkh/modal_run.py::iter_12h \\
            --force-recollect

    Re-runs **without** ``--force-recollect`` reuse existing ``per_bench/*.pt``
    (must match the target example count) and finish in ~4–5 h (rounds only).

    Tuning vs this preset: use ``iter_`` with explicit flags, or edit
    ``LONG_RUN_12H`` in this file.
    """
    params = {**LONG_RUN_12H, "force_recollect": force_recollect, "seed": seed}
    print("=== iter_12h preset (~12 h target) ===")
    _spawn_iter(run_iter, "run_iter", **params)


@app.local_entrypoint()
def iter_medium(seed: int = 43) -> None:
    """~4 h parallel-safe training (separate ``/output/iter_medium/`` tree).

    Reuses ``/output/iter/per_bench/*.pt`` from the main run (no collection).
    Does **not** touch ``/output/iter/checkpoints`` or ``history.json``.
    Safe to run while ``iter_12h`` is active on the same volume.

        uv run modal run --detach submissions/lkh/modal_run.py::iter_medium

    After completion::

        modal volume get lkh-results /iter_medium ./modal_output_medium
        cp modal_output_medium/checkpoints/*.pt submissions/lkh/checkpoints_medium/
    """
    params = {**MEDIUM_RUN_4H, "seed": seed}
    print("=== iter_medium preset (<= 4 h, reads iter/per_bench) ===")
    _spawn_iter(run_iter_medium, "run_iter_medium", **params)
