"""Run the LKH training pipeline on Modal cloud.

App definition (one ``modal.App``, one ``modal.Image``, one persistent
``modal.Volume``) and four entrypoints — one per training stage plus a
smoke test:

    smoke    : confirm image + mounts + benchmark loading work
    train    : train CostApproximator
    policy   : train ChainPolicy via PPO
    iter_    : full iterative loop (cost-approximator + policy per round)

Speed: ``iter_`` runs data collection in parallel via Modal's
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

    # Cost-approximator training across all 17 benchmarks (per-benchmark example count = N)
    uv run modal run --detach submissions/lkh/modal_run.py::train \\
        --benchmark all --num-examples 300 --epochs 80

    # Policy training across all 17 benchmarks, real-scale training
    uv run modal run --detach submissions/lkh/modal_run.py::policy \\
        --benchmark all --iterations 3000

    # Full iterative pipeline (long-running)
    uv run modal run --detach submissions/lkh/modal_run.py::iter_ \\
        --benchmark all --rounds 3 --examples 200 --policy-iterations 1500

After completion, all checkpoints, training data, and visualizations are
persisted to the Modal Volume ``lkh-results``. Download with:

    modal volume get lkh-results /iter/checkpoints ./checkpoints

Iterative presets (``ITER_PRESETS``) and ``iter_ --preset <name>`` for parallel runs
with different hyperparameters / output trees.
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
              cpu=8.0, memory=8192, timeout=6 * 3600,
              nonpreemptible=True)
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
              cpu=8.0, memory=8192, timeout=12 * 3600,
              nonpreemptible=True)
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
# so the next stage (cost-approximator training + PPO) sees a fully populated cache.
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={"/output": volume},
              cpu=2.0, memory=4096, timeout=8 * 3600,
              nonpreemptible=True, max_containers=20)
def collect_one_benchmark_modal(name: str, num_examples: int, seed: int) -> dict:
    import time as _time
    _setup_env()
    import torch as _torch
    import train as _train

    cache_dir = Path("/output/iter/per_bench")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{name}.pt"

    # Idempotency: if someone managed to commit this benchmark already
    # (e.g. a prior partial run), skip without recomputing.
    if cache_file.exists():
        try:
            d = _torch.load(str(cache_file), weights_only=False)
            return {"name": name, "n": int(len(d["features"])),
                    "wall_s": 0.0, "from_cache": True}
        except Exception:
            print(f"[{name}] cache exists but unreadable; re-collecting")

    t0 = _time.time()
    print(f"[{name}] collecting {num_examples} examples...")
    # _collect_one_benchmark signature: (benchmark_name, num_examples, seed, drift_prob).
    # The outer `collect_data` renames this to `num_examples_per_benchmark`;
    # don't conflate them here or Modal raises TypeError on the worker.
    feats, targets = _train._collect_one_benchmark(
        name, num_examples=num_examples, seed=seed,
    )
    wall = _time.time() - t0

    tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
    _torch.save({"features": feats, "targets": targets,
                 "num_examples": num_examples, "name": name},
                str(tmp))
    tmp.replace(cache_file)
    volume.commit()
    print(f"[{name}] done: {len(feats)} examples in "
          f"{_train._fmt_time(wall)} ({_train._fmt_rate(len(feats), wall)})")
    return {"name": name, "n": int(len(feats)),
            "wall_s": float(wall), "from_cache": False}


# ---------------------------------------------------------------------------
# Multi-seed inference overfit (no training). Each container runs ablate.py
# once on (benchmark, seed) with a fat per-seed time budget; the orchestrator
# fans out via .starmap() and takes the lex-best across seeds.
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={"/output": volume},
              cpu=2.0, memory=4096, timeout=2 * 3600,
              max_containers=100)
def run_one_overfit_seed(benchmark: str, seed: int, time_budget_s: float,
                         max_chains: int, max_chain_length: int,
                         reg_weight: float, use_wiremask: bool,
                         use_position_mask: bool, milestone: str,
                         ckpt_source: str) -> dict:
    """One (benchmark, seed) inference run via ablate.py.

    If ``ckpt_source`` is non-empty, files under that volume path are copied
    into ``submissions/lkh/checkpoints/`` so the placer loads our best
    cost-approximator / chain-policy instead of the repo defaults.
    """
    import json
    import time as _t
    from pathlib import Path
    _setup_env()

    if ckpt_source:
        src = Path(ckpt_source)
        dst = Path("/root/repo/submissions/lkh/checkpoints")
        dst.mkdir(parents=True, exist_ok=True)
        for fname in ("cost_approximator.pt", "chain_policy.pt"):
            sf = src / fname
            if sf.exists():
                shutil.copy2(sf, dst / fname)
                print(f"[seed {seed}] using {sf}")

    history_path = Path(f"/tmp/overfit_{benchmark}_seed{seed}.json")
    if history_path.exists():
        history_path.unlink()

    cmd = ["python", "submissions/lkh/ablate.py",
           "--benchmark", benchmark,
           "--seed", str(seed),
           "--time-budget", str(time_budget_s),
           "--max-chains", str(max_chains),
           "--max-chain-length", str(max_chain_length),
           "--milestone", milestone,
           "--history-file", str(history_path)]
    if reg_weight > 0:
        cmd += ["--reg-weight", str(reg_weight)]
    if use_wiremask:
        cmd += ["--use-wiremask"]
    if use_position_mask:
        cmd += ["--use-position-mask"]

    t0 = _t.time()
    _run(cmd)
    wall = _t.time() - t0

    history = json.loads(history_path.read_text())
    record = history[-1] if isinstance(history, list) else history
    per_bench = record["per_benchmark"][benchmark]
    return {"seed": seed, "wall_s": wall, **per_bench}


@app.function(image=image, volumes={"/output": volume},
              cpu=2.0, memory=4096, timeout=12 * 3600,
              nonpreemptible=True)
def run_overfit_sweep_all(benchmarks: list, n_seeds: int, base_seed: int,
                          time_budget_s: float, max_chains: int,
                          max_chain_length: int, reg_weight: float,
                          use_wiremask: bool, use_position_mask: bool,
                          milestone: str, output_tag: str,
                          ckpt_source: str) -> None:
    """All-benchmarks multi-seed overfit. Fans out (bench × seed), lex-bests per bench."""
    import json
    from pathlib import Path
    from collections import defaultdict

    out_dir = Path(f"/output/{output_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for bench in benchmarks:
        for i in range(n_seeds):
            configs.append((bench, base_seed + i, time_budget_s, max_chains,
                            max_chain_length, reg_weight, use_wiremask,
                            use_position_mask, milestone, ckpt_source))

    print(f"=== overfit sweep across {len(benchmarks)} benches × {n_seeds} seeds "
          f"= {len(configs)} containers ===")
    print(f"  time_budget={time_budget_s}s  milestone={milestone}  "
          f"wm={use_wiremask}  pm={use_position_mask}  reg={reg_weight}")
    print(f"  ckpt_source={ckpt_source or '(repo default)'}")

    # Re-map results to (bench, seed) -> record. starmap returns in input order,
    # so we zip back with configs.
    all_results: dict[str, list[dict]] = defaultdict(list)
    n_done = 0
    for cfg, res in zip(configs, run_one_overfit_seed.starmap(configs)):
        bench = cfg[0]
        res = dict(res)
        res["benchmark"] = bench
        all_results[bench].append(res)
        n_done += 1
        if n_done % 10 == 0 or n_done == len(configs):
            (out_dir / "all_seeds.json").write_text(
                json.dumps({b: rs for b, rs in all_results.items()}, indent=2))
            volume.commit()
            print(f"  [{n_done}/{len(configs)}] {bench} seed={res['seed']} "
                  f"proxy={res['proxy']:.4f} overlaps={res.get('overlaps', 0)}")

    # Final per-bench best (lex by overlaps then proxy)
    per_bench_best = {}
    for bench, rs in all_results.items():
        b = min(rs, key=lambda r: (r.get("overlaps", 0), r["proxy"]))
        per_bench_best[bench] = b

    summary = {
        "best_per_benchmark": per_bench_best,
        "n_seeds_per_benchmark": n_seeds,
        "config": {
            "time_budget_s": time_budget_s,
            "max_chains": max_chains,
            "max_chain_length": max_chain_length,
            "reg_weight": reg_weight,
            "use_wiremask": use_wiremask,
            "use_position_mask": use_position_mask,
            "milestone": milestone,
            "ckpt_source": ckpt_source,
        },
    }
    (out_dir / "best_per_bench.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "all_seeds.json").write_text(
        json.dumps({b: rs for b, rs in all_results.items()}, indent=2))
    volume.commit()

    print(f"\n=== best per benchmark ===")
    for bench in sorted(per_bench_best):
        b = per_bench_best[bench]
        print(f"  {bench}: seed={b['seed']:>3}  proxy={b['proxy']:.4f}  "
              f"overlaps={b.get('overlaps', 0):>3}  wall={b['wall_s']:.1f}s")


@app.function(image=image, volumes={"/output": volume},
              cpu=2.0, memory=4096, timeout=12 * 3600,
              nonpreemptible=True)
def run_overfit_sweep(benchmark: str, n_seeds: int, base_seed: int,
                      time_budget_s: float, max_chains: int,
                      max_chain_length: int, reg_weight: float,
                      use_wiremask: bool, use_position_mask: bool,
                      milestone: str, output_tag: str,
                      ckpt_source: str) -> None:
    """Orchestrator: fan out N seeds, write seeds.json + a brief summary."""
    import json
    from pathlib import Path

    out_dir = Path(f"/output/{output_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [(benchmark, base_seed + i, time_budget_s, max_chains,
                max_chain_length, reg_weight, use_wiremask,
                use_position_mask, milestone, ckpt_source)
               for i in range(n_seeds)]

    print(f"=== overfit sweep on {benchmark} ===")
    print(f"  {n_seeds} seeds @ {time_budget_s}s each, milestone={milestone}, "
          f"wm={use_wiremask}, pm={use_position_mask}, reg={reg_weight}")
    print(f"  ckpt_source={ckpt_source or '(repo default)'}")
    print(f"  -> {out_dir}/seeds.json")

    results: list[dict] = []
    for res in run_one_overfit_seed.starmap(configs):
        results.append(res)
        (out_dir / "seeds.json").write_text(json.dumps(results, indent=2))
        volume.commit()
        print(f"  seed={res['seed']:>4}  proxy={res['proxy']:.4f}  "
              f"overlaps={res.get('overlaps', 0):>3}  "
              f"wall={res['wall_s']:.1f}s")

    best = min(results,
               key=lambda r: (r.get("overlaps", 0), r["proxy"]))
    print(f"\n=== best of {len(results)} seeds ===")
    print(f"  seed={best['seed']}  proxy={best['proxy']:.4f}  "
          f"overlaps={best.get('overlaps', 0)}")
    (out_dir / "best.json").write_text(json.dumps(best, indent=2))
    volume.commit()


# Named hyperparameter bundles for ``iter_ --preset <name>``. Override any field
# on the CLI (e.g. ``--output-tag iter_exp2``). Use distinct ``output_tag`` values
# when launching parallel jobs on the same volume.
ITER_PRESETS: dict[str, dict[str, Any]] = {
    "long12h": {
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
    },
    "medium4h": {
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
    },
}

_DEFAULT_ITER_KWARGS: dict[str, Any] = {
    "benchmark": "ibm01",
    "rounds": 2,
    "examples": 400,
    "cost_epochs": 60,
    "policy_iterations": 1000,
    "trajectories_per_iter": 4,
    "eval_time_budget": 20.0,
    "calibration_samples_per_bench": 50,
    "calibration_time_budget_s": 10.0,
    "seed": 42,
    "force_recollect": False,
    "output_tag": "iter",
    "cache_read_dir": "",
    "skip_collection": False,
}


def _resolve_iter_kwargs(
    *,
    preset: str = "",
    benchmark: str = "",
    rounds: int = 0,
    examples: int = 0,
    cost_epochs: int = 0,
    policy_iterations: int = 0,
    trajectories_per_iter: int = 0,
    eval_time_budget: float = 0.0,
    calibration_samples_per_bench: int = 0,
    calibration_time_budget_s: float = 0.0,
    seed: int = 0,
    force_recollect: bool = False,
    output_tag: str = "",
    cache_read_dir: str = "",
    skip_collection: bool = False,
) -> dict[str, Any]:
    """Merge defaults, optional preset, then explicit CLI overrides (non-zero / non-empty)."""
    if preset and preset not in ITER_PRESETS:
        raise ValueError(
            f"unknown preset {preset!r}; choose from {sorted(ITER_PRESETS)}"
        )
    out = dict(_DEFAULT_ITER_KWARGS)
    if preset:
        out.update(ITER_PRESETS[preset])
    if benchmark:
        out["benchmark"] = benchmark
    if rounds > 0:
        out["rounds"] = rounds
    if examples > 0:
        out["examples"] = examples
    if cost_epochs > 0:
        out["cost_epochs"] = cost_epochs
    if policy_iterations > 0:
        out["policy_iterations"] = policy_iterations
    if trajectories_per_iter > 0:
        out["trajectories_per_iter"] = trajectories_per_iter
    if eval_time_budget > 0:
        out["eval_time_budget"] = eval_time_budget
    if calibration_samples_per_bench > 0:
        out["calibration_samples_per_bench"] = calibration_samples_per_bench
    if calibration_time_budget_s > 0:
        out["calibration_time_budget_s"] = calibration_time_budget_s
    if seed != 0:
        out["seed"] = seed
    if output_tag:
        out["output_tag"] = output_tag
    if cache_read_dir:
        out["cache_read_dir"] = cache_read_dir
    if force_recollect:
        out["force_recollect"] = True
    if skip_collection:
        out["skip_collection"] = True
    return out


def _run_iter_impl(benchmark: str, rounds: int, examples: int, cost_epochs: int,
                   policy_iterations: int, trajectories_per_iter: int,
                   eval_time_budget: float, seed: int, force_recollect: bool,
                   calibration_samples_per_bench: int,
                   calibration_time_budget_s: float,
                   output_tag: str, cache_read_dir: str,
                   skip_collection: bool,
                   # Mask + regularization opt-ins
                   gate_mode: str = "hpwl",
                   reg_weight: float = 0.0,
                   use_wiremask: bool = False,
                   use_position_mask: bool = False,
                   use_reg_feature: bool = False,
                   # Encoder / seed / reward opt-ins
                   feature_mode: str = "handcrafted",
                   encoder_kind: str = "gnn",
                   encoder_ckpt: str = "",
                   scalar_lam: float = 0.01,
                   seed_mode: str = "heuristic",
                   terminal_reward_mode: str = "committed_gain") -> None:
    import json as _json
    _setup_env()
    import train as _train
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
        missing = [n for n in benchmarks
                   if not (per_benchmark_cache_dir / f"{n}.pt").exists()]
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
        todo = [n for n in benchmarks
                if not (per_benchmark_cache_dir / f"{n}.pt").exists()]
        if todo:
            configs = [(n, examples, seed + i * 7919) for i, n in enumerate(todo)]
            print(f"[Modal] parallel collection: {len(todo)} benchmarks")
            for res in collect_one_benchmark_modal.starmap(configs):
                tag = "cached" if res.get("from_cache") else "collected"
                print(f"  {tag}: {res['name']:>6}  n={res['n']:>4}  "
                      f"wall={_train._fmt_time(res.get('wall_s', 0.0))}")
            volume.reload()
        else:
            print(f"[Modal] all {len(benchmarks)} benchmarks already cached.")

    print(f"=== iterative training on Modal ({output_tag}) ===")
    print(f"  benchmarks={benchmarks}  rounds={rounds}  output={output_root}")
    print(f"  examples/bench={examples}  cost_epochs={cost_epochs}  "
          f"policy_iters={policy_iterations}  traj/iter={trajectories_per_iter}")
    print(f"  eval_time_budget={eval_time_budget}s  "
          f"calibration={calibration_samples_per_bench}/bench @ "
          f"{calibration_time_budget_s}s")
    print(f"  per-benchmark cache: {per_benchmark_cache_dir}")

    history: list[dict] = []
    for r in range(rounds):
        record = _ti.run_round(
            r, benchmarks=benchmarks,
            num_examples_per_benchmark=examples,
            cost_epochs=cost_epochs,
            policy_iterations=policy_iterations,
            trajectories_per_iter=trajectories_per_iter,
            seed=seed, data_path=data_path,
            cost_ckpt=cost_ckpt, policy_ckpt=policy_ckpt,
            force_recollect=False,
            eval_time_budget_s=eval_time_budget,
            output_root=output_root,
            per_benchmark_cache_dir=per_benchmark_cache_dir,
            calibration_samples_per_bench=calibration_samples_per_bench,
            calibration_time_budget_s=calibration_time_budget_s,
            # Mask + regularization flags
            gate_mode=gate_mode,
            reg_weight=reg_weight,
            use_wiremask=use_wiremask,
            use_position_mask=use_position_mask,
            use_reg_feature=use_reg_feature,
            # Encoder / seed / reward flags
            feature_mode=feature_mode,
            encoder_kind=encoder_kind,
            encoder_ckpt=encoder_ckpt or None,
            scalar_lam=scalar_lam,
            seed_mode=seed_mode,
            terminal_reward_mode=terminal_reward_mode,
        )
        history.append(record)
        (output_root / "history.json").write_text(
            _json.dumps(history, indent=2, default=str)
        )
        _persist(output_tag)
        volume.commit()
        print(f"[Modal] round {r} committed ({len(history)} rounds done).")


@app.function(image=image, volumes={"/output": volume},
              cpu=8.0, memory=16384, timeout=24 * 3600,
              nonpreemptible=True)
def run_iter(benchmark: str, rounds: int, examples: int, cost_epochs: int,
             policy_iterations: int, trajectories_per_iter: int,
             eval_time_budget: float, seed: int, force_recollect: bool,
             calibration_samples_per_bench: int = 50,
             calibration_time_budget_s: float = 10.0,
             output_tag: str = "iter", cache_read_dir: str = "",
             skip_collection: bool = False,
             # Mask + regularization flags
             gate_mode: str = "hpwl",
             reg_weight: float = 0.0,
             use_wiremask: bool = False,
             use_position_mask: bool = False,
             use_reg_feature: bool = False,
             # Encoder / seed / reward flags
             feature_mode: str = "handcrafted",
             encoder_kind: str = "gnn",
             encoder_ckpt: str = "",
             scalar_lam: float = 0.01,
             seed_mode: str = "heuristic",
             terminal_reward_mode: str = "committed_gain") -> None:
    _run_iter_impl(
        benchmark, rounds, examples, cost_epochs, policy_iterations,
        trajectories_per_iter, eval_time_budget, seed, force_recollect,
        calibration_samples_per_bench, calibration_time_budget_s,
        output_tag, cache_read_dir, skip_collection,
        # Mask + regularization flags
        gate_mode=gate_mode,
        reg_weight=reg_weight,
        use_wiremask=use_wiremask,
        use_position_mask=use_position_mask,
        use_reg_feature=use_reg_feature,
        # Encoder / seed / reward flags
        feature_mode=feature_mode,
        encoder_kind=encoder_kind,
        encoder_ckpt=encoder_ckpt,
        scalar_lam=scalar_lam,
        seed_mode=seed_mode,
        terminal_reward_mode=terminal_reward_mode,
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
    """Dispatch an iterative-training Modal function and print hyperparameters."""
    call = fn.spawn(**kwargs)
    _print_spawn_handle(call, label)
    print("\n--- Hyperparameters ---")
    for key in (
        "benchmark", "rounds", "examples", "cost_epochs", "policy_iterations",
        "trajectories_per_iter", "eval_time_budget", "calibration_samples_per_bench",
        "calibration_time_budget_s", "seed", "force_recollect",
        "output_tag", "cache_read_dir", "skip_collection",
        # Mask + regularization toggles
        "gate_mode", "reg_weight", "use_wiremask", "use_position_mask",
        "use_reg_feature",
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
def train(benchmark: str = "ibm01", num_examples: int = 1500, epochs: int = 60,
          seed: int = 42, force_recollect: bool = False) -> None:
    """Train CostApproximator on Modal (background spawn).

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
    """Train ChainPolicy via PPO on Modal (background spawn).

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
def iter_(preset: str = "", benchmark: str = "", rounds: int = 0,
          examples: int = 0, cost_epochs: int = 0, policy_iterations: int = 0,
          trajectories_per_iter: int = 0, eval_time_budget: float = 0.0,
          calibration_samples_per_bench: int = 0,
          calibration_time_budget_s: float = 0.0,
          seed: int = 0, force_recollect: bool = False,
          output_tag: str = "", cache_read_dir: str = "",
          skip_collection: bool = False,
          # Mask + regularization flags
          gate_mode: str = "hpwl",
          reg_weight: float = 0.0,
          use_wiremask: bool = False,
          use_position_mask: bool = False,
          use_reg_feature: bool = False,
          # Encoder / seed / reward flags
          feature_mode: str = "handcrafted",
          encoder_kind: str = "gnn",
          encoder_ckpt: str = "",
          scalar_lam: float = 0.01,
          seed_mode: str = "heuristic",
          terminal_reward_mode: str = "committed_gain") -> None:
    """Iterative training (background spawn). All knobs configurable.

    Presets: ``long12h``, ``medium4h`` (see ``ITER_PRESETS``). Override any CLI
    flag after ``--preset``. Use a distinct ``--output-tag`` per parallel job.

    Examples::

        modal run --detach submissions/lkh/modal_run.py::iter_ \\
            --preset long12h --force-recollect

        modal run --detach submissions/lkh/modal_run.py::iter_ \\
            --benchmark all --rounds 3 --examples 240 \\
            --output-tag iter_custom --skip-collection \\
            --cache-read-dir /output/iter/per_bench

        # Encoder features + scalar gate + post-leg reward path
        modal run --detach submissions/lkh/modal_run.py::iter_ \\
            --benchmark all --feature-mode encoder --seed-mode policy \\
            --gate-mode scalar_penalty \\
            --terminal-reward-mode predicted_proxy_with_postleg

    Always use ``--detach`` with ``.spawn()`` so the job survives CLI exit.
    """
    params = _resolve_iter_kwargs(
        preset=preset, benchmark=benchmark, rounds=rounds, examples=examples,
        cost_epochs=cost_epochs, policy_iterations=policy_iterations,
        trajectories_per_iter=trajectories_per_iter,
        eval_time_budget=eval_time_budget,
        calibration_samples_per_bench=calibration_samples_per_bench,
        calibration_time_budget_s=calibration_time_budget_s,
        seed=seed, force_recollect=force_recollect,
        output_tag=output_tag, cache_read_dir=cache_read_dir,
        skip_collection=skip_collection,
        # Encoder / seed / reward flags (forwarded through preset resolver)
        feature_mode=feature_mode, encoder_kind=encoder_kind,
        encoder_ckpt=encoder_ckpt,
        scalar_lam=scalar_lam, seed_mode=seed_mode,
        terminal_reward_mode=terminal_reward_mode,
    )
    # Mask + regularization toggles pass through unconditionally — they're
    # not part of the preset bundle (so a preset doesn't accidentally turn
    # them on); flipping them is an explicit per-launch decision.
    params["gate_mode"] = gate_mode
    params["reg_weight"] = float(reg_weight)
    params["use_wiremask"] = bool(use_wiremask)
    params["use_position_mask"] = bool(use_position_mask)
    params["use_reg_feature"] = bool(use_reg_feature)
    label = f"run_iter ({preset or 'custom'})"
    print(f"=== iter_ → /output/{params['output_tag']} ===")
    _spawn_iter(run_iter, label, **params)


@app.local_entrypoint()
def overfit_inference(benchmark: str = "ibm09", n_seeds: int = 32,
                      base_seed: int = 42, time_budget: float = 300.0,
                      max_chains: int = 2000, max_chain_length: int = 8,
                      reg_weight: float = 0.05,
                      use_wiremask: bool = True,
                      use_position_mask: bool = True,
                      milestone: str = "E",
                      output_tag: str = "overfit_inf_ibm09",
                      ckpt_source: str = "/output/iter_v5_pp_reg/checkpoints") -> None:
    """Multi-seed inference overfit on a single benchmark (background spawn).

    Default config uses the predicted_proxy + policy milestone with wiremask,
    position-mask, and regularization enabled, using the v5 round-1
    checkpoints, 32 seeds × 300 s on ibm09. Total wall-time ≈ 5-10 min
    (50 containers in parallel).
    """
    call = run_overfit_sweep.spawn(
        benchmark=benchmark, n_seeds=n_seeds, base_seed=base_seed,
        time_budget_s=time_budget, max_chains=max_chains,
        max_chain_length=max_chain_length, reg_weight=reg_weight,
        use_wiremask=use_wiremask, use_position_mask=use_position_mask,
        milestone=milestone, output_tag=output_tag,
        ckpt_source=ckpt_source,
    )
    _print_spawn_handle(call, f"overfit_inference ({benchmark}, {n_seeds} seeds)")
    print(f"\n  benchmark={benchmark}  n_seeds={n_seeds}  "
          f"time_budget={time_budget}s  milestone={milestone}")
    print(f"  wm={use_wiremask}  pm={use_position_mask}  reg={reg_weight}")
    print(f"  ckpt_source={ckpt_source or '(repo default)'}")
    print(f"  output -> /output/{output_tag}/")


@app.local_entrypoint()
def overfit_inference_all(n_seeds: int = 24, base_seed: int = 100,
                          time_budget: float = 240.0,
                          max_chains: int = 3000,
                          max_chain_length: int = 10,
                          reg_weight: float = 0.05,
                          use_wiremask: bool = True,
                          use_position_mask: bool = True,
                          milestone: str = "E",
                          output_tag: str = "overfit_inf_all",
                          ckpt_source: str = "/output/iter_v5_pp_reg/checkpoints") -> None:
    """Multi-seed inference overfit across ALL 17 IBM benchmarks (background spawn).

    Fans out ``17 * n_seeds`` containers in parallel (50 at a time), each
    running ablate.py once on (benchmark, seed). Reports per-bench lex-best.

    Default: 17 benches × 24 seeds × 240s, predicted_proxy + policy milestone
    with wiremask, position-mask, and regularization enabled.
    Wall-time ≈ 25-40 min with max_containers=50.
    """
    benches = ["ibm01","ibm02","ibm03","ibm04","ibm06","ibm07","ibm08","ibm09",
               "ibm10","ibm11","ibm12","ibm13","ibm14","ibm15","ibm16","ibm17","ibm18"]
    call = run_overfit_sweep_all.spawn(
        benchmarks=benches, n_seeds=n_seeds, base_seed=base_seed,
        time_budget_s=time_budget, max_chains=max_chains,
        max_chain_length=max_chain_length, reg_weight=reg_weight,
        use_wiremask=use_wiremask, use_position_mask=use_position_mask,
        milestone=milestone, output_tag=output_tag,
        ckpt_source=ckpt_source,
    )
    _print_spawn_handle(call, f"overfit_inference_all ({len(benches)}b × {n_seeds}s)")
    print(f"\n  benchmarks={len(benches)}  n_seeds={n_seeds}  "
          f"total={len(benches) * n_seeds} containers")
    print(f"  time_budget={time_budget}s  milestone={milestone}")
    print(f"  wm={use_wiremask}  pm={use_position_mask}  reg={reg_weight}")
    print(f"  ckpt_source={ckpt_source or '(repo default)'}")
    print(f"  output -> /output/{output_tag}/")
