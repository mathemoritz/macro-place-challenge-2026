# LKH Placer — How it Works and How to Use It

> **Repo homepage:** This file is the source for the repository root
> [`README.md`](../../README.md), which shows this documentation first and the
> original challenge readme below when you scroll down on GitHub.

A macro placement system that refines an analytical seed placement
(RePlAce or DREAMPlace) via a Lin-Kernighan-style cascading local search
guided by three learned components: a GNN state encoder, an MLP cost
approximator, and a PPO actor-critic policy. Training infrastructure runs
either locally or on Modal cloud.

> **Two docs.** This `README.md` is the **operational guide** — commands,
> tuning, troubleshooting. For the **architecture walkthrough** — what each
> component does and why — see [`ARCHITECTURE.md`](./ARCHITECTURE.md).

## TL;DR — the 3 main commands

**1. Start a training run on Modal** (one-time `uv add modal && uv run modal setup` first):

```bash
# ~3 h (modest data + short eval)
uv run modal run --detach submissions/lkh/modal_run.py::iter_ \
    --benchmark all --rounds 4 --examples 75 \
    --policy-iterations 1500 --eval-time-budget 20

# ~12 h overnight (more data, longer PPO, 60s eval) — first time add --force-recollect
uv run modal run --detach submissions/lkh/modal_run.py::iter_ --preset long12h --force-recollect
```

Returns in ~10 seconds; the actual training runs in the background for ~3 hours.
You can close your laptop. Modal will email you when it finishes.

**2. Watch progress** (from any terminal, anytime):

```bash
modal app list                               # copy the ap-... ID for your run
modal app logs ap-XXXXXXXX -f                # live log stream (use ID, not app name)
modal volume ls lkh-results /iter/per_bench  # which benchmarks finished collecting
modal volume ls lkh-results /iter            # which rounds finished
```

**3. Collect results** (after Modal emails "completed"):

```bash
# Pull everything from the volume
modal volume get lkh-results /iter ./modal_output

# Promote trained checkpoints to local (canonical path on the volume)
cp modal_output/iter/checkpoints/*.pt submissions/lkh/checkpoints/

# Verify the placer with the new models
uv run python -m macro_place.evaluate submissions/lkh/placer.py --all
```

Everything else below is reference material for tuning, debugging, and local development.

---

## Pipeline at a Glance

```
initial.plc                                                     valid
(input)                                                         placement
   |                                                              ^
   v                                                              |
PlacementState                                               Legalization
(numpy arrays)                                               (outward spiral)
   |                                                              ^
   v                                                              |
LK Chain ----score candidates-----> +-- ChainPolicy (PPO)         |
(cascading                          |   if checkpoint loaded      |
 macro moves)                       +-- CostApproximator (MLP)    |
   |                                |   if checkpoint loaded      |
   |                                +-- HPWL surrogate            |
   |                                    (default fallback)        |
   |                                                              |
   +---chain commit gate---> best placement -> legalize ----------+
       (lex: overlap_pairs,
        then HPWL)
```

The placer runs a wall-clock-bounded loop of chains; each chain cascades
moves through the state. After the loop, a legalization pass forces
overlaps to zero so the output is a valid competition submission.

## What's in This Directory

| File | Role |
|---|---|
| `placer.py` | The placer (`LKHPlacer`). Contains `PlacementState`, `LKChain`, `_legalize`, checkpoint loaders. This is what the competition evaluator calls. |
| `lkh_model.py` | The two MLP/actor-critic models: `CostApproximator` and `ChainPolicy`. |
| `chain_env.py` | `ChainEnv` — wraps `PlacementState` as a Gym-style RL environment for PPO. |
| `encoder.py` | `PlacementGNN` + `CanvasCNN` + `StateEncoder`. Consumed by the runtime wrappers and training scripts. |
| `encoder_runtime.py` | GNN-only encoder runtime used at inference and during joint training. |
| `encoder_runtime_gnncnn.py` | Optional GNN+CNN encoder runtime variant. |
| `seed_head.py` | Learned seed-selection head; co-stored with the chain policy checkpoint. |
| `seed_loader.py` | Loads an analytical seed placement (RePlAce / DREAMPlace) from `seeds/`. |
| `fast_legalize.py` | Fast in-loop legalizer used for the post-legalization reward signal. |
| `commit_gate_scalar.py` | Scalar commit-gate scoring: predicted Δproxy + overlap penalty. |
| `state_snapshot.py` | Frozen state snapshots used by the joint encoder + approximator regression trainer. |
| `train.py` | Trainer for `CostApproximator`. Collects `(features, exact_Δproxy_cost)` triples and trains an MLP. |
| `train_encoder_joint.py` | Joint regression trainer for encoder + cost approximator. |
| `train_policy.py` | PPO trainer for `ChainPolicy` (and seed head when enabled). |
| `train_iter.py` | Iterative loop: cost-model training + PPO + per-benchmark evaluation + calibration. |
| `ablate.py` | Ablation runner: compares placer configurations on a benchmark. |
| `visualize.py` | Side-by-side renders: initial placement vs LKHPlacer output, with displacement arrows. |
| `modal_run.py` | Modal entry points (smoke, train, policy, iter\_). Long-running training jobs go through this. |
| `seeds/` | Drop pre-computed analytical seed `.plc` files here (one per benchmark). |
| `checkpoints/` | Trained weights (`encoder.pt`, `cost_approximator.pt`, `chain_policy.pt`). The placer auto-loads if present. |
| `data/` | Cached training data. `chain_data.pt` is the aggregated dataset; `per_bench/` has per-benchmark caches that survive preemption. |
| `iter_output/` | Per-round archival from `train_iter.py` (`round_<i>/`, `history.json`). |

## Local Workflows

### Evaluate the placer

```bash
# Single benchmark (uses whatever checkpoints exist in checkpoints/)
uv run python -m macro_place.evaluate submissions/lkh/placer.py -b ibm01

# All 17 IBM benchmarks
uv run python -m macro_place.evaluate submissions/lkh/placer.py --all

# Hide checkpoints to test the no-learning baseline
mv submissions/lkh/checkpoints submissions/lkh/checkpoints.bak
uv run python -m macro_place.evaluate submissions/lkh/placer.py -b ibm01
mv submissions/lkh/checkpoints.bak submissions/lkh/checkpoints
```

The evaluator prints `proxy=<cost>  overlaps=<n>  VALID/INVALID` per benchmark.
A valid submission has `overlaps=0` on every benchmark.

### Ablation (compare A/B/C/D)

```bash
uv run python submissions/lkh/ablate.py --time-budget 30 --benchmark ibm01
```

Prints a comparison table of the four configurations.

### Visualize a placement

```bash
uv run python submissions/lkh/visualize.py --benchmark ibm01 --time-budget 30
# -> vis/ibm01_compare.png  (initial | placer output, with displacement arrows)

# Multiple benchmarks
uv run python submissions/lkh/visualize.py --benchmark ibm01,ibm07 --no-arrows
```

### Train locally (for quick experiments)

```bash
# Cost approximator alone (on ibm01)
uv run python submissions/lkh/train.py --benchmark ibm01 --num-examples 200

# PPO policy alone (on ibm01)
uv run python submissions/lkh/train_policy.py --benchmark ibm01 --iterations 200

# Full iterative pipeline (small, ~5 min)
uv run python submissions/lkh/train_iter.py \
    --benchmark ibm01,ibm07 --rounds 1 --examples 20 --policy-iterations 100
```

Local training is fine for development but slow at scale. For real training,
use Modal.

## Modal Workflows (the main path)

Modal lets us run training on rented CPUs in parallel. Used for any training
that needs more than a few minutes.

### One-time setup

```bash
# Install the Modal CLI in this project
uv add modal

# Authenticate (opens browser to your Modal account)
uv run modal setup
```

### Smoke-test the Modal image (~30 sec the second time, ~3 min first)

```bash
uv run modal run submissions/lkh/modal_run.py::smoke
```

This builds the Docker image, mounts the repo, loads ibm01 on a Modal worker,
and prints `OK`. Run this once after any change to `modal_run.py` to confirm
the environment still works.

### Full training run (the main workflow)

**~3 h (default-scale):**

```bash
uv run modal run --detach submissions/lkh/modal_run.py::iter_ \
    --benchmark all \
    --rounds 4 \
    --examples 75 \
    --policy-iterations 1500 \
    --eval-time-budget 20
```

**~12 h overnight preset** (`--preset long12h` — see `ITER_PRESETS` in `modal_run.py`):

```bash
uv run modal run --detach submissions/lkh/modal_run.py::iter_ --preset long12h --force-recollect
```

| Knob | ~3 h run | `long12h` |
|------|----------|------------|
| `examples` / bench | 75 | **240** |
| `rounds` | 4 | **5** |
| `cost-epochs` | 60 | **100** |
| `policy-iterations` | 1500 | **3500** |
| `trajectories-per-iter` | 4 | **16** |
| `eval-time-budget` | 20 s | **60 s** |
| calibration / bench | 50 @ 10 s | **75 @ 15 s** |

**`--force-recollect`:** archives the current `iter/` tree to
`iter/archives/<UTC-timestamp>/` on the volume, clears working files, then runs
**parallel** `.starmap()` collection only. Iterative rounds **load** those caches
(they do not re-collect sequentially on Modal).
Use this when upgrading from a shorter run (e.g. 75-example caches).

Omit `--force-recollect` only when `per_bench/*.pt` already matches the
target example count (saves ~7 h; rounds-only ~4–5 h).

### Parallel runs (different hyperparameters / output trees)

Use a **unique `--output-tag`** per job on the same volume. Presets: `long12h`, `medium4h`.

```bash
# Main overnight run
uv run modal run --detach submissions/lkh/modal_run.py::iter_ --preset long12h

# Parallel experiment: reuse iter/per_bench caches, separate output
uv run modal run --detach submissions/lkh/modal_run.py::iter_ \
  --preset medium4h --output-tag iter_exp_a --seed 44

# Fully custom (no preset)
uv run modal run --detach submissions/lkh/modal_run.py::iter_ \
  --benchmark all --rounds 2 --examples 240 \
  --output-tag iter_custom --skip-collection \
  --cache-read-dir /output/iter/per_bench
```

`medium4h` skips collection and reads `/output/iter/per_bench`. All jobs use one
`run_iter` worker (24 h Modal timeout); plan wall time from calibration volume.

Both `--detach` and `.spawn()` are needed:
- `.spawn()` (in modal\_run.py) makes the call async — survives local terminal disconnect.
- `--detach` keeps the App alive after `modal run` exits.

The CLI returns within ~10 seconds with a `function_call_id`. You can close
your laptop; the run continues on Modal.

### Tuning knobs

| Flag | Default | Tradeoff |
|---|---:|---|
| `--benchmark` | `ibm01` | `all` = full IBM suite (17). Comma list for subsets. |
| `--rounds` | `2` | More rounds = longer PPO warm-start. Each round after the first is fast (data cached). |
| `--examples` | `400` | Per-benchmark sample count for cost-approximator data collection. `compute_proxy_cost` cost scales with examples × benchmark size. |
| `--policy-iterations` | `1000` | PPO update count per round. 1500-3000 is reasonable. |
| `--trajectories-per-iter` | `4` | Higher = better gradient estimates but slower PPO. |
| `--eval-time-budget` | `20.0` | Seconds per benchmark during eval phase. Bigger = better placement quality. |
| `--preset` | (none) | `long12h` or `medium4h`; base hyperparameters from `ITER_PRESETS`. |
| `--output-tag` | `iter` | Volume subtree `/output/<tag>/` (required for parallel jobs). |
| `--cache-read-dir` | `` | e.g. `/output/iter/per_bench` to reuse another run's caches. |
| `--skip-collection` | off | Require existing per-bench `.pt` under `cache-read-dir`. |
| `--calibration-samples-per-bench` | `50` | Calibration samples after each round. |
| `--calibration-time-budget-s` | `10.0` | Placer seconds per benchmark during calibration. |
| `--force-recollect` | off | Wipes per-benchmark cache before collection. Use on first `long12h` run. |

### Wall-time estimates on Modal

For `--benchmark all`:

| examples | Parallel data collection | per round (PPO+eval) | rounds | total |
|---:|---:|---:|---:|---:|
| 50 | ~1.5 h | ~7 min | 3 | ~2.0 h |
| 75 | ~2.2 h | ~7 min | 4 | ~3.0 h |
| 100 | ~3.0 h | ~7 min | 4 | ~3.7 h |
| 150 | ~4.5 h | ~7 min | 3 | ~5.0 h |
| **240** (`long12h`) | **~7 h** | **~50 min** | **5** | **~11–12 h** |

Parallel collection runs once; Round 0 then **loads** those caches (no second
pass over ``compute_proxy_cost``). Subsequent rounds reuse ``chain_data.pt``.
Cost: `nonpreemptible=True` is set on long-running functions and bills at
3× the list CPU rate — expect a few credits per full run.

### Watching it run

In another terminal (your laptop can sleep — the job won't die):

```bash
# Confirm it's actually running (note the ap-... App ID column)
modal app list

# Live log stream (Ctrl-C just detaches the viewer, doesn't stop the run)
# Detached runs are looked up by App ID, not by name — copy ap-... from app list
modal app logs ap-XXXXXXXX -f

# Which benchmarks have finished data collection
modal volume ls lkh-results /iter/per_bench
# Each name.pt that appears = one benchmark fully collected

# Check overall progress
modal volume ls lkh-results /iter
# After round 0 finishes: round_0/, checkpoints/, history.json, data/
# After round N: round_0/, ..., round_N/
```

You will get a Modal email when the run finishes.

### After it finishes

```bash
# Pull everything from the volume
modal volume get lkh-results /iter ./modal_output

# Inspect
ls modal_output/iter/
# checkpoints/, data/, history.json, per_bench/, round_0/, round_1/, ...

# See training progress numerically
cat modal_output/iter/history.json

# Promote the trained checkpoints to local
cp modal_output/iter/checkpoints/*.pt submissions/lkh/checkpoints/

# Test the placer locally
uv run python -m macro_place.evaluate submissions/lkh/placer.py -b ibm01
uv run python -m macro_place.evaluate submissions/lkh/placer.py --all

# Visualize on a few benchmarks
uv run python submissions/lkh/visualize.py --benchmark ibm01,ibm10,ibm17
```

You can also inspect or test the per-round snapshots:

```bash
# Round 2 vs round 3 on ibm01
cp modal_output/iter/round_2/*.pt submissions/lkh/checkpoints/
uv run python -m macro_place.evaluate submissions/lkh/placer.py -b ibm01
cp modal_output/iter/round_3/*.pt submissions/lkh/checkpoints/
uv run python -m macro_place.evaluate submissions/lkh/placer.py -b ibm01
```

### Stopping a run

```bash
modal app stop lkh-macro-place
# If you have multiple lkh-macro-place apps in 'app list', use the app ID:
modal app stop ap-XXXX...
```

Any completed rounds remain on the volume (per-round `volume.commit()`).

## How the Pipeline Recovers from Failure

Three layers of resilience, in order of how often they help:

1. **Per-benchmark cache** (`/output/iter/per_bench/<name>.pt`)
   Each benchmark's data is committed immediately after collection. If a
   container crashes or is killed mid-collection, only that one benchmark's
   in-flight work is lost; the others stay durable. A relaunch resumes from
   wherever the cache stopped.

2. **Per-round volume commit + archival** (`/output/iter/round_N/`)
   After each complete round, the canonical checkpoints AND a per-round
   snapshot are committed to the volume. If a run dies between rounds, all
   completed rounds are preserved.

3. **`nonpreemptible=True`**
   Modal won't preempt the workers (previously the cause of mid-collection
   restarts). Costs 3× CPU rate but eliminates the failure mode entirely.

4. **`--force-recollect` archives before overwrite**
   Prior `iter/` outputs are copied to `iter/archives/<UTC-timestamp>/`
   (per-bench caches, rounds, `history.json`, checkpoints) before the
   working tree is cleared. List archives with
   ``modal volume ls lkh-results /iter/archives``.

## Hyperparameter Cheat Sheet

| Where | Knob | Default | When to change |
|---|---|---|---|
| `LKHPlacer.__init__` | `time_budget_s` | 60 | Raise for higher-quality placements at evaluation time |
| `LKHPlacer.__init__` | `max_chain_length` | 8 | 5-10 reasonable; longer chains rarely help |
| `LKHPlacer.__init__` | `stagnation_threshold` | 50 | Chains without improvement before random perturbation |
| `LKHPlacer.__init__` | `perturb_macros` | 5 | How many macros to randomly move on stagnation |
| `train.py` `--num-examples` | 1500 | Per-benchmark data collection count |
| `train.py` `--epochs` | 60 | Cost approximator training epochs |
| `train.py` `--hidden` | 64 | CostApproximator hidden dim |
| `train_policy.py` `--iterations` | 200 | PPO iterations (1500-3000 for real runs) |
| `train_policy.py` `--ent-coef` | 0.01 | Bump up if policy collapses too fast |
| `train_iter.py` `--rounds` | 2 | Iterative refinement count |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `INVALID (N overlaps)` from evaluator | Legalization didn't reach 0 | Check the legalization log line — if it stopped at >0 overlaps, bump `max_search_radius` in `placer._legalize` |
| `rate=0.0/s` in old logs | Old format bug, fixed | Pull latest; format now shows `s/ex` for slow rates |
| Modal "Worker preemption" message | Old runs without `nonpreemptible=True` | Already fixed in `modal_run.py` |
| Modal "Cancellation signal" mid-collection | `.remote()` instead of `.spawn()` or attached `modal run` instead of `--detach` | Use the documented launch command (`.spawn()` + `--detach`) |
| Modal `App ... has no attribute 'Mount'` | Modal 1.x removed `Mount.from_local_dir` | Already using `image.add_local_dir(...)` |
| `modal call logs` errors with "No such command" | Modal 1.x uses `modal app logs`, not `modal call` | Use `modal app logs lkh-macro-place` |
| Cost approximator Pearson stays low | Not enough multi-benchmark data | Bump `--examples` |
| PPO commit rate stuck at 0% | Entropy collapsed too fast | Increase `--ent-coef` to 0.05 |
| Long benchmark hangs at "first call: > 100s" | `compute_proxy_cost` natural slowness on big designs (ibm17, ibm18) | Expected. Parallel mode bounds wall time. |

## Future Directions

- **Gradient-based local refinement.** The cost approximator is
  differentiable; gradient steps through it could refine macro positions
  locally, using the cost model as a continuous optimizer rather than
  only a per-candidate scorer.
- **Wider models.** Defaults are deliberately small (hidden=64);
  higher-capacity approximator + wider policy should fit the proxy more
  closely.
- **Larger neighborhoods.** Extending the maximum chain length and
  proposing more candidate positions per step would let one cascade rework
  a larger neighborhood.

## Reference

- Macro Placement Challenge: `CHALLENGE_README.md`, `SETUP.md`, `SCORING.md`
- Evaluator: `macro_place/evaluate.py`
- Baseline placers: `submissions/examples/`, `submissions/will_seed/`
