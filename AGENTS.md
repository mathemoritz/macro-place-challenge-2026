# AGENTS.md — Session-start context for this repo

This file orients future agent sessions. It captures what isn't obvious from
just reading the code: who owns what, where the boundaries are, and the
non-obvious invariants the submission depends on. For the architectural
walkthrough see `submissions/lkh/ARCHITECTURE.md`; for the operational
guide see `submissions/lkh/README.md`.

## What this repo is

The Partcl/HRT Macro Placement Challenge 2026 working tree. The challenge
asks for a placer that improves on `initial.plc` proxy cost across 17 IBM
ICCAD04 benchmarks (Tier 1) without introducing hard-macro overlaps. The
top 7 by proxy go through the OpenROAD flow on NG45 designs (Tier 2,
WNS/TNS/Area).

Our submission is **LKH** — a Lin-Kernighan chain placer with two learned
models (`CostApproximator` MLP for Δproxy prediction; `ChainPolicy` PPO
actor-critic for in-chain move selection + STOP) on top of a hand-coded
HPWL surrogate, cascade follow rule, and outward-spiral legalizer.

## Upstream vs ours — boundary and invariant

**Boundary commit:** `7d32ac9` ("checkpoint: upstream harness baseline", Danny Hagenlocker,
2026-05-20). Everything in that commit and earlier is the upstream
Partcl/HRT challenge harness. Everything after it is ours.

**Invariant we maintain: zero edits to upstream Python.** Verified by
`git diff 7d32ac9 HEAD -- macro_place/ scripts/ src/ test/
submissions/examples/ submissions/will_seed/ eval_docker/ benchmarks/
baselines/ SCORING.md SETUP.md` returning empty.

If you find a bug that seems to be in upstream code (`macro_place/*`,
`scripts/*`, etc.), do not patch it in-place. Either work around it in our
code or surface it as a question — modifying the evaluator path means
diverging from what judges run.

### Upstream (don't modify)

| Path | What |
|---|---|
| `macro_place/benchmark.py` | `Benchmark` dataclass: positions/sizes/fixed, `net_nodes` (deduped owners), `net_pin_nodes` ([pins,2] preserving multi-pin), helper masks |
| `macro_place/loader.py` | `load_benchmark`/`load_benchmark_from_dir` — extracts tensors from TILOS `PlacementCost`; owner-index convention is `[hard][soft][ports]` |
| `macro_place/objective.py` | `compute_proxy_cost` = `1.0×WL + 0.5×density + 0.5×congestion`; contains monkey-patch on `PlacementCost.__get_grid_cell_location` (row/col clamp) |
| `macro_place/utils.py` | `validate_placement` (strict overlap, NaN/Inf, bounds, fixed-macro check) and `visualize_placement` |
| `macro_place/_plc.py` | 26-line `sys.path` shim → `external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py` |
| `macro_place/evaluate.py` | `uv run evaluate <placer.py>` CLI; owns `IBM_BENCHMARKS`, `NG45_BENCHMARKS`, `SA_BASELINES`, `REPLACE_BASELINES`; this is what judges invoke |
| `macro_place/def_writer.py` | DEF export for downstream EDA tools |
| `external/MacroPlacement` | TILOS evaluator submodule (partcleda fork, `fix-scientific-notation-parsing`) |
| `external/circuit_training` | Google RL training submodule, unused by us |
| `submissions/examples/*` | Demo placers (`greedy_row_placer`, `simple_random_placer`) |
| `submissions/will_seed/placer.py` | Partcl internal seed (legalize + SA refinement) |
| `test/test_smoke.py` | Competition smoke tests; greedy placer is the canary |
| `eval_docker/{Dockerfile,run_eval.sh}` | Judges' `--network none` harness |
| `scripts/*`, `src/orfs_integration/*` | Optional Tier-2 ORFS local eval |
| `benchmarks/processed/public/*.pt`, `baselines/ng45_baselines.csv` | Cached benchmarks + NG45 baselines |
| `CHALLENGE_README.md`, `SETUP.md`, `SCORING.md`, `LICENSE.md` | Partcl docs |

### Ours

All of `submissions/lkh/`. Key files:

| Path | Role |
|---|---|
| `placer.py` | `LKHPlacer` (Phase 6 inference), `PlacementState`, `LKChain`, `_legalize`, `_features_for_move` (16-dim), checkpoint loaders |
| `lkh_model.py` | `CostApproximator` (16→64→64→1 MLP), `ChainPolicy` (actor-critic, K+1 logits) |
| `chain_env.py` | `ChainEnv` Gym wrapper; `compute_global/macro/chain_features` |
| `train.py` | Phase 3: state-drift data collection + C.2 mini-cascade pre-roll + C.3 calibration sampling + Tier-1 rank loss / Spearman-best selection |
| `train_policy.py` | Phase 4: PPO + Tier-1 Fix C (best-by-commit-EMA snapshot, 20-iter warmup) |
| `train_iter.py` | Phase 5: round = load/collect → train approx → PPO → per-bench eval → calibration |
| `modal_run.py` | Modal cloud app `lkh-macro-place`, `lkh-results` volume, `iter_`/`iter_12h`/`iter_medium` entry points |
| `ablate.py` | Phase 7: legacy A/B/C/D + milestone-mode A→E with history-file deltas |
| `encoder.py` | Phase 2 pure-PyTorch GNN+CNN — **implemented, smoke-tested, not wired in** |
| `visualize.py`, `visualize_comprehensive.py` | Side-by-side and panel renders |
| `ARCHITECTURE.md`, `README.md` | The walkthrough and the operational guide |

Shared files we touched (additive only):
- `pyproject.toml` — one line: `"modal>=0.65.66"`
- `.gitignore` — `.cursor/` and `background/`
- `README.md` (top-level) — prepended LKH docs above the original challenge content (still verbatim below)
- `CHALLENGE_README.md` — created as the original `README.md` content + a 2-line cross-reference
- `modal_output/*` — checked-in training-run artifacts (`history.json`, per-round records)

## Non-obvious invariants

These are the gotchas that bit us once and the comments warn about. Keep
them in mind when changing related code.

1. **Strict overlap gap = 1e-6 for the chain's cached `overlap_pairs`.**
   Matches the evaluator's `<` (not `<=`) semantics up to float precision.
   The pre-fix `gap=0.05` inflated the count, letting the chain commit
   states that reduced the inflated count while *increasing* the true
   count. `_legalize` uses `gap=0.001` (1 nm breathing room).
2. **Cache contract:** any code that writes `state.pos` directly *must*
   call `state.rebuild_caches()` afterward, or `_total_hpwl` /
   `_overlap_partners` / `_total_overlap_pairs` / `_total_overlap_area`
   drift. `apply_move` handles this incrementally and is the preferred
   path.
3. **Best-PREFIX commit, not endpoint.** Both `LKChain.run_greedy` and
   `ChainEnv._finalize` restore the lex-best prefix snapshot. Reverting
   to the endpoint loses LK's main mechanism.
4. **Lex tuple is `(overlap_pairs, overlap_area, third)`.** The outer
   placer's `best_key` must match the chain's gate or you'll undo work
   in legalization. `third` is `hpwl` (default `gate_mode="hpwl"`) or
   `cumulative_predicted_Δproxy` (`"predicted_proxy"`, requires
   approximator).
5. **`placer.py` does `sys.path.insert(0, _HERE)` at the top.**
   `evaluate.py` loads it via `importlib.util.spec_from_file_location`,
   which doesn't set the sibling-import path. Without the insert,
   `from lkh_model import ...` fails at judge time.
6. **`FEATURE_DIM = 16` is shared.** Defined in `placer.py` and
   `lkh_model.py`; the assertion in `_features_for_move` guards it.
   Changing the feature schema requires bumping both constants and
   retraining `CostApproximator` and `ChainPolicy` (the policy reads
   per-cand features through `_features_for_move` too).
7. **Soft macros are not moved.** `LKHPlacer` only optimizes hard macros
   `[0, num_hard_macros)`; soft macros keep their `initial.plc` positions.
   Matches `submissions/will_seed`.
8. **Checkpoints are gitignored** by the `*.pt` rule. After a Modal run,
   download with `modal volume get lkh-results /iter ./modal_output`, then
   `cp modal_output/iter/checkpoints/*.pt submissions/lkh/checkpoints/`.
   The judges expect to find them at `submissions/lkh/checkpoints/`.
9. **Tier-1 selection metric is Spearman ρ, not Smooth-L1.** The placer
   uses `CostApproximator` as an argmin-over-candidates ranker; rank
   correlation, not amplitude, is what predicts downstream proxy gain.
10. **PPO ships the best-by-commit-EMA snapshot, not the last iteration.**
    Late PPO can collapse (entropy → 0). EMA α=0.1, warmup 20 iters.

## How to run common things

```bash
# Single-benchmark evaluation (auto-loads checkpoints if present)
uv run python -m macro_place.evaluate submissions/lkh/placer.py -b ibm01

# All 17 IBM benchmarks
uv run python -m macro_place.evaluate submissions/lkh/placer.py --all

# Force no-learning baseline
mv submissions/lkh/checkpoints submissions/lkh/checkpoints.bak
uv run python -m macro_place.evaluate submissions/lkh/placer.py -b ibm01
mv submissions/lkh/checkpoints.bak submissions/lkh/checkpoints

# Modal training (the main path — ~3 h, --detach + .spawn)
uv run modal run --detach submissions/lkh/modal_run.py::iter_ \
    --benchmark all --rounds 4 --examples 75 \
    --policy-iterations 1500 --eval-time-budget 20

# Modal logs / progress
modal app logs lkh-macro-place
modal volume ls lkh-results /iter/per_bench
```

## Pointers

- Operational guide and tuning knobs: `submissions/lkh/README.md`
- Architectural walkthrough (Phase 1–7 + Modal arch + feature schema): `submissions/lkh/ARCHITECTURE.md`
- Challenge rules + leaderboard: `CHALLENGE_README.md`
- Tier-2 scoring formula: `SCORING.md`
- Repo API reference: `SETUP.md`
