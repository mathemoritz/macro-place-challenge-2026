"""Phase 4 — PPO trainer for the LK chain policy.

Algorithm: standard PPO with GAE. Rollouts collected from ChainEnv (HPWL
surrogate reward). Each chain step has a variable number of candidates K,
so transitions are processed individually inside the update — for the
modest batch sizes implied by the plan (~40 transitions/iter), no padding
is needed.

Multi-benchmark support (plan §4.3): all benchmarks are pre-loaded into a
lightweight cache (the PlacementCost object is dropped — PPO only needs the
HPWL surrogate, not exact proxy cost), and each rollout samples a random
benchmark from the cache. State is reset to the benchmark's initial.plc
positions before every rollout so trajectories are independent.

Usage:
    uv run python submissions/lkh/train_policy.py --iterations 200
    uv run python submissions/lkh/train_policy.py --benchmark ibm01,ibm02,ibm03
    uv run python submissions/lkh/train_policy.py --benchmark all --iterations 100
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent.parent))

from submissions.lkh.model.lkh_model import (
    ChainPolicy, ChainPolicyWithGNN, ProxyCostPredictor, StateEncoder,
)


def parse_benchmarks(arg: str) -> list[str]:
    arg = (arg or "").strip()
    if arg == "all":
        from macro_place.evaluate import IBM_BENCHMARKS
        return list(IBM_BENCHMARKS)
    names = [n.strip() for n in arg.split(",") if n.strip()]
    if not names:
        raise ValueError(f"--benchmark cannot be empty")
    return names

_spec_placer = importlib.util.spec_from_file_location("lkh_placer", str(_HERE.parent / "placer.py"))
_placer = importlib.util.module_from_spec(_spec_placer)
_spec_placer.loader.exec_module(_placer)

_spec_env = importlib.util.spec_from_file_location("chain_env", str(_HERE.parent / "environment" / "chain_env.py"))
_env_mod = importlib.util.module_from_spec(_spec_env)
_spec_env.loader.exec_module(_env_mod)
ChainEnv = _env_mod.ChainEnv

from macro_place.loader import load_benchmark_from_dir


def _load_proxy_predictor_for_policy(ckpt_path: Path):
    """Load ProxyCostPredictor from a proxy_cost_encoder checkpoint.

    Returns (proxy_predictor_dict, encoder, gnn_H) or (None, None, None) on hard failure.

    If the proxy head weights don't fit the current architecture (e.g., the
    encoder was saved before gnn_global dim changed from 5H to 7H), the encoder
    backbone is still loaded for ChainPolicyWithGNN state features, but
    proxy_predictor is None so the gate falls back to hpwl. Retrain the encoder
    with train_encoder.py to restore full gate scoring.
    """
    if not ckpt_path.exists():
        print(f"  [warn] encoder ckpt not found: {ckpt_path}")
        return None, None, None
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if ckpt.get("type") != "proxy_cost_encoder":
        print(f"  [warn] unexpected ckpt type: {ckpt.get('type')}")
        return None, None, None

    gnn_H = int(ckpt["hidden_dim"])
    proxy_predictor = None

    try:
        model = ProxyCostPredictor(
            hidden_dim=ckpt["hidden_dim"],
            num_gnn_layers=ckpt["num_gnn_layers"],
            edge_fc_layers=ckpt.get("edge_fc_layers", 1),
            grid_size=ckpt.get("grid_size", 128),
            use_cnn=ckpt.get("use_cnn", True),
        )
        model.encoder.load_state_dict(ckpt["encoder_state"])
        model.proxy_head.load_state_dict(ckpt["head_state"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        proxy_predictor = {
            "model": model,
            "norm_stats": ckpt.get("norm_stats", {}),
            "proxy_weights": ckpt.get("proxy_weights", {"wl": 1.0, "den": 0.5, "cong": 0.5}),
            "pearson_r_wl": ckpt.get("pearson_r_wl"),
            "trained_on": ckpt.get("trained_on"),
        }
        encoder = model.encoder  # shared backbone — no duplication
        print(f"  proxy_predictor + encoder loaded: H={gnn_H}  "
              f"pearson_r_wl={ckpt.get('pearson_r_wl')}  trained_on={ckpt.get('trained_on')}")
    except RuntimeError:
        # Head architecture mismatch (checkpoint predates gnn_global=7H change).
        # Encoder weights are still usable for GNN state features.
        print(f"  [warn] proxy head mismatch (checkpoint predates current architecture)")
        print(f"  [warn] encoder loaded for state features only; gate_mode falls back to hpwl")
        print(f"  [warn] retrain encoder with train_encoder.py to enable predicted_proxy gate")
        encoder = StateEncoder(
            hidden_dim=ckpt["hidden_dim"],
            num_gnn_layers=ckpt["num_gnn_layers"],
            edge_fc_layers=ckpt.get("edge_fc_layers", 1),
            grid_size=ckpt.get("grid_size", 128),
        )
        encoder.load_state_dict(ckpt["encoder_state"])
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        print(f"  encoder loaded: H={gnn_H}  trained_on={ckpt.get('trained_on')}")

    return proxy_predictor, encoder, gnn_H


def _load_benchmark_cache(benchmark_names: list[str]) -> dict[str, dict]:
    """Pre-load all benchmarks into memory for fast per-trajectory sampling.

    PPO training uses ChainEnv (HPWL surrogate reward), which doesn't need
    PlacementCost. We bind ``_plc`` to ``_`` so it can be garbage-collected
    immediately, keeping the cache lightweight (≈1–2 MB per benchmark).
    """
    cache: dict[str, dict] = {}
    for name in benchmark_names:
        bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        benchmark, _plc = load_benchmark_from_dir(str(bench_dir))
        del _plc
        hpwl_edges = _placer._hard_macro_edges(benchmark)
        state = _placer.PlacementState(benchmark, hpwl_edges)
        cache[name] = {
            "state": state,
            "init_pos": state.pos.copy(),
            "movable": np.where(state.movable)[0].tolist(),
        }
    return cache


# ── Rollout collection ─────────────────────────────────────────────────────


def collect_rollout(env: ChainEnv, policy) -> list[dict]:
    """One chain episode using stochastic policy sampling."""
    transitions: list[dict] = []
    while not env.done:
        sp = env.state_for_policy()
        K = sp["candidates"].shape[0]
        if K == 0:
            env.step(0, [])  # step() routes K==0 through _finalize
            break

        global_t = torch.from_numpy(sp["global"])
        macro_t = torch.from_numpy(sp["macro"])
        cands_t = torch.from_numpy(sp["candidates"])
        chain_t = torch.from_numpy(sp["chain"])
        gnn_g   = sp.get("gnn_global")
        macro_g = sp.get("macro_gnn_emb")

        with torch.no_grad():
            if isinstance(policy, ChainPolicyWithGNN):
                logits, value = policy(global_t, macro_t, cands_t, chain_t,
                                       gnn_global=gnn_g, macro_gnn_emb=macro_g)
            else:
                logits, value = policy(global_t, macro_t, cands_t, chain_t)
        probs = F.softmax(logits, dim=-1)
        action = int(torch.multinomial(probs, 1).item())
        log_prob_old = F.log_softmax(logits, dim=-1)[action].detach()

        reward, _done, _info = env.step(action, sp["raw_candidates"])

        transitions.append(
            {
                "global": global_t,
                "macro": macro_t,
                "candidates": cands_t,
                "chain": chain_t,
                "gnn_global":    gnn_g,
                "macro_gnn_emb": macro_g,
                "action": action,
                "log_prob_old": log_prob_old,
                "value_old": value.detach(),
                "reward": float(reward),
            }
        )
    return transitions


def compute_gae(transitions: list[dict], gamma: float = 0.99, lam: float = 0.95) -> list[dict]:
    """Generalized Advantage Estimation. Terminal value is 0."""
    last_adv = 0.0
    last_val = 0.0
    for t in reversed(range(len(transitions))):
        v = float(transitions[t]["value_old"].item())
        delta = transitions[t]["reward"] + gamma * last_val - v
        last_adv = delta + gamma * lam * last_adv
        transitions[t]["advantage"] = last_adv
        transitions[t]["return"] = last_adv + v
        last_val = v
    return transitions


# ── PPO update ─────────────────────────────────────────────────────────────


def ppo_update(
    policy,
    optimizer: torch.optim.Optimizer,
    batch: list[dict],
    clip: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    epochs: int = 4,
) -> dict:
    if not batch:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "grad_norm": 0.0, "clip_fraction": 0.0, "approx_kl": 0.0}

    advs = torch.tensor([t["advantage"] for t in batch], dtype=torch.float32)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    for i, t in enumerate(batch):
        t["adv_norm"] = float(advs[i].item())

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
               "grad_norm": 0.0, "clip_fraction": 0.0, "approx_kl": 0.0}
    n_updates = 0
    indices = list(range(len(batch)))
    for _ in range(epochs):
        np.random.shuffle(indices)
        for idx in indices:
            t = batch[idx]
            if isinstance(policy, ChainPolicyWithGNN):
                logits, value = policy(
                    t["global"], t["macro"], t["candidates"], t["chain"],
                    gnn_global=t.get("gnn_global"),
                    macro_gnn_emb=t.get("macro_gnn_emb"),
                )
            else:
                logits, value = policy(t["global"], t["macro"], t["candidates"], t["chain"])
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            log_prob_new = log_probs[t["action"]]

            ratio = torch.exp(log_prob_new - t["log_prob_old"])
            adv = torch.tensor(t["adv_norm"], dtype=torch.float32)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
            policy_loss = -torch.min(surr1, surr2)

            ret = torch.tensor(t["return"], dtype=torch.float32)
            value_loss = F.smooth_l1_loss(value, ret)
            entropy = -(probs * log_probs).sum()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(policy.parameters(), 0.5))
            optimizer.step()

            metrics["policy_loss"]   += float(policy_loss.item())
            metrics["value_loss"]    += float(value_loss.item())
            metrics["entropy"]       += float(entropy.item())
            metrics["grad_norm"]     += grad_norm
            metrics["clip_fraction"] += float((ratio.detach().abs() - 1.0).abs().item() > clip)
            metrics["approx_kl"]     += float((t["log_prob_old"] - log_prob_new).detach().item())
            n_updates += 1

    if n_updates > 0:
        for k in metrics:
            metrics[k] /= n_updates
    return metrics


# ── Training loop ──────────────────────────────────────────────────────────


def train_chain_policy(
    benchmarks: list[str],
    *,
    n_iterations: int,
    trajectories_per_iter: int,
    max_chain_length: int,
    max_candidates: int,
    terminal_commit_bonus: float,
    seed: int,
    lr: float,
    gamma: float,
    lam: float,
    clip: float,
    ent_coef: float,
    vf_coef: float,
    ppo_epochs: int,
    output_ckpt: Path,
    initial_policy_ckpt: Path | None = None,
    log_every: int = 5,
    terminal_reward_mode: str = "committed_gain",
    encoder_ckpt: Path | None = None,
    proj_dim: int = 16,
    gate_mode: str = "hpwl",
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> dict:
    """PPO training across one or more benchmarks. Each trajectory samples
    a random benchmark from ``benchmarks`` (plan §4.3 semantics)."""
    import random as _random

    if not benchmarks:
        raise ValueError("train_chain_policy: benchmarks list is empty")

    rng = _random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"=== Phase 4: PPO chain policy ===")
    print(
        f"  benchmarks={benchmarks}  iterations={n_iterations}  "
        f"traj/iter={trajectories_per_iter}"
    )
    print(f"  max_chain_length={max_chain_length}  max_candidates={max_candidates}")
    print(f"  pre-loading {len(benchmarks)} benchmark(s) into RAM...")
    bench_cache = _load_benchmark_cache(benchmarks)

    proxy_predictor, frozen_encoder, gnn_H = None, None, None
    if encoder_ckpt is not None:
        proxy_predictor, frozen_encoder, gnn_H = _load_proxy_predictor_for_policy(encoder_ckpt)

    if frozen_encoder is not None:
        policy = ChainPolicyWithGNN(hidden=64, gnn_H=gnn_H, proj_dim=proj_dim)
        print(f"  policy = ChainPolicyWithGNN (gnn_H={gnn_H}, proj_dim={proj_dim})")
    else:
        policy = ChainPolicy(hidden=64)

    effective_gate_mode = gate_mode
    if gate_mode == "predicted_proxy" and proxy_predictor is None:
        print("  [warn] gate_mode=predicted_proxy but no predictor loaded; using hpwl")
        effective_gate_mode = "hpwl"
    print(f"  gate_mode = {effective_gate_mode}")

    if initial_policy_ckpt is not None and initial_policy_ckpt.exists():
        ckpt = torch.load(initial_policy_ckpt, map_location="cpu", weights_only=False)
        saved_cls = ckpt.get("policy_class", "ChainPolicy")
        if saved_cls == type(policy).__name__:
            policy.load_state_dict(ckpt["state_dict"])
            print(f"  warm-started policy from {initial_policy_ckpt}")
        else:
            print(f"  [warn] ckpt class={saved_cls} != {type(policy).__name__}; skipping warm-start")
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # ── WandB ──────────────────────────────────────────────────────────────
    _wrun = None
    if wandb_project is not None:
        if _wandb is None:
            print("  [warn] wandb not installed; run `pip install wandb` to enable logging")
        else:
            _wrun = _wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "benchmarks":            benchmarks,
                    "n_iterations":          n_iterations,
                    "trajectories_per_iter": trajectories_per_iter,
                    "max_chain_length":      max_chain_length,
                    "max_candidates":        max_candidates,
                    "terminal_reward_mode":  terminal_reward_mode,
                    "gate_mode":             effective_gate_mode,
                    "lr":                    lr,
                    "gamma":                 gamma,
                    "lam":                   lam,
                    "clip":                  clip,
                    "ent_coef":              ent_coef,
                    "vf_coef":               vf_coef,
                    "ppo_epochs":            ppo_epochs,
                    "seed":                  seed,
                    "policy_class":          type(policy).__name__,
                    "gnn_H":                 gnn_H,
                    "proj_dim":              proj_dim if frozen_encoder is not None else None,
                    "encoder_ckpt":          str(encoder_ckpt) if encoder_ckpt else None,
                    "encoder_pearson_r_wl":  (proxy_predictor.get("pearson_r_wl")
                                             if proxy_predictor else None),
                    "encoder_trained_on":    (proxy_predictor.get("trained_on")
                                             if proxy_predictor else None),
                    "warm_start":            str(initial_policy_ckpt) if initial_policy_ckpt else None,
                },
                reinit=True,
            )
            print(f"  wandb run: {_wrun.url}")

    # Per-benchmark counters for diagnostics (only emitted at end-of-run).
    per_bench_traj_count = {n: 0 for n in benchmarks}
    per_bench_commit_count = {n: 0 for n in benchmarks}

    history = []
    t_start = time.time()
    for it in range(n_iterations):
        t_iter = time.time()
        all_transitions: list[dict] = []
        commit_count = 0
        stop_count = 0      # trajectories where policy explicitly chose STOP
        length_total = 0
        gain_total = 0.0
        best_prefix_total = 0.0
        iter_bench_traj: dict[str, int] = {n: 0 for n in benchmarks}
        iter_bench_commit: dict[str, int] = {n: 0 for n in benchmarks}

        for _ in range(trajectories_per_iter):
            name = rng.choice(benchmarks)
            bc = bench_cache[name]
            # B.3: per-rollout reset bypasses apply_move; rebuild caches.
            bc["state"].pos[:] = bc["init_pos"]
            bc["state"].rebuild_caches()
            seed_macro = rng.choice(bc["movable"])
            env = ChainEnv(
                bc["state"],
                seed_macro,
                rng=rng,
                max_chain_length=max_chain_length,
                max_candidates=max_candidates,
                terminal_commit_bonus=terminal_commit_bonus,
                terminal_reward_mode=terminal_reward_mode,
                gate_mode=effective_gate_mode,
                proxy_predictor=(proxy_predictor
                                 if effective_gate_mode == "predicted_proxy" else None),
                encoder=frozen_encoder,
            )
            traj = collect_rollout(env, policy)
            per_bench_traj_count[name] += 1
            iter_bench_traj[name] += 1
            if not traj:
                continue
            # Track explicit STOP: last action == number of candidates
            last_t = traj[-1]
            if last_t["action"] == last_t["candidates"].shape[0]:
                stop_count += 1

            traj = compute_gae(traj, gamma=gamma, lam=lam)
            all_transitions.extend(traj)
            # env._finalize ran (env.done is True); read its best-prefix
            # outcome directly so the commit accounting matches the same
            # lex (overlap_pairs, overlap_area, third) gate the env
            # enforced. start_hpwl - state.hpwl() works for both gate
            # modes because state.pos has been restored to the committed
            # prefix; under predicted_proxy gate, best_key[2] would be
            # cumulative predicted Δproxy, not HPWL, so subtracting it
            # would give nonsense.
            length_total += env.chain_length
            best_prefix_total += env.best_prefix_index
            committed = env.best_prefix_index > 0
            if committed:
                commit_count += 1
                gain_total += env.start_hpwl - env.state.hpwl()
                per_bench_commit_count[name] += 1
                iter_bench_commit[name] += 1

        metrics = ppo_update(
            policy,
            optimizer,
            all_transitions,
            clip=clip,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            epochs=ppo_epochs,
        )

        n_traj = max(trajectories_per_iter, 1)
        avg_traj_reward = sum(t["reward"] for t in all_transitions) / n_traj
        avg_traj_length  = length_total / n_traj
        avg_best_prefix  = best_prefix_total / n_traj
        commit_rate      = commit_count / n_traj
        explicit_stop_rate = stop_count / n_traj
        avg_gain         = gain_total / max(commit_count, 1)
        iter_time        = time.time() - t_iter
        elapsed          = time.time() - t_start

        record = {
            "iter": it,
            "avg_reward_per_traj": avg_traj_reward,
            "avg_length":          avg_traj_length,
            "commit_rate":         commit_rate,
            "avg_gain":            avg_gain,
            **metrics,
        }
        history.append(record)

        if _wrun is not None:
            log_dict: dict = {
                # ── rollout quality ──────────────────────────────────────
                "train/reward_per_traj":    avg_traj_reward,
                "train/commit_rate":        commit_rate,
                "train/avg_chain_length":   avg_traj_length,
                "train/avg_best_prefix":    avg_best_prefix,
                "train/avg_hpwl_gain":      avg_gain if commit_count > 0 else 0.0,
                "train/explicit_stop_rate": explicit_stop_rate,
                "train/n_transitions":      len(all_transitions),
                # ── PPO losses ───────────────────────────────────────────
                "train/policy_loss":        metrics["policy_loss"],
                "train/value_loss":         metrics["value_loss"],
                "train/entropy":            metrics["entropy"],
                # ── PPO diagnostics ──────────────────────────────────────
                "train/grad_norm":          metrics["grad_norm"],
                "train/clip_fraction":      metrics["clip_fraction"],
                "train/approx_kl":          metrics["approx_kl"],
                # ── timing ───────────────────────────────────────────────
                "train/iter_time_s":        iter_time,
                "train/total_elapsed_s":    elapsed,
            }
            # Per-benchmark commit rate for this iteration
            for name in benchmarks:
                n_b = max(iter_bench_traj[name], 1)
                log_dict[f"bench/{name}/commit_rate"] = iter_bench_commit[name] / n_b
            _wrun.log(log_dict, step=it)

        if it % log_every == 0 or it == n_iterations - 1:
            print(
                f"  it {it:4d}  R/traj={avg_traj_reward:+.4f}  "
                f"len={avg_traj_length:.1f}  commit={commit_rate:.0%}  "
                f"stop={explicit_stop_rate:.0%}  "
                f"gain={avg_gain:+.4f}  "
                f"pi={metrics['policy_loss']:+.3f}  "
                f"vf={metrics['value_loss']:.3f}  "
                f"ent={metrics['entropy']:.3f}  "
                f"kl={metrics['approx_kl']:.4f}  "
                f"[{elapsed:.1f}s]"
            )

    # Per-benchmark commit summary (helps diagnose generalization gaps).
    if len(benchmarks) > 1:
        print(f"  per-benchmark commit rates:")
        for name in benchmarks:
            n_traj = per_bench_traj_count[name]
            n_commit = per_bench_commit_count[name]
            rate = n_commit / max(n_traj, 1)
            print(f"    {name:>10}  {n_commit}/{n_traj} = {rate:.0%}")
        if _wrun is not None:
            _wrun.summary.update({
                f"final_commit_rate/{name}": per_bench_commit_count[name] / max(per_bench_traj_count[name], 1)
                for name in benchmarks
            })

    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict":   policy.state_dict(),
            "policy_class": type(policy).__name__,
            "trained_on":   benchmarks,
            "n_iterations": n_iterations,
            "hidden":       64,
            "encoder_ckpt": str(encoder_ckpt) if encoder_ckpt else None,
            "gnn_H":        gnn_H,
            "proj_dim":     proj_dim if frozen_encoder is not None else None,
            "gate_mode":    effective_gate_mode,
        },
        output_ckpt,
    )
    print(f"  policy -> {output_ckpt}")
    if _wrun is not None:
        _wrun.finish()
    return {"history": history, "policy": policy}


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--benchmark",
        default="ibm01",
        help="Single name, comma-separated list, or 'all' for the 17 IBM benchmarks.",
    )
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--trajectories-per-iter", type=int, default=4)
    p.add_argument("--max-chain-length", type=int, default=8)
    p.add_argument("--max-candidates", type=int, default=8)
    p.add_argument("--terminal-commit-bonus", type=float, default=0.0)
    p.add_argument(
        "--terminal-reward-mode",
        choices=["committed_gain", "hpwl_telescope_legacy"],
        default="committed_gain",
        help="D.1: 'committed_gain' = per-step 0 reward + terminal "
        "= best-prefix gain (recommended). 'hpwl_telescope_legacy' "
        "= per-step -Δhpwl + terminal commit bonus (pre-fix shape).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--initial-policy", default=None, help="Warm-start checkpoint path (optional)")
    p.add_argument("--encoder-ckpt", default=None,
                   help="Path to proxy_cost_encoder.pt. Enables ChainPolicyWithGNN.")
    p.add_argument("--proj-dim", type=int, default=16,
                   help="GNN projection dimension in ChainPolicyWithGNN.")
    p.add_argument("--gate-mode", default="hpwl",
                   choices=["hpwl", "predicted_proxy"],
                   help="Commit gate: hpwl (default) or predicted_proxy (uses encoder head).")
    p.add_argument("--output", default=str(_HERE.parent / "checkpoints" / "chain_policy.pt"))
    p.add_argument("--wandb-project", default=None,
                   help="WandB project name. Omit to disable WandB logging.")
    p.add_argument("--wandb-run-name", default=None,
                   help="WandB run name (auto-generated if omitted).")
    args = p.parse_args()

    benchmarks = parse_benchmarks(args.benchmark)

    train_chain_policy(
        benchmarks,
        n_iterations=args.iterations,
        trajectories_per_iter=args.trajectories_per_iter,
        max_chain_length=args.max_chain_length,
        max_candidates=args.max_candidates,
        terminal_commit_bonus=args.terminal_commit_bonus,
        terminal_reward_mode=args.terminal_reward_mode,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip=args.clip,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        ppo_epochs=args.ppo_epochs,
        output_ckpt=Path(args.output),
        initial_policy_ckpt=Path(args.initial_policy) if args.initial_policy else None,
        encoder_ckpt=Path(args.encoder_ckpt) if args.encoder_ckpt else None,
        proj_dim=args.proj_dim,
        gate_mode=args.gate_mode,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
