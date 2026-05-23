"""Phase 4 — PPO trainer for the LK chain policy.

Algorithm: standard PPO with GAE. Rollouts collected from ChainEnv (HPWL
surrogate reward). Each chain step has a variable number of candidates K,
so transitions are processed individually inside the update — for the
modest batch sizes implied by the plan (~40 transitions/iter), no padding
is needed.

Usage:
    uv run python submissions/lkh/train_policy.py --iterations 200
    uv run python submissions/lkh/train_policy.py --iterations 100 --benchmark ibm03
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

from lkh_model import ChainPolicy

_spec_placer = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
_placer = importlib.util.module_from_spec(_spec_placer)
_spec_placer.loader.exec_module(_placer)

_spec_env = importlib.util.spec_from_file_location("chain_env", str(_HERE / "chain_env.py"))
_env_mod = importlib.util.module_from_spec(_spec_env)
_spec_env.loader.exec_module(_env_mod)
ChainEnv = _env_mod.ChainEnv

from macro_place.loader import load_benchmark_from_dir


# ── Rollout collection ─────────────────────────────────────────────────────

def collect_rollout(env: ChainEnv, policy: ChainPolicy) -> list[dict]:
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

        with torch.no_grad():
            logits, value = policy(global_t, macro_t, cands_t, chain_t)
        probs = F.softmax(logits, dim=-1)
        action = int(torch.multinomial(probs, 1).item())
        log_prob_old = F.log_softmax(logits, dim=-1)[action].detach()

        reward, _done, _info = env.step(action, sp["raw_candidates"])

        transitions.append({
            "global": global_t, "macro": macro_t,
            "candidates": cands_t, "chain": chain_t,
            "action": action,
            "log_prob_old": log_prob_old,
            "value_old": value.detach(),
            "reward": float(reward),
        })
    return transitions


def compute_gae(transitions: list[dict], gamma: float = 0.99,
                lam: float = 0.95) -> list[dict]:
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

def ppo_update(policy: ChainPolicy, optimizer: torch.optim.Optimizer,
               batch: list[dict], clip: float = 0.2, ent_coef: float = 0.01,
               vf_coef: float = 0.5, epochs: int = 4) -> dict:
    if not batch:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    advs = torch.tensor([t["advantage"] for t in batch], dtype=torch.float32)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    for i, t in enumerate(batch):
        t["adv_norm"] = float(advs[i].item())

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    n_updates = 0
    indices = list(range(len(batch)))
    for _ in range(epochs):
        np.random.shuffle(indices)
        for idx in indices:
            t = batch[idx]
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
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.item())
            n_updates += 1

    if n_updates > 0:
        for k in metrics:
            metrics[k] /= n_updates
    return metrics


# ── Training loop ──────────────────────────────────────────────────────────

def train_chain_policy(benchmark_name: str, *, n_iterations: int,
                        trajectories_per_iter: int, max_chain_length: int,
                        max_candidates: int, terminal_commit_bonus: float,
                        seed: int, lr: float, gamma: float, lam: float,
                        clip: float, ent_coef: float, vf_coef: float,
                        ppo_epochs: int, output_ckpt: Path,
                        initial_policy_ckpt: Path | None = None,
                        log_every: int = 5) -> dict:
    import random as _random

    rng = _random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark_name
    benchmark, _plc = load_benchmark_from_dir(str(bench_dir))
    edges, edge_weights = _placer._hard_macro_edges(benchmark)
    base_state = _placer.PlacementState(benchmark, edges, edge_weights)
    init_pos = base_state.pos.copy()
    movable_indices = np.where(base_state.movable)[0].tolist()

    policy = ChainPolicy(hidden=64)
    if initial_policy_ckpt is not None and initial_policy_ckpt.exists():
        ckpt = torch.load(initial_policy_ckpt, map_location="cpu", weights_only=False)
        policy.load_state_dict(ckpt["state_dict"])
        print(f"  warm-started policy from {initial_policy_ckpt}")
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    print(f"=== Phase 4: PPO chain policy ===")
    print(f"  benchmark={benchmark_name}  iterations={n_iterations}  traj/iter={trajectories_per_iter}")
    print(f"  max_chain_length={max_chain_length}  max_candidates={max_candidates}")

    history = []
    t_start = time.time()
    for it in range(n_iterations):
        all_transitions: list[dict] = []
        commit_count = 0
        length_total = 0
        gain_total = 0.0
        for _ in range(trajectories_per_iter):
            base_state.pos[:] = init_pos
            seed_macro = rng.choice(movable_indices)
            env = ChainEnv(
                base_state, seed_macro, rng=rng,
                max_chain_length=max_chain_length,
                max_candidates=max_candidates,
                terminal_commit_bonus=terminal_commit_bonus,
            )
            traj = collect_rollout(env, policy)
            if not traj:
                continue
            traj = compute_gae(traj, gamma=gamma, lam=lam)
            all_transitions.extend(traj)
            # env._finalize ran (env.done is True); inspect the resulting state.
            length_total += env.chain_length
            end_hpwl = env.state.hpwl()
            end_overlaps = env.state.overlap_pairs()
            committed = (
                end_overlaps < env.start_overlaps
                or (end_overlaps == env.start_overlaps and end_hpwl < env.start_hpwl)
            )
            if committed:
                commit_count += 1
                gain_total += env.start_hpwl - end_hpwl

        metrics = ppo_update(policy, optimizer, all_transitions,
                             clip=clip, ent_coef=ent_coef, vf_coef=vf_coef,
                             epochs=ppo_epochs)

        avg_traj_reward = (sum(t["reward"] for t in all_transitions)
                           / max(trajectories_per_iter, 1))
        avg_traj_length = length_total / max(trajectories_per_iter, 1)
        commit_rate = commit_count / max(trajectories_per_iter, 1)
        avg_gain = gain_total / max(commit_count, 1)

        record = {
            "iter": it,
            "avg_reward_per_traj": avg_traj_reward,
            "avg_length": avg_traj_length,
            "commit_rate": commit_rate,
            "avg_gain": avg_gain,
            **metrics,
        }
        history.append(record)

        if it % log_every == 0 or it == n_iterations - 1:
            elapsed = time.time() - t_start
            print(f"  it {it:4d}  R/traj={avg_traj_reward:+.4f}  "
                  f"len={avg_traj_length:.1f}  commit={commit_rate:.0%}  "
                  f"gain={avg_gain:+.4f}  "
                  f"pi={metrics['policy_loss']:+.3f}  "
                  f"vf={metrics['value_loss']:.3f}  "
                  f"ent={metrics['entropy']:.3f}  "
                  f"[{elapsed:.1f}s]")

    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "trained_on": benchmark_name,
        "n_iterations": n_iterations,
        "hidden": 64,
    }, output_ckpt)
    print(f"  policy -> {output_ckpt}")
    return {"history": history, "policy": policy}


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="ibm01")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--trajectories-per-iter", type=int, default=4)
    p.add_argument("--max-chain-length", type=int, default=8)
    p.add_argument("--max-candidates", type=int, default=8)
    p.add_argument("--terminal-commit-bonus", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--initial-policy",
                   default=None,
                   help="Warm-start checkpoint path (optional)")
    p.add_argument("--output",
                   default=str(_HERE / "checkpoints" / "chain_policy.pt"))
    args = p.parse_args()

    train_chain_policy(
        args.benchmark,
        n_iterations=args.iterations,
        trajectories_per_iter=args.trajectories_per_iter,
        max_chain_length=args.max_chain_length,
        max_candidates=args.max_candidates,
        terminal_commit_bonus=args.terminal_commit_bonus,
        seed=args.seed, lr=args.lr, gamma=args.gamma, lam=args.lam,
        clip=args.clip, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        ppo_epochs=args.ppo_epochs,
        output_ckpt=Path(args.output),
        initial_policy_ckpt=Path(args.initial_policy) if args.initial_policy else None,
    )


if __name__ == "__main__":
    main()
