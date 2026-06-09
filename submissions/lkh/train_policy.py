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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

from lkh_model import ChainPolicy
from train import parse_benchmarks   # shared CLI helper

_spec_placer = importlib.util.spec_from_file_location("lkh_placer", str(_HERE / "placer.py"))
_placer = importlib.util.module_from_spec(_spec_placer)
_spec_placer.loader.exec_module(_placer)

_spec_env = importlib.util.spec_from_file_location("chain_env", str(_HERE / "chain_env.py"))
_env_mod = importlib.util.module_from_spec(_spec_env)
_spec_env.loader.exec_module(_env_mod)
ChainEnv = _env_mod.ChainEnv

from macro_place.loader import load_benchmark_from_dir


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

def collect_rollout(env: ChainEnv, policy: ChainPolicy, *,
                    encoder=None, enc_mod=None, enc_grid: int = 64) -> list[dict]:
    """One chain episode using stochastic policy sampling.

    Step 3 (encoder wiring): when ``encoder`` is supplied we feed the policy
    the GNN+CNN embeddings alongside the hand-crafted features. The GNN inputs
    (node features + edges) are built ONCE at chain start and reused across the
    chain's steps — identical to the inference-time caching in
    ``ChainEnv.state_for_policy`` — while the CNN per-macro raster is rebuilt
    each step. We store the *raw* encoder inputs (not the embeddings) in each
    transition so ``ppo_update`` can recompute the encoder forward WITH
    gradients and actually train it; the no-grad forward here is only for
    action sampling. The encoder is intentionally NOT attached to ``env`` so
    its inference path (detached, numpy) and this training path don't collide.
    """
    transitions: list[dict] = []
    enc_nf = enc_ei = enc_ea = None
    if encoder is not None:
        enc_nf = enc_mod.build_node_features(env.state)
        enc_ei, enc_ea = enc_mod.build_edge_index_and_attr(env.state)
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

        enc_g_t = enc_m_t = None
        enc_canvas = None
        enc_macro_idx = None
        if encoder is not None:
            enc_macro_idx = int(env.current_macro)
            enc_canvas = enc_mod.rasterize_canvas(
                env.state, idx=enc_macro_idx, grid_size=enc_grid)
            with torch.no_grad():
                _h, _g = encoder.gnn(enc_nf, enc_ei, enc_ea)
                _cnn = encoder.cnn(enc_canvas)
                enc_g_t = torch.cat([_g, _cnn], dim=-1)
                enc_m_t = _h[enc_macro_idx]

        with torch.no_grad():
            logits, value = policy(global_t, macro_t, cands_t, chain_t,
                                   encoder_global=enc_g_t, encoder_macro=enc_m_t)
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
            # Step 3: raw encoder inputs for the grad-enabled recompute.
            "enc_nf": enc_nf, "enc_ei": enc_ei, "enc_ea": enc_ea,
            "enc_canvas": enc_canvas, "enc_macro_idx": enc_macro_idx,
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
               vf_coef: float = 0.5, epochs: int = 4, encoder=None) -> dict:
    if not batch:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    advs = torch.tensor([t["advantage"] for t in batch], dtype=torch.float32)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    for i, t in enumerate(batch):
        t["adv_norm"] = float(advs[i].item())

    # Step 3: when an encoder is co-trained, its parameters join the policy's
    # in the gradient-clip set (the optimizer already covers both).
    clip_params = list(policy.parameters())
    if encoder is not None:
        clip_params = clip_params + list(encoder.parameters())

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    n_updates = 0
    indices = list(range(len(batch)))
    for _ in range(epochs):
        np.random.shuffle(indices)
        for idx in indices:
            t = batch[idx]
            # Step 3: recompute encoder embeddings WITH gradients from the raw
            # inputs stored at rollout time. This is what actually trains the
            # encoder — the rollout's forward was no-grad (sampling only).
            enc_g = enc_m = None
            if encoder is not None and t.get("enc_nf") is not None:
                _h, _g = encoder.gnn(t["enc_nf"], t["enc_ei"], t["enc_ea"])
                _cnn = encoder.cnn(t["enc_canvas"])
                enc_g = torch.cat([_g, _cnn], dim=-1)
                enc_m = _h[t["enc_macro_idx"]]
            logits, value = policy(t["global"], t["macro"], t["candidates"],
                                   t["chain"], encoder_global=enc_g,
                                   encoder_macro=enc_m)
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
            nn.utils.clip_grad_norm_(clip_params, 0.5)
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

def train_chain_policy(benchmarks: list[str], *, n_iterations: int,
                        trajectories_per_iter: int, max_chain_length: int,
                        max_candidates: int, terminal_commit_bonus: float,
                        seed: int, lr: float, gamma: float, lam: float,
                        clip: float, ent_coef: float, vf_coef: float,
                        ppo_epochs: int, output_ckpt: Path,
                        initial_policy_ckpt: Path | None = None,
                        log_every: int = 5,
                        terminal_reward_mode: str = "committed_gain",
                        # MaskRegulate flags
                        gate_mode: str = "hpwl",
                        reg_weight: float = 0.0,
                        use_wiremask: bool = False,
                        use_position_mask: bool = False,
                        approximator_ckpt: str | None = None,
                        use_reg_feature: bool = False,
                        use_encoder: bool = False,
                        encoder_hidden: int = 64,
                        encoder_grid: int = 64,
                        # Session building-block opt-ins
                        feature_mode: str = "handcrafted",
                        encoder_kind: str = "gnn",
                        encoder_ckpt: str | None = None,
                        scalar_lam: float = 0.01,
                        seed_mode: str = "heuristic",
                        seed_head_hidden: int = 64,
                        encoder_fine_tune_in_ppo: bool = False) -> dict:
    """PPO training across one or more benchmarks. Each trajectory samples
    a random benchmark from ``benchmarks`` (plan §4.3 semantics).

    Step 3 (encoder wiring): when ``use_encoder`` is True a ``StateEncoder``
    (GNN over the netlist + CNN over the per-macro mask raster) is co-trained
    with the policy and its weights are saved into the same checkpoint, so the
    placer's loader can reconstruct it and feed its embeddings at inference.

    Session building-block opt-ins (all default off → legacy behavior):
    - ``feature_mode="encoder"``: load encoder ckpt, build per-rollout
      encoder cache, use encoder-aware ChainPolicy dims.
    - ``gate_mode="scalar_penalty"`` + ``approximator_ckpt``: scalar gate.
    - ``terminal_reward_mode="predicted_proxy_with_postleg"``: post-leg
      reward via predicted Δproxy.
    - ``seed_mode="policy"``: add a SeedSelectionHead, train it jointly
      with the policy.
    """
    import random as _random

    if not benchmarks:
        raise ValueError("train_chain_policy: benchmarks list is empty")

    rng = _random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"=== Phase 4: PPO chain policy ===")
    print(f"  benchmarks={benchmarks}  iterations={n_iterations}  "
          f"traj/iter={trajectories_per_iter}")
    print(f"  max_chain_length={max_chain_length}  max_candidates={max_candidates}")
    print(f"  feature_mode={feature_mode}  gate_mode={gate_mode}  "
          f"seed_mode={seed_mode}  reward={terminal_reward_mode}")
    print(f"  pre-loading {len(benchmarks)} benchmark(s) into RAM...")
    bench_cache = _load_benchmark_cache(benchmarks)

    # Task 1c: cand_dim follows the approximator's schema so the policy and
    # approximator agree on per-candidate feature size. Default 16; bumps
    # to 17 when use_reg_feature is on.
    from lkh_model import FEATURE_DIM
    cand_dim = (FEATURE_DIM + 1) if use_reg_feature else FEATURE_DIM

    # ── Encoder: MaskRegulate (use_encoder) path is the primary integration.
    # If our session building-block path (feature_mode="encoder") is requested,
    # promote it to use_encoder so the two stay in sync. The runtime-wrapper
    # files (encoder_runtime.py / encoder_runtime_gnncnn.py) are alternative
    # *inference-time* loaders for the placer; PPO co-training always goes
    # through the StateEncoder directly here.
    if feature_mode == "encoder" and not use_encoder:
        use_encoder = True
        print(f"  feature_mode='encoder' → use_encoder=True (joint path)")

    encoder = None
    enc_mod = None
    encoder_global_dim = 0
    encoder_macro_dim = 0
    if use_encoder:
        _spec_enc = importlib.util.spec_from_file_location(
            "lkh_encoder", str(_HERE / "encoder.py"))
        enc_mod = importlib.util.module_from_spec(_spec_enc)
        _spec_enc.loader.exec_module(enc_mod)
        encoder = enc_mod.StateEncoder(hidden_dim=encoder_hidden,
                                       num_gnn_layers=3, grid_size=encoder_grid)
        # MaskRegulate trains encoder by default; the session opt-in
        # ``encoder_fine_tune_in_ppo`` keeps backward compatibility with
        # the frozen-encoder convention from the wrapper path.
        if encoder_fine_tune_in_ppo or not Path(encoder_ckpt or "").exists():
            encoder.train()
        else:
            # If a pretrained encoder was supplied and fine-tune is off,
            # honor the freeze (matches the session-building-block default).
            try:
                ck = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
                if "state_dict" in ck:
                    encoder.load_state_dict(ck["state_dict"], strict=False)
                    print(f"  encoder warm-started from {encoder_ckpt}")
            except Exception as _e:
                print(f"  encoder ckpt unreadable; starting fresh ({_e})")
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad_(False)
        encoder_global_dim = 2 * encoder_hidden
        encoder_macro_dim = encoder_hidden
        n_enc_params = sum(p.numel() for p in encoder.parameters())
        print(f"  encoder ENABLED: StateEncoder(hidden={encoder_hidden}, "
              f"grid={encoder_grid}) — {n_enc_params:,} params; "
              f"policy encoder dims = global:{encoder_global_dim} "
              f"macro:{encoder_macro_dim}")

    policy = ChainPolicy(hidden=64, cand_dim=cand_dim,
                         encoder_global_dim=encoder_global_dim,
                         encoder_macro_dim=encoder_macro_dim)
    if initial_policy_ckpt is not None and initial_policy_ckpt.exists():
        ckpt = torch.load(initial_policy_ckpt, map_location="cpu", weights_only=False)
        warm_cand_dim = int(ckpt.get("cand_dim", FEATURE_DIM))
        warm_enc_g = int(ckpt.get("encoder_global_dim", 0))
        warm_enc_m = int(ckpt.get("encoder_macro_dim", 0))
        if (warm_cand_dim == cand_dim and warm_enc_g == encoder_global_dim
                and warm_enc_m == encoder_macro_dim):
            policy.load_state_dict(ckpt["state_dict"])
            print(f"  warm-started policy from {initial_policy_ckpt}")
            if encoder is not None and ckpt.get("encoder_state_dict") is not None:
                encoder.load_state_dict(ckpt["encoder_state_dict"])
                print(f"  warm-started encoder from {initial_policy_ckpt}")
        else:
            print(f"  skip warm-start: ckpt shape "
                  f"(cand_dim={warm_cand_dim}, enc_g={warm_enc_g}, "
                  f"enc_m={warm_enc_m}) != run "
                  f"(cand_dim={cand_dim}, enc_g={encoder_global_dim}, "
                  f"enc_m={encoder_macro_dim})")

    # ── Approximator: needed for MaskRegulate's predicted_proxy gate, the
    # session's scalar_penalty gate, or the post-leg predicted-proxy reward.
    approx_bundle = None
    if (approximator_ckpt is not None and Path(approximator_ckpt).exists()) or \
            gate_mode in ("predicted_proxy", "scalar_penalty") or \
            terminal_reward_mode == "predicted_proxy_with_postleg":
        a_ckpt = Path(approximator_ckpt) if approximator_ckpt else (
            _HERE / "checkpoints" / "cost_approximator.pt"
        )
        approx_bundle = _placer._load_cost_approximator(a_ckpt)
        if approx_bundle is not None:
            print(f"  approximator loaded "
                  f"(r={approx_bundle.get('pearson_r_val', float('nan')):.3f}, "
                  f"use_reg_feature={approx_bundle.get('use_reg_feature', False)})")
        else:
            print(f"  using fallback scorer")
    # Alias for downstream code that referenced our earlier name.
    approximator = approx_bundle

    # ── Optional learned seed-selection head (session building-block).
    # Requires encoder embeddings; falls back to heuristic when disabled.
    seed_head = None
    if seed_mode == "policy":
        if not use_encoder or encoder is None:
            print(f"  using default seed selection")
            seed_mode = "heuristic"
        else:
            from seed_head import SeedSelectionHead
            seed_head = SeedSelectionHead(
                per_macro_dim=encoder_macro_dim,
                global_dim=encoder_global_dim,
                hidden=seed_head_hidden,
            )
            print(f"  seed head built")

    # Optimizer: union of policy + encoder (if joint) + seed_head params.
    opt_params = list(policy.parameters())
    if encoder is not None and (encoder_fine_tune_in_ppo or encoder.training):
        opt_params = opt_params + list(encoder.parameters())
    if seed_head is not None:
        opt_params = opt_params + list(seed_head.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    # Per-benchmark counters for diagnostics (only emitted at end-of-run).
    per_bench_traj_count = {n: 0 for n in benchmarks}
    per_bench_commit_count = {n: 0 for n in benchmarks}

    # Tier-1 Fix C: track the best policy by EMA-smoothed commit rate, not
    # the last iteration's weights. PPO can collapse late in training
    # (entropy → 0, policy outputs the same action everywhere) and the
    # last-iter checkpoint then ships that collapsed policy. The README
    # explicitly lists "best-policy tracking" as deferred work; this is it.
    #
    # Commit rate (per-iter fraction of trajectories that committed an
    # improvement) is the right proxy for "how often is this policy doing
    # useful work" — higher = better. Smooth with an EMA so a single
    # lucky iteration doesn't pin the best checkpoint.
    best_commit_ema = -float("inf")
    best_state_dict = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    best_encoder_state_dict = (
        {k: v.detach().clone() for k, v in encoder.state_dict().items()}
        if encoder is not None else None)
    best_iter = 0
    commit_ema: float | None = None
    EMA_ALPHA = 0.1
    # Skip the first few iterations so the EMA isn't pinned to a noisy
    # warm-start. 20 iters is ~2-5% of typical runs.
    WARMUP_ITERS = 20

    history = []
    t_start = time.time()
    for it in range(n_iterations):
        all_transitions: list[dict] = []
        commit_count = 0
        length_total = 0
        gain_total = 0.0
        for _ in range(trajectories_per_iter):
            name = rng.choice(benchmarks)
            bc = bench_cache[name]
            # B.3: per-rollout reset bypasses apply_move; rebuild caches.
            bc["state"].pos[:] = bc["init_pos"]
            bc["state"].rebuild_caches()

            # Fix 2: refresh encoder cache once per rollout (frozen
            # encoder, no_grad). per_node and graph_vec are reused across
            # every step of this episode.
            encoder_cache = None
            if encoder is not None:
                per_node, graph_vec = encode_fn(bc["state"], encoder, with_grad=False)
                encoder_cache = {"per_node": per_node, "graph_vec": graph_vec}

            # Fix 3: seed pick. Either heuristic (default) or seed-head.
            if seed_head is not None and encoder_cache is not None:
                from seed_head import choose_seed_by_policy
                seed_macro, _logp = choose_seed_by_policy(
                    encoder_cache["per_node"], encoder_cache["graph_vec"],
                    bc["state"].movable, seed_head,
                    stochastic=True, rng=rng,
                )
            else:
                seed_macro = rng.choice(bc["movable"])

            env = ChainEnv(
                bc["state"], seed_macro, rng=rng,
                max_chain_length=max_chain_length,
                max_candidates=max_candidates,
                terminal_commit_bonus=terminal_commit_bonus,
                terminal_reward_mode=terminal_reward_mode,
                gate_mode=gate_mode,
                approximator=approx_bundle,
                reg_weight=reg_weight,
                use_wiremask=use_wiremask,
                use_position_mask=use_position_mask,
                feature_mode=feature_mode,
                lam=scalar_lam,
            )
            traj = collect_rollout(env, policy, encoder=encoder,
                                   enc_mod=enc_mod, enc_grid=encoder_grid)
            per_bench_traj_count[name] += 1
            if not traj:
                continue
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
            committed = env.best_prefix_index > 0
            if committed:
                commit_count += 1
                gain_total += env.start_hpwl - env.state.hpwl()
                per_bench_commit_count[name] += 1

        metrics = ppo_update(policy, optimizer, all_transitions,
                             clip=clip, ent_coef=ent_coef, vf_coef=vf_coef,
                             epochs=ppo_epochs, encoder=encoder)

        avg_traj_reward = (sum(t["reward"] for t in all_transitions)
                           / max(trajectories_per_iter, 1))
        avg_traj_length = length_total / max(trajectories_per_iter, 1)
        commit_rate = commit_count / max(trajectories_per_iter, 1)
        avg_gain = gain_total / max(commit_count, 1)

        # Tier-1 Fix C: smooth commit rate via EMA, and snapshot the policy
        # at the running peak. After WARMUP_ITERS we treat the EMA as
        # stable enough to act on; before that we just initialize it.
        commit_ema = (commit_rate if commit_ema is None
                       else EMA_ALPHA * commit_rate + (1 - EMA_ALPHA) * commit_ema)
        if it >= WARMUP_ITERS and commit_ema > best_commit_ema:
            best_commit_ema = commit_ema
            best_iter = it
            best_state_dict = {k: v.detach().clone()
                               for k, v in policy.state_dict().items()}
            if encoder is not None:
                best_encoder_state_dict = {k: v.detach().clone()
                                           for k, v in encoder.state_dict().items()}

        record = {
            "iter": it,
            "avg_reward_per_traj": avg_traj_reward,
            "avg_length": avg_traj_length,
            "commit_rate": commit_rate,
            "commit_ema": commit_ema,
            "avg_gain": avg_gain,
            **metrics,
        }
        history.append(record)

        if it % log_every == 0 or it == n_iterations - 1:
            elapsed = time.time() - t_start
            star = " *" if it == best_iter else ""
            print(f"  it {it:4d}  R/traj={avg_traj_reward:+.4f}  "
                  f"len={avg_traj_length:.1f}  commit={commit_rate:.0%}"
                  f"(ema={commit_ema:.0%}){star}  "
                  f"gain={avg_gain:+.4f}  "
                  f"pi={metrics['policy_loss']:+.3f}  "
                  f"vf={metrics['value_loss']:.3f}  "
                  f"ent={metrics['entropy']:.3f}  "
                  f"[{elapsed:.1f}s]")

    # Per-benchmark commit summary (helps diagnose generalization gaps).
    if len(benchmarks) > 1:
        print(f"  per-benchmark commit rates:")
        for name in benchmarks:
            n_traj = per_bench_traj_count[name]
            n_commit = per_bench_commit_count[name]
            rate = n_commit / max(n_traj, 1)
            print(f"    {name:>10}  {n_commit}/{n_traj} = {rate:.0%}")

    # Tier-1 Fix C: ship the best-by-commit-EMA snapshot, not the last
    # iteration's weights. Fall back to the last state only if we never
    # cleared the warmup gate (very short training runs).
    if best_commit_ema > -float("inf"):
        policy.load_state_dict(best_state_dict)
        ship_state_dict = best_state_dict
        ship_encoder_state_dict = best_encoder_state_dict
        print(f"  shipping best-by-commit-EMA policy "
              f"(iter {best_iter}, EMA={best_commit_ema:.0%})")
    else:
        ship_state_dict = policy.state_dict()
        ship_encoder_state_dict = (encoder.state_dict()
                                   if encoder is not None else None)
        print(f"  shipping last-iter policy (no warmup-cleared best yet)")

    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": ship_state_dict,
        "trained_on": benchmarks,
        "n_iterations": n_iterations,
        "hidden": 64,
        "best_commit_ema": best_commit_ema,
        "best_iter": best_iter,
        # MaskRegulate schema flags: cand_dim, gate_mode, reg/mask flags,
        # encoder config — so the placer's loader rebuilds the policy at
        # the right shape and routes ChainEnv inputs correctly.
        "cand_dim": cand_dim,
        "use_reg_feature": bool(use_reg_feature),
        "gate_mode": gate_mode,
        "reg_weight": float(reg_weight),
        "use_wiremask": bool(use_wiremask),
        "use_position_mask": bool(use_position_mask),
        "encoder_global_dim": encoder_global_dim,
        "encoder_macro_dim": encoder_macro_dim,
        "encoder_hidden": int(encoder_hidden) if encoder is not None else 0,
        "encoder_grid_size": int(encoder_grid) if encoder is not None else 0,
        "encoder_state_dict": ship_encoder_state_dict,
        # Session building-block dims (for ChainPolicy reconstruction
        # under the runtime-wrapper code path).
        "global_dim": policy.global_dim,
        "macro_dim": policy.macro_dim,
        "chain_dim": policy.chain_dim,
        "feature_mode": feature_mode,
    }
    if seed_head is not None:
        payload["seed_head_state_dict"] = seed_head.state_dict()
        payload["seed_head_per_macro_dim"] = seed_head.per_macro_dim
        payload["seed_head_global_dim"] = seed_head.global_dim
        payload["seed_head_hidden"] = seed_head.hidden
    torch.save(payload, output_ckpt)
    print(f"  policy -> {output_ckpt}  (cand_dim={cand_dim}, "
          f"gate_mode={gate_mode}, reg_weight={reg_weight}, "
          f"use_wiremask={use_wiremask}, use_position_mask={use_position_mask}, "
          f"encoder={'on' if encoder is not None else 'off'}, "
          f"seed_head={'on' if seed_head is not None else 'off'})")
    return {"history": history, "policy": policy,
            "encoder": encoder, "seed_head": seed_head}


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="ibm01",
                   help="Single name, comma-separated list, or 'all' for the 17 IBM benchmarks.")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--trajectories-per-iter", type=int, default=4)
    p.add_argument("--max-chain-length", type=int, default=8)
    p.add_argument("--max-candidates", type=int, default=8)
    p.add_argument("--terminal-commit-bonus", type=float, default=0.0)
    p.add_argument("--terminal-reward-mode",
                   choices=["committed_gain", "hpwl_telescope_legacy",
                            "predicted_proxy_with_postleg"],
                   default="committed_gain",
                   help="D.1 + Fix 4: 'committed_gain' = per-step 0 + terminal "
                        "= best-prefix gain (default). 'hpwl_telescope_legacy' "
                        "= per-step -Δhpwl + terminal bonus. "
                        "'predicted_proxy_with_postleg' = post-leg predicted Δproxy "
                        "reward (Fix 4, poster-aligned).")
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
    # MaskRegulate encoder wiring: co-train the GNN+CNN StateEncoder with
    # the policy. Default off preserves the legacy hand-crafted path.
    p.add_argument("--use-encoder", action="store_true",
                   help="Co-train the GNN+CNN StateEncoder and feed its "
                        "embeddings to the policy (and at inference).")
    p.add_argument("--encoder-hidden", type=int, default=64,
                   help="StateEncoder hidden dim (GNN + CNN). Default 64.")
    p.add_argument("--encoder-grid", type=int, default=64,
                   help="Per-macro mask raster resolution for the CNN. Default 64.")
    # Session building-block flags. Defaults preserve legacy behavior.
    p.add_argument("--feature-mode",
                   choices=["handcrafted", "encoder"],
                   default="handcrafted",
                   help="'encoder' uses the GNN encoder embedding for features.")
    p.add_argument("--encoder-kind",
                   choices=["gnn", "gnncnn"],
                   default="gnn")
    p.add_argument("--encoder-ckpt", default=None,
                   help="Path to encoder checkpoint (default: checkpoints/encoder.pt).")
    p.add_argument("--approximator-ckpt", default=None,
                   help="Path to cost approximator (default: checkpoints/cost_approximator.pt).")
    # gate-mode supports MaskRegulate's hpwl/predicted_proxy + the session
    # building-block 'scalar_penalty' (single scalar score).
    p.add_argument("--gate-mode",
                   choices=["hpwl", "predicted_proxy", "scalar_penalty"],
                   default="hpwl",
                   help="Third lex coord of the commit gate, or scalar "
                        "predicted_Δproxy + lam × n_new_overlap_pairs.")
    p.add_argument("--scalar-lam", type=float, default=0.01,
                   help="Overlap penalty weight for the scalar gate.")
    p.add_argument("--seed-mode",
                   choices=["heuristic", "policy"],
                   default="heuristic",
                   help="'policy' adds a learned SeedSelectionHead.")
    p.add_argument("--encoder-fine-tune-in-ppo", action="store_true",
                   help="Experimental: also update encoder parameters during "
                        "PPO. Default off — encoder is frozen after regression "
                        "pretraining for stability.")
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
        seed=args.seed, lr=args.lr, gamma=args.gamma, lam=args.lam,
        clip=args.clip, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        ppo_epochs=args.ppo_epochs,
        output_ckpt=Path(args.output),
        initial_policy_ckpt=Path(args.initial_policy) if args.initial_policy else None,
        # MaskRegulate
        use_encoder=args.use_encoder,
        encoder_hidden=args.encoder_hidden,
        encoder_grid=args.encoder_grid,
        # Session building-block flags
        feature_mode=args.feature_mode,
        encoder_kind=args.encoder_kind,
        encoder_ckpt=args.encoder_ckpt,
        approximator_ckpt=args.approximator_ckpt,
        gate_mode=args.gate_mode,
        scalar_lam=args.scalar_lam,
        seed_mode=args.seed_mode,
        encoder_fine_tune_in_ppo=args.encoder_fine_tune_in_ppo,
    )


if __name__ == "__main__":
    main()
