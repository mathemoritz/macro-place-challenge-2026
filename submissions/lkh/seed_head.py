"""Learned seed-selection head.

The chain policy decides three things: which macro to start a chain
from (seed), which action to take at each step, and when to stop. The
existing ``ChainPolicy`` (lkh_model.py) handles action + STOP via a
single softmax over K+1 logits. Seed selection defaults to an
HPWL-weighted heuristic in placer.py.

This module adds an optional learned seed head as a separate
``nn.Module`` so ``ChainPolicy`` stays untouched. The head consumes
the encoder's per-macro embeddings + the global vector and emits one
score per macro. At chain spawn time the placer (or PPO rollout)
consults the head, samples or argmaxes, and uses the chosen macro as
the seed.

Storage: the seed head's weights are co-saved inside ``chain_policy.pt``
under the key ``seed_head_state_dict`` so the placer loads them in one
call. A standalone ``save_seed_head`` / ``load_seed_head`` is also
provided for unit-test convenience.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeedSelectionHead(nn.Module):
    """Per-macro seed score conditioned on encoder embeddings.

    Input layout per macro: ``cat(per_node[m], broadcast(global_vec))``.
    Two-layer MLP → single scalar logit per macro. We mask non-movable
    macros to ``-inf`` so softmax respects movability.
    """

    def __init__(self, per_macro_dim: int, global_dim: int, hidden: int = 64):
        super().__init__()
        self.per_macro_dim = int(per_macro_dim)
        self.global_dim = int(global_dim)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.per_macro_dim + self.global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, per_node: torch.Tensor, global_vec: torch.Tensor,
                movable_mask: torch.Tensor) -> torch.Tensor:
        """per_node [N, per_macro_dim], global_vec [global_dim],
        movable_mask [N] (bool/int). Returns logits [N] with ``-inf`` at
        non-movable positions.
        """
        N = per_node.shape[0]
        g = global_vec.unsqueeze(0).expand(N, -1)
        x = torch.cat([per_node, g], dim=-1)
        logits = self.net(x).squeeze(-1)
        # Mask non-movable to -inf so softmax + multinomial both respect it.
        mov = movable_mask.to(dtype=torch.bool)
        neg_inf = torch.full_like(logits, float("-inf"))
        logits = torch.where(mov, logits, neg_inf)
        return logits


def sample_seed(logits: torch.Tensor, rng) -> tuple[int, torch.Tensor]:
    """Stochastic seed pick. Returns (macro_idx, log_prob).

    ``rng`` accepted for signature consistency with the placer's existing
    callers; we use ``torch.multinomial`` which uses the torch global RNG.
    Callers that want determinism should ``torch.manual_seed`` upstream.
    """
    probs = F.softmax(logits, dim=-1)
    macro_idx = int(torch.multinomial(probs, 1).item())
    log_prob = F.log_softmax(logits, dim=-1)[macro_idx].detach()
    return macro_idx, log_prob


def argmax_seed(logits: torch.Tensor) -> int:
    """Deterministic (inference-time) seed pick."""
    return int(torch.argmax(logits).item())


def save_seed_head(head: SeedSelectionHead, path: Path) -> None:
    """Standalone save (mostly for unit tests; production co-saves with policy)."""
    payload = {
        "state_dict": head.state_dict(),
        "per_macro_dim": head.per_macro_dim,
        "global_dim": head.global_dim,
        "hidden": head.hidden,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_seed_head(path: Path) -> dict | None:
    """Standalone loader. See ``rebuild_seed_head_from_policy_ckpt`` for the
    co-stored variant used by the placer."""
    path = Path(path)
    if not path.exists():
        return None
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    head = SeedSelectionHead(
        per_macro_dim=int(ckpt["per_macro_dim"]),
        global_dim=int(ckpt["global_dim"]),
        hidden=int(ckpt.get("hidden", 64)),
    )
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return {
        "head": head,
        "per_macro_dim": head.per_macro_dim,
        "global_dim": head.global_dim,
    }


def rebuild_seed_head_from_policy_ckpt(policy_ckpt: dict) -> SeedSelectionHead | None:
    """Reconstruct the seed head from a ``chain_policy.pt`` ckpt that
    co-stored its weights. Returns ``None`` if absent.

    The PPO trainer writes ``seed_head_state_dict``, ``seed_head_per_macro_dim``,
    ``seed_head_global_dim``, ``seed_head_hidden`` into the policy ckpt
    when seed-selection training is on.
    """
    sd = policy_ckpt.get("seed_head_state_dict")
    if sd is None:
        return None
    head = SeedSelectionHead(
        per_macro_dim=int(policy_ckpt["seed_head_per_macro_dim"]),
        global_dim=int(policy_ckpt["seed_head_global_dim"]),
        hidden=int(policy_ckpt.get("seed_head_hidden", 64)),
    )
    head.load_state_dict(sd)
    head.eval()
    return head


def choose_seed_by_policy(per_node: torch.Tensor, global_vec: torch.Tensor,
                            movable_mask: np.ndarray,
                            head: SeedSelectionHead, *,
                            stochastic: bool = False, rng=None
                            ) -> tuple[int, torch.Tensor | None]:
    """End-to-end seed selection helper used by the placer + rollout collector.

    Returns ``(macro_idx, log_prob_or_None)``. ``log_prob`` is non-None
    only in stochastic mode (where it's needed for the PPO trajectory's
    log-prob).
    """
    mov_t = torch.tensor(movable_mask, dtype=torch.bool)
    with torch.no_grad() if not stochastic else torch.enable_grad():
        logits = head(per_node, global_vec, mov_t)
    if stochastic:
        return sample_seed(logits, rng)
    return argmax_seed(logits), None
