"""S6 — Decision-Distance Vector (DDV) fingerprinting.

The ModelDiff signal (Shah et al., ICML 2023): for each probe input `x` and
each of K random unit-norm directions `d`, find the smallest perturbation
magnitude `α` such that the model's prediction at `x + α·d` differs from
its prediction at `x`. The vector `[α₁, α₂, …, α_K]` is the model's
decision-distance vector at `x` — a local fingerprint of its boundary
geometry.

A stolen model (especially a distilled student) inherits the target's
boundary geometry, so its DDV correlates with the target's DDV on the same
inputs and directions. An independent model — even one trained on identical
data with the same recipe (the "evil twin") — converges to a different
solution and has a systematically different DDV.

We use a fixed grid of `α`s for efficiency (avoiding per-direction binary
search). Within a shard, the target's DDV is computed once per probe split
and re-used for every suspect, so the per-suspect cost is one DDV pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_directions(
    n_samples: int, n_per_sample: int, image_shape: tuple[int, int, int],
    seed: int, device: str,
) -> torch.Tensor:
    """Random unit-norm directions in input space, shape [N, K, C, H, W]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = torch.randn(n_samples, n_per_sample, *image_shape, generator=g)
    d = d.to(device)
    flat = d.reshape(n_samples, n_per_sample, -1)
    norm = flat.norm(dim=2).clamp_min(1e-9)
    d = d / norm.view(n_samples, n_per_sample, 1, 1, 1)
    return d


@torch.no_grad()
def compute_ddv(
    model: nn.Module,
    probe_x: torch.Tensor,
    anchor_pred: torch.Tensor,
    directions: torch.Tensor,
    alphas: torch.Tensor,
    device: str,
    chunk: int = 512,
) -> torch.Tensor:
    """Sweep alphas in increasing order; first crossing wins.

    Returns:
        ddv: [N, K] smallest α at which model's prediction differs from
             `anchor_pred[n]`. If no flip is observed within the alpha grid,
             we record `alphas[-1]` (the maximum tested perturbation).
    """
    model.eval()
    N, K = directions.size(0), directions.size(1)
    alpha_max = float(alphas[-1].item())
    ddv = torch.full((N, K), alpha_max, device=device)

    for alpha in alphas:
        x_pert = probe_x.unsqueeze(1) + float(alpha.item()) * directions
        x_flat = x_pert.reshape(N * K, *probe_x.shape[1:])

        preds_chunks = []
        for i in range(0, x_flat.size(0), chunk):
            logits = model(x_flat[i:i + chunk])
            preds_chunks.append(logits.argmax(dim=1))
        preds = torch.cat(preds_chunks, dim=0).view(N, K)

        flipped = preds != anchor_pred.unsqueeze(1)
        not_yet = ddv >= (alpha_max - 1e-9)
        update = flipped & not_yet
        ddv[update] = float(alpha.item())

    return ddv


def ddv_similarity_features(
    ddv_target: torch.Tensor,
    ddv_suspect: torch.Tensor,
    prefix: str,
    alpha_max: float,
) -> dict[str, float]:
    """Three summary statistics comparing target's DDV with suspect's DDV.

        s6_ddv_corr_{prefix}        mean per-sample Pearson correlation across
                                    the K directions (the canonical ModelDiff
                                    similarity)
        s6_ddv_cos_{prefix}         cosine similarity of the flattened DDV
                                    matrices (more robust to scale differences)
        s6_ddv_flip_agree_{prefix}  fraction of probes where target and
                                    suspect agree on whether *any* direction
                                    flipped the prediction (gross boundary
                                    presence/absence agreement)
    """
    eps = 1e-9
    # Per-sample Pearson over K
    tm = ddv_target.mean(dim=1, keepdim=True)
    sm = ddv_suspect.mean(dim=1, keepdim=True)
    td = ddv_target - tm
    sd = ddv_suspect - sm
    cov = (td * sd).sum(dim=1)
    ts = td.pow(2).sum(dim=1).clamp_min(eps).sqrt()
    ss = sd.pow(2).sum(dim=1).clamp_min(eps).sqrt()
    per_sample = cov / (ts * ss)
    valid = torch.isfinite(per_sample)
    ddv_corr = float(per_sample[valid].mean().item()) if valid.any() else 0.0

    # Overall cosine
    t_flat = ddv_target.reshape(-1)
    s_flat = ddv_suspect.reshape(-1)
    cos = F.cosine_similarity(t_flat.unsqueeze(0), s_flat.unsqueeze(0)).item()

    # Flip-set agreement (did target/suspect have *any* boundary crossing?)
    target_any = (ddv_target < alpha_max - 1e-9).any(dim=1)
    suspect_any = (ddv_suspect < alpha_max - 1e-9).any(dim=1)
    flip_agree = float((target_any == suspect_any).float().mean().item()) \
        if ddv_target.size(0) > 0 else 0.0

    return {
        f"s6_ddv_corr_{prefix}": float(ddv_corr),
        f"s6_ddv_cos_{prefix}": float(cos),
        f"s6_ddv_flip_agree_{prefix}": float(flip_agree),
    }
