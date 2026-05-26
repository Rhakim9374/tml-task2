"""S5 — decision boundary similarity via input gradients and adversarial transfer.

Two models with similar decision functions respond similarly to input
perturbations. Gradient-cosine measures this locally (same direction of
maximum loss increase); adversarial transfer rate measures it for actual
small input attacks (a perturbation that flips the target's prediction is
also likely to flip a stolen model's prediction).

Both metrics use the *target's predicted class* as the pseudo-label, so the
features work on any probe set — labeled (train/holdout/test) or OOD
(CIFAR-10 for our case).

Reference: ModelDiff (Shah et al., ICML 2023) — decision-distance-vector
similarity is one of the strongest fingerprinting signals known to date.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def grad_and_adv_features(
    target: nn.Module,
    suspect: nn.Module,
    probe_x: torch.Tensor,
    device: str,
    prefix: str,
    epsilon: float = 0.06,
) -> dict[str, float]:
    """Two features per probe set:
        s5_grad_cos_{prefix}        mean cosine(∂L/∂x_target, ∂L/∂x_suspect)
        s5_adv_transfer_{prefix}    fraction of target-fooling FGSM examples
                                    that fool the suspect into the *same* wrong class

    Args:
        epsilon: FGSM step size in *normalized* input space. 0.06 ≈ 4/255 in
                 raw pixels for CIFAR-100 normalization stats.

    All ops are batched and per-example; the function returns scalar means.
    """
    target.eval()
    suspect.eval()
    target.to(device)
    suspect.to(device)
    probe_x = probe_x.to(device)

    # Pseudo-label = target's clean prediction. Identical for both models'
    # loss computations so the gradients are comparable directionally.
    with torch.no_grad():
        target_clean = target(probe_x)
        pseudo_y = target_clean.argmax(dim=1)

    # ---- target gradient ----
    x_t = probe_x.clone().detach().requires_grad_(True)
    loss_t = F.cross_entropy(target(x_t), pseudo_y)
    (grad_t,) = torch.autograd.grad(loss_t, x_t)

    # ---- suspect gradient (same pseudo-label) ----
    x_s = probe_x.clone().detach().requires_grad_(True)
    loss_s = F.cross_entropy(suspect(x_s), pseudo_y)
    (grad_s,) = torch.autograd.grad(loss_s, x_s)

    # cosine per example, then mean (NaN-safe via clamp on norms)
    t_flat = grad_t.reshape(grad_t.size(0), -1)
    s_flat = grad_s.reshape(grad_s.size(0), -1)
    t_norm = t_flat.norm(dim=1).clamp_min(1e-12)
    s_norm = s_flat.norm(dim=1).clamp_min(1e-12)
    cos = ((t_flat * s_flat).sum(dim=1) / (t_norm * s_norm))
    grad_cos = float(cos.mean().item()) if torch.isfinite(cos).any() else 0.0

    # ---- FGSM adversarial transfer ----
    adv_x = (probe_x + epsilon * grad_t.sign()).detach()
    with torch.no_grad():
        adv_pred_t = target(adv_x).argmax(dim=1)
        adv_pred_s = suspect(adv_x).argmax(dim=1)
        fooled = (adv_pred_t != pseudo_y)  # target moved to a different class
        if int(fooled.sum().item()) > 0:
            transfer = (
                adv_pred_s[fooled] == adv_pred_t[fooled]
            ).float().mean().item()
        else:
            transfer = 0.0

    return {
        f"s5_grad_cos_{prefix}": float(grad_cos),
        f"s5_adv_transfer_{prefix}": float(transfer),
    }
