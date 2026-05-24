"""S4 — Centered Kernel Alignment between target and suspect activations.

Linear CKA is invariant to orthogonal transformations of the feature space, so
it scores function-preserving channel permutations and rotations as perfectly
similar (CKA = 1) while still distinguishing structurally different networks
(CKA << 1 for independent inits). This complements S3 well: where S3 needs an
explicit alignment, S4 is alignment-free.

We capture the post-stage activations of layer1..layer4 and the pooled
512-D penultimate feature, then average CKA across them.

Returns:
    s4_cka_stem, s4_cka_layer{1..4}, s4_cka_penult, s4_cka_mean
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _gather_grams(model: nn.Module, probe: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    """Run probe through model, return CPU activations at key points,
    each pooled and flattened to [N, D]."""
    buf: dict[str, torch.Tensor] = {}
    handles = []

    def cap(key, pool: bool):
        def hook(_m, _i, out):
            if pool and out.dim() == 4:
                # global average pool over spatial dims → [N, C]
                buf[key] = out.mean(dim=(2, 3)).detach().float().cpu()
            else:
                if out.dim() == 4:
                    buf[key] = out.flatten(1).detach().float().cpu()
                else:
                    buf[key] = out.detach().float().cpu()
        return hook

    handles.append(model.bn1.register_forward_hook(cap("stem", pool=True)))
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        handles.append(getattr(model, name).register_forward_hook(cap(name, pool=True)))
    handles.append(model.avgpool.register_forward_hook(cap("penult", pool=False)))

    model.eval()
    model.to(device)
    with torch.no_grad():
        model(probe.to(device, non_blocking=True))
    for h in handles:
        h.remove()
    return buf


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between [N, D1] and [N, D2] feature matrices."""
    assert x.shape[0] == y.shape[0], (x.shape, y.shape)
    # center features along N
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xtY = (x.T @ y).norm().item() ** 2
    xtX = (x.T @ x).norm().item()
    ytY = (y.T @ y).norm().item()
    denom = xtX * ytY
    if denom <= 0:
        return 0.0
    return float(xtY / denom)


def cka_features(target: nn.Module, suspect: nn.Module, probe: torch.Tensor, device: str) -> dict[str, float]:
    t = _gather_grams(target, probe, device)
    s = _gather_grams(suspect, probe, device)
    keys = ["stem", "layer1", "layer2", "layer3", "layer4", "penult"]
    out = {}
    vals = []
    for k in keys:
        v = linear_cka(t[k], s[k])
        out[f"s4_cka_{k}"] = v
        vals.append(v)
    out["s4_cka_mean"] = sum(vals) / len(vals)
    return out
