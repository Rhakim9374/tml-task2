"""S1 — output agreement between target and suspect on a probe set.

For each (suspect, probe) pair we compute several agreement statistics over
the precomputed target logits. All scores are oriented so that "higher =
more likely stolen".

Features returned (per suspect, per probe-set name):
    s1_cos        mean cosine similarity of raw logits
    s1_nkl_t_s    mean -KL(softmax(target) || softmax(suspect))  (higher = more agree)
    s1_nl2_prob   mean -||softmax(target) - softmax(suspect)||_2
    s1_top1       top-1 argmax agreement rate
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def forward_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    amp: bool = True,
) -> torch.Tensor:
    """Run model on the entire loader and return concatenated logits on CPU."""
    model.eval()
    out_chunks = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], enabled=amp and device != "cpu"):
            logits = model(x)
        out_chunks.append(logits.float().cpu())
    return torch.cat(out_chunks, dim=0)


def agreement_features(
    target_logits: torch.Tensor,
    suspect_logits: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """Compute the four S1 sub-features. All tensors are CPU float32, shape [N, 100]."""
    assert target_logits.shape == suspect_logits.shape, (target_logits.shape, suspect_logits.shape)
    t = target_logits
    s = suspect_logits

    # cosine on logits
    cos = F.cosine_similarity(t, s, dim=1).mean().item()

    # KL on softmax probs (use a tiny floor for numerical stability)
    tp = F.softmax(t, dim=1).clamp_min(1e-12)
    sp = F.softmax(s, dim=1).clamp_min(1e-12)
    kl_t_s = (tp * (tp.log() - sp.log())).sum(dim=1).mean().item()

    # L2 on softmax probs
    l2_prob = (tp - sp).pow(2).sum(dim=1).sqrt().mean().item()

    # top-1 agreement
    top1 = (t.argmax(dim=1) == s.argmax(dim=1)).float().mean().item()

    return {
        f"s1_cos_{prefix}": float(cos),
        f"s1_nkl_t_s_{prefix}": float(-kl_t_s),  # negate so higher = more agree
        f"s1_nl2_prob_{prefix}": float(-l2_prob),  # negate so higher = more agree
        f"s1_top1_{prefix}": float(top1),
    }
