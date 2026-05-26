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


# --------------------------------------------------------------------------
# Fine-grained agreement (added 2026-05-26): mistake / top-K / low-confidence
# --------------------------------------------------------------------------
#
# Independents converge to *different* solutions even on identical training
# data (different random init → different local minimum). They agree with the
# target on easy examples (everyone gets those right) but disagree on the hard
# ones. Stolen models — especially distilled — inherit the target's idiosyncratic
# behavior on those exact hard examples. These three features isolate the
# discriminative subset.


def mistake_agreement_features(
    target_logits: torch.Tensor,
    suspect_logits: torch.Tensor,
    labels: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """Agreement *only* on inputs where target is wrong.

    Strongest fingerprinting signal we have: a stolen model copies the
    target's wrong predictions; an independent model makes different mistakes.
    """
    target_pred = target_logits.argmax(dim=1)
    suspect_pred = suspect_logits.argmax(dim=1)
    mistake_mask = (target_pred != labels)
    n_mistakes = int(mistake_mask.sum().item())
    if n_mistakes == 0:
        # Target perfect on this probe — degenerate; default to 1.0 (most-stolen).
        return {f"s1_mistake_{prefix}": 1.0, f"s1_mistake_{prefix}_n": 0.0}
    agree = (suspect_pred[mistake_mask] == target_pred[mistake_mask]).float().mean().item()
    return {f"s1_mistake_{prefix}": float(agree),
            f"s1_mistake_{prefix}_n": float(n_mistakes)}


def topk_overlap_features(
    target_logits: torch.Tensor,
    suspect_logits: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """Top-3 and top-5 overlap of target vs suspect predictions.

    Catches "secondary preferences" — even when the top-1 differs, distilled
    students inherit the target's *runner-up* classes; independents have
    different runner-ups.
    """
    out: dict[str, float] = {}
    for k in (3, 5):
        t_topk = target_logits.topk(k, dim=1).indices  # [N, k]
        s_topk = suspect_logits.topk(k, dim=1).indices  # [N, k]
        # For each row, count how many of target's top-k appear in suspect's top-k.
        match = (t_topk.unsqueeze(2) == s_topk.unsqueeze(1)).any(dim=2).float()  # [N, k]
        overlap = match.mean(dim=1).mean().item()  # mean over rows of fraction-matched
        out[f"s1_top{k}_overlap_{prefix}"] = float(overlap)
    return out


def low_conf_agreement_features(
    target_logits: torch.Tensor,
    suspect_logits: torch.Tensor,
    prefix: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Agreement only where target's max softmax probability is below `threshold`.

    Where target hedges, the chosen class is idiosyncratic (multiple plausible
    classes; target picked one). Stolen models inherit this confusion structure;
    independents resolve differently.
    """
    target_probs = F.softmax(target_logits, dim=1)
    target_max = target_probs.max(dim=1).values
    target_pred = target_logits.argmax(dim=1)
    suspect_pred = suspect_logits.argmax(dim=1)
    mask = (target_max < threshold)
    n_low = int(mask.sum().item())
    if n_low == 0:
        return {f"s1_low_conf_agree_{prefix}": 1.0}
    agree = (suspect_pred[mask] == target_pred[mask]).float().mean().item()
    return {f"s1_low_conf_agree_{prefix}": float(agree)}
