"""S2 — Dataset Inference adapted to supervised: train/test gap of the suspect.

The target memorized its 40k training indices. A stolen suspect (direct copy,
fine-tuned, or distilled with target outputs) inherits some of that memorization
so its loss is systematically lower on the target's 40k than on a 10k holdout
that the target never saw. An independent suspect trained on overlapping data
spreads memorization more uniformly across the train split, so the gap that is
specific to the target's 40k subset is much smaller.

Features returned (per suspect):
    s2_loss_train, s2_loss_holdout, s2_loss_test     mean CE loss on each split
    s2_acc_train,  s2_acc_holdout,  s2_acc_test      top-1 accuracy on each split
    s2_conf_train, s2_conf_holdout, s2_conf_test     mean max softmax prob
    s2_loss_gap_h_t                                  holdout_loss - train_loss   (higher = more stolen)
    s2_loss_gap_te_t                                 test_loss - train_loss      (higher = more stolen)
    s2_conf_gap_t_h                                  train_conf - holdout_conf   (higher = more stolen)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def split_features(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """Return CE loss, top-1 accuracy and mean max-softmax confidence."""
    loss = F.cross_entropy(logits, labels, reduction="mean").item()
    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    conf = F.softmax(logits, dim=1).max(dim=1).values.mean().item()
    return {
        f"s2_loss_{prefix}": float(loss),
        f"s2_acc_{prefix}": float(acc),
        f"s2_conf_{prefix}": float(conf),
    }


def gap_features(per_split: dict[str, float]) -> dict[str, float]:
    """Combine per-split features into stolen-oriented gap scores.

    Expects keys s2_loss_train, s2_loss_holdout, s2_loss_test,
    s2_conf_train, s2_conf_holdout.
    """
    loss_train = per_split["s2_loss_train"]
    loss_holdout = per_split["s2_loss_holdout"]
    loss_test = per_split["s2_loss_test"]
    conf_train = per_split["s2_conf_train"]
    conf_holdout = per_split["s2_conf_holdout"]

    return {
        "s2_loss_gap_h_t": float(loss_holdout - loss_train),
        "s2_loss_gap_te_t": float(loss_test - loss_train),
        "s2_conf_gap_t_h": float(conf_train - conf_holdout),
    }
