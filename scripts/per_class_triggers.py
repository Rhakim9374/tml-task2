"""Per-class boundary triggers — find triggers separately for each of target's
predicted classes, then aggregate scores across classes.

Standard triggers (from find_triggers.py) pool all probes together and find a
single top-K. Per-class triggers ask: within the probes where target predicts
class c, which agreement patterns are most discriminative? Then aggregate
contributions across all 100 classes.

Intuition: target may have idiosyncratic prediction patterns for specific
classes — e.g., it might confuse class 23 with 24 in a particular way, or
predict class 7 with high confidence on specific inputs. Stolen models inherit
these per-class patterns; independents have their own. By looking class-by-
class we expose patterns that get washed out in the global pool.

Outputs:
    submissions/submission_perclass_triggers.csv  (id, score)

Uses the same shadow/real prediction npz files as the comprehensive trigger
pipeline (cluster/triggers.sub output). No need to re-extract.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def per_class_score(
    shadow_preds: np.ndarray,        # [N_shadow, P]
    target_pred: np.ndarray,         # [P]
    labels: np.ndarray,              # [N_shadow]
    score_preds: np.ndarray,         # [N_score, P]   suspects to score
    n_classes: int = 100,
    min_probes_per_class: int = 10,
    max_triggers_per_class: int = 20,
    min_fire_rate: float = 0.05,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Per-class trigger selection and scoring.

    Returns (scores, info_dict). scores is [N_score].
    """
    from sklearn.metrics import roc_auc_score

    N_score = score_preds.shape[0]
    scores = np.zeros(N_score, dtype=np.float64)
    total_w = 0.0
    used_classes = 0
    total_triggers = 0
    per_class_summary: list[dict] = []

    for c in range(n_classes):
        class_probes = np.where(target_pred == c)[0]
        n_probes = len(class_probes)
        if n_probes < min_probes_per_class:
            continue

        # Fires on this class's probes (target_pred is c on all of them, so
        # fires == (shadow_pred == c) == (shadow_pred == target_pred))
        train_fires = (shadow_preds[:, class_probes] == c).astype(np.float32)
        fire_rate = train_fires.mean(axis=0)
        has_contrast = (fire_rate >= min_fire_rate) & (fire_rate <= 1 - min_fire_rate)

        aucs = np.full(n_probes, 0.5, dtype=np.float64)
        for li in range(n_probes):
            if not has_contrast[li]:
                continue
            try:
                aucs[li] = roc_auc_score(labels, train_fires[:, li])
            except ValueError:
                pass

        discrim = np.abs(aucs - 0.5)
        n_top = min(max_triggers_per_class, n_probes)
        top_local = np.argsort(discrim)[::-1][:n_top]

        score_fires = (score_preds[:, class_probes] == c).astype(np.float64)

        class_triggers_used = 0
        class_max_auc = 0.5
        for li in top_local:
            if discrim[li] < 1e-6:
                break
            w = float(discrim[li])
            if aucs[li] > 0.5:
                scores += w * score_fires[:, li]
            else:
                scores += w * (1.0 - score_fires[:, li])
            total_w += w
            class_triggers_used += 1
            total_triggers += 1
            class_max_auc = max(class_max_auc, float(aucs[li]) if aucs[li] >= 0.5 else 1 - float(aucs[li]))
        if class_triggers_used > 0:
            used_classes += 1
            per_class_summary.append({
                "class": int(c),
                "n_probes": int(n_probes),
                "n_triggers": class_triggers_used,
                "max_auc": class_max_auc,
            })

    if total_w > 0:
        scores /= total_w

    info = {
        "used_classes": used_classes,
        "n_classes": n_classes,
        "total_triggers": total_triggers,
        "total_weight": total_w,
        "per_class": per_class_summary,
    }
    if verbose:
        print(f"used {used_classes}/{n_classes} classes, {total_triggers} total triggers")
        # Show the top-10 classes by max-AUC
        top_classes = sorted(per_class_summary, key=lambda x: x["max_auc"], reverse=True)[:10]
        print("Top-10 most-discriminative classes:")
        for x in top_classes:
            print(f"  class {x['class']:3d}: n_probes={x['n_probes']:4d}  "
                  f"n_triggers={x['n_triggers']:2d}  max_auc={x['max_auc']:.3f}")
    return scores, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-preds", default="shadows/predictions.npz")
    ap.add_argument("--real-preds", default="real_predictions.npz")
    ap.add_argument("--labels", default="shadows/labels.csv")
    ap.add_argument("--out", default="submissions/submission_perclass_triggers.csv")
    ap.add_argument("--min-probes-per-class", type=int, default=10)
    ap.add_argument("--max-triggers-per-class", type=int, default=20)
    ap.add_argument("--n-classes", type=int, default=100)
    args = ap.parse_args()

    sd = np.load(args.shadow_preds, allow_pickle=True)
    rd = np.load(args.real_preds, allow_pickle=True)
    shadow_preds = sd["suspect_preds"]
    shadow_ids = sd["suspect_ids"]
    target_pred = sd["target_pred"]
    real_preds = rd["suspect_preds"]
    real_ids = rd["suspect_ids"]

    labels_df = pd.read_csv(args.labels).set_index("suspect_id")
    labels = np.array([int(labels_df.loc[int(i), "label"]) for i in shadow_ids])
    print(f"shadows: {len(shadow_ids)} ({int(labels.sum())} stolen, {int((1-labels).sum())} not)")
    print(f"real: {len(real_ids)}  probes: {shadow_preds.shape[1]}")

    scores, info = per_class_score(
        shadow_preds, target_pred, labels,
        real_preds,
        n_classes=args.n_classes,
        min_probes_per_class=args.min_probes_per_class,
        max_triggers_per_class=args.max_triggers_per_class,
    )

    out_df = pd.DataFrame({"id": real_ids.astype(int), "score": scores})
    out_df = out_df.sort_values("id").reset_index(drop=True)

    expected = set(range(360))
    have = set(out_df["id"].tolist())
    missing = expected - have
    if missing:
        med = float(out_df["score"].median())
        pad = pd.DataFrame({"id": sorted(missing), "score": [med] * len(missing)})
        out_df = pd.concat([out_df, pad], ignore_index=True).sort_values("id").reset_index(drop=True)
        print(f"[warn] padded {len(missing)} missing suspects with median {med:.3f}")

    assert len(out_df) == 360
    assert out_df["id"].min() == 0 and out_df["id"].max() == 359
    assert out_df["id"].is_unique
    assert np.isfinite(out_df["score"].values).all()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\n[done] wrote {args.out}")
    print(out_df.describe())


if __name__ == "__main__":
    main()
