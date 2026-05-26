"""Train meta-classifier on (raw features + per-variant trigger scores).

The existing train_meta_classifier.py trains LR on 65 extracted features.
This script extends that by adding the 6 boundary-trigger scores (one per
probe-set variant: ood / holdout / test / train / eval / all) as additional
columns. The classifier learns the OPTIMAL combination of (raw features +
triggers) — replacing our manual blend weights with a learned mix.

To avoid data leakage on the shadow side, shadow trigger scores are computed
via 5-fold CV (find triggers from 4/5 of shadows, score the held-out 1/5;
repeat for all 5 folds). Real-suspect trigger scores already exist in
submissions/triggers_<variant>_k500.csv from the overnight pipeline.

Output:
    submissions/submission_meta_with_triggers.csv  (LR decision function)

Run after the comprehensive trigger pipeline (cluster/triggers.sub) has
produced shadows/predictions.npz and all 6 submissions/triggers_*_k500.csv.
"""

from __future__ import annotations

import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = ["ood", "holdout", "test", "train", "eval", "all"]
VARIANT_SPLITS = {
    "ood": ["ood"],
    "holdout": ["holdout"],
    "test": ["test"],
    "train": ["train"],
    "eval": ["ood", "holdout", "test"],
    "all": ["train", "ood", "holdout", "test"],
}


def find_triggers_and_score(
    shadow_preds_train: np.ndarray,
    target_pred: np.ndarray,
    labels_train: np.ndarray,
    suspect_preds_score: np.ndarray,
    top_k: int = 500,
    min_fire_rate: float = 0.05,
) -> np.ndarray:
    """Find top-K triggers from training shadows; score held-out suspects."""
    from sklearn.metrics import roc_auc_score

    fires_train = (shadow_preds_train == target_pred[None, :]).astype(np.float32)
    fire_rate = fires_train.mean(axis=0)
    has_contrast = (fire_rate >= min_fire_rate) & (fire_rate <= 1 - min_fire_rate)

    aucs = np.full(fires_train.shape[1], 0.5, dtype=np.float64)
    for p in np.where(has_contrast)[0]:
        try:
            aucs[p] = roc_auc_score(labels_train, fires_train[:, p])
        except ValueError:
            aucs[p] = 0.5

    discrim = np.abs(aucs - 0.5)
    top_idx = np.argsort(discrim)[::-1][:top_k]

    fires_score = (suspect_preds_score == target_pred[None, :]).astype(np.float64)
    scores = np.zeros(suspect_preds_score.shape[0], dtype=np.float64)
    total_w = 0.0
    for p in top_idx:
        if discrim[p] < 1e-6:
            break
        w = discrim[p]
        if aucs[p] > 0.5:
            scores += w * fires_score[:, p]
        else:
            scores += w * (1.0 - fires_score[:, p])
        total_w += w
    if total_w > 0:
        scores /= total_w
    return scores


def compute_shadow_trigger_scores_cv(
    shadow_preds: np.ndarray,      # [N_shadows, n_probes_total]
    target_pred: np.ndarray,       # [n_probes_total]
    labels: np.ndarray,            # [N_shadows]
    split_names: list[str],
    split_offsets: np.ndarray,
    variants: list[str],
    n_folds: int = 5,
    top_k: int = 500,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """5-fold CV to get unbiased trigger scores per shadow per variant."""
    from sklearn.model_selection import StratifiedKFold

    N = shadow_preds.shape[0]
    results = {v: np.zeros(N, dtype=np.float64) for v in variants}

    for variant in variants:
        splits = VARIANT_SPLITS[variant]
        keep = np.zeros(shadow_preds.shape[1], dtype=bool)
        for s in splits:
            i = split_names.index(s)
            keep[split_offsets[i]:split_offsets[i + 1]] = True
        preds_v = shadow_preds[:, keep]
        target_v = target_pred[keep]

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        print(f"  variant {variant:>10s}: {keep.sum()} probes — running {n_folds}-fold CV …",
              flush=True)
        for fold_i, (tr, va) in enumerate(skf.split(preds_v, labels)):
            scores = find_triggers_and_score(
                preds_v[tr], target_v, labels[tr],
                preds_v[va],
                top_k=top_k,
            )
            results[variant][va] = scores
    return results


def main() -> None:
    print("[1/6] loading raw features.csv (shadow + real) ...", flush=True)
    shadow_df = pd.concat(
        [pd.read_csv(p) for p in sorted(glob("checkpoints/shadow_features_shard_*.csv"))],
        ignore_index=True,
    )
    shadow_df = (shadow_df.drop_duplicates("suspect_id", keep="last")
                 .sort_values("suspect_id").reset_index(drop=True))

    real_df = pd.concat(
        [pd.read_csv(p) for p in sorted(glob("checkpoints/features_shard_*.csv"))],
        ignore_index=True,
    )
    real_df = (real_df.drop_duplicates("suspect_id", keep="last")
               .sort_values("suspect_id").reset_index(drop=True))
    print(f"        shadow features: {len(shadow_df)} rows × {len(shadow_df.columns)-1} cols")
    print(f"        real   features: {len(real_df)} rows × {len(real_df.columns)-1} cols")

    print("[2/6] loading labels and aligning ...", flush=True)
    labels_df = pd.read_csv("shadows/labels.csv").sort_values("suspect_id").reset_index(drop=True)
    common_ids = sorted(set(shadow_df["suspect_id"]) & set(labels_df["suspect_id"]))
    shadow_df = shadow_df[shadow_df["suspect_id"].isin(common_ids)].sort_values("suspect_id").reset_index(drop=True)
    labels_df = labels_df[labels_df["suspect_id"].isin(common_ids)].sort_values("suspect_id").reset_index(drop=True)
    labels = labels_df["label"].values.astype(int)
    print(f"        aligned shadows: {len(shadow_df)}  ({int(labels.sum())} stolen, {int((1-labels).sum())} not)")

    print("[3/6] loading shadow predictions and computing CV trigger scores ...", flush=True)
    preds_data = np.load("shadows/predictions.npz", allow_pickle=True)
    shadow_preds = preds_data["suspect_preds"]
    shadow_pred_ids = preds_data["suspect_ids"]
    target_pred = preds_data["target_pred"]
    split_names = list(preds_data["split_names"])
    split_offsets = np.asarray(preds_data["split_offsets"])

    # Align shadow_preds with shadow_df. Take intersection of all three
    # (shadow_df, labels_df, shadow_preds), filter all consistently.
    pred_id_set = set(int(s) for s in shadow_pred_ids)
    mask = shadow_df["suspect_id"].isin(pred_id_set).values
    if not mask.all():
        n_drop = int((~mask).sum())
        print(f"        dropping {n_drop} shadows that lack predictions")
        shadow_df = shadow_df[mask].reset_index(drop=True)
        labels = labels[mask]
    pred_id_to_row = {int(sid): i for i, sid in enumerate(shadow_pred_ids)}
    rows = [pred_id_to_row[int(sid)] for sid in shadow_df["suspect_id"]]
    shadow_preds_aligned = shadow_preds[rows]
    print(f"        final aligned shadow set: {len(shadow_df)} suspects "
          f"({int(labels.sum())} stolen, {int((1-labels).sum())} not)")

    shadow_trig = compute_shadow_trigger_scores_cv(
        shadow_preds_aligned, target_pred, labels, split_names, split_offsets,
        variants=VARIANTS, n_folds=5, top_k=500,
    )

    print("[4/6] loading real-suspect trigger scores from submissions/ ...", flush=True)
    real_trig: dict[str, np.ndarray] = {}
    for v in VARIANTS:
        path = f"submissions/triggers_{v}_k500.csv"
        if not Path(path).exists():
            print(f"        [warn] {path} missing; using zeros", file=sys.stderr)
            real_trig[v] = np.zeros(len(real_df))
            continue
        td = pd.read_csv(path).sort_values("id").reset_index(drop=True)
        # Align by id
        score_by_id = dict(zip(td["id"].astype(int), td["score"].astype(float)))
        real_trig[v] = np.array([score_by_id.get(int(i), 0.0) for i in real_df["suspect_id"]])

    print("[5/6] augmenting features and training LR ...", flush=True)
    for v in VARIANTS:
        shadow_df[f"trig_{v}"] = shadow_trig[v]
        real_df[f"trig_{v}"] = real_trig[v]

    feat_cols_s = [c for c in shadow_df.columns if c != "suspect_id"]
    feat_cols_r = [c for c in real_df.columns if c != "suspect_id"]
    common = sorted(set(feat_cols_s) & set(feat_cols_r))
    print(f"        using {len(common)} features (including 6 new trig_* columns)")

    X_shadow = shadow_df[common].values.astype(np.float64)
    X_real = real_df[common].values.astype(np.float64)
    X_shadow = np.nan_to_num(X_shadow, nan=0.0, posinf=0.0, neginf=0.0)
    X_real = np.nan_to_num(X_real, nan=0.0, posinf=0.0, neginf=0.0)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    scaler = StandardScaler()
    X_shadow_s = scaler.fit_transform(X_shadow)
    X_real_s = scaler.transform(X_real)

    print("\n=== 5-fold CV on augmented features ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold_aucs = []
    for fi, (tr, va) in enumerate(skf.split(X_shadow_s, labels)):
        m = LogisticRegression(C=1.0, max_iter=2000, penalty="l2", solver="lbfgs",
                               class_weight="balanced")
        m.fit(X_shadow_s[tr], labels[tr])
        p = m.predict_proba(X_shadow_s[va])[:, 1]
        fold_aucs.append(roc_auc_score(labels[va], p))
        print(f"  fold {fi}: AUC={fold_aucs[-1]:.4f}")
    print(f"  mean AUC={np.mean(fold_aucs):.4f}  std={np.std(fold_aucs):.4f}")

    print("[6/6] fitting on all shadows, predicting on real ...", flush=True)
    final = LogisticRegression(C=1.0, max_iter=2000, penalty="l2", solver="lbfgs",
                               class_weight="balanced")
    final.fit(X_shadow_s, labels)
    scores = final.decision_function(X_real_s)

    out_df = pd.DataFrame({"id": real_df["suspect_id"].astype(int),
                           "score": scores.astype(float)})
    out_df = out_df.sort_values("id").reset_index(drop=True)
    assert len(out_df) == 360
    assert out_df["id"].min() == 0 and out_df["id"].max() == 359
    assert out_df["id"].is_unique
    assert np.isfinite(out_df["score"].values).all()

    Path("submissions").mkdir(exist_ok=True)
    out_df.to_csv("submissions/submission_meta_with_triggers.csv", index=False)

    print("\n=== Top 20 LR coefficients (signed; higher → more stolen) ===")
    coef = pd.Series(final.coef_[0], index=common).sort_values(
        key=lambda s: s.abs(), ascending=False)
    print(coef.head(20).to_string())

    print(f"\n[done] wrote submissions/submission_meta_with_triggers.csv")
    print(out_df.describe())


if __name__ == "__main__":
    main()
