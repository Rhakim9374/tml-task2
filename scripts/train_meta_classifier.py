"""Train an L2-logistic-regression meta-classifier on the shadow features
and predict on the real suspects.

Inputs:
    checkpoints/shadow_features_shard_*.csv   features for the 195 shadow models
    shadows/labels.csv                        suspect_id, label (0/1), kind
    checkpoints/features_shard_*.csv          features for the 360 real suspects

Output:
    submissions/submission_meta_lr.csv        id, score (LR decision function)

LR's smooth decision_function preserves rank better than a sigmoid-saturated
probability and, with only ~195 shadow samples, doesn't overfit the way an
XGBoost meta-classifier did in earlier iterations.
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def load_concat(glob_pattern: str) -> pd.DataFrame:
    paths = sorted(glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {glob_pattern}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates("suspect_id", keep="last")
    df = df.sort_values("suspect_id").reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-features", default="checkpoints/shadow_features_shard_*.csv")
    ap.add_argument("--shadow-labels", default="shadows/labels.csv")
    ap.add_argument("--real-features", default="checkpoints/features_shard_*.csv")
    ap.add_argument("--out", default="submissions/submission_meta_lr.csv")
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    shadow_df = load_concat(args.shadow_features)
    labels_df = pd.read_csv(args.shadow_labels).sort_values("suspect_id").reset_index(drop=True)

    # Inner-join: any shadow that failed during extract is silently dropped.
    common_ids = sorted(set(shadow_df["suspect_id"]) & set(labels_df["suspect_id"]))
    missing_feat = sorted(set(labels_df["suspect_id"]) - set(shadow_df["suspect_id"]))
    if missing_feat:
        print(f"[warn] {len(missing_feat)} shadow(s) lack features (skipped): {missing_feat}")
    shadow_df = shadow_df[shadow_df["suspect_id"].isin(common_ids)].sort_values("suspect_id").reset_index(drop=True)
    labels_df = labels_df[labels_df["suspect_id"].isin(common_ids)].sort_values("suspect_id").reset_index(drop=True)
    print(f"using {len(common_ids)} shadows with both features and labels")

    real_df = load_concat(args.real_features)

    feat_cols = sorted(set(shadow_df.columns) & set(real_df.columns) - {"suspect_id"})
    print(f"using {len(feat_cols)} features common to shadow + real")

    X_shadow = shadow_df[feat_cols].values
    y_shadow = labels_df["label"].values
    X_real = real_df[feat_cols].values

    if np.isnan(X_shadow).any() or np.isinf(X_shadow).any() \
       or np.isnan(X_real).any() or np.isinf(X_real).any():
        print("[warn] non-finite values present; replacing with 0")
        X_shadow = np.nan_to_num(X_shadow, nan=0.0, posinf=0.0, neginf=0.0)
        X_real = np.nan_to_num(X_real, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_shadow_s = scaler.fit_transform(X_shadow)
    X_real_s = scaler.transform(X_real)

    # Stratified CV to estimate generalization on the shadow population.
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    fold_aucs = []
    for fi, (tr_idx, va_idx) in enumerate(skf.split(X_shadow_s, y_shadow)):
        m = LogisticRegression(C=1.0, max_iter=2000, penalty="l2", solver="lbfgs",
                               class_weight="balanced")
        m.fit(X_shadow_s[tr_idx], y_shadow[tr_idx])
        p = m.predict_proba(X_shadow_s[va_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y_shadow[va_idx], p))
        print(f"  fold {fi}: AUC={fold_aucs[-1]:.4f}")
    print(f"mean AUC={np.mean(fold_aucs):.4f}  std={np.std(fold_aucs):.4f}")

    # Final model trained on all shadows; emit decision_function on real.
    final = LogisticRegression(C=1.0, max_iter=2000, penalty="l2", solver="lbfgs",
                               class_weight="balanced")
    final.fit(X_shadow_s, y_shadow)
    scores = final.decision_function(X_real_s)

    out_df = pd.DataFrame({"id": real_df["suspect_id"].astype(int),
                           "score": scores.astype(float)})
    out_df = out_df.sort_values("id").reset_index(drop=True)
    assert len(out_df) == 360
    assert out_df["id"].min() == 0 and out_df["id"].max() == 359
    assert out_df["id"].is_unique
    assert np.isfinite(out_df["score"].values).all()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\n=== Top-15 LR coefficients (|w| descending) ===")
    coef = pd.Series(final.coef_[0], index=feat_cols).sort_values(
        key=lambda s: s.abs(), ascending=False)
    print(coef.head(15).to_string())
    print(f"\n[done] wrote {len(out_df)} rows → {args.out}")
    print(out_df.describe())


if __name__ == "__main__":
    main()
