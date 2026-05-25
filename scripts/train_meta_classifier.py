"""Train an XGBoost meta-classifier on shadow features, then predict on real.

Inputs:
    checkpoints/shadow_features_shard_*.csv   per-shadow features (any sharding)
    shadows/labels.csv                        suspect_id, label, kind
    checkpoints/features_shard_*.csv          per-real-suspect features

Output:
    submissions/submission_meta.csv           id, score (XGBoost predict_proba)

The columns in shadow_features and real_features must be identical (same
extract_signals.py was used for both).
"""

from __future__ import annotations

import argparse
import sys
from glob import glob

import numpy as np
import pandas as pd


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
    ap.add_argument("--out", default="submissions/submission_meta.csv")
    ap.add_argument("--n-folds", type=int, default=5)
    # Tuned for small shadow sets (~40 samples): shallow trees, modest count, mild lr.
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    args = ap.parse_args()

    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost not installed; run:", file=sys.stderr)
        print("    ~/.tml-venv/bin/pip install xgboost", file=sys.stderr)
        sys.exit(2)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    shadow_df = load_concat(args.shadow_features)
    labels_df = pd.read_csv(args.shadow_labels)
    labels_df = labels_df.sort_values("suspect_id").reset_index(drop=True)

    # Inner-join on suspect_id so suspects that failed during extract just
    # get dropped (rather than killing the whole run).
    common_ids = sorted(set(shadow_df["suspect_id"]) & set(labels_df["suspect_id"]))
    missing_feat = sorted(set(labels_df["suspect_id"]) - set(shadow_df["suspect_id"]))
    missing_lbl = sorted(set(shadow_df["suspect_id"]) - set(labels_df["suspect_id"]))
    if missing_feat:
        print(f"[warn] {len(missing_feat)} shadow suspect(s) lack features "
              f"(probably failed during extract): {missing_feat}")
    if missing_lbl:
        print(f"[warn] {len(missing_lbl)} shadow suspect(s) lack labels: {missing_lbl}")
    shadow_df = shadow_df[shadow_df["suspect_id"].isin(common_ids)].sort_values("suspect_id").reset_index(drop=True)
    labels_df = labels_df[labels_df["suspect_id"].isin(common_ids)].sort_values("suspect_id").reset_index(drop=True)
    print(f"using {len(common_ids)} shadow suspects with both features and labels")

    # Per-kind feature evolution: shows how each derivation type changes each
    # signal, so we can sanity-check that the pipeline behaves as expected.
    merged = shadow_df.merge(labels_df, on="suspect_id")
    print("\n=== PER-KIND FEATURE MEANS (how stealing methods evolve each signal) ===")
    summary_cols = [
        "s1_cos_train", "s1_top1_train", "s2_loss_gap_h_t", "s2_conf_gap_t_h",
        "s3_raw_l2", "s3_perm_l2", "s4_cka_mean", "s4_cka_penult",
    ]
    summary_cols = [c for c in summary_cols if c in shadow_df.columns]
    if summary_cols:
        per_kind = merged.groupby("kind")[summary_cols].mean().round(4)
        print(per_kind.to_string())

    real_df = load_concat(args.real_features)

    feat_cols = [c for c in shadow_df.columns if c != "suspect_id"]
    real_feat_cols = [c for c in real_df.columns if c != "suspect_id"]
    common = sorted(set(feat_cols) & set(real_feat_cols))
    if len(common) < len(feat_cols):
        print(f"[warn] dropping {len(feat_cols) - len(common)} shadow-only columns: "
              f"{sorted(set(feat_cols) - set(real_feat_cols))}")
    if len(common) < len(real_feat_cols):
        print(f"[warn] dropping {len(real_feat_cols) - len(common)} real-only columns: "
              f"{sorted(set(real_feat_cols) - set(feat_cols))}")
    feat_cols = common
    print(f"using {len(feat_cols)} features in common")

    X_shadow = shadow_df[feat_cols].values
    y_shadow = labels_df["label"].values
    X_real = real_df[feat_cols].values

    # quick sanity: any nans/infs?
    bad_shadow = np.isnan(X_shadow).any() | np.isinf(X_shadow).any()
    bad_real = np.isnan(X_real).any() | np.isinf(X_real).any()
    if bad_shadow or bad_real:
        print(f"[warn] non-finite values present (shadow={bad_shadow}, real={bad_real}); replacing with 0")
        X_shadow = np.nan_to_num(X_shadow, nan=0.0, posinf=0.0, neginf=0.0)
        X_real = np.nan_to_num(X_real, nan=0.0, posinf=0.0, neginf=0.0)

    # Stratified CV to estimate generalization
    print(f"\n=== {args.n_folds}-fold CV on shadows ===")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    fold_aucs = []
    for fi, (tr_idx, va_idx) in enumerate(skf.split(X_shadow, y_shadow)):
        m = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            eval_metric="logloss",
            n_jobs=4,
            verbosity=0,
        )
        m.fit(X_shadow[tr_idx], y_shadow[tr_idx])
        p = m.predict_proba(X_shadow[va_idx])[:, 1]
        auc = roc_auc_score(y_shadow[va_idx], p)
        fold_aucs.append(auc)
        print(f"  fold {fi}: AUC={auc:.4f}")
    print(f"mean AUC={np.mean(fold_aucs):.4f}  std={np.std(fold_aucs):.4f}")

    # Fit on full shadow set, predict on real
    print("\n=== fitting on all shadows, predicting on real ===")
    final = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        eval_metric="logloss",
        n_jobs=4,
        verbosity=0,
    )
    final.fit(X_shadow, y_shadow)
    scores = final.predict_proba(X_real)[:, 1]

    out_df = pd.DataFrame({"id": real_df["suspect_id"].astype(int), "score": scores.astype(float)})
    out_df = out_df.sort_values("id").reset_index(drop=True)
    assert len(out_df) == 360, f"expected 360 rows, got {len(out_df)}"
    assert out_df["id"].min() == 0 and out_df["id"].max() == 359
    assert out_df["score"].notna().all() and np.isfinite(out_df["score"].values).all()

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\n=== feature importances (top 15) ===")
    importances = pd.Series(final.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(importances.head(15).to_string())
    print(f"\n[done] wrote {len(out_df)} rows → {args.out}")
    print(out_df.describe())


if __name__ == "__main__":
    main()
