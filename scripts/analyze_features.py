"""Diagnostic for checkpoints/features_shard_*.csv.

Identifies which sub-features are likely informative vs noise. We have no
ground-truth labels, so we use proxies:
  - Low variance → likely uninformative (constant-ish across suspects)
  - Negative within-group rank correlation → likely inverted or broken
  - Bimodality of distribution → suggests two populations (stolen vs not)

Also prints, for each candidate `--groups` subset, the score histogram and
the top-/bottom-K suspects, so we can sanity-check the resulting ranking
before re-submitting.
"""

from __future__ import annotations

import argparse
import sys
from glob import glob
from itertools import combinations

import numpy as np
import pandas as pd


GROUPS: dict[str, list[str]] = {
    "S1": [
        "s1_cos_train", "s1_nkl_t_s_train", "s1_nl2_prob_train", "s1_top1_train",
        "s1_cos_holdout", "s1_nkl_t_s_holdout", "s1_nl2_prob_holdout", "s1_top1_holdout",
        "s1_cos_test", "s1_nkl_t_s_test", "s1_nl2_prob_test", "s1_top1_test",
        "s1_cos_ood", "s1_nkl_t_s_ood", "s1_nl2_prob_ood", "s1_top1_ood",
    ],
    "S2": ["s2_loss_gap_h_t", "s2_loss_gap_te_t", "s2_conf_gap_t_h"],
    "S3": ["s3_raw_l2", "s3_raw_cos", "s3_svd_dist", "s3_perm_l2", "s3_perm_cos"],
    "S4": ["s4_cka_stem", "s4_cka_layer1", "s4_cka_layer2", "s4_cka_layer3",
           "s4_cka_layer4", "s4_cka_penult", "s4_cka_mean"],
}


def rank_norm(s: pd.Series) -> pd.Series:
    r = s.rank(method="average", ascending=True, na_option="bottom")
    if len(r) <= 1:
        return r * 0 + 0.5
    return (r - 1) / (len(r) - 1)


def ensemble(df: pd.DataFrame, groups: list[str]) -> pd.Series:
    group_scores: dict[str, pd.Series] = {}
    for g in groups:
        cols = [c for c in GROUPS[g] if c in df.columns]
        if not cols:
            continue
        sub_ranks = pd.DataFrame({c: rank_norm(df[c]) for c in cols})
        group_scores[g] = sub_ranks.mean(axis=1)
    if not group_scores:
        return pd.Series(np.full(len(df), 0.5))
    return pd.DataFrame({g: rank_norm(s) for g, s in group_scores.items()}).mean(axis=1)


def hist(scores: pd.Series, bins: int = 20) -> None:
    counts, edges = np.histogram(scores, bins=np.linspace(0, 1, bins + 1))
    max_c = max(counts) or 1
    for i, c in enumerate(counts):
        bar = "█" * int(40 * c / max_c)
        print(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}] {bar} ({c})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-glob", default="checkpoints/features_shard_*.csv")
    args = ap.parse_args()

    paths = sorted(glob(args.features_glob))
    if not paths:
        print(f"no files match {args.features_glob}", file=sys.stderr)
        sys.exit(1)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates("suspect_id", keep="last")
    df = df.sort_values("suspect_id").reset_index(drop=True)
    print(f"Loaded {len(df)} suspects from {len(paths)} shards")

    feat_cols = [c for c in df.columns if c != "suspect_id"]

    # 1) Per-feature variance — features with tiny std are useless
    print("\n=== PER-FEATURE STATS (sorted by std ascending; low std = noise) ===")
    print(f"{'feature':<25} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} {'p10':>10} {'p90':>10}")
    rows = []
    for c in feat_cols:
        s = df[c]
        rows.append((c, s.min(), s.max(), s.mean(), s.std(), s.quantile(0.1), s.quantile(0.9)))
    for row in sorted(rows, key=lambda r: r[4]):
        print(f"{row[0]:<25} {row[1]:>10.4f} {row[2]:>10.4f} {row[3]:>10.4f} {row[4]:>10.4f} {row[5]:>10.4f} {row[6]:>10.4f}")

    # 2) Within-group rank correlation — features inside a group should agree
    print("\n=== WITHIN-GROUP RANK CORRELATIONS (avg corr of each feature to its groupmates) ===")
    print("flag ⚠ low/inverted → consider dropping that feature")
    for g, cols in GROUPS.items():
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 2:
            continue
        print(f"\n[{g}] {len(cols)} features")
        corr = df[cols].rank().corr()
        for c in cols:
            others = [x for x in cols if x != c]
            mean_to_group = corr.loc[c, others].mean()
            flag = " ⚠" if mean_to_group < 0.2 else ""
            print(f"  {c:<25} mean-corr-to-group: {mean_to_group:>+.3f}{flag}")

    # 3) Cross-group: correlation of each group's score to the others
    print("\n=== CROSS-GROUP RANK CORRELATIONS ===")
    group_scores = {}
    for g in GROUPS:
        cols = [c for c in GROUPS[g] if c in df.columns]
        if cols:
            group_scores[g] = pd.DataFrame({c: rank_norm(df[c]) for c in cols}).mean(axis=1)
    gs_df = pd.DataFrame(group_scores)
    print(gs_df.rank().corr().round(3).to_string())

    # 4) Single-feature ensemble correlation to the full ensemble
    full_score = ensemble(df, list(GROUPS.keys()))
    print("\n=== EACH FEATURE'S CORR WITH FULL ENSEMBLE SCORE ===")
    print("(negative ⇒ that feature is *fighting* the ensemble)")
    full_rank = full_score.rank()
    rho_rows = []
    for c in feat_cols:
        rho = df[c].rank().corr(full_rank)
        rho_rows.append((c, rho))
    for c, rho in sorted(rho_rows, key=lambda x: x[1]):
        flag = " ⚠ NEGATIVE" if rho < 0 else ""
        print(f"  {c:<25} {rho:>+.4f}{flag}")

    # 5) Try every group-subset combo and print score histogram + spread
    print("\n=== ENSEMBLE SCORE PER GROUP SUBSET ===")
    print(f"{'subset':<20} {'std':>8} {'p90-p10':>10}  histogram")
    keys = list(GROUPS.keys())
    subsets = []
    for r in range(1, len(keys) + 1):
        subsets.extend(combinations(keys, r))
    for subset in subsets:
        score = ensemble(df, list(subset))
        spread = score.quantile(0.9) - score.quantile(0.1)
        label = "+".join(subset)
        print(f"\n{label:<20} std={score.std():.3f}  p90-p10={spread:.3f}")
        hist(score, bins=20)
        # Also: top 10 and bottom 10 suspect IDs for this subset
        ranked = pd.DataFrame({"id": df["suspect_id"], "score": score}).sort_values("score", ascending=False)
        top = ranked.head(10)["id"].tolist()
        bot = ranked.tail(10)["id"].tolist()
        print(f"  top-10 ids:    {top}")
        print(f"  bottom-10 ids: {bot}")


if __name__ == "__main__":
    main()
