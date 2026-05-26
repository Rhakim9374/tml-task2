"""Combine S1+S2+S3+S4 features into a single stealing score and write
submission.csv. All sub-features in features.csv are already oriented so
higher = more likely stolen.

Strategy:
    1. Rank each sub-feature across the 360 suspects (rank in [0, 1]).
    2. Average the ranks within each signal group (S1, S2, S3, S4)
       → per-group score in [0, 1].
    3. Rank-average the four group scores → final score in [0, 1].

This gives every group equal weight, which is more robust than naively
averaging features that have wildly different scales or where one group
contributes many more features than another.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# core sub-features used in the ensemble. Every column listed must exist in
# features.csv and must be oriented so higher = more stolen.
GROUPS: dict[str, list[str]] = {
    "S1": [
        # output agreement on each probe split
        "s1_cos_train", "s1_nkl_t_s_train", "s1_nl2_prob_train", "s1_top1_train",
        "s1_cos_holdout", "s1_nkl_t_s_holdout", "s1_nl2_prob_holdout", "s1_top1_holdout",
        "s1_cos_test", "s1_nkl_t_s_test", "s1_nl2_prob_test", "s1_top1_test",
        "s1_cos_ood", "s1_nkl_t_s_ood", "s1_nl2_prob_ood", "s1_top1_ood",
    ],
    "S1f": [
        # fine-grained agreement: mistake-copying, top-k overlap, low-confidence agreement
        "s1_mistake_train", "s1_mistake_holdout", "s1_mistake_test",
        "s1_top3_overlap_train", "s1_top3_overlap_holdout",
        "s1_top3_overlap_test", "s1_top3_overlap_ood",
        "s1_top5_overlap_train", "s1_top5_overlap_holdout",
        "s1_top5_overlap_test", "s1_top5_overlap_ood",
        "s1_low_conf_agree_train", "s1_low_conf_agree_holdout", "s1_low_conf_agree_test",
    ],
    "S2": [
        "s2_loss_gap_h_t",
        "s2_loss_gap_te_t",
        "s2_conf_gap_t_h",
    ],
    "S3": [
        "s3_raw_l2", "s3_raw_cos",
        "s3_svd_dist",
        "s3_perm_l2", "s3_perm_cos",
    ],
    "S4": [
        "s4_cka_stem",
        "s4_cka_layer1", "s4_cka_layer2", "s4_cka_layer3", "s4_cka_layer4",
        "s4_cka_penult", "s4_cka_mean",
    ],
    "S5": [
        # decision-boundary fingerprint: input-gradient cosine + FGSM transfer rate
        "s5_grad_cos_train", "s5_grad_cos_holdout",
        "s5_grad_cos_test", "s5_grad_cos_ood",
        "s5_adv_transfer_train", "s5_adv_transfer_holdout",
        "s5_adv_transfer_test", "s5_adv_transfer_ood",
    ],
    "S6": [
        # ModelDiff-style decision-distance vector (DDV) similarity
        "s6_ddv_corr_train", "s6_ddv_corr_holdout", "s6_ddv_corr_test", "s6_ddv_corr_ood",
        "s6_ddv_cos_train", "s6_ddv_cos_holdout", "s6_ddv_cos_test", "s6_ddv_cos_ood",
        "s6_ddv_flip_agree_train", "s6_ddv_flip_agree_holdout",
        "s6_ddv_flip_agree_test", "s6_ddv_flip_agree_ood",
    ],
}


def rank_norm(series: pd.Series) -> pd.Series:
    """Average-rank in [0, 1] where the largest value gets 1."""
    # method="average" handles ties evenly; ascending=True → smallest gets rank 1
    r = series.rank(method="average", ascending=True, na_option="bottom")
    if len(r) <= 1:
        return r * 0 + 0.5
    return (r - 1) / (len(r) - 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="checkpoints/features.csv")
    p.add_argument("--out", default="submissions/submission.csv")
    p.add_argument(
        "--groups",
        default="S1,S1f,S2,S3,S4,S5,S6",
        help="comma-separated subset of groups to include in the ensemble",
    )
    p.add_argument(
        "--mode",
        choices=["mean", "max", "temperature"],
        default="mean",
        help="how to combine across groups: equal/weighted mean, max of ranks, "
             "or per-suspect temperature-softmax (emphasises confident high ranks)",
    )
    p.add_argument(
        "--group-weights",
        default=None,
        help="for mode=mean: per-group weights, e.g. 'S1=1.5,S2=0.3,S3=1.0,S4=1.5'. "
             "Missing groups default to 1.0. Ignored for mode=max.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=3.0,
        help="for mode=temperature: beta in exp(beta * (rank - 0.5)); larger → more max-like",
    )
    return p.parse_args()


def main():
    args = parse_args()
    # --features may be a single path or a glob (e.g., "checkpoints/features_shard_*.csv")
    if any(c in args.features for c in "*?["):
        from glob import glob
        paths = sorted(glob(args.features))
        if not paths:
            raise FileNotFoundError(f"no files match {args.features}")
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        print(f"[combine] read {len(df)} rows from {len(paths)} shard files")
    else:
        df = pd.read_csv(args.features)
    assert "suspect_id" in df.columns, "features.csv missing suspect_id"
    df = df.drop_duplicates(subset="suspect_id", keep="last")
    df = df.sort_values("suspect_id").reset_index(drop=True)

    selected_groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    group_scores: dict[str, pd.Series] = {}
    for g in selected_groups:
        cols = [c for c in GROUPS[g] if c in df.columns]
        if not cols:
            print(f"[warn] group {g}: no columns present in features.csv, skipping")
            continue
        missing = [c for c in GROUPS[g] if c not in df.columns]
        if missing:
            print(f"[warn] group {g}: missing {missing}")
        sub_ranks = pd.DataFrame({c: rank_norm(df[c]) for c in cols})
        group_scores[g] = sub_ranks.mean(axis=1)
        print(f"[group {g}] used {len(cols)} sub-features")

    if not group_scores:
        raise RuntimeError("no signal groups available — re-run extract_signals.py")

    # rank-average across groups (each group has equal weight by default)
    group_rank_df = pd.DataFrame({g: rank_norm(s) for g, s in group_scores.items()})

    if args.mode == "max":
        final_score = group_rank_df.max(axis=1)
        print(f"[mode] max over groups: {list(group_rank_df.columns)}")
    elif args.mode == "temperature":
        import numpy as np
        ranks = group_rank_df.values
        w = np.exp(args.temperature * (ranks - 0.5))
        final_score = pd.Series((w * ranks).sum(axis=1) / w.sum(axis=1),
                                index=group_rank_df.index)
        print(f"[mode] temperature={args.temperature} over groups: "
              f"{list(group_rank_df.columns)}")
    else:  # mean
        if args.group_weights:
            weights = {g: 1.0 for g in group_rank_df.columns}
            for spec in args.group_weights.split(","):
                if "=" not in spec:
                    raise ValueError(f"bad --group-weights spec: {spec!r}")
                g, w = spec.split("=")
                weights[g.strip()] = float(w)
            total = sum(weights[g] for g in group_rank_df.columns)
            final_score = sum(weights[g] * group_rank_df[g]
                              for g in group_rank_df.columns) / total
            print(f"[mode] weighted mean: " +
                  ", ".join(f"{g}={weights[g]}" for g in group_rank_df.columns))
        else:
            final_score = group_rank_df.mean(axis=1)
            print(f"[mode] equal mean over groups: {list(group_rank_df.columns)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({"id": df["suspect_id"].astype(int), "score": final_score.astype(float)})
    out_df = out_df.sort_values("id").reset_index(drop=True)

    # sanity checks before writing
    assert len(out_df) == 360, f"expected 360 rows, got {len(out_df)}"
    assert out_df["id"].min() == 0 and out_df["id"].max() == 359
    assert out_df["id"].is_unique
    assert out_df["score"].notna().all() and (out_df["score"].abs() < float("inf")).all()

    out_df.to_csv(args.out, index=False)
    print(f"[done] wrote {len(out_df)} rows → {args.out}")
    print(out_df.describe())


if __name__ == "__main__":
    main()
