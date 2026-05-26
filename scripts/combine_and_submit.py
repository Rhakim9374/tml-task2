"""Heuristic ensemble of the extracted feature groups → submission CSV.

For each requested group we rank-normalize every sub-feature across the 360
suspects, then average the ranks within the group. We then rank-normalize each
group score and take the **max** across groups: a suspect is considered
suspicious if ANY single signal group ranks it near the top.

This is the "anchor" submission in the blend that produced 0.703704 — robust
to scale differences between feature groups and aggressive at promoting any
suspect that one signal strongly suspects.

Usage (the exact call from the best-run recipe):

    python -m scripts.combine_and_submit \\
        --features "checkpoints/features_shard_*.csv" \\
        --groups "S1,S1f,S3,S4,S5,S6" \\
        --out submissions/heur_v4_max.csv
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import pandas as pd


# All sub-features that extract_signals.py produces, grouped by signal.
# Every column is oriented so higher = more likely stolen.
GROUPS: dict[str, list[str]] = {
    "S1": [  # output agreement on each probe split
        "s1_cos_train", "s1_nkl_t_s_train", "s1_nl2_prob_train", "s1_top1_train",
        "s1_cos_holdout", "s1_nkl_t_s_holdout", "s1_nl2_prob_holdout", "s1_top1_holdout",
        "s1_cos_test", "s1_nkl_t_s_test", "s1_nl2_prob_test", "s1_top1_test",
        "s1_cos_ood", "s1_nkl_t_s_ood", "s1_nl2_prob_ood", "s1_top1_ood",
    ],
    "S1f": [  # fine-grained: mistake-copying, top-k overlap, low-confidence agreement
        "s1_mistake_train", "s1_mistake_holdout", "s1_mistake_test",
        "s1_top3_overlap_train", "s1_top3_overlap_holdout",
        "s1_top3_overlap_test", "s1_top3_overlap_ood",
        "s1_top5_overlap_train", "s1_top5_overlap_holdout",
        "s1_top5_overlap_test", "s1_top5_overlap_ood",
        "s1_low_conf_agree_train", "s1_low_conf_agree_holdout", "s1_low_conf_agree_test",
    ],
    "S2": [  # dataset-inference gaps
        "s2_loss_gap_h_t", "s2_loss_gap_te_t", "s2_conf_gap_t_h",
    ],
    "S3": [  # weight-space distance (raw + SVD + activation-aligned permutation)
        "s3_raw_l2", "s3_raw_cos", "s3_svd_dist", "s3_perm_l2", "s3_perm_cos",
    ],
    "S4": [  # linear CKA at each stage + penultimate
        "s4_cka_stem",
        "s4_cka_layer1", "s4_cka_layer2", "s4_cka_layer3", "s4_cka_layer4",
        "s4_cka_penult", "s4_cka_mean",
    ],
    "S5": [  # decision-boundary fingerprint: input-gradient cosine + FGSM transfer
        "s5_grad_cos_train", "s5_grad_cos_holdout",
        "s5_grad_cos_test", "s5_grad_cos_ood",
        "s5_adv_transfer_train", "s5_adv_transfer_holdout",
        "s5_adv_transfer_test", "s5_adv_transfer_ood",
    ],
    "S6": [  # ModelDiff decision-distance vector similarity
        "s6_ddv_corr_train", "s6_ddv_corr_holdout", "s6_ddv_corr_test", "s6_ddv_corr_ood",
        "s6_ddv_cos_train", "s6_ddv_cos_holdout", "s6_ddv_cos_test", "s6_ddv_cos_ood",
        "s6_ddv_flip_agree_train", "s6_ddv_flip_agree_holdout",
        "s6_ddv_flip_agree_test", "s6_ddv_flip_agree_ood",
    ],
}


def rank_norm(series: pd.Series) -> pd.Series:
    """Average-rank in [0, 1] where the largest value gets 1."""
    r = series.rank(method="average", ascending=True, na_option="bottom")
    if len(r) <= 1:
        return r * 0 + 0.5
    return (r - 1) / (len(r) - 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="checkpoints/features_shard_*.csv",
                   help="path or glob to per-shard features CSVs from extract_signals.py")
    p.add_argument("--groups", default="S1,S1f,S3,S4,S5,S6",
                   help="comma-separated subset of GROUPS to include (best run drops S2)")
    p.add_argument("--out", default="submissions/heur_v4_max.csv")
    args = p.parse_args()

    paths = sorted(glob(args.features)) if any(c in args.features for c in "*?[") else [args.features]
    if not paths:
        raise FileNotFoundError(f"no files match {args.features}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates(subset="suspect_id", keep="last")
    df = df.sort_values("suspect_id").reset_index(drop=True)
    print(f"[combine] read {len(df)} rows from {len(paths)} shard file(s)")

    selected = [g.strip() for g in args.groups.split(",") if g.strip()]
    group_scores: dict[str, pd.Series] = {}
    for g in selected:
        cols = [c for c in GROUPS[g] if c in df.columns]
        if not cols:
            print(f"[warn] group {g}: no columns in features, skipping")
            continue
        missing = [c for c in GROUPS[g] if c not in df.columns]
        if missing:
            print(f"[warn] group {g}: missing {missing}")
        sub_ranks = pd.DataFrame({c: rank_norm(df[c]) for c in cols})
        group_scores[g] = sub_ranks.mean(axis=1)
        print(f"[group {g}] {len(cols)} sub-features")
    if not group_scores:
        raise RuntimeError("no signal groups available")

    # max over rank-normalized group scores: best run's "anchor" submission
    group_rank_df = pd.DataFrame({g: rank_norm(s) for g, s in group_scores.items()})
    final_score = group_rank_df.max(axis=1)
    print(f"[mode] max over groups: {list(group_rank_df.columns)}")

    out_df = pd.DataFrame({"id": df["suspect_id"].astype(int),
                           "score": final_score.astype(float)})
    out_df = out_df.sort_values("id").reset_index(drop=True)
    assert len(out_df) == 360, f"expected 360 rows, got {len(out_df)}"
    assert out_df["id"].min() == 0 and out_df["id"].max() == 359
    assert out_df["id"].is_unique
    assert out_df["score"].notna().all() and (out_df["score"].abs() < float("inf")).all()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[done] wrote {len(out_df)} rows → {args.out}")
    print(out_df.describe())


if __name__ == "__main__":
    main()
