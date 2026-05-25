"""Blend multiple submission CSVs by weighted rank-average.

Use this to combine the heuristic ensemble (combine_and_submit.py output)
with the shadow-trained meta-classifier (train_meta_classifier.py output),
or any other pair of submissions. Each input is rank-normalized independently
(so wildly different score scales align), then a weighted average is taken.

Example:
    python -m scripts.blend_submissions \\
        --inputs "submissions/submission.csv:1.0,submissions/submission_meta.csv:2.0" \\
        --out submissions/submission_blended.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def rank_norm(s: pd.Series) -> pd.Series:
    r = s.rank(method="average", ascending=True, na_option="bottom")
    if len(r) <= 1:
        return r * 0 + 0.5
    return (r - 1) / (len(r) - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        required=True,
        help="Comma-separated 'path:weight' pairs, e.g. "
             "'submissions/submission.csv:1.0,submissions/submission_meta.csv:2.0'. "
             "Each file must have id,score columns; ids must match across all inputs.",
    )
    ap.add_argument("--out", default="submissions/submission_blended.csv")
    args = ap.parse_args()

    pairs: list[tuple[str, float]] = []
    for spec in args.inputs.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            raise ValueError(f"missing ':weight' in input spec: {spec!r}")
        path, w_str = spec.rsplit(":", 1)
        pairs.append((path.strip(), float(w_str)))
    if not pairs:
        raise ValueError("--inputs is empty")

    # Load + validate
    dfs: list[tuple[pd.DataFrame, float]] = []
    for path, w in pairs:
        d = pd.read_csv(path)
        if "id" not in d.columns or "score" not in d.columns:
            raise ValueError(f"{path} missing id/score columns; has {list(d.columns)}")
        d = d.sort_values("id").reset_index(drop=True)
        dfs.append((d, w))

    base_ids = dfs[0][0]["id"].values
    for d, _ in dfs[1:]:
        if not (d["id"].values == base_ids).all():
            raise ValueError("submission id columns must all match across inputs")

    # Rank-normalize each input, then weighted-mean
    total_w = sum(w for _, w in dfs)
    blended = sum(w * rank_norm(d["score"]) for d, w in dfs) / total_w

    out_df = pd.DataFrame({"id": base_ids.astype(int), "score": blended.astype(float)})
    out_df = out_df.sort_values("id").reset_index(drop=True)

    # Submission sanity checks (mirror combine_and_submit.py)
    assert len(out_df) == 360, f"expected 360 rows, got {len(out_df)}"
    assert out_df["id"].min() == 0 and out_df["id"].max() == 359
    assert out_df["id"].is_unique
    assert out_df["score"].notna().all() and (out_df["score"].abs() < float("inf")).all()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print("[blend] inputs:")
    for path, w in pairs:
        print(f"    {path}  weight={w}")
    print(f"[done] wrote {len(out_df)} rows → {args.out}")
    print(out_df.describe())


if __name__ == "__main__":
    main()
