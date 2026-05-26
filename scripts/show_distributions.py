"""Compare distributions and rank overlaps across multiple submission CSVs.

Used before submitting a blend, to sanity-check that:
  - The blend's score spread isn't degenerate (no piling-up at one end).
  - The candidate submissions agree on the obvious cases (large bottom-K overlap)
    but disagree on the hard cases (top-K overlap measures classifier consensus).
  - Spearman correlations between candidate rankings tell us if any one
    candidate is wildly off (low correlation = outlier signal).

Usage:
    python -m scripts.show_distributions \\
        --inputs submissions/heur_weighted.csv submissions/heur_max.csv \\
                 submissions/submission_meta.csv submissions/final.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs", nargs="+",
        default=[
            "submissions/heur_weighted.csv",
            "submissions/heur_max.csv",
            "submissions/submission_meta.csv",
            "submissions/final.csv",
        ],
        help="submission CSV paths to compare",
    )
    ap.add_argument("--top-k", type=int, default=18,
                    help="number of top/bottom suspects to compare (default 18 = top 5%% of 360)")
    args = ap.parse_args()

    paths = []
    data: dict[str, pd.DataFrame] = {}
    for p in args.inputs:
        if not Path(p).exists():
            print(f"[skip] {p} not found", file=sys.stderr)
            continue
        name = Path(p).stem
        paths.append(name)
        data[name] = pd.read_csv(p).sort_values("id").reset_index(drop=True)

    if not paths:
        print("no input files found", file=sys.stderr)
        sys.exit(1)

    # -------- distributions --------
    print("=" * 100)
    print("Score-distribution stats")
    print("=" * 100)
    print(f"{'name':<22} {'mean':>8} {'std':>8} "
          f"{'min':>8} {'p10':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p90':>8} {'max':>8}")
    for p in paths:
        s = data[p]["score"]
        print(f"{p:<22} "
              f"{s.mean():>8.3f} {s.std():>8.3f} "
              f"{s.min():>8.3f} "
              f"{s.quantile(.10):>8.3f} {s.quantile(.25):>8.3f} {s.quantile(.50):>8.3f} "
              f"{s.quantile(.75):>8.3f} {s.quantile(.90):>8.3f} "
              f"{s.max():>8.3f}")

    # -------- pairwise Spearman correlation (rank-based) --------
    print()
    print("=" * 100)
    print(f"Pairwise Spearman correlation of rankings "
          f"(low = candidates disagree about the order)")
    print("=" * 100)
    header = " " * 22 + "  ".join(f"{p:>14}" for p in paths)
    print(header)
    for p1 in paths:
        r1 = data[p1]["score"].rank()
        row = f"{p1:<22}"
        for p2 in paths:
            r2 = data[p2]["score"].rank()
            rho = r1.corr(r2)
            row += f"  {rho:>14.3f}"
        print(row)

    # -------- top-K overlap --------
    print()
    print("=" * 100)
    print(f"Overlap of top-{args.top_k} (≈top 5%%) suspect sets "
          f"(high = candidates agree on most-stolen)")
    print("=" * 100)
    top_sets = {p: set(data[p].nlargest(args.top_k, "score")["id"].tolist()) for p in paths}
    print(header)
    for p1 in paths:
        row = f"{p1:<22}"
        for p2 in paths:
            row += f"  {len(top_sets[p1] & top_sets[p2]):>14d}"
        print(row)

    # -------- bottom-K overlap --------
    print()
    print("=" * 100)
    print(f"Overlap of bottom-{args.top_k} suspect sets "
          f"(high = candidates agree on least-stolen)")
    print("=" * 100)
    bot_sets = {p: set(data[p].nsmallest(args.top_k, "score")["id"].tolist()) for p in paths}
    print(header)
    for p1 in paths:
        row = f"{p1:<22}"
        for p2 in paths:
            row += f"  {len(bot_sets[p1] & bot_sets[p2]):>14d}"
        print(row)

    # -------- top-K IDs per submission --------
    print()
    print("=" * 100)
    print(f"Top-{args.top_k} suspect IDs per submission")
    print("=" * 100)
    for p in paths:
        ids = sorted(top_sets[p])
        print(f"{p:<22} {ids}")


if __name__ == "__main__":
    main()
