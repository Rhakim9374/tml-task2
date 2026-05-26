"""Score real suspects via the shadow-selected boundary triggers.

For each real suspect, compute a weighted-average fingerprint score:
    score = sum_t w_t * indicator_t / sum_t w_t

where
    indicator_t = (suspect_pred[t.probe_idx] == target_pred[t.probe_idx])
                  if t.direction == 'stolen'
                  else (suspect_pred[t.probe_idx] != target_pred[t.probe_idx])
    w_t = |AUC_t - 0.5|   (more-discriminative triggers get more weight)

Higher = more likely stolen. Writes a standard `id, score` submission CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-preds", default="real_predictions_ood.npz")
    ap.add_argument("--triggers", default="shadows/triggers.json")
    ap.add_argument("--top-k", type=int, default=200,
                    help="use top-K from triggers.json (capped at how many were saved)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = np.load(args.real_preds, allow_pickle=True)
    suspect_preds = data["suspect_preds"]    # [N, n_probes]
    suspect_ids = data["suspect_ids"]        # [N]
    target_pred = data["target_pred"]        # [n_probes]
    N, n_probes = suspect_preds.shape

    with open(args.triggers) as f:
        trig_data = json.load(f)
    triggers = trig_data["triggers"][:args.top_k]
    print(f"using top-{len(triggers)} triggers (of {len(trig_data['triggers'])} saved)")

    fires = (suspect_preds == target_pred[None, :]).astype(np.float64)  # [N, n_probes]

    scores = np.zeros(N, dtype=np.float64)
    total_w = 0.0
    n_stolen_dir = 0
    n_not_dir = 0
    for t in triggers:
        p = t["probe_idx"]
        w = t["discrim"]  # |AUC - 0.5|
        if t["direction"] == "stolen":
            scores += w * fires[:, p]
            n_stolen_dir += 1
        else:
            scores += w * (1.0 - fires[:, p])
            n_not_dir += 1
        total_w += w
    if total_w > 0:
        scores /= total_w

    print(f"direction split: {n_stolen_dir} stolen-direction triggers, "
          f"{n_not_dir} not-stolen-direction triggers")

    out_df = pd.DataFrame({"id": suspect_ids.astype(int), "score": scores.astype(float)})
    out_df = out_df.sort_values("id").reset_index(drop=True)

    # The set of real suspect IDs should be 0..359 (or whatever predictions cover)
    # but pad missing if any (every assignment expects 360 rows)
    expected = set(range(360))
    have = set(out_df["id"].tolist())
    missing = expected - have
    if missing:
        # Pad missing suspects with the median score so they don't appear at extremes.
        med = float(out_df["score"].median())
        pad = pd.DataFrame({"id": sorted(missing), "score": [med] * len(missing)})
        out_df = pd.concat([out_df, pad], ignore_index=True).sort_values("id").reset_index(drop=True)
        print(f"[warn] padded {len(missing)} missing suspects with median score {med:.3f}")

    assert len(out_df) == 360
    assert out_df["id"].min() == 0 and out_df["id"].max() == 359
    assert out_df["id"].is_unique
    assert np.isfinite(out_df["score"].values).all()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[done] wrote {len(out_df)} rows → {args.out}")
    print(out_df.describe())


if __name__ == "__main__":
    main()
