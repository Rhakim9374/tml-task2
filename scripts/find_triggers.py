"""Select boundary-trigger probes from shadow predictions.

For each probe input `p`, compute the discrimination AUC of
    fires[s, p] = (suspect_pred[s, p] == target_pred[p])
against the binary shadow label (1=stolen, 0=not stolen) across the shadow
suspects.

Probes with AUC near 1.0 are "stolen-positive triggers" — agreeing with
target on that input is strong evidence of stealing. Probes with AUC near
0.0 are "anti-triggers" — *disagreement* is the signal.

Output: a sorted list of triggers with their AUC + direction, ready for
score_with_triggers.py to apply to the real suspect set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-preds", default="shadows/predictions_ood.npz")
    ap.add_argument("--labels", default="shadows/labels.csv")
    ap.add_argument("--top-k", type=int, default=500)
    ap.add_argument("--min-fire-rate", type=float, default=0.05,
                    help="ignore probes where fewer than this fraction of shadows fire "
                         "(or fewer than 1 - this fraction); they have no contrast")
    ap.add_argument("--include-splits", default=None,
                    help="comma-separated subset of split names from the npz "
                         "(e.g. 'ood' or 'ood,holdout,test'). If unset, all splits used.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        import sys
        print("install sklearn first:  ~/.tml-venv/bin/pip install scikit-learn",
              file=sys.stderr)
        sys.exit(2)

    data = np.load(args.shadow_preds, allow_pickle=True)
    all_suspect_preds = data["suspect_preds"]    # [N, n_probes_total]
    suspect_ids = data["suspect_ids"]            # [N]
    all_target_pred = data["target_pred"]        # [n_probes_total]
    split_names = list(data["split_names"])
    split_offsets = np.asarray(data["split_offsets"])
    N_total_probes = all_suspect_preds.shape[1]

    # Optional split filtering — keep probe indices in the ORIGINAL coordinate
    # system so trigger probe_idx maps back to the full prediction matrix
    # (which is what score_with_triggers.py loads). The orig_indices array
    # below is the mapping filtered_idx → original_idx.
    if args.include_splits:
        wanted = [s.strip() for s in args.include_splits.split(",") if s.strip()]
        for w in wanted:
            if w not in split_names:
                raise ValueError(
                    f"split '{w}' not in npz (available: {split_names})")
        keep = np.zeros(N_total_probes, dtype=bool)
        for w in wanted:
            i = split_names.index(w)
            keep[split_offsets[i]:split_offsets[i + 1]] = True
        suspect_preds = all_suspect_preds[:, keep]
        target_pred = all_target_pred[keep]
        orig_indices = np.where(keep)[0]
        print(f"include-splits filter: kept {keep.sum()}/{N_total_probes} probes "
              f"from {wanted}")
    else:
        suspect_preds = all_suspect_preds
        target_pred = all_target_pred
        orig_indices = np.arange(N_total_probes)
    N, n_probes = suspect_preds.shape

    labels_df = pd.read_csv(args.labels).set_index("suspect_id")
    labels = np.array([int(labels_df.loc[int(i), "label"]) for i in suspect_ids])
    n_pos = int(labels.sum())
    n_neg = int(N - n_pos)
    print(f"loaded {N} shadows × {n_probes} probes  (stolen={n_pos}, not-stolen={n_neg})")

    # fires[s, p] = whether shadow s predicted target's class on probe p
    fires = (suspect_preds == target_pred[None, :]).astype(np.float32)

    fire_rate = fires.mean(axis=0)  # [n_probes]
    # Skip probes that are nearly-constant across shadows (no signal)
    has_contrast = (fire_rate >= args.min_fire_rate) & (fire_rate <= 1.0 - args.min_fire_rate)
    n_contrast = int(has_contrast.sum())
    print(f"probes with contrast (fire rate in [{args.min_fire_rate}, {1 - args.min_fire_rate}]): {n_contrast}")

    aucs = np.full(n_probes, 0.5, dtype=np.float64)
    for p in np.where(has_contrast)[0]:
        try:
            aucs[p] = roc_auc_score(labels, fires[:, p])
        except ValueError:
            aucs[p] = 0.5

    # Discrimination = |AUC - 0.5|. Direction = sign of (AUC - 0.5).
    discrim = np.abs(aucs - 0.5)
    order = np.argsort(discrim)[::-1]  # most discriminative first

    triggers = []
    for p in order[:args.top_k]:
        if discrim[p] < 1e-6:
            break
        triggers.append({
            # Store the ORIGINAL probe index so score_with_triggers can index
            # directly into the full-matrix real_predictions.npz (which has all splits).
            "probe_idx": int(orig_indices[p]),
            "auc": float(aucs[p]),
            "discrim": float(discrim[p]),
            "target_pred": int(target_pred[p]),
            "fire_rate_pos": float(fires[labels == 1, p].mean()),
            "fire_rate_neg": float(fires[labels == 0, p].mean()),
            "direction": "stolen" if aucs[p] > 0.5 else "not_stolen",
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "triggers": triggers,
            "summary": {
                "n_shadows": N,
                "n_probes": n_probes,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "auc_max": float(aucs.max()),
                "auc_min": float(aucs.min()),
                "auc_mean_top100": float(aucs[order[:100]].mean()),
                "n_above_0.7": int((aucs > 0.7).sum()),
                "n_above_0.8": int((aucs > 0.8).sum()),
                "n_below_0.3": int((aucs < 0.3).sum()),
            },
        }, f, indent=2)

    print(f"\nTop 15 triggers (probe_idx, AUC, direction, fire-rate stolen/not):")
    for t in triggers[:15]:
        print(f"  probe {t['probe_idx']:>5}  AUC={t['auc']:.3f}  "
              f"{t['direction']:<11}  fire={t['fire_rate_pos']:.2f}/{t['fire_rate_neg']:.2f}  "
              f"target_pred={t['target_pred']}")
    print(f"\nAUC distribution across {n_probes} probes:")
    print(f"  max={aucs.max():.3f}  min={aucs.min():.3f}  "
          f"|>0.7|={(aucs > 0.7).sum()}  |>0.8|={(aucs > 0.8).sum()}  "
          f"|<0.3|={(aucs < 0.3).sum()}  |<0.2|={(aucs < 0.2).sum()}")
    print(f"[done] wrote {len(triggers)} triggers → {args.out}")


if __name__ == "__main__":
    main()
