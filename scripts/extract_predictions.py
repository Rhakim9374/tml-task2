"""Save per-suspect argmax predictions on probe sets for trigger selection.

Used by find_triggers.py to identify boundary-trigger inputs — probes where
'suspect_pred == target_pred' is highly discriminative of stolen vs not-stolen
across the shadow set. Stolen models inherit the target's idiosyncratic
prediction on those probes; independents don't.

Output `.npz` file contains:
    target_pred:    [n_probes_total] int64   target's argmax per probe
    suspect_ids:    [N]              int32   suspect IDs that succeeded
    suspect_preds:  [N, n_probes_total] int64   each suspect's argmax per probe
    split_names:    object array            ['train', 'ood', 'holdout', 'test']
    split_offsets:  int32 array             cumulative split boundaries

The trigger pipeline runs this on both the shadow and real suspect sets with
the same probe configuration so find_triggers can mine on shadow predictions
and score_with_triggers can apply those probe indices to the real matrix.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.data import get_cifar100_splits, get_ood_probe, make_loader
from src.model import load_weights
from src.signals.output_agreement import forward_logits


def build_loader(split: str, args):
    """Return a DataLoader for the requested split."""
    if split == "ood":
        ds = get_ood_probe(args.ood_root, args.ood_n)
    else:
        train, holdout, test = get_cifar100_splits(args.data_root, args.target_dir)
        if split == "train":
            ds = train
        elif split == "holdout":
            ds = holdout
        elif split == "test":
            ds = test
        else:
            raise ValueError(f"unknown split: {split}")
    return make_loader(ds, args.batch_size, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-path", default="target_model/weights.safetensors")
    ap.add_argument("--target-dir", default="target_model")
    ap.add_argument("--suspects-dir", required=True,
                    help="directory containing suspect_NNN.safetensors")
    ap.add_argument("--num-suspects", type=int, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--splits", default="ood",
                    help="comma-separated probe splits to use (ood/holdout/test/train)")
    ap.add_argument("--data-root", default="./cifar100_data")
    ap.add_argument("--ood-root", default="./cifar10_data")
    ap.add_argument("--ood-n", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print(f"[setup] device={args.device}  splits={splits}")

    # Build a loader per split and concatenate all probes for one big argmax pass
    loaders = {s: build_loader(s, args) for s in splits}
    split_sizes = {s: len(loaders[s].dataset) for s in splits}
    total = sum(split_sizes.values())
    print(f"        probe counts: {split_sizes} (total={total})")

    print(f"[target] loading {args.target_path}")
    target = load_weights(args.target_path, device=args.device)
    target_pred_parts = []
    for s in splits:
        t0 = time.time()
        logits = forward_logits(target, loaders[s], args.device)
        target_pred_parts.append(logits.argmax(dim=1).numpy().astype(np.int64))
        print(f"[target] {s}: {logits.shape} in {time.time() - t0:.1f}s")
    target_pred = np.concatenate(target_pred_parts, axis=0)

    end = args.end if args.end is not None else args.num_suspects
    print(f"[range] processing suspects [{args.start}, {end})")

    suspect_ids: list[int] = []
    suspect_preds_rows: list[np.ndarray] = []
    for i in range(args.start, end):
        path = Path(args.suspects_dir) / f"suspect_{i:03d}.safetensors"
        if not path.exists():
            print(f"[skip] suspect_{i:03d}: missing")
            continue
        try:
            suspect = load_weights(str(path), device=args.device)
            parts = []
            for s in splits:
                logits = forward_logits(suspect, loaders[s], args.device)
                parts.append(logits.argmax(dim=1).numpy().astype(np.int64))
            suspect_preds_rows.append(np.concatenate(parts, axis=0))
            suspect_ids.append(i)
            del suspect
            if args.device != "cpu":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[suspect {i:3d}] FAILED ({type(e).__name__}: {e}); skipping",
                  flush=True)

    if not suspect_ids:
        raise RuntimeError("no suspects processed successfully")

    suspect_preds_arr = np.stack(suspect_preds_rows, axis=0)
    suspect_ids_arr = np.array(suspect_ids, dtype=np.int32)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Also save split boundary offsets so downstream can subset to e.g. just OOD.
    offsets = np.cumsum([0] + [split_sizes[s] for s in splits])
    np.savez(
        args.out,
        suspect_preds=suspect_preds_arr,
        suspect_ids=suspect_ids_arr,
        target_pred=target_pred,
        split_names=np.array(splits, dtype=object),
        split_offsets=offsets.astype(np.int32),
    )
    print(f"[done] saved {len(suspect_ids)} suspects × {total} probes → {args.out}")


if __name__ == "__main__":
    main()
