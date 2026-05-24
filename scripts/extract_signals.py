"""Extract S1+S2+S3+S4 features for all 360 suspect models.

Pipeline (per suspect):
  1. Forward suspect on train_main / holdout / test / OOD loaders → logits
  2. S1: output agreement vs cached target logits on each split
  3. S2: cross-entropy loss + accuracy + confidence + gap features
  4. Forward suspect on a small probe batch with hooks → activations
  5. S4: linear CKA at stem / layer1..4 / penultimate (uses pristine suspect)
  6. S3: raw L2/cos on flat params, SVD spectrum distance, then mutate suspect
        via activation-matched permutation and recompute L2/cos

The target is loaded once and its logits cached. Suspects are loaded one at
a time; mutated weights are never written back to disk.

Output: checkpoints/features.csv (one row per suspect, all sub-features)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import (
    get_cifar100_splits,
    get_ood_probe,
    make_loader,
)
from src.model import load_weights, make_model
from src.signals.cka import cka_features
from src.signals.dataset_inference import gap_features, split_features
from src.signals.output_agreement import agreement_features, forward_logits
from src.signals.weight_align import (
    align_suspect_to_target,
    aligned_weight_features,
    raw_weight_features,
    svd_spectrum_distance,
)


def collect_labels(loader: DataLoader) -> torch.Tensor:
    ys = []
    for batch in loader:
        y = batch[1] if isinstance(batch, (list, tuple)) else None
        if y is None:
            raise RuntimeError("loader returned no labels")
        ys.append(y)
    return torch.cat(ys, dim=0)


def collect_first_images(loader: DataLoader, n: int) -> torch.Tensor:
    """Concatenate first n images across batches into a single CPU tensor."""
    chunks = []
    got = 0
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        take = min(x.shape[0], n - got)
        chunks.append(x[:take])
        got += take
        if got >= n:
            break
    return torch.cat(chunks, dim=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target-path", default="target_model/weights.safetensors")
    p.add_argument("--target-dir", default="target_model")
    p.add_argument("--suspects-dir", default="suspect_models")
    p.add_argument("--num-suspects", type=int, default=360,
                   help="upper bound on suspect index (used when --end is not given)")
    p.add_argument("--start", type=int, default=0,
                   help="first suspect index to process (inclusive)")
    p.add_argument("--end", type=int, default=None,
                   help="one past the last suspect index to process; default = --num-suspects")
    p.add_argument("--data-root", default="./cifar100_data")
    p.add_argument("--ood-root", default="./cifar10_data")
    p.add_argument("--ood-n", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--align-probe-n", type=int, default=512, help="batch size for S3 alignment")
    p.add_argument("--cka-probe-n", type=int, default=1024, help="batch size for S4 CKA")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="checkpoints/features.csv")
    p.add_argument("--no-ood", action="store_true", help="skip OOD probe (S1 only)")
    p.add_argument("--no-align", action="store_true", help="skip S3 permutation alignment")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] device={device}")
    print(f"[setup] loading CIFAR-100 splits …")
    train_main, holdout, test = get_cifar100_splits(args.data_root, args.target_dir)
    print(f"        train_main={len(train_main)} holdout={len(holdout)} test={len(test)}")

    loaders = {
        "train": make_loader(train_main, args.batch_size, args.num_workers),
        "holdout": make_loader(holdout, args.batch_size, args.num_workers),
        "test": make_loader(test, args.batch_size, args.num_workers),
    }
    if not args.no_ood:
        ood = get_ood_probe(args.ood_root, args.ood_n)
        loaders["ood"] = make_loader(ood, args.batch_size, args.num_workers)
        print(f"        ood(cifar10)={len(ood)}")

    # cache target logits and labels per split
    print(f"[target] loading {args.target_path}")
    target = load_weights(args.target_path, device=device)
    target_logits = {}
    labels = {}
    for split, loader in loaders.items():
        t0 = time.time()
        target_logits[split] = forward_logits(target, loader, device)
        if split in {"train", "holdout", "test"}:
            labels[split] = collect_labels(loader)
        print(f"[target] {split}: logits={tuple(target_logits[split].shape)} in {time.time() - t0:.1f}s")

    # small fixed probe for S3 alignment and S4 CKA — drawn from the test set for cleanliness
    print(f"[probe] gathering align/cka batches (align_n={args.align_probe_n}, cka_n={args.cka_probe_n})")
    align_probe = collect_first_images(loaders["test"], args.align_probe_n)
    cka_probe = collect_first_images(loaders["test"], args.cka_probe_n)

    # free target weights from GPU between uses to keep memory low (we keep on CPU between suspects)
    # but for S3 alignment we re-call its forward, so keep on GPU
    target.to(device)
    target.eval()

    end = args.end if args.end is not None else args.num_suspects
    print(f"[range] processing suspects [{args.start}, {end})")

    rows: list[dict] = []
    for i in range(args.start, end):
        path = Path(args.suspects_dir) / f"suspect_{i:03d}.safetensors"
        t_susp = time.time()
        suspect = load_weights(str(path), device=device)

        # === forward suspect on all loaders → logits ===
        susp_logits = {}
        for split, loader in loaders.items():
            susp_logits[split] = forward_logits(suspect, loader, device)

        # === S1 output agreement ===
        s1 = {}
        for split in loaders.keys():
            s1.update(agreement_features(target_logits[split], susp_logits[split], prefix=split))

        # === S2 dataset inference ===
        s2_per = {}
        for split in ["train", "holdout", "test"]:
            s2_per.update(split_features(susp_logits[split], labels[split], prefix=split))
        s2 = {**s2_per, **gap_features(s2_per)}

        # === S4 CKA (pristine suspect — do BEFORE alignment mutation) ===
        s4 = cka_features(target, suspect, cka_probe, device)

        # === S3 weight features ===
        s3 = {}
        s3.update(raw_weight_features(target, suspect))
        s3.update(svd_spectrum_distance(target, suspect))
        if not args.no_align:
            # mutates suspect in place
            suspect = align_suspect_to_target(target, suspect, align_probe, device)
            s3.update(aligned_weight_features(target, suspect))

        row = {"suspect_id": i, **s1, **s2, **s3, **s4}
        rows.append(row)

        # free suspect from GPU before loading the next
        del suspect
        if device != "cpu":
            torch.cuda.empty_cache()

        if (i + 1) % 10 == 0 or i == end - 1:
            dt = time.time() - t_susp
            df_partial = pd.DataFrame(rows)
            df_partial.to_csv(args.out, index=False)
            print(f"[suspect {i:3d}] done in {dt:.1f}s. wrote partial → {args.out}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[done] wrote {len(df)} suspects × {len(df.columns) - 1} features → {args.out}")


if __name__ == "__main__":
    main()
