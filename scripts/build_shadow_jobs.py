"""Generate the shadow-model job plan.

Outputs:
    shadows/jobs.json   — list of shadow-suspect jobs; each entry has
                          {id, kind, label, out, cmd}. Job idx is the
                          HTCondor Process ID that runs it.
    shadows/labels.csv  — suspect_id, label, kind. Used by the meta-classifier.

Composition (defaults):
    20 independents (train from scratch on different 40k CIFAR-100 subsets)
    5  noise        (derive_shadow.py noise@varying scales)
    5  quant        (derive_shadow.py quant@varying bits)
    5  fine-tune    (train_shadow.py --init-from target_model, fewer epochs)
    5  distill      (train_shadow.py --mode distill --distill-target target_model)

Crucially, all derivative jobs derive from the REAL target_model/weights.safetensors
(not a separately trained "fake target"). Doing so makes the shadow feature
distribution identical to the real suspect feature distribution — the
meta-classifier never has to cross a domain gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

# All shadow derivatives use the REAL target as their starting point.
TARGET = "target_model/weights.safetensors"
SUSPECTS_DIR = "shadows/suspects"


def shadow_out(idx: int) -> str:
    return f"{SUSPECTS_DIR}/shadow_{idx:03d}.safetensors"


def make_independent(idx: int, seed: int, epochs: int) -> dict:
    return {
        "id": idx, "kind": "independent", "label": 0,
        "out": shadow_out(idx),
        "cmd": [
            "python", "-m", "scripts.train_shadow",
            "--mode", "train",
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--out", shadow_out(idx),
        ],
    }


def make_noise(idx: int, seed: int, scale: float) -> dict:
    return {
        "id": idx, "kind": "noise", "label": 1,
        "out": shadow_out(idx),
        "cmd": [
            "python", "-m", "scripts.derive_shadow",
            "--kind", "noise",
            "--target-path", TARGET,
            "--seed", str(seed),
            "--noise-scale", str(scale),
            "--out", shadow_out(idx),
        ],
    }


def make_quant(idx: int, bits: int) -> dict:
    return {
        "id": idx, "kind": "quant", "label": 1,
        "out": shadow_out(idx),
        "cmd": [
            "python", "-m", "scripts.derive_shadow",
            "--kind", "quant",
            "--target-path", TARGET,
            "--bits", str(bits),
            "--out", shadow_out(idx),
        ],
    }


def make_finetune(idx: int, seed: int, epochs: int) -> dict:
    return {
        "id": idx, "kind": "fine_tune", "label": 1,
        "out": shadow_out(idx),
        "cmd": [
            "python", "-m", "scripts.train_shadow",
            "--mode", "train",
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--init-from", TARGET,
            "--lr", "0.01",
            "--out", shadow_out(idx),
        ],
    }


def make_distill(idx: int, seed: int, epochs: int) -> dict:
    return {
        "id": idx, "kind": "distill", "label": 1,
        "out": shadow_out(idx),
        "cmd": [
            "python", "-m", "scripts.train_shadow",
            "--mode", "distill",
            "--distill-target", TARGET,
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--out", shadow_out(idx),
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-independent", type=int, default=20)
    ap.add_argument("--n-noise", type=int, default=5)
    ap.add_argument("--n-quant", type=int, default=5)
    ap.add_argument("--n-finetune", type=int, default=5)
    ap.add_argument("--n-distill", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25, help="default epochs for from-scratch trainings")
    ap.add_argument("--ft-epochs", type=int, default=8, help="epochs for fine-tune derivatives")
    ap.add_argument("--out-dir", default="shadows")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.out_dir}/suspects").mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    idx = 0
    for s in range(args.n_independent):
        jobs.append(make_independent(idx, seed=2000 + s, epochs=args.epochs))
        idx += 1
    for s in range(args.n_noise):
        scale = [0.02, 0.05, 0.08, 0.12, 0.2][s % 5]
        jobs.append(make_noise(idx, seed=3000 + s, scale=scale))
        idx += 1
    for s in range(args.n_quant):
        bits = [4, 5, 6, 7, 8][s % 5]
        jobs.append(make_quant(idx, bits=bits))
        idx += 1
    for s in range(args.n_finetune):
        jobs.append(make_finetune(idx, seed=4000 + s, epochs=args.ft_epochs))
        idx += 1
    for s in range(args.n_distill):
        jobs.append(make_distill(idx, seed=5000 + s, epochs=args.epochs))
        idx += 1

    with open(f"{args.out_dir}/jobs.json", "w") as f:
        json.dump(jobs, f, indent=2)

    with open(f"{args.out_dir}/labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suspect_id", "label", "kind"])
        for j in jobs:
            w.writerow([j["id"], j["label"], j["kind"]])

    n_total = len(jobs)
    n_pos = sum(j["label"] for j in jobs)
    print(f"Wrote {args.out_dir}/jobs.json: {n_total} jobs ({n_pos} stolen, {n_total - n_pos} not)")
    print(f"  edit cluster/shadow_suspects.sub: set `queue {n_total}` to match")


if __name__ == "__main__":
    main()
