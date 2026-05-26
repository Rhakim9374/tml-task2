"""Generate the shadow-model job plan (v2 — 70 shadows, varied derivatives).

Designed to cover the full space of "stolen vs not-stolen" cases described in
the assignment, with enough variety that the meta-classifier doesn't trivially
overfit (the v1 set produced CV AUC 1.0 because every kind had a cartoonish
signature). In particular:

  * 25 independents — varied seeds × epochs × subsets, with target's exact
    training recipe (biased crop + SGD + cosine), so they look like
    well-trained-but-not-stolen models.
  * 5 "evil twin" independents — trained on the *exact* same train indices
    as target with target's recipe and a different seed. Hardest case to
    distinguish: same data, same recipe, just a different random init.
    Forces the meta-classifier to learn that "same data ≠ stolen".
  * 10 fine-tunes — start from target_model, train more on CIFAR-100 with
    varied epoch counts (1, 2, 3, 5, 8, 12, 15, 20, 1, 3) and LRs.
  * 10 distillations — train fresh student on target's outputs (KD loss),
    varied temperatures (1, 2, 4, 8) and epoch counts.
  * 10 partial fine-tunes — start from target, freeze all but `fc` (last
    layer only) or `fc,layer4` (last block) and train a bit. Common stealing
    pattern (LoRA-style head-swap or output-recalibration).
  * 5 small-noise + 5 quant — function-preserving perturbations.

All trained models that go through gradient descent use:
  * Biased random crop (bias_x=0.5, bias_y=-0.25, jitter=0.25, reflect-pad=4)
  * SGD momentum 0.9, weight decay 5e-4
  * LR 0.1 cosine, batch size 256
  * No label smoothing
  ... matching the target's recipe verbatim per the assignment PDF.

Outputs:
    shadows/jobs.json  — one entry per shadow with {id, kind, label, out, cmd}
    shadows/labels.csv — suspect_id, label, kind (for the meta-classifier)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# All shadow derivatives use the REAL target as their starting point.
TARGET = "target_model/weights.safetensors"
TARGET_INDICES = "target_model/train_main_idx.json"
SUSPECTS_DIR = "shadows/suspects"


def shadow_out(idx: int) -> str:
    # extract_signals.py expects `suspect_NNN.safetensors` — must match.
    return f"{SUSPECTS_DIR}/suspect_{idx:03d}.safetensors"


def cmd_train(out: str, *, seed: int, epochs: int, lr: float = 0.1,
              init_from: str | None = None, train_indices: str | None = None,
              freeze_except: str | None = None) -> list[str]:
    c = [
        "python", "-m", "scripts.train_shadow",
        "--mode", "train",
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch-size", "256",
        "--out", out,
    ]
    if init_from:
        c += ["--init-from", init_from]
    if train_indices:
        c += ["--train-indices-json", train_indices]
    if freeze_except:
        c += ["--freeze-except", freeze_except]
    return c


def cmd_distill(out: str, *, seed: int, epochs: int, T: float,
                target_path: str = TARGET) -> list[str]:
    return [
        "python", "-m", "scripts.train_shadow",
        "--mode", "distill",
        "--distill-target", target_path,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--distill-temp", str(T),
        "--batch-size", "256",
        "--out", out,
    ]


def cmd_noise(out: str, *, seed: int, scale: float,
              target_path: str = TARGET) -> list[str]:
    return [
        "python", "-m", "scripts.derive_shadow",
        "--kind", "noise",
        "--target-path", target_path,
        "--seed", str(seed),
        "--noise-scale", str(scale),
        "--out", out,
    ]


def cmd_quant(out: str, *, bits: int, target_path: str = TARGET) -> list[str]:
    return [
        "python", "-m", "scripts.derive_shadow",
        "--kind", "quant",
        "--target-path", target_path,
        "--bits", str(bits),
        "--out", out,
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="shadows")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.out_dir}/suspects").mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    idx = 0

    # --- 25 independents: varied seeds / epochs / random subsets --------------
    # Epochs sweep (20, 25, 30, 35, 40) — captures both undertrained and full convergence.
    indep_epochs = [20, 25, 30, 35, 40]
    for s in range(25):
        ep = indep_epochs[s % 5]
        jobs.append({
            "id": idx, "kind": "independent", "label": 0,
            "out": shadow_out(idx),
            "cmd": cmd_train(shadow_out(idx), seed=2000 + s, epochs=ep),
        })
        idx += 1

    # --- 5 evil twins: SAME train indices as target, different seeds ----------
    # Worst-case false-positive scenario: same data, same recipe, different init.
    for s in range(5):
        jobs.append({
            "id": idx, "kind": "evil_twin", "label": 0,
            "out": shadow_out(idx),
            "cmd": cmd_train(shadow_out(idx), seed=7000 + s, epochs=30,
                             train_indices=TARGET_INDICES),
        })
        idx += 1

    # --- 10 fine-tunes: start from target, train more on CIFAR-100 ------------
    ft_specs = [
        (1, 1e-3), (2, 1e-3), (3, 5e-4), (5, 1e-3), (8, 5e-3),
        (12, 1e-3), (15, 5e-4), (20, 1e-3), (1, 1e-4), (3, 1e-3),
    ]
    for s, (ep, lr) in enumerate(ft_specs):
        jobs.append({
            "id": idx, "kind": "fine_tune", "label": 1,
            "out": shadow_out(idx),
            "cmd": cmd_train(shadow_out(idx), seed=4000 + s, epochs=ep,
                             lr=lr, init_from=TARGET),
        })
        idx += 1

    # --- 10 distillations: vary temperature and epoch count -------------------
    distill_specs = [
        (1.0, 15), (2.0, 20), (4.0, 25), (8.0, 30),
        (1.0, 25), (2.0, 30), (4.0, 30), (2.0, 20),
        (4.0, 15), (2.0, 25),
    ]
    for s, (T, ep) in enumerate(distill_specs):
        jobs.append({
            "id": idx, "kind": "distill", "label": 1,
            "out": shadow_out(idx),
            "cmd": cmd_distill(shadow_out(idx), seed=5000 + s, epochs=ep, T=T),
        })
        idx += 1

    # --- 10 partial fine-tunes: freeze backbone, train last layer(s) ----------
    partial_specs = [
        ("fc", 5, 1e-2), ("fc", 10, 1e-2), ("fc", 20, 1e-2),
        ("fc", 5, 1e-3), ("fc", 10, 1e-3),
        ("fc,layer4", 5, 1e-3), ("fc,layer4", 10, 1e-3),
        ("fc,layer4", 15, 1e-3), ("fc,layer4", 5, 5e-3),
        ("fc,layer4", 10, 5e-3),
    ]
    for s, (free, ep, lr) in enumerate(partial_specs):
        jobs.append({
            "id": idx, "kind": "partial_finetune", "label": 1,
            "out": shadow_out(idx),
            "cmd": cmd_train(shadow_out(idx), seed=6000 + s, epochs=ep, lr=lr,
                             init_from=TARGET, freeze_except=free),
        })
        idx += 1

    # --- 5 small-noise derivatives (function-preserving) ----------------------
    noise_scales = [0.005, 0.01, 0.02, 0.03, 0.05]
    for s, scale in enumerate(noise_scales):
        jobs.append({
            "id": idx, "kind": "noise", "label": 1,
            "out": shadow_out(idx),
            "cmd": cmd_noise(shadow_out(idx), seed=3000 + s, scale=scale),
        })
        idx += 1

    # --- 5 quantization derivatives ------------------------------------------
    quant_bits = [4, 5, 6, 7, 8]
    for bits in quant_bits:
        jobs.append({
            "id": idx, "kind": "quant", "label": 1,
            "out": shadow_out(idx),
            "cmd": cmd_quant(shadow_out(idx), bits=bits),
        })
        idx += 1

    # Write outputs
    with open(f"{args.out_dir}/jobs.json", "w") as f:
        json.dump(jobs, f, indent=2)

    with open(f"{args.out_dir}/labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suspect_id", "label", "kind"])
        for j in jobs:
            w.writerow([j["id"], j["label"], j["kind"]])

    n_total = len(jobs)
    n_pos = sum(j["label"] for j in jobs)
    print(f"Wrote {args.out_dir}/jobs.json: {n_total} jobs "
          f"({n_pos} stolen, {n_total - n_pos} not-stolen)")
    print(f"  Composition by kind:")
    from collections import Counter
    for k, c in Counter(j["kind"] for j in jobs).items():
        print(f"    {k:<18} {c}")
    print(f"  edit cluster/shadow_suspects.sub: set `queue {n_total}` to match")


if __name__ == "__main__":
    main()
