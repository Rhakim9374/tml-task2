"""Generate the shadow-model job plan (195 shadows, hard-case focus).

  HARD FALSE POSITIVES (label = 0; must learn: NOT stolen)
    50 independent       — recipe variation (aug × opt × lr × wd × ls × ep × subset)
    25 evil_twin         — same train_main_idx as target, varied seeds AND recipes
    15 near_target_indep — random 40k that overlaps target's 40k by ~80%

  HARD TRUE POSITIVES (label = 1; must learn: stolen)
    15 fine_tune         — VERY light (1-3 ep) all the way up to heavy
    25 partial_finetune  — fc-only / fc+layer4 / fc+layer4+layer3, varied LRs
    30 distill           — varied T (1..32), varied transfer data
    15 mixed_kd          — α∈{0.3,0.5,0.7} mixed KD+CE on target

  EASIER CASES (label = 1; function-preserving derivatives)
    10 noise + 10 quant

Total: 50 + 25 + 15 + 15 + 25 + 30 + 15 + 10 + 10 = 195
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TARGET = "target_model/weights.safetensors"
TARGET_INDICES = "target_model/train_main_idx.json"
SUSPECTS_DIR = "shadows/suspects"


def shadow_out(idx: int) -> str:
    return f"{SUSPECTS_DIR}/suspect_{idx:03d}.safetensors"


def cmd_train(
    out: str, *, seed: int, epochs: int, lr: float = 0.1,
    init_from: str | None = None, train_indices: str | None = None,
    freeze_except: str | None = None,
    augmentation: str = "target", optimizer: str = "sgd",
    weight_decay: float = 5e-4, label_smoothing: float = 0.0,
    num_train: int = 40000, batch_size: int = 256,
    mode: str = "train", distill_target: str | None = None,
    distill_temp: float = 4.0, kd_weight: float = 0.5,
) -> list[str]:
    c = [
        "python", "-m", "scripts.train_shadow",
        "--mode", mode,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--weight-decay", str(weight_decay),
        "--label-smoothing", str(label_smoothing),
        "--optimizer", optimizer,
        "--augmentation", augmentation,
        "--num-train", str(num_train),
        "--out", out,
    ]
    if init_from:
        c += ["--init-from", init_from]
    if train_indices:
        c += ["--train-indices-json", train_indices]
    if freeze_except:
        c += ["--freeze-except", freeze_except]
    if distill_target:
        c += ["--distill-target", distill_target,
              "--distill-temp", str(distill_temp),
              "--kd-weight", str(kd_weight)]
    return c


def cmd_distill(
    out: str, *, seed: int, epochs: int, T: float,
    dataset: str = "cifar100-train", target_path: str = TARGET,
    batch_size: int = 256,
) -> list[str]:
    return [
        "python", "-m", "scripts.train_shadow",
        "--mode", "distill",
        "--distill-target", target_path,
        "--distill-dataset", dataset,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--distill-temp", str(T),
        "--batch-size", str(batch_size),
        "--out", out,
    ]


def cmd_noise(out: str, *, seed: int, scale: float, target_path: str = TARGET) -> list[str]:
    return ["python", "-m", "scripts.derive_shadow",
            "--kind", "noise", "--target-path", target_path,
            "--seed", str(seed), "--noise-scale", str(scale), "--out", out]


def cmd_quant(out: str, *, bits: int, target_path: str = TARGET) -> list[str]:
    return ["python", "-m", "scripts.derive_shadow",
            "--kind", "quant", "--target-path", target_path,
            "--bits", str(bits), "--out", out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="shadows")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.out_dir}/suspects").mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    idx = 0

    # =====================================================================
    # HARD FALSE POSITIVES (label=0): must learn to NOT call these stolen
    # =====================================================================

    # --- 50 independents with recipe variation ---
    indep_recipes = [
        # target recipes (still vary lr/wd/ls/ep)
        ("target",   "sgd",  0.1,    5e-4,  0.0,  30, 40000),
        ("target",   "sgd",  0.1,    5e-4,  0.0,  25, 40000),
        ("target",   "sgd",  0.05,   5e-4,  0.0,  30, 40000),
        ("target",   "sgd",  0.1,    1e-4,  0.0,  30, 40000),
        ("target",   "sgd",  0.1,    5e-4,  0.1,  30, 40000),
        ("target",   "sgd",  0.1,    5e-4,  0.0,  40, 40000),
        ("target",   "sgd",  0.1,    5e-4,  0.0,  20, 40000),
        ("target",   "sgd",  0.05,   1e-4,  0.0,  30, 40000),
        # standard
        ("standard", "sgd",  0.1,    5e-4,  0.0,  30, 40000),
        ("standard", "sgd",  0.1,    5e-4,  0.0,  25, 40000),
        ("standard", "sgd",  0.1,    5e-4,  0.1,  30, 40000),
        ("standard", "sgd",  0.05,   5e-4,  0.0,  30, 40000),
        ("standard", "sgd",  0.1,    1e-4,  0.0,  30, 40000),
        ("standard", "sgd",  0.1,    5e-4,  0.0,  40, 40000),
        ("standard", "sgd",  0.2,    5e-4,  0.0,  30, 40000),
        # strong
        ("strong",   "sgd",  0.1,    5e-4,  0.0,  30, 40000),
        ("strong",   "sgd",  0.1,    5e-4,  0.1,  30, 40000),
        ("strong",   "sgd",  0.05,   5e-4,  0.0,  30, 40000),
        ("strong",   "sgd",  0.1,    5e-4,  0.0,  40, 40000),
        ("strong",   "sgd",  0.1,    1e-4,  0.0,  30, 40000),
        # adamw
        ("standard", "adamw", 1e-3,  1e-3,  0.0,  30, 40000),
        ("standard", "adamw", 1e-3,  1e-2,  0.0,  30, 40000),
        ("standard", "adamw", 5e-4,  1e-3,  0.0,  30, 40000),
        ("strong",   "adamw", 1e-3,  1e-3,  0.0,  30, 40000),
        ("strong",   "adamw", 1e-3,  5e-4,  0.1,  30, 40000),
        # no aug
        ("none",     "sgd",  0.1,    5e-4,  0.0,  30, 40000),
        ("none",     "sgd",  0.1,    5e-4,  0.0,  20, 40000),
        # subset sizes
        ("target",   "sgd",  0.1,    5e-4,  0.0,  30, 30000),
        ("target",   "sgd",  0.1,    5e-4,  0.0,  30, 50000),
        ("standard", "sgd",  0.1,    5e-4,  0.0,  30, 25000),
        ("standard", "sgd",  0.1,    5e-4,  0.0,  30, 50000),
        # short / long
        ("target",   "sgd",  0.1,    5e-4,  0.0,  10, 40000),
        ("target",   "sgd",  0.1,    5e-4,  0.0,  15, 40000),
        ("target",   "sgd",  0.1,    5e-4,  0.0,  50, 40000),
        ("standard", "sgd",  0.1,    5e-4,  0.0,  15, 40000),
        ("standard", "sgd",  0.1,    5e-4,  0.0,  50, 40000),
        # more label smoothing
        ("target",   "sgd",  0.1,    5e-4,  0.2,  30, 40000),
        ("standard", "sgd",  0.1,    5e-4,  0.2,  30, 40000),
        ("strong",   "sgd",  0.1,    5e-4,  0.2,  30, 40000),
        # higher LR
        ("target",   "sgd",  0.3,    5e-4,  0.0,  30, 40000),
        ("standard", "sgd",  0.3,    5e-4,  0.0,  30, 40000),
        ("target",   "adamw", 1e-3,  1e-3,  0.0,  30, 40000),
        ("target",   "adamw", 5e-4,  1e-3,  0.0,  30, 40000),
        ("strong",   "sgd",  0.2,    5e-4,  0.0,  30, 40000),
        ("standard", "sgd",  0.05,   1e-4,  0.0,  30, 40000),
        ("target",   "sgd",  0.1,    1e-3,  0.0,  30, 40000),
        ("standard", "sgd",  0.1,    1e-3,  0.0,  30, 40000),
        ("target",   "sgd",  0.1,    5e-4,  0.0,  30, 35000),
        ("standard", "sgd",  0.1,    5e-4,  0.0,  25, 45000),
        ("strong",   "adamw", 5e-4,  1e-3,  0.0,  30, 40000),
    ]
    assert len(indep_recipes) == 50
    for s, (aug, opt, lr, wd, ls, ep, nt) in enumerate(indep_recipes):
        jobs.append({"id": idx, "kind": "independent", "label": 0,
                     "out": shadow_out(idx),
                     "cmd": cmd_train(shadow_out(idx), seed=2000 + s, epochs=ep,
                                      lr=lr, augmentation=aug, optimizer=opt,
                                      weight_decay=wd, label_smoothing=ls, num_train=nt)})
        idx += 1

    # --- 25 EVIL TWINS (same train indices, varied seeds AND recipes) ---
    # HARDEST FP case: same data, similar recipe, different seed.
    evil_twin_recipes = [
        ("target", "sgd",  0.1,    5e-4, 0.0,  30),
        ("target", "sgd",  0.1,    5e-4, 0.0,  25),
        ("target", "sgd",  0.1,    5e-4, 0.0,  40),
        ("target", "sgd",  0.05,   5e-4, 0.0,  30),
        ("target", "sgd",  0.1,    1e-4, 0.0,  30),
        ("target", "sgd",  0.1,    5e-4, 0.1,  30),
        ("target", "sgd",  0.1,    5e-4, 0.0,  20),
        ("target", "sgd",  0.1,    5e-4, 0.0,  15),
        ("target", "sgd",  0.1,    5e-4, 0.0,  50),
        ("standard", "sgd", 0.1,   5e-4, 0.0,  30),
        ("standard", "sgd", 0.1,   5e-4, 0.0,  25),
        ("standard", "sgd", 0.1,   5e-4, 0.0,  40),
        ("standard", "sgd", 0.05,  5e-4, 0.0,  30),
        ("strong", "sgd",  0.1,    5e-4, 0.0,  30),
        ("strong", "sgd",  0.1,    5e-4, 0.0,  25),
        ("target", "sgd",  0.2,    5e-4, 0.0,  30),
        ("target", "sgd",  0.1,    5e-4, 0.2,  30),
        ("target", "adamw", 1e-3,  1e-3, 0.0,  30),
        ("standard", "adamw", 1e-3, 1e-3, 0.0, 30),
        ("target", "sgd",  0.1,    1e-3, 0.0,  30),
        ("standard", "sgd", 0.1,   1e-3, 0.0,  30),
        ("target", "sgd",  0.1,    5e-4, 0.0,  35),
        ("target", "sgd",  0.05,   1e-4, 0.0,  30),
        ("strong", "sgd",  0.05,   5e-4, 0.0,  30),
        ("target", "sgd",  0.3,    5e-4, 0.0,  30),
    ]
    assert len(evil_twin_recipes) == 25
    for s, (aug, opt, lr, wd, ls, ep) in enumerate(evil_twin_recipes):
        jobs.append({"id": idx, "kind": "evil_twin", "label": 0,
                     "out": shadow_out(idx),
                     "cmd": cmd_train(shadow_out(idx), seed=7000 + s, epochs=ep,
                                      lr=lr, augmentation=aug, optimizer=opt,
                                      weight_decay=wd, label_smoothing=ls,
                                      train_indices=TARGET_INDICES)})
        idx += 1

    # --- 15 "near-target" independents (random 40k that overlaps target's 40k) ---
    # Different from evil_twin: chosen RANDOMLY (~80% overlap by chance) rather
    # than explicitly using target's exact indices.
    for s in range(15):
        jobs.append({"id": idx, "kind": "near_target_indep", "label": 0,
                     "out": shadow_out(idx),
                     "cmd": cmd_train(shadow_out(idx), seed=8000 + s, epochs=30,
                                      augmentation="target")})
        idx += 1

    # =====================================================================
    # HARD TRUE POSITIVES (label=1)
    # =====================================================================

    # --- 15 fine-tunes including 5 VERY LIGHT (1-3 epochs) ---
    ft_specs = [
        # Very light — weights barely moved (HARDEST TP)
        (1, 1e-3), (1, 5e-4), (2, 1e-3), (2, 5e-4), (3, 1e-3),
        # Light
        (3, 5e-4), (5, 1e-3), (5, 5e-3), (8, 1e-3), (10, 5e-4),
        # Medium-heavy
        (12, 1e-3), (15, 5e-4), (20, 1e-3), (25, 5e-4), (15, 1e-3),
    ]
    assert len(ft_specs) == 15
    for s, (ep, lr) in enumerate(ft_specs):
        jobs.append({"id": idx, "kind": "fine_tune", "label": 1,
                     "out": shadow_out(idx),
                     "cmd": cmd_train(shadow_out(idx), seed=4000 + s, epochs=ep,
                                      lr=lr, init_from=TARGET, augmentation="target")})
        idx += 1

    # --- 25 PARTIAL fine-tunes (LoRA-style: fc-only, fc+layer4, fc+layer4+layer3) ---
    partial_specs = [
        # fc-only
        ("fc", 5, 1e-2), ("fc", 10, 1e-2), ("fc", 20, 1e-2), ("fc", 30, 1e-2),
        ("fc", 5, 1e-3), ("fc", 10, 1e-3), ("fc", 20, 1e-3), ("fc", 5, 5e-3),
        # fc + layer4
        ("fc,layer4", 5, 1e-3), ("fc,layer4", 10, 1e-3), ("fc,layer4", 15, 1e-3),
        ("fc,layer4", 20, 1e-3), ("fc,layer4", 5, 5e-3), ("fc,layer4", 10, 5e-3),
        ("fc,layer4", 15, 5e-3), ("fc,layer4", 5, 1e-2), ("fc,layer4", 10, 1e-2),
        # fc + layer4 + layer3 (more of the network unfrozen)
        ("fc,layer4,layer3", 5, 1e-3), ("fc,layer4,layer3", 10, 1e-3),
        ("fc,layer4,layer3", 15, 1e-3), ("fc,layer4,layer3", 5, 5e-3),
        ("fc,layer4,layer3", 10, 5e-3), ("fc,layer4,layer3", 15, 5e-3),
        ("fc,layer4,layer3", 5, 1e-2), ("fc,layer4,layer3", 10, 1e-2),
    ]
    assert len(partial_specs) == 25
    for s, (free, ep, lr) in enumerate(partial_specs):
        jobs.append({"id": idx, "kind": "partial_finetune", "label": 1,
                     "out": shadow_out(idx),
                     "cmd": cmd_train(shadow_out(idx), seed=6000 + s, epochs=ep,
                                      lr=lr, init_from=TARGET, freeze_except=free,
                                      augmentation="target")})
        idx += 1

    # --- 30 distillations: T × dataset × epochs ---
    distill_specs = [
        # CIFAR-100 train, varied T (10)
        (1.0,  15, "cifar100-train"),  (1.0,  25, "cifar100-train"),
        (2.0,  15, "cifar100-train"),  (2.0,  25, "cifar100-train"),
        (4.0,  15, "cifar100-train"),  (4.0,  25, "cifar100-train"),
        (8.0,  15, "cifar100-train"),  (8.0,  25, "cifar100-train"),
        (16.0, 20, "cifar100-train"),  (32.0, 20, "cifar100-train"),
        # CIFAR-10 OOD transfer (10)
        (2.0,  15, "cifar10-train"),   (2.0,  25, "cifar10-train"),
        (4.0,  15, "cifar10-train"),   (4.0,  25, "cifar10-train"),
        (8.0,  15, "cifar10-train"),   (8.0,  20, "cifar10-train"),
        (8.0,  25, "cifar10-train"),   (16.0, 20, "cifar10-train"),
        (16.0, 25, "cifar10-train"),   (32.0, 20, "cifar10-train"),
        # CIFAR-100 test transfer (5) — small transfer set, more overfit-prone student
        (2.0,  15, "cifar100-test"),   (4.0,  15, "cifar100-test"),
        (8.0,  20, "cifar100-test"),   (16.0, 20, "cifar100-test"),
        (4.0,  25, "cifar100-test"),
        # More variations (5)
        (1.0,  10, "cifar100-train"),  (4.0,  10, "cifar100-train"),
        (2.0,  30, "cifar100-train"),  (8.0,  30, "cifar100-train"),
        (4.0,  30, "cifar100-train"),
    ]
    assert len(distill_specs) == 30
    for s, (T, ep, ds) in enumerate(distill_specs):
        jobs.append({"id": idx, "kind": "distill", "label": 1,
                     "out": shadow_out(idx),
                     "cmd": cmd_distill(shadow_out(idx), seed=5000 + s,
                                        epochs=ep, T=T, dataset=ds)})
        idx += 1

    # --- 15 MIXED KD+CE: a fresh student trained with α·KD + (1-α)·CE -----
    # Boundary case between "pure distillation" (KD only, label=1) and
    # "pure independent" (CE only, label=0). Per spec these still count as
    # stolen (anything derived from target's outputs is stealing).
    mixed_specs = [
        # (kd_weight, T, epochs)
        (0.3, 2.0, 20), (0.3, 4.0, 20), (0.3, 8.0, 20),
        (0.5, 2.0, 20), (0.5, 4.0, 20), (0.5, 8.0, 20),
        (0.5, 4.0, 25), (0.5, 4.0, 30),
        (0.7, 2.0, 20), (0.7, 4.0, 20), (0.7, 8.0, 20),
        (0.7, 4.0, 25), (0.5, 1.0, 25), (0.3, 4.0, 25), (0.5, 16.0, 25),
    ]
    assert len(mixed_specs) == 15
    for s, (kdw, T, ep) in enumerate(mixed_specs):
        jobs.append({"id": idx, "kind": "mixed_kd", "label": 1,
                     "out": shadow_out(idx),
                     "cmd": cmd_train(shadow_out(idx), seed=9000 + s, epochs=ep,
                                      lr=0.1, augmentation="target",
                                      mode="mixed_kd", distill_target=TARGET,
                                      distill_temp=T, kd_weight=kdw)})
        idx += 1

    # --- 10 noise + 10 quant function-preserving ---
    noise_scales = [0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    for s, scale in enumerate(noise_scales):
        jobs.append({"id": idx, "kind": "noise", "label": 1,
                     "out": shadow_out(idx),
                     "cmd": cmd_noise(shadow_out(idx), seed=3000 + s, scale=scale)})
        idx += 1
    quant_bits = [3, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    for s, bits in enumerate(quant_bits):
        jobs.append({"id": idx, "kind": "quant", "label": 1,
                     "out": shadow_out(idx),
                     "cmd": cmd_quant(shadow_out(idx), bits=bits)})
        idx += 1

    # Write outputs
    with open(f"{args.out_dir}/jobs.json", "w") as f:
        json.dump(jobs, f, indent=2)
    with open(f"{args.out_dir}/labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suspect_id", "label", "kind"])
        for j in jobs:
            w.writerow([j["id"], j["label"], j["kind"]])

    from collections import Counter
    n_total = len(jobs)
    n_pos = sum(j["label"] for j in jobs)
    print(f"Wrote {args.out_dir}/jobs.json: {n_total} jobs "
          f"({n_pos} stolen, {n_total - n_pos} not-stolen)")
    print("  Composition by kind (focus on hard cases):")
    for k, c in Counter(j["kind"] for j in jobs).items():
        lbl = "stolen" if any(j["label"] == 1 and j["kind"] == k for j in jobs) else "not"
        print(f"    {k:<20} {c:>3}  ({lbl})")
    print(f"\n  Edit cluster/shadow_suspects.sub: set `queue {n_total}`.")
    print(f"  Edit cluster/shadow_extract.sub: set `arguments ... {n_total}`"
          f" and a divisor of {n_total} for `queue`.")


if __name__ == "__main__":
    main()
