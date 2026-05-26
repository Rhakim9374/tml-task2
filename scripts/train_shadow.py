"""Train one ResNet-18 shadow model on CIFAR-100.

Modes:
    train   — supervised cross-entropy on a (subset of) CIFAR-100 train set.
              Use --init-from to start from existing weights (fine-tuning).
    distill — knowledge distillation against a teacher (--distill-target).
              Student matches teacher's softmaxed logits; no ground-truth labels.

Architecture and normalization match task_template.py / submission.py exactly
(CIFAR-style ResNet-18 with 3x3 stride-1 conv1, no maxpool, fc=100).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

from src.data import BiasedRandomCrop  # target's exact bias_x=0.5, bias_y=-0.25 recipe

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def make_model() -> nn.Module:
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, 100)
    return m


def get_transforms(kind: str = "target") -> tuple[transforms.Compose, transforms.Compose]:
    """Train-time augmentation pipeline. `kind` controls how aggressive it is.

    target   -- target's exact recipe (biased crop bias_x=0.5, bias_y=-0.25,
                jitter=0.25 + reflect-pad-4 + hflip). For shadows that should
                look "produced under target's training conditions".
    standard -- plain RandomCrop(32, pad=4) + hflip. Most public CIFAR recipes
                use this; produces shadows that drift from target's recipe.
    strong   -- standard + colorjitter; aggressive augmentation that some
                independents would use.
    none     -- no augmentation. Tests undertrained / overfit models.
    """
    if kind == "target":
        train_aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            BiasedRandomCrop(size=32, pad=4, bias_x=0.5, bias_y=-0.25, jitter=0.25),
        ]
    elif kind == "standard":
        train_aug = [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    elif kind == "strong":
        train_aug = [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    elif kind == "none":
        train_aug = []
    else:
        raise ValueError(f"unknown augmentation kind: {kind}")
    train_tfm = transforms.Compose(train_aug + [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    eval_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    return train_tfm, eval_tfm


def get_train_set(train_tfm, data_root: str, dataset_name: str = "cifar100-train"):
    """Return a CIFAR dataset by short name.

    Used to vary the *distillation transfer set* — distilled stolen models
    in the real world are often trained on a different dataset than the
    target (e.g., OOD transfer). Modelling that here:
        cifar100-train  same data target was trained on (50k images, 100 classes)
        cifar100-test   CIFAR-100 test split (10k images, 100 classes)
        cifar10-train   CIFAR-10 (50k images, 10 classes — labels ignored in distill)
    """
    if dataset_name == "cifar100-train":
        return datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_tfm)
    if dataset_name == "cifar100-test":
        return datasets.CIFAR100(root=data_root, train=False, download=True, transform=train_tfm)
    if dataset_name == "cifar10-train":
        cifar10_root = data_root.replace("cifar100_data", "cifar10_data")
        return datasets.CIFAR10(root=cifar10_root, train=True, download=True, transform=train_tfm)
    raise ValueError(f"unknown distill dataset: {dataset_name}")


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.log_softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean", log_target=True) * (T * T)


def train_one_epoch(model, loader, opt, device, teacher=None, T=4.0,
                    label_smoothing: float = 0.0, kd_weight: float = 1.0):
    """Train one epoch.

    kd_weight controls the loss mix when teacher is set:
        1.0 → pure distillation (KD only)
        0.0 → pure CE (effectively no distillation; only meaningful if teacher is None)
        0<α<1 → mixed: α·KD + (1-α)·CE     ← used for the boundary "mixed_kd" kind
    """
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        y = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
        x = x.to(device, non_blocking=True)
        if y is not None:
            y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        if teacher is not None:
            with torch.no_grad():
                t_logits = teacher(x)
            l_kd = kd_loss(logits, t_logits, T)
            if kd_weight >= 1.0 or y is None:
                loss = l_kd
                with torch.no_grad():
                    correct += (logits.argmax(-1) == t_logits.argmax(-1)).sum().item()
            else:
                l_ce = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
                loss = kd_weight * l_kd + (1.0 - kd_weight) * l_ce
                with torch.no_grad():
                    correct += (logits.argmax(-1) == y).sum().item()
        else:
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            correct += (logits.argmax(-1) == y).sum().item()
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        correct += (model(x).argmax(-1) == y).sum().item()
        n += x.size(0)
    return correct / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["train", "distill", "mixed_kd"], default="train",
                    help="train=CE only; distill=KD only; mixed_kd=alpha·KD + (1-alpha)·CE")
    ap.add_argument("--kd-weight", type=float, default=0.5,
                    help="for mode=mixed_kd, the weight on the KD loss (0=CE-only, 1=KD-only)")
    ap.add_argument("--train-indices-json", default=None,
                    help="JSON list of CIFAR-100 train indices; if absent, sample 40k by --seed")
    ap.add_argument("--save-indices", default=None,
                    help="if set, also write the chosen train indices to this path")
    ap.add_argument("--num-train", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=30)
    # Target's recipe uses batch_size=256, lr=0.1, SGD momentum=0.9, wd=5e-4, cosine.
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--nesterov", action="store_true",
                    help="enable Nesterov momentum (off by default — target spec doesn't specify)")
    ap.add_argument("--init-from", default=None,
                    help="safetensors path to initialize from (fine-tuning)")
    ap.add_argument("--freeze-except", default=None,
                    help="comma-separated submodule prefixes to keep trainable; "
                         "all others get requires_grad=False. E.g. 'fc' for last-layer-only "
                         "fine-tuning, 'fc,layer4' to also unfreeze the final residual block.")
    ap.add_argument("--distill-target", default=None,
                    help="safetensors path to the teacher (used in distill mode)")
    ap.add_argument("--distill-temp", type=float, default=4.0)
    ap.add_argument("--data-root", default="./cifar100_data")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="cross-entropy label smoothing (target uses 0.0)")
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--augmentation", default="target",
                    choices=["target", "standard", "strong", "none"],
                    help="train-time augmentation: target=biased-crop (matches target), "
                         "standard=plain RandomCrop+hflip, strong=+colorjitter, none=none")
    ap.add_argument("--distill-dataset", default="cifar100-train",
                    choices=["cifar100-train", "cifar100-test", "cifar10-train"],
                    help="transfer set for distillation (default cifar100-train)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_tfm, eval_tfm = get_transforms(kind=args.augmentation)
    print(f"[aug] using augmentation kind: {args.augmentation}", flush=True)

    if args.mode == "distill" and args.distill_dataset != "cifar100-train":
        # Distillation transfer set may differ from target's training data.
        train_full = get_train_set(train_tfm, args.data_root, args.distill_dataset)
        print(f"[distill] transfer set: {args.distill_dataset} ({len(train_full)} samples)", flush=True)
        # subset only applies when using cifar100-train (40k-of-50k); otherwise use all
        indices = list(range(len(train_full))) if args.distill_dataset != "cifar100-train" else None
    else:
        train_full = datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=train_tfm)
        indices = None

    test_full = datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=eval_tfm)

    if indices is None:
        if args.train_indices_json and Path(args.train_indices_json).exists():
            with open(args.train_indices_json) as f:
                indices = json.load(f)
        else:
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(len(train_full), size=min(args.num_train, len(train_full)),
                                 replace=False).tolist()

    if args.save_indices:
        Path(args.save_indices).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_indices, "w") as f:
            json.dump([int(i) for i in indices], f)

    train_set = Subset(train_full, indices)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_full, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = make_model().to(args.device)
    if args.init_from:
        sd = load_file(args.init_from)
        model.load_state_dict(sd, strict=True)
        print(f"[init] loaded {args.init_from}", flush=True)

    if args.freeze_except:
        trainable_prefixes = {s.strip() for s in args.freeze_except.split(",") if s.strip()}
        n_train = 0
        n_freeze = 0
        for name, param in model.named_parameters():
            top = name.split(".")[0]
            if top in trainable_prefixes:
                param.requires_grad_(True)
                n_train += param.numel()
            else:
                param.requires_grad_(False)
                n_freeze += param.numel()
        print(f"[freeze] kept trainable: {sorted(trainable_prefixes)} "
              f"({n_train:,} params trainable, {n_freeze:,} frozen)", flush=True)

    teacher = None
    if args.mode in ("distill", "mixed_kd"):
        if not args.distill_target:
            raise ValueError(f"--distill-target required for mode={args.mode}")
        teacher = make_model().to(args.device)
        teacher.load_state_dict(load_file(args.distill_target), strict=True)
        teacher.eval()
        kdw = 1.0 if args.mode == "distill" else args.kd_weight
        print(f"[{args.mode}] teacher from {args.distill_target} (kd_weight={kdw})", flush=True)

    # Only optimize parameters that aren't frozen (matters for --freeze-except).
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.SGD(trainable, lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay, nesterov=args.nesterov)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    t0 = time.time()
    kdw_run = 1.0 if args.mode == "distill" else (args.kd_weight if args.mode == "mixed_kd" else 0.0)
    for ep in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, opt, args.device,
            teacher=teacher, T=args.distill_temp,
            label_smoothing=args.label_smoothing,
            kd_weight=kdw_run,
        )
        sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            te_acc = evaluate(model, test_loader, args.device)
            print(f"[ep {ep:3d}] tr_loss={tr_loss:.3f} tr_acc={tr_acc:.3f} te_acc={te_acc:.3f} "
                  f"lr={sched.get_last_lr()[0]:.4f} t={time.time()-t0:.0f}s", flush=True)

    final_acc = evaluate(model, test_loader, args.device)
    print(f"[done] final test acc={final_acc:.3f} in {time.time()-t0:.0f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), args.out)
    print(f"[done] saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
