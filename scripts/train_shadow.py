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


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    # Matches the target's recipe from the assignment PDF: hflip + biased crop
    # (bias_x=0.5, bias_y=-0.25, jitter=0.25, reflect-pad=4) + standard normalize.
    train_tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        BiasedRandomCrop(size=32, pad=4, bias_x=0.5, bias_y=-0.25, jitter=0.25),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    eval_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    return train_tfm, eval_tfm


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.log_softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean", log_target=True) * (T * T)


def train_one_epoch(model, loader, opt, device, teacher=None, T=4.0):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        if teacher is not None:
            with torch.no_grad():
                t_logits = teacher(x)
            loss = kd_loss(logits, t_logits, T)
        else:
            loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
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
    ap.add_argument("--mode", choices=["train", "distill"], default="train")
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
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_tfm, eval_tfm = get_transforms()
    train_full = datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=train_tfm)
    test_full = datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=eval_tfm)

    if args.train_indices_json and Path(args.train_indices_json).exists():
        with open(args.train_indices_json) as f:
            indices = json.load(f)
    else:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(train_full), size=args.num_train, replace=False).tolist()

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
    if args.mode == "distill":
        if not args.distill_target:
            raise ValueError("--distill-target required for distill mode")
        teacher = make_model().to(args.device)
        teacher.load_state_dict(load_file(args.distill_target), strict=True)
        teacher.eval()
        print(f"[distill] teacher loaded from {args.distill_target}", flush=True)

    # Only optimize parameters that aren't frozen (matters for --freeze-except).
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    t0 = time.time()
    for ep in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, args.device,
                                          teacher=teacher, T=args.distill_temp)
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
