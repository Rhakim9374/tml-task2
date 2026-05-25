"""Apply a non-training derivation to a target model.

Kinds:
    noise — add Gaussian noise with std = scale * tensor.std() to each float tensor
    quant — round each float tensor to `bits`-bit precision (then dequantize)

These produce "stolen-by-copy-with-perturbation" shadows that should be highly
similar to the target on every signal (S1/S2 trivially, S3 raw-weight, S4 CKA),
but not bit-identical. Used to populate the derivative half of the shadow set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def add_noise(sd: dict[str, torch.Tensor], scale: float, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in sd.items():
        if v.dtype.is_floating_point and v.numel() > 1:
            std = v.std().item() if v.std().item() > 0 else 1e-6
            noise = torch.randn(v.shape, generator=g, dtype=v.dtype) * (std * scale)
            out[k] = v + noise
        else:
            out[k] = v.clone()
    return out


def quantize(sd: dict[str, torch.Tensor], bits: int) -> dict[str, torch.Tensor]:
    levels = 2 ** bits
    out = {}
    for k, v in sd.items():
        if v.dtype.is_floating_point and v.numel() > 1:
            vmin = v.min().item()
            vmax = v.max().item()
            if vmax - vmin < 1e-12:
                out[k] = v.clone()
                continue
            scale = (vmax - vmin) / (levels - 1)
            q = torch.round((v - vmin) / scale)
            out[k] = q * scale + vmin
        else:
            out[k] = v.clone()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=["noise", "quant"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-scale", type=float, default=0.05,
                    help="relative noise std (fraction of tensor std)")
    ap.add_argument("--bits", type=int, default=6, help="quantization bits")
    args = ap.parse_args()

    sd = load_file(args.target_path)

    if args.kind == "noise":
        new_sd = add_noise(sd, scale=args.noise_scale, seed=args.seed)
    elif args.kind == "quant":
        new_sd = quantize(sd, bits=args.bits)
    else:
        raise ValueError(f"unknown kind: {args.kind}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(new_sd, args.out)
    print(f"[done] {args.kind} (seed={args.seed}) saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
