"""S3 — weight-space distance between target and suspect.

We compute three flavors of weight similarity, each catching a different theft
pattern:

    s3_raw_l2 / s3_raw_cos
        Raw ||W_s - W_t|| and cosine over the concatenated flat parameter
        vector. Catches direct copies and lightly fine-tuned models.

    s3_perm_l2 / s3_perm_cos
        Same metrics after a per-layer activation-matched channel permutation
        of the suspect's weights. The permutation is the linear assignment that
        maximizes target/suspect activation cross-correlation at each
        permutable point in the ResNet-18 architecture. Catches
        function-preserving channel permutations (a common obfuscation).

    s3_svd_dist
        L2 distance between sorted singular value spectra per layer, summed.
        Permutation- and orthogonal-rotation-invariant signature — catches
        broader function-preserving transforms beyond plain permutation.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.models.resnet import BasicBlock


# ----- raw / svd features (no alignment needed) ---------------------------


def flat_params(model: nn.Module) -> torch.Tensor:
    """Concatenated CPU float32 vector of all parameters (in module order)."""
    return torch.cat([p.detach().cpu().float().flatten() for p in model.parameters()])


def raw_weight_features(target: nn.Module, suspect: nn.Module) -> dict[str, float]:
    wt = flat_params(target)
    ws = flat_params(suspect)
    l2 = (wt - ws).norm().item()
    cos = F.cosine_similarity(wt.unsqueeze(0), ws.unsqueeze(0)).item()
    return {
        "s3_raw_l2": float(-l2),  # higher = more stolen
        "s3_raw_cos": float(cos),
    }


def _layer_weight_matrices(model: nn.Module) -> dict[str, torch.Tensor]:
    """Per-conv/fc weights reshaped to 2-D for SVD."""
    out: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.detach().cpu().float()
            out[name] = w.reshape(w.shape[0], -1)
    return out


def svd_spectrum_distance(target: nn.Module, suspect: nn.Module) -> dict[str, float]:
    """L2 between sorted-descending singular value spectra, summed over layers."""
    tw = _layer_weight_matrices(target)
    sw = _layer_weight_matrices(suspect)
    assert tw.keys() == sw.keys(), "architectures differ"
    total = 0.0
    for k in tw:
        s_t = torch.linalg.svdvals(tw[k]).sort(descending=True).values
        s_s = torch.linalg.svdvals(sw[k]).sort(descending=True).values
        total += float((s_t - s_s).norm().item())
    return {"s3_svd_dist": float(-total)}  # higher = more stolen


# ----- activation-matched permutation alignment ---------------------------
#
# ResNet-18 CIFAR-style permutation groups (using CIFAR-style conv1):
#   P0  conv1 + bn1 output, == residual stream inside layer1 (64 channels)
#   P1  residual stream inside layer2 (128)
#   P2  residual stream inside layer3 (256)
#   P3  residual stream inside layer4 (512)
#   M_layer.block  intermediate channels between conv1 and conv2 of each block
#
# Within each `nn.Sequential` (layer1..4), all block outputs share one
# permutation because of the residual `+` connection. The intermediate
# channels are independent per block.


def _hook_save(buf: dict, key: str):
    def hook(_module, _inp, out):
        # out can be a tensor or a tuple; for conv it's a tensor
        buf[key] = out.detach()
    return hook


@torch.no_grad()
def _capture_activations(
    model: nn.Module,
    probe: torch.Tensor,
    device: str,
) -> dict[str, torch.Tensor]:
    """Run probe through model, capturing pre-residual conv2 outputs and
    layer outputs at every block. Returns dict of {key: NCHW tensor on CPU}."""
    buf: dict[str, torch.Tensor] = {}
    handles = []

    # capture layer (Sequential) outputs — these define the residual stream
    for stage_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, stage_name)
        handles.append(layer.register_forward_hook(_hook_save(buf, stage_name)))
        # capture intermediate channels: bn1 inside each BasicBlock (post conv1+bn1, pre relu+conv2)
        for bi, block in enumerate(layer):
            assert isinstance(block, BasicBlock)
            handles.append(block.bn1.register_forward_hook(_hook_save(buf, f"{stage_name}.{bi}.bn1")))

    # capture stem (conv1 + bn1) — feeds layer1
    handles.append(model.bn1.register_forward_hook(_hook_save(buf, "stem_bn1")))

    model.eval()
    model.to(device)
    probe = probe.to(device, non_blocking=True)
    model(probe)
    for h in handles:
        h.remove()

    # move to CPU float32 for downstream alignment
    return {k: v.float().cpu() for k, v in buf.items()}


def _channel_corr(t_act: torch.Tensor, s_act: torch.Tensor) -> np.ndarray:
    """Cross-correlation between channels of two NCHW activation maps.
    Returns C_t x C_s matrix of Pearson correlations averaged over (N, H, W)."""
    # flatten spatial: [N, C, H*W] then concat over N → [C, N*H*W]
    t = t_act.permute(1, 0, 2, 3).reshape(t_act.shape[1], -1)
    s = s_act.permute(1, 0, 2, 3).reshape(s_act.shape[1], -1)
    t = t - t.mean(dim=1, keepdim=True)
    s = s - s.mean(dim=1, keepdim=True)
    t = t / (t.norm(dim=1, keepdim=True) + 1e-8)
    s = s / (s.norm(dim=1, keepdim=True) + 1e-8)
    return (t @ s.T).numpy()


def _best_perm(corr: np.ndarray) -> np.ndarray:
    """Hungarian: return permutation P (length C) such that suspect_channel = P[target_channel] maximizes total corr.

    Returns array `perm` where perm[i] gives the suspect channel index assigned
    to target channel i. Equivalently, suspect_weights_aligned[i] = suspect_weights[perm[i]].
    """
    if not np.all(np.isfinite(corr)):
        # Dead/quantized channels can give 0/0 correlations → NaN. Treat those
        # cells as "no information" (corr=0); the Hungarian then assigns them
        # to whatever leftover slot, which is fine since their downstream
        # contribution is also ~0.
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    row_ind, col_ind = linear_sum_assignment(-corr)
    # row_ind is just np.arange when corr is square; col_ind is the chosen suspect channel per target row
    assert np.all(row_ind == np.arange(corr.shape[0]))
    return col_ind.astype(np.int64)


def _apply_perm_out(conv: nn.Conv2d | nn.Linear, bn: nn.BatchNorm2d | None, perm: torch.Tensor):
    """Permute the OUTPUT channels of a conv (or linear) + its following BN."""
    conv.weight.data = conv.weight.data[perm]
    if conv.bias is not None:
        conv.bias.data = conv.bias.data[perm]
    if bn is not None:
        bn.weight.data = bn.weight.data[perm]
        bn.bias.data = bn.bias.data[perm]
        bn.running_mean.data = bn.running_mean.data[perm]
        bn.running_var.data = bn.running_var.data[perm]


def _apply_perm_in(conv: nn.Conv2d | nn.Linear, perm: torch.Tensor):
    """Permute the INPUT channels of a conv (or linear)."""
    conv.weight.data = conv.weight.data[:, perm]


def align_suspect_to_target(
    target: nn.Module,
    suspect: nn.Module,
    probe: torch.Tensor,
    device: str,
) -> nn.Module:
    """Permute suspect's channels in place to maximize activation match with target.
    Returns the same suspect object after mutation.

    Channel-group structure of CIFAR-style ResNet-18:
      group 0 (64 ch):  stem (conv1+bn1) and ALL block outputs of layer1
      group 1 (128 ch): layer2.0 downsample output and ALL block outputs of layer2
      group 2 (256 ch): layer3.0 downsample output and ALL block outputs of layer3
      group 3 (512 ch): layer4.0 downsample output and ALL block outputs of layer4

    Within a group the residual `+` connection forces the same permutation on
    every contributing tensor. The stem and layer1 share group 0 because
    layer1.0 has no downsample (input dim == output dim == 64).
    The intermediate channels between each block's conv1 and conv2 are
    independent (one permutation per block, 8 total).
    """
    t_acts = _capture_activations(target, probe, device)
    s_acts = _capture_activations(suspect, probe, device)

    # Compute the per-group residual-stream permutation from the output of each
    # group (layer1..4). This implicitly aligns the stem too via group 0.
    group_perms = {
        name: torch.from_numpy(_best_perm(_channel_corr(t_acts[name], s_acts[name])))
        for name in ["layer1", "layer2", "layer3", "layer4"]
    }

    # 1) STEM uses group 0's permutation (same residual stream as layer1).
    _apply_perm_out(suspect.conv1, suspect.bn1, group_perms["layer1"])

    # 2) layer1..4 — apply group permutation to every block's output + intermediates
    residual_perm_prev = group_perms["layer1"]  # input to layer1.0.conv1 is the stem output
    for stage_name in ["layer1", "layer2", "layer3", "layer4"]:
        perm_stage = group_perms[stage_name]

        stage = getattr(suspect, stage_name)
        # For each block in this stage
        for bi, block in enumerate(stage):
            # intermediate channels between conv1 and conv2 → independent permutation
            perm_mid = torch.from_numpy(_best_perm(_channel_corr(
                t_acts[f"{stage_name}.{bi}.bn1"], s_acts[f"{stage_name}.{bi}.bn1"]
            )))

            # block.conv1: in = residual_perm_prev, out = perm_mid
            _apply_perm_in(block.conv1, residual_perm_prev)
            _apply_perm_out(block.conv1, block.bn1, perm_mid)

            # block.conv2: in = perm_mid, out = perm_stage
            _apply_perm_in(block.conv2, perm_mid)
            _apply_perm_out(block.conv2, block.bn2, perm_stage)

            # downsample if present: in = residual_perm_prev, out = perm_stage
            if block.downsample is not None:
                ds_conv = block.downsample[0]
                ds_bn = block.downsample[1]
                _apply_perm_in(ds_conv, residual_perm_prev)
                _apply_perm_out(ds_conv, ds_bn, perm_stage)

            # the next block's residual stream input is the same as this block's residual stream output
            # (since each block adds residual to identity, after the first block the residual stream
            # already has perm_stage; the NEXT block's conv1 input must be perm_stage too).
            # Update for the next iteration:
            residual_perm_prev = perm_stage

    # 3) fc: input channels permuted by final P3 (layer4 output)
    _apply_perm_in(suspect.fc, residual_perm_prev)

    return suspect


def aligned_weight_features(target: nn.Module, suspect_aligned: nn.Module) -> dict[str, float]:
    """Compute L2 and cosine on flat params AFTER alignment. Same orientation as raw."""
    wt = flat_params(target)
    ws = flat_params(suspect_aligned)
    l2 = (wt - ws).norm().item()
    cos = F.cosine_similarity(wt.unsqueeze(0), ws.unsqueeze(0)).item()
    return {
        "s3_perm_l2": float(-l2),
        "s3_perm_cos": float(cos),
    }
