"""DINOv3 ViT-S/16 student for the learned semantic master BEV.

Input per sample: both cameras' RGB-D at 240x424::

    color  [B, 2, 3, 240, 424]   ImageNet-normalised RGB
    depth  [B, 2, 1, 240, 424]   metres / 8, invalid = 0

The shared DINOv3 ViT-S/16 encoder (with a zero-initialised 4th depth
channel) produces patch tokens from layers 3/6/9/12; a small FPN brings
them to 60x106x128.  A fixed bilinear LUT projects 60x30 BEV cell centres
at z = 0.0/0.3/0.8/1.5 m, per camera, into the image features (Gather +
Mul + Add, no ``grid_sample`` in the deployed path).  The lifted 8 groups
(near/far x 4 heights) plus a per-group depth-validity channel go through a
1x1 reduction, a small U-Net, and three heads:

    semantic [B, 4, 120, 60]   floor/wall/obstacle/hazard logits
    ego      [B, 1, 120, 60]
    coverage [B, 1, 120, 60]

The LUT is built from the same rig YAML / intrinsics as the geometric BEV
and the teacher labels.  It is registered as model buffers so it exports as
constants with the graph.

The module is deliberately independent of SAM (teacher-side only).  A tiny
random DINOv3 backbone is available for CPU tests so CI does not need the
gated weights.
"""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path

# Keep the Thor OpenBLAS pool asleep; this module imports numpy for LUTs.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bev_student_labels import GridSpec, specs_from_config

HEIGHTS = (0.0, 0.3, 0.8, 1.5)
FEAT_HW = (60, 106)
DEPTH_HW = (240, 424)
BEV_HW = (60, 30)
ROLES = ("near", "far")
NUM_CLASSES = 4
DEFAULT_WEIGHTS = (Path(__file__).resolve().parents[1]
                   / "weights" / "dinov3-vits16")


# ---------------------------------------------------------------------------
# Fixed lift LUT
# ---------------------------------------------------------------------------


def _bilinear(nu, nv, width, height):
    """Normalised image coords -> 4 flat indices + weights + valid mask."""
    xf = nu * width - 0.5
    yf = nv * height - 0.5
    x0 = np.floor(xf).astype(np.int64)
    y0 = np.floor(yf).astype(np.int64)
    wx = (xf - x0).astype(np.float32)
    wy = (yf - y0).astype(np.float32)
    inside = ((nu >= 0.0) & (nu <= 1.0)
              & (nv >= 0.0) & (nv <= 1.0)
              & (x0 >= 0) & (x0 < width - 1)
              & (y0 >= 0) & (y0 < height - 1))
    x0c = np.clip(x0, 0, max(width - 2, 0))
    y0c = np.clip(y0, 0, max(height - 2, 0))
    idx = np.stack([
        y0c * width + x0c,
        y0c * width + x0c + 1,
        (y0c + 1) * width + x0c,
        (y0c + 1) * width + x0c + 1,
    ], axis=0)
    w = np.stack([
        (1.0 - wy) * (1.0 - wx),
        (1.0 - wy) * wx,
        wy * (1.0 - wx),
        wy * wx,
    ], axis=0).astype(np.float32)
    idx[:, ~inside] = 0
    w[:, ~inside] = 0.0
    return idx, w, inside


def build_lift_lut(rig_cfg=None, config_dir=None, heights=HEIGHTS,
                   feat_hw=FEAT_HW, depth_hw=DEPTH_HW, bev_hw=BEV_HW):
    """Project the BEV feature grid into both cameras at each height.

    Returns ``{"tensors": {...}, "meta": {...}}``.  Group order is
    ``near z0..z3, far z0..z3``.  The color LUT samples the 60x106 feature
    map; the depth LUT samples the 240x424 depth input (separate intrinsics
    and a separate projection).
    """
    specs, cfg, grid, _label_cfg = specs_from_config(
        rig_cfg=rig_cfg, config_dir=config_dir, roles=ROLES)
    bev_r, bev_c = int(bev_hw[0]), int(bev_hw[1])
    range_m, width_m = float(grid.range_m), float(grid.width_m)
    x = range_m - (np.arange(bev_r) + 0.5) * (range_m / bev_r)
    y = (np.arange(bev_c) + 0.5) * (width_m / bev_c) - width_m / 2.0
    xx, yy = np.meshgrid(x, y, indexing="ij")
    xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)  # (1800,2)

    feat_idx, feat_w, feat_valid = [], [], []
    depth_idx, depth_w = [], []
    fh, fw = int(feat_hw[0]), int(feat_hw[1])
    dh, dw = int(depth_hw[0]), int(depth_hw[1])

    for role in ROLES:
        spec = specs[role]
        # body -> color, and body -> depth optical
        R_bc = spec.R_dc.astype(np.float64) @ spec.R_body.T.astype(np.float64)
        t_bc = spec.t_dc.astype(np.float64) - R_bc @ spec.t_body.astype(
            np.float64)
        R_bd = spec.R_body.T.astype(np.float64)
        for z in heights:
            p = np.concatenate(
                [xy, np.full((xy.shape[0], 1), float(z))], axis=1)
            # color projection
            pc = p @ R_bc.T + t_bc
            zc = pc[:, 2]
            u = spec.fx_c * pc[:, 0] / np.maximum(zc, 1e-9) + spec.cx_c
            v = spec.fy_c * pc[:, 1] / np.maximum(zc, 1e-9) + spec.cy_c
            nu = u / spec.width_c
            nv = v / spec.height_c
            fi, fwgt, fin = _bilinear(nu, nv, fw, fh)
            # depth projection (the depth frame is not the color frame)
            pd = (p - spec.t_body) @ spec.R_body
            zd = pd[:, 2]
            ud = spec.fx_d * pd[:, 0] / np.maximum(zd, 1e-9) + spec.cx_d
            vd = spec.fy_d * pd[:, 1] / np.maximum(zd, 1e-9) + spec.cy_d
            ndu = ud / spec.width_d
            ndv = vd / spec.height_d
            di, dwgt, din = _bilinear(ndu, ndv, dw, dh)
            ok = fin & din & (zc > 1e-6) & (zd > 1e-6)
            di[:, ~ok] = 0
            dwgt[:, ~ok] = 0.0
            fi[:, ~ok] = 0
            fwgt[:, ~ok] = 0.0
            feat_idx.append(fi)
            feat_w.append(fwgt)
            feat_valid.append(ok)
            depth_idx.append(di)
            depth_w.append(dwgt)

    tensors = {
        "feat_idx": torch.from_numpy(np.stack(feat_idx)).long(),
        "feat_w": torch.from_numpy(np.stack(feat_w)).float(),
        "feat_valid": torch.from_numpy(np.stack(feat_valid)).bool(),
        "depth_idx": torch.from_numpy(np.stack(depth_idx)).long(),
        "depth_w": torch.from_numpy(np.stack(depth_w)).float(),
    }
    meta = {
        "roles": list(ROLES),
        "heights": [float(z) for z in heights],
        "feat_hw": [fh, fw],
        "depth_hw": [dh, dw],
        "bev_hw": [bev_r, bev_c],
        "range_m": range_m,
        "width_m": width_m,
        "groups": [(r, float(z)) for r in ROLES for z in heights],
        "grid_rows": grid.rows,
        "grid_cols": grid.cols,
        "cell_m": grid.cell_m,
    }
    return {"tensors": tensors, "meta": meta}


def dummy_lift_lut(groups=8, n_cells=BEV_HW[0] * BEV_HW[1]):
    """Shape-only LUT for CPU tests: every cell samples feature (0,0)."""
    idx = torch.zeros(groups, 4, n_cells, dtype=torch.long)
    w = torch.zeros(groups, 4, n_cells, dtype=torch.float32)
    w[:, 0] = 1.0
    valid = torch.ones(groups, n_cells, dtype=torch.bool)
    return {
        "tensors": {
            "feat_idx": idx,
            "feat_w": w,
            "feat_valid": valid,
            "depth_idx": idx.clone(),
            "depth_w": w.clone(),
        },
        "meta": {
            "roles": list(ROLES), "heights": list(HEIGHTS),
            "feat_hw": list(FEAT_HW), "depth_hw": list(DEPTH_HW),
            "bev_hw": list(BEV_HW), "range_m": 6.0, "width_m": 3.0,
            "groups": [(r, float(z)) for r in ROLES for z in HEIGHTS],
            "grid_rows": 120, "grid_cols": 60, "cell_m": 0.05,
        },
    }


# ---------------------------------------------------------------------------
# Encoder / FPN
# ---------------------------------------------------------------------------


def _adapt_patch_embedding(backbone):
    """Widen the DINOv3 patch conv from 3 to 4 channels (depth zero)."""
    holder = backbone.embeddings.patch_embeddings
    # HuggingFace's DINOv3 embedding stores the Conv2d directly; older
    # layouts wrapped it as `.projection`.
    proj = holder if isinstance(holder, nn.Conv2d) else holder.projection
    new = nn.Conv2d(4, proj.out_channels, kernel_size=proj.kernel_size,
                    stride=proj.stride, padding=proj.padding,
                    bias=proj.bias is not None)
    with torch.no_grad():
        new.weight[:, :3].copy_(proj.weight)
        new.weight[:, 3:].zero_()
        if proj.bias is not None:
            new.bias.copy_(proj.bias)
    if isinstance(holder, nn.Conv2d):
        backbone.embeddings.patch_embeddings = new
    else:
        holder.projection = new
    try:
        backbone.config.num_channels = 4
    except Exception:
        pass
    return backbone


def _build_backbone(pretrained_dir=None, tiny=False):
    if tiny:
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        cfg = DINOv3ViTConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=256, patch_size=16, image_size=224,
            num_register_tokens=4)
        backbone = DINOv3ViTModel(cfg)
    else:
        from transformers import AutoModel
        path = Path(pretrained_dir or DEFAULT_WEIGHTS)
        backbone = AutoModel.from_pretrained(str(path))
    return _adapt_patch_embedding(backbone)


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(8, cout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BevStudent(nn.Module):
    """DINOv3 ViT-S/16 + FPN + fixed-LUT BEV lift + U-Net heads."""

    def __init__(self, pretrained_dir=None, tiny=False, backbone=None,
                 backbone_layers=None, fpn_dim=128, lut=None,
                 feat_hw=FEAT_HW, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_hw = tuple(int(v) for v in feat_hw)
        self.backbone = (backbone if backbone is not None
                         else _build_backbone(pretrained_dir, tiny))
        self.backbone_layers = tuple(
            backbone_layers
            or ((1, 2, 2, 2) if tiny else (3, 6, 9, 12)))
        hidden = int(self.backbone.config.hidden_size)
        self.patch_size = int(self.backbone.config.patch_size)
        self.num_prefix = 1 + int(self.backbone.config.num_register_tokens)
        self.proj = nn.ModuleList(
            [nn.Conv2d(hidden, fpn_dim, 1)
             for _ in self.backbone_layers])
        self.fpn_conv = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.fpn_norm = nn.GroupNorm(8, fpn_dim)
        self.cam_emb = nn.Embedding(2, 16)
        self.cam_proj = nn.Linear(16, fpn_dim)
        nn.init.normal_(self.cam_emb.weight, std=0.02)
        nn.init.zeros_(self.cam_proj.bias)

        lut = lut or build_lift_lut()
        for name, tensor in lut["tensors"].items():
            self.register_buffer(name, tensor)
        self.lut_meta = lut["meta"]
        self.heights = tuple(self.lut_meta["heights"])
        self.bev_hw = tuple(self.lut_meta["bev_hw"])
        self.n_groups = len(self.lut_meta["groups"])

        groups = self.n_groups
        self.reduce = nn.Conv2d(groups * (fpn_dim + 1), fpn_dim, 1)
        self.reduce_norm = nn.GroupNorm(8, fpn_dim)
        self.reduce_act = nn.GELU()
        self.dec1 = _ConvBlock(fpn_dim, 128)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                              align_corners=False)
        self.dec2 = _ConvBlock(128, 128)
        self.dec3 = _ConvBlock(128, 128)
        self.sem_head = nn.Conv2d(128, self.num_classes, 1)
        self.ego_head = nn.Conv2d(128, 1, 1)
        self.cov_head = nn.Conv2d(128, 1, 1)

    # -- encoder -----------------------------------------------------------

    def encode(self, x):
        """[2B,4,H,W] -> [2B,128,60,106]."""
        out = self.backbone(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states
        bh, bw = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        feats = []
        for i, layer in enumerate(self.backbone_layers):
            t = hs[layer][:, self.num_prefix:]
            t = t.reshape(t.shape[0], bh, bw, -1).permute(0, 3, 1, 2)
            f = self.proj[i](t)
            f = F.interpolate(f, size=self.feat_hw, mode="bilinear",
                              align_corners=False)
            feats.append(f)
        f = torch.stack(feats, dim=0).sum(dim=0)
        f = self.fpn_conv(f)
        return F.gelu(self.fpn_norm(f))

    @staticmethod
    def _gather(feats, idx, weights):
        """feats [B,C,N] + fixed 4-neighbour LUT -> [B,C,1800]."""
        sampled = feats[:, :, idx] * weights[None, None]
        return sampled.sum(dim=2)

    # -- forward -----------------------------------------------------------

    def forward(self, color, depth):
        if color.ndim != 5 or color.shape[1] != 2:
            raise ValueError("color must be [B,2,3,H,W]")
        if depth.ndim != 5 or depth.shape[1] != 2:
            raise ValueError("depth must be [B,2,1,H,W]")
        b, _, _, h, w = color.shape
        x = torch.cat([color, depth], dim=2).reshape(b * 2, 4, h, w)
        feats = self.encode(x).reshape(b, 2, -1, *self.feat_hw)
        cam = self.cam_proj(self.cam_emb(
            torch.arange(2, device=color.device)))  # [2,128]
        feats = feats + cam[None, :, :, None, None]
        dflat = depth.reshape(b, 2, 1, h * w)
        groups = []
        for g in range(self.n_groups):
            role = self.lut_meta["groups"][g][0]
            cam_i = 0 if role == "near" else 1
            f = feats[:, cam_i].reshape(b, -1, feats.shape[-2] * feats.shape[-1])
            sampled = self._gather(
                f, self.feat_idx[g], self.feat_w[g])
            sampled = sampled.reshape(b, -1, *self.bev_hw)
            ds = dflat[:, cam_i]
            d_sampled = self._gather(
                ds, self.depth_idx[g], self.depth_w[g])
            d_sampled = d_sampled.reshape(b, 1, *self.bev_hw)
            valid = (self.feat_valid[g].reshape(1, 1, *self.bev_hw)
                     & (d_sampled > 1e-6)).float()
            groups.append(torch.cat([sampled, valid], dim=1))
        z = torch.cat(groups, dim=1)
        z = self.reduce_act(self.reduce_norm(self.reduce(z)))
        z = self.dec1(z)
        z = self.up(z)
        z = self.dec2(z)
        z = self.dec3(z)
        return {
            "semantic": self.sem_head(z),
            "ego": self.ego_head(z),
            "coverage": self.cov_head(z),
        }

    # -- convenience -------------------------------------------------------

    def set_encoder_frozen(self, frozen=True):
        for p in self.backbone.parameters():
            p.requires_grad = not frozen

    def encoder_parameters(self):
        return list(self.backbone.parameters())

    def head_parameters(self):
        return [p for n, p in self.named_parameters()
                if not n.startswith("backbone.")]

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
