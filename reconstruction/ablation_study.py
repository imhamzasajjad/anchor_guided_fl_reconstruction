#!/usr/bin/env python3
"""
Ablation study runner for anchor-guided federated reconstruction on CIFAR-10.

This script reuses the reconstruction pipeline from recon_multianchor.py and
adds four ablation methods over the same checkpoints and evaluation samples:

Method A: Multi-Anchor (5 anchors, feature-bank matching enabled)
Method B: Single Anchor (1 anchor, feature-bank matching enabled)
Method C: Random Initialization (no anchor, random init + same optimization budget)
Method D: Anchor Without Feature Bank (5 anchors, feature-bank matching disabled)

Outputs:
- ablation_results.csv
- ablation_summary.csv
- ablation_significance.csv
- table_ablation.tex
- table_significance.tex
"""

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind
from skimage.metrics import structural_similarity as compute_ssim
from torchvision.models import resnet34


# ----------------------------- Defaults -------------------------------------

DEFAULT_EXPERIMENTS = {
    "E1": {"heterogeneity": "IID", "ckpt": "gm_best_100_IID.pt"},
    "E2": {"heterogeneity": "Mild", "ckpt": "gm_best_100_Non_IID(Mild).pt"},
    "E3": {"heterogeneity": "Strong", "ckpt": "gm_best_100_Non_IID(Moderate).pt"},
    "E4": {"heterogeneity": "Extreme", "ckpt": "gm_best_100_Non_IID(Extreme).pt"},
}

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

ANCHOR_POSITIONS_5 = ["top_left", "top_right", "center", "bottom_left", "bottom_right"]


@dataclass(frozen=True)
class MethodConfig:
    name: str
    anchor_positions: Sequence[str]
    use_anchor_init: bool
    use_feature_bank_guidance: bool


METHODS: Sequence[MethodConfig] = (
    MethodConfig(
        name="Multi-Anchor",
        anchor_positions=ANCHOR_POSITIONS_5,
        use_anchor_init=True,
        use_feature_bank_guidance=True,
    ),
    MethodConfig(
        name="Single Anchor",
        anchor_positions=("center",),
        use_anchor_init=True,
        use_feature_bank_guidance=True,
    ),
    MethodConfig(
        name="Random Initialization",
        anchor_positions=("random",),
        use_anchor_init=False,
        use_feature_bank_guidance=True,
    ),
    MethodConfig(
        name="Anchor Without Feature Bank",
        anchor_positions=ANCHOR_POSITIONS_5,
        use_anchor_init=True,
        use_feature_bank_guidance=False,
    ),
)


def method_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")

METHOD_ALIASES = {
    "multi-anchor": "Multi-Anchor",
    "multi": "Multi-Anchor",
    "single-anchor": "Single Anchor",
    "single": "Single Anchor",
    "random-initialization": "Random Initialization",
    "random": "Random Initialization",
    "anchor-without-feature-bank": "Anchor Without Feature Bank",
    "no-feature-bank": "Anchor Without Feature Bank",
    "nofeaturebank": "Anchor Without Feature Bank",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation study for reconstruction methods")
    parser.add_argument("--ckpt-dir", default=".", help="Directory containing trained federated checkpoints")
    parser.add_argument("--output-dir", default=".", help="Output directory for CSV and LaTeX files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--images-per-class", type=int, default=10, help="Evaluation samples per class")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of CIFAR-10 classes")
    parser.add_argument("--target-classes", default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated class IDs")
    parser.add_argument("--steps", type=int, default=3000, help="Optimization steps")
    parser.add_argument("--restarts", type=int, default=1, help="Number of reconstruction restarts")
    parser.add_argument("--img-lr", type=float, default=0.05, help="Reconstruction learning rate")
    parser.add_argument("--max-real-per-class", type=int, default=2000, help="Max train images per class for bank")
    parser.add_argument("--topk-rerank", type=int, default=25, help="Top-k candidate rerank size")
    parser.add_argument("--anchor-pool", type=int, default=64, help="Anchor candidate pool size")
    parser.add_argument("--anchor-topk", type=int, default=8, help="Top-k anchors used for deterministic pick")
    parser.add_argument("--anchor-frac", type=float, default=0.20, help="Patch area fraction kept from anchor")
    parser.add_argument("--anchor-preserve-w", type=float, default=0.15, help="Anchor patch preservation loss weight")
    parser.add_argument("--init-noise", type=float, default=0.05, help="Init noise added to seeded anchor")
    parser.add_argument("--outside-noise", type=float, default=0.08, help="Noise level outside anchor patch")
    parser.add_argument("--outside-fill-mode", default="anchor_noise", choices=["noise", "anchor_noise"], help="Outside patch init mode")
    parser.add_argument("--prototype-topk", type=int, default=16, help="Top-k class confident images for prototype targets")
    parser.add_argument("--jitter-px", type=int, default=2, help="Random translation jitter")
    parser.add_argument("--aug-views", type=int, default=2, help="Augmented views per optimization step")
    parser.add_argument("--hflip-prob", type=float, default=0.0, help="Horizontal flip probability for aug")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip norm")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA for final image")
    parser.add_argument("--ema-decay", type=float, default=0.995, help="EMA decay")

    # Loss weights reused from recon_multianchor.py
    parser.add_argument("--ce-w", type=float, default=1.0)
    parser.add_argument("--bn-w", type=float, default=0.1)
    parser.add_argument("--tv-w", type=float, default=0.005)
    parser.add_argument("--tv2-w", type=float, default=0.001)
    parser.add_argument("--l2-w", type=float, default=1e-4)
    parser.add_argument("--feat-w", type=float, default=0.2)
    parser.add_argument("--perp-w", type=float, default=1.5)
    parser.add_argument("--color-w", type=float, default=0.05)
    parser.add_argument("--hf-w", type=float, default=0.002)
    parser.add_argument("--consistency-w", type=float, default=0.05)
    parser.add_argument("--logit-margin-w", type=float, default=0.15)

    parser.add_argument(
        "--experiments",
        default="E1,E2,E3,E4",
        help="Comma-separated experiment keys from default map (E1..E4)",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated methods to run. Use 'all' or names/aliases such as "
            "multi-anchor,single-anchor,random-initialization,no-feature-bank"
        ),
    )
    return parser.parse_args()


def resolve_methods(method_arg: str) -> List[MethodConfig]:
    text = (method_arg or "all").strip().lower()
    if text == "all":
        return list(METHODS)

    requested = [m.strip().lower() for m in text.split(",") if m.strip()]
    canonical = {m.name: m for m in METHODS}
    picked_names: List[str] = []

    for key in requested:
        resolved = METHOD_ALIASES.get(key)
        if resolved is None:
            # Allow exact method names (case-insensitive)
            for name in canonical:
                if key == name.lower():
                    resolved = name
                    break
        if resolved is None:
            valid = sorted(set(["all", *METHOD_ALIASES.keys(), *[m.name for m in METHODS]]))
            raise ValueError(f"Unknown method '{key}'. Valid options include: {valid}")
        if resolved not in picked_names:
            picked_names.append(resolved)

    return [canonical[name] for name in picked_names]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_model_input(x_01: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x_01 - mean) / std


def image_to_param(x_01: torch.Tensor) -> torch.Tensor:
    x_01 = x_01.clamp(1e-4, 1 - 1e-4)
    return torch.logit(x_01)


class BNStatHook:
    def __init__(self, model: nn.Module, device: str):
        self._losses: List[torch.Tensor] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.device = device
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                self._hooks.append(mod.register_forward_pre_hook(self._make_hook(mod)))

    def _make_hook(self, mod: nn.BatchNorm2d):
        def fn(_m, inp):
            x = inp[0]
            b_mean = x.mean([0, 2, 3])
            b_var = x.var([0, 2, 3]) + 1e-8
            r_mean = mod.running_mean.detach()
            r_var = mod.running_var.detach() + 1e-8
            self._losses.append(F.mse_loss(b_mean, r_mean) + F.mse_loss(b_var, r_var))

        return fn

    def loss(self) -> torch.Tensor:
        if not self._losses:
            return torch.tensor(0.0, device=self.device)
        v = sum(self._losses)
        self._losses.clear()
        return v

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
        + torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean()
    )


def tv2_loss(x: torch.Tensor) -> torch.Tensor:
    dx2 = x[:, :, :, 2:] - 2.0 * x[:, :, :, 1:-1] + x[:, :, :, :-2]
    dy2 = x[:, :, 2:, :] - 2.0 * x[:, :, 1:-1, :] + x[:, :, :-2, :]
    return dx2.abs().mean() + dy2.abs().mean()


def highfreq_loss(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=x.device,
    ).view(1, 1, 3, 3)
    k = k.repeat(x.shape[1], 1, 1, 1)
    y = F.conv2d(x, k, padding=1, groups=x.shape[1])
    return y.abs().mean()


def color_stats_loss(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x.mean((2, 3)), ref.mean((2, 3))) + F.mse_loss(x.std((2, 3)), ref.std((2, 3)))


def _anchor_rect(h: int, w: int, frac: float) -> Tuple[int, int]:
    frac = float(np.clip(frac, 1e-4, 1.0))
    area = max(1, int(round(h * w * frac)))
    side_h = max(1, int(round(np.sqrt(area * h / w))))
    side_w = max(1, int(round(area / side_h)))
    return min(side_h, h), min(side_w, w)


def make_anchor_mask(h: int, w: int, frac: float, mode: str, dev: str) -> torch.Tensor:
    side_h, side_w = _anchor_rect(h, w, frac)
    mode = (mode or "random").lower()
    if mode == "center":
        y0 = (h - side_h) // 2
        x0 = (w - side_w) // 2
    elif mode == "top_left":
        y0, x0 = 0, 0
    elif mode == "top_right":
        y0, x0 = 0, w - side_w
    elif mode == "bottom_left":
        y0, x0 = h - side_h, 0
    elif mode == "bottom_right":
        y0, x0 = h - side_h, w - side_w
    else:
        y0 = random.randint(0, h - side_h)
        x0 = random.randint(0, w - side_w)

    m = torch.zeros(1, 1, h, w, device=dev)
    m[:, :, y0 : y0 + side_h, x0 : x0 + side_w] = 1.0
    return m


def masked_anchor_init(
    anchor: torch.Tensor,
    frac: float,
    mode: str,
    outside_mode: str,
    out_noise: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if outside_mode == "anchor_noise":
        noise = (anchor + out_noise * torch.randn_like(anchor)).clamp(0, 1)
    else:
        noise = torch.rand_like(anchor)
    m = make_anchor_mask(anchor.shape[2], anchor.shape[3], frac=frac, mode=mode, dev=str(anchor.device))
    x0 = noise * (1.0 - m) + anchor * m
    return x0.clamp(0, 1), m


def make_aug_view(x: torch.Tensor, jitter_px: int, hflip_prob: float) -> torch.Tensor:
    y = x
    if jitter_px > 0:
        jx = random.randint(-jitter_px, jitter_px)
        jy = random.randint(-jitter_px, jitter_px)
        y = torch.roll(y, shifts=(jy, jx), dims=(2, 3))
    if hflip_prob > 0 and random.random() < hflip_prob:
        y = torch.flip(y, dims=(3,))
    return y


def get_model(device: str, ckpt_path: str) -> nn.Module:
    m = resnet34(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, 10)
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    return m.to(device).eval()


def get_features(model: nn.Module, x_norm: torch.Tensor) -> torch.Tensor:
    x = model.conv1(x_norm)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


def get_feature_pyramid(model: nn.Module, x_norm: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
    x = model.conv1(x_norm)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    f1 = model.layer1(x)
    f2 = model.layer2(f1)
    f3 = model.layer3(f2)
    f4 = model.layer4(f3)
    pooled = [
        F.adaptive_avg_pool2d(f1, 8),
        F.adaptive_avg_pool2d(f2, 4),
        F.adaptive_avg_pool2d(f3, 2),
        F.adaptive_avg_pool2d(f4, 1),
    ]
    pen = torch.flatten(model.avgpool(f4), 1)
    return pooled, pen


@torch.no_grad()
def build_real_bank(
    model: nn.Module,
    device: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    num_classes: int,
    max_real_per_class: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=T.ToTensor())
    bank: Dict[int, Dict[str, List]] = {c: {"imgs": [], "idxs": []} for c in range(num_classes)}

    for idx, (img, lbl) in enumerate(dataset):
        if len(bank[lbl]["imgs"]) < max_real_per_class:
            bank[lbl]["imgs"].append(img)
            bank[lbl]["idxs"].append(idx)

    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for c in range(num_classes):
        imgs = torch.stack(bank[c]["imgs"]).to(device)
        feats, class_p = [], []
        for i in range(0, imgs.size(0), 256):
            b = imgs[i : i + 256]
            bn = to_model_input(b, mean, std)
            feats.append(get_features(model, bn).cpu())
            class_p.append(model(bn).softmax(1)[:, c].cpu())
        out[c] = {
            "imgs": torch.stack(bank[c]["imgs"]),
            "idxs": torch.tensor(bank[c]["idxs"], dtype=torch.long),
            "feats": torch.cat(feats, dim=0),
            "class_p": torch.cat(class_p, dim=0),
        }
        out[c]["feat_mean"] = out[c]["feats"].mean(0)
    return out


@torch.no_grad()
def select_anchor_index(bank: Dict[int, Dict[str, torch.Tensor]], cls: int, sample_idx: int, anchor_pool: int, anchor_topk: int) -> int:
    class_p = bank[cls]["class_p"]
    feats = bank[cls]["feats"]
    feat_mean = bank[cls]["feat_mean"].unsqueeze(0)
    candidate_count = min(anchor_pool, class_p.numel())
    top = torch.argsort(class_p, descending=True)[:candidate_count]
    d = torch.cdist(feat_mean, feats[top]).squeeze(0)
    ordered = top[torch.argsort(d)]
    pool = ordered[: min(anchor_topk, ordered.numel())]
    return int(pool[sample_idx % len(pool)])


@torch.no_grad()
def build_anchor_targets(
    model: nn.Module,
    bank: Dict[int, Dict[str, torch.Tensor]],
    cls: int,
    sample_idx: int,
    anchor_pool: int,
    anchor_topk: int,
    device: str,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    idx = select_anchor_index(bank, cls, sample_idx, anchor_pool, anchor_topk)
    anchor_img = bank[cls]["imgs"][idx].to(device)
    pyr, feat = get_feature_pyramid(model, to_model_input(anchor_img.unsqueeze(0), mean, std))
    return anchor_img.cpu(), feat.squeeze(0).cpu(), [p.cpu() for p in pyr]


@torch.no_grad()
def build_class_prototype_targets(
    model: nn.Module,
    bank: Dict[int, Dict[str, torch.Tensor]],
    cls: int,
    topk: int,
    device: str,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    class_p = bank[cls]["class_p"]
    k = min(topk, class_p.numel())
    top = torch.argsort(class_p, descending=True)[:k]
    imgs = bank[cls]["imgs"][top].to(device)
    pyr, pen = get_feature_pyramid(model, to_model_input(imgs, mean, std))
    return pen.mean(0).cpu(), [p.mean(0, keepdim=True).cpu() for p in pyr]


@torch.no_grad()
def find_nearest(
    model: nn.Module,
    recon_01: torch.Tensor,
    cls: int,
    bank: Dict[int, Dict[str, torch.Tensor]],
    topk_rerank: int,
    device: str,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[torch.Tensor, int, float]:
    rf = get_features(model, to_model_input(recon_01.unsqueeze(0).to(device), mean, std)).cpu()
    fd = torch.cdist(rf, bank[cls]["feats"]).squeeze(0)

    k = min(topk_rerank, fd.numel())
    tk = torch.topk(fd, k=k, largest=False).indices
    top_imgs = bank[cls]["imgs"][tk]

    pix_d = ((top_imgs - recon_01.cpu()) ** 2).mean((1, 2, 3))
    best_loc = int(pix_d.argmin())
    best_g = int(tk[best_loc])

    real_img = bank[cls]["imgs"][best_g]
    real_dsidx = int(bank[cls]["idxs"][best_g].item())
    dist = float(fd[best_g].item())
    return real_img, real_dsidx, dist


def reconstruct_image(
    model: nn.Module,
    target_class: int,
    target_feat: Optional[torch.Tensor],
    target_pyramid: Optional[List[torch.Tensor]],
    init_img: Optional[torch.Tensor],
    sample_idx: int,
    anchor_patch_mode: str,
    args: argparse.Namespace,
    device: str,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    target_t = torch.tensor([target_class], device=device)
    best_img = None
    best_conf = -1.0
    bn_hook = BNStatHook(model, device)
    init_ref = None if init_img is None else init_img.unsqueeze(0).to(device)

    for restart in range(args.restarts):
        mode_key = str(anchor_patch_mode or "none")
        mode_hash = abs(hash(mode_key)) % 100000
        torch.manual_seed(args.seed + restart * 997 + target_class * 31 + sample_idx * 10007 + mode_hash)

        if init_ref is None:
            x0 = torch.rand(1, 3, 32, 32, device=device)
            anchor_mask = None
        else:
            base = (init_ref + args.init_noise * torch.randn_like(init_ref)).clamp(0, 1)
            x0, anchor_mask = masked_anchor_init(
                base,
                frac=args.anchor_frac,
                mode=anchor_patch_mode,
                outside_mode=args.outside_fill_mode,
                out_noise=args.outside_noise,
            )

        u = image_to_param(x0).detach().clone().requires_grad_(True)
        ema_u = u.detach().clone()
        opt = optim.Adam([u], lr=args.img_lr, betas=(0.9, 0.999))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-4)

        for _step in range(args.steps):
            opt.zero_grad()
            bn_hook._losses.clear()

            x = torch.sigmoid(u)
            v_count = max(1, args.aug_views)
            logits_list, feat_list, pyr_list = [], [], []

            for _ in range(v_count):
                xj = make_aug_view(x.clamp(0, 1), args.jitter_px, args.hflip_prob)
                xjn = to_model_input(xj, mean, std)
                logits_v = model(xjn)
                pyr_v, feat_v = get_feature_pyramid(model, xjn)
                logits_list.append(logits_v)
                feat_list.append(feat_v)
                pyr_list.append(pyr_v)

            logits = torch.stack(logits_list, dim=0).mean(0)
            feat = torch.stack(feat_list, dim=0).mean(0)
            pyr = [torch.stack([p[li] for p in pyr_list], dim=0).mean(0) for li in range(len(pyr_list[0]))]

            l_ce = F.cross_entropy(logits, target_t)
            target_logit = logits[0, target_class]
            other_logits = logits.clone()
            other_logits[0, target_class] = -1e9
            max_other = other_logits.max(1).values[0]
            l_margin = F.relu(0.30 - (target_logit - max_other))

            l_bn = bn_hook.loss() / max(1, v_count)
            l_tv = tv_loss(x)
            l_tv2 = tv2_loss(x)
            l_hf = highfreq_loss(x)
            l_l2 = (x**2).mean()

            l_feat = torch.tensor(0.0, device=device)
            if target_feat is not None:
                l_feat = F.mse_loss(feat, target_feat.unsqueeze(0).to(device))

            l_perp = torch.tensor(0.0, device=device)
            if target_pyramid is not None:
                l_perp = sum(F.mse_loss(cur, ref.to(device)) for cur, ref in zip(pyr, target_pyramid))

            l_color = color_stats_loss(x, init_ref) if init_ref is not None else torch.tensor(0.0, device=device)
            l_anchor = (
                F.mse_loss(x * anchor_mask, init_ref * anchor_mask)
                if (init_ref is not None and anchor_mask is not None)
                else torch.tensor(0.0, device=device)
            )

            if len(feat_list) > 1:
                fstack = torch.stack(feat_list, dim=0)
                l_cons = fstack.var(dim=0, unbiased=False).mean()
            else:
                l_cons = torch.tensor(0.0, device=device)

            if init_ref is None:
                bn_coef, tv_coef, feat_coef, perp_coef = 1.35, 2.25, 1.40, 1.50
            else:
                bn_coef, tv_coef, feat_coef, perp_coef = 1.00, 1.00, 1.00, 1.00

            loss = (
                args.ce_w * l_ce
                + args.logit_margin_w * l_margin
                + (args.bn_w * bn_coef) * l_bn
                + (args.tv_w * tv_coef) * l_tv
                + args.tv2_w * l_tv2
                + args.hf_w * l_hf
                + args.l2_w * l_l2
                + (args.feat_w * feat_coef) * l_feat
                + (args.perp_w * perp_coef) * l_perp
                + args.color_w * l_color
                + args.anchor_preserve_w * l_anchor
                + args.consistency_w * l_cons
            )

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([u], args.grad_clip)
            opt.step()
            sch.step()
            if args.use_ema:
                with torch.no_grad():
                    ema_u.mul_(args.ema_decay).add_(u.detach(), alpha=(1.0 - args.ema_decay))

        with torch.no_grad():
            final = torch.sigmoid(ema_u if args.use_ema else u).clamp(0, 1).squeeze(0)
            conf = model(to_model_input(final.unsqueeze(0), mean, std)).softmax(1)[0, target_class].item()

        if conf > best_conf:
            best_conf = conf
            best_img = final.detach().clone()

    bn_hook.remove()
    assert best_img is not None
    return best_img


def save_comparison_panel(
    recon_01: torch.Tensor,
    real_01: torch.Tensor,
    path: str,
    cls: int,
    exp_name: str,
    method_name: str,
    anchor_pos: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(recon_01.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title(f"Reconstructed ({CLASS_NAMES[cls]})", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(real_01.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Nearest Real Training Image", fontsize=11)
    axes[1].axis("off")
    fig.suptitle(f"{exp_name} | {method_name} | class {cls}: {CLASS_NAMES[cls]} | anchor={anchor_pos}", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_anchor_preview(anchor_01: torch.Tensor, frac: float, mode: str, device: str) -> torch.Tensor:
    m = make_anchor_mask(anchor_01.shape[1], anchor_01.shape[2], frac=frac, mode=mode, dev=device)
    preview = anchor_01.to(device) * (0.35 + 0.65 * m.squeeze(0))
    return preview.clamp(0, 1).cpu()


def save_anchor_triplet_grid(
    rows: List[Dict[str, torch.Tensor]],
    path: str,
    cls: int,
    exp_name: str,
    method_name: str,
    img_idx: int,
) -> None:
    n = len(rows)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.0 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, row in enumerate(rows):
        ax0, ax1, ax2 = axes[r, 0], axes[r, 1], axes[r, 2]

        ax0.imshow(row["anchor_preview"].permute(1, 2, 0).cpu().numpy())
        ax0.set_title(f"Anchor ({row['anchor_pos']})", fontsize=10)
        ax0.axis("off")

        ax1.imshow(row["recon"].permute(1, 2, 0).cpu().numpy())
        ax1.set_title("Generated", fontsize=10)
        ax1.axis("off")

        nearest = row.get("nearest")
        if nearest is not None:
            ax2.imshow(nearest.permute(1, 2, 0).cpu().numpy())
            ax2.set_title("Nearest Real", fontsize=10)
        else:
            ax2.imshow(np.zeros((32, 32, 3), dtype=np.float32))
            ax2.set_title("Nearest Real (N/A)", fontsize=10)
        ax2.axis("off")

    fig.suptitle(
        f"{exp_name} | {method_name} | class {cls}: {CLASS_NAMES[cls]} | sample {img_idx}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_generated_montage(
    anchor_batches: List[Tuple[str, torch.Tensor]],
    path: str,
    cls: int,
    exp_name: str,
    method_name: str,
    img_idx: int,
) -> None:
    n = len(anchor_batches)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.8))
    if n == 1:
        axes = [axes]

    for i, (anchor_pos, img) in enumerate(anchor_batches):
        axes[i].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(anchor_pos, fontsize=9)
        axes[i].axis("off")

    fig.suptitle(
        f"Generated Images | {exp_name} | {method_name} | class {cls}: {CLASS_NAMES[cls]} | sample {img_idx}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    nx, ny = len(x), len(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(1, (nx + ny - 2))
    if pooled <= 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def fmt_stat(mean_v: float, std_v: float, digits: int = 4) -> str:
    return f"{mean_v:.{digits}f} +- {std_v:.{digits}f}"


def write_latex_ablation(path: str, summary_rows: List[Dict[str, float]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation study results across experiments.}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{llcccc}\n")
        f.write("\\toprule\n")
        f.write("Experiment & Method & MSE & SSIM & Feature Dist. & Unique Nearest \\\\ \n")
        f.write("\\midrule\n")
        for r in summary_rows:
            f.write(
                f"{r['Experiment']} & {r['Method']} & "
                f"{fmt_stat(r['Mean_MSE'], r['Std_MSE'])} & "
                f"{fmt_stat(r['Mean_SSIM'], r['Std_SSIM'])} & "
                f"{fmt_stat(r['Mean_Feature_Distance'], r['Std_Feature_Distance'])} & "
                f"{fmt_stat(r['Mean_Unique_Nearest'], r['Std_Unique_Nearest'])} \\\\ \n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def write_latex_significance(path: str, sig_rows: List[Dict[str, float]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Significance tests against Multi-Anchor baseline.}\n")
        f.write("\\label{tab:ablation-significance}\n")
        f.write("\\begin{tabular}{lllccc}\n")
        f.write("\\toprule\n")
        f.write("Experiment & Comparison & Metric & Welch p & Mann-Whitney p & Cohen's d \\\\ \n")
        f.write("\\midrule\n")
        for r in sig_rows:
            f.write(
                f"{r['Experiment']} & {r['Comparison']} & {r['Metric']} & "
                f"{r['Welch_p']:.6g} & {r['MannWhitney_p']:.6g} & {r['Cohens_d']:.6g} \\\\ \n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    target_classes = [int(x.strip()) for x in args.target_classes.split(",") if x.strip()]
    selected_experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
    selected_methods = resolve_methods(args.methods)

    experiments = {}
    for e in selected_experiments:
        if e not in DEFAULT_EXPERIMENTS:
            raise ValueError(f"Unknown experiment '{e}'. Available: {list(DEFAULT_EXPERIMENTS.keys())}")
        experiments[e] = DEFAULT_EXPERIMENTS[e]

    os.makedirs(args.output_dir, exist_ok=True)

    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)

    ablation_rows: List[Dict[str, float]] = []

    for exp_name, exp_info in experiments.items():
        ckpt_path = os.path.join(args.ckpt_dir, exp_info["ckpt"])
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found for {exp_name}: {ckpt_path}")

        exp_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        print(f"[Experiment {exp_name}] loading model from {ckpt_path}")
        model = get_model(device, ckpt_path)
        print("  building feature/real bank...")
        bank = build_real_bank(model, device, mean, std, args.num_classes, args.max_real_per_class)

        for method in selected_methods:
            print(f"  method={method.name}")
            method_dir = os.path.join(exp_dir, method_slug(method.name))
            os.makedirs(method_dir, exist_ok=True)
            for cls in target_classes:
                for img_idx in range(args.images_per_class):
                    if method.use_anchor_init:
                        anchor_img, target_feat, target_pyr = build_anchor_targets(
                            model,
                            bank,
                            cls,
                            img_idx,
                            args.anchor_pool,
                            args.anchor_topk,
                            device,
                            mean,
                            std,
                        )
                    else:
                        anchor_img = None
                        target_feat, target_pyr = build_class_prototype_targets(
                            model,
                            bank,
                            cls,
                            args.prototype_topk,
                            device,
                            mean,
                            std,
                        )

                    if not method.use_feature_bank_guidance:
                        target_feat = None
                        target_pyr = None

                    recon_batch: List[Tuple[str, torch.Tensor]] = []
                    for anchor_pos in method.anchor_positions:
                        recon = reconstruct_image(
                            model=model,
                            target_class=cls,
                            target_feat=target_feat,
                            target_pyramid=target_pyr,
                            init_img=anchor_img,
                            sample_idx=img_idx,
                            anchor_patch_mode=anchor_pos,
                            args=args,
                            device=device,
                            mean=mean,
                            std=std,
                        )
                        recon_batch.append((anchor_pos, recon))

                    mse_vals: List[float] = []
                    ssim_vals: List[float] = []
                    feat_dists: List[float] = []
                    nearest_ids: List[int] = []
                    group_panel_rows: List[Dict[str, torch.Tensor]] = []

                    for _pos, recon in recon_batch:
                        stem = f"{exp_name}_{method_slug(method.name)}_class{cls}_img{img_idx}_anchor-{_pos}"
                        plt.imsave(
                            os.path.join(method_dir, f"{stem}.png"),
                            recon.detach().cpu().permute(1, 2, 0).numpy(),
                        )

                        real_img, ridx, fd = find_nearest(
                            model,
                            recon,
                            cls,
                            bank,
                            args.topk_rerank,
                            device,
                            mean,
                            std,
                        )
                        mse_v = float(((recon.cpu() - real_img.cpu()) ** 2).mean().item())
                        ssim_v = float(
                            compute_ssim(
                                recon.permute(1, 2, 0).cpu().numpy(),
                                real_img.permute(1, 2, 0).cpu().numpy(),
                                channel_axis=2,
                                data_range=1.0,
                            )
                        )
                        mse_vals.append(mse_v)
                        ssim_vals.append(ssim_v)
                        feat_dists.append(fd)
                        nearest_ids.append(ridx)

                        plt.imsave(
                            os.path.join(method_dir, f"{stem}_nearest_real.png"),
                            real_img.permute(1, 2, 0).cpu().numpy(),
                        )
                        save_comparison_panel(
                            recon.detach().cpu(),
                            real_img.detach().cpu(),
                            os.path.join(method_dir, f"{stem}_comparison.png"),
                            cls,
                            exp_name,
                            method.name,
                            _pos,
                        )

                        if anchor_img is not None and _pos in {"top_left", "top_right", "center", "bottom_left", "bottom_right", "random"}:
                            anchor_preview = make_anchor_preview(anchor_img, frac=args.anchor_frac, mode=_pos, device=device)
                        elif anchor_img is not None:
                            anchor_preview = anchor_img.detach().cpu()
                        else:
                            anchor_preview = recon.detach().cpu()

                        group_panel_rows.append(
                            {
                                "anchor_pos": _pos,
                                "anchor_preview": anchor_preview,
                                "recon": recon.detach().cpu(),
                                "nearest": real_img.detach().cpu(),
                            }
                        )

                    mean_mse = float(np.mean(mse_vals))
                    mean_ssim = float(np.mean(ssim_vals))
                    mean_fd = float(np.mean(feat_dists))
                    min_fd = float(np.min(feat_dists))
                    max_fd = float(np.max(feat_dists))
                    unique_nearest = int(len(set(nearest_ids)))

                    ablation_rows.append(
                        {
                            "Experiment": exp_name,
                            "Method": method.name,
                            "Class": cls,
                            "Class_Name": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls),
                            "Image": img_idx,
                            "MSE": mean_mse,
                            "SSIM": mean_ssim,
                            "Feature_Distance": mean_fd,
                            "Min_Feature_Distance": min_fd,
                            "Max_Feature_Distance": max_fd,
                            "Unique_Nearest_Count": unique_nearest,
                        }
                    )
                    print(
                        f"    cls={cls:02d} img={img_idx:02d} "
                        f"mse={mean_mse:.4f} ssim={mean_ssim:.4f} fd={mean_fd:.4f} unique={unique_nearest}"
                    )

                    group_stem = f"{exp_name}_{method_slug(method.name)}_class{cls}_img{img_idx}"
                    save_anchor_triplet_grid(
                        group_panel_rows,
                        os.path.join(method_dir, f"{group_stem}_anchors_generated_nearest.png"),
                        cls,
                        exp_name,
                        method.name,
                        img_idx,
                    )
                    save_generated_montage(
                        recon_batch,
                        os.path.join(method_dir, f"{group_stem}_generated_montage.png"),
                        cls,
                        exp_name,
                        method.name,
                        img_idx,
                    )

    results_path = os.path.join(args.output_dir, "ablation_results.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Experiment",
                "Method",
                "Class",
                "Class_Name",
                "Image",
                "MSE",
                "SSIM",
                "Feature_Distance",
                "Min_Feature_Distance",
                "Max_Feature_Distance",
                "Unique_Nearest_Count",
            ],
        )
        writer.writeheader()
        writer.writerows(ablation_rows)

    # Summary rows
    summary_rows: List[Dict[str, float]] = []
    for exp_name in experiments.keys():
        for method in [m.name for m in selected_methods]:
            rows = [r for r in ablation_rows if r["Experiment"] == exp_name and r["Method"] == method]
            if not rows:
                continue

            mse = np.array([r["MSE"] for r in rows], dtype=np.float64)
            ssim = np.array([r["SSIM"] for r in rows], dtype=np.float64)
            fd = np.array([r["Feature_Distance"] for r in rows], dtype=np.float64)
            unq = np.array([r["Unique_Nearest_Count"] for r in rows], dtype=np.float64)

            summary_rows.append(
                {
                    "Experiment": exp_name,
                    "Method": method,
                    "Mean_MSE": float(np.mean(mse)),
                    "Std_MSE": float(np.std(mse, ddof=1) if len(mse) > 1 else 0.0),
                    "Mean_SSIM": float(np.mean(ssim)),
                    "Std_SSIM": float(np.std(ssim, ddof=1) if len(ssim) > 1 else 0.0),
                    "Mean_Feature_Distance": float(np.mean(fd)),
                    "Std_Feature_Distance": float(np.std(fd, ddof=1) if len(fd) > 1 else 0.0),
                    "Mean_Unique_Nearest": float(np.mean(unq)),
                    "Std_Unique_Nearest": float(np.std(unq, ddof=1) if len(unq) > 1 else 0.0),
                }
            )

    summary_path = os.path.join(args.output_dir, "ablation_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Experiment",
                "Method",
                "Mean_MSE",
                "Std_MSE",
                "Mean_SSIM",
                "Std_SSIM",
                "Mean_Feature_Distance",
                "Std_Feature_Distance",
                "Mean_Unique_Nearest",
                "Std_Unique_Nearest",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Significance tests
    comparisons = [
        ("Multi-Anchor", "Single Anchor", "Multi-Anchor vs Single Anchor"),
        ("Multi-Anchor", "Random Initialization", "Multi-Anchor vs Random Initialization"),
        ("Multi-Anchor", "Anchor Without Feature Bank", "Multi-Anchor vs No Feature Bank"),
    ]

    sig_rows: List[Dict[str, float]] = []
    for exp_name in experiments.keys():
        for base, other, label in comparisons:
            base_rows = [r for r in ablation_rows if r["Experiment"] == exp_name and r["Method"] == base]
            oth_rows = [r for r in ablation_rows if r["Experiment"] == exp_name and r["Method"] == other]
            if not base_rows or not oth_rows:
                continue

            for metric in ["SSIM", "Feature_Distance"]:
                x = np.array([r[metric] for r in base_rows], dtype=np.float64)
                y = np.array([r[metric] for r in oth_rows], dtype=np.float64)

                welch_p = float(ttest_ind(x, y, equal_var=False, nan_policy="omit").pvalue)
                mann_p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
                d = cohens_d(x, y)

                sig_rows.append(
                    {
                        "Experiment": exp_name,
                        "Comparison": label,
                        "Metric": metric,
                        "Welch_p": welch_p,
                        "MannWhitney_p": mann_p,
                        "Cohens_d": d,
                    }
                )

    sig_path = os.path.join(args.output_dir, "ablation_significance.csv")
    with open(sig_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Experiment",
                "Comparison",
                "Metric",
                "Welch_p",
                "MannWhitney_p",
                "Cohens_d",
            ],
        )
        writer.writeheader()
        writer.writerows(sig_rows)

    # LaTeX tables
    summary_sorted = sorted(summary_rows, key=lambda r: (r["Experiment"], r["Method"]))
    sig_sorted = sorted(sig_rows, key=lambda r: (r["Experiment"], r["Comparison"], r["Metric"]))

    tex_ablation = os.path.join(args.output_dir, "table_ablation.tex")
    tex_significance = os.path.join(args.output_dir, "table_significance.tex")
    write_latex_ablation(tex_ablation, summary_sorted)
    write_latex_significance(tex_significance, sig_sorted)

    print("\nAblation study completed.")
    print(f"  Results CSV       : {results_path}")
    print(f"  Summary CSV       : {summary_path}")
    print(f"  Significance CSV  : {sig_path}")
    print(f"  LaTeX (ablation)  : {tex_ablation}")
    print(f"  LaTeX (signif.)   : {tex_significance}")


if __name__ == "__main__":
    main()
