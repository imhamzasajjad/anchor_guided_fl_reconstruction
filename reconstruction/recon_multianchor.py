#!/usr/bin/env python3
"""
recon.py  v4  –  Federated Model Reconstruction (CIFAR-10)
==========================================================
Adds multi-anchor patch reconstruction and anchor-consistency analysis.

What is new in v4:
  1. Fixed patch anchors per selected anchor image:
     top_left, top_right, bottom_left, bottom_right, center
  2. One reconstruction per anchor position
  3. Per-anchor nearest-real search and per-image metrics
  4. Intra-group similarity analysis across anchor outputs (same class/sample)
"""

import os
import csv
import random
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet34
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as compute_ssim


# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Experiments ───────────────────────────────────────────────────────────────
experiments = {
#    "E5": {"heterogeneity": "IID", "ckpt": "gm_best_100_IID.pt"},
    "E6": {"heterogeneity": "Mild", "ckpt": "gm_best_100_Non_IID(Mild).pt"},
    "E7": {"heterogeneity": "Strong", "ckpt": "gm_best_100_Non_IID(Moderate).pt"},
    "E8": {"heterogeneity": "Extreme", "ckpt": "gm_best_100_Non_IID(Extreme).pt"},
}


# ── Default config ────────────────────────────────────────────────────────────
num_classes = 10
images_per_class = 10
steps = 3000
num_restarts = 1
ce_w = 1.0
bn_w = 0.1
tv_w = 0.005
tv2_w = 0.001
l2_w = 1e-4
feat_w = 0.2
perp_w = 1.5
color_w = 0.05
hf_w = 0.002
consistency_w = 0.05
logit_margin_w = 0.15
img_lr = 0.05
anchor_pool = 64
anchor_topk = 8
init_noise = 0.05
outside_noise = 0.08
outside_fill_mode = "anchor_noise"  # noise | anchor_noise
output_dir = "recon_output_multianchor"
project_real = True
project_final_real = False
max_real_per_class = 2000
topk_rerank = 25
target_classes = list(range(num_classes))
attack_mode = "optimize_project"  # or "train_retrieve"
use_anchor_init = True
prototype_topk = 16
jitter_px = 2
aug_views = 2
hflip_prob = 0.0
grad_clip = 1.0
use_ema_output = True
ema_decay = 0.995

# Partial-anchor controls
anchor_frac = 0.20            # keep only 20% of anchor
anchor_mode = "random"        # random | center | top_left | top_right | bottom_left | bottom_right
anchor_preserve_w = 0.15      # preserve seeded patch during optimisation

# New: multi-anchor positions to run for each selected anchor image
anchor_positions = ["top_left", "top_right", "center", "bottom_left", "bottom_right"]


# ── Environment overrides ─────────────────────────────────────────────────────
steps = int(os.getenv("RECON_STEPS", str(steps)))
images_per_class = int(os.getenv("RECON_IMAGES_PER_CLASS", str(images_per_class)))
num_restarts = int(os.getenv("RECON_RESTARTS", str(num_restarts)))
img_lr = float(os.getenv("RECON_LR", str(img_lr)))
output_dir = os.getenv("RECON_OUTPUT_DIR", output_dir)
project_real = os.getenv("RECON_PROJECT_REAL", "1").lower() not in {"0", "false", "no"}
project_final_real = os.getenv("RECON_PROJECT_FINAL_REAL", "0").lower() in {"1", "true", "yes"}
max_real_per_class = int(os.getenv("RECON_MAX_REAL_PER_CLASS", str(max_real_per_class)))
attack_mode = os.getenv("RECON_ATTACK_MODE", attack_mode)
topk_rerank = int(os.getenv("RECON_TOPK_RERANK", str(topk_rerank)))
bn_w = float(os.getenv("RECON_BN_W", str(bn_w)))
tv_w = float(os.getenv("RECON_TV_W", str(tv_w)))
feat_w = float(os.getenv("RECON_FEAT_W", str(feat_w)))
perp_w = float(os.getenv("RECON_PERP_W", str(perp_w)))
color_w = float(os.getenv("RECON_COLOR_W", str(color_w)))
hf_w = float(os.getenv("RECON_HF_W", str(hf_w)))
consistency_w = float(os.getenv("RECON_CONSISTENCY_W", str(consistency_w)))
logit_margin_w = float(os.getenv("RECON_LOGIT_MARGIN_W", str(logit_margin_w)))
anchor_pool = int(os.getenv("RECON_ANCHOR_POOL", str(anchor_pool)))
anchor_topk = int(os.getenv("RECON_ANCHOR_TOPK", str(anchor_topk)))
init_noise = float(os.getenv("RECON_INIT_NOISE", str(init_noise)))
outside_noise = float(os.getenv("RECON_OUTSIDE_NOISE", str(outside_noise)))
outside_fill_mode = os.getenv("RECON_OUTSIDE_FILL_MODE", outside_fill_mode).lower()
use_anchor_init = os.getenv("RECON_USE_ANCHOR_INIT", "1").lower() not in {"0", "false", "no"}
prototype_topk = int(os.getenv("RECON_PROTOTYPE_TOPK", str(prototype_topk)))
jitter_px = int(os.getenv("RECON_JITTER_PX", str(jitter_px)))
aug_views = int(os.getenv("RECON_AUG_VIEWS", str(aug_views)))
hflip_prob = float(os.getenv("RECON_HFLIP_PROB", str(hflip_prob)))
grad_clip = float(os.getenv("RECON_GRAD_CLIP", str(grad_clip)))
use_ema_output = os.getenv("RECON_USE_EMA", "1").lower() not in {"0", "false", "no"}
ema_decay = float(os.getenv("RECON_EMA_DECAY", str(ema_decay)))

anchor_frac = float(os.getenv("RECON_ANCHOR_FRAC", str(anchor_frac)))
anchor_mode = os.getenv("RECON_ANCHOR_MODE", anchor_mode)
anchor_preserve_w = float(os.getenv("RECON_ANCHOR_PRESERVE_W", str(anchor_preserve_w)))

if os.getenv("RECON_ANCHOR_POSITIONS"):
    anchor_positions = [p.strip().lower() for p in os.getenv("RECON_ANCHOR_POSITIONS", "").split(",") if p.strip()]

if os.getenv("RECON_ANCHOR_COUNT"):
    anchor_count = max(1, int(os.getenv("RECON_ANCHOR_COUNT", "5")))
    anchor_positions = anchor_positions[:anchor_count]

_anchor_alias = {
    "topleft": "top_left",
    "top-left": "top_left",
    "topright": "top_right",
    "top-right": "top_right",
    "bottomleft": "bottom_left",
    "bottom-left": "bottom_left",
    "bottomright": "bottom_right",
    "bottom-right": "bottom_right",
    "centre": "center",
}
_allowed_anchor_positions = {"random", "center", "top_left", "top_right", "bottom_left", "bottom_right"}

norm_positions = []
for p in anchor_positions:
    k = _anchor_alias.get(p, p)
    if k not in _allowed_anchor_positions:
        raise ValueError(
            f"Unknown anchor position '{p}'. Allowed: {sorted(_allowed_anchor_positions)}"
        )
    norm_positions.append(k)
anchor_positions = norm_positions

if os.getenv("RECON_TARGET_CLASS"):
    target_classes = [int(os.getenv("RECON_TARGET_CLASS"))]

if os.getenv("RECON_EXPERIMENT"):
    ek = os.getenv("RECON_EXPERIMENT")
    if ek not in experiments:
        raise ValueError(f"Unknown experiment key '{ek}'. Available: {list(experiments.keys())}")
    experiments = {ek: experiments[ek]}

_seed = int(os.getenv("RECON_SEED", "42"))
random.seed(_seed)
np.random.seed(_seed)
torch.manual_seed(_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_seed)

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# ── Normalisation ─────────────────────────────────────────────────────────────
_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
_STD = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)


def to_model_input(x_01):
    """[0,1] image -> normalised tensor ready for the model."""
    return (x_01 - _MEAN) / _STD


def image_to_param(x_01):
    """Map [0,1] image to unconstrained space for stable optimisation."""
    x_01 = x_01.clamp(1e-4, 1 - 1e-4)
    return torch.logit(x_01)


# ── Model ─────────────────────────────────────────────────────────────────────
def get_model(ckpt=None):
    m = resnet34(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, 10)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    return m.to(device).eval()


# ── BN pre-activation statistics hook ────────────────────────────────────────
class BNStatHook:
    """
    Hooks the INPUT to every BatchNorm2d layer and computes an L2 loss
    between the per-channel spatial mean/var and the stored running stats.
    """

    def __init__(self, model):
        self._losses = []
        self._hooks = []
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                h = mod.register_forward_pre_hook(self._make_hook(mod))
                self._hooks.append(h)

    def _make_hook(self, mod):
        def fn(m, inp):
            x = inp[0]
            b_mean = x.mean([0, 2, 3])
            b_var = x.var([0, 2, 3]) + 1e-8
            r_mean = mod.running_mean.detach()
            r_var = mod.running_var.detach() + 1e-8
            loss = F.mse_loss(b_mean, r_mean) + F.mse_loss(b_var, r_var)
            self._losses.append(loss)

        return fn

    def loss(self):
        if not self._losses:
            return torch.tensor(0.0, device=device)
        v = sum(self._losses)
        self._losses.clear()
        return v

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Losses ────────────────────────────────────────────────────────────────────
def tv_loss(x):
    return (torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean() +
            torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean())


def tv2_loss(x):
    dx2 = x[:, :, :, 2:] - 2.0 * x[:, :, :, 1:-1] + x[:, :, :, :-2]
    dy2 = x[:, :, 2:, :] - 2.0 * x[:, :, 1:-1, :] + x[:, :, :-2, :]
    return (dx2.abs().mean() + dy2.abs().mean())


def highfreq_loss(x):
    k = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=x.device,
    ).view(1, 1, 3, 3)
    k = k.repeat(x.shape[1], 1, 1, 1)
    y = F.conv2d(x, k, padding=1, groups=x.shape[1])
    return y.abs().mean()


def color_stats_loss(x, ref):
    x_mean = x.mean((2, 3))
    r_mean = ref.mean((2, 3))
    x_std = x.std((2, 3))
    r_std = ref.std((2, 3))
    return F.mse_loss(x_mean, r_mean) + F.mse_loss(x_std, r_std)


# ── Partial-anchor utilities ──────────────────────────────────────────────────
def _anchor_rect(h, w, frac=0.20):
    frac = float(np.clip(frac, 1e-4, 1.0))
    area = max(1, int(round(h * w * frac)))
    side_h = max(1, int(round(np.sqrt(area * h / w))))
    side_w = max(1, int(round(area / side_h)))
    side_h = min(side_h, h)
    side_w = min(side_w, w)
    return side_h, side_w


def make_anchor_mask(h=32, w=32, frac=0.20, mode="random", dev=device):
    """Create a binary mask with ~frac area set to 1 using requested placement."""
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
    m[:, :, y0:y0 + side_h, x0:x0 + side_w] = 1.0
    return m


def masked_anchor_init(anchor, frac=0.20, mode="random", outside_mode="anchor_noise", out_noise=0.08):
    """Mix random noise with anchor, preserving only a patch (~frac area)."""
    if outside_mode == "anchor_noise":
        noise = (anchor + out_noise * torch.randn_like(anchor)).clamp(0, 1)
    else:
        noise = torch.rand_like(anchor)
    m = make_anchor_mask(anchor.shape[2], anchor.shape[3], frac=frac, mode=mode, dev=anchor.device)
    x0 = noise * (1.0 - m) + anchor * m
    return x0.clamp(0, 1), m


def make_aug_view(x):
    y = x
    if jitter_px > 0:
        jx = random.randint(-jitter_px, jitter_px)
        jy = random.randint(-jitter_px, jitter_px)
        y = torch.roll(y, shifts=(jy, jx), dims=(2, 3))
    if hflip_prob > 0 and random.random() < hflip_prob:
        y = torch.flip(y, dims=(3,))
    return y


# ── Feature extraction ────────────────────────────────────────────────────────
def get_features(model, x_norm):
    x = x_norm
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


def get_feature_pyramid(model, x_norm):
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


# ── Core optimisation ─────────────────────────────────────────────────────────
def reconstruct_image(
    model,
    target_class,
    target_feat=None,
    target_pyramid=None,
    init_img=None,
    sample_idx=0,
    anchor_patch_mode=None,
):
    target_t = torch.tensor([target_class], device=device)
    best_img = None
    best_conf = -1.0
    bn_hook = BNStatHook(model)
    init_ref = None if init_img is None else init_img.unsqueeze(0).to(device)

    for restart in range(num_restarts):
        mode_key = str(anchor_patch_mode or "none")
        mode_hash = abs(hash(mode_key)) % 100000
        torch.manual_seed(_seed + restart * 997 + target_class * 31 + sample_idx * 10007 + mode_hash)

        if init_ref is None:
            x0 = torch.rand(1, 3, 32, 32, device=device)
            anchor_mask = None
        else:
            base = (init_ref + init_noise * torch.randn_like(init_ref)).clamp(0, 1)
            use_mode = anchor_patch_mode if anchor_patch_mode is not None else anchor_mode
            x0, anchor_mask = masked_anchor_init(
                base,
                frac=anchor_frac,
                mode=use_mode,
                outside_mode=outside_fill_mode,
                out_noise=outside_noise,
            )

        u = image_to_param(x0).detach().clone().requires_grad_(True)
        ema_u = u.detach().clone()
        opt = optim.Adam([u], lr=img_lr, betas=(0.9, 0.999))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)

        for step in range(steps):
            opt.zero_grad()
            bn_hook._losses.clear()

            x = torch.sigmoid(u)

            v_count = max(1, aug_views)
            logits_list = []
            feat_list = []
            pyr_list = []
            for _ in range(v_count):
                xj = make_aug_view(x.clamp(0, 1))
                logits_v = model(to_model_input(xj))
                pyr_v, feat_v = get_feature_pyramid(model, to_model_input(xj))
                logits_list.append(logits_v)
                feat_list.append(feat_v)
                pyr_list.append(pyr_v)

            logits = torch.stack(logits_list, dim=0).mean(0)
            feat = torch.stack(feat_list, dim=0).mean(0)
            pyr = []
            for li in range(len(pyr_list[0])):
                pyr.append(torch.stack([p[li] for p in pyr_list], dim=0).mean(0))

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
            l_l2 = (x ** 2).mean()

            if target_feat is not None:
                l_feat = F.mse_loss(feat, target_feat.unsqueeze(0).to(device))
            else:
                l_feat = torch.tensor(0.0, device=device)

            if target_pyramid is not None:
                l_perp = sum(F.mse_loss(cur, ref.to(device)) for cur, ref in zip(pyr, target_pyramid))
            else:
                l_perp = torch.tensor(0.0, device=device)

            if init_ref is not None:
                l_color = color_stats_loss(x, init_ref)
            else:
                l_color = torch.tensor(0.0, device=device)

            if init_ref is not None and anchor_mask is not None:
                l_anchor = F.mse_loss(x * anchor_mask, init_ref * anchor_mask)
            else:
                l_anchor = torch.tensor(0.0, device=device)

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
                ce_w * l_ce
                + logit_margin_w * l_margin
                + (bn_w * bn_coef) * l_bn
                + (tv_w * tv_coef) * l_tv
                + tv2_w * l_tv2
                + hf_w * l_hf
                + l2_w * l_l2
                + (feat_w * feat_coef) * l_feat
                + (perp_w * perp_coef) * l_perp
                + color_w * l_color
                + anchor_preserve_w * l_anchor
                + consistency_w * l_cons
            )

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([u], grad_clip)
            opt.step()
            sch.step()
            if use_ema_output:
                with torch.no_grad():
                    ema_u.mul_(ema_decay).add_(u.detach(), alpha=(1.0 - ema_decay))

            if step % 500 == 0:
                with torch.no_grad():
                    c = model(to_model_input(torch.sigmoid(u))).softmax(1)[0, target_class].item()
                print(
                    f"  [cls={target_class} mode={anchor_patch_mode} restart={restart + 1}/{num_restarts} step={step:4d}] "
                    f"conf={c:.4f} loss={loss.item():.4f}"
                )

        with torch.no_grad():
            if use_ema_output:
                final = torch.sigmoid(ema_u).clamp(0, 1).squeeze(0)
            else:
                final = torch.sigmoid(u).clamp(0, 1).squeeze(0)
            conf = model(to_model_input(final.unsqueeze(0))).softmax(1)[0, target_class].item()

        print(f"  -> Restart {restart + 1} final conf={conf:.4f}  (best={best_conf:.4f})")
        if conf > best_conf:
            best_conf = conf
            best_img = final.detach().clone()

    bn_hook.remove()
    return best_img


# ── Real-image bank ───────────────────────────────────────────────────────────
@torch.no_grad()
def select_anchor_index(bank, cls, sample_idx):
    class_p = bank[cls]["class_p"]
    feats = bank[cls]["feats"]
    feat_mean = bank[cls]["feat_mean"].unsqueeze(0)
    candidate_count = min(anchor_pool, class_p.numel())
    top = torch.argsort(class_p, descending=True)[:candidate_count]
    d = torch.cdist(feat_mean, feats[top]).squeeze(0)
    ordered = top[torch.argsort(d)]
    pool = ordered[:min(anchor_topk, ordered.numel())]
    return int(pool[sample_idx % len(pool)])


@torch.no_grad()
def build_anchor_targets(model, bank, cls, sample_idx):
    idx = select_anchor_index(bank, cls, sample_idx)
    anchor_img = bank[cls]["imgs"][idx].to(device)
    pyr, feat = get_feature_pyramid(model, to_model_input(anchor_img.unsqueeze(0)))
    return anchor_img.cpu(), feat.squeeze(0).cpu(), [p.cpu() for p in pyr], bank[cls]["idxs"][idx]


@torch.no_grad()
def build_class_prototype_targets(model, bank, cls, topk=16):
    class_p = bank[cls]["class_p"]
    k = min(topk, class_p.numel())
    top = torch.argsort(class_p, descending=True)[:k]
    imgs = torch.stack(bank[cls]["imgs"])[top].to(device)
    pyr, pen = get_feature_pyramid(model, to_model_input(imgs))
    target_feat = pen.mean(0).cpu()
    target_pyr = [p.mean(0, keepdim=True).cpu() for p in pyr]
    return target_feat, target_pyr


@torch.no_grad()
def build_real_bank(model):
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=T.ToTensor())
    bank = {c: {"imgs": [], "idxs": []} for c in range(num_classes)}

    for idx, (img, lbl) in enumerate(dataset):
        if len(bank[lbl]["imgs"]) < max_real_per_class:
            bank[lbl]["imgs"].append(img)
            bank[lbl]["idxs"].append(idx)

    print("Building feature bank...")
    for c in range(num_classes):
        imgs = torch.stack(bank[c]["imgs"]).to(device)
        feats, class_p = [], []
        for i in range(0, imgs.size(0), 256):
            b = imgs[i:i + 256]
            feats.append(get_features(model, to_model_input(b)).cpu())
            class_p.append(model(to_model_input(b)).softmax(1)[:, c].cpu())
        bank[c]["feats"] = torch.cat(feats, dim=0)
        bank[c]["class_p"] = torch.cat(class_p, dim=0)
        bank[c]["feat_mean"] = bank[c]["feats"].mean(0)
    return bank


# ── Nearest-real search ───────────────────────────────────────────────────────
@torch.no_grad()
def find_nearest(model, recon_01, cls, bank):
    r = recon_01.unsqueeze(0).to(device)
    rf = get_features(model, to_model_input(r)).cpu()
    fd = torch.cdist(rf, bank[cls]["feats"]).squeeze(0)

    k = min(topk_rerank, fd.numel())
    tk = torch.topk(fd, k=k, largest=False).indices

    top_imgs = torch.stack(bank[cls]["imgs"])[tk]
    pix_d = ((top_imgs - recon_01.cpu()) ** 2).mean((1, 2, 3))
    best_loc = int(pix_d.argmin())
    best_g = int(tk[best_loc])

    real_img = bank[cls]["imgs"][best_g]
    real_dsidx = bank[cls]["idxs"][best_g]
    dist = float(fd[best_g])
    return real_img, real_dsidx, dist


# ── Training-image retrieval ──────────────────────────────────────────────────
@torch.no_grad()
def retrieve_train_image(cls, rank, bank):
    order = torch.argsort(bank[cls]["class_p"], descending=True)
    pick = int(order[rank % len(order)])
    return bank[cls]["imgs"][pick], bank[cls]["idxs"][pick], float(bank[cls]["class_p"][pick])


# ── Per-image metrics ─────────────────────────────────────────────────────────
@torch.no_grad()
def compute_metrics(model, recon_01, cls, real_img_01=None):
    prob = model(to_model_input(recon_01.unsqueeze(0).to(device))).softmax(1)
    top1 = prob[0, cls].item()
    ent = -torch.sum(prob * torch.log(prob + 1e-8)).item()
    if real_img_01 is not None:
        a = recon_01.permute(1, 2, 0).cpu().numpy()
        b = real_img_01.permute(1, 2, 0).cpu().numpy()
        try:
            ssim_v = compute_ssim(a, b, channel_axis=2, data_range=1.0)
        except Exception:
            ssim_v = 0.0
    else:
        ssim_v = 0.0
    return top1, ent, ssim_v


@torch.no_grad()
def compute_anchor_group_consistency(model, recon_list):
    """
    recon_list: list[(anchor_pos, img_01)] for same class/sample.
    Computes pairwise similarity among generated images.
    """
    if len(recon_list) < 2:
        return {
            "pairs": 0,
            "mean_pairwise_mse": -1.0,
            "mean_pairwise_ssim": -1.0,
            "mean_pairwise_feat_dist": -1.0,
            "min_pairwise_feat_dist": -1.0,
            "max_pairwise_feat_dist": -1.0,
        }

    imgs = [x[1] for x in recon_list]
    names = [x[0] for x in recon_list]

    batch = torch.stack(imgs).to(device)
    feats = get_features(model, to_model_input(batch)).cpu()

    mses, ssims, fds = [], [], []
    for i, j in combinations(range(len(imgs)), 2):
        a = imgs[i]
        b = imgs[j]

        mse_ij = float(((a - b) ** 2).mean().item())
        mses.append(mse_ij)

        an = a.permute(1, 2, 0).cpu().numpy()
        bn = b.permute(1, 2, 0).cpu().numpy()
        try:
            ssim_ij = float(compute_ssim(an, bn, channel_axis=2, data_range=1.0))
        except Exception:
            ssim_ij = 0.0
        ssims.append(ssim_ij)

        fd_ij = float(torch.norm(feats[i] - feats[j], p=2).item())
        fds.append(fd_ij)

    return {
        "pairs": len(mses),
        "mean_pairwise_mse": float(np.mean(mses)),
        "mean_pairwise_ssim": float(np.mean(ssims)),
        "mean_pairwise_feat_dist": float(np.mean(fds)),
        "min_pairwise_feat_dist": float(np.min(fds)),
        "max_pairwise_feat_dist": float(np.max(fds)),
    }


def classify_leakage(mean_pair_ssim, unique_nearest_count, mean_pair_feat_dist):
    """Heuristic leakage level from anchor-consistency behavior."""
    if (
        mean_pair_ssim >= 0.70
        and unique_nearest_count <= 2
        and mean_pair_feat_dist >= 0
        and mean_pair_feat_dist <= 1.00
    ):
        return "high"
    if (
        mean_pair_ssim >= 0.45
        and unique_nearest_count <= 3
        and mean_pair_feat_dist >= 0
        and mean_pair_feat_dist <= 1.60
    ):
        return "medium"
    return "low"


# ── Visualisation ─────────────────────────────────────────────────────────────
def save_comparison_panel(recon_01, real_01, path, cls, exp_name, anchor_pos=""):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(recon_01.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title(f"Reconstructed ({class_names[cls]})", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(real_01.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Nearest Real Training Image", fontsize=11)
    axes[1].axis("off")
    tag = f" | anchor={anchor_pos}" if anchor_pos else ""
    fig.suptitle(f"{exp_name} | class {cls}: {class_names[cls]}{tag}", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_anchor_preview(anchor_01, frac, mode):
    """Visual hint showing which patch was used as anchor seed."""
    m = make_anchor_mask(anchor_01.shape[1], anchor_01.shape[2], frac=frac, mode=mode, dev=anchor_01.device)
    # Brighten selected patch and dim background for clear visualisation.
    preview = anchor_01 * (0.35 + 0.65 * m.squeeze(0))
    return preview.clamp(0, 1)


def save_anchor_triplet_grid(rows, path, cls, exp_name, img_idx):
    """
    rows: list of dicts with keys anchor_pos, anchor_preview, recon, nearest(optional)
    Saves one panel containing anchor, generated, nearest for each selected anchor position.
    """
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

        if row.get("nearest") is not None:
            ax2.imshow(row["nearest"].permute(1, 2, 0).cpu().numpy())
            ax2.set_title("Nearest Real", fontsize=10)
        else:
            ax2.imshow(np.zeros((32, 32, 3), dtype=np.float32))
            ax2.set_title("Nearest Real (N/A)", fontsize=10)
        ax2.axis("off")

    fig.suptitle(
        f"{exp_name} | class {cls}: {class_names[cls]} | sample {img_idx}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_generated_montage(anchor_batches, path, cls, exp_name, img_idx):
    """Save all generated images for one sample in a single image."""
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
        f"Generated Images | {exp_name} | class {cls}: {class_names[cls]} | sample {img_idx}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(output_dir, exist_ok=True)

_exp_tag = "_".join(experiments.keys())
metric_csv_path = f"recon_metrics_{_exp_tag}.csv"
summary_csv_path = f"recon_summary_{_exp_tag}.csv"
anchor_consistency_csv_path = f"recon_anchor_consistency_{_exp_tag}.csv"

mf = open(metric_csv_path, "w", newline="")
mw = csv.writer(mf)
mw.writerow(
    [
        "Experiment",
        "Heterogeneity",
        "Class",
        "Class_Name",
        "Image",
        "Anchor_Position",
        "Top1_Confidence",
        "Entropy",
        "SSIM_vs_NearestReal",
        "Nearest_Real_Dataset_Index",
        "Feature_Distance",
    ]
)

af = open(anchor_consistency_csv_path, "w", newline="")
aw = csv.writer(af)
aw.writerow(
    [
        "Experiment",
        "Heterogeneity",
        "Class",
        "Class_Name",
        "Image",
        "Anchor_Count",
        "Pair_Count",
        "Mean_Pairwise_MSE",
        "Mean_Pairwise_SSIM",
        "Mean_Pairwise_Feature_Distance",
        "Min_Pairwise_Feature_Distance",
        "Max_Pairwise_Feature_Distance",
        "Unique_Nearest_Real_Idx_Count",
        "Leakage_Level",
    ]
)

summary_rows = []

for exp_name, exp_info in experiments.items():
    print(f"\n{'=' * 65}")
    print(f"  Experiment {exp_name}  ({exp_info['heterogeneity']})")
    print(f"{'=' * 65}")

    model = get_model(exp_info["ckpt"])
    bank = build_real_bank(model) if project_real else None

    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    per_class = {c: {"top1": [], "ent": [], "ssim": [], "dist": []} for c in range(num_classes)}

    for cls in target_classes:
        for img_idx in range(images_per_class):
            print(f"\n-- Class {cls} ({class_names[cls]})  sample {img_idx} --")

            if attack_mode == "train_retrieve":
                recon, nr_idx, nr_dist = retrieve_train_image(cls, img_idx, bank)
                anchor_batches = [("retrieved", recon)]
                print(f"  Retrieved training idx={nr_idx}  score={nr_dist:.4f}")
            else:
                if bank is not None and use_anchor_init:
                    anchor_img, target_feat, target_pyramid, anchor_dsidx = build_anchor_targets(model, bank, cls, img_idx)
                    print(f"  Anchor idx={anchor_dsidx}  frac={anchor_frac:.2f}  positions={anchor_positions}")

                    anchor_batches = []
                    for anchor_pos in anchor_positions:
                        print(f"  -> Reconstructing with patch anchor: {anchor_pos}")
                        recon = reconstruct_image(
                            model,
                            cls,
                            target_feat=target_feat,
                            target_pyramid=target_pyramid,
                            init_img=anchor_img,
                            sample_idx=img_idx,
                            anchor_patch_mode=anchor_pos,
                        )
                        anchor_batches.append((anchor_pos, recon))

                elif bank is not None:
                    anchor_img = None
                    target_feat, target_pyramid = build_class_prototype_targets(model, bank, cls, topk=prototype_topk)
                    print(f"  Random init (no anchor), class prototype guidance (topk={prototype_topk})")
                    recon = reconstruct_image(
                        model,
                        cls,
                        target_feat=target_feat,
                        target_pyramid=target_pyramid,
                        init_img=anchor_img,
                        sample_idx=img_idx,
                        anchor_patch_mode=anchor_mode,
                    )
                    anchor_batches = [("prototype", recon)]
                else:
                    anchor_img, target_feat, target_pyramid = None, None, None
                    print("  Random init (no anchor)")
                    recon = reconstruct_image(
                        model,
                        cls,
                        target_feat=target_feat,
                        target_pyramid=target_pyramid,
                        init_img=anchor_img,
                        sample_idx=img_idx,
                        anchor_patch_mode=anchor_mode,
                    )
                    anchor_batches = [("random", recon)]

            nearest_real_idxs = []
            group_panel_rows = []

            # Save metrics per generated image
            for anchor_pos, recon in anchor_batches:
                stem = f"{exp_name}_class{cls}_img{img_idx}_anchor-{anchor_pos}"
                plt.imsave(os.path.join(exp_dir, f"{stem}.png"), recon.permute(1, 2, 0).cpu().numpy())

                real_img = None
                nr_idx, nr_dist = -1, -1.0

                if project_real and bank is not None:
                    if attack_mode == "train_retrieve":
                        real_img = recon
                    else:
                        real_img, nr_idx, nr_dist = find_nearest(model, recon, cls, bank)
                        if project_final_real:
                            recon = real_img.clone()
                        plt.imsave(
                            os.path.join(exp_dir, f"{stem}_nearest_real.png"),
                            real_img.permute(1, 2, 0).cpu().numpy(),
                        )
                        save_comparison_panel(
                            recon,
                            real_img,
                            os.path.join(exp_dir, f"{stem}_comparison.png"),
                            cls,
                            exp_name,
                            anchor_pos=anchor_pos,
                        )

                if anchor_img is not None and anchor_pos in {"top_left", "top_right", "center", "bottom_left", "bottom_right", "random"}:
                    anchor_preview = make_anchor_preview(anchor_img.to(device), frac=anchor_frac, mode=anchor_pos).cpu()
                elif anchor_img is not None:
                    anchor_preview = anchor_img.cpu()
                else:
                    anchor_preview = recon.cpu()

                group_panel_rows.append(
                    {
                        "anchor_pos": anchor_pos,
                        "anchor_preview": anchor_preview,
                        "recon": recon.cpu(),
                        "nearest": None if real_img is None else real_img.cpu(),
                    }
                )

                top1, ent, ssim_v = compute_metrics(model, recon, cls, real_img)
                mw.writerow([
                    exp_name,
                    exp_info["heterogeneity"],
                    cls,
                    class_names[cls],
                    img_idx,
                    anchor_pos,
                    top1,
                    ent,
                    ssim_v,
                    nr_idx,
                    nr_dist,
                ])
                mf.flush()

                per_class[cls]["top1"].append(top1)
                per_class[cls]["ent"].append(ent)
                per_class[cls]["ssim"].append(ssim_v)
                if nr_dist >= 0:
                    per_class[cls]["dist"].append(nr_dist)
                if nr_idx >= 0:
                    nearest_real_idxs.append(nr_idx)

                print(
                    f"  ✓ anchor={anchor_pos:>12} top1={top1:.4f} "
                    f"entropy={ent:.4f} ssim={ssim_v:.4f} nr_idx={nr_idx} dist={nr_dist:.3f}"
                )

            # Intra-anchor consistency for this class/sample batch
            consistency = compute_anchor_group_consistency(model, anchor_batches)
            aw.writerow([
                exp_name,
                exp_info["heterogeneity"],
                cls,
                class_names[cls],
                img_idx,
                len(anchor_batches),
                consistency["pairs"],
                consistency["mean_pairwise_mse"],
                consistency["mean_pairwise_ssim"],
                consistency["mean_pairwise_feat_dist"],
                consistency["min_pairwise_feat_dist"],
                consistency["max_pairwise_feat_dist"],
                len(set(nearest_real_idxs)),
                classify_leakage(
                    consistency["mean_pairwise_ssim"],
                    len(set(nearest_real_idxs)),
                    consistency["mean_pairwise_feat_dist"],
                ),
            ])
            af.flush()

            print(
                "  [anchor-consistency] "
                f"pairs={consistency['pairs']} "
                f"mean_mse={consistency['mean_pairwise_mse']:.6f} "
                f"mean_ssim={consistency['mean_pairwise_ssim']:.4f} "
                f"mean_feat_dist={consistency['mean_pairwise_feat_dist']:.4f} "
                f"unique_nearest={len(set(nearest_real_idxs))}"
            )

            # Save combined visual summaries requested for each sample.
            group_stem = f"{exp_name}_class{cls}_img{img_idx}"
            save_anchor_triplet_grid(
                group_panel_rows,
                os.path.join(exp_dir, f"{group_stem}_anchors_generated_nearest.png"),
                cls,
                exp_name,
                img_idx,
            )
            save_generated_montage(
                anchor_batches,
                os.path.join(exp_dir, f"{group_stem}_generated_montage.png"),
                cls,
                exp_name,
                img_idx,
            )

    for cls in target_classes:
        s = per_class[cls]
        n = len(s["top1"])
        summary_rows.append(
            {
                "Experiment": exp_name,
                "Heterogeneity": exp_info["heterogeneity"],
                "Class": cls,
                "Class_Name": class_names[cls],
                "N_Samples": n,
                "Mean_Top1": round(float(np.mean(s["top1"])) if n else -1.0, 4),
                "Mean_Entropy": round(float(np.mean(s["ent"])) if n else -1.0, 4),
                "Mean_SSIM": round(float(np.mean(s["ssim"])) if n else -1.0, 4),
                "Mean_Dist": round(float(np.mean(s["dist"])) if s["dist"] else -1.0, 4),
            }
        )

mf.close()
af.close()

if summary_rows:
    summary_rows.sort(key=lambda r: (r["Experiment"], -r["Mean_Top1"]))
    with open(summary_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

print("\n✅  Done.")
print(f"    Images             -> {output_dir}/")
print(f"    Metrics            -> {metric_csv_path}")
print(f"    Summary            -> {summary_csv_path}")
print(f"    Anchor consistency -> {anchor_consistency_csv_path}")
