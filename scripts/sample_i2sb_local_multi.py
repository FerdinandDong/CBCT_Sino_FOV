# scripts/sample_i2sb_local_multi.py
# -*- coding: utf-8 -*-
"""
I2SB local multi-step sampler script (aligned to configs/sample_i2sb_local_multi.yaml)

Key features (cfg-aligned):
- Uses cfg.sample.split to pick loader from make_dataloader() dict
- Loads cfg.sample.ckpt (supports new trainer ckpt dict with "state_dict", and legacy state_dict)
- Supports two inference modes:
    (A) Deterministic multi-step "shared-eps" update
    (B) Bridge sampler consistent with trainer._i2sb_sample (supports stochastic + clamp_known)
  Controlled by cfg.sample.val_infer / cfg.sample.sample_* (optional). If not provided, defaults to (A).
- Supports pick_id / pick_angle / "mid" selection
- Saves volume tif/npy, metrics.csv, and (optional) quad + tiles
- Computes PSNR/SSIM on missing-only region by default (cfg.sample.metric_missing_only)

NEW (minimal-intrusion):
- Optional snapshots (t-trajectory) export under out_dir/<snapshots.dir>/... for ONLY the selected (pick_id + pick_angle) case.
  Configure:
    sample:
      snapshots:
        enable: true
        dir: snapshots
        ts: [0.6667, 0.5, 0.3333]
        include_input: true
        include_output: true
        include_gt: true
        save_strip: true
        save_individual: true

NEW (minimal-intrusion):
- soft_clamp_valid (cfg.sample.soft_clamp_valid):
  When sample_clamp_known=false, softly blends x towards x1 in valid region each step:
    x <- x + a * mask * (x1 - x),  a in (0,1]
  This helps stabilize sampling without hard clamping.

FIX (minimal-intrusion, for legacy recon3d pipeline compatibility):
- Always export noisy_volume.npy and gt_volume.npy (in addition to *.tiff)
- Export *_volume_raw.(npy/tiff) when norm stats exist
- pred_raw restore no longer requires gt_i (only needs gt_lo/gt_hi)
- all-mode also exports pred_volume.(npy/tiff) + (optional) noisy/gt + metrics.csv
"""

import os
import re
import csv
import math
import argparse
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import tifffile as tiff
from tqdm import tqdm

# run: python -m scripts.sample_i2sb_local_multi
from ctprojfix.data.dataset import make_dataloader
from ctprojfix.models.i2sb_unet import I2SBUNet


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_cfg(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml_safe_load(f)

def yaml_safe_load(f):
    import yaml
    return yaml.safe_load(f)

def _ceil_to(v, m):
    return ((v + m - 1) // m) * m if m > 0 else v

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]

def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def _percentile_lohi(a: np.ndarray, p_lo=1.0, p_hi=99.0, eps=1e-6) -> Tuple[float, float]:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.percentile(a, [p_lo, p_hi])
    lo, hi = float(lo), float(hi)
    if hi - lo < eps:
        lo = float(a.min())
        hi = float(a.max() + eps)
    return lo, hi

def _norm01_with_lohi(a: np.ndarray, lo: float, hi: float, eps=1e-6) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    y = (a - lo) / (hi - lo + eps)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

def _center_crop_to(arr: Optional[np.ndarray], out_h: int, out_w: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = arr
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    H, W = a.shape[-2], a.shape[-1]
    if (H, W) == (out_h, out_w):
        return a.astype(np.float32, copy=False)
    top  = max((H - out_h) // 2, 0)
    left = max((W - out_w) // 2, 0)
    return a[top:top+out_h, left:left+out_w].astype(np.float32, copy=False)

def _match_to_pred_size(pred_hw: Tuple[int, int],
                        noisy: Optional[np.ndarray],
                        gt: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    ph, pw = int(pred_hw[0]), int(pred_hw[1])
    return _center_crop_to(noisy, ph, pw), _center_crop_to(gt, ph, pw)

def _save_tile_gray01(img01: np.ndarray, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img01, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def _save_tile_heat(err: np.ndarray, out_path: str, cmap: str, with_cb: bool):
    _ensure_dir(os.path.dirname(out_path))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if with_cb:
        fig = plt.figure(figsize=(4.4, 4), dpi=200)
        ax = fig.add_axes([0.0, 0.0, 0.9, 1.0])
        im = ax.imshow(err, cmap=cmap, vmin=0.0, vmax=None)
        ax.axis("off")
        cax = fig.add_axes([0.91, 0.1, 0.03, 0.8])
        plt.colorbar(im, cax=cax)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(4, 4), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(err, cmap=cmap, vmin=0.0, vmax=None)
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

def save_quad_and_tiles(noisy01: np.ndarray,
                        pred01: np.ndarray,
                        gt01: Optional[np.ndarray],
                        out_dir: str,
                        name: str,
                        tiles_cfg: dict,
                        use_gt_percentile_norm: bool = True):
    """
    Writes:
      outputs/.../figs/{name}.png
      outputs/.../figs/{name}_tiles/*
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(out_dir)
    fig_dir = os.path.join(out_dir, "figs")
    _ensure_dir(fig_dir)

    n0 = np.asarray(noisy01, dtype=np.float32)
    p0 = np.asarray(pred01, dtype=np.float32)
    g0 = np.asarray(gt01, dtype=np.float32) if gt01 is not None else None

    if g0 is not None and use_gt_percentile_norm:
        lo, hi = _percentile_lohi(g0, 1.0, 99.0)
        n = _norm01_with_lohi(n0, lo, hi)
        p = _norm01_with_lohi(p0, lo, hi)
        g = _norm01_with_lohi(g0, lo, hi)
    else:
        n = np.clip(n0, 0, 1)
        p = np.clip(p0, 0, 1)
        g = np.clip(g0, 0, 1) if g0 is not None else None

    if g is not None:
        err = np.abs(p - g).astype(np.float32)
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        axes[0].imshow(n, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Noisy"); axes[0].axis("off")
        axes[1].imshow(p, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Pred"); axes[1].axis("off")
        axes[2].imshow(g, cmap="gray", vmin=0, vmax=1); axes[2].set_title("GT"); axes[2].axis("off")
        im = axes[3].imshow(err, cmap=str((tiles_cfg or {}).get("cmap","magma")), vmin=0.0, vmax=None)
        axes[3].set_title("|Pred-GT|"); axes[3].axis("off")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        plt.tight_layout()
        out_path = os.path.join(fig_dir, f"{name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[FIG] {out_path}")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(n, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Noisy"); axes[0].axis("off")
        axes[1].imshow(p, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Pred"); axes[1].axis("off")
        plt.tight_layout()
        out_path = os.path.join(fig_dir, f"{name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[FIG] {out_path}")
        return

    if not bool((tiles_cfg or {}).get("enable", False)):
        return

    tiles_dir = (tiles_cfg or {}).get("dir", None)
    if not tiles_dir:
        tiles_dir = os.path.join(fig_dir, f"{name}_tiles")
    _ensure_dir(tiles_dir)

    prefix = str((tiles_cfg or {}).get("prefix", ""))
    fmt = str((tiles_cfg or {}).get("format", "png"))
    cmap = str((tiles_cfg or {}).get("cmap", "magma"))
    with_cb = bool((tiles_cfg or {}).get("heat_with_colorbar", False))

    _save_tile_gray01(n, os.path.join(tiles_dir, f"{prefix}Noisy.{fmt}"))
    _save_tile_gray01(p, os.path.join(tiles_dir, f"{prefix}Pred.{fmt}"))
    _save_tile_gray01(g, os.path.join(tiles_dir, f"{prefix}GT.{fmt}"))
    _save_tile_heat(np.abs(p-g), os.path.join(tiles_dir, f"{prefix}AbsDiff_Pred_vs_GT.{fmt}"),
                    cmap=cmap, with_cb=with_cb)
    print(f"[TILE] -> {tiles_dir}")


# ---------------- NEW: snapshot export ----------------

def save_snapshots(out_dir: str,
                   subdir: str,
                   base_name: str,
                   imgs01: List[np.ndarray],
                   tags: List[str],
                   gt01: Optional[np.ndarray] = None,
                   use_gt_percentile_norm: bool = True,
                   save_strip: bool = True,
                   save_individual: bool = True):
    """
    Save snapshot images into: out_dir/subdir/base_name/*
      - strip: .../strip.png
      - individual: .../{tag}.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.join(out_dir, str(subdir))
    case_dir = os.path.join(root, str(base_name))
    _ensure_dir(case_dir)

    imgs = [np.asarray(x, dtype=np.float32) for x in imgs01]
    g0 = np.asarray(gt01, dtype=np.float32) if gt01 is not None else None

    if (g0 is not None) and use_gt_percentile_norm:
        lo, hi = _percentile_lohi(g0, 1.0, 99.0)
        imgs = [_norm01_with_lohi(im, lo, hi) for im in imgs]
    else:
        imgs = [np.clip(im, 0.0, 1.0) for im in imgs]

    if save_individual:
        for im, tag in zip(imgs, tags):
            p = os.path.join(case_dir, f"{tag}.png")
            fig = plt.figure(figsize=(4, 4), dpi=200)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(im, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            fig.savefig(p, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

    if save_strip:
        n = len(imgs)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        if n == 1:
            axes = [axes]
        for ax, im, tag in zip(axes, imgs, tags):
            ax.imshow(im, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(tag)
            ax.axis("off")
        plt.tight_layout()
        p = os.path.join(case_dir, "strip.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[SNAP] saved -> {case_dir}")

# ------------------------------------------------------

def _psnr01(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    mse = float(np.mean((x - y) ** 2))
    if mse < eps:
        return 99.0
    return 20.0 * float(np.log10(1.0 / np.sqrt(mse)))

def _ssim2d(x: np.ndarray, y: np.ndarray, dr: float = 1.0,
            K1: float = 0.01, K2: float = 0.03, sigma: float = 1.5) -> float:
    from scipy.ndimage import gaussian_filter
    x = np.clip(x, 0, dr).astype(np.float32)
    y = np.clip(y, 0, dr).astype(np.float32)
    C1, C2 = (K1 * dr) ** 2, (K2 * dr) ** 2
    mu1, mu2 = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = gaussian_filter(x * x, sigma) - mu1s
    s2 = gaussian_filter(y * y, sigma) - mu2s
    s12 = gaussian_filter(x * y, sigma) - mu12
    num = (2 * mu12 + C1) * (2 * s12 + C2)
    den = (mu1s + mu2s + C1) * (s1 + s2 + C2) + 1e-12
    return float(np.mean(num / den))

def _get_scalar_from_batch(batch: Dict[str, Any], key: str, idx: int, default=None):
    if key not in batch:
        return default
    v = batch[key]
    if isinstance(v, torch.Tensor):
        if v.ndim == 0:
            return v.item()
        if v.ndim >= 1 and v.shape[0] > idx:
            return v[idx].item()
        return default
    if isinstance(v, (list, tuple)):
        try:
            return v[idx]
        except Exception:
            return v[0] if len(v) > 0 else default
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return float(v)
        if v.ndim >= 1 and v.shape[0] > idx:
            return v[idx]
        return default
    return v

def _unet_factor_from_model(net: torch.nn.Module) -> int:
    core = net.module if hasattr(net, "module") else net
    n_down = len(getattr(core, "downs", []))
    return 2 ** max(int(n_down), 0)

def _save_stack_both(out_dir: str, stem: str, vol: np.ndarray, imagej: bool = True):
    """Save <stem>.tiff and <stem>.npy"""
    vol = np.asarray(vol, dtype=np.float32)
    tiff.imwrite(os.path.join(out_dir, f"{stem}.tiff"), vol, imagej=imagej)
    np.save(os.path.join(out_dir, f"{stem}.npy"), vol)

def _try_save_raw_stack(out_dir: str, stem_raw: str, frames_raw: List[Optional[np.ndarray]]):
    """
    Save raw stack only if all frames are available (not None).
    """
    if len(frames_raw) == 0:
        return False
    ok = all(x is not None for x in frames_raw)
    if not ok:
        print(f"[WARN] skip saving {stem_raw}: missing raw frames (None exists).")
        return False
    vol = np.stack([np.asarray(x, dtype=np.float32) for x in frames_raw], axis=0).astype(np.float32)
    tiff.imwrite(os.path.join(out_dir, f"{stem_raw}.tiff"), vol, imagej=True)
    np.save(os.path.join(out_dir, f"{stem_raw}.npy"), vol)
    return True


# -----------------------------------------------------------------------------
# Samplers
# -----------------------------------------------------------------------------

def _pad_inputs_for_unet(x1_img_m1: torch.Tensor,
                         mask01: torch.Tensor,
                         angle01: Optional[torch.Tensor],
                         factor: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, int]:
    """
    Pad to multiples of factor.
    - image: reflect
    - mask:  constant 0
    - angle: replicate (if exists)
    """
    B, _, H, W = x1_img_m1.shape
    Ht = _ceil_to(H, factor)
    Wt = _ceil_to(W, factor)
    if (Ht, Wt) == (H, W):
        return x1_img_m1, mask01, angle01, Ht, Wt

    pad = (0, Wt - W, 0, Ht - H)
    x1_img_m1 = F.pad(x1_img_m1, pad, mode="reflect")
    mask01 = F.pad(mask01, pad, mode="constant", value=0.0)
    if angle01 is not None:
        angle01 = F.pad(angle01, pad, mode="replicate")
    return x1_img_m1, mask01, angle01, Ht, Wt


@torch.no_grad()
def sample_multi_step_shared_eps(
    net: torch.nn.Module,
    x1_img_m1: torch.Tensor,    # (B,1,H,W) in [-1,1]
    mask01: torch.Tensor,       # (B,1,H,W) in [0,1]
    angle01: Optional[torch.Tensor],  # (B,1,H,W) in [0,1] or None
    steps: int,
    sigma_T: float,
) -> torch.Tensor:
    """
    Deterministic multi-step sampling with shared epsilon estimate:
      x_t = (1-t)x0 + t*x1 + sigma_T*sqrt(t(1-t))*eps
    """
    device = x1_img_m1.device
    B, _, H, W = x1_img_m1.shape

    factor = _unet_factor_from_model(net)
    x1p, mp, ap, Ht, Wt = _pad_inputs_for_unet(x1_img_m1, mask01, angle01, factor)

    xt = x1p.clone()
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)

    eps = 1e-6
    for k in range(steps):
        t = float(ts[k].item())
        s = float(ts[k + 1].item())

        t_map = torch.full((B, 1, Ht, Wt), t, device=device, dtype=torch.float32)

        if ap is None:
            net_in = torch.cat([xt, x1p, mp, t_map], dim=1)  # 4ch
        else:
            net_in = torch.cat([xt, x1p, mp, ap, t_map], dim=1)  # 5ch

        x0_hat = net(net_in)
        if x0_hat.shape[-2:] != (Ht, Wt):
            x0_hat = F.interpolate(x0_hat, size=(Ht, Wt), mode="bilinear", align_corners=False)

        sig_t = sigma_T * math.sqrt(max(t * (1.0 - t), 0.0))
        sig_s = sigma_T * math.sqrt(max(s * (1.0 - s), 0.0))

        if sig_t < 1e-8:
            eps_hat = torch.zeros_like(xt)
        else:
            eps_hat = (xt - (1.0 - t) * x0_hat - t * x1p) / (sig_t + eps)

        xt = (1.0 - s) * x0_hat + s * x1p + sig_s * eps_hat

    xt = xt[..., :H, :W]
    return xt.clamp(-1.0, 1.0)


@torch.no_grad()
def sample_bridge_conditional(
    net: torch.nn.Module,
    x1_img01: torch.Tensor,        # (B,1,H,W) in [0,1]
    conds01: torch.Tensor,         # (B,Cc,H,W) in [0,1]  (mask(+angle))
    steps: int,
    sigma_T: float,
    stochastic: bool,
    clamp_known: bool,
    cond_has_angle: bool,
    soft_clamp_valid: float = 0.0,          # NEW
    save_ts: Optional[List[float]] = None,  # NEW
    return_states: bool = False,            # NEW
):
    """
    Bridge sampler consistent with trainer._i2sb_sample().

    NEW:
      - soft_clamp_valid: when clamp_known=false, softly clamps valid region:
          x <- x + a * mask * (x1 - x) , a in (0,1]
      - save_ts: list of t in [0,1] to snapshot (nearest grid point on linspace(1,0,steps+1))
      - return_states: if True return (x_T, saved_dict) where saved_dict maps requested tt -> x_tt
    """
    device = x1_img01.device
    B, _, H, W = x1_img01.shape

    x1m = x1_img01.float() * 2.0 - 1.0
    conds = conds01.float()

    factor = _unet_factor_from_model(net)
    Ht = _ceil_to(H, factor)
    Wt = _ceil_to(W, factor)
    if (Ht, Wt) != (H, W):
        pad = (0, Wt - W, 0, Ht - H)
        x1m = F.pad(x1m, pad, mode="reflect")
        mask = conds[:, 0:1, ...]
        mask = F.pad(mask, pad, mode="constant", value=0.0)
        if cond_has_angle and conds.shape[1] > 1:
            ang = conds[:, 1:, ...]
            ang = F.pad(ang, pad, mode="replicate")
            conds = torch.cat([mask, ang], dim=1)
        else:
            conds = mask

    mask = conds[:, 0:1, ...].clamp(0.0, 1.0)
    eps = 1e-6
    sigma2 = float(sigma_T) ** 2

    a_soft = float(soft_clamp_valid) if soft_clamp_valid is not None else 0.0
    a_soft = max(0.0, min(1.0, a_soft))

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    x = x1m  # init at t=1

    saved = {}
    save_idx = {}
    if save_ts:
        ts_cpu = ts.detach().cpu()
        for tt in save_ts:
            tt = float(tt)
            idx = int(torch.argmin(torch.abs(ts_cpu - tt)).item())
            save_idx[idx] = tt
        if 0 in save_idx:
            saved[save_idx[0]] = x.detach().clone()

    for k in range(steps):
        t = float(ts[k].item())
        s = float(ts[k + 1].item())

        t_map = x.new_full((B, 1, Ht, Wt), fill_value=t)
        net_in = torch.cat([x, x1m, conds, t_map], dim=1)
        x0_hat = net(net_in)

        if s <= 0.0:
            x = x0_hat
        else:
            mt = (1.0 - t) * x0_hat + t * x1m
            ms = (1.0 - s) * x0_hat + s * x1m
            r = x - mt

            if t >= 1.0 - 1e-8:
                mean = ms
                var_t = x.new_tensor(sigma2) * (x.new_tensor(s) * (1.0 - x.new_tensor(s)))
            else:
                t_t = x.new_tensor(t).clamp_min(eps)
                s_t = x.new_tensor(s).clamp_min(0.0)
                one_minus_t = (1.0 - t_t).clamp_min(eps)

                mean = ms + (s_t / t_t) * r
                var_t = x.new_tensor(sigma2) * (
                    s_t * (1.0 - s_t)
                    - (s_t * (1.0 - t_t)) ** 2 / (t_t * one_minus_t + eps)
                )

            if stochastic:
                std = torch.sqrt(torch.clamp(var_t, min=0.0))
                x = mean + std * torch.randn_like(x)
            else:
                x = mean

        # soft clamp (only when hard clamp is disabled)
        if (not clamp_known) and (a_soft > 0.0):
            x = x + a_soft * mask * (x1m - x)

        # hard clamp
        if clamp_known:
            x = mask * x1m + (1.0 - mask) * x

        if save_ts and ((k + 1) in save_idx):
            saved[save_idx[k + 1]] = x.detach().clone()

    if (Ht, Wt) != (H, W):
        x = x[..., :H, :W]
        if save_ts:
            saved = {tt: vv[..., :H, :W] for tt, vv in saved.items()}

    x = x.clamp(-1.0, 1.0)
    if return_states:
        return x, saved
    return x


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@torch.no_grad()
def run(cfg_path: str, ckpt: Optional[str] = None, out: Optional[str] = None):
    import random
    cfg = load_cfg(cfg_path)

    seed = int((cfg.get("sample", {}) or {}).get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_cfg = cfg.get("train", {})
    dev_str = str(train_cfg.get("device", "cuda")).strip().lower()
    if dev_str.startswith("cuda") and not torch.cuda.is_available():
        dev = torch.device("cpu")
    else:
        dev = torch.device(dev_str if dev_str else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[DEVICE] {dev}")

    # 1) DataLoader (aligned to sample cfg)
    data_cfg = dict(cfg.get("data", {}))
    data_cfg["shuffle"] = False
    data_cfg["return_norm_stats"] = True  # FIX: ensure *_lo/hi are returned for raw export
    dl = make_dataloader(data_cfg)

    sample_cfg = cfg.get("sample", {}) or {}
    want = str(sample_cfg.get("split", "") or "").lower().strip()

    if isinstance(dl, dict):
        if not want:
            raise RuntimeError("[sample_i2sb_local_multi] make_dataloader returned split dict, but sample.split not set.")
        if want not in dl or dl[want] is None:
            raise RuntimeError(f"[sample_i2sb_local_multi] split '{want}' not found in dataloader dict.")
        loader = dl[want]
        print(f"[DATA] split='{want}' size={len(loader.dataset)}")
    else:
        loader = dl
        print(f"[DATA] single loader size={len(loader.dataset)}")
        if len(loader.dataset) <= 0:
            raise RuntimeError("[DATA] DataLoader is empty.")

    # 2) Model & Checkpoint
    model_cfg = cfg.get("model", {}) or {}
    mp = (model_cfg.get("params", {}) or {})

    net = I2SBUNet(
        in_ch=int(mp.get("in_ch", 5)),
        base=int(mp.get("base", 64)),
        depth=int(mp.get("depth", 4)),
        emb_dim=int(mp.get("emb_dim", 256)),
        dropout=float(mp.get("dropout", 0.0)),
    ).to(dev)

    ckpt_path = ckpt or sample_cfg.get("ckpt", None)
    if ckpt_path is None:
        ckpt_dir = train_cfg.get("ckpt_dir", "checkpoints/i2sb_local_multi")
        ckpt_prefix = train_cfg.get("ckpt_prefix", "i2sb_local_multi")
        ckpt_path = os.path.join(ckpt_dir, f"{ckpt_prefix}_best.pth")

    if ckpt_path and os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        state = ck.get("state_dict", ck)
        net.load_state_dict(state, strict=True)
        print(f"[CKPT] loaded: {ckpt_path}")
    else:
        print(f"[WARN] ckpt not found: {ckpt_path} (net output may be random).")
    net.eval()

    # 3) Output & sampling params
    out_dir = out or str(sample_cfg.get("out_dir", "outputs/i2sb_local_multi")).strip()
    _ensure_dir(out_dir)

    pick_id = sample_cfg.get("pick_id", None)
    pick_angle = sample_cfg.get("pick_angle", None)

    steps = int(sample_cfg.get("steps", 10))
    sigma_T = float(sample_cfg.get("sigma_T", train_cfg.get("sigma_T", 1.0)))

    save_quad = bool(sample_cfg.get("save_quad", True))
    tiles_cfg = sample_cfg.get("tiles", {}) or {}

    blend_valid = bool(sample_cfg.get("blend_valid", False))
    metric_missing_only = bool(sample_cfg.get("metric_missing_only", True))

    val_infer = str(sample_cfg.get("val_infer", "shared_eps")).lower().strip()
    sample_stochastic = bool(sample_cfg.get("sample_stochastic", False))
    sample_clamp_known = bool(sample_cfg.get("sample_clamp_known", True))
    soft_clamp_valid = float(sample_cfg.get("soft_clamp_valid", 0.0) or 0.0)

    add_angle_channel = bool((cfg.get("data", {}) or {}).get("add_angle_channel", True))
    cond_has_angle = add_angle_channel

    snap_cfg = sample_cfg.get("snapshots", {}) or {}
    snap_enable = bool(snap_cfg.get("enable", False))
    snap_dir = str(snap_cfg.get("dir", "snapshots"))
    snap_ts = snap_cfg.get("ts", None) or [2.0/3.0, 1.0/3.0]
    snap_ts = [float(x) for x in snap_ts]
    snap_inc_in = bool(snap_cfg.get("include_input", True))
    snap_inc_out = bool(snap_cfg.get("include_output", True))
    snap_inc_gt = bool(snap_cfg.get("include_gt", True))
    snap_save_strip = bool(snap_cfg.get("save_strip", True))
    snap_save_ind = bool(snap_cfg.get("save_individual", True))

    print(f"[RUN] split={want or '(single)'} out={out_dir}")
    print(f"[RUN] steps={steps} sigma_T={sigma_T} val_infer={val_infer} "
          f"(stochastic={sample_stochastic}, clamp_known={sample_clamp_known}, soft_clamp_valid={soft_clamp_valid}) "
          f"blend_valid={blend_valid} metric_missing_only={metric_missing_only}")
    if snap_enable:
        print(f"[SNAP CFG] dir={snap_dir} ts={snap_ts} include_in={snap_inc_in} include_out={snap_inc_out} include_gt={snap_inc_gt} "
              f"save_strip={snap_save_strip} save_individual={snap_save_ind}")

    # 4) Inference helpers
    def infer_batch(inp: torch.Tensor,
                    want_states: bool = False,
                    snap_mid_ts: Optional[List[float]] = None):
        if add_angle_channel:
            if inp.shape[1] < 3:
                raise RuntimeError("cfg.data.add_angle_channel=true but batch['inp'] has <3 channels.")
            x1_img01 = inp[:, 0:1, ...]
            mask01 = inp[:, 1:2, ...].clamp(0.0, 1.0)
            angle01 = inp[:, 2:3, ...].clamp(0.0, 1.0)
        else:
            if inp.shape[1] < 2:
                raise RuntimeError("cfg.data.add_angle_channel=false but batch['inp'] has <2 channels.")
            x1_img01 = inp[:, 0:1, ...]
            mask01 = inp[:, 1:2, ...].clamp(0.0, 1.0)
            angle01 = None

        states01 = None

        if val_infer in ("sample", "bridge"):
            # --- Bridge sampler (trainer-aligned), deterministic when sample_stochastic=False ---
            conds = mask01 if angle01 is None else torch.cat([mask01, angle01], dim=1)

            if want_states:
                mid_ts = [float(x) for x in (snap_mid_ts or [])]
                want_ts = [1.0] + mid_ts + [0.0]
                x0_hat_m1, saved = sample_bridge_conditional(
                    net=net,
                    x1_img01=x1_img01.to(dev),
                    conds01=conds.to(dev),
                    steps=steps,
                    sigma_T=sigma_T,
                    stochastic=sample_stochastic,
                    clamp_known=sample_clamp_known,
                    cond_has_angle=cond_has_angle,
                    soft_clamp_valid=soft_clamp_valid,
                    save_ts=want_ts,
                    return_states=True,
                )
                states01 = {float(tt): ((vv.clamp(-1, 1) + 1.0) * 0.5) for tt, vv in saved.items()}
            else:
                x0_hat_m1 = sample_bridge_conditional(
                    net=net,
                    x1_img01=x1_img01.to(dev),
                    conds01=conds.to(dev),
                    steps=steps,
                    sigma_T=sigma_T,
                    stochastic=sample_stochastic,
                    clamp_known=sample_clamp_known,
                    cond_has_angle=cond_has_angle,
                    soft_clamp_valid=soft_clamp_valid,
                )

        elif val_infer in ("shared_eps", "eps"):
            # --- Deterministic shared-eps sampler ---
            x1_img_m1 = x1_img01.to(dev) * 2.0 - 1.0
            x0_hat_m1 = sample_multi_step_shared_eps(
                net=net,
                x1_img_m1=x1_img_m1,
                mask01=mask01.to(dev),
                angle01=None if angle01 is None else angle01.to(dev),
                steps=steps,
                sigma_T=sigma_T,
            )
        else:
            raise RuntimeError(f"[sample_i2sb_local_multi] Unknown val_infer='{val_infer}'. Use 'sample/bridge' or 'shared_eps'.")

        pred01 = ((x0_hat_m1.clamp(-1, 1) + 1.0) * 0.5)
        if blend_valid:
            pred01 = pred01 * (1.0 - mask01.to(pred01.device)) + x1_img01.to(pred01.device) * mask01.to(pred01.device)

        if want_states:
            return pred01, x1_img01.to(pred01.device), mask01.to(pred01.device), states01
        return pred01, x1_img01.to(pred01.device), mask01.to(pred01.device)

    # 5) Run: pick_id or all
    if pick_id is not None:
        pick_id = int(pick_id)
        print(f"[MODE] pick_id={pick_id}")

        preds, gts, noisies = [], [], []
        preds_raw, gts_raw, noisies_raw = [], [], []
        id_rows, angle_rows, A_rows = [], [], []
        noisy_lo_rows, noisy_hi_rows = [], []
        gt_lo_rows, gt_hi_rows = [], []
        psnr_rows, ssim_rows = [], []

        best_one = None  # (diff, rec)

        pbar = tqdm(loader, desc=f"ID={pick_id}", ncols=100)
        for batch in pbar:
            bid = batch.get("id", None)
            if torch.is_tensor(bid):
                ids = bid.detach().cpu().numpy().tolist()
            elif isinstance(bid, (list, tuple, np.ndarray)):
                ids = list(bid)
            else:
                ids = [bid]

            hit = [i for i, v in enumerate(ids) if v is not None and int(v) == pick_id]
            if not hit:
                continue

            inp = batch["inp"].to(dev).float()
            pred01, x1_img01, mask01 = infer_batch(inp)

            pred_np = pred01.detach().cpu().numpy()
            noisy_np = x1_img01.detach().cpu().numpy()
            mask_np = mask01.detach().cpu().numpy()

            has_gt = ("gt" in batch) and (batch["gt"] is not None)
            gt_np = batch["gt"].detach().cpu().numpy() if has_gt else None

            angles = batch.get("angle", [None] * inp.shape[0])
            if torch.is_tensor(angles):
                angles = angles.detach().cpu().numpy().tolist()

            A_arr = batch.get("A", [None] * inp.shape[0])
            if torch.is_tensor(A_arr):
                A_arr = A_arr.detach().cpu().numpy().tolist()

            for i in hit:
                pred_i = np.squeeze(pred_np[i, 0]).astype(np.float32)
                noz_i = np.squeeze(noisy_np[i, 0]).astype(np.float32)
                m_i = np.squeeze(mask_np[i, 0]).astype(np.float32)
                gt_i = np.squeeze(gt_np[i, 0]).astype(np.float32) if has_gt else None

                preds.append(pred_i)
                noisies.append(noz_i)
                if has_gt:
                    gts.append(gt_i)

                ai = angles[i] if angles[i] is not None else -1
                Ai = A_arr[i] if A_arr[i] is not None else 360
                angle_rows.append(ai)
                A_rows.append(Ai)
                id_rows.append(pick_id)

                nlo = _get_scalar_from_batch(batch, "noisy_lo", i)
                nhi = _get_scalar_from_batch(batch, "noisy_hi", i)
                glo = _get_scalar_from_batch(batch, "gt_lo", i)
                ghi = _get_scalar_from_batch(batch, "gt_hi", i)
                noisy_lo_rows.append(nlo); noisy_hi_rows.append(nhi)
                gt_lo_rows.append(glo); gt_hi_rows.append(ghi)

                # ---------------- FIX: raw restore ----------------
                # pred_raw: only needs gt_lo/gt_hi (DO NOT require gt_i)
                if (glo is not None) and (ghi is not None):
                    pr_raw = pred_i * (float(ghi) - float(glo)) + float(glo)
                else:
                    pr_raw = None

                # gt_raw: needs gt + gt_lo/gt_hi
                if (glo is not None) and (ghi is not None) and (gt_i is not None):
                    g_raw = gt_i * (float(ghi) - float(glo)) + float(glo)
                else:
                    g_raw = None

                # noisy_raw: needs noisy_lo/noisy_hi
                if (nlo is not None) and (nhi is not None):
                    n_raw = noz_i * (float(nhi) - float(nlo)) + float(nlo)
                else:
                    n_raw = None

                preds_raw.append(pr_raw)
                gts_raw.append(g_raw)
                noisies_raw.append(n_raw)

                # metrics (optionally missing-only)
                if has_gt and gt_i is not None:
                    if metric_missing_only:
                        miss = (1.0 - m_i).astype(np.float32)
                        if miss.mean() > 1e-6:
                            p_use = pred_i * miss
                            g_use = gt_i * miss
                        else:
                            p_use, g_use = pred_i, gt_i
                    else:
                        p_use, g_use = pred_i, gt_i

                    ps = _psnr01(p_use, g_use)
                    ss = _ssim2d(p_use, g_use)
                else:
                    ps, ss = None, None

                psnr_rows.append(ps)
                ssim_rows.append(ss)

                # pick_angle selection (cache inp for snapshots)
                if pick_angle is not None:
                    if isinstance(pick_angle, str) and pick_angle.lower().strip() == "mid":
                        pass
                    else:
                        want_a = int(pick_angle)
                        diff = abs(int(ai) - want_a) if ai >= 0 else 9999
                        if best_one is None or diff < best_one[0]:
                            best_one = (diff, dict(
                                pred=pred_i, noisy=noz_i, gt=gt_i, mask=m_i, angle=ai,
                                inp_cpu=inp[i].detach().cpu(),
                            ))

        if len(preds) == 0:
            raise RuntimeError(f"[ERROR] pick_id={pick_id} not found in split '{want}'.")

        # sort by angle
        order = list(range(len(preds)))
        order.sort(key=lambda k: (angle_rows[k] < 0, angle_rows[k]))

        def reorder(lst):
            return [lst[k] for k in order]

        preds = reorder(preds)
        noisies = reorder(noisies)
        if gts:
            gts = reorder(gts)
        preds_raw = reorder(preds_raw)
        gts_raw = reorder(gts_raw)
        noisies_raw = reorder(noisies_raw)
        id_rows = reorder(id_rows)
        angle_rows = reorder(angle_rows)
        A_rows = reorder(A_rows)
        noisy_lo_rows = reorder(noisy_lo_rows)
        noisy_hi_rows = reorder(noisy_hi_rows)
        gt_lo_rows = reorder(gt_lo_rows)
        gt_hi_rows = reorder(gt_hi_rows)
        psnr_rows = reorder(psnr_rows)
        ssim_rows = reorder(ssim_rows)

        # ---------------- FIX: save volumes (both tiff + npy) ----------------
        vol_pred = np.stack(preds, axis=0).astype(np.float32)
        _save_stack_both(out_dir, "pred_volume", vol_pred)

        vol_nozy = np.stack(noisies, axis=0).astype(np.float32)
        _save_stack_both(out_dir, "noisy_volume", vol_nozy)

        if gts:
            vol_gt = np.stack(gts, axis=0).astype(np.float32)
            _save_stack_both(out_dir, "gt_volume", vol_gt)

        # raw volumes (only if complete)
        _try_save_raw_stack(out_dir, "pred_volume_raw", preds_raw)
        _try_save_raw_stack(out_dir, "noisy_volume_raw", noisies_raw)
        _try_save_raw_stack(out_dir, "gt_volume_raw", gts_raw)

        # metrics.csv
        csv_path = os.path.join(out_dir, "metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "id", "angle", "A", "nlo", "nhi", "glo", "ghi", "PSNR", "SSIM"])
            for idx, val in enumerate(zip(id_rows, angle_rows, A_rows,
                                          noisy_lo_rows, noisy_hi_rows,
                                          gt_lo_rows, gt_hi_rows,
                                          psnr_rows, ssim_rows)):
                w.writerow([idx] + [x if x is not None else "" for x in val])
        print(f"[DONE] Metrics -> {csv_path}")

        ps_valid = [x for x in psnr_rows if x is not None]
        ss_valid = [x for x in ssim_rows if x is not None]
        if ps_valid:
            print(f"[RESULT] ID={pick_id} | steps={steps} | PSNR={np.mean(ps_valid):.3f} | SSIM={np.mean(ss_valid):.4f}")

        # quad + tiles (pick angle)
        if save_quad and pick_angle is not None:
            if isinstance(pick_angle, str) and pick_angle.lower().strip() == "mid":
                mid = len(preds) // 2
                rec = dict(pred=preds[mid], noisy=noisies[mid], gt=gts[mid] if gts else None, angle=angle_rows[mid])
                inp_cpu_for_snap = None
            else:
                rec = best_one[1] if best_one is not None else None
                if rec is None:
                    mid = len(preds) // 2
                    rec = dict(pred=preds[mid], noisy=noisies[mid], gt=gts[mid] if gts else None, angle=angle_rows[mid])
                    inp_cpu_for_snap = None
                else:
                    inp_cpu_for_snap = rec.get("inp_cpu", None)

            ph, pw = rec["pred"].shape
            nn, gg = _match_to_pred_size((ph, pw), rec["noisy"], rec["gt"])
            aname = f"{int(rec['angle']):03d}" if rec["angle"] >= 0 else "xxx"
            fig_name = f"id{pick_id}_a{aname}_step{steps}"
            save_quad_and_tiles(
                nn, rec["pred"], gg,
                out_dir,
                fig_name,
                tiles_cfg,
                use_gt_percentile_norm=True
            )

            # snapshots (only for bridge/sample mode and numeric pick_angle)
            if snap_enable and (val_infer in ("sample", "bridge")):
                if inp_cpu_for_snap is None:
                    print("[SNAP] skipped: no cached inp for selected rec (use numeric pick_angle, not 'mid').")
                else:
                    inp_one = inp_cpu_for_snap.unsqueeze(0).to(dev).float()
                    pred_b, x1_b, _, states01 = infer_batch(inp_one, want_states=True, snap_mid_ts=snap_ts)

                    imgs, tags = [], []
                    if snap_inc_in:
                        imgs.append(x1_b[0, 0].detach().cpu().numpy().astype(np.float32))
                        tags.append("t=1.00_in")

                    for tt in snap_ts:
                        tt = float(tt)
                        if states01 is not None and (tt in states01):
                            imgs.append(states01[tt][0, 0].detach().cpu().numpy().astype(np.float32))
                            tags.append(f"t={tt:.4f}")

                    if snap_inc_out:
                        imgs.append(pred_b[0, 0].detach().cpu().numpy().astype(np.float32))
                        tags.append("t=0.00_out")

                    gt_for_norm = rec.get("gt", None)
                    if snap_inc_gt and (gt_for_norm is not None):
                        imgs.append(np.asarray(gt_for_norm, dtype=np.float32))
                        tags.append("GT")

                    snap_name = f"{fig_name}_sig{sigma_T}"
                    save_snapshots(
                        out_dir=out_dir,
                        subdir=snap_dir,
                        base_name=snap_name,
                        imgs01=imgs,
                        tags=tags,
                        gt01=gt_for_norm,
                        use_gt_percentile_norm=True,
                        save_strip=snap_save_strip,
                        save_individual=snap_save_ind,
                    )

    else:
        # ---------------- FIX: all-mode also export recon3d-compatible outputs ----------------
        print(f"[MODE] process all data in split '{want}'")

        preds_all, noisies_all, gts_all = [], [], []
        preds_raw_all, noisies_raw_all, gts_raw_all = [], [], []

        id_rows, angle_rows, A_rows = [], [], []
        noisy_lo_rows, noisy_hi_rows = [], []
        gt_lo_rows, gt_hi_rows = [], []
        psnr_rows, ssim_rows = [], []

        pbar = tqdm(loader, ncols=100)
        for batch in pbar:
            inp = batch["inp"].to(dev).float()
            pred01, x1_img01, mask01 = infer_batch(inp)

            pred_np = pred01.detach().cpu().numpy()
            noisy_np = x1_img01.detach().cpu().numpy()
            mask_np = mask01.detach().cpu().numpy()

            has_gt = ("gt" in batch) and (batch["gt"] is not None)
            gt_np = batch["gt"].detach().cpu().numpy() if has_gt else None

            bid = batch.get("id", None)
            if torch.is_tensor(bid):
                ids = bid.detach().cpu().numpy().tolist()
            elif isinstance(bid, (list, tuple, np.ndarray)):
                ids = list(bid)
            else:
                ids = [bid] * pred_np.shape[0]

            angles = batch.get("angle", [None] * pred_np.shape[0])
            if torch.is_tensor(angles):
                angles = angles.detach().cpu().numpy().tolist()

            A_arr = batch.get("A", [None] * pred_np.shape[0])
            if torch.is_tensor(A_arr):
                A_arr = A_arr.detach().cpu().numpy().tolist()

            for i in range(pred_np.shape[0]):
                pred_i = np.squeeze(pred_np[i, 0]).astype(np.float32)
                noz_i = np.squeeze(noisy_np[i, 0]).astype(np.float32)
                m_i = np.squeeze(mask_np[i, 0]).astype(np.float32)
                gt_i = np.squeeze(gt_np[i, 0]).astype(np.float32) if has_gt else None

                preds_all.append(pred_i)
                noisies_all.append(noz_i)
                if has_gt:
                    gts_all.append(gt_i)

                idi = ids[i]
                ai = angles[i] if angles[i] is not None else -1
                Ai = A_arr[i] if A_arr[i] is not None else 360

                id_rows.append(idi if idi is not None else "")
                angle_rows.append(ai)
                A_rows.append(Ai)

                nlo = _get_scalar_from_batch(batch, "noisy_lo", i)
                nhi = _get_scalar_from_batch(batch, "noisy_hi", i)
                glo = _get_scalar_from_batch(batch, "gt_lo", i)
                ghi = _get_scalar_from_batch(batch, "gt_hi", i)
                noisy_lo_rows.append(nlo); noisy_hi_rows.append(nhi)
                gt_lo_rows.append(glo); gt_hi_rows.append(ghi)

                # raw restore
                if (glo is not None) and (ghi is not None):
                    pr_raw = pred_i * (float(ghi) - float(glo)) + float(glo)
                else:
                    pr_raw = None
                if (glo is not None) and (ghi is not None) and (gt_i is not None):
                    g_raw = gt_i * (float(ghi) - float(glo)) + float(glo)
                else:
                    g_raw = None
                if (nlo is not None) and (nhi is not None):
                    n_raw = noz_i * (float(nhi) - float(nlo)) + float(nlo)
                else:
                    n_raw = None

                preds_raw_all.append(pr_raw)
                noisies_raw_all.append(n_raw)
                gts_raw_all.append(g_raw)

                # metrics
                if has_gt and gt_i is not None:
                    if metric_missing_only:
                        miss = (1.0 - m_i).astype(np.float32)
                        if miss.mean() > 1e-6:
                            p_use = pred_i * miss
                            g_use = gt_i * miss
                        else:
                            p_use, g_use = pred_i, gt_i
                    else:
                        p_use, g_use = pred_i, gt_i
                    ps = _psnr01(p_use, g_use)
                    ss = _ssim2d(p_use, g_use)
                else:
                    ps, ss = None, None
                psnr_rows.append(ps)
                ssim_rows.append(ss)

        if not preds_all:
            raise RuntimeError("[ERROR] No predictions generated.")

        vol_pred = np.stack(preds_all, axis=0).astype(np.float32)
        _save_stack_both(out_dir, "pred_volume", vol_pred)

        vol_nozy = np.stack(noisies_all, axis=0).astype(np.float32)
        _save_stack_both(out_dir, "noisy_volume", vol_nozy)

        if len(gts_all) == len(preds_all) and len(gts_all) > 0:
            vol_gt = np.stack(gts_all, axis=0).astype(np.float32)
            _save_stack_both(out_dir, "gt_volume", vol_gt)

        _try_save_raw_stack(out_dir, "pred_volume_raw", preds_raw_all)
        _try_save_raw_stack(out_dir, "noisy_volume_raw", noisies_raw_all)
        _try_save_raw_stack(out_dir, "gt_volume_raw", gts_raw_all)

        # metrics.csv (for recon3d angles)
        csv_path = os.path.join(out_dir, "metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "id", "angle", "A", "nlo", "nhi", "glo", "ghi", "PSNR", "SSIM"])
            for idx, val in enumerate(zip(id_rows, angle_rows, A_rows,
                                          noisy_lo_rows, noisy_hi_rows,
                                          gt_lo_rows, gt_hi_rows,
                                          psnr_rows, ssim_rows)):
                w.writerow([idx] + [x if x is not None else "" for x in val])
        print(f"[DONE] Saved all predictions -> {out_dir}")
        print(f"[DONE] Metrics -> {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/sample_i2sb_local_multi.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args.cfg, ckpt=args.ckpt, out=args.out)


if __name__ == "__main__":
    main()
