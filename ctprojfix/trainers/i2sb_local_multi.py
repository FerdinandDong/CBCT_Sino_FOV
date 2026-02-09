# ctprojfix/trainers/i2sb_local_multi.py
# multi step I2SB trainer (random-t bridge training, multi-step sampling val/preview)
# -*- coding: utf-8 -*-
import os, csv, sys
import glob, re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ctprojfix.evals.metrics import psnr as psnr_fn, ssim as ssim_fn


# ---------------- AMP 兼容封装 ----------------
def _create_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


class _autocast_ctx:
    def __init__(self, enabled: bool, device_type: str = "cuda"):
        self.enabled = enabled
        self.device_type = device_type
        self.ctx = None

    def __enter__(self):
        if self.enabled:
            try:
                self.ctx = torch.amp.autocast(self.device_type)
            except Exception:
                self.ctx = torch.cuda.amp.autocast()
            return self.ctx.__enter__()
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.ctx is not None:
            return self.ctx.__exit__(exc_type, exc_val, exc_tb)
        return False


# ---------------- EMA ----------------
class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    def update(self, model: nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert n in self.shadow
                self.shadow[n].mul_(d).add_(p.data, alpha=1.0 - d)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


# ---------------- 配置 ----------------
@dataclass
class SchedCfg:
    type: Optional[str] = None
    T_max: int = 150
    eta_min: float = 1e-6
    step_size: int = 50
    gamma: float = 0.5


# ---------------- 工具 ----------------
def _ceil_to(v: int, m: int) -> int:
    return ((v + m - 1) // m) * m if m > 0 else v


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _latest_ckpt(ckpt_dir: str, prefix: str) -> Optional[str]:
    patt = os.path.join(ckpt_dir, f"{prefix}_epoch*.pth")
    files = glob.glob(patt)
    if not files:
        return None
    files.sort(key=_natural_key)
    return files[-1]


def _tag_ckpt(ckpt_dir: str, prefix: str, tag: str) -> Optional[str]:
    p = os.path.join(ckpt_dir, f"{prefix}_{tag}.pth")
    return p if os.path.isfile(p) else None


def _auto_resume_ckpt(ckpt_dir: str, prefix: str) -> Optional[str]:
    """
    auto resume 优先级：
      1) prefix_last.pth
      2) prefix_best.pth
      3) prefix_epoch*.pth 最新
    """
    p = _tag_ckpt(ckpt_dir, prefix, "last")
    if p:
        return p
    p = _tag_ckpt(ckpt_dir, prefix, "best")
    if p:
        return p
    return _latest_ckpt(ckpt_dir, prefix)


def _pad_conds(mask_and_maybe_angle: torch.Tensor, pad, has_angle: bool) -> torch.Tensor:
    """
    conds: [mask, (angle)] in [0,1]
    - mask pad: constant 0  (pad 区必须视为 missing)
    - angle pad: replicate (边缘延拓)
    """
    if mask_and_maybe_angle is None:
        return None
    conds = mask_and_maybe_angle
    if conds.shape[1] <= 0:
        return conds
    mask = conds[:, 0:1, ...]
    mask = F.pad(mask, pad, mode="constant", value=0.0)

    if has_angle and conds.shape[1] > 1:
        ang = conds[:, 1:, ...]
        ang = F.pad(ang, pad, mode="replicate")
        return torch.cat([mask, ang], dim=1)
    return mask


def _resize_max_hw(x: torch.Tensor, max_hw: Optional[int]) -> torch.Tensor:
    """
    把张量的 H/W 按比例缩到 max(H,W) <= max_hw（只缩小，不放大）。
    """
    if (max_hw is None) or (max_hw <= 0):
        return x
    H, W = x.shape[-2], x.shape[-1]
    m = max(H, W)
    if m <= max_hw:
        return x
    scale = float(max_hw) / float(m)
    newH = max(1, int(round(H * scale)))
    newW = max(1, int(round(W * scale)))
    return F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)


# ---------------- preview utils ----------------
def _percentile_lohi(a, p_lo=1.0, p_hi=99.0):
    import numpy as np
    a = a.astype("float32")
    lo, hi = np.percentile(a, [p_lo, p_hi])
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max() + 1e-6)
    return float(lo), float(hi)


def _norm01_with_lohi(a, lo, hi):
    import numpy as np
    a = a.astype("float32")
    return ((a - lo) / (hi - lo + 1e-6)).clip(0, 1).astype("float32")


def _save_quad_preview(noisy01, pred01, gt01, out_path, title=None):
    """
    4-panel: Noisy / Pred / GT / |Pred-GT| + colorbar
    使用 GT 的 percentile 强度范围统一归一化
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = _percentile_lohi(gt01, 1.0, 99.0)

    n = _norm01_with_lohi(noisy01, lo, hi)
    p = _norm01_with_lohi(pred01,  lo, hi)
    g = _norm01_with_lohi(gt01,    lo, hi)

    diff = np.abs(p - g).astype("float32")
    d_lo, d_hi = _percentile_lohi(diff, 0.0, 99.0)
    d_hi = max(d_hi, 1e-6)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(n, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Noisy"); axes[0].axis("off")
    axes[1].imshow(p, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Pred");  axes[1].axis("off")
    axes[2].imshow(g, cmap="gray", vmin=0, vmax=1); axes[2].set_title("GT");    axes[2].axis("off")
    im = axes[3].imshow(diff, cmap="magma", vmin=0, vmax=d_hi); axes[3].set_title("|Pred-GT|"); axes[3].axis("off")
    cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PREVIEW] {out_path}")


# =============================================================================
# Perceptual loss (VGG16 features)  —— mask-aware, grayscale->RGB, optional downsample
# =============================================================================
def _to_vgg_input_from_m1(x_m1: torch.Tensor) -> torch.Tensor:
    """
    x_m1: (B,1,H,W) in [-1,1]  -> VGG input (B,3,H,W), ImageNet normalized.
    """
    x01 = ((x_m1 + 1.0) * 0.5).clamp(0.0, 1.0)
    x3 = x01.repeat(1, 3, 1, 1)
    mean = x3.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = x3.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x3 - mean) / std


class VGGPerceptualLoss(nn.Module):
    """
    VGG16 feature perceptual loss.
    默认层：relu1_2, relu2_2, relu3_3（省显存/更快）。
    支持 weight_map（B,1,H,W）做空间加权。
    """
    def __init__(self, layers: Optional[List[int]] = None, use_pretrained: bool = True):
        super().__init__()
        try:
            from torchvision import models
            from torchvision.models import VGG16_Weights
        except Exception as e:
            raise RuntimeError("Perceptual loss 需要 torchvision。请先安装 torchvision，或关闭 use_percep。") from e

        if layers is None:
            layers = [3, 8, 15]  # relu1_2, relu2_2, relu3_3

        self.layers = list(layers)

        if use_pretrained:
            vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        else:
            vgg = models.vgg16(weights=None).features

        for p in vgg.parameters():
            p.requires_grad = False
        vgg.eval()
        self.vgg = vgg

    def _extract_feats(self, x_vgg: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        h = x_vgg
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.layers:
                feats.append(h)
        return feats

    def forward(
        self,
        pred_m1: torch.Tensor,
        tgt_m1: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        pred_m1/tgt_m1: (B,1,H,W) in [-1,1]
        weight_map: (B,1,H,W) in [0,1] or any positive weights
        """
        pred_vgg = _to_vgg_input_from_m1(pred_m1.float())
        tgt_vgg  = _to_vgg_input_from_m1(tgt_m1.float())

        pred_feats = self._extract_feats(pred_vgg)
        with torch.no_grad():
            tgt_feats = self._extract_feats(tgt_vgg)

        loss = pred_vgg.new_tensor(0.0)
        eps = 1e-6

        for fp, ft in zip(pred_feats, tgt_feats):
            if weight_map is None:
                loss = loss + F.l1_loss(fp, ft)
            else:
                wm = F.interpolate(weight_map.float(), size=fp.shape[-2:], mode="bilinear", align_corners=False)
                wm = wm.clamp_min(0.0)
                wm = wm.expand(-1, fp.shape[1], -1, -1)
                diff = (fp - ft).abs()
                loss = loss + (diff * wm).sum() / wm.sum().clamp_min(eps)

        return loss


class I2SBLocalTrainer:
    def __init__(
        self,
        device: str = "cuda",
        lr: float = 3e-4,
        epochs: int = 150,
        sigma_T: float = 1.0,
        t0: float = 1e-4,
        ema_decay: float = 0.999,
        ckpt_dir: str = "checkpoints/i2sb_local",
        ckpt_prefix: str = "i2sb_local",
        save_every: int = 1,
        max_keep: int = 5,
        log_dir: str = "logs/i2sb_local",
        cond_has_angle: bool = True,
        val_metric: str = "loss",
        maximize_metric: Optional[bool] = None,
        sched: Optional[Dict[str, Any]] = None,
        dump_preview_every: int = 0,
        log_interval: int = 100,
        # --- outpainting 权重 ---
        w_valid: float = 1.0,
        w_missing: float = 5.0,
        # --- perceptual loss ---
        use_percep: bool = False,
        w_percep: float = 0.02,
        percep_layers: Optional[List[int]] = None,
        percep_use_pretrained: bool = True,
        percep_region: str = "missing",   # "missing" | "both" | "valid"
        percep_max_hw: int = 256,         # perceptual 分支最大边降采样

        # --- val speed control ---
        eval_every: int = 1,              # 每 N 个 epoch 才跑一次 val
        val_max_batches: int = 0,         # val 最多跑 N 个 batch（0=全量）

        # --- resume ---
        resume: str = "auto",             # "auto" | "none" | "last" | "best"
        resume_from: Optional[str] = None,
        strict_load: bool = True,

        # --- inference / sampling (I2SB bridge sampler) ---
        val_infer: str = "one_step",      # "one_step" | "sample"
        sample_steps: int = 16,
        sample_stochastic: bool = False,
        sample_clamp_known: bool = True,

        # --- training: unroll / self-consistency ---
        train_unroll: bool = False,
        unroll_steps: int = 2,
        unroll_stopgrad: bool = True,
        unroll_use_same_eps: bool = True,
        unroll_t_mode: str = "nested",
        unroll_w: float = 1.0,
        unroll_warmup_epochs: int = 0,

        # --- NEW: unroll forward uses checkpointing (default True, cfg 不用加也能跑) ---
        unroll_use_checkpoint: bool = True,

        # --- sampling stabilization when clamp_known=false (optional) ---
        soft_clamp_valid: float = 0.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.sigma_T = float(sigma_T)
        self.t0 = float(t0)
        self.ema_decay = float(ema_decay)
        self.cond_has_angle = bool(cond_has_angle)

        # weighted pixel loss
        self.w_valid = float(w_valid)
        self.w_missing = float(w_missing)

        # perceptual
        self.use_percep = bool(use_percep)
        self.w_percep = float(w_percep)
        self.percep_region = str(percep_region).lower().strip()
        self.percep_max_hw = int(percep_max_hw) if percep_max_hw is not None else 0
        self.percep = None
        if self.use_percep:
            self.percep = VGGPerceptualLoss(
                layers=percep_layers,
                use_pretrained=bool(percep_use_pretrained),
            ).to(self.device)
            self.percep.eval()

        self.ckpt_dir = ckpt_dir
        self.ckpt_prefix = ckpt_prefix
        self.save_every = int(save_every)
        self.max_keep = int(max_keep)
        self.log_dir = log_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.log_csv_dir = os.path.join("logs", "logplots", ckpt_prefix)
        os.makedirs(self.log_csv_dir, exist_ok=True)
        self.log_csv_path = os.path.join(self.log_csv_dir, "train.csv")
        if not os.path.isfile(self.log_csv_path):
            with open(self.log_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "step", "lr", "loss", "val_loss", "psnr", "ssim"])

        self.val_metric = str(val_metric or "loss").lower()
        if maximize_metric is None:
            self.maximize_metric = (self.val_metric != "loss")
        else:
            self.maximize_metric = bool(maximize_metric)

        self.sched_cfg = SchedCfg(**(sched or {}))
        self.dump_preview_every = int(dump_preview_every)
        self.log_interval = int(log_interval)

        # val control
        self.eval_every = max(1, int(eval_every) if eval_every is not None else 1)
        self.val_max_batches = int(val_max_batches) if val_max_batches is not None else 0

        # resume control
        self.resume = str(resume or "auto").lower().strip()
        self.resume_from = resume_from
        self.strict_load = bool(strict_load)

        # inference / sampling control
        self.val_infer = str(val_infer or "one_step").lower().strip()
        self.sample_steps = max(1, int(sample_steps))
        self.sample_stochastic = bool(sample_stochastic)
        self.sample_clamp_known = bool(sample_clamp_known)

        # unroll training control
        self.train_unroll = bool(train_unroll)
        self.unroll_steps = max(2, int(unroll_steps))
        self.unroll_stopgrad = bool(unroll_stopgrad)
        self.unroll_use_same_eps = bool(unroll_use_same_eps)
        self.unroll_t_mode = str(unroll_t_mode).lower().strip()
        self.unroll_w = float(unroll_w)
        self.unroll_warmup_epochs = int(unroll_warmup_epochs)

        # NEW
        self.unroll_use_checkpoint = bool(unroll_use_checkpoint)

        # soft clamp in sampler (optional)
        self.soft_clamp_valid = float(soft_clamp_valid)

        # states
        self._recent_ckpts = []
        self.best_epoch: Optional[int] = None
        self._init_best_score()
        self.start_epoch = 1
        self.global_step = 0
        self._cur_epoch = 0

    def _init_best_score(self):
        self.best_score = -float("inf") if self.maximize_metric else float("inf")

    @staticmethod
    def _as_float_or_blank(v):
        if v is None:
            return ""
        if isinstance(v, str):
            if v.strip() == "":
                return ""
            try:
                return float(v)
            except Exception:
                return ""
        try:
            return float(v)
        except Exception:
            return ""

    def _unet_factor(self, model: nn.Module) -> int:
        m = getattr(model, "module", model)
        n_down = len(getattr(m, "downs", []))
        return 2 ** max(int(n_down), 0)

    # ---------------- Bridge helpers ----------------
    def _bridge_std(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_T * torch.sqrt(t * (1.0 - t))

    def _eps_from_xt(self, xt: torch.Tensor, x0_hat: torch.Tensor, x1m: torch.Tensor, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        std = self._bridge_std(t).clamp_min(eps)
        return (xt - (1.0 - t) * x0_hat - t * x1m) / std

    def _masked_loss(
        self,
        x_hat: torch.Tensor,
        x_ref: torch.Tensor,
        conds_crop: torch.Tensor,
        use_percep: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        x_hat/x_ref: (B,1,H,W) in [-1,1]
        conds_crop: (B,Cc,H,W) in [0,1], channel0 is mask(valid=1)
        use_percep:
          - None: use self.use_percep (default)
          - False: force disable perceptual (used for unroll to prevent OOM)
          - True: force enable
        """
        mask = conds_crop[:, 0:1, ...].clamp(0.0, 1.0)
        missing = (1.0 - mask).clamp(0.0, 1.0)
        weights = self.w_valid * mask + self.w_missing * missing

        diff2 = (x_hat - x_ref) ** 2
        loss_pix = (diff2 * weights).sum() / weights.sum().clamp_min(1e-6)

        loss = loss_pix

        do_percep = self.use_percep if use_percep is None else bool(use_percep)
        if do_percep and (self.percep is not None) and (self.w_percep > 0):
            if self.percep_region == "missing":
                wmap = missing
            elif self.percep_region == "valid":
                wmap = mask
            else:
                wmap = weights

            x_hat_p = _resize_max_hw(x_hat, self.percep_max_hw)
            x_ref_p = _resize_max_hw(x_ref, self.percep_max_hw)
            wmap_p  = _resize_max_hw(wmap,  self.percep_max_hw)

            if self.device.type == "cuda":
                try:
                    with torch.amp.autocast("cuda", enabled=False):
                        loss_perc = self.percep(x_hat_p, x_ref_p, weight_map=wmap_p)
                except Exception:
                    with torch.cuda.amp.autocast(enabled=False):
                        loss_perc = self.percep(x_hat_p, x_ref_p, weight_map=wmap_p)
            else:
                loss_perc = self.percep(x_hat_p, x_ref_p, weight_map=wmap_p)

            loss = loss + self.w_percep * loss_perc

        return loss

    # ---------------- memory-safe unroll helpers ----------------
    def _ckpt_forward(self, model: nn.Module, net_in: torch.Tensor) -> torch.Tensor:
        """Checkpointed forward for unroll (saves activation memory)."""
        if (not self.unroll_use_checkpoint) or (self.device.type != "cuda"):
            return model(net_in)

        def _fw(x):
            return model(x)

        # ensure checkpoint sees grad path
        if not net_in.requires_grad:
            net_in = net_in.requires_grad_(True)

        try:
            return checkpoint(_fw, net_in, use_reentrant=False)
        except TypeError:
            return checkpoint(_fw, net_in)

    def _tf_loss_and_cache(self, model: nn.Module, batch):
        """
        Teacher-forcing forward:
          returns (loss_tf, cache_for_unroll_detached)
        cache is detached so unroll won't keep TF graph alive.
        """
        x0 = batch["gt"].to(self.device).float()
        x1 = batch["inp"].to(self.device).float()
        B, C, H, W = x1.shape

        x1_img = x1[:, 0:1, ...]
        conds  = x1[:, 1:,  ...]

        x0_m1  = x0 * 2.0 - 1.0
        x1m_m1 = x1_img * 2.0 - 1.0

        factor = self._unet_factor(model)
        Ht = _ceil_to(H, factor)
        Wt = _ceil_to(W, factor)

        if (Ht, Wt) != (H, W):
            pad = (0, Wt - W, 0, Ht - H)
            x0_m1  = F.pad(x0_m1,  pad, mode="reflect")
            x1m_m1 = F.pad(x1m_m1, pad, mode="reflect")
            conds  = _pad_conds(conds, pad, has_angle=self.cond_has_angle)

        t = torch.rand(B, 1, 1, 1, device=self.device) * (1.0 - self.t0) + self.t0
        noise = torch.randn_like(x0_m1)
        std = self._bridge_std(t)
        xt = (1.0 - t) * x0_m1 + t * x1m_m1 + std * noise

        t_map = t.expand(B, 1, Ht, Wt)
        net_in = torch.cat([xt, x1m_m1, conds, t_map], dim=1)
        x0_hat = model(net_in)

        if (Ht, Wt) != (H, W):
            x0_hat_crop = x0_hat[..., :H, :W]
            x0_ref = x0_m1[..., :H, :W]
            conds_crop = conds[..., :H, :W]
        else:
            x0_hat_crop = x0_hat
            x0_ref = x0_m1
            conds_crop = conds

        loss_tf = self._masked_loss(x0_hat_crop, x0_ref, conds_crop, use_percep=None)

        cache = dict(
            B=B, H=H, W=W, Ht=Ht, Wt=Wt,
            # detached states for unroll
            xt=xt.detach(),
            t=t.detach(),
            x0_hat_prev=x0_hat.detach(),
            x1m_m1=x1m_m1.detach(),
            conds=conds.detach(),
            x0_ref=x0_ref.detach(),
            conds_crop=conds_crop.detach(),
        )
        return loss_tf, cache

    def _unroll_one_step_loss(self, model: nn.Module, cache, x_prev: torch.Tensor, t_prev: torch.Tensor, x0_hat_prev_det: torch.Tensor):
        """
        One unroll step (stopgrad mode only):
          - sample t_next < t_prev
          - build x_next from eps_hat (all detached)
          - forward @ t_next
          - return loss_u (percep forced OFF), and next detached states
        """
        B = cache["B"]; H = cache["H"]; W = cache["W"]; Ht = cache["Ht"]; Wt = cache["Wt"]
        x1m_m1 = cache["x1m_m1"]
        conds  = cache["conds"]
        x0_ref = cache["x0_ref"]
        conds_crop = cache["conds_crop"]

        if self.unroll_t_mode == "nested":
            u = torch.rand_like(t_prev)
            t_next = (t_prev * u).clamp_min(self.t0)
        else:
            t_next = (torch.rand_like(t_prev) * (t_prev - self.t0) + self.t0).clamp_min(self.t0)

        # stopgrad path
        x0_for_eps = x0_hat_prev_det
        x_for_eps  = x_prev

        eps_hat  = self._eps_from_xt(x_for_eps, x0_for_eps, x1m_m1, t_prev)
        std_next = self._bridge_std(t_next)
        x_next   = (1.0 - t_next) * x0_for_eps + t_next * x1m_m1 + std_next * eps_hat
        x_next   = x_next.detach()  # prevent graph chaining

        t_map_next = t_next.expand(B, 1, Ht, Wt)
        net_in_next = torch.cat([x_next, x1m_m1, conds, t_map_next], dim=1)

        x0_hat_next = self._ckpt_forward(model, net_in_next)

        if (Ht, Wt) != (H, W):
            x0_hat_next_crop = x0_hat_next[..., :H, :W]
        else:
            x0_hat_next_crop = x0_hat_next

        loss_u = self._masked_loss(x0_hat_next_crop, x0_ref, conds_crop, use_percep=False)
        return loss_u, x_next, t_next.detach(), x0_hat_next.detach()

    # --------- I2SB 训练 Loss (Random t bridge) ---------
    def _step_loss(self, model: nn.Module, batch) -> torch.Tensor:
        """
        Base teacher-forcing bridge loss.
        NOTE:
          - if unroll_stopgrad=True, unroll is done in fit() (memory-safe), NOT here.
          - if unroll_stopgrad=False, keep legacy unroll here (may be memory-heavy).
        """
        x0 = batch["gt"].to(self.device).float()
        x1 = batch["inp"].to(self.device).float()
        B, C, H, W = x1.shape

        x1_img = x1[:, 0:1, ...]
        conds  = x1[:, 1:,  ...]

        x0_m1  = x0 * 2.0 - 1.0
        x1m_m1 = x1_img * 2.0 - 1.0

        factor = self._unet_factor(model)
        Ht = _ceil_to(H, factor)
        Wt = _ceil_to(W, factor)

        if (Ht, Wt) != (H, W):
            pad = (0, Wt - W, 0, Ht - H)
            x0_m1  = F.pad(x0_m1,  pad, mode="reflect")
            x1m_m1 = F.pad(x1m_m1, pad, mode="reflect")
            conds  = _pad_conds(conds, pad, has_angle=self.cond_has_angle)

        t = torch.rand(B, 1, 1, 1, device=self.device) * (1.0 - self.t0) + self.t0
        noise = torch.randn_like(x0_m1)
        std = self._bridge_std(t)
        xt = (1.0 - t) * x0_m1 + t * x1m_m1 + std * noise

        t_map = t.expand(B, 1, Ht, Wt)
        net_in = torch.cat([xt, x1m_m1, conds, t_map], dim=1)
        x0_hat = model(net_in)

        if (Ht, Wt) != (H, W):
            x0_hat_crop = x0_hat[..., :H, :W]
            x0_ref = x0_m1[..., :H, :W]
            conds_crop = conds[..., :H, :W]
        else:
            x0_hat_crop = x0_hat
            x0_ref = x0_m1
            conds_crop = conds

        loss_tf = self._masked_loss(x0_hat_crop, x0_ref, conds_crop, use_percep=None)
        loss = loss_tf

        # legacy unroll only when stopgrad=False (graph-through chain)
        if self.train_unroll and (self.unroll_w > 0.0) and (not self.unroll_stopgrad):
            if int(getattr(self, "_cur_epoch", 0)) >= int(self.unroll_warmup_epochs):
                K = int(self.unroll_steps)
                x = xt
                t_prev = t
                x0_hat_prev = x0_hat
                loss_unroll = x0_hat.new_tensor(0.0)

                for j in range(1, K):
                    if self.unroll_t_mode == "nested":
                        u = torch.rand_like(t_prev)
                        t_next = (t_prev * u).clamp_min(self.t0)
                    else:
                        t_next = (torch.rand_like(t_prev) * (t_prev - self.t0) + self.t0).clamp_min(self.t0)

                    if self.unroll_use_same_eps:
                        eps_hat = self._eps_from_xt(x, x0_hat_prev, x1m_m1, t_prev)
                    else:
                        eps_hat = self._eps_from_xt(x.detach(), x0_hat_prev, x1m_m1, t_prev)

                    std_next = self._bridge_std(t_next)
                    x_next = (1.0 - t_next) * x0_hat_prev + t_next * x1m_m1 + std_next * eps_hat

                    t_map_next = t_next.expand(B, 1, Ht, Wt)
                    net_in_next = torch.cat([x_next, x1m_m1, conds, t_map_next], dim=1)
                    x0_hat_next = model(net_in_next)

                    if (Ht, Wt) != (H, W):
                        x0_hat_next_crop = x0_hat_next[..., :H, :W]
                    else:
                        x0_hat_next_crop = x0_hat_next

                    # perceptual off in unroll
                    loss_unroll = loss_unroll + self._masked_loss(x0_hat_next_crop, x0_ref, conds_crop, use_percep=False)

                    x = x_next
                    t_prev = t_next
                    x0_hat_prev = x0_hat_next

                loss = loss + float(self.unroll_w) * (loss_unroll / max(1, (K - 1)))

        return loss

    def _log_row(self, epoch, step, lr, loss, val_loss, psnr, ssim):
        with open(self.log_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                int(epoch),
                int(step),
                self._as_float_or_blank(lr),
                self._as_float_or_blank(loss),
                self._as_float_or_blank(val_loss),
                self._as_float_or_blank(psnr),
                self._as_float_or_blank(ssim),
            ])

    def _save_ckpt(
        self,
        model: nn.Module,
        epoch: int,
        tag: Optional[str] = None,
        opt: Optional[optim.Optimizer] = None,
        sched: Optional[optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ):
        payload = {
            "epoch": int(epoch),
            "state_dict": model.state_dict(),
            "global_step": int(getattr(self, "global_step", 0)),
            "best_score": float(getattr(self, "best_score", 0.0)),
            "best_epoch": getattr(self, "best_epoch", None),
        }

        if opt is not None:
            payload["optimizer"] = opt.state_dict()
        if sched is not None:
            try:
                payload["scheduler"] = sched.state_dict()
            except Exception:
                pass
        if scaler is not None:
            try:
                payload["scaler"] = scaler.state_dict()
            except Exception:
                pass

        if hasattr(self, "ema") and self.ema is not None:
            try:
                payload["ema_decay"] = float(self.ema.decay)
                payload["ema_shadow"] = {k: v.detach().cpu() for k, v in self.ema.shadow.items()}
            except Exception:
                pass

        name = f"{self.ckpt_prefix}_{'epoch'+str(epoch) if tag is None else tag}.pth"
        path = os.path.join(self.ckpt_dir, name)
        torch.save(payload, path)
        print(f"[CKPT] save -> {path}", flush=True)

        if tag is None:
            self._recent_ckpts.append(path)

    def _prune_ckpt(self):
        while len(self._recent_ckpts) > self.max_keep:
            old = self._recent_ckpts.pop(0)
            try:
                os.remove(old)
                print(f"[CKPT] prune -> {old}", flush=True)
            except Exception:
                pass

    def _try_resume(
        self,
        model: nn.Module,
        opt: optim.Optimizer,
        sched: Optional[optim.lr_scheduler._LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
    ) -> int:
        if self.resume in ("none", "false", "0"):
            print("[RESUME] disabled (resume=none).", flush=True)
            return 1

        ckpt_path = None
        if self.resume_from:
            ckpt_path = self.resume_from
        else:
            if self.resume == "last":
                ckpt_path = _tag_ckpt(self.ckpt_dir, self.ckpt_prefix, "last")
            elif self.resume == "best":
                ckpt_path = _tag_ckpt(self.ckpt_dir, self.ckpt_prefix, "best")
            else:
                ckpt_path = _auto_resume_ckpt(self.ckpt_dir, self.ckpt_prefix)

        if (not ckpt_path) or (not os.path.isfile(ckpt_path)):
            print("[RESUME] no checkpoint found; start from scratch.", flush=True)
            return 1

        print(f"[RESUME] loading: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=self.device)

        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=self.strict_load)

        if ("optimizer" in ckpt) and (opt is not None):
            try:
                opt.load_state_dict(ckpt["optimizer"])
                print("[RESUME] optimizer restored.", flush=True)
            except Exception as e:
                print(f"[RESUME] optimizer restore failed: {e}", flush=True)

        if ("scheduler" in ckpt) and (sched is not None):
            try:
                sched.load_state_dict(ckpt["scheduler"])
                print("[RESUME] scheduler restored.", flush=True)
            except Exception as e:
                print(f"[RESUME] scheduler restore failed: {e}", flush=True)

        if ("scaler" in ckpt) and (scaler is not None):
            try:
                scaler.load_state_dict(ckpt["scaler"])
                print("[RESUME] scaler restored.", flush=True)
            except Exception as e:
                print(f"[RESUME] scaler restore failed: {e}", flush=True)

        try:
            self.global_step = int(ckpt.get("global_step", 0))
        except Exception:
            self.global_step = 0

        if "best_score" in ckpt:
            try:
                self.best_score = float(ckpt["best_score"])
            except Exception:
                pass
        if "best_epoch" in ckpt:
            self.best_epoch = ckpt.get("best_epoch", None)

        if hasattr(self, "ema") and self.ema is not None and ("ema_shadow" in ckpt):
            try:
                shadow = ckpt["ema_shadow"]
                for k, v in shadow.items():
                    if k in self.ema.shadow:
                        self.ema.shadow[k].copy_(v.to(self.ema.shadow[k].device))
                print("[RESUME] EMA shadow restored.", flush=True)
            except Exception as e:
                print(f"[RESUME] EMA shadow restore failed: {e}", flush=True)

        last_epoch = int(ckpt.get("epoch", 0))
        start_epoch = last_epoch + 1 if last_epoch > 0 else 1
        print(f"[RESUME] resumed at epoch {last_epoch} -> start from {start_epoch} | global_step={self.global_step}", flush=True)
        return start_epoch

    # ---------------------------------------------------------------------
    # I2SB multi-step sampling (bridge sampler), network predicts x0_hat
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _i2sb_sample(
        self,
        model: nn.Module,
        x1_img01: torch.Tensor,
        conds01: torch.Tensor,
        steps: Optional[int] = None,
        stochastic: Optional[bool] = None,
        clamp_known: Optional[bool] = None,
    ) -> torch.Tensor:
        steps = int(self.sample_steps if steps is None else steps)
        steps = max(1, steps)
        stochastic = self.sample_stochastic if stochastic is None else bool(stochastic)
        clamp_known = self.sample_clamp_known if clamp_known is None else bool(clamp_known)

        B, _, H, W = x1_img01.shape
        device = self.device
        use_amp = (device.type == "cuda")

        x1m = x1_img01.to(device).float() * 2.0 - 1.0
        conds = conds01.to(device).float()

        factor = self._unet_factor(model)
        Ht = _ceil_to(H, factor)
        Wt = _ceil_to(W, factor)
        if (Ht, Wt) != (H, W):
            pad = (0, Wt - W, 0, Ht - H)
            x1m = F.pad(x1m, pad, mode="reflect")
            conds = _pad_conds(conds, pad, has_angle=self.cond_has_angle)

        mask = conds[:, 0:1, ...].clamp(0.0, 1.0)
        eps = 1e-6
        sigma2 = float(self.sigma_T) ** 2

        ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
        x = x1m

        for k in range(steps):
            t = float(ts[k].item())
            s = float(ts[k + 1].item())

            t_map = x.new_full((B, 1, Ht, Wt), fill_value=t)
            net_in = torch.cat([x, x1m, conds, t_map], dim=1)

            with _autocast_ctx(enabled=use_amp, device_type="cuda"):
                x0_hat = model(net_in)

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
                    s_t = x.new_tensor(s).clamp_min(eps)
                    one_minus_t = (1.0 - t_t).clamp_min(eps)
                    one_minus_s = (1.0 - s_t).clamp_min(eps)

                    coef = (s_t * one_minus_t) / (t_t * one_minus_s + eps)
                    mean = ms + coef * r

                    sigma2_t = x.new_tensor(sigma2)
                    var_t = sigma2_t * (s_t * one_minus_s) * ((t_t - s_t).clamp_min(0.0)) / (t_t * one_minus_s + eps)

                if stochastic:
                    std = torch.sqrt(torch.clamp(var_t, min=0.0))
                    x = mean + std * torch.randn_like(x)
                else:
                    x = mean

            if clamp_known:
                x = mask * x1m + (1.0 - mask) * x

            if (not clamp_known) and (self.soft_clamp_valid is not None) and (self.soft_clamp_valid > 0.0):
                a = float(self.soft_clamp_valid)
                a = 0.0 if a < 0 else (1.0 if a > 1.0 else a)
                x = (mask * ((1.0 - a) * x + a * x1m)) + (1.0 - mask) * x

        if (Ht, Wt) != (H, W):
            x = x[..., :H, :W]

        return x.clamp(-1.0, 1.0)

    # --------- 验证：one-step 或 multi-step sampling ---------
    @torch.no_grad()
    def _eval_val(self, model: nn.Module, val_loader, max_batches: int = 0) -> Dict[str, float]:
        if val_loader is None:
            return {"val_loss": None, "psnr": None, "ssim": None}

        model.eval()
        self.ema.apply_shadow(model)

        s_loss, s_psnr, s_ssim, n = 0.0, 0.0, 0.0, 0
        metric_missing_only = True
        factor = self._unet_factor(model)

        bcount = 0
        for batch in val_loader:
            bcount += 1
            if (max_batches is not None) and (max_batches > 0) and (bcount > max_batches):
                break

            # val loss: use _step_loss (stopgrad=True 时这里就是 TF-only)
            loss = self._step_loss(model, batch).item()
            s_loss += loss

            x0 = batch["gt"].to(self.device).float()
            x1 = batch["inp"].to(self.device).float()
            B, C, H, W = x1.shape

            x1_img = x1[:, 0:1, ...]
            conds = x1[:, 1:, ...]

            if self.val_infer == "sample":
                pred_m1 = self._i2sb_sample(
                    model,
                    x1_img01=x1_img,
                    conds01=conds,
                    steps=self.sample_steps,
                    stochastic=self.sample_stochastic,
                    clamp_known=self.sample_clamp_known,
                )
                conds_for_metric = conds
            else:
                x1m_m1 = x1_img * 2.0 - 1.0
                Ht = _ceil_to(H, factor)
                Wt = _ceil_to(W, factor)
                t_map = torch.ones((B, 1, H, W), device=self.device, dtype=torch.float32)

                if (Ht, Wt) != (H, W):
                    pad = (0, Wt - W, 0, Ht - H)
                    x1m_pad = F.pad(x1m_m1, pad, mode="reflect")
                    t_map = F.pad(t_map, pad, mode="reflect")
                    conds_pad = _pad_conds(conds, pad, has_angle=self.cond_has_angle)
                else:
                    x1m_pad = x1m_m1
                    conds_pad = conds

                xt = x1m_pad
                xin = torch.cat([xt, x1m_pad, conds_pad, t_map], dim=1)
                pred_m1 = model(xin)

                if pred_m1.shape[-2:] != (Ht, Wt):
                    pred_m1 = F.interpolate(pred_m1, size=(Ht, Wt), mode="bilinear", align_corners=False)
                if (Ht, Wt) != (H, W):
                    pred_m1 = pred_m1[..., :H, :W]
                    conds_for_metric = conds_pad[..., :H, :W]
                else:
                    conds_for_metric = conds_pad

            pred01 = ((pred_m1.clamp(-1, 1) + 1.0) * 0.5).detach()
            gt01 = x0.detach()

            if metric_missing_only:
                mask = conds_for_metric[:, 0:1, ...].clamp(0.0, 1.0)
                miss = (1.0 - mask).clamp(0.0, 1.0)

            for i in range(B):
                p = pred01[i, 0].cpu().numpy()
                g = gt01[i, 0].cpu().numpy()

                if metric_missing_only:
                    mm = miss[i, 0].cpu().numpy().astype("float32")
                    if mm.mean() > 1e-6:
                        p_use = p * mm
                        g_use = g * mm
                    else:
                        p_use, g_use = p, g
                else:
                    p_use, g_use = p, g

                s_psnr += float(psnr_fn(p_use, g_use))
                s_ssim += float(ssim_fn(p_use, g_use))
                n += 1

        self.ema.restore(model)

        if n == 0:
            return {"val_loss": (s_loss / max(1, bcount)), "psnr": None, "ssim": None}

        return {
            "val_loss": s_loss / max(1, bcount),
            "psnr": s_psnr / n,
            "ssim": s_ssim / n,
        }

    # --------- 预览图导出：one-step 或 multi-step sampling ---------
    @torch.no_grad()
    def _dump_preview(self, model: nn.Module, data_loader, epoch: int):
        if self.dump_preview_every <= 0 or data_loader is None:
            return
        if epoch % self.dump_preview_every != 0:
            return

        model.eval()
        self.ema.apply_shadow(model)

        try:
            batch = next(iter(data_loader))
        except StopIteration:
            self.ema.restore(model)
            return

        x0 = batch["gt"].to(self.device).float()
        x1 = batch["inp"].to(self.device).float()
        B, C, H, W = x1.shape

        x1_img = x1[:, 0:1, ...]
        conds = x1[:, 1:, ...]

        if self.val_infer == "sample":
            pred_m1 = self._i2sb_sample(
                model,
                x1_img01=x1_img,
                conds01=conds,
                steps=self.sample_steps,
                stochastic=self.sample_stochastic,
                clamp_known=self.sample_clamp_known,
            )
        else:
            x1m_m1 = x1_img * 2.0 - 1.0
            factor = self._unet_factor(model)
            Ht = _ceil_to(H, factor)
            Wt = _ceil_to(W, factor)

            t_map = torch.ones((B, 1, H, W), device=self.device, dtype=torch.float32)
            if (Ht, Wt) != (H, W):
                pad = (0, Wt - W, 0, Ht - H)
                x1m_pad = F.pad(x1m_m1, pad, mode="reflect")
                t_map = F.pad(t_map, pad, mode="reflect")
                conds_pad = _pad_conds(conds, pad, has_angle=self.cond_has_angle)
            else:
                x1m_pad = x1m_m1
                conds_pad = conds

            xt = x1m_pad
            xin = torch.cat([xt, x1m_pad, conds_pad, t_map], dim=1)
            pred_m1 = model(xin)

            if pred_m1.shape[-2:] != (Ht, Wt):
                pred_m1 = F.interpolate(pred_m1, size=(Ht, Wt), mode="bilinear", align_corners=False)
            if (Ht, Wt) != (H, W):
                pred_m1 = pred_m1[..., :H, :W]

        pred01 = ((pred_m1.clamp(-1, 1) + 1.0) * 0.5).detach().cpu()
        gt01 = x0.detach().cpu()
        noisy01 = x1_img.detach().cpu()

        Hm = min(pred01.shape[-2], gt01.shape[-2], noisy01.shape[-2])
        Wm = min(pred01.shape[-1], gt01.shape[-1], noisy01.shape[-1])

        p = pred01[0, 0, :Hm, :Wm].numpy()
        g = gt01[0, 0, :Hm, :Wm].numpy()
        nimg = noisy01[0, 0, :Hm, :Wm].numpy()

        out_path = os.path.join(self.ckpt_dir, f"preview_epoch{epoch:03d}.png")
        _save_quad_preview(nimg, p, g, out_path, title=f"epoch={epoch} | infer={self.val_infer}")

        self.ema.restore(model)

    def fit(self, model: nn.Module, train_loader, val_loader=None):
        from tqdm import tqdm

        model.to(self.device).train()

        self.ema = EMA(model, decay=self.ema_decay)
        opt = optim.AdamW(model.parameters(), lr=self.lr)

        if self.sched_cfg.type == "cosine":
            sched = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=int(self.sched_cfg.T_max), eta_min=float(self.sched_cfg.eta_min)
            )
        elif self.sched_cfg.type == "step":
            sched = optim.lr_scheduler.StepLR(
                opt, step_size=int(self.sched_cfg.step_size), gamma=float(self.sched_cfg.gamma)
            )
        else:
            sched = None

        use_amp = (self.device.type == "cuda")
        scaler = _create_grad_scaler(enabled=use_amp)

        n_batches = len(train_loader)

        if not hasattr(self, "global_step"):
            self.global_step = 0
        self._init_best_score()
        self.best_epoch = None

        preset_start = int(getattr(self, "start_epoch", 1) or 1)
        if self.resume in ("none", "false", "0") and preset_start > 1:
            start_epoch = preset_start
            self.start_epoch = start_epoch
            print(f"[RESUME] disabled; use preset start_epoch={start_epoch}", flush=True)
        else:
            self.start_epoch = self._try_resume(model, opt, sched, scaler)
            start_epoch = int(self.start_epoch) if self.start_epoch is not None else 1
            start_epoch = max(1, start_epoch)

        disable_tqdm = not sys.stdout.isatty()

        for epoch in range(start_epoch, self.epochs + 1):
            self._cur_epoch = int(epoch)

            running = 0.0
            pbar = tqdm(
                train_loader,
                total=n_batches,
                desc=f"Epoch {epoch}/{self.epochs}",
                ncols=100,
                disable=disable_tqdm
            )

            do_mem_safe_unroll = (
                self.train_unroll and (self.unroll_w > 0.0)
                and self.unroll_stopgrad
                and (int(epoch) >= int(self.unroll_warmup_epochs))
                and (int(self.unroll_steps) >= 2)
            )

            for it, batch in enumerate(pbar, 1):
                opt.zero_grad(set_to_none=True)

                if do_mem_safe_unroll:
                    # -------- memory-safe unroll: TF backward first, then unroll each step backward --------
                    with _autocast_ctx(enabled=use_amp, device_type="cuda"):
                        loss_tf, cache = self._tf_loss_and_cache(model, batch)

                    scaler.scale(loss_tf).backward()

                    K = int(self.unroll_steps)
                    w_each = float(self.unroll_w) / float(max(1, (K - 1)))

                    x_prev = cache["xt"]
                    t_prev = cache["t"]
                    x0_hat_prev = cache["x0_hat_prev"]

                    sum_u = 0.0
                    for _ in range(1, K):
                        with _autocast_ctx(enabled=use_amp, device_type="cuda"):
                            loss_u, x_prev, t_prev, x0_hat_prev = self._unroll_one_step_loss(
                                model, cache, x_prev, t_prev, x0_hat_prev
                            )

                        sum_u += float(loss_u.detach().item())
                        scaler.scale(loss_u * w_each).backward()

                    loss_for_log = loss_tf.detach() + loss_tf.new_tensor(sum_u * w_each)

                    # release big refs
                    del cache, x_prev, t_prev, x0_hat_prev, loss_u

                    scaler.step(opt)
                    scaler.update()
                    self.ema.update(model)

                    running += float(loss_for_log.item())
                    cur_loss_val = float(loss_for_log.item())

                else:
                    # -------- normal path (no unroll, or legacy stopgrad=False unroll inside _step_loss) --------
                    with _autocast_ctx(enabled=use_amp, device_type="cuda"):
                        loss = self._step_loss(model, batch)

                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    self.ema.update(model)

                    running += float(loss.item())
                    cur_loss_val = float(loss.item())

                self.global_step += 1

                lr_now = opt.param_groups[0]["lr"]
                avg = running / it

                if not disable_tqdm:
                    pbar.set_postfix(loss=f"{cur_loss_val:.4f}", avg=f"{avg:.4f}", lr=f"{lr_now:.2e}")

                if self.log_interval > 0 and (it % self.log_interval == 0):
                    print(f"[Epoch {epoch} | Step {it}/{n_batches}] loss={cur_loss_val:.4f} lr={lr_now:.2e}", flush=True)

            if sched is not None:
                sched.step()

            epoch_avg = running / max(1, n_batches)
            print(f"Epoch {epoch} mean loss: {epoch_avg:.4f} | lr={opt.param_groups[0]['lr']:.8e}", flush=True)

            do_val = (val_loader is not None) and (self.eval_every > 0) and ((epoch % self.eval_every) == 0)

            if do_val:
                val_res = self._eval_val(model, val_loader, max_batches=self.val_max_batches)
                val_loss, vpsnr, vssim = val_res["val_loss"], val_res["psnr"], val_res["ssim"]
                psnr_str = f"{vpsnr:.3f} dB" if vpsnr is not None else "NA"
                ssim_str = f"{vssim:.4f}" if vssim is not None else "NA"
                print(f"[VAL {epoch}] infer={self.val_infer} loss={val_loss:.6f}  PSNR={psnr_str}  SSIM={ssim_str}", flush=True)
            else:
                val_loss, vpsnr, vssim = None, None, None
                if val_loader is not None:
                    print(f"[VAL {epoch}] skipped (eval_every={self.eval_every})", flush=True)

            self._log_row(
                epoch=epoch,
                step=self.global_step,
                lr=opt.param_groups[0]["lr"],
                loss=epoch_avg,
                val_loss=val_loss,
                psnr=vpsnr,
                ssim=vssim,
            )

            score = None
            if self.val_metric == "loss" and val_loss is not None:
                score = (-val_loss) if self.maximize_metric else val_loss
            elif self.val_metric == "psnr" and vpsnr is not None:
                score = (vpsnr) if self.maximize_metric else (-vpsnr)
            elif self.val_metric == "ssim" and vssim is not None:
                score = (vssim) if self.maximize_metric else (-vssim)

            is_better = False
            if score is not None:
                if (self.maximize_metric and score > self.best_score) or ((not self.maximize_metric) and score < self.best_score):
                    self.best_score = score
                    self.best_epoch = epoch
                    is_better = True

            if is_better:
                self._save_ckpt(model, epoch, tag="best", opt=opt, sched=sched, scaler=scaler)

            if self.val_metric == "loss" and (val_loss is not None) and (self.best_epoch is not None):
                best_disp = (-self.best_score) if self.maximize_metric else self.best_score
                print(f"[BEST] metric={best_disp:.4f} @ epoch {self.best_epoch}", flush=True)
            elif self.val_metric in ("psnr", "ssim") and (self.best_epoch is not None) and (vpsnr is not None or vssim is not None):
                best_disp = (self.best_score if self.maximize_metric else -self.best_score)
                print(f"[BEST] metric={best_disp:.4f} @ epoch {self.best_epoch}", flush=True)

            self._dump_preview(model, val_loader if val_loader is not None else train_loader, epoch)

            if (epoch % self.save_every) == 0:
                self._save_ckpt(model, epoch, opt=opt, sched=sched, scaler=scaler)
                self._prune_ckpt()

            if epoch == self.epochs:
                self._save_ckpt(model, epoch, tag="last", opt=opt, sched=sched, scaler=scaler)
