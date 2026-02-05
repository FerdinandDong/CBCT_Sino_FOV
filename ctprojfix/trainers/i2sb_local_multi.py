# ctprojfix/trainers/i2sb_local_multi.py
# multi step I2SB trainer (random-t bridge training, one-step val/preview)
# -*- coding: utf-8 -*-
import os, csv, sys
import glob, re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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


def _percentile_norm01(a, p_lo=1.0, p_hi=99.0):
    import numpy as np
    a = a.astype("float32")
    lo, hi = np.percentile(a, [p_lo, p_hi])
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max() + 1e-6)
    return ((a - lo) / (hi - lo + 1e-6)).clip(0, 1).astype("float32")


def _save_triptych(noisy01, pred01, gt01, out_path, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = _percentile_norm01(noisy01)
    p = _percentile_norm01(pred01)
    g = _percentile_norm01(gt01)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(n, cmap="gray"); axes[0].set_title("Noisy"); axes[0].axis("off")
    axes[1].imshow(p, cmap="gray"); axes[1].set_title("Pred");  axes[1].axis("off")
    axes[2].imshow(g, cmap="gray"); axes[2].set_title("GT");    axes[2].axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PREVIEW] {out_path}")


def _pad_conds(mask_and_maybe_angle: torch.Tensor, pad, has_angle: bool) -> torch.Tensor:
    """
    conds: [mask, (angle)] in [0,1]
    - mask pad: constant 0  (pad 区必须视为 missing)
    - angle pad: replicate (更合理：边缘延拓)
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


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _latest_ckpt(ckpt_dir: str, prefix: str) -> Optional[str]:
    patt = os.path.join(ckpt_dir, f"{prefix}_epoch*.pth")
    files = glob.glob(patt)
    if not files:
        return None
    files.sort(key=_natural_key)
    return files[-1]


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
    VGG16 feature perceptual loss. 默认用 relu1_2, relu2_2, relu3_3（更省显存/更快）。
    支持 weight_map（B,1,H,W）做空间加权（比如只算 missing 区）。
    """
    def __init__(self, layers: Optional[List[int]] = None, use_pretrained: bool = True):
        super().__init__()
        try:
            from torchvision import models
            from torchvision.models import VGG16_Weights
        except Exception as e:
            raise RuntimeError(
                "Perceptual loss 需要 torchvision。请先安装 torchvision，或关闭 use_percep。"
            ) from e

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
        eval_every: int = 1,              # 每 N 个 epoch 才跑一次 val（1=每个 epoch）
        val_max_batches: int = 0,         # val 最多跑 N 个 batch（0=全量）

        # --- resume ---
        resume: str = "auto",             # "auto" | "none"
        resume_from: Optional[str] = None,
        strict_load: bool = True,
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

        # states
        self._recent_ckpts = []
        self.best_epoch: Optional[int] = None
        self._init_best_score()
        self.start_epoch = 1
        self.global_step = 0

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

    # --------- I2SB 训练 Loss (Random t bridge) ---------
    def _step_loss(self, model: nn.Module, batch) -> torch.Tensor:
        """
        Bridge training (I2SB-style):
        - x0: GT clean projection in [0,1]
        - x1_img: degraded/truncated/noisy projection in [0,1]
        - conds: mask(+angle) in [0,1]  (mask: 1=valid center, 0=missing sides)
        - xt = (1-t)*x0 + t*x1 + sigma_T*sqrt(t(1-t))*eps
        - net input: [xt, x1_img, conds, t_map]
        """
        x0 = batch["gt"].to(self.device).float()      # (B,1,H,W) in [0,1]
        x1 = batch["inp"].to(self.device).float()     # (B,C,H,W) in [0,1], C=2 or 3 (noisy,mask,(angle))
        B, C, H, W = x1.shape

        x1_img = x1[:, 0:1, ...]     # (B,1,H,W)
        conds  = x1[:, 1:,  ...]     # (B,C-1,H,W) -> mask(+angle)

        # images -> [-1,1]
        x0_m1  = x0 * 2.0 - 1.0
        x1m_m1 = x1_img * 2.0 - 1.0

        # pad to UNet factor
        m = getattr(model, "module", model)
        n_down = len(getattr(m, "downs", []))
        factor = 2 ** max(n_down, 0)
        Ht = ((H + factor - 1) // factor) * factor
        Wt = ((W + factor - 1) // factor) * factor

        if (Ht, Wt) != (H, W):
            pad = (0, Wt - W, 0, Ht - H)
            x0_m1  = F.pad(x0_m1,  pad, mode="reflect")
            x1m_m1 = F.pad(x1m_m1, pad, mode="reflect")
            conds  = _pad_conds(conds, pad, has_angle=self.cond_has_angle)

        # sample t and build bridge xt
        t = torch.rand(B, 1, 1, 1, device=self.device) * (1.0 - self.t0) + self.t0
        noise = torch.randn_like(x0_m1)
        std = self.sigma_T * torch.sqrt(t * (1.0 - t))
        xt = (1.0 - t) * x0_m1 + t * x1m_m1 + std * noise

        # net input
        t_map = t.expand(B, 1, Ht, Wt)
        net_in = torch.cat([xt, x1m_m1, conds, t_map], dim=1)

        # forward: predict x0 in [-1,1]
        x0_hat = model(net_in)

        # crop back
        if (Ht, Wt) != (H, W):
            x0_hat = x0_hat[..., :H, :W]
            x0_ref = x0_m1[..., :H, :W]
            conds_crop = conds[..., :H, :W]
        else:
            x0_ref = x0_m1
            conds_crop = conds

        # mask: 1=valid(center), 0=missing(sides)
        mask = conds_crop[:, 0:1, ...].clamp(0.0, 1.0)
        missing = (1.0 - mask).clamp(0.0, 1.0)

        # ---------------- pixel loss (weighted MSE) ----------------
        weights = self.w_valid * mask + self.w_missing * missing
        diff2 = (x0_hat - x0_ref) ** 2
        loss_pix = (diff2 * weights).sum() / weights.sum().clamp_min(1e-6)

        # ---------------- perceptual loss (optional) ----------------
        loss = loss_pix
        if self.use_percep and (self.percep is not None) and (self.w_percep > 0):
            if self.percep_region == "missing":
                wmap = missing
            elif self.percep_region == "valid":
                wmap = mask
            else:
                wmap = weights

            # perceptual 分支降采样（只缩小，不放大）
            x0_hat_p = _resize_max_hw(x0_hat, self.percep_max_hw)
            x0_ref_p = _resize_max_hw(x0_ref, self.percep_max_hw)
            wmap_p   = _resize_max_hw(wmap,   self.percep_max_hw)

            if self.device.type == "cuda":
                with torch.cuda.amp.autocast(enabled=False):
                    loss_perc = self.percep(x0_hat_p, x0_ref_p, weight_map=wmap_p)
            else:
                loss_perc = self.percep(x0_hat_p, x0_ref_p, weight_map=wmap_p)

            loss = loss + self.w_percep * loss_perc

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
        """
        新版 ckpt：包含 model/epoch + optimizer/scheduler/scaler/ema/global_step/best
        兼容旧版：只存 state_dict/epoch 的 ckpt 也能 load。
        """
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

        # EMA shadow
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
        """
        返回 start_epoch（续训从 last_epoch+1 开始）
        resume:
          - "none": 禁用
          - "auto": 自动找最新 prefix_epoch*.pth
        resume_from:
          - 指定 ckpt 路径（优先级最高）
        """
        if self.resume in ("none", "false", "0"):
            print("[RESUME] disabled (resume=none).", flush=True)
            return 1

        ckpt_path = None
        if self.resume_from:
            ckpt_path = self.resume_from
        elif self.resume == "auto":
            ckpt_path = _latest_ckpt(self.ckpt_dir, self.ckpt_prefix)

        if (not ckpt_path) or (not os.path.isfile(ckpt_path)):
            print("[RESUME] no checkpoint found; start from scratch.", flush=True)
            return 1

        print(f"[RESUME] loading: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # 兼容：可能直接就是 state_dict
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=self.strict_load)

        # optimizer
        if ("optimizer" in ckpt) and (opt is not None):
            try:
                opt.load_state_dict(ckpt["optimizer"])
                print("[RESUME] optimizer restored.", flush=True)
            except Exception as e:
                print(f"[RESUME] optimizer restore failed: {e}", flush=True)

        # scheduler
        if ("scheduler" in ckpt) and (sched is not None):
            try:
                sched.load_state_dict(ckpt["scheduler"])
                print("[RESUME] scheduler restored.", flush=True)
            except Exception as e:
                print(f"[RESUME] scheduler restore failed: {e}", flush=True)

        # scaler
        if ("scaler" in ckpt) and (scaler is not None):
            try:
                scaler.load_state_dict(ckpt["scaler"])
                print("[RESUME] scaler restored.", flush=True)
            except Exception as e:
                print(f"[RESUME] scaler restore failed: {e}", flush=True)

        # global_step / best
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

        # EMA shadow
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

    # --------- 验证 (one-step t=1.0 for monitoring) ---------
    @torch.no_grad()
    def _eval_val(self, model: nn.Module, val_loader, max_batches: int = 0) -> Dict[str, float]:
        if val_loader is None:
            return {"val_loss": None, "psnr": None, "ssim": None}

        model.eval()
        self.ema.apply_shadow(model)

        s_loss, s_psnr, s_ssim, n = 0.0, 0.0, 0.0, 0
        metric_missing_only = True

        m = getattr(model, "module", model)
        n_down = len(getattr(m, "downs", []))
        factor = 2 ** max(n_down, 0)

        bcount = 0
        for batch in val_loader:
            bcount += 1
            if (max_batches is not None) and (max_batches > 0) and (bcount > max_batches):
                break

            # val loss uses the same _step_loss (includes perceptual if enabled)
            loss = self._step_loss(model, batch).item()
            s_loss += loss

            # one-step inference at t=1
            x0 = batch["gt"].to(self.device).float()
            x1 = batch["inp"].to(self.device).float()
            B, C, H, W = x1.shape

            x1_img = x1[:, 0:1, ...]
            conds = x1[:, 1:, ...]  # mask(+angle)
            x1m_m1 = x1_img * 2.0 - 1.0

            Ht = _ceil_to(H, factor)
            Wt = _ceil_to(W, factor)
            t_map = torch.ones((B, 1, H, W), device=self.device, dtype=torch.float32)

            if (Ht, Wt) != (H, W):
                pad = (0, Wt - W, 0, Ht - H)
                x1m_m1 = F.pad(x1m_m1, pad, mode="reflect")
                t_map = F.pad(t_map, pad, mode="reflect")
                conds = _pad_conds(conds, pad, has_angle=self.cond_has_angle)

            xt = x1m_m1
            xin = torch.cat([xt, x1m_m1, conds, t_map], dim=1)
            pred = model(xin)

            if pred.shape[-2:] != (Ht, Wt):
                pred = F.interpolate(pred, size=(Ht, Wt), mode="bilinear", align_corners=False)
            if (Ht, Wt) != (H, W):
                pred = pred[..., :H, :W]
                conds_crop = conds[..., :H, :W]
            else:
                conds_crop = conds

            pred01 = ((pred.clamp(-1, 1) + 1.0) * 0.5).detach()
            gt01 = x0.detach()

            if metric_missing_only:
                mask = conds_crop[:, 0:1, ...].clamp(0.0, 1.0)
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

    # --------- 预览图导出 (one-step) ---------
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
        x1m_m1 = x1_img * 2.0 - 1.0

        m = getattr(model, "module", model)
        n_down = len(getattr(m, "downs", []))
        factor = 2 ** max(n_down, 0)

        with _autocast_ctx(enabled=(self.device.type == "cuda"), device_type="cuda"):
            t_map = torch.ones((B, 1, H, W), device=self.device, dtype=torch.float32)
            Ht = ((H + factor - 1) // factor) * factor
            Wt = ((W + factor - 1) // factor) * factor

            if (Ht, Wt) != (H, W):
                pad = (0, Wt - W, 0, Ht - H)
                x1m_m1 = F.pad(x1m_m1, pad, mode="reflect")
                t_map = F.pad(t_map, pad, mode="reflect")
                conds = _pad_conds(conds, pad, has_angle=self.cond_has_angle)

            xt = x1m_m1
            xin = torch.cat([xt, x1m_m1, conds, t_map], dim=1)

            pred = model(xin)
            if pred.shape[-2:] != (Ht, Wt):
                pred = F.interpolate(pred, size=(Ht, Wt), mode="bilinear", align_corners=False)
            if (Ht, Wt) != (H, W):
                pred = pred[..., :H, :W]

            pred01 = ((pred.clamp(-1, 1) + 1.0) * 0.5).detach().cpu()
            gt01 = x0.detach().cpu()
            noisy01 = x1_img.detach().cpu()

            Hm = min(pred01.shape[-2], gt01.shape[-2], noisy01.shape[-2])
            Wm = min(pred01.shape[-1], gt01.shape[-1], noisy01.shape[-1])

            p = pred01[0, 0, :Hm, :Wm].numpy()
            g = gt01[0, 0, :Hm, :Wm].numpy()
            nimg = noisy01[0, 0, :Hm, :Wm].numpy()

            out_path = os.path.join(self.ckpt_dir, f"preview_epoch{epoch:03d}.png")
            _save_triptych(nimg, p, g, out_path, title=f"epoch={epoch}")

        self.ema.restore(model)

    def fit(self, model: nn.Module, train_loader, val_loader=None):
        from tqdm import tqdm

        model.to(self.device).train()

        # init EMA/opt/sched/scaler
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

        # init states (then try resume to overwrite)
        if not hasattr(self, "global_step"):
            self.global_step = 0
        self._init_best_score()
        self.best_epoch = None

        # resume
        self.start_epoch = self._try_resume(model, opt, sched, scaler)
        start_epoch = int(self.start_epoch) if self.start_epoch is not None else 1
        start_epoch = max(1, start_epoch)

        disable_tqdm = not sys.stdout.isatty()

        for epoch in range(start_epoch, self.epochs + 1):
            running = 0.0
            pbar = tqdm(
                train_loader,
                total=n_batches,
                desc=f"Epoch {epoch}/{self.epochs}",
                ncols=100,
                disable=disable_tqdm
            )

            for it, batch in enumerate(pbar, 1):
                opt.zero_grad(set_to_none=True)
                with _autocast_ctx(enabled=use_amp, device_type="cuda"):
                    loss = self._step_loss(model, batch)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                self.ema.update(model)

                running += loss.item()
                self.global_step += 1

                lr_now = opt.param_groups[0]["lr"]
                avg = running / it

                if not disable_tqdm:
                    pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg:.4f}", lr=f"{lr_now:.2e}")

                if self.log_interval > 0 and (it % self.log_interval == 0):
                    print(f"[Epoch {epoch} | Step {it}/{n_batches}] loss={loss.item():.4f} lr={lr_now:.2e}", flush=True)

            if sched is not None:
                sched.step()

            epoch_avg = running / max(1, n_batches)
            print(f"Epoch {epoch} mean loss: {epoch_avg:.4f} | lr={opt.param_groups[0]['lr']:.8e}", flush=True)

            # ---- val control: only run every eval_every epochs ----
            do_val = (val_loader is not None) and (self.eval_every > 0) and ((epoch % self.eval_every) == 0)

            if do_val:
                val_res = self._eval_val(model, val_loader, max_batches=self.val_max_batches)
                val_loss, vpsnr, vssim = val_res["val_loss"], val_res["psnr"], val_res["ssim"]
                psnr_str = f"{vpsnr:.3f} dB" if vpsnr is not None else "NA"
                ssim_str = f"{vssim:.4f}" if vssim is not None else "NA"
                print(f"[VAL {epoch}] loss={val_loss:.6f}  PSNR={psnr_str}  SSIM={ssim_str}", flush=True)
            else:
                val_loss, vpsnr, vssim = None, None, None
                if val_loader is not None:
                    print(f"[VAL {epoch}] skipped (eval_every={self.eval_every})", flush=True)

            # log csv
            self._log_row(
                epoch=epoch,
                step=self.global_step,
                lr=opt.param_groups[0]["lr"],
                loss=epoch_avg,
                val_loss=val_loss,
                psnr=vpsnr,
                ssim=vssim,
            )

            # best selection only when val exists
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

            # preview (independent frequency)
            self._dump_preview(model, val_loader if val_loader is not None else train_loader, epoch)

            # save periodic ckpt
            if (epoch % self.save_every) == 0:
                self._save_ckpt(model, epoch, opt=opt, sched=sched, scaler=scaler)
                self._prune_ckpt()

            if epoch == self.epochs:
                self._save_ckpt(model, epoch, tag="last", opt=opt, sched=sched, scaler=scaler)
