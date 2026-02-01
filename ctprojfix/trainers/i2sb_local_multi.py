# ctprojfix/trainers/i2sb_local_multi.py
# multi step I2SB trainer
# -*- coding: utf-8 -*-
import os, csv, math, sys
from dataclasses import dataclass
from typing import Optional, Dict, Any

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
    import numpy as np

    n = _percentile_norm01(noisy01)
    p = _percentile_norm01(pred01)
    g = _percentile_norm01(gt01)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(n, cmap="gray"); axes[0].set_title("Noisy"); axes[0].axis("off")
    axes[1].imshow(p, cmap="gray"); axes[1].set_title("Pred");  axes[1].axis("off")
    axes[2].imshow(g, cmap="gray"); axes[2].set_title("GT");    axes[2].axis("off")
    if title: fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PREVIEW] {out_path}")


class I2SBLocalTrainer:
    def __init__(self,
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
                 ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.sigma_T = float(sigma_T)
        self.t0 = float(t0)
        self.ema_decay = float(ema_decay)
        self.cond_has_angle = bool(cond_has_angle)

        self.ckpt_dir = ckpt_dir
        self.ckpt_prefix = ckpt_prefix
        self.save_every = int(save_every)
        self.max_keep = int(max_keep)
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

        self._recent_ckpts = []
        self.best_epoch: Optional[int] = None
        self._init_best_score()

    def _init_best_score(self):
        if self.maximize_metric:
            self.best_score = -float("inf")
        else:
            self.best_score = float("inf")

    # --------- [修改] I2SB 训练 Loss (Multi-step Random t) ---------
    def _step_loss(self, model: nn.Module, batch) -> torch.Tensor:
        x0 = batch["gt"].to(self.device).float()      # (B,1,H,W) Clean
        x1 = batch["inp"].to(self.device).float()     # (B,C,H,W) Noisy Stack [img, mask, angle...]
        B, C, H, W = x1.shape

        # 归一化到 [-1, 1]
        x0 = x0 * 2.0 - 1.0
        x1 = x1 * 2.0 - 1.0

        # --- 1. Pad ---
        n_down = len(getattr(model, "downs", []))
        factor = 2 ** max(n_down, 0)
        Ht = ((H + factor - 1) // factor) * factor
        Wt = ((W + factor - 1) // factor) * factor

        if (Ht, Wt) != (H, W):
            pad = (0, Wt - W, 0, Ht - H)
            x0 = F.pad(x0, pad, mode="reflect")
            x1 = F.pad(x1, pad, mode="reflect")

        # --- 2. 准备 I2SB 条件 ---
        # 拆分: x1_img 是要加噪的图像, conditions (mask/angle) 保持不变
        x1_img = x1[:, 0:1, ...] 
        conditions = x1[:, 1:, ...]

        # --- 3. 随机采样 t ~ [t0, 1] ---
        # shape: (B, 1, 1, 1) 方便广播
        t = torch.rand(B, 1, 1, 1, device=self.device) * (1 - self.t0) + self.t0
        
        # --- 4. 构建 xt (Forward Process) ---
        # I2SB: xt = (1-t)x0 + t*x1 + noise
        noise = torch.randn_like(x0)
        # std = sqrt(t(1-t))，这是 I2SB 常用的一种 bridge 设定
        std = torch.sqrt(t * (1 - t))
        xt = (1 - t) * x0 + t * x1_img + std * noise

        # --- 5. 构造网络输入 ---
        # t_map 扩展到全图
        t_map = t.expand(B, 1, Ht, Wt)
        
        # 现在的 input 是: [xt, mask, angle, t_map]
        # 注意：我们将 xt 放回原 x1_img 的位置，这样通道数依然是 4
        net_in = torch.cat([xt, conditions, t_map], dim=1)

        # --- 6. 预测 & Loss ---
        # 目标：预测 x0
        x0_hat = model(net_in)

        # 裁回
        if (H, W) != (Ht, Wt):
            # 注意: 如果只求 Loss，不一定要裁回去，只要 x0 也保留 padded 状态即可
            # 但为了逻辑一致，我们通常只计算有效区域
            x0_hat = x0_hat[..., :H, :W]
            x0_orig = x0[..., :H, :W] # 取未 pad 之前的 x0 值(已norm)
        else:
            x0_orig = x0

        # L2 Loss
        loss = torch.mean((x0_hat - x0_orig) ** 2)
        return loss

    def _log_row(self, epoch, step, lr, loss, val_loss, psnr, ssim):
        with open(self.log_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                int(epoch),
                int(step),
                float(lr) if lr is not None else "",
                float(loss) if loss is not None else "",
                float(val_loss) if val_loss is not None else "",
                float(psnr) if psnr is not None else "",
                float(ssim) if ssim is not None else "",
            ])

    def _save_ckpt(self, model: nn.Module, epoch: int, tag: Optional[str] = None):
        obj = {"state_dict": model.state_dict(), "epoch": epoch}
        name = f"{self.ckpt_prefix}_{'epoch'+str(epoch) if tag is None else tag}.pth"
        path = os.path.join(self.ckpt_dir, name)
        torch.save(obj, path)
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

    # --------- 验证 (保持 One-step t=1.0 逻辑用于快速监控) ---------
    @torch.no_grad()
    def _eval_val(self, model: nn.Module, val_loader) -> Dict[str, float]:
        if val_loader is None:
            return {"val_loss": None, "psnr": None, "ssim": None}
        model.eval()
        self.ema.apply_shadow(model)
        s_loss, s_psnr, s_ssim, n = 0.0, 0.0, 0.0, 0

        for batch in val_loader:
            # Loss 依然调用 step_loss (随机 t)，但这只为了监控 loss 下降趋势
            # 如果想看确定的 loss，可以临时 mock t=1，但这里保持原样简单
            loss = self._step_loss(model, batch).item()
            s_loss += loss

            # PSNR/SSIM 计算：这里我们模拟 "One-step Inference at t=1.0"
            # 也就是把 Noisy 直接喂进去，看模型能不能直接恢复 (One-step consistency)
            x0 = batch["gt"].to(self.device).float()       # [0,1]
            x1 = batch["inp"].to(self.device).float()      # [0,1]
            B, _, H, W = x1.shape

            t_map = torch.ones((B, 1, H, W), device=self.device, dtype=torch.float32)
            x1m1 = x1 * 2.0 - 1.0

            n_down = len(getattr(model, "downs", []))
            factor = 2 ** max(n_down, 0)
            Ht = _ceil_to(H, factor)
            Wt = _ceil_to(W, factor)
            if (Ht, Wt) != (H, W):
                pad = (0, Wt - W, 0, Ht - H)
                x1m1 = F.pad(x1m1, pad, mode="reflect")
                t_map = F.pad(t_map, pad, mode="reflect")

            # 这里直接喂 x1 (相当于 t=1, xt=x1)，看 One-step 效果
            xin = torch.cat([x1m1, t_map], dim=1)
            pred = model(xin)
            if pred.shape[-2:] != (Ht, Wt):
                pred = F.interpolate(pred, size=(Ht, Wt), mode="bilinear", align_corners=False)
            if (H, W) != (Ht, Wt):
                pred = pred[..., :H, :W]

            pred01 = ((pred.clamp(-1, 1) + 1.0) * 0.5).detach()
            gt01 = x0.detach()

            for i in range(pred01.shape[0]):
                s_psnr += float(psnr_fn(pred01[i, 0].cpu().numpy(), gt01[i, 0].cpu().numpy()))
                s_ssim += float(ssim_fn(pred01[i, 0].cpu().numpy(), gt01[i, 0].cpu().numpy()))
                n += 1

        self.ema.restore(model)
        if n == 0:
            return {"val_loss": s_loss, "psnr": None, "ssim": None}
        return {"val_loss": s_loss / max(1, len(val_loader)),
                "psnr": s_psnr / n,
                "ssim": s_ssim / n}

    # --------- 预览图导出 (同理保持 One-step) ---------
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
        B, _, H, W = x1.shape

        with _autocast_ctx(enabled=(self.device.type == "cuda"), device_type="cuda"):
            # Preview 依然使用 t=1.0 展示 "单步去噪" 能力
            t_map = torch.ones((B, 1, H, W), device=self.device, dtype=torch.float32)
            x1m1 = x1 * 2.0 - 1.0

            n_down = len(getattr(model, "downs", []))
            factor = 2 ** max(n_down, 0)
            Ht = ((H + factor - 1) // factor) * factor
            Wt = ((W + factor - 1) // factor) * factor
            if (Ht, Wt) != (H, W):
                pad = (0, Wt - W, 0, Ht - H)
                x1m1 = F.pad(x1m1, pad, mode="reflect")
                t_map = F.pad(t_map, pad, mode="reflect")

            xin = torch.cat([x1m1, t_map], dim=1)
            pred = model(xin)
            if pred.shape[-2:] != (Ht, Wt):
                pred = F.interpolate(pred, size=(Ht, Wt), mode="bilinear", align_corners=False)
            if (H, W) != (Ht, Wt):
                pred = pred[..., :H, :W]

            pred01 = ((pred.clamp(-1, 1) + 1.0) * 0.5).detach().cpu()
            gt01   = x0.detach().cpu()
            noisy01= x1[:, 0:1].detach().cpu() if x1.shape[1] >= 1 else gt01

            hH, hW = pred01.shape[-2], pred01.shape[-1]
            gH, gW = gt01.shape[-2],   gt01.shape[-1]
            nH, nW = noisy01.shape[-2],noisy01.shape[-1]
            Hm, Wm = min(hH, gH, nH), min(hW, gW, nW)
            p = pred01[0, 0, :Hm, :Wm].numpy()
            g = gt01[0, 0, :Hm, :Wm].numpy()
            n = noisy01[0, 0, :Hm, :Wm].numpy()

            out_path = os.path.join(self.ckpt_dir, f"preview_epoch{epoch:03d}.png")
            _save_triptych(n, p, g, out_path, title=f"epoch={epoch}")

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
        global_step = 0

        self._init_best_score()
        self.best_epoch = None

        disable_tqdm = not sys.stdout.isatty()

        for epoch in range(1, self.epochs + 1):
            running = 0.0
            pbar = tqdm(train_loader, total=n_batches,
                        desc=f"Epoch {epoch}/{self.epochs}",
                        ncols=100, disable=disable_tqdm)

            for it, batch in enumerate(pbar, 1):
                opt.zero_grad(set_to_none=True)
                with _autocast_ctx(enabled=use_amp, device_type="cuda"):
                    loss = self._step_loss(model, batch)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                self.ema.update(model)

                running += loss.item()
                global_step += 1
                lr_now = opt.param_groups[0]["lr"]
                avg = running / it

                if not disable_tqdm:
                    pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg:.4f}", lr=f"{lr_now:.2e}")

                if self.log_interval > 0 and (it % self.log_interval == 0):
                    print(f"[Epoch {epoch} | Step {it}/{n_batches}] loss={loss.item():.4f} lr={lr_now:.2e}", flush=True)

            if sched is not None:
                sched.step()

            val_res = self._eval_val(model, val_loader) if val_loader is not None else {"val_loss": None, "psnr": None, "ssim": None}
            val_loss, vpsnr, vssim = val_res["val_loss"], val_res["psnr"], val_res["ssim"]

            epoch_avg = running / max(1, n_batches)

            print(f"Epoch {epoch} mean loss: {epoch_avg:.4f} | lr={opt.param_groups[0]['lr']:.8e}", flush=True)
            if val_loss is not None:
                psnr_str = f"{vpsnr:.3f} dB" if vpsnr is not None else "NA"
                ssim_str = f"{vssim:.4f}"    if vssim is not None else "NA"
                print(f"[VAL {epoch}] loss={val_loss:.6f}  PSNR={psnr_str}  SSIM={ssim_str}", flush=True)

            self._log_row(epoch=epoch, step=global_step, lr=opt.param_groups[0]["lr"],
                          loss=epoch_avg,
                          val_loss=(val_loss if val_loss is not None else ""),
                          psnr=(vpsnr if vpsnr is not None else ""),
                          ssim=(vssim if vssim is not None else ""))

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
                self._save_ckpt(model, epoch, tag="best")

            if self.val_metric == "loss" and val_loss is not None:
                best_disp = (-self.best_score) if self.maximize_metric else self.best_score
                be = self.best_epoch if self.best_epoch is not None else epoch
                print(f"[BEST] metric={best_disp:.4f} @ epoch {be}", flush=True)
            elif self.val_metric in ("psnr", "ssim") and (vpsnr is not None or vssim is not None):
                best_disp = (self.best_score if self.maximize_metric else -self.best_score)
                be = self.best_epoch if self.best_epoch is not None else epoch
                print(f"[BEST] metric={best_disp:.4f} @ epoch {be}", flush=True)

            self._dump_preview(model, val_loader if val_loader is not None else train_loader, epoch)

            if (epoch % self.save_every) == 0:
                self._save_ckpt(model, epoch)
                self._prune_ckpt()

            if epoch == self.epochs:
                self._save_ckpt(model, epoch, tag="last")