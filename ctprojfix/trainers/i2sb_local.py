# ctprojfix/trainers/i2sb_local.py
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
    """
    优先使用 torch>=2.0 的 torch.amp.GradScaler，旧环境回落到 torch.cuda.amp.GradScaler
    """
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)  # 新 API
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)     # 旧 API

class _autocast_ctx:
    """
    优先使用 torch>=2.0 的 torch.amp.autocast，旧环境回落到 torch.cuda.amp.autocast
    """
    def __init__(self, enabled: bool, device_type: str = "cuda"):
        self.enabled = enabled
        self.device_type = device_type
        self.ctx = None

    def __enter__(self):
        if self.enabled:
            try:
                self.ctx = torch.amp.autocast(self.device_type)   # 新 API
            except Exception:
                self.ctx = torch.cuda.amp.autocast()              # 旧 API
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
    type: Optional[str] = None        # 'cosine' | 'step' | None
    T_max: int = 150                  # for cosine
    eta_min: float = 1e-6             # for cosine
    step_size: int = 50               # for step
    gamma: float = 0.5                # for step


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
    """
    保存单张三联图（Noisy/Pred/GT），输入均为 [0,1]。
    """
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
    """
    让 I2SB(本地) 的训练“表现”与 SupervisedTrainer 保持一致：
      - 日志落地: logs/logplots/<ckpt_prefix>/train.csv
      - 指标: loss/psnr/ssim（val）
      - 最优模型: *_best.pth，最后: *_last.pth
      - sched 字段与 SupervisedTrainer 对齐
      - cfg 新增:
          log_interval: int
          val_metric: 'loss'|'psnr'|'ssim'
          maximize_metric: bool
          dump_preview_every: int (>0 时每 N 个 epoch 在 ckpt_dir 下输出预览三联图)
    """
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
                 val_metric: str = "loss",           # 'loss' | 'psnr' | 'ssim'
                 maximize_metric: Optional[bool] = None,
                 sched: Optional[Dict[str, Any]] = None,
                 dump_preview_every: int = 0,        # >0 则每 N epoch 存单张三联图
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

        # 日志路径对齐 SupervisedTrainer：logs/logplots/<prefix>/train.csv
        self.log_csv_dir = os.path.join("logs", "logplots", ckpt_prefix)
        os.makedirs(self.log_csv_dir, exist_ok=True)
        self.log_csv_path = os.path.join(self.log_csv_dir, "train.csv")
        if not os.path.isfile(self.log_csv_path):
            with open(self.log_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "step", "lr", "loss", "val_loss", "psnr", "ssim"])

        # 验证指标
        self.val_metric = str(val_metric or "loss").lower()
        if maximize_metric is None:
            self.maximize_metric = (self.val_metric != "loss")
        else:
            self.maximize_metric = bool(maximize_metric)

        self.sched_cfg = SchedCfg(**(sched or {}))
        self.dump_preview_every = int(dump_preview_every)
        self.log_interval = int(log_interval)

        # 维护最近 N 个 checkpoint 路径
        self._recent_ckpts = []
        self.best_epoch: Optional[int] = None
        # best_score 初始化
        self._init_best_score()

    # --------- best score 初始化 ---------
    def _init_best_score(self):
        if self.maximize_metric:
            # 指标越大越好
            self.best_score = -float("inf")
        else:
            # 指标越小越好（loss）
            self.best_score = float("inf")

    # --------- I2SB 本地最简 loss ---------
    def _step_loss(self, model: nn.Module, batch) -> torch.Tensor:
        x0 = batch["gt"].to(self.device).float()      # (B,1,H,W)  in [0,1]
        x1 = batch["inp"].to(self.device).float()     # (B,C,H,W)  in [0,1]
        B, _, H, W = x1.shape

        # to [-1,1]
        x0 = x0 * 2.0 - 1.0
        x1 = x1 * 2.0 - 1.0

        # 对齐到 2^(depth-1)
        n_down = len(getattr(model, "downs", []))
        factor = 2 ** max(n_down, 0)
        Ht = ((H + factor - 1) // factor) * factor
        Wt = ((W + factor - 1) // factor) * factor

        # 右/下反射 pad
        if (Ht, Wt) != (H, W):
            pad = (0, Wt - W, 0, Ht - H)
            x0 = F.pad(x0, pad, mode="reflect")
            x1 = F.pad(x1, pad, mode="reflect")

        # t-map: 1-step 近似用 1.0
        t_map = torch.ones((B, 1, Ht, Wt), device=self.device, dtype=torch.float32)
        net_in = torch.cat([x1, t_map], dim=1)

        x0_hat = model(net_in)

        # 先对齐到 (Ht, Wt)
        if x0_hat.shape[-2:] != (Ht, Wt):
            x0_hat = F.interpolate(x0_hat, size=(Ht, Wt), mode="bilinear", align_corners=False)

        # 再裁回原始 (H, W)
        if (H, W) != (Ht, Wt):
            x0_hat = x0_hat[..., :H, :W]
            x0     = x0[...,     :H, :W]   # ★ 同时把 x0 裁回原尺寸

        # 兜底：若还有一丝差异（例如奇偶/整除边界），统一裁到公共最小尺寸
        hH, hW = x0_hat.shape[-2], x0_hat.shape[-1]
        gH, gW = x0.shape[-2],     x0.shape[-1]
        if (hH != gH) or (hW != gW):
            Hm, Wm = min(hH, gH), min(hW, gW)
            x0_hat = x0_hat[..., :Hm, :Wm]
            x0     = x0[...,     :Hm, :Wm]

        # L2
        loss = torch.mean((x0_hat - x0) ** 2)
        return loss

    # --------- 日志写入，和 plot_train_log.py 对齐 ---------
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

    # --------- 计算验证指标（对齐 SupervisedEvaluator）---------
    @torch.no_grad()
    def _eval_val(self, model: nn.Module, val_loader) -> Dict[str, float]:
        if val_loader is None:
            return {"val_loss": None, "psnr": None, "ssim": None}
        model.eval()
        # 使用 EMA 权重
        self.ema.apply_shadow(model)
        s_loss, s_psnr, s_ssim, n = 0.0, 0.0, 0.0, 0

        for batch in val_loader:
            # 计算 loss
            loss = self._step_loss(model, batch).item()
            s_loss += loss

            # 计算 PSNR/SSIM：需要把 net 输出和 gt 都放回 [0,1]
            x0 = batch["gt"].to(self.device).float()       # [0,1]
            x1 = batch["inp"].to(self.device).float()      # [0,1]
            B, _, H, W = x1.shape

            t_map = torch.ones((B, 1, H, W), device=self.device, dtype=torch.float32)
            x1m1 = x1 * 2.0 - 1.0

            # pad 到整齐
            n_down = len(getattr(model, "downs", []))
            factor = 2 ** max(n_down, 0)
            Ht = _ceil_to(H, factor)
            Wt = _ceil_to(W, factor)
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

            pred01 = ((pred.clamp(-1, 1) + 1.0) * 0.5).detach()
            gt01 = x0.detach()

            # 逐样本 PSNR/SSIM 累加
            for i in range(pred01.shape[0]):
                s_psnr += float(psnr_fn(pred01[i, 0].cpu().numpy(), gt01[i, 0].cpu().numpy()))
                s_ssim += float(ssim_fn(pred01[i, 0].cpu().numpy(), gt01[i, 0].cpu().numpy()))
                n += 1

        self.ema.restore(model)
        if n == 0:  # 防止除 0
            return {"val_loss": s_loss, "psnr": None, "ssim": None}
        return {"val_loss": s_loss / max(1, len(val_loader)),
                "psnr": s_psnr / n,
                "ssim": s_ssim / n}

    # --------- 预览图导出（每 N epoch 一张）---------
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

        x0 = batch["gt"].to(self.device).float()   # [0,1]
        x1 = batch["inp"].to(self.device).float()  # [0,1]
        B, _, H, W = x1.shape

        with _autocast_ctx(enabled=(self.device.type == "cuda"), device_type="cuda"):
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

            # 再兜底：统一三者的公共最小尺寸
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

    # --------- 主训练循环（含 tqdm 打印）---------
    def fit(self, model: nn.Module, train_loader, val_loader=None):
        from tqdm import tqdm

        model.to(self.device).train()
        self.ema = EMA(model, decay=self.ema_decay)
        opt = optim.AdamW(model.parameters(), lr=self.lr)

        # scheduler 对齐
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

        # best 逻辑与 SupervisedTrainer 一致
        self._init_best_score()
        self.best_epoch = None

        # tqdm 在nohup下自动禁用
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

                # —— 每 log_interval 步打印一次（nohup 可见） ——
                if self.log_interval > 0 and (it % self.log_interval == 0):
                    print(f"[Epoch {epoch} | Step {it}/{n_batches}] loss={loss.item():.4f} lr={lr_now:.2e}", flush=True)

            # 学习率步进
            if sched is not None:
                sched.step()

            # ---- 验证与日志 ----
            val_res = self._eval_val(model, val_loader) if val_loader is not None else {"val_loss": None, "psnr": None, "ssim": None}
            val_loss, vpsnr, vssim = val_res["val_loss"], val_res["psnr"], val_res["ssim"]

            epoch_avg = running / max(1, n_batches)

            # —— 与 pconv 相同的 epoch 汇总行 ——
            print(f"Epoch {epoch} mean loss: {epoch_avg:.4f} | lr={opt.param_groups[0]['lr']:.8e}", flush=True)
            if val_loss is not None:
                psnr_str = f"{vpsnr:.3f} dB" if vpsnr is not None else "NA"
                ssim_str = f"{vssim:.4f}"    if vssim is not None else "NA"
                print(f"[VAL {epoch}] loss={val_loss:.6f}  PSNR={psnr_str}  SSIM={ssim_str}", flush=True)

            # 写 CSV 日志（和 plot_train_log.py 对齐）
            self._log_row(epoch=epoch, step=global_step, lr=opt.param_groups[0]["lr"],
                          loss=epoch_avg,
                          val_loss=(val_loss if val_loss is not None else ""),
                          psnr=(vpsnr if vpsnr is not None else ""),
                          ssim=(vssim if vssim is not None else ""))

            # ---- BEST 选择逻辑 ----
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

            # BEST 提示（与 pconv 风格一致）
            if self.val_metric == "loss" and val_loss is not None:
                # 展示真实 best 的“原值”（非取反后）
                best_disp = (-self.best_score) if self.maximize_metric else self.best_score
                be = self.best_epoch if self.best_epoch is not None else epoch
                print(f"[BEST] metric={best_disp:.4f} @ epoch {be}", flush=True)
            elif self.val_metric in ("psnr", "ssim") and (vpsnr is not None or vssim is not None):
                best_disp = (self.best_score if self.maximize_metric else -self.best_score)
                be = self.best_epoch if self.best_epoch is not None else epoch
                print(f"[BEST] metric={best_disp:.4f} @ epoch {be}", flush=True)

            # —— 预览图 ——（每 N epoch 导出一张三联图）
            self._dump_preview(model, val_loader if val_loader is not None else train_loader, epoch)

            # 定期保存
            if (epoch % self.save_every) == 0:
                self._save_ckpt(model, epoch)
                self._prune_ckpt()

            # 最后保存 last
            if epoch == self.epochs:
                self._save_ckpt(model, epoch, tag="last")
