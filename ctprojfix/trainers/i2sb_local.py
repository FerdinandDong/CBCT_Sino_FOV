# ctprojfix/trainers/i2sb_local.py
# -*- coding: utf-8 -*-
import os, csv, math, time
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
        return torch.cuda.amp.GradScaler(enabled=enabled)
    except Exception:
        return torch.amp.GradScaler("cuda", enabled=enabled)

class _autocast_ctx:
    def __init__(self, enabled: bool, device_type: str = "cuda"):
        self.enabled = enabled
        self.device_type = device_type
    def __enter__(self):
        if self.enabled:
            try:
                self.ctx = torch.cuda.amp.autocast()
            except Exception:
                self.ctx = torch.amp.autocast(self.device_type)
            return self.ctx.__enter__()
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            return self.ctx.__exit__(exc_type, exc_val, exc_tb)
        return False

# ---------------- EMA ----------------
class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = decay
        self.shadow = {}
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

class I2SBLocalTrainer:
    """
    让 I2SB(本地) 的训练“表现”与 SupervisedTrainer 保持一致：
      - 日志落地: logs/logplots/<ckpt_prefix>/train.csv
      - 指标: loss/psnr/ssim（val）
      - 最优模型: *_best.pth，最后: *_last.pth
      - sched 字段与 SupervisedTrainer 对齐
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

        # 日志路径对齐 SupervisedTrainer
        # logs/logplots/<prefix>/train.csv
        self.log_csv_dir = os.path.join("logs", "logplots", ckpt_prefix)
        os.makedirs(self.log_csv_dir, exist_ok=True)
        self.log_csv_path = os.path.join(self.log_csv_dir, "train.csv")
        # 若为空，新建表头
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

        # 维护最近 N 个 checkpoint 路径
        self._recent_ckpts = []

    # --------- I2SB 本地最简 loss ---------
    def _step_loss(self, model: nn.Module, batch) -> torch.Tensor:
        """
        x0: GT in [0,1]
        inp: [noisy, mask, (angle)?] in [0,1]  → 归一化到 [-1,1]
        拼 t_map=Uniform(0,1) 单步，或固定 1.0 做 1-step 近似
        """
        x0 = batch["gt"].to(self.device).float()             # (B,1,H,W)  in [0,1]
        x1 = batch["inp"].to(self.device).float()            # (B,C,H,W)  in [0,1] , C=2/3
        B, C, H, W = x1.shape

        # to [-1,1]
        x0 = x0 * 2.0 - 1.0
        x1 = x1 * 2.0 - 1.0

        # 对齐到 2^(depth-1)
        n_down = len(getattr(model, "downs", []))
        factor = 2 ** max(n_down, 0)
        Ht = math.ceil(H / factor) * factor
        Wt = math.ceil(W / factor) * factor

        # 右/下反射 pad
        if (Ht, Wt) != (H, W):
            pad = (0, Wt - W, 0, Ht - H)
            x0 = F.pad(x0, pad, mode="reflect")
            x1 = F.pad(x1, pad, mode="reflect")

        # t-map: 1-step 近似用 1.0；也可改成 torch.rand(B) 做随机时刻训练
        t_map = torch.ones((B, 1, Ht, Wt), device=self.device, dtype=torch.float32)
        net_in = torch.cat([x1, t_map], dim=1)  # in_ch: (2/3)+1

        x0_hat = model(net_in)

        # 裁回
        if x0_hat.shape[-2:] != (Ht, Wt):
            x0_hat = F.interpolate(x0_hat, size=(Ht, Wt), mode="bilinear", align_corners=False)
        if (H, W) != (Ht, Wt):
            x0_hat = x0_hat[..., :H, :W]

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
        print(f"[CKPT] save -> {path}")
        if tag is None:
            self._recent_ckpts.append(path)

    def _prune_ckpt(self):
        while len(self._recent_ckpts) > self.max_keep:
            old = self._recent_ckpts.pop(0)
            try:
                os.remove(old)
                print(f"[CKPT] prune -> {old}")
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
            loss = self._step_loss(model, batch).item()
            s_loss += loss

            # 计算 PSNR/SSIM：需要把 net 输出和 gt 都放回 [0,1]
            x0 = batch["gt"].to(self.device).float()
            x1 = batch["inp"].to(self.device).float()
            B, _, H, W = x1.shape
            t_map = torch.ones((B, 1, H, W), device=self.device, dtype=torch.float32)
            x0m1 = x0 * 2.0 - 1.0
            x1m1 = x1 * 2.0 - 1.0
            # pad 到整齐
            n_down = len(getattr(model, "downs", []))
            factor = 2 ** max(n_down, 0)
            Ht = math.ceil(H / factor) * factor
            Wt = math.ceil(W / factor) * factor
            if (Ht, Wt) != (H, W):
                pad = (0, Wt - W, 0, Ht - H)
                x0m1 = F.pad(x0m1, pad, mode="reflect")
                x1m1 = F.pad(x1m1, pad, mode="reflect")
                t_map = F.pad(t_map, pad, mode="reflect")

            xin = torch.cat([x1m1, t_map], dim=1)
            pred = model(xin)
            if pred.shape[-2:] != (Ht, Wt):
                pred = F.interpolate(pred, size=(Ht, Wt), mode="bilinear", align_corners=False)
            if (H, W) != (Ht, Wt):
                pred = pred[..., :H, :W]

            pred01 = ((pred.clamp(-1, 1) + 1.0) * 0.5).detach()
            gt01 = x0.detach()  # 原本就 [0,1]

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
        best_score = -float("inf") if self.maximize_metric else float("inf")

        for epoch in range(1, self.epochs + 1):
            running = 0.0
            pbar = tqdm(train_loader, total=n_batches, desc=f"Epoch {epoch}/{self.epochs}", ncols=100)

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
                pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg:.4f}", lr=f"{lr_now:.2e}")

            # 学习率步进
            if sched is not None:
                sched.step()

            # ---- 验证与日志 ----
            val_res = self._eval_val(model, val_loader) if val_loader is not None else {"val_loss": None, "psnr": None, "ssim": None}
            val_loss, vpsnr, vssim = val_res["val_loss"], val_res["psnr"], val_res["ssim"]

            epoch_avg = running / max(1, n_batches)
            print_str = f"[TRAIN] epoch {epoch:03d}  loss={epoch_avg:.6f}"
            if val_loss is not None:
                print_str += f"  [VAL] loss={val_loss:.6f}"
            if vpsnr is not None:
                print_str += f"  psnr={vpsnr:.3f}"
            if vssim is not None:
                print_str += f"  ssim={vssim:.4f}"
            print(print_str)

            # 写日志（和 plot_train_log.py 对齐）
            self._log_row(epoch=epoch, step=global_step, lr=opt.param_groups[0]["lr"],
                          loss=epoch_avg, val_loss=(val_loss if val_loss is not None else ""),
                          psnr=(vpsnr if vpsnr is not None else ""),
                          ssim=(vssim if vssim is not None else ""))

            # ---- 保存 best ----
            score = None
            if self.val_metric == "loss" and val_loss is not None:
                score = -val_loss if self.maximize_metric else val_loss
            elif self.val_metric == "psnr" and vpsnr is not None:
                score = vpsnr if self.maximize_metric else -vpsnr
            elif self.val_metric == "ssim" and vssim is not None:
                score = vssim if self.maximize_metric else -vssim

            is_better = False
            if score is not None:
                if (self.maximize_metric and score > best_score) or ((not self.maximize_metric) and score < best_score):
                    best_score = score
                    is_better = True

            if is_better:
                self._save_ckpt(model, epoch, tag="best")

            # 定期保存
            if (epoch % self.save_every) == 0:
                self._save_ckpt(model, epoch)
                self._prune_ckpt()

            # 最后保存 last
            if epoch == self.epochs:
                self._save_ckpt(model, epoch, tag="last")
