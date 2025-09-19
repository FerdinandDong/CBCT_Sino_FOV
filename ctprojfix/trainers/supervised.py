# ctprojfix/trainers/supervised.py
import os, re, glob, sys
import numpy as np  # ★ 新增
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from ctprojfix.losses.criterion import build_criterion_from_cfg
from ctprojfix.evals.metrics import psnr as _psnr, ssim as _ssim  # 在 [0,1] 归一化域上评估


# ---------- ckpt / 轮转删除 ----------
def _natural_key(s: str):
    """用自然序排序：epoch10 > epoch9"""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def _latest_ckpt(ckpt_dir: str, prefix: str | None = None):
    patt = os.path.join(ckpt_dir, f"{prefix or ''}*epoch*.pth") if prefix else os.path.join(ckpt_dir, "*epoch*.pth")
    files = glob.glob(patt)
    if not files:
        return None
    files.sort(key=_natural_key)
    return files[-1]

def _rotate_ckpts(ckpt_dir: str, prefix: str | None, max_keep: int | None):
    if not max_keep or max_keep <= 0:
        return
    patt = os.path.join(ckpt_dir, f"{prefix or ''}*epoch*.pth") if prefix else os.path.join(ckpt_dir, "*epoch*.pth")
    files = sorted(glob.glob(patt), key=_natural_key)
    while len(files) > max_keep:
        old = files.pop(0)
        try:
            os.remove(old)
            print(f"[CKPT] removed old -> {old}", flush=True)
        except Exception as e:
            print(f"[CKPT] remove failed {old}: {e}", flush=True)


class SupervisedTrainer:
    def __init__(
        self,
        device="cuda",
        lr=3e-4,
        epochs=2,
        ckpt_dir="checkpoints",
        ckpt_prefix=None,
        save_every=1,
        max_keep=5,
        # —— 续训选项 ——
        resume_from=None,
        resume="auto",
        reset_epoch=False,
        reset_optim=False,
        strict_load=True,
        # —— 损失配置 ——
        loss_cfg=None,
        # —— 日志相关 ——
        use_tqdm=True,
        log_interval=None,
        # 验证指标控制：loss | psnr | ssim | None
        val_metric: str | None = "psnr",
        maximize_metric: bool | None = None,  # None -> 根据指标自动选择
    ):
        # 设备 & 超参
        self.device = torch.device(device if torch.cuda.is_available() else "cpu") \
                      if isinstance(device, str) else device
        self.lr = float(lr)
        self.epochs = int(epochs)

        # ckpt 配置
        self.ckpt_dir = ckpt_dir
        self.ckpt_prefix = ckpt_prefix
        self.save_every = int(save_every)
        self.max_keep = int(max_keep) if max_keep is not None else None
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 续训配置
        self.resume_from = resume_from
        self.resume = resume
        self.reset_epoch = bool(reset_epoch)
        self.reset_optim = bool(reset_optim)
        self.strict_load = bool(strict_load)

        # 损失
        self.criterion = build_criterion_from_cfg(self.device, loss_cfg or {"type": "l2"})
        print(f"[LOSS] using criterion: {type(self.criterion).__name__}", flush=True)

        # GPU 信息
        self.device_current_id = torch.cuda.current_device() if self.device.type == "cuda" else None

        # 日志相关
        self.use_tqdm = bool(use_tqdm)
        self.log_interval = None if log_interval is None else int(log_interval)
        self._tqdm_disable = (not sys.stderr.isatty()) or (not self.use_tqdm)

        # 验证 & best
        self.val_metric = None if val_metric is None else str(val_metric).lower()
        # 自动决定“越大越好/越小越好”
        if maximize_metric is None:
            self.maximize_metric = False if self.val_metric == "loss" else True
        else:
            self.maximize_metric = bool(maximize_metric)
        self.best_metric = None
        self.best_epoch = None

    # ---------- ckpt 命名/保存/加载 ----------
    def _ckpt_name(self, model, epoch: int):
        base = self.ckpt_prefix or model.__class__.__name__
        return os.path.join(self.ckpt_dir, f"{base}_epoch{epoch}.pth")

    def _best_ckpt_name(self, model):
        base = self.ckpt_prefix or model.__class__.__name__
        return os.path.join(self.ckpt_dir, f"{base}_best.pth")

    def _try_resume(self, model, opt):
        """返回起始 epoch（续训时为 last_epoch+1）"""
        ckpt_path = None
        if self.resume_from:
            ckpt_path = self.resume_from
        elif self.resume == "auto":
            ckpt_path = _latest_ckpt(self.ckpt_dir, self.ckpt_prefix)

        if not ckpt_path or not os.path.isfile(ckpt_path):
            print("[RESUME] no checkpoint found; start from scratch.", flush=True)
            return 1

        print(f"[RESUME] loading: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=self.strict_load)

        if (not self.reset_optim) and ("optimizer" in ckpt):
            try:
                opt.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[RESUME] optimizer state load failed: {e}", flush=True)

        last_epoch = int(ckpt.get("epoch", 0))
        start_epoch = 1 if self.reset_epoch else (last_epoch + 1 if last_epoch > 0 else 1)
        print(f"[RESUME] resumed at epoch {last_epoch} -> start from {start_epoch}", flush=True)
        return start_epoch

    def _save_ckpt(self, model, opt, epoch: int):
        path = self._ckpt_name(model, epoch)
        payload = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
        }
        torch.save(payload, path)
        print(f"[CKPT] saved -> {path}", flush=True)
        _rotate_ckpts(self.ckpt_dir, self.ckpt_prefix, self.max_keep)

    def _save_best(self, model, metric_value: float, epoch: int):
        """仅保存 model state_dict，便于加载推理"""
        path = self._best_ckpt_name(model)
        payload = {
            "epoch": epoch,
            "metric": float(metric_value),
            "state_dict": model.state_dict(),
        }
        torch.save(payload, path)
        print(f"[CKPT] saved BEST -> {path}  (metric={metric_value:.4f} @ epoch {epoch})", flush=True)

    # ---------- 训练主循环 ----------
    def fit(self, model, loader, val_loader=None):  # 兼容：val_loader 可为 None
        model = model.to(self.device)
        opt = Adam(model.parameters(), lr=self.lr)

        # 续训
        start_epoch = self._try_resume(model, opt)

        print("device:", self.device, "current_id:", self.device_current_id, flush=True)

        for epoch in range(start_epoch, self.epochs + 1):
            model.train()
            losses = []

            data_iter = tqdm(loader, desc=f"Epoch {epoch}", disable=self._tqdm_disable) \
                        if not self._tqdm_disable else loader

            for step, batch in enumerate(data_iter, 1):
                x = batch["inp"].to(self.device, non_blocking=True)
                y = batch["gt"].to(self.device, non_blocking=True)

                pred = model(x)

                if getattr(model, "expects_mask", False):
                    M = x[:, 1:2]
                else:
                    M = torch.ones_like(y)

                try:
                    loss = self.criterion(pred, y, M)
                except TypeError:
                    loss = self.criterion(pred, y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                losses.append(float(loss.detach().cpu()))

                if not self._tqdm_disable:
                    data_iter.set_postfix(loss=f"{loss.item():.4f}")
                elif self.log_interval and (step % self.log_interval == 0):
                    print(f"[Epoch {epoch} | Step {step}/{len(loader)}] loss={loss.item():.4f}", flush=True)

            mean_loss = sum(losses) / max(1, len(losses))
            print(f"Epoch {epoch} mean loss: {mean_loss:.4f}", flush=True)

            # 验证与 best
            did_val = False
            cur_metric = None
            if val_loader is not None and self.val_metric is not None:
                did_val, cur_metric = self._validate_and_maybe_save_best(model, val_loader, epoch)

            # 常规按周期保存
            if (epoch % self.save_every) == 0:
                self._save_ckpt(model, opt, epoch)

            # 打印 best 追踪
            if did_val and (self.best_metric is not None):
                print(f"[BEST] metric={self.best_metric:.4f} @ epoch {self.best_epoch}", flush=True)

    # ---------- 验证 ----------
    @torch.no_grad()
    def _validate_and_maybe_save_best(self, model, val_loader, epoch: int):
        model.eval()
        losses, psnrs, ssims = [], [], []

        val_iter = tqdm(val_loader, desc=f"Validate@{epoch}", disable=self._tqdm_disable) \
                   if not self._tqdm_disable else val_loader

        for batch in val_iter:
            x = batch["inp"].to(self.device, non_blocking=True)
            y = batch["gt"].to(self.device, non_blocking=True)
            pred = model(x)

            # —— 验证 loss：与训练完全一致 —— 
            if getattr(model, "expects_mask", False):
                M = x[:, 1:2]
            else:
                M = torch.ones_like(y)
            try:
                loss = self.criterion(pred, y, M)
            except TypeError:
                loss = self.criterion(pred, y)
            losses.append(float(loss.detach().cpu()))

            # —— 指标：在 [0,1] 域上 —— 
            p_clamp = torch.clamp(pred, 0.0, 1.0)
            y_clamp = torch.clamp(y,    0.0, 1.0)
            p_np = p_clamp.detach().cpu().numpy()
            y_np = y_clamp.detach().cpu().numpy()
            for i in range(p_np.shape[0]):
                pi = p_np[i, 0]; yi = y_np[i, 0]
                psnrs.append(_psnr(pi, yi, data_range=1.0))
                ssims.append(_ssim(pi, yi, data_range=1.0))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_psnr = float(np.mean(psnrs)) if psnrs else 0.0
        mean_ssim = float(np.mean(ssims)) if ssims else 0.0

        # 日志（总是打印三者，便于观察）
        print(f"[VAL {epoch}] loss={mean_loss:.6f}  PSNR={mean_psnr:.3f} dB  SSIM={mean_ssim:.4f}", flush=True)

        # 选择用于比较的指标
        if self.val_metric == "loss":
            metric = mean_loss
        elif self.val_metric == "psnr":
            metric = mean_psnr
        elif self.val_metric == "ssim":
            metric = mean_ssim
        else:
            metric = None  # 不做 best

        # 维护 best
        if metric is not None:
            is_better = (self.best_metric is None) or \
                        ((metric > self.best_metric) if self.maximize_metric else (metric < self.best_metric))
            if is_better:
                self.best_metric = metric
                self.best_epoch = epoch
                self._save_best(model, metric, epoch)

        return True, metric
