# ctprojfix/trainers/supervised.py
import os, re, glob
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from ctprojfix.losses.criterion import build_criterion_from_cfg


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
            print(f"[CKPT] removed old -> {old}")
        except Exception as e:
            print(f"[CKPT] remove failed {old}: {e}")


class SupervisedTrainer:
    def __init__(
        self,
        device="cuda",
        lr=3e-4,
        epochs=2,
        ckpt_dir="checkpoints",
        ckpt_prefix=None,      # 文件名前缀
        save_every=1,          # 每几轮保存
        max_keep=5,            # 最多保留几个
        # —— 续训选项 —— 
        resume_from=None,      # 明确指定 ckpt 路径；None 表示看 resume="auto"/False
        resume="auto",         # "auto" | False
        reset_epoch=False,     # True: 从 1 开始计数
        reset_optim=False,     # True: 不加载优化器状态
        strict_load=True,      # 严格加载 state_dict
        # —— 损失配置 —— 
        loss_cfg=None,
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
        print(f"[LOSS] using criterion: {type(self.criterion).__name__}")

    # ---------- ckpt 命名/保存/加载 ----------
    def _ckpt_name(self, model, epoch: int):
        base = self.ckpt_prefix or model.__class__.__name__
        return os.path.join(self.ckpt_dir, f"{base}_epoch{epoch}.pth")

    def _try_resume(self, model, opt):
        """返回起始 epoch（续训时为 last_epoch+1）"""
        ckpt_path = None

        if self.resume_from:             # 明确指定
            ckpt_path = self.resume_from
        elif self.resume == "auto":      # 自动找最新
            ckpt_path = _latest_ckpt(self.ckpt_dir, self.ckpt_prefix)

        if not ckpt_path or not os.path.isfile(ckpt_path):
            print("[RESUME] no checkpoint found; start from scratch.")
            return 1

        print(f"[RESUME] loading: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=self.strict_load)

        if (not self.reset_optim) and ("optimizer" in ckpt):
            try:
                opt.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[RESUME] optimizer state load failed: {e}")

        last_epoch = int(ckpt.get("epoch", 0))
        start_epoch = 1 if self.reset_epoch else (last_epoch + 1 if last_epoch > 0 else 1)
        print(f"[RESUME] resumed at epoch {last_epoch} -> start from {start_epoch}")
        return start_epoch

    def _save_ckpt(self, model, opt, epoch: int):
        path = self._ckpt_name(model, epoch)
        payload = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
        }
        torch.save(payload, path)
        print(f"[CKPT] saved -> {path}")
        _rotate_ckpts(self.ckpt_dir, self.ckpt_prefix, self.max_keep)

    # ---------- 训练主循环 ----------
    def fit(self, model, loader):
        model = model.to(self.device)
        opt = Adam(model.parameters(), lr=self.lr)

        # 续训（如配置）
        start_epoch = self._try_resume(model, opt)

        for epoch in range(start_epoch, self.epochs + 1):
            model.train()
            losses = []
            pbar = tqdm(loader, desc=f"Epoch {epoch}")

            for batch in pbar:
                x = batch["inp"].to(self.device)  # (B,C,H,W)，约定 x[:,0]=noisy, x[:,1]=mask, x[:,2]=angle(可选)
                y = batch["gt"].to(self.device)   # (B,1,H,W)

                # 前向（PConv 模型会在内部拆 noisy/mask）
                pred = model(x)

                # 统一构造 M（中心有效=1），给 CombinedLoss；若 criterion 只接收 2 个参数，则降级
                if getattr(model, "expects_mask", False):
                    M = x[:, 1:2]  # PConv 模型输入包含 mask
                else:
                    M = torch.ones_like(y)

                try:
                    loss = self.criterion(pred, y, M)   # CombinedLoss 等 3参
                except TypeError:
                    loss = self.criterion(pred, y)      # L1/L2 等 2参

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            mean_loss = sum(losses) / max(1, len(losses))
            print(f"Epoch {epoch} mean loss: {mean_loss:.4f}")

            if (epoch % self.save_every) == 0:
                self._save_ckpt(model, opt, epoch)
