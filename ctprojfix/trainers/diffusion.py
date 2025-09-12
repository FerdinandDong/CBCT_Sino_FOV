# ctprojfix/trainers/diffusion.py
import os, glob, re
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

def make_beta_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

class DiffusionTrainer:
    def __init__(self,
                 device="cuda", lr=1e-4, epochs=1,
                 T=1000, beta_start=1e-4, beta_end=0.02,
                 dc_strength=1.0,
                 ckpt_dir="checkpoints/diffusion",
                 save_every=1,
                 max_keep=5):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lr, self.epochs = float(lr), int(epochs)
        self.T = int(T)
        self.betas = make_beta_schedule(self.T, beta_start, beta_end).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.dc_strength = float(dc_strength)

        self.ckpt_dir = ckpt_dir
        self.save_every = int(save_every) if save_every else 0
        self.max_keep = int(max_keep)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def q_sample(self, x0, t, eps):
        a_bar = self.alphas_bar[t].view(-1,1,1,1)
        return (a_bar.sqrt() * x0) + ((1.0 - a_bar).sqrt() * eps)

    # ---- ckpt helpers ----
    def _extract_epoch(self, path: str) -> int:
        m = re.search(r"epoch(\d+)\.pth$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    def _save_ckpt(self, model, epoch, extra=None):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f"DDPM_epoch{epoch}.pth")
        payload = {"state_dict": model.state_dict(), "epoch": epoch}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[CKPT] saved -> {path}")

        if self.max_keep > 0:
            files = glob.glob(os.path.join(self.ckpt_dir, "DDPM_epoch*.pth"))
            files_sorted = sorted(files, key=self._extract_epoch)  # sort by epoch number
            while len(files_sorted) > self.max_keep:
                old = files_sorted.pop(0)
                try:
                    os.remove(old)
                    print(f"[CKPT] removed old -> {old}")
                except Exception as e:
                    print(f"[CKPT] failed to remove {old}: {e}")

    # ---- main train loop ----
    def fit(self, model, loader):
        model = model.to(self.device)
        opt = Adam(model.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs+1):
            model.train()
            epoch_losses = []

            pbar = tqdm(loader, desc=f"DDPM Epoch {epoch}",  disable=True)  #只打印epoch级别 防止log过大
            # pbar = tqdm(loader, desc=f"DDPM Epoch {epoch}") # 进度条显示

            for batch in pbar:
                noisy = batch["inp"][:,0:1].to(self.device)   # (B,1,H,W)
                mask  = batch["inp"][:,1:2].to(self.device)   # (B,1,H,W)
                clean = batch["gt"].to(self.device)           # (B,1,H,W)

                # angle 通道（拿不到就用 0）
                B, _, H, W = clean.shape
                try:
                    ang = batch["angle"].to(self.device).float()
                    A   = batch["A"].to(self.device).float().clamp(min=1)
                    ang = ang / (A - 1 + 1e-6)
                    angle_norm = ang.view(-1,1,1,1).expand(B,1,H,W)
                except Exception:
                    angle_norm = torch.zeros(B,1,H,W, device=self.device, dtype=clean.dtype)

                # 采样时间步 & 合成 x_t
                t = torch.randint(0, self.T, (B,), device=self.device, dtype=torch.long)
                eps = torch.randn_like(clean)
                x_t = self.q_sample(clean, t, eps)

                # 条件：[noisy, mask]；预测 ε̂
                cond = torch.cat([noisy, mask], dim=1)  # (B,2,H,W)
                eps_hat = model(x_t, cond, t, angle_norm=angle_norm)

                loss = F.mse_loss(eps_hat, eps)
                opt.zero_grad(); loss.backward(); opt.step()

                epoch_losses.append(float(loss.item()))
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            print(f"[DDPM] Epoch {epoch} mean loss: {mean_loss:.4f}")

            if self.save_every and (epoch % self.save_every == 0):
                self._save_ckpt(model, epoch, extra={"mean_loss": mean_loss})
