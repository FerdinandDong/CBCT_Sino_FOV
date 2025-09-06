# ctprojfix/models/diffusion/sampler.py
import torch

@torch.no_grad()
def make_beta_schedule(T=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar

@torch.no_grad()
def ddpm_step(x_t, t, eps_hat, betas, alphas, alphas_bar, z=None):
    """标准 DDPM 反向一步"""
    a_t = alphas[t].view(-1, 1, 1, 1)
    a_bar_t = alphas_bar[t].view(-1, 1, 1, 1)
    b_t = betas[t].view(-1, 1, 1, 1)
    mean = (1.0 / a_t.sqrt()) * (x_t - ((1.0 - a_t) / (1.0 - a_bar_t).sqrt()) * eps_hat)
    if z is None:
        z = torch.randn_like(x_t)
    x_prev = mean + b_t.sqrt() * z
    return x_prev

@torch.no_grad()
def data_consistency(x, noisy, mask, mode="hard", alpha=0.5):
    """数据一致性约束"""
    if mode == "hard":
        return mask * noisy + (1.0 - mask) * x
    else:
        blend = alpha * noisy + (1.0 - alpha) * x
        return mask * blend + (1.0 - mask) * x

@torch.no_grad()
def run_sampling(model, noisy, mask,
                 T=1000, beta_start=1e-4, beta_end=2e-2,
                 dc_mode="hard", dc_alpha=0.5,
                 angle_norm=None,
                 device="cuda"):
    """
    从纯噪声迭代采样，返回修复后的完整投影
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    B, _, H, W = noisy.shape
    betas, alphas, alphas_bar = make_beta_schedule(T, beta_start, beta_end, device)

    x = torch.randn(B, 1, H, W, device=device)  # 初始纯噪声
    cond = torch.cat([noisy.to(device), mask.to(device)], dim=1)
    if angle_norm is not None:
        angle_norm = angle_norm.to(device)

    for step in reversed(range(T)):
        t = torch.full((B,), step, device=device, dtype=torch.long)
        eps_hat = model(x, cond, t, angle_norm=angle_norm)
        z = None if step == 0 else torch.randn_like(x)
        x = ddpm_step(x, t, eps_hat, betas, alphas, alphas_bar, z=z)
        x = data_consistency(x, noisy, mask, mode=dc_mode, alpha=dc_alpha)

    return x
