#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, yaml
import numpy as np
import torch
from tqdm import tqdm
import imageio.v3 as iio

# 触发模型注册
import ctprojfix.models.unet            # 以防有依赖
import ctprojfix.models.unet_res        # 以防有依赖
import ctprojfix.models.diffusion.ddpm  # 注册 "diffusion"
from ctprojfix.models.registry import build_model
from ctprojfix.data.dataset import make_dataloader

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def save_png(path, arr2d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 归一化到 0..1 再写 PNG
    a = np.asarray(arr2d, dtype=np.float32)
    lo, hi = np.percentile(a, [1, 99])
    if hi > lo:
        a = (a - lo) / (hi - lo)
    a = np.clip(a, 0, 1)
    iio.imwrite(path, (a * 255).astype(np.uint8))

def prepare_noise_schedule(T, beta_start, beta_end, device):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def apply_dc(x_t, obs_center, mask, mode="hard", alpha=0.5):
    """
    x_t, obs_center, mask: (B,1,H,W)
    mode: "hard" | "soft"
    """
    # --- 安全检查 ---
    assert mask.shape == x_t.shape, "[DC] mask/生成尺寸不一致"
    assert torch.sum(mask) > 0, "[DC] mask 全 0，请检查 downsample 与 truncate_left/right 是否匹配"

    if mode == "hard":
        # 中心区直接替换为观测
        return x_t * (1 - mask) + obs_center * mask
    elif mode == "soft":
        # 中心区向观测平滑靠拢
        return x_t * (1 - mask) + (alpha * obs_center + (1 - alpha) * x_t) * mask
    else:
        return x_t

@torch.no_grad()
def reverse_ddpm(model, cond, angle_norm, T, betas, alphas, alpha_bars,
                 dc_mode="hard", dc_alpha=0.5, device="cuda"):
    """
    cond: (B,2,H,W) = [noisy, mask]
    angle_norm: (B,1,H,W) or None
    return: x_0 (B,1,H,W)
    """
    B, _, H, W = cond.shape
    x_t = torch.randn(B, 1, H, W, device=device)

    # 观测中心（用于 DC）
    obs_center = cond[:, 0:1]  # noisy
    mask =      cond[:, 1:2]   # 1=center, 0=trunc sides

    for t in tqdm(reversed(range(T)), total=T, desc="Sample"):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        eps_hat = model(x_t, cond, t_batch, angle_norm=angle_norm)  # 预测噪声ε

        a_t = alphas[t]            # 标量
        ab_t = alpha_bars[t]       # 标量
        ab_t_prev = alpha_bars[t-1] if t > 0 else torch.tensor(1.0, device=device)

        # 公式：x_{t-1} = 1/sqrt(a_t) * (x_t - (1-a_t)/sqrt(1-ab_t) * eps_hat) + σ_t*z
        # 其中 σ_t^2 = β_t
        coef1 = (1.0 / torch.sqrt(a_t)).to(device)
        coef2 = ((1 - a_t) / torch.sqrt(1 - ab_t)).to(device)
        mean = coef1 * (x_t - coef2 * eps_hat)

        if t > 0:
            z = torch.randn_like(x_t)
            sigma_t = torch.sqrt(betas[t]).to(device)
            x_t = mean + sigma_t * z
        else:
            x_t = mean

        # 数据一致性（对中心有效区）
        x_t = apply_dc(x_t, obs_center, mask, mode=dc_mode, alpha=dc_alpha)

    # 最终视为 x_0
    return x_t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/sample_diffusion.yaml", help="采样配置")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    dev_str = cfg["sample"].get("device", "cuda")
    device = torch.device(dev_str if torch.cuda.is_available() else "cpu")

    # 模型
    model = build_model(cfg["model"]["name"], **cfg["model"]["params"]).to(device)
    ckpt_path = cfg["sample"].get("ckpt", "")
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[OK] 加载权重: {ckpt_path}")
    else:
        print(f"[INFO] 未找到权重（将用随机初始化，仅用于流程检查）: {ckpt_path}")
    model.eval()

    # 数据
    loader = make_dataloader(cfg["data"])
    assert len(loader.dataset) > 0, "[Error] 数据集为空，请检查 ids/路径/命名是否匹配"

    # 噪声调度
    T = int(cfg["sample"].get("T", 200))
    beta_start = float(cfg["sample"].get("beta_start", 1e-4))
    beta_end   = float(cfg["sample"].get("beta_end", 2e-2))
    betas, alphas, alpha_bars = prepare_noise_schedule(T, beta_start, beta_end, device)

    dc_mode  = cfg["sample"].get("dc_mode", "hard").lower()
    dc_alpha = float(cfg["sample"].get("dc_alpha", 0.5))
    out_dir  = cfg["sample"].get("out_dir", "outputs/sample")
    os.makedirs(out_dir, exist_ok=True)

    # 逐 batch 采样并保存
    global_idx = 0
    with torch.no_grad():
        for batch in loader:
            inp = batch["inp"].to(device)  # (B, C, H, W)  C=2 或 3
            B, C, H, W = inp.shape

            # 拆 cond / angle
            if C == 2:
                cond = inp                                 # [noisy, mask]
                angle_norm = None
            elif C == 3:
                cond = inp[:, :2, ...]                     # [noisy, mask]
                angle_norm = inp[:, 2:3, ...]              # [angle_map]
            else:
                raise ValueError(f"Unexpected input channels: {C}")

            # 采样
            x0 = reverse_ddpm(model, cond, angle_norm, T, betas, alphas, alpha_bars,
                              dc_mode=dc_mode, dc_alpha=dc_alpha, device=device)

            # 保存每张图（pred/noisy/mask/gt 可选）
            x0_np = x0.detach().cpu().numpy()
            noisy_np = cond[:, 0:1].detach().cpu().numpy()
            mask_np  = cond[:, 1:2].detach().cpu().numpy()

            has_gt = ("gt" in batch)
            if has_gt:
                gt_np = batch["gt"].numpy()

            for i in range(B):
                pred = x0_np[i, 0]
                noisy = noisy_np[i, 0]
                m = mask_np[i, 0]

                base = os.path.join(out_dir, f"s{global_idx:05d}")
                save_png(base + "_pred.png", pred)
                save_png(base + "_noisy.png", noisy)
                save_png(base + "_mask.png",  m)

                if has_gt:
                    save_png(base + "_gt.png", gt_np[i, 0])

                global_idx += 1

    print(f"[OK] 输出保存到：{out_dir}")

if __name__ == "__main__":
    main()
