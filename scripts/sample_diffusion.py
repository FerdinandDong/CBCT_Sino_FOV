#!/usr/bin/env python3
# CBCT_Sino_FOV/scripts/sample_diffusion.py
# -*- coding: utf-8 -*-

import os, argparse, yaml
import numpy as np
import torch
from tqdm import tqdm
import tifffile as tiff
import imageio.v3 as iio  # noqa: F401  # 可能被上游依赖

# 触发模型注册（保持你的既有依赖关系）
import ctprojfix.models.unet            # 以防有依赖
import ctprojfix.models.unet_res        # 以防有依赖
import ctprojfix.models.diffusion.ddpm  # 注册 "diffusion"

from ctprojfix.models.registry import build_model
from ctprojfix.data.dataset import make_dataloader

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def percentile_norm01(a, p_lo=1, p_hi=99):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.percentile(a, [p_lo, p_hi])
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max() + 1e-6)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

def save_triptych(noisy, pred, gt, out_path, title=None):
    """保存单张三联图（仅用于展示）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = percentile_norm01(noisy); p = percentile_norm01(pred); g = percentile_norm01(gt)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(n, cmap="gray"); axes[0].set_title("Noisy"); axes[0].axis("off")
    axes[1].imshow(p, cmap="gray"); axes[1].set_title("Pred");  axes[1].axis("off")
    axes[2].imshow(g, cmap="gray"); axes[2].set_title("GT");    axes[2].axis("off")
    if title: fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {out_path}")

def prepare_noise_schedule(T, beta_start, beta_end, device):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def apply_dc(x_t, obs_center, mask, mode="hard", alpha=0.5):
    """
    x_t, obs_center, mask: (B,1,H,W)
    mode: "hard" | "soft" | "none"
    """
    assert mask.shape == x_t.shape, "[DC] mask/生成尺寸不一致"
    assert torch.sum(mask) > 0, "[DC] mask 全 0，请检查 downsample 与 truncate_left/right 是否匹配"
    if mode == "hard":
        return x_t * (1 - mask) + obs_center * mask
    elif mode == "soft":
        return x_t * (1 - alpha * mask) + (alpha * obs_center) * mask
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

    obs_center = cond[:, 0:1]  # noisy
    mask =      cond[:, 1:2]   # 1=center, 0=trunc sides

    for t in tqdm(reversed(range(T)), total=T, desc="Sample"):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        eps_hat = model(x_t, cond, t_batch, angle_norm=angle_norm)  # 预测噪声ε

        a_t = alphas[t]
        ab_t = alpha_bars[t]
        coef1 = (1.0 / torch.sqrt(a_t)).to(device)
        coef2 = ((1 - a_t) / torch.sqrt(1 - ab_t)).to(device)
        mean = coef1 * (x_t - coef2 * eps_hat)

        if t > 0:
            z = torch.randn_like(x_t)
            sigma_t = torch.sqrt(betas[t]).to(device)
            x_t = mean + sigma_t * z
        else:
            x_t = mean

        x_t = apply_dc(x_t, obs_center, mask, mode=dc_mode, alpha=dc_alpha)

    return torch.clamp(x_t, 0.0, 1.0)

def _get_item_from_batch(batch, key, i, default=None):
    if key not in batch: return default
    v = batch[key]
    if isinstance(v, torch.Tensor):
        if v.ndim == 0: return v.item()
        return v[i].item() if v.ndim >= 1 else default
    if isinstance(v, (list, tuple)):
        try: return v[i]
        except Exception: return v if len(v) == 1 else default
    if isinstance(v, np.ndarray):
        if v.ndim == 0: return float(v)
        return v[i] if v.ndim >= 1 else default
    return v

def _str2bool(val, default=False):
    if val is None: return default
    if isinstance(val, bool): return val
    return str(val).lower() in ["1","true","yes","y"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/sample_diffusion.yaml", help="采样配置")
    # 体数据输出控制（与 evaluate.py 对齐）
    ap.add_argument("--save-volume", type=str, default=None, help="是否输出整体 3D 文件 (true/false)")
    ap.add_argument("--save-format", type=str, default=None, choices=["npy","tiff","both"], help="体数据输出格式")
    ap.add_argument("--save-gt-noisy", type=str, default=None, help="是否同时导出 GT/Noisy 整卷 (true/false)")
    ap.add_argument("--show-idx", type=str, default=None, help='三联图索引（"mid" 或整数），仅保存这一张 PNG')
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)

    # I2SB 分支
    model_name = str(cfg.get("model", {}).get("name", "")).lower().strip()
    cfg = load_cfg(args.cfg)
    
    # ======== I2SB 本地分支：直接调用 scripts/sample_i2sb_local.py ========
    if model_name == "i2sb_local":
        print("[I2SB] Detected model.name=='i2sb_local' → delegate to local sampler...")
        # robust 导入同目录脚本
        import importlib.util, sys
        this_dir = os.path.dirname(os.path.abspath(__file__))
        mod_path = os.path.join(this_dir, "sample_i2sb_local.py")
        spec = importlib.util.spec_from_file_location("sample_i2sb_local", mod_path)
        s1 = importlib.util.module_from_spec(spec)
        sys.modules["sample_i2sb_local"] = s1
        spec.loader.exec_module(s1)
        # 直接跑
        s1.run(args.cfg, ckpt=None, out=None)
        print("[I2SB] Sampling finished by local sampler.")
        return
    # ============================================================

    dev_str = cfg.get("sample", {}).get("device", "cuda")
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

    # ✅ 输出目录：默认 outputs/diffusion
    out_dir  = cfg["sample"].get("out_dir", "outputs/diffusion")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[OUT] 输出目录: {os.path.abspath(out_dir)}")

    # 体输出控制
    save_volume   = _str2bool(args.save_volume, default=bool(cfg["sample"].get("save_volume", True)))
    save_format   = args.save_format or cfg["sample"].get("save_format", "both")
    save_gt_noisy = _str2bool(args.save_gt_noisy, default=bool(cfg["sample"].get("save_gt_noisy", True)))
    show_idx = args.show_idx or cfg["sample"].get("show_idx", "mid")

    # 用于整卷写出
    preds_all, preds_raw_all = [], []
    gts_all, noisies_all = [], []
    gts_raw_all, noisies_raw_all = [], []
    keep_for_fig = {"noisy": None, "pred": None, "gt": None, "idx": None}

    # 逐 batch 采样（不保存逐帧 PNG！）
    global_idx = 0
    with torch.no_grad():
        for batch in loader:
            inp = batch["inp"].to(device)  # (B, C, H, W)  C=2 或 3
            B, C, H, W = inp.shape

            # cond / angle
            if C == 2:
                cond = inp
                angle_norm = None
            elif C == 3:
                cond = inp[:, :2, ...]
                angle_norm = inp[:, 2:3, ...]
            else:
                raise ValueError(f"Unexpected input channels: {C}")

            x0 = reverse_ddpm(model, cond, angle_norm, T, betas, alphas, alpha_bars,
                              dc_mode=dc_mode, dc_alpha=dc_alpha, device=device)

            x0_np = x0.detach().cpu().numpy()               # (B,1,H,W)
            noisy_np = cond[:, 0:1].detach().cpu().numpy()  # (B,1,H,W)
            has_gt = ("gt" in batch)
            if has_gt: gt_np = batch["gt"].cpu().numpy()

            # 收集整卷（归一化域）
            for i in range(B):
                preds_all.append(np.squeeze(x0_np[i, 0]).astype(np.float32))
                if has_gt and save_gt_noisy:
                    gts_all.append(np.squeeze(gt_np[i, 0]).astype(np.float32))
                    noisies_all.append(np.squeeze(noisy_np[i, 0]).astype(np.float32))

            # RAW 反归一
            for i in range(B):
                n_lo = _get_item_from_batch(batch, "noisy_lo", i, default=None)
                n_hi = _get_item_from_batch(batch, "noisy_hi", i, default=None)
                g_lo = _get_item_from_batch(batch, "gt_lo",    i, default=None)
                g_hi = _get_item_from_batch(batch, "gt_hi",    i, default=None)

                if (g_lo is not None) and (g_hi is not None):
                    pred_raw = x0_np[i, 0] * (float(g_hi) - float(g_lo)) + float(g_lo)
                    preds_raw_all.append(pred_raw.astype(np.float32))

                if save_gt_noisy and (g_lo is not None) and (g_hi is not None) and (n_lo is not None) and (n_hi is not None):
                    if has_gt:
                        gt_raw = gt_np[i, 0] * (float(g_hi) - float(g_lo)) + float(g_lo)
                        gts_raw_all.append(gt_raw.astype(np.float32))
                    noisy_raw = noisy_np[i, 0] * (float(n_hi) - float(n_lo)) + float(n_lo)
                    noisies_raw_all.append(noisy_raw.astype(np.float32))

            # 选择中间帧用于三联图展示（只保存一张）
            if isinstance(show_idx, str) and show_idx.lower() == "mid":
                target_local = B // 2
            else:
                target_local = 0  # 先存第一张，最后再统一挑选

            if keep_for_fig["noisy"] is None:
                keep_for_fig["noisy"] = noisy_np[target_local, 0]
                keep_for_fig["pred"]  = x0_np[target_local, 0]
                keep_for_fig["gt"]    = (gt_np[target_local, 0] if has_gt else noisy_np[target_local, 0])
                keep_for_fig["idx"]   = global_idx + target_local

            global_idx += B

    # === 写出整卷 ===
    if save_volume:
        # 归一化域
        if len(preds_all) > 0:
            vol = np.stack(preds_all, axis=0).astype(np.float32)
            if save_format in ("npy", "both"):
                np.save(os.path.join(out_dir, "pred_volume.npy"), vol)
            if save_format in ("tiff", "both"):
                tiff.imwrite(os.path.join(out_dir, "pred_volume.tiff"), vol, imagej=True)
            print(f"[OK] 预测体(归一化域) -> {os.path.join(out_dir, 'pred_volume.*')}  shape={vol.shape}")
        if save_gt_noisy and len(gts_all) > 0:
            vol_gt = np.stack(gts_all, axis=0).astype(np.float32)
            vol_nz = np.stack(noisies_all, axis=0).astype(np.float32)
            if save_format in ("npy", "both"):
                np.save(os.path.join(out_dir, "gt_volume.npy"), vol_gt)
                np.save(os.path.join(out_dir, "noisy_volume.npy"), vol_nz)
            if save_format in ("tiff", "both"):
                tiff.imwrite(os.path.join(out_dir, "gt_volume.tiff"),   vol_gt, imagej=True)
                tiff.imwrite(os.path.join(out_dir, "noisy_volume.tiff"),vol_nz, imagej=True)
            print(f"[OK] GT/Noisy(归一化域) 已保存。")

        # RAW 域
        if len(preds_raw_all) > 0:
            vol_raw = np.stack(preds_raw_all, axis=0).astype(np.float32)
            if save_format in ("npy", "both"):
                np.save(os.path.join(out_dir, "pred_volume_raw.npy"), vol_raw)
            if save_format in ("tiff", "both"):
                tiff.imwrite(os.path.join(out_dir, "pred_volume_raw.tiff"), vol_raw, imagej=True)
            print(f"[OK] 预测体(RAW域) -> {os.path.join(out_dir, 'pred_volume_raw.*')}  shape={vol_raw.shape}")
        if save_gt_noisy and len(gts_raw_all) > 0:
            vol_gt_raw = np.stack(gts_raw_all, axis=0).astype(np.float32)
            vol_nz_raw = np.stack(noisies_raw_all, axis=0).astype(np.float32)
            if save_format in ("npy", "both"):
                np.save(os.path.join(out_dir, "gt_volume_raw.npy"),   vol_gt_raw)
                np.save(os.path.join(out_dir, "noisy_volume_raw.npy"),vol_nz_raw)
            if save_format in ("tiff", "both"):
                tiff.imwrite(os.path.join(out_dir, "gt_volume_raw.tiff"),   vol_gt_raw, imagej=True)
                tiff.imwrite(os.path.join(out_dir, "noisy_volume_raw.tiff"),vol_nz_raw, imagej=True)
            print(f"[OK] GT/Noisy(RAW域) 已保存。")

    # === 只保存一张三联 PNG ===
    if keep_for_fig["noisy"] is not None:
        fig_path = os.path.join(out_dir, f"triptych_s{keep_for_fig['idx']:05d}.png")
        save_triptych(keep_for_fig["noisy"], keep_for_fig["pred"], keep_for_fig["gt"],
                      out_path=fig_path, title=f"idx={keep_for_fig['idx']}")
    else:
        print("[WARN] 未能保存三联图：没有采样结果。")

    print(f"[OK] 采样完成。输出目录：{os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()
