# scripts/sample_diffusion.py
import argparse, os, yaml, numpy as np
import torch, imageio.v3 as iio
from tqdm import tqdm

# trigger model registry
import ctprojfix.models.unet
import ctprojfix.models.diffusion.ddpm
from ctprojfix.models.registry import build_model
from ctprojfix.models.diffusion.sampler import run_sampling
from ctprojfix.data.dataset import make_dataloader, DummyDataset
from ctprojfix.evals.metrics import psnr, ssim

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_checkpoint(model, ckpt_path, device):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[INFO] 未提供有效权重，使用随机初始化（仅检查流程）: {ckpt_path}")
        return model
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    print(f"[OK] 加载权重: {ckpt_path}")
    return model

def to_np01(x):
    x = x.astype(np.float32)
    return np.clip(x, 0.0, 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/sample_diffusion.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    device = torch.device(cfg["sample"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    out_dir = cfg["sample"].get("out_dir", "outputs/sample")
    os.makedirs(out_dir, exist_ok=True)

    # 模型加载
    model = build_model(cfg["model"]["name"], **cfg["model"]["params"]).to(device)
    model = load_checkpoint(model, cfg["sample"].get("ckpt",""), device)
    model.eval()

    # 数据加载
    loader = make_dataloader(cfg["data"])
    if len(loader.dataset) == 0 or cfg["data"].get("use_dummy", False):
        print("[INFO] 使用 DummyDataset 进行采样演示")
        dummy = DummyDataset(length=cfg["data"].get("dummy_length", 4),
                             H=cfg["data"].get("dummy_H", 256),
                             W=cfg["data"].get("dummy_W", 384),
                             truncate_left=cfg["data"].get("truncate_left", 64),
                             truncate_right=cfg["data"].get("truncate_right", 64),
                             add_angle_channel=cfg["data"].get("add_angle_channel", False))
        from torch.utils.data import DataLoader
        loader = DataLoader(dummy, batch_size=cfg["data"].get("batch_size", 2), shuffle=False)

    # 采样参数
    T = int(cfg["sample"].get("T", 200))
    beta_start = float(cfg["sample"].get("beta_start", 1e-4))
    beta_end   = float(cfg["sample"].get("beta_end", 2e-2))
    dc_mode = cfg["sample"].get("dc_mode", "hard")
    dc_alpha= float(cfg["sample"].get("dc_alpha", 0.5))

    idx = 0
    all_psnr, all_ssim = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Sample"):
            noisy = batch["inp"][:,0:1].to(device)
            mask  = batch["inp"][:,1:2].to(device)
            gt    = batch.get("gt", None)
            angle_norm = None
            try:
                ang = batch["angle"].to(device).float()
                A   = batch["A"].to(device).float().clamp(min=1)
                ang = ang / (A - 1 + 1e-6)
                B, _, H, W = noisy.shape
                angle_norm = ang.view(-1,1,1,1).expand(B,1,H,W)
            except Exception:
                pass

            x0 = run_sampling(model, noisy, mask,
                              T=T, beta_start=beta_start, beta_end=beta_end,
                              dc_mode=dc_mode, dc_alpha=dc_alpha,
                              angle_norm=angle_norm, device=str(device))

            # 保存 & 评估
            x0_np = x0.detach().cpu().numpy()  # (B,1,H,W)
            for b in range(x0_np.shape[0]):
                pred = to_np01(np.squeeze(x0_np[b],0))
                iio.imwrite(os.path.join(out_dir, f"sample_{idx:05d}.png"), (pred*255).astype(np.uint8))
                if gt is not None:
                    g = to_np01(np.squeeze(gt[b].cpu().numpy(),0))
                    all_psnr.append(psnr(pred, g, data_range=1.0))
                    all_ssim.append(ssim(pred, g, data_range=1.0))
                idx += 1

    if all_psnr:
        print(f"[RESULT] PSNR: {np.mean(all_psnr):.3f} dB  |  SSIM: {np.mean(all_ssim):.4f}  (N={len(all_psnr)})")
    print(f"[OK] 输出保存到：{out_dir}")

if __name__ == "__main__":
    main()
