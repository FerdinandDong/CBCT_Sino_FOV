# scripts/evaluate.py
import argparse, yaml, os
import numpy as np
import torch
from tqdm import tqdm

# 触发模型注册
import ctprojfix.models.unet
from ctprojfix.models.registry import build_model
from ctprojfix.data.dataset import make_dataloader, DummyDataset
from ctprojfix.evals.metrics import psnr, ssim

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_checkpoint(model, ckpt_path, device):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[INFO] 未提供有效权重路径，使用随机初始化模型进行评估（仅用于测试流程检查）: {ckpt_path}")
        return model
    ckpt = torch.load(ckpt_path, map_location=device)
    # 兼容直接保存 state_dict 或包含 "state_dict"
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    print(f"[OK] 已加载权重: {ckpt_path}")
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/eval.yaml", help="评估配置")
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    # 构建模型
    device = torch.device(cfg["eval"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model  = build_model(cfg["model"]["name"], **cfg["model"]["params"]).to(device)
    model  = load_checkpoint(model, cfg["eval"].get("ckpt", ""), device)
    model.eval()

    # 数据
    loader = make_dataloader(cfg["data"])
    if len(loader.dataset) == 0 or cfg["data"].get("use_dummy", False):
        print("[INFO] 使用 DummyDataset 做离线评估。")
        dummy = DummyDataset(
            length=cfg["data"].get("dummy_length", 8),
            H=cfg["data"].get("dummy_H", 256),
            W=cfg["data"].get("dummy_W", 384),
            truncate_left=cfg["data"].get("truncate_left", 64),
            truncate_right=cfg["data"].get("truncate_right", 64),
        )
        from torch.utils.data import DataLoader
        loader = DataLoader(dummy, batch_size=cfg["data"].get("batch_size", 1), shuffle=False, num_workers=0)

    # 评估循环
    psnrs, ssim_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            x = batch["inp"].to(device)  # (B,2,H,W)
            y = batch["gt"].to(device)   # (B,1,H,W)
            pred = model(x)

            # 转成 numpy(0-1) 计算指标
            pred_np = pred.detach().cpu().numpy()
            gt_np   = y.detach().cpu().numpy()
            # 按 batch 求平均
            for i in range(pred_np.shape[0]):
                p = np.squeeze(pred_np[i], axis=0)  # (H,W)
                g = np.squeeze(gt_np[i],   axis=0)  # (H,W)
                psnrs.append(psnr(p, g, data_range=1.0))
                ssim_list.append(ssim(p, g, data_range=1.0))

    print(f"[RESULT] PSNR: {np.mean(psnrs):.3f} dB  |  SSIM: {np.mean(ssim_list):.4f}  (N={len(psnrs)})")

if __name__ == "__main__":
    main()
