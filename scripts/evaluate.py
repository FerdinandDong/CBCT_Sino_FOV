import argparse, yaml, os, csv
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # 无显示环境时避免卡住
import matplotlib.pyplot as plt

# 触发模型注册：把可能用到的模型都 import 一次
import ctprojfix.models.unet
import ctprojfix.models.unet_res
import ctprojfix.models.pconv_unet
import ctprojfix.models.diffusion.ddpm

from ctprojfix.models.registry import build_model
from ctprojfix.data.dataset import make_dataloader, DummyDataset
from ctprojfix.evals.metrics import psnr, ssim


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(model, ckpt_path, device):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[INFO] 未提供有效权重路径，使用随机初始化模型进行评估（仅用于流程检查）: {ckpt_path}")
        return model
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    print(f"[OK] 已加载权重: {ckpt_path}")
    return model


def _to_py(obj):
    """把可能是 tensor/ndarray 的标量或 1D 元素，转成 python 标量或可打印字符串"""
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            return float(obj.reshape(-1)[0])
        return obj.tolist()
    return obj


def _get_item_from_batch(batch, key, i, default=None):
    """从 batch[key] 中取第 i 个样本的值（支持标量/1D tensor/ndarray/list）"""
    if key not in batch:
        return default
    v = batch[key]
    if isinstance(v, torch.Tensor):
        if v.ndim == 0:
            return v.item()
        return v[i].item() if v.ndim >= 1 else default
    if isinstance(v, (list, tuple)):
        try:
            return v[i]
        except Exception:
            return v if len(v) == 1 else default
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return float(v)
        return v[i] if v.ndim >= 1 else default
    # 标量
    return v


def save_triptych(noisy, pred, gt, out_path, title=None):
    """保存三联图 + 误差图"""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Noisy"); axes[0].axis("off")
    axes[1].imshow(pred,  cmap="gray", vmin=0, vmax=1); axes[1].set_title("Pred");  axes[1].axis("off")
    axes[2].imshow(gt,    cmap="gray", vmin=0, vmax=1); axes[2].set_title("GT");    axes[2].axis("off")
    err = np.abs(pred - gt)
    im = axes[3].imshow(err, cmap="magma"); axes[3].set_title("|Pred-GT|"); axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    if title: fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/eval.yaml", help="评估配置")
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    device = torch.device(cfg["eval"].get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # 模型
    model = build_model(cfg["model"]["name"], **cfg["model"]["params"]).to(device)
    model = load_checkpoint(model, cfg["eval"].get("ckpt", ""), device)
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
            add_angle_channel=cfg["data"].get("add_angle_channel", False),
        )
        from torch.utils.data import DataLoader
        loader = DataLoader(dummy, batch_size=cfg["data"].get("batch_size", 1), shuffle=False, num_workers=0)

    # 输出
    save_dir = cfg["eval"].get("out_dir", "outputs/eval")
    os.makedirs(save_dir, exist_ok=True)
    trip_dir = os.path.join(save_dir, "figs")
    os.makedirs(trip_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics.csv")

    # 选择展示索引（第几张保存三联图）
    show_idx = cfg["eval"].get("show_idx", "mid")
    total = len(loader.dataset)
    if isinstance(show_idx, str) and show_idx.lower() == "mid":
        show_idx = total // 2
    else:
        show_idx = int(show_idx)

    # 评估
    psnrs, ssims = [], []
    img_count = 0

    # 写 CSV
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["idx", "id", "angle", "PSNR", "SSIM"])

        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval"):
                x = batch["inp"].to(device)  # 约定：x[:,0]=noisy, x[:,1]=mask, x[:,2]=angle(可选)
                y = batch["gt"].to(device)   # (B,1,H,W)
                pred = model(x)

                pred_np  = pred.detach().cpu().numpy()
                gt_np    = y.detach().cpu().numpy()
                noisy_np = x[:, 0:1].detach().cpu().numpy()  # (B,1,H,W)

                for i in range(pred_np.shape[0]):
                    p = np.squeeze(pred_np[i], 0)   # (H,W)
                    g = np.squeeze(gt_np[i], 0)
                    n = np.squeeze(noisy_np[i], 0)

                    # 指标范围对齐
                    p = np.clip(p, 0.0, 1.0)
                    g = np.clip(g, 0.0, 1.0)
                    n = np.clip(n, 0.0, 1.0)

                    cur_psnr = psnr(p, g, data_range=1.0)
                    cur_ssim = ssim(p, g, data_range=1.0)
                    psnrs.append(cur_psnr); ssims.append(cur_ssim)

                    id_val = _get_item_from_batch(batch, "id", i, default="?")
                    angle  = _get_item_from_batch(batch, "angle", i, default=-1)
                    writer.writerow([img_count, str(id_val), int(angle), f"{cur_psnr:.4f}", f"{cur_ssim:.4f}"])

                    # 保存单张
                    plt.imsave(os.path.join(save_dir, f"pred_{img_count:05d}.png"), p, cmap="gray", vmin=0, vmax=1)
                    plt.imsave(os.path.join(save_dir, f"gt_{img_count:05d}.png"),   g, cmap="gray", vmin=0, vmax=1)
                    plt.imsave(os.path.join(save_dir, f"noisy_{img_count:05d}.png"),n, cmap="gray", vmin=0, vmax=1)

                    # 保存一张三联+误差
                    if img_count == show_idx:
                        save_triptych(n, p, g,
                                      os.path.join(trip_dir, f"triptych_{img_count:05d}.png"),
                                      title=f"id={id_val}, angle={angle}, idx={img_count}")

                    img_count += 1

    print(f"[RESULT] PSNR: {np.mean(psnrs):.3f} dB  |  SSIM: {np.mean(ssims):.4f}  (N={len(psnrs)})")
    print(f"[OK] 结果保存目录：{save_dir}\n[OK] 指标 CSV：{csv_path}")


if __name__ == "__main__":
    main()
