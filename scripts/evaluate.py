# -*- coding: utf-8 -*-
"""
- 维持原有评估/指标/三联图逻辑；
- 不再逐张保存 pred_00000；
- 一次性导出整卷预测（pred_volume.*），并可选整卷导出 GT / Noisy；
- 若 batch 提供 lo/hi 标注，额外导出 RAW 域整卷（*_raw.*）。
  eval:
    save_volume: true        # 是否导出整卷预测（默认 true）
    save_format: both        # npy|tiff|both（默认 both）
    save_gt_noisy: true      # 是否同时导出 GT / Noisy 整卷（默认 true）

"""

import argparse, yaml, os, csv
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # 无显示环境时避免卡住
import matplotlib.pyplot as plt
import tifffile as tiff  # 用于一次性保存多页 TIF

# 触发模型注册：把可能用到的模型都 import 一次
import ctprojfix.models.unet
import ctprojfix.models.unet_res
import ctprojfix.models.pconv_unet
import ctprojfix.models.diffusion.ddpm

from ctprojfix.models.registry import build_model
from ctprojfix.data.dataset import make_dataloader, DummyDataset
from ctprojfix.evals.metrics import psnr, ssim

# 更灵活的 device 解析
def _resolve_device(cfg_eval: dict, cli_device: str | None, cli_gpu: str | int | None) -> torch.device:
    """
    优先级：--device > --gpu > cfg.eval.device > auto
      --device: 例如 "cuda:1" / "cpu"
      --gpu:    整数索引（等价于 device=f"cuda:{idx}"）
    """
    # 1) CLI --device
    if cli_device:
        d = str(cli_device).strip().lower()
        if d == "cpu" or (d.startswith("cuda") and torch.cuda.is_available()):
            return torch.device(d)
        # 容错, 传了 "cuda" 但当前不可用 → 退回 cpu
        return torch.device("cuda" if (d.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    # 2) CLI --gpu
    if cli_gpu is not None:
        try:
            idx = int(cli_gpu)
            if torch.cuda.is_available():
                return torch.device(f"cuda:{idx}")
        except Exception:
            pass
        return torch.device("cpu")

    # 3) cfg.eval.device
    d = str(cfg_eval.get("device", "") or "").strip().lower()
    if d:
        if d == "cpu":
            return torch.device("cpu")
        if d.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(d)
            else:
                print("[WARN] cfg.eval.device 要求 CUDA，但本机不可用，退回 CPU。")
                return torch.device("cpu")

    # 4) auto
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    return v


def save_triptych(noisy, pred, gt, out_path, title=None):
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


def _get_bool(val, default=False):
    if val is None: return default
    if isinstance(val, bool): return val
    return str(val).lower() in ["1","true","yes","y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/eval.yaml", help="评估配置")
    # 是否保存体数据的控制开关
    ap.add_argument("--save-volume", type=str, default=None, help="是否输出整体 3D 文件 (true/false)")
    ap.add_argument("--save-format", type=str, default=None, choices=["npy","tiff","both"], help="体数据输出格式")
    ap.add_argument("--save-gt-noisy", type=str, default=None, help="是否同时导出 GT/Noisy 整卷 (true/false)")


    #设备选择
    ap.add_argument("--device", type=str, default=None, help='计算设备，如 "cuda:0" / "cuda:1" / "cpu"')
    ap.add_argument("--gpu", type=str, default=None, help="GPU 编号（等价于 --device cuda:{idx}）")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    eval_cfg = cfg.get("eval", {})
    print(f"[CFG] loaded config from {args.cfg}")

    device = _resolve_device(eval_cfg, args.device, args.gpu)
    if device.type == "cuda":
        try:
            torch.cuda.set_device(device)  #将当前 CUDA 上下文切到该卡
        except Exception as e:
            print(f"[WARN] torch.cuda.set_device 失败：{e}")

    print(f"[DEVICE] using device = {device}", flush=True)


    # device = torch.device(cfg.get("eval",{}).get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # 模型
    model = build_model(cfg["model"]["name"], **cfg["model"]["params"]).to(device)
    model = load_checkpoint(model, eval_cfg.get("ckpt", ""), device)
    model.eval()

    # 数据（评估时建议在 cfg.data 里加 return_norm_stats: true）
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
    save_dir = eval_cfg.get("out_dir", "outputs/eval")
    os.makedirs(save_dir, exist_ok=True)
    trip_dir = os.path.join(save_dir, "figs")
    os.makedirs(trip_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics.csv")

    # 选择展示索引（第几张保存三联图）
    show_idx = eval_cfg.get("show_idx", "mid")
    total = len(loader.dataset)
    if isinstance(show_idx, str) and show_idx.lower() == "mid":
        show_idx = total // 2
    else:
        show_idx = int(show_idx)

    # 开关合并：优先用命令行，其次 cfg，最后默认
    save_volume = _get_bool(args.save_volume, default=bool(eval_cfg.get("save_volume", True)))
    save_format = args.save_format or eval_cfg.get("save_format", "both")
    save_gt_noisy = _get_bool(args.save_gt_noisy, default=bool(eval_cfg.get("save_gt_noisy", True)))

    psnrs, ssims = [], []
    img_count = 0

    # 新增列表收集整卷输出
    preds_all = []         # 归一化域 [0,1]
    preds_raw_all = []     # 若有 RAW 反归一（与 GT 尺度一致）
    ids_all = []           # 记录 id 顺序（便于排查）

    gts_all = []           # GT 整卷（归一化域）
    noisies_all = []       # Noisy 整卷（归一化域）
    gts_raw_all = []       # GT RAW（若有 lo/hi）
    noisies_raw_all = []   # Noisy RAW（若有 lo/hi）

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["idx","id","angle","A","noisy_lo","noisy_hi","gt_lo","gt_hi","PSNR","SSIM"])

        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval"):
                x = batch["inp"].to(device)  # 约定：x[:,0]=noisy, x[:,1]=mask, x[:,2]=angle
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
                    A_val  = _get_item_from_batch(batch, "A", i, default=-1)

                    # 原本：逐张保存 PNG（注释掉，改为整卷写出）
                    # plt.imsave(os.path.join(save_dir, f"pred_{img_count:05d}.png"), p, cmap="gray", vmin=0, vmax=1)  #
                    # plt.imsave(os.path.join(save_dir, f"gt_{img_count:05d}.png"),   g, cmap="gray", vmin=0, vmax=1)  #
                    # plt.imsave(os.path.join(save_dir, f"noisy_{img_count:05d}.png"),n, cmap="gray", vmin=0, vmax=1)  #

                    # 收集体数据（归一化域）
                    preds_all.append(p.astype(np.float32))
                    ids_all.append(id_val)
                    if save_gt_noisy:
                        gts_all.append(g.astype(np.float32))
                        noisies_all.append(n.astype(np.float32))

                    # 若 batch 提供归一化统计，则反归一得到 RAW，以 GT 尺度为准
                    n_lo = _get_item_from_batch(batch, "noisy_lo", i, default=None)
                    n_hi = _get_item_from_batch(batch, "noisy_hi", i, default=None)
                    g_lo = _get_item_from_batch(batch, "gt_lo",    i, default=None)
                    g_hi = _get_item_from_batch(batch, "gt_hi",    i, default=None)

                    if (n_lo is not None) and (n_hi is not None) and (g_lo is not None) and (g_hi is not None):
                        pred_raw  = p * (g_hi - g_lo) + g_lo
                        # 原本 逐张保存 RAW npy（注释掉，改为整卷写出）
                        # np.save(os.path.join(save_dir, f"pred_raw_{img_count:05d}.npy"),  pred_raw.astype(np.float32))  #
                        preds_raw_all.append(pred_raw.astype(np.float32))  # 修改：收集 RAW 体

                        if save_gt_noisy:
                            gt_raw    = g * (g_hi - g_lo) + g_lo
                            noisy_raw = n * (n_hi - n_lo) + n_lo
                            gts_raw_all.append(gt_raw.astype(np.float32))
                            noisies_raw_all.append(noisy_raw.astype(np.float32))

                    # CSV 指标
                    writer.writerow([
                        img_count, str(id_val), int(angle), int(A_val),
                        "" if n_lo is None else f"{float(n_lo):.7g}",
                        "" if n_hi is None else f"{float(n_hi):.7g}",
                        "" if g_lo is None else f"{float(g_lo):.7g}",
                        "" if g_hi is None else f"{float(g_hi):.7g}",
                        f"{cur_psnr:.4f}", f"{cur_ssim:.4f}"
                    ])

                    # 保存一张三联+误差（保留）
                    if img_count == show_idx:
                        save_triptych(n, p, g,
                                      os.path.join(trip_dir, f"triptych_{img_count:05d}.png"),
                                      title=f"id={id_val}, angle={angle}, A={A_val}, idx={img_count}")

                    img_count += 1


    # 循环结束后统一写出卷数据

    if save_volume:
        # 归一化域：Pred 必写
        if len(preds_all) > 0:
            vol = np.stack(preds_all, axis=0).astype(np.float32)  # [N,H,W]
            if save_format in ("npy", "both"):
                np.save(os.path.join(save_dir, "pred_volume.npy"), vol)
            if save_format in ("tiff", "both"):
                tiff.imwrite(os.path.join(save_dir, "pred_volume.tiff"), vol, imagej=True)
            print(f"[OK] 预测体(归一化域) 已保存: pred_volume.npy / pred_volume.tiff  形状={vol.shape}")
        # RAW 域（若有）
        if len(preds_raw_all) > 0:
            vol_raw = np.stack(preds_raw_all, axis=0).astype(np.float32)
            if save_format in ("npy", "both"):
                np.save(os.path.join(save_dir, "pred_volume_raw.npy"), vol_raw)
            if save_format in ("tiff", "both"):
                tiff.imwrite(os.path.join(save_dir, "pred_volume_raw.tiff"), vol_raw, imagej=True)
            print(f"[OK] 预测体(RAW域) 已保存: pred_volume_raw.npy / pred_volume_raw.tiff  形状={vol_raw.shape}")

        # GT / Noisy cfg决定
        if save_gt_noisy:
            if len(gts_all) > 0:
                vol_gt = np.stack(gts_all, axis=0).astype(np.float32)
                if save_format in ("npy", "both"):
                    np.save(os.path.join(save_dir, "gt_volume.npy"), vol_gt)
                if save_format in ("tiff", "both"):
                    tiff.imwrite(os.path.join(save_dir, "gt_volume.tiff"), vol_gt, imagej=True)
                print(f"[OK] GT 体(归一化域) 已保存: gt_volume.npy / gt_volume.tiff  形状={vol_gt.shape}")

            if len(noisies_all) > 0:
                vol_noisy = np.stack(noisies_all, axis=0).astype(np.float32)
                if save_format in ("npy", "both"):
                    np.save(os.path.join(save_dir, "noisy_volume.npy"), vol_noisy)
                if save_format in ("tiff", "both"):
                    tiff.imwrite(os.path.join(save_dir, "noisy_volume.tiff"), vol_noisy, imagej=True)
                print(f"[OK] Noisy 体(归一化域) 已保存: noisy_volume.npy / noisy_volume.tiff  形状={vol_noisy.shape}")

            if len(gts_raw_all) > 0:
                vol_gt_raw = np.stack(gts_raw_all, axis=0).astype(np.float32)
                if save_format in ("npy", "both"):
                    np.save(os.path.join(save_dir, "gt_volume_raw.npy"), vol_gt_raw)
                if save_format in ("tiff", "both"):
                    tiff.imwrite(os.path.join(save_dir, "gt_volume_raw.tiff"), vol_gt_raw, imagej=True)
                print(f"[OK] GT 体(RAW域) 已保存: gt_volume_raw.npy / gt_volume_raw.tiff  形状={vol_gt_raw.shape}")

            if len(noisies_raw_all) > 0:
                vol_noisy_raw = np.stack(noisies_raw_all, axis=0).astype(np.float32)
                if save_format in ("npy", "both"):
                    np.save(os.path.join(save_dir, "noisy_volume_raw.npy"), vol_noisy_raw)
                if save_format in ("tiff", "both"):
                    tiff.imwrite(os.path.join(save_dir, "noisy_volume_raw.tiff"), vol_noisy_raw, imagej=True)
                print(f"[OK] Noisy 体(RAW域) 已保存: noisy_volume_raw.npy / noisy_volume_raw.tiff  形状={vol_noisy_raw.shape}")

    print(f"[RESULT] PSNR: {np.mean(psnrs):.3f} dB  |  SSIM: {np.mean(ssims):.4f}  (N={len(psnrs)})")
    print(f"[OK] 结果保存目录：{save_dir}\n[OK] 指标 CSV：{csv_path}")


if __name__ == "__main__":
    main()
