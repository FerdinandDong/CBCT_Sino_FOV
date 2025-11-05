# -*- coding: utf-8 -*-
# CBCT_Sino_FOV/scripts/evaluate.py
import argparse, yaml, os, csv, random
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile as tiff

# 触发模型注册
import ctprojfix.models.unet
import ctprojfix.models.unet_res
import ctprojfix.models.pconv_unet
import ctprojfix.models.diffusion.ddpm

from ctprojfix.models.registry import build_model
from ctprojfix.data.dataset import make_dataloader, DummyDataset
try:
    from ctprojfix.data.dataset import ProjectionAnglesDataset
except Exception:
    ProjectionAnglesDataset = None

from ctprojfix.evals.metrics import psnr, ssim
from torch.utils.data import DataLoader


# -------------------- 简易工具 --------------------
def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def _save_tile_gray01(img01: np.ndarray, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img01, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def _save_tile_heat(err: np.ndarray, out_path: str, cmap: str, with_cb: bool):
    _ensure_dir(os.path.dirname(out_path))
    if with_cb:
        fig = plt.figure(figsize=(4.4, 4), dpi=200)
        ax = fig.add_axes([0.0, 0.0, 0.9, 1.0])
        im = ax.imshow(err, cmap=cmap, vmin=0.0, vmax=None)  # 自适应
        ax.axis("off")
        cax = fig.add_axes([0.91, 0.1, 0.03, 0.8])
        plt.colorbar(im, cax=cax)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(4, 4), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(err, cmap=cmap, vmin=0.0, vmax=None)  # 自适应
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def _resolve_device(cfg_eval: dict, cli_device: str | None, cli_gpu: str | int | None) -> torch.device:
    if cli_device:
        d = str(cli_device).strip().lower()
        if d == "cpu":
            return torch.device("cpu")
        if d.startswith("cuda") and torch.cuda.is_available():
            return torch.device(d)
        return torch.device("cpu")
    if cli_gpu is not None:
        try:
            idx = int(cli_gpu)
            if torch.cuda.is_available():
                return torch.device(f"cuda:{idx}")
        except Exception:
            pass
        return torch.device("cpu")
    d = str(cfg_eval.get("device", "") or "").strip().lower()
    if d:
        if d == "cpu":
            return torch.device("cpu")
        if d.startswith("cuda") and torch.cuda.is_available():
            return torch.device(d)
        if d.startswith("cuda"):
            print("[WARN] cfg.eval.device 要求 CUDA，但本机不可用，退回 CPU。")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    print(f"[SEED] set to {seed}")

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_checkpoint(model, ckpt_path, device):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[INFO] 未提供有效权重，使用随机初始化：{ckpt_path}")
        return model
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    print(f"[OK] 加载权重: {ckpt_path}")
    return model

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

def _get_bool(val, default=False):
    if val is None: return default
    if isinstance(val, bool): return val
    return str(val).lower() in ["1","true","yes","y"]

def save_triptych(noisy, pred, gt, out_path, title=None, cmap="magma"):
    """保存三联图 + 误差；误差热力图自适应范围。"""
    err = np.abs(pred - gt)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Noisy"); axes[0].axis("off")
    axes[1].imshow(pred,  cmap="gray", vmin=0, vmax=1); axes[1].set_title("Pred");  axes[1].axis("off")
    axes[2].imshow(gt,    cmap="gray", vmin=0, vmax=1); axes[2].set_title("GT");    axes[2].axis("off")
    im = axes[3].imshow(err, cmap=cmap, vmin=0.0, vmax=None)  # 自适应
    axes[3].set_title("|Pred-GT|"); axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    if title: fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {out_path}")


# -------------------- 数据集打印 --------------------
def _unwrap_subset(ds):
    try:
        from torch.utils.data import Subset
        if isinstance(ds, Subset): return ds.dataset
    except Exception:
        pass
    return ds

def _print_dataset_brief(ds):
    base = _unwrap_subset(ds)
    if isinstance(base, DummyDataset):
        print("[DATA] DummyDataset"); print(f"[DATA] HxW={base.H}x{base.W}, length={len(base)}"); return
    if ProjectionAnglesDataset is not None and isinstance(base, ProjectionAnglesDataset):
        ids = list(base.ids); ids_preview = ids[:12]
        print("[DATA] ProjectionAnglesDataset")
        print(f"[DATA] root_noisy = {base.root_noisy}")
        print(f"[DATA] root_clean = {base.root_clean}")
        print(f"[DATA] ids(count={len(ids)}): {ids_preview}{' ...' if len(ids)>len(ids_preview) else ''}")
        return
    print(f"[DATA] dataset type = {type(base)}")


# -------------------- 重建一个不shuffle的 eval loader --------------------
from torch.utils.data import DataLoader
def _rebuild_eval_loader(original_loader, data_cfg):
    ds = original_loader.dataset
    bs = int(data_cfg.get("batch_size", 1))
    nw = int(data_cfg.get("num_workers", 0))
    pin = bool(data_cfg.get("pin_memory", False))
    persistent_workers = bool(data_cfg.get("persistent_workers", (nw > 0)))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
    seed = int(data_cfg.get("seed", 2024))
    g = torch.Generator(); g.manual_seed(seed)
    kwargs = dict(batch_size=bs, shuffle=False, num_workers=nw,
                  pin_memory=pin, drop_last=False, generator=g)
    if nw > 0:
        kwargs.update(dict(persistent_workers=persistent_workers,
                           prefetch_factor=max(2, prefetch_factor)))
    return DataLoader(ds, **kwargs)


# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/eval.yaml", help="评估配置")
    ap.add_argument("--save-volume", type=str, default=None)
    ap.add_argument("--save-format", type=str, default=None, choices=["npy","tiff","both"])
    ap.add_argument("--save-gt-noisy", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    eval_cfg = cfg.get("eval", {})
    data_cfg = cfg.get("data", {})
    seed = int(eval_cfg.get("seed", data_cfg.get("seed", 2024)))
    _set_seed(seed)

    device = _resolve_device(eval_cfg, args.device, args.gpu)
    if device.type == "cuda":
        try: torch.cuda.set_device(device.index if device.index is not None else 0)
        except Exception as e: print(f"[WARN] set_device 失败：{e}")
    print(f"[DEVICE] {device}")

    model = build_model(cfg["model"]["name"], **cfg["model"]["params"]).to(device)
    model = load_checkpoint(model, eval_cfg.get("ckpt", ""), device)
    model.eval()

    loader = make_dataloader(cfg["data"])
    if isinstance(loader, dict):
        _print_dataset_brief(loader.get("test", loader.get("val", loader.get("train"))).dataset)
        loader = loader.get("test", next(iter(loader.values())))
    else:
        _print_dataset_brief(loader.dataset)

    if not bool(data_cfg.get("shuffle", True)):
        print("[INFO] data.shuffle=False → 重建不打乱的 DataLoader")
        loader = _rebuild_eval_loader(loader, data_cfg)

    save_dir = eval_cfg.get("out_dir", "outputs/eval")
    _ensure_dir(save_dir)
    trip_dir = os.path.join(save_dir, "figs")
    _ensure_dir(trip_dir)
    csv_path = os.path.join(save_dir, "metrics.csv")

    # tiles 导出（仅简单开关）
    tiles_cfg = (eval_cfg.get("tiles", {}) or {})
    tiles_enable = bool(tiles_cfg.get("enable", False))
    tiles_dir    = tiles_cfg.get("dir", None)      # 默认: fig 同名子目录
    tiles_prefix = str(tiles_cfg.get("prefix", ""))
    tiles_format = str(tiles_cfg.get("format", "png"))
    tiles_heat_with_cb = bool(tiles_cfg.get("heat_with_colorbar", False))
    tiles_cmap   = str(tiles_cfg.get("cmap", "magma"))

    show_idx = eval_cfg.get("show_idx", "mid")
    show_id  = eval_cfg.get("show_id", None)
    show_ang = eval_cfg.get("show_angle", None)

    total = len(loader.dataset)
    show_idx = (total // 2) if (isinstance(show_idx, str) and show_idx.lower()=="mid") else int(show_idx)

    save_volume   = _get_bool(args.save_volume,  default=bool(eval_cfg.get("save_volume", True)))
    save_format   = args.save_format or eval_cfg.get("save_format", "both")
    save_gt_noisy = _get_bool(args.save_gt_noisy, default=bool(eval_cfg.get("save_gt_noisy", True)))

    psnrs, ssims = [], []
    img_count = 0
    preds_all, preds_raw_all, ids_all = [], [], []
    gts_all, noisies_all, gts_raw_all, noisies_raw_all = [], [], [], []
    saved_flag = False

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["idx","id","angle","A","noisy_lo","noisy_hi","gt_lo","gt_hi","PSNR","SSIM"])

        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval"):
                x = batch["inp"].to(device)
                y = batch["gt"].to(device)
                pred = model(x)

                pred_np  = pred.detach().cpu().numpy()
                gt_np    = y.detach().cpu().numpy()
                noisy_np = x[:, 0:1].detach().cpu().numpy()

                for i in range(pred_np.shape[0]):
                    p = np.squeeze(pred_np[i], 0)
                    g = np.squeeze(gt_np[i], 0)
                    n = np.squeeze(noisy_np[i], 0)
                    p = np.clip(p, 0.0, 1.0); g = np.clip(g, 0.0, 1.0); n = np.clip(n, 0.0, 1.0)

                    cur_psnr = psnr(p, g, data_range=1.0)
                    cur_ssim = ssim(p, g, data_range=1.0)
                    psnrs.append(cur_psnr); ssims.append(cur_ssim)

                    id_val = _get_item_from_batch(batch, "id", i, default="?")
                    angle  = _get_item_from_batch(batch, "angle", i, default=-1)
                    A_val  = _get_item_from_batch(batch, "A", i, default=-1)

                    preds_all.append(p.astype(np.float32))
                    ids_all.append(id_val)
                    if save_gt_noisy:
                        gts_all.append(g.astype(np.float32))
                        noisies_all.append(n.astype(np.float32))

                    n_lo = _get_item_from_batch(batch, "noisy_lo", i, default=None)
                    n_hi = _get_item_from_batch(batch, "noisy_hi", i, default=None)
                    g_lo = _get_item_from_batch(batch, "gt_lo",    i, default=None)
                    g_hi = _get_item_from_batch(batch, "gt_hi",    i, default=None)
                    if (n_lo is not None) and (n_hi is not None) and (g_lo is not None) and (g_hi is not None):
                        pred_raw  = p * (g_hi - g_lo) + g_lo
                        preds_raw_all.append(pred_raw.astype(np.float32))
                        if save_gt_noisy:
                            gt_raw    = g * (g_hi - g_lo) + g_lo
                            noisy_raw = n * (n_hi - n_lo) + n_lo
                            gts_raw_all.append(gt_raw.astype(np.float32))
                            noisies_raw_all.append(noisy_raw.astype(np.float32))

                    writer.writerow([
                        img_count, str(id_val), int(angle), int(A_val),
                        "" if n_lo is None else f"{float(n_lo):.7g}",
                        "" if n_hi is None else f"{float(n_hi):.7g}",
                        "" if g_lo is None else f"{float(g_lo):.7g}",
                        "" if g_hi is None else f"{float(g_hi):.7g}",
                        f"{cur_psnr:.4f}", f"{cur_ssim:.4f}"
                    ])

                    # ——保存三联图 & 小图——
                    save_this = False
                    if (show_id is not None) and (show_ang is not None):
                        if int(id_val) == int(show_id) and int(angle) == int(show_ang):
                            save_this = True
                    else:
                        if img_count == show_idx:
                            save_this = True

                    if save_this and not saved_flag:
                        out_name = (
                            f"triptych_id{int(id_val)}_a{int(angle):03d}.png"
                            if (show_id is not None) else
                            f"triptych_{img_count:05d}.png"
                        )
                        trip_path = os.path.join(trip_dir, out_name)
                        save_triptych(n, p, g, trip_path, title=f"id={id_val}, angle={angle}, A={A_val}, idx={img_count}")

                        if tiles_enable:
                            if tiles_dir:
                                td = tiles_dir
                            else:
                                base = os.path.splitext(os.path.basename(trip_path))[0]
                                td = os.path.join(trip_dir, base + "_tiles")
                            _ensure_dir(td)

                            tag = (
                                f"id{int(id_val)}_a{int(angle):03d}_idx{img_count}"
                                if (show_id is not None) else
                                f"idx{img_count:05d}"
                            )
                            _save_tile_gray01(n, os.path.join(td, f"{tiles_prefix}Noisy_{tag}.{tiles_format}"))
                            _save_tile_gray01(p, os.path.join(td, f"{tiles_prefix}Pred_{tag}.{tiles_format}"))
                            _save_tile_gray01(g, os.path.join(td, f"{tiles_prefix}GT_{tag}.{tiles_format}"))
                            _save_tile_heat(np.abs(p-g), os.path.join(td, f"{tiles_prefix}AbsDiff_Pred_vs_GT_{tag}.{tiles_format}"),
                                            cmap=tiles_cmap, with_cb=tiles_heat_with_cb)

                        saved_flag = True

                    img_count += 1

    # -------- 写出卷数据 --------
    if save_volume:
        if len(preds_all) > 0:
            vol = np.stack(preds_all, axis=0).astype(np.float32)
            if save_format in ("npy", "both"):  np.save(os.path.join(save_dir, "pred_volume.npy"), vol)
            if save_format in ("tiff","both"): tiff.imwrite(os.path.join(save_dir, "pred_volume.tiff"), vol, imagej=True)
            print(f"[OK] 预测体(0~1) 已保存: pred_volume.*  形状={vol.shape}")

        if len(preds_raw_all) > 0:
            vol_raw = np.stack(preds_raw_all, axis=0).astype(np.float32)
            if save_format in ("npy", "both"):  np.save(os.path.join(save_dir, "pred_volume_raw.npy"), vol_raw)
            if save_format in ("tiff","both"): tiff.imwrite(os.path.join(save_dir, "pred_volume_raw.tiff"), vol_raw, imagej=True)
            print(f"[OK] 预测体(RAW) 已保存: pred_volume_raw.*  形状={vol_raw.shape}")

        if save_gt_noisy:
            if len(gts_all) > 0:
                vol_gt = np.stack(gts_all, axis=0).astype(np.float32)
                if save_format in ("npy", "both"):  np.save(os.path.join(save_dir, "gt_volume.npy"), vol_gt)
                if save_format in ("tiff","both"): tiff.imwrite(os.path.join(save_dir, "gt_volume.tiff"), vol_gt, imagej=True)
                print(f"[OK] GT 体(0~1) 已保存: gt_volume.*  形状={vol_gt.shape}")

            if len(noisies_all) > 0:
                vol_noisy = np.stack(noisies_all, axis=0).astype(np.float32)
                if save_format in ("npy", "both"):  np.save(os.path.join(save_dir, "noisy_volume.npy"), vol_noisy)
                if save_format in ("tiff","both"): tiff.imwrite(os.path.join(save_dir, "noisy_volume.tiff"), vol_noisy, imagej=True)
                print(f"[OK] Noisy 体(0~1) 已保存: noisy_volume.*  形状={vol_noisy.shape}")

            if len(gts_raw_all) > 0:
                vol_gt_raw = np.stack(gts_raw_all, axis=0).astype(np.float32)
                if save_format in ("npy", "both"):  np.save(os.path.join(save_dir, "gt_volume_raw.npy"), vol_gt_raw)
                if save_format in ("tiff","both"): tiff.imwrite(os.path.join(save_dir, "gt_volume_raw.tiff"), vol_gt_raw, imagej=True)
                print(f"[OK] GT 体(RAW) 已保存: gt_volume_raw.*  形状={vol_gt_raw.shape}")

            if len(noisies_raw_all) > 0:
                vol_noisy_raw = np.stack(noisies_raw_all, axis=0).astype(np.float32)
                if save_format in ("npy", "both"):  np.save(os.path.join(save_dir, "noisy_volume_raw.npy"), vol_noisy_raw)
                if save_format in ("tiff","both"): tiff.imwrite(os.path.join(save_dir, "noisy_volume_raw.tiff"), vol_noisy_raw, imagej=True)
                print(f"[OK] Noisy 体(RAW) 已保存: noisy_volume_raw.*  形状={vol_noisy_raw.shape}")

    print(f"[RESULT] PSNR: {np.mean(psnrs):.3f} dB  |  SSIM: {np.mean(ssims):.4f}  (N={len(psnrs)})")
    print(f"[OK] 结果目录：{save_dir}\n[OK] 指标 CSV：{csv_path}")


if __name__ == "__main__":
    main()
