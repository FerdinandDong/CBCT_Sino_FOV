# -*- coding: utf-8 -*-
# CBCT_Sino_FOV/scripts/evaluate.py
"""
- 维持原有评估/指标/三联图逻辑；
- 不再逐张保存 pred_00000；
- 一次性导出整卷预测（pred_volume.*），并可选整卷导出 GT / Noisy；
- 若 batch 提供 lo/hi 标注，额外导出 RAW 域整卷（*_raw.*）。
- 新增：
  * 明确打印使用的 cfg/数据根/ids/几何/输出目录；
  * 支持 eval.show_id + eval.show_angle 精确点名保存样本；
  * 读取 data.shuffle，若为 False 则强制不打乱（并固定随机种子）；
  * 固定随机种子，提升可复现性。
"""

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


# -------------------- 实用函数 --------------------
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 如需极致确定性可开启下两行（会降速）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"[SEED] set to {seed}")


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
        return obj.item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return float(obj.reshape(-1)[0]) if obj.size == 1 else obj.tolist()
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


# -------------------- 数据集打印 --------------------
def _unwrap_subset(ds):
    try:
        from torch.utils.data import Subset
        if isinstance(ds, Subset):
            return ds.dataset
    except Exception:
        pass
    return ds


def _print_dataset_brief(ds):
    base = _unwrap_subset(ds)

    if isinstance(base, DummyDataset):
        print("[DATA] 使用 DummyDataset")
        print(f"[DATA] HxW={base.H}x{base.W}, length={len(base)}, truncate_left={base.truncate_left}, truncate_right={base.truncate_right}, add_angle={base.add_angle}")
        return

    if ProjectionAnglesDataset is not None and isinstance(base, ProjectionAnglesDataset):
        ids = list(base.ids)
        ids_preview = ids[:12]
        print("[DATA] 使用 ProjectionAnglesDataset")
        print(f"[DATA] root_noisy = {base.root_noisy}")
        print(f"[DATA] root_clean = {base.root_clean}")
        print(f"[DATA] ids(count={len(ids)}): {ids_preview}{' ...' if len(ids)>len(ids_preview) else ''}")
        print(f"[DATA] step={base.step}, downsample={base.downsample}, mask_mode={base.mask_mode}, normalize={base.normalize}, add_angle_channel={base.add_angle_channel}")
        print(f"[DATA] return_norm_stats={base.return_norm_stats}, cache_strategy={base.cache_strategy}, max_cached_ids={base.max_cached_ids}")
        try:
            meta_items = list(base.meta.items())[:5]
            for k, v in meta_items:
                print(f"[DATA] id={k} -> A={v.get('A')}, H={v.get('H')}, W={v.get('W')}")
            if len(base.meta) > 5:
                print(f"[DATA] ... ({len(base.meta)-5} more ids)")
        except Exception:
            pass
        try:
            print(f"[DATA] total indexed samples (angles) = {len(base.index)}")
        except Exception:
            pass
        return

    print(f"[DATA] dataset type = {type(base)} (未识别的自定义类型，跳过详细字段打印)")


# -------------------- 重建一个不shuffle的 eval loader --------------------
def _rebuild_eval_loader(original_loader, data_cfg):
    ds = original_loader.dataset
    bs = int(data_cfg.get("batch_size", 1))
    nw = int(data_cfg.get("num_workers", 0))
    pin = bool(data_cfg.get("pin_memory", False))
    persistent_workers = bool(data_cfg.get("persistent_workers", (nw > 0)))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
    seed = int(data_cfg.get("seed", 2024))
    g = torch.Generator()
    g.manual_seed(seed)
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
    # 体数据输出控制
    ap.add_argument("--save-volume", type=str, default=None, help="是否输出整体 3D 文件 (true/false)")
    ap.add_argument("--save-format", type=str, default=None, choices=["npy","tiff","both"], help="体数据输出格式")
    ap.add_argument("--save-gt-noisy", type=str, default=None, help="是否同时导出 GT/Noisy 整卷 (true/false)")
    # 设备选择
    ap.add_argument("--device", type=str, default=None, help='计算设备，如 "cuda:0" / "cuda:1" / "cpu"')
    ap.add_argument("--gpu", type=str, default=None, help="GPU 编号（等价于 --device cuda:{idx}）")
    args = ap.parse_args()

    # 读取 cfg + 顶层日志
    cfg = load_cfg(args.cfg)
    eval_cfg = cfg.get("eval", {})
    data_cfg = cfg.get("data", {})
    seed = int(eval_cfg.get("seed", data_cfg.get("seed", 2024)))
    _set_seed(seed)

    print("=" * 80)
    print(f"[CFG] loaded from: {args.cfg}")
    print(f"[MODEL] name={cfg.get('model',{}).get('name')} params={cfg.get('model',{}).get('params')}")
    print(f"[EVAL] out_dir={eval_cfg.get('out_dir','outputs/eval')} ckpt={eval_cfg.get('ckpt','')}")
    print(f"[EVAL] save_volume={eval_cfg.get('save_volume', True)} save_format={eval_cfg.get('save_format','both')} save_gt_noisy={eval_cfg.get('save_gt_noisy', True)}")
    # 展示策略
    print(f"[EVAL] show_idx={eval_cfg.get('show_idx','mid')}  show_id={eval_cfg.get('show_id', None)}  show_angle={eval_cfg.get('show_angle', None)}")
    # 原始cfg数据字段
    print(f"[DATA(cfg)] root_noisy={data_cfg.get('root_noisy')}  root_clean={data_cfg.get('root_clean')}")
    print(f"[DATA(cfg)] ids={data_cfg.get('ids')}  exclude_ids={data_cfg.get('exclude_ids')}")
    print(f"[DATA(cfg)] step={data_cfg.get('step')} downsample={data_cfg.get('downsample')} mask_mode={data_cfg.get('mask_mode')} normalize={data_cfg.get('normalize')} shuffle={data_cfg.get('shuffle', None)}")
    print("=" * 80)

    # 设备
    device = _resolve_device(eval_cfg, args.device, args.gpu)
    if device.type == "cuda":
        try:
            torch.cuda.set_device(device.index if device.index is not None else 0)
        except Exception as e:
            print(f"[WARN] torch.cuda.set_device 失败：{e}")
    print(f"[DEVICE] using device = {device}", flush=True)

    # 模型
    model = build_model(cfg["model"]["name"], **cfg["model"]["params"]).to(device)
    model = load_checkpoint(model, eval_cfg.get("ckpt", ""), device)
    model.eval()

    # 数据
    loader = make_dataloader(cfg["data"])
    # 打印数据摘要 & 统一选取一个 loader
    if isinstance(loader, dict):
        print("[INFO] Detected split loaders (train/val/test). 使用 test loader 进行元信息打印。")
        _print_dataset_brief(loader.get("test", loader.get("val", loader.get("train"))).dataset)
        loader = loader.get("test", next(iter(loader.values())))
    else:
        _print_dataset_brief(loader.dataset)

    # 若 cfg.data.shuffle == false，则强制重建一个“不打乱”的评估 loader
    shuffle_flag = bool(data_cfg.get("shuffle", True))
    if not shuffle_flag:
        print("[INFO] data.shuffle=False → 重建一个不打乱的评估 DataLoader。")
        loader = _rebuild_eval_loader(loader, data_cfg)

    # 输出路径
    save_dir = eval_cfg.get("out_dir", "outputs/eval")
    os.makedirs(save_dir, exist_ok=True)
    trip_dir = os.path.join(save_dir, "figs")
    os.makedirs(trip_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics.csv")
    print(f"[IO] save_dir={save_dir}")
    print(f"[IO] metrics_csv={csv_path}")
    print("=" * 80)

    # 展示选择（支持 idx 或 id+angle）
    show_idx = eval_cfg.get("show_idx", "mid")
    show_id  = eval_cfg.get("show_id", None)
    show_ang = eval_cfg.get("show_angle", None)

    total = len(loader.dataset)
    if isinstance(show_idx, str) and show_idx.lower() == "mid":
        show_idx = total // 2
    else:
        show_idx = int(show_idx)

    if show_id is not None and show_ang is not None:
        print(f"[SHOW] 使用精确点名：id={show_id}, angle={show_ang}")
    else:
        print(f"[SHOW] 使用索引：show_idx={show_idx}  (data.shuffle=False 确保稳定)")
    print("=" * 80)

    # 开关合并：优先命令行
    save_volume = _get_bool(args.save_volume, default=bool(eval_cfg.get("save_volume", True)))
    save_format = args.save_format or eval_cfg.get("save_format", "both")
    save_gt_noisy = _get_bool(args.save_gt_noisy, default=bool(eval_cfg.get("save_gt_noisy", True)))

    psnrs, ssims = [], []
    img_count = 0

    preds_all, preds_raw_all, ids_all = [], [], []
    gts_all, noisies_all, gts_raw_all, noisies_raw_all = [], [], [], []

    saved_flag = False  # 是否已经保存过 triptych

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

                    p = np.clip(p, 0.0, 1.0)
                    g = np.clip(g, 0.0, 1.0)
                    n = np.clip(n, 0.0, 1.0)

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

                    # ——保存三联图——
                    save_this = False
                    if (show_id is not None) and (show_ang is not None):
                        # 精确点名：id+angle
                        if int(id_val) == int(show_id) and int(angle) == int(show_ang):
                            save_this = True
                    else:
                        # 按 idx：需确保遍历顺序稳定（data.shuffle=False）
                        if img_count == show_idx:
                            save_this = True

                    if save_this and not saved_flag:
                        out_name = f"triptych_id{int(id_val)}_a{int(angle):03d}.png" if (show_id is not None) else f"triptych_{img_count:05d}.png"
                        save_triptych(n, p, g, os.path.join(trip_dir, out_name),
                                      title=f"id={id_val}, angle={angle}, A={A_val}, idx={img_count}")
                        saved_flag = True

                    img_count += 1

    # -------- 写出卷数据 --------
    if save_volume:
        if len(preds_all) > 0:
            vol = np.stack(preds_all, axis=0).astype(np.float32)
            if save_format in ("npy", "both"):
                np.save(os.path.join(save_dir, "pred_volume.npy"), vol)
            if save_format in ("tiff", "both"):
                tiff.imwrite(os.path.join(save_dir, "pred_volume.tiff"), vol, imagej=True)
            print(f"[OK] 预测体(归一化域) 已保存: pred_volume.npy / pred_volume.tiff  形状={vol.shape}")

        if len(preds_raw_all) > 0:
            vol_raw = np.stack(preds_raw_all, axis=0).astype(np.float32)
            if save_format in ("npy", "both"):
                np.save(os.path.join(save_dir, "pred_volume_raw.npy"), vol_raw)
            if save_format in ("tiff", "both"):
                tiff.imwrite(os.path.join(save_dir, "pred_volume_raw.tiff"), vol_raw, imagej=True)
            print(f"[OK] 预测体(RAW域) 已保存: pred_volume_raw.npy / pred_volume_raw.tiff  形状={vol_raw.shape}")

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
