# scripts/sample_i2sb_local.py
import os, argparse, yaml, csv, torch, numpy as np, tifffile as tiff
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, List

from ctprojfix.data.dataset import make_dataloader
from ctprojfix.models.i2sb_unet import I2SBUNet

def load_cfg(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _ceil_to(v, m): return ((v + m - 1) // m) * m if m > 0 else v

def _percentile_norm01(a: np.ndarray, p_lo=1, p_hi=99, eps=1e-6) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.percentile(a, [p_lo, p_hi])
    if hi - lo < eps:
        return np.zeros_like(a, dtype=np.float32)
    y = (a - lo) / (hi - lo + eps)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

def _center_crop_to(arr: Optional[np.ndarray], out_h: int, out_w: int) -> Optional[np.ndarray]:
    if arr is None: return None
    a = arr
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    H, W = a.shape[-2], a.shape[-1]
    if (H, W) == (out_h, out_w):
        return a.astype(np.float32, copy=False)
    top  = max((H - out_h) // 2, 0)
    left = max((W - out_w) // 2, 0)
    return a[top:top+out_h, left:left+out_w].astype(np.float32, copy=False)

def _match_to_pred_size(pred_hw: Tuple[int,int],
                        noisy: Optional[np.ndarray],
                        gt: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    ph, pw = int(pred_hw[0]), int(pred_hw[1])
    return _center_crop_to(noisy, ph, pw), _center_crop_to(gt, ph, pw)

def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def _save_tile_gray01(img01: np.ndarray, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img01, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def _save_tile_heat(err: np.ndarray, out_path: str, cmap: str, with_cb: bool):
    _ensure_dir(os.path.dirname(out_path))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if with_cb:
        fig = plt.figure(figsize=(4.4, 4), dpi=200)
        ax = fig.add_axes([0.0, 0.0, 0.9, 1.0])
        im = ax.imshow(err, cmap=cmap, vmin=0.0, vmax=None)
        ax.axis("off")
        cax = fig.add_axes([0.91, 0.1, 0.03, 0.8])
        plt.colorbar(im, cax=cax)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(4, 4), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(err, cmap=cmap, vmin=0.0, vmax=None)
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

def save_quad_and_tiles(noisy01: np.ndarray,
                        pred01: np.ndarray,
                        gt01: Optional[np.ndarray],
                        out_dir: str,
                        name: str,
                        tiles_cfg: dict):
    """
    输出：
    - out_dir/figs/{name}.png : 1x4 四连图（Noisy/Pred/GT/AbsDiff+colorbar）
    - 可选 out_dir/figs/{name}_tiles/* : Noisy/Pred/GT/AbsDiff 小图
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(out_dir)
    fig_dir = os.path.join(out_dir, "figs")
    _ensure_dir(fig_dir)

    n = np.clip(noisy01.astype(np.float32), 0, 1)
    p = np.clip(pred01.astype(np.float32),  0, 1)
    g = np.clip(gt01.astype(np.float32),    0, 1) if gt01 is not None else None

    # ---- 四连图 ----
    if g is not None:
        err = np.abs(p - g)
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        axes[0].imshow(n, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Noisy"); axes[0].axis("off")
        axes[1].imshow(p, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Pred");  axes[1].axis("off")
        axes[2].imshow(g, cmap="gray", vmin=0, vmax=1); axes[2].set_title("GT");    axes[2].axis("off")
        im = axes[3].imshow(err, cmap=str(tiles_cfg.get("cmap","magma")), vmin=0.0, vmax=None)
        axes[3].set_title("|Pred-GT|"); axes[3].axis("off")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        plt.tight_layout()
        out_path = os.path.join(fig_dir, f"{name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[FIG] {out_path}")
    else:
        # 没GT就退化成三联图
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(n, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Noisy"); axes[0].axis("off")
        axes[1].imshow(p, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Pred");  axes[1].axis("off")
        plt.tight_layout()
        out_path = os.path.join(fig_dir, f"{name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[FIG] {out_path}")
        return

    # ---- tiles（可选）----
    if not bool((tiles_cfg or {}).get("enable", False)):
        return
    tiles_dir = tiles_cfg.get("dir", None)
    if not tiles_dir:
        tiles_dir = os.path.join(fig_dir, f"{name}_tiles")
    _ensure_dir(tiles_dir)

    prefix = str(tiles_cfg.get("prefix", ""))
    fmt = str(tiles_cfg.get("format", "png"))
    cmap = str(tiles_cfg.get("cmap", "magma"))
    with_cb = bool(tiles_cfg.get("heat_with_colorbar", False))

    _save_tile_gray01(n, os.path.join(tiles_dir, f"{prefix}Noisy.{fmt}"))
    _save_tile_gray01(p, os.path.join(tiles_dir, f"{prefix}Pred.{fmt}"))
    _save_tile_gray01(g, os.path.join(tiles_dir, f"{prefix}GT.{fmt}"))
    _save_tile_heat(np.abs(p-g), os.path.join(tiles_dir, f"{prefix}AbsDiff_Pred_vs_GT.{fmt}"),
                    cmap=cmap, with_cb=with_cb)
    print(f"[TILE] -> {tiles_dir}")

# ---- 的指标 & 辅助工具 ----
def _psnr01(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """在[0,1]归一域上算 PSNR."""
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    mse = float(np.mean((x - y) ** 2))
    if mse < eps:
        return 99.0
    return 20.0 * float(np.log10(1.0 / np.sqrt(mse)))

def _ssim2d(x: np.ndarray, y: np.ndarray, dr: float = 1.0,
            K1: float = 0.01, K2: float = 0.03, sigma: float = 1.5) -> float:
    """简单 SSIM（和 eval 里的实现同风格）."""
    from scipy.ndimage import gaussian_filter
    x = np.clip(x, 0, dr).astype(np.float32)
    y = np.clip(y, 0, dr).astype(np.float32)
    C1, C2 = (K1*dr)**2, (K2*dr)**2
    mu1, mu2 = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
    mu1s, mu2s, mu12 = mu1*mu1, mu2*mu2, mu1*mu2
    s1 = gaussian_filter(x*x, sigma) - mu1s
    s2 = gaussian_filter(y*y, sigma) - mu2s
    s12 = gaussian_filter(x*y, sigma) - mu12
    num = (2*mu12 + C1) * (2*s12 + C2)
    den = (mu1s + mu2s + C1) * (s1 + s2 + C2) + 1e-12
    return float(np.mean(num / den))

def _get_scalar_from_batch(batch: Dict[str, Any], key: str, idx: int, default=None):
    if key not in batch:
        return default
    v = batch[key]
    if isinstance(v, torch.Tensor):
        if v.ndim == 0:
            return v.item()
        if v.ndim >= 1 and v.shape[0] > idx:
            return v[idx].item()
        return default
    if isinstance(v, (list, tuple)):
        try:
            return v[idx]
        except Exception:
            return v[0] if len(v) > 0 else default
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return float(v)
        if v.ndim >= 1 and v.shape[0] > idx:
            return v[idx]
        return default
    return v

@torch.no_grad()
def run(cfg_path: str, ckpt: Optional[str] = None, out: Optional[str] = None):
    cfg = load_cfg(cfg_path)
    train_cfg = cfg.get("train", {})
    dev = torch.device(train_cfg.get("device","cuda") if torch.cuda.is_available() else "cpu")

    # --------- dataloader（不要pop了 混亂了）---------
    data_cfg = dict(cfg["data"])
    # 评估采样时强制不打乱再兜底一次
    data_cfg["shuffle"] = bool(data_cfg.get("shuffle", False))
    dl = make_dataloader(data_cfg)

    # 选择 split如果存在 且 sample.split 指定
    want = str(cfg.get("sample", {}).get("split", "") or "").lower().strip()

    if isinstance(dl, dict):
        # split 模式：严格按 want 取；取不到就报错（不 silent fallback）
        if not want:
            raise RuntimeError("[I2SB local] make_dataloader returned split loaders, but sample.split not set.")
        if want not in dl or dl[want] is None:
            raise RuntimeError(f"[I2SB local] split '{want}' not found or empty. Available: {list(dl.keys())}")
        loader = dl[want]
        print(f"[I2SB local] using split='{want}' size={len(loader.dataset)}")
    else:
        # 非 split：就是单 loader（ids:[10] 这种）
        loader = dl
        print(f"[I2SB local] single loader size={len(loader.dataset)}")
        if len(loader.dataset) <= 0:
            raise RuntimeError("DataLoader is empty.")

    # --------- model & ckpt ----------
    mp = cfg["model"]["params"]
    net = I2SBUNet(in_ch=mp.get("in_ch",4), base=mp.get("base",64), depth=mp.get("depth",4),
                   emb_dim=mp.get("emb_dim",256), dropout=mp.get("dropout",0.0)).to(dev)
    ckpt_path = ckpt
    if ckpt_path is None:
        ckpt_dir = train_cfg.get("ckpt_dir", "checkpoints/i2sb_local")
        ckpt_prefix = train_cfg.get("ckpt_prefix", "i2sb_local")
        ckpt_path = cfg.get("sample", {}).get("ckpt", os.path.join(ckpt_dir, f"{ckpt_prefix}_best.pth"))
    if ckpt_path and os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        state = ck.get("state_dict", ck)
        net.load_state_dict(state, strict=True)
        print(f"[OK] load ckpt: {ckpt_path}")
    else:
        print(f"[WARN] ckpt not found: {ckpt_path} → use random init")
    net.eval()

    # --------- out dir ----------
    out_dir = out or cfg.get("sample", {}).get("out_dir", "outputs/diffusion/i2sb_local_1step")
    os.makedirs(out_dir, exist_ok=True)

    pick_id    = cfg.get("sample", {}).get("pick_id", None)
    pick_angle = cfg.get("sample", {}).get("pick_angle", None)

    # ------ helper: 前向并裁回原尺寸 ------
    def forward_and_restore(x1: torch.Tensor, depth_down:int, H:int, W:int) -> torch.Tensor:
        n_down = depth_down
        factor = 2 ** max(n_down, 0)
        Ht, Wt = _ceil_to(H, factor), _ceil_to(W, factor)
        if (Ht, Wt) != (H, W):
            pad = (0, Wt - W, 0, Ht - H)
            x1 = F.pad(x1, pad, mode="reflect")
        t_map = torch.ones((x1.shape[0],1,Ht,Wt), device=dev, dtype=torch.float32)
        xin = torch.cat([x1, t_map], dim=1)
        x0_hat = net(xin)
        # 裁回 H,W 并兜底插值
        if x0_hat.shape[-2] >= Ht and x0_hat.shape[-1] >= Wt:
            x0_hat = x0_hat[..., :Ht, :Wt]
        if x0_hat.shape[-2] >= H and x0_hat.shape[-1] >= W:
            x0_hat = x0_hat[..., :H, :W]
        if x0_hat.shape[-2:] != (H, W):
            x0_hat = F.interpolate(x0_hat, size=(H, W), mode="bilinear", align_corners=False)
        return x0_hat

    depth_down = len(getattr(net, "downs", []))

    # ---------- pick_id 模式：导出整卷 + 单张三联图 + metrics ----------
    if pick_id is not None:
        pick_id = int(pick_id)
        print(f"[I2SB local] single-id volume mode: pick_id={pick_id}, pick_angle={pick_angle}")

        preds, gts, noisies = [], [], []
        preds_raw, gts_raw, noisies_raw = [], [], []
        id_rows:     List[Any]      = []
        angle_rows:  List[int]      = []
        A_rows:      List[int]      = []
        noisy_lo_rows: List[Optional[float]] = []
        noisy_hi_rows: List[Optional[float]] = []
        gt_lo_rows:    List[Optional[float]] = []
        gt_hi_rows:    List[Optional[float]] = []
        psnr_rows:   List[Optional[float]] = []
        ssim_rows:   List[Optional[float]] = []

        best_one = None  # (diff, rec_dict)

        for batch in loader:
            # 解析 id 和 angle/A
            bid = batch.get("id", None)
            if bid is None:
                continue
            if torch.is_tensor(bid):
                ids = bid.detach().cpu().numpy().tolist()
            elif isinstance(bid, (list, tuple, np.ndarray)):
                ids = list(bid)
            else:
                ids = [bid]

            angles = batch.get("angle", None)
            if torch.is_tensor(angles):
                angles_np = angles.detach().cpu().numpy().tolist()
            elif isinstance(angles, (list, tuple, np.ndarray)):
                angles_np = list(angles)
            else:
                angles_np = [None] * len(ids)

            A_arr = batch.get("A", None)
            if torch.is_tensor(A_arr):
                A_np = A_arr.detach().cpu().numpy().tolist()
            elif isinstance(A_arr, (list, tuple, np.ndarray)):
                A_np = list(A_arr)
            else:
                A_np = [360] * len(ids)

            # 找到本 batch 中所有属于 pick_id 的索引
            hit = [i for i, v in enumerate(ids) if int(v) == pick_id]
            if not hit:
                continue

            # 前向
            x1_full = batch["inp"].to(dev).float() * 2.0 - 1.0  # (B, Cx1, H, W)
            B, _, H, W = x1_full.shape
            x0_hat = forward_and_restore(x1_full, depth_down, H, W)  # (B,1,H,W)
            x0 = ((x0_hat.clamp(-1,1) + 1.0) * 0.5).cpu().numpy()     # (B,1,H,W)
            noisy = batch["inp"][:, 0:1].cpu().numpy()                # (B,1,H,W)
            has_gt = ("gt" in batch)
            gt_np = batch["gt"].cpu().numpy() if has_gt else None

            for i in hit:
                pred_i = np.squeeze(x0[i,0]).astype(np.float32)
                noz_i  = np.squeeze(noisy[i,0]).astype(np.float32)
                gt_i   = (np.squeeze(gt_np[i,0]).astype(np.float32) if has_gt else None)

                preds.append(pred_i)
                noisies.append(noz_i)
                if has_gt:
                    gts.append(gt_i)

                # meta
                ai = angles_np[i] if angles_np[i] is not None else -1
                Ai = A_np[i]      if A_np[i]      is not None else 360
                try: ai = int(ai)
                except Exception: ai = -1
                try: Ai = int(Ai)
                except Exception: Ai = 360
                angle_rows.append(ai)
                A_rows.append(Ai)
                id_rows.append(int(pick_id))

                # scalar ranges
                noisy_lo = _get_scalar_from_batch(batch, "noisy_lo", i, default=None)
                noisy_hi = _get_scalar_from_batch(batch, "noisy_hi", i, default=None)
                gt_lo    = _get_scalar_from_batch(batch, "gt_lo",    i, default=None)
                gt_hi    = _get_scalar_from_batch(batch, "gt_hi",    i, default=None)
                noisy_lo_rows.append(noisy_lo)
                noisy_hi_rows.append(noisy_hi)
                gt_lo_rows.append(gt_lo)
                gt_hi_rows.append(gt_hi)

                # raw 重建
                if (gt_lo is not None) and (gt_hi is not None):
                    pr_raw = pred_i * (float(gt_hi) - float(gt_lo)) + float(gt_lo)
                    g_raw  = (gt_i * (float(gt_hi) - float(gt_lo)) + float(gt_lo)) if gt_i is not None else None
                else:
                    pr_raw, g_raw = None, None
                if (noisy_lo is not None) and (noisy_hi is not None):
                    n_raw = noz_i * (float(noisy_hi) - float(noisy_lo)) + float(noisy_lo)
                else:
                    n_raw = None
                preds_raw.append(pr_raw)
                gts_raw.append(g_raw)
                noisies_raw.append(n_raw)

                # 指标（归一化域上）
                if has_gt and gt_i is not None:
                    ps = _psnr01(pred_i, gt_i)
                    ss = _ssim2d(pred_i, gt_i, dr=1.0)
                else:
                    ps, ss = None, None
                psnr_rows.append(ps)
                ssim_rows.append(ss)

                # 供三联图挑选最接近 pick_angle 的帧
                if pick_angle is not None:
                    if isinstance(pick_angle, str) and pick_angle.lower() == "mid":
                        diff = 0  # mid 另行处理，通过最后取中位
                    else:
                        want = int(pick_angle)
                        diff = abs(ai - want) if ai >= 0 else 10**9
                    rec = dict(pred=pred_i, noisy=noz_i, gt=gt_i, angle=ai)
                    if (best_one is None) or (diff < best_one[0]):
                        best_one = (diff, rec)

        # 检查是否收集到任何帧
        if len(preds) == 0:
            raise RuntimeError(f"[I2SB local] pick_id={pick_id} not found in split or produced 0 frames.")

        # 角度排序（按 angle 升序，未知角度置后）
        order = list(range(len(preds)))
        order.sort(key=lambda k: (angle_rows[k] < 0, angle_rows[k]))
        preds        = [preds[k]        for k in order]
        noisies      = [noisies[k]      for k in order]
        if len(gts) > 0:
            gts      = [gts[k]          for k in order]
        preds_raw    = [preds_raw[k]    for k in order]
        gts_raw      = [gts_raw[k]      for k in order]
        noisies_raw  = [noisies_raw[k]  for k in order]
        id_rows      = [id_rows[k]      for k in order]
        angle_rows   = [angle_rows[k]   for k in order]
        A_rows       = [A_rows[k]       for k in order]
        noisy_lo_rows = [noisy_lo_rows[k] for k in order]
        noisy_hi_rows = [noisy_hi_rows[k] for k in order]
        gt_lo_rows    = [gt_lo_rows[k]    for k in order]
        gt_hi_rows    = [gt_hi_rows[k]    for k in order]
        psnr_rows     = [psnr_rows[k]     for k in order]
        ssim_rows     = [ssim_rows[k]     for k in order]

        # 导出整卷（归一化域）
        vol_pred = np.stack(preds,   axis=0).astype(np.float32)
        vol_nozy = np.stack(noisies, axis=0).astype(np.float32)
        tiff.imwrite(os.path.join(out_dir, "pred_volume.tiff"),  vol_pred, imagej=True)
        tiff.imwrite(os.path.join(out_dir, "noisy_volume.tiff"), vol_nozy, imagej=True)
        np.save(os.path.join(out_dir, "pred_volume.npy"),  vol_pred)
        np.save(os.path.join(out_dir, "noisy_volume.npy"), vol_nozy)
        if len(gts) > 0:
            vol_gt = np.stack(gts, axis=0).astype(np.float32)
            tiff.imwrite(os.path.join(out_dir, "gt_volume.tiff"), vol_gt, imagej=True)
            np.save(os.path.join(out_dir, "gt_volume.npy"), vol_gt)
        else:
            vol_gt = None
        print(f"[I2SB local][OK] saved volumes -> {out_dir}")

        # RAW 体（若上下界都存在）
        if all(v is not None for v in preds_raw):
            vol_pr = np.stack(preds_raw, axis=0).astype(np.float32)
            tiff.imwrite(os.path.join(out_dir, "pred_volume_raw.tiff"), vol_pr, imagej=True)
            np.save(os.path.join(out_dir, "pred_volume_raw.npy"), vol_pr)
        if vol_gt is not None and all(v is not None for v in gts_raw):
            vol_gr = np.stack(gts_raw, axis=0).astype(np.float32)
            tiff.imwrite(os.path.join(out_dir, "gt_volume_raw.tiff"), vol_gr, imagej=True)
            np.save(os.path.join(out_dir, "gt_volume_raw.npy"), vol_gr)
        if all(v is not None for v in noisies_raw):
            vol_nr = np.stack(noisies_raw, axis=0).astype(np.float32)
            tiff.imwrite(os.path.join(out_dir, "noisy_volume_raw.tiff"), vol_nr, imagej=True)
            np.save(os.path.join(out_dir, "noisy_volume_raw.npy"), vol_nr)

        # 写出 metrics.csv（和 pconv 一样的列）
        csv_path = os.path.join(out_dir, "metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx","id","angle","A",
                        "noisy_lo","noisy_hi","gt_lo","gt_hi",
                        "PSNR","SSIM"])
            for idx, (sid, ang, A_val,
                      nlo, nhi, glo, ghi,
                      ps, ss) in enumerate(zip(
                          id_rows, angle_rows, A_rows,
                          noisy_lo_rows, noisy_hi_rows,
                          gt_lo_rows, gt_hi_rows,
                          psnr_rows, ssim_rows
                      )):
                w.writerow([
                    idx,
                    sid,
                    ang,
                    A_val,
                    (float(nlo) if nlo is not None else ""),
                    (float(nhi) if nhi is not None else ""),
                    (float(glo) if glo is not None else ""),
                    (float(ghi) if ghi is not None else ""),
                    (float(ps)  if ps  is not None else ""),
                    (float(ss)  if ss  is not None else ""),
                ])
        print(f"[I2SB local][OK] saved metrics -> {csv_path}")

        # 选一张保存三联图 + 热力图
        if pick_angle is not None:
            if isinstance(pick_angle, str) and pick_angle.lower() == "mid":
                mid_idx = len(preds) // 2
                rec = dict(
                    pred=preds[mid_idx],
                    noisy=noisies[mid_idx],
                    gt=(gts[mid_idx] if len(gts) > 0 else None),
                    angle=angle_rows[mid_idx]
                )
            else:
                if best_one is None:
                    # 如果全部角度未知，退化为中位
                    mid_idx = len(preds) // 2
                    rec = dict(
                        pred=preds[mid_idx],
                        noisy=noisies[mid_idx],
                        gt=(gts[mid_idx] if len(gts) > 0 else None),
                        angle=angle_rows[mid_idx]
                    )
                else:
                    rec = best_one[1]

            ph, pw = rec["pred"].shape[-2], rec["pred"].shape[-1]
            nn, gg = _match_to_pred_size((ph, pw), rec["noisy"], rec["gt"])
            a = int(rec["angle"]) if rec["angle"] is not None else -1
            a3 = f"{max(a,0):03d}"

            tiles_cfg = (cfg.get("sample", {}).get("tiles", {}) or {})
            name = f"id{pick_id}_a{a3}"
            save_quad_and_tiles(nn, rec["pred"], gg, out_dir, name=name, tiles_cfg=tiles_cfg)

        print(f"[I2SB local][OK] single-id volume ready at: {out_dir}")

        # 统计整卷平均（只对 has_gt 的帧）
        ps_valid = [v for v in psnr_rows if v is not None]
        ss_valid = [v for v in ssim_rows if v is not None]
        if len(ps_valid) > 0:
            mean_psnr = float(np.mean(ps_valid))
            mean_ssim = float(np.mean(ss_valid))
            print(f"[RESULT][id={pick_id}] PSNR: {mean_psnr:.3f} dB  |  SSIM: {mean_ssim:.4f}  (N={len(ps_valid)})")
        else:
            print(f"[RESULT][id={pick_id}] no GT found; cannot compute mean metrics.")

        return

    # ---------- 否则：整卷导出（全 split） ----------
    preds = []
    for batch in loader:
        x1 = batch["inp"].to(dev).float() * 2.0 - 1.0
        B, _, H, W = x1.shape
        x0_hat = forward_and_restore(x1, depth_down, H, W)
        x0 = ((x0_hat.clamp(-1,1) + 1.0) * 0.5).cpu().numpy()
        preds += [np.squeeze(x0[i,0]).astype(np.float32) for i in range(B)]

    if len(preds) == 0:
        raise RuntimeError("[I2SB local] no frames predicted; empty split?")
    vol = np.stack(preds, axis=0).astype(np.float32)
    out_path = os.path.join(out_dir, "pred_volume.tiff")
    tiff.imwrite(out_path, vol, imagej=True)
    print("[I2SB local][OK] saved:", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/sample_i2sb_local.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args.cfg, ckpt=args.ckpt, out=args.out)

if __name__ == "__main__":
    main()
