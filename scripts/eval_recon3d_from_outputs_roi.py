# -*- coding: utf-8 -*-
# eval_recon3d_from_outputs.py  HU输出 + ROI标注 + ROI tiles + pipeline计时
# CBCT_Sino_FOV/scripts/eval_recon3d_from_outputs_roi.py
# 训练后的recon评估

import os, glob, csv, argparse, yaml, math, json, time
import numpy as np
import tifffile as tiff
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.ndimage import gaussian_filter

# Core recon APIs (wrappers)
from ctprojfix.recon import run_fdk, wce_fdk_reconstruct, wce_hsieh_fdk_reconstruct


# ========== IO & CFG ==========
def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def imread01(p):
    im = Image.open(p)
    if im.mode not in ("L", "I;16", "I"):
        im = im.convert("L")
    arr = np.array(im)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        mx = arr.max() if arr.max() > 0 else 1.0
        arr = arr / mx
    return arr.astype(np.float32)

def read_angles_from_csv(csv_path, A_default=360, return_index=False):
    a_list, A_list = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        R = csv.DictReader(f)
        for r in R:
            a = int(float(r.get("angle", 0)))
            A = r.get("A", None)
            A = int(float(A)) if (A is not None and str(A).strip() != "") else int(A_default)
            a_list.append(a); A_list.append(A)
    a = np.asarray(a_list, np.int32)
    A = np.asarray(A_list, np.int32)
    ang = (2.0 * np.pi * (a / np.maximum(1, A))).astype(np.float32)
    return (ang, A, a) if return_index else (ang, A)


# ========== 体堆栈读取 ==========
PRIORITY_SETS = [
    ("_raw.npy", np.load),
    ("_raw.tiff", tiff.imread),
    (".npy", np.load),
    (".tiff", tiff.imread),
]

def find_volume(from_dir, stem: str, explicit_path: str | None):
    if explicit_path:
        p = explicit_path if os.path.isabs(explicit_path) else os.path.join(from_dir, explicit_path)
        if os.path.isfile(p):
            v = (np.load(p) if p.endswith('.npy') else tiff.imread(p)).astype(np.float32)
            return v, p
    for suf, loader in PRIORITY_SETS:
        patt = os.path.join(from_dir, f"{stem}{suf}")
        for cand in glob.glob(patt):
            try:
                v = loader(cand).astype(np.float32)
                return v, cand
            except Exception:
                pass
    return None, None

def stack_with_pattern(dir_path, patt, loader):
    files = sorted(glob.glob(os.path.join(dir_path, patt)))
    if not files:
        return None, []
    arrs = [loader(f) for f in files]
    return np.stack(arrs, axis=0).astype(np.float32), files

def stack_raw_npy(dir_path, patt):
    return stack_with_pattern(dir_path, patt, lambda p: np.load(p).astype(np.float32))

def stack_png_01(dir_path, patt):
    return stack_with_pattern(dir_path, patt, imread01)


# ========== 工具 ==========
def wrap_0_2pi(x):
    return np.mod(x, 2*np.pi).astype(np.float32)

def apply_uv_ops(stack_TVU, angles_rad, rot180=False, swap_uv=False, flip_u=False, flip_v=False):
    S = stack_TVU
    ang = angles_rad.copy()
    if rot180:
        S = S[:, ::-1, ::-1]; ang = wrap_0_2pi(ang + np.pi)
    if swap_uv:
        S = np.transpose(S, (0, 2, 1))
    if flip_v:
        S = S[:, ::-1, :]
    if flip_u:
        S = S[:, :, ::-1]
    return S, ang

def dedup_by_angle(a_idx, angles, tol_rad):
    keep = []
    if a_idx is not None and len(a_idx) == len(angles):
        seen = set()
        for i, a in enumerate(a_idx):
            if a not in seen:
                keep.append(i); seen.add(a)
        return np.array(keep, dtype=np.int32)
    last = None
    for i, ang in enumerate(angles):
        if last is None or abs(ang - last) > tol_rad:
            keep.append(i); last = ang
    return np.array(keep, dtype=np.int32)


# ========== 显示/指标 ==========
def ssim2d(x, y, dr=1.0, K1=0.01, K2=0.03, sigma=1.5):
    C1, C2 = (K1*dr)**2, (K2*dr)**2
    mu1, mu2 = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
    mu1s, mu2s, mu12 = mu1*mu1, mu2*mu2, mu1*mu2
    s1 = gaussian_filter(x*x, sigma) - mu1s
    s2 = gaussian_filter(y*y, sigma) - mu2s
    s12 = gaussian_filter(x*y, sigma) - mu12
    num = (2*mu12 + C1) * (2*s12 + C2)
    den = (mu1s + mu2s + C1) * (s1 + s2 + C2) + 1e-12
    return float(np.mean(num/den))

def vol_psnr(x, y, dr=1.0):
    x = np.clip(x, 0, dr); y = np.clip(y, 0, dr)
    mse = np.mean((x - y) ** 2, dtype=np.float64)
    return 99.0 if mse < 1e-12 else float(20.0 * math.log10(dr / math.sqrt(mse)))

def norm01(v):
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max() + 1e-6)
    return np.clip((v - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)

def parse_slices(spec, Z):
    if isinstance(spec, (int, np.integer)):
        idxs = [int(spec)]
    elif isinstance(spec, str):
        idxs = [Z//2] if spec.lower() == "mid" else [int(spec)]
    elif isinstance(spec, (list, tuple)):
        idxs = []
        for s in spec:
            if isinstance(s, str) and s.lower() == "mid":
                idxs.append(Z//2)
            else:
                idxs.append(int(s))
    else:
        idxs = [Z//2]
    seen, out = set(), []
    for i in idxs:
        ii = min(max(0, int(i)), Z-1)
        if ii not in seen:
            out.append(ii); seen.add(ii)
    return out


# ====== HU 可视化窗口 ======
def _get_vmin_vmax_for_hu(vol_hu: np.ndarray, display_cfg: dict):
    mode = str((display_cfg or {}).get("mode", "percentile")).lower()
    if mode == "window":
        wl = float((display_cfg or {}).get("wl", 40.0))
        ww = float((display_cfg or {}).get("ww", 400.0))
        vmin, vmax = wl - ww/2.0, wl + ww/2.0
        return vmin, vmax
    # percentile
    p = (display_cfg or {}).get("percentile", [1, 99])
    p1, p2 = float(p[0]), float(p[1])
    vmin = np.percentile(vol_hu, p1)
    vmax = np.percentile(vol_hu, p2)
    if vmax <= vmin:
        vmin, vmax = float(vol_hu.min()), float(vol_hu.max() + 1e-6)
    return vmin, vmax

def save_axial_triptych_hu(vol_noisy_hu, vol_pred_hu, vol_gt_hu, z_indices, out_dir, tag, display_cfg):
    """
    原始输出：保持不变（不画 ROI，不导出 tiles）
    """
    os.makedirs(out_dir, exist_ok=True)
    for z in z_indices:
        n = vol_noisy_hu[z]; p = vol_pred_hu[z]; g = vol_gt_hu[z]
        vmin, vmax = _get_vmin_vmax_for_hu(np.concatenate([n.ravel(), p.ravel(), g.ravel()]), display_cfg)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(n, cmap="gray", vmin=vmin, vmax=vmax); axes[0].set_title(f"Noisy (HU)  z={z} {tag}"); axes[0].axis("off")
        axes[1].imshow(p, cmap="gray", vmin=vmin, vmax=vmax); axes[1].set_title(f"Pred  (HU)  z={z} {tag}"); axes[1].axis("off")
        axes[2].imshow(g, cmap="gray", vmin=vmin, vmax=vmax); axes[2].set_title(f"GT    (HU)  z={z} {tag}"); axes[2].axis("off")
        plt.tight_layout()
        fp = os.path.join(out_dir, f"axial_triptych_hu{tag}_z{z:04d}.png")
        plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[FIG] {fp}")


# ROI helpers 
def _clip_int(v, lo, hi):
    return int(max(lo, min(hi, v)))

def _roi_from_cfg(roi_item: dict, H: int, W: int):
    """
    ROI item supports:
      - center_xy_frac: [x_frac, y_frac] in [0,1]   (x=col, y=row)
      - center_xy_px:   [x_px, y_px]
      - size_wh_px:     [w, h]
    returns: (x0, y0, w, h) in pixel coords (x=col, y=row)
    """
    if roi_item is None:
        return None

    if "center_xy_px" in roi_item:
        xc, yc = roi_item["center_xy_px"]
        xc, yc = float(xc), float(yc)
    else:
        fx, fy = roi_item.get("center_xy_frac", [0.5, 0.5])
        xc, yc = float(fx) * (W - 1), float(fy) * (H - 1)

    w, h = roi_item.get("size_wh_px", [128, 128])
    w, h = int(w), int(h)

    x0 = int(round(xc - w / 2))
    y0 = int(round(yc - h / 2))

    # clamp to image bounds
    x0 = _clip_int(x0, 0, W - 1)
    y0 = _clip_int(y0, 0, H - 1)
    # ensure roi fits
    if x0 + w > W: x0 = max(0, W - w)
    if y0 + h > H: y0 = max(0, H - h)

    x0 = _clip_int(x0, 0, max(0, W - w))
    y0 = _clip_int(y0, 0, max(0, H - h))
    return x0, y0, w, h

def _crop(img2d: np.ndarray, x0: int, y0: int, w: int, h: int):
    return img2d[y0:y0+h, x0:x0+w]

def save_axial_triptych_hu_with_rois(vol_noisy_hu, vol_pred_hu, vol_gt_hu,
                                     z_indices, out_dir, tag, display_cfg,
                                     roi_cfg: dict | None):
    """
    新增输出（不破坏原始 triptych）：
      1) axial_triptych_hu*_roi_zXXXX.png : 三联图上画 ROI 框
      2) axial_triptych_hu*_roi_zXXXX_tiles/ : 保存 Noisy/Pred/GT 的 ROI crops
    """
    if not (roi_cfg or {}).get("enable", False):
        return

    rois = (roi_cfg or {}).get("list", []) or []
    if len(rois) == 0:
        return

    tiles_cfg = (roi_cfg or {}).get("tiles", {}) or {}
    tiles_enable = bool(tiles_cfg.get("enable", True))
    tiles_format = str(tiles_cfg.get("format", "png")).lower()
    same_range = bool(tiles_cfg.get("use_same_vminvmax_as_main", True))

    os.makedirs(out_dir, exist_ok=True)

    for z in z_indices:
        n = vol_noisy_hu[z]; p = vol_pred_hu[z]; g = vol_gt_hu[z]

        # 主图显示范围：与原始一致
        if same_range:
            vmin, vmax = _get_vmin_vmax_for_hu(np.concatenate([n.ravel(), p.ravel(), g.ravel()]), display_cfg)
        else:
            vmin, vmax = _get_vmin_vmax_for_hu(g, display_cfg)

        H, W = g.shape

        # ---- 1) ROI 标注 triptych ----
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ims = [
            (axes[0], n, f"Noisy (HU)  z={z} {tag}"),
            (axes[1], p, f"Pred  (HU)  z={z} {tag}"),
            (axes[2], g, f"GT    (HU)  z={z} {tag}"),
        ]
        for ax, img, title in ims:
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")

            for item in rois:
                r = _roi_from_cfg(item, H, W)
                if r is None:
                    continue
                x0, y0, w, h = r

                color = item.get("color", "lime")   # 默认绿
                rect = Rectangle((x0, y0), w, h, fill=False, linewidth=2,
                                edgecolor=color, linestyle='-')
                ax.add_patch(rect)


                # 标注文字（ROI-1/ROI-2 论文）
                name = str(item.get("name", "ROI"))
                ax.text(x0, max(0, y0-6), name, fontsize=8,
                        bbox=dict(facecolor='black', alpha=0.3, pad=1),
                        color='white')

        plt.tight_layout()
        fp_roi = os.path.join(out_dir, f"axial_triptych_hu{tag}_roi_z{z:04d}.png")
        plt.savefig(fp_roi, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[FIG] {fp_roi}")

        # ---- 2) ROI tiles ----
        if tiles_enable:
            base = os.path.splitext(os.path.basename(fp_roi))[0]
            tiles_dir = tiles_cfg.get("dir", None)
            if tiles_dir is None:
                tiles_dir = os.path.join(out_dir, f"{base}_tiles")
            os.makedirs(tiles_dir, exist_ok=True)

            for item in rois:
                r = _roi_from_cfg(item, H, W)
                if r is None:
                    continue
                x0, y0, w, h = r
                name = str(item.get("name", "ROI"))

                cn = _crop(n, x0, y0, w, h)
                cp = _crop(p, x0, y0, w, h)
                cg = _crop(g, x0, y0, w, h)

                # tiles 显示范围：默认用主图范围（保证横向可比）
                tvmin, tvmax = (vmin, vmax) if same_range else _get_vmin_vmax_for_hu(
                    np.concatenate([cn.ravel(), cp.ravel(), cg.ravel()]), display_cfg
                )

                for lab, im in [("noisy", cn), ("pred", cp), ("gt", cg)]:
                    fpt = os.path.join(
                        tiles_dir,
                        f"{name}_{lab}_z{z:04d}_x{x0}_y{y0}_w{w}_h{h}.{tiles_format}"
                    )
                    # plt.figure(figsize=(3.2, 3.2))
                    # plt.imshow(im, cmap="gray", vmin=tvmin, vmax=tvmax)
                    # plt.title(f"{name} | {lab} | z={z}")
                    # plt.axis("off")
                    # plt.tight_layout()
                    # plt.savefig(fpt, dpi=200, bbox_inches="tight")
                    # plt.close()
                    plt.figure(figsize=(3.2, 3.2))
                    plt.imshow(im, cmap="gray", vmin=tvmin, vmax=tvmax)
                    plt.axis("off")
                    plt.savefig(fpt, dpi=200, bbox_inches="tight", pad_inches=0)
                    plt.close()


            print(f"[TILES] {tiles_dir}")


# ===== 强度域->线积分域（若需）=====
def _to_line_if_needed(stack_TVU: np.ndarray, need: bool, eps: float = 1e-6) -> np.ndarray:
    """若 need=True，则把强度域 I 转线积分 L=-log(I)；对<=0做裁剪。"""
    if not need:
        return stack_TVU
    S = np.asarray(stack_TVU, dtype=np.float32, order="C")
    S = np.clip(S, float(eps), None)
    L = -np.log(S)
    return L.astype(np.float32)


# ====== HU 标定与映射 ======
def estimate_hu_calib_from_gt(vol_gt: np.ndarray) -> dict:
    """
    基于 GT 重建体自动估计空气/水参考：
    - 先用百分位分割：mask = vol > P60 近似人体（去空气）
    - air_ref = (~mask) 体素的中位数（若几乎没有空气体素，则退化为全体 P1）
    - water_ref = mask 体素的中位数（若掩膜很小则退化为全体中位）
    - HU = 1000 * (x - water_ref) / (water_ref - air_ref)
    """
    v = np.asarray(vol_gt, dtype=np.float32)
    p60 = np.percentile(v, 60.0)
    mask = v > p60
    if np.count_nonzero(~mask) < max(100, int(v.size * 0.001)):  # 空气太少
        air_ref = float(np.percentile(v, 1.0))
    else:
        air_ref = float(np.median(v[~mask]))
    if np.count_nonzero(mask) < max(100, int(v.size * 0.001)):    # 组织太少
        water_ref = float(np.median(v))
    else:
        water_ref = float(np.median(v[mask]))
    if abs(water_ref - air_ref) < 1e-6:
        air_ref = float(np.percentile(v, 1.0))
        water_ref = float(np.percentile(v, 80.0))
    scale = 1000.0 / (water_ref - air_ref + 1e-12)
    calib = {"air_ref": air_ref, "water_ref": water_ref, "scale": scale}
    return calib

def to_hu(vol: np.ndarray, calib: dict, clamp=(-1000.0, 1000.0)) -> np.ndarray:
    water_ref = float(calib["water_ref"])
    scale = float(calib["scale"])
    hu = scale * (vol.astype(np.float32) - water_ref)
    if clamp is not None:
        hu = np.clip(hu, clamp[0], clamp[1])
    return hu.astype(np.float32)


# ========== main ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/eval.yaml")
    ap.add_argument("--pred", default=None)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--noisy", default=None)
    ap.add_argument("--with_wce", action="store_true", help="同时运行 WCE + FDK（Gaussian/Hsieh 取决于 config）")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    recon = cfg.get("recon3d", {})

    # 域开关
    need_log = bool(recon.get("input_is_intensity", False))
    log_eps  = float(recon.get("log_eps", 1.0e-6))
    print(f"[CFG] input_is_intensity={need_log}  log_eps={log_eps:g}")

    from_dir = recon.get("from_dir", os.path.join(cfg.get("eval", {}).get("out_dir", "outputs/eval")))
    save_dir = recon.get("save_dir", os.path.join(from_dir, "recon3d"))
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(from_dir, recon.get("metrics_csv", "metrics.csv"))

    pred_file = args.pred or recon.get("pred_file")
    gt_file   = args.gt   or recon.get("gt_file")
    noz_file  = args.noisy or recon.get("noisy_file")

    pred, pred_src = find_volume(from_dir, "pred_volume", pred_file)
    gt,   gt_src   = find_volume(from_dir, "gt_volume", gt_file)
    noz,  noz_src  = find_volume(from_dir, "noisy_volume", noz_file)

    print(f"[PATH] pred_volume -> {pred_src}")
    print(f"[PATH] gt_volume   -> {gt_src}")
    print(f"[PATH] noisy_volume-> {noz_src}")

    use_raw = any(s and ("_raw" in os.path.basename(s)) for s in [pred_src, gt_src, noz_src])

    if pred is None or gt is None or noz is None:
        print("[WARN] 整卷体文件未完整找到，尝试逐帧堆栈兜底...")
        p2, pfiles = stack_raw_npy(from_dir, recon.get("pattern_pred_raw", "pred_raw_*.npy"))
        g2, gfiles = stack_raw_npy(from_dir, recon.get("pattern_gt_raw", "gt_raw_*.npy"))
        n2, nfiles = stack_raw_npy(from_dir, recon.get("pattern_noisy_raw", "noisy_raw_*.npy"))
        if p2 is None or g2 is None or n2 is None:
            print("[WARN] RAW 逐帧不存在，改用 PNG 归一化域兜底（仅排错）")
            p2, pfiles = stack_png_01(from_dir, recon.get("pattern_pred_png", "pred_*.png"))
            g2, gfiles = stack_png_01(from_dir, recon.get("pattern_gt_png", "gt_*.png"))
            n2, nfiles = stack_png_01(from_dir, recon.get("pattern_noisy_png", "noisy_*.png"))
        pred, gt, noz = p2, g2, n2
        if pfiles: print(f"[PATH] pred frames: {pfiles[0]} ... {pfiles[-1]}  (total {len(pfiles)})")
        if gfiles: print(f"[PATH] gt   frames: {gfiles[0]} ... {gfiles[-1]}  (total {len(gfiles)})")
        if nfiles: print(f"[PATH] noisy frames: {nfiles[0]} ... {nfiles[-1]}  (total {len(nfiles)})")
        assert pred is not None and gt is not None and noz is not None, "未找到任何可用投影文件."

    def to_TVU(v):
        v = np.asarray(v, dtype=np.float32)
        if v.ndim == 3: return v
        raise ValueError(f"期望三维数组 (T,V,U)，收到形状: {v.shape}")

    pred = to_TVU(pred); gt = to_TVU(gt); noz = to_TVU(noz)
    N = min(pred.shape[0], gt.shape[0], noz.shape[0])
    pred, gt, noz = pred[:N], gt[:N], noz[:N]

    A_default = int(recon.get("A_default", 360))
    angles, A_arr, a_idx = read_angles_from_csv(csv_path, A_default=A_default, return_index=True)
    if len(angles) != N:
        M = min(len(angles), N)
        print(f"[WARN] 角度数 {len(angles)} 与帧数 {N} 不一致，裁剪到 {M}.")
        angles = angles[:M]; A_arr = A_arr[:M]; a_idx = a_idx[:M]
        pred = pred[:M]; gt = gt[:M]; noz = noz[:M]; N = M

    angle_direction = int(recon.get("angle_direction", 1))
    angle_offset_deg = float(recon.get("angle_offset_deg", 0.0))
    ang = wrap_0_2pi(angle_direction * (angles + np.deg2rad(angle_offset_deg)))

    if bool(recon.get("dedup", True)):
        tol_deg = float(recon.get("dedup_tol_deg", 0.2))
        keep = dedup_by_angle(a_idx, ang, np.deg2rad(tol_deg))
        if keep.size < N:
            print(f"[FIX] 去重：从 {N} -> {keep.size}（移除重叠视角）")
            pred = pred[keep]; gt = gt[keep]; noz = noz[keep]
            ang = ang[keep]; A_arr = A_arr[keep]; a_idx = a_idx[keep]
            N = keep.size

    du   = float(recon.get("du", 2.0))
    dv   = float(recon.get("dv", 2.0))
    SOD  = float(recon.get("SOD", 600.0))
    ODD  = float(recon.get("ODD", 900.0))
    vol_shape = [int(x) for x in recon.get("vol_shape", [512,512,512])]
    filt = str(recon.get("filter", "ram-lak"))
    gpu  = int(recon.get("gpu", 0))

    sort_by_angle= bool(recon.get("sort_by_angle", False))
    rot180  = bool(recon.get("rot180", False))
    swap_uv = bool(recon.get("swap_uv", False))
    flip_u  = bool(recon.get("flip_u", False))
    flip_v  = bool(recon.get("flip_v", False))

    pred, ang = apply_uv_ops(pred, ang, rot180=rot180, swap_uv=swap_uv, flip_u=flip_u, flip_v=flip_v)
    gt,   _   = apply_uv_ops(gt,   ang, rot180=rot180, swap_uv=swap_uv, flip_u=flip_u, flip_v=flip_v)
    noz,  _   = apply_uv_ops(noz,  ang, rot180=rot180, swap_uv=swap_uv, flip_u=flip_u, flip_v=flip_v)

    det_v, det_u = pred.shape[1], pred.shape[2]
    print(f"[CFG] use_raw={use_raw}  views={N}  det(V,U)=({det_v},{det_u})  filter={filt} gpu={gpu}")
    print(f"[CFG] du={du} dv={dv} SOD={SOD} ODD={ODD} vol_shape[Z,Y,X]={vol_shape}")

    display_cfg = recon.get("display", {}) or {}
    roi_cfg = recon.get("roi", {}) or {}

    # ======= FDK 基线（Pred / GT / Noisy）=======
    def recon_and_save(tag, P, G, N_, A_):
        t0_all = time.perf_counter()

        if P is None or G is None or N_ is None or A_ is None or len(A_) == 0:
            print(f"[SKIP] {tag}: empty set."); return

        if sort_by_angle:
            order = np.argsort(A_)
            P, G, N_, A_ = P[order], G[order], N_[order], A_[order]

        print(f"[INFO] FDK_CUDA [{tag}] ... (views={len(A_)})")

        # 若输入是强度域，这里统一转线积分域
        P_line = _to_line_if_needed(P, need=need_log, eps=log_eps)
        G_line = _to_line_if_needed(G, need=need_log, eps=log_eps)
        N_line = _to_line_if_needed(N_, need=need_log, eps=log_eps)

        vol_gt, t_gt = run_fdk(G_line, vol_shape, du, dv, det_u, det_v, A_, SOD, ODD, gpu, filt)
        vol_pr, t_pr = run_fdk(P_line, vol_shape, du, dv, det_u, det_v, A_, SOD, ODD, gpu, filt)
        vol_nz, t_nz = run_fdk(N_line, vol_shape, du, dv, det_u, det_v, A_, SOD, ODD, gpu, filt)
        print(f"[TIME][{tag}] GT={t_gt:.3f}s, PRED={t_pr:.3f}s, NOISY={t_nz:.3f}s")

        # ---- HU 标定（基于 GT），三者共用同一组参数 ----
        hu_calib = estimate_hu_calib_from_gt(vol_gt)
        with open(os.path.join(save_dir, f"hu_calib_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(hu_calib, f, ensure_ascii=False, indent=2)

        vol_gt_hu = to_hu(vol_gt, hu_calib)
        vol_pr_hu = to_hu(vol_pr, hu_calib)
        vol_nz_hu = to_hu(vol_nz, hu_calib)

        # 保存 HU 体数据（float32）
        tiff.imwrite(os.path.join(save_dir, f"recon_gt_{tag}_hu.tif"),   vol_gt_hu.astype(np.float32))
        tiff.imwrite(os.path.join(save_dir, f"recon_pred_{tag}_hu.tif"), vol_pr_hu.astype(np.float32))
        tiff.imwrite(os.path.join(save_dir, f"recon_noisy_{tag}_hu.tif"),vol_nz_hu.astype(np.float32))
        np.save(os.path.join(save_dir, f"recon_gt_{tag}_hu.npy"),   vol_gt_hu.astype(np.float32))
        np.save(os.path.join(save_dir, f"recon_pred_{tag}_hu.npy"), vol_pr_hu.astype(np.float32))
        np.save(os.path.join(save_dir, f"recon_noisy_{tag}_hu.npy"),vol_nz_hu.astype(np.float32))

        # HU 三联图（原图，保持不变）
        z_list = parse_slices(recon.get("slices", ["mid"]), vol_gt_hu.shape[0])
        save_axial_triptych_hu(vol_nz_hu, vol_pr_hu, vol_gt_hu, z_list, save_dir, tag=f"_{tag}", display_cfg=display_cfg)

        # ROI 标注 + tiles
        save_axial_triptych_hu_with_rois(
            vol_nz_hu, vol_pr_hu, vol_gt_hu,
            z_list, save_dir, tag=f"_{tag}", display_cfg=display_cfg,
            roi_cfg=roi_cfg
        )

        # 指标仍按[0,1]归一矩阵计算，保持与历史对比一致
        gtn = norm01(vol_gt); prn = norm01(vol_pr); nzn = norm01(vol_nz)
        psnr_pred = vol_psnr(prn, gtn, 1.0)
        ssim_pred = float(np.mean([ssim2d(prn[i], gtn[i], dr=1.0) for i in range(prn.shape[0])]))
        psnr_nozy = vol_psnr(nzn, gtn, 1.0)
        ssim_nozy = float(np.mean([ssim2d(nzn[i], gtn[i], dr=1.0) for i in range(nzn.shape[0])]))
        with open(os.path.join(save_dir, f"recon_metrics_{tag}.txt"), "w") as f:
            f.write(f"[{tag}] Pred vs GT:  PSNR={psnr_pred:.3f} dB  SSIM_Axial={ssim_pred:.4f}\n")
            f.write(f"[{tag}] Noisy vs GT: PSNR={psnr_nozy:.3f} dB  SSIM_Axial={ssim_nozy:.4f}\n")

        # pipeline 总计时（更工程化：io+fdk+hu+save）
        t_all = time.perf_counter() - t0_all
        with open(os.path.join(save_dir, f"recon_timing_{tag}.txt"), "w", encoding="utf-8") as f:
            f.write(f"[{tag}] pipeline_total_sec={t_all:.6f}\n")
            f.write(f"[{tag}] fdk_sec_gt={t_gt:.6f} fdk_sec_pred={t_pr:.6f} fdk_sec_noisy={t_nz:.6f}\n")
        print(f"[TIME][{tag}] pipeline_total={t_all:.3f}s (io+fdk+hu+save)")

    # —— WCE 统一调度（支持 gaussian/hsieh）——
    def recon_and_save_wce(tag, noisy_TVU, gt_TVU, A_):
        t0_all = time.perf_counter()

        if noisy_TVU is None or gt_TVU is None or A_ is None or len(A_) == 0:
            print(f"[SKIP] {tag}: empty set."); return
        if sort_by_angle:
            order = np.argsort(A_)
            noisy_TVU, gt_TVU, A_ = noisy_TVU[order], gt_TVU[order], A_[order]

        geom = {
            "du": du, "dv": dv,
            "det_u": det_u, "det_v": det_v,
            "SOD": SOD, "ODD": ODD,
            "vol_shape": vol_shape,
            "filt": filt, "gpu": gpu,
        }

        wcfg = recon.get("wce", {}) or {}
        method = str(wcfg.get("method", "hsieh")).lower()
        ds_data = int(cfg.get("data", {}).get("downsample", 1))

        if method == "hsieh":
            hsieh_params = dict(wcfg)
            hsieh_params["trunc_left"]  = int(wcfg.get("trunc_left",  0))
            hsieh_params["trunc_right"] = int(wcfg.get("trunc_right", 0))
            hsieh_params["edge_band"]   = int(wcfg.get("edge_band", 16))
            hsieh_params["radius_px_cap"] = int(wcfg.get("radius_px_cap", 4096))
            hsieh_params["min_valid"]   = float(wcfg.get("min_valid", 1e-6))
            hsieh_params["clip_exp"]    = float(wcfg.get("clip_exp", 20.0))
            hsieh_params["input_domain"] = str(wcfg.get("input_domain", "intensity")).lower()
            hsieh_params["fdk_expect"]   = str(wcfg.get("fdk_expect",   "line")).lower()
            if not bool(wcfg.get("pixels_are_post_downsample", False)):
                hsieh_params["pixels_are_post_downsample"] = False
                hsieh_params["downsample"] = ds_data
            else:
                hsieh_params["pixels_are_post_downsample"] = True
                hsieh_params["downsample"] = 1

            print("[INFO] WCE method = Hsieh-known-trunc  params=" +
                  json.dumps(hsieh_params, ensure_ascii=False))

            input_hint = noz_src or from_dir
            vol_wce, t_wce, _ = wce_hsieh_fdk_reconstruct(
                noisy_TVU, A_, geom,
                hsieh_params=hsieh_params,
                input_file_hint=input_hint
            )

        elif method == "gaussian":
            wce_params = dict(wcfg)
            print("[INFO] WCE method = Gaussian  params=" +
                  json.dumps(wce_params, ensure_ascii=False))
            vol_wce, t_wce, _ = wce_fdk_reconstruct(
                noisy_TVU, A_, geom, wce_params=wce_params
            )
        else:
            print(f"[WARN] Unknown WCE method={method}, skip.")
            return

        # baseline 的 GT/NOISY：按 need_log 决定是否转 log
        G_line = _to_line_if_needed(gt_TVU,    need=need_log, eps=log_eps)
        N_line = _to_line_if_needed(noisy_TVU, need=need_log, eps=log_eps)
        vol_gt, t_gt = run_fdk(G_line, vol_shape, du, dv, det_u, det_v, A_, SOD, ODD, gpu, filt)
        vol_nz, t_nz = run_fdk(N_line, vol_shape, du, dv, det_u, det_v, A_, SOD, ODD, gpu, filt)
        print(f"[TIME][WCE {tag}] GT={t_gt:.3f}s, WCE={t_wce:.3f}s, NOISY={t_nz:.3f}s")

        # HU 标定（基于 GT）
        hu_calib = estimate_hu_calib_from_gt(vol_gt)
        with open(os.path.join(save_dir, f"hu_calib_wce_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(hu_calib, f, ensure_ascii=False, indent=2)

        vol_gt_hu  = to_hu(vol_gt, hu_calib)
        vol_wce_hu = to_hu(vol_wce, hu_calib)
        vol_nz_hu  = to_hu(vol_nz, hu_calib)

        # 保存 HU 体数据
        tiff.imwrite(os.path.join(save_dir, f"recon_gt_wce_{method}_{tag}_hu.tif"),   vol_gt_hu.astype(np.float32))
        tiff.imwrite(os.path.join(save_dir, f"recon_wce_{method}_{tag}_hu.tif"),      vol_wce_hu.astype(np.float32))
        tiff.imwrite(os.path.join(save_dir, f"recon_noisy_wce_{method}_{tag}_hu.tif"),vol_nz_hu.astype(np.float32))
        np.save(os.path.join(save_dir, f"recon_gt_wce_{method}_{tag}_hu.npy"),   vol_gt_hu.astype(np.float32))
        np.save(os.path.join(save_dir, f"recon_wce_{method}_{tag}_hu.npy"),      vol_wce_hu.astype(np.float32))
        np.save(os.path.join(save_dir, f"recon_noisy_wce_{method}_{tag}_hu.npy"),vol_nz_hu.astype(np.float32))

        # HU 三联图（WCE：原图）
        z_list = parse_slices(recon.get("slices", ["mid"]), vol_gt_hu.shape[0])
        save_axial_triptych_hu(vol_nz_hu, vol_wce_hu, vol_gt_hu, z_list, save_dir, tag=f"_wce_{method}_{tag}", display_cfg=display_cfg)

        # ROI 标注 + tiles（WCE）
        save_axial_triptych_hu_with_rois(
            vol_nz_hu, vol_wce_hu, vol_gt_hu,
            z_list, save_dir, tag=f"_wce_{method}_{tag}", display_cfg=display_cfg,
            roi_cfg=roi_cfg
        )

        # 指标仍按[0,1]归一矩阵
        gtn = norm01(vol_gt); wcn = norm01(vol_wce); nzn = norm01(vol_nz)
        psnr_wce = vol_psnr(wcn, gtn, 1.0)
        ssim_wce = float(np.mean([ssim2d(wcn[i], gtn[i], dr=1.0) for i in range(wcn.shape[0])]))
        psnr_noz = vol_psnr(nzn, gtn, 1.0)
        ssim_noz = float(np.mean([ssim2d(nzn[i], gtn[i], dr=1.0) for i in range(nzn.shape[0])]))
        with open(os.path.join(save_dir, f"recon_metrics_wce_{method}_{tag}.txt"), "w") as f:
            f.write(f"[WCE {method} {tag}] WCE vs GT:  PSNR={psnr_wce:.3f} dB  SSIM_Axial={ssim_wce:.4f}\n")
            f.write(f"[WCE {method} {tag}] Noisy vs GT: PSNR={psnr_noz:.3f} dB  SSIM_Axial={ssim_noz:.4f}\n")

        # pipeline 总计时
        t_all = time.perf_counter() - t0_all
        with open(os.path.join(save_dir, f"recon_timing_wce_{method}_{tag}.txt"), "w", encoding="utf-8") as f:
            f.write(f"[WCE {method} {tag}] pipeline_total_sec={t_all:.6f}\n")
            f.write(f"[WCE {method} {tag}] fdk_sec_gt={t_gt:.6f} fdk_sec_wce={t_wce:.6f} fdk_sec_noisy={t_nz:.6f}\n")
        print(f"[TIME][WCE {method} {tag}] pipeline_total={t_all:.3f}s (io+wce+fdk+hu+save)")

    # 整圈
    recon_and_save("all", pred, gt, noz, ang)

    with_wce = args.with_wce or bool((recon.get("wce", {}) or {}).get("enabled", False))
    if with_wce:
        recon_and_save_wce("all", noz, gt, ang)

    print(f"[OK] done -> {save_dir}")


if __name__ == "__main__":
    main()
