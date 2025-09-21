# -*- coding: utf-8 -*-

import os, glob, csv, argparse, yaml, math, time
import numpy as np
import tifffile as tiff
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import astra

# ------------------ IO & CFG ------------------

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


# 体堆栈读取整卷优先 

PRIORITY_SETS = [
    ("_raw.npy", np.load),
    ("_raw.tiff", tiff.imread),
    (".npy", np.load),
    (".tiff", tiff.imread),
]


def find_volume(from_dir, stem: str, explicit_path: str | None):
    """按优先级查找体文件（stem ∈ {pred_volume, gt_volume, noisy_volume}）。"""
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


# ------------------ 工具/排错 ------------------

def wrap_0_2pi(x):
    return np.mod(x, 2*np.pi).astype(np.float32)


def apply_uv_ops(stack_TVU, angles_rad, rot180=False, swap_uv=False, flip_u=False, flip_v=False):
    S = stack_TVU
    ang = angles_rad.copy()
    if rot180:
        S = S[:, ::-1, ::-1]
        ang = wrap_0_2pi(ang + np.pi)
    if swap_uv:
        S = np.transpose(S, (0, 2, 1))  # (T,V,U) <-> (T,U,V)
    if flip_v:
        S = S[:, ::-1, :]
    if flip_u:
        S = S[:, :, ::-1]
    return S, ang


def maybe_sort_by_angle(stack_TVU, angles_rad):
    idx = np.argsort(angles_rad)
    return stack_TVU[idx], angles_rad[idx]


def dedup_by_angle(a_idx, angles, tol_rad):
    """按 a 索引优先去重；若无 a 则按角度合并接近的重复（容差 tol_rad）。返回保留索引。"""
    keep = []
    if a_idx is not None and len(a_idx) == len(angles):
        seen = set()
        for i, a in enumerate(a_idx):
            if a not in seen:
                keep.append(i); seen.add(a)
        return np.array(keep, dtype=np.int32)
    # 退化：按弧度聚类（近似）
    last = None
    for i, ang in enumerate(angles):
        if last is None or abs(ang - last) > tol_rad:
            keep.append(i); last = ang
    return np.array(keep, dtype=np.int32)


# 简单体指标/可视化

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


def save_axial_triptych(vol_noisy, vol_pred, vol_gt, z_indices, out_dir, tag=""):
    os.makedirs(out_dir, exist_ok=True)
    for z in z_indices:
        n = vol_noisy[z]; p = vol_pred[z]; g = vol_gt[z]
        vmin, vmax = np.percentile(np.concatenate([n.ravel(), p.ravel(), g.ravel()]), [1, 99])
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(n, cmap="gray", vmin=vmin, vmax=vmax); axes[0].set_title(f"Noisy  z={z} {tag}"); axes[0].axis("off")
        axes[1].imshow(p, cmap="gray", vmin=vmin, vmax=vmax); axes[1].set_title(f"Pred   z={z} {tag}"); axes[1].axis("off")
        axes[2].imshow(g, cmap="gray", vmin=vmin, vmax=vmax); axes[2].set_title(f"GT     z={z} {tag}"); axes[2].axis("off")
        plt.tight_layout()
        fp = os.path.join(out_dir, f"axial_triptych{tag}_z{z:04d}.png")
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[FIG] {fp}")


# ------------------ ASTRA FDK ------------------

def run_fdk(sino_TVU, vol_shape_zyx, du, dv, det_u, det_v, angles_rad, SOD, ODD, gpu, filt="ram-lak"):
    nx, ny, nz = int(vol_shape_zyx[2]), int(vol_shape_zyx[1]), int(vol_shape_zyx[0])
    vg = astra.create_vol_geom(nx, ny, nz)

    angles = angles_rad.astype(np.float32)
    pg = astra.create_proj_geom('cone', float(dv), float(du), int(det_v), int(det_u),
                                angles, float(SOD), float(ODD))

    sino_VTU = np.transpose(sino_TVU, (1, 0, 2))
    if not (sino_VTU.shape == (det_v, len(angles), det_u)):
        raise ValueError(f"[ASTRA] sino shape {sino_VTU.shape} != (V={det_v}, T={len(angles)}, U={det_u})")

    sid = astra.data3d.create('-proj3d', pg, sino_VTU)
    rid = astra.data3d.create('-vol', vg)

    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ProjectionDataId'] = sid
    cfg['ReconstructionDataId'] = rid
    cfg['option'] = {'GPUindex': int(gpu), 'FilterType': str(filt)}

    t0 = time.time()
    alg = astra.algorithm.create(cfg)
    astra.algorithm.run(alg)
    dt = time.time() - t0

    vol = astra.data3d.get(rid).astype(np.float32)
    astra.algorithm.delete(alg)
    astra.data3d.delete([sid, rid])
    return vol, dt


# ------------------ 主流程 ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/eval.yaml")
    # 允许命令行覆盖三个体文件
    ap.add_argument("--pred", default=None)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--noisy", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    recon = cfg.get("recon3d", {})

    from_dir = recon.get("from_dir", os.path.join(cfg.get("eval", {}).get("out_dir", "outputs/eval")))
    save_dir = recon.get("save_dir", os.path.join(from_dir, "recon3d"))
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(from_dir, recon.get("metrics_csv", "metrics.csv"))

    # 新：优先读整卷文件
    pred_file = args.pred or recon.get("pred_file")
    gt_file   = args.gt   or recon.get("gt_file")
    noz_file  = args.noisy or recon.get("noisy_file")

    pred, pred_src = find_volume(from_dir, "pred_volume", pred_file)
    gt,   gt_src   = find_volume(from_dir, "gt_volume", gt_file)
    noz,  noz_src  = find_volume(from_dir, "noisy_volume", noz_file)

    use_raw = any(s and ("_raw" in os.path.basename(s)) for s in [pred_src, gt_src, noz_src])

    # 兜底：逐帧
    if pred is None or gt is None or noz is None:
        print("[WARN] 整卷体文件未完整找到，尝试逐帧堆栈兜底...")
        p2, _ = stack_raw_npy(from_dir, "pred_raw_*.npy")
        g2, _ = stack_raw_npy(from_dir, "gt_raw_*.npy")
        n2, _ = stack_raw_npy(from_dir, "noisy_raw_*.npy")
        if p2 is None or g2 is None or n2 is None:
            print("[WARN] RAW 逐帧不存在，改用 PNG 归一化域兜底（仅排错）")
            p2, _ = stack_png_01(from_dir, "pred_*.png")
            g2, _ = stack_png_01(from_dir, "gt_*.png")
            n2, _ = stack_png_01(from_dir, "noisy_*.png")
        pred, gt, noz = p2, g2, n2
        assert pred is not None and gt is not None and noz is not None, "未找到任何可用投影文件。"

    # 体数据统一为 (T,V,U)
    def to_TVU(v):
        v = np.asarray(v, dtype=np.float32)
        if v.ndim == 3:
            return v  # 形如 (T,V,U)
        raise ValueError(f"期望三维数组 (T,V,U)，收到形状: {v.shape}")

    pred = to_TVU(pred)
    gt   = to_TVU(gt)
    noz  = to_TVU(noz)

    # 对齐三路长度
    N = min(pred.shape[0], gt.shape[0], noz.shape[0])
    pred, gt, noz = pred[:N], gt[:N], noz[:N]

    # 角度
    A_default = int(recon.get("A_default", 360))
    angles, A_arr, a_idx = read_angles_from_csv(csv_path, A_default=A_default, return_index=True)
    if len(angles) != N:
        M = min(len(angles), N)
        print(f"[WARN] 角度数 {len(angles)} 与帧数 {N} 不一致，裁剪到 {M}.")
        angles = angles[:M]; A_arr = A_arr[:M]; a_idx = a_idx[:M]
        pred = pred[:M]; gt = gt[:M]; noz = noz[:M]; N = M

    # 方向与相位
    angle_direction = int(recon.get("angle_direction", 1))
    angle_offset_deg = float(recon.get("angle_offset_deg", 0.0))
    ang = wrap_0_2pi(angle_direction * (angles + np.deg2rad(angle_offset_deg)))

    # 去重（修复“重叠”）
    if bool(recon.get("dedup", True)):
        tol_deg = float(recon.get("dedup_tol_deg", 0.2))
        keep = dedup_by_angle(a_idx, ang, np.deg2rad(tol_deg))
        if keep.size < N:
            print(f"[FIX] 去重：从 {N} -> {keep.size}（移除重叠视角）")
            pred = pred[keep]; gt = gt[keep]; noz = noz[keep]
            ang = ang[keep]; A_arr = A_arr[keep]; a_idx = a_idx[keep]
            N = keep.size

    # 几何与排错开关
    du   = float(recon.get("du", 0.4))
    dv   = float(recon.get("dv", 0.4))
    SOD  = float(recon.get("SOD", 750.0))
    ODD  = float(recon.get("ODD", 450.0))
    vol_shape = [int(x) for x in recon.get("vol_shape", [512,512,512])]
    filt = str(recon.get("filter", "ram-lak"))
    gpu  = int(recon.get("gpu", 0))

    do_half      = bool(recon.get("split_half", False))
    sort_by_angle= bool(recon.get("sort_by_angle", False))
    rot180  = bool(recon.get("rot180", False))
    swap_uv = bool(recon.get("swap_uv", False))
    flip_u  = bool(recon.get("flip_u", False))
    flip_v  = bool(recon.get("flip_v", False))

    # UV 操作
    pred, ang = apply_uv_ops(pred, ang, rot180=rot180, swap_uv=swap_uv, flip_u=flip_u, flip_v=flip_v)
    gt,   _   = apply_uv_ops(gt,   ang, rot180=rot180, swap_uv=swap_uv, flip_u=flip_u, flip_v=flip_v)
    noz,  _   = apply_uv_ops(noz,  ang, rot180=rot180, swap_uv=swap_uv, flip_u=flip_u, flip_v=flip_v)

    det_v, det_u = pred.shape[1], pred.shape[2]

    print(f"[CFG] use_raw={use_raw}  views={N}  det(V,U)=({det_v},{det_u})  filter={filt} gpu={gpu}")
    print(f"[CFG] du={du} dv={dv} SOD={SOD} ODD={ODD} vol_shape[Z,Y,X]={vol_shape}")

    def recon_and_save(tag, P, G, N_, A_):
        if P is None or G is None or N_ is None or A_ is None or len(A_) == 0:
            print(f"[SKIP] {tag}: empty set.")
            return
        if sort_by_angle:
            order = np.argsort(A_)
            P, G, N_, A_ = P[order], G[order], N_[order], A_[order]

        print(f"[INFO] FDK_CUDA [{tag}] ... (views={len(A_)})")
        vol_gt, t_gt = run_fdk(G, vol_shape, du, dv, det_u, det_v, A_, SOD, ODD, gpu, filt)
        vol_pr, t_pr = run_fdk(P, vol_shape, du, dv, det_u, det_v, A_, SOD, ODD, gpu, filt)
        vol_nz, t_nz = run_fdk(N_, vol_shape, du, dv, det_u, det_v, A_, SOD, ODD, gpu, filt)
        print(f"[TIME][{tag}] GT={t_gt:.3f}s, PRED={t_pr:.3f}s, NOISY={t_nz:.3f}s")

        # 归一化后存 tiff，便于快速看
        gtn = norm01(vol_gt); prn = norm01(vol_pr); nzn = norm01(vol_nz)
        tiff.imwrite(os.path.join(save_dir, f"recon_gt_{tag}.tif"),   gtn.astype(np.float32))
        tiff.imwrite(os.path.join(save_dir, f"recon_pred_{tag}.tif"), prn.astype(np.float32))
        tiff.imwrite(os.path.join(save_dir, f"recon_noisy_{tag}.tif"),nzn.astype(np.float32))

        z_spec = recon.get("slices", ["mid"])  # ["mid"] or [10,20]
        z_list = parse_slices(z_spec, gtn.shape[0])
        save_axial_triptych(nzn, prn, gtn, z_list, save_dir, tag=f"_{tag}")

        psnr_pred = vol_psnr(prn, gtn, 1.0)
        ssim_pred = float(np.mean([ssim2d(prn[i], gtn[i], dr=1.0) for i in range(prn.shape[0])]))
        psnr_nozy = vol_psnr(nzn, gtn, 1.0)
        ssim_nozy = float(np.mean([ssim2d(nzn[i], gtn[i], dr=1.0) for i in range(nzn.shape[0])]))
        with open(os.path.join(save_dir, f"recon_metrics_{tag}.txt"), "w") as f:
            f.write(f"[{tag}] Pred vs GT:  PSNR={psnr_pred:.3f} dB  SSIM_Axial={ssim_pred:.4f}\n")
            f.write(f"[{tag}] Noisy vs GT: PSNR={psnr_nozy:.3f} dB  SSIM_Axial={ssim_nozy:.4f}\n")
        print(f"[METRIC][{tag}] Pred vs GT:  PSNR={psnr_pred:.3f} dB  SSIM_Axial={ssim_pred:.4f}")
        print(f"[METRIC][{tag}] Noisy vs GT: PSNR={psnr_nozy:.3f} dB  SSIM_Axial={ssim_nozy:.4f}")

    # 半圈或整圈
    if bool(recon.get("split_half", False)):
        A0 = int(A_arr[0]) if len(A_arr) > 0 else 360
        half = A0 // 2
        idx_half1 = np.where(a_idx <  half)[0]
        idx_half2 = np.where(a_idx >= half)[0]
        recon_and_save("half1", pred[idx_half1], gt[idx_half1], noz[idx_half1], ang[idx_half1])
        recon_and_save("half2", pred[idx_half2], gt[idx_half2], noz[idx_half2], ang[idx_half2])
    else:
        recon_and_save("all", pred, gt, noz, ang)

    print(f"[OK] done -> {save_dir}")


if __name__ == "__main__":
    main()
