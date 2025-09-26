# -*- coding: utf-8 -*-

import os, glob, csv, argparse, yaml, math, time, json
import numpy as np
import tifffile as tiff
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 
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

# ========== 指标/可视化 ==========
def ssim2d(x, y, dr=1.0, K1=0.01, K2=0.03, sigma=1.5):
    from scipy.ndimage import gaussian_filter
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

    from_dir = recon.get("from_dir", os.path.join(cfg.get("eval", {}).get("out_dir", "outputs/eval")))
    save_dir = recon.get("save_dir", os.path.join(from_dir, "recon3d"))
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(from_dir, recon.get("metrics_csv", "metrics.csv"))

    # —— 打印读取路径（整卷优先）——
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

    # 兜底：逐帧（并打印）
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
        # 打印首尾两条，避免刷屏
        if pfiles: print(f"[PATH] pred frames: {pfiles[0]} ... {pfiles[-1]}  (total {len(pfiles)})")
        if gfiles: print(f"[PATH] gt   frames: {gfiles[0]} ... {gfiles[-1]}  (total {len(gfiles)})")
        if nfiles: print(f"[PATH] noisy frames: {nfiles[0]} ... {nfiles[-1]}  (total {len(nfiles)})")
        assert pred is not None and gt is not None and noz is not None, "未找到任何可用投影文件。"

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

    du   = float(recon.get("du", 0.4))
    dv   = float(recon.get("dv", 0.4))
    SOD  = float(recon.get("SOD", 750.0))
    ODD  = float(recon.get("ODD", 450.0))
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

    # FDK 基线
    def recon_and_save(tag, P, G, N_, A_):
        if P is None or G is None or N_ is None or A_ is None or len(A_) == 0:
            print(f"[SKIP] {tag}: empty set."); return
        if sort_by_angle:
            order = np.argsort(A_); P, G, N_, A_ = P[order], G[order], N_[order], A_[order]
        print(f"[INFO] FDK_CUDA [{tag}] ... (views={len(A_)})")
        vol_gt, t_gt = run_fdk(G, vol_shape, du, dv, det_u, det_v, A_, SOD, ODD, gpu, filt)
        vol_pr, t_pr = run_fdk(P, vol_shape, du, dv, det_u, det_v, A_, SOD, ODD, gpu, filt)
        vol_nz, t_nz = run_fdk(N_, vol_shape, du, dv, det_u, det_v, A_, SOD, ODD, gpu, filt)
        print(f"[TIME][{tag}] GT={t_gt:.3f}s, PRED={t_pr:.3f}s, NOISY={t_nz:.3f}s")
        gtn = norm01(vol_gt); prn = norm01(vol_pr); nzn = norm01(vol_nz)
        tiff.imwrite(os.path.join(save_dir, f"recon_gt_{tag}.tif"),   gtn.astype(np.float32))
        tiff.imwrite(os.path.join(save_dir, f"recon_pred_{tag}.tif"), prn.astype(np.float32))
        tiff.imwrite(os.path.join(save_dir, f"recon_noisy_{tag}.tif"),nzn.astype(np.float32))
        z_list = parse_slices(recon.get("slices", ["mid"]), gtn.shape[0])
        save_axial_triptych(nzn, prn, gtn, z_list, save_dir, tag=f"_{tag}")
        psnr_pred = vol_psnr(prn, gtn, 1.0)
        ssim_pred = float(np.mean([ssim2d(prn[i], gtn[i], dr=1.0) for i in range(prn.shape[0])]))
        psnr_nozy = vol_psnr(nzn, gtn, 1.0)
        ssim_nozy = float(np.mean([ssim2d(nzn[i], gtn[i], dr=1.0) for i in range(nzn.shape[0])]))
        with open(os.path.join(save_dir, f"recon_metrics_{tag}.txt"), "w") as f:
            f.write(f"[{tag}] Pred vs GT:  PSNR={psnr_pred:.3f} dB  SSIM_Axial={ssim_pred:.4f}\n")
            f.write(f"[{tag}] Noisy vs GT: PSNR={psnr_nozy:.3f} dB  SSIM_Axial={ssim_nozy:.4f}\n")

    # —— WCE 统一调度（支持 gaussian/hsieh 两种）——
    def recon_and_save_wce(tag, noisy_TVU, gt_TVU, A_):
        if noisy_TVU is None or gt_TVU is None or A_ is None or len(A_) == 0:
            print(f"[SKIP] {tag}: empty set."); return
        if sort_by_angle:
            order = np.argsort(A_); noisy_TVU, gt_TVU, A_ = noisy_TVU[order], gt_TVU[order], A_[order]

        wcfg = recon.get("wce", {}) or {}
        method = str(wcfg.get("method", "hsieh")).lower()   # 默认用 Hsieh（已知截断）
        geom = { "du": du, "dv": dv, "SOD": SOD, "ODD": ODD, "vol_shape": vol_shape, "gpu": gpu, "filter": filt }

        if method == "hsieh":
            # 把截断宽度与输入域传下去（来自 config）
            hsieh_params = dict(
                trunc_left = int(wcfg.get("trunc_left", recon.get("trunc_left", 0))),
                trunc_right= int(wcfg.get("trunc_right", recon.get("trunc_right", 0))),
                input_domain= str(wcfg.get("input_domain", "line")),
                min_valid = float(wcfg.get("min_valid", 1e-6)),
                clip_exp  = float(wcfg.get("clip_exp", 20.0)),
                radius_px_cap = int(wcfg.get("radius_px_cap", 4096)),
                edge_band = int(wcfg.get("edge_band", 16)),
            )
            print(f"[INFO] WCE method = Hsieh-known-trunc  params={json.dumps(hsieh_params)}")
            input_hint = noz_src or from_dir
            vol_wce, t_wce, sino_corr = wce_hsieh_fdk_reconstruct(noisy_TVU, A_, geom,
                                                                hsieh_params=hsieh_params,
                                                                input_file_hint=input_hint)
        else:
            print(f"[INFO] WCE method = Gaussian  params={json.dumps(wcfg)}")
            vol_wce, t_wce, sino_corr = wce_fdk_reconstruct(noisy_TVU, A_, geom, wce_params=wcfg)

        vol_gt, t_gt = run_fdk(gt_TVU,    vol_shape, du, dv, det_u, det_v, A_, SOD, ODD, gpu, filt)
        vol_nz, t_nz = run_fdk(noisy_TVU, vol_shape, du, dv, det_u, det_v, A_, SOD, ODD, gpu, filt)
        print(f"[TIME][WCE {tag}] GT={t_gt:.3f}s, WCE={t_wce:.3f}s, NOISY={t_nz:.3f}s")

        if bool(wcfg.get("save_corr", False)):
            np.save(os.path.join(save_dir, f"wce_corr_sino_{method}_{tag}.npy"), sino_corr.astype(np.float32))

        gtn = norm01(vol_gt); wcn = norm01(vol_wce); nzn = norm01(vol_nz)
        tiff.imwrite(os.path.join(save_dir, f"recon_gt_wce_{method}_{tag}.tif"),   gtn.astype(np.float32))
        tiff.imwrite(os.path.join(save_dir, f"recon_wce_{method}_{tag}.tif"),      wcn.astype(np.float32))
        tiff.imwrite(os.path.join(save_dir, f"recon_noisy_wce_{method}_{tag}.tif"),nzn.astype(np.float32))

        z_list = parse_slices(recon.get("slices", ["mid"]), gtn.shape[0])
        save_axial_triptych(nzn, wcn, gtn, z_list, save_dir, tag=f"_wce_{method}_{tag}")

        psnr_wce = vol_psnr(wcn, gtn, 1.0)
        ssim_wce = float(np.mean([ssim2d(wcn[i], gtn[i], dr=1.0) for i in range(wcn.shape[0])]))
        psnr_noz = vol_psnr(nzn, gtn, 1.0)
        ssim_noz = float(np.mean([ssim2d(nzn[i], gtn[i], dr=1.0) for i in range(nzn.shape[0])]))
        with open(os.path.join(save_dir, f"recon_metrics_wce_{method}_{tag}.txt"), "w") as f:
            f.write(f"[WCE {method} {tag}] WCE vs GT:  PSNR={psnr_wce:.3f} dB  SSIM_Axial={ssim_wce:.4f}\n")
            f.write(f"[WCE {method} {tag}] Noisy vs GT: PSNR={psnr_noz:.3f} dB  SSIM_Axial={ssim_noz:.4f}\n")


    # —— 半圈或整圈 —— 
    do_half = bool(recon.get("split_half", False))
    # 读角度所需变量在上面
    if do_half:
        A0 = int(A_arr[0]) if len(A_arr) > 0 else 360
        half = A0 // 2
        idx_half1 = np.where(a_idx <  half)[0]
        idx_half2 = np.where(a_idx >= half)[0]
        recon_and_save("half1", pred[idx_half1], gt[idx_half1], noz[idx_half1], ang[idx_half1])
        recon_and_save("half2", pred[idx_half2], gt[idx_half2], noz[idx_half2], ang[idx_half2])
        with_wce = args.with_wce or bool((recon.get("wce", {}) or {}).get("enabled", False))
        if with_wce:
            recon_and_save_wce("half1", noz[idx_half1], gt[idx_half1], ang[idx_half1])
            recon_and_save_wce("half2", noz[idx_half2], gt[idx_half2], ang[idx_half2])
    else:
        recon_and_save("all", pred, gt, noz, ang)
        with_wce = args.with_wce or bool((recon.get("wce", {}) or {}).get("enabled", False))
        if with_wce:
            recon_and_save_wce("all", noz, gt, ang)

    print(f"[OK] done -> {save_dir}")


if __name__ == "__main__":
    main()
