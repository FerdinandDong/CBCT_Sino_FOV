#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_compare_metrics.py
Quantitative evaluation for CBCT Inpainting/Reconstruction.
Calculates RMSE, PSNR, SSIM, LPIPS for:
  1. Global (Whole Volume)
  2. FOV-Inner (Limited Field of View)
  3. FOV-Outer (Extended Field of View / Inpainted Region)

Requirements:
  pip install lpips scikit-image pyyaml
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import warnings
from tqdm import tqdm

# Metrics
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
import lpips

# Suppress skimage warnings for small data ranges
warnings.filterwarnings("ignore")

# ==============================================================================
#  1. Helper Functions (IO & Math)
# ==============================================================================

def load_npy(path):
    if not os.path.exists(path):
        return None
    try:
        arr = np.load(path)
        arr = np.squeeze(arr) # (D, H, W)
        return arr.astype(np.float32)
    except Exception as e:
        print(f"[ERR] Failed to load {path}: {e}")
        return None

def _recursive_format(obj, vars_dict):
    """
    Template substitution for BOTH values AND keys.
    Supports {root} and {root_name}.
    """
    if isinstance(obj, str):
        s = obj
        for k, v in vars_dict.items():
            # Replace {key} with value
            s = s.replace("{" + k + "}", str(v))
        return s
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # 1. 替换 Key 里的变量 (让 yaml 里的 Key 可以动态变化)
            new_k = _recursive_format(k, vars_dict) if isinstance(k, str) else k
            # 2. 替换 Value 里的变量
            new_v = _recursive_format(v, vars_dict)
            new_dict[new_k] = new_v
        return new_dict
    elif isinstance(obj, list):
        return [_recursive_format(i, vars_dict) for i in obj]
    else:
        return obj

def create_masks(shape, mask_def):
    """
    Generate boolean masks for Inner (FOV) and Outer (Extended).
    Returns: (mask_inner, mask_outer) where True indicates the region of interest.
    """
    D, H, W = shape
    mask_inner = np.zeros((H, W), dtype=bool)
    
    mtype = mask_def.get("type", "none")
    
    if mtype == "col_truncate":
        # Projection domain: Left/Right truncation
        left = int(mask_def.get("truncate_left", 0))
        right = int(mask_def.get("truncate_right", 0))
        x_start = left
        x_end = W - right
        mask_inner[:, x_start:x_end] = True
        
    elif mtype == "circle":
        # Recon domain: Central circle is valid
        rad = float(mask_def.get("radius", min(H,W)//2))
        cy, cx = H//2, W//2
        y, x = np.ogrid[:H, :W]
        dist_sq = (x - cx)**2 + (y - cy)**2
        mask_inner = dist_sq <= rad**2
        
    else:
        # Default: everything is inner
        mask_inner[:] = True

    mask_outer = ~mask_inner
    return mask_inner, mask_outer

# ==============================================================================
#  2. Metric Calculators
# ==============================================================================

class MetricEvaluator:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[INIT] Loading LPIPS on {self.device}...")
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_fn.eval()

    def calc_rmse(self, pred, gt, mask=None):
        """Root Mean Squared Error."""
        diff = pred - gt
        if mask is not None:
            diff = diff[..., mask] # flatten selection
        mse = np.mean(diff ** 2)
        return np.sqrt(mse)

    def calc_psnr(self, pred, gt, data_range, mask=None):
        """PSNR with support for masking."""
        diff = pred - gt
        if mask is not None:
            diff = diff[..., mask]
        mse = np.mean(diff ** 2)
        if mse < 1e-12: return 99.99
        return 20 * np.log10(data_range / np.sqrt(mse))

    def calc_ssim_3d(self, pred_vol, gt_vol, data_range, mask=None):
        """
        Calculate SSIM slice-by-slice.
        If mask is provided (2D), it is applied to the SSIM map of each slice.
        """
        ssim_vals = []
        for i in range(pred_vol.shape[0]):
            p = pred_vol[i]
            g = gt_vol[i]
            
            # Use skimage structural_similarity with full=True to get the map
            score, ssim_map = ssim_func(g, p, data_range=data_range, full=True)
            
            if mask is not None:
                # Average SSIM only inside the mask
                score = np.mean(ssim_map[mask])
            
            ssim_vals.append(score)
        
        return np.mean(ssim_vals)

    @torch.no_grad()
    def calc_lpips_3d(self, pred_vol, gt_vol, norm_range=(-1, 1), mask=None, batch_size=32):
        """
        Calculate LPIPS slice-by-slice using batching.
        norm_range: tuple (min, max) of the input data to map to [-1, 1].
        """
        # 1. Pre-process: Normalize to [-1, 1]
        d_min, d_max = norm_range
        def _norm(v):
            v = np.clip(v, d_min, d_max)
            return 2.0 * (v - d_min) / (d_max - d_min) - 1.0

        scores = []
        N = pred_vol.shape[0]
        
        # 2. Loop in batches
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            
            p_batch = _norm(pred_vol[i:end]) # (B, H, W)
            g_batch = _norm(gt_vol[i:end])
            
            t_p = torch.from_numpy(p_batch).unsqueeze(1).to(self.device)
            t_g = torch.from_numpy(g_batch).unsqueeze(1).to(self.device)
            
            # Replicate channels 1->3
            t_p = t_p.repeat(1, 3, 1, 1)
            t_g = t_g.repeat(1, 3, 1, 1)
            
            # Apply Mask Strategy: Replace Outer with GT
            # So LPIPS diff is 0 outside mask
            if mask is not None:
                t_mask = torch.from_numpy(mask).to(self.device).float()
                t_mask = t_mask.unsqueeze(0).unsqueeze(0).repeat(end-i, 3, 1, 1)
                t_p = t_p * t_mask + t_g * (1 - t_mask)
            
            # Forward
            val = self.lpips_fn(t_p, t_g, normalize=False)
            scores.append(val.cpu().numpy().flatten())
            
        return np.mean(np.concatenate(scores))


# ==============================================================================
#  3. Main Pipeline
# ==============================================================================

def run_task(task, global_vars, evaluator: MetricEvaluator, f_log):
    # 递归替换配置中的变量（包括 Key 和 Value）
    task = _recursive_format(task, global_vars)
    
    name = task.get("name", "Task")
    domain = task.get("domain", "projection")
    data_range = float(task.get("data_range", 1.0))
    lpips_norm = task.get("lpips_norm", [0.0, 1.0])
    
    print(f"\n>> Running Task: {name} ({domain})")
    
    # 1. Load GT
    gt_path = task["paths"]["GT"]
    vol_gt = load_npy(gt_path)
    
    if vol_gt is None:
        msg = f"[SKIP] GT not found: {gt_path}"
        print(msg); f_log.write(msg + "\n")
        return

    # 2. Create Masks
    D, H, W = vol_gt.shape
    mask_inner, mask_outer = create_masks((D, H, W), task.get("mask_def", {}))
    
    px_total = H * W
    px_inner = np.count_nonzero(mask_inner)
    px_outer = np.count_nonzero(mask_outer)
    print(f"   Mask Info (2D): Total={px_total}, Inner={px_inner}, Outer={px_outer}")

    regions = [
        ("Global", None),
        ("Inner",  mask_inner),
        ("Outer",  mask_outer)
    ]

    # 3. Iterate Targets
    targets = task["paths"].get("targets", {})
    
    # Header for Log (Increased width for Method name)
    header = f"{'Method':<30} | {'Region':<8} | {'RMSE':<8} | {'PSNR':<8} | {'SSIM':<8} | {'LPIPS':<8}"
    print("-" * 90)
    print(header)
    print("-" * 90)
    f_log.write(f"\n--- Task: {name} ---\n")
    f_log.write(header + "\n")
    f_log.write("-" * 90 + "\n")

    for method_name, path in targets.items():
        vol_pred = load_npy(path)
        
        if vol_pred is None:
            # print(f"   [WARN] Missing: {method_name} at {path}")
            continue
            
        if vol_pred.shape != vol_gt.shape:
            print(f"   [ERR] Shape mismatch {method_name}: {vol_pred.shape} vs GT {vol_gt.shape}")
            continue

        for region_name, mask in regions:
            if mask is not None and np.count_nonzero(mask) == 0:
                continue

            # Calculate Metrics
            rmse = evaluator.calc_rmse(vol_pred, vol_gt, mask)
            psnr = evaluator.calc_psnr(vol_pred, vol_gt, data_range, mask)
            ssim = evaluator.calc_ssim_3d(vol_pred, vol_gt, data_range, mask)
            lpips_val = evaluator.calc_lpips_3d(vol_pred, vol_gt, norm_range=lpips_norm, mask=mask)

            # Log
            res_str = f"{method_name:<30} | {region_name:<8} | {rmse:<8.3f} | {psnr:<8.3f} | {ssim:<8.4f} | {lpips_val:<8.4f}"
            print(res_str)
            f_log.write(res_str + "\n")

    print("-" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/compare_metrics.yml")
    args = parser.parse_args()

    if not os.path.exists(args.cfg):
        print(f"[ERR] Config not found: {args.cfg}")
        sys.exit(1)
        
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    root = cfg.get("root", ".")
    
    # 自动从 root 路径提取文件夹名
    # e.g., "outputs/i2sb_local_1step" -> "i2sb_local_1step"
    root_name = os.path.basename(root.rstrip("/\\"))
    
    save_name = cfg.get("save_file", "metrics.txt")
    save_path = os.path.join(root, save_name)
    
    evaluator = MetricEvaluator(device=cfg.get("device", "cuda"))

    print(f"==================================================")
    print(f" Metrics Evaluation ")
    print(f" Root: {root}")
    print(f" Name: {root_name}")
    print(f" Saving to: {save_path}")
    print(f"==================================================")

    with open(save_path, "w", encoding="utf-8") as f_log:
        f_log.write(f"Evaluation Report\nRoot: {root}\nMethod: {root_name}\n")
        
        # 注入 root_name 供 yaml 里的 {root_name} 使用
        global_vars = {"root": root, "root_name": root_name}
        
        for task in cfg.get("tasks", []):
            run_task(task, global_vars, evaluator, f_log)

    print(f"\n[DONE] Metrics saved to {save_path}")

if __name__ == "__main__":
    main()