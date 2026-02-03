#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recon/Projection montage maker with Batch Slicing & Template Support.

Key Features:
1. Batch Slicing: Supports 'slice_list': [128, 256] -> loops automatically.
2. Variable Substitution: Supports '{root}/path/...' -> easier config management.
3. Flexible Heatmap: Supports 'fixed' mode with custom vmax per task.
4. ROI Pipeline: Draws boxes, saves cropped tiles, and boxed full panels.
"""

import os
import sys
import argparse
import copy
import numpy as np

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ==============================================================================
#  Section 1: Basic IO & Helpers
# ==============================================================================

def load_npy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    arr = np.load(path)
    arr = np.squeeze(arr)
    # Check dims
    if arr.ndim not in (2, 3):
        # Fallback: maybe (1, H, W) or (C, H, W) -> take [0]
        if arr.ndim == 4: arr = arr[0]
    return arr.astype(np.float32, copy=False)

def pick_slice(vol: np.ndarray, axis: int, index) -> np.ndarray:
    """Pick a 2D slice from 2D/3D volume-like array."""
    if vol.ndim == 2:
        return vol
    D = vol.shape[axis]
    if isinstance(index, str):
        if index.lower() in ("middle", "mid"):
            idx = D // 2
        else:
            try:
                idx = int(index)
            except ValueError:
                raise ValueError(f"Unsupported slice spec: {index}")
    else:
        idx = int(index)
    
    if not (0 <= idx < D):
        # Clip to boundary to prevent crash
        idx = max(0, min(idx, D-1))
        print(f"[WARN] Slice index clipped to {idx}")
    
    return np.take(vol, idx, axis=axis)

def _get_vmin_vmax_from_slice_list(slices, display_cfg: dict):
    """
    Compute display window (vmin, vmax).
    """
    mode = str((display_cfg or {}).get("mode", "percentile")).lower()
    
    # 1. Window Mode (Medical standard)
    if mode == "window":
        wl = float((display_cfg or {}).get("wl", 40.0))
        ww = float((display_cfg or {}).get("ww", 400.0))
        vmin, vmax = wl - ww / 2.0, wl + ww / 2.0
        return float(vmin), float(vmax)

    # 2. Fixed [0, 1]
    if mode == "fixed01":
        return 0.0, 1.0

    # 3. Fixed manual range
    if mode == "fixed":
        flat = slices[0] # reference
        vmin = float((display_cfg or {}).get("vmin", float(np.min(flat))))
        vmax = float((display_cfg or {}).get("vmax", float(np.max(flat))))
        if vmax <= vmin: vmax = vmin + 1e-6
        return vmin, vmax

    # 4. Percentile (Default)
    flat = np.concatenate([s.ravel() for s in slices], axis=0).astype(np.float32)
    p = (display_cfg or {}).get("percentile", [1, 99])
    p1, p2 = float(p[0]), float(p[1])
    vmin = np.percentile(flat, p1)
    vmax = np.percentile(flat, p2)
    if vmax <= vmin:
        vmin, vmax = float(flat.min()), float(flat.max() + 1e-6)
    return float(vmin), float(vmax)

def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


# ==============================================================================
#  Section 2: Plotting & Saving (Tiles / ROIs)
# ==============================================================================

def save_tile_gray(img: np.ndarray, path: str, vmin: float, vmax: float):
    _ensure_dir(os.path.dirname(path))
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_tile_heat(img: np.ndarray, path: str, cmap: str, vmax: float, with_colorbar: bool):
    _ensure_dir(os.path.dirname(path))
    if with_colorbar:
        fig = plt.figure(figsize=(4.4, 4), dpi=200)
        ax = fig.add_axes([0.0, 0.0, 0.9, 1.0])
        im = ax.imshow(img, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.axis("off")
        cax = fig.add_axes([0.91, 0.1, 0.03, 0.8])
        plt.colorbar(im, cax=cax)
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(4, 4), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

def save_tile_gray_with_rois(img: np.ndarray, path: str, vmin: float, vmax: float, rois: list):
    _ensure_dir(os.path.dirname(path))
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    draw_rois(ax, rois)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_tile_heat_with_rois(img: np.ndarray, path: str, cmap: str, vmax: float, rois: list, with_colorbar: bool):
    _ensure_dir(os.path.dirname(path))
    if with_colorbar:
        fig = plt.figure(figsize=(4.4, 4), dpi=200)
        ax = fig.add_axes([0.0, 0.0, 0.9, 1.0])
        im = ax.imshow(img, cmap=cmap, vmin=0.0, vmax=vmax)
        draw_rois(ax, rois)
        ax.axis("off")
        cax = fig.add_axes([0.91, 0.1, 0.03, 0.8])
        plt.colorbar(im, cax=cax)
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(4, 4), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img, cmap=cmap, vmin=0.0, vmax=vmax)
        draw_rois(ax, rois)
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


# ==============================================================================
#  Section 3: ROI Logic
# ==============================================================================

def _clip_int(v, lo, hi):
    return int(max(lo, min(hi, v)))

def _clamp_roi(x0, y0, w, h, W, H):
    x0 = float(x0); y0 = float(y0); w = float(w); h = float(h)
    if w <= 1 or h <= 1: return None
    
    # Boundary check
    x0 = max(0.0, min(x0, W - 1.0))
    y0 = max(0.0, min(y0, H - 1.0))
    x1 = max(0.0, min(x0 + w, W * 1.0))
    y1 = max(0.0, min(y0 + h, H * 1.0))
    
    w2 = x1 - x0
    h2 = y1 - y0
    if w2 <= 1 or h2 <= 1: return None
    return x0, y0, w2, h2

def parse_rois(roi_cfg: dict, shape_hw):
    H, W = int(shape_hw[0]), int(shape_hw[1])
    rois = []
    if not roi_cfg or not bool(roi_cfg.get("enable", False)):
        return rois

    items = roi_cfg.get("list", []) or []
    default_lw = float(roi_cfg.get("linewidth", 2.0))
    
    for it in items:
        name = str(it.get("name", "ROI"))
        color = str(it.get("color", "lime"))
        lw = float(it.get("linewidth", default_lw))

        if "center_xy_frac" in it:
            fx, fy = it["center_xy_frac"]
            cx = float(fx) * (W - 1)
            cy = float(fy) * (H - 1)
        elif "center_xy_px" in it:
            cx, cy = it["center_xy_px"]
            cx = float(cx); cy = float(cy)
        else:
            cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

        if "size_wh_px" in it:
            w, h = it["size_wh_px"]
            w, h = float(w), float(h)
        elif "size_wh_frac" in it:
            fw, fh = it["size_wh_frac"]
            w = float(fw) * W
            h = float(fh) * H
        else:
            w, h = 128.0, 128.0

        x0 = cx - w / 2.0
        y0 = cy - h / 2.0
        
        packed = _clamp_roi(x0, y0, w, h, W, H)
        if packed is None: continue
        
        rx, ry, rw, rh = packed
        rois.append({"name": name, "color": color, "lw": lw, 
                     "x0": rx, "y0": ry, "w": rw, "h": rh})
    return rois

def draw_rois(ax, rois):
    for r in rois:
        rect = Rectangle((r["x0"], r["y0"]), r["w"], r["h"],
                         fill=False, linewidth=r["lw"], edgecolor=r["color"])
        ax.add_patch(rect)

def crop_roi(img: np.ndarray, roi: dict) -> np.ndarray:
    x0 = int(round(roi["x0"]))
    y0 = int(round(roi["y0"]))
    x1 = int(round(roi["x0"] + roi["w"]))
    y1 = int(round(roi["y0"] + roi["h"]))
    # Double check bounds
    H, W = img.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    return img[y0:y1, x0:x1]


# ==============================================================================
#  Section 4: Templating & Processing Logic (FIXED)
# ==============================================================================

def _recursive_format(obj, vars_dict):
    """
    Recursively format strings using .replace() to handle partial substitution.
    This fixes the KeyError when '{slice}' is present but not in vars_dict.
    """
    if isinstance(obj, str):
        # 使用 replace 而不是 format，这样即使字符串里有 {slice} 
        # 而 vars_dict 里没有 slice，也不会报错，只会替换掉 {root}
        s = obj
        for k, v in vars_dict.items():
            s = s.replace("{" + k + "}", str(v))
        return s
        
    elif isinstance(obj, dict):
        return {k: _recursive_format(v, vars_dict) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_recursive_format(i, vars_dict) for i in obj]
    else:
        return obj

def _process_single_view(task_cfg: dict):
    """
    Execute ONE montage generation (single slice/angle).
    """
    # 1. Config Parsing
    out_path = task_cfg.get("out_path", "outputs/fig.png")
    paths = task_cfg.get("paths", {})
    slice_idx_raw = task_cfg.get("slice_index", "mid")
    
    print(f"   -> View: {slice_idx_raw} | Saving: {out_path}")

    # 2. Load Data
    if "GT" not in paths:
        raise KeyError("Config 'paths' must contain 'GT'.")
    
    include_names = task_cfg.get("include", [k for k in paths.keys() if k != "GT"])
    col_order = ["GT"] + [k for k in include_names if k in paths and k != "GT"]

    slice_axis = int(task_cfg.get("slice_axis", 0))

    slices = {}
    for name in col_order:
        p = paths[name]
        try:
            vol = load_npy(p)
            slices[name] = pick_slice(vol, slice_axis, slice_idx_raw)
        except Exception as e:
            print(f"[ERROR] Failed to load {name} from {p}: {e}")
            return # Skip this view

    # 3. Compute Display Range (Gray)
    display_cfg = task_cfg.get("display", {}) or {}
    vmin, vmax = _get_vmin_vmax_from_slice_list([slices[n] for n in col_order], display_cfg)

    # 4. Compute Difference & Heatmap Range
    heat_cfg = task_cfg.get("heatmap", {}) or {}
    show_heat = bool(heat_cfg.get("enable", False))
    heat_cmap = str(heat_cfg.get("cmap", "magma"))
    show_cb   = bool(heat_cfg.get("colorbar", True))
    
    # [FEATURE] Fixed Heatmap Range
    heat_mode = str(heat_cfg.get("heat_mode", "adaptive")).lower()
    if heat_mode == "fixed":
        diff_vmax = float(heat_cfg.get("vmax", 100.0))
    else:
        diff_vmax = None # Calculated later

    diffs = {}
    if show_heat:
        gt = slices["GT"].astype(np.float32)
        for name in col_order:
            if name == "GT": continue
            diffs[name] = np.abs(slices[name] - gt)
        
        # Calculate adaptive vmax if needed
        if diff_vmax is None and len(diffs) > 0:
            all_diff = np.concatenate([d.ravel() for d in diffs.values()], axis=0)
            diff_vmax = float(np.percentile(all_diff, 99.0))
        
        if diff_vmax is None: diff_vmax = 1.0

    # 5. Prepare ROIs
    roi_cfg = task_cfg.get("roi", {}) or {}
    rois = parse_rois(roi_cfg, slices[col_order[0]].shape)

    # 6. Plotting Setup
    dpi = int(task_cfg.get("dpi", 220))
    figsize = task_cfg.get("figsize", [14, 7])
    title_fs = int(task_cfg.get("title_fontsize", 12))
    suptitle = task_cfg.get("suptitle", None)

    # -------------------------------------------------------------
    # A. Draw Main Montage (Clean, No Boxes)
    # -------------------------------------------------------------
    ncols = len(col_order)
    nrows = 2 if show_heat else 1
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, constrained_layout=True)
    
    # Standardize axes array
    if nrows == 1 and ncols == 1: axes = np.array([[axes]])
    elif nrows == 1: axes = np.array([axes])
    elif ncols == 1: axes = np.array([[axes[0]], [axes[1]]])

    # Row 1: Gray Images
    for j, name in enumerate(col_order):
        ax = axes[0, j]
        ax.imshow(slices[name], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}", fontsize=title_fs)
        ax.axis("off")

    # Row 2: Heatmaps
    if show_heat:
        for j, name in enumerate(col_order):
            ax = axes[1, j]
            if name == "GT":
                ax.axis("off"); continue
            
            im = ax.imshow(diffs[name], cmap=heat_cmap, vmin=0.0, vmax=diff_vmax)
            ax.set_title(f"|{name}-GT|", fontsize=title_fs)
            ax.axis("off")
            if show_cb:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Global Title (Supports {slice} formatting if passed in string)
    if suptitle:
        # Try formatting slice into title if placeholder exists
        try:
            st = str(suptitle).replace("{slice}", str(slice_idx_raw))
        except:
            st = str(suptitle)
        fig.suptitle(st, fontsize=title_fs + 2)

    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------
    # B. Save Clean Tiles (No Boxes)
    # -------------------------------------------------------------
    tiles_cfg = task_cfg.get("tiles", {}) or {}
    if bool(tiles_cfg.get("enable", False)):
        tiles_dir = tiles_cfg.get("dir", None)
        if not tiles_dir:
            base_dir = os.path.dirname(out_path)
            base_name = os.path.splitext(os.path.basename(out_path))[0]
            tiles_dir = os.path.join(base_dir, base_name + "_tiles")
        _ensure_dir(tiles_dir)

        prefix = str(tiles_cfg.get("prefix", ""))
        fmt = str(tiles_cfg.get("format", "png"))
        heat_cb_tile = bool(tiles_cfg.get("heat_with_colorbar", False))
        tag = str(slice_idx_raw).lower()
        
        for name in col_order:
            fname = f"{prefix}{name}_z{tag}.{fmt}"
            save_tile_gray(slices[name], os.path.join(tiles_dir, fname), vmin, vmax)

        if show_heat:
            for name in diffs:
                fname = f"{prefix}{name}_vs_GT_diff_z{tag}.{fmt}"
                save_tile_heat(diffs[name], os.path.join(tiles_dir, fname), 
                               heat_cmap, diff_vmax, heat_cb_tile)

    # -------------------------------------------------------------
    # C. ROI Pipeline (Montage with Boxes + Cropped Tiles)
    # -------------------------------------------------------------
    if bool(roi_cfg.get("enable", False)) and len(rois) > 0:
        out_suffix = str(roi_cfg.get("out_suffix", "_roi"))
        base, ext = os.path.splitext(out_path)
        out_path_roi = base + out_suffix + ext

        # C.1 Draw Montage with Boxes
        fig2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, constrained_layout=True)
        if nrows == 1 and ncols == 1: axes2 = np.array([[axes2]])
        elif nrows == 1: axes2 = np.array([axes2])
        elif ncols == 1: axes2 = np.array([[axes2[0]], [axes2[1]]])

        # Gray + ROI Boxes
        for j, name in enumerate(col_order):
            ax = axes2[0, j]
            ax.imshow(slices[name], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"{name}", fontsize=title_fs)
            draw_rois(ax, rois)
            ax.axis("off")

        # Heat + ROI Boxes
        if show_heat:
            for j, name in enumerate(col_order):
                ax = axes2[1, j]
                if name == "GT":
                    ax.axis("off"); continue
                
                im = ax.imshow(diffs[name], cmap=heat_cmap, vmin=0.0, vmax=diff_vmax)
                ax.set_title(f"|{name}-GT|", fontsize=title_fs)
                draw_rois(ax, rois)
                ax.axis("off")
                if show_cb:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if suptitle:
            try: st = str(suptitle).replace("{slice}", str(slice_idx_raw))
            except: st = str(suptitle)
            fig2.suptitle(st + " (ROI)", fontsize=title_fs + 2)

        _ensure_dir(os.path.dirname(out_path_roi))
        fig2.savefig(out_path_roi, bbox_inches="tight")
        plt.close(fig2)

        # C.2 Save ROI Cropped Tiles & Boxed Full Tiles
        roi_tiles_cfg = roi_cfg.get("tiles", {}) or {}
        if bool(roi_tiles_cfg.get("enable", True)):
            
            # Paths
            roi_tiles_dir = roi_tiles_cfg.get("dir", None)
            if not roi_tiles_dir:
                roi_tiles_dir = os.path.join(os.path.dirname(out_path), 
                                             os.path.splitext(os.path.basename(out_path))[0] + "_roi_tiles")
            
            boxed_dir = roi_tiles_cfg.get("boxed_dir", None)
            boxed_enable = bool(roi_tiles_cfg.get("boxed_enable", True))
            if boxed_enable and not boxed_dir:
                boxed_dir = roi_tiles_dir + "_boxed"

            _ensure_dir(roi_tiles_dir)
            if boxed_enable: _ensure_dir(boxed_dir)

            roi_fmt = str(roi_tiles_cfg.get("format", "png"))
            roi_prefix = str(roi_tiles_cfg.get("prefix", ""))
            roi_heat_cb = bool(roi_tiles_cfg.get("heat_with_colorbar", False))
            
            tag = str(slice_idx_raw).lower()

            # --- Save Cropped ROIs ---
            for r in rois:
                sub = os.path.join(roi_tiles_dir, r["name"])
                _ensure_dir(sub)
                
                for name in col_order:
                    crop = crop_roi(slices[name], r)
                    fname = f"{roi_prefix}{name}_{r['name']}_z{tag}.{roi_fmt}"
                    save_tile_gray(crop, os.path.join(sub, fname), vmin, vmax)
                
                if show_heat:
                    for name in diffs:
                        cropd = crop_roi(diffs[name], r)
                        fname = f"{roi_prefix}{name}_diff_{r['name']}_z{tag}.{roi_fmt}"
                        save_tile_heat(cropd, os.path.join(sub, fname), heat_cmap, diff_vmax, roi_heat_cb)

            # --- Save Full Boxed Tiles (Optional) ---
            if boxed_enable:
                gray_box = os.path.join(boxed_dir, "gray")
                _ensure_dir(gray_box)
                
                for name in col_order:
                    fname = f"{roi_prefix}{name}_boxed_z{tag}.{roi_fmt}"
                    save_tile_gray_with_rois(slices[name], os.path.join(gray_box, fname), vmin, vmax, rois)
                
                if show_heat:
                    heat_box = os.path.join(boxed_dir, "heat")
                    _ensure_dir(heat_box)
                    for name in diffs:
                        fname = f"{roi_prefix}{name}_diff_boxed_z{tag}.{roi_fmt}"
                        save_tile_heat_with_rois(diffs[name], os.path.join(heat_box, fname), 
                                                 heat_cmap, diff_vmax, rois, roi_heat_cb)


# ==============================================================================
#  Section 5: Batch Runner
# ==============================================================================

def run_batch_tasks(task_cfg: dict, global_vars: dict):
    """
    Handle one task block.
    1. Apply global vars ({root}).
    2. Check 'slice_list' for batch looping.
    3. Call _process_single_view.
    """
    # 1. Variable Substitution
    task_cfg = _recursive_format(task_cfg, global_vars)
    name = task_cfg.get("name", "unnamed_task")
    print(f"\n[TASK] {name}")

    # 2. Check Batch Mode
    slice_list = task_cfg.get("slice_list", None)
    
    if slice_list is not None and isinstance(slice_list, list):
        print(f"   -> Batch mode detected: {slice_list}")
        raw_out_path = task_cfg.get("out_path", "")
        
        for s_idx in slice_list:
            sub_cfg = copy.deepcopy(task_cfg)
            sub_cfg["slice_index"] = s_idx
            
            # Manually replace {slice} using .replace()
            # This avoids format() errors if {root} was already replaced
            if "{slice}" in raw_out_path:
                sub_cfg["out_path"] = raw_out_path.replace("{slice}", str(s_idx))
            else:
                base, ext = os.path.splitext(raw_out_path)
                sub_cfg["out_path"] = f"{base}_z{s_idx}{ext}"
            
            _process_single_view(sub_cfg)
    else:
        _process_single_view(task_cfg)


def main():
    parser = argparse.ArgumentParser(description="Batch Compare Figs")
    parser.add_argument("--cfg", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()

    if yaml is None:
        raise RuntimeError("PyYAML not installed. `pip install pyyaml`")

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    global_vars = {}
    if "root" in cfg:
        global_vars["root"] = cfg["root"]
    
    tasks = cfg.get("tasks", None)
    if tasks is not None:
        if not isinstance(tasks, list):
            raise ValueError("cfg.tasks must be a list")
        for t in tasks:
            run_batch_tasks(t, global_vars)
    else:
        run_batch_tasks(cfg, global_vars)

    print("\n[DONE] All tasks finished.")


if __name__ == "__main__":
    main()