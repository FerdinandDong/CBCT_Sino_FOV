#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recon/Projection montage maker (HU-consistent for recon; robust for projections).
Base behavior preserved:
- Original montage saved to out_path (NO ROI boxes).
- Optional tiles saved to tiles.dir (NO ROI boxes).

New ROI pipeline (does NOT change original outputs):
- Extra ROI montage saved: out_path with suffix (default "_roi") (WITH ROI boxes).
- ROI tiles (cropped ROI only) saved to roi.tiles.dir (pure image, no title).
- ROI boxed tiles (full panel WITH ROI boxes) saved to roi.tiles.boxed_dir.
  Includes both gray panels and heat panels.

Also supports two tasks in one YAML:
- tasks: [ {name: recon, ...}, {name: proj, ...} ]
If tasks not provided, falls back to legacy single-task keys.
"""

import os
import sys
import argparse
import numpy as np

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ---------------------- Basic IO ----------------------
def load_npy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    arr = np.load(path)
    arr = np.squeeze(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expect 2D or 3D array, got {arr.shape} from {path}")
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
            raise ValueError(f"Unsupported slice spec: {index}")
    else:
        idx = int(index)
    if not (0 <= idx < D):
        raise IndexError(f"slice_index {idx} out of range [0,{D-1}] for axis={axis}")
    return np.take(vol, idx, axis=axis)


# ---------------------- Display scaling helpers ----------------------
def _get_vmin_vmax_from_slice_list(slices, display_cfg: dict):
    """
    Generic vmin/vmax estimator.
    - mode: window (wl/ww) OR percentile ([p1,p2]) OR fixed01 OR fixed(vmin/vmax)
    """
    mode = str((display_cfg or {}).get("mode", "percentile")).lower()
    flat = np.concatenate([s.ravel() for s in slices], axis=0).astype(np.float32)

    if mode == "window":
        wl = float((display_cfg or {}).get("wl", 40.0))
        ww = float((display_cfg or {}).get("ww", 400.0))
        vmin, vmax = wl - ww / 2.0, wl + ww / 2.0
        return float(vmin), float(vmax)

    if mode == "fixed01":
        return 0.0, 1.0

    if mode == "fixed":
        vmin = float((display_cfg or {}).get("vmin", float(np.min(flat))))
        vmax = float((display_cfg or {}).get("vmax", float(np.max(flat))))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        return vmin, vmax

    # default: percentile
    p = (display_cfg or {}).get("percentile", [1, 99])
    p1, p2 = float(p[0]), float(p[1])
    vmin = np.percentile(flat, p1)
    vmax = np.percentile(flat, p2)
    if vmax <= vmin:
        vmin, vmax = float(flat.min()), float(flat.max() + 1e-6)
    return float(vmin), float(vmax)


# ---------------------- Save tiles (pure image, no title) ----------------------
def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

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
    """Full panel tile WITH ROI rectangles. Pure image, no title."""
    _ensure_dir(os.path.dirname(path))
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    draw_rois(ax, rois)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_tile_heat_with_rois(img: np.ndarray, path: str, cmap: str, vmax: float, rois: list, with_colorbar: bool):
    """Full heat tile WITH ROI rectangles. Pure image, no title."""
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


# ---------------------- ROI parse/draw/crop ----------------------
def _clamp_roi(x0, y0, w, h, W, H):
    x0 = float(x0); y0 = float(y0); w = float(w); h = float(h)
    if w <= 1 or h <= 1:
        return None
    x0 = max(0.0, min(x0, W - 1.0))
    y0 = max(0.0, min(y0, H - 1.0))
    x1 = max(0.0, min(x0 + w, W * 1.0))
    y1 = max(0.0, min(y0 + h, H * 1.0))
    w2 = x1 - x0
    h2 = y1 - y0
    if w2 <= 1 or h2 <= 1:
        return None
    return x0, y0, w2, h2

def parse_rois(roi_cfg: dict, shape_hw):
    """
    roi_cfg:
      enable: true
      linewidth: 2
      list:
        - name: ROI-1
          color: red
          center_xy_frac: [0.282, 0.50]   # (x/W, y/H)
          size_wh_px: [128, 128]          # (w, h)
    """
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
            cx = (W - 1) / 2.0
            cy = (H - 1) / 2.0

        if "size_wh_px" in it:
            w, h = it["size_wh_px"]
            w = float(w); h = float(h)
        elif "size_wh_frac" in it:
            fw, fh = it["size_wh_frac"]
            w = float(fw) * W
            h = float(fh) * H
        else:
            w, h = 128.0, 128.0

        x0 = cx - w / 2.0
        y0 = cy - h / 2.0
        packed = _clamp_roi(x0, y0, w, h, W, H)
        if packed is None:
            continue
        x0, y0, w, h = packed
        rois.append({"name": name, "color": color, "lw": lw, "x0": x0, "y0": y0, "w": w, "h": h})
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
    return img[y0:y1, x0:x1]


# ---------------------- One task runner ----------------------
def run_one_task(task_cfg: dict):
    """
    task_cfg minimal fields (compatible with your old cfg):
      out_path, dpi, figsize, title_fontsize, suptitle
      slice_axis, slice_index
      paths: {GT: ..., Noisy: ..., ...}
      include: [...]
      display: {...}           (window/percentile/fixed01/fixed)
      heatmap: {...}
      tiles: {...}
      roi: {...}               (new pipeline, does not alter old outputs)
    """
    out_path   = task_cfg.get("out_path", "outputs/fig_compare.png")
    dpi        = int(task_cfg.get("dpi", 220))
    figsize    = task_cfg.get("figsize", [14, 7])
    title_fs   = int(task_cfg.get("title_fontsize", 12))
    suptitle   = task_cfg.get("suptitle", None)
    slice_axis = int(task_cfg.get("slice_axis", 0))
    slice_index= task_cfg.get("slice_index", "middle")

    tiles_cfg = task_cfg.get("tiles", {}) or {}
    save_tiles = bool(tiles_cfg.get("enable", False))
    tiles_dir  = tiles_cfg.get("dir", None)
    tiles_prefix = str(tiles_cfg.get("prefix", ""))
    tiles_format = str(tiles_cfg.get("format", "png"))
    tiles_heat_with_cb = bool(tiles_cfg.get("heat_with_colorbar", False))

    paths: dict = task_cfg["paths"]
    if "GT" not in paths:
        raise KeyError("cfg.paths 必须包含 'GT'。")

    include_names = task_cfg.get("include", None)
    if include_names is None:
        include = [k for k in paths.keys() if k != "GT"]
    else:
        include = []
        for k in include_names:
            if k not in paths:
                raise KeyError(f"cfg.include 包含未知键：{k}")
            if k != "GT":
                include.append(k)
    col_order = ["GT"] + include

    # load + slice
    slices = {}
    for name in col_order:
        vol = load_npy(paths[name])
        slices[name] = pick_slice(vol, slice_axis, slice_index)

    # display scaling
    display_cfg = task_cfg.get("display", {}) or {}
    vmin, vmax = _get_vmin_vmax_from_slice_list([slices[n] for n in col_order], display_cfg)

    # heat
    heat_cfg = task_cfg.get("heatmap", {}) or {}
    show_heat = bool(heat_cfg.get("enable", False))
    heat_cmap = str(heat_cfg.get("cmap", "magma"))
    show_cb   = bool(heat_cfg.get("colorbar", True))
    diff_vmax = heat_cfg.get("vmax", None)

    diffs = {}
    if show_heat:
        gt = slices["GT"].astype(np.float32)
        for name in include:
            diffs[name] = np.abs(slices[name].astype(np.float32) - gt)
        if diff_vmax is None and len(diffs) > 0:
            all_diff = np.concatenate([d.ravel() for d in diffs.values()], axis=0)
            diff_vmax = float(np.percentile(all_diff, 99.0))
        if diff_vmax is None:
            diff_vmax = 1.0

    # ROI parse
    roi_cfg = task_cfg.get("roi", {}) or {}
    rois = parse_rois(roi_cfg, slices[col_order[0]].shape)

    # ----------------- Original montage (NO ROI) -----------------
    ncols = len(col_order)
    nrows = 2 if show_heat else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, constrained_layout=True)

    # normalize axes shape
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for j, name in enumerate(col_order):
        ax = axes[0, j]
        ax.imshow(slices[name], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}", fontsize=title_fs)
        ax.axis("off")

    if show_heat:
        for j, name in enumerate(col_order):
            ax = axes[1, j]
            if name == "GT":
                ax.axis("off")
                continue
            im = ax.imshow(diffs[name], cmap=heat_cmap, vmin=0.0, vmax=diff_vmax)
            ax.set_title(f"|{name} - GT|", fontsize=title_fs)
            ax.axis("off")
            if show_cb:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle, fontsize=title_fs + 2)

    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {out_path}")

    # ----------------- Original tiles (NO ROI) -----------------
    if save_tiles:
        if not tiles_dir:
            base_dir = os.path.dirname(out_path)
            base_name = os.path.splitext(os.path.basename(out_path))[0]
            tiles_dir = os.path.join(base_dir, base_name + "_tiles")
        _ensure_dir(tiles_dir)

        slice_tag = slice_index.lower() if isinstance(slice_index, str) else f"{int(slice_index)}"

        for name in col_order:
            fname = f"{tiles_prefix}{name}_z{slice_tag}.{tiles_format}"
            fpath = os.path.join(tiles_dir, fname)
            save_tile_gray(slices[name], fpath, vmin=vmin, vmax=vmax)
            print(f"[TILE] {fpath}")

        if show_heat:
            for name in include:
                fname = f"{tiles_prefix}{name}_vs_GT_absdiff_z{slice_tag}.{tiles_format}"
                fpath = os.path.join(tiles_dir, fname)
                save_tile_heat(diffs[name], fpath, cmap=heat_cmap, vmax=diff_vmax, with_colorbar=tiles_heat_with_cb)
                print(f"[TILE] {fpath}")

    # ----------------- ROI pipeline (NEW, does not affect originals) -----------------
    if bool(roi_cfg.get("enable", False)) and len(rois) > 0:
        out_suffix = str(roi_cfg.get("out_suffix", "_roi"))
        base, ext = os.path.splitext(out_path)
        out_path_roi = base + out_suffix + ext

        fig2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, constrained_layout=True)
        if nrows == 1 and ncols == 1:
            axes2 = np.array([[axes2]])
        elif nrows == 1:
            axes2 = np.array([axes2])
        elif ncols == 1:
            axes2 = np.array([[axes2[0]], [axes2[1]]])

        # gray + ROI
        for j, name in enumerate(col_order):
            ax = axes2[0, j]
            ax.imshow(slices[name], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"{name}", fontsize=title_fs)
            draw_rois(ax, rois)
            ax.axis("off")

        # heat + ROI
        if show_heat:
            for j, name in enumerate(col_order):
                ax = axes2[1, j]
                if name == "GT":
                    ax.axis("off")
                    continue
                im = ax.imshow(diffs[name], cmap=heat_cmap, vmin=0.0, vmax=diff_vmax)
                ax.set_title(f"|{name} - GT|", fontsize=title_fs)
                draw_rois(ax, rois)
                ax.axis("off")
                if show_cb:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if suptitle:
            fig2.suptitle(str(suptitle) + " (ROI)", fontsize=title_fs + 2)

        _ensure_dir(os.path.dirname(out_path_roi))
        fig2.savefig(out_path_roi, bbox_inches="tight")
        plt.close(fig2)
        print(f"[ROI] Saved -> {out_path_roi}")

        # ROI tiles config
        roi_tiles_cfg = (roi_cfg.get("tiles", {}) or {})
        roi_tiles_enable = bool(roi_tiles_cfg.get("enable", True))

        # NEW: boxed tiles (full panel WITH ROI boxes)
        boxed_enable = bool((roi_tiles_cfg.get("boxed_enable", True)))
        boxed_dir = roi_tiles_cfg.get("boxed_dir", None)

        if roi_tiles_enable:
            # default roi tiles dir
            roi_tiles_dir = roi_tiles_cfg.get("dir", None)
            if not roi_tiles_dir:
                if save_tiles and tiles_dir:
                    roi_tiles_dir = tiles_dir + "_roi"
                else:
                    base_dir = os.path.dirname(out_path)
                    base_name = os.path.splitext(os.path.basename(out_path))[0]
                    roi_tiles_dir = os.path.join(base_dir, base_name + "_roi_tiles")
            _ensure_dir(roi_tiles_dir)

            # default boxed dir
            if boxed_enable and (not boxed_dir):
                boxed_dir = roi_tiles_dir + "_boxed"
            if boxed_enable:
                _ensure_dir(boxed_dir)

            slice_tag = slice_index.lower() if isinstance(slice_index, str) else f"{int(slice_index)}"
            roi_fmt = str(roi_tiles_cfg.get("format", tiles_format))
            roi_prefix = str(roi_tiles_cfg.get("prefix", ""))
            roi_heat_with_cb = bool(roi_tiles_cfg.get("heat_with_colorbar", False))

            # (A) ROI cropped tiles (per ROI subfolder)
            for r in rois:
                sub = os.path.join(roi_tiles_dir, r["name"])
                _ensure_dir(sub)
                for name in col_order:
                    crop = crop_roi(slices[name], r)
                    fname = f"{roi_prefix}{name}_roi_{r['name']}_z{slice_tag}.{roi_fmt}"
                    fpath = os.path.join(sub, fname)
                    save_tile_gray(crop, fpath, vmin=vmin, vmax=vmax)
                    print(f"[ROI-TILE] {fpath}")

                if show_heat:
                    for name in include:
                        cropd = crop_roi(diffs[name], r)
                        fname = f"{roi_prefix}{name}_vs_GT_absdiff_roi_{r['name']}_z{slice_tag}.{roi_fmt}"
                        fpath = os.path.join(sub, fname)
                        save_tile_heat(cropd, fpath, cmap=heat_cmap, vmax=diff_vmax, with_colorbar=roi_heat_with_cb)
                        print(f"[ROI-TILE] {fpath}")

            # (B) ROI boxed tiles (full panels WITH ROI boxes) 补的
            if boxed_enable:
                # gray row boxed tiles
                gray_sub = os.path.join(boxed_dir, "gray")
                heat_sub = os.path.join(boxed_dir, "heat") if show_heat else None
                _ensure_dir(gray_sub)
                if heat_sub:
                    _ensure_dir(heat_sub)

                for name in col_order:
                    fname = f"{roi_prefix}{name}_boxedROI_z{slice_tag}.{roi_fmt}"
                    fpath = os.path.join(gray_sub, fname)
                    save_tile_gray_with_rois(slices[name], fpath, vmin=vmin, vmax=vmax, rois=rois)
                    print(f"[ROI-BOXED] {fpath}")

                if show_heat:
                    for name in include:
                        fname = f"{roi_prefix}{name}_vs_GT_absdiff_boxedROI_z{slice_tag}.{roi_fmt}"
                        fpath = os.path.join(heat_sub, fname)
                        save_tile_heat_with_rois(diffs[name], fpath, cmap=heat_cmap, vmax=diff_vmax, rois=rois,
                                                 with_colorbar=roi_heat_with_cb)
                        print(f"[ROI-BOXED] {fpath}")


# ---------------------- main ----------------------
def main(cfg_path: str):
    if yaml is None:
        raise RuntimeError("PyYAML 未安装。请先 `pip install pyyaml`")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # New mode: tasks list
    tasks = cfg.get("tasks", None)
    if tasks is not None:
        if not isinstance(tasks, list) or len(tasks) == 0:
            raise ValueError("cfg.tasks 必须是非空 list")
        for t in tasks:
            name = str(t.get("name", "task"))
            print(f"\n========== [TASK] {name} ==========")
            run_one_task(t)
        return

    # Legacy single-task: use cfg as task
    run_one_task(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make comparison montages for recon(HU)/proj with optional ROI pipeline.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()
    try:
        main(args.cfg)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
