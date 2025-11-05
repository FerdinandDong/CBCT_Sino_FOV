#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成重建域对比图（单行多列），并可选第二行加入“与GT的|差值|热力图”。
新增：可将每个小图单独另存为文件（含灰度面板与热力图面板）。
HU 可视化窗口与 eval_recon3d_from_outputs.py 保持一致：
- display.mode: "window" 使用 wl/ww；否则使用 percentile: [p1,p2]
- vmin/vmax 在当前切片上“所有方法的像素集合”上统一估计，保证列间一致
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


# ---------------------- 基础 IO ----------------------
def load_npy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    arr = np.load(path)
    arr = np.squeeze(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expect 2D or 3D array, got {arr.shape} from {path}")
    return arr.astype(np.float32, copy=False)

def pick_slice(vol: np.ndarray, axis: int, index) -> np.ndarray:
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


# ---------------------- 与 eval_ 脚本一致的 HU 显示策略 ----------------------
def _get_vmin_vmax_for_hu_from_slice_list(slices: list[np.ndarray], display_cfg: dict):
    """
    从一组同切片的图（GT/Noisy/Pred/...）联合估计统一 vmin/vmax。
    逻辑与 eval_recon3d_from_outputs._get_vmin_vmax_for_hu 保持一致。
    """
    mode = str((display_cfg or {}).get("mode", "percentile")).lower()
    flat = np.concatenate([s.ravel() for s in slices], axis=0).astype(np.float32)

    if mode == "window":
        wl = float((display_cfg or {}).get("wl", 40.0))
        ww = float((display_cfg or {}).get("ww", 400.0))
        vmin, vmax = wl - ww / 2.0, wl + ww / 2.0
        return float(vmin), float(vmax)

    # percentile 模式
    p = (display_cfg or {}).get("percentile", [1, 99])
    p1, p2 = float(p[0]), float(p[1])
    vmin = np.percentile(flat, p1)
    vmax = np.percentile(flat, p2)
    if vmax <= vmin:
        vmin, vmax = float(flat.min()), float(flat.max() + 1e-6)
    return float(vmin), float(vmax)


# ---------------------- 辅助：保存单个小图 ----------------------
def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def save_tile_gray(img: np.ndarray, path: str, vmin: float, vmax: float):
    _ensure_dir(os.path.dirname(path))
    # 使用无轴的小画布保存（保持与大图一致的灰度窗口）
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])  # 填满
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_tile_heat(img: np.ndarray, path: str, cmap: str, vmax: float, with_colorbar: bool):
    _ensure_dir(os.path.dirname(path))
    if with_colorbar:
        fig = plt.figure(figsize=(4.4, 4), dpi=200)
        # 预留右侧窄色条
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


# ---------------------- 主流程 ----------------------
def main(cfg_path: str):
    if yaml is None:
        raise RuntimeError("PyYAML 未安装。请先 `pip install pyyaml`")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 基本参数
    out_path   = cfg.get("out_path", "outputs/fig_compare.png")
    dpi        = int(cfg.get("dpi", 220))
    figsize    = cfg.get("figsize", [14, 7])
    title_fs   = cfg.get("title_fontsize", 12)
    suptitle   = cfg.get("suptitle", None)
    slice_axis = int(cfg.get("slice_axis", 0))
    slice_index= cfg.get("slice_index", "middle")

    # 单图保存参数
    tiles_cfg = cfg.get("tiles", {}) or {}
    save_tiles = bool(tiles_cfg.get("enable", False))
    tiles_dir  = tiles_cfg.get("dir", None)  # 如未设，则默认与 out_path 同级目录 + "_tiles"
    tiles_prefix = tiles_cfg.get("prefix", "")  # 统一前缀，可为空
    tiles_format = tiles_cfg.get("format", "png")
    tiles_heat_with_cb = bool(tiles_cfg.get("heat_with_colorbar", False))

    paths: dict = cfg["paths"]
    if "GT" not in paths:
        raise KeyError("cfg.paths 必须包含 'GT'。")

    # 展示顺序：GT + include
    include_names = cfg.get("include", None)
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

    # 读取体并抽取切片
    slices = {}
    for name in col_order:
        vol = load_npy(paths[name])
        slices[name] = pick_slice(vol, slice_axis, slice_index)

    # 与 eval_ 一致的 vmin/vmax（统一到所有方法的像素集合）
    display_cfg = cfg.get("display", {}) or {}
    vmin, vmax = _get_vmin_vmax_for_hu_from_slice_list([slices[n] for n in col_order], display_cfg)

    # 差值热力图参数
    heat_cfg = cfg.get("heatmap", {}) or {}
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
            # 统一用 99 分位防极值；可在 cfg 中手动指定 vmax 覆盖
            diff_vmax = float(np.percentile(all_diff, 99.0))
        if diff_vmax is None:
            diff_vmax = 1.0  # 兜底

    # 画大图（拼版）
    ncols = len(col_order)
    nrows = 2 if show_heat else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, constrained_layout=True)

    # 兼容 axes 形状
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    # 第一行：灰度显示（统一 vmin/vmax）
    for j, name in enumerate(col_order):
        ax = axes[0, j]
        ax.imshow(slices[name], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}", fontsize=title_fs)
        ax.axis("off")

    # 第二行：|差值|热力图（仅非 GT）
    if show_heat:
        for j, name in enumerate(col_order):
            ax = axes[1, j]
            if name == "GT":
                ax.axis("off")
                continue
            im = ax.imshow(diffs[name], cmap=heat_cmap, vmin=0.0, vmax=diff_vmax)
            ax.set_title(f"|{name} - GT| (HU)", fontsize=title_fs)
            ax.axis("off")
            if show_cb:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle, fontsize=title_fs + 2)

    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {out_path}")

    # ====== 单图另存（可选）======
    if save_tiles:
        # 默认 tiles_dir：使用拼版图同目录，文件名前缀去扩展、加 _tiles
        if not tiles_dir:
            base_dir = os.path.dirname(out_path)
            base_name = os.path.splitext(os.path.basename(out_path))[0]
            tiles_dir = os.path.join(base_dir, base_name + "_tiles")
        _ensure_dir(tiles_dir)

        # 生成文件名的切片索引文本
        if isinstance(slice_index, str):
            slice_tag = slice_index.lower()
        else:
            slice_tag = f"{int(slice_index)}"

        # 灰度面板单独导出
        for name in col_order:
            fname = f"{tiles_prefix}{name}_z{slice_tag}.{tiles_format}"
            fpath = os.path.join(tiles_dir, fname)
            save_tile_gray(slices[name], fpath, vmin=vmin, vmax=vmax)
            print(f"[TILE] {fpath}")

        # 热力图面板单独导出（仅非 GT 且启用热力图）
        if show_heat:
            for name in include:
                fname = f"{tiles_prefix}{name}_vs_GT_absdiff_z{slice_tag}.{tiles_format}"
                fpath = os.path.join(tiles_dir, fname)
                save_tile_heat(diffs[name], fpath, cmap=heat_cmap, vmax=diff_vmax, with_colorbar=tiles_heat_with_cb)
                print(f"[TILE] {fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make HU-consistent comparison figures from recon .npy results (with optional per-tile saving).")
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()
    try:
        main(args.cfg)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
