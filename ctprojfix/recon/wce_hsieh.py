# ctprojfix/recon/wce_hsieh.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from .fdk_astra import fdk_reconstruct


@dataclass
class HsiehParams:
    # 截断宽度（像素）
    trunc_left: int = 0
    trunc_right: int = 0

    # 输入域：原始投影是 "line"(已 -log) 还是 "intensity"
    input_domain: str = "line"

    # FDK 期望的域："line"（线积分，常见）或 "intensity"
    fdk_expect: str = "line"

    # 数值稳定
    min_valid: float = 1e-6
    clip_exp: float = 20.0  # 从 log 回强度域时的上限裁剪：exp(-clip)

    # 半径/边带（像素）
    radius_px_cap: int = 4096
    edge_band: int = 16

    # —— 像素语义与下采样换算 ——
    # config 里的 trunc/edge_band 是否已经是“下采样后像素”？
    pixels_are_post_downsample: bool = False
    # 若否，则按 downsample 把像素除以 ds
    downsample: int = 1


# ---------- 域变换 ----------

def _to_line_domain(sino: np.ndarray, domain: str, eps: float) -> np.ndarray:
    """确保返回线积分域（log 域）"""
    if str(domain).lower() == "line":
        return sino.astype(np.float32, copy=False)
    s = np.clip(sino.astype(np.float32), eps, None)
    return -np.log(s)


def _to_intensity_from_line(L: np.ndarray, clip_exp: float) -> np.ndarray:
    """从线积分域回到强度域，并做上限裁剪以避免溢出"""
    return np.exp(-np.clip(L, 0.0, float(clip_exp))).astype(np.float32)


# ---------- 行处理工具 ----------

def _row_boundary_mean(L_row: np.ndarray, u0: int, u1: int) -> float:
    u0 = max(0, min(u0, L_row.size - 1))
    u1 = max(0, min(u1, L_row.size - 1))
    if u0 > u1:
        u0, u1 = u1, u0
    seg = L_row[u0:u1 + 1]
    return float(seg.mean()) if seg.size > 0 else 0.0


def _extrapolate_known_trunc_row(
    L_row: np.ndarray,
    trunc_left: int,
    trunc_right: int,
    edge_band: int,
    radius_px: float,
) -> None:
    """
    在 log(线积分)域逐行外推（原地修改 L_row）。
    仅填充被截断的两侧区间：
      左：[0, trunc_left)；右：[U - trunc_right, U)
    """
    U = L_row.size
    if trunc_left <= 0 and trunc_right <= 0:
        return

    c = (U - 1) * 0.5
    R = float(max(radius_px, 1.0))  # 保底，避免 sqrt 负数

    # 左边界：有效区起点 uL = trunc_left
    if trunc_left > 0:
        uL = trunc_left
        L_edge = _row_boundary_mean(L_row, uL, min(U - 1, uL + edge_band))
        xL = abs(uL - c)
        denom = 2.0 * np.sqrt(max(R * R - xL * xL, 1e-6))
        muL = L_edge / denom
        for u in range(0, trunc_left):
            x = abs(u - c)
            if x >= R:
                L_row[u] = 0.0
            else:
                L_row[u] = float(muL * 2.0 * np.sqrt(max(R * R - x * x, 0.0)))

    # 右边界：有效区终点 uR = U - trunc_right - 1
    if trunc_right > 0:
        uR = U - trunc_right - 1
        L_edge = _row_boundary_mean(L_row, max(0, uR - edge_band), uR)
        xR = abs(uR - c)
        denom = 2.0 * np.sqrt(max(R * R - xR * xR, 1e-6))
        muR = L_edge / denom
        for u in range(U - trunc_right, U):
            x = abs(u - c)
            if x >= R:
                L_row[u] = 0.0
            else:
                L_row[u] = float(muR * 2.0 * np.sqrt(max(R * R - x * x, 0.0)))


# ---------- 像素参数换算 ----------

def _effective_pixels(hp: HsiehParams, U: int) -> Dict[str, int]:
    """
    将 cfg 中的像素参数换算为『当前 sinogram 分辨率』下的像素值。
    - 若 pixels_are_post_downsample=True：参数已是“下采样后像素”，不再缩放。
    - 否则按 downsample 做 /ds。
    """
    ds = max(1, int(hp.downsample))
    if hp.pixels_are_post_downsample:
        L_eff = int(hp.trunc_left)
        R_eff = int(hp.trunc_right)
        edge_eff = int(max(1, hp.edge_band))
        rad_cap_eff = int(max(1, hp.radius_px_cap))
    else:
        L_eff = int(round(hp.trunc_left / ds))
        R_eff = int(round(hp.trunc_right / ds))
        edge_eff = int(max(1, round(hp.edge_band / ds)))
        rad_cap_eff = int(max(1, round(hp.radius_px_cap / ds)))

    # 限幅，保证有效
    L_eff = max(0, min(L_eff, U // 2))
    R_eff = max(0, min(R_eff, U // 2))
    edge_eff = max(1, min(edge_eff, U // 4))
    rad_cap_eff = max(2, min(rad_cap_eff, 4 * U))  # 给个合理上限

    # 避免“中心有效宽度”为 0
    if (U - L_eff - R_eff) <= 0:
        # 保底让中心有 2 像素
        shrink = max(0, 2 - (U - L_eff - R_eff))
        # 左右对称缩一点
        L_eff = max(0, L_eff - (shrink // 2))
        R_eff = max(0, R_eff - (shrink - (shrink // 2)))

    return dict(L=L_eff, R=R_eff, edge=edge_eff, rad_cap=rad_cap_eff)


# ---------- 主流程 ----------

def wce_hsieh_extrapolate_known_trunc(
    sino_TVU: np.ndarray,
    params: Optional[dict] = None,
    input_file_hint: Optional[str] = None,
) -> np.ndarray:
    """
    Hsieh 风格确定性外推：
      - 在 log(线积分)域拟合两侧被截断区域（逐行）
      - 有效区原样保留
      - 返回值的“域”由 fdk_expect 决定（"line" 或 "intensity"）
    """
    p = params or {}
    hp = HsiehParams(
        trunc_left=int(p.get("trunc_left", 0)),
        trunc_right=int(p.get("trunc_right", 0)),
        input_domain=str(p.get("input_domain", "line")),
        fdk_expect=str(p.get("fdk_expect", "line")),
        min_valid=float(p.get("min_valid", 1e-6)),
        clip_exp=float(p.get("clip_exp", 20.0)),
        radius_px_cap=int(p.get("radius_px_cap", 4096)),
        edge_band=int(p.get("edge_band", 16)),
        pixels_are_post_downsample=bool(p.get("pixels_are_post_downsample", False)),
        downsample=int(p.get("downsample", 1)),
    )

    S = np.asarray(sino_TVU, dtype=np.float32)
    if S.ndim != 3:
        raise ValueError(f"[WCE-Hsieh] Expect (T,V,U), got {S.shape}")
    T, V, U = S.shape

    if input_file_hint:
        print(f"[WCE-Hsieh] Input sinogram: {input_file_hint}")
    print(f"[WCE-Hsieh] shape(T,V,U)=({T},{V},{U})  input_domain={hp.input_domain}  fdk_expect={hp.fdk_expect}")

    # 统一转到 log(线积分)域做外推
    L = _to_line_domain(S, hp.input_domain, hp.min_valid)  # (T,V,U)

    # 像素参数换算到『当前 U 分辨率』
    eff = _effective_pixels(hp, U)
    L_eff, R_eff = eff["L"], eff["R"]
    edge_eff, rad_cap_eff = eff["edge"], eff["rad_cap"]

    # 半径（像素）
    eff_half = max(1.0, (U - L_eff - R_eff) * 0.5)
    R0 = min(float(rad_cap_eff), eff_half + max(L_eff, R_eff))

    print(f"[WCE-Hsieh] trunc(L,R) raw=({hp.trunc_left},{hp.trunc_right}), "
          f"post-ds=({L_eff},{R_eff}), edge_band={edge_eff}, R0≈{R0:.2f}, "
          f"ds={hp.downsample}, pixels_are_post={hp.pixels_are_post_downsample}")

    # 外推（原地改拷贝）
    Lc = L.copy()
    if L_eff > 0 or R_eff > 0:
        for t in range(T):
            Lv = Lc[t]  # (V,U)
            for v in range(V):
                _extrapolate_known_trunc_row(
                    Lv[v],
                    trunc_left=L_eff,
                    trunc_right=R_eff,
                    edge_band=edge_eff,
                    radius_px=R0,
                )

    # 输出域按照 fdk_expect 返回
    fdk_expect = hp.fdk_expect.lower()
    if fdk_expect == "line":
        S_out = Lc
    elif fdk_expect == "intensity":
        S_out = _to_intensity_from_line(Lc, hp.clip_exp)
    else:
        raise ValueError(f"[WCE-Hsieh] Unknown fdk_expect={hp.fdk_expect} (use 'line' or 'intensity')")

    return S_out.astype(np.float32)


def wce_hsieh_fdk_reconstruct(
    noisy_sino_TVU: np.ndarray,
    angles_rad: np.ndarray,
    geom: Dict[str, Any],
    hsieh_params: Optional[dict] = None,
    input_file_hint: Optional[str] = None,
):
    """
    Hsieh 外推 + FDK
    返回: (volume, elapsed_sec, sino_corr)
    """
    sino_corr = wce_hsieh_extrapolate_known_trunc(
        noisy_sino_TVU, params=hsieh_params, input_file_hint=input_file_hint
    )
    print(f"[WCE-Hsieh] -> FDK  (domain fed to FDK: {(hsieh_params or {}).get('fdk_expect','line')})")
    vol, dt = fdk_reconstruct(sino_corr, angles_rad, geom)
    return vol, dt, sino_corr
