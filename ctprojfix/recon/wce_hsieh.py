# ctprojfix/recon/wce_hsieh.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .fdk_astra import fdk_reconstruct


@dataclass
class HsiehParams:
    # 已知截断宽度（像素）
    trunc_left: int = 0
    trunc_right: int = 0
    # 输入域: "line" 表示线积分 (已 -log)；"intensity" 表示强度域
    input_domain: str = "line"
    # 数值稳定
    min_valid: float = 1e-6
    clip_exp: float = 20.0
    # 半径选择：默认取“有效区半宽”，也可限制最大半径
    radius_px_cap: int = 4096
    # 边界拟合时采用的行内均值窗口
    edge_band: int = 16


def _to_line_domain(sino: np.ndarray, domain: str, eps: float) -> np.ndarray:
    if domain.lower() == "line":
        return sino.astype(np.float32)
    # intensity -> line integral
    s = np.clip(sino.astype(np.float32), eps, None)
    return -np.log(s)


def _to_intensity_domain(L: np.ndarray, clip_exp: float) -> np.ndarray:
    return np.exp(-np.clip(L, 0.0, clip_exp)).astype(np.float32)


def _row_boundary_mean(L_row: np.ndarray, u0: int, u1: int) -> float:
    u0 = max(0, min(u0, L_row.size - 1))
    u1 = max(0, min(u1, L_row.size - 1))
    if u0 > u1: u0, u1 = u1, u0
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
    已知左右截断宽度的单行外推（原地修改 L_row）：
      - 仅填充 [0:trunc_left) 与 [U-trunc_right:U)
      - 半径 R 以中心 c 计算弦长：Lcyl(u) = μ * 2 * sqrt(R^2 - (u-c)^2)
      - μ 通过“边界连续”求解：令 Lcyl 在边界像素等于该边界附近原始平均值
    """
    U = L_row.size
    if trunc_left <= 0 and trunc_right <= 0:
        return

    c = (U - 1) * 0.5
    R = min(radius_px, (U - trunc_left - trunc_right) * 0.5 + max(trunc_left, trunc_right))

    # 左边界：有效区起点 uL = trunc_left
    if trunc_left > 0:
        uL = trunc_left
        L_edge = _row_boundary_mean(L_row, uL, min(U - 1, uL + edge_band))
        xL = abs(uL - c)
        denom = 2.0 * np.sqrt(max(R * R - xL * xL, 1e-6))
        muL = L_edge / denom
        # 只在截断区填充
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


def wce_hsieh_extrapolate_known_trunc(
    sino_TVU: np.ndarray,
    params: Optional[dict] = None,
    input_file_hint: Optional[str] = None,
) -> np.ndarray:
    """
    Hsieh 风格，基于“已知左右截断宽度”的确定性外推。
    - 仅填充被截断的列；有效区原样保留
    - 在 log(线积分)域做边界连续拟合，逐行解出 μ
    """
    if input_file_hint:
        print(f"[WCE-Hsieh] Input sinogram file: {input_file_hint}")

    p = params or {}
    hp = HsiehParams(
        trunc_left=int(p.get("trunc_left", 0)),
        trunc_right=int(p.get("trunc_right", 0)),
        input_domain=str(p.get("input_domain", "line")),
        min_valid=float(p.get("min_valid", 1e-6)),
        clip_exp=float(p.get("clip_exp", 20.0)),
        radius_px_cap=int(p.get("radius_px_cap", 4096)),
        edge_band=int(p.get("edge_band", 16)),
    )

    S = np.asarray(sino_TVU, dtype=np.float32)
    L = _to_line_domain(S, hp.input_domain, hp.min_valid)  # (T,V,U)
    T, V, U = L.shape

    # 半径：以像素为单位（适度宽一些，避免 sqrt 为 0）
    eff_half = (U - hp.trunc_left - hp.trunc_right) * 0.5
    R0 = min(float(hp.radius_px_cap), eff_half + max(hp.trunc_left, hp.trunc_right))

    Lc = L.copy()
    for t in range(T):
        Lv = Lc[t]  # (V,U)
        for v in range(V):
            _extrapolate_known_trunc_row(
                Lv[v],
                hp.trunc_left, hp.trunc_right,
                edge_band=hp.edge_band,
                radius_px=R0,
            )

    # 回强度域（如果后续 FDK 期望强度域，则返回强度；若 FDK 期望线积分域，请直接返回 Lc）
    # 流程里 FDK 直接吃当前数据（不做 log），所以按“强度域”走：
    # 但既然我们刚在 log 域外推，这里需要回到“强度域”
    S_corr = _to_intensity_domain(Lc, hp.clip_exp) if hp.input_domain != "line" else Lc
    return S_corr.astype(np.float32)


def wce_hsieh_fdk_reconstruct(
    noisy_sino_TVU: np.ndarray,
    angles_rad: np.ndarray,
    geom: dict,
    hsieh_params: Optional[dict] = None,
    input_file_hint: Optional[str] = None,
):
    """
    Hsieh 版（已知截断宽度）外推 + FDK
    返回: (volume, elapsed_sec, sino_corr)
    """
    sino_corr = wce_hsieh_extrapolate_known_trunc(
        noisy_sino_TVU, params=hsieh_params, input_file_hint=input_file_hint
    )
    vol, dt = fdk_reconstruct(sino_corr, angles_rad, geom)
    return vol, dt, sino_corr
