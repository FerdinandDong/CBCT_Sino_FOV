# ctprojfix/recon/wce_baseline.py
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from .fdk_astra import fdk_reconstruct


def wce_estimate_bg(sino_TVU: np.ndarray, sigma_v: float = 12.0, sigma_u: float = 12.0) -> np.ndarray:
    """
    估计低频背景（近似水等效项）：仅在探测器 (V,U) 方向高斯平滑，不跨视角 T。
    """
    S = np.asarray(sino_TVU, dtype=np.float32)
    bg = gaussian_filter(S, sigma=(0.0, float(sigma_v), float(sigma_u)), mode="nearest")
    return bg


def wce_apply(
    sino_TVU: np.ndarray,
    domain: str = "linear",
    alpha: float = 1.0,
    beta: float = 1.0,
    sigma_v: float = 12.0,
    sigma_u: float = 12.0,
    clip: float = 99.9,
) -> np.ndarray:
    """
    WCE 前处理：
      - 背景 bg
      - linear：S' = (S - alpha*bg) * beta
      - log：   L=-log(S)；L'=(L - alpha*bg)*beta；S'=exp(-clip(L'))
    """
    S = np.asarray(sino_TVU, dtype=np.float32)
    bg = wce_estimate_bg(S, sigma_v=sigma_v, sigma_u=sigma_u)

    if str(domain).lower() == "log":
        S_safe = np.clip(S, 1e-6, None)
        L = -np.log(S_safe)
        Lc = (L - float(alpha) * bg) * float(beta)
        S_corr = np.exp(-np.clip(Lc, 0.0, 20.0))
    else:
        S_corr = (S - float(alpha) * bg) * float(beta)

    hi = np.percentile(S_corr, float(clip))
    S_corr = np.clip(S_corr, 0.0, hi).astype(np.float32)
    return S_corr


def wce_fdk_reconstruct(
    noisy_sino_TVU: np.ndarray,
    angles_rad: np.ndarray,
    geom: dict,
    wce_params: dict | None = None,
):
    """
    WCE 前处理 + FDK：
      - 对 noisy 投影做 WCE 校正
      - 调用 FDK 重建
    返回 (vol_wce, elapsed_sec, sino_corrected)
    """
    p = wce_params or {}
    sino_corr = wce_apply(
        noisy_sino_TVU,
        domain=p.get("domain", "linear"),
        alpha=float(p.get("alpha", 1.0)),
        beta=float(p.get("beta", 1.0)),
        sigma_v=float(p.get("sigma_v", 12.0)),
        sigma_u=float(p.get("sigma_u", 12.0)),
        clip=float(p.get("clip", 99.9)),
    )
    vol, dt = fdk_reconstruct(sino_corr, angles_rad, geom)
    return vol, dt, sino_corr
