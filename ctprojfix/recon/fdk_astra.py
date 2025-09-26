# ctprojfix/recon/fdk_astra.py
from __future__ import annotations
import time
import numpy as np
import astra


def run_fdk(
    sino_TVU: np.ndarray,
    vol_shape_zyx,
    du: float,
    dv: float,
    det_u: int,
    det_v: int,
    angles_rad: np.ndarray,
    SOD: float,
    ODD: float,
    gpu: int,
    filt: str = "ram-lak",
):
    """
    三维锥束 FDK（ASTRA CUDA）。
    几何与数据维度：
      - 投影输入 (T,V,U)；ASTRA 期望 (V,T,U)，内部已转置。
      - create_proj_geom('cone', dv, du, det_v, det_u, angles, SOD, ODD)
    """
    nx, ny, nz = int(vol_shape_zyx[2]), int(vol_shape_zyx[1]), int(vol_shape_zyx[0])
    vg = astra.create_vol_geom(nx, ny, nz)

    angles = angles_rad.astype(np.float32)
    pg = astra.create_proj_geom(
        "cone", float(dv), float(du), int(det_v), int(det_u),
        angles, float(SOD), float(ODD)
    )

    sino_VTU = np.transpose(sino_TVU, (1, 0, 2))
    if not (sino_VTU.shape == (det_v, len(angles), det_u)):
        raise ValueError(f"[ASTRA] sino shape {sino_VTU.shape} != (V={det_v}, T={len(angles)}, U={det_u})")

    sid = astra.data3d.create("-proj3d", pg, sino_VTU)
    rid = astra.data3d.create("-vol", vg)

    cfg = astra.astra_dict("FDK_CUDA")
    cfg["ProjectionDataId"] = sid
    cfg["ReconstructionDataId"] = rid
    cfg["option"] = {"GPUindex": int(gpu), "FilterType": str(filt)}

    t0 = time.time()
    alg = astra.algorithm.create(cfg)
    astra.algorithm.run(alg)
    dt = time.time() - t0

    vol = astra.data3d.get(rid).astype(np.float32)
    astra.algorithm.delete(alg)
    astra.data3d.delete([sid, rid])
    return vol, dt


def fdk_reconstruct(sino_TVU: np.ndarray, angles_rad: np.ndarray, geom: dict):
    """
    便捷封装：从 geom 读取参数并调用 run_fdk。
    需要 geom 包含：vol_shape, du, dv, SOD, ODD, gpu, 可选filter
    """
    required = ["vol_shape", "du", "dv", "SOD", "ODD"]
    for k in required:
        if k not in geom:
            raise ValueError(f"geom 缺少必要字段: {k}")

    det_v, det_u = sino_TVU.shape[1], sino_TVU.shape[2]
    return run_fdk(
        sino_TVU, tuple(geom["vol_shape"]),
        float(geom["du"]), float(geom["dv"]),
        int(det_u), int(det_v),
        angles_rad, float(geom["SOD"]), float(geom["ODD"]),
        int(geom.get("gpu", 0)), str(geom.get("filter", "ram-lak")),
    )
