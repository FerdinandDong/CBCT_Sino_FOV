import os
import numpy as np
import imageio.v3 as iio
import torch
from torch.utils.data import Dataset, DataLoader

#加DummyDataset 本地测试跑通空转
import random

class ProjFrameDataset(Dataset):
    """
    加载投影对 (noisy vs full)。
    默认逐帧读取，节省内存。
    """
    def __init__(self, root_noisy, root_clean, ids, truncate_left=350, truncate_right=350,
                 normalize="percentile", split="train"):
        self.root_noisy = root_noisy
        self.root_clean = root_clean
        self.ids = ids
        self.truncate_left = truncate_left
        self.truncate_right = truncate_right
        self.normalize = normalize
        self.split = split

        self.samples = []
        for ID in ids:
            noisy_path = os.path.join(root_noisy, f"projectionNoisy{ID}.tif")
            clean_path = os.path.join(root_clean, f"projection{ID}.tif")
            if os.path.exists(noisy_path) and os.path.exists(clean_path):
                self.samples.append((ID, noisy_path, clean_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ID, noisy_path, clean_path = self.samples[idx]

        # 读取一帧（test第0帧，扩展到全角度）
        noisy = iio.imread(noisy_path, index=0).astype(np.float32)
        clean = iio.imread(clean_path, index=0).astype(np.float32)

        # 归一化
        if self.normalize == "percentile":
            p1, p99 = np.percentile(noisy, (1, 99))
            noisy = (noisy - p1) / (p99 - p1 + 1e-6)
            noisy = np.clip(noisy, 0, 1)
            p1, p99 = np.percentile(clean, (1, 99))
            clean = (clean - p1) / (p99 - p1 + 1e-6)
            clean = np.clip(clean, 0, 1)

        # mask 通道
        H, W = noisy.shape
        mask = np.zeros((H, W), dtype=np.float32)
        mask[:, self.truncate_left:W-self.truncate_right] = 1.0

        # 拼接通道：noisy + mask
        x = np.stack([noisy, mask], axis=0)  # (2,H,W)
        y = clean[None, ...]  # (1,H,W)

        return {"inp": torch.from_numpy(x), "gt": torch.from_numpy(y), "id": ID}

# 加DummyDataset 本地测试跑通空转
class DummyDataset(Dataset):
    """
    没有真实数据时用的占位数据集：随机张量，尺寸与通道和 mask 规则一致。
    """
    def __init__(self, length=8, H=256, W=384, truncate_left=64, truncate_right=64, add_angle_channel=False):
        
        self.add_angle = add_angle_channel
        self.length = int(length)
        self.H = int(H); self.W = int(W)
        self.truncate_left = int(truncate_left); self.truncate_right = int(truncate_right)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        noisy = np.random.rand(self.H, self.W).astype(np.float32)
        clean = np.random.rand(self.H, self.W).astype(np.float32)

        mask = np.zeros((self.H, self.W), dtype=np.float32)
        L, R = self.truncate_left, self.W - self.truncate_right
        L = max(0, min(L, self.W)); R = max(0, min(R, self.W))
        if R > L:
            mask[:, L:R] = 1.0

        x = np.stack([noisy, mask], axis=0)  # (2,H,W)
        y = clean[None, ...]                 # (1,H,W)
        return {"inp": torch.from_numpy(x), "gt": torch.from_numpy(y), "id": -1}


# --- 无数据时回退 Dummy ---
def make_dataloader(cfg):
    use_dummy = cfg.get("use_dummy", False)

    if not use_dummy:
        ds = ProjFrameDataset(
            root_noisy=cfg["root_noisy"],
            root_clean=cfg["root_clean"],
            ids=cfg["ids"],
            truncate_left=cfg.get("truncate_left", 350),
            truncate_right=cfg.get("truncate_right", 350),
            normalize=cfg.get("normalize", "percentile"),
            split="train",
        )
        if len(ds) == 0:
            # 自动回退（打印提示）
            print("[INFO] ProjFrameDataset 为空，自动回退 DummyDataset。可在 configs 里设置 data.use_dummy=true 关闭提示。")
            use_dummy = True

    if use_dummy:
        ds = DummyDataset(
            length=cfg.get("dummy_length", 8),
            H=cfg.get("dummy_H", 256),
            W=cfg.get("dummy_W", 384),
            truncate_left=cfg.get("truncate_left", 64),
            truncate_right=cfg.get("truncate_right", 64),
        )

    return DataLoader(ds, batch_size=cfg.get("batch_size", 1), shuffle=True, num_workers=0)

# --- end ---

# # ctprojfix/data/dataset.py
# import os
# import math
# import numpy as np
# import imageio.v3 as iio
# import torch
# from torch.utils.data import Dataset, DataLoader

# # ---------- 辅助 ----------
# def _safe_read_frame(tif_path: str, k: int):
#     """
#     安全读取第 k 帧：
#     - 如果是 3D (A,H,W)，用 index=k 读一帧
#     - 如果是 2D (H,W)（有些数据集每文件只一帧），k 只能是 0
#     """
#     try:
#         arrk = iio.imread(tif_path, index=k)
#     except Exception:
#         arr = iio.imread(tif_path)
#         if arr.ndim == 3:
#             arrk = arr[k]
#         elif arr.ndim == 2:
#             if k != 0:
#                 raise IndexError(f"{tif_path} 只有单帧，但请求 k={k}")
#             arrk = arr
#         else:
#             raise ValueError(f"Unsupported ndim={arr.ndim} for {tif_path}")
#     return arrk.astype(np.float32)


# def _count_frames(tif_path: str):
#     """估计帧数 A（若为 2D 文件则返回 1）"""
#     try:
#         arr = iio.imread(tif_path)
#         return arr.shape[0] if arr.ndim == 3 else 1
#     except Exception:
#         # 退化：逐索引探测（几乎用不到）
#         k = 0
#         while True:
#             try:
#                 _ = iio.imread(tif_path, index=k)
#                 k += 1
#             except Exception:
#                 break
#         return k


# def _normalize_percentile(img: np.ndarray, p1=1.0, p99=99.0):
#     v1, v99 = np.percentile(img[np.isfinite(img)], (p1, p99))
#     if v99 <= v1:
#         v1, v99 = float(np.nanmin(img)), float(np.nanmax(img))
#     out = (img - v1) / (v99 - v1 + 1e-6)
#     return np.clip(out, 0.0, 1.0).astype(np.float32)


# def _make_mask(W: int, left: int, right: int):
#     """固定截断掩膜：中间 1，两侧 0"""
#     mask = np.zeros((W,), dtype=np.float32)
#     L, R = int(left), int(W - right)
#     L = max(0, min(L, W)); R = max(0, min(R, W))
#     if R > L:
#         mask[L:R] = 1.0
#     return mask


# def _auto_mask_from_nonzero(img: np.ndarray, thresh: float = 1e-6):
#     """自动掩膜：非零为 1（对全 0 行也鲁棒）"""
#     nz = (np.abs(img) > thresh).astype(np.float32)
#     # 聚合到列维度，得到每列是否“曾经非零”
#     col = (nz.sum(axis=0) > 0).astype(np.float32)
#     return col


# # ---------- 主数据集：逐角度 ----------
# class ProjectionAnglesDataset(Dataset):
#     """
#     遍历 (ID, angle_k) 样本：
#       inp 通道: [noisy, mask] (+ 可选 angle_norm)
#       gt  通道: [clean]
#     支持：
#       - step: 抽帧（如 1/2/3）
#       - downsample: H/W 方向按步长采样
#       - mask_mode: fixed(默认) / auto_nonzero
#     """
#     def __init__(
#         self,
#         root_noisy: str,
#         root_clean: str,
#         ids,
#         step: int = 1,
#         downsample: int = 1,
#         truncate_left: int = 350,
#         truncate_right: int = 350,
#         mask_mode: str = "fixed",  # or "auto_nonzero"
#         normalize: str = "percentile",
#         add_angle_channel: bool = False,
#     ):
#         self.root_noisy = root_noisy
#         self.root_clean = root_clean
#         self.ids = list(ids)
#         self.step = int(step)
#         self.down = int(downsample)
#         self.truncL = int(truncate_left)
#         self.truncR = int(truncate_right)
#         self.mask_mode = mask_mode
#         self.normalize = normalize
#         self.add_angle = add_angle_channel

#         # 预扫描每个 ID 的帧数（默认 360）
#         self.index = []  # [(ID, k, A), ...]
#         for ID in self.ids:
#             noisy_path = os.path.join(root_noisy, f"projectionNoisy{ID}.tif")
#             clean_path = os.path.join(root_clean, f"projection{ID}.tif")
#             if not (os.path.isfile(noisy_path) and os.path.isfile(clean_path)):
#                 # 缺文件就跳过该 ID
#                 continue
#             A = min(_count_frames(noisy_path), _count_frames(clean_path))
#             for k in range(0, A, self.step):
#                 self.index.append((ID, k, A))

#         if len(self.index) == 0:
#             print("[WARN] ProjectionAnglesDataset 构建为空（可能本地无数据）。")

#     def __len__(self):
#         return len(self.index)

#     def __getitem__(self, idx):
#         ID, k, A = self.index[idx]
#         noisy_path = os.path.join(self.root_noisy, f"projectionNoisy{ID}.tif")
#         clean_path = os.path.join(self.root_clean, f"projection{ID}.tif")

#         noisy = _safe_read_frame(noisy_path, k)  # (H, W)
#         clean = _safe_read_frame(clean_path, k)  # (H, W)

#         # 归一化（逐帧）
#         if self.normalize == "percentile":
#             noisy = _normalize_percentile(noisy)
#             clean = _normalize_percentile(clean)

#         H, W = noisy.shape

#         # 掩膜
#         if self.mask_mode == "auto_nonzero":
#             col = _auto_mask_from_nonzero(noisy)
#         else:  # "fixed"
#             col = _make_mask(W, self.truncL, self.truncR)
#         mask = np.broadcast_to(col[None, :], (H, W)).astype(np.float32)

#         # 降采样
#         if self.down > 1:
#             noisy = noisy[::self.down, ::self.down]
#             clean = clean[::self.down, ::self.down]
#             mask  = mask[::self.down, ::self.down]

#         # 组装通道
#         chans = [noisy, mask]
#         if self.add_angle:
#             angle_norm = np.full_like(noisy, fill_value=(k / max(1, A - 1)), dtype=np.float32)
#             chans.append(angle_norm)

#         x = np.stack(chans, axis=0)  # (C,H,W), C=2 或 3
#         y = clean[None, ...]         # (1,H,W)

#         sample = {
#             "inp": torch.from_numpy(x),     # float32
#             "gt": torch.from_numpy(y),      # float32
#             "id": int(ID),
#             "angle": int(k),
#             "A": int(A),
#         }
#         return sample


# # ---------- Dummy（本地无数据时空跑） ----------
# class DummyDataset(Dataset):
#     def __init__(self, length=16, H=256, W=384, truncate_left=64, truncate_right=64, add_angle_channel=False):
#         self.length = int(length)
#         self.H, self.W = int(H), int(W)
#         self.L, self.R = int(truncate_left), int(truncate_right)
#         self.add_angle = add_angle_channel

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         noisy = np.random.rand(self.H, self.W).astype(np.float32)
#         clean = np.random.rand(self.H, self.W).astype(np.float32)

#         mask = np.zeros((self.H, self.W), dtype=np.float32)
#         L = max(0, min(self.L, self.W)); R = max(0, min(self.W - self.R, self.W))
#         if R > L: mask[:, L:R] = 1.0

#         chans = [noisy, mask]
#         if self.add_angle:
#             angle_norm = np.full_like(noisy, fill_value=(idx / max(1, self.length - 1)), dtype=np.float32)
#             chans.append(angle_norm)

#         x = np.stack(chans, axis=0)
#         y = clean[None, ...]
#         return {"inp": torch.from_numpy(x), "gt": torch.from_numpy(y), "id": -1, "angle": idx, "A": self.length}


# # ---------- 工厂 ----------
# def make_dataloader(cfg):
#     """
#     cfg.data 示例：
#       root_noisy, root_clean, ids, batch_size, step, downsample,
#       truncate_left/right, mask_mode, normalize, add_angle_channel,
#       use_dummy, dummy_*
#     """
#     use_dummy = cfg.get("use_dummy", False)

#     if not use_dummy:
#         ds = ProjectionAnglesDataset(
#             root_noisy   = cfg["root_noisy"],
#             root_clean   = cfg["root_clean"],
#             ids          = cfg["ids"],
#             step         = cfg.get("step", 1),
#             downsample   = cfg.get("downsample", 1),
#             truncate_left= cfg.get("truncate_left", 350),
#             truncate_right=cfg.get("truncate_right", 350),
#             mask_mode    = cfg.get("mask_mode", "fixed"),
#             normalize    = cfg.get("normalize", "percentile"),
#             add_angle_channel = cfg.get("add_angle_channel", False),
#         )
#         if len(ds) == 0:
#             print("[INFO] 数据集为空，自动回退 DummyDataset（本地无数据调试）。")
#             use_dummy = True

#     if use_dummy:
#         ds = DummyDataset(
#             length = cfg.get("dummy_length", 16),
#             H      = cfg.get("dummy_H", 256),
#             W      = cfg.get("dummy_W", 384),
#             truncate_left  = cfg.get("truncate_left", 64),
#             truncate_right = cfg.get("truncate_right", 64),
#             add_angle_channel = cfg.get("add_angle_channel", False),
#         )

#     return DataLoader(
#         ds,
#         batch_size=cfg.get("batch_size", 2),
#         shuffle=True,
#         num_workers=cfg.get("num_workers", 0),
#         pin_memory=cfg.get("pin_memory", False),
#     )
