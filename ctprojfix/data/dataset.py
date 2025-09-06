# ctprojfix/data/dataset.py
import os
import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def _read_tif(path):
    arr = iio.imread(path)  # 可能返回 (A,H,W) 或 (H,W)
    # 统一为 float32
    arr = np.asarray(arr, dtype=np.float32)
    return arr

def _percentile_norm(x, p1=1.0, p99=99.0, eps=1e-6):
    lo, hi = np.percentile(x, [p1, p99])
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

class DummyDataset(Dataset):
    def __init__(self, length=8, H=256, W=384, truncate_left=64, truncate_right=64):
        self.length = length
        self.H = H
        self.W = W
        self.truncate_left = truncate_left
        self.truncate_right = truncate_right

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        noisy = np.random.rand(self.H, self.W).astype(np.float32)
        mask = np.zeros((self.H, self.W), dtype=np.float32)
        mask[:, self.truncate_left:self.W-self.truncate_right] = 1.0
        x = np.stack([noisy, mask], axis=0)  # (2,H,W)
        y = np.random.rand(1, self.H, self.W).astype(np.float32)  # (1,H,W)
        return {"inp": torch.from_numpy(x), "gt": torch.from_numpy(y), "id": idx}
        
class ProjectionAnglesDataset(Dataset):
    """
    逐角度 2D 样本：
      - noisy_path: /data_shared/.../projectionNoisy{ID}.tif  (A,H,W)
      - clean_path: /data_shared/.../projection{ID}.tif       (A,H,W)
    每个样本返回一个角度帧 (H,W)：
      inp = [noisy_norm, mask]  -> (2,H,W)
      gt  = clean_norm          -> (1,H,W)
    """

    def __init__(self, cfg):
        self.root_noisy = cfg.get("root_noisy")
        self.root_clean = cfg.get("root_clean")
        self.ids = list(cfg.get("ids", []))
        self.step = int(cfg.get("step", 1))
        self.downsample = int(cfg.get("downsample", 1))
        self.truncate_left = int(cfg.get("truncate_left", 350))
        self.truncate_right = int(cfg.get("truncate_right", 350))
        self.mask_mode = str(cfg.get("mask_mode", "fixed")).lower()  # "fixed" / "auto_nonzero"
        self.normalize = str(cfg.get("normalize", "percentile")).lower()
        self.add_angle_channel = bool(cfg.get("add_angle_channel", False))

        # 扫描每个 ID 的角度数，构建样本索引表
        self.index = []  # [(id, a, A), ...]
        self.meta = {}   # id -> dict(A,H,W)
        for id_ in self.ids:
            noisy_path = os.path.join(self.root_noisy, f"projectionNoisy{id_}.tif")
            clean_path = os.path.join(self.root_clean, f"projection{id_}.tif")
            if not (os.path.isfile(noisy_path) and os.path.isfile(clean_path)):
                continue
            arr = _read_tif(noisy_path)
            if arr.ndim == 3:
                A, H, W = arr.shape
            elif arr.ndim == 2:
                A, (H, W) = 1, arr.shape
            else:
                raise ValueError(f"Unsupported tif ndim={arr.ndim} for {noisy_path}")
            self.meta[id_] = dict(A=A, H=H//self.downsample, W=W//self.downsample)

            # 逐角度建样本，支持 step 抽帧
            for a in range(0, A, self.step):
                self.index.append((id_, a, A))

        # 简单缓存，避免每次重复读整叠
        self.cache_noisy = {}
        self.cache_clean = {}

    def __len__(self):
        return len(self.index)

    def _load_frame(self, id_, a, kind="noisy"):
        if kind == "noisy":
            path = os.path.join(self.root_noisy, f"projectionNoisy{id_}.tif")
            cache = self.cache_noisy
        else:
            path = os.path.join(self.root_clean, f"projection{id_}.tif")
            cache = self.cache_clean

        if id_ not in cache:
            cache[id_] = _read_tif(path)  # (A,H,W) or (H,W)

        arr = cache[id_]
        if arr.ndim == 3:
            frame = arr[a]  # (H,W)
        else:
            frame = arr     # (H,W)

        if self.downsample > 1:
            ds = self.downsample
            frame = frame[::ds, ::ds]

        return frame.astype(np.float32)

    def _make_mask(self, H, W):
        if self.mask_mode == "fixed":
            L, R = self.truncate_left, self.truncate_right
            mask = np.zeros((H, W), dtype=np.float32)
            left = max(0, L); right = max(0, R)
            mask[:, left:W-right] = 1.0
            return mask
        else:  # auto_nonzero: 用非零区近似
            # 注意：这种方式依赖 noisy 帧；在 __getitem__ 里再按 noisy 构造更稳
            raise NotImplementedError

    def __getitem__(self, idx):
        id_, a, A = self.index[idx]
        noisy = self._load_frame(id_, a, kind="noisy")  # (H,W)
        clean = self._load_frame(id_, a, kind="clean")  # (H,W)

        H, W = noisy.shape  # 这里保证是 2D
        if self.normalize == "percentile":
            noisy_n = _percentile_norm(noisy)
            clean_n = _percentile_norm(clean)
        else:
            noisy_n = noisy.astype(np.float32)
            clean_n = clean.astype(np.float32)

        # mask：固定截断左右宽度
        mask = self._make_mask(H, W)

        # 打包 tensor
        inp_list = [noisy_n[None, ...], mask[None, ...]]  # (2,H,W)
        sample = {
            "inp": torch.from_numpy(np.concatenate(inp_list, axis=0)),  # (2,H,W)
            "gt":  torch.from_numpy(clean_n[None, ...]),                # (1,H,W)
            "angle": torch.tensor(a, dtype=torch.long),
            "A": torch.tensor(A, dtype=torch.long),
            "id": id_,
        }
        return sample

def make_dataloader(cfg):
    ds = ProjectionAnglesDataset(cfg)
    bs = int(cfg.get("batch_size", 1))
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=False)
