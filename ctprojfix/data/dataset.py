# ctprojfix/data/dataset.py
import os
import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------- 基础工具 ----------

def _read_tif(path):
    """读取 tif，返回 float32 的 np.ndarray，形状可能是 (A,H,W) 或 (H,W)"""
    arr = iio.imread(path)
    return np.asarray(arr, dtype=np.float32)

def _percentile_norm(x, p1=1.0, p99=99.0, eps=1e-6):
    """按百分位归一化到 [0,1]"""
    lo, hi = np.percentile(x, [p1, p99])
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

def _make_mask_fixed(H, W, left, right):
    """固定截断宽度的中心有效区 mask（1=中心有效，0=两侧）"""
    left = max(0, int(left)); right = max(0, int(right))
    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, left:W-right] = 1.0 if (W - right) > left else 0.0
    return mask

def _make_mask_auto_nonzero(noisy):
    """
    从 noisy 自动推断中心有效区：
    - 找到每列是否存在非零
    - 取非零列的最左/最右，填满中间为 1
    """
    H, W = noisy.shape
    col_has = (noisy > 0).any(axis=0)
    idx = np.where(col_has)[0]
    if idx.size == 0:
        return np.zeros((H, W), dtype=np.float32)
    L, R = int(idx.min()), int(idx.max())
    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, L:R+1] = 1.0
    return mask


# ---------- DummyDataset（本地/流程测试） ----------

class DummyDataset(Dataset):
    """
    生成伪造的 [noisy, mask] (+ angle) 与 gt，便于本地/流程测试
    """
    def __init__(self, length=8, H=256, W=384,
                 truncate_left=64, truncate_right=64,
                 add_angle_channel=False):
        self.length = int(length)
        self.H, self.W = int(H), int(W)
        self.truncate_left = int(truncate_left)
        self.truncate_right = int(truncate_right)
        self.add_angle = bool(add_angle_channel)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        H, W = self.H, self.W
        noisy = np.random.rand(H, W).astype(np.float32)
        mask = _make_mask_fixed(H, W, self.truncate_left, self.truncate_right)
        # 目标（随便生成一张更平滑的图以模拟“干净”）
        gt = _percentile_norm(noisy + 0.1*np.random.randn(H, W).astype(np.float32))

        inp_list = [noisy[None, ...], mask[None, ...]]
        angle = idx % 360
        A = 360
        if self.add_angle:
            angle_norm = float(angle) / max(1, A - 1)
            angle_map = np.full((H, W), angle_norm, dtype=np.float32)
            inp_list.append(angle_map[None, ...])

        sample = {
            "inp": torch.from_numpy(np.concatenate(inp_list, axis=0)),  # (2 or 3, H, W)
            "gt":  torch.from_numpy(gt[None, ...]),                     # (1, H, W)
            "id":  idx,
            "angle": torch.tensor(angle, dtype=torch.long),
            "A": torch.tensor(A, dtype=torch.long),
        }
        return sample


# ---------- 真实数据：逐角度 2D 样本 ----------

class ProjectionAnglesDataset(Dataset):
    """
    逐角度 2D 样本：
      - noisy_path: /.../projectionNoisy{ID}.tif (A,H,W)
      - clean_path: /.../projection{ID}.tif      (A,H,W)
    返回：
      inp = [noisy_norm, mask] (+ angle_map)  -> (C,H,W), C=2/3
      gt  = clean_norm                         -> (1,H,W)
      以及 angle, A, id 等元信息
    """

    def __init__(self, cfg):
        # 路径 & 基本配置
        self.root_noisy = cfg.get("root_noisy")
        self.root_clean = cfg.get("root_clean")
        self.ids = list(cfg.get("ids", []))
        self.step = int(cfg.get("step", 1))
        self.downsample = int(cfg.get("downsample", 1))
        self.truncate_left = int(cfg.get("truncate_left", 350))
        self.truncate_right = int(cfg.get("truncate_right", 350))
        self.mask_mode = str(cfg.get("mask_mode", "fixed")).lower()  # "fixed" or "auto_nonzero"
        self.normalize = str(cfg.get("normalize", "percentile")).lower()
        self.add_angle_channel = bool(cfg.get("add_angle_channel", False))

        # 索引构建：[(id, a, A), ...]
        self.index = []
        self.meta = {}  # id -> dict(A,H,W)
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
            for a in range(0, A, self.step):
                self.index.append((id_, a, A))

        # 简单缓存（整叠），按需取帧
        self.cache_noisy = {}
        self.cache_clean = {}
        



    def __len__(self):
        return len(self.index)

    def _load_frame(self, id_, a, kind="noisy"):
        if kind == "noisy":
            path = os.path.join(self.root_noisy, f"projectionNoisy{id_:d}.tif")
            cache = self.cache_noisy
        else:
            path = os.path.join(self.root_clean, f"projection{id_:d}.tif")
            cache = self.cache_clean

        if id_ not in cache:
            cache[id_] = _read_tif(path)  # (A,H,W) or (H,W)

        arr = cache[id_]
        frame = arr[a] if arr.ndim == 3 else arr  # (H,W)

        if self.downsample > 1:
            ds = self.downsample
            frame = frame[::ds, ::ds]

        return frame.astype(np.float32, copy=False)

    def __getitem__(self, idx):
        id_, a, A = self.index[idx]
        noisy = self._load_frame(id_, a, kind="noisy")  # (H,W)
        clean = self._load_frame(id_, a, kind="clean")  # (H,W)
        H, W = noisy.shape

        ds = max(1, self.downsample)
        # 缩放后的截断宽度
        L_eff = int(round(self.truncate_left / ds))
        R_eff = int(round(self.truncate_right / ds))

        # 归一化
        if self.normalize == "percentile":
            noisy_n = _percentile_norm(noisy)
            clean_n = _percentile_norm(clean)
        else:
            noisy_n = noisy.astype(np.float32, copy=False)
            clean_n = clean.astype(np.float32, copy=False)

        if self.mask_mode == "fixed":
            mask = _make_mask_fixed(H, W, L_eff, R_eff)
        elif self.mask_mode == "auto_nonzero":
            mask = _make_mask_auto_nonzero(noisy)
        else:
            raise ValueError(f"Unknown mask_mode: {self.mask_mode}")
        ##需要做一次检查 mask是否全0 在sample里做


        # 角度通道
        inp_list = [noisy_n[None, ...], mask[None, ...]]  # (2,H,W)
        if self.add_angle_channel:
            angle_norm = float(a) / max(1, A - 1)
            angle_map = np.full((H, W), angle_norm, dtype=np.float32)
            inp_list.append(angle_map[None, ...])         # (3,H,W)

        sample = {
            "inp": torch.from_numpy(np.concatenate(inp_list, axis=0)),
            "gt":  torch.from_numpy(clean_n[None, ...]),
            "angle": torch.tensor(a, dtype=torch.long),
            "A": torch.tensor(A, dtype=torch.long),
            "id": id_,
        }
        return sample


# ---------- DataLoader 构造 ----------

def make_dataloader(cfg):
    """
    从 cfg 构造 DataLoader：
      - batch_size / num_workers / pin_memory 都会读取
    """
    use_dummy = bool(cfg.get("use_dummy", False))
    bs = int(cfg.get("batch_size", 1))
    nw = int(cfg.get("num_workers", 0))
    pin = bool(cfg.get("pin_memory", False))

    if use_dummy:
        ds = DummyDataset(length=cfg.get("dummy_length", 8),
                          H=cfg.get("dummy_H", 256),
                          W=cfg.get("dummy_W", 384),
                          truncate_left=cfg.get("truncate_left", 64),
                          truncate_right=cfg.get("truncate_right", 64),
                          add_angle_channel=cfg.get("add_angle_channel", False))
    else:
        ds = ProjectionAnglesDataset(cfg)

    return DataLoader(ds, batch_size=bs, shuffle=True,
                      num_workers=nw, pin_memory=pin, drop_last=False)
