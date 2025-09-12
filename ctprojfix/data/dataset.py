# ctprojfix/data/dataset.py
import os, re, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 逐帧稳定读取（优先 tifffile；否则回退到 imageio）
try:
    import tifffile
    _HAS_TIFFFILE = True
except Exception:
    tifffile = None
    _HAS_TIFFFILE = False

try:
    import imageio.v3 as iio
    _HAS_IMAGEIO = True
except Exception:
    iio = None
    _HAS_IMAGEIO = False

from collections import OrderedDict


# ---------- 基础工具 ----------

def _discover_ids(root_noisy, root_clean):
    """
    扫描两个目录，解析出可用的 id 列表（取交集）。
    支持文件名：
      projectionNoisy0.tif / projectionNoisy05.tif
      projection0.tif      / projection05.tif
    """
    def parse_ids(root, prefix):
        patt = os.path.join(root, f"{prefix}*.tif")
        ids = set()
        for p in glob.glob(patt):
            fn = os.path.basename(p)
            m = re.search(rf"^{re.escape(prefix)}(\d+)\.tif$", fn)
            if m:
                ids.add(int(m.group(1)))
        return ids

    noisy_ids = parse_ids(root_noisy, "projectionNoisy")
    clean_ids = parse_ids(root_clean, "projection")
    return sorted(noisy_ids & clean_ids)


def _read_tif_stack(path):
    """整叠读取：只有在 cache_strategy='per_id' 时会用到。"""
    if _HAS_TIFFFILE:
        return np.asarray(tifffile.imread(path), dtype=np.float32)
    elif _HAS_IMAGEIO:
        return np.asarray(iio.imread(path), dtype=np.float32)
    else:
        raise RuntimeError("Neither tifffile nor imageio is available to read tif.")


def _read_tif_frame(path, a):
    """只读取第 a 帧，避免一次性把整叠载入内存。"""
    if _HAS_TIFFFILE:
        with tifffile.TiffFile(path) as tif:
            return np.asarray(tif.pages[a].asarray(), dtype=np.float32)
    elif _HAS_IMAGEIO:
        # imageio 无法精准“单帧”读取多页 tiff，只能整叠；尽量避免 fallback 到这里
        arr = iio.imread(path)
        arr = np.asarray(arr, dtype=np.float32)
        return arr[a] if arr.ndim == 3 else arr
    else:
        raise RuntimeError("Neither tifffile nor imageio is available to read tif.")


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
    """从 noisy 自动推断中心有效区"""
    H, W = noisy.shape
    col_has = (noisy > 0).any(axis=0)
    idx = np.where(col_has)[0]
    if idx.size == 0:
        return np.zeros((H, W), dtype=np.float32)
    L, R = int(idx.min()), int(idx.max())
    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, L:R+1] = 1.0
    return mask


# ---------- DummyDataset ----------
class DummyDataset(Dataset):
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
        gt = _percentile_norm(noisy + 0.1*np.random.randn(H, W).astype(np.float32))

        inp_list = [noisy[None, ...], mask[None, ...]]
        angle = idx % 360
        A = 360
        if self.add_angle:
            angle_norm = float(angle) / max(1, A - 1)
            angle_map = np.full((H, W), angle_norm, dtype=np.float32)
            inp_list.append(angle_map[None, ...])

        sample = {
            "inp": torch.from_numpy(np.concatenate(inp_list, axis=0)),
            "gt":  torch.from_numpy(gt[None, ...]),
            "id":  idx,
            "angle": torch.tensor(angle, dtype=torch.long),
            "A": torch.tensor(A, dtype=torch.long),
        }
        return sample


# ---------- ProjectionAnglesDataset ----------
class ProjectionAnglesDataset(Dataset):
    """
    逐角度 2D 样本：
      noisy: projectionNoisy{ID}.tif (A,H,W) 或 (H,W)
      clean: projection{ID}.tif      (A,H,W) 或 (H,W)

    新增：
      - cache_strategy: "none" | "lru" | "per_id"
        * none：完全不缓存（最稳最省内存）
        * lru：按 id 做 LRU 缓存（默认 2 个 id）
        * per_id：整叠缓存（占内存大；不建议多进程时用）
      - max_cached_ids: lru 模式下最多缓存的 id 数
    """
    def __init__(self, cfg):
        # 路径
        self.root_noisy = cfg.get("root_noisy")
        self.root_clean = cfg.get("root_clean")
        ids_cfg = cfg.get("ids", None)

        # 自动发现 ids
        if ids_cfg in (None, [], "all", "ALL"):
            self.ids = _discover_ids(self.root_noisy, self.root_clean)
        else:
            self.ids = list(ids_cfg)

        # 基本参数
        self.step = int(cfg.get("step", 1))
        self.downsample = int(cfg.get("downsample", 1))
        self.truncate_left = int(cfg.get("truncate_left", 350))
        self.truncate_right = int(cfg.get("truncate_right", 350))
        self.mask_mode = str(cfg.get("mask_mode", "fixed")).lower()
        self.normalize = str(cfg.get("normalize", "percentile")).lower()
        self.add_angle_channel = bool(cfg.get("add_angle_channel", False))

        # 缓存策略
        self.cache_strategy = str(cfg.get("cache_strategy", "lru")).lower()  # "none"|"lru"|"per_id"
        self.max_cached_ids = int(cfg.get("max_cached_ids", 2))
        self.cache_noisy = OrderedDict()  # id -> stack 或 list[frame]
        self.cache_clean = OrderedDict()

        # 索引构建
        self.index, self.meta = [], {}
        for id_ in self.ids:
            noisy_path = os.path.join(self.root_noisy, f"projectionNoisy{id_}.tif")
            clean_path = os.path.join(self.root_clean, f"projection{id_}.tif")
            if not (os.path.isfile(noisy_path) and os.path.isfile(clean_path)):
                continue

            # 用 tifffile 只取页数，避免整叠读入
            if _HAS_TIFFFILE:
                with tifffile.TiffFile(noisy_path) as tif:
                    A = len(tif.pages) if len(tif.pages) > 1 else 1
                    H, W = tif.pages[0].shape
            else:
                # 回退：读一次（可能整叠），仅用于拿形状
                arr_probe = _read_tif_stack(noisy_path)
                if arr_probe.ndim == 3:
                    A, H, W = arr_probe.shape
                elif arr_probe.ndim == 2:
                    A, (H, W) = 1, arr_probe.shape
                else:
                    raise ValueError(f"Unsupported tif ndim={arr_probe.ndim} for {noisy_path}")

            self.meta[id_] = dict(A=A, H=H//self.downsample, W=W//self.downsample)
            for a in range(0, A, self.step):
                self.index.append((id_, a, A))

        if len(self.index) == 0:
            noisy_cnt = len(glob.glob(os.path.join(self.root_noisy, "projectionNoisy*.tif")))
            clean_cnt = len(glob.glob(os.path.join(self.root_clean, "projection*.tif")))
            raise ValueError(
                f"[Dataset] 索引为空. noisy={noisy_cnt}, clean={clean_cnt}, ids={self.ids}\n"
                f"root_noisy={self.root_noisy}\nroot_clean={self.root_clean}"
            )

    # ---- LRU 工具 ----
    @staticmethod
    def _lru_get(cache: OrderedDict, key):
        if key in cache:
            val = cache.pop(key)
            cache[key] = val
            return val
        return None

    def _lru_put(self, cache: OrderedDict, key, value):
        cache[key] = value
        while len(cache) > self.max_cached_ids:
            cache.popitem(last=False)  # pop oldest

    # ---- 统一的逐帧/缓存读取 ----
    def _load_frame(self, id_, a, kind="noisy"):
        if kind == "noisy":
            path = os.path.join(self.root_noisy, f"projectionNoisy{id_:d}.tif")
            cache = self.cache_noisy
        else:
            path = os.path.join(self.root_clean, f"projection{id_:d}.tif")
            cache = self.cache_clean

        strat = self.cache_strategy
        if strat == "none":
            frame = _read_tif_frame(path, a)  # 每次只读一帧
        elif strat == "per_id":
            stack = self._lru_get(cache, id_)
            if stack is None:
                stack = _read_tif_stack(path)  # 整叠（占内存大）
                self._lru_put(cache, id_, stack)
            frame = stack[a] if stack.ndim == 3 else stack
        else:
            # lru：为每个 id 维护一个 list 占位；访问到哪帧就填哪帧
            holder = self._lru_get(cache, id_)
            if holder is None:
                if _HAS_TIFFFILE:
                    with tifffile.TiffFile(path) as tif:
                        holder = [None] * max(1, len(tif.pages))
                else:
                    # 无法只读一帧时退化为整叠（建议安装 tifffile）
                    holder = _read_tif_stack(path)
                self._lru_put(cache, id_, holder)

            if isinstance(holder, list):
                if holder[a] is None:
                    holder[a] = _read_tif_frame(path, a)
                frame = holder[a]
            else:
                frame = holder[a] if holder.ndim == 3 else holder

        # 下采样
        if self.downsample > 1:
            ds = self.downsample
            frame = frame[::ds, ::ds]

        return frame.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        id_, a, A = self.index[idx]
        noisy = self._load_frame(id_, a, "noisy")  # (H,W)
        clean = self._load_frame(id_, a, "clean")
        H, W = noisy.shape

        ds = max(1, self.downsample)
        L_eff = int(round(self.truncate_left  / ds))
        R_eff = int(round(self.truncate_right / ds))

        # 归一化
        if self.normalize == "percentile":
            noisy_n = _percentile_norm(noisy)
            clean_n = _percentile_norm(clean)
        else:
            noisy_n = noisy.astype(np.float32, copy=False)
            clean_n = clean.astype(np.float32, copy=False)

        # mask
        if self.mask_mode == "fixed":
            mask = _make_mask_fixed(H, W, L_eff, R_eff)
        elif self.mask_mode == "auto_nonzero":
            mask = _make_mask_auto_nonzero(noisy)
        else:
            raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

        if mask.sum() <= 0:
            raise RuntimeError(f"[DATA] mask is all zeros for id={id_}, angle={a}; "
                               f"check truncate_left/right vs downsample.")

        # 角度通道
        inp_list = [noisy_n[None, ...], mask[None, ...]]
        if self.add_angle_channel:
            angle_norm = float(a) / max(1, A - 1)
            angle_map = np.full((H, W), angle_norm, dtype=np.float32)
            inp_list.append(angle_map[None, ...])

        sample = {
            "inp": torch.from_numpy(np.concatenate(inp_list, axis=0)),  # (2/3,H,W)
            "gt":  torch.from_numpy(clean_n[None, ...]),                # (1,H,W)
            "angle": torch.tensor(a, dtype=torch.long),
            "A": torch.tensor(A, dtype=torch.long),
            "id": id_,
        }
        return sample


# ---------- DataLoader 构造 ----------
def make_dataloader(cfg):
    """
    读取 cfg 并构造 DataLoader。
    新增可选参数：
      - persistent_workers: bool（默认 True 当 num_workers>0 时）
      - prefetch_factor: int（默认 2，仅当 num_workers>0 时生效）
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

    persistent_workers = bool(cfg.get("persistent_workers", (nw > 0)))
    prefetch_factor = int(cfg.get("prefetch_factor", 2))
    dl_kwargs = dict(
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        drop_last=False,
    )
    if nw > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        dl_kwargs["prefetch_factor"] = max(2, prefetch_factor)

    return DataLoader(ds, **dl_kwargs)
