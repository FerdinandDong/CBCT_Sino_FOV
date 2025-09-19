# ctprojfix/data/dataset.py
import os, re, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset  # >>> split: Subset

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


# 工具组模块

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
    """按百分位归一化到 [0,1]（老实现，保留以保持兼容）"""
    lo, hi = np.percentile(x, [p1, p99])
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _percentile_norm_stats(x, p1=1.0, p99=99.0, eps=1e-6):
    """按百分位归一化到 [0,1]，同时返回 (lo, hi) 以便后续反归一化"""
    lo, hi = np.percentile(x, [p1, p99])
    if hi - lo < eps:
        y = np.zeros_like(x, dtype=np.float32)
    else:
        y = (x - lo) / (hi - lo)
        y = np.clip(y, 0.0, 1.0).astype(np.float32)
    return y, float(lo), float(hi)


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
    def __init__(self, length=8, H=960, W=1240,
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

    可选开关（向后兼容）：
      - return_norm_stats: bool（默认 False）
        * False：行为与旧版本完全一致（训练不受影响）
        * True：额外返回每帧的 noisy_lo/noisy_hi/gt_lo/gt_hi 四个标量
    """
    def __init__(self, cfg):
        # 路径
        self.root_noisy = cfg.get("root_noisy")
        self.root_clean = cfg.get("root_clean")
        ids_cfg = cfg.get("ids", None)

        # 自动发现 ids + 排除列表  # >>> split: exclude_ids 支持
        if ids_cfg in (None, [], "all", "ALL"):
            ids = _discover_ids(self.root_noisy, self.root_clean)
        else:
            ids = list(ids_cfg)
        exclude = set(cfg.get("exclude_ids", []))
        if exclude:
            ids = [i for i in ids if i not in exclude]
        self.ids = ids

        # 基本参数
        self.step = int(cfg.get("step", 1))
        self.downsample = int(cfg.get("downsample", 1))
        self.truncate_left = int(cfg.get("truncate_left", 350))
        self.truncate_right = int(cfg.get("truncate_right", 350))
        self.mask_mode = str(cfg.get("mask_mode", "fixed")).lower()
        self.normalize = str(cfg.get("normalize", "percentile")).lower()
        self.add_angle_channel = bool(cfg.get("add_angle_channel", False))

        # 仅评估时会打开；训练默认 False
        self.return_norm_stats = bool(cfg.get("return_norm_stats", False))

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

        # 归一化（是否返回统计由 return_norm_stats 决定；默认 False 以保持训练兼容）
        if self.normalize == "percentile":
            if self.return_norm_stats:
                noisy_n, n_lo, n_hi = _percentile_norm_stats(noisy)
                clean_n, g_lo, g_hi = _percentile_norm_stats(clean)
            else:
                noisy_n = _percentile_norm(noisy); n_lo, n_hi = 0.0, 1.0
                clean_n = _percentile_norm(clean); g_lo, g_hi = 0.0, 1.0
        else:
            noisy_n = noisy.astype(np.float32, copy=False); n_lo, n_hi = 0.0, 1.0
            clean_n = clean.astype(np.float32, copy=False); g_lo, g_hi = 0.0, 1.0

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

        # 仅在评估开关打开时附加统计量（训练不会看到）
        if self.return_norm_stats:
            sample.update({
                "noisy_lo": torch.tensor(n_lo, dtype=torch.float32),
                "noisy_hi": torch.tensor(n_hi, dtype=torch.float32),
                "gt_lo":    torch.tensor(g_lo, dtype=torch.float32),
                "gt_hi":    torch.tensor(g_hi, dtype=torch.float32),
            })

        return sample


# ---------- Split 辅助（files/ratio） ----------  # >>> split

def _read_list_file(path):  # >>> split
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            items.append(ln)
    return items


def _make_index_keys(ds: ProjectionAnglesDataset):  # >>> split
    """统一索引键："id,a"（角度级别）。"""
    return [f"{id_},{a}" for (id_, a, A) in ds.index]


def _subset_indices_from_files(ds: ProjectionAnglesDataset, split_cfg):  # >>> split
    list_format = str(split_cfg.get('list_format', 'id')).lower()
    keys = _make_index_keys(ds)
    # 按 ID 聚类，便于快捷选择
    by_id = {}
    for pos, k in enumerate(keys):
        sid, sa = k.split(',')
        by_id.setdefault(sid, []).append((pos, int(sa)))

    def collect(path):
        lines = _read_list_file(path)
        if list_format == 'id':
            pos = []
            for sid in lines:
                if sid in by_id:
                    pos.extend([p for (p, _) in by_id[sid]])
            return sorted(pos)
        elif list_format in ('id_a', 'id,angle'):
            want = set(lines)
            return sorted([i for i, k in enumerate(keys) if k in want])
        else:
            raise ValueError(f"Unknown list_format: {list_format}")

    return (
        collect(split_cfg['train_list']),
        collect(split_cfg['val_list']),
        collect(split_cfg['test_list']),
    )


def _subset_indices_by_ratio(ds: ProjectionAnglesDataset, split_cfg):  # >>> split
    rng = np.random.default_rng(int(split_cfg.get('seed', 2024)))
    keys = _make_index_keys(ds)
    group_by = str(split_cfg.get('group_by', 'id')).lower()

    if group_by == 'none':
        idx = np.arange(len(keys))
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(n * float(split_cfg.get('train', 0.8)))
        n_va = int(n * float(split_cfg.get('val', 0.1)))
        tr = idx[:n_tr].tolist()
        va = idx[n_tr:n_tr+n_va].tolist()
        te = idx[n_tr+n_va:].tolist()
        return sorted(tr), sorted(va), sorted(te)

    # group_by = 'id'：按体 ID 分组
    groups = {}
    for pos, k in enumerate(keys):
        sid = k.split(',')[0]
        groups.setdefault(sid, []).append(pos)
    gids = list(groups.keys())
    rng.shuffle(gids)
    n = len(gids)
    n_tr = int(n * float(split_cfg.get('train', 0.8)))
    n_va = int(n * float(split_cfg.get('val', 0.1)))
    g_tr = set(gids[:n_tr])
    g_va = set(gids[n_tr:n_tr+n_va])
    g_te = set(gids[n_tr+n_va:])
    tr = [p for g in g_tr for p in groups[g]]
    va = [p for g in g_va for p in groups[g]]
    te = [p for g in g_te for p in groups[g]]
    return sorted(tr), sorted(va), sorted(te)


# ---------- DataLoader 构造 ----------

def make_dataloader(cfg):
    """
    读取 cfg 并构造 DataLoader。
    支持：
      - split: files | ratio（返回 {train,val,test} 字典）
      - 无 split：返回单个 DataLoader（兼容旧代码）
      - persistent_workers / prefetch_factor
      - exclude_ids: [0,10,11]
    """
    use_dummy = bool(cfg.get("use_dummy", False))
    bs = int(cfg.get("batch_size", 1))
    nw = int(cfg.get("num_workers", 0))
    pin = bool(cfg.get("pin_memory", False))

    persistent_workers = bool(cfg.get("persistent_workers", (nw > 0)))
    prefetch_factor = int(cfg.get("prefetch_factor", 2))

    # Dummy：保持单 Loader 行为
    if use_dummy:
        ds = DummyDataset(length=cfg.get("dummy_length", 8),
                          H=cfg.get("dummy_H", 256),
                          W=cfg.get("dummy_W", 384),
                          truncate_left=cfg.get("truncate_left", 64),
                          truncate_right=cfg.get("truncate_right", 64),
                          add_angle_channel=cfg.get("add_angle_channel", False))
        dl_kwargs = dict(batch_size=bs, shuffle=True, num_workers=nw,
                         pin_memory=pin, drop_last=False)
        if nw > 0:
            dl_kwargs.update({"persistent_workers": persistent_workers,
                              "prefetch_factor": max(2, prefetch_factor)})
        return DataLoader(ds, **dl_kwargs)

    # 真数据：构建全量 Dataset
    ds_full = ProjectionAnglesDataset(cfg)

    split_cfg = cfg.get('split', None)
    if not split_cfg:
        # 没有 split：返回单 Loader（兼容旧训练脚本）
        dl_kwargs = dict(batch_size=bs, shuffle=True, num_workers=nw,
                         pin_memory=pin, drop_last=False)
        if nw > 0:
            dl_kwargs.update({"persistent_workers": persistent_workers,
                              "prefetch_factor": max(2, prefetch_factor)})
        return DataLoader(ds_full, **dl_kwargs)

    # 有 split：生成三份子集
    mode = str(split_cfg.get('mode', 'ratio')).lower()
    if mode == 'files':
        idx_tr, idx_va, idx_te = _subset_indices_from_files(ds_full, split_cfg)
    elif mode == 'ratio':
        idx_tr, idx_va, idx_te = _subset_indices_by_ratio(ds_full, split_cfg)
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    ds_tr = Subset(ds_full, idx_tr)
    ds_va = Subset(ds_full, idx_va)
    ds_te = Subset(ds_full, idx_te)

    common = dict(num_workers=nw, pin_memory=pin, drop_last=False)
    if nw > 0:
        common.update({"persistent_workers": persistent_workers,
                       "prefetch_factor": max(2, prefetch_factor)})

    train_loader = DataLoader(ds_tr, batch_size=bs, shuffle=True,  **common)
    val_loader   = DataLoader(ds_va, batch_size=int(cfg.get('val_batch_size', bs)),  shuffle=False, **common)
    test_loader  = DataLoader(ds_te, batch_size=int(cfg.get('test_batch_size', bs)), shuffle=False, **common)

    return {"train": train_loader, "val": val_loader, "test": test_loader}
