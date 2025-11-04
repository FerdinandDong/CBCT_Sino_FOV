#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
python scripts/check_data_units.py \
  --root-noisy /data/shared_folder/projectionsNoisy \
  --root-clean /data/shared_folder/projectionsNoisy \
  --max-ids 0 --sample-frames 8
快速检查投影数据是 HU / 强度 / 线积分 域的。
'''
import os, re, glob, argparse, json
import numpy as np

try:
    import tifffile
    HAS_TIFF = True
except Exception:
    tifffile = None
    HAS_TIFF = False

def parse_args():
    ap = argparse.ArgumentParser(description="Quick check: are your projections in HU / intensity / log?")
    ap.add_argument("--root-noisy", default="/data/shared_folder/projectionsNoisy",
                    help="目录：projectionNoisy{id}.tif（截断+噪声）")
    ap.add_argument("--root-clean", default="/data/shared_folder/projectionsNoisy",
                    help="目录：projection{id}.tif（GT，无噪）")
    ap.add_argument("--ids", type=int, nargs="*", default=None,
                    help="要抽查的 ID 列表；不填则自动发现前几个")
    ap.add_argument("--sample-frames", type=int, default=8,
                    help="每个 tiff 抽查的帧数（均匀采样）")
    ap.add_argument("--max-ids", type=int, default=3,
                    help="最多检查多少个 ID（自动发现模式生效）")
    return ap.parse_args()

def _discover_ids(root_noisy, root_clean):
    def pick(root, prefix):
        patt = os.path.join(root, f"{prefix}*.tif")
        ids = []
        for p in glob.glob(patt):
            fn = os.path.basename(p)
            m = re.match(rf"^{re.escape(prefix)}(\d+)\.tif$", fn)
            if m: ids.append(int(m.group(1)))
        return set(ids)
    a = pick(root_noisy, "projectionNoisy")
    b = pick(root_clean, "projection")
    ids = sorted(a & b)
    return ids

def _read_tiff_stats(path, sample_frames=8):
    """只抽样读取若干页做统计，返回 dict。"""
    if not HAS_TIFF:
        raise RuntimeError("tifffile 未安装：pip install tifffile")

    with tifffile.TiffFile(path) as tif:
        pages = tif.pages
        A = len(pages) if len(pages) > 1 else 1
        H, W = pages[0].shape
        if A <= sample_frames:
            idx = list(range(A))
        else:
            # 均匀采样
            idx = np.linspace(0, A-1, sample_frames, dtype=int).tolist()

        vals = []
        zeros = 0
        negs = 0
        for i in idx:
            arr = pages[i].asarray()
            a = np.asarray(arr, dtype=np.float32)
            vals.append(a.reshape(-1))
            zeros += int(np.sum(a == 0))
            negs  += int(np.sum(a < 0))

        v = np.concatenate(vals, axis=0) if vals else np.array([], np.float32)
        stats = dict(
            shape=(A, H, W),
            dtype=str(pages[0].dtype),
            min=float(np.min(v)) if v.size else None,
            max=float(np.max(v)) if v.size else None,
            mean=float(np.mean(v)) if v.size else None,
            std=float(np.std(v)) if v.size else None,
            p01=float(np.percentile(v, 1)) if v.size else None,
            p50=float(np.percentile(v, 50)) if v.size else None,
            p99=float(np.percentile(v, 99)) if v.size else None,
            zeros=int(zeros),
            negatives=int(negs),
            sampled=int(v.size),
        )
        return v, stats

def _heuristic_hu_like(stats):
    """非常粗的 HU 判断：仅用于提示，不当真。"""
    if stats["min"] is None: return "unknown"

    mn, mx = stats["min"], stats["max"]
    has_neg = stats["negatives"] > 0
    # HU 常见范围（粗略）：空气 ~ -1000, 软组织 ~ 0±200, 骨头可到 1000~3000+
    if has_neg and mn < -200 and mx > 200:
        return "HU-like (possible)"
    if not has_neg and mn >= 0:
        return "non-HU (likely intensity or log scaled positive)"
    return "unknown"

def _try_log_transform(v):
    """把正数当作强度做 -log，返回简单统计。"""
    v = v[v > 0]  # 只对正值有效
    if v.size == 0:
        return dict(info="no positive values to log", ok=False)
    L = -np.log(np.clip(v, 1e-6, None))
    return dict(
        ok=True,
        min=float(L.min()),
        max=float(L.max()),
        mean=float(L.mean()),
        std=float(L.std()),
        p01=float(np.percentile(L, 1)),
        p50=float(np.percentile(L, 50)),
        p99=float(np.percentile(L, 99)),
    )

def main():
    args = parse_args()

    # 发现 ID
    ids = args.ids
    if not ids:
        ids = _discover_ids(args.root_noisy, args.root_clean)[:args.max_ids]
        if not ids:
            print("[ERR] 在目录中没有发现匹配的 tiff：projectionNoisy*.tif / projection*.tif")
            return
    print(f"[CHECK] IDs = {ids}")

    for id_ in ids:
        noisy_path = os.path.join(args.root_noisy, f"projectionNoisy{id_}.tif")
        clean_path = os.path.join(args.root_clean, f"projection{id_}.tif")

        print(f"\n===== ID {id_} =====")
        if not os.path.isfile(noisy_path):
            print(f"[MISS] {noisy_path}")
            continue
        if not os.path.isfile(clean_path):
            print(f"[MISS] {clean_path}")
            continue

        # 1) Noisy
        noisy_vals, noisy_stats = _read_tiff_stats(noisy_path, sample_frames=args.sample_frames)
        print("[NOISY] stats:", json.dumps(noisy_stats, ensure_ascii=False))
        print("[NOISY] HU guess:", _heuristic_hu_like(noisy_stats))
        log_noisy = _try_log_transform(noisy_vals)
        if log_noisy.get("ok", False):
            print("[NOISY] if -log(intensity):", json.dumps(log_noisy, ensure_ascii=False))
        else:
            print("[NOISY] -log check:", log_noisy.get("info"))

        # 2) Clean (GT)
        clean_vals, clean_stats = _read_tiff_stats(clean_path, sample_frames=args.sample_frames)
        print("[CLEAN] stats:", json.dumps(clean_stats, ensure_ascii=False))
        print("[CLEAN] HU guess:", _heuristic_hu_like(clean_stats))
        log_clean = _try_log_transform(clean_vals)
        if log_clean.get("ok", False):
            print("[CLEAN] if -log(intensity):", json.dumps(log_clean, ensure_ascii=False))
        else:
            print("[CLEAN] -log check:", log_clean.get("info"))

        # 3) 小结：对比 noisy / clean
        #    - 如果两者范围都在 [0, MAX] 且无负值，多半不是 HU。
        #    - 若 -log 后分布更“像”（例如标准差变大且范围合理），说明原始是强度域。
        if (noisy_stats["negatives"] == 0 and clean_stats["negatives"] == 0):
            print("[HINT] 两个投影都无负值，极可能不在 HU；更像 强度域 或 归一化强度。")
        if log_noisy.get("ok", False) and log_clean.get("ok", False):
            print("[HINT] -log 后范围/对比更合理的话，FDK 应该以 log(线积分) 作为输入。")

    print("\n[OK] done.")

if __name__ == "__main__":
    main()


