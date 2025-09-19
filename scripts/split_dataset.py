# -*- coding: utf-8 -*-
"""
split_dataset.py
划分 projection 数据集 ID 为 train/val/test 三个 txt 文件
"""
import os, glob, re, random

ROOT_NOISY = "/data/shared_folder/projectionsNoisy"
ROOT_CLEAN = "/data/shared_folder/projectionsNoisy"
OUT_DIR = "splits"
os.makedirs(OUT_DIR, exist_ok=True)

# 扫描所有可用 ID（取 noisy 和 clean 的交集）
def discover_ids():
    def parse_ids(root, prefix):
        patt = os.path.join(root, f"{prefix}*.tif")
        ids = set()
        for p in glob.glob(patt):
            fn = os.path.basename(p)
            m = re.match(rf"^{re.escape(prefix)}(\d+)\.tif$", fn)
            if m:
                ids.add(int(m.group(1)))
        return ids
    return sorted(parse_ids(ROOT_NOISY, "projectionNoisy") &
                  parse_ids(ROOT_CLEAN, "projection"))

ids = discover_ids()
print(f"[INFO] 总共发现 {len(ids)} 个样本 ID: {ids[:10]}...")

# 手动指定 test ids 
test_ids = [0, 10, 11]
remain = [i for i in ids if i not in test_ids]

# 随机划分 train/val
random.seed(42)
random.shuffle(remain)
n_val = max(1, int(0.05 * len(remain)))  # 10% val
val_ids = remain[:n_val]
train_ids = remain[n_val:]


def write_ids(fname, id_list):
    with open(fname, "w") as f:
        for i in sorted(id_list):
            f.write(f"{i}\n")
    print(f"[OK] 写入 {fname}, 共 {len(id_list)} 个 ID")

write_ids(os.path.join(OUT_DIR, "train.txt"), train_ids)
write_ids(os.path.join(OUT_DIR, "val.txt"),   val_ids)
write_ids(os.path.join(OUT_DIR, "test.txt"),  test_ids)
