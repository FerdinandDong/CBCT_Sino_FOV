# -*- coding: utf-8 -*-
"""
解析 CBCT_Sino_FOV/logs/train_xxx.log，输出：
1) CSV: epoch_summary.csv / step_samples.csv
2) PNG: train_val_loss.png / psnr.png / ssim.png / lr.png / generalization_gap.png
3) 控制台打印：过拟合检查（EMA 平滑 + 幅度阈值 + PSNR 连降）

用法：
python scripts/plot_train_log.py --log logs/train_pconv.log \
                                 --outdir outputs/logplots/pconv
"""

import os, re, argparse, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- 正则：适配日志格式 ----------------
RE_STEP = re.compile(
    r'^\[Epoch\s+(?P<ep>\d+)\s*\|\s*Step\s+(?P<st>\d+)\s*/\s*(?P<tot>\d+)\]\s*loss=(?P<loss>[\d\.]+)\s*lr=(?P<lr>[\deE\+\-\.]+)'
)
RE_EPOCH_MEAN = re.compile(
    r'^Epoch\s+(?P<ep>\d+)\s+mean loss:\s+(?P<loss>[\d\.]+)\s*\|\s*lr=(?P<lr>[\deE\+\-\.]+)'
)
RE_VAL = re.compile(
    r'^\[VAL\s+(?P<ep>\d+)\]\s*loss=(?P<loss>[\d\.]+)\s*PSNR=(?P<psnr>[\d\.]+)\s*dB\s*SSIM=(?P<ssim>[\d\.]+)'
)
RE_CKPT_BEST = re.compile(
    r'^\[CKPT\]\s*saved BEST.*metric=(?P<metric>[\d\.]+)\s*@\s*epoch\s+(?P<ep>\d+)'
)

# ---------------- 解析日志 ----------------
def parse_log(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    step_rows = []  # 采样若干 step 点用于平滑画图
    steps_per_epoch = defaultdict(list)

    epochs = {}      # ep -> dict(train_mean, lr, val_loss, psnr, ssim, is_best)
    best_eps = set()

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            ln = ln.strip()

            m = RE_STEP.match(ln)
            if m:
                ep = int(m['ep'])
                st = int(m['st'])
                tot = int(m['tot'])
                loss = float(m['loss'])
                lr = float(m['lr'])
                if (st % 200) == 0 or st == 1 or st == tot:
                    step_rows.append((ep, st, tot, loss, lr))
                steps_per_epoch[ep].append(loss)
                continue

            m = RE_EPOCH_MEAN.match(ln)
            if m:
                ep = int(m['ep'])
                loss = float(m['loss'])
                lr = float(m['lr'])
                epochs.setdefault(ep, {})
                epochs[ep]['train_mean'] = loss
                epochs[ep]['lr'] = lr
                continue

            m = RE_VAL.match(ln)
            if m:
                ep = int(m['ep'])
                vloss = float(m['loss'])
                psnr = float(m['psnr'])
                ssim = float(m['ssim'])
                epochs.setdefault(ep, {})
                epochs[ep]['val_loss'] = vloss
                epochs[ep]['val_psnr'] = psnr
                epochs[ep]['val_ssim'] = ssim
                continue

            m = RE_CKPT_BEST.match(ln)
            if m:
                ep = int(m['ep'])
                best_eps.add(ep)
                epochs.setdefault(ep, {})
                epochs[ep]['best_metric'] = float(m['metric'])
                epochs[ep]['is_best'] = True
                continue

    # 若缺少 epoch 均值，用 step 的均值补
    for ep, lst in steps_per_epoch.items():
        if ep not in epochs:
            epochs[ep] = {}
        if 'train_mean' not in epochs[ep] and len(lst) > 0:
            epochs[ep]['train_mean'] = float(np.mean(lst))

    # 排序输出
    eps = sorted(epochs.keys())
    E = []
    for ep in eps:
        r = epochs[ep]
        E.append({
            'epoch': ep,
            'train_mean': float(r.get('train_mean', np.nan)),
            'val_loss': float(r.get('val_loss', np.nan)),
            'val_psnr': float(r.get('val_psnr', np.nan)),
            'val_ssim': float(r.get('val_ssim', np.nan)),
            'lr': float(r.get('lr', np.nan)),
            'is_best': bool(r.get('is_best', False)),
            'best_metric': float(r.get('best_metric', np.nan)),
        })
    return E, step_rows

# ---------------- 保存 CSV ----------------
def save_csv(E, step_rows, outdir):
    os.makedirs(outdir, exist_ok=True)
    ep_csv = os.path.join(outdir, 'epoch_summary.csv')
    with open(ep_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_mean', 'val_loss', 'val_psnr', 'val_ssim', 'lr', 'is_best', 'best_metric'])
        for r in E:
            w.writerow([r['epoch'], r['train_mean'], r['val_loss'], r['val_psnr'], r['val_ssim'], r['lr'], int(r['is_best']), r['best_metric']])
    st_csv = os.path.join(outdir, 'step_samples.csv')
    with open(st_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'step', 'steps_total', 'loss', 'lr'])
        for ep, st, tot, loss, lr in step_rows:
            w.writerow([ep, st, tot, loss, lr])
    print(f"[OK] CSV -> {ep_csv}\n[OK] CSV -> {st_csv}")

# ---------------- 画图 ----------------
def plot_series(E, outdir):
    os.makedirs(outdir, exist_ok=True)
    epoch = np.array([r['epoch'] for r in E], dtype=np.int32)
    train = np.array([r['train_mean'] for r in E], dtype=np.float64)
    vloss = np.array([r['val_loss'] for r in E], dtype=np.float64)
    psnr = np.array([r['val_psnr'] for r in E], dtype=np.float64)
    ssim = np.array([r['val_ssim'] for r in E], dtype=np.float64)
    lr   = np.array([r['lr'] for r in E], dtype=np.float64)
    is_best = np.array([int(r['is_best']) for r in E], dtype=np.int32)

    # 1) Train / Val Loss（纵轴强制 0-3）
    plt.figure(figsize=(8,5))
    plt.plot(epoch, train, label='train_mean loss')
    plt.plot(epoch, vloss, label='val loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Train / Val Loss')
    plt.ylim(0, 2.0)  # 范围
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'train_val_loss.png'), dpi=160); plt.close()

    # 2) PSNR 单图
    plt.figure(figsize=(8,5))
    plt.plot(epoch, psnr, label='val PSNR (dB)')
    plt.xlabel('epoch'); plt.ylabel('PSNR (dB)')
    plt.title('Validation PSNR'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'psnr.png'), dpi=160); plt.close()

    # 3) SSIM 单图
    plt.figure(figsize=(8,5))
    plt.plot(epoch, ssim, label='val SSIM')
    plt.xlabel('epoch'); plt.ylabel('SSIM')
    plt.title('Validation SSIM'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'ssim.png'), dpi=160); plt.close()

    # 4) LR
    plt.figure(figsize=(8,4))
    plt.plot(epoch, lr, label='lr')
    plt.xlabel('epoch'); plt.ylabel('lr'); plt.title('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'lr.png'), dpi=160); plt.close()

    # 5) Generalization gap = val - train
    gap = vloss - train
    plt.figure(figsize=(8,4))
    plt.plot(epoch, gap, label='val - train loss')
    plt.axhline(0, linestyle='--')
    if np.any(is_best>0):
        best_epochs = epoch[is_best>0]
        best_gaps = gap[is_best>0]
        plt.scatter(best_epochs, best_gaps, marker='*', s=80, label='BEST ckpt')
    plt.xlabel('epoch'); plt.ylabel('gap'); plt.title('Generalization Gap')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'generalization_gap.png'), dpi=160); plt.close()

    return gap

# ---------------- 过拟合分析（稳健版） ----------------
def analyze_overfitting(E, gap):
    """
    更稳的判定：
      - 对 train/val/psnr 做 EMA 平滑（alpha=0.3）
      - 窗口=5：train 下降≥1% 且 val 上升≥1% 才报警
      - 或 val PSNR 连续下降≥0.2 dB 报警
    """
    def ema(x, alpha=0.3):
        y = np.array(x, dtype=np.float64)
        out = np.zeros_like(y)
        m = 0.0
        for i, v in enumerate(y):
            m = alpha * v + (1 - alpha) * (m if i > 0 else v)
            out[i] = m
        return out

    epoch = np.array([r['epoch'] for r in E], dtype=np.int32)
    train = np.array([r['train_mean'] for r in E], dtype=np.float64)
    vloss = np.array([r['val_loss'] for r in E], dtype=np.float64)
    vpsnr = np.array([r['val_psnr'] for r in E], dtype=np.float64)

    t_s = ema(train); v_s = ema(vloss); p_s = ema(vpsnr)

    win = 5
    alerts = []
    for i in range(len(epoch) - win):
        t0, t1 = t_s[i], t_s[i+win]
        v0, v1 = v_s[i], v_s[i+win]
        if t0 > 0 and v0 > 0:
            t_drop = (t0 - t1) / t0
            v_rise = (v1 - v0) / v0
            if (t_drop >= 0.01) and (v_rise >= 0.01):  # 1% 阈值
                alerts.append((int(epoch[i]), int(epoch[i+win])))

    psnr_alerts = []
    for i in range(len(epoch) - win):
        if p_s[i+win] <= p_s[i] - 0.2:  # 连续窗口 PSNR 下降 ≥ 0.2 dB
            psnr_alerts.append((int(epoch[i]), int(epoch[i+win])))

    report = {
        "trend_alerts": alerts,
        "psnr_alerts": psnr_alerts,
        "gap_last": float(gap[-1]) if len(gap) else float('nan'),
    }
    return report

# ---------------- 主函数 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="训练日志路径，如 logs/train_pconv.log")
    ap.add_argument("--outdir", default=None, help="输出目录（默认根据日志名自动生成）")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join("outputs", "logplots", os.path.splitext(os.path.basename(args.log))[0])
    os.makedirs(outdir, exist_ok=True)

    E, step_rows = parse_log(args.log)
    if not E:
        print("[ERR] 没有解析到任何 epoch 记录，请检查日志格式。")
        return

    save_csv(E, step_rows, outdir)
    gap = plot_series(E, outdir)
    report = analyze_overfitting(E, gap)

    print("\n========== Overfitting Check ==========")
    if report["trend_alerts"]:
        for st, ed in report["trend_alerts"]:
            print(f"[ALERT] 1%级别：train↓ & val↑  epoch {st} → {ed}")
    else:
        print("[INFO] 未触发 1% 级别的 train↓&val↑ 报警。")

    if report["psnr_alerts"]:
        for st, ed in report["psnr_alerts"]:
            print(f"[ALERT] PSNR 连降≥0.2 dB  epoch {st} → {ed}")
    else:
        print("[INFO] 未出现 PSNR 连续显著下降。")

    best_list = [r for r in E if r['is_best']]
    if best_list:
        last_best = best_list[-1]
        print(f"[BEST] epoch={last_best['epoch']}  metric={last_best['best_metric']:.4f}  "
              f"val_loss={last_best['val_loss']:.4f}  psnr={last_best['val_psnr']:.3f}dB  ssim={last_best['val_ssim']:.4f}")
    print(f"[OK] 图表与 CSV 已保存到：{outdir}")

if __name__ == "__main__":
    main()
