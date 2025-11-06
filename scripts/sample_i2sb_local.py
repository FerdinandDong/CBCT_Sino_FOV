# scripts/sample_i2sb_local.py
import os, argparse, yaml, csv, torch, numpy as np, tifffile as tiff
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, List

from ctprojfix.data.dataset import make_dataloader
from ctprojfix.models.i2sb_unet import I2SBUNet

def load_cfg(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _ceil_to(v, m): return ((v + m - 1) // m) * m if m > 0 else v

def _percentile_norm01(a: np.ndarray, p_lo=1, p_hi=99, eps=1e-6) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.percentile(a, [p_lo, p_hi])
    if hi - lo < eps:
        return np.zeros_like(a, dtype=np.float32)
    y = (a - lo) / (hi - lo + eps)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

def _center_crop_to(arr: Optional[np.ndarray], out_h: int, out_w: int) -> Optional[np.ndarray]:
    if arr is None: return None
    a = arr
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    H, W = a.shape[-2], a.shape[-1]
    if (H, W) == (out_h, out_w):
        return a.astype(np.float32, copy=False)
    top  = max((H - out_h) // 2, 0)
    left = max((W - out_w) // 2, 0)
    return a[top:top+out_h, left:left+out_w].astype(np.float32, copy=False)

def _match_to_pred_size(pred_hw: Tuple[int,int],
                        noisy: Optional[np.ndarray],
                        gt: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    ph, pw = int(pred_hw[0]), int(pred_hw[1])
    return _center_crop_to(noisy, ph, pw), _center_crop_to(gt, ph, pw)

def _save_triptych_and_heat(noisy: np.ndarray,
                            pred: np.ndarray,
                            gt: Optional[np.ndarray],
                            save_dir: str,
                            prefix: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = _percentile_norm01(noisy)
    p = _percentile_norm01(pred)
    g = _percentile_norm01(gt) if gt is not None else n

    os.makedirs(save_dir, exist_ok=True)
    # 三联图
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(n, cmap="gray"); axes[0].set_title("Noisy"); axes[0].axis("off")
    axes[1].imshow(p, cmap="gray"); axes[1].set_title("Pred");  axes[1].axis("off")
    axes[2].imshow(g, cmap="gray"); axes[2].set_title("GT");    axes[2].axis("off")
    plt.tight_layout()
    tpath = os.path.join(save_dir, f"{prefix}.png")
    plt.savefig(tpath, dpi=150, bbox_inches="tight")
    plt.close()

    # 误差热力
    if gt is not None:
        err = np.abs(p - g)
        fig = plt.figure(figsize=(5, 4))
        import matplotlib.pyplot as plt
        plt.imshow(err, cmap="magma")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")
        hpath = os.path.join(save_dir, f"{prefix.replace('triptych', 'heat')}.png")
        plt.savefig(hpath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[FIG] saved: {tpath}, {hpath}")
    else:
        print(f"[FIG] saved: {tpath}")

@torch.no_grad()
def run(cfg_path: str, ckpt: Optional[str] = None, out: Optional[str] = None):
    cfg = load_cfg(cfg_path)
    train_cfg = cfg.get("train", {})
    dev = torch.device(train_cfg.get("device","cuda") if torch.cuda.is_available() else "cpu")

    # --------- dataloader（尊重 sample.split；忽略 exclude_ids）---------
    data_cfg = dict(cfg["data"])
    data_cfg.pop("exclude_ids", None)
    data_cfg["shuffle"] = False
    dl = make_dataloader(data_cfg)
    if isinstance(dl, dict):
        want = str(cfg.get("sample", {}).get("split", "test")).lower()
        loader = dl.get(want) or dl.get("test") or dl.get("val") or dl.get("train")
        if loader is None:
            raise RuntimeError("All splits empty. Check splits/*.txt and data paths.")
        try: n = len(loader.dataset)
        except Exception: n = 0
        print(f"[I2SB local] using split='{want}' size={n}")
    else:
        loader = dl
        try: n = len(loader.dataset)
        except Exception: n = 0
        print(f"[I2SB local] single loader size={n}")
        if n <= 0:
            raise RuntimeError("DataLoader is empty.")

    # --------- model & ckpt ----------
    mp = cfg["model"]["params"]
    net = I2SBUNet(in_ch=mp.get("in_ch",4), base=mp.get("base",64), depth=mp.get("depth",4),
                   emb_dim=mp.get("emb_dim",256), dropout=mp.get("dropout",0.0)).to(dev)
    ckpt_path = ckpt
    if ckpt_path is None:
        ckpt_dir = train_cfg.get("ckpt_dir", "checkpoints/i2sb_local")
        ckpt_prefix = train_cfg.get("ckpt_prefix", "i2sb_local")
        ckpt_path = cfg.get("sample", {}).get("ckpt", os.path.join(ckpt_dir, f"{ckpt_prefix}_best.pth"))
    if ckpt_path and os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        state = ck.get("state_dict", ck)
        net.load_state_dict(state, strict=True)
        print(f"[OK] load ckpt: {ckpt_path}")
    else:
        print(f"[WARN] ckpt not found: {ckpt_path} → use random init")
    net.eval()

    # --------- out dir ----------
    out_dir = out or cfg.get("sample", {}).get("out_dir", "outputs/diffusion/i2sb_local_1step")
    os.makedirs(out_dir, exist_ok=True)

    pick_id    = cfg.get("sample", {}).get("pick_id", None)
    pick_angle = cfg.get("sample", {}).get("pick_angle", None)

    # ------ helper: 前向并裁回原尺寸 ------
    def forward_and_restore(x1: torch.Tensor, depth_down:int, H:int, W:int) -> torch.Tensor:
        n_down = depth_down
        factor = 2 ** max(n_down, 0)
        Ht, Wt = _ceil_to(H, factor), _ceil_to(W, factor)
        if (Ht, Wt) != (H, W):
            pad = (0, Wt - W, 0, Ht - H)
            x1 = F.pad(x1, pad, mode="reflect")
        t_map = torch.ones((x1.shape[0],1,Ht,Wt), device=dev, dtype=torch.float32)
        xin = torch.cat([x1, t_map], dim=1)
        x0_hat = net(xin)
        # 裁回 H,W 并兜底插值
        if x0_hat.shape[-2] >= Ht and x0_hat.shape[-1] >= Wt:
            x0_hat = x0_hat[..., :Ht, :Wt]
        if x0_hat.shape[-2] >= H and x0_hat.shape[-1] >= W:
            x0_hat = x0_hat[..., :H, :W]
        if x0_hat.shape[-2:] != (H, W):
            x0_hat = F.interpolate(x0_hat, size=(H, W), mode="bilinear", align_corners=False)
        return x0_hat

    depth_down = len(getattr(net, "downs", []))

    # ---------- pick_id 模式：导出整卷 + 单张三联图 ----------
    if pick_id is not None:
        pick_id = int(pick_id)
        print(f"[I2SB local] single-id volume mode: pick_id={pick_id}, pick_angle={pick_angle}")

        preds, gts, noisies = [], [], []
        angle_rows: List[Tuple[int,int]] = []  # (angle, A)
        best_one = None  # (diff, rec_dict)

        for batch in loader:
            # 解析 id 和 angle/A
            bid = batch.get("id", None)
            if bid is None: continue
            if torch.is_tensor(bid): ids = bid.detach().cpu().numpy().tolist()
            elif isinstance(bid, (list, tuple, np.ndarray)): ids = list(bid)
            else: ids = [bid]

            angles = batch.get("angle", None)
            if torch.is_tensor(angles): angles_np = angles.detach().cpu().numpy().tolist()
            elif isinstance(angles, (list, tuple, np.ndarray)): angles_np = list(angles)
            else: angles_np = [None] * len(ids)

            A_arr = batch.get("A", None)
            if torch.is_tensor(A_arr): A_np = A_arr.detach().cpu().numpy().tolist()
            elif isinstance(A_arr, (list, tuple, np.ndarray)): A_np = list(A_arr)
            else: A_np = [360] * len(ids)

            # 找到本 batch 中所有属于 pick_id 的索引
            hit = [i for i, v in enumerate(ids) if int(v) == pick_id]
            if not hit: continue

            # 前向
            x1_full = batch["inp"].to(dev).float() * 2.0 - 1.0  # (B, Cx1, H, W)
            B, _, H, W = x1_full.shape
            x0_hat = forward_and_restore(x1_full, depth_down, H, W)  # (B,1,H,W)
            x0 = ((x0_hat.clamp(-1,1) + 1.0) * 0.5).cpu().numpy()     # (B,1,H,W)
            noisy = batch["inp"][:, 0:1].cpu().numpy()                # (B,1,H,W)
            has_gt = ("gt" in batch)
            gt_np = batch["gt"].cpu().numpy() if has_gt else None

            # 收集该 id 的所有帧（整卷）
            for i in hit:
                pred_i = np.squeeze(x0[i,0]).astype(np.float32)
                noz_i  = np.squeeze(noisy[i,0]).astype(np.float32)
                gt_i   = (np.squeeze(gt_np[i,0]).astype(np.float32) if has_gt else None)

                preds.append(pred_i)
                noisies.append(noz_i)
                if has_gt: gts.append(gt_i)

                ai = angles_np[i] if angles_np[i] is not None else -1
                Ai = A_np[i] if A_np[i] is not None else 360
                try:
                    ai = int(ai)
                except Exception:
                    ai = -1
                try:
                    Ai = int(Ai)
                except Exception:
                    Ai = 360
                angle_rows.append((ai, Ai))

                # 供三联图挑选最接近 pick_angle 的帧
                if pick_angle is not None:
                    if isinstance(pick_angle, str) and pick_angle.lower() == "mid":
                        diff = 0  # mid 另行处理，通过最后取中位
                    else:
                        want = int(pick_angle)
                        diff = abs(ai - want) if ai >= 0 else 10**9
                    rec = dict(pred=pred_i, noisy=noz_i, gt=gt_i, angle=ai)
                    if (best_one is None) or (diff < best_one[0]):
                        best_one = (diff, rec)

        # 检查是否收集到任何帧
        if len(preds) == 0:
            raise RuntimeError(f"[I2SB local] pick_id={pick_id} not found in split or produced 0 frames.")

        # 角度排序（按 angle 升序，未知角度置后）
        order = list(range(len(preds)))
        order.sort(key=lambda k: (angle_rows[k][0] < 0, angle_rows[k][0]))
        preds   = [preds[k]   for k in order]
        noisies = [noisies[k] for k in order]
        if len(gts) > 0:
            gts = [gts[k] for k in order]
        angle_rows = [angle_rows[k] for k in order]

        # 导出整卷
        vol_pred = np.stack(preds,   axis=0).astype(np.float32)
        vol_nozy = np.stack(noisies, axis=0).astype(np.float32)
        tiff.imwrite(os.path.join(out_dir, "pred_volume.tiff"),  vol_pred, imagej=True)
        tiff.imwrite(os.path.join(out_dir, "noisy_volume.tiff"), vol_nozy, imagej=True)
        if len(gts) > 0:
            vol_gt = np.stack(gts, axis=0).astype(np.float32)
            tiff.imwrite(os.path.join(out_dir, "gt_volume.tiff"), vol_gt, imagej=True)
        else:
            vol_gt = None
        print(f"[I2SB local][OK] saved volumes -> {out_dir}")

        # 写出 metrics.csv（angle 与 A）
        csv_path = os.path.join(out_dir, "metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["angle", "A"])
            for ai, Ai in angle_rows:
                w.writerow([ai, Ai])
        print(f"[I2SB local][OK] saved angles -> {csv_path}")

        # 选一张保存三联图 + 热力图
        if pick_angle is not None:
            if isinstance(pick_angle, str) and pick_angle.lower() == "mid":
                mid_idx = len(preds) // 2
                rec = dict(
                    pred=preds[mid_idx],
                    noisy=noisies[mid_idx],
                    gt=(gts[mid_idx] if len(gts) > 0 else None),
                    angle=angle_rows[mid_idx][0]
                )
            else:
                # best_one 在采样循环内已记录（按角度最近）
                if best_one is None:
                    # 如果全部角度未知，退化为中位
                    mid_idx = len(preds) // 2
                    rec = dict(
                        pred=preds[mid_idx],
                        noisy=noisies[mid_idx],
                        gt=(gts[mid_idx] if len(gts) > 0 else None),
                        angle=angle_rows[mid_idx][0]
                    )
                else:
                    rec = best_one[1]

            ph, pw = rec["pred"].shape[-2], rec["pred"].shape[-1]
            nn, gg = _match_to_pred_size((ph, pw), rec["noisy"], rec["gt"])
            a = int(rec["angle"]) if rec["angle"] is not None else -1
            a3 = f"{max(a,0):03d}"
            prefix = f"triptych_id{pick_id}_a{a3}"
            _save_triptych_and_heat(nn, rec["pred"], gg, out_dir, prefix)

        print(f"[I2SB local][OK] single-id volume ready at: {out_dir}")
        return

    # ---------- 否则：整卷导出（全 split） ----------
    preds = []
    for batch in loader:
        x1 = batch["inp"].to(dev).float() * 2.0 - 1.0
        B, _, H, W = x1.shape
        x0_hat = forward_and_restore(x1, depth_down, H, W)
        x0 = ((x0_hat.clamp(-1,1) + 1.0) * 0.5).cpu().numpy()
        preds += [np.squeeze(x0[i,0]).astype(np.float32) for i in range(B)]

    if len(preds) == 0:
        raise RuntimeError("[I2SB local] no frames predicted; empty split?")
    vol = np.stack(preds, axis=0).astype(np.float32)
    out_path = os.path.join(out_dir, "pred_volume.tiff")
    tiff.imwrite(out_path, vol, imagej=True)
    print("[I2SB local][OK] saved:", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/sample_i2sb_local.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args.cfg, ckpt=args.ckpt, out=args.out)

if __name__ == "__main__":
    main()
