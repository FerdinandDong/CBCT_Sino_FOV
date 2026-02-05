# scripts/sample_i2sb_local_multi.py
# -*- coding: utf-8 -*-
import os, argparse, yaml, csv, math
import torch, numpy as np
import tifffile as tiff
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from tqdm import tqdm

# run: python -m scripts.sample_i2sb_local_multi
from ctprojfix.data.dataset import make_dataloader
from ctprojfix.models.i2sb_unet import I2SBUNet


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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

def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def _save_tile_gray01(img01: np.ndarray, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img01, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def _save_tile_heat(err: np.ndarray, out_path: str, cmap: str, with_cb: bool):
    _ensure_dir(os.path.dirname(out_path))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if with_cb:
        fig = plt.figure(figsize=(4.4, 4), dpi=200)
        ax = fig.add_axes([0.0, 0.0, 0.9, 1.0])
        im = ax.imshow(err, cmap=cmap, vmin=0.0, vmax=None)
        ax.axis("off")
        cax = fig.add_axes([0.91, 0.1, 0.03, 0.8])
        plt.colorbar(im, cax=cax)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(4, 4), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(err, cmap=cmap, vmin=0.0, vmax=None)
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

def save_quad_and_tiles(noisy01: np.ndarray,
                        pred01: np.ndarray,
                        gt01: Optional[np.ndarray],
                        out_dir: str,
                        name: str,
                        tiles_cfg: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(out_dir)
    fig_dir = os.path.join(out_dir, "figs")
    _ensure_dir(fig_dir)

    n = np.clip(noisy01.astype(np.float32), 0, 1)
    p = np.clip(pred01.astype(np.float32),  0, 1)
    g = np.clip(gt01.astype(np.float32),    0, 1) if gt01 is not None else None

    if g is not None:
        err = np.abs(p - g)
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        axes[0].imshow(n, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Noisy"); axes[0].axis("off")
        axes[1].imshow(p, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Pred (Multi)");  axes[1].axis("off")
        axes[2].imshow(g, cmap="gray", vmin=0, vmax=1); axes[2].set_title("GT");    axes[2].axis("off")
        im = axes[3].imshow(err, cmap=str((tiles_cfg or {}).get("cmap","magma")), vmin=0.0, vmax=None)
        axes[3].set_title("|Pred-GT|"); axes[3].axis("off")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        plt.tight_layout()
        out_path = os.path.join(fig_dir, f"{name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[FIG] {out_path}")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(n, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Noisy"); axes[0].axis("off")
        axes[1].imshow(p, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Pred (Multi)");  axes[1].axis("off")
        plt.tight_layout()
        out_path = os.path.join(fig_dir, f"{name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[FIG] {out_path}")
        return

    if not bool((tiles_cfg or {}).get("enable", False)):
        return
    tiles_dir = (tiles_cfg or {}).get("dir", None)
    if not tiles_dir:
        tiles_dir = os.path.join(fig_dir, f"{name}_tiles")
    _ensure_dir(tiles_dir)

    prefix = str((tiles_cfg or {}).get("prefix", ""))
    fmt = str((tiles_cfg or {}).get("format", "png"))
    cmap = str((tiles_cfg or {}).get("cmap", "magma"))
    with_cb = bool((tiles_cfg or {}).get("heat_with_colorbar", False))

    _save_tile_gray01(n, os.path.join(tiles_dir, f"{prefix}Noisy.{fmt}"))
    _save_tile_gray01(p, os.path.join(tiles_dir, f"{prefix}Pred.{fmt}"))
    _save_tile_gray01(g, os.path.join(tiles_dir, f"{prefix}GT.{fmt}"))
    _save_tile_heat(np.abs(p-g), os.path.join(tiles_dir, f"{prefix}AbsDiff_Pred_vs_GT.{fmt}"),
                    cmap=cmap, with_cb=with_cb)
    print(f"[TILE] -> {tiles_dir}")

def _psnr01(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    mse = float(np.mean((x - y) ** 2))
    if mse < eps: return 99.0
    return 20.0 * float(np.log10(1.0 / np.sqrt(mse)))

def _ssim2d(x: np.ndarray, y: np.ndarray, dr: float = 1.0,
            K1: float = 0.01, K2: float = 0.03, sigma: float = 1.5) -> float:
    from scipy.ndimage import gaussian_filter
    x = np.clip(x, 0, dr).astype(np.float32)
    y = np.clip(y, 0, dr).astype(np.float32)
    C1, C2 = (K1*dr)**2, (K2*dr)**2
    mu1, mu2 = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
    mu1s, mu2s, mu12 = mu1*mu1, mu2*mu2, mu1*mu2
    s1 = gaussian_filter(x*x, sigma) - mu1s
    s2 = gaussian_filter(y*y, sigma) - mu2s
    s12 = gaussian_filter(x*y, sigma) - mu12
    num = (2*mu12 + C1) * (2*s12 + C2)
    den = (mu1s + mu2s + C1) * (s1 + s2 + C2) + 1e-12
    return float(np.mean(num / den))

def _get_scalar_from_batch(batch: Dict[str, Any], key: str, idx: int, default=None):
    if key not in batch: return default
    v = batch[key]
    if isinstance(v, torch.Tensor):
        if v.ndim == 0: return v.item()
        if v.ndim >= 1 and v.shape[0] > idx: return v[idx].item()
        return default
    if isinstance(v, (list, tuple)):
        try: return v[idx]
        except Exception: return v[0] if len(v) > 0 else default
    if isinstance(v, np.ndarray):
        if v.ndim == 0: return float(v)
        if v.ndim >= 1 and v.shape[0] > idx: return v[idx]
        return default
    return v


# -----------------------------------------------------------------------------
# Core: deterministic multi-step bridge sampling
# net input is 5ch: [xt, x1_img, mask, angle, t_map]
# add_angle_channel == True 恒成立 => inp has 3 channels: [noisy, mask, angle] in [0,1]
# -----------------------------------------------------------------------------

def _pad_inputs_for_unet(x1_img_m1: torch.Tensor,
                         mask01: torch.Tensor,
                         angle01: torch.Tensor,
                         factor: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
    Pad to multiples of factor.
    - image: reflect
    - mask:  constant 0
    - angle: replicate
    """
    B, _, H, W = x1_img_m1.shape
    Ht = _ceil_to(H, factor)
    Wt = _ceil_to(W, factor)
    if (Ht, Wt) == (H, W):
        return x1_img_m1, mask01, angle01, Ht, Wt

    pad = (0, Wt - W, 0, Ht - H)
    x1_img_m1 = F.pad(x1_img_m1, pad, mode="reflect")
    mask01    = F.pad(mask01,    pad, mode="constant", value=0.0)
    angle01   = F.pad(angle01,   pad, mode="replicate")
    return x1_img_m1, mask01, angle01, Ht, Wt


def sample_multi_step(net: torch.nn.Module,
                      x1_img_m1: torch.Tensor,   # (B,1,H,W) in [-1,1]
                      mask01: torch.Tensor,      # (B,1,H,W) in [0,1]
                      angle01: torch.Tensor,     # (B,1,H,W) in [0,1]
                      depth_down: int,
                      steps: int,
                      sigma_T: float) -> torch.Tensor:
    """
    Deterministic multi-step sampling with shared epsilon:
      x_t = (1-t)x0 + t*x1 + sigma_T*sqrt(t(1-t))*eps
    Start from t=1: x_1 = x1
    Update towards t=0 using predicted x0.
    Return x0_hat in [-1,1] (same H,W as input).
    """
    device = x1_img_m1.device
    B, _, H, W = x1_img_m1.shape

    factor = 2 ** max(depth_down, 0)
    x1p, mp, ap, Ht, Wt = _pad_inputs_for_unet(x1_img_m1, mask01, angle01, factor)

    xt = x1p.clone()  # init at t=1
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    eps = 1e-6
    for k in range(steps):
        t = float(ts[k].item())
        s = float(ts[k + 1].item())

        t_map = torch.full((B, 1, Ht, Wt), t, device=device, dtype=torch.float32)

        # 5ch: [xt, x1_img, mask, angle, t_map]
        net_in = torch.cat([xt, x1p, mp, ap, t_map], dim=1)
        x0_hat = net(net_in)  # (B,1,Ht,Wt) in [-1,1]

        if x0_hat.shape[-2:] != (Ht, Wt):
            x0_hat = F.interpolate(x0_hat, size=(Ht, Wt), mode="bilinear", align_corners=False)

        sig_t = sigma_T * math.sqrt(max(t * (1.0 - t), 0.0))
        sig_s = sigma_T * math.sqrt(max(s * (1.0 - s), 0.0))

        if sig_t < 1e-8:
            eps_hat = torch.zeros_like(xt)
        else:
            eps_hat = (xt - (1.0 - t) * x0_hat - t * x1p) / (sig_t + eps)

        xt = (1.0 - s) * x0_hat + s * x1p + sig_s * eps_hat

    xt = xt[..., :H, :W]
    return xt


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@torch.no_grad()
def run(cfg_path: str, ckpt: Optional[str] = None, out: Optional[str] = None):
    cfg = load_cfg(cfg_path)
    train_cfg = cfg.get("train", {})
    dev = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # 1) DataLoader
    data_cfg = dict(cfg["data"])
    data_cfg["shuffle"] = False
    dl = make_dataloader(data_cfg)

    want = str(cfg.get("sample", {}).get("split", "") or "").lower().strip()
    if isinstance(dl, dict):
        if not want:
            raise RuntimeError("[I2SB Multi] make_dataloader returned split dict, but sample.split not set.")
        if want not in dl or dl[want] is None:
            raise RuntimeError(f"[I2SB Multi] split '{want}' not found.")
        loader = dl[want]
        print(f"[I2SB Multi] using split='{want}' size={len(loader.dataset)}")
    else:
        loader = dl
        print(f"[I2SB Multi] single loader size={len(loader.dataset)}")
        if len(loader.dataset) <= 0:
            raise RuntimeError("DataLoader is empty.")

    # 2) Model & Checkpoint
    mp = cfg["model"]["params"]

    in_ch = int(mp.get("in_ch", 5))
    if in_ch != 5:
        print(f"[WARN] model.in_ch={in_ch}, but sampling expects 5ch input. For add_angle_channel=True it MUST be 5.")
    net = I2SBUNet(
        in_ch=in_ch,
        base=int(mp.get("base", 64)),
        depth=int(mp.get("depth", 4)),
        emb_dim=int(mp.get("emb_dim", 256)),
        dropout=float(mp.get("dropout", 0.0))
    ).to(dev)

    ckpt_path = ckpt
    if ckpt_path is None:
        ckpt_dir = train_cfg.get("ckpt_dir", "checkpoints/i2sb_local_multi")
        ckpt_prefix = train_cfg.get("ckpt_prefix", "i2sb_local_multi")
        ckpt_path = cfg.get("sample", {}).get("ckpt", os.path.join(ckpt_dir, f"{ckpt_prefix}_best.pth"))

    if ckpt_path and os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        state = ck.get("state_dict", ck)
        net.load_state_dict(state, strict=True)
        print(f"[OK] load ckpt: {ckpt_path}")
    else:
        print(f"[WARN] ckpt not found: {ckpt_path} !!! Result might be random noise.")
    net.eval()

    # depth_down
    net_core = net.module if hasattr(net, "module") else net
    depth_down = len(getattr(net_core, "downs", []))

    # 3) Output Dir & Params
    out_dir = out or cfg.get("sample", {}).get("out_dir", "outputs/i2sb_local_multi")
    os.makedirs(out_dir, exist_ok=True)

    pick_id    = cfg.get("sample", {}).get("pick_id", None)
    pick_angle = cfg.get("sample", {}).get("pick_angle", None)

    steps = int(cfg.get("sample", {}).get("steps", 10))
    sigma_T = float(cfg.get("sample", {}).get("sigma_T", cfg.get("train", {}).get("sigma_T", 1.0)))

    # !!!!!!!!强制有效区保持原输入（切割中间fov调试用；默认 False）!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    blend_valid = bool(cfg.get("sample", {}).get("blend_valid", False))

    print(f"[I2SB Multi] Inference using K={steps} | sigma_T={sigma_T} | blend_valid={blend_valid}")

    # 4) Inference
    if pick_id is not None:
        pick_id = int(pick_id)
        print(f"[I2SB Multi] Processing single volume ID={pick_id}...")

        preds, gts, noisies = [], [], []
        preds_blend = []  # optional
        preds_raw, gts_raw, noisies_raw = [], [], []
        id_rows, angle_rows, A_rows = [], [], []

        noisy_lo_rows, noisy_hi_rows = [], []
        gt_lo_rows, gt_hi_rows = [], []
        psnr_rows, ssim_rows = [], []

        best_one = None

        pbar = tqdm(loader, desc=f"ID={pick_id}", ncols=100)
        for batch in pbar:
            bid = batch.get("id", None)
            if torch.is_tensor(bid): ids = bid.detach().cpu().numpy().tolist()
            elif isinstance(bid, (list, tuple, np.ndarray)): ids = list(bid)
            else: ids = [bid]

            hit = [i for i, v in enumerate(ids) if int(v) == pick_id]
            if not hit:
                continue

            inp = batch["inp"].to(dev).float()  # (B,3,H,W) in [0,1] => [noisy, mask, angle]
            if inp.shape[1] < 3:
                raise RuntimeError("add_angle_channel must be true: inp should be 3ch [noisy, mask, angle].")

            x1_img01 = inp[:, 0:1, ...]
            mask01   = inp[:, 1:2, ...].clamp(0.0, 1.0)
            angle01  = inp[:, 2:3, ...].clamp(0.0, 1.0)

            x1_img_m1 = x1_img01 * 2.0 - 1.0

            x0_hat_m1 = sample_multi_step(
                net=net,
                x1_img_m1=x1_img_m1,
                mask01=mask01,
                angle01=angle01,
                depth_down=depth_down,
                steps=steps,
                sigma_T=sigma_T
            )

            pred01 = ((x0_hat_m1.clamp(-1, 1) + 1.0) * 0.5)  # (B,1,H,W) [0,1]

            # optional: keep valid region from input x1 (debug/visual)
            if blend_valid:
                pred01_blend = pred01 * (1.0 - mask01) + x1_img01 * mask01
            else:
                pred01_blend = None

            pred_np = pred01.detach().cpu().numpy()
            noisy_np = x1_img01.detach().cpu().numpy()
            pred_bl_np = pred01_blend.detach().cpu().numpy() if pred01_blend is not None else None

            has_gt = ("gt" in batch)
            gt_np = batch["gt"].cpu().numpy() if has_gt else None

            angles = batch.get("angle", [None] * inp.shape[0])
            if torch.is_tensor(angles): angles = angles.cpu().numpy().tolist()
            A_arr = batch.get("A", [None] * inp.shape[0])
            if torch.is_tensor(A_arr): A_arr = A_arr.cpu().numpy().tolist()

            for i in hit:
                pred_i = np.squeeze(pred_np[i, 0])
                predb_i = np.squeeze(pred_bl_np[i, 0]) if pred_bl_np is not None else None
                noz_i  = np.squeeze(noisy_np[i, 0])
                gt_i   = np.squeeze(gt_np[i, 0]) if has_gt else None

                preds.append(pred_i)
                noisies.append(noz_i)
                if predb_i is not None:
                    preds_blend.append(predb_i)
                if has_gt:
                    gts.append(gt_i)

                ai = angles[i] if angles[i] is not None else -1
                Ai = A_arr[i] if A_arr[i] is not None else 360
                angle_rows.append(ai)
                A_rows.append(Ai)
                id_rows.append(pick_id)

                nlo = _get_scalar_from_batch(batch, "noisy_lo", i)
                nhi = _get_scalar_from_batch(batch, "noisy_hi", i)
                glo = _get_scalar_from_batch(batch, "gt_lo", i)
                ghi = _get_scalar_from_batch(batch, "gt_hi", i)
                noisy_lo_rows.append(nlo); noisy_hi_rows.append(nhi)
                gt_lo_rows.append(glo); gt_hi_rows.append(ghi)

                if (glo is not None) and (ghi is not None) and (gt_i is not None):
                    pr_raw = pred_i * (ghi - glo) + glo
                    g_raw  = gt_i   * (ghi - glo) + glo
                else:
                    pr_raw, g_raw = None, None

                if (nlo is not None) and (nhi is not None):
                    n_raw = noz_i * (nhi - nlo) + nlo
                else:
                    n_raw = None

                preds_raw.append(pr_raw)
                gts_raw.append(g_raw)
                noisies_raw.append(n_raw)

                if has_gt and gt_i is not None:
                    ps = _psnr01(pred_i, gt_i)
                    ss = _ssim2d(pred_i, gt_i)
                else:
                    ps, ss = None, None
                psnr_rows.append(ps); ssim_rows.append(ss)

                if pick_angle is not None and str(pick_angle).lower() != "mid":
                    want_a = int(pick_angle)
                    diff = abs(int(ai) - want_a) if ai >= 0 else 9999
                    if best_one is None or diff < best_one[0]:
                        best_one = (diff, dict(pred=pred_i, noisy=noz_i, gt=gt_i, angle=ai))

        if len(preds) == 0:
            raise RuntimeError(f"[I2SB Multi] pick_id={pick_id} not found in split.")

        order = list(range(len(preds)))
        order.sort(key=lambda k: (angle_rows[k] < 0, angle_rows[k]))

        def reorder(lst): return [lst[k] for k in order]
        preds = reorder(preds); noisies = reorder(noisies)
        if preds_blend:
            preds_blend = reorder(preds_blend)
        if gts:
            gts = reorder(gts)
        preds_raw = reorder(preds_raw); gts_raw = reorder(gts_raw); noisies_raw = reorder(noisies_raw)
        id_rows = reorder(id_rows); angle_rows = reorder(angle_rows); A_rows = reorder(A_rows)
        noisy_lo_rows = reorder(noisy_lo_rows); noisy_hi_rows = reorder(noisy_hi_rows)
        gt_lo_rows = reorder(gt_lo_rows); gt_hi_rows = reorder(gt_hi_rows)
        psnr_rows = reorder(psnr_rows); ssim_rows = reorder(ssim_rows)

        vol_pred = np.stack(preds, axis=0).astype(np.float32)
        tiff.imwrite(os.path.join(out_dir, "pred_volume.tiff"), vol_pred, imagej=True)
        np.save(os.path.join(out_dir, "pred_volume.npy"), vol_pred)

        if preds_blend:
            vol_pb = np.stack(preds_blend, axis=0).astype(np.float32)
            tiff.imwrite(os.path.join(out_dir, "pred_volume_blend.tiff"), vol_pb, imagej=True)
            np.save(os.path.join(out_dir, "pred_volume_blend.npy"), vol_pb)

        vol_nozy = np.stack(noisies, axis=0).astype(np.float32)
        tiff.imwrite(os.path.join(out_dir, "noisy_volume.tiff"), vol_nozy, imagej=True)

        if gts:
            vol_gt = np.stack(gts, axis=0).astype(np.float32)
            tiff.imwrite(os.path.join(out_dir, "gt_volume.tiff"), vol_gt, imagej=True)

        if len(preds_raw) > 0 and all(x is not None for x in preds_raw):
            vol_pr = np.stack(preds_raw, axis=0).astype(np.float32)
            tiff.imwrite(os.path.join(out_dir, "pred_volume_raw.tiff"), vol_pr, imagej=True)
            np.save(os.path.join(out_dir, "pred_volume_raw.npy"), vol_pr)

        csv_path = os.path.join(out_dir, "metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx","id","angle","A","nlo","nhi","glo","ghi","PSNR","SSIM"])
            for idx, val in enumerate(zip(id_rows, angle_rows, A_rows,
                                          noisy_lo_rows, noisy_hi_rows,
                                          gt_lo_rows, gt_hi_rows,
                                          psnr_rows, ssim_rows)):
                w.writerow([idx] + [x if x is not None else "" for x in val])
        print(f"[Done] Metrics saved to {csv_path}")

        ps_valid = [x for x in psnr_rows if x is not None]
        ss_valid = [x for x in ssim_rows if x is not None]
        if ps_valid:
            print(f"[Result] ID={pick_id} | K={steps} | PSNR={np.mean(ps_valid):.3f} | SSIM={np.mean(ss_valid):.4f}")

        if pick_angle is not None:
            if best_one:
                rec = best_one[1]
            else:
                mid = len(preds)//2
                rec = dict(pred=preds[mid], noisy=noisies[mid], gt=gts[mid] if gts else None, angle=angle_rows[mid])

            tiles_cfg = (cfg.get("sample", {}).get("tiles", {}) or {})
            ph, pw = rec["pred"].shape
            nn, gg = _match_to_pred_size((ph, pw), rec["noisy"], rec["gt"])
            aname = f"{int(rec['angle']):03d}" if rec["angle"] >= 0 else "xxx"
            save_quad_and_tiles(nn, rec["pred"], gg, out_dir, f"id{pick_id}_a{aname}_step{steps}", tiles_cfg)

    else:
        print(f"[I2SB Multi] Processing ALL data in split '{want}'...")
        preds = []

        pbar = tqdm(loader, ncols=80)
        for batch in pbar:
            inp = batch["inp"].to(dev).float()
            if inp.shape[1] < 3:
                raise RuntimeError("add_angle_channel must be true: inp should be 3ch [noisy, mask, angle].")

            x1_img01 = inp[:, 0:1, ...]
            mask01   = inp[:, 1:2, ...].clamp(0.0, 1.0)
            angle01  = inp[:, 2:3, ...].clamp(0.0, 1.0)
            x1_img_m1 = x1_img01 * 2.0 - 1.0

            x0_hat_m1 = sample_multi_step(
                net=net,
                x1_img_m1=x1_img_m1,
                mask01=mask01,
                angle01=angle01,
                depth_down=depth_down,
                steps=steps,
                sigma_T=sigma_T
            )
            pred01 = ((x0_hat_m1.clamp(-1, 1) + 1.0) * 0.5).cpu().numpy()

            for i in range(pred01.shape[0]):
                preds.append(np.squeeze(pred01[i, 0]))

        if not preds:
            raise RuntimeError("No predictions generated.")
        vol = np.stack(preds, axis=0).astype(np.float32)
        tiff.imwrite(os.path.join(out_dir, "pred_volume_all.tiff"), vol, imagej=True)
        print(f"[Done] Saved all predictions to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/sample_i2sb_local_multi.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args.cfg, ckpt=args.ckpt, out=args.out)

if __name__ == "__main__":
    main()
