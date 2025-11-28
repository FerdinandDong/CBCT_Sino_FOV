# scripts/train.py
# -*- coding: utf-8 -*-
import argparse, yaml, os
from typing import Any, List, Optional
import torch

# 触发模型注册
import ctprojfix.models.unet
import ctprojfix.models.unet_res
import ctprojfix.models.pconv_unet
import ctprojfix.models.diffusion.ddpm

from ctprojfix.models.registry import build_model
from ctprojfix.data.dataset import make_dataloader
from ctprojfix.trainers.supervised import SupervisedTrainer
from ctprojfix.trainers.diffusion import DiffusionTrainer
from ctprojfix.models.diffusion.ddpm import DDPMProj


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# 设备解析与多卡封装测试
def _resolve_device(cfg_train: dict, cli_device: Optional[str], cli_gpu: Optional[str|int]) -> torch.device:
    # 1) CLI --device 优先
    if cli_device:
        d = str(cli_device).strip().lower()
        if d == "cpu":
            return torch.device("cpu")
        if d.startswith("cuda") and torch.cuda.is_available():
            return torch.device(d)
        print(f"[WARN] --device={d} 不可用，退回 CPU")
        return torch.device("cpu")

    # 2) CLI --gpu 紧随其后
    if cli_gpu is not None:
        try:
            idx = int(cli_gpu)
            if torch.cuda.is_available():
                return torch.device(f"cuda:{idx}")
        except Exception:
            pass
        print(f"[WARN] --gpu={cli_gpu} 不可用，退回 CPU")
        return torch.device("cpu")

    # 3) cfg.train.device
    d = str(cfg_train.get("device", "") or "").strip().lower()
    if d:
        if d == "cpu":
            return torch.device("cpu")
        if d.startswith("cuda") and torch.cuda.is_available():
            return torch.device(d)
        if d.startswith("cuda"):
            print(f"[WARN] cfg.train.device={d} 要求 CUDA，但本机不可用，退回 CPU。")
            return torch.device("cpu")

    # 4) auto
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _maybe_wrap_dataparallel(model: torch.nn.Module, cfg_train: dict) -> torch.nn.Module:
    """
    可选 DataParallel：
      train:
        data_parallel: true
        gpu_ids: [0,1,2]
    仅用于单机多卡的简易并行；真要多机/高性能请改 DDP。
    """
    use_dp = bool(cfg_train.get("data_parallel", False))
    gpu_ids: List[int] = cfg_train.get("gpu_ids", [])
    if not use_dp:
        return model
    if not torch.cuda.is_available():
        print("[WARN] data_parallel=True 但 CUDA 不可用，忽略。")
        return model
    if not gpu_ids:
        # 若未显式给出，使用所有可见 GPU
        gpu_ids = list(range(torch.cuda.device_count()))
    if len(gpu_ids) <= 1:
        return model
    print(f"[DP] Using DataParallel on GPU ids: {gpu_ids}")
    model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/train_unet.yaml")
    ap.add_argument("--device", type=str, default=None, help='如 "cuda:3" / "cpu"（优先级最高）')
    ap.add_argument("--gpu", type=str, default=None, help="等价于 --device cuda:{idx}")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    tr = cfg.get("train", {})  # 统一取一遍 train 配置


    # I2SB(本地) 分支：直接用我数据，无需外部库
    name_lower = str(cfg.get("model", {}).get("name", "")).lower().strip()
    if name_lower == "i2sb_local":
        from ctprojfix.trainers.i2sb_local import I2SBLocalTrainer
        from ctprojfix.models.i2sb_unet import I2SBUNet
        device = _resolve_device(tr, args.device, args.gpu)
        if device.type == "cuda":
            try:
                torch.cuda.set_device(device.index if device.index is not None else 0)
            except Exception as e:
                print(f"[WARN] torch.cuda.set_device 失败：{e}")
        print(f"[DEVICE] using device = {device}")

        # 数据
        loaders = make_dataloader(cfg["data"])
        if isinstance(loaders, dict):
            train_loader = loaders.get("train")
            val_loader = loaders.get("val")
        else:
            train_loader, val_loader = loaders, None

        # 模型（通道= x0(1) + x1(2/3)）
        mparam = cfg["model"]["params"]
        model = I2SBUNet(
            in_ch=int(mparam.get("in_ch", 4)),   # 默认 noisy+mask+angle + x0_embed
            base=int(mparam.get("base", 64)),
            depth=int(mparam.get("depth", 4)),
            dropout=float(mparam.get("dropout", 0.0)),
        ).to(device)
        model = _maybe_wrap_dataparallel(model, tr)

        # 训练器
        trainer = I2SBLocalTrainer(
            device=str(device),
            lr=float(tr.get("lr", 5e-5)),
            epochs=int(tr.get("epochs", 150)),
            sigma_T=float(tr.get("sigma_T", 1.0)),
            t0=float(tr.get("t0", 1e-4)),
            ema_decay=float(tr.get("ema", 0.999)),
            ckpt_dir=tr.get("ckpt_dir", "checkpoints/diffusion/i2sb_local"),
            ckpt_prefix=tr.get("ckpt_prefix", "i2sb_local"),
            save_every=int(tr.get("save_every", 1)),
            max_keep=int(tr.get("max_keep", 5)),
            log_dir=tr.get("log_dir", "logs/i2sb_local"),
            cond_has_angle=bool(cfg["data"].get("add_angle_channel", False)),

            # 把调度与度量策略透传进去
            val_metric=str(tr.get("val_metric", "loss")),
            maximize_metric=bool(tr.get("maximize_metric", False)),
            sched=tr.get("sched", None),
            dump_preview_every=int(tr.get("dump_preview_every", 0)),
        )

        trainer.fit(model, train_loader, val_loader)
        return

    # 解析设备
    device = _resolve_device(tr, args.device, args.gpu)
    if device.type == "cuda":
        try:
            torch.cuda.set_device(device.index if device.index is not None else 0)
        except Exception as e:
            print(f"[WARN] torch.cuda.set_device 失败：{e}")
    print(f"[DEVICE] using device = {device}")
    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(device)
            print(f"[DEVICE] GPU name = {name}")
            print(f"[DEVICE] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
        except Exception:
            pass

    # 模型 & 数据
    model = build_model(cfg["model"]["name"], **cfg["model"]["params"]).to(device)

    # 多卡 DataParallel 包装 试试
    model = _maybe_wrap_dataparallel(model, tr)

    loaders: Any = make_dataloader(cfg["data"])  # 可能返回 DataLoader 或 dict 支持三分
    if isinstance(loaders, dict):
        train_loader = loaders.get("train")
        val_loader = loaders.get("val")
        test_loader = loaders.get("test")
        try:
            n_tr = len(train_loader.dataset) if train_loader is not None else 0
            n_va = len(val_loader.dataset) if val_loader is not None else 0
            n_te = len(test_loader.dataset) if test_loader is not None else 0
            print(f"[DATA] split sizes -> train:{n_tr}  val:{n_va}  test:{n_te}")
        except Exception:
            pass
    else:
        train_loader = loaders
        val_loader = None

    name = str(cfg["model"]["name"]).lower().strip()

    if isinstance(model, DDPMProj) or name in ("diffusion", "ddpm", "ldm"):
        trainer = DiffusionTrainer(
            device=str(device),
            lr=float(tr.get("lr", 1e-4)),
            epochs=int(tr.get("epochs", 1)),
            T=int(tr.get("T", 1000)),
            beta_start=float(tr.get("beta_start", 1e-4)),
            beta_end=float(tr.get("beta_end", 2e-2)),
            dc_strength=float(tr.get("dc_strength", 1.0)),
            ckpt_dir=tr.get("ckpt_dir", "checkpoints/diffusion"),
            ckpt_prefix=tr.get("ckpt_prefix", "DDPM"),
            save_every=int(tr.get("save_every", 1)),
            max_keep=int(tr.get("max_keep", 5)),
            resume=tr.get("resume", False),
            strict_load=bool(tr.get("strict_load", True)),
        )
    else:
        trainer = SupervisedTrainer(
            device=str(device),
            lr=float(tr.get("lr", 3e-4)),
            epochs=int(tr.get("epochs", 2)),
            ckpt_dir=tr.get("ckpt_dir", "checkpoints"),
            ckpt_prefix=tr.get("ckpt_prefix", None),
            save_every=int(tr.get("save_every", 1)),
            max_keep=int(tr.get("max_keep", 5)),

            # ---- 续训选项 ----
            resume_from=tr.get("resume_from", None),
            resume=tr.get("resume", "auto"),
            reset_epoch=bool(tr.get("reset_epoch", False)),
            reset_optim=bool(tr.get("reset_optim", False)),
            strict_load=bool(tr.get("strict_load", True)),

            # ---- 损失配置 ----
            loss_cfg=tr.get("loss", {"type": "combined"}),

            # ---- 日志选项 ----
            use_tqdm=bool(tr.get("use_tqdm", True)),
            log_interval=tr.get("log_interval", None),

            # 验证指标与 LR 调度 
            val_metric=tr.get("val_metric", "loss"),   # loss | psnr | ssim | None
            sched=tr.get("sched", None),               # e.g. {type: cosine, T_max: 150, eta_min: 1e-6}
        )

    print(f"[DEBUG] using trainer: {type(trainer).__name__}")

    # 训练
    try:
        trainer.fit(model, train_loader, val_loader)
    except TypeError:
        trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
