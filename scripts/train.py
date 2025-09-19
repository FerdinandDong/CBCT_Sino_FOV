# scripts/train.py
import argparse, yaml
from typing import Any

# 触发模型注册（保持不变）
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/train_unet.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    tr = cfg.get("train", {})  # 统一取一遍 train 配置

    # 模型 & 数据
    model = build_model(cfg["model"]["name"], **cfg["model"]["params"])  # 原样
    loaders: Any = make_dataloader(cfg["data"])  # 可能返回 DataLoader 或 dict  # 兼容三分

    # 兼容 {train,val,test} 字典或单个 DataLoader
    if isinstance(loaders, dict):
        train_loader = loaders.get("train")
        val_loader = loaders.get("val")
        test_loader = loaders.get("test")  # 目前未在训练中使用，仅保留占位
        try:
            n_tr = len(train_loader.dataset)
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
            device=tr.get("device", "cuda"),
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
            device=tr.get("device", "cuda"),
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
        )

    print(f"[DEBUG] using trainer: {type(trainer).__name__}")

    # 优先调用支持 val_loader 的新签名；否则回退老签名
    try:
        trainer.fit(model, train_loader, val_loader)  # 传入验证集
    except TypeError:
        trainer.fit(model, train_loader)              # 兼容旧实现


if __name__ == "__main__":
    main()
