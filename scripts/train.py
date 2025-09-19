# scripts/train.py
import argparse, yaml

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/train_unet.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    # 统一取一遍 train 配置，避免分支里重复取/漏取
    tr = cfg.get("train", {})

    # 模型 & 数据
    model = build_model(cfg["model"]["name"], **cfg["model"]["params"])
    loader = make_dataloader(cfg["data"])

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
    trainer.fit(model, loader)


if __name__ == "__main__":
    main()
