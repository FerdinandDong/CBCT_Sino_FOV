import argparse, yaml

# 触发模型注册
import ctprojfix.models.unet
import ctprojfix.models.unet_res
import ctprojfix.models.diffusion.ddpm

from ctprojfix.models.registry import build_model
from ctprojfix.data.dataset import make_dataloader
from ctprojfix.trainers.supervised import SupervisedTrainer
from ctprojfix.trainers.diffusion import DiffusionTrainer
from ctprojfix.models.diffusion.ddpm import DDPMProj  # 用于 isinstance 判别

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/train_unet.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    model = build_model(cfg["model"]["name"], **cfg["model"]["params"])
    loader = make_dataloader(cfg["data"])

    name = str(cfg["model"]["name"]).lower().strip()
    if isinstance(model, DDPMProj) or name in ("diffusion", "ddpm", "ldm"):
        tr = cfg["train"]
        trainer = DiffusionTrainer(
            device=tr.get("device","cuda"),
            lr=float(tr.get("lr",1e-4)),
            epochs=int(tr.get("epochs",1)),
            T=int(tr.get("T",1000)),
            beta_start=float(tr.get("beta_start",1e-4)),
            beta_end=float(tr.get("beta_end",2e-2)),
            dc_strength=float(tr.get("dc_strength",1.0)),
            ckpt_dir=tr.get("ckpt_dir", "checkpoints/diffusion"),
            save_every=int(tr.get("save_every", 1)),
            max_keep=int(tr.get("max_keep", 5)),
        )

    else:
        trainer = SupervisedTrainer(
            device=cfg["train"].get("device","cuda"),
            lr=float(cfg["train"].get("lr",3e-4)),
            epochs=int(cfg["train"].get("epochs",2)),
        )

    print(f"[DEBUG] using trainer: {type(trainer).__name__}")
    trainer.fit(model, loader)

if __name__ == "__main__":
    main()
