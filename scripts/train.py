# def main():
#     print("Training entry (use configs/train_unet.yaml or train_diffusion.yaml).")

# if __name__ == "__main__":
#     main()
import argparse, yaml
import ctprojfix.models.unet                     # 触发注册
from ctprojfix.models.registry import build_model
from ctprojfix.data.dataset import make_dataloader
from ctprojfix.trainers.supervised import SupervisedTrainer

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
    trainer_cfg = cfg["train"]
    trainer = SupervisedTrainer(
        device=trainer_cfg.get("device", "cuda"),
        lr=float(trainer_cfg.get("lr", 3e-4)),
        epochs=int(trainer_cfg.get("epochs", 2)),
    )
    print("train cfg types:", type(cfg["train"]["lr"]), type(cfg["train"]["epochs"]))

    trainer.fit(model, loader)

if __name__ == "__main__":
    main()
