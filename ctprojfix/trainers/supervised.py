import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

class SupervisedTrainer:
    def __init__(self, device="cuda", lr=3e-4, epochs=2, ckpt_dir="checkpoints"):
        # 强制类型转换，避免 YAML/环境把数值当成字符串
        self.device = torch.device(device if torch.cuda.is_available() else "cpu") \
                      if isinstance(device, str) else device
        self.lr = float(lr)
        self.epochs = int(epochs)
        #checkpoint 目录
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def fit(self, model, loader):
        model = model.to(self.device)
        opt = Adam(model.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            model.train()
            losses = []
            pbar = tqdm(loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                x = batch["inp"].to(self.device)
                y = batch["gt"].to(self.device)

                pred = model(x)
                loss = F.l1_loss(pred, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            # 每个 epoch 保存一次权重
            ckpt_path = os.path.join(self.ckpt_dir, f"{model.__class__.__name__}_epoch{epoch}.pth")
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, ckpt_path)
            print(f"[OK] Saved checkpoint: {ckpt_path}")
            print(f"Epoch {epoch} mean loss: {sum(losses)/len(losses):.4f}")
