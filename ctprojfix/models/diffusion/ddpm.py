import torch, torch.nn as nn
from ..unet import UNetProj
from ..registry import register

def timestep_embedding(t, dim):
    # 简单版正弦时间步嵌入
    device = t.device
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=device) * (torch.log(torch.tensor(10000.0))/half))
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2: emb = torch.nn.functional.pad(emb, (0,1))
    return emb

@register("diffusion")
class DDPMProj(nn.Module):
    """
    条件扩散（投影修复）：UNet 预测噪声 ε
    输入通道：x_t (1) + noisy (1) + mask (1) [+ angle (1, 可选)]
    输出通道：1（噪声）
    """
    def __init__(self, add_angle_channel=False, base=32):
        super().__init__()
        in_ch = 3 + (1 if add_angle_channel else 0)  # x_t + noisy + mask (+ angle)
        self.noise_pred = UNetProj(in_ch=in_ch, out_ch=1, base=base)

    def forward(self, x_t, cond, t, angle_norm=None):
        # cond: (B,2,H,W) = [noisy, mask];  x_t: (B,1,H,W)
        xs = [x_t, cond]
        if angle_norm is not None:
            xs.append(angle_norm)  # (B,1,H,W)
        x = torch.cat(xs, dim=1)

        # --- 自适配：根据 UNet 首层 in_channels 补/裁通道 ---
        exp_c = self.noise_pred.enc1[0][0].in_channels if hasattr(self.noise_pred.enc1[0], '__getitem__') else self.noise_pred.enc1[0].in_channels
        cur_c = x.shape[1]
        if cur_c < exp_c:
            pad = torch.zeros(x.size(0), exp_c - cur_c, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif cur_c > exp_c:
            x = x[:, :exp_c, :, :]

        return self.noise_pred(x)  # ε̂