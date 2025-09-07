import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import register


def timestep_embedding(t, dim, max_period=10000):
    """
    t: (B,) long
    return: (B, dim)
    标准正弦时间步嵌入（与 DDPM/StableDiffusion 相同形式）
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B,half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # (B,dim or dim-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb


class FiLM(nn.Module):
    """
    将一个 embedding 向量  ->  生成 scale, shift
    用于在 ResBlock 中做 AdaGN（对 GN 输出做 y = y*(1+scale) + shift）
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, out_dim * 2)
        )

    def forward(self, emb):
        h = self.net(emb)  # (B, 2*out_dim)
        scale, shift = torch.chunk(h, 2, dim=1)
        return scale, shift


class ResBlock(nn.Module):
    """
    ResBlock + GroupNorm + SiLU，支持 FiLM 时间步调制
    """
    def __init__(self, in_ch, out_ch, emb_dim, groups=8, dropout=0.0):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 1e-8 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb_to_scale_shift = FiLM(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        """
        x: (B,C,H,W)
        emb: (B, emb_dim)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # FiLM: 对中间特征注入 (scale, shift)
        scale, shift = self.emb_to_scale_shift(emb)  # (B, C)
        # reshape -> (B,C,1,1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 2, stride=2)

    def forward(self, x):
        return self.op(x)


# Diffusion UNet 主体

class TimeCondUNet(nn.Module):
    """
    UNet with time-step FiLM conditioning.
    输入通道： x_t(1) + noisy(1) + mask(1) [+ angle(1)]
    """
    def __init__(self, in_ch=3, out_ch=1, base=32, depth=4, emb_dim=256, dropout=0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        # 时间步嵌入投影到 emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.emb_dim = emb_dim

        ch = base
        self.in_conv = nn.Conv2d(in_ch, ch, 3, padding=1)

        # Encoder
        enc_blocks = []
        downs = []
        chs = [ch]
        for _ in range(depth):
            enc_blocks.append(ResBlock(ch, ch, emb_dim=emb_dim, dropout=dropout))
            ch_next = ch * 2
            enc_blocks.append(ResBlock(ch, ch_next, emb_dim=emb_dim, dropout=dropout))
            downs.append(Down(ch_next))
            ch = ch_next
            chs.append(ch)
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.downs = nn.ModuleList(downs)

        # Bottleneck
        self.mid1 = ResBlock(ch, ch, emb_dim=emb_dim, dropout=dropout)
        self.mid2 = ResBlock(ch, ch, emb_dim=emb_dim, dropout=dropout)

        # Decoder
        dec_blocks = []
        ups = []
        for i in reversed(range(depth)):
            ch_skip = chs[i + 1]  # ✅ 与 skips 对齐：第一次应为最深的 16b
            ups.append(Up(ch))
            dec_blocks.append(ResBlock(ch + ch_skip, ch // 2, emb_dim=emb_dim, dropout=dropout))
            dec_blocks.append(ResBlock(ch // 2, ch // 2, emb_dim=emb_dim, dropout=dropout))
            ch = ch // 2

        self.ups = nn.ModuleList(ups)
        self.dec_blocks = nn.ModuleList(dec_blocks)

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 1)

    def forward(self, x, t_emb):
        """
        x: (B, in_ch, H, W)
        t_emb: (B,) long
        """
        t = timestep_embedding(t_emb, self.emb_dim)   # (B, emb_dim)
        t = self.time_mlp(t)                          # (B, emb_dim)

        x = self.in_conv(x)
        skips = []
        # Encoder
        it = iter(self.enc_blocks)
        for down in self.downs:
            b1 = next(it)(x, t)
            b2 = next(it)(b1, t)
            skips.append(b2)
            x = down(b2)

        # Mid
        x = self.mid1(x, t)
        x = self.mid2(x, t)

        # Decoder
        it = iter(self.dec_blocks)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x)
            # 对齐尺寸（防止奇偶性）
            if x.shape[-2:] != skip.shape[-2:]:
                diffY = skip.size(2) - x.size(2)
                diffX = skip.size(3) - x.size(3)
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
            x = torch.cat([x, skip], dim=1)
            x = next(it)(x, t)
            x = next(it)(x, t)

        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)
        return x


# 训练不变

@register("diffusion")
class DDPMProj(nn.Module):
    """
    条件扩散（投影修复）：UNet 预测噪声 ε
    输入：
      x_t: (B,1,H,W)
      cond: (B,2,H,W) = [noisy, mask]
      angle_norm: (B,1,H,W) 可选
      t: (B,) long
    """
    def __init__(self, add_angle_channel=False, base=32, depth=4, emb_dim=256, dropout=0.0):
        super().__init__()
        self.add_angle = bool(add_angle_channel)
        in_ch = 3 + (1 if self.add_angle else 0)  # x_t + noisy + mask (+ angle)
        self.net = TimeCondUNet(in_ch=in_ch, out_ch=1, base=base, depth=depth, emb_dim=emb_dim, dropout=dropout)

    def forward(self, x_t, cond, t, angle_norm=None):
        xs = [x_t, cond]  # x_t: (B,1,H,W), cond: (B,2,H,W)
        if self.add_angle and (angle_norm is not None):
            xs.append(angle_norm)  # (B,1,H,W)
        x = torch.cat(xs, dim=1)  # (B, in_ch, H, W)
        return self.net(x, t)     # 预测 ε̂

# import torch, torch.nn as nn
# from ..unet import UNetProj
# from ..registry import register

# def timestep_embedding(t, dim):
#     # 简单版正弦时间步嵌入
#     device = t.device
#     half = dim // 2
#     freqs = torch.exp(-torch.arange(half, device=device) * (torch.log(torch.tensor(10000.0))/half))
#     args = t[:, None].float() * freqs[None]
#     emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
#     if dim % 2: emb = torch.nn.functional.pad(emb, (0,1))
#     return emb

# @register("diffusion")
# class DDPMProj(nn.Module):
#     """
#     条件扩散（投影修复）：UNet 预测噪声 ε
#     输入通道：x_t (1) + noisy (1) + mask (1) [+ angle (1, 可选)]
#     输出通道：1（噪声）
#     """
#     def __init__(self, add_angle_channel=False, base=32):
#         super().__init__()
#         in_ch = 3 + (1 if add_angle_channel else 0)  # x_t + noisy + mask (+ angle)
#         self.noise_pred = UNetProj(in_ch=in_ch, out_ch=1, base=base)

#     def forward(self, x_t, cond, t, angle_norm=None):
#         # cond: (B,2,H,W) = [noisy, mask];  x_t: (B,1,H,W)
#         xs = [x_t, cond]
#         if angle_norm is not None:
#             xs.append(angle_norm)  # (B,1,H,W)
#         x = torch.cat(xs, dim=1)

#         # --- 自适配：根据 UNet 首层 in_channels 补/裁通道 ---
#         exp_c = self.noise_pred.enc1[0][0].in_channels if hasattr(self.noise_pred.enc1[0], '__getitem__') else self.noise_pred.enc1[0].in_channels
#         cur_c = x.shape[1]
#         if cur_c < exp_c:
#             pad = torch.zeros(x.size(0), exp_c - cur_c, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
#             x = torch.cat([x, pad], dim=1)
#         elif cur_c > exp_c:
#             x = x[:, :exp_c, :, :]

#         return self.noise_pred(x)  # ε̂