# ctprojfix/models/i2sb_unet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Utils ----------
def _safe_num_groups(num_channels: int, prefer: int = 8) -> int:
    """返回能整除 num_channels 的 GroupNorm 分组数（尽量接近 prefer）。"""
    g = min(prefer, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return g if g > 0 else 1

def _center_crop_to(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """中心裁剪 x 到指定 (H, W)。"""
    Ht, Wt = target_hw
    H, W = x.shape[-2], x.shape[-1]
    if (H, W) == (Ht, Wt):
        return x
    top = max((H - Ht) // 2, 0)
    left = max((W - Wt) // 2, 0)
    return x[..., top:top + Ht, left:left + Wt]

# --- Sinusoidal time embedding ---
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,) in [0,1] —— 将标量时间映射到固定维度向量
    return: (B, dim)
    """
    t = t.to(dtype=torch.float32)
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(start=0, end=half, device=device, dtype=torch.float32)
        * -(math.log(1e4) / (max(half - 1, 1)))
    )  # (half,)
    args = t[:, None] * freqs[None, :]  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb  # (B, dim)

# ---------- Blocks ----------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0, gn_groups=8):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        g1 = _safe_num_groups(in_ch, gn_groups)
        g2 = _safe_num_groups(out_ch, gn_groups)

        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, emb):
        # x: (B, C, H, W), emb: (B, emb_dim)
        h = self.conv1(self.act(self.norm1(x)))
        emb_add = self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)  # (B, out_ch, 1, 1)
        h = h + emb_add
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout, gn_groups=8):
        super().__init__()
        self.block1 = ResBlock(in_ch, out_ch, emb_dim, dropout, gn_groups)
        self.block2 = ResBlock(out_ch, out_ch, emb_dim, dropout, gn_groups)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, emb):
        x = self.block1(x, emb)
        x = self.block2(x, emb)
        x = self.pool(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout, gn_groups=8):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.block1 = ResBlock(in_ch, out_ch, emb_dim, dropout, gn_groups)
        self.block2 = ResBlock(out_ch, out_ch, emb_dim, dropout, gn_groups)

    def forward(self, x, hskip, emb):
        # 上采样
        x = self.up(x)
        # 尺寸对齐：先把 skip 裁到 x 的大小；仍不等则把 x 插到 skip 的大小（双保险）
        H, W = x.shape[-2], x.shape[-1]
        hskip = _center_crop_to(hskip, (H, W))
        if hskip.shape[-2:] != x.shape[-2:]:
            x = F.interpolate(x, size=hskip.shape[-2:], mode="nearest")
        # 拼接 + 两个残差块
        x = torch.cat([x, hskip], dim=1)
        x = self.block1(x, emb)
        x = self.block2(x, emb)
        return x

# ---------- U-Net ----------
class I2SBUNet(nn.Module):
    """
    轻量 U-Net（I2SB 本地版）：
      输入: concat([x1(2或3通道: noisy+mask(+angle)), t_map(1)])  → in_ch = (2/3)+1
      输出: x0_hat (1通道)
    """
    def __init__(self, in_ch=4, base=64, depth=4, emb_dim=256, dropout=0.0, gn_groups=8):
        super().__init__()
        self.in_ch = in_ch
        self.emb_dim = emb_dim

        # 架构通道
        ch = [base * (2 ** i) for i in range(depth)]  # e.g. [64,128,256,512]
        self.in_conv = nn.Conv2d(in_ch, ch[0], 3, padding=1)

        # 下采样堆叠
        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(Down(ch[i], ch[i + 1], emb_dim, dropout, gn_groups))

        # Bottleneck
        self.mid1 = ResBlock(ch[-1], ch[-1], emb_dim, dropout, gn_groups)
        self.mid2 = ResBlock(ch[-1], ch[-1], emb_dim, dropout, gn_groups)

        # 上采样堆叠（注意 in_ch = 上采样通道 + skip 通道）
        self.ups = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.ups.append(Up(ch[i + 1] + ch[i], ch[i], emb_dim, dropout, gn_groups))

        # 头/尾
        g0 = _safe_num_groups(ch[0], gn_groups)
        self.out_norm = nn.GroupNorm(g0, ch[0])
        self.out_act = nn.SiLU()
        self.out = nn.Conv2d(ch[0], 1, 3, padding=1)

        # 时间嵌入 MLP（输入为 timestep_embedding 的 (B, emb_dim)）
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x1_with_t):  # (B, in_ch, H, W)
        B = x1_with_t.shape[0]

        # 拆出 t_map（最后 1 通道），并得到标量 t ∈ [0,1]（B,）
        t_map = x1_with_t[:, -1:, :, :]
        t = t_map.mean(dim=(2, 3)).squeeze(1).clamp(0.0, 1.0)

        # 时间嵌入
        emb = timestep_embedding(t, self.emb_dim)  # (B, emb_dim)
        emb = self.time_mlp(emb)                   # (B, emb_dim)

        # 编码
        hs = []
        h = self.in_conv(x1_with_t)  # 用完整输入（含 t_map）
        hs.append(h)
        for down in self.downs:
            h = down(h, emb)
            hs.append(h)

        # Bottleneck
        h = self.mid1(h, emb)
        h = self.mid2(h, emb)

        # 解码（逐级与 skip 连接）
        for up, hskip in zip(self.ups, reversed(hs[:-1])):
            h = up(h, hskip, emb)

        # 输出
        h = self.out(self.out_act(self.out_norm(h)))  # (B,1,H,W)
        return h
