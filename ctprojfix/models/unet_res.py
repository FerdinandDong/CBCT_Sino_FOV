import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register


class ResBlock(nn.Module):
    """Residual Block with 2 conv layers"""
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.bn1 = nn.BatchNorm2d(out_ch) if norm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_ch) if norm else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class Down(nn.Module):
    """Downsample block"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ResBlock(in_ch, out_ch),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upsample block with skip connection"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.res = ResBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        if x.shape[-2:] != skip.shape[-2:]:
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.res(x)


@register("unet_res")
class UNetRes(nn.Module):
    """
    ResNet-UNet baseline for projection repair
    """
    def __init__(self, in_ch=2, out_ch=1, base=64, depth=4):
        super().__init__()
        self.inc = ResBlock(in_ch, base)

        # Encoder
        self.downs = nn.ModuleList()
        ch = base
        for i in range(depth):
            self.downs.append(Down(ch, ch * 2))
            ch *= 2

        # Bottleneck
        self.bottleneck = ResBlock(ch, ch)

        # Decoder
        self.ups = nn.ModuleList()
        for i in range(depth):
            self.ups.append(Up(ch, ch // 2))
            ch //= 2

        # Final conv
        self.outc = nn.Conv2d(ch, out_ch, 1)

    def forward(self, x):
        skips = []
        x = self.inc(x)
        skips.append(x)

        # down path
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # bottleneck
        x = self.bottleneck(x)

        # up path
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]  # reverse order
            x = up(x, skip)

        return self.outc(x)
