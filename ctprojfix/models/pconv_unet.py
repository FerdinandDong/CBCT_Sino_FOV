# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register  # 你的注册表

# ---------------------------
# Partial Convolution (single-channel mask version)
# ---------------------------
class PartialConv2d(nn.Conv2d):
    """
    Partial Convolution (single-channel mask) 版本：
    - mask: (B,1,H,W)，同一张 mask 作用于所有输入通道
    - 每次前向会同步更新下一层要用的 mask（update_mask）
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=True, eps=1e-6):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.register_buffer("weight_mask",
                             torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1]))
        self.slide_win = self.weight_mask.numel()
        self.eps = eps

        self._last_hw = None
        self._update_mask = None
        self._mask_ratio = None

    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        if mask is None:
            mask = x.new_ones(B, 1, H, W)

        need_recalc = (
            self._last_hw != (H, W)
            or self._update_mask is None
            or self._update_mask.dtype != x.dtype
            or self._update_mask.device != x.device
        )
        if need_recalc:
            self._last_hw = (H, W)
            with torch.no_grad():
                update_mask = F.conv2d(
                    mask, self.weight_mask.to(x), bias=None,
                    stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1
                )
                mask_ratio = self.slide_win / (update_mask + self.eps)
                update_mask = torch.clamp(update_mask, 0, 1)
                mask_ratio = mask_ratio * update_mask
                self._update_mask = update_mask
                self._mask_ratio = mask_ratio

        x_masked = x * mask
        raw = super().forward(x_masked)

        if self.bias is not None:
            b = self.bias.view(1, -1, 1, 1)
            out = (raw - b) * self._mask_ratio + b
        else:
            out = raw * self._mask_ratio

        out = out * self._update_mask
        return out, self._update_mask


# ---------------------------
# Blocks
# ---------------------------
class PConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, bn=True, act="relu"):
        super().__init__()
        self.pconv = PartialConv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=(not bn))
        self.bn = nn.BatchNorm2d(cout) if bn else None
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leaky":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act is None:
            self.act = None
        else:
            raise ValueError(f"Unknown act: {act}")

    def forward(self, x, m):
        x, m = self.pconv(x, m)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x, m


class DownP(nn.Module):
    """ 下采样：PConv(stride=2) """
    def __init__(self, cin, cout):
        super().__init__()
        self.block = PConvBNAct(cin, cout, k=3, s=2, p=1, bn=True, act="leaky")
    def forward(self, x, m):
        return self.block(x, m)


class UpP(nn.Module):
    """ 上采样：Nearest/Bilinear 上采样 + PConv """
    def __init__(self, cin, cout, mode="nearest"):
        super().__init__()
        self.mode = mode
        self.block = PConvBNAct(cin, cout, k=3, s=1, p=1, bn=True, act="leaky")

    def _interp_img(self, x, size):
        if self.mode in ("bilinear", "bicubic"):
            return F.interpolate(x, size=size, mode=self.mode, align_corners=False)
        return F.interpolate(x, size=size, mode=self.mode)

    def forward(self, x, m, skip_x=None, skip_m=None):
        if skip_x is not None:
            target_hw = skip_x.shape[-2:]                # 直接对齐到 skip 的 H,W
            x = self._interp_img(x, target_hw)
            m = F.interpolate(m, size=target_hw, mode="nearest")
            x = torch.cat([x, skip_x], dim=1)
            m = torch.cat([m, skip_m], dim=1)
            m = m.amax(dim=1, keepdim=True)             # 合并多通道 mask
        else:
            # 最后一层没有 skip，就常规上采样一倍
            if self.mode in ("bilinear", "bicubic"):
                x = F.interpolate(x, scale_factor=2, mode=self.mode, align_corners=False)
            else:
                x = F.interpolate(x, scale_factor=2, mode=self.mode)
            m = F.interpolate(m, scale_factor=2, mode="nearest")

        return self.block(x, m)

# ---------------------------
# PConv U-Net for Projection Restoration
# ---------------------------
@register("pconv_unet")
class PConvUNetProj(nn.Module):
    """
    输入：inp = [noisy(1), mask(1), (angle)(1 可选)]
    输出：clean/full (B,1,H,W)
    """
    def __init__(self, in_ch_img=1, add_angle_channel=True,
                 base=64, depth=4, up_mode="nearest", out_act=None):
        super().__init__()
        self.expects_mask = True   # 训练器用到：我吃 (image, mask)
        self.use_angle = bool(add_angle_channel)
        img_c = in_ch_img + (1 if self.use_angle else 0)
        self.depth = int(depth)

        # Encoder
        feat = base
        enc = []
        enc.append(PConvBNAct(img_c, feat, k=7, s=2, p=3, bn=True, act="leaky"))  # e1: /2
        ch = feat
        for _ in range(1, self.depth):
            enc.append(DownP(ch, min(ch*2, base*8)))  # 逐层 /2
            ch = min(ch*2, base*8)
        self.encoders = nn.ModuleList(enc)

        # Bottleneck（不再下采样）
        self.bott = PConvBNAct(ch, ch, k=3, s=1, p=1, bn=True, act="leaky")

        # Decoder：共 depth 次上采样；前 (depth-1) 次带 skip，最后 1 次无 skip
        dec = []
        ch_dec = ch
        for i in range(self.depth-1, -1, -1):
            # i>0 → 有对应 skip；i==0 → 无 skip
            if i > 0:
                skip_c = (self.encoders[i].pconv.out_channels
                          if isinstance(self.encoders[i], PConvBNAct)
                          else self.encoders[i].block.pconv.out_channels)
                in_c = ch_dec + skip_c
                out_c = max(skip_c, base)
            else:
                in_c = ch_dec
                out_c = max(base, base)
            dec.append(UpP(in_c, out_c, mode=up_mode))
            ch_dec = out_c
        self.decoders = nn.ModuleList(dec)

        # Head
        self.head = nn.Conv2d(ch_dec, 1, kernel_size=3, padding=1)
        if out_act == "tanh":
            self.out_act = nn.Tanh()
        elif out_act == "sigmoid":
            self.out_act = nn.Sigmoid()
        else:
            self.out_act = None

    def _split_inputs(self, x):
        # x: (B,C,H,W) ; C=2/3
        if x.size(1) == 2:
            noisy = x[:, 0:1]
            mask  = x[:, 1:2]
            img   = noisy
        elif x.size(1) >= 3:
            noisy = x[:, 0:1]
            mask  = x[:, 1:2]
            angle = x[:, 2:3]
            img   = torch.cat([noisy, angle], dim=1) if self.use_angle else noisy
        else:
            raise ValueError(f"Expect inp channels >=2 (noisy,mask[,angle]), got {x.shape}")
        return img, mask

    def forward(self, inp):
        H0, W0 = inp.shape[-2:]          # 记录输入尺寸
        img, mask = self._split_inputs(inp)

        # Encoder（保存 skip）
        feats, masks = [], []
        x, m = img, mask
        for enc in self.encoders:
            x, m = enc(x, m)
            feats.append(x)
            masks.append(m)

        # Bottleneck
        x, m = self.bott(x, m)

        # Decoder：第 i 次上采样对应 skip 索引 idx = depth-2-i；若 idx<0 则无 skip
        for i, dec in enumerate(self.decoders):
            if i < self.depth - 1:
                enc_idx = self.depth - 1 - i   # 3,2,1 对应 e4,e3,e2（以 depth=4 为例）
                skip_x = feats[enc_idx]
                skip_m = masks[enc_idx]
                x, m = dec(x, m, skip_x, skip_m)
            else:
                # 最后一层不上 skip
                x, m = dec(x, m, None, None)


        # Decoder 循环结束后，确保回到原尺寸
        if x.shape[-2:] != (H0, W0):
            if self.decoders[0].mode in ("bilinear", "bicubic"):
                x = F.interpolate(x, size=(H0, W0), mode=self.decoders[0].mode, align_corners=False)
            else:
                x = F.interpolate(x, size=(H0, W0), mode=self.decoders[0].mode)
            m = F.interpolate(m, size=(H0, W0), mode="nearest")

        out = self.head(x)
        if self.out_act is not None:
            out = self.out_act(out)
        return out
