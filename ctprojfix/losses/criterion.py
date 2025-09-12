# ctprojfix/losses/criterion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- optional deps (safe import) ---
try:
    from pytorch_msssim import ssim
except Exception:
    ssim = None

try:
    import lpips
except Exception:
    lpips = None

try:
    import torchvision.models as models
    from torchvision.models.vgg import VGG16_Weights
except Exception:
    models, VGG16_Weights = None, None


# ---------------- Perceptual (VGG16 features) ----------------
class PerceptualLoss(nn.Module):
    def __init__(self, device, use_pretrained=True):
        super().__init__()
        if models is None:
            raise RuntimeError("torchvision 未安装，无法使用 PerceptualLoss/StyleLoss")
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if use_pretrained else None).features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(16)]).to(device)
        self.slice1.eval()
        for p in self.slice1.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        xf = self.slice1(x)
        yf = self.slice1(y)
        return F.mse_loss(xf, yf)


# ---------------- Style (Gram matrix on VGG features) ----------------
class StyleLoss(nn.Module):
    def __init__(self, perceptual_loss: PerceptualLoss):
        super().__init__()
        self.vgg16_slice = perceptual_loss.slice1  # 共享 VGG 特征提取器

    @staticmethod
    def gram_matrix(feat: torch.Tensor):
        b, c, h, w = feat.size()
        f = feat.view(b, c, h * w)
        gram = torch.bmm(f, f.transpose(1, 2)) / (c * h * w)
        return gram

    def forward(self, x, y):
        x3 = x.repeat(1, 3, 1, 1)
        y3 = y.repeat(1, 3, 1, 1)
        xf = self.vgg16_slice(x3)
        yf = self.vgg16_slice(y3)
        return F.mse_loss(self.gram_matrix(xf), self.gram_matrix(yf))


# ---------------- SSIM ----------------
class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        if ssim is None:
            raise RuntimeError("pytorch_msssim 未安装，无法使用 SSIMLoss")

    def forward(self, x, y):
        return 1.0 - ssim(x, y, data_range=1.0)


# ---------------- Edge (Sobel, buffer version) ----------------
class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1., 0., -1.],
                           [2., 0., -2.],
                           [1., 0., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1., 2., 1.],
                           [0., 0., 0.],
                           [-1., -2., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x, y):
        ex_x = F.conv2d(x, self.kx, padding=1)
        ey_x = F.conv2d(x, self.ky, padding=1)
        ex_y = F.conv2d(y, self.kx, padding=1)
        ey_y = F.conv2d(y, self.ky, padding=1)
        return 0.5 * (F.mse_loss(ex_x, ex_y) + F.mse_loss(ey_x, ey_y))


# ---------------- L1 in hole/valid masked regions ----------------
class L1HoleValidLoss(nn.Module):
    def forward(self, x, y, M):
        hole = torch.mean((1 - M) * torch.abs(x - y))
        valid = torch.mean(M * torch.abs(x - y))
        return hole, valid


# ---------------- LPIPS ----------------
class LPIPSLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        if lpips is None:
            raise RuntimeError("lpips 未安装，无法使用 LPIPS")
        self.lp = lpips.LPIPS(net='alex').to(device).eval()

    def forward(self, x, y):
        x3 = torch.clamp(x.repeat(1, 3, 1, 1), -1.0, 1.0)
        y3 = torch.clamp(y.repeat(1, 3, 1, 1), -1.0, 1.0)
        return self.lp(x3, y3).mean()


# ---------------- L2 ----------------
class L2Loss(nn.Module):
    def forward(self, x, y):
        return F.mse_loss(x, y)


# ---------------- Combined ----------------
class CombinedLoss(nn.Module):
    def __init__(self, device,
                 use_perceptual=False, w_perc=20.0,
                 use_style=False,      w_style=200.0,
                 use_ssim=False,       w_ssim=0.001,
                 use_edge=False,       w_edge=0.4,
                 use_lpips=False,      w_lpips=0.8,
                 w_l1_hole=0.1, w_l1_valid=0.01,
                 use_l2=True,         w_l2=1.0):
        super().__init__()
        self.device = device

        # components
        self.perc = PerceptualLoss(device) if (use_perceptual or use_style) else None
        self.style = StyleLoss(self.perc) if (use_style and self.perc is not None) else None
        self.ssim = SSIMLoss() if use_ssim else None
        self.edge = EdgeLoss().to(device) if use_edge else None
        self.lpips = LPIPSLoss(device) if use_lpips else None
        self.l1_hole_valid = L1HoleValidLoss()
        self.l2 = L2Loss() if use_l2 else None

        # flags & weights
        self.use_perc = use_perceptual
        self.use_style = use_style
        self.use_ssim = use_ssim
        self.use_edge = use_edge
        self.use_lpips = use_lpips
        self.use_l2 = use_l2

        self.w_perc = w_perc
        self.w_style = w_style
        self.w_ssim = w_ssim
        self.w_edge = w_edge
        self.w_lpips = w_lpips
        self.w_l1_h = w_l1_hole
        self.w_l1_v = w_l1_valid
        self.w_l2 = w_l2

    def forward(self, x, y, M):
        loss = 0.0

        if self.use_l2 and self.l2 is not None:
            loss += self.w_l2 * self.l2(x, y)

        if self.use_perc and self.perc is not None:
            loss += self.w_perc * self.perc(x, y)

        if self.use_style and self.style is not None:
            loss += self.w_style * self.style(x, y)

        if self.use_ssim and self.ssim is not None:
            loss += self.w_ssim * self.ssim(x, y)

        if self.use_edge and self.edge is not None:
            loss += self.w_edge * self.edge(x, y)

        if self.l1_hole_valid is not None:
            l1h, l1v = self.l1_hole_valid(x, y, M)
            loss += self.w_l1_h * l1h + self.w_l1_v * l1v

        if self.use_lpips and self.lpips is not None:
            loss += self.w_lpips * self.lpips(x, y)

        return loss


def build_criterion_from_cfg(device, loss_cfg: dict):
    t = (loss_cfg.get("type", "l2") or "l2").lower()
    if t == "combined":
        return CombinedLoss(device, **(loss_cfg.get("opts", {})))
    elif t == "l1":
        return nn.L1Loss()
    elif t == "l2":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {t}")
