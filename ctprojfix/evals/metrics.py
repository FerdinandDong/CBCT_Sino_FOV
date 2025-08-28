# ctprojfix/evals/metrics.py
import numpy as np
from scipy.ndimage import gaussian_filter

def _to_float01(x):
    x = np.asarray(x, dtype=np.float32)
    if x.min() < 0.0 or x.max() > 1.0:
        # 这里只是对0-1范围，如果超出可在外面做归一化；这里简单clip
        x = np.clip(x, 0.0, 1.0)
    return x

def psnr(pred, gt, data_range=1.0):
    pred = _to_float01(pred); gt = _to_float01(gt)
    mse = np.mean((pred - gt) ** 2, dtype=np.float64)
    if mse <= 1e-12:
        return 99.0
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)

def ssim(pred, gt, data_range=1.0, sigma=1.5):
    """
    简化版 SSIM（高斯核平滑，单通道 2D），适合 0-1 归一化图像。
    """
    pred = _to_float01(pred); gt = _to_float01(gt)
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu_x = gaussian_filter(pred, sigma=sigma)
    mu_y = gaussian_filter(gt,   sigma=sigma)
    mu_x2, mu_y2 = mu_x**2, mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x2 = gaussian_filter(pred*pred, sigma=sigma) - mu_x2
    sigma_y2 = gaussian_filter(gt*gt,     sigma=sigma) - mu_y2
    sigma_xy = gaussian_filter(pred*gt,   sigma=sigma) - mu_xy

    num = (2*mu_xy + C1) * (2*sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-12)
    return float(np.mean(ssim_map))
