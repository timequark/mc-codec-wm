"""质量与鲁棒性指标：PSNR / SSIM / 误码率。"""

import numpy as np
from scipy.ndimage import gaussian_filter


def _gray_f64(img):
    if img.ndim == 3:
        img = img[:, :, :3].mean(axis=2)
    return img.astype(np.float64)


def psnr(img_a, img_b, data_range=255.0):
    """峰值信噪比 (dB)。完全相同返回 inf。"""
    a = np.asarray(img_a, np.float64)
    b = np.asarray(img_b, np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def ssim(img_a, img_b, data_range=255.0, sigma=1.5):
    """高斯加权全局 SSIM（单通道灰度）。"""
    a = _gray_f64(img_a)
    b = _gray_f64(img_b)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_a = gaussian_filter(a, sigma)
    mu_b = gaussian_filter(b, sigma)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    va = gaussian_filter(a * a, sigma) - mu_a2
    vb = gaussian_filter(b * b, sigma) - mu_b2
    vab = gaussian_filter(a * b, sigma) - mu_ab

    s = ((2 * mu_ab + c1) * (2 * vab + c2)) / \
        ((mu_a2 + mu_b2 + c1) * (va + vb + c2))
    return float(s.mean())


def ber(bits_a, bits_b):
    """误码率 (bit error rate)，按较短长度比较。"""
    n = min(len(bits_a), len(bits_b))
    if n == 0:
        return 0.0
    errs = sum(1 for i in range(n) if bits_a[i] != bits_b[i])
    return errs / n
