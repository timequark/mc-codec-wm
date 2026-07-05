"""Alpha 通道处理：提取 alpha>threshold 的可嵌入区域掩码。"""

import numpy as np

from .. import config


def get_alpha_mask(image, threshold=config.ALPHA_THRESHOLD):
    """返回 alpha > threshold 的布尔掩码；无 alpha 通道则全 True。"""
    h, w = image.shape[:2]
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, 3] > threshold
    return np.ones((h, w), dtype=bool)
