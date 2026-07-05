"""图像读写与通道拆分（保留 alpha）。"""

import cv2
import numpy as np


def imread(path, with_alpha=True):
    """读取图像，返回 ndarray（BGR / BGRA / 灰度）。"""
    flag = cv2.IMREAD_UNCHANGED if with_alpha else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def imwrite(path, image):
    """保存图像。"""
    if not cv2.imwrite(str(path), image):
        raise IOError(f"无法写入图像: {path}")


def split_alpha(image):
    """拆出 (bgr, alpha)。无 alpha 时 alpha 为 None；灰度图升为 BGR。"""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
    if image.shape[2] == 4:
        return image[:, :, :3].copy(), image[:, :, 3].copy()
    return image[:, :, :3].copy(), None


def merge_alpha(bgr, alpha):
    """把 alpha 合并回 BGR；alpha 为 None 时原样返回。"""
    if alpha is None:
        return bgr
    return np.dstack([bgr, alpha])
