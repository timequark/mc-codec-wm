"""DCT 分块：在 alpha>0 区域按 block_size 自左向右、自上而下划分块网格。"""

import cv2
import numpy as np


def iter_blocks(shape, block_size, mask=None):
    """生成可用块的左上角坐标（左->右、上->下）。

    一个块可用的条件：整块完全落在图像内，且（若给 mask）块内所有像素
    mask 均为 True。

    Parameters
    ----------
    shape : (h, w)            图像尺寸
    block_size : int          块大小
    mask : ndarray(bool), optional  可嵌入掩码；None 表示全图可用

    Yields
    ------
    (y, x)                    块左上角坐标
    """
    h, w = shape[0], shape[1]
    bs = block_size
    for y in range(0, h - bs + 1, bs):
        for x in range(0, w - bs + 1, bs):
            if mask is None or mask[y:y + bs, x:x + bs].all():
                yield (y, x)


def count_blocks(shape, block_size, mask=None) -> int:
    """统计可用块数量。"""
    return sum(1 for _ in iter_blocks(shape, block_size, mask))


def dct2(block):
    """二维 DCT（输入自动转 float32）。"""
    return cv2.dct(np.asarray(block, dtype=np.float32))


def idct2(coeff):
    """二维 IDCT。"""
    return cv2.idct(np.asarray(coeff, dtype=np.float32))
