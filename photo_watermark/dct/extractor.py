"""块级提取：逐块读出 bit 流（顺序与 embedder 完全一致）。"""

import numpy as np

from .block import iter_blocks, dct2
from .coeff import mid_band_pair
from .strategy1 import extract_bit


def extract_bits(plane, n_bits=None, block_size=8, mask=None) -> list:
    """从 plane 的可用 DCT 块提取 bit 流。

    Parameters
    ----------
    plane : ndarray(float)    亮度平面
    n_bits : int, optional    需提取的 bit 数；None 表示取满所有可用块
    block_size : int          块大小
    mask : ndarray(bool), optional  可嵌入掩码

    Returns
    -------
    list                      提取的 bit 流
    """
    plane = np.asarray(plane, dtype=np.float32)
    pos_a, pos_b = mid_band_pair(block_size)
    bs = block_size
    bits = []
    for (y, x) in iter_blocks(plane.shape, bs, mask):
        if n_bits is not None and len(bits) >= n_bits:
            break
        coeff = dct2(plane[y:y + bs, x:x + bs])
        bits.append(extract_bit(coeff, pos_a, pos_b))
    return bits
