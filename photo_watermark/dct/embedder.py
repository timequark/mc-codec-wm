"""块级嵌入：把 bit 流逐块写入亮度平面的 DCT 系数。"""

import numpy as np

from .block import iter_blocks, dct2, idct2
from .coeff import mid_band_pair
from .strategy1 import embed_bit


def embed_bits(plane, bits, block_size, mask=None, delta=12.0):
    """将 bits 依次嵌入 plane 的可用 DCT 块。

    Parameters
    ----------
    plane : ndarray(float)    亮度平面（会被复制，不就地修改入参）
    bits : list               待嵌入 bit 流
    block_size : int          块大小 8/12/16
    mask : ndarray(bool), optional  可嵌入掩码
    delta : float             能量强度

    Returns
    -------
    ndarray(float)            嵌入后的平面
    """
    out = np.array(plane, dtype=np.float32, copy=True)
    pos_a, pos_b = mid_band_pair(block_size)
    bs = block_size
    it = iter_blocks(out.shape, bs, mask)
    for bit in bits:
        try:
            y, x = next(it)
        except StopIteration:
            raise ValueError("可用块不足以容纳全部 bit")
        coeff = dct2(out[y:y + bs, x:x + bs])
        embed_bit(coeff, bit, pos_a, pos_b, delta)
        out[y:y + bs, x:x + bs] = idct2(coeff)
    return out
