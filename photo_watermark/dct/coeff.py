"""中频系数对选取：依据块尺寸动态计算，不写死。

取块内穿过中心的反对角线 r+c = n-1（既非 DC 低频，也非最高频），
在其上选取跨主对角线、径向频率相等的一对位置 A / B：
    A = (i,   n-1-i)
    B = (i+1, n-2-i),  i = (n-1)//2
这样 |A|、|B| 的频率幅度相同，未嵌入时两系数统计相近，
比较 coef[B] 与 coef[A] 的大小即可稳健判 bit。

当 block_size >= config.MULTI_SITE_MIN_BLOCK（默认 24）时，块内空间足够，
沿反对角线（频率轴 d = r+c）在中频窗口 [n/2, 3n/2] 内三等分，取 3 对系数，
每对写同一 bit、解码多数投票，提升块内鲁棒性（见 mid_band_sites）。
"""

from .. import config


def mid_band_pair(block_shape):
    """返回中频系数对 (A, B)。

    Parameters
    ----------
    block_shape : int 或 (n, n)   块尺寸或块形状

    Returns
    -------
    (A, B) : 两个 (row, col) 坐标
    """
    n = block_shape if isinstance(block_shape, int) else block_shape[0]
    if n < 2:
        raise ValueError("块尺寸须 >= 2")
    d = n - 1                    # 反对角线：r + c = n-1，穿过中心且避开 DC
    i = (n - 1) // 2
    j = i + 1 if i + 1 <= d else i - 1
    a = (i, d - i)
    b = (j, d - j)
    return a, b


def _pair_on_diag(d, n):
    """反对角线 r+c=d 上取一对跨主对角线的相邻镜像系数 (A, B)，裁剪到块内。"""
    i = d // 2

    def clamp(p):
        return (min(max(p[0], 0), n - 1), min(max(p[1], 0), n - 1))

    return clamp((i, d - i)), clamp((i + 1, d - i - 1))


def mid_band_sites(block_shape):
    """返回块内的中频系数对列表。

    block_size < MULTI_SITE_MIN_BLOCK 时返回单对（与 mid_band_pair 一致）；
    否则沿频率轴 d=r+c 的中频窗口 [n/2, 3n/2] 三等分，返回 MULTI_SITE_COUNT 对，
    各对分布在低中频/中频/高中频，供块内同 bit 冗余 + 多数投票。

    Returns
    -------
    list[(A, B)]
    """
    n = block_shape if isinstance(block_shape, int) else block_shape[0]
    if n < config.MULTI_SITE_MIN_BLOCK:
        return [mid_band_pair(n)]
    d_lo, d_hi = n // 2, 3 * n // 2
    k_n = config.MULTI_SITE_COUNT
    return [_pair_on_diag(d_lo + int(round((d_hi - d_lo) * (k + 0.5) / k_n)), n)
            for k in range(k_n)]


def read_pair(coeff, pos_a, pos_b):
    """从 DCT 系数矩阵读取 (coef[A], coef[B])。"""
    return coeff[pos_a[0], pos_a[1]], coeff[pos_b[0], pos_b[1]]
