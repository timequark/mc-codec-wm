"""中频系数对选取：依据块尺寸动态计算，不写死。

基本单元 mid_band_pair：在一条反对角线 r+c=d 上取跨主对角线、径向频率相近的
一对位置 A / B（|A|≈|B|，未嵌入时两系数统计相近），比较 coef[B] 与 coef[A]
的大小即可稳健判 bit；其默认取穿过中心的 r+c=n-1（既非 DC 低频，也非最高频）。

mid_band_sites：沿频率轴 d=r+c 在中频窗口 [n/2, 3n/2] 内等分，返回多对系数——
小块（< MULTI_SITE_MIN_BLOCK）2 对、大块 3 对，分布在低中频/中频/高中频。
嵌入/解码端按块序 i % 对数 轮换选用其中一对（见 embedder / extractor），
每块仍只嵌 1 bit，借块间位置轮换打散相邻块的规律。
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


def _band_window(n, band_mode):
    """按频带模式给出频率轴 d=r+c 的取值窗口 (d_lo, d_hi)。"""
    if band_mode == "low":
        # 低频窗口 [n/5, n/2]：抗印刷拍照的低通信道，代价是低频含更多图像内容
        d_lo = max(2, n // 5)
        return d_lo, max(d_lo + 1, n // 2)
    # "mid"（默认）：中频窗口 [n/2, 3n/2]，数字域最优
    return n // 2, 3 * n // 2


def mid_band_sites(block_shape, band_mode=None):
    """返回块内的系数对列表（频带由 band_mode 决定）。

    沿频率轴 d=r+c 在所选频带窗口内等分：
    小块（< MULTI_SITE_MIN_BLOCK）返回 MULTI_SITE_COUNT_SMALL 对，
    大块返回 MULTI_SITE_COUNT 对，各对分布在窗口内低/中/高处。
    嵌入/解码端按块序轮换选用其中一对（每块仍只嵌 1 bit）。

    band_mode : "mid"（默认，中频）/ "low"（低频，抗印刷拍照）；
                None 时取 config.BAND_MODE。嵌入与解码须用同一档。

    Returns
    -------
    list[(A, B)]
    """
    n = block_shape if isinstance(block_shape, int) else block_shape[0]
    band_mode = band_mode or config.BAND_MODE
    k_n = (config.MULTI_SITE_COUNT if n >= config.MULTI_SITE_MIN_BLOCK
           else config.MULTI_SITE_COUNT_SMALL)
    d_lo, d_hi = _band_window(n, band_mode)
    return [_pair_on_diag(d_lo + int(round((d_hi - d_lo) * (k + 0.5) / k_n)), n)
            for k in range(k_n)]


def read_pair(coeff, pos_a, pos_b):
    """从 DCT 系数矩阵读取 (coef[A], coef[B])。"""
    return coeff[pos_a[0], pos_a[1]], coeff[pos_b[0], pos_b[1]]
