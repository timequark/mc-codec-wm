"""中频系数对选取：依据块尺寸动态计算，不写死。

取块内穿过中心的反对角线 r+c = n-1（既非 DC 低频，也非最高频），
在其上选取跨主对角线、径向频率相等的一对位置 A / B：
    A = (i,   n-1-i)
    B = (i+1, n-2-i),  i = (n-1)//2
这样 |A|、|B| 的频率幅度相同，未嵌入时两系数统计相近，
比较 coef[B] 与 coef[A] 的大小即可稳健判 bit。
"""


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


def read_pair(coeff, pos_a, pos_b):
    """从 DCT 系数矩阵读取 (coef[A], coef[B])。"""
    return coeff[pos_a[0], pos_a[1]], coeff[pos_b[0], pos_b[1]]
