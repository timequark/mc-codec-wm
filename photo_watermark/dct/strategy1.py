"""策略1：中频系数对 A/B 的能量差调制。

bit == 1: 使 coef[A] + delta <= coef[B]   （A->B 方向增大 delta）
bit == 0: 使 coef[A] - delta >= coef[B]   （A->B 方向减小 delta）

对称调整（保持两系数均值不变）以最小化画质损失。
解码不用 delta：coef[B] > coef[A] -> 1，否则 -> 0。

sites 为一个或多个系数对（见 coeff.mid_band_sites）：
嵌入时每对都写同一 bit；解码时每对硬判决后多数投票得 1 bit。
单对时与原硬判决完全等价。
"""


def _embed_one(coeff, bit, pos_a, pos_b, delta):
    """在单个系数对 (A,B) 上就地嵌入 1 bit。"""
    a = float(coeff[pos_a[0], pos_a[1]])
    b = float(coeff[pos_b[0], pos_b[1]])
    mean = 0.5 * (a + b)
    half = 0.5 * delta
    if bit:
        # 需要 b - a >= delta
        if b - a < delta:
            a, b = mean - half, mean + half
    else:
        # 需要 a - b >= delta
        if a - b < delta:
            a, b = mean + half, mean - half
    coeff[pos_a[0], pos_a[1]] = a
    coeff[pos_b[0], pos_b[1]] = b


def embed_bit(coeff, bit, sites, delta):
    """就地按策略1 在 coeff 的所有系数对上嵌入同一 bit，返回 coeff。"""
    for pos_a, pos_b in sites:
        _embed_one(coeff, bit, pos_a, pos_b, delta)
    return coeff


def extract_bit(coeff, sites) -> int:
    """每对硬判决 coef[B] > coef[A]，多数投票得 1 bit（单对时即原硬判决）。"""
    votes = sum(1 for pos_a, pos_b in sites
                if coeff[pos_b[0], pos_b[1]] > coeff[pos_a[0], pos_a[1]])
    return 1 if votes * 2 >= len(sites) else 0
