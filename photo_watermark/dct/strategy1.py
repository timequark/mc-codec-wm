"""策略1：中频系数对 A/B 的能量差调制。

bit == 1: 使 coef[A] + delta <= coef[B]   （A->B 方向增大 delta）
bit == 0: 使 coef[A] - delta >= coef[B]   （A->B 方向减小 delta）

对称调整（保持两系数均值不变）以最小化画质损失。
解码不用 delta：coef[B] > coef[A] -> 1，否则 -> 0。
"""


def embed_bit(coeff, bit, pos_a, pos_b, delta):
    """就地按策略1 在 coeff 上嵌入 1 bit，返回 coeff。"""
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
    return coeff


def extract_bit(coeff, pos_a, pos_b) -> int:
    """硬判决：coef[B] > coef[A] -> 1，否则 -> 0。"""
    a = coeff[pos_a[0], pos_a[1]]
    b = coeff[pos_b[0], pos_b[1]]
    return 1 if b > a else 0
