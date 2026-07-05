"""Zigzag 顺序：块内 (row,col) <-> zigzag 序号，用于定位中频带。"""


def zigzag_order(n: int) -> list:
    """返回 n x n 块的 zigzag 顺序坐标列表 [(r,c), ...]，共 n*n 项。"""
    coords = []
    for s in range(2 * n - 1):
        # s = r + c 反对角线；偶数向上、奇数向下走
        if s % 2 == 0:
            r = min(s, n - 1)
            c = s - r
            while r >= 0 and c <= n - 1:
                coords.append((r, c))
                r -= 1
                c += 1
        else:
            c = min(s, n - 1)
            r = s - c
            while c >= 0 and r <= n - 1:
                coords.append((r, c))
                r += 1
                c -= 1
    return coords


def zigzag_index(rc, n: int) -> int:
    """(row,col) -> zigzag 序号。"""
    return zigzag_order(n).index(tuple(rc))
