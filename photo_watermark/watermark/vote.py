"""投票：结果频率投票 + 逐 bit 多数投票。"""

from collections import Counter


def majority_result(results: list):
    """对若干条成功解码结果统计频率，返回出现最多的一条；空则 None。"""
    if not results:
        return None
    return Counter(results).most_common(1)[0][0]


def bit_vote(groups: list) -> list:
    """对多份等长 bit 流逐位多数投票，返回单份 bit 流。

    平票（含偶数份对半）时判为 1。
    """
    groups = [g for g in groups if g]
    if not groups:
        return []
    length = min(len(g) for g in groups)
    n = len(groups)
    voted = []
    for i in range(length):
        ones = sum(g[i] for g in groups)
        voted.append(1 if ones * 2 >= n else 0)
    return voted
