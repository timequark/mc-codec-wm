"""冗余复制与分组。

嵌入：单份 bit 流顺序重复 repl 份铺开。
解码：把提取到的 bit 流按单份长度切成 repl 组。
"""


def replicate(bits: list, repl: int) -> list:
    """将单份 bits 重复 repl 份拼接。"""
    if repl < 1:
        raise ValueError("repl 须 >= 1")
    return list(bits) * repl


def split_groups(bits: list, unit_len: int) -> list:
    """按 unit_len 切分成若干份；丢弃末尾不足一份的残余。"""
    if unit_len < 1:
        raise ValueError("unit_len 须 >= 1")
    n = len(bits) // unit_len
    return [bits[i * unit_len:(i + 1) * unit_len] for i in range(n)]
