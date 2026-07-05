"""字节 <-> bit 列表转换（MSB first）。"""


def bytes_to_bits(data: bytes) -> list:
    """bytes -> [0/1, ...]，每字节高位在前。"""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_bytes(bits: list) -> bytes:
    """[0/1, ...] -> bytes，每 8 bit 一字节，高位在前。

    长度不足 8 的余数按缺高位补 0 处理。
    """
    out = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        byte = 0
        for b in chunk:
            byte = (byte << 1) | (b & 1)
        # 补齐末尾不足 8 位
        byte <<= (8 - len(chunk))
        out.append(byte)
    return bytes(out)
