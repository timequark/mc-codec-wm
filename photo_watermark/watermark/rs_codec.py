"""ReedSolomon 编解码：水印(16字节) + ECC(16字节)。"""

from reedsolo import RSCodec, ReedSolomonError

from .. import config

_cache = {}


def _codec(ecc_bytes: int) -> RSCodec:
    rsc = _cache.get(ecc_bytes)
    if rsc is None:
        rsc = RSCodec(ecc_bytes)
        _cache[ecc_bytes] = rsc
    return rsc


def rs_encode(data: bytes, ecc_bytes: int = config.RS_ECC_BYTES) -> bytes:
    """对 data 追加 ECC，返回 data+ecc（长度 = len(data)+ecc_bytes）。"""
    return bytes(_codec(ecc_bytes).encode(bytes(data)))


def rs_decode(codeword: bytes, ecc_bytes: int = config.RS_ECC_BYTES):
    """校验纠错。返回 (data, ok)；ok=False 表示不可纠错。"""
    try:
        decoded = _codec(ecc_bytes).decode(bytes(codeword))[0]
        return bytes(decoded), True
    except ReedSolomonError:
        return None, False
