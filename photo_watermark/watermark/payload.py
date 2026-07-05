"""水印载荷：文本 <-> 单份 bit 流（含 RS-ECC）。"""

from .. import config
from .rs_codec import rs_encode, rs_decode
from .bitstream import bytes_to_bits, bits_to_bytes


def build_payload(text: str, cfg: config.Config = None) -> list:
    """水印文本 -> RS 编码 -> bit 流（单份，长度 = cfg.payload_bits）。"""
    cfg = cfg or config.Config()
    data = text.encode("ascii")
    if len(data) != cfg.watermark_len:
        raise ValueError(
            f"水印长度须为 {cfg.watermark_len}，实际 {len(data)}")
    codeword = rs_encode(data, cfg.rs_ecc_bytes)
    return bytes_to_bits(codeword)


def _is_valid_watermark(text: str, cfg: config.Config) -> bool:
    """水印须为 cfg.watermark_len 位 ASCII 字母或数字。

    用于拒绝"垃圾比特恰好通过 RS 校验"的误判（如全 0 码字本身即合法码字）。
    """
    return len(text) == cfg.watermark_len and text.isascii() and text.isalnum()


def parse_payload(bits: list, cfg: config.Config = None):
    """单份 bit 流 -> RS 解码 -> (text, ok)。

    仅当 RS 校验通过且结果为合法水印（16 位字母数字）时才算成功，
    避免误判。
    """
    cfg = cfg or config.Config()
    codeword = bits_to_bytes(bits[:cfg.payload_bits])
    data, ok = rs_decode(codeword, cfg.rs_ecc_bytes)
    if not ok:
        return None, False
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        return None, False
    if not _is_valid_watermark(text, cfg):
        return None, False
    return text, True
