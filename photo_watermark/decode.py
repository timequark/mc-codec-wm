"""解码入口。

流程（见 CLAUDE.md 解码规则）：
    读 Block -> 判 bit(coefB>coefA ->1) -> 按 repl 分组
        -> 每份 ECC：成功记录 / 失败保留
        -> 所有成功结果统计频率取最多者输出
        -> 若全部失败：Bit Vote -> ECC
"""

import cv2

from . import config
from .image import io, alpha as alpha_mod, roi as roi_mod
from .watermark.payload import parse_payload
from .watermark.redundancy import split_groups
from .watermark.vote import majority_result, bit_vote
from .dct.extractor import extract_bits
from .utils.logger import get_logger

_log = get_logger("photo_watermark.decode")


def decode(image_path, block_size=config.DEFAULT_BLOCK_SIZE,
           repl=config.DEFAULT_REPL, mask_path=None, band_mode=None, cfg=None):
    """从图片文件中解码水印文本。

    Parameters
    ----------
    image_path : str          待解码图片（已矫正对齐）
    block_size : int          DCT 块大小 8/12/16（须与嵌入一致）
    repl : int                冗余份数（须与嵌入一致）
    mask_path : str, optional 蒙版路径
    band_mode : str, optional 频带档 "mid"/"low"，须与嵌入端一致
    cfg : Config, optional    参数配置

    Returns
    -------
    str or None               解码水印文本，失败返回 None
    """
    img = io.imread(image_path, with_alpha=True)
    mask_img = io.imread(mask_path, with_alpha=True) if mask_path else None
    return decode_image(img, block_size, repl, mask_img, band_mode, cfg)


def decode_image(img, block_size=config.DEFAULT_BLOCK_SIZE,
                 repl=config.DEFAULT_REPL, mask_img=None, band_mode=None,
                 cfg=None):
    """从图像数组解码水印文本（供 web / 已对齐数组直接调用）。

    Parameters
    ----------
    img : ndarray             待解码图像（已矫正对齐，BGR/BGRA）
    block_size, repl : 见 decode()
    mask_img : ndarray, optional  蒙版图像数组
    band_mode : str, optional 频带档 "mid"/"low"，须与嵌入端一致
    cfg : Config, optional

    Returns
    -------
    str or None
    """
    cfg = cfg or config.Config()
    band_mode = band_mode or cfg.band_mode

    # 1. 可嵌入区域（须与嵌入端一致）
    amask = alpha_mod.get_alpha_mask(img, cfg.alpha_threshold)
    sensitive = None
    if mask_img is not None:
        sensitive = alpha_mod.get_alpha_mask(mask_img, 0)
    roi = roi_mod.build_roi_mask(amask, sensitive)

    # 2. 提取 bit 流
    bgr, _ = io.split_alpha(img)
    y_plane = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    n_needed = repl * cfg.payload_bits
    bits = extract_bits(y_plane, n_needed, block_size, roi, band_mode)

    # 3. 按 repl 分组
    groups = split_groups(bits, cfg.payload_bits)

    # 4. 逐份 ECC，成功者收集
    successes = []
    for g in groups:
        text, ok = parse_payload(g, cfg)
        if ok:
            successes.append(text)

    if successes:
        result = majority_result(successes)
        _log.info("逐份 ECC 成功 %d/%d 份, 结果=%s",
                  len(successes), len(groups), result)
        return result

    # 5. 全部失败 -> Bit Vote -> ECC
    _log.info("逐份 ECC 全失败, 降级 Bit Vote")
    voted = bit_vote(groups)
    text, ok = parse_payload(voted, cfg)
    if ok:
        _log.info("Bit Vote 成功, 结果=%s", text)
        return text
    _log.warning("解码失败")
    return None


if __name__ == "__main__":
    pass
