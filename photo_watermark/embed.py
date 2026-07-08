"""嵌入入口。

流程：
    水印文本 -> RS-ECC -> bit 流 -> 按 repl 冗余复制
        -> alpha>0 区域按 block_size 划分 DCT 块
        -> 逐块用策略1 嵌入 1 bit -> 输出图片
"""

import cv2
import numpy as np

from . import config
from .image import io, alpha as alpha_mod, roi as roi_mod
from .watermark.payload import build_payload
from .watermark.redundancy import replicate
from .dct.block import count_blocks
from .dct.embedder import embed_bits
from .utils.logger import get_logger

_log = get_logger("photo_watermark.embed")


def embed(image_path, watermark_text, output_path,
          block_size=config.DEFAULT_BLOCK_SIZE, mask_path=None,
          repl=config.DEFAULT_REPL, delta=config.DELTA, band_mode=None,
          cfg=None):
    """将水印嵌入图片并保存。

    Parameters
    ----------
    image_path : str          原图路径
    watermark_text : str      16 位字母或数字水印
    output_path : str         输出图片路径（建议 PNG 无损）
    block_size : int          DCT 块大小 8/12/16
    mask_path : str, optional 蒙版路径（alpha==0 区域跳过）
    repl : int                冗余份数
    delta : float             能量强度
    band_mode : str, optional 频带档 "mid"/"low"，None 取 cfg.band_mode
    cfg : Config, optional    参数配置

    Returns
    -------
    ndarray                   嵌入后的图像
    """
    cfg = cfg or config.Config()
    band_mode = band_mode or cfg.band_mode
    img = io.imread(image_path, with_alpha=True)

    # 1. 可嵌入区域
    amask = alpha_mod.get_alpha_mask(img, cfg.alpha_threshold)
    sensitive = None
    if mask_path:
        sm = io.imread(mask_path, with_alpha=True)
        sensitive = alpha_mod.get_alpha_mask(sm, 0)
    roi = roi_mod.build_roi_mask(amask, sensitive)

    # 2. 载荷 bit 流（含冗余）
    bits = replicate(build_payload(watermark_text, cfg), repl)
    n_blocks = count_blocks(img.shape[:2], block_size, roi)
    if len(bits) > n_blocks:
        raise ValueError(
            f"容量不足: 需 {len(bits)} 块, 仅有 {n_blocks} 块 "
            f"(单份 {cfg.payload_bits} bit x repl {repl})")
    _log.info("嵌入 %d bit 到 %d 可用块 (block_size=%d, repl=%d)",
              len(bits), n_blocks, block_size, repl)

    # 3. 在亮度平面嵌入
    bgr, alpha_ch = io.split_alpha(img)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ycrcb[:, :, 0] = np.clip(
        embed_bits(ycrcb[:, :, 0], bits, block_size, roi, delta, band_mode),
        0, 255)
    out_bgr = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    out = io.merge_alpha(out_bgr, alpha_ch)

    # io.imwrite(output_path, out)
    # 保存为 600 DPI
    from PIL import Image

    if out.shape[2] == 4:
        # BGRA -> RGBA
        out_save = cv2.cvtColor(out, cv2.COLOR_BGRA2RGBA)
        Image.fromarray(out_save).save(
            output_path,
            dpi=(600, 600),
            compress_level=0      # PNG，无损
        )
    else:
        # BGR -> RGB
        out_save = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        Image.fromarray(out_save).save(
            output_path,
            dpi=(600, 600),
            compress_level=0
        )

    return out


if __name__ == "__main__":
    pass
