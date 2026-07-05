from PIL import Image
from pathlib import Path


def resize_by_width(
    input_file,
    output_file,
    target_width,
    dpi=300,
    resample=Image.Resampling.LANCZOS,
):
    """
    按指定宽度等比例缩放图片，并保存指定DPI。

    Parameters
    ----------
    input_file : str
        输入图片

    output_file : str
        输出图片

    target_width : int
        缩放后的宽度（像素）

    dpi : int or tuple
        保存DPI，例如：
            300
            (300,300)

    resample :
        缩放算法
        Image.Resampling.LANCZOS（推荐）
        Image.Resampling.BICUBIC
        Image.Resampling.BILINEAR
        Image.Resampling.NEAREST
    """

    img = Image.open(input_file)

    src_w, src_h = img.size

    # 等比例计算高度
    scale = target_width / src_w
    target_height = int(round(src_h * scale))

    img_resize = img.resize(
        (target_width, target_height),
        resample=resample
    )

    # dpi参数
    if isinstance(dpi, int):
        dpi = (dpi, dpi)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    img_resize.save(output_file, dpi=dpi)

    print("原图尺寸 :", (src_w, src_h))
    print("目标尺寸 :", img_resize.size)
    print("保存DPI :", dpi)