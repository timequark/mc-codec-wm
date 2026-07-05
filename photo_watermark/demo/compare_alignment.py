"""矫正对比示例：拍照图经矫正管线后与模板对齐。

运行方式（项目根目录）：
    python -m photo_watermark.demo.compare_alignment
    或   python photo_watermark/demo/compare_alignment.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from photo_watermark.align.pipeline import align
from photo_watermark.image import io

def build_path(fname):
    return os.path.join(os.path.dirname(__file__), "..", "..", fname)

def main():
    template = io.imread(build_path("images/mkking/mkking-02.png"))
    photo = io.imread(build_path("images/mkking/mkking-02-cap-01.png"))
    aligned, status = align(photo, template)
    print("align status:", status)
    if aligned is not None:
        io.imwrite("images/mkking/mkking-02-aligned.png", aligned)
    else:
        print("未检出目标")


if __name__ == "__main__":
    main()
