"""嵌入示例。

运行方式（项目根目录）：
    python -m photo_watermark.demo.embed_demo
    或   python photo_watermark/demo/embed_demo.py
"""

import sys
from pathlib import Path

# 允许直接以脚本方式运行：把项目根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from photo_watermark.embed import embed


def main():
    embed(
        image_path="images/mkking/mkking-02.png",
        watermark_text="ABCD1234EFGH5678",
        output_path="images/mkking/mkking-02-wm.png",
        block_size=12,
        repl=8,
        delta=60
    )


if __name__ == "__main__":
    main()
