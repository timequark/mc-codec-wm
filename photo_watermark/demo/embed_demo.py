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

pic_tag = '03'

def main():
    embed(
        image_path=f"images/mkking/mkking-{pic_tag}.png",
        watermark_text="ABCD1234EFGH5678",
        output_path=f"images/mkking/mkking-{pic_tag}-wm-low.png",
        block_size=12,
        repl=30,
        delta=200,
        band_mode="low"
    )


if __name__ == "__main__":
    main()
