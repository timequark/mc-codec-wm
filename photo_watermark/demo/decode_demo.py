"""解码示例。

运行方式（项目根目录）：
    python -m photo_watermark.demo.decode_demo
    或   python photo_watermark/demo/decode_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from photo_watermark.decode import decode

pic_tag = '03'

def main():
    input_path = f"images/mkking/mkking-{pic_tag}-wm-low.png"
    # input_path = "photo_watermark/web/uploads/2026-07-08_18-41-56.png"
    text = decode(image_path=input_path, block_size=12, repl=30)
    print("decoded:", text)


if __name__ == "__main__":
    main()
