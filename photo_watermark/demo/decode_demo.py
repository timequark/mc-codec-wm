"""解码示例。

运行方式（项目根目录）：
    python -m photo_watermark.demo.decode_demo
    或   python photo_watermark/demo/decode_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from photo_watermark.decode import decode


def main():
    text = decode(image_path="images/mkking/mkking-02-wm.png", block_size=12, repl=8)
    print("decoded:", text)


if __name__ == "__main__":
    main()
