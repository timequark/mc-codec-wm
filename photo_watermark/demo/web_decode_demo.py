"""测试 web/app.py 的 run_decode：输入一张拍摄的图像。

复用 app 的解码入口（直接解码 -> 失败则矫正后解码），
只是把 web 请求换成本地脚本调用，方便离线调试拍照图。

运行方式（项目根目录）：
    python -m photo_watermark.demo.web_decode_demo
    或   python photo_watermark/demo/web_decode_demo.py

说明：
    template 用【原始图】mkking-03.png（与水印无关），作为矫正模板与 ROI；
    photo 用拍摄后经服务保存的上传图（web/uploads 下）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2

from photo_watermark.web import app

# 模板（原始图，非 -wm）与拍摄图
TEMPLATE_PATH = "images/mkking/mkking-03.png"
PHOTO_PATH = "photo_watermark/web/uploads/2026-07-08_18-41-56.png"

BLOCK_SIZE = 12
REPL = 30


def main():
    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)
    if template is None:
        raise FileNotFoundError(f"无法读取模板: {TEMPLATE_PATH}")
    photo = cv2.imread(PHOTO_PATH, cv2.IMREAD_UNCHANGED)
    if photo is None:
        raise FileNotFoundError(f"无法读取拍摄图: {PHOTO_PATH}")

    # run_decode 从 app.STATE 读取模板，这里手动注入（等价于 app.main 的初始化）
    app.STATE["template"] = template
    app.STATE["template_path"] = TEMPLATE_PATH
    app.STATE["block_size"] = BLOCK_SIZE
    app.STATE["repl"] = REPL

    print(f"template={TEMPLATE_PATH} {template.shape}")
    print(f"photo   ={PHOTO_PATH} {photo.shape}")

    photoname = Path(PHOTO_PATH).name[:Path(PHOTO_PATH).name.rfind(".")]
    result = app.run_decode(photo, BLOCK_SIZE, REPL, photoname=photoname, use_align=True)

    print("-" * 40)
    print("ok    :", result["ok"])
    print("method:", result["method"])
    print("text  :", result["text"])
    if result.get("stage"):
        print("stage :", result["stage"])
    if result.get("reason"):
        print("reason:", result["reason"])


if __name__ == "__main__":
    main()
