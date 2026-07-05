"""可视化：块网格、特征匹配连线等调试图。"""

import cv2
import numpy as np

from ..dct.block import iter_blocks


def draw_block_grid(image, block_size, mask=None, color=(0, 255, 0)):
    """在图上绘制可用 DCT 块网格，返回 BGR 可视化图。"""
    vis = image[:, :, :3].copy() if image.ndim == 3 else \
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bs = block_size
    for (y, x) in iter_blocks(vis.shape[:2], bs, mask):
        cv2.rectangle(vis, (x, y), (x + bs - 1, y + bs - 1), color, 1)
    return vis


def draw_matches(template, photo, pts_a, pts_b):
    """并排绘制特征匹配连线，返回可视化图。"""
    ta = template[:, :, :3] if template.ndim == 3 else \
        cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    pb = photo[:, :, :3] if photo.ndim == 3 else \
        cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
    h = max(ta.shape[0], pb.shape[0])
    canvas = np.zeros((h, ta.shape[1] + pb.shape[1], 3), np.uint8)
    canvas[:ta.shape[0], :ta.shape[1]] = ta
    canvas[:pb.shape[0], ta.shape[1]:] = pb
    off = ta.shape[1]
    for (ax, ay), (bx, by) in zip(pts_a, pts_b):
        cv2.circle(canvas, (int(ax), int(ay)), 3, (0, 255, 0), -1)
        cv2.circle(canvas, (int(bx) + off, int(by)), 3, (0, 255, 0), -1)
        cv2.line(canvas, (int(ax), int(ay)),
                 (int(bx) + off, int(by)), (0, 200, 255), 1)
    return canvas
