"""目标定位与单应校验（路线 A：内容特征抠图+矫正）。

拍照图里目标只占一部分、周围是背景。本模块负责：
1) 粗定位：降采样后与模板特征匹配，估计单应，把模板四角映射回拍照图，
   得到目标四边形 quad（拍照图坐标系）；
2) 校验：内点数 / 凸性 / 面积占比 / 长宽比，判断是否真的检出目标；
3) 裁剪：按 quad 外扩 margin 裁出 ROI，去掉大部分背景，供精配准使用。

抠图与矫正本质是同一步——最终 warpPerspective 到模板尺寸即完成，
本模块只是先把目标从背景中定位/裁出来，提升精配准的稳健性与精度。
"""

import cv2
import numpy as np

from .. import config
from .sift import detect_and_match
from .homography import find_homography


def _resize_max(img, max_side):
    """按最长边缩放到 max_side 以内，返回 (small, scale)。scale<=1。"""
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img, 1.0
    small = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                       interpolation=cv2.INTER_AREA)
    return small, s


def _template_corners(template_shape):
    h, w = template_shape[:2]
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)


def _quad_aspect(quad):
    """quad 顺序为 [TL, TR, BR, BL]，返回宽/高。"""
    tl, tr, br, bl = quad
    # tr - tl = [x_tr - x_tl,  y_tr - y_tl]，得到 tl 指向 tr 的向量
    # np.linalg.norm(tr - tl) 求向量模长（L2 范数），即勾股定理，即上边的长度
    # np.linalg.norm(br - bl) 求向量模长（L2 范数），即勾股定理，即下边的长度
    # 取上边和下边的平均值作为宽度
    width = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    height = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    return width / height if height > 1e-6 else 0.0


def locate_target(photo, template, cfg=None, method="SIFT"):
    """粗定位目标四边形。

    Returns
    -------
    dict {quad(Nx2), inliers, scale} 或 None（匹配点/单应不足）
    """
    cfg = cfg or config.Config()
    small, scale = _resize_max(photo, cfg.coarse_max_side)

    # 模板 -> 缩小拍照图（便于把模板四角映射到拍照图）
    pts_small, pts_tpl = detect_and_match(small, template, method=method)
    if len(pts_tpl) < 4:
        return None
    H_tpl2small, mask = find_homography(pts_tpl, pts_small)
    if H_tpl2small is None:
        return None

    inliers = int(mask.sum()) if mask is not None else 0
    # "模板的四个角"通过单应矩阵投影到拍照图上，从而框出目标在照片里的位置
    quad_small = cv2.perspectiveTransform(
        _template_corners(template.shape),
        H_tpl2small
    ).reshape(-1, 2)
    # 换算回全分辨率拍照图
    quad = quad_small / scale
    return {"quad": quad, "inliers": inliers, "scale": scale}


def validate_quad(quad, photo_shape, template_shape, inliers, cfg=None):
    """校验目标四边形是否合理。返回 (ok, reason)。"""
    cfg = cfg or config.Config()

    if inliers < cfg.locate_inliers_min:
        return False, f"匹配内点过少({inliers}<{cfg.locate_inliers_min})"

    q = quad.astype(np.float32)
    if not cv2.isContourConvex(q):
        return False, "目标四边形非凸(透视估计异常)"

    area = abs(cv2.contourArea(q))
    frame = float(photo_shape[0] * photo_shape[1])
    ratio = area / frame if frame > 0 else 0.0
    if ratio < cfg.quad_area_min:
        return False, f"目标过小(占画面{ratio:.3%})，请拉近"
    if ratio > cfg.quad_area_max:
        return False, f"目标异常过大(占画面{ratio:.3%})"

    tpl_aspect = template_shape[1] / template_shape[0]
    q_aspect = _quad_aspect(quad)
    if tpl_aspect > 0 and abs(q_aspect - tpl_aspect) / tpl_aspect > cfg.quad_aspect_tol:
        return False, f"长宽比偏差过大(目标{q_aspect:.2f} vs 模板{tpl_aspect:.2f})"

    return True, "ok"


def crop_roi(photo, quad, margin):
    """按 quad 外接矩形外扩 margin 裁剪，返回 (roi, (x0, y0))。"""
    h, w = photo.shape[:2]
    x, y, ww, hh = cv2.boundingRect(quad.astype(np.float32))
    m = int(round(margin * max(ww, hh)))
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(w, x + ww + m), min(h, y + hh + m)
    return photo[y0:y1, x0:x1], (x0, y0)
