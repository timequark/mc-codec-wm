"""矫正总管线（路线 A）：定位抠图 -> 精配准 -> 相位相关 -> ECC。

去畸变
  -> 粗定位(降采样特征匹配) 得目标四边形 quad
  -> 校验 quad(内点/凸性/面积/长宽比)；不合格 -> 判"未检出目标"
  -> 按 quad 裁 ROI(去背景)
  -> ROI 与模板精匹配 -> 单应 -> warpPerspective 到模板尺寸(抠图+矫正一步完成)
  -> Phase Correlation 估残余位移 -> warpAffine 补偿
  -> ECC 精配准 -> 亚像素对齐

返回 (aligned 或 None, status)。status 含 detected/stage/inliers/reason，
供上层（web）展示"未检出目标，请重拍"等提示。

关于像素位移（CLAUDE.md 关注点）：warpPerspective 后与模板做相位相关得残余
整体平移(dx,dy)，超阈值则 warpAffine 补偿，再 ECC 消除亚像素残差，
保证 DCT 块网格与嵌入端对齐。
"""

import cv2
import numpy as np

from .. import config
from .undistort import undistort
from .sift import detect_and_match
from .homography import find_homography, warp_perspective
from .phase import estimate_shift, translate
from .ecc import refine_ecc
from .locate import locate_target, validate_quad, crop_roi, _template_corners
from ..utils.logger import get_logger

_log = get_logger("photo_watermark.align")


def _final_ok(H_img2tpl, template_shape, inliers, cfg):
    """校验精配准单应：把模板四角经 H⁻¹ 映回 ROI，检查凸性与内点。"""
    if inliers < cfg.locate_inliers_min:
        return False, f"精配准内点过少({inliers}<{cfg.locate_inliers_min})"
    try:
        Hinv = np.linalg.inv(H_img2tpl)
    except np.linalg.LinAlgError:
        return False, "单应不可逆"
    quad = cv2.perspectiveTransform(
        _template_corners(template_shape), Hinv).reshape(-1, 2)
    if not cv2.isContourConvex(quad.astype(np.float32)):
        return False, "精配准四边形非凸"
    return True, "ok"


def align(photo, template, cfg=None, camera_matrix=None, dist_coeffs=None,
          method="SIFT", use_phase=True, use_ecc=True, shift_tol=0.3):
    """将拍照图中的目标抠出并矫正到与模板对齐。

    Parameters
    ----------
    photo : ndarray           拍照图（可含背景）
    template : ndarray        模板（原始图，与水印无关）
    cfg : Config, optional
    camera_matrix, dist_coeffs : 去畸变标定参数（可选）
    method : str              "SIFT" 或 "ORB"
    use_phase, use_ecc : bool 是否做相位相关补偿 / ECC 精配准
    shift_tol : float         触发平移补偿的像素阈值

    Returns
    -------
    (aligned, status) : aligned 为与模板对齐的图像（尺寸同模板）或 None；
                        status = {detected, stage, inliers, reason}
    """
    cfg = cfg or config.Config()
    h, w = template.shape[:2]
    status = {"detected": False, "stage": None, "inliers": 0, "reason": ""}

    # 1. 去畸变
    img = undistort(photo, camera_matrix, dist_coeffs)

    # 2. 粗定位 + 校验 -> 决定用裁剪 ROI(fine) 还是整图(single)
    roi = img
    stage = "single"
    loc = locate_target(img, template, cfg, method)
    print("loc:", loc)
    if loc is not None:
        ok, reason = validate_quad(
            loc["quad"], img.shape, template.shape, loc["inliers"], cfg)
        if ok:
            roi, _ = crop_roi(img, loc["quad"], cfg.roi_margin)
            stage = "fine"
            _log.info("粗定位成功 inliers=%d，裁剪 ROI %s", loc["inliers"], roi.shape[:2])
        else:
            status["reason"] = reason
            _log.info("粗定位校验未过：%s（回退整图精匹配）", reason)

    # 3. 精匹配（ROI 或整图）-> 单应 -> 抠图+矫正
    pts_roi, pts_tpl = detect_and_match(roi, template, method=method)
    H, mask = find_homography(pts_roi, pts_tpl)
    if H is None:
        status["reason"] = status["reason"] or "匹配点不足，未检出目标"
        _log.warning("未检出目标：%s", status["reason"])
        return None, status
    inliers = int(mask.sum()) if mask is not None else 0

    ok, reason = _final_ok(H, template.shape, inliers, cfg)
    if not ok:
        status["reason"] = reason
        _log.warning("精配准校验未过：%s", reason)
        return None, status

    warped = warp_perspective(roi, H, (w, h))
    status.update(detected=True, stage=stage, inliers=inliers, reason="ok")

    # 4. 相位相关估残余整体位移 -> 平移补偿
    if use_phase:
        dx, dy, resp = estimate_shift(template, warped)
        _log.info("相位相关残余位移 dx=%.3f dy=%.3f (resp=%.3f)", dx, dy, resp)
        if abs(dx) > shift_tol or abs(dy) > shift_tol:
            warped = translate(warped, dx, dy)

    # 5. ECC 精配准 -> 亚像素对齐
    if use_ecc:
        warped = refine_ecc(template, warped)

    return warped, status
