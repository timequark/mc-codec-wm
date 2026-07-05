"""ECC 精配准（cv2.findTransformECC），达到亚像素级对齐。"""

import cv2
import numpy as np


def _to_gray_f32(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return np.float32(img) / 255.0


def refine_ecc(template, image, warp_mode=cv2.MOTION_TRANSLATION,
               iterations=50, eps=1e-4):
    """以 template 为基准精配准 image，返回对齐后图像。

    warp_mode 可选 MOTION_TRANSLATION / MOTION_EUCLIDEAN /
    MOTION_AFFINE / MOTION_HOMOGRAPHY。失败时返回原图副本。

    findTransformECC 函数实现了一种基于区域的对齐方法，该方法基于强度相似性。本质上，该函数更新初始变换，
    该变换大致对齐图像。如果缺少这些信息，则使用单位变换（单位矩阵）作为初始化。注意，如果图像经历了强烈的位移/旋转，
    需要一个初始变换来大致对齐图像（例如，一个简单的欧氏变换/相似变换，允许图像显示大致相同的内容）。在第二幅图像中使用逆向变形，以使图像接近第一幅图像，
    即在使用 warpAffine 或 warpPerspective 时使用 WARP_INVERSE_MAP 标志。
    """
    t = _to_gray_f32(template)
    m = _to_gray_f32(image)
    h, w = t.shape[:2]

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, 3, dtype=np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                iterations, eps)
    try:
        cv2.findTransformECC(t, m, warp, warp_mode, criteria, None, 5)
    except cv2.error:
        return image.copy()
    
    # ┌────────┬──────────────────────────┬──────────────────────────┐
    # │        │      cv2.warpAffine      │   cv2.warpPerspective    │
    # ├────────┼──────────────────────────┼──────────────────────────┤
    # │ 矩阵    │ 2×3 仿射矩阵 M            │ 3×3 单应矩阵 H            │
    # ├────────┼──────────────────────────┼──────────────────────────┤
    # │ 自由度  │ 6（平移+旋转+缩放+错切）    │ 8（再加透视）              │
    # ├────────┼──────────────────────────┼──────────────────────────┤
    # │ 能表达  │ 平行线保持平行             │ 平行线可汇聚（近大远小）     │
    # ├────────┼──────────────────────────┼──────────────────────────┤
    # │ 计算    │ 无除法，快                │ 每点要除以第三分量，稍慢     │
    # └────────┴──────────────────────────┴──────────────────────────┘
    # 前面已经进行过 粗矫正，使用透视 warpPerspective
    #
    # 相位相关操作后 只进行 warpAffine 就可以。注意：这步只需消除残余的亚像素平移，用最少自由度的模型最稳、最不容易发散。
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        return cv2.warpPerspective(
            image, warp, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return cv2.warpAffine(
        image, warp, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
