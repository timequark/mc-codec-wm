"""单应估计与透视矫正。"""

import cv2


def find_homography(pts_src, pts_dst, ransac_thresh=3.0):
    """RANSAC 估计单应矩阵。返回 (H, mask)。点太少返回 (None, None)。"""
    if len(pts_src) < 4 or len(pts_dst) < 4:
        return None, None
    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, ransac_thresh)
    return H, mask


def warp_perspective(image, H, size):
    """透视变换粗配准。size=(w, h)。"""
    return cv2.warpPerspective(image, H, size)
