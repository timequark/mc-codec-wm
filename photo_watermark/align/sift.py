"""SIFT / ORB 特征检测与匹配（参考 demo_compare_SIFT_vulcan_hat.py）。"""

import cv2
import numpy as np

from .. import config


def _to_gray(img):
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _match_once(g1, g2, method, ratio, nfeatures):
    if method.upper() == "ORB":
        det = cv2.ORB_create(nfeatures=nfeatures)
        norm = cv2.NORM_HAMMING
    else:
        det = cv2.SIFT_create(nfeatures=nfeatures)
        norm = cv2.NORM_L2

    k1, d1 = det.detectAndCompute(g1, None)
    k2, d2 = det.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(d1, d2, k=2)
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]

    pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pts1, pts2


def detect_and_match(template, photo, method="SIFT", ratio=0.75,
                     nfeatures=config.SIFT_NFEATURES, allow_fallback=True):
    """检测特征并按 Lowe ratio 匹配；主方法匹配不足时自动回退另一方法。

    Returns
    -------
    (pts_template, pts_photo) : 两个 Nx2 float32 数组（一一对应的匹配点）
    """
    g1, g2 = _to_gray(template), _to_gray(photo)
    pts1, pts2 = _match_once(g1, g2, method, ratio, nfeatures)

    if len(pts1) < 4 and allow_fallback:
        alt = "ORB" if method.upper() == "SIFT" else "SIFT"
        pts1, pts2 = _match_once(g1, g2, alt, ratio, nfeatures)
    return pts1, pts2
