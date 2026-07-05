"""相位相关：估计粗配准后剩余整体像素位移 (dx, dy) 并平移补偿。

用于解决 CLAUDE.md 提出的"矫正后整体像素位移"问题：
warpPerspective 后与模板做相位相关，得到亚像素级残余平移，再 warpAffine 补偿，
避免 DCT 块网格因整体偏移而错位。
"""

import cv2
import numpy as np


def _to_gray_f32(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return np.float32(img)


def estimate_shift(template, image):
    """相位相关估计剩余平移量 (dx, dy) 与响应峰值。

    Returns
    -------
    (dx, dy, response)        image 相对 template 的平移与置信度
    """
    t = _to_gray_f32(template)
    m = _to_gray_f32(image)
    h = min(t.shape[0], m.shape[0])
    w = min(t.shape[1], m.shape[1])
    t, m = t[:h, :w], m[:h, :w]
    # 汉宁窗（Hann window）是一种常用的窗函数，主要用于信号处理和图像处理等领域。
    # 它是一种平滑的窗口函数，可以减少边界效应（如吉布斯现象），并提高频谱分辨率。
    # 汉宁窗在频域处理中特别有用，例如在傅里叶变换前后应用以减少泄漏效应。
    # 汉宁窗在以下场景中有广泛的应用：
    #   - 频域滤波：在傅里叶变换前后应用汉宁窗可以减少频谱泄漏。
    #   - 特征提取：在图像处理中，可以应用于图像的频域分析，减少边界效应。
    #   - 噪声去除：通过对频谱应用汉宁窗，可以有效地减少噪声的影响
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    # cv2.phaseCorrelate 是 OpenCV 中用于检测两幅图像之间相对位移量的函数。
    # 它基于傅立叶变换的位移定理，通过计算图像的互功率谱来确定图像的平移量。该方法在图像对齐和运动估计中非常有效。
    (dx, dy), response = cv2.phaseCorrelate(t, m, win)
    return dx, dy, response


def translate(image, dx, dy):
    """warpAffine 平移补偿（将 image 平移 -dx,-dy 回到模板坐标）。"""
    h, w = image.shape[:2]
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
