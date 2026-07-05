"""镜头去畸变。"""

import cv2


def undistort(image, camera_matrix=None, dist_coeffs=None):
    """对拍照图去畸变；无标定参数时原样返回副本。"""
    if camera_matrix is None or dist_coeffs is None:
        return image.copy()
    return cv2.undistort(image, camera_matrix, dist_coeffs)
