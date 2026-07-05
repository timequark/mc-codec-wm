"""ROI：合并 alpha 掩码与敏感区蒙版，得到最终可嵌入掩码。

规则：蒙版中 alpha==0 的区域视为敏感区，跳过不嵌入。
"""

import numpy as np


def build_roi_mask(alpha_mask, sensitive_mask=None):
    """合并 alpha 掩码与敏感区蒙版。

    Parameters
    ----------
    alpha_mask : ndarray(bool)          原图 alpha>0 掩码
    sensitive_mask : ndarray(bool), optional
        蒙版可用区（True=可嵌入，False=敏感跳过）；None 表示不限制

    Returns
    -------
    ndarray(bool)                       最终可嵌入掩码
    """
    if sensitive_mask is None:
        return alpha_mask
    if sensitive_mask.shape != alpha_mask.shape:
        raise ValueError("蒙版尺寸与原图不一致")
    return np.logical_and(alpha_mask, sensitive_mask)
