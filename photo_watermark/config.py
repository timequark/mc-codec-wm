"""全局参数配置。

注意：
- BLOCK_SIZE 不写死，由 embed/decode 入口参数指定（此处仅给默认值）。
- 中频系数对 A/B 不写死，由 dct.coeff.mid_band_pair 依据块尺寸动态计算。
"""

from dataclasses import dataclass


# ---- 水印 / ECC ----
WATERMARK_LEN = 16           # 水印文本长度（16 位字母或数字）
RS_ECC_BYTES = 16            # ReedSolomon 校验字节数
DEFAULT_REPL = 8             # 默认冗余份数 repl

# ---- DCT ----
DEFAULT_BLOCK_SIZE = 12       # 默认块大小（可由入口改为 8/12/16）
DELTA = 60.0                 # 策略1 能量强度 delta（仅嵌入使用，解码不用）
MULTI_SITE_MIN_BLOCK = 24    # block_size >= 此值用 MULTI_SITE_COUNT 对，否则用 MULTI_SITE_COUNT_SMALL 对
MULTI_SITE_COUNT = 3         # 大块（>=MULTI_SITE_MIN_BLOCK）的中频对数
MULTI_SITE_COUNT_SMALL = 2   # 小块（<MULTI_SITE_MIN_BLOCK）的中频对数

# 嵌入频带开关：站点频率窗口 d=r+c 的取值区间（见 dct.coeff.mid_band_sites）
#   "mid" 中频 [n/2, 3n/2]：数字域最优，但印刷+拍照会低通滤除，鲁棒性差
#   "low" 低频 [n/5, n/2]：抗印刷拍照，但低频含更多图像内容，需较大 delta 压制
# 嵌入端与解码端必须一致；解码不知道嵌入用了哪档，故用全局开关保证对齐。
BAND_MODE = "low"

# ---- 对齐 / ROI ----
ALPHA_THRESHOLD = 0          # alpha > ALPHA_THRESHOLD 的区域参与嵌入

# ---- 对齐定位（路线 A：内容特征抠图+矫正）----
COARSE_MAX_SIDE = 1200       # 粗定位时拍照图降采样的最长边（像素）
LOCATE_INLIERS_MIN = 15      # 单应估计判定"检出"所需最少 RANSAC 内点
QUAD_AREA_MIN = 0.01         # 目标四边形面积占画面最小比例（低于视为误检）
QUAD_AREA_MAX = 1.20         # 最大比例（略大于 1 容忍目标出血/超出边界）
QUAD_ASPECT_TOL = 0.35       # 目标四边形长宽比相对模板的容差
ROI_MARGIN = 0.08            # 依粗定位裁剪 ROI 时的外扩比例
SIFT_NFEATURES = 8000        # 特征点上限


@dataclass
class Config:
    watermark_len: int = WATERMARK_LEN
    rs_ecc_bytes: int = RS_ECC_BYTES
    repl: int = DEFAULT_REPL
    block_size: int = DEFAULT_BLOCK_SIZE
    delta: float = DELTA
    band_mode: str = BAND_MODE
    alpha_threshold: int = ALPHA_THRESHOLD
    # 对齐定位
    coarse_max_side: int = COARSE_MAX_SIDE
    locate_inliers_min: int = LOCATE_INLIERS_MIN
    quad_area_min: float = QUAD_AREA_MIN
    quad_area_max: float = QUAD_AREA_MAX
    quad_aspect_tol: float = QUAD_ASPECT_TOL
    roi_margin: float = ROI_MARGIN
    sift_nfeatures: int = SIFT_NFEATURES

    @property
    def payload_bytes(self) -> int:
        """单份码字字节数 = 水印 + ECC。"""
        return self.watermark_len + self.rs_ecc_bytes

    @property
    def payload_bits(self) -> int:
        """单份码字 bit 数。"""
        return self.payload_bytes * 8
