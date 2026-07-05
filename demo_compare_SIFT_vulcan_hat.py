import os
import glob
import sys
import random
from typing import Any, Dict, List
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_sauvola
import alphashape
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union
from pytorch_msssim import ms_ssim
import torch
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
# Get the parent directory by going one level up
parent_dir = os.path.dirname(parent_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from codec.common import logger


'''
FUNC:
    cv2.findHomography(srcPoints, dstPoints, method=cv2.RANSAC, ransacReprojThreshold=3, mask=None, maxIters=2000, confidence=0.995)
DESC:
    找到两个图像之间的单应性矩阵（Homography matrix）

    单应性矩阵是一个3x3的矩阵，它描述了两个平面之间的相对位置关系；通过 cv2.wrapPerspective() 函数，应用单映射矩阵进行透视变换.
'''

DATA_ROOT = "watermark_graph/pic-hvs"

# PROD_NAME = "MonkeyKing"
PROD_NAME = "vulcan-hat"

# PROD_SUB_BOTTLE_FRONT = 'bottle-front'
# PROD_SUB_BOTTLE_BACK  = 'bottle-back'
# PROD_SUB_BOX_FRONT    = 'box-front'
# PROD_SUB_BOX_BACK     = 'box-back'
# PROD_SUB_CASE_TAPE    = 'case-tape'

PROD_ORIG_USE_ROI = False            # 开关：是否使用原图 ROI
PROD_SIMILARITY_CONTOUR_EN = True    # 开关：粗筛相似度轮廓比对
PROD_SIMILARITY_CONTOUR_PARALLEL_EN = False      # 开关：粗筛相似度轮廓比 - 并行-
PROD_DETECT_EDGE_ALG = 'hsv'                     # 边缘检测扣图算法, hsv - HSV 色彩检测, default - Canny 检测
PROD_DETECT_EDGE_LOCATE_ICON_EN = False           # 开关：HSV 方式定位有效图标区域

# 局部高重复纹理 / 小区域特征塌缩（feature collapse）导致的伪单应性（degenerate homography）
INLIERS_COVERAGE_CHECK = True    # 是否启用内点覆盖率检查
INLIERS_COVERAGE_THRESHOLD = 0.06   # 内点覆盖率阈值，过低可能是伪单应性

PROD_MATCH_SIMILITY_EN = False # 开关：形态比对

# TODO: 待优化并发方式
#   注意：并行量过大时会影响 knnMatch 输出结果的质量
PROD_MATCH_QUERY_WORKERS = 1        # 并行数量, 1 - for 循环线性执行, >1 - 并行


'''
cv2.BFMatcher.knnMatch 在关键点特别多（比如几千甚至上万）时确实会很耗时，因为它是 暴力匹配：每个描述子都会跟另一组所有描述子做距离计算。
复杂度是 O(N × M)，如果两个集合都很大，性能骤降！

FLANN 是 近似最近邻搜索，适合大规模特征匹配，速度比暴力匹配快很多，精度也足够。
    参数: checks
        近似最近邻搜索 时，每次查询特征点，FLANN 会在索引树（如 KD-Tree）中 随机选择若干条路径进行搜索。
        checks 就是允许搜索的 候选路径数。
        值 越大 → 搜得越全 → 匹配结果更接近精确暴力匹配，但速度更慢。
        值 越小 → 搜得越少 → 速度更快，但可能漏掉正确匹配。
        实际建议:
            checks=32 ~ 64：一般场景，准确率和速度平衡。
            checks=128 ~ 256：要求更高精度时。
            checks=10：实时应用（比如视频帧匹配），牺牲点精度换速度。

'''
MATCHER_ALG = 'FLANN' # BFMatcher | FLANN

DPI = 600

DEBUG               = False
DEBUG_DRAW_MATCHES  = True


MAP_ID = {
    '001': '0000000000001501',
    '002': '0000000000001501',
    '003': '0000000000001501',
    '004': '0000000000001501'
}

def build_path(f):
    return os.path.join(DATA_ROOT, f)

def save_image(image, filename):
    if image.ndim == 3:
        # 将 BGR 转换为 RGB
        rgb_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_array = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # 转换为 PIL Image
    pil_img = Image.fromarray(rgb_array)
    pil_img.save(filename, format='png', dpi=(DPI, DPI))
    # cv2.imwrite(filename, image)

def random_rgb():
    # 生成一个随机的 RGB 颜色
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def rects_intersect(r1, r2):
    """
    判断两个矩形是否相交
    rect: (x, y, w, h)
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

def rects_closure(rect_inner, rect_outer):
    """
    判断一个矩形是否在另一个矩形的内部
    rect: (x, y, w, h)
    """
    ax, ay, aw, ah = rect_inner
    bx, by, bw, bh = rect_outer

    return ax >= bx and (ax+aw) <= (bx+bw) and ay >= by and (ay+ah) <= (by+bh)

def order_box_points(box):
    """
    输入：box，shape (4,2) float/int
    输出：按左上→右上→右下→左下顺序排列的顶点
    """
    box = np.array(box)
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).ravel()

    ordered = np.zeros_like(box)
    ordered[0] = box[np.argmin(s)]     # 左上，x+y最小
    ordered[2] = box[np.argmax(s)]     # 右下，x+y最大
    ordered[1] = box[np.argmin(diff)]  # 右上，x-y最小
    ordered[3] = box[np.argmax(diff)]  # 左下，x-y最大

    return ordered.astype(int)

def bound_rect(contours):
    """
    所有轮廓的最外层外截矩形
    """
    if not contours:
        return None

    # 合并所有点
    all_points = np.vstack(contours)
    # 计算外接矩形
    x, y, w, h = cv2.boundingRect(all_points)
    return (x, y, w, h)

def min_area_rect_for_contour(contour, return_angle = False):
    """
    输入：单个轮廓 contour（numpy array of shape (N,1,2) 或 (N,2)）
    返回：字典，包含中心、宽高、角度、4个顶点、面积、长宽交换后的规范 angle（0..180）
    """
    if contour is None or len(contour) == 0:
        raise ValueError("空的 contour")

    # OpenCV 要求 contour 形状为 (N,1,2) 或 (N,2)
    rect = cv2.minAreaRect(contour)  # ((cx, cy), (w, h), angle)
    (cx, cy), (w, h), angle = rect

    # 得到 4 个顶点（浮点）并转为 int（用于绘图）
    box = cv2.boxPoints(rect)  # float32, shape (4,2)
    box_int = np.int0(box)

    if return_angle:

        # 规范化 angle，使得 w >= h，角度变为直观的 0..180 范围（可选）
        # OpenCV 返回 angle 的意义：
        #   如果 w >= h: angle 是矩形相对于水平线的逆时针角（通常在 -90..0）
        #   如果 w <  h: angle 会有偏移，因此常见做法是保证 w>=h
        if w < h:
            w, h = h, w
            angle = angle + 90.0

        # 将 angle 规范到 [0, 180)
        angle = angle % 180.0
        area = w * h
        aspect_ratio = w / h if h != 0 else float('inf')

        return {
            "center": (cx, cy),
            "width": w,
            "height": h,
            "angle": angle,        # 规范化角度 (0..180)
            "box": box,            # 4x2 float32 顶点
            "box_int": order_box_points(box_int),    # 4x2 整数顶点（用于绘图）
            "area": area,
            "aspect_ratio": aspect_ratio,
            "raw_rect": rect       # 原始 ((cx,cy),(w,h),angle)
        }
    else:
        return {
            "box_int": order_box_points(box_int)
        }

def merge_contours(contours):
    """
    合并所有轮廓：凸包
    """
    if not contours:
        return None
    
    # 1. 合并所有轮廓点
    all_points = np.vstack(contours)

    # 2. 计算凸包，得到一个合并后的 contour
    merged_contour = cv2.convexHull(all_points)

    return merged_contour

def find_valid_alpha_shape(points, alpha_start=0.5, alpha_end=5.0, step=0.5):
    """
    从较小 alpha 开始逐步增大，直到生成 Polygon / MultiPolygon
    """
    for a in np.arange(alpha_start, alpha_end, step):
        shape = alphashape.alphashape(points, a)
        if shape.geom_type in ("Polygon", "MultiPolygon"): # "Polygon", "MultiPolygon"
            return shape, a
    return None, None

def multipolygon_to_concave_contour(multipoly, alpha=2.0):
    """
    将 MultiPolygon 合并为一个外层凹包 contour
    """
    if multipoly.is_empty:
        return None

    # 收集所有外边界点
    all_points = []
    for poly in multipoly.geoms:
        all_points.extend(np.array(poly.exterior.coords))
    all_points = np.array(all_points)

    # 重新计算 alpha shape，得到外层包围
    merged = alphashape.alphashape(all_points, alpha)

    # 如果仍然是 MultiPolygon，取面积最大的部分
    # if merged.geom_type == 'MultiPolygon':
    #     merged = max(merged.geoms, key=lambda p: p.area)

    if merged.geom_type == 'Polygon':
        x, y = merged.exterior.coords.xy
        merged_contour = np.array(list(zip(x, y)), dtype=np.int32).reshape(-1, 1, 2)
        return merged_contour
    else:
        # 没有多边形，退化到 convex hull
        hull = cv2.convexHull(all_points.astype(np.float32))
        return hull

def merge_contours_concave(contours, alpha=1.5):
    """
    合并所有轮廓：凹包 (Concave Hull)
    alpha 值越小，轮廓越贴近点云（更“凹”）
    """
    if not contours:
        return None

    # 合并所有轮廓点
    # all_points = np.vstack([c.reshape(-1, 2) for c in contours])
    all_points = np.vstack(contours).reshape(-1, 2)

    shape, used_alpha = find_valid_alpha_shape(all_points, alpha_start=1.0, alpha_end=5.0, step=0.5)

    if shape is None:
        # 没有多边形，退化到 convex hull
        hull = cv2.convexHull(all_points.astype(np.float32))
        return hull
    
    concave_hull = shape

    # 如果断裂为多个 Polygon，则重新合并所有外边界再计算一次
    if concave_hull.geom_type == 'MultiPolygon':
        concave_hull = multipolygon_to_concave_contour(concave_hull, alpha=used_alpha)
        return concave_hull

    elif concave_hull.geom_type == 'GeometryCollection':
        '''
        此分支逻辑仅供参考
            find_valid_alpha_shape 里的条件限制，导致不会走这里。
        '''
        # 提取里面的多边形部分
        polys = [g for g in concave_hull.geoms if isinstance(g, Polygon)]
        if len(polys) > 0:
            hull = unary_union(polys)
        else:
            # 没有多边形，退化到 convex hull
            hull = cv2.convexHull(all_points.astype(np.float32))
        return hull
    
    elif concave_hull.geom_type == 'Polygon':
        x, y = concave_hull.exterior.coords.xy
        merged_contour = np.array(list(zip(x, y)), dtype=np.int32).reshape(-1, 1, 2)
        return merged_contour
    
    return None

def merge_contours_by_bbox(contours, merge_cross=False):
    # 1. 计算所有 boundingRect
    rects = [cv2.boundingRect(c) for c in contours]

    # 2. 构建邻接关系（相交的矩形归为一类）
    n = len(rects)
    visited = [False] * n
    groups = []

    def dfs(i, group):
        visited[i] = True
        group.append(i)
        for j in range(n):
            if not visited[j] and merge_cross and rects_intersect(rects[i], rects[j]):
                dfs(j, group)

    for i in range(n):
        if not visited[i]:
            group = []
            dfs(i, group)
            groups.append(group)

    # 3. 合并每组内的 contour
    contour_groups_list = []
    for group in groups:
        pts = np.vstack([contours[i] for i in group])
        contour_group_info = (pts, group)
        contour_groups_list.append(contour_group_info)

    return contour_groups_list

class SIFTMatcher:
    def __init__(self, cache_candidate_features="candidate_features.pkl"):
        # SIFT: 调整参数以提高关键点数量
        self.sift = cv2.SIFT_create(
            nfeatures=10000,           # 特征点数量，0表示不限制
            nOctaveLayers=3,       # 每octave中的层数（默认3），增加可检测更多尺度
            contrastThreshold=0.02, # 对比度阈值（默认0.04），降低可检测更多点
            edgeThreshold=20,      # 边缘阈值（默认10），增加可保留更多边缘点
            sigma=1.0              # 高斯模糊sigma（默认1.6），减小可保留更多细节
        )
        # SIFT: 默认
        # self.sift = cv2.SIFT_create()

        if MATCHER_ALG == 'BFMatcher':
            self.bf = cv2.BFMatcher()
        else:
            # 创建 FLANN matcher
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # 增大 checks 更精确，减少更快
            self.bf = cv2.FlannBasedMatcher(index_params, search_params)
            
        self.candidate_features = {}  # 存储候选图像的特征信息
        self.cache_candidate_features = cache_candidate_features
        self.img_height = 0
        self.img_width  = 0
        
    def initialize_candidates(self, candidate_img_paths: List[str]) -> Dict[str, Any]:
        """
        一次性初始化所有候选原图的特征
        
        参数:
        candidate_img_paths: 候选原图路径列表
        
        返回:
        初始化结果信息
        """
        self.candidate_features.clear()
        results = {}
        
        for i, img_path in enumerate(candidate_img_paths):
            if not os.path.exists(img_path):
                logger.info(f"警告: 文件不存在 {img_path}")
                results[img_path] = {"status": "error", "message": "文件不存在"}
                continue
                
            try:
                # 读取图像并转换为灰度图
                img = cv2.imread(img_path)
                if img is None:
                    logger.info(f"警告: 无法读取图像 {img_path}")
                    results[img_path] = {"status": "error", "message": "无法读取图像"}
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                
                # 检测关键点和计算描述符
                keypoints, descriptors = self.sift.detectAndCompute(gray, None)
                
                if descriptors is None:
                    logger.info(f"警告: 无法在图像中检测到关键点 {img_path}")
                    results[img_path] = {"status": "error", "message": "无关键点"}
                    continue
                
                # 存储特征信息
                candidate_img = cv2.imread(img_path)
                candidate_img_gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
                self.candidate_features[img_path] = {
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'image_shape': img.shape[:2],  # 存储图像尺寸 (height, width)
                    'index': i,
                    'candidate_img_gray': candidate_img_gray
                }
                
                results[img_path] = {
                    "status": "success",
                    "keypoints_count": len(keypoints),
                    "image_shape": img.shape[:2]
                }

                self.img_height, self.img_width = img.shape[:2]
                
                logger.info(f"初始化成功: {img_path} - 关键点: {len(keypoints)}")
                
            except Exception as e:
                logger.info(f"初始化失败 {img_path}: {str(e)}")
                results[img_path] = {"status": "error", "message": str(e)}
        
        return results
    
    def save_features(self) -> bool:
        """
        保存候选图像特征到文件
        
        参数:
        file_path: 保存路径
        
        返回:
        是否保存成功
        """
        try:
            # 将关键点转换为可序列化的格式
            serializable_features = {}
            for path, features in self.candidate_features.items():
                # 将关键点转换为可序列化的格式
                keypoints_data = []
                for kp in features['keypoints']:
                    keypoints_data.append((
                        kp.pt, kp.size, kp.angle, kp.response, 
                        kp.octave, kp.class_id
                    ))
                
                serializable_features[path] = {
                    'keypoints_data': keypoints_data,
                    'descriptors': features['descriptors'],
                    'image_shape': features['image_shape'],
                    'index': features['index']
                }
            
            with open(self.cache_candidate_features, 'wb') as f:
                pickle.dump(serializable_features, f)
            return True
        except Exception as e:
            logger.info(f"保存特征失败: {str(e)}")
            return False
    
    def load_features(self) -> bool:
        """
        从文件加载候选图像特征
        
        参数:
        file_path: 加载路径
        
        返回:
        是否加载成功
        """
        try:
            # 避免重复加载
            if len(self.candidate_features) > 0:
                return True
            
            with open(self.cache_candidate_features, 'rb') as f:
                serializable_features = pickle.load(f)
            
            self.candidate_features.clear()
            
            for path, features in serializable_features.items():
                # 将序列化数据转换回KeyPoint对象
                keypoints = []
                for kp_data in features['keypoints_data']:
                    kp = cv2.KeyPoint(
                        x=kp_data[0][0], y=kp_data[0][1],
                        size=kp_data[1], angle=kp_data[2],
                        response=kp_data[3], octave=int(kp_data[4]),
                        class_id=int(kp_data[5])
                    )
                    keypoints.append(kp)
                
                candidate_img = cv2.imread(path)
                candidate_img_gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
                self.candidate_features[path] = {
                    'keypoints': keypoints,
                    'descriptors': features['descriptors'],
                    'image_shape': features['image_shape'],
                    'index': features['index'],
                    'candidate_img_gray': candidate_img_gray
                }
                self.img_height, self.img_width = features['image_shape']
            
            logger.info(f"加载成功: {len(self.candidate_features)} 个候选图像特征")
            return True
        except Exception as e:
            logger.info(f"加载特征失败: {str(e)}")
            return False
    
    def locate_target(self, img_bgr, dst_size=None):
        if PROD_DETECT_EDGE_LOCATE_ICON_EN:
            # BGR -> HSV
            img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # 蓝色范围
            lower_blue = np.array([80, 80, 80])
            upper_blue = np.array([140, 255, 255])
            mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
            contour_blue_groups, valid_contours_blue = self.find_contour_blocks(img_gray, mask_blue, filter_area=True)
            if not contour_blue_groups or len(contour_blue_groups) > 1:
                logger.error("A: no BLUE targets found !")
                return False, None, None
            
            # 优先匹配大面积区域
            contour_blue_groups = sorted(contour_blue_groups, key=lambda x: cv2.contourArea(x[0]), reverse=True)
            blue_big_contour = contour_blue_groups[0][0]

            # 红色范围 (低区间)
            lower_red1 = np.array([0, 80, 80])
            upper_red1 = np.array([10, 255, 255])
            mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
            # 红色范围 (高区间)
            lower_red2 = np.array([170, 80, 80])
            upper_red2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
            # 合并两段红色掩膜
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            contour_red_groups, contour_red_list = self.find_contour_blocks(img_gray, mask_red, filter_area=True, merge_cross=False)
            n_red = len(contour_red_groups)
            if not contour_red_groups or n_red > 1:
                logger.error("A: no RED targets found !")
                return False, None, None
            
            # 优先匹配大面积区域
            contour_red_groups = sorted(contour_red_groups, key=lambda x: cv2.contourArea(x[0]), reverse=True)
            red_big_contour = contour_red_groups[0][0]

            area_red  = cv2.contourArea(contour_red_groups[0][0])
            area_blue = cv2.contourArea(contour_blue_groups[0][0])
            ratio_blue2red = area_blue / area_red

            find_target_icon = False
            if 0.2 <= ratio_blue2red and ratio_blue2red <= 0.35:
                find_target_icon = True
            
            if not find_target_icon:
                logger.info("A: no target icon found !")
                return False, None, None
            
            # 限定在红色 + 蓝色区域内
            mask_copper = np.zeros(img_gray.shape, dtype=np.uint8)
            red_blue_contour = merge_contours([blue_big_contour, red_big_contour])
            cv2.fillPoly(mask_copper, [red_blue_contour], 255)

            img_bgr_copy = img_bgr.copy()
            img_bgr_copy[mask_copper == 0] = 255
            x, y, w, h = cv2.boundingRect(red_blue_contour) # 不用 copper_big_contour，可以大致去 人手 区域
            roi_bgr = img_bgr_copy[y:y+h, x:x+w]
            if dst_size is not None:
                dst_h, dst_w = dst_size
                roi_h, roi_w = roi_bgr.shape[:2]
                fx = dst_w / roi_w
                fy = fx
                roi_bgr = cv2.resize(roi_bgr, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
            return True, roi_bgr, red_blue_contour
        else:
            return True, img_bgr.copy(), None
    
    def extract_contours(self, img_any):
        """提取图像中的主要轮廓"""
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # # 多种预处理方法组合
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # # 自适应阈值处理，更好地处理光照变化
        # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                             cv2.THRESH_BINARY_INV, 11, 2)
        
        # # 形态学操作去除噪声
        # kernel = np.ones((3, 3), np.uint8)
        # cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # # 查找轮廓
        # contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # # 过滤掉太小的轮廓
        # min_area = image.shape[0] * image.shape[1] * 0.1  # 至少占图像面积的10%
        # filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # return filtered_contours

        if img_any.ndim == 3:
            # 1. 转换颜色空间
            # ycbcr = cv2.cvtColor(img_any, cv2.COLOR_BGR2YCrCb)
            # Y, Cr, Cb = cv2.split(ycbcr)
            img_gray = cv2.cvtColor(img_any, cv2.COLOR_BGR2GRAY)
        else:
            # Y = img_any
            img_gray = img_any
        
        # 2. 边缘检测
        # edges = cv2.Canny(Y, 100, 255) # threshold1, threshold2 越小，捕捉更多的边缘信息

        # if DEBUG:
        #     cv2.imshow('edges', edges)
        #     cv2.waitKey(0)

        mean_gray = np.mean(img_gray)
        # 图像二值化
        if mean_gray <= 80:
            ret, binary_image = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th1 = 255 - binary_image # 反色处理
        elif 80 < mean_gray  and mean_gray <= 100:
            ret,binary_image = cv2.threshold(img_gray, mean_gray, 255, cv2.THRESH_BINARY)
            th1 = binary_image
        elif 100 < mean_gray and mean_gray <= 140:
            # 深色背景
            ret, binary_image = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th1 = 255 - binary_image # 反色处理
        elif 140 < mean_gray and mean_gray <= 160:
            # ret,binary_image = cv2.threshold(img_gray, mean_gray / 2, 255, cv2.THRESH_BINARY)
            ret,binary_image = cv2.threshold(img_gray, mean_gray, 255, cv2.THRESH_BINARY)
            th1 = 255 - binary_image
        else:
            '''
            浅色背景（偏亮）
            '''
            thresh_sauvola = threshold_sauvola(img_gray, window_size=31)
            binary_image0 = img_gray > thresh_sauvola                              # 取出偏白的像素
            th1 = np.array((binary_image0 + 0) * 255, dtype='uint8')          # 偏白的像素直接拉满到 255, 背景变白
            th1 = 255 - th1

        if DEBUG:
            cv2.imshow('th1', th1)
            cv2.waitKey(0)
        
        # 查找轮廓
        contours, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if DEBUG:
            _rgb = img_any.copy()
            for contour in contours:
                cv2.drawContours(_rgb, [contour], -1, random_rgb(), 1)
            cv2.imshow('+++', _rgb)
            cv2.waitKey(0)
        
        # 过滤掉太小的轮廓
        # min_area = img_gray.shape[0] * img_gray.shape[1] * 0.01  # 至少占图像面积的1%
        min_area = 10
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        return filtered_contours

    def compare_contours_hu_moments(self, query_img_path, query_img, candidate_img_path, candidate_img, query_main_contour=None):
        """
        使用Hu矩进行轮廓相似度比较
        Hu矩对平移、旋转、缩放具有不变性
        """
        if query_main_contour is None:
            # 读取图像并提取轮廓
            if query_img is None:
                query_img = cv2.imread(query_img_path)
            contours1 = self.extract_contours(query_img)
            query_main_contour = max(contours1, key=cv2.contourArea)
        
        if candidate_img is None:
            candidate_img = cv2.imread(candidate_img_path)
        contours2 = self.extract_contours(candidate_img)
        
        if query_main_contour is None or not contours2:
            logger.info('无法检测到轮廓')
            return {"error": "无法检测到轮廓"}
        
        # 取最大的轮廓进行比较
        main_contour2 = max(contours2, key=cv2.contourArea)
        if DEBUG:
            _rgb_img = query_img.copy()
            cv2.drawContours(_rgb_img, [query_main_contour], -1, (255, 0, 0), 3)
            cv2.imshow('contour 1', _rgb_img)
            cv2.waitKey(0)

            if candidate_img.ndim < 3:
                _rgb_img = cv2.cvtColor(candidate_img, cv2.COLOR_GRAY2BGR)
            else:
                _rgb_img = candidate_img.copy()
            cv2.drawContours(_rgb_img, [main_contour2], -1, (255, 0, 0), 3)
            cv2.imshow('contour 2', _rgb_img)
            cv2.waitKey(0)
        
        # 计算Hu矩
        hu_moments1 = cv2.HuMoments(cv2.moments(query_main_contour)).flatten()
        hu_moments2 = cv2.HuMoments(cv2.moments(main_contour2)).flatten()
        
        # 计算相似度（使用对数变换避免小值的影响）
        similarity = cv2.matchShapes(query_main_contour, main_contour2, cv2.CONTOURS_MATCH_I2, 0)
        
        # Hu矩距离（值越小越相似）
        hu_distance = np.sum(np.abs(
                                np.log(np.abs(hu_moments1) + 1e-10) -
                                np.log(np.abs(hu_moments2) + 1e-10)))
        logger.info(f'similarity_score: {similarity}, hu_distance: {hu_distance}')

        return {
            'similarity_score': similarity,  # 0表示完全匹配，值越大差异越大
            'hu_distance': hu_distance,
            'query_main_contour': query_main_contour,
            'contour1_area': cv2.contourArea(query_main_contour),
            'contour2_area': cv2.contourArea(main_contour2),
            'is_similar': similarity < 0.2  # 经验阈值
        }
    
    def ssim_similarity(self, query_img_path, candidate_img_path):
        '''
        缺点：
            SSIM 比对没有考虑图像尺度问题
        '''
        query_img = cv2.imread(query_img_path)
        candidate_img = cv2.imread(candidate_img_path)
        query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        candidate_img_gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)

        wavelet = 'haar'
        query_img_LL, _ = pywt.dwt2(query_img_gray, wavelet)
        candidate_img_LL, _ = pywt.dwt2(candidate_img_gray, wavelet)

        def normalize(img):
            return (img - img.min()) / (img.max() - img.min() + 1e-6)

        query_img_LL = normalize(query_img_LL)
        candidate_img_LL = normalize(candidate_img_LL)

        score, diff = ssim(query_img_LL, candidate_img_LL, data_range=1.0, full=True)
        logger.info(f"SSIM: {score}")
        return score
        
    def ssim_similarity_ms(self, query_img_path, query_img, candidate_img_path, candidate_img_gray):
        '''
        MS - SSIM (Multi-Scale SSIM)
        '''
        if query_img is None:
            query_img = cv2.imread(query_img_path)
        if candidate_img_gray is None:
            candidate_img = cv2.imread(candidate_img_path)
            candidate_img_gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
        query_img_h, query_img_w = query_img.shape[:2]
        candidate_img_h, candidate_img_w = candidate_img_gray.shape[:2]
        if query_img_h != candidate_img_h or query_img_w != candidate_img_w:
            query_img = cv2.resize(query_img, (candidate_img_w, candidate_img_h), interpolation=cv2.INTER_LINEAR)
        
        if query_img.ndim == 3:
            query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        else:
            query_img_gray = query_img
        

        '''
        比对 LL 带，效果差
        '''
        # wavelet = 'haar'
        # query_img_LL, _ = pywt.dwt2(query_img_gray, wavelet)
        # candidate_img_LL, _ = pywt.dwt2(candidate_img_gray, wavelet)
        # def normalize(img):
        #     return (img - img.min()) / (img.max() - img.min() + 1e-6)
        # query_img_LL = normalize(query_img_LL)
        # candidate_img_LL = normalize(candidate_img_LL)
        # 转为 torch Tensor [N,C,H,W]
        # t1 = torch.tensor(query_img_LL).unsqueeze(0).unsqueeze(0)
        # t2 = torch.tensor(candidate_img_LL).unsqueeze(0).unsqueeze(0)
        # t = ms_ssim(t1, t2, data_range=1.0).item()

        '''
        比对灰度图，效果不错
        '''
        # 灰度图归一化
        t1 = torch.tensor(query_img_gray, dtype=torch.float32) / 255.0
        t2 = torch.tensor(candidate_img_gray, dtype=torch.float32) / 255.0
        # 转成 [N,C,H,W]
        t1 = t1.unsqueeze(0).unsqueeze(0)
        t2 = t2.unsqueeze(0).unsqueeze(0)
        t = ms_ssim(t1, t2, data_range=1.0).item()
        logger.info(f"MS-SSIM Tensor Score {t}")
        return t
    
    def detect_edge(self, img_file, img_bgr=None, roi_save=False, dst_size=None, correct=False):
        if img_bgr is None:
            img_bgr = cv2.imread(img_file)
        
        if not roi_save and not correct:
            return img_bgr
        
        if not roi_save and correct and img_file and os.path.exists(img_file + '.corrected.png') and os.path.getsize(img_file + '.corrected.png') > 0:
            return
        
        # 1. 转换颜色空间
        if img_bgr.ndim == 3:
            ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
            Y, Cr, Cb = cv2.split(ycbcr)
        else:
            Y = img_bgr
        
        # 2. 边缘检测
        edges = cv2.Canny(Y, 10, 150) # threshold1, threshold2 越小，捕捉更多的边缘信息

        if DEBUG:
            cv2.imshow('_', edges)
            cv2.waitKey(0)
        
        dot_sets_x = np.sum(edges, axis=0)                     # X轴投影（沿 Y 轴求和）
        dot_sets_y = np.sum(edges, axis=1)                     # Y轴投影（沿 X 轴求和）
        
        # np.where(condition, x, y)
        #     满足条件(condition)，输出x，不满足输出y
        #
        # np.where(condition)
        #     只有条件 (condition)，没有x和y，则输出满足条件元素的坐标。这里的坐标以tuple的形式给出，通常原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标
        
        dot_coods_x = np.where(dot_sets_x > 0)[0]
        dot_coods_y = np.where(dot_sets_y > 0)[0]
        if len(dot_coods_x) == 0 or len(dot_coods_y) == 0:
            raise Exception('No valid contour detected.')
        img_roi = img_bgr[dot_coods_y[0]:dot_coods_y[-1]+1, dot_coods_x[0]:dot_coods_x[-1]+1]
        if roi_save and img_file is not None:
            save_image(img_roi, img_file + '.roi.png')
        
        if DEBUG:
            cv2.imshow('roi image', img_roi)
            cv2.waitKey(0)
        
        logger.info(f'ROI size is {dot_coods_y[-1]-dot_coods_y[0]+1} x {dot_coods_x[-1]-dot_coods_x[0]+1}')
        logger.info('')

        img_corrected = None

        if correct and dst_size is not None:
            logger.info('开始矫正...')
            dst_h, dst_w = dst_size
            dst = np.float32([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h]])
            roi_h, roi_w = img_roi.shape[:2]
            ordered_points = np.float32([[0, 0], [roi_w-1, 0], [roi_w-1, roi_h-1], [0, roi_h-1]])
            matrix = cv2.getPerspectiveTransform(ordered_points, dst)
            img_corrected = cv2.warpPerspective(img_roi, matrix, (dst_w, dst_h)) # 默认: cv2.INTER_LINEAR
            if roi_save:
                save_image(img_corrected, img_file + '.corrected.png')
            if DEBUG:
                cv2.imshow('corrected image', img_corrected)
                cv2.waitKey(0)
        else:
            img_corrected = img_bgr.copy()
        
        logger.info('detect & correct done')
        return img_corrected
    
    def find_contour_blocks(self, img_gray, mask, filter_area=True, merge_cross=False):
        img_h, img_w = img_gray.shape[:2]

        # 保留原图中的 mask 区域，其余白色
        white_background = np.full_like(img_gray, 255)
        # result = np.where(mask[:, :, np.newaxis] == 255, img_gray, white_background) # 彩图时加 np.newaxis
        result = np.where(mask[:, :] == 255, img_gray, white_background)
        if DEBUG:
            cv2.imshow("Mask Image Only", result)
            cv2.waitKey(0)
        
        _, im_bw = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if DEBUG:
            cv2.imshow("Mask Binary", im_bw)
            cv2.waitKey(0)

        # 查找轮廓
        # 外轮廓 + 洞的轮廓
        # contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 外轮廓，忽略内部空洞（洞的轮廓）
        im_bw = cv2.bitwise_not(im_bw)   # 反转黑白
        contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for i, cnt in enumerate(contours):
            # hierarchy[i][3] == -1 表示最外层轮廓（通常是边框），跳过
            # if hierarchy[0][i][3] == -1:
            #     continue
            valid_contours.append(cnt)
        if filter_area:
            th_area_lower = 2000 # 1000
            th_area_upper = 0.9 * img_w * img_h

            # 存在多个相对独立、不够聚集的轮廓时会有问题！
            # valid_contours = [cnt for cnt in valid_contours if cv2.contourArea(cnt) < th_area_upper]
            # merged_contour = merge_contours(valid_contours)
            # return [merged_contour]

            valid_contours = [
                cnt for cnt in valid_contours
                if th_area_lower <= (area := cv2.contourArea(cnt)) <= th_area_upper
            ]
            if DEBUG:
                logger.info('contour area threshold upper: {}'.format(th_area_upper))
                for cnt in valid_contours:
                    logger.info('contour area: {}'.format(cv2.contourArea(cnt)))
        if DEBUG:
            _rgb = cv2.cvtColor(im_bw, cv2.COLOR_GRAY2BGR)
            for cnt in valid_contours:
                cv2.drawContours(_rgb, [cnt], -1, random_rgb(), -1)
            cv2.imshow('valid contours', _rgb)
            cv2.waitKey(0)
        
        # 幼稚 聚类 算法
        contour_groups_list = merge_contours_by_bbox(valid_contours, merge_cross=merge_cross)
        return contour_groups_list, valid_contours
    
    def detect_edge_hsv(self, img_file, img_bgr=None, roi_save=False, dst_size=None, correct=False):
        if img_bgr is None:
            img_bgr = cv2.imread(img_file)
        
        if not roi_save and not correct:
            return img_bgr
        
        if not roi_save and correct and img_file and os.path.exists(img_file + '.corrected.png') and os.path.getsize(img_file + '.corrected.png') > 0:
            return cv2.imread(img_file + '.corrected.png')
        
        # 彩图
        if img_bgr.ndim != 3:
            raise Exception('Not support GRAY image.')
        
        located, img_bgr, contour_copper = self.locate_target(img_bgr=img_bgr, dst_size=dst_size)
        if not located:
            return None
        
        if DEBUG:
            cv2.imshow("Copper Filtered", img_bgr)
            cv2.waitKey(0)
        
        # img_h, img_w = img_bgr.shape[:2]

        # 注意:
        #   target_rect 只是最小外截、旋转正矩形，并没有考虑 倾 斜
        # info = min_area_rect_for_contour(contour_copper) # 注意：这里最小外截矩形，不是 boundingRect
        # target_rect = info['box_int']
        # logger.info(f"A: find target icon. {target_rect}")

        # 不矫正了，直接使用 contour_copper 区域 roi 图像
        pad = 5
        img_corrected = np.pad(
            img_bgr,
            pad_width=((pad, pad), (pad, pad), (0, 0)),  # 只对 H, W pad，不对通道 pad
            mode='constant',
            constant_values=255  # 白色边框
        )

        # img_corrected = None
        # if correct and dst_size is not None:
        #     logger.info('开始矫正...')
        #     dst_h, dst_w = dst_size
        #     dst = np.float32([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h]])
        #     # ordered_points = np.float32([[0, 0], [roi_w-1, 0], [roi_w-1, roi_h-1], [0, roi_h-1]])
        #     ordered_points = np.float32([target_rect[0], target_rect[1], target_rect[2], target_rect[3]])
        #     matrix = cv2.getPerspectiveTransform(ordered_points, dst)
        #     img_corrected = cv2.warpPerspective(img_bgr, matrix, (dst_w, dst_h)) # 默认: cv2.INTER_LINEAR
        #     pad = 10
        #     img_corrected = np.pad(
        #         img_corrected,
        #         pad_width=((pad, pad), (pad, pad), (0, 0)),  # 只对 H, W pad，不对通道 pad
        #         mode='constant',
        #         constant_values=255  # 白色边框
        #     )
        #     if roi_save:
        #         save_image(img_corrected, img_file + '.corrected.png')
        #     if DEBUG:
        #         cv2.imshow('corrected image', img_corrected)
        #         cv2.waitKey(0)
        # else:
        #     img_corrected = img_bgr.copy()
        # logger.info('detect & correct done')

        return img_corrected
    
    def match_prod(self, query_img):
        # 与每个候选图像进行匹配
        scores = []
        max_score_ms_ssis = 0
        min_score_contour = 0

        if PROD_MATCH_QUERY_WORKERS == 1:
            for candidate_path, candidate_feat in self.candidate_features.items():
                score = self.ssim_similarity_ms(None, query_img, candidate_path, candidate_feat['candidate_img_gray'])
                scores.append(score)
            max_score_ms_ssis = np.max(scores)
        
            if PROD_SIMILARITY_CONTOUR_EN:
                contour_scores = []
                query_main_contour = None

                for candidate_path, candidate_feat in self.candidate_features.items():
                    similarity_info = self.compare_contours_hu_moments(None, query_img, candidate_path, candidate_feat['candidate_img_gray'], query_main_contour=query_main_contour)
                    if 'query_main_contour' in similarity_info:
                        query_main_contour = similarity_info['query_main_contour']
                    contour_scores.append(similarity_info)
                contour_scores = [s for s in contour_scores if 'error' not in s]
                if contour_scores:
                    min_score_contour = np.min([s['similarity_score'] for s in contour_scores])
        else:
            with ThreadPoolExecutor(max_workers=PROD_MATCH_QUERY_WORKERS) as executor:
                # ---------- 1. 提交 SSIM 任务 ----------
                ssim_futures = {
                    executor.submit(self.ssim_similarity_ms, None, query_img, path, feat['candidate_img_gray']): path
                    for path, feat in self.candidate_features.items()
                }
                ssim_scores = []
                for future in as_completed(ssim_futures):
                    ssim_scores.append(future.result())
                if ssim_scores:
                    max_score_ms_ssis = np.max(ssim_scores)

                if PROD_SIMILARITY_CONTOUR_EN:
                    '''
                    弃用： 该步骤线程启动时需要等待几秒，耗时
                    
                    保持单线程
                    '''
                    if PROD_SIMILARITY_CONTOUR_PARALLEL_EN:
                        # ---------- 2. 并行轮廓匹配 ----------
                        contour_scores = []
                        query_main_contour = None

                        # 先获取 query_main_contour
                        first_item = next(iter(self.candidate_features.items()))
                        first_similarity_info = self.compare_contours_hu_moments(
                            None, query_img, first_item[0], first_item[1]['candidate_img_gray'], query_main_contour=None
                        )
                        if 'query_main_contour' in first_similarity_info:
                            query_main_contour = first_similarity_info['query_main_contour']
                        if 'error' not in first_similarity_info:
                            contour_scores.append(first_similarity_info)

                        def process_contour(candidate_path, candidate_feat):
                            return self.compare_contours_hu_moments(
                                None, query_img, candidate_path, candidate_feat['candidate_img_gray'], query_main_contour=query_main_contour
                            )
                    
                        futures = [
                            executor.submit(process_contour, path, feat)
                            for path, feat in list(self.candidate_features.items())[1:]  # 剩下的候选
                        ]
                        for future in as_completed(futures):
                            res = future.result()
                            if 'error' not in res:
                                contour_scores.append(res)
                        if contour_scores:
                            min_score_contour = np.min([s['similarity_score'] for s in contour_scores])
                    else:
                        # 单线程
                        contour_scores = []
                        query_main_contour = None

                        for candidate_path, candidate_feat in self.candidate_features.items():
                            similarity_info = self.compare_contours_hu_moments(None, query_img, candidate_path, candidate_feat['candidate_img_gray'], query_main_contour=query_main_contour)
                            if 'query_main_contour' in similarity_info:
                                query_main_contour = similarity_info['query_main_contour']
                            contour_scores.append(similarity_info)
                        contour_scores = [s for s in contour_scores if 'error' not in s]
                        if contour_scores:
                            min_score_contour = np.min([s['similarity_score'] for s in contour_scores])
        
        # 综合评估
        logger.info(f"综合评估 - MS-SSIM max score: {max_score_ms_ssis}, Contour min similarity score: {min_score_contour}")
        if max_score_ms_ssis < 0.5 and min_score_contour > 1.0:
            return False
        
        return True
        
    def match_query_image(self, query_img_path: str, query_img:None, ratio_thresh: float = 0.75, ransac_thresh: float = 5.0) -> Dict[str, Any]:
        """
        匹配查询图像与已初始化的候选原图
        
        参数:
        query_img_path: 查询图像路径
        ratio_thresh: Lowe's比率测试的阈值
        ransac_thresh: RANSAC算法的阈值
        
        返回:
        匹配结果和相关信息
        """
        if not self.candidate_features:
            return {"error": "请先初始化候选图像特征"}
        
        # 读取查询图像
        if query_img is None:
            query_img = cv2.imread(query_img_path)
        if query_img is None:
            return {"error": f"无法读取查询图像: {query_img_path}"}
        
        if query_img.ndim == 3:
            query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        else:
            query_gray = query_img
        if DEBUG:
            cv2.imshow('orig', cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY))
            cv2.waitKey(0)
            cv2.imshow('query', query_gray)
            cv2.waitKey(0)

        # 检测查询图像的关键点和描述符
        kp_query, des_query = self.sift.detectAndCompute(query_gray, None)
        if des_query is None:
            return {"error": "无法在查询图像中检测到关键点"}
        
        # 存储所有候选图像的结果
        candidate_results = []
        
        if PROD_MATCH_QUERY_WORKERS == 1:
            # 与每个候选图像进行匹配
            for candidate_path, candidate_feat in self.candidate_features.items():
                kp_candidate = candidate_feat['keypoints']
                des_candidate = candidate_feat['descriptors']
                
                des_query_copy = des_query.copy()

                # 进行特征匹配
                matches = self.bf.knnMatch(des_query_copy, des_candidate, k=2)
                
                # 应用比率测试
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) < 2:
                        continue
                    m, n = match_pair
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)
                
                # 计算RANSAC内点数
                inliers_count = 0
                mask = None
                homography = None
                
                if len(good_matches) >= 4:
                    src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_candidate[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
                    inliers_count = np.sum(mask) if mask is not None else 0

                    logger.info(f"To CHECK Kp-Candidate: {len(kp_candidate)}, Kp-Query: {len(kp_query)}, Matches: {len(good_matches)}, Inliers: {inliers_count}")

                    # 可视化匹配结果（可选）
                    if DEBUG_DRAW_MATCHES:
                        candidate_img = cv2.imread(candidate_path)
                        self.draw_and_save_matches(query_img, kp_query, candidate_img, kp_candidate, good_matches, f"{candidate_path}_matches.jpg", mask)
                    
                    # 新增：覆盖率检测
                    if INLIERS_COVERAGE_CHECK and mask is not None:
                        inliers_count = int(np.sum(mask))

                        # =========================
                        # ✅ 1. 提取 inliers 点
                        # =========================
                        inlier_dst = dst_pts[mask.ravel() == 1].reshape(-1, 2)

                        if len(inlier_dst) >= 4:
                            h, w = candidate_feat['image_shape']

                            # =========================
                            # ✅ 2. coverage（网格覆盖率）
                            # =========================
                            grid_size = 16
                            grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

                            for pt in inlier_dst:
                                gx = int(pt[0] / w * grid_size)
                                gy = int(pt[1] / h * grid_size)

                                gx = np.clip(gx, 0, grid_size - 1)
                                gy = np.clip(gy, 0, grid_size - 1)

                                grid[gy, gx] = 1

                            coverage = np.sum(grid) / (grid_size * grid_size)

                            # =========================
                            # ✅ 3. area ratio（辅助）
                            # =========================
                            # x, y, bw, bh = cv2.boundingRect(inlier_dst.astype(np.float32))
                            # area_ratio = (bw * bh) / (w * h)

                            # =========================
                            # ✅ 4. 过滤逻辑（关键）
                            # =========================
                            logger.info(f"Candidate: {candidate_path}, Inliers: {inliers_count}, Coverage: {coverage:.2f}, Inliers Coverage Threshold: {INLIERS_COVERAGE_THRESHOLD} ")
                            # if coverage < 0.1 or area_ratio < 0.05:
                            if coverage < INLIERS_COVERAGE_THRESHOLD:
                                # 判定为误匹配
                                inliers_count = 0
                                homography = None
                                mask = None
                    
                # 存储该候选图像的结果
                candidate_result = {
                    'path': candidate_path,
                    'index': candidate_feat['index'],
                    'keypoints_count': len(kp_candidate),
                    'matches_count': len(good_matches),
                    'inliers_count': inliers_count,
                    'homography': homography,
                    'mask': mask
                }
                candidate_results.append(candidate_result)
        else:
            # 定义单个候选处理函数
            def process_candidate(candidate_path, candidate_feat):
                kp_candidate = candidate_feat['keypoints']
                des_candidate = candidate_feat['descriptors']

                # 特征匹配
                matches = matcher.bf.knnMatch(des_query, des_candidate, k=2)

                # 比率测试
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) < 2:
                        continue
                    m, n = match_pair
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)

                # 计算 RANSAC 内点数
                inliers_count = 0
                mask = None
                homography = None
                if len(good_matches) >= 4:
                    src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_candidate[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
                    inliers_count = np.sum(mask) if mask is not None else 0

                return {
                    'path': candidate_path,
                    'index': candidate_feat['index'],
                    'keypoints_count': len(kp_candidate),
                    'matches_count': len(good_matches),
                    'inliers_count': inliers_count,
                    'homography': homography,
                    'mask': mask
                }
            # 多线程执行
            candidate_results = []
            with ThreadPoolExecutor(max_workers=PROD_MATCH_QUERY_WORKERS) as executor:
                future_to_path = {
                    executor.submit(process_candidate, path, feat): path
                    for path, feat in self.candidate_features.items()
                }
                for future in as_completed(future_to_path):
                    candidate_results.append(future.result())
        
        # 找到最佳匹配
        if not candidate_results:
            return {"error": "没有成功的候选图像匹配"}
        
        best_match = max(candidate_results, key=lambda x: x['inliers_count'])
        second_best = None
        
        # 如果有多个候选，找到第二好的匹配
        if len(candidate_results) > 1:
            sorted_candidates = sorted(candidate_results, key=lambda x: x['inliers_count'], reverse=True)
            best_match = sorted_candidates[0]
            second_best = sorted_candidates[1]
        
        # 计算置信度
        confidence = '低'
        confidence_high_type = ''
        # if second_best is not None:
        #     if best_match['inliers_count'] > 100:
        #         if best_match['inliers_count'] > second_best['inliers_count'] * 1.5: # 1.5
        #             confidence = '高'
        #             confidence_high_type = 'inliers count'
        #         elif best_match['matches_count'] > second_best['matches_count'] * 1.5:
        #             confidence = '高'
        #             confidence_high_type = 'matches count'
        #         # elif best_match['inliers_count'] > second_best['inliers_count'] * 1.2: # 微点LOGO
        #         elif best_match['inliers_count'] > second_best['inliers_count'] * 1.02:  # 伊利
        #             confidence = '中'
        #             confidence_high_type = 'inliers count'
        # # elif best_match['inliers_count'] > 100:  # 如果只有一个候选，但有足够的内点
        if best_match['inliers_count'] > (best_match['keypoints_count'] * 0.6):  # 如果只有一个候选，但有足够的内点
            confidence = '高'
            confidence_high_type = 'inliers count'
        elif best_match['inliers_count'] > (best_match['keypoints_count'] * 0.012) or best_match['inliers_count'] > 50:  # 如果只有一个候选，但有足够的内点. android: 500, ios: 150
            confidence = '中'
            confidence_high_type = 'inliers count'
        
        # 准备最终结果
        result = {
            'query_keypoints': len(kp_query),
            'best_match': {
                'path': best_match['path'],
                'index': best_match['index'],
                'keypoints_count': best_match['keypoints_count'],
                'matches_count': best_match['matches_count'],
                'inliers_count': best_match['inliers_count'],
                'confidence': confidence,
                'confidence_high_type': confidence_high_type
            },
            'all_candidates': candidate_results,
            'total_candidates': len(candidate_results)
        }
        
        return result
    
    def draw_and_save_matches(self, img1, kp1, img2, kp2, matches, output_path, mask=None):
        """
        绘制匹配结果并保存为图像文件
        """
        if mask is not None:
            matches_mask = mask.ravel().tolist()
            draw_params = dict(matchColor=(0, 255, 0),  # 绘制绿色匹配线
                            singlePointColor=None,
                            matchesMask=matches_mask,  # 只绘制内点
                            flags=2)
        else:
            draw_params = dict(matchColor=(0, 255, 0),
                            singlePointColor=None,
                            flags=2)
        
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
        cv2.imwrite(output_path, img_matches)

def matcher_go(query_image, correct=True):
    success = False
    msg = '无效的图像！请对准图像并保持手机稳定再拍照。'

    if not matcher.load_features():
        # 候选原图列表
        
        # candidate_images = [
        #     build_path(f'{PROD_NAME}/candidate01.png' + ('.roi.png' if PROD_ORIG_USE_ROI else '')),
        #     build_path(f'{PROD_NAME}/candidate02.png' + ('.roi.png' if PROD_ORIG_USE_ROI else '')),
        #     build_path(f'{PROD_NAME}/candidate03.png' + ('.roi.png' if PROD_ORIG_USE_ROI else ''))
        # ]

        if PROD_ORIG_USE_ROI:
            files = glob.glob(os.path.join(build_path(f'{PROD_NAME}'), "*.roi.png"))
        else:
            files = glob.glob(os.path.join(build_path(f'{PROD_NAME}'), "candidate[0-9][0-9].png"))
        files.sort()
        candidate_images = list(files)
        
        # 一次性初始化所有候选原图
        logger.info("正在初始化候选图像特征...")
        init_results = matcher.initialize_candidates(candidate_images)
        logger.info(f"初始化完成: {len([r for r in init_results.values() if r['status'] == 'success'])} 个成功")
        
        # 保存特征到文件，下次可以直接加载
        matcher.save_features()

        matcher.load_features()
    
    if PROD_DETECT_EDGE_ALG == 'hsv':
        query_image_corrected = matcher.detect_edge_hsv(None, query_image, roi_save=False, dst_size=(matcher.img_height, matcher.img_width), correct=correct)
    else:
        query_image_corrected = matcher.detect_edge(None, query_image, roi_save=False, dst_size=(matcher.img_height, matcher.img_width), correct=correct)

    if query_image_corrected is None:
        logger.info(f'\n无效的产品样例图像')
        return success, msg
    
    # Similarity
    if PROD_MATCH_SIMILITY_EN:
        is_prod = matcher.match_prod(query_image_corrected)
        if not is_prod:
            logger.info(f'\n无效的产品样例图像')
            return success, msg

    logger.info(f"\n正在匹配查询图像...")
    result = matcher.match_query_image(None, query_image_corrected)
    
    matched_image_id = ''
    # 打印结果
    if 'error' in result:
        logger.info(f"错误: {result['error']}")
    else:
        logger.info("=== 匹配结果 ===")
        logger.info(f"查询图像关键点数量: {result['query_keypoints']}")
        best = result['best_match']
        if best['confidence'] == '低':
            logger.info(f"置信度低，请尽量调整拍照角度，保持光线充足，并且保持图像画面水平。")
        else:
            logger.info(f"图像路径: {best['path']}")
            logger.info(f"图像索引: {best['index']}")
            logger.info(f"关键点数量: {best['keypoints_count']}")
            logger.info(f"匹配点数: {best['matches_count']}")
            logger.info(f"RANSAC内点数: {best['inliers_count']}")
            logger.info(f"置信度: {best['confidence']}")
            logger.info(f"置信度匹配类型: {best['confidence_high_type']}")
            matched_image_id = '00' + str(best['index'] + 1)
            success = True
            if matched_image_id in MAP_ID.keys():
                # msg = matched_image_id
                msg = MAP_ID[matched_image_id]
        
        logger.info("\n--- 所有候选图像结果 ---")
        for i, candidate in enumerate(result['all_candidates']):
            logger.info(f"{i+1}. {candidate['path']} - "
                  f"关键点: {candidate['keypoints_count']}, "
                  f"匹配点: {candidate['matches_count']}, "
                  f"内点: {candidate['inliers_count']}, "
                  f"内比: {(candidate['inliers_count']/candidate['keypoints_count']):.3f}")
    
    logger.info(f'\n匹配到标识: {matched_image_id}')
    return success, msg

def generate_edges_png():
    orig_img_path = build_path('MonkeyKing/P20220725-A5-01.png.roi.png')
    img_bgr = cv2.imread(orig_img_path)
    ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr)

    # 2. 边缘检测
    edges = cv2.Canny(Y, 220, 255) # threshold1, threshold2 越小，捕捉更多的边缘信息

    # ------------------------
    # 生 成 轮 廓 图
    # ------------------------
    edges = 255 - edges
    # 构造 RGBA
    #   - 黑色像素 (0) → 不透明
    #   - 白色像素 (255) → 完全透明
    alpha = np.where(edges == 0, 255, 0).astype(np.uint8)
    rgba = np.dstack([edges, edges, edges, alpha])
    # 保存为 PNG
    Image.fromarray(rgba).save(orig_img_path + '-edges.png', format="png", dpi=(DPI, DPI))

    logger.info('')

# 创建匹配器实例
# matcher = SIFTMatcher(build_path('MonkeyKing/candidate_features.pkl'))
matcher = SIFTMatcher(build_path(f'{PROD_NAME}/candidate_features.pkl'))

# 使用示例
if __name__ == "__main__":
    '''
    matcher.detect_edge(build_path(f'{PROD_NAME}/candidate02.png'), None, roi_save=True)
    matcher.detect_edge(build_path(f'{PROD_NAME}/candidate03.png'), None, roi_save=True)
    '''
    # files = glob.glob(os.path.join(build_path(f'{PROD_NAME}'), "candidate[0-9][0-9].png"))
    # files.sort()
    # for _f in files:
    #     matcher.detect_edge(_f, None, roi_save=True)
    
    # 匹配查询图像
    # query_image_path = build_path('MonkeyKing/001_IMG_20250822_143418/IMG_20250822_143255_cut.png.corrected.png')
    # query_image_path = build_path('MonkeyKing/002_IMG_20250821_191439/IMG_20250821_191423_1_cut.png.corrected.png')
    # query_image_path = build_path('MonkeyKing/003_IMG_20250822_113301/IMG_20250822_113223_1_cut.png.corrected.png')
    # query_image_path = build_path('MonkeyKing/001_verify_IMG_20250822_182237/IMG_20250822_182144_cut.png.corrected.png')
    # query_image_path = build_path('MonkeyKing/002_verify_IMG_20250822_183125/IMG_20250822_183120_cut.png.corrected.png')

    # query_image_path = build_path(f'{PROD_NAME}/candidate01.png')
    
    # query_image_path = build_path(f'{PROD_NAME}/../error-wm_2025-09-08_090556.png')
    # query_image_path = build_path(f'{PROD_NAME}/../error-wm_2025-09-05_111016.png')
    # query_image_path = build_path(f'{PROD_NAME}/../error-wm_2025-09-08_130033.png')
    # query_image_path = build_path(f'/Users/qchen/Downloads/ok-wm_2025-09-09_103150.png')
    # query_image_path = build_path(f'{PROD_NAME}/candidate01.png')
    # query_image_path = build_path(f'/Users/qchen/Downloads/ok-wm_2025-09-16_110721.png') # 复制品
    # query_image_path = build_path(f'{PROD_NAME}/../ok-wm_2025-09-19_175324.png')
    # query_image_path = build_path(f'{PROD_NAME}/../ok-wm_2025-09-19_193855-is-not-blue.png')
    # query_image_path = build_path(f'{PROD_NAME}/../../../tmp-images/2025-09-19/error-wm_2025-09-19_203418.png')
    # query_image_path = build_path(f'{PROD_NAME}/../../../tmp-images/2025-09-20/error-wm_2025-09-20_132701_crop.png')

    # query_image_path = build_path(f'{PROD_NAME}/IMG_20250920_131800.jpg') # 打印厂翻拍 -> 打印

    # query_image_path = build_path(f'{PROD_NAME}/candidate01.png')
    query_image_path = build_path(f'{PROD_NAME}/../../../tmp-images/2025-11-27/error-wm_2025-11-27_165409.png')
    
    
    
    
    
    
    logger.info(f"\n正在匹配查询图像: {query_image_path}")
    img_bgr  = cv2.imread(query_image_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    matched_image_id = matcher_go(img_bgr, correct=True) # correct True / False
    