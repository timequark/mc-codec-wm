import os
import glob
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pywt
import matplotlib.pyplot as plt
from pathlib import Path

# 允许直接以脚本方式运行：把项目根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

'''
随机噪点 、高斯模糊、分形噪声叠加
'''

DATA_ROOT = "images/mkking"

DPI = 600

DEBUG = True

def build_path(f):
    return os.path.join(DATA_ROOT, f)

def save_image(image, filename):
    # 将 BGR 转换为 RGB
    rgb_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 转换为 PIL Image
    pil_img = Image.fromarray(rgb_array)
    pil_img.save(filename, format='png', dpi=(DPI, DPI))
    # cv2.imwrite(filename, image)

def bgra_to_bgr_white(bg_img):
    """
    将 BGRA 图像转换为 BGR，alpha=0 的区域填充为白色
    :param bg_img: numpy.ndarray, shape=(H, W, 4), dtype=uint8
    :return: numpy.ndarray, shape=(H, W, 3), dtype=uint8
    """
    if bg_img.shape[2] != 4:
        raise ValueError("输入图像必须是 BGRA 格式")

    # 拆通道
    b, g, r, a = cv2.split(bg_img)
    alpha = a.astype(np.float32) / 255.0  # 归一化 alpha 到 [0,1]

    # 原始 BGR 转 float
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    # 背景白色 (255)
    b = b * alpha + 255 * (1 - alpha)
    g = g * alpha + 255 * (1 - alpha)
    r = r * alpha + 255 * (1 - alpha)

    # 合并
    bgr = cv2.merge([b, g, r]).astype(np.uint8)
    return bgr

def detect_edge(img_bgr):
    # 1. 转换颜色空间
    ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr)
    # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('_', img_gray)
    # cv2.waitKey(0)

    # 2. 边缘检测
    edges = cv2.Canny(Y, 200, 255) # threshold1, threshold2 越小，捕捉更多的边缘信息
    # edges = cv2.Canny(img_gray, 200, 255) # threshold1, threshold2 越小，捕捉更多的边缘信息

    if DEBUG:
        cv2.imshow('_', edges)
        cv2.waitKey(0)
    return edges

# def add_alpha(image, alpha_value=128):
#     """
#     给前景图手动修改 alpha 值：
#     - 原本完全不透明（alpha=255）的像素设置为 alpha_value
#     - 原本部分透明或透明的像素保持不变
#     :param foreground: PIL.Image, RGB 或 RGBA
#     :param alpha_value: 0~255, 用于原本 alpha=255 的像素
#     :return: RGBA 图像
#     """
#     # 确保有 alpha 通道
#     if image.mode != 'RGBA':
#         image = image.convert('RGBA')

#     # 转成 numpy 数组
#     fg_array = np.array(image)
#     # fg_array shape: (H, W, 4)

#     # 找到原本 alpha=255 的像素
#     alpha_channel = fg_array[:, :, 3]
#     mask = (alpha_channel == 255)

#     # 设置新的 alpha 值
#     alpha_channel[mask] = alpha_value

#     # 更新 alpha 通道
#     fg_array[:, :, 3] = alpha_channel

#     # 转回 PIL.Image
#     return Image.fromarray(fg_array, mode='RGBA')

def add_alpha(image, max_to_white=False, alpha_value=128, threshold=80):
    """
    给前景图手动修改 alpha 值：
    - 原本完全不透明（alpha=255）的像素设置为 alpha_value
    - 原本部分透明或透明的像素保持不变
    - 亮度小于阈值（如80）的像素设为完全透明
    :param image: PIL.Image, RGB 或 RGBA
    :param alpha_value: 0~255, 用于原本 alpha=255 的像素
    :param threshold: int, 灰度阈值，小于此值的像素完全透明
    :return: RGBA 图像
    """
    # 确保有 alpha 通道
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # 转成 numpy 数组
    fg_array = np.array(image)  # (H, W, 4)

    # 拆分通道
    r, g, b, a = fg_array[:,:,0], fg_array[:,:,1], fg_array[:,:,2], fg_array[:,:,3]
    if max_to_white:
        max_rgb = np.maximum.reduce([r, g, b])  # 每个像素点的 max(r,g,b)
        # 找到最小值
        min_val = max_rgb.min()
        # 把等于 min_val 的元素替换为 1e6
        max_rgb[max_rgb == min_val] = 1e6

        # 避免除零
        scale = np.where(max_rgb > 0, 255.0 / max_rgb, 1.0)

        # 等比放大
        r = r * scale
        g = g * scale
        b = b * scale
        r = np.clip(r, 0, 255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)

    # 灰度计算 (ITU-R BT.601)
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # 找到原本 alpha=255 的像素
    mask_opaque = (a == 255)
    a[mask_opaque] = alpha_value

    # 找到灰度 < threshold 的像素
    mask_dark = (gray < threshold)
    a[mask_dark] = 0  # 完全透明

    # 更新 alpha
    fg_array[:,:,0] = r
    fg_array[:,:,1] = g
    fg_array[:,:,2] = b
    fg_array[:,:,3] = a

    return Image.fromarray(fg_array, mode='RGBA')

def merge_bg_foreground(background, foreground, crop_center=False, border=True, corner_radius=30, mask_file=None, fill_color=(255, 255, 255)):
    # 1. 合成前景
    if foreground:
        if foreground.mode != 'RGBA':
            foreground = foreground.convert('RGBA')

        bg_width, bg_height = background.size
        fg_width, fg_height = foreground.size

        # 裁剪 background 中心区域
        if crop_center:
            left   = (bg_width - fg_width) // 2
            top    = (bg_height - fg_height) // 2
            right  = left + fg_width
            bottom = top + fg_height
            background = background.crop((left, top, right, bottom)).convert('RGBA')
            bg_width, bg_height = background.size
        
        fg_layer = Image.new('RGBA', background.size, (0, 0, 0, 0))
        pos = ((bg_width - fg_width) // 2, (bg_height - fg_height) // 2)
        fg_layer.paste(foreground, pos, foreground)

        composite = Image.alpha_composite(background.convert('RGBA'), fg_layer)
    else:
        composite = background.convert('RGBA')  # 保留 alpha

    # 2. 添加边框
    if border:
        border_color = (255, 0, 0, 255)  # RGBA
        composite = ImageOps.expand(composite, border=6, fill=(255,255,255,255))  # 白色外边
        composite = ImageOps.expand(composite, border=2, fill=border_color)        # 红色内边

    # 3. 圆角裁剪
    if corner_radius > 0:
        w, h = composite.size
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, w, h), radius=corner_radius, fill=255)
        composite.putalpha(mask)
    
    # 4. 用 foreground 蒙板中的全黑区域作为 mask
    if foreground is not None and mask_file is not None:
        mask_gray = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        # 创建 alpha 通道：黑色=255，其它=0
        alpha = np.where(mask_gray == 0, 255, 0).astype(np.uint8)
        # 把 alpha 转换为 PIL Image
        alpha_img = Image.fromarray(alpha)
        # 把前景放在 composite 的对应位置
        mask_full = Image.new('L', composite.size, 0)
        # 粘贴到 mask_full
        mask_full.paste(alpha_img, pos)

        # 创建填充背景
        filled_bg = Image.new('RGBA', composite.size, fill_color + (255,))
        # 按 mask 混合
        composite = Image.composite(composite, filled_bg, mask_full)

    return composite  # 返回 RGBA，可直接保存 PNG

def merge_foreground_bg(background, foreground, crop_center=False, border=True, corner_radius=30, foreground_as_mask=False, mask_file=None, fill_color=(255, 255, 255), fg_weight=1.0):
    # 1. 合成前景
    if foreground:
        if foreground.mode != 'RGBA':
            foreground = foreground.convert('RGBA')

        # 仅用于视觉合成的前景副本，按前景比重 fg_weight(0~1) 缩放 alpha；
        # 原始 foreground 保持不变，供后面 foreground_as_mask 使用。
        fg_visual = foreground
        if fg_weight != 1.0:
            wgt = max(0.0, min(1.0, fg_weight))
            r, g, b, a = foreground.split()
            a = a.point(lambda v: int(round(v * wgt)))
            fg_visual = Image.merge('RGBA', (r, g, b, a))

        bg_width, bg_height = background.size
        fg_width, fg_height = foreground.size

        # 裁剪 background 中心区域
        if crop_center:
            left   = (bg_width - fg_width) // 2
            top    = (bg_height - fg_height) // 2
            right  = left + fg_width
            bottom = top + fg_height
            background = background.crop((left, top, right, bottom)).convert('RGBA')
            bg_width, bg_height = background.size
        
        fg_layer = Image.new('RGBA', background.size, (0, 0, 0, 0))
        bg_width, bg_height = background.size
        fg_width, fg_height = foreground.size
        pos = ((bg_width - fg_width) // 2, (bg_height - fg_height) // 2)
        fg_layer.paste(fg_visual, pos, fg_visual)

        # 前景 alpha==0 的位置，把背景对应位置的 alpha 也设为 0（用前景透明区在背景上打洞）
        bg_arr = np.array(background.convert('RGBA'))
        fa = np.array(foreground.split()[-1])                 # 前景 alpha (fg 尺寸)
        x0, y0 = pos
        sub = bg_arr[y0:y0 + fg_height, x0:x0 + fg_width, 3]  # 背景 alpha 在前景覆盖处的窗口
        sub[fa == 0] = 0
        bg_arr[y0:y0 + fg_height, x0:x0 + fg_width, 3] = sub
        background = Image.fromarray(bg_arr, 'RGBA')

        # 需要控制好 im2 的 alpha 值
        # composite = Image.alpha_composite(fg_layer, background.convert('RGBA')) # 背景 盖 前景
        composite = Image.alpha_composite(background.convert('RGBA'), fg_layer) # 前景 盖 背景
    else:
        composite = background.convert('RGBA')  # 保留 alpha

    # 2. 添加边框
    if border:
        border_color = (255, 0, 0, 255)  # RGBA
        composite = ImageOps.expand(composite, border=6, fill=(255,255,255,255))  # 白色外边
        composite = ImageOps.expand(composite, border=2, fill=border_color)        # 红色内边

    # 3. 圆角裁剪
    if corner_radius > 0:
        w, h = composite.size
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, w, h), radius=corner_radius, fill=255)
        composite.putalpha(mask)
    
    # 4. 用 foreground 的透明度作为最终 mask
    if foreground is not None and foreground_as_mask:
        # foreground alpha 通道（非透明区域为 255）
        fg_alpha = foreground.split()[-1]
        # 把前景放在 composite 的对应位置
        mask_full = Image.new('L', composite.size, 0)
        mask_full.paste(fg_alpha, pos)  # 把 alpha 放到 composite 的中心位置

        # 创建填充背景
        filled_bg = Image.new('RGBA', composite.size, fill_color + (255,))
        # 按 mask 混合
        composite = Image.composite(composite, filled_bg, mask_full)

    return composite  # 返回 RGBA，可直接保存 PNG

def motion_kernel(length=35, angle_deg=20):
    """
    生成一个任意角度的线性运动模糊核（用于“擦拭感”）
    """
    length = max(3, int(length))
    k = np.zeros((length, length), np.float32)
    cv2.line(k, (0, length // 2), (length - 1, length // 2), 1, 1)
    # 旋转到指定角度
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length), flags=cv2.INTER_LINEAR, borderValue=0)
    k /= max(k.sum(), 1e-6)  # 归一化
    return k

def wiped_noise_image(
    size=(800, 1200),
    density=0.1,            # 噪点密度（0~1），越小越稀疏
    delta_range=(8, 60),      # 每个噪点的“变暗幅度”范围（对255的减值）
    motion_len=45,            # 擦拭长度（核尺寸）
    motion_angle=20,          # 擦拭角度（度）
    alpha=0.85,               # 擦拭后噪点整体强度（0~1）
    soften=3,                 # 额外软化模糊（高斯核尺寸，奇数；0表示不软化）
    seed=42                   # 随机种子，复现实验
):
    h, w = size
    rng = np.random.default_rng(seed)

    # 1) 白底
    base = np.full((h, w), 255, np.uint8)

    # 2) 低密度随机噪点（先做“减白”的强度图）
    mask = rng.random((h, w)) < density
    deltas = np.zeros((h, w), np.float32)
    if mask.any():
        deltas[mask] = rng.integers(delta_range[0], delta_range[1], size=mask.sum())

    # 3) 方向性“擦拭”模糊（运动模糊核）
    k = motion_kernel(motion_len, motion_angle)
    wiped = cv2.filter2D(deltas, -1, k, borderType=cv2.BORDER_REPLICATE)

    # 4) 可选：柔化边缘，让擦痕更像被“抹开”
    if soften and soften >= 3:
        soften = soften + (1 - soften % 2)  # 保证奇数
        wiped = cv2.GaussianBlur(wiped, (soften, soften), 0)

    # 5) 强度缩放并叠加到白底（从255减去）
    wiped *= float(alpha)
    out = base.astype(np.float32) - wiped
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out

def stone_texture_on_white(
    size=(800, 800),
    scale=6.0,
    octaves=3,
    persistence=0.5,
    contrast=1.6,
    dark_ratio=0.3,
    texture_strength=60,
    zero_ratio=0.8,
    blur_kernel_size=0,  # 新增：高斯模糊核大小，必须是正奇数
    blur_sigma=0.8,      # 新增：高斯模糊的标准差，控制模糊程度
    blur_iter=1,         # 新增：高斯模糊轮数
    seed=42
):
    '''
    :param scale: 降低噪声频率 → 纹理更大块，会让噪声“颗粒”更大，石头块更稀疏
    :param octaves: 减少噪声层数 (octaves) → 减少细节，会让纹理更单一，没有那么复杂的细碎纹理
    :param contrast: 调低对比度 (contrast) → 让明暗差别小，降低 contrast (如 1.6 → 1.2 或 1.0)，让噪声分布更均匀，避免小斑点过多
    :param texture_strength: 调低纹理强度 (texture_strength) → 减少视觉干扰，会让石头纹理更浅，整体更接近白底
    :param zero_ratio: 直接控制非噪点比例 0 ~ 1（自定义）
    :param blur_kernel_size: 高斯模糊核大小，必须是正奇数 (如3,5,7)。越大越模糊，设为0或1则不模糊
    :param blur_sigma: 高斯模糊的标准差，控制模糊程度。0表示自动计算
    :param blur_iter: 高斯模糊轮数
    '''
    h, w = size
    rng = np.random.default_rng(seed)

    # 白底
    base = np.full((h, w), 255, np.float32)

    # 基础噪声
    noise_base = rng.random((h, w)).astype(np.float32)
    
    # mask = rng.random((h, w)) < zero_ratio
    # noise_base = rng.random((h, w)).astype(np.float32)
    # noise_base[mask] = 0.0

    total_pixels = h * w
    num_zero = int(total_pixels * zero_ratio)
    # 随机挑选 num_zero 个像素置 0
    idx = rng.choice(total_pixels, num_zero, replace=False)

    # 方案一：单点
    # noise_base.flat[idx] = 0

    # 方案二：半径范围内的点群
    mask = np.ones((h, w), np.uint8)
    mask.flat[idx] = 0  # 先放中心点
    radius = 1
    # 定义膨胀结构元素（radius越大区域越大）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    mask = cv2.dilate(mask, kernel)  # 扩大为圆形范围
    # 应用在 noise_base
    noise_base[mask == 0] = 0

    # 分形噪声叠加
    noise = np.zeros_like(noise_base)
    freq, amp = 1.0, 1.0
    for _ in range(octaves):
        small = cv2.resize(noise_base, (max(1, int(w/freq)), max(1, int(h/freq))), interpolation=cv2.INTER_LINEAR)
        big   = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        noise += big * amp
        freq *= scale
        amp  *= persistence

    def safe_normalize(arr):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)  # 清 NaN/Inf
        return cv2.normalize(arr, None, 0, 1, cv2.NORM_MINMAX)
    
    # 归一化到 [0,1]
    noise = safe_normalize(noise)

    # 对比度增强
    noise = np.power(noise, contrast)
    noise = safe_normalize(noise)

    # 暗部稀疏
    thresh = np.quantile(noise, dark_ratio)
    noise = (noise - thresh) / (1 - thresh)
    noise = np.clip(noise, 0, 1)

    # 关键步骤：应用轻度高斯模糊来平滑颗粒感
    if blur_kernel_size > 1 and blur_sigma > 0:
        for _ in range(0, blur_iter):
            noise = cv2.GaussianBlur(noise, (blur_kernel_size, blur_kernel_size), blur_sigma)
        # 重新归一化以确保值在[0,1]范围内
        noise = safe_normalize(noise)
    
    # 转成纹理：白底 - 扰动
    texture = noise * texture_strength
    result = base - texture

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def stone_texture_on_black(
    size=(800, 800),
    scale=6.0,
    octaves=3,
    persistence=0.5,
    contrast=1.6,
    dark_ratio=0.3,
    texture_strength=60,
    zero_ratio=0.8,
    en_up_then_down_sample=True,  # 是否启用 Upsample → Downsample 模糊
    blur_kernel_size=0,  # 新增：高斯模糊核大小，必须是正奇数
    blur_sigma=0.8,      # 新增：高斯模糊的标准差，控制模糊程度
    blur_iter=1,         # 新增：高斯模糊轮数
    seed=42
):
    '''
    :param scale: 降低噪声频率 → 纹理更大块，会让噪声“颗粒”更大，石头块更稀疏
    :param octaves: 减少噪声层数 (octaves) → 减少细节，会让纹理更单一，没有那么复杂的细碎纹理
    :param contrast: 调低对比度 (contrast) → 让明暗差别小，降低 contrast (如 1.6 → 1.2 或 1.0)，让噪声分布更均匀，避免小斑点过多
    :param texture_strength: 调低纹理强度 (texture_strength) → 减少视觉干扰，会让石头纹理更浅，整体更接近白底
    :param zero_ratio: 直接控制非噪点比例 0 ~ 1（自定义）
    :param en_up_then_down_sample: 是否启用 Upsample → Downsample 模糊，这是一种在高分辨率下更自然的模糊方法，推荐开启，尤其是当 blur_kernel_size 较大时
    :param blur_kernel_size: 高斯模糊核大小，必须是正奇数 (如3,5,7)。越大越模糊，设为0或1则不模糊
    :param blur_sigma: 高斯模糊的标准差，控制模糊程度。0表示自动计算
    :param blur_iter: 高斯模糊轮数
    '''
    h, w = size
    rng = np.random.default_rng(seed)

    # 白底
    base = np.full((h, w), 0, np.float32)

    # 基础噪声
    noise_base = rng.random((h, w)).astype(np.float32)
    
    # mask = rng.random((h, w)) < zero_ratio
    # noise_base = rng.random((h, w)).astype(np.float32)
    # noise_base[mask] = 0.0

    total_pixels = h * w
    num_zero = int(total_pixels * zero_ratio)
    # 随机挑选 num_zero 个像素置 0
    idx = rng.choice(total_pixels, num_zero, replace=False)

    # 方案一：单点
    # noise_base.flat[idx] = 0

    # 方案二：半径范围内的点群
    mask = np.ones((h, w), np.uint8)
    mask.flat[idx] = 0  # 先放中心点
    radius = 1
    # 定义膨胀结构元素（radius越大区域越大）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    mask = cv2.dilate(mask, kernel)  # 扩大为圆形范围
    # 应用在 noise_base
    noise_base[mask == 0] = 0

    # 分形噪声叠加
    noise = np.zeros_like(noise_base)
    freq, amp = 1.0, 1.0
    for _ in range(octaves):
        small = cv2.resize(noise_base, (max(1, int(w/freq)), max(1, int(h/freq))), interpolation=cv2.INTER_LINEAR)
        big   = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        noise += big * amp
        freq *= scale
        amp  *= persistence

    def safe_normalize(arr):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)  # 清 NaN/Inf
        return cv2.normalize(arr, None, 0, 1, cv2.NORM_MINMAX)
    
    # 归一化到 [0,1]
    noise = safe_normalize(noise)

    # 对比度增强
    noise = np.power(noise, contrast)
    noise = safe_normalize(noise)

    # 暗部稀疏
    thresh = np.quantile(noise, dark_ratio)
    noise = (noise - thresh) / (1 - thresh)
    noise = np.clip(noise, 0, 1)

    # ⭐ Upsample → Downsample 模糊
    if en_up_then_down_sample:
        scale_up = 2.0   # 推荐 1.5 ~ 3.0

        H, W = noise.shape

        # 放大（用线性或立方插值）
        up = cv2.resize(noise, (int(W*scale_up), int(H*scale_up)),
                        interpolation=cv2.INTER_CUBIC)

        # 可选：轻微 Gaussian（在高分辨率下更自然）
        up = cv2.GaussianBlur(up, (3, 3), 0.5)

        # 缩小（关键：用 AREA）
        noise = cv2.resize(up, (W, H),
                        interpolation=cv2.INTER_AREA)

    # 关键步骤：应用轻度高斯模糊来平滑颗粒感
    if blur_kernel_size > 1 and blur_sigma > 0:
        for _ in range(0, blur_iter):
            noise = cv2.GaussianBlur(noise, (blur_kernel_size, blur_kernel_size), blur_sigma)
        # 重新归一化以确保值在[0,1]范围内
        noise = safe_normalize(noise)
    
    # 转成纹理：白底 - 扰动
    texture = noise * texture_strength
    result = base + texture

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

# def stone_texture_on_black(
#     size=(800, 800),
#     scale=6.0,
#     octaves=3,
#     persistence=0.5,
#     contrast=1.6,
#     dark_ratio=0.3,
#     texture_strength=60,
#     zero_ratio=0.8,
#     blur_kernel_size=0,  # 新增：高斯模糊核大小，必须是正奇数
#     blur_sigma=0.8,      # 新增：高斯模糊的标准差，控制模糊程度
#     blur_iter=1,         # 新增：高斯模糊轮数
#     seed=42
# ):
#     result = stone_texture_on_white(
#         size=size,
#         scale=scale,
#         octaves=octaves,
#         persistence=persistence,
#         contrast=contrast,
#         dark_ratio=dark_ratio,
#         texture_strength=texture_strength,
#         zero_ratio=zero_ratio,
#         blur_kernel_size=blur_kernel_size,
#         blur_sigma=blur_sigma,
#         blur_iter=blur_iter,
#         seed=seed
#     )
#     return 255 - result

# def generate_sparse_noise_texture(width=800, height=800,
#                                   bg_white=True,
#                                   seed=11,
#                                   density=0.02,
#                                   dot_min=60, dot_max=140,
#                                   blur_ksize=5):
#     """
#     生成稀疏颗粒噪点纹理（带强度范围）
#     :param width: 纹理宽
#     :param height: 纹理高
#     :param density: 噪点密度 (0~1)
#     :param dot_min: 噪点最低强度（越小越浅）
#     :param dot_max: 噪点最高强度（越大越深）
#     :param blur_ksize: 模糊柔化噪点（奇数）
#     """

#     rng = np.random.default_rng(seed)

#     # 1) 白背景或黑背景
#     base = np.full((height, width), 255 if bg_white else 0, dtype=np.uint8)

#     # 2) 随机选点
#     total_pixels = width * height
#     num_points = int(total_pixels * density)
#     idxs = rng.choice(total_pixels, num_points, replace=False)

#     # 3) 随机噪点强度范围
#     dot_values = rng.integers(dot_min, dot_max + 1, size=num_points)
#     base.flat[idxs] = 255 - dot_values if bg_white else dot_values

#     # 4) 模糊柔化
#     if blur_ksize > 1:
#         # base = cv2.GaussianBlur(base, (blur_ksize, blur_ksize), cv2.BORDER_DEFAULT)
#         base = cv2.GaussianBlur(base, (blur_ksize, blur_ksize), 0)

#     # 5) 转三通道
#     texture = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
#     return texture

def generate_sparse_noise_texture(width=800, height=800,
                                  bg_white=True,
                                  seed=11,
                                  density=0.02,
                                  dot_size=2,
                                  dot_min=60, dot_max=140,
                                  blur_ksize=5):
    """
    生成稀疏颗粒噪点纹理（带强度范围）,
    可以噪点尺寸

    :param width: 纹理宽
    :param height: 纹理高
    :param density: 噪点密度 (0~1)
    :param dot_size: 点尺寸
    :param dot_min: 噪点最低强度（越小越浅）
    :param dot_max: 噪点最高强度（越大越深）
    :param blur_ksize: 模糊柔化噪点（奇数）
    """
    rng = np.random.default_rng(seed)

    # 背景
    base = np.full((height, width), 255 if bg_white else 0, dtype=np.uint8)

    # 随机中心点
    total_pixels = width * height
    num_points = int(total_pixels * density)

    xs = rng.integers(0, width, size=num_points)
    ys = rng.integers(0, height, size=num_points)

    dot_values = rng.integers(dot_min, dot_max + 1, size=num_points)

    half = dot_size // 2

    for x, y, val in zip(xs, ys, dot_values):
        color = 255 - val if bg_white else val

        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(width, x1 + dot_size)
        y2 = min(height, y1 + dot_size)

        base[y1:y2, x1:x2] = color

    if blur_ksize > 1:
        base = cv2.GaussianBlur(base, (blur_ksize, blur_ksize), 0)

    texture = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    return texture

import cv2
import numpy as np


def generate_sparse_noise_texture_transparent(
        width=800,
        height=800,
        seed=11,
        density=0.02,
        dot_size=2,
        dot_min=60,
        dot_max=140,
        blur_ksize=0):
    """
    生成透明背景噪点图

    返回:
        BGRA (uint8)

    alpha:
        背景 = 0
        噪点 = 255
    """

    rng = np.random.default_rng(seed)

    # BGRA
    texture = np.zeros((height, width, 4), dtype=np.uint8)

    total_pixels = width * height
    num_points = int(total_pixels * density)

    xs = rng.integers(0, width, size=num_points)
    ys = rng.integers(0, height, size=num_points)

    dot_values = rng.integers(
        dot_min,
        dot_max + 1,
        size=num_points)

    half = dot_size // 2

    for x, y, val in zip(xs, ys, dot_values):

        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(width, x1 + dot_size)
        y2 = min(height, y1 + dot_size)

        # 灰度颜色
        texture[y1:y2, x1:x2, 0] = val  # B
        texture[y1:y2, x1:x2, 1] = val  # G
        texture[y1:y2, x1:x2, 2] = val  # R
        texture[y1:y2, x1:x2, 3] = 255  # A

    # 可选：柔化边缘
    if blur_ksize > 1:
        alpha = texture[:, :, 3]
        alpha = cv2.GaussianBlur(
            alpha,
            (blur_ksize, blur_ksize),
            0)

        texture[:, :, 3] = alpha

    return texture

def merge_by_mask(
    texture_bgra,
    texture_bgra_merged,
    mask_bgra,
    use_blur=False,           # 开关
    blur_ksize=5,             # 高斯核大小（必须是奇数）
    blur_sigma=1.0            # sigma
):
    """
    根据 mask alpha 选择使用哪张图：
    - mask alpha == 0 -> texture_bgra_merged（可选 GaussianBlur）
    - mask alpha > 0 -> texture_bgra

    如果 texture_bgra_merged 通道数不为4，则自动加 alpha=255
    """

    # 检查通道数
    if texture_bgra.shape[2] != 4 or mask_bgra.shape[2] != 4:
        raise ValueError("texture_bgra 和 mask_bgra 必须为 BGRA 格式")

    # 处理 texture_bgra_merged
    if texture_bgra_merged.shape[2] == 4:
        merged_bgra = texture_bgra_merged.copy()
    elif texture_bgra_merged.shape[2] == 3:
        alpha = np.full(texture_bgra_merged.shape[:2], 255, dtype=np.uint8)
        merged_bgra = np.dstack((texture_bgra_merged, alpha))
    else:
        raise ValueError("texture_bgra_merged 必须是 BGR 或 BGRA 格式")

    # === 可选 GaussianBlur ===
    if use_blur:
        if blur_ksize % 2 == 0 or blur_ksize <= 0:
            raise ValueError("blur_ksize 必须是正奇数")

        # 只对 BGR 做 blur，alpha 保持不变（更合理）
        bgr = merged_bgra[..., :3]
        alpha = merged_bgra[..., 3:]

        bgr_blur = cv2.GaussianBlur(bgr, (blur_ksize, blur_ksize), blur_sigma)
        merged_bgra = np.dstack((bgr_blur, alpha))

    # mask alpha
    mask_alpha = mask_bgra[..., 3]
    valid_mask = (mask_alpha == 0)[..., None]  # HxWx1

    # 合成
    result = np.where(valid_mask, merged_bgra, texture_bgra)

    return result

def clip_by_mask(texture_bgra, mask_bgra, reverse=False):
    """
    根据 mask alpha 裁剪 texture

    mask alpha > 0:
        保留 texture

    mask alpha == 0:
        设置为透明

    return:
        BGRA
    """

    if mask_bgra is None:
        return texture_bgra.copy()
    
    if texture_bgra.shape[2] != 4:
        raise ValueError("texture_bgra 必须是 BGRA")

    if mask_bgra.shape[2] != 4:
        raise ValueError("mask_bgra 必须是 BGRA")

    result = texture_bgra.copy()

    mask_alpha = mask_bgra[..., 3]

    if not reverse:
        result[..., 3] = np.where(
            mask_alpha > 0,
            result[..., 3],
            0
        )
    else:
        result[..., 3] = np.where(
            mask_alpha == 0,
            result[..., 3],
            0
        )

    return result

def save_image_pil(image, filename):
    if image.shape[2] == 4:
        # BGRA → RGBA
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        pil_img = Image.fromarray(rgba)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

    pil_img.save(filename, format='PNG', dpi=(DPI, DPI))

if __name__ == "__main__":
    background = cv2.imread(build_path('candidate-01-raw.png'), cv2.IMREAD_UNCHANGED)
    mask_bgra = cv2.imread(build_path('mask.png'), cv2.IMREAD_UNCHANGED) if os.path.exists(build_path('mask.png')) else None
    h, w = background.shape[:2]

    # wiped_ = wiped_noise_image(
    #     size=(480, 320),
    #     density=0.006,        # 更稀疏
    #     delta_range=(200, 220), # 局部可更暗一些
    #     motion_len=55,        # 擦痕更长
    #     motion_angle=15,      # 擦拭方向略倾斜
    #     alpha=0.9,
    #     soften=5,
    #     seed=2025
    # )

    # 示例
    # stone = stone_texture_on_white(
    #     size=(480, 480),
    #     scale=5.0,
    #     octaves=4,
    #     persistence=0.55,
    #     contrast=1.8,
    #     dark_ratio=0.15,
    #     texture_strength=70,
    #     seed=2025
    # )

    # logo-wecode-001.png, logo-wecode-002.png, logo-wecode-003.png
    # cfgs = [
    #     # seed, scale,  octaves,    persistence,    contrast,   noise_ratio, texture_strength
    #     (2024,  12.0,   10,         0.75,           3.0,        0.2,       120),
    #     (2025,  16.0,   12,         0.75,           2.0,        0.10,       90),
    #     (1990,  14.0,   12,         0.75,           2.5,        0.10,       110)
    # ]
    
    
    cfgs = [
        # seed, scale,  octaves,    persistence,    contrast,   noise_ratio, texture_strength
        (1901,  5.0,   1,         0.20,           2.0,        0.2,        280),
        # (1911,  5.0,   2,         0.20,           2.0,        0.05,        255),
    ]
    
    # test GaussianBlur
    # cfgs = [
    #     # seed, scale,  octaves,    persistence,    contrast,   noise_ratio, texture_strength
    #     (1901,  5.0,   2,         0.20,           2.0,        0.02,        160),
    #     (1911,  5.0,   2,         0.20,           2.0,        0.05,        160),
    #     (1921,  5.0,   2,         0.20,           2.0,        0.05,        160),
    #     (1905,  5.0,   2,         0.20,           2.0,        0.05,        160),
    #     (1908,  5.0,   2,         0.20,           2.0,        0.05,        160),
    #     (1912,  5.0,   2,         0.20,           2.0,        0.05,        160),
    #     (1919,  5.0,   2,         0.20,           2.0,        0.05,        160),
    # ]

    id_offset = 0
    
    background_transparent = False

    for i, _tp in enumerate(cfgs):
        '''
        stone_texture_on_white
        stone_texture_on_black
        '''
        # 方式一：Perlin 分形噪声纹理（更自然，但参数较多，调试较复杂）
        # stone = stone_texture_on_black(
        # # stone = stone_texture_on_white(
        #     size=(h, w),
        #     scale=_tp[1],
        #     octaves=_tp[2],
        #     persistence=_tp[3],
        #     contrast=_tp[4],
        #     dark_ratio=_tp[5],
        #     texture_strength=_tp[6],
        #     zero_ratio=0.9,
        #     en_up_then_down_sample=True, # 上采样+下采样模糊，低通滤波（但更接近真实视觉模糊）
        #     blur_kernel_size=0,
        #     blur_sigma=1,
        #     blur_iter=1,
        #     seed=_tp[0]
        # )

        # 方式二：使用简单随机噪点纹理、加高斯模糊，铺开 gray 灰度
        '''
        width=w, height=h,
            bg_white=False,
            seed=11000,
            density=0.5,
            dot_min=200, dot_max=255,
            blur_ksize=3
        hat:
            - v1
            density=0.04,
            dot_min=200, dot_max=255,
            blur_ksize=0

            blur: 3, sigma 1

            - v2
            density=0.02,
            dot_min=200, dot_max=255,
            blur_ksize=0

            blur: 3, sigma 0.5

            - v2.1
            density=0.02,
            dot_size=2,
            dot_min=200, dot_max=255,
            blur_ksize=0

            blur: 3, sigma 0.5

            - v2.2
            density=0.01,
            dot_size=4,
            dot_min=200, dot_max=255,
            blur_ksize=0

            blur: 3, sigma 0.5

            - v2.3
            density=0.01,
            dot_size=3,
            dot_min=200, dot_max=255,
            blur_ksize=0

            blur: 3, sigma 0.5
        
        pregnant:
            - v1
            density=0.04,
            dot_min=200, dot_max=255,
            blur_ksize=0

            blur: 3, sigma 1

            - v2
            density=0.02,
            dot_min=200, dot_max=255,
            blur_ksize=0

            blur: 3, sigma 0.5
        '''
        tag = '01'
        if not background_transparent:
            stone = generate_sparse_noise_texture(
                width=w, height=h,
                bg_white=True,
                seed=11000,
                density=0.06,
                dot_size=1,
                dot_min=220, dot_max=255,
                blur_ksize=0
            )
            id = '00' + str(i + 1 + id_offset)
            # cv2.imshow(id, stone)
            # cv2.waitKey(0)
            save_image(stone, build_path(f'noise-{id}.png'))
        else:
            stone = generate_sparse_noise_texture_transparent(
                width=w, height=h,
                seed=11000,
                density=0.01,
                dot_size=3,
                dot_min=200, dot_max=255,
                blur_ksize=0)
            id = '00' + str(i + 1 + id_offset)
            cv2.imwrite(build_path(f'noise-{id}.png'), stone)

    cv2.destroyAllWindows()

    for i, _tp in enumerate(cfgs):
        id = '00' + str(i + 1 + id_offset)
        
        if not background_transparent:
            # 前景 -> 背景
            pil_img_background = Image.open(build_path(f'noise-{id}.png'))
            
            # pil_img_background = add_alpha(pil_img_background, alpha_value=255, threshold=200)

            # 模糊：只对 BGR 做 blur，alpha 保持不变（更合理）
            merged_bgra = cv2.imread(build_path(f'noise-{id}.png'), cv2.IMREAD_UNCHANGED)
            blur_ksize, blur_sigma = 5, 0.0
            bgr = merged_bgra[..., :3]
            alpha = merged_bgra[..., 3:]
            bgr_blur = cv2.GaussianBlur(bgr, (blur_ksize, blur_ksize), blur_sigma)
            # 依据灰度重建 alpha：GRAY>125 设为 0(透明)，否则 255(不透明)
            gray = cv2.cvtColor(bgr_blur, cv2.COLOR_BGR2GRAY)
            alpha = np.where(gray > 255, 0, 255).astype(np.uint8)
            merged_bgra = np.dstack((bgr_blur, alpha))
            cv2.imshow(f'Blurred noise-{id}', merged_bgra)
            cv2.waitKey(0)
            save_image_pil(merged_bgra, build_path(f'noise-{id}.png'))
            pil_img_background = Image.open(build_path(f'noise-{id}.png'))

            pil_img_background.show()
            pil_img_front = add_alpha(Image.open(build_path(f'candidate-{tag}-raw.png')), alpha_value=255, threshold=0)

            composite = merge_foreground_bg(pil_img_background, pil_img_front, crop_center=False, border=False, corner_radius=0, mask_file=None, fg_weight=0.9)
            composite.show()
            composite.save(build_path(f'candidate-{tag}-raw-composite.png'), format='png', dpi=(DPI, DPI))

            # Mask Clipping
            texture_bgra = cv2.imread(build_path(f'candidate-{tag}-raw-composite.png'), cv2.IMREAD_UNCHANGED)
            image_with_noise = merge_by_mask(texture_bgra=background, texture_bgra_merged=texture_bgra, mask_bgra=mask_bgra, use_blur=False) if mask_bgra is not None else texture_bgra
            cv2.imshow('Image with noise', image_with_noise)
            cv2.waitKey(0)
            save_image_pil(image_with_noise, build_path(f'candidate-{tag}-raw-dpi{DPI}.png'))

            # - Mask Clipping
            # - Mask 区域再次 GaussianBlur, 感觉更自然，但会导致原图颜色发生变化
            # texture_bgra = cv2.imread(build_path(f'candidate-{tag}-raw-composite.png'), cv2.IMREAD_UNCHANGED)
            # image_with_noise = merge_by_mask(texture_bgra=background, texture_bgra_merged=texture_bgra, mask_bgra=mask_bgra, use_blur=True, blur_ksize=3, blur_sigma=0.5) if mask_bgra is not None else texture_bgra
            # cv2.imshow('Image with noise blurred', image_with_noise)
            # cv2.waitKey(0)
            # save_image_pil(image_with_noise, build_path(f'candidate-{tag}-raw-dpi{DPI}-blurred.png'))
        else:
            # 直接使用带 alpha 的噪点图作为前景，合成到原图上
            pil_img_front = Image.open(build_path(f'noise-{id}.png'))
            composite = pil_img_front
            composite.save(build_path(f'candidate-{tag}-raw-composite.png'), format='png', dpi=(DPI, DPI))

            # - Mask Clipping
            # - Mask 区域再次 GaussianBlur, 感觉更自然，但会导致原图颜色发生变化
            texture_bgra = cv2.imread(build_path(f'candidate-{tag}-raw-composite.png'), cv2.IMREAD_UNCHANGED)
            image_with_noise = clip_by_mask(texture_bgra=texture_bgra, mask_bgra=mask_bgra, reverse=True)
            cv2.imshow('Image with noise blurred', image_with_noise)
            cv2.waitKey(0)
            save_image_pil(image_with_noise, build_path(f'candidate-{tag}-raw-dpi{DPI}.png'))
    
    cv2.destroyAllWindows()

