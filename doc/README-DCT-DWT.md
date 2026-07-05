# DCT / DCT with quantization / DWT


https://cloud.tencent.com/developer/article/2589211

## DCT
```
import numpy as np
from scipy.fft import dctn, idctn
from PIL import Image
import matplotlib.pyplot as plt

def image_to_dct_blocks(image, block_size=8):
    """
    将图像分割成块并对每个块进行DCT变换
    
    参数:
    image: 输入图像（PIL Image对象或NumPy数组）
    block_size: DCT块大小，通常为8
    
    返回:
    dct_blocks: DCT变换后的系数块
    image_shape: 原始图像形状
    """
    # 转换为NumPy数组
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert('L'))  # 转换为灰度图像
    else:
        img_array = np.array(image)
    
    # 获取图像形状
    image_shape = img_array.shape
    height, width = image_shape
    
    # 计算需要处理的块数
    num_blocks_h = (height + block_size - 1) // block_size
    num_blocks_w = (width + block_size - 1) // block_size
    
    # 创建足够大的数组来存储所有块
    padded_height = num_blocks_h * block_size
    padded_width = num_blocks_w * block_size
    padded_img = np.zeros((padded_height, padded_width), dtype=np.float64)
    padded_img[:height, :width] = img_array
    
    # 初始化DCT系数块数组
    dct_blocks = np.zeros((padded_height, padded_width), dtype=np.float64)
    
    # 对每个块进行DCT变换
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # 提取块
            block = padded_img[i*block_size:(i+1)*block_size, 
                              j*block_size:(j+1)*block_size]
            
            # 应用DCT变换
            dct_block = dctn(block, norm='ortho')
            
            # 存储变换后的块
            dct_blocks[i*block_size:(i+1)*block_size, 
                      j*block_size:(j+1)*block_size] = dct_block
    
    return dct_blocks, image_shape

def dct_blocks_to_image(dct_blocks, original_shape, block_size=8):
    """
    将DCT系数块转换回图像
    
    参数:
    dct_blocks: DCT变换后的系数块
    original_shape: 原始图像形状
    block_size: DCT块大小，通常为8
    
    返回:
    重建的图像（NumPy数组）
    """
    # 获取DCT块的形状
    padded_height, padded_width = dct_blocks.shape
    
    # 计算块数
    num_blocks_h = padded_height // block_size
    num_blocks_w = padded_width // block_size
    
    # 初始化重建图像数组
    reconstructed_img = np.zeros((padded_height, padded_width), dtype=np.float64)
    
    # 对每个块进行IDCT变换
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # 提取DCT块
            dct_block = dct_blocks[i*block_size:(i+1)*block_size, 
                                 j*block_size:(j+1)*block_size]
            
            # 应用IDCT变换
            block = idctn(dct_block, norm='ortho')
            
            # 存储重建的块
            reconstructed_img[i*block_size:(i+1)*block_size, 
                             j*block_size:(j+1)*block_size] = block
    
    # 裁剪到原始大小
    height, width = original_shape
    reconstructed_img = reconstructed_img[:height, :width]
    
    # 确保像素值在有效范围内
    reconstructed_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8)
    
    return reconstructed_img

def visualize_dct_coefficients(dct_blocks, num_blocks=8):
    """
    可视化DCT系数
    
    参数:
    dct_blocks: DCT系数块
    num_blocks: 要显示的块数
    """
    # 只取前几个块进行可视化
    display_blocks = dct_blocks[:num_blocks*8, :num_blocks*8]
    
    # 取对数以增强可视化效果
    log_magnitude = np.log(1 + np.abs(display_blocks))
    
    plt.figure(figsize=(10, 10))
    plt.imshow(log_magnitude, cmap='gray')
    plt.title('DCT Coefficients (Log Magnitude)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('dct_coefficients.png')
    plt.close()
    
    return log_magnitude
```
### DCT系数进行嵌入
在频域隐写中，选择合适的DCT系数进行修改至关重要。根据人类视觉系统的特性，我们通常选择以下区域：

- 中频系数区域：人眼对中频系数的变化相对不敏感
- 避开DC系数：DC系数（左上角）代表块的平均值，修改会明显影响图像质量
- 避开高频系数：高频系数虽然人眼不敏感，但在压缩时容易被丢弃

```
def select_embedding_positions(block_size=8, avoid_dc=True, use_middle_freq=True):
    """
    选择适合嵌入信息的DCT系数位置
    
    参数:
    block_size: DCT块大小
    avoid_dc: 是否避开DC系数
    use_middle_freq: 是否只使用中频系数
    
    返回:
    positions: 适合嵌入的系数位置列表
    """
    positions = []
    
    if use_middle_freq:
        # 对于8x8块，中频区域通常是(2,2)到(5,5)之间的系数
        # 这些系数在视觉上相对不重要，但在压缩时不会被丢弃
        start_freq = 2
        end_freq = 5
        
        for i in range(start_freq, end_freq + 1):
            for j in range(start_freq, end_freq + 1):
                # 可以根据需要排除某些对角位置
                if avoid_dc or (i != 0 and j != 0):
                    positions.append((i, j))
    else:
        # 使用除DC系数外的所有系数
        for i in range(block_size):
            for j in range(block_size):
                if avoid_dc or (i != 0 and j != 0):
                    positions.append((i, j))
    
    return positions

def calculate_embedding_capacity(image_shape, positions_per_block, block_size=8):
    """
    计算图像的嵌入容量
    
    参数:
    image_shape: 图像形状 (高度, 宽度)
    positions_per_block: 每个块中用于嵌入的位置数量
    block_size: DCT块大小
    
    返回:
    capacity_bits: 可嵌入的比特数
    capacity_bytes: 可嵌入的字节数
    """
    height, width = image_shape
    
    # 计算块数
    num_blocks_h = (height + block_size - 1) // block_size
    num_blocks_w = (width + block_size - 1) // block_size
    total_blocks = num_blocks_h * num_blocks_w
    
    # 计算容量
    capacity_bits = total_blocks * positions_per_block
    capacity_bytes = capacity_bits // 8
    
    return capacity_bits, capacity_bytes
```

### DCT域LSB隐写实现
DCT域LSB隐写是对空间域LSB隐写的频域扩展。它在选定的DCT系数的最低有效位中嵌入信息。

```
def message_to_bits(message):
    """
    将消息转换为比特流
    
    参数:
    message: 要转换的字符串
    
    返回:
    bits: 比特流列表
    """
    bits = []
    for char in message:
        # 将字符转换为8位二进制
        byte = bin(ord(char))[2:].zfill(8)
        for bit in byte:
            bits.append(int(bit))
    
    # 添加结束标记（8个1）
    for _ in range(8):
        bits.append(1)
    
    return bits

def bits_to_message(bits):
    """
    将比特流转换回消息
    
    参数:
    bits: 比特流列表
    
    返回:
    message: 解码后的消息
    """
    message = ""
    i = 0
    
    # 处理比特流，直到遇到结束标记
    while i + 8 <= len(bits):
        # 检查是否是结束标记
        if bits[i:i+8] == [1]*8:
            break
        
        # 将8位转换为一个字符
        byte = bits[i:i+8]
        char_code = 0
        for bit in byte:
            char_code = (char_code << 1) | bit
        
        # 添加字符到消息
        message += chr(char_code)
        i += 8
    
    return message

def dct_lsb_embedding(image_path, message, output_path, block_size=8, positions=None):
    """
    在DCT域中使用LSB方法嵌入消息
    
    参数:
    image_path: 原始图像路径
    message: 要嵌入的消息
    output_path: 输出图像路径
    block_size: DCT块大小
    positions: 用于嵌入的DCT系数位置，如果为None则使用默认位置
    
    返回:
    是否成功嵌入
    """
    # 加载图像
    img = Image.open(image_path).convert('L')  # 转换为灰度图像
    
    # 进行DCT变换
    dct_blocks, original_shape = image_to_dct_blocks(img)
    
    # 准备嵌入位置
    if positions is None:
        positions = select_embedding_positions(block_size)
    
    # 计算嵌入容量
    capacity_bits, capacity_bytes = calculate_embedding_capacity(original_shape, len(positions), block_size)
    
    # 将消息转换为比特流
    bits = message_to_bits(message)
    
    # 检查消息是否太大
    if len(bits) > capacity_bits:
        print(f"错误: 消息太大，最大容量为{capacity_bytes}字节")
        return False
    
    print(f"消息大小: {len(message)}字节, 可用容量: {capacity_bytes}字节")
    
    # 执行嵌入
    bit_index = 0
    height, width = dct_blocks.shape
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size
    
    for block_row in range(num_blocks_h):
        for block_col in range(num_blocks_w):
            # 遍历当前块中的嵌入位置
            for (i, j) in positions:
                # 检查是否还有比特需要嵌入
                if bit_index >= len(bits):
                    break
                
                # 计算系数在整个数组中的位置
                coeff_i = block_row * block_size + i
                coeff_j = block_col * block_size + j
                
                # 获取当前DCT系数
                coeff_value = dct_blocks[coeff_i, coeff_j]
                
                # 修改LSB
                if coeff_value >= 0:
                    dct_blocks[coeff_i, coeff_j] = float(int(coeff_value) & ~1 | bits[bit_index])
                else:
                    # 对于负值，我们需要特别处理
                    abs_value = abs(int(coeff_value))
                    new_abs_value = abs_value & ~1 | bits[bit_index]
                    dct_blocks[coeff_i, coeff_j] = -float(new_abs_value)
                
                # 移动到下一个比特
                bit_index += 1
            
            # 检查是否嵌入完成
            if bit_index >= len(bits):
                break
        
        # 检查是否嵌入完成
        if bit_index >= len(bits):
            break
    
    # 使用IDCT重建图像
    stego_image = dct_blocks_to_image(dct_blocks, original_shape)
    
    # 保存隐写图像
    stego_pil = Image.fromarray(stego_image)
    stego_pil.save(output_path)
    
    print(f"消息已成功嵌入到 {output_path}")
    return True

def dct_lsb_extraction(stego_path, block_size=8, positions=None):
    """
    从DCT域中使用LSB方法提取消息
    
    参数:
    stego_path: 隐写图像路径
    block_size: DCT块大小
    positions: 用于嵌入的DCT系数位置，如果为None则使用默认位置
    
    返回:
    提取的消息
    """
    # 加载隐写图像
    img = Image.open(stego_path).convert('L')
    
    # 进行DCT变换
    dct_blocks, original_shape = image_to_dct_blocks(img)
    
    # 准备嵌入位置
    if positions is None:
        positions = select_embedding_positions(block_size)
    
    # 提取比特
    bits = []
    height, width = dct_blocks.shape
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size
    
    # 我们需要收集足够的比特直到找到结束标记
    # 假设消息不会超过容量的一半，以避免无限循环
    max_bits = (num_blocks_h * num_blocks_w * len(positions)) // 2
    
    for block_row in range(num_blocks_h):
        for block_col in range(num_blocks_w):
            # 遍历当前块中的嵌入位置
            for (i, j) in positions:
                # 计算系数在整个数组中的位置
                coeff_i = block_row * block_size + i
                coeff_j = block_col * block_size + j
                
                # 获取当前DCT系数
                coeff_value = dct_blocks[coeff_i, coeff_j]
                
                # 提取LSB
                bit = int(abs(coeff_value)) & 1
                bits.append(bit)
                
                # 检查是否达到最大比特数
                if len(bits) >= max_bits:
                    break
                
                # 每8个比特检查一次是否是结束标记
                if len(bits) % 8 == 0 and len(bits) >= 8:
                    if bits[-8:] == [1]*8:
                        break
            
            # 检查是否达到最大比特数或找到结束标记
            if len(bits) >= max_bits or (len(bits) % 8 == 0 and bits[-8:] == [1]*8):
                break
        
        # 检查是否达到最大比特数或找到结束标记
        if len(bits) >= max_bits or (len(bits) % 8 == 0 and bits[-8:] == [1]*8):
            break
    
    # 将比特流转换回消息
    message = bits_to_message(bits)
    
    return message
```

### 测试DCT域LSB隐写的效果
```
def test_dct_lsb_stego(original_image, stego_image, test_message="这是一条测试消息，用于验证DCT域LSB隐写的效果。"):
    """
    测试DCT域LSB隐写
    """
    # 执行嵌入
    success = dct_lsb_embedding(original_image, test_message, stego_image)
    
    if success:
        # 执行提取
        extracted_message = dct_lsb_extraction(stego_image)
        
        print(f"原始消息: {test_message}")
        print(f"提取消息: {extracted_message}")
        print(f"消息匹配: {test_message == extracted_message}")
        
        # 计算嵌入前后的图像差异
        original = np.array(Image.open(original_image).convert('L'))
        stego = np.array(Image.open(stego_image).convert('L'))
        
        mse = np.mean((original.astype(np.float64) - stego.astype(np.float64)) ** 2)
        psnr = 10 * np.log10(255**2 / mse)
        
        print(f"嵌入前后MSE: {mse:.4f}")
        print(f"嵌入前后PSNR: {psnr:.2f} dB")
        
        return extracted_message, mse, psnr
    else:
        print("隐写失败")
        return None, None, None

# 测试代码（需要替换为实际的图像路径）
# extracted, mse, psnr = test_dct_lsb_stego('lena.jpg', 'lena_dct_lsb_stego.png')
```

## DCT with quantization
基于量化的DCT隐写，它利用JPEG压缩中的量化步骤来嵌入信息。
```
def create_quantization_table(quality=75):
    """
    创建JPEG量化表
    
    参数:
    quality: JPEG质量因子 (1-100)
    
    返回:
    quantization_table: 8x8量化表
    """
    # 标准JPEG亮度量化表
    base_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    # 根据质量因子调整量化表
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    
    quantization_table = np.floor((base_table * scale + 50) / 100)
    quantization_table = np.clip(quantization_table, 1, 255).astype(np.uint8)
    
    return quantization_table

def quantize_dct_blocks(dct_blocks, quantization_table, block_size=8):
    """
    对DCT系数进行量化
    
    参数:
    dct_blocks: DCT系数
    quantization_table: 量化表
    block_size: 块大小
    
    返回:
    quantized_blocks: 量化后的DCT系数
    """
    height, width = dct_blocks.shape
    quantized_blocks = np.zeros((height, width), dtype=np.int32)
    
    # 对每个块进行量化
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # 提取块
            block = dct_blocks[i:i+block_size, j:j+block_size]
            
            # 量化
            quantized_block = np.round(block / quantization_table)
            
            # 存储量化后的块
            quantized_blocks[i:i+block_size, j:j+block_size] = quantized_block
    
    return quantized_blocks

def dequantize_dct_blocks(quantized_blocks, quantization_table, block_size=8):
    """
    对量化后的DCT系数进行反量化
    
    参数:
    quantized_blocks: 量化后的DCT系数
    quantization_table: 量化表
    block_size: 块大小
    
    返回:
    dequantized_blocks: 反量化后的DCT系数
    """
    height, width = quantized_blocks.shape
    dequantized_blocks = np.zeros((height, width), dtype=np.float64)
    
    # 对每个块进行反量化
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # 提取块
            block = quantized_blocks[i:i+block_size, j:j+block_size]
            
            # 反量化
            dequantized_block = block * quantization_table
            
            # 存储反量化后的块
            dequantized_blocks[i:i+block_size, j:j+block_size] = dequantized_block
    
    return dequantized_blocks

def quantization_based_steganography(image_path, message, output_path, quality=75, positions=None):
    """
    基于量化的DCT隐写
    
    参数:
    image_path: 原始图像路径
    message: 要嵌入的消息
    output_path: 输出图像路径
    quality: JPEG质量因子
    positions: 用于嵌入的DCT系数位置
    
    返回:
    是否成功嵌入
    """
    # 加载图像
    img = Image.open(image_path).convert('L')
    
    # 进行DCT变换
    dct_blocks, original_shape = image_to_dct_blocks(img)
    
    # 创建量化表
    quantization_table = create_quantization_table(quality)
    
    # 量化DCT系数
    quantized_blocks = quantize_dct_blocks(dct_blocks, quantization_table)
    
    # 准备嵌入位置
    if positions is None:
        positions = select_embedding_positions()
    
    # 计算嵌入容量
    capacity_bits, capacity_bytes = calculate_embedding_capacity(original_shape, len(positions))
    
    # 将消息转换为比特流
    bits = message_to_bits(message)
    
    # 检查消息是否太大
    if len(bits) > capacity_bits:
        print(f"错误: 消息太大，最大容量为{capacity_bytes}字节")
        return False
    
    print(f"消息大小: {len(message)}字节, 可用容量: {capacity_bytes}字节")
    
    # 执行嵌入
    bit_index = 0
    height, width = quantized_blocks.shape
    num_blocks_h = height // 8
    num_blocks_w = width // 8
    
    for block_row in range(num_blocks_h):
        for block_col in range(num_blocks_w):
            # 遍历当前块中的嵌入位置
            for (i, j) in positions:
                # 检查是否还有比特需要嵌入
                if bit_index >= len(bits):
                    break
                
                # 计算系数在整个数组中的位置
                coeff_i = block_row * 8 + i
                coeff_j = block_col * 8 + j
                
                # 获取当前量化后的DCT系数
                coeff_value = quantized_blocks[coeff_i, coeff_j]
                
                # 根据要嵌入的比特值调整量化系数
                # 如果系数不为0，我们可以通过奇偶性来嵌入信息
                if coeff_value != 0:
                    # 如果当前系数的奇偶性与要嵌入的比特不同，则调整
                    if (abs(coeff_value) % 2) != bits[bit_index]:
                        if coeff_value > 0:
                            quantized_blocks[coeff_i, coeff_j] += 1 if bits[bit_index] else -1
                        else:
                            quantized_blocks[coeff_i, coeff_j] -= 1 if bits[bit_index] else -1
                else:
                    # 如果系数为0，我们可以将其设置为±1来嵌入信息
                    quantized_blocks[coeff_i, coeff_j] = 1 if bits[bit_index] else -1
                
                # 移动到下一个比特
                bit_index += 1
            
            # 检查是否嵌入完成
            if bit_index >= len(bits):
                break
        
        # 检查是否嵌入完成
        if bit_index >= len(bits):
            break
    
    # 反量化
    dequantized_blocks = dequantize_dct_blocks(quantized_blocks, quantization_table)
    
    # 重建图像
    stego_image = dct_blocks_to_image(dequantized_blocks, original_shape)
    
    # 保存隐写图像
    stego_pil = Image.fromarray(stego_image)
    stego_pil.save(output_path)
    
    print(f"消息已成功嵌入到 {output_path}")
    return True

def quantization_based_extraction(stego_path, quality=75, positions=None):
    """
    从基于量化的DCT隐写图像中提取消息
    
    参数:
    stego_path: 隐写图像路径
    quality: JPEG质量因子
    positions: 用于嵌入的DCT系数位置
    
    返回:
    提取的消息
    """
    # 加载隐写图像
    img = Image.open(stego_path).convert('L')
    
    # 进行DCT变换
    dct_blocks, _ = image_to_dct_blocks(img)
    
    # 创建量化表
    quantization_table = create_quantization_table(quality)
    
    # 量化DCT系数
    quantized_blocks = quantize_dct_blocks(dct_blocks, quantization_table)
    
    # 准备嵌入位置
    if positions is None:
        positions = select_embedding_positions()
    
    # 提取比特
    bits = []
    height, width = quantized_blocks.shape
    num_blocks_h = height // 8
    num_blocks_w = width // 8
    
    # 我们需要收集足够的比特直到找到结束标记
    max_bits = (num_blocks_h * num_blocks_w * len(positions)) // 2
    
    for block_row in range(num_blocks_h):
        for block_col in range(num_blocks_w):
            # 遍历当前块中的嵌入位置
            for (i, j) in positions:
                # 计算系数在整个数组中的位置
                coeff_i = block_row * 8 + i
                coeff_j = block_col * 8 + j
                
                # 获取当前量化后的DCT系数
                coeff_value = quantized_blocks[coeff_i, coeff_j]
                
                # 提取比特：使用系数的奇偶性
                bit = 1 if abs(coeff_value) % 2 == 1 else 0
                bits.append(bit)
                
                # 检查是否达到最大比特数
                if len(bits) >= max_bits:
                    break
                
                # 每8个比特检查一次是否是结束标记
                if len(bits) % 8 == 0 and len(bits) >= 8:
                    if bits[-8:] == [1]*8:
                        break
            
            # 检查是否达到最大比特数或找到结束标记
            if len(bits) >= max_bits or (len(bits) % 8 == 0 and bits[-8:] == [1]*8):
                break
        
        # 检查是否达到最大比特数或找到结束标记
        if len(bits) >= max_bits or (len(bits) % 8 == 0 and bits[-8:] == [1]*8):
            break
    
    # 将比特流转换回消息
    message = bits_to_message(bits)
    
    return message
```

### 测试基于量化的DCT隐写

```
def test_quantization_stego(original_image, stego_image, quality=75, 
                          test_message="这是一条测试消息，用于验证基于量化的DCT隐写的效果。"):
    """
    测试基于量化的DCT隐写
    """
    # 执行嵌入
    success = quantization_based_steganography(original_image, test_message, stego_image, quality)
    
    if success:
        # 执行提取
        extracted_message = quantization_based_extraction(stego_image, quality)
        
        print(f"原始消息: {test_message}")
        print(f"提取消息: {extracted_message}")
        print(f"消息匹配: {test_message == extracted_message}")
        
        # 计算嵌入前后的图像差异
        original = np.array(Image.open(original_image).convert('L'))
        stego = np.array(Image.open(stego_image).convert('L'))
        
        mse = np.mean((original.astype(np.float64) - stego.astype(np.float64)) ** 2)
        psnr = 10 * np.log10(255**2 / mse)
        
        print(f"嵌入前后MSE: {mse:.4f}")
        print(f"嵌入前后PSNR: {psnr:.2f} dB")
        
        return extracted_message, mse, psnr
    else:
        print("隐写失败")
        return None, None, None
```

## 基于DWT的小波隐写
离散小波变换(DWT)在隐写术中具有独特的优势，它能够在不同频率分量上进行数据嵌入，从而提高隐写的隐蔽性

```
import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt

def image_to_dwt(image, wavelet='haar', level=1):
    """
    对图像进行离散小波变换
    
    参数:
    image: 输入图像（PIL Image或numpy数组）
    wavelet: 小波基函数，默认为'haar'
    level: 分解级别
    
    返回:
    coeffs: DWT系数
    original_shape: 原始图像形状
    """
    # 如果输入是PIL Image，转换为numpy数组
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    original_shape = image.shape
    
    # 如果是彩色图像，转换为灰度
    if len(original_shape) > 2:
        image = np.mean(image, axis=2).astype(np.float64)
    else:
        image = image.astype(np.float64)
    
    # 执行离散小波变换
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    return coeffs, original_shape

def dwt_to_image(coeffs, original_shape, wavelet='haar'):
    """
    从小波系数重建图像
    
    参数:
    coeffs: DWT系数
    original_shape: 原始图像形状
    wavelet: 小波基函数，默认为'haar'
    
    返回:
    image: 重建后的图像
    """
    # 从小波系数重建图像
    reconstructed = pywt.waverec2(coeffs, wavelet)
    
    # 裁剪到原始尺寸（避免由于小波变换可能导致的尺寸变化）
    reconstructed = reconstructed[:original_shape[0], :original_shape[1]]
    
    # 确保像素值在有效范围内
    reconstructed = np.clip(reconstructed, 0, 255)
    
    return reconstructed.astype(np.uint8)

def visualize_dwt_coeffs(coeffs):
    """
    可视化DWT系数
    
    参数:
    coeffs: DWT系数
    """
    # 解包DWT系数
    cA = coeffs[0]  # 近似系数
    details = coeffs[1:]  # 细节系数
    
    # 绘制近似系数
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(cA, cmap='gray')
    plt.title('近似系数 (Approximation)')
    plt.axis('off')
    
    # 绘制细节系数（仅显示第一级）
    if details:
        cH, cV, cD = details[0]  # 水平、垂直、对角线细节系数
        
        plt.subplot(2, 2, 2)
        plt.imshow(cH, cmap='gray')
        plt.title('水平细节系数 (Horizontal)')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(cV, cmap='gray')
        plt.title('垂直细节系数 (Vertical)')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(cD, cmap='gray')
        plt.title('对角线细节系数 (Diagonal)')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 测试代码（需要替换为实际的图像路径）
# img = Image.open('lena.jpg')
# coeffs, shape = image_to_dwt(img, wavelet='haar', level=1)
# visualize_dwt_coeffs(coeffs)
# reconstructed = dwt_to_image(coeffs, shape)
# Image.fromarray(reconstructed).save('reconstructed.jpg')
```

### 小波域嵌入策略
在小波域中进行隐写需要考虑以下几个关键点：

- 嵌入位置选择：通常选择中频分量进行嵌入，这些分量对人类视觉不敏感但又足够稳定。
- 嵌入强度控制：避免过度修改导致隐写图像明显失真。
- 嵌入容量计算：根据小波分解级别和选择的嵌入区域计算可用容量。

常用的小波域隐写策略

```
小波域隐写位置优先级：

优先级1: 低频近似系数的非零区域 (对视觉影响小且稳定)
优先级2: 水平和垂直细节系数 (对纹理区域影响小)
优先级3: 对角线细节系数 (通常较少使用)
优先级4: 高频区域 (仅适用于需要大容量的情况)
```

### 小波域嵌入
```
def dwt_lsb_steganography(image_path, message, output_path, wavelet='haar', level=1, 
                         embedding_strength=0.5):
    """
    基于DWT的LSB隐写
    
    参数:
    image_path: 原始图像路径
    message: 要嵌入的消息
    output_path: 输出图像路径
    wavelet: 小波基函数
    level: 分解级别
    embedding_strength: 嵌入强度 (0-1)
    
    返回:
    是否成功嵌入
    """
    # 加载图像
    img = Image.open(image_path)
    
    # 进行DWT变换
    coeffs, original_shape = image_to_dwt(img, wavelet, level)
    
    # 将消息转换为比特流
    bits = message_to_bits(message)
    
    # 计算嵌入容量
    cA = coeffs[0]  # 近似系数
    capacity_bits = int(cA.size * 0.8 * embedding_strength)  # 使用近似系数的80%
    capacity_bytes = capacity_bits // 8
    
    # 检查消息是否太大
    if len(bits) > capacity_bits:
        print(f"错误: 消息太大，最大容量为{capacity_bytes}字节")
        return False
    
    print(f"消息大小: {len(message)}字节, 可用容量: {capacity_bytes}字节")
    
    # 执行嵌入（在近似系数上）
    cA_flat = cA.flatten()
    bit_index = 0
    
    # 找到非零系数进行嵌入
    for i in range(len(cA_flat)):
        if bit_index >= len(bits):
            break
        
        # 只在非零系数上嵌入
        if abs(cA_flat[i]) > 0.1:  # 使用一个小阈值来选择明显非零的系数
            # 转换为整数进行LSB操作
            coeff_val = int(cA_flat[i])
            
            # 修改LSB位
            if (coeff_val % 2) != bits[bit_index]:
                coeff_val += 1 if bits[bit_index] else -1
                
            # 保存回系数数组
            cA_flat[i] = coeff_val
            bit_index += 1
    
    # 重新构建近似系数
    coeffs[0] = cA_flat.reshape(cA.shape)
    
    # 如果还有比特未嵌入，尝试在细节系数中嵌入
    if bit_index < len(bits):
        for detail in coeffs[1:]:
            if bit_index >= len(bits):
                break
            
            # 处理每个子带
            for band in detail:
                band_flat = band.flatten()
                
                for i in range(len(band_flat)):
                    if bit_index >= len(bits):
                        break
                    
                    # 只在非零系数上嵌入
                    if abs(band_flat[i]) > 0.1:
                        # 转换为整数进行LSB操作
                        coeff_val = int(band_flat[i])
                        
                        # 修改LSB位
                        if (coeff_val % 2) != bits[bit_index]:
                            coeff_val += 1 if bits[bit_index] else -1
                            
                        # 保存回系数数组
                        band_flat[i] = coeff_val
                        bit_index += 1
    
    # 重建图像
    stego_image = dwt_to_image(coeffs, original_shape, wavelet)
    
    # 保存隐写图像
    if len(original_shape) > 2 and original_shape[2] == 3:
        # 如果原始图像是彩色的，转换回RGB
        stego_pil = Image.fromarray(stego_image)
        stego_pil = stego_pil.convert('RGB')
    else:
        stego_pil = Image.fromarray(stego_image)
    
    stego_pil.save(output_path)
    
    print(f"消息已成功嵌入到 {output_path}")
    return True

def dwt_lsb_extraction(stego_path, wavelet='haar', level=1, message_length=None):
    """
    从基于DWT的LSB隐写图像中提取消息
    
    参数:
    stego_path: 隐写图像路径
    wavelet: 小波基函数
    level: 分解级别
    message_length: 消息长度（字节），如果为None则尝试自动检测
    
    返回:
    提取的消息
    """
    # 加载隐写图像
    img = Image.open(stego_path)
    
    # 进行DWT变换
    coeffs, _ = image_to_dwt(img, wavelet, level)
    
    # 提取比特
    bits = []
    
    # 首先从近似系数中提取
    cA = coeffs[0]
    cA_flat = cA.flatten()
    
    # 计算需要提取的比特数
    if message_length is not None:
        target_bits = message_length * 8 + 8  # 加上结束标记
    else:
        target_bits = int(cA.size * 0.8 * 2)  # 设置一个较大的默认值
    
    # 从非零系数中提取
    for coeff in cA_flat:
        if len(bits) >= target_bits:
            break
            
        if abs(coeff) > 0.1:
            # 提取LSB位
            bit = 1 if int(abs(coeff)) % 2 == 1 else 0
            bits.append(bit)
    
    # 如果还需要更多比特，从细节系数中提取
    if len(bits) < target_bits:
        for detail in coeffs[1:]:
            if len(bits) >= target_bits:
                break
                
            for band in detail:
                band_flat = band.flatten()
                
                for coeff in band_flat:
                    if len(bits) >= target_bits:
                        break
                        
                    if abs(coeff) > 0.1:
                        # 提取LSB位
                        bit = 1 if int(abs(coeff)) % 2 == 1 else 0
                        bits.append(bit)
    
    # 将比特流转换回消息
    message = bits_to_message(bits)
    
    return message
```

### 测试基于DWT的LSB隐写
```
def test_dwt_lsb_stego(original_image, stego_image, wavelet='haar', level=1, 
                     test_message="这是一条测试消息，用于验证基于DWT的LSB隐写的效果。"):
    """
    测试基于DWT的LSB隐写
    """
    # 执行嵌入
    success = dwt_lsb_steganography(original_image, test_message, stego_image, 
                                   wavelet, level)
    
    if success:
        # 执行提取
        extracted_message = dwt_lsb_extraction(stego_image, wavelet, level)
        
        print(f"原始消息: {test_message}")
        print(f"提取消息: {extracted_message}")
        print(f"消息匹配: {test_message == extracted_message}")
        
        # 计算嵌入前后的图像差异
        original = np.array(Image.open(original_image))
        stego = np.array(Image.open(stego_image))
        
        mse = np.mean((original.astype(np.float64) - stego.astype(np.float64)) ** 2)
        psnr = 10 * np.log10(255**2 / mse)
        
        print(f"嵌入前后MSE: {mse:.4f}")
        print(f"嵌入前后PSNR: {psnr:.2f} dB")
        
        return extracted_message, mse, psnr
    else:
        print("隐写失败")
        return None, None, None

# 测试代码（需要替换为实际的图像路径）
# extracted, mse, psnr = test_dwt_lsb_stego('lena.jpg', 'lena_dwt_stego.png', wavelet='db4', level=2)
```

## 基于DWT的自适应隐写
自适应隐写是一种更先进的方法，它会根据图像内容的特性自动调整嵌入强度，在纹理复杂的区域嵌入更多信息，在平滑区域嵌入更少信息。

```
def calculate_texture_energy(coeffs, window_size=3):
    """
    计算图像的纹理能量
    
    参数:
    coeffs: DWT系数
    window_size: 计算纹理能量的窗口大小
    
    返回:
    texture_map: 纹理能量图
    """
    cA = coeffs[0]  # 近似系数
    
    # 创建纹理能量图
    texture_map = np.zeros_like(cA)
    height, width = cA.shape
    
    # 使用Sobel算子计算梯度
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # 计算水平和垂直梯度
    grad_x = np.zeros_like(cA)
    grad_y = np.zeros_like(cA)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            window = cA[i-1:i+2, j-1:j+2]
            grad_x[i, j] = np.sum(window * sobel_x)
            grad_y[i, j] = np.sum(window * sobel_y)
    
    # 计算梯度幅值作为纹理能量
    texture_map = np.sqrt(grad_x**2 + grad_y**2)
    
    # 归一化
    max_energy = np.max(texture_map)
    if max_energy > 0:
        texture_map = texture_map / max_energy
    
    return texture_map

def adaptive_dwt_steganography(image_path, message, output_path, wavelet='haar', level=1):
    """
    基于DWT的自适应隐写
    
    参数:
    image_path: 原始图像路径
    message: 要嵌入的消息
    output_path: 输出图像路径
    wavelet: 小波基函数
    level: 分解级别
    
    返回:
    是否成功嵌入
    """
    # 加载图像
    img = Image.open(image_path)
    
    # 进行DWT变换
    coeffs, original_shape = image_to_dwt(img, wavelet, level)
    
    # 计算纹理能量
    texture_map = calculate_texture_energy(coeffs)
    
    # 将消息转换为比特流
    bits = message_to_bits(message)
    
    # 计算嵌入容量（基于纹理能量）
    cA = coeffs[0]  # 近似系数
    
    # 纹理能量阈值，用于确定哪些区域可以嵌入信息
    threshold = 0.1
    
    # 计算可用的嵌入位置数量
    available_positions = np.sum(texture_map > threshold)
    capacity_bits = available_positions  # 每个可用位置嵌入1比特
    capacity_bytes = capacity_bits // 8
    
    # 检查消息是否太大
    if len(bits) > capacity_bits:
        print(f"错误: 消息太大，最大容量为{capacity_bytes}字节")
        return False
    
    print(f"消息大小: {len(message)}字节, 可用容量: {capacity_bytes}字节")
    
    # 执行自适应嵌入
    cA_flat = cA.flatten()
    texture_flat = texture_map.flatten()
    bit_index = 0
    
    # 只在纹理丰富的区域嵌入
    for i in range(len(cA_flat)):
        if bit_index >= len(bits):
            break
        
        # 只在纹理丰富的区域嵌入
        if texture_flat[i] > threshold and abs(cA_flat[i]) > 0.1:
            # 转换为整数进行操作
            coeff_val = int(cA_flat[i])
            
            # 根据纹理能量计算嵌入强度
            # 纹理能量越高，嵌入强度越大
            embedding_factor = 1 + texture_flat[i] * 0.5  # 1到1.5的范围
            
            # 修改系数值
            if bits[bit_index]:
                if coeff_val % 2 == 0:
                    coeff_val += int(embedding_factor)
            else:
                if coeff_val % 2 == 1:
                    coeff_val -= int(embedding_factor)
            
            # 保存回系数数组
            cA_flat[i] = coeff_val
            bit_index += 1
    
    # 重新构建近似系数
    coeffs[0] = cA_flat.reshape(cA.shape)
    
    # 重建图像
    stego_image = dwt_to_image(coeffs, original_shape, wavelet)
    
    # 保存隐写图像
    if len(original_shape) > 2 and original_shape[2] == 3:
        # 如果原始图像是彩色的，转换回RGB
        stego_pil = Image.fromarray(stego_image)
        stego_pil = stego_pil.convert('RGB')
    else:
        stego_pil = Image.fromarray(stego_image)
    
    stego_pil.save(output_path)
    
    print(f"消息已成功嵌入到 {output_path}")
    return True
```

# 频域隐写安全性分析和检测技术
频域隐写虽然比空间域隐写更加隐蔽，但仍然可能被检测出来。

频域隐写技术虽然在隐蔽性方面有所提升，但仍然存在以下安全隐患：

1. **统计分布异常**：嵌入数据会改变频域系数的统计分布，尤其是当嵌入量较大时。
2. **量化表不匹配**：在JPEG相关的隐写中，自定义量化表与标准量化表的差异可能被检测。
3. **DCT系数相关性变化**：相邻DCT系数之间的相关性在隐写后可能发生变化。
4. **小波系数能量分布异常**：小波系数的能量分布在隐写后可能偏离正常模式。
5. **嵌入模式固定**：固定的嵌入位置和强度可能形成可识别的模式。

## DCT域隐写检测技术

### 直方图分析检测

```
def dct_histogram_analysis(original_image, stego_image, block_size=8):
    """
    通过分析DCT系数直方图来检测隐写
    
    参数:
    original_image: 原始图像路径
    stego_image: 隐写图像路径
    block_size: 块大小
    
    返回:
    similarity_score: 直方图相似度分数 (0-1)
    """
    # 加载图像
    img1 = Image.open(original_image).convert('L')
    img2 = Image.open(stego_image).convert('L')
    
    # 转换为numpy数组
    arr1 = np.array(img1).astype(np.float64)
    arr2 = np.array(img2).astype(np.float64)
    
    # 进行DCT变换
    dct_blocks1, _ = image_to_dct_blocks(Image.fromarray(arr1.astype(np.uint8)))
    dct_blocks2, _ = image_to_dct_blocks(Image.fromarray(arr2.astype(np.uint8)))
    
    # 展平DCT系数
    dct_flat1 = dct_blocks1.flatten()
    dct_flat2 = dct_blocks2.flatten()
    
    # 过滤掉DC系数，只保留AC系数
    height, width = dct_blocks1.shape
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size
    
    ac_coeffs1 = []
    ac_coeffs2 = []
    
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block1 = dct_blocks1[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            block2 = dct_blocks2[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            
            # 跳过DC系数 (0,0)，只收集AC系数
            for x in range(block_size):
                for y in range(block_size):
                    if x != 0 or y != 0:
                        ac_coeffs1.append(block1[x, y])
                        ac_coeffs2.append(block2[x, y])
    
    # 创建直方图
    hist_range = (-200, 200)
    num_bins = 100
    hist1, _ = np.histogram(ac_coeffs1, bins=num_bins, range=hist_range, density=True)
    hist2, _ = np.histogram(ac_coeffs2, bins=num_bins, range=hist_range, density=True)
    
    # 计算直方图相似度 (使用相关系数)
    similarity_score = np.corrcoef(hist1, hist2)[0, 1]
    
    print(f"直方图相关系数: {similarity_score:.4f}")
    print(f"可能包含隐写: {similarity_score < 0.98}")
    
    return similarity_score
```

### 卡方检测 (Chi-Square Attack)
卡方检测是一种统计检测方法，用于检测JPEG图像中是否存在基于量化的隐写。它通过分析量化后DCT系数对的分布特性来判断是否存在隐写。

```
def chi_square_attack(image_path, block_size=8, coefficient_positions=None):
    """
    使用卡方检测方法检测JPEG隐写
    
    参数:
    image_path: 待检测图像路径
    block_size: 块大小
    coefficient_positions: 要分析的DCT系数位置列表
    
    返回:
    chi_square_scores: 卡方检测分数
    p_values: 对应的p值
    """
    import scipy.stats as stats
    
    # 加载图像并转换为灰度
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    
    # 进行DCT变换
    dct_blocks, _ = image_to_dct_blocks(img)
    
    # 如果未指定系数位置，使用中频系数
    if coefficient_positions is None:
        coefficient_positions = [(1,1), (1,2), (2,1), (2,2), (1,3), (3,1)]
    
    height, width = dct_blocks.shape
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size
    
    chi_square_scores = []
    p_values = []
    
    # 对每个指定位置的系数进行卡方检测
    for (x, y) in coefficient_positions:
        # 收集所有块中该位置的系数
        coeffs = []
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                coeff = dct_blocks[i*block_size + x, j*block_size + y]
                # 只考虑非零系数
                if coeff != 0:
                    coeffs.append(coeff)
        
        # 创建系数对 (k, k+1) 和 (k, k-1) 的计数
        pair_counts_plus = {}
        pair_counts_minus = {}
        
        # 初始化计数
        for k in range(-20, 21):
            pair_counts_plus[k] = 0
            pair_counts_minus[k] = 0
        
        # 统计相邻系数对的频率
        for coeff in coeffs:
            k = int(round(coeff))
            # 统计 (k, k+1) 对
            if (k+1) in pair_counts_plus:
                pair_counts_plus[k] += 1
            # 统计 (k, k-1) 对
            if (k-1) in pair_counts_minus:
                pair_counts_minus[k] += 1
        
        # 计算卡方统计量
        chi_square = 0
        n = len(coeffs)
        
        for k in range(-19, 20):
            observed_plus = pair_counts_plus[k]
            observed_minus = pair_counts_minus[k]
            
            # 期望频率应该相等（如果没有隐写）
            expected = (observed_plus + observed_minus) / 2
            
            if expected > 0:
                chi_square += ((observed_plus - expected)**2 + (observed_minus - expected)**2) / expected
        
        # 自由度为系数范围的大小
        degrees_of_freedom = 38  # 从-19到19共39个点，但因为和为定值，所以自由度减1
        
        # 计算p值
        p_value = 1 - stats.chi2.cdf(chi_square, degrees_of_freedom)
        
        chi_square_scores.append(chi_square)
        p_values.append(p_value)
        
        print(f"位置 ({x},{y}) - 卡方分数: {chi_square:.4f}, p值: {p_value:.6f}")
        print(f"\t 可能包含隐写: {p_value < 0.05}")
    
    return chi_square_scores, p_values
```

### DCT系数相关性检测

```
def dct_coefficient_correlation(image_path1, image_path2, block_size=8):
    """
    通过分析DCT系数的相关性来检测隐写
    
    参数:
    image_path1: 第一张图像路径
    image_path2: 第二张图像路径
    block_size: 块大小
    
    返回:
    correlation_matrix: 相关性矩阵
    """
    # 加载图像
    img1 = Image.open(image_path1).convert('L')
    img2 = Image.open(image_path2).convert('L')
    
    # 进行DCT变换
    dct_blocks1, _ = image_to_dct_blocks(img1)
    dct_blocks2, _ = image_to_dct_blocks(img2)
    
    height, width = dct_blocks1.shape
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size
    
    # 收集每个位置的DCT系数
    coefficients1 = {}
    coefficients2 = {}
    
    # 初始化字典
    for i in range(block_size):
        for j in range(block_size):
            coefficients1[(i,j)] = []
            coefficients2[(i,j)] = []
    
    # 收集系数
    for block_i in range(num_blocks_h):
        for block_j in range(num_blocks_w):
            for i in range(block_size):
                for j in range(block_size):
                    coeff1 = dct_blocks1[block_i*block_size + i, block_j*block_size + j]
                    coeff2 = dct_blocks2[block_i*block_size + i, block_j*block_size + j]
                    coefficients1[(i,j)].append(coeff1)
                    coefficients2[(i,j)].append(coeff2)
    
    # 计算相邻系数之间的相关性变化
    correlation_changes = []
    positions = []
    
    # 检查水平相邻系数
    for i in range(block_size):
        for j in range(block_size - 1):
            pos1 = (i, j)
            pos2 = (i, j+1)
            
            # 计算第一张图像的相关性
            corr1 = np.corrcoef(coefficients1[pos1], coefficients1[pos2])[0, 1]
            
            # 计算第二张图像的相关性
            corr2 = np.corrcoef(coefficients2[pos1], coefficients2[pos2])[0, 1]
            
            # 计算相关性变化
            change = abs(corr1 - corr2)
            correlation_changes.append(change)
            positions.append(f"{pos1}->{pos2}")
    
    # 检查垂直相邻系数
    for i in range(block_size - 1):
        for j in range(block_size):
            pos1 = (i, j)
            pos2 = (i+1, j)
            
            # 计算第一张图像的相关性
            corr1 = np.corrcoef(coefficients1[pos1], coefficients1[pos2])[0, 1]
            
            # 计算第二张图像的相关性
            corr2 = np.corrcoef(coefficients2[pos1], coefficients2[pos2])[0, 1]
            
            # 计算相关性变化
            change = abs(corr1 - corr2)
            correlation_changes.append(change)
            positions.append(f"{pos1}->{pos2}")
    
    # 找到相关性变化最大的位置
    max_change_idx = np.argmax(correlation_changes)
    max_change_pos = positions[max_change_idx]
    max_change_value = correlation_changes[max_change_idx]
    
    # 计算平均相关性变化
    avg_change = np.mean(correlation_changes)
    
    print(f"最大相关性变化: {max_change_value:.4f} 在 {max_change_pos}")
    print(f"平均相关性变化: {avg_change:.4f}")
    print(f"可能包含隐写: {avg_change > 0.05}")
    
    return avg_change
```

## DWT域隐写检测技术

### 小波系数统计检测

```
def dwt_statistical_analysis(original_image, stego_image, wavelet='haar', level=1):
    """
    通过分析小波系数的统计特性来检测隐写
    
    参数:
    original_image: 原始图像路径
    stego_image: 隐写图像路径
    wavelet: 小波基函数
    level: 分解级别
    
    返回:
    feature_difference: 特征差异值
    """
    # 加载图像
    img1 = Image.open(original_image)
    img2 = Image.open(stego_image)
    
    # 进行DWT变换
    coeffs1, _ = image_to_dwt(img1, wavelet, level)
    coeffs2, _ = image_to_dwt(img2, wavelet, level)
    
    # 提取近似系数
    cA1 = coeffs1[0]
    cA2 = coeffs2[0]
    
    # 计算统计特征
    features1 = {
        'mean': np.mean(cA1),
        'std': np.std(cA1),
        'skew': stats.skew(cA1.flatten()),
        'kurtosis': stats.kurtosis(cA1.flatten())
    }
    
    features2 = {
        'mean': np.mean(cA2),
        'std': np.std(cA2),
        'skew': stats.skew(cA2.flatten()),
        'kurtosis': stats.kurtosis(cA2.flatten())
    }
    
    # 计算特征差异
    mean_diff = abs(features1['mean'] - features2['mean'])
    std_diff = abs(features1['std'] - features2['std'])
    skew_diff = abs(features1['skew'] - features2['skew'])
    kurtosis_diff = abs(features1['kurtosis'] - features2['kurtosis'])
    
    # 计算总差异
    feature_difference = mean_diff + std_diff + skew_diff + kurtosis_diff
    
    print(f"统计特征差异:")
    print(f"均值差异: {mean_diff:.4f}")
    print(f"标准差差异: {std_diff:.4f}")
    print(f"偏度差异: {skew_diff:.4f}")
    print(f"峰度差异: {kurtosis_diff:.4f}")
    print(f"总差异: {feature_difference:.4f}")
    print(f"可能包含隐写: {feature_difference > 5.0}")
    
    return feature_difference
```

### 小波系数熵分析
```
def calculate_entropy(data, bins=256):
    """
    计算数据的熵
    
    参数:
    data: 输入数据
    bins: 直方图 bins 数量
    
    返回:
    entropy: 熵值
    """
    # 计算直方图
    hist, _ = np.histogram(data, bins=bins, density=True)
    
    # 过滤掉零概率
    hist = hist[hist > 0]
    
    # 计算熵
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy

def dwt_entropy_analysis(original_image, stego_image, wavelet='haar', level=1):
    """
    通过分析小波系数的熵来检测隐写
    
    参数:
    original_image: 原始图像路径
    stego_image: 隐写图像路径
    wavelet: 小波基函数
    level: 分解级别
    
    返回:
    entropy_diff: 熵差异值
    """
    # 加载图像
    img1 = Image.open(original_image)
    img2 = Image.open(stego_image)
    
    # 进行DWT变换
    coeffs1, _ = image_to_dwt(img1, wavelet, level)
    coeffs2, _ = image_to_dwt(img2, wavelet, level)
    
    # 提取所有系数
    all_coeffs1 = []
    all_coeffs2 = []
    
    # 添加近似系数
    all_coeffs1.extend(coeffs1[0].flatten())
    all_coeffs2.extend(coeffs2[0].flatten())
    
    # 添加细节系数
    for details in coeffs1[1:]:
        for band in details:
            all_coeffs1.extend(band.flatten())
    
    for details in coeffs2[1:]:
        for band in details:
            all_coeffs2.extend(band.flatten())
    
    # 计算熵
    entropy1 = calculate_entropy(all_coeffs1)
    entropy2 = calculate_entropy(all_coeffs2)
    
    # 计算熵差异
    entropy_diff = abs(entropy1 - entropy2)
    
    print(f"原始图像小波系数熵: {entropy1:.4f}")
    print(f"检测图像小波系数熵: {entropy2:.4f}")
    print(f"熵差异: {entropy_diff:.4f}")
    print(f"可能包含隐写: {entropy_diff > 0.1}")
    
    return entropy_diff
```
## 综合检测方法
将多种检测方法结合使用可以提高检测的准确性。下面是一个综合检测系统的实现：

```
def steganalysis_detector(original_image, stego_image, wavelet='haar', level=1):
    """
    综合检测系统，结合多种方法检测隐写
    
    参数:
    original_image: 原始图像路径
    stego_image: 隐写图像路径
    wavelet: 小波基函数
    level: 分解级别
    
    返回:
    detection_result: 检测结果字典
    is_suspicious: 是否包含隐写
    """
    import scipy.stats as stats
    
    detection_result = {}
    
    # 1. DCT直方图分析
    print("\n1. DCT直方图分析")
    hist_similarity = dct_histogram_analysis(original_image, stego_image)
    detection_result['histogram_similarity'] = hist_similarity
    
    # 2. 卡方检测
    print("\n2. 卡方检测")
    chi_scores, p_values = chi_square_attack(stego_image)
    detection_result['chi_square_scores'] = chi_scores
    detection_result['p_values'] = p_values
    
    # 3. DCT系数相关性分析
    print("\n3. DCT系数相关性分析")
    corr_change = dct_coefficient_correlation(original_image, stego_image)
    detection_result['correlation_change'] = corr_change
    
    # 4. DWT统计分析
    print("\n4. DWT统计分析")
    stat_diff = dwt_statistical_analysis(original_image, stego_image, wavelet, level)
    detection_result['statistical_difference'] = stat_diff
    
    # 5. DWT熵分析
    print("\n5. DWT熵分析")
    entropy_diff = dwt_entropy_analysis(original_image, stego_image, wavelet, level)
    detection_result['entropy_difference'] = entropy_diff
    
    # 计算总体可疑度得分（0-1）
    suspicious_score = 0
    
    # 基于直方图相似度的得分
    if hist_similarity < 0.98:
        suspicious_score += 0.2
    
    # 基于卡方检测的得分
    if any(p < 0.05 for p in p_values):
        suspicious_score += 0.2
    
    # 基于相关性变化的得分
    if corr_change > 0.05:
        suspicious_score += 0.2
    
    # 基于统计差异的得分
    if stat_diff > 5.0:
        suspicious_score += 0.2
    
    # 基于熵差异的得分
    if entropy_diff > 0.1:
        suspicious_score += 0.2
    
    detection_result['suspicious_score'] = suspicious_score
    
    # 判断是否包含隐写
    is_suspicious = suspicious_score > 0.5
    
    print(f"\n=== 检测结果摘要 ===")
    print(f"可疑度得分: {suspicious_score:.2f}/1.00")
    print(f"综合判断: {'可能包含隐写内容' if is_suspicious else '未检测到隐写内容'}")
    
    return detection_result, is_suspicious

# 测试代码（需要替换为实际的图像路径）
# result, is_suspicious = steganalysis_detector('lena.jpg', 'lena_stego.png')
```

# 基于FFT的频域隐写技术
FFT（快速傅里叶变换）可将图像从空间域映射到频率域。与DCT不同，FFT提供全局频谱表示，适合在中频带嵌入信息以兼顾隐蔽性与鲁棒性。

## FFT基础与频域带选择
- 频谱分解：F = FFT2(I)，得到复数频谱，包含幅度谱与相位谱。
- 频带选择：常选用环形中频带（避开低频直流与高频噪声区）。
- 嵌入载体：相位域对视觉更不敏感，幅度域可通过量化隐写实现较好鲁棒性。

```
import numpy as np
from PIL import Image

def fft2_image(image: Image.Image):
    arr = np.array(image.convert('L'), dtype=np.float64)
    F = np.fft.fftshift(np.fft.fft2(arr))
    magnitude = np.abs(F)
    phase = np.angle(F)
    return F, magnitude, phase, arr.shape

def ifft2_image(F, shape):
    img_rec = np.fft.ifft2(np.fft.ifftshift(F))
    img_rec = np.real(img_rec)
    img_rec = np.clip(img_rec, 0, 255).astype(np.uint8)
    return Image.fromarray(img_rec[:shape[0], :shape[1]])

def ring_mask(shape, r_inner=20, r_outer=60):
    h, w = shape
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y-cy)**2 + (X-cx)**2)
    mask = (dist >= r_inner) & (dist <= r_outer)
    return mask
```

## 基于相位的隐藏实现（Phase Embedding）
相位域对微小改动不敏感。我们可对选定频带的相位进行奇偶量化，将比特编码为相位的量化索引。

```
def fft_phase_embed(image_path, message, output_path, r_inner=20, r_outer=60, delta=np.pi/8):
    img = Image.open(image_path)
    F, mag, phase, shape = fft2_image(img)
    mask = ring_mask(shape, r_inner, r_outer)

    bits = message_to_bits(message)
    bit_idx = 0

    # 在掩模区域的相位进行量化嵌入
    for y in range(shape[0]):
        for x in range(shape[1]):
            if not mask[y, x]:
                continue
            if bit_idx >= len(bits):
                break
            # 量化相位到步长 delta 的最近值，并设置奇偶索引为比特
            q = np.round(phase[y, x] / delta)
            # 将量化索引的奇偶性调整为当前比特
            if (q % 2) != bits[bit_idx]:
                q += 1 if bits[bit_idx] else -1
            phase[y, x] = q * delta
            bit_idx += 1
        if bit_idx >= len(bits):
            break

    # 组合修改后的频谱
    F_new = mag * np.exp(1j * phase)
    out = ifft2_image(F_new, shape)
    out.save(output_path)
    return True

def fft_phase_extract(stego_path, bit_count, r_inner=20, r_outer=60, delta=np.pi/8):
    img = Image.open(stego_path)
    F, mag, phase, shape = fft2_image(img)
    mask = ring_mask(shape, r_inner, r_outer)
    bits = []
    for y in range(shape[0]):
        for x in range(shape[1]):
            if not mask[y, x]:
                continue
            if len(bits) >= bit_count:
                break
            q = int(np.round(phase[y, x] / delta))
            bits.append(int(q % 2))
        if len(bits) >= bit_count:
            break
    return bits_to_message(bits)
```

## 幅度域量化索引调制（QIM）隐写
在幅度谱上进行量化索引调制，将比特映射到不同量化格栅。相较相位嵌入，QIM在压缩与噪声环境下更具鲁棒性。

```
def fft_magnitude_qim_embed(image_path, message, output_path, r_inner=20, r_outer=60, step=2.0):
    img = Image.open(image_path)
    F, mag, phase, shape = fft2_image(img)
    mask = ring_mask(shape, r_inner, r_outer)
    bits = message_to_bits(message)
    bit_idx = 0

    mag_new = mag.copy()
    for y in range(shape[0]):
        for x in range(shape[1]):
            if not mask[y, x]:
                continue
            if bit_idx >= len(bits):
                break
            b = bits[bit_idx]
            # 两套格栅：偶数格栅编码0，奇数格栅编码1
            q = np.floor(mag_new[y, x] / step)
            if (q % 2) != b:
                # 调整到最近满足比特的格栅
                mag_new[y, x] = (q + (1 if b else 0)) * step
            else:
                mag_new[y, x] = q * step
            bit_idx += 1
        if bit_idx >= len(bits):
            break

    F_new = mag_new * np.exp(1j * phase)
    out = ifft2_image(F_new, shape)
    out.save(output_path)
    return True

def fft_magnitude_qim_extract(stego_path, bit_count, r_inner=20, r_outer=60, step=2.0):
    img = Image.open(stego_path)
    F, mag, phase, shape = fft2_image(img)
    mask = ring_mask(shape, r_inner, r_outer)
    bits = []
    for y in range(shape[0]):
        for x in range(shape[1]):
            if not mask[y, x]:
                continue
            if len(bits) >= bit_count:
                break
            q = int(np.floor(mag[y, x] / step))
            bits.append(int(q % 2))
        if len(bits) >= bit_count:
            break
    return bits_to_message(bits)
```

# 混合频域隐写（DCT+DWT）

将DCT的块状能量压缩特性与DWT的多尺度特性结合，可实现容量、隐蔽性与鲁棒性的折中。

- 流程：图像 → DWT分解 → 在中尺度近似系数上执行块状DCT → 在选定中频DCT系数执行量化或LSB隐写 → 逆DCT → 逆DWT。
- 优点：在纹理区域获得更高容量，在平滑区域保持较低可感知性。

```
def hybrid_dct_dwt_embed(image_path, message, output_path, wavelet='haar', level=1, positions=None, quality=75):
    img = Image.open(image_path).convert('L')
    coeffs, shape = image_to_dwt(img, wavelet, level)
    cA = coeffs[0]

    # 在近似系数上做块状DCT
    cA_img = Image.fromarray(np.clip(cA, 0, 255).astype(np.uint8))
    dct_blocks, cA_shape = image_to_dct_blocks(cA_img)

    # 量化表与位置
    qtbl = create_quantization_table(quality)
    qblocks = quantize_dct_blocks(dct_blocks, qtbl)
    if positions is None:
        positions = select_embedding_positions()

    bits = message_to_bits(message)
    bit_idx = 0
    h, w = qblocks.shape
    nb_h, nb_w = h//8, w//8

    for br in range(nb_h):
        for bc in range(nb_w):
            for (i, j) in positions:
                if bit_idx >= len(bits):
                    break
                ci = br*8 + i
                cj = bc*8 + j
                val = qblocks[ci, cj]
                if val != 0:
                    if (abs(val) % 2) != bits[bit_idx]:
                        qblocks[ci, cj] += 1 if val > 0 else -1
                else:
                    qblocks[ci, cj] = 1 if bits[bit_idx] else -1
                bit_idx += 1
            if bit_idx >= len(bits):
                break
        if bit_idx >= len(bits):
            break

    dct_deq = dequantize_dct_blocks(qblocks, qtbl)
    # 回写到近似系数，并重建图像
    coeffs[0] = dct_deq[:cA.shape[0], :cA.shape[1]]
    stego = dwt_to_image(coeffs, shape, wavelet)
    Image.fromarray(stego).save(output_path)
    return True
```

# 隐写容量与质量评估
- 容量：以比特或每像素比特（bpp）度量，受嵌入位置与强度影响。
- 质量：PSNR、SSIM等指标评估视觉失真。
- 可靠性：误码率（BER）衡量提取正确性。

```
import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None

def compute_psnr(img_path_a, img_path_b):
    a = np.array(Image.open(img_path_a).convert('L'), dtype=np.float64)
    b = np.array(Image.open(img_path_b).convert('L'), dtype=np.float64)
    mse = np.mean((a-b)**2)
    if mse == 0:
        return float('inf')
    return 10*np.log10(255**2/mse)

def compute_ssim(img_path_a, img_path_b):
    if ssim is None:
        print('SSIM依赖未安装（scikit-image），仅返回None')
        return None
    a = np.array(Image.open(img_path_a).convert('L'))
    b = np.array(Image.open(img_path_b).convert('L'))
    val = ssim(a, b, data_range=255)
    return val
```

# 安全性与抗检测策略
- 密钥控制的随机位置选择：通过伪随机序列选择嵌入位置，降低固定模式被检测的风险。
- 自适应嵌入强度：依据纹理/能量图动态调整。
- 统计平衡：在系数的正负与奇偶上保持平衡，抑制直方图与相关性异常。

```
import hashlib

def prng_positions(shape, count, key, block_size=8):
    h, w = shape
    seed = int(hashlib.sha256(key.encode('utf-8')).hexdigest(), 16) % (2**32-1)
    rng = np.random.default_rng(seed)
    pos = []
    for _ in range(count):
        i = rng.integers(1, block_size-1)
        j = rng.integers(1, block_size-1)
        pos.append((int(i), int(j)))
    return pos
```

# 频域隐写检测基线（机器学习）
结合统计特征与简单分类器，构建可操作的检测基线：

- 特征：DCT AC系数直方图、相邻相关性、DWT能量统计。
- 分类器：SVM或随机森林进行可疑样本判别。

```
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC

def extract_simple_features(image_path):
    # DCT AC直方图 + DWT统计
    img = Image.open(image_path).convert('L')
    dct_blocks, _ = image_to_dct_blocks(img)
    h, w = dct_blocks.shape
    nb_h, nb_w = h//8, w//8
    ac = []
    for bi in range(nb_h):
        for bj in range(nb_w):
            for i in range(8):
                for j in range(8):
                    if i==0 and j==0:
                        continue
                    ac.append(dct_blocks[bi*8+i, bj*8+j])
    hist, _ = np.histogram(ac, bins=64, range=(-100, 100), density=True)

    coeffs, _ = image_to_dwt(img, wavelet='haar', level=1)
    cA = coeffs[0]
    feats = {
        'dct_hist_mean': float(np.mean(hist)),
        'dct_hist_std': float(np.std(hist)),
        'dwt_mean': float(np.mean(cA)),
        'dwt_std': float(np.std(cA)),
        'dwt_kurtosis': float(np.mean(((cA-np.mean(cA))/ (np.std(cA)+1e-6))**4)),
    }
    return feats

def train_simple_detector(image_paths, labels):
    vec = DictVectorizer(sparse=False)
    X = [extract_simple_features(p) for p in image_paths]
    Xv = vec.fit_transform(X)
    clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    clf.fit(Xv, labels)
    return clf, vec

def predict_suspicion(clf, vec, image_path):
    x = vec.transform([extract_simple_features(image_path)])
    prob = clf.predict_proba(x)[0,1]
    return prob
```

# 工具与资源速查
- 依赖库：numpy、scipy、pywt、Pillow、scikit-image（SSIM）、scikit-learn。
- 数据集：BOSSBase、ALASKA、StegoAppDB，用于训练与评估。
- 常用工具：StegExpose（检测）、ImageMagick（预处理）、Matlab/Python工具箱（频域处理）。

# 法律与合规声明
- 本文内容仅用于教学与研究，不得用于规避版权保护或非法数据隐藏。
- 使用公共数据集与自有数据进行实验，遵守著作权法与隐私保护条例。
- 在受控环境开展测试，不得将隐写技术用于违法用途。

# 取证与报告模板要点（建议）
- 记录隐写载体来源、生成过程、参数与密钥管理方式。
- 评估视觉质量（PSNR/SSIM）与提取可靠性（BER），给出实验配置与可重复性说明。
- 归档脚本与版本依赖清单，附上样例输入/输出与检测结论。

# 参考文献与数据集
- Fridrich, J. Steganography in Digital Media: Principles, Algorithms, and Applications.
- Kodovsky, J., Fridrich, J. Steganalysis of content-adaptive steganography in spatial domain.
- BOSSBase, ALASKA Dataset.