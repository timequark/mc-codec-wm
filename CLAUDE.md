# 图片水印嵌入与解码
输入：
- 水印文本, 16位字母或数字组成
- 冗余份数 repl
- 原图
- 原图蒙版（或选）, 后面升级版中考虑使用蒙版中alpah==0的区域跳过图片中敏感区域

## 规则
- 水印（16字节） + reedsolo ecc（16字节） 后转成bit位1或0
- DCT BLOCK 大小 BLOCK_SIZE 可以为8、12、16
- 原图alpha>0的区域按 BLOCK_SIZE 自左向右、自上而下划分出DCT块网格
- 水印总计冗余份数 * 单条水印+ECC后的bit长度 不超过总的 DCT BLOCK 数量
- 一个 DCT 块嵌入一个 bit
- 嵌入时使用策略1
- 解码参考解码规则

## 拍照解码前的图像矫正
参考 demo_compare_SIFT_vulcan_hat.py 的 SIFT 特征点思想，进行矫正。

需要考虑的问题：
- 矫正完成后可能出现的整体像素位移问题
- 如何判断像素位移，如果出现像素位移，如何解决

考虑下面的矫正方案：
```
拍照图
      │
      ▼
去畸变（undistort）
      │
      ▼
SIFT / ORB 特征匹配
      │
      ▼
findHomography(RANSAC)
      │
      ▼
warpPerspective（粗配准）
      │
      ▼
Phase Correlation（估计剩余 dx、dy）
      │
      ▼
warpAffine（平移补偿）
      │
      ▼
ECC 精配准（可选，但推荐）
      │
      ▼
最终与模板达到亚像素级对齐
      │
      ▼
DWT / 水印解码
```

## DCT BLOCK 的嵌入策略

### 策略 1
DCT BLOCK （8、12、16）的中频带先一对像素位置，例如位置 A / B，
根据能量强度 delta 参数，
bit == 1 时，如果 coef[A] + delta < coef[B], 动态调整 coef[A] 、coef[B] 值，使得 coef[A] + delta <= coef[B], A -> B 方向性增大 delta

bit == 0 时，如果 coef[A] - delta > coef[B], 动态调整 coef[A] 、coef[B] 值，使得 coef[A] - delta >= coef[B], A -> B 方向性减小 delta

### 策略 2


## 解码规则
流程
```
读取Block

↓

得到bit

↓

按repl分组

↓

每份ECC

↓

成功直接记录

↓

失败保留
```
最后
```
所有成功结果

↓

统计频率

↓

最多的

↓

输出
```
如果全部失败，再进行
```
Bit Vote

↓

ECC
```

解码不用 delta
```
coefB > coefA

→1

否则

→0
```