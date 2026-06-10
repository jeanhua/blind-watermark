# Blind-Watermark

Go 实现的盲水印 CLI 工具，基于 DWT + DCT + SVD 算法，将文本水印隐式嵌入图像，肉眼不可见，支持提取还原。

**参考算法来源**： [guofei9987/blind_watermark](https://github.com/guofei9987/blind_watermark)

## 原理

对图像的 YUV 三个通道分别执行 Haar 小波分解（DWT），取低频分量 CA，切分为 4×4 块。每块依次做 DCT → 随机置乱（种子加密）→ SVD → 修改奇异值嵌入 1 bit 信息 → 逆变换还原。提取时通过读取奇异值恢复水印位，经 K-means 阈值化得到最终结果。


## 安装

### 预编译二进制【推荐】

从 [GitHub Releases]() 下载对应平台二进制，直接运行。

### 从源码编译

```bash
git clone https://github.com/jeanhua/blind-watermark.git
cd blind-watermark
go build .
```

## 使用

### 嵌入水印

```bash
blind_watermark_go embed --pwd 1234 input.jpg "secret message" output.png
```

参数：
- `--pwd` / `-p`：嵌入密码，用于种子化随机置乱（默认 1）
- `--pwd_wm`：水印加密密码（可选，默认与 `--pwd`相同）
- 位置参数：`<原图> <水印文本> <输出图>`

### 提取水印

```bash
blind_watermark_go extract --pwd 1234 --wm_shape 112 output.png
```

参数：
- `--pwd` / `-p`：提取密码，必须与嵌入时一致
- `--pwd_wm`：水印加密密码（可选）
- `--wm_shape`：水印位长度（嵌入时会输出该值，提取必须一致）
- 位置参数：`<含水印图片>`

### 示例

```bash
# 嵌入
$ blind_watermark_go embed --pwd 42 photo.jpg "你好世界" marked.png
Watermark bits length: 12
Embedded bits count: 96
Embed succeeded! Output: marked.png

# 提取
$ blind_watermark_go extract --pwd 42 --wm_shape 96 marked.png
Extract succeeded! Watermark is:
你好世界
```

## 相比 Python 原版的优化

| 类别 | 优化 | 说明 |
|------|------|------|
| **语言** | Python → Go | 编译为单二进制，无运行时依赖，部署简单 |
| **并发** | 通道级并行 | Y/U/V 三个通道的嵌入、提取、DWT 分解/重构全部 goroutine 并发 |
| **并发** | 块级并行 | 每个通道内数千个 4×4 块按行分片并行处理 |
| **并发** | 像素级并行 | BGR↔YUV 颜色转换、像素裁剪均按行并行 |
| **依赖** | 极简依赖 | 核心算法仅用 Go 标准库，仅 CLI 框架依赖 cobra，无需 NumPy / OpenCV / PyWavelets |

