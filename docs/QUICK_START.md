# 快速入门指南

本指南将帮助您快速上手视频监控危险行为检测系统，包括安装、配置和基本使用方法。

## 目录

- [系统要求](#系统要求)
- [安装步骤](#安装步骤)
- [基本使用](#基本使用)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

## 系统要求

### 硬件要求

- **CPU**: 至少4核处理器
- **内存**: 至少4GB RAM
- **存储**: 至少1GB可用空间
- **摄像头**: 兼容的USB摄像头或IP摄像头

### 软件要求

- **操作系统**: Windows 10/11, Ubuntu 18.04/20.04, macOS 10.15+
- **Python**: 3.7或更高版本
- **依赖库**: 见`requirements.txt`文件

## 安装步骤

### 1. 克隆或下载项目

```bash
git clone https://github.com/yourusername/video-danger-detection.git
cd video-danger-detection
```

### 2. 创建虚拟环境 (可选但推荐)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 检查摄像头

确保您的摄像头已连接并且可以被系统识别。

## 基本使用

### 启动系统

**Windows**:
直接双击运行`run_all_in_one.bat`文件。

**Linux/Mac**:
```bash
python src/all_in_one_system.py
```

### 命令行参数

系统支持多种命令行参数以自定义运行方式：

```bash
python src/all_in_one_system.py --source 0 --width 640 --height 480 --save_alerts
```

常用参数:
- `--source`: 视频源（0表示第一个摄像头，也可以是视频文件路径）
- `--width` 和 `--height`: 视频分辨率
- `--save_alerts`: 保存告警帧到文件
- `--use_gpu`: 启用GPU加速（如果有CUDA支持）
- `--feature_threshold`: 特征点阈值，控制灵敏度
- `--web_interface`: 启用Web界面

### 系统操作

运行后，系统将打开一个窗口显示视频流和检测结果。您可以使用以下键盘快捷键：

- `ESC`: 退出程序
- `s`: 保存当前帧
- `r`: 重置统计信息
- `p`: 暂停/继续

## 配置说明

### 修改默认配置

您可以通过以下两种方式修改系统配置：

1. **命令行参数**:
   在启动时使用命令行参数覆盖默认设置。

2. **配置文件**:
   编辑`src/config/default.yaml`文件设置默认参数。

### 主要配置项

```yaml
# 视频配置
video:
  source: 0           # 视频源
  width: 640          # 宽度
  height: 480         # 高度
  fps: 30             # 目标帧率

# 危险行为检测配置
danger_detection:
  feature_threshold: 100  # 特征点阈值
  area_threshold: 0.05    # 区域阈值
  alert_cooldown: 10      # 告警冷却时间(帧)

# 系统配置
system:
  process_every: 3    # 每N帧处理一次
  use_gpu: false      # 是否使用GPU
  save_alerts: true   # 是否保存告警帧
  output_dir: "output" # 输出目录
```

### 自定义警戒区域

警戒区域是您特别关注的视频区域，可以通过两种方式定义：

1. **命令行参数**:
   ```bash
   python src/all_in_one_system.py --alert_region "[(100,100), (300,100), (300,300), (100,300)]"
   ```

2. **配置文件**:
   ```yaml
   alert_regions:
     - name: "入口区域"
       points: [[100, 100], [300, 100], [300, 300], [100, 300]]
     - name: "窗户区域"
       points: [[400, 150], [500, 150], [500, 250], [400, 250]]
   ```

## Web界面

如果启用了Web界面，您可以通过浏览器访问系统：

1. 启动系统时添加`--web_interface`参数：
   ```bash
   python src/all_in_one_system.py --web_interface
   ```

2. 打开浏览器访问：http://localhost:5000

Web界面提供以下功能：
- 实时视频流查看
- 系统状态监控
- 告警历史记录
- 开始/停止/暂停系统

## 常见问题

### 摄像头无法启动

**问题**: 运行程序时出现"无法打开摄像头"错误。

**解决方案**:
- 检查摄像头是否已连接
- 确保没有其他程序正在使用摄像头
- 尝试使用不同的摄像头ID（例如 `--source 1`）

### 系统运行缓慢

**问题**: 系统帧率低或响应迟缓。

**解决方案**:
- 降低视频分辨率（例如 `--width 320 --height 240`）
- 增加处理间隔（例如 `--process_every 5`）
- 关闭不必要的检测功能
- 如果有NVIDIA显卡，启用GPU加速 (`--use_gpu`)

### 误报过多

**问题**: 系统产生太多错误告警。

**解决方案**:
- 增加特征点阈值（例如 `--feature_threshold 150`）
- 增加运动区域阈值（例如 `--area_threshold 0.1`）
- 增加告警冷却时间（例如 `--alert_cooldown 20`）

### 无法检测到危险行为

**问题**: 系统未能检测到明显的危险行为。

**解决方案**:
- 降低特征点阈值（例如 `--feature_threshold 60`）
- 降低运动区域阈值（例如 `--area_threshold 0.03`）
- 确保摄像头视角合适，能够清晰捕捉到整个场景

### 内存使用过高

**问题**: 长时间运行后内存占用过高。

**解决方案**:
- 减少历史帧数量（编辑`danger_recognizer.py`中的`history_length`参数）
- 定期重启系统
- 检查是否有内存泄漏（开发者可使用memory_profiler工具） 