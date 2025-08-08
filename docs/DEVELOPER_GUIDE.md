# 开发者指南

本文档提供视频监控危险行为检测系统的开发指南，帮助开发者理解系统架构、构建流程、测试方法，以及如何扩展系统功能。

## 目录

- [环境设置](#环境设置)
- [系统架构](#系统架构)
- [开发工作流](#开发工作流)
- [测试指南](#测试指南)
- [扩展系统](#扩展系统)
- [代码规范](#代码规范)
- [问题排查](#问题排查)

## 环境设置

### 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# 安装基本依赖
pip install -r requirements.txt

# 安装开发依赖（如果有）
pip install -r requirements-dev.txt
```

### 配置开发环境

推荐使用PyCharm或VS Code作为开发环境，配置如下：

- **PyCharm配置**:
  1. 打开项目目录
  2. 配置Python解释器（使用前面创建的虚拟环境）
  3. 将`src`目录标记为"Sources Root"
  4. 将`test`目录标记为"Test Sources Root"

- **VS Code配置**:
  1. 安装Python和Pylance扩展
  2. 配置`.vscode/settings.json`:
  ```json
  {
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.pythonPath": "venv/bin/python"
  }
  ```

## 系统架构

### 核心组件

![系统架构图](images/architecture.png)

系统由以下核心组件构成：

1. **视频输入模块**：处理来自摄像头或视频文件的输入
2. **特征提取模块**：从视频帧中提取运动特征
3. **行为分析模块**：分析运动特征，识别危险行为
4. **告警系统**：管理和分发告警信息
5. **Web界面**：提供用户交互界面

### 数据流

系统的数据流如下：

1. 视频输入 → 特征提取 → 行为分析 → 告警生成
2. 告警事件 → 通知分发 → 用户通知
3. 视频帧 → Web界面 → 用户

### 模块依赖

```
AllInOneSystem
├── VideoCapture
├── MotionFeatureManager
│   ├── OpticalFlowExtractor
│   └── MotionHistoryExtractor
├── DangerRecognizer
└── AlertSystem
    ├── AlertProcessor
    └── NotificationManager
```

## 开发工作流

### 分支管理

请遵循以下Git分支模型：

- `main`: 生产环境分支，保持稳定
- `develop`: 开发分支，所有特性开发完成后合并到这里
- `feature/*`: 特性分支，用于开发新功能
- `bugfix/*`: 用于修复问题
- `hotfix/*`: 用于生产环境紧急修复

### 提交代码

```bash
# 创建特性分支
git checkout -b feature/new-feature

# 开发完成后提交代码
git add .
git commit -m "feat: 添加新功能XX"

# 合并到develop分支
git checkout develop
git merge feature/new-feature
```

### 提交消息规范

请使用以下格式的提交消息：

- `feat`: 新功能
- `fix`: 修复问题
- `docs`: 文档更新
- `style`: 代码风格调整
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建或辅助工具变动

例如：`feat: 添加摔倒检测功能`

## 测试指南

### 单元测试

系统使用pytest进行单元测试：

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest test/test_motion_features.py

# 生成测试覆盖率报告
pytest --cov=src
```

### 集成测试

集成测试位于`test/integration`目录下：

```bash
# 运行集成测试
pytest test/integration
```

### 模拟数据

对于测试，系统提供了一系列模拟数据：

- `src/preprocessing_samples/`: 预处理的样本图像
- `src/video_samples/`: 测试视频样本

## 扩展系统

### 添加新的危险行为检测

在`src/danger_recognizer.py`中添加新的危险行为检测逻辑：

```python
# 在_analyze_danger方法中添加
def _analyze_danger(self, frame, features, object_detections=None):
    # 现有代码...
    
    # 添加新的行为检测，例如"异常聚集"
    if self._detect_abnormal_gathering(features, object_detections):
        alerts.append({
            'type': '异常聚集',
            'confidence': 0.75,
            'frame': frame
        })
    
    return alerts

# 添加新的检测方法
def _detect_abnormal_gathering(self, features, object_detections):
    # 实现检测逻辑
    if object_detections and len(object_detections) > 5:
        # 计算物体之间的距离
        # 检查是否形成聚集
        return True
    return False
```

### 添加新的特征提取器

1. 在`src/models/motion`目录下创建新的特征提取器文件，例如`crowd_density.py`
2. 实现从`MotionFeatureExtractor`继承的新类
3. 在`motion_manager.py`中集成新的特征提取器

```python
# 新的特征提取器
class CrowdDensityExtractor(MotionFeatureExtractor):
    def __init__(self, config=None):
        super().__init__()
        # 初始化...
    
    def _extract_impl(self, frame, prev_frame=None, **kwargs):
        # 实现特征提取逻辑
        return features
```

### 添加新的告警通知渠道

在`src/models/alert/notification_manager.py`中添加新的通知渠道：

```python
def send_notification(self, alert_event):
    # 现有代码...
    
    # 添加新的通知方式，例如Telegram
    if 'telegram' in self.channels:
        self._send_telegram_notification(alert_event)
    
    return True

def _send_telegram_notification(self, alert_event):
    # 实现Telegram通知逻辑
    # ...
```

## 代码规范

### 代码风格

本项目遵循PEP 8代码风格规范，建议使用`black`或`autopep8`进行格式化：

```bash
# 使用black格式化代码
black src/

# 使用flake8检查代码风格
flake8 src/
```

### 文档注释

请使用Google风格的文档注释：

```python
def function_name(param1, param2):
    """函数简短描述。

    详细描述（可选）...

    Args:
        param1: 参数1的描述
        param2: 参数2的描述

    Returns:
        返回值描述

    Raises:
        ValueError: 异常说明
    """
    # 函数实现...
```

### 类设计原则

- 遵循单一职责原则
- 使用组合优于继承
- 面向接口编程
- 保持类的方法和属性数量合理

## 问题排查

### 常见问题

1. **ModuleNotFoundError**:
   - 检查PYTHONPATH环境变量是否正确设置
   - 确认所有`__init__.py`文件都存在

2. **视频输入问题**:
   - 检查摄像头设备是否被其他程序占用
   - 尝试降低视频分辨率

3. **性能问题**:
   - 使用`cProfile`进行性能分析: `python -m cProfile -o output.prof src/all_in_one_system.py`
   - 使用`snakeviz output.prof`可视化性能分析结果

### 调试技巧

1. 在`src/models/motion/motion_manager.py`中添加详细日志：
```python
import logging
logging.getLogger("MotionFeatureManager").setLevel(logging.DEBUG)
```

2. 使用`cv2.imwrite`保存中间帧结果进行分析：
```python
cv2.imwrite("debug_frames/frame_{}.jpg".format(frame_count), frame)
```

3. 单独测试特定模块：
```python
# 测试光流提取
if __name__ == "__main__":
    import numpy as np
    import cv2
    
    extractor = OpticalFlowExtractor()
    cap = cv2.VideoCapture(0)
    
    ret, prev_frame = cap.read()
    while True:
        ret, frame = cap.read()
        features = extractor.extract(frame, prev_frame)
        # 可视化结果...
        prev_frame = frame.copy()
``` 