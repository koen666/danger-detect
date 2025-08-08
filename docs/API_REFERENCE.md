# API 参考文档

本文档提供了视频监控危险行为检测系统的主要组件和API说明。

## 目录

- [系统主模块](#系统主模块)
- [危险行为识别模块](#危险行为识别模块)
- [运动特征提取模块](#运动特征提取模块)
- [告警系统模块](#告警系统模块)
- [视频处理模块](#视频处理模块)
- [配置管理模块](#配置管理模块)
- [工具函数模块](#工具函数模块)

## 系统主模块

`src.all_in_one_system.AllInOneSystem`

系统的核心类，集成了所有功能模块，提供完整的危险行为检测系统。

### 初始化

```python
system = AllInOneSystem(args)
```

参数:
- `args`: 配置参数对象，包含系统运行所需的各种参数

### 主要方法

#### start()

启动系统，开始视频处理和分析。

```python
system.start()
```

#### stop()

停止系统运行。

```python
system.stop()
```

#### process_frame(frame)

处理单帧图像，执行运动分析和危险行为检测。

```python
alerts = system.process_frame(frame)
```

参数:
- `frame`: 输入的视频帧（numpy数组）

返回:
- 检测到的告警列表

#### generate_report()

生成系统运行报告。

```python
report = system.generate_report()
```

返回:
- 包含统计信息的报告字符串

## 危险行为识别模块

`src.danger_recognizer.DangerRecognizer`

负责识别视频中的危险行为。

### 初始化

```python
recognizer = DangerRecognizer(config)
```

参数:
- `config`: 配置字典，包含检测参数和阈值

### 主要方法

#### process_frame(frame, features, object_detections=None)

处理当前帧，分析是否存在危险行为。

```python
alerts = recognizer.process_frame(frame, features, object_detections)
```

参数:
- `frame`: 当前视频帧
- `features`: 从运动特征管理器获取的特征列表
- `object_detections`: 可选的物体检测结果

返回:
- 检测到的告警列表

#### add_alert_region(region, name="警戒区")

添加自定义警戒区域。

```python
region_id = recognizer.add_alert_region([(100,100), (300,100), (300,300), (100,300)], "入口区域")
```

参数:
- `region`: 多边形区域的顶点坐标列表
- `name`: 区域名称

返回:
- 区域ID

## 运动特征提取模块

### MotionFeatureManager

`src.models.motion.motion_manager.MotionFeatureManager`

运动特征管理器，负责协调和管理各种运动特征提取器。

#### 初始化

```python
manager = MotionFeatureManager(
    use_optical_flow=True,
    use_motion_history=False,
    optical_flow_method='farneback',
    use_gpu=False
)
```

参数:
- `use_optical_flow`: 是否使用光流
- `use_motion_history`: 是否使用运动历史
- `optical_flow_method`: 光流方法 ('farneback', 'sparse', 'dense_pyr_lk')
- `use_gpu`: 是否使用GPU加速

#### 主要方法

##### extract_features(frame, prev_frame=None)

从帧中提取特征。

```python
features = manager.extract_features(frame, prev_frame)
```

参数:
- `frame`: 当前帧
- `prev_frame`: 前一帧（可选）

返回:
- 提取的特征字典

### OpticalFlowExtractor

`src.models.motion.optical_flow.OpticalFlowExtractor`

光流特征提取器，用于计算连续帧之间的移动。

#### 初始化

```python
extractor = OpticalFlowExtractor(
    method='farneback',
    use_gpu=False
)
```

参数:
- `method`: 光流方法 ('farneback', 'pyr_lk')
- `use_gpu`: 是否使用GPU加速

#### 主要方法

##### extract(frame, prev_frame=None, **kwargs)

提取光流特征。

```python
features = extractor.extract(frame, prev_frame)
```

参数:
- `frame`: 当前帧
- `prev_frame`: 前一帧

返回:
- 光流特征列表

### MotionHistoryExtractor

`src.models.motion.motion_history.MotionHistoryExtractor`

运动历史提取器，用于分析一段时间内的运动模式。

#### 初始化

```python
extractor = MotionHistoryExtractor(history_length=20, threshold=30)
```

参数:
- `history_length`: 历史长度（帧数）
- `threshold`: 运动检测阈值

#### 主要方法

##### extract(frame, prev_frame=None, **kwargs)

提取运动历史特征。

```python
features = extractor.extract(frame, prev_frame)
```

参数:
- `frame`: 当前帧
- `prev_frame`: 前一帧

返回:
- 运动历史特征列表

## 告警系统模块

### AlertSystem

`src.models.alert.alert_system.AlertSystem`

告警系统，负责管理和处理各类告警事件。

#### 初始化

```python
alert_system = AlertSystem(config)
```

参数:
- `config`: 告警系统配置

#### 主要方法

##### add_alert(alert_event)

添加新的告警事件。

```python
alert_id = alert_system.add_alert(alert_event)
```

参数:
- `alert_event`: 告警事件对象

返回:
- 告警ID

##### get_alerts(filter_func=None)

获取告警列表。

```python
alerts = alert_system.get_alerts(lambda a: a.confidence > 0.8)
```

参数:
- `filter_func`: 过滤函数（可选）

返回:
- 告警列表

### NotificationManager

`src.models.alert.notification_manager.NotificationManager`

通知管理器，负责发送告警通知。

#### 初始化

```python
manager = NotificationManager(config)
```

参数:
- `config`: 通知配置

#### 主要方法

##### send_notification(alert_event)

发送告警通知。

```python
success = manager.send_notification(alert_event)
```

参数:
- `alert_event`: 告警事件对象

返回:
- 是否发送成功

## 视频处理模块

### VideoCapture

`src.models.video.video_capture.VideoCapture`

视频捕获类，用于从摄像头或视频文件获取视频帧。

#### 初始化

```python
capture = VideoCapture(source=0, width=640, height=480)
```

参数:
- `source`: 视频源（0表示摄像头，也可以是视频文件路径）
- `width`: 视频宽度
- `height`: 视频高度

#### 主要方法

##### read()

读取下一帧。

```python
success, frame = capture.read()
```

返回:
- `success`: 是否成功读取
- `frame`: 视频帧

##### release()

释放视频源。

```python
capture.release()
```

### FrameProcessor

`src.models.video.frame_processor.FrameProcessor`

帧处理器，用于图像处理和优化。

#### 初始化

```python
processor = FrameProcessor(config)
```

参数:
- `config`: 处理配置

#### 主要方法

##### process(frame)

处理视频帧。

```python
processed_frame = processor.process(frame)
```

参数:
- `frame`: 输入帧

返回:
- 处理后的帧

## 配置管理模块

### Config

`src.config.Config`

配置管理类，用于加载和保存配置。

#### 主要方法

##### load_config(config_path)

加载配置文件。

```python
config = Config.load_config("config.yaml")
```

参数:
- `config_path`: 配置文件路径

返回:
- 配置字典

##### save_config(config, config_path)

保存配置到文件。

```python
Config.save_config(config, "config.yaml")
```

参数:
- `config`: 配置字典
- `config_path`: 配置文件保存路径

## 工具函数模块

### frame_validation

`src.utils.frame_validation`

帧验证工具函数。

#### 主要函数

##### validate_frame(frame)

验证帧是否有效。

```python
is_valid, error_message = validate_frame(frame)
```

参数:
- `frame`: 视频帧

返回:
- `is_valid`: 是否有效
- `error_message`: 错误信息（如果无效）

### motion_utils

`src.utils.motion_utils`

运动分析工具函数。

#### 主要函数

##### calculate_motion_metrics(motion_vectors)

计算运动特征统计信息。

```python
metrics = calculate_motion_metrics(motion_vectors)
```

参数:
- `motion_vectors`: 运动矢量列表

返回:
- 统计指标字典

### preprocessing

`src.utils.preprocessing`

图像预处理工具函数。

#### 主要函数

##### preprocess_frame(frame)

预处理视频帧。

```python
processed_frame = preprocess_frame(frame)
```

参数:
- `frame`: 输入帧

返回:
- 预处理后的帧 