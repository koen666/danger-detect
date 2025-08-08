#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模块连接脚本 - 用于验证和连接各个功能模块
"""

import os
import sys
import time
import logging
import argparse
import importlib
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("connect_modules.log")
    ]
)
logger = logging.getLogger(__name__)

# 定义可用模块类型
MODULE_TYPES = {
    'video': {
        'path': 'models.video.video_capture',
        'class': 'VideoCaptureManager',
        'status': False,
        'instance': None
    },
    'frame': {
        'path': 'models.frame.frame_processor',
        'class': 'FrameProcessor',
        'status': False,
        'instance': None
    },
    'detection': {
        'path': 'models.detection.object_detector',
        'class': 'create_detector',
        'status': False,
        'instance': None
    },
    'tracking': {
        'path': 'models.tracking.object_tracker',
        'class': 'ObjectTracker',
        'status': False,
        'instance': None
    },
    'motion': {
        'path': 'models.motion.motion_manager',
        'class': 'MotionFeatureManager',
        'status': False,
        'instance': None
    },
    'behavior': {
        'path': 'models.behavior.behavior_analyzer',
        'class': 'BehaviorAnalyzer',
        'status': False,
        'instance': None
    },
    'relation': {
        'path': 'models.relation.rga',
        'class': 'RelationGraphAnalyzer',
        'status': False,
        'instance': None
    },
    'vl_model': {
        'path': 'models.vision_language.vl_model',
        'class': 'VisionLanguageModel',
        'status': False,
        'instance': None
    },
    'alert': {
        'path': 'models.alert.alert_system',
        'class': 'AlertSystem',
        'status': False,
        'instance': None
    }
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='模块连接脚本')
    
    # 模块检查参数
    parser.add_argument('--check_modules', action='store_true', help='检查可用模块')
    parser.add_argument('--test_connection', action='store_true', help='测试模块连接')
    parser.add_argument('--list_dependencies', action='store_true', help='列出所有依赖')
    
    # 测试参数
    parser.add_argument('--test_video', action='store_true', help='测试视频模块')
    parser.add_argument('--test_motion', action='store_true', help='测试运动特征模块')
    parser.add_argument('--test_detection', action='store_true', help='测试对象检测模块')
    parser.add_argument('--test_alert', action='store_true', help='测试告警模块')
    
    # 视频源参数
    parser.add_argument('--source', type=str, default='0', help='视频源 (0表示摄像头, 或者是视频文件路径)')
    
    return parser.parse_args()

def check_module(module_name):
    """检查模块是否可用"""
    module_info = MODULE_TYPES.get(module_name)
    if not module_info:
        logger.error(f"未知模块: {module_name}")
        return False
    
    try:
        # 尝试导入模块
        module = importlib.import_module(module_info['path'])
        
        # 获取类或函数
        if hasattr(module, module_info['class']):
            module_info['status'] = True
            logger.info(f"模块 {module_name} ({module_info['path']}.{module_info['class']}) 可用")
            return True
        else:
            logger.warning(f"模块 {module_name} 导入成功，但找不到类/函数 {module_info['class']}")
            return False
            
    except ImportError as e:
        logger.warning(f"模块 {module_name} ({module_info['path']}) 不可用: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"检查模块 {module_name} 时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def check_all_modules():
    """检查所有模块"""
    logger.info("开始检查所有模块...")
    available_modules = []
    unavailable_modules = []
    
    for module_name in MODULE_TYPES:
        if check_module(module_name):
            available_modules.append(module_name)
        else:
            unavailable_modules.append(module_name)
    
    logger.info("模块检查完成")
    logger.info(f"可用模块 ({len(available_modules)}): {', '.join(available_modules)}")
    logger.info(f"不可用模块 ({len(unavailable_modules)}): {', '.join(unavailable_modules)}")
    
    return available_modules, unavailable_modules

def test_module_connection():
    """测试模块之间的连接"""
    logger.info("开始测试模块连接...")
    
    # 首先检查所有模块
    available_modules, _ = check_all_modules()
    
    # 测试视频捕获模块和运动特征模块的连接
    if 'video' in available_modules and 'motion' in available_modules:
        logger.info("测试视频捕获模块和运动特征模块的连接...")
        
        try:
            # 导入模块
            from models.video.video_capture import VideoCaptureManager
            from models.motion.motion_manager import MotionFeatureManager
            
            # 初始化模块
            video_manager = VideoCaptureManager(source='0')
            motion_manager = MotionFeatureManager(use_optical_flow=True)
            
            logger.info("✓ 视频捕获模块和运动特征模块可以连接")
        except Exception as e:
            logger.error(f"测试视频捕获模块和运动特征模块连接时出错: {str(e)}")
    
    # 测试运动特征模块和行为分析模块的连接
    if 'motion' in available_modules and 'behavior' in available_modules:
        logger.info("测试运动特征模块和行为分析模块的连接...")
        
        try:
            # 导入模块
            from models.motion.motion_manager import MotionFeatureManager
            from models.behavior.behavior_analyzer import BehaviorAnalyzer
            
            # 初始化模块
            motion_manager = MotionFeatureManager(use_optical_flow=True)
            behavior_analyzer = BehaviorAnalyzer()
            
            logger.info("✓ 运动特征模块和行为分析模块可以连接")
        except Exception as e:
            logger.error(f"测试运动特征模块和行为分析模块连接时出错: {str(e)}")
    
    # 测试对象检测和追踪模块的连接
    if 'detection' in available_modules and 'tracking' in available_modules:
        logger.info("测试对象检测和追踪模块的连接...")
        
        try:
            # 导入模块
            from models.detection.object_detector import create_detector
            from models.tracking.object_tracker import ObjectTracker
            
            # 初始化模块
            detector = create_detector(detector_type='yolov4')
            tracker = ObjectTracker(detector=detector)
            
            logger.info("✓ 对象检测和追踪模块可以连接")
        except Exception as e:
            logger.error(f"测试对象检测和追踪模块连接时出错: {str(e)}")
    
    # 测试行为分析和告警模块的连接
    if 'behavior' in available_modules and 'alert' in available_modules:
        logger.info("测试行为分析和告警模块的连接...")
        
        try:
            # 导入模块
            from models.behavior.behavior_analyzer import BehaviorAnalyzer
            from models.alert.alert_system import AlertSystem
            
            # 初始化模块
            behavior_analyzer = BehaviorAnalyzer()
            alert_system = AlertSystem()
            
            logger.info("✓ 行为分析和告警模块可以连接")
        except Exception as e:
            logger.error(f"测试行为分析和告警模块连接时出错: {str(e)}")
    
    logger.info("模块连接测试完成")

def list_dependencies():
    """列出所有依赖"""
    logger.info("系统依赖库:")
    logger.info("1. 基础库:")
    logger.info("   - opencv-python (cv2): 用于图像处理和视频捕获")
    logger.info("   - numpy: 用于数值计算")
    logger.info("   - Flask: Web界面支持")
    
    logger.info("2. 模型库:")
    logger.info("   - torch: PyTorch深度学习框架（对象检测、行为分析）")
    logger.info("   - onnxruntime: ONNX模型推理（可选）")
    
    logger.info("3. 工具库:")
    logger.info("   - tqdm: 进度条显示")
    logger.info("   - matplotlib: 可视化支持")
    logger.info("   - requests: HTTP请求（用于通知）")
    logger.info("   - pillow: 图像处理")
    
    try:
        import pkg_resources
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        
        logger.info("\n已安装的包:")
        for package, version in sorted(installed_packages.items()):
            logger.info(f"   - {package}: {version}")
    except Exception as e:
        logger.error(f"获取已安装包列表时出错: {str(e)}")

def test_video_module(args):
    """测试视频模块"""
    logger.info("测试视频捕获模块...")
    
    try:
        from models.video.video_capture import VideoCaptureManager
        import cv2
        
        # 初始化视频捕获管理器
        video_manager = VideoCaptureManager(
            source=args.source,
            width=640,
            height=480
        )
        
        if not video_manager.open():
            logger.error("无法打开视频源")
            return
        
        logger.info("✓ 视频捕获模块初始化成功")
        
        # 读取几帧并显示
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 100:  # 读取100帧
            ret, frame = video_manager.read()
            if not ret:
                break
            
            # 显示帧
            cv2.imshow('视频测试', frame)
            
            # 检查退出条件
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc键退出
                break
            
            frame_count += 1
        
        # 计算帧率
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        logger.info(f"读取了 {frame_count} 帧，平均帧率: {fps:.2f} FPS")
        
        # 清理资源
        video_manager.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"测试视频模块时出错: {str(e)}")
        logger.error(traceback.format_exc())

def test_motion_module(args):
    """测试运动特征模块"""
    logger.info("测试运动特征模块...")
    
    try:
        from models.motion.motion_manager import MotionFeatureManager
        import cv2
        import numpy as np
        
        # 初始化运动特征管理器
        motion_manager = MotionFeatureManager(
            use_optical_flow=True,
            use_motion_history=True
        )
        
        logger.info("✓ 运动特征模块初始化成功")
        
        # 创建测试帧
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame1, (320, 240), 50, (255, 255, 255), -1)
        
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame2, (340, 250), 50, (255, 255, 255), -1)
        
        # 提取特征
        features1 = motion_manager.extract_features(frame1, None)
        features2 = motion_manager.extract_features(frame2, frame1)
        
        logger.info(f"第一帧特征数量: {len(features1)}")
        logger.info(f"第二帧特征数量: {len(features2)}")
        
        # 可视化特征
        vis_frame = frame2.copy()
        if features2:
            vis_frame = motion_manager.visualize_features(vis_frame, features2)
        
        # 显示结果
        cv2.imshow('运动特征测试', vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        logger.info("✓ 运动特征测试完成")
        
    except Exception as e:
        logger.error(f"测试运动特征模块时出错: {str(e)}")
        logger.error(traceback.format_exc())

def test_detection_module(args):
    """测试对象检测模块"""
    logger.info("测试对象检测模块...")
    
    try:
        from models.detection.object_detector import create_detector
        import cv2
        
        # 初始化检测器
        detector = create_detector(detector_type='yolov4')
        
        logger.info("✓ 对象检测模块初始化成功")
        
        # 打开视频源
        if args.source.isdigit():
            source = int(args.source)
        else:
            source = args.source
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"无法打开视频源: {source}")
            return
        
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            logger.error("无法读取视频帧")
            return
        
        # 执行检测
        detections = detector.detect(frame)
        
        logger.info(f"检测到 {len(detections)} 个对象")
        
        # 可视化检测结果
        for det in detections:
            bbox = det['bbox']
            label = det['label']
            confidence = det['confidence']
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", 
                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('对象检测测试', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 清理资源
        cap.release()
        
        logger.info("✓ 对象检测测试完成")
        
    except Exception as e:
        logger.error(f"测试对象检测模块时出错: {str(e)}")
        logger.error(traceback.format_exc())

def test_alert_module(args):
    """测试告警模块"""
    logger.info("测试告警模块...")
    
    try:
        from models.alert.alert_system import AlertSystem
        
        # 初始化告警系统
        alert_system = AlertSystem()
        
        logger.info("✓ 告警模块初始化成功")
        
        # 创建测试告警
        test_alert = {
            'type': '测试告警',
            'confidence': 0.95,
            'timestamp': time.time(),
            'details': '这是一个测试告警，用于验证告警系统功能'
        }
        
        # 处理告警
        alert_system.process_alert(test_alert)
        
        logger.info("✓ 告警测试完成")
        
    except Exception as e:
        logger.error(f"测试告警模块时出错: {str(e)}")
        logger.error(traceback.format_exc())

def main():
    """主函数"""
    args = parse_args()
    
    try:
        # 检查模块
        if args.check_modules:
            check_all_modules()
        
        # 测试模块连接
        if args.test_connection:
            test_module_connection()
        
        # 列出依赖
        if args.list_dependencies:
            list_dependencies()
        
        # 测试视频模块
        if args.test_video:
            test_video_module(args)
        
        # 测试运动特征模块
        if args.test_motion:
            test_motion_module(args)
        
        # 测试对象检测模块
        if args.test_detection:
            test_detection_module(args)
        
        # 测试告警模块
        if args.test_alert:
            test_alert_module(args)
        
        # 如果没有指定任何操作，则打印帮助信息
        if not any([
            args.check_modules, args.test_connection, args.list_dependencies,
            args.test_video, args.test_motion, args.test_detection, args.test_alert
        ]):
            logger.info("没有指定任何操作。使用 --help 查看可用选项。")
            logger.info("可用操作:")
            logger.info("  --check_modules     - 检查可用模块")
            logger.info("  --test_connection   - 测试模块连接")
            logger.info("  --list_dependencies - 列出所有依赖")
            logger.info("  --test_video        - 测试视频模块")
            logger.info("  --test_motion       - 测试运动特征模块")
            logger.info("  --test_detection    - 测试对象检测模块")
            logger.info("  --test_alert        - 测试告警模块")
    
    except Exception as e:
        logger.error(f"系统运行出错: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 