#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
24小时视频监控告警系统 - 灵活版
根据可用模块动态启用/禁用功能
"""

import os
import sys
import time
import logging
import argparse
import cv2
import numpy as np
import traceback
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("flexible_system.log")
    ]
)
logger = logging.getLogger(__name__)

# 定义可用模块列表
AVAILABLE_MODULES = {
    'video_capture': False,
    'frame_processor': False,
    'object_detector': False,
    'object_tracker': False,
    'motion_manager': False,
    'behavior_analyzer': False,
    'rga': False,
    'vl_model': False,
    'alert_system': False
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='24小时视频监控告警系统 - 灵活版')
    
    # 视频源参数
    parser.add_argument('--source', type=str, default='0', help='视频源 (0表示摄像头, 或者是视频文件路径)')
    parser.add_argument('--width', type=int, default=640, help='视频宽度')
    parser.add_argument('--height', type=int, default=480, help='视频高度')
    
    # 功能启用/禁用
    parser.add_argument('--use_optical_flow', action='store_true', help='启用光流特征提取')
    parser.add_argument('--use_motion_history', action='store_true', help='启用运动历史特征提取')
    parser.add_argument('--use_object_detection', action='store_true', help='启用对象检测')
    parser.add_argument('--use_behavior', action='store_true', help='启用行为分析')
    parser.add_argument('--use_alert', action='store_true', help='启用告警系统')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--save_frames', action='store_true', help='保存处理后的帧')
    parser.add_argument('--save_interval', type=int, default=30, help='保存帧的间隔')
    parser.add_argument('--display', action='store_true', help='显示实时处理结果')
    
    return parser.parse_args()

def check_available_modules():
    """检查哪些模块可用"""
    global AVAILABLE_MODULES
    
    # 检查视频捕获模块
    try:
        from models.video.video_capture import VideoCaptureManager
        AVAILABLE_MODULES['video_capture'] = True
        logger.info("视频捕获模块可用")
    except ImportError:
        logger.warning("视频捕获模块不可用，将使用OpenCV直接读取视频")
    
    # 检查帧处理模块
    try:
        from models.frame.frame_processor import FrameProcessor
        AVAILABLE_MODULES['frame_processor'] = True
        logger.info("帧处理模块可用")
    except ImportError:
        logger.warning("帧处理模块不可用，将跳过帧预处理")
    
    # 检查对象检测模块
    try:
        from models.detection.object_detector import create_detector
        AVAILABLE_MODULES['object_detector'] = True
        logger.info("对象检测模块可用")
    except ImportError:
        logger.warning("对象检测模块不可用，将跳过对象检测")
    
    # 检查对象追踪模块
    try:
        from models.tracking.object_tracker import ObjectTracker
        AVAILABLE_MODULES['object_tracker'] = True
        logger.info("对象追踪模块可用")
    except ImportError:
        logger.warning("对象追踪模块不可用，将跳过对象追踪")
    
    # 检查运动特征管理器
    try:
        from models.motion.motion_manager import MotionFeatureManager
        AVAILABLE_MODULES['motion_manager'] = True
        logger.info("运动特征管理器可用")
    except ImportError:
        logger.warning("运动特征管理器不可用，将跳过运动特征提取")
    
    # 检查行为分析器
    try:
        from models.behavior.behavior_analyzer import BehaviorAnalyzer
        AVAILABLE_MODULES['behavior_analyzer'] = True
        logger.info("行为分析器可用")
    except ImportError:
        logger.warning("行为分析器不可用，将跳过行为分析")
    
    # 检查RGA
    try:
        from models.relation.rga import RelationGraphAnalyzer
        AVAILABLE_MODULES['rga'] = True
        logger.info("RGA图关系分析器可用")
    except ImportError:
        logger.warning("RGA图关系分析器不可用，将跳过关系分析")
    
    # 检查视觉语言模型
    try:
        from models.vision_language.vl_model import VisionLanguageModel
        AVAILABLE_MODULES['vl_model'] = True
        logger.info("视觉语言模型可用")
    except ImportError:
        logger.warning("视觉语言模型不可用，将跳过场景描述")
    
    # 检查告警系统
    try:
        from models.alert.alert_system import AlertSystem
        AVAILABLE_MODULES['alert_system'] = True
        logger.info("告警系统可用")
    except ImportError:
        logger.warning("告警系统不可用，将跳过告警处理")

def main():
    """主函数"""
    args = parse_args()
    
    # 检查可用模块
    check_available_modules()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化组件
    video_manager = None
    frame_processor = None
    detector = None
    tracker = None
    motion_manager = None
    behavior_analyzer = None
    rga = None
    vl_model = None
    alert_system = None
    
    # 初始化视频捕获
    if AVAILABLE_MODULES['video_capture']:
        from models.video.video_capture import VideoCaptureManager
        video_manager = VideoCaptureManager(
            source=args.source,
            width=args.width,
            height=args.height
        )
    
    # 初始化帧处理器
    if AVAILABLE_MODULES['frame_processor']:
        from models.frame.frame_processor import FrameProcessor
        frame_processor = FrameProcessor(
            resize_dim=(args.width, args.height)
        )
    
    # 初始化对象检测和追踪
    if AVAILABLE_MODULES['object_detector'] and args.use_object_detection:
        from models.detection.object_detector import create_detector
        detector = create_detector(detector_type='yolov4')
        
        if AVAILABLE_MODULES['object_tracker']:
            from models.tracking.object_tracker import ObjectTracker
            tracker = ObjectTracker(detector=detector)
    
    # 初始化运动特征管理器
    if AVAILABLE_MODULES['motion_manager'] and (args.use_optical_flow or args.use_motion_history):
        from models.motion.motion_manager import MotionFeatureManager
        motion_manager = MotionFeatureManager(
            use_optical_flow=args.use_optical_flow,
            use_motion_history=args.use_motion_history
        )
    
    # 初始化行为分析器
    if AVAILABLE_MODULES['behavior_analyzer'] and args.use_behavior:
        from models.behavior.behavior_analyzer import BehaviorAnalyzer
        behavior_analyzer = BehaviorAnalyzer()
    
    # 初始化告警系统
    if AVAILABLE_MODULES['alert_system'] and args.use_alert:
        from models.alert.alert_system import AlertSystem
        alert_system = AlertSystem()
    
    # 打开视频源
    cap = None
    if video_manager:
        if not video_manager.open():
            logger.error("无法打开视频源通过VideoCaptureManager")
            logger.info("尝试使用OpenCV直接打开视频源")
            video_manager = None
    
    if not video_manager:
        # 如果专用管理器不可用，则使用OpenCV直接打开
        if args.source.isdigit():
            source = int(args.source)
        else:
            source = args.source
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"无法打开视频源: {source}")
            return
        
        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # 创建显示窗口
    if args.display:
        cv2.namedWindow('24小时告警系统', cv2.WINDOW_NORMAL)
    
    # 初始化状态变量
    frame_count = 0
    start_time = time.time()
    prev_frame = None
    
    logger.info("开始处理视频流...")
    
    try:
        while True:
            # 读取帧
            if video_manager:
                ret, frame = video_manager.read()
            else:
                ret, frame = cap.read()
            
            if not ret:
                break
            
            # 处理帧
            if frame_processor:
                processed_frame = frame_processor.process(frame)
            else:
                processed_frame = frame
            
            # 对象检测和追踪
            tracks = {}
            if tracker:
                tracks = tracker.update(processed_frame)
            
            # 提取运动特征
            motion_features = []
            if motion_manager:
                motion_features = motion_manager.extract_features(processed_frame, prev_frame, tracks)
            
            # 行为分析
            behaviors = []
            if behavior_analyzer and (tracks or motion_features):
                behaviors = behavior_analyzer.analyze(tracks, motion_features)
            
            # 处理告警
            alerts = []
            if alert_system:
                alerts = alert_system.process(
                    frame=processed_frame,
                    tracks=tracks,
                    motion_features=motion_features,
                    behaviors=behaviors
                )
            
            # 可视化结果
            vis_frame = processed_frame.copy()
            
            # 绘制对象框和ID
            for track_id, track in tracks.items():
                bbox = track['bbox']
                label = track.get('label', 'Object')
                confidence = track.get('confidence', 1.0)
                
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(vis_frame, f"{label} {track_id} ({confidence:.2f})", 
                          (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 可视化运动特征
            if motion_manager and motion_features:
                vis_frame = motion_manager.visualize_features(vis_frame, motion_features)
            
            # 可视化行为
            for behavior in behaviors:
                obj_id = behavior['object_id']
                behavior_type = behavior['type']
                confidence = behavior.get('confidence', 1.0)
                
                if obj_id in tracks:
                    bbox = tracks[obj_id]['bbox']
                    x1, y1 = int(bbox[0]), int(bbox[1])
                    cv2.putText(vis_frame, f"{behavior_type} ({confidence:.2f})",
                              (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 可视化告警
            for alert in alerts:
                alert_type = alert['type']
                confidence = alert.get('confidence', 1.0)
                
                # 在顶部显示告警信息
                cv2.putText(vis_frame, f"ALERT: {alert_type} ({confidence:.2f})",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 显示功能状态
            status_text = []
            if args.use_optical_flow:
                status_text.append("光流:开")
            if args.use_motion_history:
                status_text.append("运动历史:开")
            if args.use_object_detection:
                status_text.append("对象检测:开")
            if args.use_behavior:
                status_text.append("行为分析:开")
            if args.use_alert:
                status_text.append("告警:开")
            
            status_str = " | ".join(status_text)
            cv2.putText(vis_frame, status_str, (10, vis_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 添加帧率信息
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, vis_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示结果
            if args.display:
                cv2.imshow('24小时告警系统', vis_frame)
            
            # 保存结果
            if args.save_frames and frame_count % args.save_interval == 0:
                output_path = os.path.join(args.output_dir, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(output_path, vis_frame)
                logger.info(f"保存帧到: {output_path}")
            
            # 保存告警截图
            if alerts:
                alert_dir = os.path.join(args.output_dir, 'alerts')
                os.makedirs(alert_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                alert_path = os.path.join(alert_dir, f"alert_{timestamp}_{alerts[0]['type']}.jpg")
                cv2.imwrite(alert_path, vis_frame)
                logger.info(f"保存告警截图到: {alert_path}")
            
            # 更新状态
            prev_frame = processed_frame.copy()
            frame_count += 1
            
            # 检查退出条件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q或Esc键退出
                break
    
    except Exception as e:
        logger.error(f"系统运行出错: {str(e)}")
        logger.error(traceback.format_exc())
    
    finally:
        # 清理资源
        if video_manager:
            video_manager.release()
        elif cap and cap.isOpened():
            cap.release()
        
        if args.display:
            cv2.destroyAllWindows()
        
        # 显示统计信息
        elapsed = time.time() - start_time
        logger.info(f"处理完成: {frame_count} 帧在 {elapsed:.2f} 秒内 (平均 {frame_count/elapsed:.2f} FPS)")

if __name__ == "__main__":
    main() 