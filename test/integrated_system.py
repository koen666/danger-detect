#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
24小时视频监控告警系统 - 整合版
整合了所有功能模块：
1. 视频捕获
2. 帧处理
3. 对象检测和追踪
4. 运动特征提取
5. 行为识别
6. RGA图关系建模
7. 视觉语言理解
8. 告警系统
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
        logging.FileHandler("integrated_system.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='24小时视频监控告警系统')
    
    # 视频源参数
    parser.add_argument('--source', type=str, default='0', help='视频源 (0表示摄像头, 或者是视频文件路径)')
    parser.add_argument('--width', type=int, default=640, help='视频宽度')
    parser.add_argument('--height', type=int, default=480, help='视频高度')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率')
    
    # 帧处理参数
    parser.add_argument('--denoise', action='store_true', help='启用去噪处理')
    parser.add_argument('--enhance', action='store_true', help='启用图像增强')
    
    # 硬件加速
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU加速（如果可用）')
    
    # 对象检测和追踪参数
    parser.add_argument('--detector', type=str, default='yolov4', choices=['yolov4', 'yolov5', 'fasterrcnn'], help='对象检测器')
    parser.add_argument('--confidence', type=float, default=0.5, help='对象检测置信度阈值')
    parser.add_argument('--nms_threshold', type=float, default=0.4, help='非极大值抑制阈值')
    parser.add_argument('--max_disappeared', type=int, default=50, help='对象消失最大帧数')
    
    # 运动特征提取参数
    parser.add_argument('--use_optical_flow', action='store_true', help='启用光流特征提取')
    parser.add_argument('--use_motion_history', action='store_true', help='启用运动历史特征提取')
    parser.add_argument('--optical_flow_method', type=str, default='farneback', choices=['farneback', 'lucas_kanade'], help='光流算法')
    
    # 行为识别参数
    parser.add_argument('--trajectory_length', type=int, default=30, help='轨迹最大长度')
    parser.add_argument('--speed_threshold', type=float, default=2.0, help='速度阈值')
    parser.add_argument('--interaction_threshold', type=float, default=100.0, help='互动距离阈值')
    
    # RGA和视觉语言模型参数
    parser.add_argument('--enable_rga', action='store_true', help='启用RGA图关系建模')
    parser.add_argument('--enable_vl', action='store_true', help='启用视觉语言模型')
    parser.add_argument('--model_version', type=str, default='Qwen-VL-Chat', help='视觉语言模型版本')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='模型运行设备')
    parser.add_argument('--vl_interval', type=int, default=30, help='视觉语言处理帧间隔')
    parser.add_argument('--num_entities', type=int, default=5, help='场景图中的最大实体数')
    
    # 告警系统参数
    parser.add_argument('--rules_config', type=str, default='config/rules.json', help='规则配置文件路径')
    parser.add_argument('--notification_config', type=str, default='config/notification.json', help='通知配置文件路径')
    parser.add_argument('--alert_interval', type=int, default=60, help='告警间隔（秒）')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--save_frames', action='store_true', help='保存处理后的帧')
    parser.add_argument('--save_interval', type=int, default=30, help='保存帧的间隔')
    parser.add_argument('--display', action='store_true', help='显示实时处理结果')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    try:
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 导入必要的模块
        try:
            from models.video.video_capture import VideoCaptureManager
            from models.frame.frame_processor import FrameProcessor
            from models.detection.object_detector import create_detector
            from models.tracking.object_tracker import ObjectTracker
            from models.motion.motion_manager import MotionFeatureManager
            from models.behavior.behavior_analyzer import BehaviorAnalyzer
            from models.relation.rga import RelationGraphAnalyzer
            from models.vision_language.vl_model import VisionLanguageModel
            from models.alert.alert_system import AlertSystem
            
            logger.info("成功导入所有模块")
        except ImportError as e:
            logger.error(f"导入模块失败: {str(e)}")
            return
        
        # 初始化视频捕获管理器
        video_manager = VideoCaptureManager(
            source=args.source,
            width=args.width,
            height=args.height,
            fps=args.fps
        )
        
        # 初始化帧处理器
        frame_processor = FrameProcessor(
            denoise=args.denoise,
            enhance=args.enhance,
            resize_dim=(args.width, args.height)
        )
        
        # 初始化对象检测器
        detector = create_detector(
            detector_type=args.detector,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms_threshold,
            use_gpu=args.use_gpu
        )
        
        # 初始化对象追踪器
        tracker = ObjectTracker(
            detector=detector,
            max_disappeared=args.max_disappeared
        )
        
        # 初始化运动特征管理器
        motion_manager = MotionFeatureManager(
            use_optical_flow=args.use_optical_flow,
            use_motion_history=args.use_motion_history,
            optical_flow_method=args.optical_flow_method,
            use_gpu=args.use_gpu
        )
        
        # 初始化行为分析器
        behavior_analyzer = BehaviorAnalyzer(
            trajectory_length=args.trajectory_length,
            speed_threshold=args.speed_threshold,
            interaction_threshold=args.interaction_threshold
        )
        
        # 初始化RGA（如果启用）
        rga = None
        if args.enable_rga:
            rga = RelationGraphAnalyzer(
                num_entities=args.num_entities,
                use_gpu=args.use_gpu
            )
        
        # 初始化视觉语言模型（如果启用）
        vl_model = None
        if args.enable_vl:
            vl_model = VisionLanguageModel(
                model_version=args.model_version,
                device=args.device
            )
        
        # 初始化告警系统
        alert_system = AlertSystem(
            rules_config=args.rules_config,
            notification_config=args.notification_config,
            alert_interval=args.alert_interval
        )
        
        # 打开视频源
        if not video_manager.open():
            logger.error("无法打开视频源")
            return
        
        # 创建显示窗口
        if args.display:
            cv2.namedWindow('24小时告警系统', cv2.WINDOW_NORMAL)
        
        # 初始化状态变量
        frame_count = 0
        start_time = time.time()
        prev_frame = None
        
        logger.info("开始处理视频流...")
        
        while video_manager.is_opened():
            # 读取帧
            ret, frame = video_manager.read()
            if not ret:
                break
            
            # 处理帧
            processed_frame = frame_processor.process(frame)
            
            # 获取对象检测和追踪结果
            tracks = tracker.update(processed_frame)
            
            # 提取运动特征
            motion_features = motion_manager.extract_features(processed_frame, prev_frame, tracks)
            
            # 分析行为
            behaviors = behavior_analyzer.analyze(tracks, motion_features)
            
            # 构建关系图（如果启用）
            relations = []
            if rga and len(tracks) > 1:
                relations = rga.analyze(tracks, behaviors)
            
            # 视觉语言分析（如果启用且满足间隔条件）
            scene_description = None
            if vl_model and frame_count % args.vl_interval == 0:
                scene_description = vl_model.analyze(processed_frame, tracks, behaviors)
            
            # 处理告警
            alerts = alert_system.process(
                frame=processed_frame,
                tracks=tracks,
                motion_features=motion_features,
                behaviors=behaviors,
                relations=relations,
                scene_description=scene_description
            )
            
            # 可视化结果
            vis_frame = processed_frame.copy()
            
            # 绘制对象框和ID
            for track_id, track in tracks.items():
                bbox = track['bbox']
                label = track['label']
                confidence = track['confidence']
                
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(vis_frame, f"{label} {track_id} ({confidence:.2f})", 
                           (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 可视化运动特征
            if motion_features:
                vis_frame = motion_manager.visualize_features(vis_frame, motion_features)
            
            # 可视化行为
            for behavior in behaviors:
                obj_id = behavior['object_id']
                behavior_type = behavior['type']
                confidence = behavior['confidence']
                
                if obj_id in tracks:
                    bbox = tracks[obj_id]['bbox']
                    x1, y1 = int(bbox[0]), int(bbox[1])
                    cv2.putText(vis_frame, f"{behavior_type} ({confidence:.2f})",
                               (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 可视化告警
            for alert in alerts:
                alert_type = alert['type']
                confidence = alert['confidence']
                
                # 在顶部显示告警信息
                cv2.putText(vis_frame, f"ALERT: {alert_type} ({confidence:.2f})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
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
        
        # 清理资源
        video_manager.release()
        if args.display:
            cv2.destroyAllWindows()
        
        # 显示统计信息
        elapsed = time.time() - start_time
        logger.info(f"处理完成: {frame_count} 帧在 {elapsed:.2f} 秒内 (平均 {frame_count/elapsed:.2f} FPS)")
    
    except Exception as e:
        logger.error(f"系统运行出错: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 