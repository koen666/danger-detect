#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频运动分析系统 - 简化版
专注于视频捕获和运动特征提取
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
        logging.FileHandler("motion_video_system.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='视频运动分析系统 - 简化版')
    
    # 视频源参数
    parser.add_argument('--source', type=str, default='0', help='视频源 (0表示摄像头, 或者是视频文件路径)')
    parser.add_argument('--width', type=int, default=640, help='视频宽度')
    parser.add_argument('--height', type=int, default=480, help='视频高度')
    
    # 运动特征参数
    parser.add_argument('--use_optical_flow', action='store_true', help='启用光流特征提取')
    parser.add_argument('--use_motion_history', action='store_true', help='启用运动历史特征提取')
    parser.add_argument('--optical_flow_method', type=str, default='farneback', 
                       choices=['farneback', 'lucas_kanade'], help='光流算法')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU加速（如果可用）')
    
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
        
        # 导入运动特征管理器
        try:
            from models.motion.motion_manager import MotionFeatureManager
            logger.info("成功导入MotionFeatureManager")
        except ImportError as e:
            logger.error(f"导入MotionFeatureManager失败: {str(e)}")
            return
        
        # 初始化运动特征管理器
        motion_manager = MotionFeatureManager(
            use_optical_flow=args.use_optical_flow,
            use_motion_history=args.use_motion_history,
            optical_flow_method=args.optical_flow_method,
            use_gpu=args.use_gpu
        )
        
        # 打开视频源
        if args.source.isdigit():
            source = int(args.source)
        else:
            source = args.source
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"无法打开视频源: {source}")
            # 使用测试模式
            logger.info("切换到测试模式")
            frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            test_mode = True
        else:
            # 设置视频分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            ret, frame = cap.read()
            if not ret:
                logger.error("无法读取视频帧")
                return
            test_mode = False
            logger.info(f"成功打开视频源: {source}")
        
        # 创建显示窗口
        if args.display:
            cv2.namedWindow('视频运动分析', cv2.WINDOW_NORMAL)
        
        # 初始化状态变量
        frame_count = 0
        start_time = time.time()
        prev_frame = None
        
        logger.info("开始处理视频流...")
        
        while True:
            # 获取帧
            if test_mode:
                # 在测试模式下生成移动的圆
                frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                center_x = int(args.width/2 + args.width/4 * np.sin(frame_count / 30.0))
                center_y = int(args.height/2 + args.height/4 * np.cos(frame_count / 30.0))
                cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            
            # 提取运动特征
            features = motion_manager.extract_features(frame, prev_frame)
            
            # 可视化特征
            vis_frame = frame.copy()
            if features:
                vis_frame = motion_manager.visualize_features(vis_frame, features)
            
            # 获取运动历史图像
            if args.use_motion_history:
                mhi = motion_manager.get_motion_history_image()
                if mhi is not None:
                    try:
                        # 确保MHI是8位格式
                        if mhi.dtype != np.uint8:
                            mhi_norm = cv2.normalize(mhi, None, 0, 255, cv2.NORM_MINMAX)
                            mhi_8bit = np.uint8(mhi_norm)
                        else:
                            mhi_8bit = mhi
                        
                        # 转换为彩色显示
                        mhi_color = cv2.applyColorMap(mhi_8bit, cv2.COLORMAP_JET)
                        # 调整大小
                        mhi_color = cv2.resize(mhi_color, (vis_frame.shape[1] // 4, vis_frame.shape[0] // 4))
                        # 放在右上角
                        vis_frame[10:10+mhi_color.shape[0], vis_frame.shape[1]-mhi_color.shape[1]-10:vis_frame.shape[1]-10] = mhi_color
                    except Exception as e:
                        logger.error(f"可视化运动历史图像时出错: {str(e)}")
            
            # 显示特征统计
            # 统计特征类型
            feature_types = {}
            for feature in features:
                feature_type = feature.type
                if feature_type in feature_types:
                    feature_types[feature_type] += 1
                else:
                    feature_types[feature_type] = 1
            
            # 显示帧信息和特征统计
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Features: {len(features)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            y_pos = 120
            for feature_type, count in feature_types.items():
                cv2.putText(vis_frame, f"{feature_type}: {count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30
            
            # 显示启用的功能
            status_text = []
            if args.use_optical_flow:
                status_text.append("光流:开")
            if args.use_motion_history:
                status_text.append("运动历史:开")
            
            status_str = " | ".join(status_text)
            cv2.putText(vis_frame, status_str, (10, vis_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示结果
            if args.display:
                cv2.imshow('视频运动分析', vis_frame)
            
            # 保存结果
            if args.save_frames and frame_count % args.save_interval == 0:
                output_path = os.path.join(args.output_dir, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(output_path, vis_frame)
                logger.info(f"保存帧到: {output_path}")
            
            # 更新状态
            prev_frame = frame.copy()
            frame_count += 1
            
            # 检查退出条件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q或Esc键退出
                break
        
        # 清理资源
        if not test_mode and cap.isOpened():
            cap.release()
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